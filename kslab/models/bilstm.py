"""소형 양방향 LSTM 슬롯 복원기 (torch). 기본은 CPU, `device`로 GPU를 쓴다.

MLP와 결정적으로 다른 점: MLP는 고정 폭 문자 창(+-4자)만 보지만 BiLSTM은 **문장 전체**를
양방향으로 읽는다. 연음처럼 받침이 어디로 넘어갔는지 알려면 인접 음절 너머의 문맥이
필요한 경우에 이게 유리할 것으로 예상되고, E2/E5b가 그 가설을 검증한다.

CPU 예산에 맞춰 크게 줄였다(embed 96 / hidden 128 / 2층, 시퀀스 상한 160). 참조한 대회
솔루션은 embed 512 / hidden 512 / 4층이었지만 그건 A100 기준이다.

인터페이스는 선형·MLP 복원기와 같다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from .. import features as F
from .. import tracking
from . import device as _device
from ..jamo import CHO_INDEX, N_CHO, N_JONG, N_JUNG, is_syllable, join, split

_SLOT_SIZES = (N_CHO, N_JUNG, N_JONG)
IGNORE = -1


def _input_slots(char: str) -> tuple[int, int, int]:
    """입력 문자에서 이미 알고 있는 자모. 모르는 자리는 -1."""
    if is_syllable(char):
        return split(char)
    if char in CHO_INDEX:
        return (CHO_INDEX[char], -1, -1)   # 초성체: 초성만 주어짐
    return (-1, -1, -1)


def _torch():
    import torch

    return torch


class _BiLstmNet:
    def __new__(cls, vocab_size: int, embed_dim: int, hidden: int, layers: int,
                unknown_slots: Sequence[int], dropout: float):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=F.PAD)
                self.norm_in = nn.LayerNorm(embed_dim)
                self.lstm = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=hidden,
                    num_layers=layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.norm_out = nn.LayerNorm(hidden * 2)
                self.drop = nn.Dropout(dropout)
                self.heads = nn.ModuleDict(
                    {str(slot): nn.Linear(hidden * 2, _SLOT_SIZES[slot]) for slot in unknown_slots}
                )

            def forward(self, tokens):
                x = self.norm_in(self.embedding(tokens))
                out, _ = self.lstm(x)
                out = self.drop(self.norm_out(out))
                return {int(k): head(out) for k, head in self.heads.items()}

        return Net()


@dataclass
class BiLstmSlotRestorer:
    """문장 전체를 읽는 위치별 슬롯 복원기."""

    technique: str
    max_len: int = 160
    embed_dim: int = 96
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    epochs: int = 6
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    random_state: int = 2026
    threads: int | None = None
    # "cpu" | "cuda" | "auto". 기본 CPU — kslab/models/device.py 참고.
    device: str = "cpu"
    vocab: F.CharVocab = field(default_factory=F.CharVocab)
    net: object | None = None
    _fit_log: list = field(default_factory=list)
    # 선형·MLP와 인터페이스를 맞추기 위한 자리. BiLSTM은 창을 쓰지 않는다.
    window: int = 0

    @property
    def unknown_slots(self) -> tuple[int, ...]:
        return F.UNKNOWN_SLOTS[self.technique]

    def _fit_vocab(self, texts: Sequence[str]) -> None:
        # 문장 단위 모델이라 창 기반 CharVocab.fit을 그대로 못 쓴다. 문자 전체를 넣는다.
        fake = [
            F.SlotExample("_", "_", self.technique, 0, 0, tuple(t), 0.0, (-1, -1, -1), (-1, -1, -1), False)
            for t in texts
        ]
        self.vocab.fit(fake).freeze()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        tokens = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for r, text in enumerate(texts):
            for c, ch in enumerate(text[: self.max_len]):
                tokens[r, c] = self.vocab.encode(ch)
        return tokens

    def _targets(self, obfuscated: Sequence[str], clean: Sequence[str]) -> dict[int, np.ndarray]:
        """위치별 정답. 후보 위치가 아니거나 정답이 없으면 IGNORE."""
        out = {
            slot: np.full((len(obfuscated), self.max_len), IGNORE, dtype=np.int64)
            for slot in self.unknown_slots
        }
        for r, (obf, ref) in enumerate(zip(obfuscated, clean)):
            candidates = set(F.candidate_positions(obf, self.technique))
            for c in range(min(len(obf), self.max_len)):
                if c not in candidates:
                    continue
                gold = ref[c]
                if not is_syllable(gold):
                    continue
                slots = split(gold)
                for slot in self.unknown_slots:
                    out[slot][r, c] = slots[slot]
        return out

    def _checkpoint_signature(self, n_pairs: int) -> tuple:
        """체크포인트가 지금 이 설정·데이터에서 나온 게 맞는지 확인하는 지문.

        다른 technique이나 다른 페어 세트로 만든 체크포인트를 잘못 이어받으면 텐서 크기가
        안 맞거나(즉시 에러) 조용히 틀린 상태에서 이어 학습하는(더 위험한) 경우가 생긴다.
        vocab 크기까지 포함해 페어 내용이 달라졌는지도 간접적으로 잡는다.
        """
        return (
            self.technique, self.max_len, self.embed_dim, self.hidden, self.layers,
            round(self.dropout, 6), self.batch_size, self.epochs, tuple(self.unknown_slots),
            n_pairs, len(self.vocab),
        )

    def fit(
        self,
        pairs,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 200,
        show_progress: bool = True,
    ) -> "BiLstmSlotRestorer":
        """선형·MLP와 달리 슬롯 예제가 아니라 **페어**를 받는다 (문장 단위 모델이라서).

        `checkpoint_path`를 주면 `checkpoint_every` 배치마다(그리고 매 에폭 경계마다)
        모델·옵티마이저 상태를 저장하고, 그 경로에 파일이 이미 있으면 거기서부터 이어
        학습한다. 노트북처럼 중간에 죽을 수 있는 환경에서 몇 시간짜리 학습을 처음부터
        다시 하지 않기 위한 것 — 저장은 임시 파일에 쓴 뒤 원자적으로 교체해 저장 도중
        죽어도 이전 체크포인트가 깨지지 않는다.
        """
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)

        obfuscated = [p.obfuscated for p in pairs]
        clean = [p.clean for p in pairs]
        if not obfuscated:
            raise ValueError("학습 페어가 비었습니다.")
        fit_log = tracking.begin_fit("bilstm", self, n_pairs=len(obfuscated))
        self._fit_vocab(obfuscated + clean)

        tokens = torch.from_numpy(self._encode(obfuscated))
        targets = {s: torch.from_numpy(v) for s, v in self._targets(obfuscated, clean).items()}

        dev = _device.resolve(self.device)
        self.net = _BiLstmNet(
            len(self.vocab), self.embed_dim, self.hidden, self.layers,
            self.unknown_slots, self.dropout,
        ).to(dev)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        generator = torch.Generator().manual_seed(self.random_state)
        n = len(obfuscated)
        signature = self._checkpoint_signature(n)

        start_epoch, resume_batch, order = 0, 0, None
        ckpt_path = Path(checkpoint_path) if checkpoint_path else None
        if ckpt_path and ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if checkpoint.get("signature") != signature:
                print(
                    f"[bilstm:{self.technique}] 체크포인트가 현재 설정·데이터와 달라 무시합니다: "
                    f"{ckpt_path}",
                    flush=True,
                )
            else:
                self.net.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                self._fit_log = checkpoint["fit_log"]
                start_epoch = checkpoint["epoch"]
                resume_batch = checkpoint["batch"]
                order = checkpoint["order"]
                # 재개하는 에폭의 순열(order)은 그대로 이어 쓰고, generator 상태는
                # "그 순열을 뽑은 직후" 시점으로 되돌린다. 그래야 재개 이후 새로 도는
                # 에폭들의 순열이 끊기지 않고 이어진다 — 안 그러면 generator가 다시
                # 처음부터 뽑아, 재개 후 epoch들이 원래 실행과 다른(하지만 이미 예전에
                # 나왔던) 순열을 반복하게 된다.
                generator.set_state(checkpoint["generator_state"])
                # Dropout은 explicit generator가 아니라 **전역** torch RNG를 쓴다.
                # 위의 torch.manual_seed(self.random_state) 가 이미 전역 상태를
                # "처음부터 다시"로 되돌려놨으므로, 저장해 둔 전역 상태로 다시 덮어써야
                # 재개 이후 배치들의 드롭아웃 마스크가 크래시 없이 이어 돈 것과 같아진다.
                torch.set_rng_state(checkpoint["torch_rng_state"])
                # GPU에서는 드롭아웃이 CUDA RNG를 쓴다. 같은 이유로 그것도 되돌린다.
                if dev.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], dev)
                print(
                    f"[bilstm:{self.technique}] 체크포인트에서 재개: "
                    f"epoch {start_epoch + 1}/{self.epochs}, batch {resume_batch}",
                    flush=True,
                )

        def save_checkpoint(epoch: int, batch: int, order_tensor) -> None:
            if ckpt_path is None:
                return
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            torch.save(
                {
                    "signature": signature, "epoch": epoch, "batch": batch,
                    "order": order_tensor, "generator_state": generator.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
                    ),
                    "model_state": self.net.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "fit_log": self._fit_log,
                },
                tmp,
            )
            os.replace(tmp, ckpt_path)   # 같은 파일시스템 내 rename은 원자적

        self.net.train()
        epochs_iter = range(start_epoch, self.epochs)
        if show_progress:
            epochs_iter = tqdm(
                epochs_iter, desc=f"{self.technique} epoch",
                initial=start_epoch, total=self.epochs, unit="epoch",
            )
        for epoch in epochs_iter:
            if order is None:
                order = torch.randperm(n, generator=generator)
            starts = list(range(0, n, self.batch_size))
            skip = resume_batch if epoch == start_epoch else 0
            total, batches = 0.0, skip
            batch_iter = starts[skip:]
            if show_progress:
                batch_iter = tqdm(
                    batch_iter, desc="  batch", leave=False,
                    initial=skip, total=len(starts), unit="batch",
                )
            for start in batch_iter:
                idx = order[start : start + self.batch_size]
                logits = self.net(tokens[idx].to(dev))
                loss = sum(
                    loss_fn(
                        logits[slot].reshape(-1, _SLOT_SIZES[slot]),
                        targets[slot][idx].to(dev).reshape(-1),
                    )
                    for slot in self.unknown_slots
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total += float(loss)
                batches += 1
                if show_progress:
                    batch_iter.set_postfix(loss=f"{float(loss):.4f}")
                if checkpoint_every and batches % checkpoint_every == 0:
                    save_checkpoint(epoch, batches, order)
            # 재개한 에폭은 크래시 이전 배치의 손실을 못 들고 와서 이 평균이 그 에폭
            # 전체가 아니라 재개 이후분만 반영한다 — 학습 자체(가중치 갱신)는 전부 반영됨.
            self._fit_log.append(round(total / max(batches, 1), 4))
            fit_log.epoch(epoch + 1, self._fit_log[-1])
            if show_progress:
                epochs_iter.set_postfix(avg_loss=self._fit_log[-1])
            order, resume_batch = None, 0
            save_checkpoint(epoch + 1, 0, None)
        self.net.eval()
        return self

    def _forward(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        torch = _torch()
        tokens = torch.from_numpy(self._encode(texts))
        dev = next(self.net.parameters()).device
        chunks: dict[int, list] = {slot: [] for slot in self.unknown_slots}
        with torch.no_grad():
            for start in range(0, len(texts), 128):
                logits = self.net(tokens[start : start + 128].to(dev))
                for slot, value in logits.items():
                    chunks[slot].append(torch.softmax(value, dim=-1).cpu().numpy())
        return {slot: np.concatenate(parts, axis=0) for slot, parts in chunks.items()}

    def restore_many(
        self, texts: Sequence[str], threshold: float = 0.0
    ) -> list[tuple[str, list[float]]]:
        """threshold 이상 확신하는 자리만 바꾼다. 0.0이면 argmax 그대로."""
        if self.net is None:
            raise RuntimeError("먼저 fit을 호출하세요.")
        proba = self._forward(texts)
        results = [list(t) for t in texts]
        confidences: list[list[float]] = [[] for _ in texts]
        for r, text in enumerate(texts):
            for c in F.candidate_positions(text, self.technique):
                if c >= self.max_len:
                    continue      # 시퀀스 상한을 넘는 꼬리는 건드리지 않는다
                slots = list(_input_slots(text[c]))
                conf = 1.0
                for slot in self.unknown_slots:
                    probs = proba[slot][r, c]
                    best = int(probs.argmax())
                    slots[slot] = best
                    conf *= float(probs[best])
                confidences[r].append(conf)
                if conf < threshold:
                    continue      # abstention
                if min(slots) < 0:
                    continue      # 예측이 불완전한 자리는 원문 그대로 둔다
                results[r][c] = join(*slots)
        return [("".join(chars), conf) for chars, conf in zip(results, confidences)]

    def restore(self, text: str, threshold: float = 0.0) -> tuple[str, list[float]]:
        return self.restore_many([text], threshold)[0]

    def predict_slots(self, examples: Sequence[F.SlotExample]) -> np.ndarray:
        """슬롯 예제 인터페이스. 문장을 다시 모아 한 번에 통과시킨다."""
        out = np.array([ex.input_slots for ex in examples], dtype=np.int64)
        if not len(examples):
            return out
        # 예제는 sentence_index/char_index를 들고 있으나 원문 텍스트는 없다.
        # E2 러너가 텍스트를 함께 넘기도록 `predict_slots_for` 를 쓴다.
        raise NotImplementedError(
            "BiLstmSlotRestorer는 문장 단위 모델입니다. predict_slots_for(texts, examples)를 쓰세요."
        )

    def predict_slots_for(
        self, texts: Sequence[str], examples: Sequence[F.SlotExample]
    ) -> np.ndarray:
        """(문장 목록, 그 문장들에서 뽑은 예제) -> (n, 3) 예측 슬롯."""
        out = np.array([ex.input_slots for ex in examples], dtype=np.int64)
        if not len(examples):
            return out
        proba = self._forward(texts)
        for row, ex in enumerate(examples):
            if ex.char_index >= self.max_len:
                continue
            for slot in self.unknown_slots:
                out[row, slot] = int(proba[slot][ex.sentence_index, ex.char_index].argmax())
        return out

    def parameter_count(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters())
