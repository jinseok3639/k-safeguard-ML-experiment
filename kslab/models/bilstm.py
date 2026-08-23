"""소형 양방향 LSTM 슬롯 복원기 (torch, CPU 전용).

MLP와 결정적으로 다른 점: MLP는 고정 폭 문자 창(+-4자)만 보지만 BiLSTM은 **문장 전체**를
양방향으로 읽는다. 연음처럼 받침이 어디로 넘어갔는지 알려면 인접 음절 너머의 문맥이
필요한 경우에 이게 유리할 것으로 예상되고, E2/E5b가 그 가설을 검증한다.

CPU 예산에 맞춰 크게 줄였다(embed 96 / hidden 128 / 2층, 시퀀스 상한 160). 참조한 대회
솔루션은 embed 512 / hidden 512 / 4층이었지만 그건 A100 기준이다.

인터페이스는 선형·MLP 복원기와 같다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import features as F
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

    def fit(self, pairs) -> "BiLstmSlotRestorer":
        """선형·MLP와 달리 슬롯 예제가 아니라 **페어**를 받는다 (문장 단위 모델이라서)."""
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)

        obfuscated = [p.obfuscated for p in pairs]
        clean = [p.clean for p in pairs]
        if not obfuscated:
            raise ValueError("학습 페어가 비었습니다.")
        self._fit_vocab(obfuscated + clean)

        tokens = torch.from_numpy(self._encode(obfuscated))
        targets = {s: torch.from_numpy(v) for s, v in self._targets(obfuscated, clean).items()}

        self.net = _BiLstmNet(
            len(self.vocab), self.embed_dim, self.hidden, self.layers,
            self.unknown_slots, self.dropout,
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        generator = torch.Generator().manual_seed(self.random_state)
        n = len(obfuscated)

        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            total, batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                logits = self.net(tokens[idx])
                loss = sum(
                    loss_fn(
                        logits[slot].reshape(-1, _SLOT_SIZES[slot]),
                        targets[slot][idx].reshape(-1),
                    )
                    for slot in self.unknown_slots
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total += float(loss)
                batches += 1
            self._fit_log.append(round(total / max(batches, 1), 4))
        self.net.eval()
        return self

    def _forward(self, texts: Sequence[str]) -> dict[int, np.ndarray]:
        torch = _torch()
        tokens = torch.from_numpy(self._encode(texts))
        chunks: dict[int, list] = {slot: [] for slot in self.unknown_slots}
        with torch.no_grad():
            for start in range(0, len(texts), 128):
                logits = self.net(tokens[start : start + 128])
                for slot, value in logits.items():
                    chunks[slot].append(torch.softmax(value, dim=-1).numpy())
        return {slot: np.concatenate(parts, axis=0) for slot, parts in chunks.items()}

    def restore_many(self, texts: Sequence[str]) -> list[tuple[str, list[float]]]:
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
                if min(slots) < 0:
                    continue      # 예측이 불완전한 자리는 원문 그대로 둔다
                results[r][c] = join(*slots)
        return [("".join(chars), conf) for chars, conf in zip(results, confidences)]

    def restore(self, text: str) -> tuple[str, list[float]]:
        return self.restore_many([text])[0]

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
