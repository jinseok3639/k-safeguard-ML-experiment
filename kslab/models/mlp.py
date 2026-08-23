"""자모 임베딩 + MLP 슬롯 복원기 (torch, CPU 전용).

선형 모델과 같은 인터페이스(`fit` / `predict_slots` / `restore_many`)를 지켜서 E2에서
그대로 바꿔 끼울 수 있게 했다.

설계 근거는 저장소의 `experiments/참조용/README.md`가 정리해 둔 검토 결과다 — 자모 단위
표현, 어절 내 상대 위치를 특징으로 쓰기, 비대상 문자 보존. 노트북 구현 코드는 가져오지
않았다(저장소 방침). 실제 구조는 이 프로젝트의 후보 규칙에 맞춰 다시 짰다.

선형 모델과 다른 점 두 가지:
- 문자 창을 one-hot 대신 **학습되는 임베딩**으로 표현한다. 어휘가 커져도 파라미터가
  선형으로만 늘고, 비슷한 문자끼리 표현을 공유할 수 있다.
- 초성·중성·종성 head가 **몸통을 공유**한다. 세 슬롯이 같은 문맥 신호를 쓰므로.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .. import features as F
from ..jamo import N_CHO, N_JONG, N_JUNG, join

_SLOT_SIZES = (N_CHO, N_JUNG, N_JONG)


def _torch():
    import torch

    return torch


class _SlotNet:
    """지연 생성되는 torch 모듈. torch import를 클래스 정의 시점에서 떼어놓는다."""

    def __new__(cls, vocab_size: int, window_width: int, embed_dim: int, hidden: int,
                unknown_slots: Sequence[int], dropout: float):
        torch = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=F.PAD)
                # 입력 자모 슬롯도 특징으로 넣는다. '모름'을 위해 클래스 수 + 1.
                self.slot_embeds = nn.ModuleList(
                    [nn.Embedding(size + 1, 8) for size in _SLOT_SIZES]
                )
                body_in = window_width * embed_dim + 3 * 8 + 1  # +1 은 어절 내 상대 위치
                self.body = nn.Sequential(
                    nn.Linear(body_in, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.residual = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.heads = nn.ModuleDict(
                    {str(slot): nn.Linear(hidden, _SLOT_SIZES[slot]) for slot in unknown_slots}
                )

            def forward(self, chars, pos, slots):
                x = self.embedding(chars).flatten(1)
                slot_feats = [emb(slots[:, i]) for i, emb in enumerate(self.slot_embeds)]
                x = torch.cat([x, *slot_feats, pos], dim=1)
                h = self.body(x)
                h = h + self.residual(h)   # skip connection
                return {int(k): head(h) for k, head in self.heads.items()}

        return Net()


@dataclass
class MlpSlotRestorer:
    """기법 하나에 대한 자모 MLP 슬롯 복원기."""

    technique: str
    window: int = 4
    embed_dim: int = 48
    hidden: int = 256
    dropout: float = 0.1
    epochs: int = 8
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-2
    random_state: int = 2026
    threads: int | None = None
    vocab: F.CharVocab = field(default_factory=F.CharVocab)
    net: object | None = None
    _fit_log: list = field(default_factory=list)

    @property
    def unknown_slots(self) -> tuple[int, ...]:
        return F.UNKNOWN_SLOTS[self.technique]

    def _arrays(self, examples):
        chars, pos, slots = F.jamo_window_arrays(examples, self.vocab)
        return chars, pos, slots

    def fit(self, examples: Sequence[F.SlotExample]) -> "MlpSlotRestorer":
        torch = _torch()
        if self.threads:
            torch.set_num_threads(self.threads)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if not examples:
            raise ValueError("학습 예제가 비었습니다.")
        if not self.unknown_slots:
            raise ValueError(f"{self.technique}에는 예측할 슬롯이 없습니다.")

        self.vocab.fit(examples).freeze()
        chars, pos, slots = self._arrays(examples)
        targets = {slot: F.targets_array(examples, slot) for slot in self.unknown_slots}

        width = chars.shape[1]
        self.net = _SlotNet(
            len(self.vocab), width, self.embed_dim, self.hidden, self.unknown_slots, self.dropout
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

        tensors = {
            "chars": torch.from_numpy(chars),
            "pos": torch.from_numpy(pos),
            "slots": torch.from_numpy(slots),
        }
        target_tensors = {s: torch.from_numpy(v) for s, v in targets.items()}
        n = len(examples)
        generator = torch.Generator().manual_seed(self.random_state)

        self.net.train()
        for epoch in range(self.epochs):
            order = torch.randperm(n, generator=generator)
            total, batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                if len(idx) < 2:
                    continue  # BatchNorm은 배치 1을 못 받는다
                logits = self.net(
                    tensors["chars"][idx], tensors["pos"][idx], tensors["slots"][idx]
                )
                loss = sum(
                    loss_fn(logits[slot], target_tensors[slot][idx])
                    for slot in self.unknown_slots
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
                batches += 1
            self._fit_log.append(round(total / max(batches, 1), 4))
        self.net.eval()
        return self

    def predict_slots(self, examples: Sequence[F.SlotExample]) -> np.ndarray:
        out = np.array([ex.input_slots for ex in examples], dtype=np.int64)
        if not len(examples) or self.net is None:
            return out
        proba = self.predict_proba_slots(examples)
        for slot, probs in proba.items():
            out[:, slot] = probs.argmax(axis=1)
        return out

    def predict_proba_slots(self, examples: Sequence[F.SlotExample]) -> dict[int, np.ndarray]:
        torch = _torch()
        chars, pos, slots = self._arrays(examples)
        result: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for start in range(0, len(examples), 4096):
                stop = start + 4096
                logits = self.net(
                    torch.from_numpy(chars[start:stop]),
                    torch.from_numpy(pos[start:stop]),
                    torch.from_numpy(slots[start:stop]),
                )
                for slot, value in logits.items():
                    chunk = torch.softmax(value, dim=1).numpy()
                    result[slot] = (
                        chunk if slot not in result else np.vstack([result[slot], chunk])
                    )
        return result

    def restore_many(
        self, texts: Sequence[str], threshold: float = 0.0
    ) -> list[tuple[str, list[float]]]:
        """threshold 이상 확신하는 자리만 바꾼다.

        E3에서 확인했듯 ΔFPR이 구속 조건인 상황에서는 "모르면 손대지 않는" 편이
        항상 낫다. threshold=0.0 이면 종전대로 argmax를 그대로 적용한다.
        """
        from .. import pairs as _pairs

        pairs = [_pairs.Pair("_", "_", self.technique, 1.0, t, t) for t in texts]
        examples = F.extract_examples(pairs, window=self.window)
        results = [list(t) for t in texts]
        confidences: list[list[float]] = [[] for _ in texts]
        if not examples:
            return [("".join(c), conf) for c, conf in zip(results, confidences)]

        proba = self.predict_proba_slots(examples)
        predicted = np.array([ex.input_slots for ex in examples], dtype=np.int64)
        for slot, probs in proba.items():
            predicted[:, slot] = probs.argmax(axis=1)

        for row, ex in enumerate(examples):
            conf = 1.0
            for slot, probs in proba.items():
                conf *= float(probs[row, predicted[row, slot]])
            confidences[ex.sentence_index].append(conf)
            if conf < threshold:
                continue      # abstention — 이 자리는 원문 그대로 둔다
            cho, jung, jong = (int(v) for v in predicted[row])
            if min(cho, jung, jong) < 0:
                continue
            results[ex.sentence_index][ex.char_index] = join(cho, jung, jong)
        return [("".join(chars), conf) for chars, conf in zip(results, confidences)]

    def restore(self, text: str, threshold: float = 0.0) -> tuple[str, list[float]]:
        return self.restore_many([text], threshold)[0]

    def parameter_count(self) -> int:
        if self.net is None:
            return 0
        return sum(p.numel() for p in self.net.parameters())
