"""로지스틱 회귀 슬롯 복원기 — 대조군이자 코퍼스 스케일링 실험의 고정 아키텍처.

E1에서 코퍼스만 바꾸며 곡선을 그릴 때 아키텍처를 이걸로 고정한다. 학습이 빨라
505문장부터 30만문장까지 여러 지점을 하루 안에 돌릴 수 있는 게 유일한 선택 이유다.

`SlotRestorer`는 기법별로 "모르는 슬롯"에만 분류기를 붙인다(`features.UNKNOWN_SLOTS`).
아는 슬롯은 입력에서 그대로 복사한다 — 초성체는 초성이, 된소리는 중성·종성이 이미 주어진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .. import features as F
from ..jamo import N_CHO, N_JONG, N_JUNG, compose, decompose, is_syllable, split

_SLOT_SIZES = (N_CHO, N_JUNG, N_JONG)
_SLOT_NAMES = ("cho", "jung", "jong")


@dataclass
class SlotRestorer:
    """기법 하나에 대한 위치별 슬롯 복원기."""

    technique: str
    window: int = 4
    C: float = 2.0
    max_iter: int = 400
    vocab: F.CharVocab = field(default_factory=F.CharVocab)
    heads: dict[int, LogisticRegression] = field(default_factory=dict)
    _constant: dict[int, int] = field(default_factory=dict)

    @property
    def unknown_slots(self) -> tuple[int, ...]:
        return F.UNKNOWN_SLOTS[self.technique]

    def fit(self, examples: Sequence[F.SlotExample]) -> "SlotRestorer":
        if not examples:
            raise ValueError("학습 예제가 비었습니다.")
        self.vocab.fit(examples).freeze()
        matrix = F.char_window_matrix(examples, self.vocab)
        for slot in self.unknown_slots:
            y = F.targets_array(examples, slot)
            mask = y >= 0
            if not mask.any():
                raise ValueError(f"슬롯 {_SLOT_NAMES[slot]}의 정답이 하나도 없습니다.")
            y_slot = y[mask]
            classes = np.unique(y_slot)
            if len(classes) == 1:
                # 한 클래스뿐이면 LogisticRegression이 학습을 거부한다. 상수 예측기로 대체.
                self._constant[slot] = int(classes[0])
                continue
            model = LogisticRegression(max_iter=self.max_iter, C=self.C)
            model.fit(matrix[mask], y_slot)
            self.heads[slot] = model
        return self

    def predict_slots(self, examples: Sequence[F.SlotExample]) -> np.ndarray:
        """(n, 3) 예측 슬롯. 아는 슬롯은 입력값을 그대로 둔다."""
        out = np.array([ex.input_slots for ex in examples], dtype=np.int64)
        if not len(examples):
            return out
        matrix = F.char_window_matrix(examples, self.vocab)
        for slot in self.unknown_slots:
            if slot in self._constant:
                out[:, slot] = self._constant[slot]
            elif slot in self.heads:
                out[:, slot] = self.heads[slot].predict(matrix)
        return out

    def predict_proba_slots(self, examples: Sequence[F.SlotExample]) -> dict[int, np.ndarray]:
        """슬롯별 클래스 확률. abstention 임계값 스윕에 쓴다."""
        matrix = F.char_window_matrix(examples, self.vocab)
        return {
            slot: model.predict_proba(matrix)
            for slot, model in self.heads.items()
        }

    def restore(self, text: str) -> tuple[str, list[float]]:
        """문자열 하나를 복원. (복원문, 후보 위치별 confidence) 반환.

        confidence는 예측한 슬롯들의 확률 곱이다. 후보 위치가 없으면 빈 목록.
        """
        positions = F.candidate_positions(text, self.technique)
        if not positions:
            return text, []
        from .. import pairs as _pairs

        pair = _pairs.Pair("_", "_", self.technique, 1.0, text, text)
        examples = F.extract_examples([pair], window=self.window)
        predicted = self.predict_slots(examples)
        probs = self.predict_proba_slots(examples) if self.heads else {}

        slots = decompose(text)
        confidences: list[float] = []
        for row, ex in enumerate(examples):
            conf = 1.0
            for slot in self.unknown_slots:
                if slot in probs:
                    model = self.heads[slot]
                    idx = int(np.searchsorted(model.classes_, predicted[row, slot]))
                    if idx < len(model.classes_) and model.classes_[idx] == predicted[row, slot]:
                        conf *= float(probs[slot][row, idx])
            confidences.append(conf)
            cho, jung, jong = (int(v) for v in predicted[row])
            if min(cho, jung, jong) < 0:
                continue      # 예측이 불완전하면 그 자리는 건드리지 않는다
            slots[ex.char_index] = (cho, jung, jong)

        # 초성체처럼 입력이 음절이 아니었던 자리는 compose가 원문 문자를 되돌리므로,
        # 예측한 슬롯을 반영하려면 그 자리를 음절로 채워 넣어야 한다.
        chars = list(text)
        for row, ex in enumerate(examples):
            s = slots[ex.char_index]
            if s is None:
                continue
            cho, jung, jong = s
            if min(cho, jung, jong) < 0:
                continue
            from ..jamo import join

            chars[ex.char_index] = join(cho, jung, jong)
        return "".join(chars), confidences
