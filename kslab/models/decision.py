"""된소리 되돌림 **판단** 모델 — 위치별 이진 분류 + abstention.

된소리는 다른 기법과 성질이 다르다. 무엇으로 되돌릴지는 `ㄲ→ㄱ`로 결정적이라 이미 풀려
있고(저장소 실측 NRR 100%), 남은 문제는 **어느 위치를 되돌릴지** 하나뿐이다.

현행 `TensifyInverseProvider`는 이 질문에 답하지 않고 모든 부분집합 `2^n-1`개를 후보로
뿌린 뒤 하위 가드레일에 OR로 넘긴다. 회복은 완벽했지만 ΔFPR-obf가 +14.29%p 올라
`DO_NOT_PROMOTE`가 됐다. 위치별 이진 판단은 그 후보를 1개로 붕괴시킨다.

**abstention이 핵심이다.** ΔFPR이 구속 조건인 상황에서는 "모르면 후보를 아예 안 내는" 편이
"찍는" 것보다 항상 낫다. 확률이 임계값 아래면 그 위치는 손대지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .. import features as F
from .. import tracking
from ..jamo import (
    CHO_INDEX,
    COMPAT_CHO,
    TENSE_CHO_INDICES,
    TENSE_TO_LAX,
    is_syllable,
    join,
    split,
)

KEEP, REVERT = 0, 1


def tense_positions(text: str) -> list[int]:
    """된소리 초성을 가진 음절 위치. 현행 provider의 `_tense_positions`와 같은 규칙."""
    return F.candidate_positions(text, "tensify")


def detensify(text: str, positions: Sequence[int]) -> str:
    """지정한 위치의 경음 초성을 평음으로. 중성·종성은 보존."""
    chars = list(text)
    for i in positions:
        cho, jung, jong = split(text[i])
        lax = TENSE_TO_LAX.get(COMPAT_CHO[cho])
        if lax is None:
            continue
        chars[i] = join(CHO_INDEX[lax], jung, jong)
    return "".join(chars)


def hangul_syllable_count(text: str) -> int:
    return sum(1 for ch in text if is_syllable(ch))


def current_policy_candidates(
    text: str,
    max_candidates: int = 9,
    min_tense_syllables: int = 1,
    min_tense_ratio: float = 0.10,
) -> list[str]:
    """현행 `TensifyInverseProvider` 정책의 재현 — 비교 대조군.

    activation gate를 통과하면 되돌릴 위치의 모든 부분집합을 '많이 되돌리는 것부터'
    생성하고 `max_candidates`에서 자른다.
    """
    positions = tense_positions(text)
    total = hangul_syllable_count(text)
    ratio = len(positions) / total if total else 0.0
    if len(positions) < min_tense_syllables or ratio < min_tense_ratio:
        return []
    out: list[str] = []
    for count in range(len(positions), 0, -1):
        for selected in combinations(positions, count):
            out.append(detensify(text, selected))
            if len(out) >= max_candidates:
                return out
    return out


@dataclass
class TensifyDecider:
    """위치별 '되돌릴까/둘까' 이진 분류기."""

    window: int = 4
    C: float = 2.0
    max_iter: int = 60
    solver: str = "sgd"
    random_state: int = 2026
    vocab: F.CharVocab = field(default_factory=F.CharVocab)
    head: object | None = None
    _constant: int | None = None

    def _new_head(self, n_samples: int):
        if self.solver == "sgd":
            return SGDClassifier(
                loss="log_loss",
                alpha=1.0 / max(self.C * max(n_samples, 1), 1e-12),
                max_iter=self.max_iter,
                tol=1e-4,
                random_state=self.random_state,
            )
        return LogisticRegression(max_iter=400, C=self.C)

    def fit(self, pairs) -> "TensifyDecider":
        """페어에서 이진 예제를 뽑아 학습.

        라벨은 입력만으로 정해지지 않고 (난독화문, 원문) 쌍에서 나온다: 그 자리가 원문과
        다르면 공격자가 만든 된소리(REVERT), 같으면 원래 그런 단어(KEEP).
        identity 페어(원문 그대로)의 된소리 위치가 KEEP 표본을 공급한다 — `깨끗`·`씨앗`
        같은 정상 어휘를 지키는 신호가 여기서 나온다.
        """
        examples, labels = [], []
        for pair in pairs:
            if len(pair.obfuscated) != len(pair.clean):
                continue
            for example in F.extract_examples([pair], self.window):
                examples.append(example)
                labels.append(REVERT if example.changed else KEEP)
        tracking.begin_fit("tensify_decider", self, n_examples=len(examples))
        if not examples:
            raise ValueError("학습 예제가 비었습니다.")

        self.vocab.fit(examples).freeze()
        matrix = F.char_window_matrix(examples, self.vocab)
        y = np.asarray(labels)
        if len(np.unique(y)) == 1:
            self._constant = int(y[0])
            return self
        self.head = self._new_head(len(y))
        self.head.fit(matrix, y)
        return self

    def _revert_probabilities(self, examples) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(examples), float(self._constant))
        matrix = F.char_window_matrix(examples, self.vocab)
        classes = list(self.head.classes_)
        probs = self.head.predict_proba(matrix)
        return probs[:, classes.index(REVERT)] if REVERT in classes else np.zeros(len(examples))

    def restore_many(
        self, texts: Sequence[str], threshold: float = 0.5
    ) -> list[tuple[str, dict]]:
        """임계값 이상인 위치만 되돌린다. 후보는 문장당 최대 1개.

        임계값이 높을수록 abstention이 늘고, 손대지 않은 문장은 아예 후보를 만들지 않는다
        (=하위 가드레일에 추가 view가 안 감).
        """
        from .. import pairs as _pairs

        pairs = [_pairs.Pair("_", "_", "tensify", 1.0, t, t) for t in texts]
        examples = F.extract_examples(pairs, self.window)
        results = [list(t) for t in texts]
        stats = [{"candidates": 0, "reverted": 0, "abstained": 0} for _ in texts]
        if not examples:
            return [("".join(c), s) for c, s in zip(results, stats)]

        probabilities = self._revert_probabilities(examples)
        for row, example in enumerate(examples):
            index = example.sentence_index
            stats[index]["candidates"] += 1
            if probabilities[row] >= threshold:
                char = texts[index][example.char_index]
                cho, jung, jong = split(char)
                lax = TENSE_TO_LAX.get(COMPAT_CHO[cho])
                if lax is not None:
                    results[index][example.char_index] = join(CHO_INDEX[lax], jung, jong)
                    stats[index]["reverted"] += 1
            else:
                stats[index]["abstained"] += 1
        return [("".join(chars), s) for chars, s in zip(results, stats)]

    def restore(self, text: str, threshold: float = 0.5) -> tuple[str, dict]:
        return self.restore_many([text], threshold)[0]
