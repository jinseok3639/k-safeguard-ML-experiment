"""문자 n-gram 게이트 — "이 어절에 복원 모델을 돌릴 필요가 있는가".

초성체는 호환 자모가 눈에 보이고 된소리는 경음 밀도라는 값싼 신호가 있다. 연음과 종성
크래밍은 그런 게 없다 — 결과물이 문법적으로 멀쩡한 한글 음절이라 입력만 봐서는 구별이
안 된다. 그래서 `features.candidate_positions`가 모든 음절을 후보로 돌려주고, 그 결과
clean mutation이 90%대로 뛴다.

게이트는 이 문제를 앞단에서 자른다. 정상 한국어 코퍼스로 문자 n-gram 통계 언어모델을
학습해 두고, **어절 단위로 부자연스러운 곳만** 무거운 모델로 보낸다. 정상 문장의 어절은
대부분 게이트를 통과하지 못하므로 애초에 손을 안 대게 된다.

GPU도 신경망도 필요 없다. stupid backoff는 계산이 값싸고 결정론적이다.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

BOS = "\x02"
EOS = "\x03"


@dataclass
class CharNgramLM:
    """문자 n-gram 언어모델 (stupid backoff).

    stupid backoff를 쓰는 이유는 정규화된 확률이 필요 없기 때문이다. 우리가 쓰는 건
    어절끼리의 **상대 점수**뿐이라, Kneser-Ney 같은 제대로 된 스무딩의 비용을 치를 이유가 없다.
    """

    order: int = 5
    alpha: float = 0.4          # backoff 할인 계수
    min_count: int = 2          # 최고차 n-gram 가지치기 (메모리 절약)
    counts: list = field(default_factory=list)   # counts[k] = k+1-gram Counter
    total_chars: int = 0
    vocab: set = field(default_factory=set)

    def fit(self, texts: Iterable[str]) -> "CharNgramLM":
        self.counts = [Counter() for _ in range(self.order)]
        for text in texts:
            padded = BOS * (self.order - 1) + text + EOS
            self.vocab.update(text)
            self.total_chars += len(text) + 1
            for k in range(self.order):
                size = k + 1
                for i in range(len(padded) - size + 1):
                    gram = padded[i : i + size]
                    if BOS in gram[1:]:
                        continue     # 패딩이 중간에 낀 gram은 세지 않는다
                    self.counts[k][gram] += 1
        if self.min_count > 1:
            top = self.counts[self.order - 1]
            self.counts[self.order - 1] = Counter(
                {g: c for g, c in top.items() if c >= self.min_count}
            )
        return self

    def _prob(self, context: str, char: str) -> float:
        """stupid backoff 점수. 확률은 아니지만 순위는 맞다."""
        weight = 1.0
        for size in range(min(len(context), self.order - 1), -1, -1):
            ctx = context[len(context) - size :] if size else ""
            gram = ctx + char
            numerator = self.counts[len(gram) - 1].get(gram, 0)
            if numerator:
                denominator = self.counts[len(ctx) - 1].get(ctx, 0) if ctx else self.total_chars
                if denominator:
                    return weight * numerator / denominator
            weight *= self.alpha
        # 완전 미등록 문자: 어휘 크기로 최소 확률을 준다
        return weight / max(len(self.vocab), 1)

    def logscore(self, text: str) -> float:
        """문자당 평균 음의 로그 점수. 높을수록 '부자연스럽다'."""
        if not text:
            return 0.0
        padded = BOS * (self.order - 1) + text
        total = 0.0
        for i in range(len(text)):
            context = padded[i : i + self.order - 1]
            total -= math.log(max(self._prob(context, text[i]), 1e-12))
        return total / len(text)

    def size(self) -> dict:
        return {
            "order": self.order,
            "vocab": len(self.vocab),
            "grams": [len(c) for c in self.counts],
            "total_chars": self.total_chars,
        }


def word_spans(text: str) -> list[tuple[int, int]]:
    """공백 기준 어절 경계 (start, end). 게이트의 판정 단위."""
    spans, start = [], None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


@dataclass
class PerplexityGate:
    """어절 점수가 임계값을 넘는 구간만 통과시킨다."""

    lm: CharNgramLM
    threshold: float
    context: int = 1     # 앞뒤로 함께 넘길 인접 어절 수

    def suspicious_spans(self, text: str) -> list[tuple[int, int]]:
        spans = word_spans(text)
        if not spans:
            return []
        scores = [self.lm.logscore(text[s:e]) for s, e in spans]
        flagged = {i for i, score in enumerate(scores) if score >= self.threshold}
        if not flagged:
            return []
        # 인접 어절까지 포함해 병합한다 — 연음은 어절 경계를 넘나든다.
        widened = set()
        for i in flagged:
            for j in range(max(0, i - self.context), min(len(spans), i + self.context + 1)):
                widened.add(j)
        merged: list[tuple[int, int]] = []
        for i in sorted(widened):
            start, end = spans[i]
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def selected_positions(self, text: str) -> set[int]:
        """게이트를 통과한 문자 인덱스 집합."""
        return {i for start, end in self.suspicious_spans(text) for i in range(start, end)}

    def flags(self, text: str) -> bool:
        return bool(self.suspicious_spans(text))


def gated_restore(model, gate: PerplexityGate, texts: Sequence[str]) -> list[str]:
    """게이트를 통과한 위치만 모델 결과를 채택한다.

    모델은 문장 전체에 대해 한 번 돌리고(배치가 싸다), 게이트가 통과시키지 않은 자리는
    원문 문자를 되돌려 놓는다. 이렇게 하면 게이트가 clean mutation을 직접 줄인다.
    """
    restored = [r[0] for r in model.restore_many(texts)]
    out = []
    for original, candidate in zip(texts, restored):
        if len(original) != len(candidate):
            out.append(candidate)
            continue
        allowed = gate.selected_positions(original)
        out.append(
            "".join(
                candidate[i] if i in allowed else original[i] for i in range(len(original))
            )
        )
    return out
