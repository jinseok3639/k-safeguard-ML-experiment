"""난독화 레시피 — 문장 하나를 다시 만들 수 있는 기록(데이터 계보).

레시피는 원문에 거는 단계 목록이다. 단계마다 생성기(`kslab` 내부 또는 `kslab.external_gen`의 외부), 기법,
강도, 시드를 적는다. 같은 레시피를 같은 코드·외부 저장소 커밋에서 다시 실행하면 같은 문장이 나온다.

단계마다 바로 앞 문장과 글자 단위로 비교해(difflib) 바뀐 구간을 기록하고, 그 구간을 원문 위치로 되돌려
연결한다. 같은 글자를 여러 단계가 건드려도 모든 단계가 남는다(`compose.provenance`는 마지막 단계만 남긴다).

한계: 길이가 바뀌는 단계는 difflib이 최소 편집으로 구간을 고르므로, 같은 글자가 반복되는 곳에서 위치가 한두 글자
어긋날 수 있다. 길이가 같은 단계는 글자별로 비교해 정확하다.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass

from kslab import obfuscate

INTERNAL = "kslab"


@dataclass(frozen=True)
class Step:
    generator: str        # "kslab" 또는 external_gen.GENERATORS의 이름
    op: str               # 내부는 기법 이름, 외부는 드라이버 op
    intensity: float = 0.0   # 내부만 쓴다
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Edit:
    step: int             # 레시피에서 몇 번째 단계인가
    op: str
    before_span: tuple[int, int]    # 단계 전 문장의 위치
    after_span: tuple[int, int]     # 단계 후 문장의 위치
    orig_span: tuple[int, int]      # 원문 위치. 원문에 없던 글자를 고친 삽입이면 (p, p)
    before: str
    after: str


def diff_spans(before: str, after: str) -> list[tuple[int, int, int, int]]:
    """바뀐 구간 (i1, i2, j1, j2) 목록. before[i1:i2] 가 after[j1:j2] 로 바뀌었다.

    길이가 같으면 글자별로 비교한다(정확). 길이가 다를 때만 difflib을 쓴다.
    """
    if len(before) == len(after):
        spans, start = [], None
        for i, (a, b) in enumerate(zip(before, after)):
            if a != b and start is None:
                start = i
            elif a == b and start is not None:
                spans.append((start, i, start, i))
                start = None
        if start is not None:
            spans.append((start, len(before), start, len(before)))
        return spans
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]


def _orig_span(origin: list[int], i1: int, i2: int) -> tuple[int, int]:
    mapped = [o for o in origin[i1:i2] if o >= 0]
    if mapped:
        return min(mapped), max(mapped) + 1
    # 원문에 없는 자리: 앞쪽에서 가장 가까운 원문 글자 바로 뒤
    prev = next((o for o in reversed(origin[:i1]) if o >= 0), -1)
    return prev + 1, prev + 1


def track(step_index: int, op: str, before: str, after: str, origin: list[int]) -> tuple[list[Edit], list[int]]:
    """한 단계의 바뀐 구간을 기록하고, 단계 후 문장의 글자별 원문 위치를 돌려준다(-1 = 원문에 없음)."""
    edits, new_origin, last = [], [], 0
    for i1, i2, j1, j2 in diff_spans(before, after):
        new_origin += origin[last:i1]
        edits.append(Edit(step_index, op, (i1, i2), (j1, j2), _orig_span(origin, i1, i2),
                          before[i1:i2], after[j1:j2]))
        if i2 - i1 == j2 - j1:
            new_origin += origin[i1:i2]
        else:
            anchor = next((o for o in origin[i1:i2] if o >= 0), -1)
            new_origin += [anchor] * (j2 - j1)
        last = i2
    new_origin += origin[last:]
    return edits, new_origin


@dataclass
class Result:
    text: str
    applied: list[int]          # 실제로 문장을 바꾼 단계 번호
    edits: list[Edit]
    steps_text: list[str]       # 단계마다 끝난 뒤의 문장


def run_internal(text: str, steps: list[Step]) -> Result:
    """내부 기법만으로 된 레시피를 실행한다."""
    origin = list(range(len(text)))
    current, applied, edits, texts = text, [], [], []
    for index, step in enumerate(steps):
        if step.generator != INTERNAL:
            raise ValueError(f"내부 레시피에 외부 단계가 있습니다: {step}")
        nxt = obfuscate.TRANSFORMS[step.op](current, step.intensity, seed=step.seed)
        if nxt != current:
            step_edits, origin = track(index, step.op, current, nxt, origin)
            edits += step_edits
            applied.append(index)
        current = nxt
        texts.append(current)
    return Result(current, applied, edits, texts)


def from_texts(text: str, ops: list[str], texts: list[str], step_offset: int = 0) -> Result:
    """외부 생성기가 돌려준 단계별 문장으로 구간을 기록한다."""
    origin = list(range(len(text)))
    current, applied, edits = text, [], []
    for index, (op, nxt) in enumerate(zip(ops, texts)):
        if nxt != current:
            step_edits, origin = track(step_offset + index, op, current, nxt, origin)
            edits += step_edits
            applied.append(step_offset + index)
        current = nxt
    return Result(current, applied, edits, list(texts))
