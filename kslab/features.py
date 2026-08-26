"""슬롯 예제 추출과 특징 인코딩.

가장 중요한 규칙: **후보 위치는 입력만 보고 정해야 한다.** 학습할 때 "clean과 다른 위치"를
후보로 쓰면 추론 때 알 수 없는 정보를 쓰는 셈이고, clean mutation rate가 실제보다 좋게 나온다.
그래서 `candidate_positions()`는 난독화된 문자열만 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import phonology as PH
from .jamo import (
    CHO_INDEX,
    N_CHO,
    N_JONG,
    N_JUNG,
    TENSE_CHO_INDICES,
    is_syllable,
    split,
    word_relative_positions,
)
from .pairs import Pair

PAD = 0
UNK = 1
_SPECIAL = 2

# 조합 기법 이름은 단일 기법 이름을 이 문자로 이어 붙인다. `liaison+tensify` 처럼.
COMPOSITE_SEP = "+"


# --- 후보 위치 규칙 -------------------------------------------------------
#
# 규칙은 두 종류다.
#  - **좁은 규칙**: 난독화 결과에 눈에 보이는 흔적이 남는 경우(초성체의 호환 자모, tensify의
#    경음 초성, aspirate의 거센소리 초성). 방문할 자리가 적어 clean mutation 노출이 작다.
#  - **전수 검사**: 결과물이 문법적으로 정상인 한글이라 값싼 신호가 없는 경우. 비용이 크고
#    오탐 노출이 크다 — 이걸 줄이는 게 E6 perplexity 게이트의 존재 이유다.

def _all_syllables(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if is_syllable(ch)]


def _compat_jamo(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if ch in CHO_INDEX]


def _onset_in(indices):
    def rule(text: str) -> list[int]:
        return [
            i for i, ch in enumerate(text)
            if is_syllable(ch) and split(ch)[0] in indices
        ]
    return rule


def _nucleus_in(indices):
    def rule(text: str) -> list[int]:
        return [
            i for i, ch in enumerate(text)
            if is_syllable(ch) and split(ch)[1] in indices
        ]
    return rule


def _onset_in_with_predecessor(indices):
    """초성이 `indices`인 음절과 **그 바로 앞 음절**.

    격음화처럼 두 음절에 걸쳐 일어나는 변환에 필요하다. 격음화는 앞 음절의 받침을 없애고
    뒤 음절의 초성을 거센소리로 만드는데, 거센소리 초성만 후보로 잡으면 받침이 빠진 앞
    음절을 방문조차 못 해 손상의 절반이 구조적으로 도달 불가가 된다(실측 46.6%).

    앞 음절을 후보에 넣는 것은 난독화된 문자열만 보고 판단할 수 있으므로 입력 전용 규칙이라는
    불변식을 깨지 않는다.
    """
    def rule(text: str) -> list[int]:
        picked: set[int] = set()
        for i, ch in enumerate(text):
            if is_syllable(ch) and split(ch)[0] in indices:
                picked.add(i)
                if i > 0 and is_syllable(text[i - 1]):
                    picked.add(i - 1)
        return sorted(picked)
    return rule


# technique -> (모르는 슬롯, 후보 위치 규칙).
# 슬롯 인덱스는 (초성, 중성, 종성) = (0, 1, 2). 나머지 슬롯은 입력에서 그대로 복사한다.
TECHNIQUE_SPEC: dict[str, tuple[tuple[int, ...], object]] = {
    "chosung": ((1, 2), _compat_jamo),                       # 초성은 주어짐
    "tensify": ((0,), _onset_in(TENSE_CHO_INDICES)),          # 사실상 이진 되돌림 판단
    "liaison": ((0, 2), _all_syllables),                      # 받침이 뒤로 넘어감
    "palatalize": ((0, 2), _all_syllables),
    "jongseong_cram": ((2,), _all_syllables),
    "identity": ((), _all_syllables),
    # --- E9 추가 ---
    "liaison_correct": ((0, 2), _all_syllables),
    "h_deletion": ((0, 2), _all_syllables),
    # 거센소리 초성이 남으므로 tensify와 같은 구조의 좁은 규칙을 쓸 수 있다.
    # 단 격음화는 두 음절에 걸치므로 앞 음절까지 후보에 넣어야 한다 — 안 그러면
    # 받침이 빠진 앞 음절이 도달 불가가 된다(실측 46.6%).
    "aspirate": ((0, 2), _onset_in_with_predecessor(PH.ASPIRATE_CHO_INDICES)),
    # 혼동 가능한 모음을 가진 음절만. 중성 슬롯을 건드리는 유일한 기법이다.
    "vowel_confuse": ((1,), _nucleus_in(PH.CONFUSABLE_JUNG_INDICES)),
    "coda_neutralize": ((2,), _all_syllables),
    "cluster_reduce": ((2,), _all_syllables),
    "nasalize": ((0, 2), _all_syllables),
    "lateralize": ((0, 2), _all_syllables),
    "tensify_rule": ((0,), _onset_in(TENSE_CHO_INDICES)),
}


def resolve_spec(technique: str) -> tuple[tuple[int, ...], object]:
    """단일 또는 조합 technique -> (모르는 슬롯, 후보 규칙).

    조합 이름(`liaison+tensify`)은 구성 기법의 **합집합**으로 푼다. 모르는 슬롯은 슬롯의
    합집합이고, 후보 위치는 각 규칙이 고른 자리의 합집합이다. 조합 난독화를 맞은 입력에서
    어느 기법이 어느 자리를 건드렸는지 추론 시점에는 알 수 없으므로 합집합이 유일하게
    정직한 선택이다.
    """
    if technique in TECHNIQUE_SPEC:
        return TECHNIQUE_SPEC[technique]
    parts = technique.split(COMPOSITE_SEP)
    unknown = [p for p in parts if p not in TECHNIQUE_SPEC]
    if len(parts) < 2 or unknown:
        raise ValueError(f"후보 규칙이 정의되지 않은 technique: {technique}")
    slots = tuple(sorted({s for p in parts for s in TECHNIQUE_SPEC[p][0]}))

    def union_rule(text: str, _parts=tuple(parts)) -> list[int]:
        seen: set[int] = set()
        for name in _parts:
            seen.update(TECHNIQUE_SPEC[name][1](text))
        return sorted(seen)

    return slots, union_rule


def composite_name(techniques) -> str:
    """조합 technique의 표준 이름. 구성 기법을 정렬해 이어 붙인다 — 순서가 달라도 같은 이름."""
    names = sorted(set(techniques))
    if not names:
        raise ValueError("technique이 비었습니다.")
    return names[0] if len(names) == 1 else COMPOSITE_SEP.join(names)


class _UnknownSlots(dict):
    """`UNKNOWN_SLOTS[technique]` 조회. 조합 이름은 합집합으로 계산해 캐시한다."""

    def __missing__(self, technique: str) -> tuple[int, ...]:
        slots = resolve_spec(technique)[0]
        self[technique] = slots
        return slots


# 기법별로 "모르는 슬롯". 나머지 슬롯은 입력에서 그대로 복사하면 된다.
UNKNOWN_SLOTS = _UnknownSlots({name: spec[0] for name, spec in TECHNIQUE_SPEC.items()})


def candidate_positions(text: str, technique: str) -> list[int]:
    """입력만 보고 고른 후보 위치. 학습과 추론이 반드시 같은 규칙을 써야 한다."""
    return resolve_spec(technique)[1](text)


@dataclass(frozen=True)
class SlotExample:
    cluster_id: str
    source: str
    technique: str
    sentence_index: int
    char_index: int
    window_chars: tuple[str, ...]
    rel_position: float
    input_slots: tuple[int, int, int]     # 입력 자모 (모르면 -1)
    target_slots: tuple[int, int, int]    # 정답 자모 (clean 음절), 정답이 없으면 (-1,-1,-1)
    changed: bool                          # 실제로 난독화가 걸린 자리인가 (평가·분석용)


def _slots_of(char: str) -> tuple[int, int, int]:
    if is_syllable(char):
        return split(char)
    if char in CHO_INDEX:
        return (CHO_INDEX[char], -1, -1)   # 초성체: 초성만 알려짐
    return (-1, -1, -1)


def extract_examples(pairs: Sequence[Pair], window: int = 4) -> list[SlotExample]:
    """페어 목록 -> 후보 위치마다 슬롯 예제 하나."""
    examples: list[SlotExample] = []
    for sent_idx, pair in enumerate(pairs):
        obf, clean = pair.obfuscated, pair.clean
        if len(obf) != len(clean):
            raise ValueError(f"길이 불일치 페어: {pair.technique} {pair.cluster_id}")
        rel = word_relative_positions(obf)
        for i in candidate_positions(obf, pair.technique):
            gold = clean[i]
            target = split(gold) if is_syllable(gold) else (-1, -1, -1)
            examples.append(
                SlotExample(
                    cluster_id=pair.cluster_id,
                    source=pair.source,
                    technique=pair.technique,
                    sentence_index=sent_idx,
                    char_index=i,
                    window_chars=tuple(
                        obf[i + d] if 0 <= i + d < len(obf) else "\x00"
                        for d in range(-window, window + 1)
                    ),
                    rel_position=rel[i],
                    input_slots=_slots_of(obf[i]),
                    target_slots=target,
                    changed=obf[i] != gold,
                )
            )
    return examples


class CharVocab:
    """문자 -> id. 학습 데이터에서만 자란다(`fit=True`), 평가에서는 미등록 문자를 UNK로."""

    def __init__(self) -> None:
        self._ids: dict[str, int] = {"\x00": PAD, "\uffff": UNK}
        self._frozen = False

    def fit(self, examples: Sequence[SlotExample]) -> "CharVocab":
        if self._frozen:
            raise RuntimeError("이미 freeze된 vocab에는 fit할 수 없습니다.")
        for ex in examples:
            for ch in ex.window_chars:
                if ch not in self._ids:
                    self._ids[ch] = len(self._ids)
        return self

    def freeze(self) -> "CharVocab":
        self._frozen = True
        return self

    def __len__(self) -> int:
        return len(self._ids)

    def encode(self, char: str) -> int:
        return self._ids.get(char, UNK)

    def encode_window(self, chars: Sequence[str]) -> list[int]:
        return [self.encode(c) for c in chars]


def char_window_matrix(examples: Sequence[SlotExample], vocab: CharVocab):
    """선형 모델용 희소 one-hot. (n, window_len * |vocab|)"""
    import numpy as np
    from scipy.sparse import csr_matrix

    if not examples:
        raise ValueError("예제가 비었습니다.")
    width = len(examples[0].window_chars)
    size = len(vocab)
    rows, cols = [], []
    for r, ex in enumerate(examples):
        for p, cid in enumerate(vocab.encode_window(ex.window_chars)):
            rows.append(r)
            cols.append(p * size + cid)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(examples), width * size))


def jamo_window_arrays(examples: Sequence[SlotExample], vocab: CharVocab):
    """신경망용 밀집 배열. (문자 id 창, 상대 위치, 입력 자모 슬롯)"""
    import numpy as np

    width = len(examples[0].window_chars)
    chars = np.zeros((len(examples), width), dtype=np.int64)
    pos = np.zeros((len(examples), 1), dtype=np.float32)
    slots = np.zeros((len(examples), 3), dtype=np.int64)
    for r, ex in enumerate(examples):
        chars[r] = vocab.encode_window(ex.window_chars)
        pos[r, 0] = ex.rel_position
        # -1(모름)은 각 슬롯 클래스 수를 sentinel 인덱스로 쓴다.
        slots[r] = [
            ex.input_slots[0] if ex.input_slots[0] >= 0 else N_CHO,
            ex.input_slots[1] if ex.input_slots[1] >= 0 else N_JUNG,
            ex.input_slots[2] if ex.input_slots[2] >= 0 else N_JONG,
        ]
    return chars, pos, slots


def targets_array(examples: Sequence[SlotExample], slot: int):
    """지정 슬롯의 정답 배열. 정답이 없는(-1) 예제는 -1로 남는다."""
    import numpy as np

    return np.array([ex.target_slots[slot] for ex in examples], dtype=np.int64)


def subsample(examples, cap: int, seed: int = 2026):
    """슬롯 예제를 결정론적으로 cap개까지 줄인다.

    E1이 재는 것은 '어휘 폭'이지 '기울기 스텝 수'가 아니다. 학습 예제 수를 고정한 채
    문장 수만 늘리면, 10만 문장에서 뽑은 30만 슬롯(넓은 어휘)과 8천 문장에서 뽑은
    30만 슬롯(좁고 반복적인 어휘)을 같은 계산량에서 비교할 수 있다.
    """
    import random as _random

    if cap <= 0 or len(examples) <= cap:
        return list(examples)
    indices = sorted(_random.Random(seed).sample(range(len(examples)), cap))
    return [examples[i] for i in indices]
