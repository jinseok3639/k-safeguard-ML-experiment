"""슬롯 예제 추출과 특징 인코딩.

가장 중요한 규칙: **후보 위치는 입력만 보고 정해야 한다.** 학습할 때 "clean과 다른 위치"를
후보로 쓰면 추론 때 알 수 없는 정보를 쓰는 셈이고, clean mutation rate가 실제보다 좋게 나온다.
그래서 `candidate_positions()`는 난독화된 문자열만 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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

# 기법별로 "모르는 슬롯". 나머지 슬롯은 입력에서 그대로 복사하면 된다.
UNKNOWN_SLOTS = {
    "chosung": (1, 2),          # 초성은 주어짐, 중성·종성을 예측
    "tensify": (0,),            # 초성만 (사실상 이진 되돌림 판단)
    "liaison": (0, 2),          # 받침이 뒤로 넘어가 초성·종성이 흔들림
    "palatalize": (0, 2),
    "jongseong_cram": (2,),     # 종성만
    "identity": (),
}


def candidate_positions(text: str, technique: str) -> list[int]:
    """입력만 보고 고른 후보 위치. 학습과 추론이 반드시 같은 규칙을 써야 한다."""
    if technique == "chosung":
        # 초성체는 완성형 음절이 호환 초성 낱자 하나로 바뀐다 — 눈에 보이는 값싼 신호.
        return [i for i, ch in enumerate(text) if ch in CHO_INDEX]
    if technique == "tensify":
        # 경음 초성을 가진 음절만. 정상 된소리 단어도 여기 걸리는 게 문제의 핵심이다.
        return [
            i for i, ch in enumerate(text)
            if is_syllable(ch) and split(ch)[0] in TENSE_CHO_INDICES
        ]
    if technique in ("liaison", "palatalize", "jongseong_cram", "identity"):
        # 결과물이 문법적으로 정상인 한글 음절이라 값싼 신호가 없다 -> 전수 검사.
        # 이 비용을 줄이는 게 E6 perplexity 게이트의 존재 이유다.
        return [i for i, ch in enumerate(text) if is_syllable(ch)]
    raise ValueError(f"후보 규칙이 정의되지 않은 technique: {technique}")


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
