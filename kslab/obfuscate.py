"""한글 표기 난독화 생성기 — `hf_repo/ko_obfuscator.py` 확장판.

원본 5종(jamo_decompose·chosung·tensify·break_spacing·zwsp_inject)은 저장소 main과 동작이
같다. 여기에 세 가지를 더한다.

- `liaison` / `palatalize`: `experiment/disparity` 브랜치에 있던 구현을 가져오되, 그 구현이
  TODO로 남겨둔 **겹받침**을 처리하도록 고쳤다. 원본은 `JONG_TO_CHO`에 단자음만 담아
  `읽어서`·`값을`·`앉아서`·`삶은`·`닭을`이 전부 변환되지 않았다.
- `jongseong_cram`: 신규. 저장소에 구현이 아예 없었다.

규약은 원본 그대로 `(text, intensity=1.0, seed=0) -> str`. 각 변환은 진입 시 자기
`random.Random(seed)`를 새로 만들어 서로 RNG 상태를 공유하지 않는다.
"""

from __future__ import annotations

import random

from .jamo import (
    CHO_INDEX,
    COMPAT_CHO,
    COMPAT_JONG,
    COMPAT_JUNG,
    COMPOUND_JONG,
    JONG_INDEX,
    JUNG_INDEX,
    LAX_TO_TENSE,
    is_syllable,
    join,
    split,
)

ZWSP = "\u200b"

_ONSET_IEUNG = CHO_INDEX["ㅇ"]
_VOWEL_I = JUNG_INDEX["ㅣ"]

# 연음에서 뒤 음절 초성으로 넘어갈 수 있는 단자음 종성. ㅇ은 넘어가지 않는다.
_SIMPLE_JONG_TO_CHO = {
    JONG_INDEX[j]: CHO_INDEX[j]
    for j in COMPAT_JONG
    if j and j in CHO_INDEX and j != "ㅇ"
}
# 겹받침: 앞 자음은 종성으로 남고 뒤 자음이 초성으로 넘어간다. 읽어 -> 일거
_COMPOUND_JONG_TO_CHO = {
    JONG_INDEX[j]: (JONG_INDEX[front], CHO_INDEX[back])
    for j, (front, back) in COMPOUND_JONG.items()
    if back in CHO_INDEX
}
# 구개음화: 종성 ㄷ/ㅌ + '이' -> 초성 ㅈ/ㅊ
_PALATAL = {JONG_INDEX["ㄷ"]: CHO_INDEX["ㅈ"], JONG_INDEX["ㅌ"]: CHO_INDEX["ㅊ"]}

# 종성 크래밍에 쓰는 받침 풀. 실제 한국어에서 흔한 단자음 받침으로 제한해
# "그럴듯하지만 틀린" 표기를 만든다.
_CRAM_JONG = tuple(JONG_INDEX[j] for j in ("ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ"))


def _pick(text, intensity, rng):
    """intensity 비율만큼 음절 인덱스를 결정론적으로 선택."""
    idxs = [i for i, ch in enumerate(text) if is_syllable(ch)]
    k = round(len(idxs) * intensity)
    return set(rng.sample(idxs, k)) if k else set()


def _choose(sites, intensity, rng):
    """후보 위치 목록에서 intensity 비율만큼 고른다."""
    k = round(len(sites) * intensity)
    return set(rng.sample(sites, k)) if k else set()


def _syllables(text):
    """텍스트 -> 음절은 가변 [초,중,종] 리스트, 그 외 문자는 None."""
    return [list(split(ch)) if is_syllable(ch) else None for ch in text]


def _rebuild(text, syl):
    return "".join(join(*s) if s else ch for ch, s in zip(text, syl))


def jamo_decompose(text, intensity=1.0, seed=0):
    """자모분해: 안녕 -> ㅇㅏㄴㄴㅕㅇ. 길이 보존 안 됨."""
    rng = random.Random(seed)
    pick = _pick(text, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        if is_syllable(ch) and i in pick:
            c, j, t = split(ch)
            out.append(COMPAT_CHO[c] + COMPAT_JUNG[j] + COMPAT_JONG[t])
        else:
            out.append(ch)
    return "".join(out)


def chosung(text, intensity=1.0, seed=0):
    """초성체: 안녕하세요 -> ㅇㄴㅎㅅㅇ. 음절당 1문자라 길이 보존."""
    rng = random.Random(seed)
    pick = _pick(text, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        if is_syllable(ch) and i in pick:
            out.append(COMPAT_CHO[split(ch)[0]])
        else:
            out.append(ch)
    return "".join(out)


def tensify(text, intensity=1.0, seed=0):
    """된소리/쌍자음화: 시스템 -> 씨스템 (초성 평음->경음). 길이 보존."""
    rng = random.Random(seed)
    cand = [
        i for i, ch in enumerate(text)
        if is_syllable(ch) and COMPAT_CHO[split(ch)[0]] in LAX_TO_TENSE
    ]
    pick = _choose(cand, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        if i in pick:
            c, j, t = split(ch)
            out.append(join(CHO_INDEX[LAX_TO_TENSE[COMPAT_CHO[c]]], j, t))
        else:
            out.append(ch)
    return "".join(out)


def break_spacing(text, intensity=1.0, seed=0):
    """띄어쓰기 파괴. intensity>=0.5면 모든 공백 제거, 아니면 글자 사이 공백 삽입."""
    if intensity >= 0.5:
        return text.replace(" ", "")
    rng = random.Random(seed)
    pick = _pick(text, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if i in pick:
            out.append(" ")
    return "".join(out)


def zwsp_inject(text, intensity=1.0, seed=0):
    """음절 사이 zero-width space 삽입 (토크나이저 교란)."""
    rng = random.Random(seed)
    pick = _pick(text, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if i in pick:
            out.append(ZWSP)
    return "".join(out)


def liaison(text, intensity=1.0, seed=0):
    """연음: 받침을 뒤 음절 초성으로 넘겨 소리대로 표기. 먹을게 -> 머글게, 읽어 -> 일거.

    겹받침은 앞 자음을 종성에 남기고 뒤 자음만 넘긴다(disparity 원본이 빠뜨린 부분).
    음절 수가 안 변하므로 길이 보존.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = [
        i for i in range(len(syl) - 1)
        if syl[i]
        and (syl[i][2] in _SIMPLE_JONG_TO_CHO or syl[i][2] in _COMPOUND_JONG_TO_CHO)
        and syl[i + 1]
        and syl[i + 1][0] == _ONSET_IEUNG
    ]
    for i in _choose(sites, intensity, rng):
        jong = syl[i][2]
        if jong in _COMPOUND_JONG_TO_CHO:
            keep, move = _COMPOUND_JONG_TO_CHO[jong]
            syl[i][2] = keep
        else:
            move = _SIMPLE_JONG_TO_CHO[jong]
            syl[i][2] = 0
        syl[i + 1][0] = move
    return _rebuild(text, syl)


def palatalize(text, intensity=1.0, seed=0):
    """구개음화: 받침 ㄷ/ㅌ + '이' -> ㅈ/ㅊ. 굳이 -> 구지, 같이 -> 가치. 길이 보존."""
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = [
        i for i in range(len(syl) - 1)
        if syl[i] and syl[i][2] in _PALATAL
        and syl[i + 1] and syl[i + 1][0] == _ONSET_IEUNG and syl[i + 1][1] == _VOWEL_I
    ]
    for i in _choose(sites, intensity, rng):
        syl[i + 1][0] = _PALATAL[syl[i][2]]
        syl[i][2] = 0
    return _rebuild(text, syl)


def jongseong_cram(text, intensity=1.0, seed=0):
    """종성 크래밍: 받침에 불필요한 자음을 채워 넣는다. 안녕 -> 안녕ㄱ 이 아니라 안녕->앉녕 식.

    받침 없는 음절에는 받침을 새로 넣고, 받침 있는 음절은 다른 단자음 받침으로 바꾼다.
    어느 쪽이든 음절 하나는 음절 하나로 남으므로 길이가 보존된다. 결과물이 문법적으로는
    정상인 한글 음절이라, 초성체의 호환 자모 같은 값싼 탐지 신호가 없는 것이 이 기법의 요점이다.
    """
    rng = random.Random(seed)
    cand = [i for i, ch in enumerate(text) if is_syllable(ch)]
    pick = _choose(cand, intensity, rng)
    out = []
    for i, ch in enumerate(text):
        if i not in pick:
            out.append(ch)
            continue
        c, j, t = split(ch)
        options = [x for x in _CRAM_JONG if x != t]
        out.append(join(c, j, rng.choice(options)))
    return "".join(out)


TRANSFORMS = {
    "jamo_decompose": jamo_decompose,
    "chosung": chosung,
    "tensify": tensify,
    "break_spacing": break_spacing,
    "zwsp_inject": zwsp_inject,
    "liaison": liaison,
    "palatalize": palatalize,
    "jongseong_cram": jongseong_cram,
}

# 'phonetic' = 정규화로 되돌릴 대상이 없는 음운 변형, 'visual' = 정규화-해결 계열 대조군.
# 저장소의 experiments/benchmark/run_normalizer_evaluation.py:59 FAMILY_MAP 과 같은 축이다.
FAMILY = {
    "jamo_decompose": "visual",
    "chosung": "visual",
    "break_spacing": "visual",
    "zwsp_inject": "visual",
    "tensify": "phonetic",
    "liaison": "phonetic",
    "palatalize": "phonetic",
    "jongseong_cram": "phonetic",
}

# 입력과 출력의 문자 수가 항상 같은 변환. 위치별 슬롯 분류가 성립하는 대상이다.
LENGTH_PRESERVING = frozenset(
    {"chosung", "tensify", "liaison", "palatalize", "jongseong_cram"}
)
