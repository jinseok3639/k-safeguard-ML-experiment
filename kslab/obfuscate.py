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

from . import phonology as PH
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


def _adjacent_sites(syl, predicate):
    """앞뒤 음절 쌍에 predicate(앞, 뒤)가 참인 앞 음절 인덱스 목록."""
    return [
        i for i in range(len(syl) - 1)
        if syl[i] and syl[i + 1] and predicate(syl[i], syl[i + 1])
    ]


def _unary_sites(syl, predicate):
    """음절 하나만 보고 predicate(음절)이 참인 인덱스 목록."""
    return [i for i, s in enumerate(syl) if s and predicate(s)]


def liaison_correct(text, intensity=1.0, seed=0):
    """표준발음법을 지키는 연음. `liaison`과 달리 받침 ㅎ을 뒤로 넘기지 않고 탈락시킨다.

    좋아 -> 조아, 많아 -> 마나, 싫어 -> 시러 (제12항 4). 나머지 받침은 `liaison`과 같다.

    `liaison`을 고치지 않고 따로 둔 이유: `liaison`이 만드는 조하·실허 같은 표기도 실제
    오타로 흔히 나타나므로 그 자체로 유효한 공격이고, 무엇보다 E1~E8의 기준선이 그 동작
    위에서 측정됐다. 둘을 나란히 두면 "표준 발음형"과 "오타형"을 각각 다룰 수 있다.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _adjacent_sites(
        syl,
        lambda a, b: b[0] == _ONSET_IEUNG and (
            a[2] in PH.H_CODA_BEFORE_VOWEL_IDX
            or a[2] in _SIMPLE_JONG_TO_CHO
            or a[2] in _COMPOUND_JONG_TO_CHO
        ),
    )
    for i in sorted(_choose(sites, intensity, rng)):
        jong = syl[i][2]
        if jong in PH.H_CODA_BEFORE_VOWEL_IDX:
            keep, move = PH.H_CODA_BEFORE_VOWEL_IDX[jong]
            syl[i][2] = keep
            if move is not None:
                syl[i + 1][0] = move
        elif jong in _COMPOUND_JONG_TO_CHO:
            keep, move = _COMPOUND_JONG_TO_CHO[jong]
            syl[i][2] = keep
            syl[i + 1][0] = move
        else:
            syl[i + 1][0] = _SIMPLE_JONG_TO_CHO[jong]
            syl[i][2] = 0
    return _rebuild(text, syl)


def h_deletion(text, intensity=1.0, seed=0):
    """ㅎ 탈락만 따로. 종성 ㅎ/ㄶ/ㅀ + 모음 -> ㅎ이 사라진다 (제12항 4).

    좋아 -> 조아, 많아 -> 마나, 싫어 -> 시러, 낳은 -> 나은. 길이 보존.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _adjacent_sites(
        syl, lambda a, b: a[2] in PH.H_CODA_BEFORE_VOWEL_IDX and b[0] == _ONSET_IEUNG
    )
    for i in sorted(_choose(sites, intensity, rng)):
        keep, move = PH.H_CODA_BEFORE_VOWEL_IDX[syl[i][2]]
        syl[i][2] = keep
        if move is not None:
            syl[i + 1][0] = move
    return _rebuild(text, syl)


def aspirate(text, intensity=1.0, seed=0):
    """격음화(유기음화). ㅎ과 평음이 만나 거센소리로 축약된다 (제12항 1). 길이 보존.

    양방향이다.
    - 종성 ㅎ계열 + 초성 평음: 놓고 -> 노코, 좋다 -> 조타, 많지 -> 만치, 닳지 -> 달치
    - 종성 평음 + 초성 ㅎ:   국화 -> 구콰, 맏형 -> 마텽, 좁히다 -> 조피다, 밝히다 -> 발키다

    결과물에 거센소리 초성이 남으므로 `tensify`의 경음 초성처럼 값싼 후보 규칙을 쓸 수 있다.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    cho_h = CHO_INDEX["ㅎ"]
    sites = _adjacent_sites(
        syl,
        lambda a, b: (
            (a[2] in PH.H_CODA_RESIDUE_IDX and b[0] in PH.H_FOLLOWING_ONSET_IDX)
            or (a[2] in PH.CODA_PLUS_H_IDX and b[0] == cho_h)
        ),
    )
    for i in sorted(_choose(sites, intensity, rng)):
        jong, nxt = syl[i][2], syl[i + 1][0]
        if jong in PH.H_CODA_RESIDUE_IDX and nxt in PH.H_FOLLOWING_ONSET_IDX:
            syl[i][2] = PH.H_CODA_RESIDUE_IDX[jong]
            syl[i + 1][0] = PH.H_FOLLOWING_ONSET_IDX[nxt]
        else:
            keep, asp = PH.CODA_PLUS_H_IDX[jong]
            syl[i][2] = keep
            syl[i + 1][0] = asp
    return _rebuild(text, syl)


def vowel_confuse(text, intensity=1.0, seed=0):
    """모음 혼동 표기. ㅐ/ㅔ, ㅒ/ㅖ, ㅚ/ㅙ/ㅞ, ㅢ/ㅣ/ㅔ 를 서로 바꾼다. 길이 보존.

    표준발음법 조항이 아니라 실제 화자가 구분 못 하거나 자주 틀리는 **표기 혼동**이다.
    재미있게도 이게 길이 보존 기법 중 **중성 슬롯을 건드리는 유일한 기법**이다 — 기존
    5종은 전부 초성·종성만 바꾸므로 중성 헤드가 학습 신호를 못 받는다.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _unary_sites(syl, lambda s: s[1] in PH.CONFUSABLE_JUNG_INDICES)
    for i in sorted(_choose(sites, intensity, rng)):
        syl[i][1] = rng.choice(PH.VOWEL_CONFUSION_IDX[syl[i][1]])
    return _rebuild(text, syl)


def coda_neutralize(text, intensity=1.0, seed=0):
    """평파열음화(음절의 끝소리 규칙). 종성을 7종성으로 중화한다 (제8·9항). 길이 보존.

    부엌 -> 부억, 옷 -> 옫, 빛 -> 빋, 앞 -> 압.
    ㅅㅆㅈㅊㅌㅎ이 전부 ㄷ으로 가는 **다대일** 사상이라 원리적으로 되돌릴 수 없는 정보가
    생긴다 — 복원 실험에서는 상한(ceiling)을 반드시 함께 봐야 하는 기법이다.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _unary_sites(syl, lambda s: s[2] in PH.CODA_NEUTRALIZE_IDX)
    for i in sorted(_choose(sites, intensity, rng)):
        syl[i][2] = PH.CODA_NEUTRALIZE_IDX[syl[i][2]]
    return _rebuild(text, syl)


def cluster_reduce(text, intensity=1.0, seed=0):
    """자음군 단순화. 겹받침 둘 중 하나만 남긴다 (제10·11항). 길이 보존.

    값 -> 갑, 넋 -> 넉, 닭 -> 닥, 삶 -> 삼, 여덟 -> 여덜.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _unary_sites(syl, lambda s: s[2] in PH.CLUSTER_REDUCE_IDX)
    for i in sorted(_choose(sites, intensity, rng)):
        syl[i][2] = PH.CLUSTER_REDUCE_IDX[syl[i][2]]
    return _rebuild(text, syl)


def nasalize(text, intensity=1.0, seed=0):
    """비음화 (제18·19항). 길이 보존.

    - 파열음의 비음화: 국물 -> 궁물, 닫는 -> 단는, 밥물 -> 밤물
    - ㄹ의 비음화:     종로 -> 종노, 강릉 -> 강능, 담력 -> 담녁
    - 상호 비음화:     백리 -> 뱅니, 섭리 -> 섬니
    """
    rng = random.Random(seed)
    syl = _syllables(text)

    def is_site(a, b):
        if a[2] in PH.CODA_TO_NASAL_IDX and b[0] in PH.NASAL_ONSET_INDICES:
            return True                                   # 파열음의 비음화
        if b[0] == PH.CHO_R:
            return (
                a[2] in PH.NASAL_CODA_BEFORE_R_INDICES     # ㄹ의 비음화
                or a[2] in PH.CODA_TO_NASAL_IDX            # 상호 비음화
            )
        return False

    for i in sorted(_choose(_adjacent_sites(syl, is_site), intensity, rng)):
        jong, nxt = syl[i][2], syl[i + 1][0]
        if nxt == PH.CHO_R:
            syl[i + 1][0] = PH.CHO_N
            if jong in PH.CODA_TO_NASAL_IDX:
                syl[i][2] = PH.CODA_TO_NASAL_IDX[jong]
        else:
            syl[i][2] = PH.CODA_TO_NASAL_IDX[jong]
    return _rebuild(text, syl)


def lateralize(text, intensity=1.0, seed=0):
    """유음화 (제20항). ㄴ과 ㄹ이 만나면 ㄹㄹ이 된다. 길이 보존.

    신라 -> 실라, 난로 -> 날로, 칼날 -> 칼랄, 설날 -> 설랄, 권력 -> 궐력.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _adjacent_sites(
        syl,
        lambda a, b: (a[2] == PH.JONG_N and b[0] == PH.CHO_R)
        or (a[2] == PH.JONG_R and b[0] == PH.CHO_N),
    )
    for i in sorted(_choose(sites, intensity, rng)):
        if syl[i][2] == PH.JONG_N:
            syl[i][2] = PH.JONG_R          # 신라 -> 실라
        else:
            syl[i + 1][0] = PH.CHO_R       # 칼날 -> 칼랄
    return _rebuild(text, syl)


def tensify_rule(text, intensity=1.0, seed=0):
    """규칙 경음화 (제23~28항). 장애음 종성 뒤 평음 초성만 경음이 된다. 길이 보존.

    국밥 -> 국빱, 학교 -> 학꾜, 먹다 -> 먹따, 옷고름 -> 옫꼬름.

    기존 `tensify`가 문맥과 무관하게 아무 평음 초성이나 바꾸는 것과 대비된다 — 이쪽은
    실제 한국어 화자가 그렇게 **발음하는** 자리만 골라서 표기에 반영한다.
    """
    rng = random.Random(seed)
    syl = _syllables(text)
    sites = _adjacent_sites(
        syl,
        lambda a, b: a[2] in PH.OBSTRUENT_CODA_INDICES
        and b[0] in PH.TENSIFIABLE_ONSET_INDICES,
    )
    for i in sorted(_choose(sites, intensity, rng)):
        syl[i + 1][0] = PH.LAX_TO_TENSE_ONSET_IDX[syl[i + 1][0]]
    return _rebuild(text, syl)


TRANSFORMS = {
    "jamo_decompose": jamo_decompose,
    "chosung": chosung,
    "tensify": tensify,
    "break_spacing": break_spacing,
    "zwsp_inject": zwsp_inject,
    "liaison": liaison,
    "palatalize": palatalize,
    "jongseong_cram": jongseong_cram,
    # --- E9 추가: 표준발음법 규칙 + 표기 혼동 ---
    "liaison_correct": liaison_correct,
    "h_deletion": h_deletion,
    "aspirate": aspirate,
    "vowel_confuse": vowel_confuse,
    "coda_neutralize": coda_neutralize,
    "cluster_reduce": cluster_reduce,
    "nasalize": nasalize,
    "lateralize": lateralize,
    "tensify_rule": tensify_rule,
}

# 'phonetic' = 정규화로 되돌릴 대상이 없는 음운 변형, 'visual' = 정규화-해결 계열 대조군.
# k-safeguard f6ee374의 experiments/benchmark/run_normalizer_evaluation.py:59 FAMILY_MAP 과 같은 축이다.
FAMILY = {
    "jamo_decompose": "visual",
    "chosung": "visual",
    "break_spacing": "visual",
    "zwsp_inject": "visual",
    "tensify": "phonetic",
    "liaison": "phonetic",
    "palatalize": "phonetic",
    "jongseong_cram": "phonetic",
    "liaison_correct": "phonetic",
    "h_deletion": "phonetic",
    "aspirate": "phonetic",
    "vowel_confuse": "phonetic",
    "coda_neutralize": "phonetic",
    "cluster_reduce": "phonetic",
    "nasalize": "phonetic",
    "lateralize": "phonetic",
    "tensify_rule": "phonetic",
}

# 입력과 출력의 문자 수가 항상 같은 변환. 위치별 슬롯 분류가 성립하는 대상이다.
LENGTH_PRESERVING = frozenset(
    {
        "chosung", "tensify", "liaison", "palatalize", "jongseong_cram",
        "liaison_correct", "h_deletion", "aspirate", "vowel_confuse",
        "coda_neutralize", "cluster_reduce", "nasalize", "lateralize",
        "tensify_rule",
    }
)

# 표준발음법 조항 근거가 있는 기법 -> 조항 번호. 보고서에서 근거를 밝히는 데 쓴다.
# vowel_confuse는 조항이 아니라 표기 혼동이므로 여기 없다.
STANDARD_PRONUNCIATION_CLAUSE = {
    "liaison": "제13·14항",
    "liaison_correct": "제13·14항 + 제12항 4",
    "h_deletion": "제12항 4",
    "aspirate": "제12항 1",
    "palatalize": "제17항",
    "coda_neutralize": "제8·9항",
    "cluster_reduce": "제10·11항",
    "nasalize": "제18·19항",
    "lateralize": "제20항",
    "tensify_rule": "제23~28항",
}

# 복원 모델을 붙이는 기법. 나머지는 생성기(레드팀 데이터)로만 쓴다.
#
# 기준은 **신호 밀도** = 실제 변형된 자리 / 모델이 방문하는 후보 자리. 밀도가 낮으면
# 모델이 identity를 학습하고 정확도 99.9%를 보고한다 — palatalize(문장 0.9% 발동)가
# 어느 실험에도 못 들어간 이유다.
MODELED = frozenset(
    {
        "chosung", "tensify", "liaison", "jongseong_cram",   # 기존
        "vowel_confuse",   # 중성 슬롯을 건드리는 유일한 기법
        "aspirate",        # 거센소리 초성이라는 값싼 후보 규칙이 있어 밀도 9.2% 확보
    }
)
