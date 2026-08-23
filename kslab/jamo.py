"""완성형 한글 음절과 자모 슬롯(초성·중성·종성) 사이의 무손실 변환.

테이블은 `k-safeguard/src/k_safeguard/normalization.py:18-34`의 호환 자모 상수와 동일하다.
저장소를 import하지 않고 값을 복제한 이유는 이 샌드박스가 저장소에 의존하지 않아야 하기
때문이다 — 값이 갈라지면 `tests/test_jamo.py`가 저장소 원본과 대조해서 잡는다.
"""

from __future__ import annotations

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
JUNG_COUNT = 21
JONG_COUNT = 28
SYLLABLES_PER_INITIAL = JUNG_COUNT * JONG_COUNT  # 588

COMPAT_CHO = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
COMPAT_JUNG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
COMPAT_JONG = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

CHO_INDEX = {char: index for index, char in enumerate(COMPAT_CHO)}
JUNG_INDEX = {char: index for index, char in enumerate(COMPAT_JUNG)}
JONG_INDEX = {char: index for index, char in enumerate(COMPAT_JONG) if char}

# 슬롯 클래스 수. Dacon 1위가 쓴 68(19+21+28)과 같은 크기이나, 종성 0번을 "받침 없음"
# 클래스로 그대로 쓰므로 별도 sentinel 문자가 필요 없다.
N_CHO = len(COMPAT_CHO)      # 19
N_JUNG = len(COMPAT_JUNG)    # 21
N_JONG = len(COMPAT_JONG)    # 28
N_SLOT_CLASSES = N_CHO + N_JUNG + N_JONG  # 68

# 겹받침 -> (앞 자음, 뒤로 넘어갈 자음). 연음에서 뒤 음절 초성으로 이월되는 것은 뒤쪽이다.
# 예: 읽어 -> 일거 (ㄺ 에서 ㄹ 은 남고 ㄱ 이 넘어간다)
COMPOUND_JONG = {
    "ㄳ": ("ㄱ", "ㅅ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅅ"),
}

TENSE_TO_LAX = {"ㄲ": "ㄱ", "ㄸ": "ㄷ", "ㅃ": "ㅂ", "ㅆ": "ㅅ", "ㅉ": "ㅈ"}
LAX_TO_TENSE = {lax: tense for tense, lax in TENSE_TO_LAX.items()}
TENSE_CHO_INDICES = frozenset(CHO_INDEX[c] for c in TENSE_TO_LAX)


def is_syllable(char: str) -> bool:
    """완성형 한글 음절인지."""
    return len(char) == 1 and HANGUL_BASE <= ord(char) <= HANGUL_END


def split(char: str) -> tuple[int, int, int]:
    """완성형 음절 -> (초성, 중성, 종성) 인덱스. 종성 0은 받침 없음."""
    code = ord(char) - HANGUL_BASE
    return code // SYLLABLES_PER_INITIAL, (code % SYLLABLES_PER_INITIAL) // JONG_COUNT, code % JONG_COUNT


def join(cho: int, jung: int, jong: int = 0) -> str:
    """(초성, 중성, 종성) 인덱스 -> 완성형 음절."""
    if not (0 <= cho < N_CHO and 0 <= jung < N_JUNG and 0 <= jong < N_JONG):
        raise ValueError(f"자모 인덱스 범위를 벗어났습니다: ({cho}, {jung}, {jong})")
    return chr(HANGUL_BASE + cho * SYLLABLES_PER_INITIAL + jung * JONG_COUNT + jong)


def decompose(text: str) -> list[tuple[int, int, int] | None]:
    """문자열 -> 음절마다 [초,중,종] 인덱스, 음절이 아니면 None. 길이는 원문과 같다."""
    return [split(ch) if is_syllable(ch) else None for ch in text]


def compose(text: str, slots: list[tuple[int, int, int] | None]) -> str:
    """`decompose`의 역. 음절이 아니던 자리는 원문 문자를 그대로 되돌린다."""
    if len(text) != len(slots):
        raise ValueError("slots 길이가 원문과 다릅니다.")
    return "".join(join(*s) if s is not None else ch for ch, s in zip(text, slots))


def syllable_positions(text: str) -> list[int]:
    """완성형 음절이 있는 인덱스 목록."""
    return [i for i, ch in enumerate(text) if is_syllable(ch)]


def word_relative_positions(text: str) -> list[float]:
    """각 문자의 '어절 내 상대 위치'(0<v<=1). 공백은 0.0.

    Dacon 1위 1단계가 유일한 문맥 신호로 쓴 특징이다. 어절 경계는 공백 기준.
    """
    out = [0.0] * len(text)
    start = None
    for i, ch in enumerate(text + " "):
        if ch.isspace() or i == len(text):
            if start is not None:
                span = i - start
                for k in range(start, i):
                    out[k] = round((k - start + 1) / span, 4)
                start = None
        elif start is None:
            start = i
    return out
