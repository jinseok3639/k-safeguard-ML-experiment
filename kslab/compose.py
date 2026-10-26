"""다중(조합) 난독화 생성.

기존 `pairs.make_pairs`는 **항상 clean 텍스트에** 변환 하나만 건다. 실제 공격자는 그럴
이유가 없다 — 연음을 걸고 그 위에 된소리를 얹을 수 있다. 이 모듈은 변환을 순서대로 겹쳐
거는 생성기와, 그 결과를 공정하게 비교하기 위한 장치를 제공한다.

## 왜 그냥 겹쳐 걸면 안 되는가

1. **깊이와 손상량이 뒤섞인다.** 각 단계를 intensity 1.0으로 걸면 깊이 3은 깊이 1보다 훨씬
   많이 망가진 문자열이 된다. 그때 모델이 못 맞히는 건 "조합이라서"가 아니라 "더 망가져서"다.
   `calibrate` 가 단계별 강도를 낮춰 **손상량을 고정한 채 깊이만 바꾸는** 비교를 만든다.
2. **시드가 상관된다.** 모든 변환이 진입 시 `random.Random(seed)`를 새로 만들기 때문에
   같은 seed를 단계마다 그대로 주면 후보 위치 목록이 겹칠 때 같은 자리를 고른다.
   `stage_seed` 가 단계마다 다른 시드를 만든다.
3. **변경 자리의 출처를 잃는다.** `SlotExample.changed`는 "원문과 다른가"만 알려주므로
   조합에서는 어느 단계가 그 자리를 건드렸는지 알 수 없다. `provenance` 가 자리마다
   **마지막으로 쓴 단계**를 기록한다.

## 조합해도 의미 없는 짝 (실측 확인)

- `chosung`은 뒤에 무엇을 걸어도 흡수한다. 음절이 호환 자모 낱자로 바뀌어 이후 변환의
  후보 자리가 사라진다 — 말뭉치 2000문장에서 `chosung∘jongseong_cram == chosung` 2000/2000.
- `tensify`와 `jongseong_cram`은 교환법칙이 성립한다(2000/2000 동일). 초성만 보는 규칙과
  종성만 쓰는 규칙이라 서로의 후보 집합을 바꾸지 않는다. 두 순서를 다 재는 건 낭비다.
- `palatalize`는 문장 0.9%에서만 발동해 사실상 죽어 있다.

그래서 조합 대상에서 `chosung`을 빼고, `tensify·jongseong_cram` 순서쌍은 하나만 둔다.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

from . import obfuscate

# 표준발음법 도출 순서에 맞춘 정준 순서.
#
# 연음·ㅎ탈락이 먼저다(뒤에 모음이 오면 받침이 넘어가 이후 받침 규칙의 입력이 사라진다).
# 그다음 ㅎ 축약, 받침 정리(자음군 단순화 -> 평파열음화), 동화(비음화·유음화), 경음화 순.
# 음운 규칙이 아닌 잡음(모음 혼동, 종성 크래밍)은 맨 뒤에 얹는다.
CANONICAL_ORDER = (
    "liaison_correct", "liaison", "h_deletion", "palatalize", "aspirate",
    "cluster_reduce", "coda_neutralize", "nasalize", "lateralize",
    "tensify_rule", "tensify", "vowel_confuse", "jongseong_cram",
)

# 조합에 쓰면 안 되는 기법.
NOT_COMPOSABLE = frozenset({
    "chosung",                                   # 뒤 단계를 전부 흡수한다
    "jamo_decompose", "break_spacing", "zwsp_inject",   # 길이 보존이 아니다
})


@dataclass(frozen=True)
class Profile:
    """이름 붙인 조합 공격. `stages` 순서대로 겹쳐 건다."""

    name: str
    stages: tuple[str, ...]

    @property
    def depth(self) -> int:
        return len(self.stages)

    def technique(self) -> str:
        """이 프로파일을 맞은 입력을 다룰 모델의 technique 이름(구성 기법의 합집합)."""
        from .features import composite_name

        return composite_name(self.stages)


@dataclass(frozen=True)
class Composed:
    """조합 난독화 결과 한 건."""

    text: str
    provenance: tuple[str, ...]      # 문자마다 마지막으로 쓴 단계 이름 ("" = 손대지 않음)
    stage_intensity: float           # 각 단계에 실제로 넣은 강도

    @property
    def touched(self) -> int:
        return sum(1 for p in self.provenance if p)


def canonical_sort(stages: Sequence[str]) -> tuple[str, ...]:
    """정준 순서로 정렬. 목록에 없는 이름은 뒤로 밀되 원래 순서를 지킨다."""
    rank = {name: i for i, name in enumerate(CANONICAL_ORDER)}
    return tuple(sorted(stages, key=lambda s: (rank.get(s, len(rank)), s)))


def stage_seed(gen_seed: int, stage_index: int, name: str) -> int:
    """단계마다 다른 결정론적 시드.

    `hash()`는 프로세스마다 달라지므로(PYTHONHASHSEED) 쓰지 않는다. crc32는 안정적이다.
    """
    return (gen_seed * 1_000_003 + stage_index * 7919 + zlib.crc32(name.encode())) % (2**31)


def validate_stages(stages: Sequence[str]) -> None:
    unknown = [s for s in stages if s not in obfuscate.TRANSFORMS]
    if unknown:
        raise ValueError(f"알 수 없는 technique: {unknown}")
    blocked = [s for s in stages if s in NOT_COMPOSABLE]
    if blocked:
        raise ValueError(
            f"조합에 쓸 수 없는 technique: {blocked} "
            "(chosung은 뒤 단계를 흡수하고, 나머지는 길이 보존이 아니다)"
        )
    if not stages:
        raise ValueError("stages가 비었습니다.")


def apply_stages(
    text: str, stages: Sequence[str], intensity: float, gen_seed: int
) -> Composed:
    """변환을 순서대로 겹쳐 건다. 자리마다 마지막으로 쓴 단계를 함께 돌려준다."""
    validate_stages(stages)
    current = text
    provenance = [""] * len(text)
    for index, name in enumerate(stages):
        fn = obfuscate.TRANSFORMS[name]
        nxt = fn(current, intensity, seed=stage_seed(gen_seed, index, name))
        if len(nxt) != len(current):
            raise ValueError(f"{name}이 길이를 바꿨습니다: {current!r} -> {nxt!r}")
        for i, (a, b) in enumerate(zip(current, nxt)):
            if a != b:
                provenance[i] = name
        current = nxt
    return Composed(current, tuple(provenance), intensity)


def damage_rate(clean: str, obfuscated: str) -> float:
    """문자 단위 손상률. 길이 보존이므로 단순 불일치 비율이다."""
    if not clean:
        return 0.0
    return sum(1 for a, b in zip(clean, obfuscated) if a != b) / len(clean)


def measure_damage(
    texts: Sequence[str], stages: Sequence[str], intensity: float, gen_seed: int
) -> float:
    """말뭉치 전체에서 실제로 난 손상률(문자 기준)."""
    changed = total = 0
    for text in texts:
        out = apply_stages(text, stages, intensity, gen_seed).text
        changed += sum(1 for a, b in zip(text, out) if a != b)
        total += len(text)
    return changed / total if total else 0.0


def calibrate(
    texts: Sequence[str],
    stages: Sequence[str],
    budget: float,
    gen_seed: int = 1234,
    tolerance: float = 0.002,
    max_steps: int = 18,
) -> tuple[float, float]:
    """모든 단계에 공통으로 넣을 강도를 이분 탐색해 손상률을 `budget`에 맞춘다.

    반환 `(강도, 실측 손상률)`. 단계들이 서로 간섭하기 때문에(연음이 비운 받침을 종성
    크래밍이 다시 채우는 식) 해석적으로 계산하지 않고 말뭉치에서 직접 잰다.

    강도 1.0으로도 budget에 못 미치면 1.0을 돌려준다 — 그 조합의 손상 상한이다.
    깊이별 비교를 할 때는 반드시 실측 손상률을 함께 보고해야 한다. 두 조건의 실측값이
    다르면 그 비교는 '깊이 비교'가 아니라 '손상량 비교'가 되어버린다.
    """
    validate_stages(stages)
    ceiling = measure_damage(texts, stages, 1.0, gen_seed)
    if ceiling <= budget + tolerance:
        return 1.0, ceiling
    low, high = 0.0, 1.0
    realized = ceiling
    for _ in range(max_steps):
        mid = (low + high) / 2
        realized = measure_damage(texts, stages, mid, gen_seed)
        if abs(realized - budget) <= tolerance:
            return mid, realized
        if realized > budget:
            high = mid
        else:
            low = mid
    return (low + high) / 2, realized


# --- 프로파일 --------------------------------------------------------------
#
# 균등 랜덤 부분집합이 아니라 **이름 붙인 프로파일**을 쓴다. 조합 가능한 기법이 몇 개뿐이라
# 랜덤 표본은 얻는 게 없고, 배치가 정체불명의 혼합이 되어 귀속(attribution)을 잃는다.
# 프로파일은 그 자체로 "이런 공격이 가능하다"는 레드팀 산출물이기도 하다.

_T, _L, _C = "tensify", "liaison", "jongseong_cram"
_V, _A = "vowel_confuse", "aspirate"

PROFILES: tuple[Profile, ...] = (
    # 깊이 1 — 단일 난독화. 기존 E0~E8과 직접 비교되는 기준선.
    Profile("d1_tensify", (_T,)),
    Profile("d1_liaison", (_L,)),
    Profile("d1_cram", (_C,)),
    Profile("d1_vowel", (_V,)),
    Profile("d1_aspirate", (_A,)),
    # 깊이 2 — 순서가 의미 있는 짝만. tensify·cram은 교환법칙이 성립해 한 순서만 둔다.
    Profile("d2_liaison_then_tensify", (_L, _T)),   # 예: 연음 후 된소리
    Profile("d2_tensify_then_liaison", (_T, _L)),
    Profile("d2_liaison_then_cram", (_L, _C)),
    Profile("d2_cram_then_liaison", (_C, _L)),
    Profile("d2_tensify_cram", (_T, _C)),           # 교환법칙 성립 — 한 순서만
    Profile("d2_liaison_then_vowel", (_L, _V)),
    Profile("d2_aspirate_then_liaison", (_A, _L)),
    # 깊이 3 — 정준 순서와 그 역순.
    Profile("d3_canonical", (_T, _L, _C)),
    Profile("d3_reversed", (_C, _L, _T)),
    Profile("d3_with_vowel", (_L, _V, _C)),
    # 깊이 5 — 예: 연음 + 구개음화 + ㅎ결합 후 된소리 + 종성 크래밍.
    Profile("d5_full_attack", ("liaison_correct", "palatalize", _A, _T, _C)),
    # 음운 규칙만으로 쌓은 정준 도출. 결과가 실제 표준 발음 표기에 가깝다.
    # 옷고름 -> 옫꼬름, 꽃다발 -> 꼳따발, 읽는 -> 잉는.
    Profile(
        "d5_phonological",
        ("cluster_reduce", "coda_neutralize", "nasalize", "lateralize", "tensify_rule"),
    ),
)

PROFILES_BY_NAME = {p.name: p for p in PROFILES}


def profiles_of_depth(depth: int) -> tuple[Profile, ...]:
    return tuple(p for p in PROFILES if p.depth == depth)
