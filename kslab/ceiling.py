"""복원 상한(ceiling) — 학습하지 않고 계산하는 성능 천장.

모델을 하나도 학습시키지 않고 "이 모델 구조로 이 입력을 얼마나 고칠 수 있는가"의 **상한**을
구한다. 조합 난독화 실험을 시작하기 전에 이걸 먼저 돌리는 이유는, 상한이 승격 기준에 못
미치면 어떤 학습도 무의미하기 때문이다. 몇 분이면 실험 전체를 접을 수 있다.

## 상한이 존재하는 이유 — 구조적 제약 두 겹

복원기는 (`models/linear.py:83-91`, `mlp.py:210-212`, `bilstm.py:211-216`) 예외 없이
다음 두 단계를 밟는다.

1. `candidate_positions(obf, technique)`가 고른 자리만 **방문**한다. 나머지 자리는
   쳐다보지도 않는다.
2. 방문한 자리에서 `UNKNOWN_SLOTS[technique]`에 든 슬롯만 **예측**하고, 나머지 슬롯은
   입력값을 그대로 복사한다.

그래서 어떤 자리는 모델이 아무리 똑똑해도 못 고친다.

- **도달 불가(unreachable)**: 손상됐는데 후보 규칙이 그 자리를 안 고른다.
  예) `tensify` 모델은 경음 초성만 보므로, 종성 크래밍이 망친 평음 음절은 방문조차 안 한다.
- **표현 불가(inexpressible)**: 방문은 하는데 손상이 모델의 담당 슬롯 **밖**에 있다.
  예) `tensify` 모델의 담당은 초성뿐이라 종성이 틀린 자리는 종성을 입력 그대로 복사해
  다시 써넣는다.

이 둘은 확률 문제가 아니라 구조 문제다 — 학습 데이터를 늘려도, 아키텍처를 키워도, 임계값을
낮춰도 변하지 않는다. 단일 기법 모델에 조합 입력을 주면 이 손실이 얼마나 되는지가
"조합을 따로 다뤄야 하는가"에 대한 가장 값싼 답이다.

## 주의 — 이건 낙관적 상한이다

계산에서 모델은 **담당 슬롯을 항상 정답으로 맞힌다**고 가정한다. 실제 정확도는 이보다
낮다. 따라서 상한이 기준에 미달하면 확실히 실패고, 상한이 기준을 넘어도 통과가 보장되지는
않는다. 한쪽 방향으로만 결론을 낼 수 있는 지표다.
"""

from __future__ import annotations

from typing import Sequence

from . import features as F
from .jamo import is_syllable, split


def _slot_values(char: str) -> tuple[int, int, int] | None:
    return split(char) if is_syllable(char) else None


def classify_sites(obfuscated: str, clean: str, technique: str) -> dict[str, list[int]]:
    """손상된 자리를 '고칠 수 있는가' 기준으로 가른다.

    반환 키: `fixable`(방문하고 담당 슬롯 안), `unreachable`(후보 규칙 밖),
    `inexpressible`(방문하지만 담당 슬롯 밖).
    """
    if len(obfuscated) != len(clean):
        raise ValueError("길이가 다른 페어에는 상한을 계산할 수 없습니다.")
    unknown = set(F.UNKNOWN_SLOTS[technique])
    visited = set(F.candidate_positions(obfuscated, technique))
    out: dict[str, list[int]] = {"fixable": [], "unreachable": [], "inexpressible": []}
    for i, (o, c) in enumerate(zip(obfuscated, clean)):
        if o == c:
            continue
        if i not in visited:
            out["unreachable"].append(i)
            continue
        o_slots, c_slots = _slot_values(o), _slot_values(c)
        if o_slots is None or c_slots is None:
            # 완성형 음절이 아닌 자리(초성체의 호환 자모 등)는 슬롯 단위로 쓸 수 없다.
            out["inexpressible"].append(i)
            continue
        # 담당 슬롯 밖의 슬롯이 하나라도 틀리면, 그 슬롯은 입력에서 복사되므로 못 고친다.
        if any(o_slots[s] != c_slots[s] for s in (0, 1, 2) if s not in unknown):
            out["inexpressible"].append(i)
        else:
            out["fixable"].append(i)
    return out


def ceiling_for(
    cleans: Sequence[str], obfuscated: Sequence[str], technique: str
) -> dict[str, float | int]:
    """말뭉치 전체에 대한 상한 요약.

    `ceiling_cer_reduction`이 핵심이다 — E8의 승격 기준(CER을 최소 2%p 줄일 것)과 직접
    비교할 수 있는 값이고, 이게 0.02 미만이면 그 조합에 그 모델을 붙이는 건 가망이 없다.
    """
    counts = {"fixable": 0, "unreachable": 0, "inexpressible": 0}
    chars = 0
    raw_errors = 0
    exact_possible = 0
    for clean, obf in zip(cleans, obfuscated):
        groups = classify_sites(obf, clean, technique)
        for key in counts:
            counts[key] += len(groups[key])
        chars += len(clean)
        raw_errors += sum(len(v) for v in groups.values())
        if not groups["unreachable"] and not groups["inexpressible"]:
            exact_possible += 1
    damaged = sum(counts.values())
    residual = counts["unreachable"] + counts["inexpressible"]
    n = len(cleans) or 1
    chars = chars or 1
    return {
        **counts,
        "damaged_sites": damaged,
        "sentences": len(cleans),
        "raw_cer": raw_errors / chars,
        "ceiling_residual_cer": residual / chars,
        "ceiling_cer_reduction": (raw_errors - residual) / chars,
        "ceiling_net_repair": counts["fixable"] / damaged if damaged else 0.0,
        "ceiling_exact_restoration": exact_possible / n,
        "unreachable_share": counts["unreachable"] / damaged if damaged else 0.0,
        "inexpressible_share": counts["inexpressible"] / damaged if damaged else 0.0,
    }
