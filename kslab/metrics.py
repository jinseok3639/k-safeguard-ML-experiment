"""proxy 지표 — 가드레일 추론 없이 복원 품질을 재는 값싼 신호.

`EVALUATION_SPEC.md`의 NRR/ΔFPR은 가드레일 모델이 있어야 계산된다. 이 폴더에는 GPU도
모델도 붙이지 않기로 했으므로, 여기서는 모델 자체의 복원 품질과 **clean mutation rate**만
본다. clean mutation rate는 스펙 §11.4의 실제 승격 기준(≤1%)을 그대로 쓴다 — 된소리
provider를 DO_NOT_PROMOTE로 만든 축이 정확히 이것이기 때문이다.

`summarize_observations`는 `experiments/benchmark/run_normalizer_evaluation.py:298-332`의
seed-cluster bootstrap을 그대로 이식한 것이다(stdlib만 사용, 동작 동일).
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence


def percentile(values: list[float], proportion: float) -> float | None:
    """run_clean_baseline.py:242-247 과 동일한 구현."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * proportion + 0.5)))
    return ordered[index]


def _bootstrap_mean_ci(
    seed_values: list[float], samples: int, random_seed: int
) -> tuple[float | None, float | None]:
    if not seed_values or samples < 1:
        return None, None
    rng = random.Random(random_seed)
    size = len(seed_values)
    estimates = [statistics.fmean(rng.choices(seed_values, k=size)) for _ in range(samples)]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_observations(
    observations: Sequence[tuple[str, float]],
    bootstrap_samples: int = 10_000,
    random_seed: int = 2026,
) -> dict[str, Any]:
    """(cluster_id, value) 목록 -> cluster-balanced 점추정 + bootstrap 95% CI.

    cluster_id는 통계적 독립 단위다. 505 시드 평가에서는 `seed_id`, 코퍼스 평가에서는
    문서/출처 id를 넣는다. 같은 cluster 안의 값을 먼저 평균낸 뒤 cluster를 재표집한다.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for cluster_id, value in observations:
        grouped[cluster_id].append(value)
    seed_values = [statistics.fmean(values) for values in grouped.values()]
    ci_low, ci_high = _bootstrap_mean_ci(seed_values, bootstrap_samples, random_seed)
    return {
        "rows": len(observations),
        "clusters": len(seed_values),
        "micro_estimate": statistics.fmean(v for _, v in observations) if observations else None,
        "cluster_balanced_estimate": statistics.fmean(seed_values) if seed_values else None,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "bootstrap_samples": bootstrap_samples if seed_values else 0,
    }


def char_error_rate(reference: str, hypothesis: str) -> float:
    """문자 단위 편집거리 / 참조 길이. 길이가 같으면 단순 불일치 비율과 같다."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    if len(reference) == len(hypothesis):
        return sum(a != b for a, b in zip(reference, hypothesis)) / len(reference)
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        cur = [i]
        for j, h in enumerate(hypothesis, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(reference)


def slot_accuracy(
    predictions: Iterable[tuple[int, int, int]],
    targets: Iterable[tuple[int, int, int]],
) -> dict[str, float]:
    """초성/중성/종성 각각의 정확도와 음절 exact(3슬롯 모두 일치)."""
    n = 0
    hits = [0, 0, 0]
    exact = 0
    for pred, gold in zip(predictions, targets):
        n += 1
        ok = True
        for k in range(3):
            if pred[k] == gold[k]:
                hits[k] += 1
            else:
                ok = False
        exact += ok
    if not n:
        return {"cho": 0.0, "jung": 0.0, "jong": 0.0, "syllable_exact": 0.0, "n": 0}
    return {
        "cho": hits[0] / n,
        "jung": hits[1] / n,
        "jong": hits[2] / n,
        "syllable_exact": exact / n,
        "n": n,
    }


def clean_mutation_rate(originals: Sequence[str], restored: Sequence[str]) -> dict[str, Any]:
    """clean 입력에 복원기를 그대로 돌렸을 때 바뀐 문장 비율. 승격 1차 게이트(≤1%).

    문장 단위 비율과 문자 단위 비율을 함께 낸다 — 문장 하나에서 한 글자만 바뀌어도
    가드레일 판정은 뒤집힐 수 있으므로 문장 비율이 주 지표다.
    """
    if len(originals) != len(restored):
        raise ValueError("originals 와 restored 의 길이가 다릅니다.")
    n = len(originals)
    if not n:
        return {"sentence_rate": 0.0, "char_rate": 0.0, "sentences": 0, "mutated": 0}
    mutated = sum(1 for a, b in zip(originals, restored) if a != b)
    total_chars = sum(len(a) for a in originals) or 1
    changed_chars = sum(
        sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        for a, b in zip(originals, restored)
    )
    return {
        "sentence_rate": mutated / n,
        "char_rate": changed_chars / total_chars,
        "sentences": n,
        "mutated": mutated,
    }


def restoration_metrics(
    cleans: Sequence[str],
    obfuscated: Sequence[str],
    restored: Sequence[str],
    cluster_ids: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
    random_seed: int = 2026,
) -> dict[str, Any]:
    """문장 단위 복원 품질: exact restoration, residual CER, CER 개선폭."""
    if not (len(cleans) == len(obfuscated) == len(restored)):
        raise ValueError("입력 시퀀스 길이가 서로 다릅니다.")
    ids = list(cluster_ids) if cluster_ids is not None else [str(i) for i in range(len(cleans))]
    exact, residual, raw, gain = [], [], [], []
    for cid, clean, obf, rest in zip(ids, cleans, obfuscated, restored):
        exact.append((cid, float(rest == clean)))
        r_cer = char_error_rate(clean, rest)
        o_cer = char_error_rate(clean, obf)
        residual.append((cid, r_cer))
        raw.append((cid, o_cer))
        gain.append((cid, o_cer - r_cer))
    return {
        "exact_restoration": summarize_observations(exact, bootstrap_samples, random_seed),
        "residual_cer": summarize_observations(residual, bootstrap_samples, random_seed),
        "raw_cer": summarize_observations(raw, bootstrap_samples, random_seed),
        "cer_reduction": summarize_observations(gain, bootstrap_samples, random_seed),
    }
