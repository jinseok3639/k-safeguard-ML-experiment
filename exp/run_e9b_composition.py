"""E9b — 단일 난독화로 학습한 복원기가 조합 난독화를 다루는가, 혼합 학습의 대가는 무엇인가.

E9a(Stage 0)에서 학습 없이 구한 구조적 상한이 합집합 모델에 대해 기준을 넘었으므로, 남은
질문은 순수하게 통계적이다 — **그 분포로 학습했는가**. 세 가지를 묻는다.

1. 단일 난독화로 학습한 모델이 조합 난독화를 복원하는가?
2. 단일 + 조합을 섞어 학습해야 하는가?
3. 섞으면 무엇을 잃는가? (trade-off)

## 설계에서 조심한 것

- **손상량 고정(iso-damage)**: 각 단계를 강도 1.0으로 걸면 깊이 3이 문자의 76%를 파괴한다.
  그러면 모델이 못 맞히는 이유가 "조합이라서"인지 "더 망가져서"인지 구분되지 않는다.
  프로파일마다 강도를 이분 탐색해 실측 손상률을 예산에 맞추고, 그 실측값을 함께 보고한다.
- **임계값을 분위수로**: confidence는 담당 슬롯 수만큼 확률을 곱한 값이라 슬롯 1개짜리
  모델과 3개짜리 모델이 서로 다른 척도를 쓴다(E8에서 통과 임계값이 기법마다 0.99~0.999999로
  갈린 원인). 절대값 격자 대신 **"방문한 자리 중 몇 %를 포기할 것인가"** 로 임계값을 잡으면
  척도가 달라도 비교가 성립한다.
- **자리 단위 4분류**: `slot_accuracy_changed_only`는 `changed`(원문과 다른가)에 기대는데
  조합에서는 그 플래그가 출처를 잃는다. repair/miss/botch/damage로 나눈 `net_repair`를 쓴다.
- **cascade 대조군**: 조합을 다루는 가장 값싼 방법은 새 모델이 아니라 기존 단일 모델 3개를
  순서대로 돌리는 것이다. 학습 비용이 0이므로 이게 통하면 혼합 학습 자체가 불필요하다.

## 단계별 종료 조건

- Gate 1: 단일 기법 모델이 깊이 1에서 E8을 재현하지 못하면 harness가 깨진 것 — 중단.
- Gate 2: `universal/S`가 조합에서 자기 깊이-1 성능의 80% 이상을 유지하면 조합은 문제가
  아니다 — 거기서 결론 내고 중단(이 결과도 유효한 답이다).
- Gate 3: cascade가 `universal/S+M`에 필적하면 혼합 학습은 불필요하다.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import ceiling as CE
from kslab import compose as X
from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.models.bilstm import BiLstmSlotRestorer
from kslab.models.mlp import MlpSlotRestorer

# 조합에 투입하는 기법. chosung은 뒤 단계를 흡수하므로 제외(E9a에서 2000/2000 확인),
# palatalize는 문장 0.9%에서만 발동해 사실상 죽어 있다.
BASE_TECHNIQUES = ("tensify", "liaison", "jongseong_cram", "vowel_confuse", "aspirate")

# "어떤 공격인지 모른다"는 배치 상황을 나타내는 단일 모델. 담당 슬롯과 후보 위치가
# 구성 기법의 합집합이라 초성·중성·종성을 전부 예측하고 모든 음절을 방문한다.
UNIVERSAL = F.composite_name(BASE_TECHNIQUES)

INTENSITIES = (0.5, 1.0)

# 평가 프로파일. 깊이 1은 E0~E8과 직접 비교되는 기준선이다.
EVAL_PROFILES = (
    "d1_tensify", "d1_liaison", "d1_cram", "d1_vowel", "d1_aspirate",
    "d2_liaison_then_tensify", "d2_liaison_then_cram", "d2_tensify_cram",
    "d2_liaison_then_vowel",
    "d3_canonical", "d3_with_vowel",
)

# 학습용 조합 프로파일(S+M 레짐). 평가 프로파일과 일부러 완전히 겹치지 않게 둔다 —
# 겹치면 "조합을 배웠다"가 아니라 "이 프로파일을 외웠다"를 재게 된다.
TRAIN_PROFILES = (
    "d2_tensify_then_liaison", "d2_cram_then_liaison",
    "d2_aspirate_then_liaison", "d3_reversed",
)

DAMAGE_BUDGET = 0.08

# BiLSTM 체크포인트 위치. run-id 아래(`run.directory`)가 아니라 **run-id와 무관하게
# 고정된 경로**에 둔다 — `Run.start`는 같은 run-id 디렉터리를 재사용하지 못하게 막으므로,
# 죽은 뒤 새 run-id로 다시 돌려도 여기 체크포인트가 남아 있어야 그 자리부터 이어진다.
CHECKPOINT_DIR = R.RESULTS_DIR / "_checkpoints"

# 포기율(abstain rate) 기준 임계값. 절대 confidence 격자와 달리 슬롯 수가 달라도 비교된다.
ABSTAIN_QUANTILES = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

MUTATION_BAR = 0.01        # E8과 동일: benign 문장 변형률 ≤1%
MIN_CER_REDUCTION = 0.02   # E8과 동일: CER을 최소 2%p는 줄일 것
RETENTION_BAR = 0.80       # Gate 2: 깊이 1 대비 유지율


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--calibration-sentences", type=int, default=1500)
    parser.add_argument("--damage-budget", type=float, default=DAMAGE_BUDGET)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument(
        "--stages", nargs="*", default=["1", "2", "3"],
        help="실행할 단계. 3은 BiLSTM 확인(기법당 30분)이라 기본에서 빼고 싶을 수 있다.",
    )
    parser.add_argument("--skip-bilstm", action="store_true")
    parser.add_argument(
        "--bilstm-checkpoint-every", type=int, default=200,
        help="BiLSTM 학습 중 이 배치 수마다 체크포인트를 저장한다. 0이면 에폭 경계에서만 저장.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="tqdm 진행률 표시줄을 끈다(로그 파일로 리다이렉트할 때 등).",
    )
    return parser.parse_args()


# --- 학습 데이터 구성 ------------------------------------------------------

def _relabel(pairs, technique: str):
    """페어의 technique을 통일한다. 합집합 모델은 어떤 공격인지 모르는 채로 학습해야 한다."""
    return [dataclasses.replace(p, technique=technique) for p in pairs]


# identity 페어 규칙: **난독화 조건 하나당 identity 하나**. `make_pairs`가 기법마다
# identity를 하나씩 내는 것과 같은 규칙을 조합 프로파일에도 그대로 적용한다.
#
# 이 비율을 모델 간에 맞춰야 하는 이유: identity 페어가 "정상 입력은 건드리지 마라"는
# 신호이고, E2의 identity ablation이 보였듯 clean mutation을 직접 좌우한다. 비율이
# 다르면 합집합 모델의 오변경이 나쁘게 나와도 그게 명세(전 슬롯 · 전수 검사) 탓인지
# identity 신호가 적어서인지 가를 수 없다.
#
# 주의: 보고서의 `identity 비율`은 `Pair.is_identity`(난독화 결과 == 원문)로 세므로
# 의도한 identity 페어뿐 아니라 **변환이 발동하지 않은 경우**도 포함한다. 그래서 값이
# 1.0을 넘을 수 있다 — 기법이 자주 불발할수록 커진다. 그 값 자체가 모델이 실제로 본
# "안 바뀐 입력"의 비중이므로 비교에 쓰기에는 오히려 이쪽이 맞다.


def _single_pairs(items, args, technique: str | None = None):
    """단일 난독화 페어. `technique`을 주면 그 이름으로 다시 라벨링한다.

    relabel할 때 identity 페어를 중복 제거하지 **않는다**. 기법마다 하나씩 나온 identity를
    합쳐야 기법별 기준 모델과 같은 비율이 된다. 중복 제거하면 1/5로 줄어 조건이 달라진다.
    """
    pairs = P.make_pairs(
        items, BASE_TECHNIQUES, INTENSITIES,
        gen_seed=args.gen_seed, include_identity=True,
    )
    return pairs if technique is None else _relabel(pairs, technique)


def _composed_pairs(items, args, intensities, technique: str):
    """조합 난독화 페어(학습용 프로파일). 프로파일마다 identity 하나씩."""
    out = []
    for name in TRAIN_PROFILES:
        profile = X.PROFILES_BY_NAME[name]
        out += P.make_composed_pairs(
            items, profile, intensities[name], gen_seed=args.gen_seed,
            include_identity=True,
        )
    return _relabel(out, technique)


def _fit(technique, pairs, args, architecture="mlp", label=None):
    """`label`은 BiLSTM 체크포인트 파일명으로만 쓰인다 — 모델 자체는 몰라도 된다.

    체크포인트가 이미 완주돼 있으면(예: 죽었다 새 run-id로 재시작) `fit()`이 재학습
    없이 거의 즉시 그 상태를 불러온다. `kslab/models/bilstm.py`의 체크포인트/재개
    구현은 무중단 실행과 비트 단위로 같은 결과를 내도록 검증돼 있다
    (`tests/test_models_and_gate.py::TestBilstmCheckpoint`).
    """
    started = time.time()
    if architecture == "bilstm":
        checkpoint_path = (
            CHECKPOINT_DIR / f"{(label or technique).replace('/', '_')}.pt"
        )
        model = BiLstmSlotRestorer(
            technique, max_len=96, embed_dim=96, hidden=112, layers=2,
            epochs=5, batch_size=128, random_state=args.random_seed, threads=args.threads,
        ).fit(
            pairs, checkpoint_path=checkpoint_path,
            checkpoint_every=args.bilstm_checkpoint_every,
            show_progress=not args.no_progress,
        )
        units = len(pairs)
    else:
        examples = F.subsample(
            F.extract_examples(pairs, args.window), args.max_slot_examples, args.random_seed
        )
        model = MlpSlotRestorer(
            technique, window=args.window, epochs=args.epochs,
            random_state=args.random_seed, threads=args.threads,
        ).fit(examples)
        units = len(examples)
    identity = sum(1 for p in pairs if p.is_identity)
    obfuscated = len(pairs) - identity
    return model, {
        "fit_seconds": round(time.time() - started, 1),
        "train_units": units,
        "pairs": len(pairs),
        # 감사용. 모든 모델에서 같아야 비교가 성립한다(§ IDENTITY_PER_OBFUSCATED).
        "identity_ratio": round(identity / obfuscated, 3) if obfuscated else 0.0,
        "composed_pairs": sum(1 for p in pairs if p.depth > 1),
    }


# --- 평가 ------------------------------------------------------------------

def _quantile_thresholds(model, texts):
    """모델의 confidence 분포에서 포기율 기준 임계값을 뽑는다.

    E8은 절대 confidence 격자(0.99, 0.999, ...)를 썼는데, confidence가 담당 슬롯 수만큼
    확률을 곱한 값이라 슬롯 수가 다르면 척도가 달라진다. 포기율로 잡으면 척도와 무관하게
    "방문한 자리 중 몇 %를 포기했는가"가 축이 되어 모델 간 비교가 성립한다.
    """
    if isinstance(model, _CascadeModel):
        # cascade는 단계마다 척도가 다른 confidence를 하나로 합칠 수 없다. 분위수 키를
        # 그대로 넘겨 각 단계가 자기 임계값으로 환산하게 한다.
        return {f"{q:.2f}": f"{q:.2f}" for q in ABSTAIN_QUANTILES}
    scores: list[float] = []
    for _text, confidences in model.restore_many(texts, 0.0):
        scores += list(confidences)
    if not scores:
        return {f"{q:.2f}": 0.0 for q in ABSTAIN_QUANTILES}
    scores.sort()
    out = {}
    for q in ABSTAIN_QUANTILES:
        if q <= 0:
            out[f"{q:.2f}"] = 0.0
        else:
            index = min(len(scores) - 1, int(q * len(scores)))
            # 이 값 미만을 버리면 대략 q 비율이 포기된다.
            out[f"{q:.2f}"] = scores[index]
    return out


def _cluster_balanced(observations) -> float:
    """클러스터별 평균의 평균. 부트스트랩 없이 점추정만 — 스윕용 저비용 경로."""
    groups: dict[str, list[float]] = {}
    for cluster_id, value in observations:
        groups.setdefault(cluster_id, []).append(value)
    if not groups:
        return 0.0
    return sum(sum(v) / len(v) for v in groups.values()) / len(groups)


def _sweep_row(model, obfuscated, cleans, clusters, provenances, benign_mutation, threshold):
    """스윕 한 칸. 부트스트랩을 돌리지 않는다.

    그리드가 모델 × 프로파일 × 분위수로 수백 칸이라 칸마다 부트스트랩(2000회 × 4지표)을
    돌리면 실행 시간이 통제 불능이 된다. 여기서는 점추정만 내고, 승격 기준을 통과한
    지점에 대해서만 `_confirm`이 신뢰구간을 계산한다.
    """
    restored = [r[0] for r in model.restore_many(obfuscated, threshold)]
    outcomes = M.site_outcomes(cleans, obfuscated, restored, provenances)
    gain, exact = [], []
    for cluster_id, clean, obf, rest in zip(clusters, cleans, obfuscated, restored):
        gain.append((cluster_id, M.char_error_rate(clean, obf) - M.char_error_rate(clean, rest)))
        exact.append((cluster_id, float(rest == clean)))
    return {
        "benign_clean_mutation": benign_mutation,
        "cer_reduction_estimate": _cluster_balanced(gain),
        "exact_restoration_estimate": _cluster_balanced(exact),
        "net_repair": outcomes["net_repair"],
        "repair_rate": outcomes["repair_rate"],
        "botch_rate": outcomes["botch_rate"],
        "damage_sites": outcomes["damage"],
        "damaged_sites": outcomes["damaged_sites"],
        "by_stage": outcomes.get("by_stage", {}),
    }


def _confirm(model, obfuscated, cleans, clusters, threshold, args):
    """승격 기준을 통과한 지점에만 신뢰구간을 붙인다."""
    restored = [r[0] for r in model.restore_many(obfuscated, threshold)]
    return M.restoration_metrics(
        cleans, obfuscated, restored, clusters, args.bootstrap_samples, args.random_seed
    )


def _evaluate_model(model, eval_sets, cleans, clusters, benign) -> dict:
    """모델 하나를 모든 프로파일 × 포기율에서 스윕한다.

    모델을 하나 학습시킬 때마다(다음 모델 학습을 시작하기 전에) 바로 이걸 불러 결과를
    `metrics.json`에 스냅샷해야 한다 — 그래야 다음 모델(특히 BiLSTM처럼 몇 시간 걸리는
    학습) 도중 죽어도 이미 끝난 모델의 평가 결과는 디스크에 남는다.
    """
    thresholds = _quantile_thresholds(model, cleans)
    # benign 복원은 프로파일과 무관하다. 임계값당 한 번만 돌려 프로파일 수만큼의
    # 중복 계산을 없앤다.
    benign_by_threshold = {
        quantile: M.clean_mutation_rate(
            benign, [r[0] for r in model.restore_many(benign, threshold)]
        )
        for quantile, threshold in thresholds.items()
    }
    profiles = {}
    for pname, data in eval_sets.items():
        rows = {
            quantile: _sweep_row(
                model, data["obfuscated"], cleans, clusters, data["provenances"],
                benign_by_threshold[quantile], threshold,
            )
            for quantile, threshold in thresholds.items()
        }
        profiles[pname] = {"depth": data["depth"], "quantiles": rows}
    return {"thresholds": thresholds, "profiles": profiles}


class _CascadeModel:
    """단일 기법 모델을 정준 순서의 역순으로 이어 돌리는 대조군. 학습 비용 0.

    조합이 정준 순서로 걸렸다면 마지막에 걸린 것부터 벗겨내는 게 자연스럽다.

    임계값은 **단계마다 자기 분위수 임계값**을 쓴다. 단계별 confidence 척도가 서로 다르므로
    (담당 슬롯 수가 다르다) 하나의 절대값을 모든 단계에 먹이면 어떤 단계는 전부 통과시키고
    어떤 단계는 전부 포기하게 된다. `restore_many`의 `threshold` 자리에 분위수 키를 받아
    각 단계가 자기 임계값으로 환산한다.

    각 단계가 clean mutation을 독립적으로 더하므로 위험이 단계 수만큼 누적된다 — 그게
    이 대조군에서 실제로 재려는 대가다.
    """

    def __init__(self, models, thresholds_by_technique):
        self.models = models
        self.thresholds = thresholds_by_technique
        self.order = tuple(reversed(X.canonical_sort(list(models))))

    def restore_many(self, texts, quantile="0.00"):
        current = list(texts)
        for technique in self.order:
            threshold = self.thresholds[technique].get(str(quantile), 0.0)
            current = [r[0] for r in self.models[technique].restore_many(current, threshold)]
        # confidence는 단계마다 척도가 달라 하나로 합칠 수 없다. 분위수 축이 이미 그 역할을
        # 하므로 여기서는 자리 단위 점수를 내지 않는다.
        return [(text, []) for text in current]


def main() -> None:
    args = parse_args()
    stages = set(args.stages)
    run = R.Run.start(
        run_id=args.run_id,
        experiment="E9b_composition",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "window": args.window,
            "max_slot_examples": args.max_slot_examples,
            "damage_budget": args.damage_budget,
            "base_techniques": list(BASE_TECHNIQUES),
            "universal_technique": UNIVERSAL,
            "eval_profiles": list(EVAL_PROFILES),
            "train_profiles": list(TRAIN_PROFILES),
            "abstain_quantiles": list(ABSTAIN_QUANTILES),
            "intensities": list(INTENSITIES),
            "epochs": args.epochs,
            "stages": sorted(stages),
            "gen_seed": args.gen_seed,
            "random_seed": args.random_seed,
            "corpus_license": C.LICENSE_NOTE,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)
    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)

    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)
    corpus_texts = [t for _, _, t in train_items]

    seeds = P.load_seeds()
    seed_items = P.seeds_as_items(seeds)
    benign = [s.text for s in seeds if s.label == "benign"]
    cleans = [s.text for s in seeds]
    clusters = [s.seed_id for s in seeds]

    metrics: dict = {
        "universal_technique": UNIVERSAL,
        "calibration": {},
        "models": {},
        "evaluations": {},
        "ceilings": {},
        "gates": {},
    }

    # --- 강도 보정: 프로파일마다 손상량을 예산에 맞춘다 ---
    print(f"[E9b] 손상량 {args.damage_budget:.0%} 보정 ...", flush=True)
    calib_texts = corpus_texts[: args.calibration_sentences]
    intensities: dict[str, float] = {}
    for name in set(EVAL_PROFILES) | set(TRAIN_PROFILES):
        profile = X.PROFILES_BY_NAME[name]
        value, realized = X.calibrate(
            calib_texts, profile.stages, args.damage_budget, args.gen_seed
        )
        intensities[name] = value
        metrics["calibration"][name] = {
            "depth": profile.depth, "stage_intensity": round(value, 4),
            "realized_damage": realized, "stages": list(profile.stages),
        }

    # --- 평가 세트: 505 dev 시드에 프로파일을 건다 ---
    eval_sets = {}
    for name in EVAL_PROFILES:
        profile = X.PROFILES_BY_NAME[name]
        composed = [
            X.apply_stages(t, profile.stages, intensities[name], args.gen_seed)
            for t in cleans
        ]
        eval_sets[name] = {
            "obfuscated": [c.text for c in composed],
            "provenances": [c.provenance for c in composed],
            "depth": profile.depth,
        }
        metrics["ceilings"][name] = {
            technique: CE.ceiling_for(cleans, eval_sets[name]["obfuscated"], technique)
            for technique in (UNIVERSAL,)
        }

    trained: dict[str, object] = {}

    def _snapshot() -> None:
        # 전체 실행이 끝나기 전에도 여기까지의 결과를 디스크에 남긴다. `run.finish()`는
        # 맨 끝에 한 번만 부르므로(manifest에 elapsed_seconds를 확정해야 해서), 그 전에
        # 죽으면 아무것도 안 남는다 — 이 스냅샷이 그 공백을 메운다.
        run.write_json("metrics.json", metrics)

    def _train_eval_store(label, technique, pairs, architecture="mlp", regime=None, extra=None):
        print(f"  학습 {label} ...", flush=True)
        model, stats = _fit(technique, pairs, args, architecture=architecture, label=label)
        if regime:
            stats["regime"] = regime
        if extra:
            stats.update(extra)
        metrics["models"][label] = stats
        trained[label] = model
        print(f"  평가 {label} ...", flush=True)
        metrics["evaluations"][label] = _evaluate_model(model, eval_sets, cleans, clusters, benign)
        _snapshot()
        return model

    # --- Stage 1: 단일 난독화로만 학습 ---
    if "1" in stages:
        print("[E9b] Stage 1 — 단일 학습 모델", flush=True)
        per_technique = {}
        per_technique_thresholds = {}
        for technique in BASE_TECHNIQUES:
            pairs = P.make_pairs(
                train_items, [technique], INTENSITIES,
                gen_seed=args.gen_seed, include_identity=True,
            )
            model = _train_eval_store(f"single/{technique}", technique, pairs)
            per_technique[technique] = model
            per_technique_thresholds[technique] = metrics["evaluations"][f"single/{technique}"]["thresholds"]

        single_pairs = _single_pairs(train_items, args, UNIVERSAL)
        _train_eval_store("universal/S", UNIVERSAL, single_pairs, regime="S (단일 난독화만)")

        cascade = _CascadeModel(per_technique, per_technique_thresholds)
        trained["cascade/S"] = cascade
        metrics["models"]["cascade/S"] = {
            "fit_seconds": 0.0, "train_units": 0,
            "regime": "S (학습 비용 0)",
            "note": "단일 모델을 정준 순서 역순으로 이어 돌린다. 단계마다 자기 분위수 임계값을 쓴다.",
        }
        print("  평가 cascade/S ...", flush=True)
        metrics["evaluations"]["cascade/S"] = _evaluate_model(cascade, eval_sets, cleans, clusters, benign)
        _snapshot()

    # --- Stage 2: 단일 + 조합 혼합 학습 ---
    if "2" in stages:
        print("[E9b] Stage 2 — 혼합 학습 모델", flush=True)
        mixed = _single_pairs(train_items, args, UNIVERSAL) + _composed_pairs(
            train_items, args, intensities, UNIVERSAL
        )
        _train_eval_store(
            "universal/S+M", UNIVERSAL, mixed, regime="S+M (단일 + 조합)",
            extra={"composed_profiles": list(TRAIN_PROFILES)},
        )

    # --- Stage 3: BiLSTM 확인 ---
    if "3" in stages and not args.skip_bilstm:
        print("[E9b] Stage 3 — BiLSTM 확인 (오래 걸립니다)", flush=True)
        print(f"  체크포인트 위치: {CHECKPOINT_DIR} (죽어도 여기서 이어집니다)", flush=True)
        for label, builder in (
            ("bilstm/S", lambda: _single_pairs(train_items, args, UNIVERSAL)),
            (
                "bilstm/S+M",
                lambda: _single_pairs(train_items, args, UNIVERSAL)
                + _composed_pairs(train_items, args, intensities, UNIVERSAL),
            ),
        ):
            _train_eval_store(label, UNIVERSAL, builder(), architecture="bilstm")

    _gates(metrics)

    # 통과 지점에만 신뢰구간을 붙인다.
    print("[E9b] 통과 지점 신뢰구간 확인 ...", flush=True)
    for label, profiles in metrics.get("passing", {}).items():
        model = trained[label]
        for pname, best in profiles.items():
            threshold = metrics["evaluations"][label]["thresholds"][best["quantile"]]
            best["confirmed"] = _confirm(
                model, eval_sets[pname]["obfuscated"], cleans, clusters, threshold, args
            )
    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e9b_composition_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E9b] 완료 -> {run.directory}  판정 {metrics['status']}")
    for key, value in metrics["gates"].items():
        print(f"  {key}: {value}")


def _best_passing(entry) -> dict | None:
    """승격 기준 두 개를 동시에 만족하는 지점 중 CER 개선이 가장 큰 것."""
    best = None
    for quantile, row in entry["quantiles"].items():
        mutation = row["benign_clean_mutation"]["sentence_rate"]
        reduction = row["cer_reduction_estimate"] or 0.0
        if mutation <= MUTATION_BAR and reduction >= MIN_CER_REDUCTION:
            if best is None or reduction > best["cer_reduction"]:
                best = {
                    "quantile": quantile, "cer_reduction": reduction,
                    "benign_clean_mutation": mutation,
                    "net_repair": row["net_repair"],
                    "benign_ci95_high": row["benign_clean_mutation"]["sentence_rate_ci95_high"],
                }
    return best


def _gates(metrics: dict) -> None:
    passing: dict[str, dict] = {}
    for label, entry in metrics["evaluations"].items():
        for pname, profile_entry in entry["profiles"].items():
            best = _best_passing(profile_entry)
            if best:
                passing.setdefault(label, {})[pname] = best
    metrics["passing"] = passing

    def reduction(label, pname):
        row = passing.get(label, {}).get(pname)
        return row["cer_reduction"] if row else 0.0

    # Gate 2 — universal/S 가 조합에서 자기 깊이-1 성능을 얼마나 유지하는가.
    #
    # 깊이 1에서조차 통과 지점이 없으면 이 질문 자체가 성립하지 않는다. 그때 유지율을
    # 0으로 계산해 "조합이 문제다"라고 말하면 거짓이다 — 조합 이전에 모델이 안 되는 것이다.
    base = max(
        (reduction("universal/S", p) for p in EVAL_PROFILES if p.startswith("d1_")),
        default=0.0,
    )
    deeper = max(
        (reduction("universal/S", p) for p in EVAL_PROFILES if not p.startswith("d1_")),
        default=0.0,
    )
    if base <= 0.0:
        metrics["gates"]["gate2_retention"] = None
        metrics["gates"]["gate2_composition_is_a_problem"] = None
        metrics["gates"]["gate2_note"] = (
            "universal/S가 깊이 1에서도 승격 기준을 통과하지 못했다. "
            "조합의 영향을 재려면 먼저 단일 난독화에서 기준을 넘어야 한다."
        )
    else:
        retention = deeper / base
        metrics["gates"]["gate2_retention"] = round(retention, 4)
        metrics["gates"]["gate2_composition_is_a_problem"] = retention < RETENTION_BAR

    # Gate 3 — cascade 가 혼합 학습에 필적하는가.
    #
    # cascade_best > 0 인데 mixed_best == 0 인 경우("승격 지점이 cascade에는 있고 혼합
    # 학습에는 없다")는 애매한 게 아니라 **cascade가 명백히 이긴다**는 강한 결과다 —
    # None으로 감추면 안 된다. 진짜로 판단이 안 서는 경우는 **둘 다** 0일 때뿐이다 —
    # 그때는 "cascade가 충분하다"도 "혼합 학습이 필요하다"도 아니고, 어느 쪽도 승격
    # 기준에 못 미친다는 별개의 사실이다.
    cascade_best = max((reduction("cascade/S", p) for p in EVAL_PROFILES), default=0.0)
    mixed_best = max((reduction("universal/S+M", p) for p in EVAL_PROFILES), default=0.0)
    metrics["gates"]["gate3_cascade_best_cer_reduction"] = round(cascade_best, 4)
    metrics["gates"]["gate3_mixed_best_cer_reduction"] = round(mixed_best, 4)
    if cascade_best <= 0.0 and mixed_best <= 0.0:
        metrics["gates"]["gate3_cascade_suffices"] = None
        metrics["gates"]["gate3_note"] = (
            "cascade와 universal/S+M 둘 다 승격 기준을 만족하는 지점이 하나도 없다. "
            "'cascade로 충분한가'는 어느 쪽도 충분하지 않다는 뜻에서 별개로 성립하지 않는다."
        )
    else:
        metrics["gates"]["gate3_cascade_suffices"] = cascade_best >= mixed_best * RETENTION_BAR

    metrics["status"] = "PROMOTE_CANDIDATE" if passing else "DO_NOT_PROMOTE"


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E9b — 조합 난독화에 대한 단일 학습 vs 혼합 학습",
        "",
        f"> 상태: `{metrics['status']}`",
        ">",
        "> 단일 난독화로 학습한 복원기가 겹쳐 걸린 난독화를 다루는지, 조합을 섞어 학습하면",
        "> 무엇을 얻고 무엇을 잃는지 잰다. E9a에서 구조적 상한이 통과했으므로 여기서 남은",
        "> 질문은 통계적인 것뿐이다.",
        "",
        "## 승격 기준 (E8과 동일)",
        "",
        f"1. benign 문장 변형률 ≤ {MUTATION_BAR:.0%}",
        f"2. **그리고** CER 개선폭 ≥ {MIN_CER_REDUCTION:.0%} (전부 포기하면 1번은 공짜로 통과하므로)",
        "",
        f"- 코퍼스 {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        "- 평가: 505 dev 시드",
        f"- 손상 예산 {args.damage_budget:.0%} — 깊이가 달라도 손상량이 같도록 강도를 보정했다",
        f"- 합집합 모델 technique: `{metrics['universal_technique']}`",
        "",
        "## 1. 강도 보정 (손상량 고정)",
        "",
        "| 프로파일 | 깊이 | 단계 강도 | 실측 손상률 |",
        "|---|---|---|---|",
    ]
    for name, row in sorted(metrics["calibration"].items()):
        lines.append(
            f"| {name} | {row['depth']} | {row['stage_intensity']:.3f} | {row['realized_damage']:.2%} |"
        )

    lines += [
        "",
        "## 2. 학습 비용과 데이터 구성",
        "",
        "`identity 비율`은 (원문 그대로 페어 / 난독화 페어)다. 이 값이 모델마다 다르면",
        "오변경 비교가 성립하지 않는다 — E2의 identity ablation이 보였듯 이 신호가 clean",
        "mutation을 직접 좌우하기 때문이다. 모든 모델에서 같은지 감사용으로 싣는다.",
        "",
        "| 모델 | 레짐 | 학습 시간(s) | 학습 단위 | identity 비율 | 조합 페어 |",
        "|---|---|---|---|---|---|",
    ]
    for label, row in metrics["models"].items():
        lines.append(
            f"| {label} | {row.get('regime', '-')} | {row['fit_seconds']} | "
            f"{row['train_units']:,} | {row.get('identity_ratio', 0):.3f} | "
            f"{row.get('composed_pairs', 0):,} |"
        )

    lines += [
        "",
        "## 3. 승격 기준을 동시에 만족하는 지점",
        "",
        "`net_repair` = (되돌린 자리 − 망친 자리) / 손상된 자리. 아무것도 안 하면 0이고,",
        "멀쩡한 데를 망치면 음수가 된다. 조합 깊이가 달라도 뜻이 흐려지지 않는 지표다.",
        "",
        "| 모델 | 프로파일 | 포기율 | CER 개선 | benign 변형률 | (95% 상한) | net_repair |",
        "|---|---|---|---|---|---|---|",
    ]
    any_pass = False
    for label, profiles in metrics.get("passing", {}).items():
        for pname, row in sorted(profiles.items()):
            any_pass = True
            lines.append(
                f"| {label} | {pname} | {row['quantile']} | {row['cer_reduction'] * 100:+.2f}%p | "
                f"{row['benign_clean_mutation']:.2%} | {row['benign_ci95_high']:.2%} | "
                f"{row['net_repair']:+.3f} |"
            )
    if not any_pass:
        lines.append("| — | — | — | — | — | — | — |")

    gates = metrics["gates"]
    if gates.get("gate2_retention") is None:
        gate2_line = f"- **Gate 2 (조합이 문제인가)**: {gates.get('gate2_note', 'universal/S가 깊이 1에서도 통과 지점이 없어 유지율을 정의할 수 없다.')}"
    else:
        gate2_line = (
            f"- **Gate 2 (조합이 문제인가)**: `universal/S`가 조합에서 깊이-1 대비 "
            f"**{gates['gate2_retention']:.1%}** 를 유지했다. "
            + (
                f"{RETENTION_BAR:.0%} 미만이므로 **조합은 실제 문제다** — 혼합 학습을 볼 이유가 있다."
                if gates.get("gate2_composition_is_a_problem")
                else f"{RETENTION_BAR:.0%} 이상이므로 **조합은 큰 문제가 아니다** — 단일 학습으로 충분하다."
            )
        )
    if gates.get("gate3_cascade_suffices") is None:
        gate3_line = f"- **Gate 3 (cascade로 충분한가)**: {gates.get('gate3_note', 'cascade와 혼합 학습 둘 다 통과 지점이 없어 비교가 성립하지 않는다.')}"
    else:
        gate3_line = (
            f"- **Gate 3 (cascade로 충분한가)**: cascade {gates.get('gate3_cascade_best_cer_reduction', 0) * 100:+.2f}%p "
            f"vs 혼합 학습 {gates.get('gate3_mixed_best_cer_reduction', 0) * 100:+.2f}%p. "
            + (
                "cascade가 필적하거나 앞서므로 **혼합 학습은 불필요하다** — 학습 비용 0으로 같은 자리에 간다."
                if gates["gate3_cascade_suffices"]
                else "혼합 학습이 cascade를 유의하게 앞선다."
            )
        )
    lines += [
        "",
        "## 판정",
        "",
        gate2_line,
        gate3_line,
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론 없음. clean mutation은 ΔFPR의 proxy다.",
        "- benign 시드가 204개뿐이라 1% 기준의 관측 해상도가 0.49%p다. 표에 Wilson 95% 상한을",
        "  함께 실었다 — 0/204도 상한은 1.85%이므로 '통과'는 약한 주장이다.",
        "- 임계값을 포기율(분위수)로 잡아 슬롯 수가 다른 모델 간 비교를 성립시켰다. 절대",
        "  confidence 값은 모델마다 척도가 달라 그대로 비교하면 안 된다.",
        "- 학습용 조합 프로파일과 평가용 조합 프로파일을 겹치지 않게 두었다. 겹치면 '조합을",
        "  일반화했다'가 아니라 '그 프로파일을 외웠다'를 재게 된다.",
        "- 혼합 학습(S+M)은 같은 예제 예산 안에서 단일 데이터 비중을 줄인다. 깊이-1 성능",
        "  하락에는 '간섭'과 '단일 데이터 감소'가 섞여 있다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
