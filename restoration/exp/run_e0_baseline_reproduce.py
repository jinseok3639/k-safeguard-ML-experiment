"""E0 — 하네스 검증 + 분할 정직성 실측 + 현상 고정.

**이 실험은 유효한 모델을 만들지 않는다.** 505개 dev 시드로 학습까지 해버리는, 일부러
잘못된 설정이다. 목적은 두 가지다.

1. seed_id 랜덤 분할과 source-disjoint 분할이 얼마나 벌어지는지 수치로 박아둔다.
   505 시드 중 256개가 `template_v1` 한 출처라 랜덤 분할은 같은 템플릿의 형제 문장을
   train/test 양쪽으로 흘린다. 이후 모든 실험이 source 분할을 쓰는 근거가 이 수치다.
2. "505 문장짜리 코퍼스"가 clean mutation rate를 어디까지밖에 못 낮추는지 보여준다.
   이게 E1 코퍼스 스케일링 곡선의 최좌단 점이 된다.

산출물에 status SEED_TRAINED_DIAGNOSTIC 을 박아 승격 후보로 오해하지 않게 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.models.linear import SlotRestorer

STATUS = "SEED_TRAINED_DIAGNOSTIC"
TECHNIQUES = ("chosung", "tensify")
INTENSITIES = (0.3, 0.5, 1.0)
HOLDOUT_SOURCE = "template_v1"
SLOT_NAMES = ("cho", "jung", "jong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument(
        "--allow-unset-cache-paths",
        action="store_true",
        help="env.ps1 없이 돌릴 때만. 이 실험은 HF/torch를 쓰지 않는다.",
    )
    return parser.parse_args()


def _train_eval(technique, train_items, test_items, args):
    """한 기법 x 한 분할에 대해 학습·평가하고 지표와 예측행을 돌려준다."""
    train_pairs = P.make_pairs(
        train_items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=True
    )
    test_pairs = P.make_pairs(
        test_items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=False
    )
    train_ex = F.extract_examples(train_pairs, args.window)
    test_ex = [
        e for e in F.extract_examples(test_pairs, args.window) if e.technique == technique
    ]
    model = SlotRestorer(technique, window=args.window).fit(train_ex)

    predicted = model.predict_slots(test_ex)
    gold = [e.target_slots for e in test_ex]
    slot_acc = M.slot_accuracy(
        [tuple(int(v) for v in row) for row in predicted], gold
    )

    # 변형이 실제로 걸린 자리만. 안 걸린 자리는 그대로 두면 맞으므로 정확도를 부풀린다.
    changed_idx = [i for i, e in enumerate(test_ex) if e.changed]
    changed_acc = M.slot_accuracy(
        [tuple(int(v) for v in predicted[i]) for i in changed_idx],
        [gold[i] for i in changed_idx],
    )

    # 다수결 baseline: 학습셋 최빈 클래스로만 찍었을 때
    baselines = {}
    for slot in F.UNKNOWN_SLOTS[technique]:
        train_y = [e.target_slots[slot] for e in train_ex if e.target_slots[slot] >= 0]
        if not train_y or not changed_idx:
            continue
        majority = max(set(train_y), key=train_y.count)
        hits = sum(1 for i in changed_idx if gold[i][slot] == majority)
        baselines[SLOT_NAMES[slot]] = hits / len(changed_idx)

    # 문장 단위 복원 품질
    obf_texts, clean_texts, restored, clusters = [], [], [], []
    for pair in test_pairs:
        if pair.technique != technique:
            continue
        restored_text, _ = model.restore(pair.obfuscated)
        obf_texts.append(pair.obfuscated)
        clean_texts.append(pair.clean)
        restored.append(restored_text)
        clusters.append(pair.cluster_id)
    restoration = M.restoration_metrics(
        clean_texts, obf_texts, restored, clusters,
        args.bootstrap_samples, args.random_seed,
    )

    # clean mutation: 난독화하지 않은 원문을 그대로 복원기에 통과시킨다.
    clean_inputs = [text for _, _, text in test_items]
    clean_outputs = [model.restore(t)[0] for t in clean_inputs]
    mutation = M.clean_mutation_rate(clean_inputs, clean_outputs)

    result = {
        "train_examples": len(train_ex),
        "test_examples": len(test_ex),
        "test_changed_examples": len(changed_idx),
        "slot_accuracy_all_candidates": slot_acc,
        "slot_accuracy_changed_only": changed_acc,
        "majority_baseline_changed_only": baselines,
        "restoration": restoration,
        "clean_mutation": mutation,
    }
    rows = list(zip(clusters, obf_texts, clean_texts, restored))
    return result, rows


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()
    items = P.seeds_as_items(seeds)

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E0_baseline_reproduce",
        config={
            "status": STATUS,
            "techniques": list(TECHNIQUES),
            "intensities": list(INTENSITIES),
            "window": args.window,
            "holdout_source": HOLDOUT_SOURCE,
            "bootstrap_samples": args.bootstrap_samples,
            "random_seed": args.random_seed,
            "gen_seed": args.gen_seed,
            "model": "sklearn.LogisticRegression(C=2.0, max_iter=400)",
            "warning": "505 dev 시드로 학습한 진단용 설정. 승격 후보 아님.",
        },
        strict_paths=not args.allow_unset_cache_paths,
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)
    run.note(
        "seed_source_counts",
        {s: sum(1 for x in seeds if x.source == s) for s in sorted({x.source for x in seeds})},
    )

    source_train, source_test = P.split_by_source(items, [HOLDOUT_SOURCE])
    P.assert_disjoint(source_train, source_test)
    random_train, random_test = P.split_random(
        items, test_size=len(source_test), random_seed=0
    )
    P.assert_disjoint(random_train, random_test)
    run.note(
        "split_sizes",
        {
            "source_disjoint": {"train": len(source_train), "test": len(source_test)},
            "random": {"train": len(random_train), "test": len(random_test)},
        },
    )

    metrics: dict = {"status": STATUS, "splits": {}}
    predictions: list[dict] = []
    for split_name, (tr, te) in (
        ("source_disjoint", (source_train, source_test)),
        ("random", (random_train, random_test)),
    ):
        metrics["splits"][split_name] = {}
        for technique in TECHNIQUES:
            print(f"[E0] {split_name} / {technique} ...", flush=True)
            result, rows = _train_eval(technique, tr, te, args)
            metrics["splits"][split_name][technique] = result
            for cluster_id, obf, clean, rest in rows:
                predictions.append(
                    {
                        "split": split_name,
                        "technique": technique,
                        "cluster_id": cluster_id,
                        "obfuscated": obf,
                        "clean": clean,
                        "restored": rest,
                        "exact": rest == clean,
                    }
                )

    inflation = {}
    for technique in TECHNIQUES:
        src = metrics["splits"]["source_disjoint"][technique]["slot_accuracy_changed_only"]
        rnd = metrics["splits"]["random"][technique]["slot_accuracy_changed_only"]
        inflation[technique] = {
            slot: round(rnd[slot] - src[slot], 4)
            for slot in ("cho", "jung", "jong", "syllable_exact")
        }
    metrics["random_split_inflation_pp"] = inflation

    run.write_jsonl("predictions.jsonl", predictions)
    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)

    R.publish_baseline(
        "e0_probe_v1.json",
        {
            "run_id": args.run_id,
            "status": STATUS,
            "splits": metrics["splits"],
            "random_split_inflation_pp": inflation,
        },
        run_id=args.run_id,
    )
    print(f"\n[E0] 완료 -> {run.directory}")
    for technique in TECHNIQUES:
        src = metrics["splits"]["source_disjoint"][technique]
        acc = src["slot_accuracy_changed_only"]
        print(
            f"  {technique:9s} source분할 changed-only  "
            f"초성 {acc['cho']:.3f} 중성 {acc['jung']:.3f} 종성 {acc['jong']:.3f}  |  "
            f"clean mutation {src['clean_mutation']['sentence_rate']:.1%}"
        )


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E0 — 하네스 검증과 분할 정직성 실측",
        "",
        f"> 상태: `{STATUS}`",
        ">",
        "> 505개 dev 시드로 **학습까지** 한 일부러 잘못된 설정이다. 유효한 모델이 아니라,",
        "> 분할 규칙의 근거와 코퍼스 스케일링 곡선의 최좌단 점을 만들기 위한 진단이다.",
        "",
        "## 실행 조건",
        "",
        f"- 기법: {', '.join(TECHNIQUES)} / 강도 {INTENSITIES}",
        f"- 모델: sklearn LogisticRegression(C=2.0, max_iter=400), 문자 창 +-{args.window}",
        f"- holdout source: `{HOLDOUT_SOURCE}` (505 중 256)",
        f"- bootstrap {args.bootstrap_samples:,}회, 난수 시드 {args.random_seed}",
        "- identity 페어(clean -> clean)를 학습에 포함",
        "",
        "## 결과 — 변형이 실제로 걸린 위치만",
        "",
        "| 기법 | 분할 | 초성 | 중성 | 종성 | 음절 exact | clean mutation |",
        "|---|---|---|---|---|---|---|",
    ]
    for technique in TECHNIQUES:
        for split_name in ("source_disjoint", "random"):
            r = metrics["splits"][split_name][technique]
            a = r["slot_accuracy_changed_only"]
            lines.append(
                f"| {technique} | {split_name} | {a['cho']:.3f} | {a['jung']:.3f} | "
                f"{a['jong']:.3f} | {a['syllable_exact']:.3f} | "
                f"{r['clean_mutation']['sentence_rate']:.1%} |"
            )
    lines += [
        "",
        "## 다수결 baseline 대비 (source 분할, changed-only)",
        "",
        "| 기법 | 슬롯 | 모델 | 다수결 |",
        "|---|---|---|---|",
    ]
    for technique in TECHNIQUES:
        r = metrics["splits"]["source_disjoint"][technique]
        for slot_name, base in r["majority_baseline_changed_only"].items():
            lines.append(
                f"| {technique} | {slot_name} | "
                f"{r['slot_accuracy_changed_only'][slot_name]:.3f} | {base:.3f} |"
            )
    lines += [
        "",
        "## 랜덤 분할이 부풀리는 양 (랜덤 − source, %p)",
        "",
        "| 기법 | 초성 | 중성 | 종성 | 음절 exact |",
        "|---|---|---|---|---|",
    ]
    for technique, d in metrics["random_split_inflation_pp"].items():
        lines.append(
            f"| {technique} | {d['cho'] * 100:+.1f} | {d['jung'] * 100:+.1f} | "
            f"{d['jong'] * 100:+.1f} | {d['syllable_exact'] * 100:+.1f} |"
        )
    lines += [
        "",
        "## 판정",
        "",
        "- 이후 모든 실험은 **source-disjoint 분할**만 주 지표로 보고한다. 랜덤 분할 수치는",
        "  대조군으로만 병기한다.",
        "- clean mutation rate가 승격 기준(<=1%)에서 얼마나 먼지가 E1의 출발점이다.",
        "",
        "## 해석 제한",
        "",
        "- 505 시드로 학습한 설정이라 어휘 폭이 극단적으로 좁다. 절대 수치를 모델 성능으로",
        "  읽으면 안 된다.",
        "- 가드레일 추론을 하지 않았으므로 NRR·ΔFPR은 이 문서에 없다.",
        "- `results/`의 `predictions.jsonl`에는 공격 원문이 들어 있어 커밋하지 않는다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
