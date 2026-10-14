"""E1 — 코퍼스 스케일링 곡선. 이 샌드박스의 결정적 실험.

E0에서 505문장으로 학습한 tensify 복원기의 clean mutation rate가 23.8%였다. 승격 기준은
1% 이하다. 질문은 하나다: **이 간극이 데이터로 메워지는가, 아니면 아키텍처를 바꿔야 하는가.**

아키텍처를 로지스틱(SGD, log loss)으로 **고정**하고 코퍼스 문장 수만 늘린다. 학습 슬롯
예제 수는 상한을 두어 고정하므로, 곡선이 재는 것은 기울기 스텝 수가 아니라 **어휘 폭**이다.

평가는 항상 505개 dev 시드로 한다 — 학습에는 절대 넣지 않는다. 코퍼스(위키·KLUE)와 시드
(대화형 명령문)는 분포가 다르고, 우리가 알고 싶은 것이 바로 그 전이 성능이다.

부가로 위키:KLUE 혼합 비율도 한 지점에서 스윕한다. 어휘 폭만이 아니라 레지스터도
clean mutation에 영향을 준다는 가설을 확인하기 위해서다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.models.linear import SlotRestorer

TECHNIQUES = ("tensify", "chosung", "liaison", "jongseong_cram")
INTENSITIES = (0.5, 1.0)
CORPUS_SIZES = (500, 2000, 8000, 30000, 100000)
RATIO_SWEEP_SIZE = 30000
RATIO_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_RATIO = 0.5
SLOT_NAMES = ("cho", "jung", "jong")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--sizes", type=int, nargs="*", default=list(CORPUS_SIZES))
    parser.add_argument("--techniques", nargs="*", default=list(TECHNIQUES))
    parser.add_argument("--skip-ratio-sweep", action="store_true")
    return parser.parse_args()


def _eval_on_seeds(model, technique, seeds, args):
    """505 dev 시드에서 슬롯 정확도와 clean mutation을 잰다."""
    items = P.seeds_as_items(seeds)
    eval_pairs = P.make_pairs(
        items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=False
    )
    examples = [
        e for e in F.extract_examples(eval_pairs, args.window) if e.technique == technique
    ]
    changed = [e for e in examples if e.changed]
    accuracy = {"cho": 0.0, "jung": 0.0, "jong": 0.0, "syllable_exact": 0.0, "n": 0}
    if changed:
        predicted = model.predict_slots(changed)
        accuracy = M.slot_accuracy(
            [tuple(int(v) for v in row) for row in predicted],
            [e.target_slots for e in changed],
        )

    # 문장 단위 복원 품질
    obf = [p.obfuscated for p in eval_pairs]
    clean = [p.clean for p in eval_pairs]
    clusters = [p.cluster_id for p in eval_pairs]
    restored = [r[0] for r in model.restore_many(obf)]
    restoration = M.restoration_metrics(
        clean, obf, restored, clusters, args.bootstrap_samples, args.random_seed
    )

    # clean mutation — benign / attack 따로. benign 쪽이 ΔFPR-clean 의 proxy다.
    mutation = {}
    for label in ("benign", "attack"):
        texts = [s.text for s in seeds if s.label == label]
        outputs = [r[0] for r in model.restore_many(texts)]
        mutation[label] = M.clean_mutation_rate(texts, outputs)
    return {
        "eval_examples": len(examples),
        "eval_changed_examples": len(changed),
        "slot_accuracy_changed_only": accuracy,
        "restoration": restoration,
        "clean_mutation": mutation,
    }


def _train_point(technique, corpus_path, args):
    """코퍼스 지점 하나에서 학습하고 학습 쪽 통계를 돌려준다."""
    rows = C.load_corpus(corpus_path)
    items = C.corpus_as_items(rows)
    train_items, _ = C.split_by_hash(items, test_fraction=0.2)
    P.assert_disjoint(train_items, _)

    train_pairs = P.make_pairs(
        train_items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=True
    )
    examples = F.extract_examples(train_pairs, args.window)
    total = len(examples)
    examples = F.subsample(examples, args.max_slot_examples, args.random_seed)

    started = time.time()
    model = SlotRestorer(
        technique, window=args.window, solver="sgd", max_iter=30, random_state=args.random_seed
    ).fit(examples)
    return model, {
        "corpus_sentences": len(rows),
        "train_sentences": len(train_items),
        "slot_examples_total": total,
        "slot_examples_used": len(examples),
        "vocab_size": len(model.vocab),
        "fit_seconds": round(time.time() - started, 2),
    }


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E1_corpus_scaling",
        config={
            "techniques": args.techniques,
            "intensities": list(INTENSITIES),
            "corpus_sizes": args.sizes,
            "corpus_wiki_ratio": DEFAULT_RATIO,
            "ratio_sweep": [] if args.skip_ratio_sweep else list(RATIO_SWEEP),
            "ratio_sweep_size": RATIO_SWEEP_SIZE,
            "window": args.window,
            "max_slot_examples": args.max_slot_examples,
            "model": "sklearn.SGDClassifier(log_loss), alpha=1/(C*n)",
            "corpus_license": C.LICENSE_NOTE,
            "eval_set": "505 dev seeds (학습 미포함)",
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    metrics: dict = {"scaling": {}, "ratio_sweep": {}}

    for size in args.sizes:
        corpus_path = C.build_corpus(size, DEFAULT_RATIO)
        run.add_input(f"corpus_{size}", corpus_path)
        metrics["scaling"][str(size)] = {}
        for technique in args.techniques:
            print(f"[E1] size={size:<7} {technique} ...", flush=True)
            model, train_stats = _train_point(technique, corpus_path, args)
            result = _eval_on_seeds(model, technique, seeds, args)
            result["train"] = train_stats
            metrics["scaling"][str(size)][technique] = result
            acc = result["slot_accuracy_changed_only"]
            mut = result["clean_mutation"]["benign"]["sentence_rate"]
            print(
                f"       vocab {train_stats['vocab_size']:5d} | "
                f"음절exact {acc['syllable_exact']:.3f} | "
                f"benign clean mutation {mut:.1%} | {train_stats['fit_seconds']}s",
                flush=True,
            )

    if not args.skip_ratio_sweep:
        for ratio in RATIO_SWEEP:
            corpus_path = C.build_corpus(RATIO_SWEEP_SIZE, ratio)
            run.add_input(f"corpus_ratio_{int(ratio * 100)}", corpus_path)
            key = f"{ratio:.2f}"
            metrics["ratio_sweep"][key] = {}
            for technique in args.techniques:
                print(f"[E1] ratio={ratio:<5} {technique} ...", flush=True)
                model, train_stats = _train_point(technique, corpus_path, args)
                result = _eval_on_seeds(model, technique, seeds, args)
                result["train"] = train_stats
                metrics["ratio_sweep"][key][technique] = result

    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e1_corpus_scaling_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E1] 완료 -> {run.directory}")


def _curve_table(scaling: dict, techniques: list[str], field) -> list[str]:
    sizes = sorted(scaling, key=int)
    lines = ["| 기법 | " + " | ".join(f"{int(s):,}문장" for s in sizes) + " |",
             "|---" * (len(sizes) + 1) + "|"]
    for technique in techniques:
        cells = []
        for size in sizes:
            entry = scaling[size].get(technique)
            cells.append(field(entry) if entry else "-")
        lines.append(f"| {technique} | " + " | ".join(cells) + " |")
    return lines


def _report(metrics: dict, args: argparse.Namespace) -> str:
    scaling = metrics["scaling"]
    techniques = args.techniques
    lines = [
        "# E1 — 코퍼스 스케일링 곡선",
        "",
        "> 상태: `PROVISIONAL_DEV_ONLY`",
        ">",
        "> 아키텍처를 고정하고 코퍼스 문장 수만 늘렸다. 학습 슬롯 예제 수에 상한을 두었으므로",
        f"> 곡선이 재는 것은 기울기 스텝 수가 아니라 **어휘 폭**이다(상한 {args.max_slot_examples:,}).",
        "",
        "## 실행 조건",
        "",
        f"- 코퍼스: {C.LICENSE_NOTE}, 위키:KLUE = {DEFAULT_RATIO:.0%}:{1 - DEFAULT_RATIO:.0%}",
        "- 모델: sklearn SGDClassifier(log_loss), alpha=1/(C·n), 전 구간 동일",
        f"- 문자 창 +-{args.window}, 강도 {INTENSITIES}, identity 페어 포함",
        "- 평가: 505개 dev 시드 (학습 미포함)",
        "",
        "## 곡선 1 — benign clean mutation rate (낮을수록 좋음, 승격 기준 <=1%)",
        "",
    ]
    lines += _curve_table(
        scaling, techniques,
        lambda e: f"{e['clean_mutation']['benign']['sentence_rate']:.1%}",
    )
    lines += [
        "",
        "## 곡선 2 — 음절 exact (변형된 위치만, 높을수록 좋음)",
        "",
    ]
    lines += _curve_table(
        scaling, techniques,
        lambda e: f"{e['slot_accuracy_changed_only']['syllable_exact']:.3f}",
    )
    lines += ["", "## 어휘 크기", ""]
    lines += _curve_table(scaling, techniques, lambda e: f"{e['train']['vocab_size']:,}")

    if metrics["ratio_sweep"]:
        lines += [
            "",
            f"## 위키:KLUE 혼합 비율 스윕 ({RATIO_SWEEP_SIZE:,}문장 고정)",
            "",
            "benign clean mutation rate / 음절 exact",
            "",
            "| 기법 | " + " | ".join(f"위키 {float(r):.0%}" for r in sorted(metrics["ratio_sweep"])) + " |",
            "|---" * (len(metrics["ratio_sweep"]) + 1) + "|",
        ]
        for technique in techniques:
            cells = []
            for ratio in sorted(metrics["ratio_sweep"]):
                entry = metrics["ratio_sweep"][ratio].get(technique)
                if not entry:
                    cells.append("-")
                    continue
                cells.append(
                    f"{entry['clean_mutation']['benign']['sentence_rate']:.1%}"
                    f" / {entry['slot_accuracy_changed_only']['syllable_exact']:.3f}"
                )
            lines.append(f"| {technique} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론을 하지 않았다. NRR·ΔFPR은 이 문서에 없다.",
        "- 코퍼스(위키·KLUE 문어체)와 평가셋(대화형 명령문)의 분포가 다르다. 전이 성능을",
        "  재는 것이 목적이지만, 시드 쪽 어휘가 코퍼스에 없으면 그만큼 불리하게 나온다.",
        "- 단일 아키텍처(선형) 결과다. 아키텍처 사다리는 E2에서 다룬다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
