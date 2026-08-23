"""E8 — 아키텍처 × abstention 임계값. E2와 E3의 합류점.

E2는 아키텍처가 오변경을 크게 줄인다는 것을(liaison 33.8%→5.9%), E3는 abstention이
오변경을 줄인다는 것을(된소리 14.7%→0.98%) 각각 보였다. 둘은 서로 다른 축이므로 함께
걸 수 있다. 이 실험은 그 조합이 승격 기준(clean mutation ≤1%)에 닿는지를 본다.

E3에서 배운 것 하나를 그대로 적용한다: **오변경 기준만 보면 안 된다.** 임계값을 충분히
올리면 어떤 모델이든 "아무것도 안 해서" 0%를 찍는다. 그래서 쓸모 기준(최소 복원율)을
함께 걸고, 두 기준을 동시에 넘는 점이 있는지로 판정한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import record as R
from kslab.models.bilstm import BiLstmSlotRestorer
from kslab.models.linear import SlotRestorer
from kslab.models.mlp import MlpSlotRestorer

TECHNIQUES = ("tensify", "liaison", "jongseong_cram")
INTENSITIES = (0.5, 1.0)
THRESHOLDS = (
    0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999,
    # 스모크에서 0.9999에서도 오변경이 2.94%였다. 1% 바에 닿는지 보려면 더 위가 필요하다.
    0.99999, 0.999999, 0.9999999,
)
MUTATION_BAR = 0.01
MIN_CER_REDUCTION = 0.02   # 최소 쓸모: CER을 최소 2%p는 줄여야 한다


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--techniques", nargs="*", default=list(TECHNIQUES))
    parser.add_argument("--architectures", nargs="*", default=["mlp", "bilstm"])
    return parser.parse_args()


def _train(architecture, technique, train_pairs, args):
    started = time.time()
    if architecture == "bilstm":
        model = BiLstmSlotRestorer(
            technique, max_len=96, embed_dim=96, hidden=112, layers=2,
            epochs=5, batch_size=128, random_state=args.random_seed, threads=args.threads,
        ).fit(train_pairs)
    else:
        examples = F.subsample(
            F.extract_examples(train_pairs, args.window),
            args.max_slot_examples, args.random_seed,
        )
        cls = MlpSlotRestorer if architecture == "mlp" else SlotRestorer
        kwargs = (
            {"epochs": 8, "threads": args.threads}
            if architecture == "mlp"
            else {"solver": "sgd", "max_iter": 30}
        )
        model = cls(
            technique, window=args.window, random_state=args.random_seed, **kwargs
        ).fit(examples)
    return model, round(time.time() - started, 1)


def _sweep(model, technique, seeds, args) -> dict:
    benign = [s.text for s in seeds if s.label == "benign"]
    clean = [s.text for s in seeds]
    clusters = [s.seed_id for s in seeds]
    fn = __import__("kslab.obfuscate", fromlist=["x"]).TRANSFORMS[technique]
    obfuscated = [fn(t, 1.0, seed=args.gen_seed) for t in clean]
    changed = [i for i, (o, c) in enumerate(zip(obfuscated, clean)) if o != c]

    rows = {}
    for threshold in THRESHOLDS:
        benign_out = [r[0] for r in model.restore_many(benign, threshold)]
        restored = [r[0] for r in model.restore_many(obfuscated, threshold)]
        exact, residual, raw, gain = [], [], [], []
        for i in changed:
            exact.append((clusters[i], float(restored[i] == clean[i])))
            r_cer = M.char_error_rate(clean[i], restored[i])
            o_cer = M.char_error_rate(clean[i], obfuscated[i])
            residual.append((clusters[i], r_cer))
            raw.append((clusters[i], o_cer))
            gain.append((clusters[i], o_cer - r_cer))
        summarize = lambda obs: M.summarize_observations(
            obs, args.bootstrap_samples, args.random_seed
        )
        touched = sum(1 for i in changed if restored[i] != obfuscated[i])
        # 키를 .4f 로 만들면 0.99999·0.999999·0.9999999 가 전부 "1.0000" 으로 뭉개져
        # 서로 덮어쓴다. 임계값을 그대로 식별할 수 있는 정밀도를 쓴다.
        rows[f"{threshold:.7f}"] = {
            "benign_clean_mutation": M.clean_mutation_rate(benign, benign_out),
            "exact_restoration": summarize(exact),
            "residual_cer": summarize(residual),
            "raw_cer": summarize(raw),
            "cer_reduction": summarize(gain),
            "touched_rate": touched / len(changed) if changed else 0.0,
        }
    return rows


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E8_abstention",
        config={
            "corpus_size": args.corpus_size,
            "techniques": args.techniques,
            "architectures": args.architectures,
            "thresholds": list(THRESHOLDS),
            "mutation_bar": MUTATION_BAR,
            "min_cer_reduction": MIN_CER_REDUCTION,
            "corpus_license": C.LICENSE_NOTE,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    metrics: dict = {"sweeps": {}, "passing": []}
    for technique in args.techniques:
        train_pairs = P.make_pairs(
            train_items, [technique], INTENSITIES,
            gen_seed=args.gen_seed, include_identity=True,
        )
        metrics["sweeps"][technique] = {}
        for architecture in args.architectures:
            print(f"[E8] {technique} / {architecture} 학습 ...", flush=True)
            model, seconds = _train(architecture, technique, train_pairs, args)
            sweep = _sweep(model, technique, seeds, args)
            metrics["sweeps"][technique][architecture] = {
                "fit_seconds": seconds, "thresholds": sweep,
            }
            for threshold, row in sweep.items():
                mutation = row["benign_clean_mutation"]["sentence_rate"]
                reduction = row["cer_reduction"]["cluster_balanced_estimate"] or 0.0
                if mutation <= MUTATION_BAR and reduction >= MIN_CER_REDUCTION:
                    metrics["passing"].append({
                        "technique": technique, "architecture": architecture,
                        "threshold": threshold, "benign_clean_mutation": mutation,
                        "cer_reduction": reduction,
                        "exact": row["exact_restoration"]["cluster_balanced_estimate"],
                    })
            best = min(
                sweep.items(),
                key=lambda kv: abs(kv[1]["benign_clean_mutation"]["sentence_rate"] - MUTATION_BAR),
            )
            print(
                f"       1% 근처 thr={best[0]}  "
                f"mutation {best[1]['benign_clean_mutation']['sentence_rate']:.2%}  "
                f"CER감소 {(best[1]['cer_reduction']['cluster_balanced_estimate'] or 0) * 100:+.2f}%p  "
                f"({seconds}s)",
                flush=True,
            )

    metrics["status"] = "PROMOTE_CANDIDATE" if metrics["passing"] else "DO_NOT_PROMOTE"
    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e8_abstention_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E8] 완료 -> {run.directory}  판정 {metrics['status']}")
    for entry in metrics["passing"]:
        print(f"  통과: {entry['technique']}/{entry['architecture']} thr={entry['threshold']} "
              f"mutation {entry['benign_clean_mutation']:.2%} CER감소 {entry['cer_reduction'] * 100:+.2f}%p")


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E8 — 아키텍처 × abstention",
        "",
        f"> 상태: `{metrics['status']}`",
        ">",
        "> E2(아키텍처)와 E3(abstention)는 서로 다른 축이라 함께 걸 수 있다. 그 조합이",
        f"> 승격 기준(clean mutation ≤{MUTATION_BAR:.0%})에 닿는지 본다.",
        "",
        "## 승격 기준",
        "",
        f"1. benign clean mutation ≤ {MUTATION_BAR:.0%}",
        f"2. **그리고** CER 감소 ≥ {MIN_CER_REDUCTION:.0%}p — 임계값을 올리면 어떤 모델이든",
        "   '아무것도 안 해서' 오변경 0%를 찍으므로 쓸모 기준을 함께 건다(E3에서 배운 것).",
        "",
        f"- 코퍼스 {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        "- 평가: 505 dev 시드, intensity 1.0",
        "",
    ]
    for technique, archs in metrics["sweeps"].items():
        lines += [f"## {technique}", ""]
        for architecture, entry in archs.items():
            lines += [
                f"### {architecture} (학습 {entry['fit_seconds']}s)",
                "",
                "| 임계값 | benign mutation | CER | CER 감소 | exact | 손댄 비율 |",
                "|---|---|---|---|---|---|",
            ]
            for threshold, row in entry["thresholds"].items():
                mutation = row["benign_clean_mutation"]["sentence_rate"]
                reduction = row["cer_reduction"]["cluster_balanced_estimate"] or 0.0
                ok = mutation <= MUTATION_BAR and reduction >= MIN_CER_REDUCTION
                lines.append(
                    f"| {threshold}{' ✅' if ok else ''} | {mutation:.2%} | "
                    f"{row['residual_cer']['cluster_balanced_estimate'] or 0:.4f} | "
                    f"{reduction * 100:+.2f}%p | "
                    f"{row['exact_restoration']['cluster_balanced_estimate'] or 0:.3f} | "
                    f"{row['touched_rate']:.1%} |"
                )
            lines.append("")

    lines += ["## 판정", ""]
    if metrics["passing"]:
        lines += ["두 기준을 동시에 넘는 조합:", "",
                  "| 기법 | 아키텍처 | 임계값 | mutation | CER 감소 | exact |", "|---|---|---|---|---|---|"]
        for e in metrics["passing"]:
            lines.append(
                f"| {e['technique']} | {e['architecture']} | {e['threshold']} | "
                f"{e['benign_clean_mutation']:.2%} | {e['cer_reduction'] * 100:+.2f}%p | "
                f"{e['exact'] or 0:.3f} |"
            )
    else:
        lines += [
            "두 기준을 동시에 넘는 조합이 **없다.** 아키텍처와 abstention을 함께 걸어도",
            "오변경을 1% 아래로 누르는 지점에서는 복원이 의미 있게 남지 않는다.",
        ]
    lines += [
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론 없음. clean mutation은 ΔFPR의 proxy다.",
        "- intensity 1.0 단일 조건이다. 약한 난독화에서는 다르게 나올 수 있다.",
        "- 단일 실행이라 임계값별 작은 차이는 분산으로 읽어야 한다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
