"""E2 — 아키텍처 사다리와 identity 페어 ablation.

같은 코퍼스·같은 분할·같은 후보 규칙에서 세 아키텍처를 비교한다.

1. `linear`  — 문자 창 one-hot + 로지스틱(SGD). E1에서 곡선을 그린 그 모델.
2. `mlp`     — 자모 임베딩 + MLP(BatchNorm + skip), 초/중/종 head가 몸통 공유.
3. `bilstm`  — 문장 전체를 양방향으로 읽는다. 고정 창을 넘는 문맥이 필요한지 보는 쪽.

ablation 하나: **identity 페어(clean→clean) 포함 여부**. clean mutation rate에 직결되는
신호라 실제로 얼마나 기여하는지 수치로 확인한다.

BiLSTM은 CPU에서 느리다. 그래서 E2는 코퍼스 크기를 하나로 고정하고(기본 8,000문장)
아키텍처 **상대 비교**만 한다 — 절대 수치는 E1의 최적 지점보다 낮게 나온다.
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
from kslab.models.bilstm import BiLstmSlotRestorer
from kslab.models.linear import SlotRestorer
from kslab.models.mlp import MlpSlotRestorer

TECHNIQUES = ("tensify", "chosung", "liaison", "jongseong_cram")
ARCHITECTURES = ("linear", "mlp", "bilstm")
INTENSITIES = (0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=8000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--techniques", nargs="*", default=list(TECHNIQUES))
    parser.add_argument("--architectures", nargs="*", default=list(ARCHITECTURES))
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="identity 페어 ablation을 건너뛴다 (시간 절약용).",
    )
    return parser.parse_args()


def _build(architecture, technique, train_pairs, args):
    """아키텍처 하나를 학습하고 (모델, 학습통계)를 돌려준다."""
    started = time.time()
    if architecture == "bilstm":
        model = BiLstmSlotRestorer(
            technique, max_len=96, embed_dim=96, hidden=112, layers=2,
            epochs=5, batch_size=128, random_state=args.random_seed, threads=args.threads,
        ).fit(train_pairs)
        used = len(train_pairs)
        params = model.parameter_count()
    else:
        examples = F.extract_examples(train_pairs, args.window)
        total = len(examples)
        examples = F.subsample(examples, args.max_slot_examples, args.random_seed)
        used = len(examples)
        if architecture == "linear":
            model = SlotRestorer(
                technique, window=args.window, solver="sgd", max_iter=30,
                random_state=args.random_seed,
            ).fit(examples)
            params = None
        else:
            model = MlpSlotRestorer(
                technique, window=args.window, epochs=8,
                random_state=args.random_seed, threads=args.threads,
            ).fit(examples)
            params = model.parameter_count()
        used = f"{used}/{total}"
    return model, {
        "fit_seconds": round(time.time() - started, 1),
        "train_units": used,
        "vocab_size": len(model.vocab),
        "parameters": params,
    }


def _evaluate(model, architecture, technique, seeds, args) -> dict:
    items = P.seeds_as_items(seeds)
    eval_pairs = P.make_pairs(
        items, [technique], INTENSITIES, gen_seed=args.gen_seed, include_identity=False
    )
    examples = [
        e for e in F.extract_examples(eval_pairs, max(args.window, 1))
        if e.technique == technique
    ]
    changed = [e for e in examples if e.changed]

    obf = [p.obfuscated for p in eval_pairs]
    clean = [p.clean for p in eval_pairs]
    clusters = [p.cluster_id for p in eval_pairs]

    started = time.time()
    restored = [r[0] for r in model.restore_many(obf)]
    infer_seconds = time.time() - started

    if architecture == "bilstm":
        predicted = model.predict_slots_for(obf, changed) if changed else []
    else:
        predicted = model.predict_slots(changed) if changed else []
    accuracy = (
        M.slot_accuracy(
            [tuple(int(v) for v in row) for row in predicted],
            [e.target_slots for e in changed],
        )
        if changed
        else {"cho": 0.0, "jung": 0.0, "jong": 0.0, "syllable_exact": 0.0, "n": 0}
    )

    mutation = {}
    for label in ("benign", "attack"):
        texts = [s.text for s in seeds if s.label == label]
        mutation[label] = M.clean_mutation_rate(
            texts, [r[0] for r in model.restore_many(texts)]
        )

    # CPU 지연: 문장 1건 기준. 승격 기준(MLP 100ms / BiLSTM 200ms)과 대조한다.
    sample = clean[:200]
    latencies = []
    for text in sample:
        t0 = time.perf_counter()
        model.restore_many([text])
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "slot_accuracy_changed_only": accuracy,
        "restoration": M.restoration_metrics(
            clean, obf, restored, clusters, args.bootstrap_samples, args.random_seed
        ),
        "clean_mutation": mutation,
        "latency_ms": {
            "p50": M.percentile(latencies, 0.50),
            "p95": M.percentile(latencies, 0.95),
            "p99": M.percentile(latencies, 0.99),
            "batched_total_seconds": round(infer_seconds, 2),
            "batched_sentences": len(obf),
        },
    }


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E2_architecture",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "techniques": args.techniques,
            "architectures": args.architectures,
            "intensities": list(INTENSITIES),
            "window": args.window,
            "max_slot_examples": args.max_slot_examples,
            "threads": args.threads,
            "corpus_license": C.LICENSE_NOTE,
            "ablation": not args.skip_ablation,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    metrics: dict = {"ladder": {}, "identity_ablation": {}}

    for technique in args.techniques:
        train_pairs = P.make_pairs(
            train_items, [technique], INTENSITIES,
            gen_seed=args.gen_seed, include_identity=True,
        )
        metrics["ladder"][technique] = {}
        for architecture in args.architectures:
            print(f"[E2] {technique} / {architecture} ...", flush=True)
            model, train_stats = _build(architecture, technique, train_pairs, args)
            result = _evaluate(model, architecture, technique, seeds, args)
            result["train"] = train_stats
            metrics["ladder"][technique][architecture] = result
            print(
                f"       음절exact {result['slot_accuracy_changed_only']['syllable_exact']:.3f}"
                f" | benign mutation {result['clean_mutation']['benign']['sentence_rate']:.1%}"
                f" | p95 {result['latency_ms']['p95']:.1f}ms"
                f" | fit {train_stats['fit_seconds']}s",
                flush=True,
            )

    if not args.skip_ablation:
        # identity 페어를 뺐을 때 clean mutation이 얼마나 나빠지는가 (linear·mlp만)
        for technique in args.techniques:
            no_identity = P.make_pairs(
                train_items, [technique], INTENSITIES,
                gen_seed=args.gen_seed, include_identity=False,
            )
            metrics["identity_ablation"][technique] = {}
            for architecture in [a for a in args.architectures if a != "bilstm"]:
                print(f"[E2-ablation] {technique} / {architecture} (identity 없음) ...", flush=True)
                model, _stats = _build(architecture, technique, no_identity, args)
                result = _evaluate(model, architecture, technique, seeds, args)
                metrics["identity_ablation"][technique][architecture] = {
                    "clean_mutation": result["clean_mutation"],
                    "slot_accuracy_changed_only": result["slot_accuracy_changed_only"],
                }

    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e2_architecture_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E2] 완료 -> {run.directory}")


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E2 — 아키텍처 사다리",
        "",
        "> 상태: `PROVISIONAL_DEV_ONLY`",
        ">",
        f"> 코퍼스를 {args.corpus_size:,}문장으로 고정한 **상대 비교**다. BiLSTM이 CPU에서",
        "> 느려 크기를 하나로 묶었으므로, 절대 수치는 E1의 최적 지점보다 낮게 나온다.",
        "",
        "## 실행 조건",
        "",
        f"- 코퍼스 {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        f"- linear/mlp: 문자 창 +-{args.window}, 슬롯 예제 상한 {args.max_slot_examples:,}",
        "- bilstm: 시퀀스 상한 96, embed 96 / hidden 112 / 2층, 5 epoch",
        f"- torch 스레드 {args.threads}, 평가는 505 dev 시드 (학습 미포함)",
        "",
        "## 결과",
        "",
        "| 기법 | 아키텍처 | 음절 exact | benign clean mutation | p95 지연 | 학습 시간 | 파라미터 |",
        "|---|---|---|---|---|---|---|",
    ]
    for technique, archs in metrics["ladder"].items():
        for architecture, r in archs.items():
            params = r["train"]["parameters"]
            param_cell = f"{params:,}" if params else "-"
            lines.append(
                f"| {technique} | {architecture} | "
                f"{r['slot_accuracy_changed_only']['syllable_exact']:.3f} | "
                f"{r['clean_mutation']['benign']['sentence_rate']:.1%} | "
                f"{r['latency_ms']['p95']:.1f}ms | "
                f"{r['train']['fit_seconds']}s | {param_cell} |"
            )

    if metrics["identity_ablation"]:
        lines += [
            "",
            "## identity 페어 ablation — benign clean mutation",
            "",
            "| 기법 | 아키텍처 | identity 포함 | identity 제외 | 차이 |",
            "|---|---|---|---|---|",
        ]
        for technique, archs in metrics["identity_ablation"].items():
            for architecture, r in archs.items():
                with_id = metrics["ladder"][technique][architecture]["clean_mutation"]["benign"][
                    "sentence_rate"
                ]
                without = r["clean_mutation"]["benign"]["sentence_rate"]
                lines.append(
                    f"| {technique} | {architecture} | {with_id:.1%} | {without:.1%} | "
                    f"{(without - with_id) * 100:+.1f}%p |"
                )

    lines += [
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론 없음. clean mutation은 ΔFPR의 proxy다.",
        "- 코퍼스 크기를 하나로 고정했으므로 아키텍처 순위는 이 크기에서만 유효하다.",
        "  더 큰 코퍼스에서 순위가 뒤집힐 수 있다.",
        "- p95 지연은 문장 1건씩 호출했을 때다. 실제 provider는 스팬을 묶어 배치로 돌린다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
