"""E6 — 문자 n-gram perplexity 게이트.

연음과 종성 크래밍은 결과물이 문법적으로 정상인 한글 음절이라, 초성체의 호환 자모나
된소리의 경음 밀도 같은 값싼 활성화 신호가 없다. 그래서 후보 위치가 '모든 음절'이 되고
clean mutation이 90%대로 뛴다.

게이트는 정상 코퍼스로 학습한 문자 n-gram LM으로 **어절 단위로 부자연스러운 곳만** 모델에
넘긴다. 재는 것은 셋이다.

1. threshold별 clean 오탐 / 난독화 탐지율 (게이트 자체의 분별력)
2. 게이트를 통과하는 **clean 문자 비율** — clean mutation에 곱해지는 계수라 가장 중요하다
3. 실제 복원기에 게이트를 씌웠을 때 clean mutation과 복원 정확도가 어떻게 움직이는지
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import gate as G
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R
from kslab.models.linear import SlotRestorer

GATED_TECHNIQUES = ("liaison", "jongseong_cram")
INTENSITIES = (0.5, 1.0)
THRESHOLDS = (9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
PROMOTION_MUTATION_BAR = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--context", type=int, default=1)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    return parser.parse_args()


def _score_distributions(lm, texts) -> dict:
    scores = [lm.logscore(t[a:b]) for t in texts for a, b in G.word_spans(t)]
    if not scores:
        return {}
    quantiles = statistics.quantiles(scores, n=100)
    return {
        "words": len(scores),
        "p50": round(statistics.median(scores), 4),
        "p90": round(quantiles[89], 4),
        "p95": round(quantiles[94], 4),
        "p99": round(quantiles[98], 4),
    }


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()
    clean_texts = [s.text for s in seeds]
    benign_texts = [s.text for s in seeds if s.label == "benign"]

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E6_gate",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "ngram_order": args.order,
            "gate_context_words": args.context,
            "thresholds": list(THRESHOLDS),
            "techniques": list(GATED_TECHNIQUES),
            "model": "stupid-backoff 문자 n-gram + 어절 단위 임계값",
            "corpus_license": C.LICENSE_NOTE,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    print(f"[E6] n-gram LM 학습 ({len(train_items):,}문장) ...", flush=True)
    lm = G.CharNgramLM(order=args.order).fit([t for _, _, t in train_items])
    run.note("lm", lm.size())

    metrics: dict = {"score_distributions": {}, "gate_only": {}, "end_to_end": {}}
    metrics["score_distributions"]["clean"] = _score_distributions(lm, clean_texts)

    obfuscated = {}
    for technique in GATED_TECHNIQUES:
        fn = O.TRANSFORMS[technique]
        intensity = 1.0 if technique == "liaison" else 0.5
        obfuscated[technique] = [fn(t, intensity, seed=args.gen_seed) for t in clean_texts]
        metrics["score_distributions"][technique] = _score_distributions(
            lm, obfuscated[technique]
        )

    # --- 1) 게이트 자체의 분별력 ---
    print("[E6] 게이트 임계값 스윕 ...", flush=True)
    for threshold in THRESHOLDS:
        gate = G.PerplexityGate(lm, threshold, context=args.context)
        clean_chars = sum(len(t) for t in clean_texts)
        passed_chars = sum(len(gate.selected_positions(t)) for t in clean_texts)
        entry = {
            "clean_sentence_flag_rate": sum(1 for t in clean_texts if gate.flags(t)) / len(clean_texts),
            "clean_char_pass_rate": passed_chars / clean_chars,
        }
        for technique in GATED_TECHNIQUES:
            changed = [
                o for o, c in zip(obfuscated[technique], clean_texts) if o != c
            ]
            entry[f"{technique}_detection_rate"] = (
                sum(1 for t in changed if gate.flags(t)) / len(changed) if changed else 0.0
            )
        metrics["gate_only"][f"{threshold:.1f}"] = entry

    # --- 2) 실제 복원기에 게이트를 씌운 효과 ---
    for technique in GATED_TECHNIQUES:
        print(f"[E6] {technique} 복원기 학습 ...", flush=True)
        train_pairs = P.make_pairs(
            train_items, [technique], INTENSITIES,
            gen_seed=args.gen_seed, include_identity=True,
        )
        examples = F.subsample(
            F.extract_examples(train_pairs, args.window),
            args.max_slot_examples, args.random_seed,
        )
        model = SlotRestorer(
            technique, window=args.window, solver="sgd", max_iter=30,
            random_state=args.random_seed,
        ).fit(examples)

        obf = obfuscated[technique]
        changed_idx = [i for i, (o, c) in enumerate(zip(obf, clean_texts)) if o != c]
        clusters = [s.seed_id for s in seeds]

        def _measure(restore_benign, restore_obf):
            # 문장 단위 exact는 연음·크래밍처럼 한 문장에서 여러 자리가 바뀌는 기법에서는
            # 거의 항상 0이라 분별력이 없다. CER 감소폭을 주 지표로 함께 본다.
            exact, residual, raw, gain = [], [], [], []
            for i in changed_idx:
                cluster = clusters[i]
                exact.append((cluster, float(restore_obf[i] == clean_texts[i])))
                r_cer = M.char_error_rate(clean_texts[i], restore_obf[i])
                o_cer = M.char_error_rate(clean_texts[i], obf[i])
                residual.append((cluster, r_cer))
                raw.append((cluster, o_cer))
                gain.append((cluster, o_cer - r_cer))
            summarize = lambda obs: M.summarize_observations(
                obs, args.bootstrap_samples, args.random_seed
            )
            return {
                "benign_clean_mutation": M.clean_mutation_rate(benign_texts, restore_benign),
                "exact_restoration": summarize(exact),
                "residual_cer": summarize(residual),
                "raw_cer": summarize(raw),
                "cer_reduction": summarize(gain),
            }

        ungated_benign = [r[0] for r in model.restore_many(benign_texts)]
        ungated_obf = [r[0] for r in model.restore_many(obf)]
        metrics["end_to_end"][technique] = {
            "ungated": _measure(ungated_benign, ungated_obf),
            "gated": {},
        }
        for threshold in THRESHOLDS:
            gate = G.PerplexityGate(lm, threshold, context=args.context)
            metrics["end_to_end"][technique]["gated"][f"{threshold:.1f}"] = _measure(
                G.gated_restore(model, gate, benign_texts),
                G.gated_restore(model, gate, obf),
            )
            row = metrics["end_to_end"][technique]["gated"][f"{threshold:.1f}"]
            print(
                f"       thr={threshold:4.1f} benign mutation "
                f"{row['benign_clean_mutation']['sentence_rate']:6.2%}"
                f"  CER {row['raw_cer']['cluster_balanced_estimate'] or 0:.3f}"
                f"->{row['residual_cer']['cluster_balanced_estimate'] or 0:.3f}"
                f"  exact {row['exact_restoration']['cluster_balanced_estimate'] or 0:.3f}",
                flush=True,
            )

    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e6_gate_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E6] 완료 -> {run.directory}")


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E6 — 문자 n-gram perplexity 게이트",
        "",
        "> 상태: `PROVISIONAL_DEV_ONLY`",
        ">",
        "> 연음·종성크래밍은 값싼 활성화 신호가 없어 후보가 '모든 음절'이 된다. 게이트는",
        "> 어절 단위로 부자연스러운 곳만 모델에 넘겨 그 비용과 오변경을 함께 줄인다.",
        "",
        "## 실행 조건",
        "",
        f"- LM: stupid-backoff 문자 {args.order}-gram, 코퍼스 {args.corpus_size:,}문장"
        f" (위키 {args.wiki_ratio:.0%})",
        f"- 게이트 단위: 어절, 인접 {args.context}어절까지 함께 통과",
        f"- {C.LICENSE_NOTE}",
        "- 평가: 505개 dev 시드 (학습 미포함)",
        "",
        "## 어절 점수 분포",
        "",
        "| 입력 | p50 | p90 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for name, dist in metrics["score_distributions"].items():
        if dist:
            lines.append(
                f"| {name} | {dist['p50']:.2f} | {dist['p90']:.2f} | "
                f"{dist['p95']:.2f} | {dist['p99']:.2f} |"
            )
    lines += [
        "",
        "## 게이트 단독 성능",
        "",
        "`clean 문자 통과율`이 가장 중요한 열이다 — clean mutation에 곱해지는 계수다.",
        "",
        "| 임계값 | clean 문장 오탐 | **clean 문자 통과율** | "
        + " | ".join(f"{t} 탐지" for t in GATED_TECHNIQUES) + " |",
        "|---|---|---|" + "---|" * len(GATED_TECHNIQUES),
    ]
    for threshold, row in metrics["gate_only"].items():
        cells = " | ".join(
            f"{row[f'{t}_detection_rate']:.1%}" for t in GATED_TECHNIQUES
        )
        lines.append(
            f"| {threshold} | {row['clean_sentence_flag_rate']:.1%} | "
            f"**{row['clean_char_pass_rate']:.1%}** | {cells} |"
        )

    lines += ["", "## 복원기 + 게이트 (end-to-end)", ""]
    for technique, entry in metrics["end_to_end"].items():
        ungated = entry["ungated"]
        def _cells(row):
            return (
                f"{row['benign_clean_mutation']['sentence_rate']:.1%}",
                f"{row['residual_cer']['cluster_balanced_estimate'] or 0:.4f}",
                f"{(row['cer_reduction']['cluster_balanced_estimate'] or 0) * 100:+.2f}%p",
                f"{row['exact_restoration']['cluster_balanced_estimate'] or 0:.3f}",
            )

        raw_cer = ungated["raw_cer"]["cluster_balanced_estimate"] or 0
        lines += [
            f"### {technique}",
            "",
            f"난독화 원본 CER {raw_cer:.4f}. CER 감소폭이 양수여야 복원이 도움이 된 것이다.",
            "",
            "| 설정 | benign clean mutation | residual CER | CER 감소 | exact |",
            "|---|---|---|---|---|",
            "| 게이트 없음 | " + " | ".join(_cells(ungated)) + " |",
        ]
        for threshold, row in entry["gated"].items():
            mutation = row["benign_clean_mutation"]["sentence_rate"]
            mark = " ✅" if mutation <= PROMOTION_MUTATION_BAR else ""
            cells = list(_cells(row))
            cells[0] += mark
            lines.append(f"| 임계값 {threshold} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## 해석 제한",
        "",
        "- 가드레일 추론 없음. clean mutation은 ΔFPR의 proxy다.",
        "- 게이트는 정상 코퍼스만 보고 학습했다. 공격자가 게이트를 알고 자연스러운 표기를",
        "  고르면 탐지율이 떨어진다 — 이 실험은 그 적응적 공격을 다루지 않는다.",
        "- 연음은 결과물이 정상 한국어와 점수 분포가 거의 겹쳐서 게이트의 분별력이 약하다.",
        "  종성 크래밍 쪽이 훨씬 잘 갈린다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
