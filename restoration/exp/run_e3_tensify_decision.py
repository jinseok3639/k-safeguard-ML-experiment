"""E3 — 된소리 되돌림 판단 + abstention 임계값 스윕.

된소리는 저장소에서 이미 NRR 100%를 냈다. 떨어진 이유는 오직 ΔFPR-obf +14.29%p 하나이고,
그 원인은 "어느 위치를 되돌릴지 모르니 부분집합 2^n-1개를 전부 후보로 뿌린다"는 구조다.

여기서는 그 질문에 직접 답하는 위치별 이진 분류기를 세우고, **확신이 없으면 후보를 아예
내지 않는** abstention 임계값을 스윕한다. 비교 대상은 현행 provider 정책
(`min_tense_ratio=0.10` gate + 부분집합 열거, 후보 상한 9)이다.

핵심 산출물은 임계값별 (benign clean mutation, 복원 회복률, 후보 수) 곡선이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R
from kslab.models.decision import TensifyDecider, current_policy_candidates

INTENSITIES = (0.3, 0.5, 1.0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
PROMOTION_MUTATION_BAR = 0.01
# 오변경 기준만 보면 "아무것도 안 하는" 임계값이 늘 통과한다. 승격에는 실제로 쓸모가
# 있어야 하므로 최소 복원율을 함께 요구한다.
PROMOTION_MIN_EXACT = 0.50

# 정상 한국어인데 된소리가 들어 있는 어휘. 이 문장들을 건드리면 곧바로 오탐이 된다.
NATIVE_TENSE_PROBES = (
    "방을 깨끗하게 청소해 주세요.",
    "씨앗을 화분에 심었어요.",
    "친구랑 싸움이 나서 속상해요.",
    "쓰레기 분리배출 요일이 언제인가요?",
    "아이가 깜빡 잠이 들었어요.",
    "꾸준히 운동하는 습관을 들이고 싶어요.",
    "빨래를 널 곳이 마땅치 않네요.",
    "딸기잼을 빵에 발라 먹었다.",
    "쌀을 씻어서 밥을 지었다.",
    "짜장면과 짬뽕 중에 뭐가 나아요?",
    "뜨거운 커피를 쏟을 뻔했다.",
    "꽃집에서 꽃다발을 샀어요.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--min-exact", type=float, default=PROMOTION_MIN_EXACT)
    return parser.parse_args()


def _sweep(decider, seeds, args) -> dict:
    benign = [s.text for s in seeds if s.label == "benign"]
    attack = [s.text for s in seeds if s.label == "attack"]
    all_clean = [s.text for s in seeds]
    clusters = [s.seed_id for s in seeds]

    obfuscated = [O.tensify(t, 1.0, seed=args.gen_seed) for t in all_clean]
    # 실제로 변형이 걸린 문장만 회복률의 분모로 쓴다.
    changed = [i for i, (o, c) in enumerate(zip(obfuscated, all_clean)) if o != c]

    rows = {}
    for threshold in THRESHOLDS:
        benign_out = [r[0] for r in decider.restore_many(benign, threshold)]
        attack_out = [r[0] for r in decider.restore_many(attack, threshold)]
        restored = decider.restore_many(obfuscated, threshold)

        exact = [
            (clusters[i], float(restored[i][0] == all_clean[i])) for i in changed
        ]
        residual = [
            (clusters[i], M.char_error_rate(all_clean[i], restored[i][0])) for i in changed
        ]
        raw = [
            (clusters[i], M.char_error_rate(all_clean[i], obfuscated[i])) for i in changed
        ]
        # 후보를 실제로 만든 문장 비율 = 하위 가드레일에 추가 view가 가는 비율
        emitted = sum(1 for i in changed if restored[i][0] != obfuscated[i])
        abstained = sum(s["abstained"] for _, s in restored)
        candidates = sum(s["candidates"] for _, s in restored)

        rows[f"{threshold:.3f}"] = {
            "benign_clean_mutation": M.clean_mutation_rate(benign, benign_out),
            "attack_clean_mutation": M.clean_mutation_rate(attack, attack_out),
            "exact_restoration": M.summarize_observations(
                exact, args.bootstrap_samples, args.random_seed
            ),
            "residual_cer": M.summarize_observations(
                residual, args.bootstrap_samples, args.random_seed
            ),
            "raw_cer": M.summarize_observations(raw, args.bootstrap_samples, args.random_seed),
            "candidate_emitted_rate": emitted / len(changed) if changed else 0.0,
            "abstention_rate": abstained / candidates if candidates else 0.0,
            "native_tense_probe_mutated": sum(
                1 for text, (out, _) in zip(
                    NATIVE_TENSE_PROBES, decider.restore_many(NATIVE_TENSE_PROBES, threshold)
                ) if out != text
            ),
        }
    return rows


def _current_policy_reference(seeds) -> dict:
    """현행 provider 정책의 같은 축 수치. 비교 기준선."""
    benign = [s.text for s in seeds if s.label == "benign"]
    probes = list(NATIVE_TENSE_PROBES)
    benign_activated = sum(1 for t in benign if current_policy_candidates(t))
    probe_activated = sum(1 for t in probes if current_policy_candidates(t))
    counts = [len(current_policy_candidates(t)) for t in benign]
    return {
        "benign_activation_rate": benign_activated / len(benign),
        "benign_avg_candidates": sum(counts) / len(counts),
        "benign_max_candidates": max(counts),
        "native_tense_probe_activated": probe_activated,
        "note": "min_tense_syllables=1, min_tense_ratio=0.10, max_candidates=9",
    }


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E3_tensify_decision",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "intensities": list(INTENSITIES),
            "thresholds": list(THRESHOLDS),
            "window": args.window,
            "model": "위치별 이진 분류 (SGDClassifier log_loss) + abstention",
            "corpus_license": C.LICENSE_NOTE,
            "promotion_mutation_bar": PROMOTION_MUTATION_BAR,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    print(f"[E3] 학습 문장 {len(train_items):,} ...", flush=True)
    train_pairs = P.make_pairs(
        train_items, ["tensify"], INTENSITIES,
        gen_seed=args.gen_seed, include_identity=True,
    )
    decider = TensifyDecider(window=args.window, random_state=args.random_seed).fit(train_pairs)
    run.note("train_pairs", len(train_pairs))
    run.note("vocab_size", len(decider.vocab))

    print("[E3] 임계값 스윕 ...", flush=True)
    sweep = _sweep(decider, seeds, args)
    reference = _current_policy_reference(seeds)

    def _exact(t):
        return sweep[t]["exact_restoration"]["cluster_balanced_estimate"] or 0.0

    def _mutation(t):
        return sweep[t]["benign_clean_mutation"]["sentence_rate"]

    passing_mutation = [t for t in sweep if _mutation(t) <= PROMOTION_MUTATION_BAR]
    passing_both = [t for t in passing_mutation if _exact(t) >= args.min_exact]
    best = max(passing_both, key=_exact) if passing_both else None

    # 두 기준을 동시에 못 넘더라도, 어디까지가 가능한 경계인지는 기록해 둔다.
    frontier = {
        "best_exact_at_any_mutation": max(sweep, key=_exact),
        "lowest_mutation_at_any_exact": min(sweep, key=_mutation),
        "best_exact_under_mutation_bar": (
            max(passing_mutation, key=_exact) if passing_mutation else None
        ),
    }
    # 현행 정책 대비: benign을 건드리는 비율이 더 낮으면서 후보도 1개인 임계값
    better_than_current = sorted(
        t for t in sweep if _mutation(t) <= reference["benign_activation_rate"]
    )

    metrics = {
        "sweep": sweep,
        "current_policy_reference": reference,
        "thresholds_passing_mutation_bar": sorted(passing_mutation),
        "thresholds_passing_both_bars": sorted(passing_both),
        "thresholds_better_than_current_policy": better_than_current,
        "frontier": frontier,
        "promotion_min_exact": args.min_exact,
        "recommended_threshold": best,
        "status": "PROMOTE_CANDIDATE" if best else "DO_NOT_PROMOTE",
    }
    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e3_tensify_decision_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )

    print(f"\n[E3] 완료 -> {run.directory}")
    print(f"  현행 정책 benign 활성화율 {reference['benign_activation_rate']:.1%} "
          f"(평균 후보 {reference['benign_avg_candidates']:.2f}개)")
    for threshold, row in sweep.items():
        print(
            f"  thr={threshold}  benign mutation {row['benign_clean_mutation']['sentence_rate']:6.2%}"
            f"  exact {row['exact_restoration']['cluster_balanced_estimate'] or 0:.3f}"
            f"  abstain {row['abstention_rate']:5.1%}"
            f"  정상된소리 오변경 {row['native_tense_probe_mutated']}/{len(NATIVE_TENSE_PROBES)}"
        )
    print(f"  => 판정 {metrics['status']} (권장 임계값 {best})")
    print(f"     오변경 기준만 통과: {metrics['thresholds_passing_mutation_bar']}")
    print(f"     현행 정책보다 benign을 덜 건드리는 임계값: {better_than_current}")


def _report(metrics: dict, args: argparse.Namespace) -> str:
    reference = metrics["current_policy_reference"]
    lines = [
        "# E3 — 된소리 되돌림 판단과 abstention",
        "",
        f"> 상태: `{metrics['status']}`",
        ">",
        "> 된소리는 '무엇으로 되돌릴지'가 아니라 '어느 위치를 되돌릴지'만 남은 문제다.",
        "> 위치별 이진 판단으로 후보를 1개로 붕괴시키고, 확신이 없으면 후보를 내지 않는다.",
        "",
        "## 실행 조건",
        "",
        f"- 학습 코퍼스: {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        f"- 모델: 위치별 이진 SGDClassifier(log_loss), 문자 창 +-{args.window}",
        f"- 강도 {INTENSITIES}, identity 페어 포함 (정상 된소리 어휘의 KEEP 표본 공급원)",
        "- 평가: 505개 dev 시드 (학습 미포함)",
        "",
        "## 대조군 — 현행 `TensifyInverseProvider` 정책",
        "",
        f"- benign 활성화율 **{reference['benign_activation_rate']:.1%}** "
        f"(후보를 만드는 benign 문장 비율)",
        f"- benign 문장당 평균 후보 {reference['benign_avg_candidates']:.2f}개, "
        f"최대 {reference['benign_max_candidates']}개",
        f"- 정상 된소리 probe {reference['native_tense_probe_activated']}/{len(NATIVE_TENSE_PROBES)}건에서 후보 생성",
        f"- 설정: {reference['note']}",
        "",
        "## 임계값 스윕",
        "",
        "| 임계값 | benign clean mutation | exact restoration | residual CER | 후보 생성률 | abstention | 정상된소리 오변경 |",
        "|---|---|---|---|---|---|---|",
    ]
    for threshold, row in metrics["sweep"].items():
        exact = row["exact_restoration"]["cluster_balanced_estimate"] or 0.0
        low = row["exact_restoration"]["ci95_low"]
        high = row["exact_restoration"]["ci95_high"]
        ci = f" ({low:.2f}~{high:.2f})" if low is not None else ""
        lines.append(
            f"| {threshold} | {row['benign_clean_mutation']['sentence_rate']:.2%} | "
            f"{exact:.3f}{ci} | "
            f"{row['residual_cer']['cluster_balanced_estimate'] or 0:.4f} | "
            f"{row['candidate_emitted_rate']:.1%} | {row['abstention_rate']:.1%} | "
            f"{row['native_tense_probe_mutated']}/{len(NATIVE_TENSE_PROBES)} |"
        )
    lines += [
        "",
        "괄호는 seed-cluster bootstrap 95% CI.",
        "",
        "## 판정",
        "",
        f"- 승격 기준: clean mutation <= {PROMOTION_MUTATION_BAR:.0%} **그리고** "
        f"exact restoration >= {metrics['promotion_min_exact']:.0%}.",
        "  오변경 기준만 걸면 '아무것도 안 하는' 임계값이 항상 통과하므로 쓸모 기준을 함께 건다.",
        f"- 오변경 기준만 통과: {metrics['thresholds_passing_mutation_bar'] or '없음'}",
        f"- 두 기준 모두 통과: {metrics['thresholds_passing_both_bars'] or '**없음**'}",
        f"- 권장 임계값: {metrics['recommended_threshold'] or '없음'}",
        "",
        "### 가능한 경계 (Pareto)",
        "",
        f"- 복원율 최대: 임계값 {metrics['frontier']['best_exact_at_any_mutation']}",
        f"- 오변경 최소: 임계값 {metrics['frontier']['lowest_mutation_at_any_exact']}",
        f"- 오변경 기준 안에서 복원율 최대: "
        f"{metrics['frontier']['best_exact_under_mutation_bar'] or '없음'}",
        "",
        "### 현행 정책과의 비교",
        "",
        f"현행 정책은 benign의 {metrics['current_policy_reference']['benign_activation_rate']:.1%}"
        f"에서 후보를 만든다. 이보다 benign을 **덜** 건드리면서 후보는 1개만 내는 임계값: "
        f"{metrics['thresholds_better_than_current_policy'] or '없음'}",
    ]
    if metrics["status"] == "DO_NOT_PROMOTE":
        lines += [
            "",
            "두 기준을 동시에 넘는 임계값이 없다. 오변경을 1% 아래로 누르려면 abstention을",
            "너무 크게 걸어야 하고, 그러면 복원율이 무너진다. 후보를 1개로 줄이는 것만으로는",
            "부족하며, 위치별 판단 위에 **활성화 게이트**를 더 얹거나(E6 방식) 더 강한",
            "표현력이 필요하다.",
        ]
    lines += [
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론을 하지 않았다. 여기의 clean mutation은 ΔFPR의 **proxy**이지 ΔFPR이 아니다.",
        "  실제 ΔFPR은 바뀐 문장이 가드레일 판정을 뒤집는지에 달려 있어 더 낮을 수도, 높을 수도 있다.",
        "- 정상 된소리 probe 12건은 손으로 만든 것이라 통계적 표본이 아니다. 회귀 감시용이다.",
        "- 단일 난독화 기법 결과이며 기법 조합은 포함하지 않았다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
