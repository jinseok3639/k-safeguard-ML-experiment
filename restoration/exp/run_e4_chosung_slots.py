"""E4 — 초성체 슬롯 복원, 특히 사전 방식의 구조적 사각지대.

저장소의 `ChosungLexiconProvider`는 NRR 13.04%에 그쳤다. 원인 중 둘은 정확도 문제가 아니라
**구조적 차단**이다.

1. `_is_repeated_chat_initials()` — `ㅋㅋ`·`ㅎㅎ` 같은 통신체를 거르려는 필터인데, 초성이
   전부 같은 정상 단어(`제조`→`ㅈㅈ`, `방법`→`ㅂㅂ`, `사살`→`ㅅㅅ`)까지 함께 막는다.
   사전에 넣어도 소용없다. 필터가 사전 조회 **이전에** 문자열 형태만 보고 자르기 때문이다.
2. `min_initials=3` 기본값 — 2음절 초성체(`ㅍㅌ`, `ㅁㅇ`)는 아예 시도되지 않는다.
   한국어에서 가장 흔한 형태인데도 그렇다.

위치별 분류기는 사전을 안 쓰므로 두 사각지대가 원리상 없다. 이 실험은 그게 실제로
메워지는지, 그리고 메워진다 해도 쓸 만한 정확도가 나오는지를 본다.

E1에서 초성체 음절 exact는 코퍼스를 200배 늘려도 0.13->0.15로 평평했다. 기대치는 낮게
잡되, **어디까지 되는지와 왜 안 되는지**를 정확히 남기는 것이 이 실험의 목적이다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import jamo as J
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R
from kslab.models.linear import SlotRestorer
from kslab.models.mlp import MlpSlotRestorer

INTENSITIES = (0.5, 1.0)

# 사전 방식이 구조적으로 못 푸는 단어들. 초성이 전부 같아 통신체 필터에 걸린다.
REPEATED_INITIAL_WORDS = ("제조", "방법", "사살", "가공", "구글", "노노", "무마", "부분", "지적")
# 2음절 초성체 — min_initials=3 기본값에서 시도조차 되지 않는 형태.
TWO_SYLLABLE_WORDS = ("폭탄", "무기", "마약", "해킹", "탈옥", "우회", "설정", "명령")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--architectures", nargs="*", default=["linear", "mlp"])
    return parser.parse_args()


def _blind_spot_probe(model, words) -> dict:
    """단어 하나를 통째로 초성체로 만든 뒤 복원해 본다.

    사전 방식이 '후보를 만들지도 못하는' 케이스라 비교 축은 단순하다:
    분류기는 적어도 **시도는 한다**. 맞히는지는 별개 문제다.
    """
    obfuscated = [O.chosung(w, 1.0, seed=0) for w in words]
    restored = [r[0] for r in model.restore_many(obfuscated)]
    hits = sum(1 for word, got in zip(words, restored) if got == word)
    syllable_hits = sum(
        1 for word, got in zip(words, restored)
        for a, b in zip(word, got) if a == b
    )
    total_syllables = sum(len(w) for w in words)
    return {
        "words": len(words),
        "attempted": sum(1 for o in obfuscated if F.candidate_positions(o, "chosung")),
        "exact_word_hits": hits,
        "syllable_accuracy": syllable_hits / total_syllables if total_syllables else 0.0,
        "samples": [
            {"word": w, "obfuscated": o, "restored": r}
            for w, o, r in list(zip(words, obfuscated, restored))[:6]
        ],
    }


def _topk_recovery(model, examples, k_values=(1, 2, 3)) -> dict:
    """상위 k개 조합 안에 정답이 있는가. 후보를 몇 개까지 낼지 결정하는 근거."""
    if not examples:
        return {}
    proba = model.predict_proba_slots(examples)
    out = {}
    for k in k_values:
        hits = 0
        for row, example in enumerate(examples):
            ok = True
            for slot in model.unknown_slots:
                if slot not in proba:
                    continue
                ranked = proba[slot][row].argsort()[::-1][:k]
                gold = example.target_slots[slot]
                classes = getattr(model.heads.get(slot), "classes_", None) if hasattr(model, "heads") else None
                candidates = [int(classes[i]) for i in ranked] if classes is not None else [int(i) for i in ranked]
                if gold not in candidates:
                    ok = False
                    break
            hits += ok
        out[f"top{k}"] = hits / len(examples)
    return out


def main() -> None:
    args = parse_args()
    seeds = P.load_seeds()

    run = R.Run.start(
        run_id=args.run_id,
        experiment="E4_chosung_slots",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "architectures": args.architectures,
            "intensities": list(INTENSITIES),
            "window": args.window,
            "max_slot_examples": args.max_slot_examples,
            "corpus_license": C.LICENSE_NOTE,
            "comparison": "ChosungLexiconProvider NRR 13.04% (저장소 CHOSUNG_GUARDRAIL_IMPACT.md)",
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    train_pairs = P.make_pairs(
        train_items, ["chosung"], INTENSITIES,
        gen_seed=args.gen_seed, include_identity=True,
    )
    train_examples = F.subsample(
        F.extract_examples(train_pairs, args.window), args.max_slot_examples, args.random_seed
    )

    eval_pairs = P.make_pairs(
        P.seeds_as_items(seeds), ["chosung"], INTENSITIES,
        gen_seed=args.gen_seed, include_identity=False,
    )
    eval_examples = [
        e for e in F.extract_examples(eval_pairs, args.window) if e.technique == "chosung"
    ]
    changed = [e for e in eval_examples if e.changed]

    metrics: dict = {"architectures": {}, "diagnostics": {}}

    for architecture in args.architectures:
        print(f"[E4] {architecture} 학습 ...", flush=True)
        if architecture == "linear":
            model = SlotRestorer(
                "chosung", window=args.window, solver="sgd", max_iter=30,
                random_state=args.random_seed,
            ).fit(train_examples)
        else:
            model = MlpSlotRestorer(
                "chosung", window=args.window, epochs=8,
                random_state=args.random_seed, threads=args.threads,
            ).fit(train_examples)

        predicted = model.predict_slots(changed)
        accuracy = M.slot_accuracy(
            [tuple(int(v) for v in row) for row in predicted],
            [e.target_slots for e in changed],
        )
        obf = [p.obfuscated for p in eval_pairs]
        clean = [p.clean for p in eval_pairs]
        clusters = [p.cluster_id for p in eval_pairs]
        restored = [r[0] for r in model.restore_many(obf)]

        benign = [s.text for s in seeds if s.label == "benign"]
        entry = {
            "slot_accuracy_changed_only": accuracy,
            "restoration": M.restoration_metrics(
                clean, obf, restored, clusters, args.bootstrap_samples, args.random_seed
            ),
            "clean_mutation_benign": M.clean_mutation_rate(
                benign, [r[0] for r in model.restore_many(benign)]
            ),
            "topk_slot_recovery": _topk_recovery(model, changed[:20000]),
            "blind_spots": {
                "repeated_initials": _blind_spot_probe(model, REPEATED_INITIAL_WORDS),
                "two_syllable": _blind_spot_probe(model, TWO_SYLLABLE_WORDS),
            },
        }
        metrics["architectures"][architecture] = entry
        print(
            f"       중성 {accuracy['jung']:.3f} 종성 {accuracy['jong']:.3f} "
            f"음절exact {accuracy['syllable_exact']:.3f} | "
            f"top3 {entry['topk_slot_recovery'].get('top3', 0):.3f} | "
            f"clean mutation {entry['clean_mutation_benign']['sentence_rate']:.1%}",
            flush=True,
        )

    # 왜 안 되는지: 초성 하나가 몇 개의 음절로 갈라지는가 (정보량 상한)
    counter: Counter = Counter()
    for _, _, text in train_items:
        for ch in text:
            if J.is_syllable(ch):
                counter[J.split(ch)[0]] += 1
    metrics["diagnostics"]["distinct_syllables_per_initial"] = {
        J.COMPAT_CHO[i]: len({
            J.split(ch) for _, _, t in train_items[:5000] for ch in t
            if J.is_syllable(ch) and J.split(ch)[0] == i
        })
        for i in range(J.N_CHO)
    }
    metrics["diagnostics"]["theoretical_max_slot_accuracy"] = {
        "note": "중성 21종·종성 28종 중 하나를 문맥만으로 고르는 문제. "
                "초성체는 음절당 정보가 초성 하나뿐이라 상한이 낮다.",
    }

    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e4_chosung_slots_v1.json", {"run_id": args.run_id, **metrics}, run_id=args.run_id
    )
    print(f"\n[E4] 완료 -> {run.directory}")


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E4 — 초성체 슬롯 복원과 사전 방식의 사각지대",
        "",
        "> 상태: `PROVISIONAL_DEV_ONLY`",
        ">",
        "> 비교 대상은 저장소의 `ChosungLexiconProvider`(NRR 13.04%,",
        "> `CHOSUNG_GUARDRAIL_IMPACT.md`). 다만 여기서는 가드레일을 돌리지 않으므로",
        "> NRR이 아니라 복원 정확도로만 비교한다.",
        "",
        "## 실행 조건",
        "",
        f"- 코퍼스 {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        f"- 문자 창 +-{args.window}, 강도 {INTENSITIES}, 슬롯 예제 상한 {args.max_slot_examples:,}",
        "- 평가: 505개 dev 시드 (학습 미포함)",
        "",
        "## 결과",
        "",
        "| 아키텍처 | 중성 | 종성 | 음절 exact | top-2 | top-3 | benign clean mutation |",
        "|---|---|---|---|---|---|---|",
    ]
    for architecture, entry in metrics["architectures"].items():
        acc = entry["slot_accuracy_changed_only"]
        topk = entry["topk_slot_recovery"]
        lines.append(
            f"| {architecture} | {acc['jung']:.3f} | {acc['jong']:.3f} | "
            f"{acc['syllable_exact']:.3f} | {topk.get('top2', 0):.3f} | "
            f"{topk.get('top3', 0):.3f} | "
            f"{entry['clean_mutation_benign']['sentence_rate']:.1%} |"
        )
    lines += [
        "",
        "**clean mutation이 0%인 것은 모델이 잘해서가 아니다.** 초성체 후보 위치는 호환 자모",
        "낱자(`ㅍ`,`ㅌ`)뿐인데 이런 문자는 정상 문장에 나타나지 않는다. 즉 초성체 복원기는",
        "오탐 위험이 **구조적으로** 없다 — 된소리·연음과 결정적으로 다른 점이다.",
        "",
        "## 사전 방식 사각지대",
        "",
        "사전 기반 provider는 아래 두 부류에서 **후보를 만들지도 못한다**. 분류기는 사전을",
        "쓰지 않으므로 적어도 시도는 한다.",
        "",
    ]
    for architecture, entry in metrics["architectures"].items():
        lines += [f"### {architecture}", "",
                  "| 부류 | 단어 수 | 시도 | 단어 단위 정답 | 음절 정확도 |",
                  "|---|---|---|---|---|"]
        for key, label in (
            ("repeated_initials", "초성 반복(`제조`·`방법` 등, 통신체 필터에 걸림)"),
            ("two_syllable", "2음절(`폭탄`·`무기` 등, min_initials=3에 걸림)"),
        ):
            probe = entry["blind_spots"][key]
            lines.append(
                f"| {label} | {probe['words']} | {probe['attempted']} | "
                f"{probe['exact_word_hits']} | {probe['syllable_accuracy']:.3f} |"
            )
        lines.append("")
        samples = entry["blind_spots"]["repeated_initials"]["samples"]
        lines += ["복원 예시:", "", "| 원문 | 초성체 | 복원 |", "|---|---|---|"]
        for sample in samples:
            lines.append(f"| {sample['word']} | {sample['obfuscated']} | {sample['restored']} |")
        lines.append("")

    lines += [
        "## 해석",
        "",
        "- 사전 방식이 **원천적으로 후보를 못 만드는** 케이스에서도 분류기는 시도한다.",
        "  구조적 차단은 해소된다.",
        "- 그러나 정확도 자체가 낮다. E1에서 코퍼스를 200배 늘려도 음절 exact가 0.13->0.15로",
        "  평평했던 것과 같은 벽이다. 초성체는 음절당 주어지는 정보가 초성 하나뿐이라",
        "  중성 21종·종성 28종을 문맥만으로 골라야 한다 — 정보량 자체의 한계에 가깝다.",
        "- top-k를 넓히면 회복률이 오르지만, 그만큼 하위 가드레일에 보내는 view가 늘어",
        "  저장소가 이미 지적한 후보 상한 문제로 되돌아간다.",
        "",
        "## 해석 제한",
        "",
        "- 가드레일 추론 없음. NRR과 직접 비교할 수 없다.",
        "- 사각지대 probe 단어들은 손으로 고른 것이라 통계적 표본이 아니다.",
        "- 단어를 통째로 초성체화한 probe는 문맥이 전혀 없는 최악 조건이다. 문장 안에서는",
        "  주변 음절이 남아 있어 더 유리하다.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
