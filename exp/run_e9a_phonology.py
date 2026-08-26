"""E9a — 음운 규칙 생성기 확장과 조합 복원의 구조적 상한 (학습 없음).

두 가지를 한다.

1. **생성기 감사**: 표준발음법 기준으로 누락돼 있던 음운 규칙 9종을 새로 넣고, 각각이
   실제로 학습 신호를 낼 만한 빈도인지, 조항대로 동작하는지 잰다. E5a가 기존 5종에
   대해 한 일을 새 기법으로 확장한 것이다.

2. **조합 복원의 구조적 상한(Stage 0 게이트)**: 모델을 하나도 학습시키지 않고, 단일 기법
   모델에 조합 난독화를 먹였을 때 **구조적으로 못 고치는 손상 비율**을 잰다. 복원기는
   후보 규칙이 고른 자리만 방문하고 담당 슬롯만 예측하므로(나머지는 입력 복사), 이 손실은
   학습으로 줄지 않는다. E8 승격 기준(CER 최소 2%p 감소)에 상한이 못 미치면 그 조합에
   그 모델을 붙이는 실험은 시작할 필요가 없다.

E9b(학습 실험)를 돌리기 전에 반드시 이걸 먼저 통과해야 한다 — 몇 분이면 며칠짜리 학습
계획을 접을 수 있다.

선행 실험에서 가져온 것:
- E5a: 생성기 자체를 지표로 감사하는 형식.
- E8: 승격 기준 두 개(benign 변형률 ≤1%, CER 개선 ≥2%p)를 그대로 쓴다.
- E2: "병목은 데이터가 아니라 아키텍처"였으므로, 상한은 아키텍처가 아니라 **명세**
  (후보 규칙 + 담당 슬롯)가 정한다는 점을 분리해서 본다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import ceiling as CE
from kslab import compose as X
from kslab import corpus as C
from kslab import features as F
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R

# 조합 상한을 잴 프로파일. 깊이 1은 기존 E0~E8과 직접 비교되는 기준선이다.
CEILING_PROFILES = (
    "d1_tensify", "d1_liaison", "d1_cram", "d1_vowel", "d1_aspirate",
    "d2_liaison_then_tensify", "d2_tensify_then_liaison",
    "d2_liaison_then_cram", "d2_cram_then_liaison", "d2_tensify_cram",
    "d2_liaison_then_vowel", "d2_aspirate_then_liaison",
    "d3_canonical", "d3_reversed", "d3_with_vowel",
    "d5_full_attack", "d5_phonological",
)

# 손상량을 고정한 채 깊이만 바꾸기 위한 예산(문자 기준). 깊이별 난이도를 비교하려면
# 필수다 — 각 단계를 강도 1.0으로 걸면 깊이 3이 문자의 76%를 파괴해, 모델이 못 맞히는
# 이유가 "조합이라서"인지 "더 망가져서"인지 구분할 수 없다.
DAMAGE_BUDGET = 0.08

MIN_CER_REDUCTION = 0.02   # E8과 같은 유용성 기준

# 표준발음법 조항 대조용 회귀 예시. 생성기가 조항대로 도는지 눈으로 확인 가능한 형태로 박아둔다.
REGRESSION = {
    "aspirate": [
        ("놓고", "노코"), ("좋다", "조타"), ("많지", "만치"), ("닳지", "달치"),
        ("국화", "구콰"), ("맏형", "마텽"), ("좁히다", "조피다"),
        ("앉히다", "안치다"), ("밝히다", "발키다"), ("축하", "추카"),
    ],
    "h_deletion": [
        ("좋아", "조아"), ("많아", "마나"), ("싫어", "시러"),
        ("낳은", "나은"), ("놓아", "노아"),
    ],
    "liaison_correct": [
        ("좋아", "조아"), ("많아", "마나"), ("싫어", "시러"),
        ("먹을", "머글"), ("읽어", "일거"), ("값을", "갑슬"),
    ],
    "coda_neutralize": [
        ("부엌", "부억"), ("옷", "옫"), ("빛", "빋"), ("앞", "압"), ("히읗", "히읃"),
    ],
    "cluster_reduce": [
        ("값", "갑"), ("넋", "넉"), ("닭", "닥"), ("삶", "삼"), ("여덟", "여덜"),
    ],
    "nasalize": [
        ("국물", "궁물"), ("닫는", "단는"), ("밥물", "밤물"), ("먹는", "멍는"),
        ("종로", "종노"), ("강릉", "강능"), ("백리", "뱅니"), ("섭리", "섬니"),
    ],
    "lateralize": [
        ("신라", "실라"), ("난로", "날로"), ("칼날", "칼랄"),
        ("설날", "설랄"), ("권력", "궐력"),
    ],
    "tensify_rule": [
        ("국밥", "국빱"), ("학교", "학꾜"), ("먹다", "먹따"), ("덮개", "덮깨"),
    ],
    # 조합이라야 나오는 실제 표준 발음. 단일 규칙으로는 도달할 수 없다는 증거다.
    "_composed": [
        ("옷고름", "옫꼬름"), ("꽃다발", "꼳따발"), ("밭머리", "반머리"),
        ("앞마당", "암마당"), ("읽는", "잉는"),
    ],
}
COMPOSED_CHECK_STAGES = ("cluster_reduce", "coda_neutralize", "nasalize", "tensify_rule")

NEW_TRANSFORMS = (
    "liaison_correct", "h_deletion", "aspirate", "vowel_confuse",
    "coda_neutralize", "cluster_reduce", "nasalize", "lateralize", "tensify_rule",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--calibration-sentences", type=int, default=1500)
    parser.add_argument("--ceiling-sentences", type=int, default=3000)
    parser.add_argument("--damage-budget", type=float, default=DAMAGE_BUDGET)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def _regression(metrics: dict) -> None:
    """조항 대조. 하나라도 어긋나면 이후 수치가 전부 의미를 잃으므로 제일 먼저 본다."""
    rows = {}
    failures = []
    for name, cases in REGRESSION.items():
        if name == "_composed":
            continue
        fn = O.TRANSFORMS[name]
        got = [(src, exp, fn(src, 1.0, seed=0)) for src, exp in cases]
        bad = [(s, e, g) for s, e, g in got if g != e]
        rows[name] = {"cases": len(got), "failures": len(bad)}
        failures += [{"technique": name, "input": s, "expected": e, "got": g} for s, e, g in bad]

    composed = []
    for src, exp in REGRESSION["_composed"]:
        got = X.apply_stages(src, COMPOSED_CHECK_STAGES, 1.0, 0).text
        composed.append({"input": src, "expected": exp, "got": got, "ok": got == exp})
        if got != exp:
            failures.append({"technique": "_composed", "input": src, "expected": exp, "got": got})
    rows["_composed"] = {
        "cases": len(composed), "failures": sum(1 for c in composed if not c["ok"])
    }
    metrics["regression"] = {
        "by_technique": rows, "failures": failures, "composed_examples": composed,
    }


def _generator_stats(texts, metrics, gen_seed) -> None:
    """새 기법이 실제로 학습 신호를 낼 빈도인지, 어느 슬롯을 건드리는지."""
    from kslab.jamo import is_syllable, split

    rows = {}
    for name in NEW_TRANSFORMS:
        fn = O.TRANSFORMS[name]
        changed_sentences = changed_chars = total_chars = 0
        slots = [0, 0, 0]
        candidate_sites = 0
        for text in texts:
            out = fn(text, 1.0, seed=gen_seed)
            if len(out) != len(text):
                raise ValueError(f"{name}이 길이를 바꿨습니다.")
            if out != text:
                changed_sentences += 1
            total_chars += len(text)
            candidate_sites += len(F.candidate_positions(out, name))
            for a, b in zip(text, out):
                if a == b:
                    continue
                changed_chars += 1
                if is_syllable(a) and is_syllable(b):
                    for k, (x, y) in enumerate(zip(split(a), split(b))):
                        if x != y:
                            slots[k] += 1
        rows[name] = {
            "clause": O.STANDARD_PRONUNCIATION_CLAUSE.get(name, "표기 혼동(조항 없음)"),
            "family": O.FAMILY[name],
            "modeled": name in O.MODELED,
            "unknown_slots": list(F.UNKNOWN_SLOTS[name]),
            "sentence_fire_rate": changed_sentences / len(texts),
            "char_change_rate": changed_chars / total_chars,
            "sites_per_1k_chars": changed_chars / total_chars * 1000,
            # 신호 밀도 = 실제 변형된 자리 / 모델이 방문하는 후보 자리.
            # 낮으면 모델이 identity를 학습하고 정확도 99.9%를 보고한다.
            "signal_density": changed_chars / candidate_sites if candidate_sites else 0.0,
            "slot_changes": {"cho": slots[0], "jung": slots[1], "jong": slots[2]},
        }
    metrics["generators"] = rows


def _liaison_variants(texts, metrics, gen_seed) -> None:
    """`liaison`(오타형)과 `liaison_correct`(표준 발음형)의 차이를 실측한다.

    기존 liaison은 받침 ㅎ을 뒤 음절 초성으로 넘겨 좋아 -> 조하 를 만든다. 표준 발음은
    조아 다. 기존 동작을 고치지 않고 변종을 따로 둔 이유는 두 가지다 — 조하·실허 같은
    표기도 실제 오타로 흔히 나타나 그 자체로 유효한 공격이고, E1~E8의 기준선이 전부
    기존 동작 위에서 측정됐다. 여기서는 "그 차이가 얼마나 되는가"만 숫자로 남긴다.
    """
    differing = 0
    diff_chars = 0
    total_chars = 0
    samples = []
    for text in texts:
        a = O.liaison(text, 1.0, seed=gen_seed)
        b = O.liaison_correct(text, 1.0, seed=gen_seed)
        total_chars += len(text)
        if a != b:
            differing += 1
            diff_chars += sum(1 for x, y in zip(a, b) if x != y)
            if len(samples) < 8:
                samples.append({"clean": text, "liaison": a, "liaison_correct": b})
    metrics["liaison_variants"] = {
        "sentences": len(texts),
        "differing_sentences": differing,
        "differing_sentence_rate": differing / len(texts),
        "differing_char_rate": diff_chars / total_chars,
        "samples": samples,
    }


def _ceilings(texts, metrics, args) -> None:
    """Stage 0 게이트 — 학습 없이 구하는 구조적 상한."""
    calib = texts[: args.calibration_sentences]
    evaluate = texts[: args.ceiling_sentences]
    rows: dict[str, dict] = {}
    for pname in CEILING_PROFILES:
        profile = X.PROFILES_BY_NAME[pname]
        intensity, realized = X.calibrate(
            calib, profile.stages, args.damage_budget, args.gen_seed
        )
        obfuscated = [
            X.apply_stages(t, profile.stages, intensity, args.gen_seed).text
            for t in evaluate
        ]
        # 후보 모델: 구성 단계 각각 + 합집합 모델.
        candidates = sorted({*profile.stages, profile.technique()})
        entry = {
            "stages": list(profile.stages),
            "depth": profile.depth,
            "stage_intensity": round(intensity, 4),
            "realized_damage": realized,
            "union_technique": profile.technique(),
            "models": {},
        }
        for technique in candidates:
            try:
                entry["models"][technique] = CE.ceiling_for(evaluate, obfuscated, technique)
            except ValueError as exc:            # 후보 규칙이 없는 조합 이름
                entry["models"][technique] = {"error": str(exc)}
        rows[pname] = entry
    metrics["ceilings"] = rows


def _verdict(metrics: dict) -> None:
    """각 조합에서 '합집합 모델이 구조적으로 가능한가'와 '단일 모델이 무너지는가'."""
    union_pass, single_fail = [], []
    for pname, entry in metrics["ceilings"].items():
        union = entry["models"].get(entry["union_technique"], {})
        reduction = union.get("ceiling_cer_reduction", 0.0)
        if reduction >= MIN_CER_REDUCTION:
            union_pass.append({"profile": pname, "ceiling_cer_reduction": reduction})
        for technique, row in entry["models"].items():
            if technique == entry["union_technique"] or "error" in row:
                continue
            if row["ceiling_cer_reduction"] < MIN_CER_REDUCTION:
                single_fail.append({
                    "profile": pname, "technique": technique,
                    "ceiling_cer_reduction": row["ceiling_cer_reduction"],
                    "unreachable_share": row["unreachable_share"],
                    "inexpressible_share": row["inexpressible_share"],
                })
    metrics["union_passes_gate"] = union_pass
    metrics["single_model_structural_failures"] = single_fail
    regression_ok = not metrics["regression"]["failures"]
    metrics["status"] = (
        "PROCEED_TO_E9B" if regression_ok and union_pass else "DO_NOT_PROMOTE"
    )


def _report(metrics: dict, args: argparse.Namespace) -> str:
    lines = [
        "# E9a — 음운 규칙 생성기 확장과 조합 복원의 구조적 상한",
        "",
        f"> 상태: `{metrics['status']}`",
        ">",
        "> 표준발음법 기준으로 누락돼 있던 음운 규칙 9종을 생성기에 넣고, 조합 난독화에 대해",
        "> **학습 없이** 구할 수 있는 복원 상한을 계산했다. 상한은 후보 위치 규칙과 담당 슬롯이",
        "> 정하는 구조적 한계이므로, 학습·아키텍처·임계값으로 넘을 수 없다.",
        "",
        "## 판정 기준",
        "",
        f"1. 표준발음법 조항 대조 회귀가 전부 통과할 것 (실패 시 이후 수치는 무의미)",
        f"2. 합집합 모델의 상한 CER 개선폭 ≥ {MIN_CER_REDUCTION:.0%} (E8과 같은 유용성 기준)",
        "",
        f"- 코퍼스 {args.corpus_size:,}문장 (위키 {args.wiki_ratio:.0%}), {C.LICENSE_NOTE}",
        f"- 상한 평가 {args.ceiling_sentences:,}문장, 강도 보정 {args.calibration_sentences:,}문장",
        f"- 손상 예산 {args.damage_budget:.0%} (깊이별 비교를 위해 손상량 고정)",
        "",
        "## 1. 조항 대조 회귀",
        "",
        "| 기법 | 조항 | 케이스 | 실패 |",
        "|---|---|---|---|",
    ]
    for name, row in metrics["regression"]["by_technique"].items():
        clause = (
            "조합 전용" if name == "_composed"
            else O.STANDARD_PRONUNCIATION_CLAUSE.get(name, "-")
        )
        mark = "" if not row["failures"] else " ⚠"
        lines.append(f"| {name} | {clause} | {row['cases']} | {row['failures']}{mark} |")

    lines += [
        "",
        "조합으로만 도달하는 표준 발음 (단일 규칙으로는 나오지 않는다):",
        "",
        "| 원문 | 표준 발음 | 생성 결과 |",
        "|---|---|---|",
    ]
    for row in metrics["regression"]["composed_examples"]:
        mark = "" if row["ok"] else " ⚠"
        lines.append(f"| {row['input']} | {row['expected']} | {row['got']}{mark} |")

    lines += [
        "",
        "## 2. 새 생성기 통계",
        "",
        "`신호 밀도` = 실제 변형된 자리 / 모델이 방문하는 후보 자리. 이게 낮으면 모델은",
        "identity를 학습하고 높은 정확도를 보고한다 — 기존 palatalize가 그 경우다.",
        "",
        "| 기법 | 조항 | 학습대상 | 담당슬롯 | 문장발동 | 1k자당 | 신호밀도 | 초성/중성/종성 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in metrics["generators"].items():
        slots = "".join("초중종"[s] for s in row["unknown_slots"]) or "-"
        sc = row["slot_changes"]
        lines.append(
            f"| {name} | {row['clause']} | {'O' if row['modeled'] else '-'} | {slots} | "
            f"{row['sentence_fire_rate']:.1%} | {row['sites_per_1k_chars']:.1f} | "
            f"{row['signal_density']:.2%} | {sc['cho']:,}/{sc['jung']:,}/{sc['jong']:,} |"
        )

    lv = metrics["liaison_variants"]
    lines += [
        "",
        "## 3. liaison 두 변종",
        "",
        f"기존 `liaison`(받침 ㅎ을 넘김, 좋아→조하)과 새 `liaison_correct`(ㅎ 탈락, 좋아→조아)는",
        f"문장의 **{lv['differing_sentence_rate']:.2%}** 에서 결과가 갈린다 (문자 기준 {lv['differing_char_rate']:.3%}).",
        "",
        "기존 동작을 고치지 않고 변종을 추가한 이유: 조하·실허 같은 표기도 실제 오타로 흔히",
        "나타나 그 자체로 유효한 공격이고, E1~E8의 기준선이 전부 기존 동작 위에서 측정됐다.",
        "",
        "| 원문 | liaison (오타형) | liaison_correct (표준) |",
        "|---|---|---|",
    ]
    for row in lv["samples"][:5]:
        lines.append(f"| {row['clean'][:28]} | {row['liaison'][:28]} | {row['liaison_correct'][:28]} |")

    lines += [
        "",
        "## 4. 조합 복원의 구조적 상한 (Stage 0 게이트)",
        "",
        "`도달불가` = 손상됐는데 후보 규칙이 그 자리를 안 고름. `표현불가` = 방문은 하는데",
        "손상이 담당 슬롯 밖이라 입력을 그대로 복사함. 둘 다 학습으로 줄지 않는다.",
        "",
        "| 프로파일 | 깊이 | 실측손상 | 모델 | 상한 CER개선 | 도달불가 | 표현불가 |",
        "|---|---|---|---|---|---|---|",
    ]
    for pname, entry in metrics["ceilings"].items():
        for technique, row in entry["models"].items():
            if "error" in row:
                continue
            union_mark = " **(합집합)**" if technique == entry["union_technique"] else ""
            ok = " ✅" if row["ceiling_cer_reduction"] >= MIN_CER_REDUCTION else " ❌"
            lines.append(
                f"| {pname} | {entry['depth']} | {entry['realized_damage']:.2%} | "
                f"{technique}{union_mark} | {row['ceiling_cer_reduction']:.2%}{ok} | "
                f"{row['unreachable_share']:.1%} | {row['inexpressible_share']:.1%} |"
            )

    fails = metrics["single_model_structural_failures"]
    lines += ["", "## 판정", ""]
    if metrics["status"] == "PROCEED_TO_E9B":
        lines += [
            f"합집합 모델은 {len(metrics['union_passes_gate'])}개 프로파일에서 상한이 기준을 넘는다 —",
            "조합 난독화를 다루는 것이 **구조적으로는** 가능하다. 남은 질문은 순수하게 통계적이다:",
            "그 분포로 학습했는가. E9b로 넘어간다.",
            "",
        ]
    else:
        lines += [
            "합집합 모델조차 상한이 기준에 못 미치거나 회귀가 깨졌다. E9b는 의미가 없다.",
            "",
        ]
    if fails:
        lines += [
            f"단일 기법 모델이 **구조적으로** 기준에 미달하는 경우가 {len(fails)}건 있다:",
            "",
            "| 프로파일 | 모델 | 상한 CER개선 | 도달불가 | 표현불가 |",
            "|---|---|---|---|---|",
        ]
        for row in sorted(fails, key=lambda r: r["ceiling_cer_reduction"])[:12]:
            lines.append(
                f"| {row['profile']} | {row['technique']} | {row['ceiling_cer_reduction']:.2%} | "
                f"{row['unreachable_share']:.1%} | {row['inexpressible_share']:.1%} |"
            )
        lines.append("")

    lines += [
        "## 해석 제한",
        "",
        "- **상한은 낙관적이다.** 모델이 담당 슬롯을 항상 정답으로 맞힌다고 가정한다. 실제",
        "  정확도는 이보다 낮으므로, 상한 미달은 확정적 실패지만 상한 통과는 통과 보장이 아니다.",
        "- 가드레일 추론 없음. 이 실험은 학습을 하지 않으므로 clean mutation도 재지 않는다.",
        "- 손상 예산을 고정했으므로 깊이 간 비교가 성립한다. 실측 손상률이 예산에서 벗어난",
        "  프로파일(그 조합의 손상 상한이 예산보다 낮은 경우)은 깊이 비교에서 제외해야 한다.",
        "- 조합 대상에서 `chosung`은 제외했다 — 뒤 단계를 전부 흡수해 조합이 성립하지 않는다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run = R.Run.start(
        run_id=args.run_id,
        experiment="E9a_phonology_ceiling",
        config={
            "corpus_size": args.corpus_size,
            "wiki_ratio": args.wiki_ratio,
            "calibration_sentences": args.calibration_sentences,
            "ceiling_sentences": args.ceiling_sentences,
            "damage_budget": args.damage_budget,
            "gen_seed": args.gen_seed,
            "random_seed": args.random_seed,
            "profiles": list(CEILING_PROFILES),
            "new_transforms": list(NEW_TRANSFORMS),
            "corpus_license": C.LICENSE_NOTE,
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)
    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)
    texts = [text for _, _, text in train_items]

    metrics: dict = {}
    print("[E9a] 조항 대조 회귀 ...", flush=True)
    _regression(metrics)
    print("[E9a] 생성기 통계 ...", flush=True)
    _generator_stats(texts[: args.ceiling_sentences], metrics, args.gen_seed)
    print("[E9a] liaison 변종 비교 ...", flush=True)
    _liaison_variants(texts[: args.ceiling_sentences], metrics, args.gen_seed)
    print(f"[E9a] 조합 상한 {len(CEILING_PROFILES)}개 프로파일 ...", flush=True)
    _ceilings(texts, metrics, args)
    _verdict(metrics)

    run.write_text("report.md", _report(metrics, args))
    run.finish(metrics)
    R.publish_baseline(
        "e9a_phonology_ceiling_v1.json",
        {"run_id": args.run_id, **metrics},
        run_id=args.run_id,
    )

    failures = len(metrics["regression"]["failures"])
    print(f"\n[E9a] 완료 -> {run.directory}  판정 {metrics['status']}")
    print(f"  조항 회귀 실패 {failures}건")
    print(f"  합집합 모델이 상한 기준을 넘는 프로파일 {len(metrics['union_passes_gate'])}개")
    print(f"  단일 모델의 구조적 미달 {len(metrics['single_model_structural_failures'])}건")


if __name__ == "__main__":
    main()
