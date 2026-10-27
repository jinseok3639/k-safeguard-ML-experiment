"""P01.2 — 새 난독화 기법의 발동률 (p01.2-new-techniques).

학습 없음. 505 시드에서 P01.2 기법마다 바뀐 문장 비율을 잰다. 바뀌지 않은 문장은 평가셋에서 "원문 그대로"가
되므로, 발동률이 낮은 기법은 P01.4 평가셋 기준(축마다 변형 안 된 문장 ≤ 5%)에 걸린다.

판정(노트에 실행 전 고정): intensity 1.0에서 기법별 바뀐 문장 비율 ≥ 95%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R
from kslab import taxonomy as T

EXPERIMENT = "P01_obf_generator"
NEW = ("semivowel_insert", "initial_insert", "jamo_copy", "symbol_insert",
       "syllable_extend", "lookalike_foreign", "lookalike_hangul")
INTENSITIES = (0.5, 1.0)
BAR = 0.95
EXAMPLE_IDS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p01.2-new-techniques")
    parser.add_argument("--gen-seed", type=int, default=2026)
    args = parser.parse_args()

    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT,
                      config={"techniques": list(NEW), "intensities": list(INTENSITIES),
                              "gen_seed": args.gen_seed, "bar": BAR})
    seeds = P.load_seeds()
    texts = [s.text for s in seeds]
    rates: dict[str, dict[str, float]] = {}
    examples = []
    for name in NEW:
        fn = O.TRANSFORMS[name]
        rates[name] = {}
        for intensity in INTENSITIES:
            outs = [fn(t, intensity, seed=args.gen_seed + i) for i, t in enumerate(texts)]
            rates[name][str(intensity)] = sum(o != t for o, t in zip(outs, texts)) / len(texts)
            if intensity == 0.5:
                examples += [(name, texts[i], outs[i]) for i in range(EXAMPLE_IDS)]
    below = [n for n in NEW if rates[n]["1.0"] < BAR]
    metrics = {"n_seeds": len(texts), "changed_rate": rates, "below_bar": below,
               "status": "pass" if not below else "review"}

    lines = ["# P01.2 발동률 (505 시드, 바뀐 문장 비율)", "",
             "| 기법 | 유형 | 범위 | 0.5 | 1.0 |", "|---|---|---|---|---|"]
    for n in NEW:
        t = T.type_of(n)
        lines.append(f"| {n} | {t.name} | {t.scope} | {rates[n]['0.5']:.1%} | {rates[n]['1.0']:.1%} |")
    lines += ["", "## 예시 (intensity 0.5)", "", "| 기법 | 원문 | 난독화 |", "|---|---|---|"]
    lines += [f"| {n} | {a} | {b} |" for n, a, b in examples]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={f"changed_1.0/{n}": rates[n]["1.0"] for n in NEW})
    print("\n".join(lines[:4 + len(NEW)]))
    print(f"\n[P01.2] {metrics['status']}  미달 {below}  -> {run.directory}")


if __name__ == "__main__":
    main()
