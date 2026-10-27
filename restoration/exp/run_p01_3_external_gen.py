"""P01.3 — 외부 생성기 커버리지 (p01.3-external-gen).

학습 없음. 외부 기법마다 505 시드에서 바뀐 문장 비율을 재고, 통합 분류표 유형별로 내부·외부 생성기 유무를
합친 커버리지 표를 만든다. 외부 생성기는 평가셋 2층(홀드아웃)이라 학습에 쓰지 않는다.

판정(노트에 실행 전 고정): 바뀐 문장 비율 ≥ 50%인 외부 기법이 있으면 그 유형을 "덮음". D5 "대상" 유형 중
덮는 유형이 절반 이상이면 통과.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import external_gen as X
from kslab import pairs as P
from kslab import record as R
from kslab import taxonomy as T

EXPERIMENT = "P01_obf_generator"
COVER_BAR = 0.5


def _key(gen: str, op: str) -> str:
    """MLflow 지표 이름에 쓸 수 없는 콜론을 바꾼다."""
    return f"{gen}/{op}".replace(":", "_")


COMPOSE = tuple(("kotox", f"compose:{n}") for n in (2, 3, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p01.3-external-gen")
    parser.add_argument("--gen-seed", type=int, default=2026)
    args = parser.parse_args()

    ops = list(X.TYPE_OF) + sorted(X.CONTROL) + list(COMPOSE)
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT,
                      config={"ops": [f"{g}/{o}" for g, o in ops], "gen_seed": args.gen_seed,
                              "cover_bar": COVER_BAR,
                              "sources": {g.name: g.source for g in X.GENERATORS.values()}})
    texts = [s.text for s in P.load_seeds()]
    rates, failed, examples = {}, {}, []
    for gen, op in ops:
        items = [X.Item(t, op, seed=args.gen_seed + i) for i, t in enumerate(texts)]
        try:
            outs = [r["out"] for r in X.run(gen, items)]
        except RuntimeError as e:
            failed[_key(gen, op)] = str(e).strip().splitlines()[-1]
            continue
        rates[_key(gen, op)] = sum(o != t for o, t in zip(outs, texts)) / len(texts)
        examples.append((gen, op, texts[0], outs[0]))

    covered: dict[str, list[str]] = {}
    for (gen, op), type_id in X.TYPE_OF.items():
        rate = rates.get(_key(gen, op))
        if rate is not None and rate >= COVER_BAR:
            covered.setdefault(type_id, []).append(_key(gen, op))
    target = [t.id for t in T.TYPES if t.scope == "target"]
    target_covered = [t for t in target if t in covered]
    metrics = {
        "changed_rate": rates, "failed": failed, "covered": covered,
        "target_covered": target_covered, "target_uncovered": [t for t in target if t not in covered],
        "n_target": len(target), "status": "pass" if len(target_covered) * 2 >= len(target) else "fail",
    }

    lines = ["# P01.3 외부 생성기 (505 시드)", "", "## 유형별 커버리지", "",
             "| 분류 | 유형 | 범위 | 내부 기법 | 외부(≥50% 발동) |", "|---|---|---|---|---|"]
    for t in T.TYPES:
        lines.append(f"| {t.category} | {t.name} | {t.scope} | {', '.join(t.techniques) or '—'} | "
                     f"{', '.join(covered.get(t.id, [])) or '—'} |")
    lines += ["", "## 외부 기법별 바뀐 문장 비율", "", "| 기법 | 유형 | 비율 |", "|---|---|---|"]
    for gen, op in ops:
        key = _key(gen, op)
        kind = X.TYPE_OF.get((gen, op)) or ("대조군(오타)" if (gen, op) in X.CONTROL else "복합")
        lines.append(f"| {key} | {kind} | {f'{rates[key]:.1%}' if key in rates else '실패: ' + failed[key]} |")
    lines += ["", "## 예시 (첫 시드)", "", "| 기법 | 원문 | 결과 |", "|---|---|---|"]
    lines += [f"| {g}/{o} | {a} | {b} |" for g, o, a, b in examples]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={"target_covered": len(target_covered), "n_target": len(target)})
    print("\n".join(lines[: lines.index("## 예시 (첫 시드)")]))
    print(f"\n[P01.3] {metrics['status']}  -> {run.directory}")


if __name__ == "__main__":
    main()
