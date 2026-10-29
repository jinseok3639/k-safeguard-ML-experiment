"""P01.1 — 통합 분류표 커버리지 (p01.1-taxonomy).

학습 없음. `kslab/taxonomy.py`의 유형마다 복원기 범위와 내부 생성기 유무를 표로 만든다. 외부 생성기
칸은 P01.3에서 채운다.

판정(노트에 실행 전 고정): `obfuscate.TRANSFORMS` 전부가 한 칸에 속하고, D5 "대상" 유형마다 내부 기법이
지금 있거나 P01.2 예정이어야 한다.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import record as R
from kslab import taxonomy as T

EXPERIMENT = "P01_obf_generator"
COLUMNS = ("분류", "유형", "KOTOX", "KoNTe", "범위", "내부 생성기", "기법")


def table(rows: list[dict[str, str]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(row[c] for c in COLUMNS) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p01.1-taxonomy")
    args = parser.parse_args()

    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT,
                      config={"types": [t.id for t in T.TYPES]})
    rows = T.coverage_rows()
    target = [t for t in T.TYPES if t.scope == "target"]
    target_uncovered = [t.id for t in target if not (t.techniques or t.planned)]
    target_existing = [t.id for t in target if t.techniques]
    metrics = {
        "n_types": len(T.TYPES),
        "scope_counts": dict(Counter(t.scope for t in T.TYPES)),
        "status_counts": dict(Counter(r["내부 생성기"] for r in rows)),
        "unassigned_transforms": T.unassigned(),
        "target_uncovered": target_uncovered,
        "target_with_existing_generator": len(target_existing),
        "n_target": len(target),
        "status": "pass" if not T.unassigned() and not target_uncovered else "fail",
    }
    run.write_text("coverage.md", "# P01.1 커버리지 (`*` = P01.2 예정)\n\n" + table(rows) + "\n")
    run.finish(metrics, headline={"n_types": len(T.TYPES), "n_target": len(target),
                                  "target_with_existing_generator": len(target_existing)})
    print(table(rows))
    print(f"\n[P01.1] {metrics['status']}  -> {run.directory}")


if __name__ == "__main__":
    main()
