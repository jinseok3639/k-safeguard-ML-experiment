"""P04.1 — 편집 태그 표현 상한 (p04.1-edit-ceiling).

학습 없음. 평가셋 v1의 레코드마다 정답을 알 때 글자 편집 태그(`kslab/edit_tags.py`)로 원문에 얼마나 가까이 되돌릴 수
있나를 재고, 1기 슬롯 모델 구조의 상한과 나란히 둔다.

판정(노트에 실행 전 고정): external 축 target 레코드의 편집 태그 상한 CER 개선 ≥ 2%p, single·compose 축 target 레코드의
표현 가능률 ≥ 95%. 둘 다 통과하면 pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import edit_tags as E
from kslab import evalset as V
from kslab import record as R
from kslab import taxonomy as T
from kslab.metrics import char_error_rate

EXPERIMENT = "P04_normalizer"
EVALSET = R.ROOT / "data" / "derived" / "evalset_v1"
AXES = ("single", "compose", "external", "control")
GAIN_BAR = 0.02
EXPRESSIBLE_BAR = 0.95
TARGET = {t.id for t in T.TYPES if t.scope == "target"}


def scope_of(types: list[str]) -> str:
    if types == ["control"]:
        return "control"
    return "target" if set(types) <= TARGET else "mixed"


class Acc:
    def __init__(self):
        self.n = self.expressible = 0
        self.raw = self.edit = self.slot = 0.0

    def add(self, raw, edit, slot, expressible):
        self.n += 1
        self.expressible += expressible
        self.raw += raw; self.edit += edit; self.slot += slot

    def row(self) -> dict:
        n = max(self.n, 1)
        return {"n": self.n, "expressible": self.expressible / n, "raw_cer": self.raw / n,
                "edit_gain": (self.raw - self.edit) / n, "slot_gain": (self.raw - self.slot) / n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p04.1-edit-ceiling")
    args = parser.parse_args()

    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT,
                      config={"evalset": "evalset_v1", "gain_bar": GAIN_BAR, "expressible_bar": EXPRESSIBLE_BAR,
                              "target_types": sorted(TARGET)})
    run.add_input("records.jsonl", EVALSET / "records.jsonl")
    run.add_input("header.json", EVALSET / "header.json")

    by_axis: dict[tuple, Acc] = defaultdict(Acc)
    by_type: dict[tuple, Acc] = defaultdict(Acc)
    by_name: dict[tuple, Acc] = defaultdict(Acc)
    misses: dict[str, list] = defaultdict(list)
    with open(EVALSET / "records.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            obf, clean = r["obfuscated"], r["clean"]
            types = V.types_of(r)
            scope = scope_of(types)
            _, residual = E.oracle_tags(obf, clean)
            n = max(len(clean), 1)
            vals = (char_error_rate(clean, obf), residual / n, E.slot_residual(obf, clean) / n, residual == 0)
            by_axis[(r["axis"], scope)].add(*vals)
            by_name[(r["axis"], r["name"])].add(*vals)
            if scope == "target":
                for t in types:
                    by_type[(t, r["axis"])].add(*vals)
            if residual and scope == "target" and len(misses[r["axis"]]) < 8:
                misses[r["axis"]].append((r["id"], clean, obf, residual))

    axis_rows = {f"{a}/{s}": acc.row() for (a, s), acc in sorted(by_axis.items())}
    ext = axis_rows.get("external/target", Acc().row())
    own = [axis_rows[k]["expressible"] for k in ("single/target", "compose/target") if k in axis_rows]
    checks = {"external_edit_gain": ext["edit_gain"] >= GAIN_BAR,
              "own_expressible": bool(own) and min(own) >= EXPRESSIBLE_BAR}
    metrics = {"axis": axis_rows,
               "type": {f"{t}/{a}": acc.row() for (t, a), acc in sorted(by_type.items())},
               "name": {f"{a}/{nm}".replace(":", "_"): acc.row() for (a, nm), acc in sorted(by_name.items())},
               "checks": checks, "status": "pass" if all(checks.values()) else "fail"}

    def pct(x):
        return f"{x * 100:.1f}"

    lines = ["# P04.1 편집 태그 표현 상한 (평가셋 v1)", "",
             "CER과 개선은 문장 평균, %p. 표현 가능 = 정답 태그로 원문을 정확히 만들 수 있는 비율.", "",
             "## 축별", "", "| 축 | 범위 | 문장 | 원래 CER | 편집 태그 표현 가능 | 편집 태그 상한 개선 | 슬롯 상한 개선 |",
             "|---|---|---|---|---|---|---|"]
    for (a, s), acc in sorted(by_axis.items(), key=lambda kv: (AXES.index(kv[0][0]), kv[0][1])):
        x = acc.row()
        lines.append(f"| {a} | {s} | {x['n']} | {pct(x['raw_cer'])} | {pct(x['expressible'])}% | "
                     f"{pct(x['edit_gain'])} | {pct(x['slot_gain'])} |")
    lines += ["", "## 유형별 (target 레코드)", "", "| 유형 | 축 | 문장 | 표현 가능 | 편집 태그 상한 개선 | 슬롯 상한 개선 |",
              "|---|---|---|---|---|---|"]
    for t in T.TYPES:
        for a in AXES:
            if (t.id, a) in by_type:
                x = by_type[(t.id, a)].row()
                lines.append(f"| {t.name} | {a} | {x['n']} | {pct(x['expressible'])}% | {pct(x['edit_gain'])} | "
                             f"{pct(x['slot_gain'])} |")
    lines += ["", "## 표현 못 한 예 (target)", "", "| id | 원문 | 난독화 | 남는 오류 |", "|---|---|---|---|"]
    for a in AXES:
        lines += [f"| {i} | {c} | {o} | {res} |" for i, c, o, res in misses.get(a, [])]
    lines += ["", f"판정: {metrics['status']} — {checks}"]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={"external_target_edit_gain": ext["edit_gain"],
                                  "external_target_slot_gain": ext["slot_gain"],
                                  "own_expressible_min": min(own) if own else 0.0})
    print("\n".join(lines[: lines.index("## 유형별 (target 레코드)")]))
    print(f"\n[P04.1] {metrics['status']} {checks} -> {run.directory}")


if __name__ == "__main__":
    main()
