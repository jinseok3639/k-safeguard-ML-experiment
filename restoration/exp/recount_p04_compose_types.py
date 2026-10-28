"""P04.3b·P04.3c 판정 3(compose 자동형 유형별 CER 개선)을 저장된 조건별 지표에서 다시 합친다.

러너가 compose 조건 이름(`auto3.1`)을 `auto/3`으로 가정해 공격자별 묶음이 비었다(러너는 고쳤다). 조건별 지표는 문장 수와
문장 평균 CER 개선으로 저장돼 있어, 문장 수로 가중 평균하면 다시 학습·추론하지 않고 같은 값이 나온다.

    python -m restoration.exp.recount_p04_compose_types p04.3b-kcelectra-ft
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import record as R
from kslab import taxonomy as T

TARGET = [t.id for t in T.TYPES if t.scope == "target"]
GAIN_BAR, TYPES_BAR = 0.02, 7


def by_attacker(types: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, list]] = {}
    for key, row in types.items():
        m = re.fullmatch(r"(\w+)/compose_(human|auto)[\d.]+", key)
        if m:
            acc = out.setdefault(m.group(2), {}).setdefault(m.group(1), [0, 0.0])
            acc[0] += row["n"]
            acc[1] += row["cer_gain"] * row["n"]
    return {a: {t: g / n for t, (n, g) in d.items()} for a, d in out.items()}


def main() -> None:
    run_dir = R.ROOT / "restoration" / "results" / sys.argv[1]
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    models = {"kcelectra": metrics["kcelectra"]} | {f"bilstm-{k}": v for k, v in metrics["bilstm"].items() if k.startswith("s")}
    result = {}
    lines = [f"# {sys.argv[1]} — compose 유형별 CER 개선 다시 합침(판정 3)", "",
             "| 모델 | 공격자 | " + " | ".join(TARGET) + " | 2%p 넘은 유형 |", "|---|---|" + "---|" * (len(TARGET) + 1)]
    for name, res in models.items():
        att = by_attacker(res["eval"]["type"])
        result[name] = {a: {"gain": g, "types_ok": [t for t in TARGET if g.get(t, 0) >= GAIN_BAR]} for a, g in att.items()}
        for a in ("human", "auto"):
            g = att.get(a, {})
            lines.append(f"| {name} | {a} | " + " | ".join(f"{g.get(t, float('nan')) * 100:.1f}" for t in TARGET)
                         + f" | {len(result[name][a]['types_ok'])}/9 |")
    ok = len(result["kcelectra"]["auto"]["types_ok"]) >= TYPES_BAR
    lines += ["", f"판정 3 (KcELECTRA, 자동형 ≥ {TYPES_BAR}/9): {'통과' if ok else '미달'}"]
    (run_dir / "recount_compose_types.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "recount_compose_types.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
