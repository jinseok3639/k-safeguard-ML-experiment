"""P02 — 최종 복원기를 앞에 붙인 가드레일 우회율: S01과 같은 평가셋 v1을 다시 태운다.

S01(`s01.1-kanana`)은 가드레일 단독이었다. 여기서는 모든 문장(원문 포함)을 팀 복원기(P04.3b)로 고친 뒤 같은
가드레일에 넣는다. 두 구성을 같은 run에서 낸다.
  복원문만 검사   복원문의 판정만 쓴다
  원문 OR 복원문  기준 run(`--base-run`, 가드레일 단독)의 판정과 복원문 판정 중 하나라도 차단이면 차단

집계는 S01과 같은 `guardrail.evalset.aggregate`(원문 단위 bootstrap)다. 지표 정의도 같다.
evasion_if_caught의 분모(원문이 차단된 공격)는 구성마다 그 구성의 원문 판정으로 정한다.

산출물
  results/<run-id>/scores.jsonl   문장별 판정(복원문·OR·복원기가 바꿨나). 문장은 없음
  results/<run-id>/metrics.json   구성별 집계, 복원 범위별 우회율, 복원기 정보
  baselines/p02_<가드레일>_restored_v1.json  집계만

    python -m guardrail.exp.run_p02_kanana_restored --run-id p02.1-kanana-restored --base-run s01.1-kanana
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guardrail import evalset as E
from guardrail import scorers as S
from guardrail.exp.p05_profanity import restorers as RS
from guardrail.exp.run_s01_obf_robustness import DEFAULT_EVALSET, score_all
from kslab import record as R
from kslab import taxonomy as T

EXPERIMENT = "p02_kanana_restorer"


def scope_rates(rows: list[dict], flags: dict, caught: set[str]) -> dict:
    """복원 범위(target/boundary/out)별 원문 차단 조건부 우회율(행 단위). 유형이 여럿이면 가장 바깥 범위로 센다."""
    order = {"target": 0, "boundary": 1, "out": 2}
    acc: dict[str, list[int]] = {}
    for r in rows:
        if r["axis"] == "clean" or r["label"] != "attack" or r["source_id"] not in caught or not r["types"]:
            continue
        if flags.get(r["id"]) is None:
            continue
        scope = max((T.BY_ID[t].scope for t in r["types"] if t in T.BY_ID), key=order.get, default="out")
        a = acc.setdefault(scope, [0, 0])
        a[0] += not flags[r["id"]]; a[1] += 1
    return {k: {"k": v[0], "n": v[1], "rate": v[0] / v[1]} for k, v in sorted(acc.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--guardrail", choices=sorted(S.GUARDRAILS), default="kanana")
    ap.add_argument("--restorer", choices=sorted(RS.RESTORERS), default="p04.3b-ft-s0")
    ap.add_argument("--base-run", required=True, help="같은 가드레일 단독 run(S01)")
    ap.add_argument("--evalset", type=Path, default=DEFAULT_EVALSET)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    a = ap.parse_args()

    g = S.GUARDRAILS[a.guardrail]
    sums = E.verify(a.evalset)
    records = E.load_records(a.evalset)
    rows = E.clean_rows(records) + records
    with open(R.RESULTS_DIR / a.base_run / "scores.jsonl", encoding="utf-8") as f:
        base = {j["id"]: j["flagged"] for j in map(json.loads, f)}
    if set(base) != {r["id"] for r in rows}:
        raise SystemExit(f"{a.base_run}의 행이 평가셋과 다릅니다")

    run = R.Run.start(run_id=a.run_id, experiment=EXPERIMENT, config={
        "guardrail": g.key, "model": g.model, "revision": g.revision, "threshold": g.threshold,
        "restorer": a.restorer, "base_run": a.base_run, "batch": a.batch, "bootstrap_samples": a.bootstrap_samples,
        "evalset_sha256": sums,
        "evalset_license_note": "KOTOX·kotka·korean-jamo-typo MIT, KoG2Padvanced 표시 없음(내부 평가 전용), "
                                "korean_guardrail_test CC BY 4.0",
    })
    run.add_input("evalset/records.jsonl", a.evalset / "records.jsonl", rows=len(records))

    print(f"[P02] {g.name} + {a.restorer}  {len(rows)}행", flush=True)
    restore = RS.RESTORERS[a.restorer]()
    for k, v in RS.INFO.items():
        run.note(f"restorer_{k}", v)
    t0 = time.time()
    texts = [r["obfuscated"] for r in rows]
    restored_texts = restore(texts)
    restore_sec = time.time() - t0
    rrows = [dict(r, obfuscated=t) for r, t in zip(rows, restored_texts)]

    import torch
    scorer = S.load(g, "cuda" if torch.cuda.is_available() else "cpu")
    scored = score_all(scorer, rrows, a.batch)
    flags_r = {r["id"]: scored[r["id"]]["flagged"] for r in rows}
    flags_or = {i: (True if base[i] or flags_r[i] else (None if base[i] is None and flags_r[i] is None else False))
                for i in flags_r}

    out = {}
    for name, flags in (("restored", flags_r), ("or", flags_or)):
        s = E.aggregate(rows, flags, bootstrap_samples=a.bootstrap_samples)
        caught = {r["source_id"] for r in rows if r["axis"] == "clean" and r["label"] == "attack" and flags.get(r["id"])}
        out[name] = {**s, "headline": E.headline(s), "by_scope": scope_rates(rows, flags, caught)}
    base_caught = {r["source_id"] for r in rows if r["axis"] == "clean" and r["label"] == "attack" and base.get(r["id"])}
    out["base_by_scope"] = scope_rates(rows, base, base_caught)
    neg = [i for i, r in enumerate(rows) if r["axis"] == "clean" and r["label"] != "attack"]
    changed = sum(restored_texts[i] != texts[i] for i in neg)

    run.write_jsonl("scores.jsonl", [{"id": r["id"], "guardrail": g.name, "flagged_base": base[r["id"]],
                                      "flagged_restored": flags_r[r["id"]], "flagged_or": flags_or[r["id"]],
                                      "changed": restored_texts[i] != texts[i]} for i, r in enumerate(rows)])
    metrics = {"guardrail": g.name, "restorer": a.restorer, "restorer_info": dict(RS.INFO), "rows": len(rows),
               "restore_seconds": round(restore_sec, 1), "benign_clean_changed": {"k": changed, "n": len(neg)}, **out}
    headline = {f"{k}_{m}": v for k in ("restored", "or") for m, v in out[k]["headline"].items()}
    lines = [f"# {a.run_id} — {g.name} + {a.restorer}", "", f"기준 run: `{a.base_run}`", "",
             "| 지표 | 복원문만 | 원문 OR 복원문 |", "|---|---|---|"]
    for m in ("clean_tpr", "clean_fpr", "evasion_all", "evasion_if_caught_all", "fpr_all"):
        lines.append(f"| {m} | {100 * out['restored']['headline'].get(m, float('nan')):.1f}% | "
                     f"{100 * out['or']['headline'].get(m, float('nan')):.1f}% |")
    run.write_text("report.md", "\n".join(lines) + "\n")
    name = f"p02_{g.key.replace('-', '_')}_restored_v1.json"
    if (R.BASELINES_DIR / name).exists():
        raise SystemExit(f"기준값은 덮어쓰지 않습니다: baselines/{name}")
    R.publish_baseline(name, {"run_id": a.run_id, "metrics": metrics, "headline": headline}, run_id=a.run_id)
    run.finish(metrics, headline=headline)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
