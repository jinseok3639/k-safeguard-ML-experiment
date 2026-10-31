"""p05 욕설 필터 run — 원문과 난독화본을 필터 하나에 넣고 유형별 회피율·오탐률을 기록한다.

`--restorer`를 주면 필터에 넣기 전에 모든 문장(원문 포함)을 복원기로 고친다. 이때 `--base-run`(같은 필터,
복원기 없음)을 함께 주면, 복원기 없이 뚫렸던 난독화본이 복원 후 다시 잡히는 비율(회복률)도 낸다.

  원문 탐지율      욕설 원문 중 걸린 비율
  원문 오탐률      정상 원문 중 걸린 비율
  회피율(유형별)   원문이 걸린 욕설 문장의 난독화본 중 안 걸린 비율. 주 지표
  난독화 오탐률    정상 문장 난독화본 중 걸린 비율
  회복률           복원기 없이 회피된 난독화본 중 복원 후 걸린 비율(--base-run 필요)
  정상 변경률      복원기가 정상 원문을 바꾼 비율(--restorer)
  지연             필터(+복원기) 처리 시간 / 행 수

산출물: results/<run-id>/scores.jsonl(행별 판정, 문장 없음), metrics.json, report.md,
baselines/p05_<run 대상>_v1.json(집계만).

    python -m guardrail.exp.p05_profanity.run_filter --run-id p05.1-korcen --filter korcen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail.exp.p05_profanity import data as D
from guardrail.exp.p05_profanity import filters as F
from kslab import record as R
from kslab.metrics import wilson_interval

EXPERIMENT = "guardrail-p05-profanity-filter"
RESTORERS: dict = {}   # 복원기 이름 -> () -> (texts -> texts). 팀 복원기가 들어오면 등록한다


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def summarize(rows: list[dict], flagged: dict[str, bool], base: dict[str, bool] | None = None) -> dict:
    src = {r["id"]: r for r in rows if r["technique"] is None}
    obf = [r for r in rows if r["technique"] is not None]
    pos = [r for r in src.values() if r["label"] == 1]
    neg = [r for r in src.values() if r["label"] == 0]
    # 회피율의 분모: 원문이 걸린 욕설 문장. 복원 run은 복원기 없는 run의 원문 판정을 기준으로 삼는다
    caught = base if base is not None else flagged
    by_type, by_tech = defaultdict(lambda: [0, 0, 0, 0]), defaultdict(lambda: [0, 0])
    for r in obf:
        t = by_type[r["type"]]
        if r["label"] == 1 and caught[r["src"]]:
            t[0] += not flagged[r["id"]]; t[1] += 1
            by_tech[r["technique"]][0] += not flagged[r["id"]]; by_tech[r["technique"]][1] += 1
        elif r["label"] == 0:
            t[2] += flagged[r["id"]]; t[3] += 1
    ev = [r for r in obf if r["label"] == 1 and caught[r["src"]]]
    out = {
        "clean_tpr": rate(sum(flagged[r["id"]] for r in pos), len(pos)),
        "clean_fpr": rate(sum(flagged[r["id"]] for r in neg), len(neg)),
        "evasion": rate(sum(not flagged[r["id"]] for r in ev), len(ev)),
        "obf_fpr": rate(sum(flagged[r["id"]] for r in obf if r["label"] == 0), sum(r["label"] == 0 for r in obf)),
        "by_type": {k: {"evasion": rate(v[0], v[1]), "obf_fpr": rate(v[2], v[3])} for k, v in sorted(by_type.items())},
        "by_technique": {k: rate(v[0], v[1]) for k, v in sorted(by_tech.items())},
    }
    if base is not None:
        lost = [r for r in ev if not base[r["id"]]]
        out["recovery"] = rate(sum(flagged[r["id"]] for r in lost), len(lost))
        out["recovery_by_type"] = {}
        for t in sorted({r["type"] for r in lost}):
            sub = [r for r in lost if r["type"] == t]
            out["recovery_by_type"][t] = rate(sum(flagged[r["id"]] for r in sub), len(sub))
    return out


def _pct(m: dict) -> str:
    return "-" if not m["n"] else f"{100 * m['rate']:.1f}% [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}] ({m['k']}/{m['n']})"


def report(name: str, s: dict, meta: dict) -> str:
    lines = [f"# {meta['run_id']} — {name}", "",
             f"- 데이터: {D.SOURCE}, 원문 {meta['sources']}문장, 난독화본 {meta['variants']}행",
             f"- 복원기: {meta['restorer'] or '없음'}", f"- 지연: {meta['ms_per_row']:.3f} ms/행 ({meta['device']})", "",
             "| 지표 | 값 |", "|---|---|",
             f"| 원문 탐지율 | {_pct(s['clean_tpr'])} |", f"| 원문 오탐률 | {_pct(s['clean_fpr'])} |",
             f"| 회피율(전체) | {_pct(s['evasion'])} |", f"| 난독화 오탐률 | {_pct(s['obf_fpr'])} |"]
    if "recovery" in s:
        lines.append(f"| 회복률 | {_pct(s['recovery'])} |")
    if "benign_changed" in s:
        lines.append(f"| 정상 원문 변경률 | {_pct(s['benign_changed'])} |")
    lines += ["", "| 유형 | 회피율 | 난독화 오탐률 |", "|---|---|---|"]
    lines += [f"| {k} | {_pct(v['evasion'])} | {_pct(v['obf_fpr'])} |" for k, v in s["by_type"].items()]
    return "\n".join(lines) + "\n"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--filter", choices=sorted(F.FILTERS), required=True)
    ap.add_argument("--restorer", default=None, help="복원기 이름(RESTORERS)")
    ap.add_argument("--base-run", default=None, help="같은 필터의 복원기 없는 run id(회복률 계산)")
    ap.add_argument("--limit", type=int, default=None, help="스모크 run에서만: 앞에서 n문장")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    if a.limit and not R.tracking.is_smoke(a.run_id, R.SMOKE_MARKERS):
        raise SystemExit("--limit은 스모크 run(run-id에 smoke)에서만 쓴다.")
    if a.restorer and a.restorer not in RESTORERS:
        raise SystemExit(f"모르는 복원기 '{a.restorer}'. 등록된 것: {', '.join(RESTORERS) or '없음'}")
    flt = F.FILTERS[a.filter]
    sources = D.load()[:a.limit] if a.limit else D.load()
    rows = sources + D.variants(sources)
    base = None
    if a.base_run:
        with open(R.RESULTS_DIR / a.base_run / "scores.jsonl", encoding="utf-8") as f:
            base = {r["id"]: r["flagged"] for r in map(json.loads, f)}
        if set(base) != {r["id"] for r in rows}:
            raise SystemExit(f"{a.base_run}의 행이 지금 데이터와 다릅니다")

    run = R.Run.start(run_id=a.run_id, experiment=EXPERIMENT, config={
        "filter": flt.key, "filter_model": flt.name, "filter_revision": flt.version, "filter_rule": flt.rule,
        "restorer": a.restorer, "base_run": a.base_run, "limit": a.limit,
        "data_source": D.SOURCE, "data_sha256": D.SHA256, "sources": len(sources), "variants": len(rows) - len(sources),
        "obfuscation": "kslab 내부 기법 24개 각각, evalset single 축 강도·시드", "data_license_note": "MIT. 문장은 커밋하지 않는다",
    })
    run.add_input("curse-detection/dataset.txt", D.DATASET, rows=len(sources))
    print(f"[P05] {flt.name}  복원기 {a.restorer or '없음'}  원문 {len(sources)} + 난독화 {len(rows) - len(sources)}행", flush=True)

    texts = [r["text"] for r in rows]
    restore_sec = 0.0
    restored = texts
    if a.restorer:
        restore = RESTORERS[a.restorer]()
        restore(texts[:8])                 # 첫 호출 준비(모델 올리기)는 지연에서 뺀다
        t0 = time.perf_counter()
        restored = restore(texts)
        restore_sec = time.perf_counter() - t0
    fn = flt.load()
    fn(restored[:8])
    t0 = time.perf_counter()
    flags = fn(restored)
    seconds = restore_sec + time.perf_counter() - t0
    run.note("restore_ms_per_row", 1000 * restore_sec / len(rows))
    flagged = {r["id"]: bool(f) for r, f in zip(rows, flags)}

    s = summarize(rows, flagged, base)
    if a.restorer:
        neg = [i for i, r in enumerate(rows) if r["technique"] is None and r["label"] == 0]
        s["benign_changed"] = rate(sum(restored[i] != texts[i] for i in neg), len(neg))
    try:
        import torch
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    if a.filter == "korcen":
        device = "cpu(규칙)"
    meta = {"run_id": a.run_id, "sources": len(sources), "variants": len(rows) - len(sources),
            "restorer": a.restorer, "ms_per_row": 1000 * seconds / len(rows), "device": device}
    run.note("device", device)
    run.write_jsonl("scores.jsonl", [{"id": r["id"], "src": r["src"], "label": r["label"], "type": r["type"],
                                      "technique": r["technique"], "flagged": flagged[r["id"]],
                                      **({"changed": restored[i] != texts[i]} if a.restorer else {})}
                                     for i, r in enumerate(rows)])
    metrics = {"filter": flt.key, "restorer": a.restorer, "rows": len(rows), "seconds": seconds,
               "ms_per_row": meta["ms_per_row"], "device": device, **s}
    headline = {"clean_tpr": s["clean_tpr"]["rate"], "clean_fpr": s["clean_fpr"]["rate"], "evasion": s["evasion"]["rate"]}
    if "recovery" in s:
        headline["recovery"] = s["recovery"]["rate"]
    run.write_text("report.md", report(flt.name, s, meta))
    target = a.run_id.split("-", 1)[-1].replace("-", "_")
    name = f"p05_{target}_v1.json"
    if not a.limit and (R.BASELINES_DIR / name).exists():
        raise SystemExit(f"기준값은 덮어쓰지 않습니다: baselines/{name}")
    R.publish_baseline(name, {"run_id": a.run_id, "metrics": metrics, "headline": headline}, run_id=a.run_id)
    run.finish(metrics, headline=headline)
    print(report(flt.name, s, meta))


if __name__ == "__main__":
    main()
