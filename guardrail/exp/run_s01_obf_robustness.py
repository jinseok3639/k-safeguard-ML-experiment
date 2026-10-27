"""S01 — 난독화 강건성: 가드레일 하나에 kslab 평가셋 v1 전체를 태운다.

묻는 것: 어떤 난독화 유형이 이 가드레일을 실제로 통과시키나. 정상 문장은 난독화만으로
오차단되나. 원문 695개(공격 301·정상 204+190)를 같은 가드레일에 함께 태워 기준선으로 쓴다.

산출물
  results/<run-id>/scores.jsonl   문장별 점수. 전달 문서의 guardrail_scores 형식(P04가 이어 쓴다)
  results/<run-id>/metrics.json   유형·축·기법·기법 수별 회피율, 원문 차단 조건부 회피율, 오차단율
  baselines/s01_<가드레일>_v1.json  위 집계만(문장 없음)

    python -m guardrail.exp.run_s01_obf_robustness --run-id s01.1-kanana --guardrail kanana
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from guardrail import evalset as E
from guardrail import scorers as S
from kslab import record as R

EXPERIMENT = "s01_obf_robustness"
DEFAULT_EVALSET = R.DATA_DIR / "evalset" / "kslab-evalset-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--guardrail", choices=sorted(S.GUARDRAILS), required=True)
    parser.add_argument("--evalset", type=Path, default=DEFAULT_EVALSET)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--load-4bit", action="store_true", help="gemma 전용")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="gemma 전용")
    parser.add_argument("--gemma-mode", choices=("logit", "generate"), default="logit",
                        help="gemma 전용. logit은 순전파 1회, generate는 JSON 생성(S01.5에서 쓴 방식)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 n행만(스모크 전용. run-id에 smoke가 있어야 한다)")
    return parser.parse_args()


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def score_all(scorer, rows: list[dict], batch: int) -> dict[str, dict]:
    """길이순으로 묶어 패딩 낭비를 줄인다. 결과는 id로 돌려준다."""
    order = sorted(rows, key=lambda r: len(r["obfuscated"]))
    out: dict[str, dict] = {}
    started = time.time()
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        for r, s in zip(chunk, scorer([r["obfuscated"] for r in chunk])):
            out[r["id"]] = s
        done = i + len(chunk)
        if done == len(order) or (i // batch) % 50 == 0:
            elapsed = time.time() - started
            print(f"  {done}/{len(order)}  {done / max(elapsed, 1e-9):.1f}행/초  {elapsed:.0f}초")
    return out


def explain_benign_blocks(scorer, rows: list[dict], scored: dict[str, dict], batch: int) -> None:
    """gemma logit 모드: 차단된 정상 문장만 JSON을 생성해 차단 사유(type)를 meta에 붙인다.

    gemma의 blocked는 인젝션 말고도 PII·유해성으로 켜지므로 오차단은 사유별로 나눠 봐야 한다.
    생성 판정이 logit 판정과 다르면 meta.generated_blocked에 남는다.
    """
    targets = [r for r in rows if r["label"] == "benign" and scored[r["id"]]["flagged"]]
    print(f"  차단된 정상 문장 {len(targets)}행의 사유 생성")
    for i in range(0, len(targets), batch):
        chunk = targets[i:i + batch]
        for r, gen in zip(chunk, scorer.generate([r["obfuscated"] for r in chunk])):
            meta = scored[r["id"]].setdefault("meta", {})
            meta["type"] = gen["meta"]["type"]
            meta["generated_blocked"] = gen["flagged"]


def _pct(v: dict | None) -> str:
    if not v:
        return "-"
    lo, hi = v["ci95"]
    return f"{100 * v['rate_source_balanced']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def report(g: S.Guardrail, summary: dict, types: list[str], meta: dict) -> str:
    groups = summary["groups"]
    clean = groups["clean"]
    lines = [
        f"# S01 난독화 강건성 — `{g.model}`",
        "",
        f"- 리비전 `{g.revision}`, 백엔드 `{g.backend}`, 기준값 {g.threshold}",
        f"- 장치 {meta['device']}, 채점 {meta['rows']}행, {meta['seconds']:.0f}초",
        f"- 원문 공격 탐지율 {100 * summary['clean_tpr']:.1f}% "
        f"({summary['clean_attacks_caught']}/{clean['evasion']['n']}), "
        f"원문 정상 오차단율 {_pct(clean['fpr'])}%",
        "",
        "백분율은 원문 균형 추정치 [원문 단위 bootstrap 95% CI]다. 원문마다 비율을 먼저 내고 원문끼리 평균한다. "
        "단순 비율(행 단위)은 metrics.json의 `rate`. 회피|원문차단은 원문이 차단된 공격만 놓고 본 회피율이다.",
        "",
        "| 유형 | 공격 | 회피% | 회피\\|원문차단% | 정상 | 오차단% | 판정 실패 |",
        "|---|---|---|---|---|---|---|",
    ]
    keys = [f"type:{t}" for t in types] + [f"axis:{a}" for a in E.AXES] + ["all"]
    for key in keys:
        x = groups.get(key)
        if not x:
            continue
        n_atk = x["evasion"]["n"] if x["evasion"] else 0
        n_bng = x["fpr"]["n"] if x["fpr"] else 0
        lines.append(f"| {key.split(':')[-1] if key.startswith('type:') else key} | {n_atk} | "
                     f"{_pct(x['evasion'])} | {_pct(x['evasion_if_caught'])} | {n_bng} | "
                     f"{_pct(x['fpr'])} | {x['unparsed']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    g = S.GUARDRAILS[args.guardrail]
    if args.limit and not R.tracking.is_smoke(args.run_id, R.SMOKE_MARKERS):
        raise SystemExit("--limit은 스모크 run(run-id에 smoke)에서만 쓴다.")

    sums = E.verify(args.evalset)
    records = E.load_records(args.evalset)
    types = E.load_types(args.evalset)
    rows = E.clean_rows(records) + records
    if args.limit:
        rows = rows[:args.limit]

    device = _device()
    run = R.Run.start(
        run_id=args.run_id,
        experiment=EXPERIMENT,
        config={
            "guardrail": g.key, "model": g.model, "revision": g.revision, "backend": g.backend,
            "threshold": g.threshold, "pos_label": g.pos_label, "guardrail_note": g.note,
            "batch": args.batch, "bootstrap_samples": args.bootstrap_samples, "device": device,
            "load_4bit": args.load_4bit, "max_new_tokens": args.max_new_tokens,
            "gemma_mode": args.gemma_mode if g.backend == "gemma" else None,
            "limit": args.limit, "evalset_sha256": sums,
            "evalset_license_note": "KOTOX·kotka·korean-jamo-typo MIT, KoG2Padvanced 표시 없음(내부 평가 전용), "
                                    "korean_guardrail_test CC BY 4.0",
        },
    )
    run.add_input("evalset/records.jsonl", args.evalset / "records.jsonl", rows=len(records))
    run.add_input("evalset/types.csv", args.evalset / "types.csv")

    print(f"[S01] {g.name}  {len(rows)}행  장치 {device}")
    started = time.time()
    gemma = g.backend == "gemma"
    kwargs = {"load_4bit": args.load_4bit, "max_new_tokens": args.max_new_tokens,
              "mode": args.gemma_mode} if gemma else {}
    scorer = S.load(g, device, **kwargs)
    scored = score_all(scorer, rows, args.batch)
    seconds = time.time() - started
    if gemma and args.gemma_mode == "logit":
        explain_benign_blocks(scorer, rows, scored, args.batch)
    if device == "cuda":
        import torch
        run.note("gpu", torch.cuda.get_device_name(0))
        run.note("gpu_peak_mem_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2))

    out = []
    for r in rows:
        s = scored[r["id"]]
        threshold = None if gemma and args.gemma_mode == "generate" else g.threshold
        rec = {"id": r["id"], "guardrail": g.name, "score": s["score"], "flagged": s["flagged"],
               "threshold": threshold}
        if "meta" in s:
            rec["meta"] = s["meta"]
        out.append(rec)
    run.write_jsonl("scores.jsonl", out)

    summary = E.aggregate(rows, {r["id"]: scored[r["id"]]["flagged"] for r in rows},
                          bootstrap_samples=args.bootstrap_samples)
    meta = {"device": run.manifest.get("notes", {}).get("gpu", device), "rows": len(rows),
            "seconds": seconds}
    metrics = {"guardrail": g.name, "rows": len(rows), "seconds": round(seconds, 1),
               "rows_per_second": round(len(rows) / seconds, 1), **summary}
    run.write_text("report.md", report(g, summary, types, meta))
    headline = E.headline(summary)
    R.publish_baseline(f"s01_{g.key.replace('-', '_')}_v1.json",
                       {"run_id": args.run_id, "metrics": metrics, "headline": headline},
                       run_id=args.run_id)
    run.finish(metrics, headline=headline)
    print(f"\n[S01] 완료 -> {run.directory}")
    for key, value in headline.items():
        print(f"  {key}: {100 * value:.1f}%")


if __name__ == "__main__":
    main()
