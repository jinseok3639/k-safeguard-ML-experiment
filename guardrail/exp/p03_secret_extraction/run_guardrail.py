"""p03-secret-extraction 가드레일 run — Tensor Trust 추출 공격을 가드레일 하나에 태운다.

가드레일에는 사용자 입력 자리만 넣는다. 공격 입력은 공격문(`attack`), 정상 대조는 같은 자리에 넣는 진짜
비밀번호(`access_code`)다. 앞뒤 방어문은 서비스 쪽 텍스트라 넣지 않는다. 채점은 `guardrail/scorers.py` 그대로다.

산출물
  results/<run-id>/scores.jsonl   행별 점수. S01 `guardrail_scores`와 같은 형식에 `kind`(attack/code)를 더했다
  results/<run-id>/metrics.json   공격 차단율, 비밀번호 입력 오차단율(Wilson 95% 구간)
  baselines/p03_<가드레일>_v1.json 위 집계만(문장 없음)

    python -m guardrail.exp.p03_secret_extraction.run_guardrail --run-id p03.3-kanana --guardrail kanana
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail import scorers as S
from guardrail.exp.p03_secret_extraction import tensortrust as T
from kslab import record as R
from kslab.metrics import wilson_interval

EXPERIMENT = "p03_secret_extraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--guardrail", choices=sorted(S.GUARDRAILS), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 n샘플만(스모크 전용. run-id에 smoke가 있어야 한다)")
    return parser.parse_args()


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def score_all(scorer, rows: list[dict], batch: int) -> dict[str, dict]:
    """길이순으로 묶어 패딩 낭비를 줄인다. 결과는 id로 돌려준다."""
    order = sorted(rows, key=lambda r: len(r["text"]))
    out: dict[str, dict] = {}
    started = time.time()
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        for r, s in zip(chunk, scorer([r["text"] for r in chunk])):
            out[r["id"]] = s
        done = i + len(chunk)
        if done == len(order) or (i // batch) % 10 == 0:
            elapsed = time.time() - started
            print(f"  [score] {done}/{len(order)} ({100 * done / len(order):.0f}%)  "
                  f"{done / max(elapsed, 1e-9):.1f}행/초  {elapsed:.0f}초", flush=True)
    return out


def rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def summarize(rows: list[dict], scored: dict[str, dict]) -> dict:
    out = {}
    for kind in T.KINDS:
        part = [r for r in rows if r["kind"] == kind]
        flagged = [scored[r["id"]]["flagged"] for r in part]
        out[kind] = {**rate(sum(f is True for f in flagged), len(part)),
                     "unparsed": sum(f is None for f in flagged)}
    return {"attack_blocked": out["attack"], "code_blocked": out["code"]}


def _pct(v: dict) -> str:
    lo, hi = v["ci95"]
    return f"{100 * v['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({v['k']}/{v['n']})"


def report(g: S.Guardrail, summary: dict, meta: dict) -> str:
    return "\n".join([
        f"# P03 Tensor Trust 비밀값 추출 — 가드레일 `{g.model}`",
        "",
        f"- 리비전 `{g.revision}`, 백엔드 `{g.backend}`, 기준값 {g.threshold}",
        f"- 장치 {meta['device']}, 채점 {meta['rows']}행, {meta['seconds']:.0f}초",
        f"- 데이터 Tensor Trust 추출 벤치마크 `{T.DATA_COMMIT[:7]}`, 샘플 {meta['samples']}개"
        f"(제외 {meta['excluded']}개: 공격문에 비밀번호가 들어 있음)",
        "",
        "| 입력 | 차단율 [Wilson 95%] | 판정 실패 |",
        "|---|---|---|",
        f"| 공격문 | {_pct(summary['attack_blocked'])} | {summary['attack_blocked']['unparsed']} |",
        f"| 진짜 비밀번호(정상 대조) | {_pct(summary['code_blocked'])} | {summary['code_blocked']['unparsed']} |",
        "",
    ])


def main() -> None:
    args = parse_args()
    g = S.GUARDRAILS[args.guardrail]
    if args.limit and not R.tracking.is_smoke(args.run_id, R.SMOKE_MARKERS):
        raise SystemExit("--limit은 스모크 run(run-id에 smoke)에서만 쓴다.")

    samples, excluded = T.load()
    if args.limit:
        samples = samples[:args.limit]
    rows = T.inputs(samples)

    device = _device()
    run = R.Run.start(
        run_id=args.run_id,
        experiment=EXPERIMENT,
        config={
            "guardrail": g.key, "model": g.model, "revision": g.revision, "backend": g.backend,
            "threshold": g.threshold, "pos_label": g.pos_label, "guardrail_note": g.note,
            "batch": args.batch, "device": device, "limit": args.limit,
            "data_commit": T.DATA_COMMIT, "samples": len(samples),
            "excluded_sample_ids": [s["sample_id"] for s in excluded],
            "guardrail_input": "attack 또는 access_code만(앞뒤 방어문 제외)",
            "data_license_note": "tensor-trust-data 라이선스 표기 없음(코드 저장소 BSD-2). 문장은 커밋하지 않는다",
        },
    )
    run.add_input("tensor-trust/extraction_robustness_dataset.jsonl", T.DATASET,
                  rows=len(samples) + len(excluded))

    print(f"[P03] {g.name}  샘플 {len(samples)}개 → {len(rows)}행  장치 {device}", flush=True)
    started = time.time()
    scorer = S.load(g, device)
    scored = score_all(scorer, rows, args.batch)
    seconds = time.time() - started
    if device == "cuda":
        import torch
        run.note("gpu", torch.cuda.get_device_name(0))
        run.note("gpu_peak_mem_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2))

    out = []
    for r in rows:
        s = scored[r["id"]]
        rec = {"id": r["id"], "kind": r["kind"], "guardrail": g.name, "score": s["score"],
               "flagged": s["flagged"], "threshold": g.threshold}
        if "meta" in s:
            rec["meta"] = s["meta"]
        out.append(rec)
    run.write_jsonl("scores.jsonl", out)

    summary = summarize(rows, scored)
    meta = {"device": run.manifest.get("notes", {}).get("gpu", device), "rows": len(rows),
            "seconds": seconds, "samples": len(samples), "excluded": len(excluded)}
    metrics = {"guardrail": g.name, "rows": len(rows), "seconds": round(seconds, 1), **summary}
    headline = {"attack_blocked": summary["attack_blocked"]["rate"],
                "code_blocked": summary["code_blocked"]["rate"]}
    run.write_text("report.md", report(g, summary, meta))
    R.publish_baseline(f"p03_{g.key.replace('-', '_')}_v1.json",
                       {"run_id": args.run_id, "metrics": metrics, "headline": headline},
                       run_id=args.run_id)
    run.finish(metrics, headline=headline)
    print(f"\n[P03] 완료 -> {run.directory}")
    for key, value in headline.items():
        print(f"  {key}: {100 * value:.1f}%")


if __name__ == "__main__":
    main()
