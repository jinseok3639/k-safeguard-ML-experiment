"""p03-secret-extraction 가드레일 run — 가드레일 하나의 점수를 집계해 run으로 기록한다.

점수는 두 길로 얻는다. 가드레일 입력과 채점 방식은 `score_guardrail.py`를 본다.
  --scores 없음   여기서 바로 채점한다(로컬 GPU)
  --scores 파일   Colab의 score_guardrail.py가 쓴 점수 파일을 읽는다. 모델을 불러오지 않는다.
                  파일이 지금 데이터의 id를 빠짐없이 한 번씩 담았는지, --data·--guardrail과 같은지 확인한다

산출물
  results/<run-id>/scores.jsonl   행별 점수. S01 `guardrail_scores`와 같은 형식에 `kind`(attack/code)를 더했다
  results/<run-id>/metrics.json   공격 차단율, 비밀번호 입력 오차단율(Wilson 95% 구간)
  baselines/p03_<run 대상>_v1.json 위 집계만(문장 없음). 이름은 run id의 대상 부분에서 딴다

    python -m guardrail.exp.p03_secret_extraction.run_guardrail --data en --run-id p03.3-kanana --guardrail kanana \\
        [--scores <받아 온 .scores.jsonl>]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail import scorers as S
from guardrail.exp.p03_secret_extraction import score_guardrail as SG
from guardrail.exp.p03_secret_extraction import tensortrust as T
from kslab import record as R
from kslab.metrics import wilson_interval

EXPERIMENT = "p03_secret_extraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=sorted(T.DATASETS), required=True,
                        help="입력 데이터: en(원문), ko(한국어 번역본), ko-<기법>(번역본 난독화, make_obf.py 참고)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--guardrail", choices=sorted(S.GUARDRAILS), required=True)
    parser.add_argument("--scores", type=Path, default=None, help="score_guardrail.py가 쓴 .scores.jsonl")
    parser.add_argument("--batch", type=int, default=32, help="여기서 채점할 때만 쓴다")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 n샘플만(스모크 전용. run-id에 smoke가 있어야 한다)")
    return parser.parse_args()


def read_scores(path: Path, expected: list[str], data: str, g: S.Guardrail) -> tuple[list[dict], dict]:
    """(행별 점수, 채점 환경). 점수 파일이 지금 데이터·가드레일과 맞지 않으면 멈춘다."""
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    counts = collections.Counter(r["id"] for r in recs)
    dup = [i for i, c in counts.items() if c > 1]
    missing = [i for i in expected if i not in counts]
    extra = [i for i in counts if i not in set(expected)]
    if dup or missing or extra:
        raise SystemExit(f"점수 파일이 데이터와 맞지 않습니다: 중복 {len(dup)}, 누락 {len(missing)}, "
                         f"데이터에 없는 id {len(extra)}")
    for field, want in (("data", data), ("guardrail", g.name)):
        got = {r.get(field) for r in recs}
        if got != {want}:
            raise SystemExit(f"점수 파일의 {field}가 {sorted(map(str, got))}인데 지금 run은 {want}입니다")
    env = recs[0].get("env", {})
    order = {i: k for k, i in enumerate(expected)}
    return [{k: v for k, v in r.items() if k != "env"} for r in sorted(recs, key=lambda r: order[r["id"]])], env


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
        f"- 데이터 `{meta['data']}` ({meta['source']}), 샘플 {meta['samples']}개"
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

    data = T.dataset(args.data)
    samples, excluded = T.load_dataset(data.name)
    if args.limit:
        samples = samples[:args.limit]
    rows = T.inputs(samples)
    # 점수 파일이 맞는지는 노트 게이트를 통과하기 전에 확인한다
    from_file = read_scores(args.scores, [r["id"] for r in rows], data.name, g) if args.scores else None

    run = R.Run.start(
        run_id=args.run_id,
        experiment=EXPERIMENT,
        config={
            "guardrail": g.key, "model": g.model, "revision": g.revision, "backend": g.backend,
            "threshold": g.threshold, "pos_label": g.pos_label, "guardrail_note": g.note,
            "scored_by": "score_guardrail.py 파일" if args.scores else "이 run",
            "batch": None if args.scores else args.batch, "limit": args.limit,
            "data": data.name, "data_source": data.source, "data_sha256": data.sha256,
            "data_commit": T.DATA_COMMIT, "samples": len(samples),
            "excluded_sample_ids": [s["sample_id"] for s in excluded],
            "guardrail_input": "attack 또는 access_code만(앞뒤 방어문 제외)",
            "data_license_note": "tensor-trust-data 라이선스 표기 없음(코드 저장소 BSD-2). 문장은 커밋하지 않는다",
        },
    )
    run.add_input(f"tensor-trust/{data.path.name}", data.path,
                  rows=len(samples) + len(excluded))

    print(f"[P03] {g.name}  데이터 {data.name}  샘플 {len(samples)}개 → {len(rows)}행", flush=True)
    if from_file:
        recs, env = from_file
        run.add_input("scores", args.scores, rows=len(recs))
        print(f"  점수 파일 {args.scores.name} 읽음 (채점 {env.get('gpu', env.get('device', '?'))})", flush=True)
    else:
        recs, env = SG.score(g, rows, data.name, args.batch)
    for key, value in env.items():
        run.note(key, value)
    run.write_jsonl("scores.jsonl", recs)

    scored = {r["id"]: r for r in recs}
    summary = summarize(rows, scored)
    seconds = env.get("seconds", 0.0)
    meta = {"device": env.get("gpu", env.get("device", "?")), "rows": len(rows),
            "seconds": seconds, "samples": len(samples), "excluded": len(excluded),
            "data": data.name, "source": data.source}
    metrics = {"guardrail": g.name, "data": data.name, "rows": len(rows), "seconds": seconds, **summary}
    headline = {"attack_blocked": summary["attack_blocked"]["rate"],
                "code_blocked": summary["code_blocked"]["rate"]}
    run.write_text("report.md", report(g, summary, meta))
    target = args.run_id.split("-", 1)[-1].replace("-", "_")
    R.publish_baseline(f"p03_{target}_v1.json",
                       {"run_id": args.run_id, "metrics": metrics, "headline": headline},
                       run_id=args.run_id)
    run.finish(metrics, headline=headline)
    print(f"\n[P03] 완료 -> {run.directory}")
    for key, value in headline.items():
        print(f"  {key}: {100 * value:.1f}%")


if __name__ == "__main__":
    main()
