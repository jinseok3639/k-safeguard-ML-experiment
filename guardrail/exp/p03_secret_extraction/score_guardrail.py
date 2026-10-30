"""p03-secret-extraction 가드레일 채점 — Tensor Trust 입력을 가드레일 하나에 넣고 점수를 jsonl로 쓴다(Colab).

가드레일에는 사용자 입력 자리만 넣는다. 공격 입력은 공격문(`attack`), 정상 대조는 같은 자리에 넣는 진짜
비밀번호(`access_code`)다. 앞뒤 방어문은 서비스 쪽 텍스트라 넣지 않는다. 채점은 `guardrail/scorers.py` 그대로다.

generate_llm.py와 같이 노트 게이트와 MLflow를 거치지 않고 점수 파일만 쓴다. 집계·기록은 받아 온 파일로
로컬의 `run_guardrail.py --scores`가 한다. 1138행이라 한 번에 끝나므로 이어 하기 없이 끝에 파일을 통째로 쓴다.

    python -m guardrail.exp.p03_secret_extraction.score_guardrail --data en --guardrail kanana \\
        --out <Drive>/kanana-en.scores.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail import scorers as S
from guardrail.exp.p03_secret_extraction import tensortrust as T


def device() -> str:
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


def score(g: S.Guardrail, rows: list[dict], data: str, batch: int) -> tuple[list[dict], dict]:
    """(행별 점수 기록, 채점 환경). 기록 형식은 S01 `guardrail_scores`에 kind·data를 더한 것이다."""
    dev = device()
    started = time.time()
    scored = score_all(S.load(g, dev), rows, batch)
    env = {"device": dev, "seconds": round(time.time() - started, 1)}
    if dev == "cuda":
        import torch
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out = []
    for r in rows:
        s = scored[r["id"]]
        rec = {"id": r["id"], "kind": r["kind"], "data": data, "guardrail": g.name, "score": s["score"],
               "flagged": s["flagged"], "threshold": g.threshold}
        if "meta" in s:
            rec["meta"] = s["meta"]
        out.append(rec)
    return out, env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=sorted(T.DATASETS), required=True,
                    help="입력 데이터: en(원문), ko(한국어 번역본), ko-<기법>(번역본 난독화, make_obf.py 참고)")
    ap.add_argument("--guardrail", choices=sorted(S.GUARDRAILS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="파이프라인 점검용: 앞에서 n샘플만")
    a = ap.parse_args()

    g = S.GUARDRAILS[a.guardrail]
    samples, _ = T.load_dataset(a.data)
    if a.limit:
        samples = samples[:a.limit]
    rows = T.inputs(samples)
    print(f"[{a.data}] {g.name}  샘플 {len(samples)}개 → {len(rows)}행", flush=True)

    recs, env = score(g, rows, a.data, a.batch)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for rec in recs:
            f.write(json.dumps({**rec, "env": env}, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"완료: {len(recs)}행, {env['seconds']:.0f}초, {env.get('gpu', env['device'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
