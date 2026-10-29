"""p03-secret-extraction LLM 생성 — Tensor Trust 입력을 LLM에 넣고 응답을 jsonl로 쓴다(Colab, vLLM).

샘플마다 공격 입력(최대 500토큰)과 진짜 비밀번호 입력(최대 50토큰)을 한 번씩 생성한다. greedy다.
메시지는 system·user·user 세 개이고 중간 assistant 차례가 없다. 맨 끝에서 한 번만 생성한다.

이 스크립트는 노트 게이트와 MLflow를 거치지 않는다(Colab에는 private/ 노트가 없다). 판정·집계·기록은
받아 온 파일로 로컬의 run_llm.py가 한다. 그래서 여기서는 응답과 생성 메타데이터만 남기고 판정하지 않는다.
이미 쓴 id는 건너뛰므로 Colab 세션이 끊겨도 같은 명령을 다시 돌리면 이어서 한다.

Colab 준비(저장소 루트에서)
    git clone https://github.com/HumanCompatibleAI/tensor-trust-data data/ext/tensor-trust-data
    git -C data/ext/tensor-trust-data checkout 747a75e096761ebc01bd3970158827326b4add23

    python -m guardrail.exp.p03_secret_extraction.generate_llm --model LGAI-EXAONE/EXAONE-4.0-32B-AWQ \\
        --revision 1ce84bc --out <Drive>/p03.6-exaone4-32b-awq.gen.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail.exp.p03_secret_extraction import tensortrust as T


def done_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def resolve_revision(model: str, revision: str | None) -> str:
    """짧은 해시나 None을 전체 커밋 해시로 바꾼다. 기록과 재현용."""
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision).sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=200, help="이 행 수마다 파일에 기록")
    ap.add_argument("--limit", type=int, default=None, help="파이프라인 점검용: 앞에서 n샘플만")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples, _ = T.load()
    if a.limit:
        samples = samples[:a.limit]
    rows = T.inputs(samples)
    done = done_ids(out)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)}행 중 {len(rows) - len(todo)}행 완료, {len(todo)}행 남음", flush=True)
    if not todo:
        return

    import torch
    from vllm import LLM, SamplingParams
    rev = resolve_revision(a.model, a.revision)
    llm = LLM(model=a.model, revision=rev, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, trust_remote_code=True)
    name = f"{a.model}@{rev}"
    t0 = time.time()
    finished = 0
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for kind in T.KINDS:
            part = [r for r in todo if r["kind"] == kind]
            if not part:
                continue
            sp = SamplingParams(temperature=0.0, max_tokens=part[0]["max_tokens"])
            print(f"[{kind}] {len(part)}행, 최대 {sp.max_tokens}토큰", flush=True)
            for i in range(0, len(part), a.chunk):
                chunk = part[i:i + a.chunk]
                # EXAONE 4.0의 추론 모드를 끈다(S02와 같다). 지원하지 않는 템플릿은 이 인자를 무시한다
                res = llm.chat([r["messages"] for r in chunk], sp,
                               chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
                for r, o in zip(chunk, res):
                    c = o.outputs[0]
                    rec = {"id": r["id"], "kind": kind, "model": name, "response": c.text,
                           "meta": {"finish_reason": c.finish_reason, "n_tokens": len(c.token_ids),
                                    "n_prompt_tokens": len(o.prompt_token_ids), "max_tokens": sp.max_tokens,
                                    "data_commit": T.DATA_COMMIT}}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                finished += len(chunk)
                el = time.time() - t0
                print(f"  [{kind}] {finished}/{len(todo)} ({100 * finished / len(todo):.0f}%)  "
                      f"{finished / el:.1f}행/초  {el:.0f}초", flush=True)
    print(f"완료: {len(todo)}행, {time.time() - t0:.0f}초, {torch.cuda.get_device_name(0)}", flush=True)


if __name__ == "__main__":
    main()
