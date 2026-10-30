"""p03 번역본 난독화 데이터 — 한국어 번역본 공격문에 kslab 난독화 기법 하나를 건 파일을 만든다.

공격자가 바꿀 수 있는 `attack`만 바꾼다. 방어문(pre_prompt·post_prompt)과 비밀번호(access_code)는 그대로다.
강도는 kslab 평가셋 single 축과 같은 0.5이고, 시드는 sample_id와 기법 이름에서 정한다. 같은 입력이면
어느 환경에서든 같은 파일이 나오므로 `tensortrust.DATASETS`에 적은 sha256으로 확인할 수 있다.

  ko-tensify  `tensify`(된소리, 유형 phon_initial)
  ko-hangul   `lookalike_hangul`(닮은 한글·야민정음, 유형 shape_hangul). KOTOX 사전
              `data/ext/kotox-src`(커밋 a01c917)가 필요하다. Windows에서는 core.autocrlf=false로 받는다
  ko-cram      `jongseong_cram`(받침 채우기, 유형 phon_insert)
  ko-semivowel `semivowel_insert`(반모음 추가, 유형 phon_insert)
  ko-foreign   `lookalike_foreign`(닮은 비한글 문자, 유형 shape_foreign)
  ko-jamo      `jamo_decompose`(자모 분해, 유형 jamo_decompose)

    python -m guardrail.exp.p03_secret_extraction.make_obf --data ko-tensify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail.exp.p03_secret_extraction import tensortrust as T
from kslab import obfuscate as O
from kslab.evalset import SINGLE_INTENSITY, stable_seed

OPS = {"ko-tensify": "tensify", "ko-hangul": "lookalike_hangul", "ko-cram": "jongseong_cram",
       "ko-semivowel": "semivowel_insert", "ko-foreign": "lookalike_foreign", "ko-jamo": "jamo_decompose"}


def obfuscate_rows(rows: list[dict], op: str) -> list[dict]:
    fn = O.TRANSFORMS[op]
    out = []
    for r in rows:
        new = dict(r)
        new["attack"] = fn(r["attack"], SINGLE_INTENSITY, stable_seed(r["sample_id"], op))
        out.append(new)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=sorted(OPS), required=True)
    a = ap.parse_args()

    src = T.dataset("ko")
    if T.sha256(src.path) != src.sha256:
        raise SystemExit(f"번역본 해시가 기록과 다릅니다: {src.path}")
    with open(src.path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out_rows = obfuscate_rows(rows, OPS[a.data])

    dst = T.obf_path(a.data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = sum(r["attack"] != o["attack"] for r, o in zip(rows, out_rows))
    print(f"{dst}\n  {len(out_rows)}행, 공격문이 바뀐 행 {changed}, sha256 {T.sha256(dst)}")


if __name__ == "__main__":
    main()
