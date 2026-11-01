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

핵심어 표적(`ko-kw-<기법>`): 공격문 전체가 아니라 인젝션 핵심어(`KEYWORDS`)가 들어 있는 어절에만 기법을 강도 1.0으로
건다. 사람이 필터를 우회할 때처럼 걸리는 단어만 바꾸고 나머지는 그대로 둔다. 핵심어 목록은 결과를 보기 전에
일반적인 인젝션 어휘로 정했고(2026-10-30), 번역본에서는 포함 비율만 확인했다(564/570 샘플, 어절의 11%).

    python -m guardrail.exp.p03_secret_extraction.make_obf --data ko-tensify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from guardrail.exp.p03_secret_extraction import tensortrust as T
from kslab import obfuscate as O
from kslab.evalset import SINGLE_INTENSITY, stable_seed

OPS = {"ko-tensify": "tensify", "ko-hangul": "lookalike_hangul", "ko-cram": "jongseong_cram",
       "ko-semivowel": "semivowel_insert", "ko-foreign": "lookalike_foreign", "ko-jamo": "jamo_decompose"}
TARGETED = {"ko-kw-cram": "jongseong_cram", "ko-kw-foreign": "lookalike_foreign",
            "ko-kw-semivowel": "semivowel_insert", "ko-kw-tensify": "tensify"}
KEYWORDS = ("무시", "지시", "명령", "비밀번호", "암호", "패스워드", "접근코드", "액세스코드", "코드", "프롬프트", "시스템",
            "규칙", "지침", "출력", "반복", "말해", "알려", "공개", "노출", "이전", "위의", "앞의", "잊어", "보안",
            "관리자", "개발자", "권한", "우회", "해제", "무효", "재정의")
TARGETED_INTENSITY = 1.0


def obfuscate_rows(rows: list[dict], op: str) -> list[dict]:
    fn = O.TRANSFORMS[op]
    out = []
    for r in rows:
        new = dict(r)
        new["attack"] = fn(r["attack"], SINGLE_INTENSITY, stable_seed(r["sample_id"], op))
        out.append(new)
    return out


def obfuscate_keywords(rows: list[dict], op: str) -> list[dict]:
    """핵심어가 든 어절만 바꾼다. 어절마다 시드를 따로 둬 같은 단어도 위치마다 다르게 바뀔 수 있다."""
    fn = O.TRANSFORMS[op]
    out = []
    for r in rows:
        n = 0

        def sub(m: re.Match) -> str:
            nonlocal n
            word = m.group(0)
            if not any(k in word for k in KEYWORDS):
                return word
            n += 1
            return fn(word, TARGETED_INTENSITY, stable_seed(r["sample_id"], f"{op}#{n}"))

        new = dict(r)
        new["attack"] = re.sub(r"\S+", sub, r["attack"])
        out.append(new)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=sorted(OPS) + sorted(TARGETED), required=True)
    a = ap.parse_args()

    src = T.dataset("ko")
    if T.sha256(src.path) != src.sha256:
        raise SystemExit(f"번역본 해시가 기록과 다릅니다: {src.path}")
    with open(src.path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out_rows = (obfuscate_keywords(rows, TARGETED[a.data]) if a.data in TARGETED
                else obfuscate_rows(rows, OPS[a.data]))

    dst = T.obf_path(a.data)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    changed = sum(r["attack"] != o["attack"] for r, o in zip(rows, out_rows))
    print(f"{dst}\n  {len(out_rows)}행, 공격문이 바뀐 행 {changed}, sha256 {T.sha256(dst)}")


if __name__ == "__main__":
    main()
