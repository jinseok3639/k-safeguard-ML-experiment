"""korean-jamo-typo(MIT, EMNLP 2026, github.com/SJLee0311/korean-jamo-typo) 드라이버. 전용 venv로 실행한다.

사용: <venv python> -X utf8 -I jamo_typo_driver.py <korean-jamo-typo 경로>  (표준입력 JSONL -> 표준출력 JSONL)

입력 한 줄: {"text": ..., "op": "T1:3", "seed": int}   T1~T5, 콜론 뒤는 바꿀 자리 수
출력 한 줄: {"out": ...}

키보드 오타라 의도적 난독화가 아니다. 평가셋에서는 "비의도 노이즈" 대조군으로만 쓴다.
"""

from __future__ import annotations

import json
import random
import sys


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    import perturbations as P

    ops = {"T1": P.jamo_substitution, "T2": P.jongseong_deletion, "T3": P.jamo_repetition,
           "T4": P.space_deletion, "T5": P.jamo_transposition}
    for line in sys.stdin:
        item = json.loads(line)
        random.seed(item["seed"])   # 원본이 전역 random을 쓴다
        name, _, count = item["op"].partition(":")
        out = ops[name](item["text"], int(count or 1))
        print(json.dumps({"out": out}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
