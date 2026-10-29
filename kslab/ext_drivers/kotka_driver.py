"""kotka(MIT, github.com/Astro36/kotka) 드라이버. kotka 전용 venv의 파이썬으로 실행한다.

사용: <kotka venv python> -X utf8 -I kotka_driver.py <kotka 경로>  (표준입력 JSONL -> 표준출력 JSONL)

입력 한 줄: {"text": ..., "op": "squaredog" | "yamin" | "shred:0.3", "seed": int}
출력 한 줄: {"out": ...}
"""

from __future__ import annotations

import json
import random
import sys


def main() -> None:
    sys.path.insert(0, sys.argv[1])
    from kotka.lipogram import replace_phoneme
    from kotka.lipogram.rules import SquareDog
    from kotka.shredder import shred_syllable
    from kotka.yaminizer import encode_yamin

    for line in sys.stdin:
        item = json.loads(line)
        random.seed(item["seed"])   # shredder는 전역 random을 쓴다
        name, _, arg = item["op"].partition(":")
        text = item["text"]
        if name == "squaredog":
            out = replace_phoneme(text, rule=SquareDog())
        elif name == "yamin":
            out = encode_yamin(text)
        elif name == "shred":
            out = shred_syllable(text, active_rate=float(arg))
        else:
            raise ValueError(f"모르는 op: {item['op']}")
        print(json.dumps({"out": out}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
