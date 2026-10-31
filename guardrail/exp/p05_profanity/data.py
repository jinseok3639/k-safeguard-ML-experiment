"""p05 욕설 필터 평가 데이터 — Curse-detection-data 원문과 kslab 단일 기법 난독화본.

원문: github.com/2runo/Curse-detection-data(MIT) `dataset.txt`를 `data/ext/curse-detection-data/`에 클론한다.
커뮤니티 댓글 5,825문장, `문장|라벨`(1 욕설, 0 정상). 라벨 기준은 단순 욕설·차별·비하·성적 표현 등을 넓게 포함한다.
문장은 저장소에 커밋하지 않는다.

난독화본: 원문마다 kslab 내부 기법 24개를 하나씩 건다(평가셋 v1 single 축과 같은 강도·시드 규칙,
`kslab.evalset.single_steps`). 문장을 바꾸지 못한 기법은 행을 만들지 않는다. 유형은 `kslab.taxonomy`로 붙인다.

id: 원문 `cd-<줄 번호>`, 난독화본 `cd-<줄 번호>/<기법>`.
"""

from __future__ import annotations

import hashlib

from kslab import taxonomy as TX
from kslab.evalset import single_steps, stable_seed
from kslab.obfuscate import TRANSFORMS
from kslab.record import DATA_DIR
from kslab.recipe import run_internal

REPO_DIR = DATA_DIR / "ext" / "curse-detection-data"
DATASET = REPO_DIR / "dataset.txt"
DATA_COMMIT = "ff24162"
SHA256 = "1c3489417e4972dbbbdde19cc47bb8638292891f7f1a443ecbdc2e3c6843545a"
SOURCE = f"2runo/Curse-detection-data@{DATA_COMMIT} dataset.txt (MIT)"
FETCH = f"    git clone https://github.com/2runo/Curse-detection-data {REPO_DIR}"

TYPE_OF = {tech: t.id for t in TX.TYPES for tech in t.techniques}


def load() -> list[dict]:
    if not DATASET.exists():
        raise SystemExit(f"Curse-detection 데이터가 없습니다: {DATASET}\n{FETCH}")
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != SHA256:
        raise SystemExit(f"데이터 해시가 기록과 다릅니다: {DATASET}")
    rows = []
    with open(DATASET, encoding="utf-8") as f:
        for i, line in enumerate(f):
            text, label = line.rstrip("\r\n").rsplit("|", 1)
            rows.append({"id": f"cd-{i}", "src": f"cd-{i}", "text": text, "label": int(label),
                         "technique": None, "type": "clean"})
    return rows


def variants(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        for tech in sorted(TRANSFORMS):
            obf = run_internal(s["text"], single_steps(tech, stable_seed(s["src"], tech))).text
            if obf != s["text"]:
                out.append({"id": f"{s['src']}/{tech}", "src": s["src"], "text": obf, "label": s["label"],
                            "technique": tech, "type": TYPE_OF.get(tech, tech)})
    return out
