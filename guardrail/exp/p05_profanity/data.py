"""p05 욕설 필터 평가 데이터 — Curse-detection-data 원문과 kslab 단일 기법 난독화본.

원문: github.com/2runo/Curse-detection-data(MIT) `dataset.txt`를 `data/ext/curse-detection-data/`에 클론한다.
커뮤니티 댓글 5,825문장, `문장|라벨`(1 욕설, 0 정상). 라벨 기준은 단순 욕설·차별·비하·성적 표현 등을 넓게 포함한다.
문장은 저장소에 커밋하지 않는다.

난독화본: 원문마다 kslab 내부 기법 24개를 하나씩 건다(평가셋 v1 single 축과 같은 강도·시드 규칙,
`kslab.evalset.single_steps`). 문장을 바꾸지 못한 기법은 행을 만들지 않는다. 유형은 `kslab.taxonomy`로 붙인다.

섞은 난독화본(compose 축): 평가셋 v1 compose 축과 같은 규칙(`kslab.evalset.compose_for`)으로 기법 n개(2·3·4)를
겹쳐 건다. 강도 0.3, 공격자 두 종류(사람형: KoNTe 실사용 빈도 가중, 자동형: 균등). 원문마다 조합(공격자 × n)당
하나씩 만든다. 행의 `type`은 실제로 걸린 기법 수(`n=2` 등), `technique`은 조합 이름(`human3` 등)이다.

P04 학습 분포 난독화본(p04mix 축): 팀 복원기 학습 데이터 생성기 `kslab.train_data.variant_steps`를 그대로 쓴다. 복원 대상
9유형에서만 기법 n개(1·2·3·4)를 고르고 강도를 기법마다 0.3·0.5·0.8에서 뽑는다. 원문마다 n당 하나. 복원기가 학습한 것과
같은 분포라 복원기에 가장 유리한 조건(상한)이다. 행 `type`은 실제로 걸린 기법 수, `technique`은 `p04n<n>`.

compose 축은 `kslab.evalset.COMPOSE_EXCLUDED`에 따라 초성체를 쓰지 않는다. 경계 유형(야민정음·닮은 외국문자·음절 늘이기)은 쓴다.

id: 원문 `cd-<줄 번호>`, 단일 난독화본 `cd-<줄 번호>/<기법>`, 섞은 난독화본 `cd-<줄 번호>/<공격자><n>`, p04mix `cd-<줄 번호>/p04n<n>`.
"""

from __future__ import annotations

import hashlib

from kslab import taxonomy as TX
from kslab import train_data as TD
from kslab.evalset import ATTACKERS, COMPOSE_SIZES, compose_for, single_steps, stable_seed
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


def compose_variants(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        for size in COMPOSE_SIZES:
            for attacker in ATTACKERS:
                name = f"{attacker}{size}"
                steps, _ = compose_for(s["text"], size, attacker, stable_seed(s["src"], name))
                res = run_internal(s["text"], steps)
                if res.applied and res.text != s["text"]:
                    out.append({"id": f"{s['src']}/{name}", "src": s["src"], "text": res.text, "label": s["label"],
                                "technique": name, "type": f"n={len(res.applied)}",
                                "ops": [steps[i].op for i in res.applied]})
    return out


def p04mix_variants(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        for size in TD.SIZES:
            steps = TD.variant_steps(s["text"], size, stable_seed(s["src"], "p04mix", size))
            res = run_internal(s["text"], steps)
            if res.applied and res.text != s["text"]:
                out.append({"id": f"{s['src']}/p04n{size}", "src": s["src"], "text": res.text, "label": s["label"],
                            "technique": f"p04n{size}", "type": f"n={len(res.applied)}",
                            "ops": [steps[i].op for i in res.applied]})
    return out


AXES = {"single": variants, "compose": compose_variants, "p04mix": p04mix_variants}
