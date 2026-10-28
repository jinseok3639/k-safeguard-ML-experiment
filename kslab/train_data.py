"""P04 정규화기 학습 데이터 — 코퍼스 문장에 복원 대상 유형만 섞어 건다(D3 자동형 ①).

평가셋(`evalset.py`)과 같은 레시피 형식을 쓰되 세 가지가 다르다.

- 원문은 위키·KLUE 코퍼스다. 평가셋 원문 695와 같은 문장은 뺀다.
- 복원 대상(`taxonomy` scope "target") 유형만 쓴다. 경계·범위 밖 유형과 외부 생성기는 쓰지 않는다(홀드아웃).
- 기법마다 강도를 `INTENSITIES`에서 무작위로 고른다. 평가셋은 단일 0.5·복합 0.3으로 고정이라, 한 강도로만 배우면 다른 강도를
  못 본다(예: `break_spacing`은 0.5 이상이면 공백을 전부 지우고 그 아래면 공백을 넣는다). 발음 규칙 기법은 평가셋처럼 1.0.

문장마다 변형 `VARIANTS`개(기법 수는 `SIZES`에서 고름)와, `IDENTITY_SHARE` 비율로 손대지 않은 쌍을 만든다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kslab import evalset as V
from kslab import taxonomy as T
from kslab.recipe import INTERNAL, Step, run_internal

TARGET_TYPES = frozenset(t.id for t in T.TYPES if t.scope == "target")
SIZES = (1, 2, 3, 4)
VARIANTS = 3
IDENTITY_SHARE = 0.2
INTENSITIES = (0.3, 0.5, 0.8)
DEV_FRACTION = 0.1
SPLIT_SALT = "p04-train-v1"


@dataclass(frozen=True)
class TrainPair:
    id: str
    source: str
    split: str
    size: int
    recipe: list[Step]
    applied_ops: list[str]
    clean: str
    obfuscated: str

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "split": self.split, "size": self.size,
                "recipe": [s.to_dict() for s in self.recipe], "applied_ops": self.applied_ops,
                "types": sorted({T.type_of(op).id for op in self.applied_ops}),
                "clean": self.clean, "obfuscated": self.obfuscated}


def variant_steps(text: str, size: int, seed: int) -> list[Step]:
    """대상 유형에서 문장에 걸리는 기법 `size`개를 고르고 강도를 다시 뽑는다."""
    steps, _ = V.compose_for(text, size, "auto", seed, types=TARGET_TYPES)
    rng = random.Random(V.stable_seed(seed, "intensity"))
    return [Step(INTERNAL, s.op, V.SPARSE_INTENSITY if s.op in V.SPARSE else rng.choice(INTENSITIES), s.seed)
            for s in steps]


def make_pairs(source: str, cluster: str, text: str, split: str, gen_seed: int) -> list[TrainPair]:
    rng = random.Random(V.stable_seed(gen_seed, source))
    out = []
    for k in range(VARIANTS):
        size = rng.choice(SIZES)
        seed = V.stable_seed(gen_seed, source, k)
        steps = variant_steps(text, size, seed)
        result = run_internal(text, steps)
        if result.text == text:
            continue
        out.append(TrainPair(f"{source}/v{k}", cluster, split, size, steps,
                             [steps[i].op for i in result.applied], text, result.text))
    # 변형 대비 IDENTITY_SHARE가 되도록(변형 3개면 0.75개) 확률로 넣는다
    if rng.random() < VARIANTS * IDENTITY_SHARE / (1 - IDENTITY_SHARE):
        out.append(TrainPair(f"{source}/id", cluster, split, 0, [], [], text, text))
    return out
