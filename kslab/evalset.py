"""평가셋 v1 정의 — 원문, 축, 복합 기법 고르기. 평가 전용이며 학습에 쓰지 않는다.

원문은 505 시드와 공개 정상 문장 190(`xxxjjhhh/korean_guardrail_test`, CC BY 4.0)이다. 공개 문장 거르기
규칙은 P01 리뷰(2026-10-27)에서 정했다.
"""

from __future__ import annotations

import json
import random
import re
import zlib
from dataclasses import dataclass
from pathlib import Path

from kslab import pairs as P
from kslab import taxonomy as T
from kslab.obfuscate import STANDARD_PRONUNCIATION_CLAUSE, TRANSFORMS
from kslab.compose import CANONICAL_ORDER
from kslab.recipe import INTERNAL, Step, run_internal

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_BENIGN = ROOT / "data" / "hf-survey" / "xxxjjhhh__korean_guardrail_test" / "main.jsonl"
# 인젝션·유해로 볼 문장(폭탄 제조, "이 시스템 정보를 출력해" 류)과 자판 난타
PUBLIC_DROP = frozenset({7, 50, 73, 79, 81, 84, 87, 89, 91, 94, 98})
PUBLIC_TEMPLATE_IDS = range(152, 349)   # 템플릿으로 찍은 블록("역사에 대한 튜토리얼…")

SINGLE_INTENSITY = 0.5
COMPOSE_INTENSITY = 0.3
# 발음 규칙 환경에서만 걸리는 기법. 문장당 후보 자리가 한두 곳이라 0.5·0.3이면 반올림으로 0곳이 되기 쉽다
# (`obfuscate._choose`, round(0.5) = 0). 이 기법들은 1.0으로 건다(P01.4b).
SPARSE = frozenset(STANDARD_PRONUNCIATION_CLAUSE) | {"aspirate"}
SPARSE_INTENSITY = 1.0
COMPOSE_ROUNDS = 6
COMPOSE_SIZES = (2, 3, 4)
ATTACKERS = ("human", "auto")
SAMPLES = 2
EXTERNAL_OPS = (("kotox", "compose:2"), ("kotox", "compose:3"), ("kotox", "compose:4"),
                ("kotka", "yamin"), ("kotka", "squaredog"), ("kotka", "shred:0.3"))
CONTROL_OPS = tuple(("jamo_typo", f"T{i}:2") for i in range(1, 6))

# 복합에서 거는 순서: 음운(길이 보존) -> 첨가 -> 모양 -> 기호 -> 띄어쓰기. 뒤 단계가 앞 단계를 지우지 않게 한다.
COMPOSE_ORDER = CANONICAL_ORDER + (
    "semivowel_insert", "initial_insert", "jamo_copy", "syllable_extend",
    "lookalike_hangul", "lookalike_foreign", "jamo_decompose", "symbol_insert", "zwsp_inject", "break_spacing",
)
COMPOSE_EXCLUDED = frozenset({"chosung"})   # 다른 기법의 흔적을 지운다


@dataclass(frozen=True)
class Source:
    source_id: str
    text: str
    label: str        # attack / benign
    origin: str       # seed / public_benign


def load_public_benign() -> list[Source]:
    rows = [json.loads(line) for line in PUBLIC_BENIGN.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, out = set(), []
    for row in rows:
        if row["is_injection"] or row["id"] in PUBLIC_DROP or row["id"] in PUBLIC_TEMPLATE_IDS:
            continue
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        out.append(Source(f"kgt-{row['id']}", row["prompt"], "benign", "public_benign"))
    return out


def load_sources() -> list[Source]:
    seeds = [Source(s.seed_id, s.text, s.label, "seed") for s in P.load_seeds()]
    return seeds + load_public_benign()


def stable_seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)


def _konte_count(konte: str) -> int:
    return sum(int(n) for n in re.findall(r"\d+", konte))


def type_weights(attacker: str) -> dict[str, float]:
    """복합에 쓸 유형과 가중치. 사람형은 KoNTe 건수(여러 유형이 한 건수를 나눠 가지면 똑같이 나눔), 자동형은 균등."""
    types = [t for t in T.TYPES if any(x in COMPOSE_ORDER and x not in COMPOSE_EXCLUDED for x in t.techniques)]
    if attacker == "auto":
        return {t.id: 1.0 for t in types}
    sharing: dict[str, int] = {}
    for t in types:
        sharing[t.konte] = sharing.get(t.konte, 0) + 1
    return {t.id: _konte_count(t.konte) / sharing[t.konte] for t in types}


def _draw(pool: dict[str, float], rng: random.Random) -> str:
    """유형 하나를 가중치대로 뽑고 그 유형의 기법 하나를 고른다. 뽑힌 유형은 pool에서 뺀다."""
    ids, w = zip(*pool.items())
    type_id = rng.choices(ids, weights=w)[0]
    del pool[type_id]
    options = sorted(x for x in T.BY_ID[type_id].techniques if x in COMPOSE_ORDER and x not in COMPOSE_EXCLUDED)
    return rng.choice(options)


def _chain(techniques: list[str], seed: int, single: bool = False) -> list[Step]:
    ordered = sorted(techniques, key=COMPOSE_ORDER.index) if not single else techniques
    base = SINGLE_INTENSITY if single else COMPOSE_INTENSITY
    return [Step(INTERNAL, name, SPARSE_INTENSITY if name in SPARSE else base, stable_seed(seed, name))
            for name in ordered]


def compose_steps(size: int, attacker: str, seed: int) -> list[Step]:
    """유형이 겹치지 않게 기법 `size`개를 골라 고정 순서로 늘어놓는다(문장을 보지 않는다)."""
    rng = random.Random(seed)
    pool = dict(type_weights(attacker))
    return _chain([_draw(pool, rng) for _ in range(size)], seed)


def compose_for(text: str, size: int, attacker: str, seed: int) -> tuple[list[Step], list[str]]:
    """문장에 실제로 걸리는 기법 `size`개를 채운다.

    고른 기법이 문장을 안 바꾸면 그 기법을 빼고 쓰지 않은 유형에서 다시 뽑는다(KOTOX도 실패한 규칙을 다른 규칙으로
    바꿔 다시 시도한다). 최종 레시피와, 시도했다가 뺀 기법 목록을 돌려준다. 유형이 바닥나면 채운 만큼만 쓴다.
    """
    rng = random.Random(seed)
    pool = dict(type_weights(attacker))
    chosen = [_draw(pool, rng) for _ in range(size)]
    dropped: list[str] = []
    for _ in range(COMPOSE_ROUNDS):
        steps = _chain(chosen, seed)
        idle = [steps[i].op for i in range(len(steps)) if i not in run_internal(text, steps).applied]
        if not idle:
            break
        dropped += idle
        chosen = [x for x in chosen if x not in idle]
        while len(chosen) < size and pool:
            chosen.append(_draw(pool, rng))
    return _chain(chosen, seed), dropped


def single_steps(technique: str, seed: int) -> list[Step]:
    return _chain([technique], seed, single=True)


def internal_techniques() -> list[str]:
    return sorted(TRANSFORMS)
