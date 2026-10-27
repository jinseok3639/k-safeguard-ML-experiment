"""학습·평가 페어 구성과 분할.

두 가지 원칙이 코드로 강제된다.

1. **505 시드는 평가 전용.** 학습 페어를 만드는 함수는 시드를 받지 않는다.
2. **분할은 source-disjoint.** 505 시드 중 256개가 `template_v1` 한 출처라, seed_id 랜덤
   분할은 같은 템플릿의 형제 문장을 train/test 양쪽에 흘려보낸다. 이번 세션 실측으로
   랜덤 분할이 source 분할보다 중성 슬롯 정확도를 15%p 부풀렸다.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import obfuscate
from .jamo import is_syllable

SEED_DIR = Path(__file__).resolve().parents[1] / "hf_repo" / "seeds"


@dataclass(frozen=True)
class Seed:
    """505 dev 시드 한 건."""

    seed_id: str
    text: str
    label: str
    category: str
    source: str


@dataclass(frozen=True)
class Pair:
    """(난독화, 원문) 학습·평가 페어. 길이 보존 변환에서만 만든다."""

    cluster_id: str      # 통계적 독립 단위 (시드면 seed_id, 코퍼스면 문서 id)
    source: str          # 분할 기준
    technique: str
    intensity: float
    obfuscated: str
    clean: str
    # 조합 난독화에서만 채워진다. 문자마다 마지막으로 그 자리를 바꾼 단계 이름("" = 무변경).
    # 단일 난독화 페어는 비워 둔다 — technique 하나가 곧 출처이므로 정보가 중복이다.
    provenance: tuple[str, ...] = ()
    # 이 페어를 만든 프로파일 이름(조합일 때만). 보고서에서 조건을 식별하는 데 쓴다.
    profile: str = ""

    @property
    def is_identity(self) -> bool:
        return self.obfuscated == self.clean

    @property
    def depth(self) -> int:
        """겹쳐 건 변환의 수. 단일 난독화는 1, identity는 0."""
        if self.is_identity:
            return 0
        return len(self.technique.split("+"))


def load_seeds(seed_dir: Path | None = None) -> list[Seed]:
    """505개 dev 시드를 읽는다. 평가 전용 — 학습에 쓰지 말 것."""
    directory = Path(seed_dir) if seed_dir else SEED_DIR
    seeds: list[Seed] = []
    for name in ("attacks.jsonl", "benign.jsonl"):
        path = directory / name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                seeds.append(
                    Seed(
                        seed_id=row["id"],
                        text=row["text"],
                        label=row["label"],
                        category=row["category"],
                        source=row["source"],
                    )
                )
    if not seeds:
        raise ValueError(f"시드를 하나도 읽지 못했습니다: {directory}")
    return seeds


def clean_sentence(text: str, min_len: int = 10, max_len: int = 120, min_hangul: float = 0.5) -> str | None:
    """코퍼스 문장 정제. 조건을 못 맞추면 None.

    한글 음절 비율이 낮은 문장(수식·표·외국어)은 자모 슬롯 학습에 신호를 주지 않으므로 뺀다.
    """
    text = unicodedata.normalize("NFC", text).strip()
    text = " ".join(text.split())
    if not (min_len <= len(text) <= max_len):
        return None
    syllables = sum(1 for ch in text if is_syllable(ch))
    if syllables / len(text) < min_hangul:
        return None
    return text


def make_pairs(
    items: Sequence[tuple[str, str, str]],
    techniques: Iterable[str],
    intensities: Iterable[float] = (1.0,),
    gen_seed: int = 1234,
    include_identity: bool = True,
) -> list[Pair]:
    """(cluster_id, source, text) 목록 -> 길이 보존 난독화 페어.

    `include_identity=True`면 (clean, clean) 페어를 함께 넣는다. 정상 입력을 건드리지
    않도록 가르치는 신호이고, clean mutation rate에 직결된다 — k-safeguard `f6ee374`의
    `experiments/참조용/README.md`가 3위 솔루션에서 짚어둔 아이디어다.
    """
    techniques = list(techniques)
    unknown = set(techniques) - set(obfuscate.TRANSFORMS)
    if unknown:
        raise ValueError(f"알 수 없는 technique: {sorted(unknown)}")
    lossy = set(techniques) - obfuscate.LENGTH_PRESERVING
    if lossy:
        raise ValueError(
            f"길이 보존이 아닌 technique은 슬롯 분류에 쓸 수 없습니다: {sorted(lossy)}"
        )

    pairs: list[Pair] = []
    for cluster_id, source, text in items:
        for technique in techniques:
            if include_identity:
                # identity 페어는 **짝이 되는 기법의 후보 규칙**을 그대로 달고 나간다.
                # 공통 "identity" 기법으로 묶으면 후보 위치가 '모든 음절'이 되어버려,
                # 추론 때 실제로 보게 될 분포(예: 된소리 초성 위치만)와 어긋난다.
                # 이 페어들이 주는 신호가 정확히 "정상 단어의 그 자리는 건드리지 마라"이므로
                # 분포가 어긋나면 clean mutation 학습이 헛돈다.
                pairs.append(Pair(cluster_id, source, technique, 0.0, text, text))
            fn = obfuscate.TRANSFORMS[technique]
            for intensity in intensities:
                obf = fn(text, intensity, seed=gen_seed)
                if len(obf) != len(text):
                    raise ValueError(
                        f"{technique}이 길이를 바꿨습니다 (len {len(text)} -> {len(obf)}): {text!r}"
                    )
                pairs.append(Pair(cluster_id, source, technique, intensity, obf, text))
    return pairs


def make_composed_pairs(
    items: Sequence[tuple[str, str, str]],
    profile,
    intensity: float = 1.0,
    gen_seed: int = 1234,
    include_identity: bool = True,
) -> list[Pair]:
    """조합(다중) 난독화 페어. `compose.Profile` 하나를 모든 아이템에 건다.

    `make_pairs`를 건드리지 않고 형제 함수로 둔 이유: `make_pairs`는 technique 이름을
    `obfuscate.TRANSFORMS`에 대조해 검증하는데 조합 이름은 거기 없다. 그 검증을 느슨하게
    하면 E0~E8의 모든 기준선이 같은 함수를 통과하는 의미가 사라진다.

    페어의 `technique`은 구성 기법의 **합집합 이름**(`liaison+tensify`)이다. 추론 시점에
    어느 단계가 어느 자리를 건드렸는지 알 수 없으므로, 모델이 다뤄야 할 슬롯과 후보 위치는
    합집합이어야 정직하다. 단계별 귀속은 `provenance`에 따로 남긴다.
    """
    from . import compose as _compose

    _compose.validate_stages(profile.stages)
    technique = profile.technique()
    pairs: list[Pair] = []
    for cluster_id, source, text in items:
        if include_identity:
            # identity 페어는 조합 이름을 그대로 달아야 후보 위치 규칙이 추론 때와 같아진다.
            # 프로파일마다 하나씩만 낸다 — 프로파일 수만큼 곱해지면 "아무것도 건드리지 마라"
            # 신호가 과대표집되어 유용성이 무너진다.
            pairs.append(
                Pair(cluster_id, source, technique, 0.0, text, text, (), profile.name)
            )
        result = _compose.apply_stages(text, profile.stages, intensity, gen_seed)
        if len(result.text) != len(text):
            raise ValueError(f"{profile.name}이 길이를 바꿨습니다: {text!r}")
        pairs.append(
            Pair(
                cluster_id, source, technique, intensity,
                result.text, text, result.provenance, profile.name,
            )
        )
    return pairs


def split_by_source(
    items: Sequence[tuple[str, str, str]], holdout_sources: Iterable[str]
) -> tuple[list, list]:
    """source 기준 disjoint 분할. holdout에 지정한 출처 전체가 평가 쪽으로 간다."""
    holdout = set(holdout_sources)
    known = {source for _, source, _ in items}
    missing = holdout - known
    if missing:
        raise ValueError(f"존재하지 않는 source를 holdout으로 지정했습니다: {sorted(missing)}")
    train = [it for it in items if it[1] not in holdout]
    test = [it for it in items if it[1] in holdout]
    if not train or not test:
        raise ValueError("분할 결과 한쪽이 비었습니다.")
    return train, test


def split_random(
    items: Sequence[tuple[str, str, str]], test_size: int, random_seed: int = 0
) -> tuple[list, list]:
    """cluster_id 랜덤 분할. **정직하지 않은 대조군** — source 분할과 대비할 때만 쓴다."""
    import random as _random

    shuffled = list(items)
    _random.Random(random_seed).shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


def assert_disjoint(train: Sequence[tuple[str, str, str]], test: Sequence[tuple[str, str, str]]) -> None:
    """분할 누출 검사. cluster_id도 문장 자체도 겹치면 안 된다."""
    train_ids = {c for c, _, _ in train}
    test_ids = {c for c, _, _ in test}
    overlap_ids = train_ids & test_ids
    if overlap_ids:
        raise AssertionError(f"cluster_id가 train/test에 모두 있습니다: {sorted(overlap_ids)[:5]}")
    train_texts = {t for _, _, t in train}
    overlap_texts = train_texts & {t for _, _, t in test}
    if overlap_texts:
        raise AssertionError(f"동일 문장이 train/test에 모두 있습니다: {sorted(overlap_texts)[:3]}")


def seeds_as_items(seeds: Sequence[Seed]) -> list[tuple[str, str, str]]:
    """Seed 목록 -> (cluster_id, source, text) 형태."""
    return [(s.seed_id, s.source, s.text) for s in seeds]
