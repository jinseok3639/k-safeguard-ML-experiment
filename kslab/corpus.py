"""학습용 clean 한국어 문장 풀 — 한국어 위키 + KLUE.

둘 다 CC BY-SA 4.0이다. 스트리밍으로 받아 **필터링된 문장만** jsonl로 남기고 원본 shard는
보관하지 않는다 — C: 여유가 21GB뿐이라 원본을 받아두면 금방 찬다.

두 출처를 섞는 이유는 레지스터 때문이다. 평가에 쓰는 505 시드는 대화형 명령문인데 위키는
백과사전체라, 위키만으로 학습하면 어휘 폭은 늘어도 정작 평가 분포와 어긋난다. KLUE의
뉴스 제목(ynat)과 구어 문장(nli)이 그 간극을 메운다. 혼합 비율 자체가 E1의 스윕 대상이다.

문서 단위로 `source`를 달아두는 게 중요하다. 같은 문서의 문장들은 서로 강하게 닮아 있어서,
문장 단위로 무작위 분할하면 E0에서 실측한 것과 같은 종류의 누출이 생긴다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .pairs import clean_sentence

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CORPUS_DIR = DATA_DIR / "corpus"

# 착수 시 실제 조회로 확인한 id/config (2026-08-23).
WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ko"
KLUE_DATASET = "klue/klue"
KLUE_CONFIGS = ("ynat", "nli")

LICENSE_NOTE = "wikimedia/wikipedia(20231101.ko) CC BY-SA 4.0 · klue/klue CC BY-SA 4.0"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
# 위키 문서에 남는 표·틀·각주 잔재. 이런 줄은 문장으로 치지 않는다.
_MARKUP = re.compile(r"^\s*(\||\{\{|\}\}|==|\*|#|:)")


@dataclass(frozen=True)
class CorpusRow:
    row_id: str
    source: str      # 문서 단위 — 분할의 기준
    origin: str      # 'wiki' | 'klue:ynat' | 'klue:nli'
    text: str

    def as_json(self) -> dict:
        return {"id": self.row_id, "source": self.source, "origin": self.origin, "text": self.text}


def _sentences(block: str) -> Iterator[str]:
    for piece in _SENTENCE_SPLIT.split(block):
        if not piece or _MARKUP.match(piece):
            continue
        cleaned = clean_sentence(piece)
        if cleaned:
            yield cleaned


def _stream_wiki(limit: int, max_per_doc: int = 6) -> Iterator[CorpusRow]:
    """위키 문서를 문장으로 쪼갠다. 한 문서에서 너무 많이 가져오면 다양성이 떨어진다."""
    from datasets import load_dataset

    stream = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)
    emitted = 0
    for doc in stream:
        if emitted >= limit:
            return
        doc_id = f"wiki:{doc['id']}"
        taken = 0
        for sentence in _sentences(doc["text"]):
            if taken >= max_per_doc or emitted >= limit:
                break
            yield CorpusRow(f"{doc_id}#{taken}", doc_id, "wiki", sentence)
            taken += 1
            emitted += 1


def _klue_rows(config: str) -> Iterator[CorpusRow]:
    """KLUE config 하나를 문장 단위로 흘린다."""
    from datasets import load_dataset

    stream = load_dataset(KLUE_DATASET, config, split="train", streaming=True)
    fields = ("title",) if config == "ynat" else ("premise", "hypothesis")
    for index, row in enumerate(stream):
        guid = row.get("guid") or f"{config}-{index}"
        for field in fields:
            value = row.get(field)
            if not value:
                continue
            cleaned = clean_sentence(value)
            if not cleaned:
                continue
            yield CorpusRow(
                f"klue:{config}:{guid}:{field}",
                f"klue:{config}:{guid}",
                f"klue:{config}",
                cleaned,
            )


def _stream_klue(limit: int) -> Iterator[CorpusRow]:
    """KLUE ynat(뉴스 제목)과 nli(구어 문장)를 **번갈아** 흘린다.

    config를 순서대로 소진하면 앞선 config가 할당량을 다 먹어 뒤쪽이 하나도 안 들어온다.
    레지스터를 섞는 게 목적이므로 round-robin 이어야 한다.
    """
    streams = [iter(_klue_rows(config)) for config in KLUE_CONFIGS]
    emitted = 0
    while streams and emitted < limit:
        for stream in list(streams):
            if emitted >= limit:
                return
            try:
                yield next(stream)
            except StopIteration:
                streams.remove(stream)
                continue
            emitted += 1


def build_corpus(
    target: int,
    wiki_ratio: float = 0.5,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """문장 `target`개를 모아 jsonl로 캐시하고 경로를 돌려준다.

    같은 (target, wiki_ratio) 조합이면 이미 만든 파일을 그대로 쓴다 — E1이 여러 지점을
    돌 때 매번 다시 받지 않게 한다.
    """
    if not 0.0 <= wiki_ratio <= 1.0:
        raise ValueError("wiki_ratio는 0~1이어야 합니다.")
    directory = Path(cache_dir) if cache_dir else CORPUS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"corpus_n{target}_wiki{int(wiki_ratio * 100)}.jsonl"
    if path.exists() and not force:
        return path

    wiki_target = int(round(target * wiki_ratio))
    klue_target = target - wiki_target

    seen: set[str] = set()
    rows: list[CorpusRow] = []
    for stream, want in ((_stream_wiki(wiki_target * 3), wiki_target),
                         (_stream_klue(klue_target * 3), klue_target)):
        taken = 0
        for row in stream:
            if taken >= want:
                break
            if row.text in seen:
                continue
            seen.add(row.text)
            rows.append(row)
            taken += 1

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_json(), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def load_corpus(path: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(CorpusRow(row["id"], row["source"], row["origin"], row["text"]))
    return rows


def corpus_as_items(rows: list[CorpusRow]) -> list[tuple[str, str, str]]:
    """(cluster_id, source, text). cluster와 source를 모두 문서 id로 둔다."""
    return [(r.source, r.source, r.text) for r in rows]


def split_by_hash(
    items: list[tuple[str, str, str]], test_fraction: float = 0.2, salt: str = "kslab-v1"
) -> tuple[list, list]:
    """source 해시로 결정론적 disjoint 분할. 같은 문서는 항상 같은 쪽으로 간다."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction은 0과 1 사이여야 합니다.")
    threshold = int(test_fraction * (1 << 32))
    train, test = [], []
    for item in items:
        digest = hashlib.sha256(f"{salt}:{item[1]}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        (test if bucket < threshold else train).append(item)
    if not train or not test:
        raise ValueError("분할 결과 한쪽이 비었습니다.")
    return train, test
