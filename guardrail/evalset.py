"""kslab 평가셋 v1(난독화 24,444문장) 읽기와 유형별 집계.

평가셋은 공격 원문을 담고 있어 `data/evalset/`(커밋 제외)에 둔다. 전달 묶음의 파일
(`records.jsonl`, `types.csv`, `SHA256SUMS.txt` 등)을 그대로 복사하면 된다.

집계 규칙은 전달 문서를 따른다.
- `types.csv`의 16유형 + 대조군으로 센다. 한 문장에 유형이 여럿이면 유형마다 한 번씩 센다.
- 원문(난독화 없음)은 `source_id`별 `clean`을 중복 없이 뽑아 id `<source_id>/clean`으로 쓴다.
- 통계 단위는 원문(`source_id`)이다. 한 원문에서 나온 문장 수십 개는 독립이 아니다.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from kslab import metrics as M

AXES = ("single", "compose", "external", "control")


def verify(directory: Path) -> dict[str, str]:
    """SHA256SUMS.txt와 대조한다. 어긋나면 멈춘다."""
    sums = {}
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*")] = digest
    for name, digest in sums.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"{name} 해시가 SHA256SUMS.txt와 다르다: {actual}")
    return sums


def load_records(directory: Path) -> list[dict[str, Any]]:
    with (directory / "records.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_types(directory: Path) -> list[str]:
    with (directory / "types.csv").open(encoding="utf-8-sig") as handle:
        return [row["id"] for row in csv.DictReader(handle)]


def clean_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for r in records:
        if r["source_id"] not in seen:
            seen[r["source_id"]] = {
                "id": f"{r['source_id']}/clean", "source_id": r["source_id"], "label": r["label"],
                "axis": "clean", "name": "clean", "types": [], "n_applied": 0,
                "obfuscated": r["clean"],
            }
    return sorted(seen.values(), key=lambda r: r["id"])


def _rate(observations: list[tuple[str, float]], samples: int) -> dict[str, Any] | None:
    if not observations:
        return None
    s = M.summarize_observations(observations, bootstrap_samples=samples)
    return {"n": s["rows"], "sources": s["clusters"], "rate": s["micro_estimate"],
            "rate_source_balanced": s["cluster_balanced_estimate"],
            "ci95": [s["ci95_low"], s["ci95_high"]]}


def _group_keys(r: dict[str, Any]) -> list[str]:
    if r["axis"] == "clean":
        return ["clean"]
    keys = [f"type:{t}" for t in r["types"]] + [f"axis:{r['axis']}", "all"]
    if r["axis"] == "single":
        keys.append(f"op:{r['name']}")
    if r["axis"] in ("compose", "external"):
        keys.append(f"n_applied:{r['n_applied']}")
    return keys


def aggregate(rows: list[dict[str, Any]], flags: dict[str, bool | None],
              bootstrap_samples: int = 2000) -> dict[str, Any]:
    """문장별 flagged를 그룹별 지표로 묶는다.

    evasion            공격 중 통과(flagged=false) 비율
    evasion_if_caught  원문이 차단된 공격만 놓고 본 통과 비율. 난독화 자체의 효과
    fpr                정상 중 차단 비율
    flagged가 None(판정을 읽지 못함)인 행은 지표에서 빼고 unparsed로 센다.
    """
    caught = {r["source_id"] for r in rows
              if r["axis"] == "clean" and r["label"] == "attack" and flags.get(r["id"])}
    evasion: dict[str, list] = collections.defaultdict(list)
    evasion_c: dict[str, list] = collections.defaultdict(list)
    fpr: dict[str, list] = collections.defaultdict(list)
    unparsed = collections.Counter()
    for r in rows:
        flag = flags.get(r["id"])
        for key in _group_keys(r):
            if flag is None:
                unparsed[key] += 1
                continue
            if r["label"] == "attack":
                evasion[key].append((r["source_id"], float(not flag)))
                if r["source_id"] in caught:
                    evasion_c[key].append((r["source_id"], float(not flag)))
            else:
                fpr[key].append((r["source_id"], float(flag)))
    groups = {}
    for key in sorted(set(evasion) | set(fpr) | set(unparsed)):
        groups[key] = {
            "evasion": _rate(evasion[key], bootstrap_samples),
            "evasion_if_caught": _rate(evasion_c[key], bootstrap_samples),
            "fpr": _rate(fpr[key], bootstrap_samples),
            "unparsed": unparsed[key],
        }
    clean = groups.get("clean", {})
    return {
        "clean_tpr": None if not clean.get("evasion") else 1 - clean["evasion"]["rate"],
        "clean_attacks_caught": len(caught),
        "groups": groups,
    }


def headline(summary: dict[str, Any]) -> dict[str, float]:
    """대표 지표. 표와 같게 원문 균형 추정치를 쓴다(원문은 원문당 1행이라 두 값이 같다)."""
    g = summary["groups"]

    def pick(key: str, metric: str) -> float | None:
        v = g.get(key, {}).get(metric)
        return None if v is None else v["rate_source_balanced"]

    values = {
        "clean_tpr": summary["clean_tpr"],
        "clean_fpr": pick("clean", "fpr"),
        "evasion_all": pick("all", "evasion"),
        "evasion_if_caught_all": pick("all", "evasion_if_caught"),
        "fpr_all": pick("all", "fpr"),
    }
    return {k: v for k, v in values.items() if v is not None}
