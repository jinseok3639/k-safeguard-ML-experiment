"""정규화기(복원기) 평가 지표 — 문장마다 계산한 뒤 평균한다.

글자 오류율(CER) 개선을 %p로 평균하면 입력이 얼마나 망가졌는지에 따라 값이 흔들리고, 많이 망가진 문장이 평균을
좌우한다. 여기 지표는 문장 하나하나를 같은 무게로 본다.

- 결과 분포(`outcome`): 완전 복원 / 나아짐 / 같음 / 나빠짐.
- 복원율(`restore_rate`): (복원 전 CER − 복원 후 CER) ÷ 복원 전 CER. 하한은 두지 않는다 — 망가뜨리면 음수다.
- 글자 단위 F1(`edit_score`): 정답 편집(난독화 → 원문)과 복원기 편집(난독화 → 복원 결과)을 같은 정렬
  (`edit_tags.oracle_tags`)로 구해 난독화 문장의 글자마다 비교한다. 모델 내부를 보지 않으므로 어떤 복원기에도 쓴다.
- 유형별 재현율(`type_recall`): 고쳐야 할 글자마다 평가셋의 글자 위치 기록(`edits`)으로 유형을 붙인다.
- 정상 문장 오변경(`benign_report`): 바뀐 문장 비율과 함께 서로 다른 오류 단어 수, 95% 신뢰구간을 낸다.

모델·러너와 상관없이 (원문, 난독화 문장, 복원 결과)만 받는다.
"""

from __future__ import annotations

import difflib
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from kslab import edit_tags as E
from kslab import taxonomy as T
from kslab.metrics import char_error_rate

OUTCOMES = ("exact", "better", "same", "worse")
# 복원 전 CER 구간. 이름은 MLflow 키에 그대로 쓴다.
CER_BINS = ((0.0, 0.1, "cer0_10"), (0.1, 0.3, "cer10_30"), (0.3, math.inf, "cer30_"))
UNKNOWN = "unknown"


def outcome(clean: str, obf: str, restored: str) -> str:
    if restored == clean:
        return "exact"
    before, after = char_error_rate(clean, obf), char_error_rate(clean, restored)
    if after < before:
        return "better"
    return "worse" if after > before else "same"


def restore_rate(clean: str, obf: str, restored: str) -> float | None:
    """망가진 정도 중 되돌린 비율. 망가지지 않은 문장(복원 전 CER 0)은 None."""
    before = char_error_rate(clean, obf)
    if before == 0:
        return None
    return (before - char_error_rate(clean, restored)) / before


def cer_bin(clean: str, obf: str) -> str:
    before = char_error_rate(clean, obf)
    return next(name for low, high, name in CER_BINS if low <= before < high)


@dataclass
class EditScore:
    """문장 하나의 글자 단위 편집 비교. 단위는 글자 자리의 편집과 그 앞 공백 넣기 두 가지다."""

    needed: int = 0          # 정답에서 고쳐야 하는 단위
    made: int = 0            # 복원기가 바꾼 단위
    correct: int = 0         # 고쳐야 했고, 정답과 똑같이 고친 단위
    collateral: int = 0      # 고칠 필요가 없던 글자를 바꾼 수
    untouched: int = 0       # 고칠 필요가 없던 글자 수

    @property
    def recall(self) -> float | None:
        return self.correct / self.needed if self.needed else None

    @property
    def precision(self) -> float | None:
        return self.correct / self.made if self.made else None

    @property
    def f1(self) -> float | None:
        """고칠 것이 없는 문장은 None. 아무것도 안 바꾸면 0."""
        if not self.needed:
            return None
        if not self.correct:
            return 0.0
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r)


def _same(gold: E.Tag, pred: E.Tag) -> bool:
    return gold.op == pred.op and (gold.op != E.SUB or gold.char == pred.char)


def edit_score(clean: str, obf: str, restored: str) -> EditScore:
    gold, _ = E.oracle_tags(obf, clean)
    pred, _ = E.oracle_tags(obf, restored)
    score = EditScore()
    for g, p in zip(gold, pred):
        need, made = g.op != E.KEEP, p.op != E.KEEP
        score.needed += need
        score.made += made
        score.correct += need and made and _same(g, p)
        if not need:
            score.untouched += 1
            score.collateral += made
        score.needed += g.space_before
        score.made += p.space_before
        score.correct += g.space_before and p.space_before
    return score


def edit_type(record: dict, op: str) -> str:
    """글자 위치 기록(`edits`)의 op가 속한 유형 id. 모르는 op는 `unknown`."""
    from kslab.external_gen import TYPE_OF

    step = record["recipe"][0]
    try:
        if step["generator"] == "kotox":
            return TYPE_OF[("kotox", f"rule:{op}")]
        if step["generator"] != "kslab":
            return TYPE_OF[(step["generator"], step["op"])]
        return T.type_of(op).id
    except KeyError:
        return UNKNOWN


def type_recall(record: dict, restored: str) -> dict[str, tuple[int, int]] | None:
    """유형별 (고쳐야 할 단위, 맞게 고친 단위). 정답 정렬에 남는 오류가 있으면 위치를 맞출 수 없어 None.

    정답 태그를 따라가며 난독화 글자마다 원문 위치를 붙이고, 그 위치를 덮는 `edits` 구간의 유형을 준다.
    지운 글자(끼워 넣은 기호 등)는 원문의 끼워 넣은 자리 p에 붙는다. 한 단위를 여러 기법이 건드렸으면 모두에 센다.
    """
    clean, obf = record["clean"], record["obfuscated"]
    gold, residual = E.oracle_tags(obf, clean)
    if residual:
        return None
    pred, _ = E.oracle_tags(obf, restored)
    spans = [(e["orig_span"][0], e["orig_span"][1], edit_type(record, e["op"])) for e in record.get("edits") or []]

    def covering(pos: int, inserted: bool) -> set[str]:
        if inserted:
            # 끼워 넣은 글자: 그 자리에 0폭으로 기록된 삽입이 있으면 그것만, 없으면 자리에 닿는 구간(늘인 음절 등).
            found = {t for a, b, t in spans if a == b == pos} or {t for a, b, t in spans if a <= pos <= b}
        else:
            found = {t for a, b, t in spans if a <= pos < b}
        return found or {UNKNOWN}

    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    j = 0
    for g, p in zip(gold, pred):
        if g.space_before:
            for t in covering(j, inserted=False):
                out[t][0] += 1
                out[t][1] += p.space_before
            j += 1
        if g.op == E.DELETE:
            for t in covering(j, inserted=True):
                out[t][0] += 1
                out[t][1] += _same(g, p)
            continue
        if g.op == E.SUB:
            for t in covering(j, inserted=False):
                out[t][0] += 1
                out[t][1] += _same(g, p)
        j += 1
    return {t: (n, c) for t, (n, c) in out.items()}


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


class Board:
    """문장들을 모아 지표를 낸다. `group`마다(예: 기법 수, CER 구간) 같은 지표를 따로 낸다."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.types: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.type_skipped = 0

    def add(self, clean: str, obf: str, restored: str, groups: dict[str, str] | None = None,
            record: dict | None = None) -> None:
        score = edit_score(clean, obf, restored)
        self.rows.append({
            "outcome": outcome(clean, obf, restored), "restore_rate": restore_rate(clean, obf, restored),
            "f1": score.f1, "precision": score.precision, "recall": score.recall,
            "collateral": score.collateral, "untouched": score.untouched, "groups": groups or {},
        })
        if record is not None:
            found = type_recall(record, restored)
            if found is None:
                self.type_skipped += 1
            else:
                for t, (n, c) in found.items():
                    self.types[t][0] += n
                    self.types[t][1] += c

    @staticmethod
    def _summary(rows: list[dict]) -> dict:
        n = len(rows)
        out: dict = {"n": n}
        for name in OUTCOMES:
            out[name] = sum(r["outcome"] == name for r in rows) / n if n else None
        for key in ("restore_rate", "f1", "precision", "recall"):
            out[key] = _mean(r[key] for r in rows)
        untouched = sum(r["untouched"] for r in rows)
        out["collateral"] = sum(r["collateral"] for r in rows) / untouched if untouched else None
        return out

    def summary(self) -> dict:
        out = self._summary(self.rows)
        by: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            for group, value in row["groups"].items():
                by[group][value].append(row)
        out["by"] = {g: {v: self._summary(rows) for v, rows in sorted(d.items())} for g, d in by.items()}
        if self.types or self.type_skipped:
            # 유형별 재현율은 단위를 모두 모아 센다(유형 하나가 문장에 몇 글자 안 걸리는 경우가 많다).
            out["type_recall"] = {t: {"needed": n, "correct": c, "recall": c / n if n else None}
                                  for t, (n, c) in sorted(self.types.items())}
            out["type_skipped"] = self.type_skipped
        return out


def wrong_chars(text: str, clean: str) -> list[bool]:
    """`text`의 글자마다 원문과 어긋났는지. 편집거리 정렬로 맞춘다.

    바뀐 글자와 더 들어간 글자는 그 글자에, 빠진 글자는 빠진 자리 바로 뒤 글자(끝이면 앞 글자)에 표시한다.
    더 들어간 공백은 양옆 글자에도 표시한다 — 토큰은 공백을 덮지 않으므로 그래야 토큰 단위로 옮겨진다.
    """
    n, m = len(text), len(clean)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][m] = n - i
    for j in range(m + 1):
        cost[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            cost[i][j] = min(cost[i + 1][j + 1] + (text[i] != clean[j]), cost[i + 1][j] + 1, cost[i][j + 1] + 1)
    wrong = [False] * n

    def mark(i: int) -> None:
        if n:
            wrong[min(max(i, 0), n - 1)] = True

    i = j = 0
    while i < n or j < m:
        if i < n and j < m and cost[i][j] == cost[i + 1][j + 1] + (text[i] != clean[j]):
            if text[i] != clean[j]:
                mark(i)
            i, j = i + 1, j + 1
        elif i < n and cost[i][j] == cost[i + 1][j] + 1:
            mark(i)
            if text[i] == " ":
                mark(i - 1)
                mark(i + 1)
            i += 1
        else:
            mark(i if i < n else n - 1)
            j += 1
    return wrong


def lenient_key(text: str) -> list[str]:
    """문장부호를 지우고 공백으로 나눈 어절. 이것이 같으면 오변경으로 세지 않는다."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P")).split()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율 k/n의 Wilson 신뢰구간."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, center - half), min(1.0, center + half))


def changed_words(clean: str, restored: str) -> list[tuple[str, str]]:
    """바뀐 어절 쌍 (원문, 복원 결과). 문장부호·공백만 다른 것은 빼고 본다."""
    a, b = lenient_key(clean), lenient_key(restored)
    pairs = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            pairs.append((" ".join(a[i1:i2]), " ".join(b[j1:j2])))
    return pairs


def benign_report(cleans: list[str], outputs: list[str]) -> dict:
    """정상 문장에 복원기를 돌린 결과. 문장 비율과 함께 서로 다른 오류 단어 수를 낸다.

    같은 단어가 여러 문장에 나오면(템플릿으로 만든 문장 등) 단어 하나의 오류가 문장 여러 개로 번진다.
    """
    n = len(cleans)
    lenient = [lenient_key(o) != lenient_key(c) for c, o in zip(cleans, outputs)]
    strict = sum(o != c for c, o in zip(cleans, outputs))
    pairs = sorted({p for c, o, bad in zip(cleans, outputs, lenient) if bad for p in changed_words(c, o)})
    k = sum(lenient)
    low, high = wilson(k, n)
    return {"n": n, "mutated": k, "mutation": k / n if n else 0.0, "mutation_strict": strict / n if n else 0.0,
            "ci_low": low, "ci_high": high, "error_words": len({w for w, _ in pairs}),
            "word_pairs": [list(p) for p in pairs]}
