"""글자 편집 태그 — P04 정규화기의 출력 표현.

난독화 문장의 글자마다 태그 하나를 붙인다.

- `KEEP`: 그대로 둔다.
- `DELETE`: 지운다(끼워 넣은 기호·자모, 늘인 음절, 넣은 공백).
- `SUB`: 완성형 한글 음절 하나로 바꾼다(음운 교체·받침 추가 등). 바꿀 글자는 `char`.

여기에 "이 글자 앞에 공백을 넣는다"는 플래그가 따로 붙는다(띄어쓰기 삭제 되돌리기). 두 글자를 하나로 합치는 것은
`DELETE` + `SUB`로 표현된다(ㅆㅡ → 쓰). 표현 못 하는 것은 공백이 아닌 글자를 새로 끼워 넣는 것과 한글 음절이 아닌
글자로 바꾸는 것이다. 1기 슬롯 모델은 길이가 같고 한글 음절끼리 바뀐 자리만 표현한다(`slot_residual`).

`oracle_tags`는 정답을 알 때 가장 적게 고치는 태그열을 정렬(DP)로 찾는다. P04.1 상한과 학습 레이블에 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

from kslab.jamo import is_syllable

KEEP, DELETE, SUB = "K", "D", "S"
_INF = float("inf")


@dataclass(frozen=True)
class Tag:
    op: str                     # KEEP / DELETE / SUB
    char: str = ""              # SUB일 때 바꿀 음절
    space_before: bool = False


def apply_tags(obf: str, tags: list[Tag]) -> str:
    if len(tags) != len(obf):
        raise ValueError("태그 수가 글자 수와 다릅니다.")
    out = []
    for ch, t in zip(obf, tags):
        if t.space_before:
            out.append(" ")
        if t.op == KEEP:
            out.append(ch)
        elif t.op == SUB:
            out.append(t.char)
    return "".join(out)


def oracle_tags(obf: str, clean: str) -> tuple[list[Tag], int]:
    """`clean`에 가장 가까운 결과를 내는 태그열과, 그래도 남는 오류 수.

    비용은 (남는 오류, 편집 수) 순으로 줄인다. 남는 오류는 태그로 만들 수 없는 원문 글자 수다 — 한글 음절이 아닌
    글자로 바꿔야 하거나, 공백이 아닌 글자를 끼워 넣어야 하는 경우. 오류가 0이면 `apply_tags(obf, tags) == clean`이다.
    """
    n, m = len(obf), len(clean)
    # cost[i][j]: obf[i:]로 clean[j:]를 만들 때의 (오류, 편집). 뒤에서부터 채운다.
    cost = [[(_INF, _INF)] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    cost[n][m] = (0, 0)
    for j in range(m - 1, -1, -1):           # obf를 다 쓴 뒤 남은 원문: 전부 오류
        cost[n][j] = (cost[n][j + 1][0] + 1, cost[n][j + 1][1])
        move[n][j] = "miss"
    for i in range(n - 1, -1, -1):
        for j in range(m, -1, -1):
            best, how = _add(cost[i + 1][j], 0, 1), "del"
            if j < m:
                c = clean[j]
                if obf[i] == c:
                    cand = cost[i + 1][j + 1]
                    if cand < best:
                        best, how = cand, "keep"
                elif is_syllable(c):
                    cand = _add(cost[i + 1][j + 1], 0, 1)
                    if cand < best:
                        best, how = cand, "sub"
                else:                         # 바꿀 수 없는 글자: obf 글자를 그대로 두고 오류 1
                    cand = _add(cost[i + 1][j + 1], 1, 0)
                    if cand < best:
                        best, how = cand, "wrong"
                if c == " " and (move[i][j + 1] in ("keep", "sub", "del", "wrong")):
                    cand = _add(cost[i][j + 1], 0, 1)
                    if cand < best:
                        best, how = cand, "space"
                cand = _add(cost[i][j + 1], 1, 0)  # 원문 글자를 못 만들고 건너뜀
                if cand < best:
                    best, how = cand, "miss"
            cost[i][j], move[i][j] = best, how

    tags: list[Tag] = []
    i = j = 0
    space = False
    while i < n:
        how = move[i][j]
        if how == "space":
            space, j = True, j + 1
            continue
        if how == "miss":
            j += 1
            continue
        if how == "keep" or how == "wrong":
            tags.append(Tag(KEEP, space_before=space)); j += 1
        elif how == "sub":
            tags.append(Tag(SUB, clean[j], space_before=space)); j += 1
        else:
            tags.append(Tag(DELETE, space_before=space))
        space = False
        i += 1
    return tags, int(cost[0][0][0])


def _add(c: tuple, err: int, edits: int) -> tuple:
    return (c[0] + err, c[1] + edits)


def slot_residual(obf: str, clean: str) -> int:
    """1기 슬롯 모델이 정답을 알아도 못 고치는 오류 수(편집거리 기준).

    길이가 다르면 아무것도 못 고친다. 길이가 같으면 한글 음절끼리 바뀐 자리만 고친다.
    """
    if len(obf) != len(clean):
        return _edit_distance(obf, clean)
    return sum(a != b and not (is_syllable(a) and is_syllable(b)) for a, b in zip(obf, clean))


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
