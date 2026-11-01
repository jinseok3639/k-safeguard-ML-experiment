"""최종 발표 그림 8장 — 시스템 2개 × 비교 2가지 × 지표 2개. matplotlib 기본 테마(한글 글꼴만 지정).

시스템
  ML 기반 욕설 필터  smilegate-ai/kor_unsmile, Curse-detection 욕설 2,044문장(p05.10·p05.12)
  Kanana 가드레일   kanana-safeguard-prompt-2.1b, 평가셋 v1 공격 301문장(s01.1·p02.1)
난독화: 두 시스템 모두 복합 기법(평가셋 v1 compose 규칙, 한 문장에 기법 2~4개)
비교
  ① 원문 vs 난독화 문장
  ② 난독화 문장 vs k-safeguard 적용 후(원문과 정규화문 중 하나라도 걸리면 차단, 실행 전 등록 구성)
지표: 탐지율 = 욕설·공격 문장 중 차단 비율, 회피율 = 통과 비율(100 − 탐지율). 행 단위

    python guardrail/exp/make_final_figures.py   # 저장소 루트에서 -> reports/figures/final/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from guardrail import evalset as E  # noqa: E402

RES, OUT = ROOT / "guardrail" / "results", ROOT / "guardrail" / "reports" / "figures" / "final"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})   # 한글 글꼴만. 나머지는 기본


def load(run: str) -> dict:
    with open(RES / run / "scores.jsonl", encoding="utf-8") as f:
        return {j["id"]: j for j in map(json.loads, f)}


def pct(ids, flag) -> float:
    return 100 * sum(bool(flag(i)) for i in ids) / len(ids)


def profanity() -> dict:
    b, r = load("p05.10-kor-unsmile-compose"), load("p05.12-kor-unsmile-compose-restored")
    src = [i for i, x in b.items() if x["label"] == 1 and x["technique"] is None]
    obf = [i for i, x in b.items() if x["label"] == 1 and x["technique"]]
    return {"원문": pct(src, lambda i: b[i]["flagged"]), "난독화 문장": pct(obf, lambda i: b[i]["flagged"]),
            "k-safeguard 적용 후": pct(obf, lambda i: b[i]["flagged"] or r[i]["flagged"]),
            "n": (len(src), len(obf))}


def kanana() -> dict:
    s = load("p02.1-kanana-restored")
    rows = E.clean_rows(E.load_records(ROOT / "data" / "evalset" / "kslab-evalset-v1"))
    rows += E.load_records(ROOT / "data" / "evalset" / "kslab-evalset-v1")
    src = [r["id"] for r in rows if r["label"] == "attack" and r["axis"] == "clean"]
    obf = [r["id"] for r in rows if r["label"] == "attack" and r["axis"] == "compose"]
    return {"원문": pct(src, lambda i: s[i]["flagged_base"]), "난독화 문장": pct(obf, lambda i: s[i]["flagged_base"]),
            "k-safeguard 적용 후": pct(obf, lambda i: s[i]["flagged_or"]), "n": (len(src), len(obf))}


def bar(system: str, metric: str, a: str, b: str, va: float, vb: float, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["C0", "C1" if b == "k-safeguard 적용 후" else "C3"]
    bars = ax.bar([a, b], [va, vb], color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=13)
    d = vb - va
    ax.set_title(f"{system} · {metric}  ({'+' if d >= 0 else '-'}{abs(d):.1f}%p)", fontsize=13)
    ax.set_ylabel(f"{metric}(%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=12)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"ml": ("ML 기반 욕설 필터", profanity()), "kanana": ("Kanana 가드레일", kanana())}
    lines = ["# 최종 발표 그림 8장", "",
             "모두 복합 기법 난독화(한 문장에 기법 2~4개)다. 탐지율은 욕설·공격 문장 중 차단된 비율, 회피율은 통과한 비율(100 − 탐지율)이다.",
             "k-safeguard 적용 후는 원문과 정규화한 문장을 둘 다 검사해 하나라도 걸리면 차단하는 구성이다(실행 전 등록).", "",
             "| 파일 | 시스템 | 비교 | 앞 | 뒤 | 변화 |", "|---|---|---|---|---|---|"]
    for key, (system, v) in data.items():
        for cmp_key, (a, b) in {"obf": ("원문", "난독화 문장"), "norm": ("난독화 문장", "k-safeguard 적용 후")}.items():
            for metric, f in (("탐지율", lambda x: x), ("회피율", lambda x: 100 - x)):
                name = f"{key}_{cmp_key}_{'detect' if metric == '탐지율' else 'evade'}.png"
                bar(system, metric, a, b, f(v[a]), f(v[b]), name)
                lines.append(f"| `{name}` | {system} | {a} → {b} | {f(v[a]):.1f}% | {f(v[b]):.1f}% | "
                             f"{f(v[b]) - f(v[a]):+.1f}%p |")
    ml_n, kn_n = data["ml"][1]["n"], data["kanana"][1]["n"]
    lines += ["", "## 각주용 설명", "",
              f"- ML 기반 욕설 필터: 공개 한국어 혐오·욕설 분류기(smilegate-ai/kor_unsmile). 평가 문장은 공개 커뮤니티 댓글 욕설 "
              f"{ml_n[0]:,}문장과 그 복합 난독화본 {ml_n[1]:,}행",
              f"- Kanana 가드레일: 카카오 프롬프트 공격 탐지 모델(kanana-safeguard-prompt-2.1b). 평가 문장은 공격 프롬프트 "
              f"{kn_n[0]}문장과 그 복합 난독화본 {kn_n[1]:,}행",
              "- 같은 조건에서 정상 문장을 잘못 막는 비율: ML 필터 12.5% → 13.0%, Kanana 3.8% → 3.8%",
              "- 이 회피율(전체 공격 중 통과)은 보고서의 '원문 차단 조건부 우회율'(Kanana 27.5% → 15.7%)과 정의가 다르다. "
              "그림과 같은 숫자로 말할 때는 이 파일의 표를 쓴다"]
    (OUT / "figures_text.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
