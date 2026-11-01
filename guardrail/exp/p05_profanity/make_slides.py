"""p05 발표용 슬라이드 그림 — 그래프와 범례만 그린다. 제목·설명·각주는 PPT에서 쓰도록 `slides_text.md`로 따로 낸다.

기존 필터는 회색, 우리 라이브러리는 파란색 하나로 강조한다. "우리 라이브러리"는 원문 OR 복원문 결합(실행 전 등록 구성,
p05-mixed-obfuscation)이고, 지표는 난독화된 욕설 중 차단된 비율(감지율)이다. 모든 수치는 results/p05.*/scores.jsonl에서
다시 계산한다.

데이터셋 표기(발표용): 데이터셋 1 = p04mix(복원 대상 9유형만 섞음), 데이터셋 2 = compose(경계 유형 포함 섞음).

    python guardrail/exp/p05_profanity/make_slides.py   # 저장소 루트에서 -> reports/figures/p05/slide_*.png, slides_text.md
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[3]
RES, OUT = ROOT / "guardrail" / "results", ROOT / "guardrail" / "reports" / "figures" / "p05"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

SURF, INK, INK2, GRID = "#ffffff", "#141413", "#5f5e5a", "#ebeae6"
OURS, BASE = "#2a78d6", "#7d7c77"
W, H = 13.33, 6.4                       # PPT 제목 아래 본문 영역(16:9 슬라이드의 약 85%)
PCT = FuncFormatter(lambda v, _: f"{v:.0f}%")

NAME = {"jamo_decompose": "자모 분해", "phon_insert": "받침·반모음 추가", "spacing": "띄어쓰기", "symbol_insert": "기호 삽입",
        "phon_vowel": "모음 교체", "phon_initial": "초성 교체·된소리", "shape_foreign": "닮은 외국문자", "chosung": "초성체",
        "phon_liaison": "연음", "phon_spelling": "소리 나는 대로", "phon_coda": "받침 교체", "syllable_extend": "음절 늘이기",
        "shape_hangul": "야민정음"}
RUNS = {  # (필터, 데이터) -> (단독 run, 복원기 run)
    ("kor_unsmile", "single"): ("p05.2-kor-unsmile", "p05.4-kor-unsmile-restored"),
    ("kor_unsmile", "d1"): ("p05.6-kor-unsmile-p04mix", "p05.8-kor-unsmile-p04mix-restored"),
    ("kor_unsmile", "d2"): ("p05.10-kor-unsmile-compose", "p05.12-kor-unsmile-compose-restored"),
    ("korcen", "single"): ("p05.1-korcen", "p05.3-korcen-restored"),
    ("korcen", "d2"): ("p05.9-korcen-compose", "p05.11-korcen-compose-restored"),
}
DATA = {"single": "단일 난독화", "d1": "섞은 난독화 · 데이터셋 1", "d2": "섞은 난독화 · 데이터셋 2"}


def load(run: str) -> dict:
    with open(RES / run / "scores.jsonl", encoding="utf-8") as f:
        return {j["id"]: j for j in map(json.loads, f)}


def detect(base: str, restored: str, key=lambda x: "all") -> dict:
    """난독화된 욕설 감지율(%): (필터 단독, 원문 OR 복원문, 행 수). key로 묶는다."""
    b, r = load(base), load(restored)
    acc = defaultdict(lambda: [0, 0, 0])
    for i, x in b.items():
        if x["technique"] and x["label"] == 1:
            a = acc[key(x)]
            a[0] += x["flagged"]; a[1] += x["flagged"] or r[i]["flagged"]; a[2] += 1
    return {k: (100 * v[0] / v[2], 100 * v[1] / v[2], v[2]) for k, v in acc.items()}


def clean_fpr(base: str, restored: str) -> tuple[float, float]:
    b, r = load(base), load(restored)
    neg = [i for i, x in b.items() if x["technique"] is None and x["label"] == 0]
    return (100 * sum(b[i]["flagged"] for i in neg) / len(neg),
            100 * sum(b[i]["flagged"] or r[i]["flagged"] for i in neg) / len(neg))


def figure(legend: bool = True):
    fig = plt.figure(figsize=(W, H), facecolor=SURF)
    if legend:
        fig.text(0.04, 0.93, "●", color=BASE, fontsize=24, va="center")
        fig.text(0.07, 0.93, "기존 필터만", color=INK, fontsize=20, va="center")
        fig.text(0.25, 0.93, "●", color=OURS, fontsize=24, va="center")
        fig.text(0.28, 0.93, "기존 필터 + 우리 라이브러리", color=INK, fontsize=20, va="center", fontweight="bold")
    return fig


def clean(ax, grid="x"):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2, labelsize=18)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=1.2)
    ax.set_axisbelow(True)


def dumbbell(ax, labels, pairs, xlim, delta_x):
    ys = list(range(len(labels)))[::-1]
    for y, (lo, hi) in zip(ys, pairs):
        ax.plot([lo, hi], [y, y], color=OURS, lw=6, alpha=0.22, solid_capstyle="round", zorder=1)
        ax.scatter([lo], [y], s=420, color=BASE, zorder=3, edgecolor=SURF, linewidth=3)
        ax.scatter([hi], [y], s=520, color=OURS, zorder=4, edgecolor=SURF, linewidth=3)
        ax.text(lo - 1.6, y, f"{lo:.0f}%", ha="right", va="center", fontsize=20, color=INK2)
        ax.text(hi + 1.6, y, f"{hi:.0f}%", ha="left", va="center", fontsize=22, color=INK, fontweight="bold")
        ax.text(delta_x, y, f"+{hi - lo:.0f}%p", ha="left", va="center", fontsize=24, color=OURS, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=19, color=INK)
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.set_major_formatter(PCT)


def slide1() -> dict:
    keys = list(RUNS)
    vals = {k: detect(*RUNS[k])["all"] for k in keys}
    fig = figure()
    ax = fig.add_axes([0.37, 0.09, 0.58, 0.75]); clean(ax)
    labels = [f"{f} · {DATA[d]}" if d != "single" else f"{f} · {DATA[d]}" for f, d in keys]
    dumbbell(ax, labels, [vals[k][:2] for k in keys], (30, 100), 90)
    ax.axhline(1.5, color=GRID, lw=2)
    fig.savefig(OUT / "slide_1_detection.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return vals


def slide2() -> dict:
    d = detect(*RUNS[("kor_unsmile", "d1")], key=lambda x: x["type"])
    ns = sorted(k for k in d if d[k][2] > 100)
    xs = [int(k.split("=")[1]) for k in ns]
    lo, hi = [d[k][0] for k in ns], [d[k][1] for k in ns]
    fig = figure()
    ax = fig.add_axes([0.08, 0.14, 0.80, 0.70]); clean(ax, "y")
    ax.fill_between(xs, lo, hi, color=OURS, alpha=0.10, lw=0)
    ax.plot(xs, lo, color=BASE, lw=5, marker="o", ms=18, mec=SURF, mew=3)
    ax.plot(xs, hi, color=OURS, lw=5.5, marker="o", ms=19, mec=SURF, mew=3)
    for x, a, b in zip(xs, lo, hi):
        ax.text(x, a - 3.6, f"{a:.0f}%", ha="center", va="top", fontsize=21, color=INK2)
        ax.text(x, b + 3.0, f"{b:.0f}%", ha="center", va="bottom", fontsize=22, color=INK, fontweight="bold")
    ax.text(xs[-1] + 0.14, (lo[-1] + hi[-1]) / 2, f"+{hi[-1] - lo[-1]:.0f}%p", fontsize=30, color=OURS,
            fontweight="bold", va="center")
    ax.set_xticks(xs); ax.set_xticklabels([f"기법 {x}개" for x in xs], fontsize=20, color=INK)
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.75); ax.set_ylim(30, 90)
    ax.yaxis.set_major_formatter(PCT)
    fig.savefig(OUT / "slide_2_mixed.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return {"ns": xs, "lo": lo, "hi": hi}


def slide3() -> dict:
    d = detect(*RUNS[("kor_unsmile", "single")], key=lambda x: x["type"])
    top = sorted(d, key=lambda k: -(d[k][1] - d[k][0]))[:6]
    fig = figure()
    ax = fig.add_axes([0.25, 0.09, 0.70, 0.75]); clean(ax)
    dumbbell(ax, [NAME[k] for k in top], [d[k][:2] for k in top], (40, 100), 91.5)
    fig.savefig(OUT / "slide_3_by_type.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return {k: d[k] for k in sorted(d, key=lambda k: -(d[k][1] - d[k][0]))}


def slide4() -> list:
    models = [("우리 라이브러리", 0.129, OURS), ("LLM 해제 모델 A", 7.6, BASE),
              ("LLM 해제 모델 B", 7.8, BASE), ("LLM 해제 모델 C", 8.0, BASE)]
    fig = figure(legend=False)
    ax = fig.add_axes([0.22, 0.10, 0.72, 0.82]); clean(ax)
    ys = list(range(len(models)))[::-1]
    for y, (n, v, c) in zip(ys, models):
        ax.barh(y, v, height=0.6, color=c, edgecolor=SURF, linewidth=3)
        ax.text(v + 0.15, y, f"{v:g}B" if v >= 1 else f"{v * 1000:.0f}M", va="center", fontsize=24,
                color=OURS if c == OURS else INK, fontweight="bold" if c == OURS else "normal")
    ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in models], fontsize=20, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlim(0, 9.2); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}B"))
    fig.savefig(OUT / "slide_4_size.png", dpi=200, facecolor=SURF)
    plt.close(fig)
    return models


def text(v1, v2, v3) -> str:
    u = lambda d: f"{v1[('kor_unsmile', d)][0]:.0f}% → {v1[('kor_unsmile', d)][1]:.0f}%"
    f1, f2 = clean_fpr(*RUNS[("kor_unsmile", "d1")])
    t = list(v3.items())
    rest = ", ".join(f"{NAME[k]} +{v[1] - v[0]:.1f}%p" for k, v in t[6:])
    return f"""# P05 발표 슬라이드 문구

그림(`slide_*.png`)에는 그래프와 범례만 있다. 아래 제목·설명·각주를 PPT에 직접 쓴다. 수치는 `make_slides.py`가 결과 파일에서
다시 계산해 이 파일을 함께 만든다.

## 공통: 데이터셋 설명 (첫 결과 슬라이드나 부록에 한 번)

- **평가 문장**: 공개 커뮤니티 댓글 욕설 데이터(Curse-detection, 5,825문장 중 욕설 2,044문장)
- **단일 난독화**: 욕설 문장마다 난독화 기법 하나씩 적용(24개 기법)
- **섞은 난독화 · 데이터셋 1**: 한 문장에 기법 1~4개를 섞음. 우리 라이브러리가 복원하도록 설계된 9개 유형만 사용
- **섞은 난독화 · 데이터셋 2**: 한 문장에 기법 2~4개를 섞음. 복원 범위 밖의 모양 바꾸기(야민정음·닮은 외국문자 등)까지 포함한 더 현실적인 조건
- **우리 라이브러리 적용 방식**: 기존 필터는 그대로 두고, 원문과 복원문을 둘 다 검사해 하나라도 걸리면 차단
- **기존 필터**: kor_unsmile(공개 ML 분류기), korcen(공개 규칙 기반 필터)

## 슬라이드 1 — `slide_1_detection.png`

- 제목: **우리 라이브러리를 붙이면 난독화된 욕설을 더 잡는다**
- 한 줄 설명: 난독화된 욕설 중 차단된 비율. 기존 필터는 바꾸지 않고 옆에 붙이기만 했다
- 말할 거리: kor_unsmile 기준 단일 난독화 {u('single')}, 데이터셋 1 {u('d1')}, 데이터셋 2 {u('d2')}
- 각주: 정상 문장을 욕설로 잘못 막는 비율은 거의 그대로(kor_unsmile {f1:.1f}% → {f2:.1f}%)

## 슬라이드 2 — `slide_2_mixed.png`

- 제목: **섞을수록 기존 필터는 무너지고, 우리 라이브러리와의 차이는 벌어진다**
- 한 줄 설명: kor_unsmile, 한 문장에 섞은 난독화 기법 수별 감지율(데이터셋 1)
- 말할 거리: 기법 4개를 섞으면 기존 필터는 {v2['lo'][-1]:.0f}%만 잡지만, 우리 라이브러리를 붙이면 {v2['hi'][-1]:.0f}%(+{v2['hi'][-1] - v2['lo'][-1]:.0f}%p)
- 각주: 데이터셋 2(복원 범위 밖 유형 포함)에서도 +12~14%p 차이 유지

## 슬라이드 3 — `slide_3_by_type.png`

- 제목: **가장 크게 되찾는 난독화 유형**
- 한 줄 설명: kor_unsmile, 난독화 유형별 감지율(단일 난독화, 향상폭 상위 6유형)
- 말할 거리: 자모 분해(ㅅ ㅣ ㅂ ㅏ ㄹ 식) {t[0][1][0]:.0f}% → {t[0][1][1]:.0f}%, 받침·반모음 추가 {t[1][1][0]:.0f}% → {t[1][1][1]:.0f}%
- 각주: 나머지 유형은 {rest}. 초성체는 현재 복원 범위 밖, 닮은 외국문자·야민정음·음절 늘이기는 경계(복원 여부 미정)

## 슬라이드 4 — `slide_4_size.png`

- 제목: **LLM 기반 해제 모델의 약 1/60 크기**
- 한 줄 설명: 난독화 복원 모델의 파라미터 수
- 말할 거리: 우리 라이브러리 1.29억 개, LLM 해제 모델 76억~80억 개. 문장 하나 처리 시간 GPU 약 7ms(p50)
- 각주: LLM 해제 모델 A·B·C는 KOTOX 연구(arXiv 2510.10961)가 공개한 Qwen2.5-7B·EXAONE-3.5-7.8B·Bllossom-8B 기반 모델. 크기만 비교했고 속도·정확도는 같은 조건에서 재지 않았다
"""


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    v1, v2, v3 = slide1(), slide2(), slide3()
    slide4()
    (OUT / "slides_text.md").write_text(text(v1, v2, v3), encoding="utf-8", newline="\n")
    print("ok")
