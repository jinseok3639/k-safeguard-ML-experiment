"""p05 발표용 슬라이드 그림(16:9) — 기존 필터는 회색, 우리 라이브러리는 파란색 하나로 강조한다.

모든 수치는 results/p05.*/scores.jsonl에서 다시 계산한다. "우리 라이브러리"는 원문 OR 복원문 결합(실행 전 등록 구성,
p05-mixed-obfuscation)이다. 지표는 난독화된 욕설 전체 중 차단된 비율(감지율)이다.

    python guardrail/exp/p05_profanity/make_slides.py   # 저장소 루트에서 -> reports/figures/p05/slide_*.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES, OUT = ROOT / "guardrail" / "results", ROOT / "guardrail" / "reports" / "figures" / "p05"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

SURF, INK, INK2, GRID = "#ffffff", "#141413", "#5f5e5a", "#ebeae6"
OURS, BASE = "#2a78d6", "#7d7c77"
W, H = 13.33, 7.5

NAME = {"jamo_decompose": "자모 분해", "phon_insert": "받침·반모음 추가", "spacing": "띄어쓰기", "symbol_insert": "기호 삽입",
        "phon_vowel": "모음 교체", "phon_initial": "초성 교체·된소리", "shape_foreign": "닮은 외국문자", "chosung": "초성체",
        "phon_liaison": "연음", "phon_spelling": "소리 나는 대로", "phon_coda": "받침 교체", "syllable_extend": "음절 늘이기",
        "shape_hangul": "야민정음"}


def load(run: str) -> dict:
    with open(RES / run / "scores.jsonl", encoding="utf-8") as f:
        return {j["id"]: j for j in map(json.loads, f)}


def detect(base: str, restored: str, key=lambda r: "all") -> dict:
    """난독화된 욕설 감지율: 필터 단독, 원문 OR 복원문. key로 묶는다."""
    b, r = load(base), load(restored)
    acc = defaultdict(lambda: [0, 0, 0])
    for i, x in b.items():
        if x["technique"] and x["label"] == 1:
            a = acc[key(x)]
            a[0] += x["flagged"]; a[1] += x["flagged"] or r[i]["flagged"]; a[2] += 1
    return {k: (100 * v[0] / v[2], 100 * v[1] / v[2], v[2]) for k, v in acc.items()}


def slide(title: str, subtitle: str, note: str):
    fig = plt.figure(figsize=(W, H), facecolor=SURF)
    fig.text(0.055, 0.915, title, fontsize=26, fontweight="bold", color=INK, va="bottom")
    fig.text(0.055, 0.87, subtitle, fontsize=15, color=INK2, va="top")
    fig.text(0.055, 0.035, note, fontsize=10.5, color=INK2)
    return fig


def clean(ax, grid="x"):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK2, labelsize=13)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)


def legend_chips(fig, y=0.80):
    fig.text(0.055, y, "●", color=BASE, fontsize=16, va="center")
    fig.text(0.075, y, "기존 필터 단독", color=INK, fontsize=14, va="center")
    fig.text(0.215, y, "●", color=OURS, fontsize=16, va="center")
    fig.text(0.235, y, "기존 필터 + 우리 라이브러리", color=INK, fontsize=14, va="center", fontweight="bold")


def dumbbell(ax, labels, pairs, xlim, delta_x):
    ys = list(range(len(labels)))[::-1]
    for y, (lo, hi) in zip(ys, pairs):
        ax.plot([lo, hi], [y, y], color=OURS, lw=4, alpha=0.25, solid_capstyle="round", zorder=1)
        ax.scatter([lo], [y], s=260, color=BASE, zorder=3, edgecolor=SURF, linewidth=2.5)
        ax.scatter([hi], [y], s=320, color=OURS, zorder=4, edgecolor=SURF, linewidth=2.5)
        ax.text(lo - 1.2, y, f"{lo:.0f}%", ha="right", va="center", fontsize=14, color=INK2)
        ax.text(hi + 1.2, y, f"{hi:.0f}%", ha="left", va="center", fontsize=15, color=INK, fontweight="bold")
        ax.text(delta_x, y, f"+{hi - lo:.1f}%p", ha="left", va="center", fontsize=16, color=OURS, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=14, color=INK)
    ax.set_xlim(*xlim); ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


def slide1():
    rows = [("kor_unsmile (ML)", "단일 기법", "p05.2-kor-unsmile", "p05.4-kor-unsmile-restored"),
            ("kor_unsmile (ML)", "섞은 난독화 · 학습 분포", "p05.6-kor-unsmile-p04mix", "p05.8-kor-unsmile-p04mix-restored"),
            ("kor_unsmile (ML)", "섞은 난독화 · 평가셋", "p05.10-kor-unsmile-compose", "p05.12-kor-unsmile-compose-restored"),
            ("korcen (규칙)", "단일 기법", "p05.1-korcen", "p05.3-korcen-restored"),
            ("korcen (규칙)", "섞은 난독화 · 평가셋", "p05.9-korcen-compose", "p05.11-korcen-compose-restored")]
    vals = [detect(b, r)["all"][:2] for _, _, b, r in rows]
    fig = slide("우리 라이브러리를 붙이면 난독화 욕설을 더 잡는다",
                "난독화된 욕설 중 차단된 비율 · 기존 필터는 그대로 두고 옆에 붙이기만 했다",
                "Curse-detection 욕설 2,044문장의 난독화본. 우리 라이브러리: 원문과 복원문 중 하나라도 걸리면 차단(실행 전 등록 구성). "
                "정상 원문 오탐 증가 ≤ 0.5%p")
    legend_chips(fig)
    ax = fig.add_axes([0.30, 0.12, 0.62, 0.62]); clean(ax)
    dumbbell(ax, [f"{f}\n{c}" for f, c, _, _ in rows], vals, (30, 100), 90)
    ax.axhline(1.5, color=GRID, lw=1.5)
    fig.savefig(OUT / "slide_1_detection.png", dpi=200, facecolor=SURF)


def slide2():
    d = detect("p05.6-kor-unsmile-p04mix", "p05.8-kor-unsmile-p04mix-restored", key=lambda x: x["type"])
    ns = sorted(k for k in d if d[k][2] > 100)
    xs = [int(k.split("=")[1]) for k in ns]
    lo, hi = [d[k][0] for k in ns], [d[k][1] for k in ns]
    fig = slide("섞을수록 기존 필터는 무너지고, 차이는 벌어진다",
                "kor_unsmile · 한 문장에 섞은 난독화 기법 수별 감지율",
                "복원기 학습 분포 섞기(p04mix, 복원 대상 9유형, 초성체 제외). 평가셋 섞기(경계 유형 포함)에서도 +12~14%p 유지. "
                "우리 라이브러리: 원문 OR 복원문")
    legend_chips(fig)
    ax = fig.add_axes([0.08, 0.17, 0.80, 0.57]); clean(ax, "y")
    ax.fill_between(xs, lo, hi, color=OURS, alpha=0.10, lw=0)
    ax.plot(xs, lo, color=BASE, lw=3.5, marker="o", ms=13, mec=SURF, mew=2.5)
    ax.plot(xs, hi, color=OURS, lw=4, marker="o", ms=14, mec=SURF, mew=2.5)
    for x, a, b in zip(xs, lo, hi):
        ax.text(x, a - 3.2, f"{a:.0f}%", ha="center", va="top", fontsize=15, color=INK2)
        ax.text(x, b + 2.6, f"{b:.0f}%", ha="center", va="bottom", fontsize=16, color=INK, fontweight="bold")
    ax.annotate(f"+{hi[-1] - lo[-1]:.0f}%p", xy=(xs[-1] + 0.12, (lo[-1] + hi[-1]) / 2), fontsize=22, color=OURS,
                fontweight="bold", va="center")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}개" for x in xs], fontsize=15, color=INK)
    ax.set_xlim(xs[0] - 0.35, xs[-1] + 0.65); ax.set_ylim(30, 90)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("한 문장에 섞은 기법 수", fontsize=13, color=INK2, labelpad=8)
    fig.savefig(OUT / "slide_2_mixed.png", dpi=200, facecolor=SURF)


def slide3():
    d = detect("p05.2-kor-unsmile", "p05.4-kor-unsmile-restored", key=lambda x: x["type"])
    top = sorted(d, key=lambda k: -(d[k][1] - d[k][0]))[:7]
    fig = slide("가장 크게 되찾는 난독화 유형",
                "kor_unsmile · 유형별 감지율 (기법 하나씩 적용, 상위 7유형)",
                "나머지 6유형(연음·소리 나는 대로·받침 교체·초성체·음절 늘이기·야민정음)은 +1.6~6.2%p. "
                "초성체·야민정음·음절 늘이기는 복원 범위 밖 또는 경계")
    legend_chips(fig)
    ax = fig.add_axes([0.24, 0.12, 0.68, 0.62]); clean(ax)
    dumbbell(ax, [NAME[k] for k in top], [d[k][:2] for k in top], (40, 100), 92)
    fig.savefig(OUT / "slide_3_by_type.png", dpi=200, facecolor=SURF)


def slide4():
    models = [("우리 라이브러리\n(KcELECTRA 태거)", 0.129, OURS), ("KOTOX 해제 모델\nQwen2.5-7B", 7.6, BASE),
              ("KOTOX 해제 모델\nEXAONE-3.5-7.8B", 7.8, BASE), ("KOTOX 해제 모델\nBllossom-8B", 8.0, BASE)]
    fig = slide("LLM 해제 모델의 약 1/60 크기",
                "난독화 복원 모델 파라미터 수(십억 개)",
                "우리 라이브러리 지연(문장 1개): GPU p50 6.7ms·p95 12.1ms, CPU p50 14.9ms. "
                "KOTOX(arXiv 2510.10961) 공개 해제 모델과 크기만 비교했다(속도·정확도는 같은 조건에서 재지 않음)")
    ax = fig.add_axes([0.24, 0.14, 0.68, 0.62]); clean(ax)
    ys = list(range(len(models)))[::-1]
    for y, (n, v, c) in zip(ys, models):
        ax.barh(y, v, height=0.55, color=c, edgecolor=SURF, linewidth=2)
        ax.text(v + 0.12, y, f"{v:g}B" if v >= 1 else f"{v * 1000:.0f}M", va="center", fontsize=16,
                color=INK, fontweight="bold" if c == OURS else "normal")
    ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in models], fontsize=14, color=INK)
    ax.set_xlim(0, 9); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}B"))
    fig.savefig(OUT / "slide_4_size.png", dpi=200, facecolor=SURF)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (slide1, slide2, slide3, slide4):
        f()
    print("ok")
