"""p05 발표용 그림 4장 — baselines/p05_*과 results/p05.*/scores.jsonl을 읽어 reports/figures/p05/에 PNG로 쓴다.

    python guardrail/exp/p05_profanity/make_figures.py   # 저장소 루트에서
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 11})
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
B = "guardrail/baselines/"
def load(n): return json.load(open(B + n, encoding="utf-8"))["metrics"]
kc, ku = load("p05_korcen_v1.json"), load("p05_kor_unsmile_v1.json")
kr, ur = load("p05_korcen_restored_v1.json"), load("p05_kor_unsmile_restored_v1.json")
OURS = "kor_unsmile + 우리 라이브러리"
NAME = {"phon_insert": "받침·반모음 추가", "shape_foreign": "닮은 외국문자", "phon_vowel": "모음 교체", "jamo_decompose": "자모 분해",
        "chosung": "초성체", "spacing": "띄어쓰기", "symbol_insert": "기호 삽입", "phon_initial": "초성 교체·된소리",
        "syllable_extend": "음절 늘이기", "phon_liaison": "연음", "phon_spelling": "소리 나는 대로", "phon_coda": "받침 교체", "shape_hangul": "야민정음"}
OUT = "guardrail/reports/figures/p05/"

def style(ax):
    ax.set_facecolor(SURF); ax.figure.set_facecolor(SURF)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

def hbars(ax, labels, series, colors, names, ref=None, fmt="{:.0f}%", legend_loc="lower right", legend_kw={}):
    n = len(series); h = 0.8 / n
    for k, (vals, c, nm) in enumerate(zip(series, colors, names)):
        ys = [i + (k - (n - 1) / 2) * h for i in range(len(labels))]
        bars = ax.barh(ys, vals, height=h - 0.06, color=c, label=nm, edgecolor=SURF, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 0.8, y, fmt.format(v), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=INK); ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(ref + 0.4, -0.75, f"기준 {ref}%", color=INK2, fontsize=9, ha="left")
    ax.legend(frameon=False, loc=legend_loc, labelcolor=INK, fontsize=10, **legend_kw)

# 1. 유형별 회피율
types = sorted(NAME, key=lambda t: -max(kc["by_type"][t]["evasion"]["rate"], ku["by_type"][t]["evasion"]["rate"]))
fig, ax = plt.subplots(figsize=(9, 8.2)); style(ax)
hbars(ax, [NAME[t] for t in types], [[100 * kc["by_type"][t]["evasion"]["rate"] for t in types],
      [100 * ku["by_type"][t]["evasion"]["rate"] for t in types], [100 * ur["by_type"][t]["evasion"]["rate"] for t in types]],
      [C1, C2, C3], ["korcen(규칙)", "kor_unsmile(ML)", OURS], ref=30)
ax.set_xlabel("회피율: 원문은 잡혔는데 난독화본은 놓친 비율(%)", color=INK2)
ax.set_title("공개 필터는 최대 39%까지 뚫린다 — 우리 라이브러리는 복원 대상 유형의 회피를 줄인다", loc="left", color=INK, fontsize=12, pad=14)
fig.text(0.01, 0.005, "Curse-detection 욕설 2,044문장 × kslab 난독화(강도 0.5). 우리 라이브러리(복원문만 검사)는 13유형 중 7개에서 줄이고, 범위 밖·경계 유형(초성체·외국문자·야민정음 등)에서는 늘림", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "1_evasion_by_type.png", dpi=200)

# 2. 정상 문장 오탐
sel = ["clean", "jamo_decompose", "shape_foreign", "chosung"]
lab = {"clean": "원문(난독화 전)", **NAME}
def fpr(m, t): return 100 * (m["clean_fpr"]["rate"] if t == "clean" else m["by_type"][t]["obf_fpr"]["rate"])
fig, ax = plt.subplots(figsize=(9, 5.4)); style(ax)
hbars(ax, [lab[t] for t in sel], [[fpr(kc, t) for t in sel], [fpr(ku, t) for t in sel], [fpr(ur, t) for t in sel]],
      [C1, C2, C3], ["korcen(규칙)", "kor_unsmile(ML)", OURS])
ax.set_xlabel("정상 문장을 욕설로 막은 비율(%)", color=INK2)
ax.set_title("난독화된 정상 문장은 최대 절반이 막힌다 — 우리 라이브러리를 붙이면 오탐이 줄어든다", loc="left", color=INK, fontsize=12, pad=14)
fig.text(0.01, 0.01, "Curse-detection 정상 3,781문장과 그 난독화본. 우리 라이브러리: 필터 앞에서 복원문만 검사", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT + "2_false_positive.png", dpi=200)

# 3. 복원기 전후
def orc(base, rest):
    b = {j["id"]: j for j in map(json.loads, open(f"guardrail/results/{base}/scores.jsonl", encoding="utf-8"))}
    r = {j["id"]: j["flagged"] for j in map(json.loads, open(f"guardrail/results/{rest}/scores.jsonl", encoding="utf-8"))}
    ev = [i for i in b if b[i]["technique"] and b[i]["label"] == 1 and b[b[i]["src"]]["flagged"]]
    return 100 * sum(not (b[i]["flagged"] or r[i]) for i in ev) / len(ev)
rows = ["korcen(규칙)", "kor_unsmile(ML)"]
fig, ax = plt.subplots(figsize=(9, 4.4)); style(ax)
hbars(ax, rows, [[100 * kc["evasion"]["rate"], 100 * ku["evasion"]["rate"]],
                 [100 * kr["evasion"]["rate"], 100 * ur["evasion"]["rate"]],
                 [orc("p05.1-korcen", "p05.3-korcen-restored"), orc("p05.2-kor-unsmile", "p05.4-kor-unsmile-restored")]],
      [C1, C2, C3], ["필터 단독", "+ 우리 라이브러리 (복원문만 검사)", "+ 우리 라이브러리 (원문 OR 복원문, 사후 분석)"], fmt="{:.1f}%",
      legend_loc="upper center", legend_kw={"bbox_to_anchor": (0.5, -0.22), "ncol": 3})
ax.set_xlabel("전체 회피율(%) — 낮을수록 좋음", color=INK2)
ax.set_title("우리 라이브러리를 필터 옆에 붙이면 회피 41~60% 감소 (복원문만 검사하면 규칙 필터에선 역효과)", loc="left", color=INK, fontsize=12.5, pad=14)
fig.text(0.01, 0.01, "우리 라이브러리: P04.3b(KcELECTRA) 복원 모델. OR 결합 시 원문 오탐 korcen 1.3→1.3%, kor_unsmile 12.5→13.0%. OR은 결과를 본 뒤 정한 구성", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT + "3_restorer_effect.png", dpi=200)

# 4. 유형별 회복률(kor_unsmile)
rec = ur["recovery_by_type"]; ts = sorted(rec, key=lambda t: -rec[t]["rate"])
fig, ax = plt.subplots(figsize=(9, 5.6)); style(ax)
vals = [100 * rec[t]["rate"] for t in ts]
bars = ax.barh(range(len(ts)), vals, height=0.62, color=C1, edgecolor=SURF, linewidth=2)
for i, v in enumerate(vals): ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=9.5, color=INK)
ax.set_yticks(range(len(ts))); ax.set_yticklabels([NAME[t] for t in ts], color=INK); ax.invert_yaxis()
ax.axvline(50, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(50.6, -0.8, "기준 50%", color=INK2, fontsize=9, ha="left")
ax.set_xlabel("회복률: 필터가 놓친 난독화 욕설 중 우리 라이브러리로 복원한 뒤 다시 잡힌 비율(%)", color=INK2)
ax.set_title("우리 라이브러리는 필터가 놓친 욕설을 복원 대상 유형에서 54~89% 되찾는다 (kor_unsmile 기준)", loc="left", color=INK, fontsize=12, pad=14)
fig.text(0.01, 0.005, "우리 라이브러리 복원 범위: 초성체는 제외, 닮은 외국문자·야민정음·음절 늘이기는 경계(복원 여부 미정). 나머지가 복원 대상", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT + "4_recovery_by_type.png", dpi=200)
print("ok")

# 5. 섞은 기법 수 n에 따른 회피율(필터 2 × 난독화 축 2)
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
AXN = {"p04mix": "복원기 학습 분포 섞기(상한)", "compose": "평가셋 섞기(경계 유형 포함)"}
for i, (fk, fname) in enumerate((("kor_unsmile", "kor_unsmile(ML)"), ("korcen", "korcen(규칙)"))):
    for j, axn in enumerate(("p04mix", "compose")):
        ax = axes[i][j]; style(ax); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.xaxis.grid(False)
        a, r = load(f"p05_{fk}_{axn}_v1.json"), load(f"p05_{fk}_{axn}_restored_v1.json")
        ns = [k for k in a["by_type"] if a["by_type"][k]["evasion"]["n"] > 100]
        xs = [int(k.split("=")[1]) for k in ns]
        for vals, c, nm in (([a["by_type"][k]["evasion"]["rate"] for k in ns], C1, "필터 단독"),
                            ([r["by_type"][k]["evasion"]["rate"] for k in ns], C2, "+ 우리 라이브러리(복원문만 검사)"),
                            ([r["or"]["by_type_evasion"][k]["rate"] for k in ns], C3, "+ 우리 라이브러리(원문 OR 복원문)")):
            ys = [100 * v for v in vals]
            ax.plot(xs, ys, color=c, lw=2, marker="o", ms=7, mec=SURF, mew=2, label=nm)
            ax.text(xs[-1] + 0.12, ys[-1], f"{ys[-1]:.0f}%", va="center", fontsize=9, color=INK)
        ax.set_xticks(xs); ax.set_xticklabels([f"{x}개" for x in xs], color=INK2); ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.6)
        ax.set_title(f"{fname} · {AXN[axn]}", loc="left", fontsize=10.5, color=INK)
        if j == 0: ax.set_ylabel("회피율(%)", color=INK2)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=10, labelcolor=INK)
fig.suptitle("기법을 섞을수록 공개 필터는 더 뚫린다 — 우리 라이브러리를 OR로 붙이면 회피가 30~58% 준다", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.text(0.01, 0.005, "x축: 한 문장에 섞은 난독화 기법 수. 초성체 제외. OR 결합은 실행 전 등록한 구성(원문 오탐 증가 0.5%p 이하)", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.09, 1, 0.95)); fig.savefig(OUT + "5_mixed_by_n.png", dpi=200)
