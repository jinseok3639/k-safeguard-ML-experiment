"""P04.10 — 저장된 정규화기 모델을 새 지표로 다시 잰다 (p04.6-reeval-metrics).

재학습·임계값 재조정 없이, 각 run이 정한 기권 임계값 그대로 평가셋 v1과 평가 원문 695에 돌린다.

1. 재현: 옛 지표(외부 생성기 CER 개선, 원문 695 오변경)를 같은 출력에서 다시 계산해 원래 run 값과 맞춘다.
   바뀐 문장 수가 다르거나 CER 개선이 0.1%p 넘게 다르면 새 지표를 해석하지 않는다.
2. 새 지표(`kslab/restore_metrics.py`): 문장 결과 분포, 문장별 복원율, 글자 단위 F1, 유형별 재현율,
   기법 수·복원 전 CER 구간별, 정상 문장 오변경(단어 수·신뢰구간).
3. 변별력: 판정용 후보 지표마다 KcELECTRA 전체 미세조정 시드 3개 범위와 BiLSTM 시드 3개 범위가 겹치는지 본다.

MLflow에는 모델마다 하위 run 하나를 같은 키로 남기고, 그림 두 장(난이도별, 유형별 재현율)을 첨부한다.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import evalset as V
from kslab import record as R
from kslab import restore_metrics as M
from kslab import taxonomy as T
from kslab.models.edit_tagger import EditTagger
from kslab.models.pretrained_tagger import PretrainedEditTagger
from restoration.exp.run_p04_3_edit_tagger import EVALSET, MODELS, TARGET, Acc, _load, scope_of

EXPERIMENT = "P04_normalizer"
RESULTS = R.ROOT / "restoration" / "results"
GAIN_TOLERANCE = 0.001
# (이름, 묶음, 모델 파일, 임계값이 든 run, metrics.json 안의 경로)
MODELS_TO_SCORE = (
    ("ft-s0", "ft", "p04.3b-kcelectra-ft-s0.pt", "p04.3b-kcelectra-ft", ("kcelectra",)),
    ("ft-s1", "ft", "p04.3d-kcelectra-ft-s1-s1.pt", "p04.3d-kcelectra-ft-s1", ("kcelectra",)),
    ("ft-s2", "ft", "p04.3e-kcelectra-ft-s2-s2.pt", "p04.3e-kcelectra-ft-s2", ("kcelectra",)),
    ("lora-s0", "lora", "p04.3c-kcelectra-lora-s0.pt", "p04.3c-kcelectra-lora", ("kcelectra",)),
    ("bilstm-s0", "bilstm", "p04.3-edit-tagger-base-s0.pt", "p04.3b-kcelectra-ft", ("bilstm", "s0")),
    ("bilstm-s1", "bilstm", "p04.3-edit-tagger-base-s1.pt", "p04.3b-kcelectra-ft", ("bilstm", "s1")),
    ("bilstm-s2", "bilstm", "p04.3-edit-tagger-base-s2.pt", "p04.3b-kcelectra-ft", ("bilstm", "s2")),
)
FAMILY_LABEL = {"ft": "KcELECTRA 전체 미세조정", "lora": "KcELECTRA LoRA", "bilstm": "BiLSTM"}
# 판정용 후보 지표: (보드, 지표)
CANDIDATES = (("ext", "restore_rate"), ("ext", "f1"), ("int", "f1"))
SIZES = ("1", "2", "3", "4")


def original(run_id: str, path: tuple[str, ...]) -> dict:
    node = json.loads((RESULTS / run_id / "metrics.json").read_text(encoding="utf-8"))
    for key in path:
        node = node[key]
    return node


def board_of(record: dict) -> str | None:
    """`ext`(외부 생성기 target), `int`(내부 단일·복합 target), 그 밖은 결과 분포만 보는 `mixed`·`control`."""
    scope = scope_of(V.types_of(record))
    if scope != "target":
        return scope
    return "ext" if record["axis"] == "external" else "int"


def score_model(model: EditTagger, threshold: float, records: list[dict], cleans: list[str], batch: int) -> dict:
    outs = [model.decide(r["obfuscated"], row, threshold)
            for r, row in zip(records, model.predict([r["obfuscated"] for r in records], batch_size=batch))]
    benign = [model.decide(c, row, threshold) for c, row in zip(cleans, model.predict(cleans, batch_size=batch))]

    old = Acc()
    boards = {name: M.Board() for name in ("ext", "int", "mixed", "control")}
    for r, out in zip(records, outs):
        name = board_of(r)
        if r["axis"] == "external" and name == "ext":
            old.add(r["clean"], r["obfuscated"], out)
        groups = {"size": str(min(r["n_applied"], 4)), "cer": M.cer_bin(r["clean"], r["obfuscated"])}
        boards[name].add(r["clean"], r["obfuscated"], out, groups=groups,
                         record=r if name in ("ext", "int") else None)
    return {"threshold": threshold, "old": {"external_gain": old.row()["cer_gain"]},
            "boards": {k: b.summary() for k, b in boards.items()}, "benign": M.benign_report(cleans, benign)}


def mlflow_numbers(result: dict) -> dict[str, float]:
    """모델 하위 run에 올릴 숫자. 모든 모델이 같은 키를 쓴다."""
    out: dict[str, float] = {}
    for board in ("ext", "int"):
        s = result["boards"][board]
        for key in ("restore_rate", "f1", "precision", "recall", "collateral", *M.OUTCOMES):
            if s.get(key) is not None:
                out[f"{board}/{key}"] = s[key]
        for t, row in s.get("type_recall", {}).items():
            if row["recall"] is not None and t != M.UNKNOWN:
                out[f"{board}/type/{t}/recall"] = row["recall"]
    b = result["benign"]
    out.update({"benign/mutation": b["mutation"], "benign/mutation_strict": b["mutation_strict"],
                "benign/error_words": b["error_words"], "benign/ci_high": b["ci_high"]})
    return out


def _family_mean(results: dict, family: str, get) -> float:
    vals = [get(r) for name, r in results.items() if r["family"] == family]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def draw(results: dict, directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if any(f.name == "Malgun Gothic" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"ft": "#2b6cb0", "lora": "#c05621", "bilstm": "#718096"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (("int", "size", SIZES, "평가셋 내부 · 기법 수"),
              ("ext", "cer", tuple(name for _, _, name in M.CER_BINS), "외부 생성기 · 복원 전 CER 구간"))
    for row, (board, group, keys, title) in enumerate(panels):
        for col, metric in enumerate(("restore_rate", "f1")):
            ax = axes[row][col]
            for name, r in results.items():
                by = r["boards"][board]["by"].get(group, {})
                ys = [by.get(k, {}).get(metric) for k in keys]
                ax.plot(range(len(keys)), [np.nan if y is None else y for y in ys], marker="o",
                        color=colors[r["family"]], alpha=0.85, label=name)
            ax.set_xticks(range(len(keys)), keys)
            ax.set_title(f"{title} — {'문장별 복원율' if metric == 'restore_rate' else 'F1'}")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "difficulty.png", dpi=120)
    plt.close(fig)

    names = {t.id: t.name for t in T.TYPES}
    types = sorted(TARGET)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, board, title in zip(axes, ("int", "ext"), ("평가셋 내부", "외부 생성기")):
        width = 0.27
        for k, family in enumerate(("ft", "lora", "bilstm")):
            ys = [_family_mean(results, family, lambda r, t=t: r["boards"][board].get("type_recall", {})
                               .get(t, {}).get("recall")) for t in types]
            ax.bar(np.arange(len(types)) + (k - 1) * width, ys, width, color=colors[family], label=FAMILY_LABEL[family])
        ax.set_xticks(range(len(types)), [names[t] for t in types], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} — 유형별 재현율(고쳐야 할 글자 중 맞게 고친 비율, 묶음별 시드 평균)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "type_recall.png", dpi=120)
    plt.close(fig)
    return ["difficulty.png", "type_recall.png"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p04.6-reeval-metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", help="이 이름의 모델만 잰다(점검용)")
    args = parser.parse_args()

    chosen = [m for m in MODELS_TO_SCORE if not args.only or m[0] in args.only]
    cfg = {"models": {name: {"file": file, "threshold_from": run} for name, _, file, run, _ in chosen},
           "gain_tolerance": GAIN_TOLERANCE, "mutation_rule": "문장부호·공백 무시(엄격 기준 병기)",
           "restore_rate_lower_bound": "none"}
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT, config=cfg)
    run.add_input("evalset_v1/records.jsonl", EVALSET / "records.jsonl")
    records = _load(EVALSET / "records.jsonl")
    cleans = [s.text for s in V.load_sources()]

    results: dict[str, dict] = {}
    for name, family, file, source_run, path in chosen:
        stored = original(source_run, path)
        cls = EditTagger if family == "bilstm" else PretrainedEditTagger
        model = cls.load(MODELS / file, device=args.device)
        result = score_model(model, stored["threshold"], records, cleans, batch=128 if family == "bilstm" else 32)
        del model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        old_gain = stored["eval"]["axis"]["external/target"]["cer_gain"]
        old_mutated = round(stored["clean_mutation"] * len(cleans))
        result.update({"family": family, "reproduce": {
            "external_gain": [old_gain, result["old"]["external_gain"]],
            "mutated": [old_mutated, result["benign"]["mutated"]],
            "ok": abs(old_gain - result["old"]["external_gain"]) <= GAIN_TOLERANCE
                  and old_mutated == result["benign"]["mutated"]}})
        results[name] = result
        e = result["boards"]["ext"]
        print(f"[{name}] reproduce={result['reproduce']['ok']} ext restore={e['restore_rate']:.4f} "
              f"f1={e['f1']:.4f} benign={result['benign']['mutated']}/{len(cleans)}", flush=True)

    reproduced = all(r["reproduce"]["ok"] for r in results.values())
    separation = {}
    for board, metric in CANDIDATES:
        ranges = {f: [r["boards"][board][metric] for r in results.values() if r["family"] == f] for f in ("ft", "bilstm")}
        if all(ranges.values()):
            ft, bi = ranges["ft"], ranges["bilstm"]
            separation[f"{board}/{metric}"] = {"ft": [min(ft), max(ft)], "bilstm": [min(bi), max(bi)],
                                               "separates": min(ft) > max(bi) or max(ft) < min(bi)}

    figures = draw(results, run.directory)
    if run.tracker is not None:
        for name, r in results.items():
            child = run.tracker.open_child(name, {"family": r["family"], "threshold": r["threshold"]},
                                           {"kslab.kind": "eval"})
            for key, value in mlflow_numbers(r).items():
                run.tracker.log_step(key, float(value), 0, run_id=child)

    def pct(x):
        return "-" if x is None else f"{x * 100:.1f}"

    lines = [f"# {args.run_id} — 저장된 모델을 새 지표로 재평가", "",
             f"재현: {'모두 맞음' if reproduced else '어긋남 — 새 지표를 해석하지 않는다'}", "",
             "| 모델 | 외부 CER 개선(원래 → 재계산) | 원문 695 바뀐 문장(원래 → 재계산) | 재현 |", "|---|---|---|---|"]
    for name, r in results.items():
        g, m = r["reproduce"]["external_gain"], r["reproduce"]["mutated"]
        lines.append(f"| {name} | {pct(g[0])} → {pct(g[1])} | {m[0]} → {m[1]} | {'예' if r['reproduce']['ok'] else '아니오'} |")
    lines += ["", "## 판정용 후보 지표의 변별력 (전체 미세조정 시드 3 vs BiLSTM 시드 3)", "",
              "| 지표 | 전체 미세조정 범위 | BiLSTM 범위 | 가르나 |", "|---|---|---|---|"]
    for key, s in separation.items():
        lines.append(f"| {key} | {pct(s['ft'][0])}~{pct(s['ft'][1])} | {pct(s['bilstm'][0])}~{pct(s['bilstm'][1])} | "
                     f"{'예' if s['separates'] else '아니오'} |")
    for board, title in (("ext", "외부 생성기"), ("int", "평가셋 내부")):
        lines += ["", f"## {title} (복원 대상 유형만 든 문장)", "",
                  "| 모델 | 문장 | 완전 복원 | 나아짐 | 같음 | 나빠짐 | 문장별 복원율 | F1 | 정밀도 | 재현율 | 멀쩡한 글자 손상 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            s = r["boards"][board]
            lines.append(f"| {name} | {s['n']} | {pct(s['exact'])} | {pct(s['better'])} | {pct(s['same'])} | "
                         f"{pct(s['worse'])} | {pct(s['restore_rate'])} | {pct(s['f1'])} | {pct(s['precision'])} | "
                         f"{pct(s['recall'])} | {pct(s['collateral'])} |")
    lines += ["", "## 정상 문장 (평가 원문 695)", "",
              "| 모델 | 오변경(문장부호 무시) | 95% 구간 | 엄격 | 서로 다른 오류 단어 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        b = r["benign"]
        lines.append(f"| {name} | {pct(b['mutation'])}% ({b['mutated']}) | {pct(b['ci_low'])}~{pct(b['ci_high'])}% | "
                     f"{pct(b['mutation_strict'])}% | {b['error_words']} |")
    lines += ["", f"그림: {', '.join(figures)}. 세부(기법 수·CER 구간별, 유형별, 오류 단어 쌍)는 metrics.json."]
    run.write_text("report.md", "\n".join(lines) + "\n")

    metrics = {"models": results, "separation": separation,
               "status": "reproduced" if reproduced else "not_reproduced"}
    headline = {}
    for family in ("ft", "bilstm", "lora"):
        for board, metric in CANDIDATES:
            value = _family_mean(results, family, lambda r: r["boards"][board][metric])
            if not np.isnan(value):
                headline[f"{family}/{board}/{metric}"] = value
    run.finish(metrics, headline=headline)
    print("\n".join(lines))
    print(f"\n[{args.run_id}] {metrics['status']} -> {run.directory}")


if __name__ == "__main__":
    main()
