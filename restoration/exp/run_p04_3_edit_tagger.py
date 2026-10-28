"""P04.3 — 글자 편집 태거 학습·평가 (p04.3-edit-tagger).

`data/derived/p04_train_v1`로 편집 태거(`kslab/models/edit_tagger.py`)를 학습하고 평가셋 v1 전 축에서 잰다.
음운 대표형 특징 끄기/켜기 × 시드 3.

기권 임계값은 dev의 손대지 않은 쌍(코퍼스 정상 문장)에서 문장 오변경이 1% 이하가 되게 정하고, 평가셋에는 고정해서 쓴다.

판정(노트에 실행 전 고정, 설정별 시드 평균): 평가셋 원문 695 오변경 ≤ 1% / external target CER 개선 ≥ 2%p /
compose 자동형 target에서 9유형 중 7개 이상 CER 개선 ≥ 2%p. 설정(특징 끄기/켜기) 선택은 dev CER 개선으로 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import evalset as V
from kslab import record as R
from kslab import taxonomy as T
from kslab.metrics import char_error_rate
from kslab.models.edit_tagger import EditTagger

EXPERIMENT = "P04_normalizer"
TRAIN = R.ROOT / "data" / "derived" / "p04_train_v1"
EVALSET = R.ROOT / "data" / "derived" / "evalset_v1"
MODELS = R.ROOT / "data" / "derived" / "p04_models"
AXES = ("single", "compose", "external", "control")
TARGET = {t.id for t in T.TYPES if t.scope == "target"}
BENIGN_BUDGET = 0.01
GAIN_BAR = 0.02
TYPES_BAR = 7


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def calibrate(confs: list[float], budget: float) -> float:
    """확신도가 이 값 이상인 문장만 바뀐다. 바뀌는 문장 비율이 `budget` 이하가 되는 가장 낮은 값."""
    ranked = sorted(confs, reverse=True)
    k = int(budget * len(ranked))
    return ranked[k] + 1e-9 if k < len(ranked) else 0.0


class Acc:
    def __init__(self):
        self.n = 0
        self.raw = self.rest = self.exact = 0.0
        self.raw_d = self.rest_d = 0.0

    def add(self, clean, obf, rest):
        raw, res = char_error_rate(clean, obf), char_error_rate(clean, rest)
        self.n += 1
        self.raw += raw
        self.rest += res
        self.exact += rest == clean
        self.raw_d += raw * max(len(clean), 1)
        self.rest_d += res * max(len(clean), 1)

    def row(self) -> dict:
        n = max(self.n, 1)
        return {"n": self.n, "raw_cer": self.raw / n, "cer_gain": (self.raw - self.rest) / n,
                "exact": self.exact / n, "err": (self.raw_d - self.rest_d) / max(self.raw_d, 1)}


def scope_of(types: list[str]) -> str:
    if types == ["control"]:
        return "control"
    return "target" if set(types) <= TARGET else "mixed"


def evaluate(model: EditTagger, records: list[dict], rows: list, thr: float) -> dict:
    out: dict[str, dict] = defaultdict(lambda: defaultdict(Acc))
    for r, row in zip(records, rows):
        rest = model.decide(r["obfuscated"], row, thr)
        types = V.types_of(r)
        scope = scope_of(types)
        out["axis"][f"{r['axis']}/{scope}"].add(r["clean"], r["obfuscated"], rest)
        if scope == "target":
            for t in types:
                out["type"][f"{t}/{r['axis']}"].add(r["clean"], r["obfuscated"], rest)
                if r["axis"] == "compose":
                    out["type"][f"{t}/compose_{r['name'].rstrip('0123456789.')}"].add(r["clean"], r["obfuscated"], rest)
    return {g: {k: a.row() for k, a in sorted(d.items())} for g, d in out.items()}


def train_one(phonetic: bool, seed: int, train: list[dict], dev: list[dict], records: list[dict],
              cleans: list[str], args) -> dict:
    tag = f"{'phon' if phonetic else 'base'}-s{seed}"
    model = EditTagger(phonetic=phonetic, random_state=2026 + seed, device=args.device, epochs=args.epochs)
    print(f"[fit] {tag}", flush=True)
    model.fit([p["obfuscated"] for p in train], [p["clean"] for p in train])
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODELS / f"{args.run_id}-{tag}.pt")

    dev_id = [p for p in dev if not p["recipe"]]
    dev_obf = [p for p in dev if p["recipe"]]
    thr = calibrate([model.edit_confidence(r) for r in model.predict([p["clean"] for p in dev_id])], BENIGN_BUDGET)
    dev_acc = Acc()
    for p, row in zip(dev_obf, model.predict([p["obfuscated"] for p in dev_obf])):
        dev_acc.add(p["clean"], p["obfuscated"], model.decide(p["obfuscated"], row, thr))
    clean_rows = model.predict(cleans)
    mutated = sum(model.decide(c, row, thr) != c for c, row in zip(cleans, clean_rows))
    rows = model.predict([r["obfuscated"] for r in records])
    result = {"threshold": thr, "dev": dev_acc.row(), "clean_mutation": mutated / len(cleans),
              "eval": evaluate(model, records, rows, thr), "eval_no_abstain": evaluate(model, records, rows, 0.0),
              "history": model.history}
    ext = result["eval"]["axis"].get("external/target", {}).get("cer_gain", 0.0)
    print(f"[done] {tag} thr={thr:.4f} dev_gain={result['dev']['cer_gain']:.4f} "
          f"clean_mut={result['clean_mutation']:.4f} ext_gain={ext:.4f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="p04.3-edit-tagger")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    defaults = EditTagger()
    cfg = {"epochs": args.epochs, "seeds": args.seeds, "device": args.device, "benign_budget": BENIGN_BUDGET,
           "model": {k: getattr(defaults, k) for k in ("embed_dim", "jamo_dim", "hidden", "layers", "dropout",
                                                       "batch_size", "lr", "weight_decay", "max_len")}}
    run = R.Run.start(run_id=args.run_id, experiment=EXPERIMENT, config=cfg)
    for name in ("train.jsonl", "dev.jsonl"):
        run.add_input(f"p04_train_v1/{name}", TRAIN / name)
    run.add_input("evalset_v1/records.jsonl", EVALSET / "records.jsonl")

    train, dev, records = _load(TRAIN / "train.jsonl"), _load(TRAIN / "dev.jsonl"), _load(EVALSET / "records.jsonl")
    cleans = [s.text for s in V.load_sources()]
    results = {f"{'phon' if ph else 'base'}-s{s}": train_one(ph, s, train, dev, records, cleans, args)
               for ph in (False, True) for s in args.seeds}

    def mean(name: str, getter) -> float:
        return float(np.mean([getter(r) for k, r in results.items() if k.startswith(name + "-")]))

    summary = {}
    for name in ("base", "phon"):
        types_ok = [t for t in sorted(TARGET)
                    if mean(name, lambda r: r["eval"]["type"].get(f"{t}/compose_auto", {}).get("cer_gain", 0)) >= GAIN_BAR]
        s = {"dev_gain": mean(name, lambda r: r["dev"]["cer_gain"]),
             "clean_mutation": mean(name, lambda r: r["clean_mutation"]),
             "external_gain": mean(name, lambda r: r["eval"]["axis"].get("external/target", {}).get("cer_gain", 0)),
             "types_ok": types_ok}
        s["checks"] = {"clean_mutation": s["clean_mutation"] <= BENIGN_BUDGET,
                       "external_gain": s["external_gain"] >= GAIN_BAR, "types": len(types_ok) >= TYPES_BAR}
        summary[name] = s
    chosen = max(summary, key=lambda k: summary[k]["dev_gain"])
    metrics = {"runs": results, "summary": summary, "chosen": chosen,
               "status": "pass" if all(summary[chosen]["checks"].values()) else "fail"}

    def pct(x):
        return f"{x * 100:.1f}"

    first = results[f"{chosen}-s{args.seeds[0]}"]
    lines = ["# P04.3 글자 편집 태거", "", f"설정별 시드 {len(args.seeds)}개 평균. 선택(dev 기준): **{chosen}**", "",
             "| 설정 | dev CER 개선 | 원문 695 오변경 | external target 개선 | compose 자동형 2%p 넘은 유형 |",
             "|---|---|---|---|---|"]
    for name, s in summary.items():
        lines.append(f"| {name} | {pct(s['dev_gain'])} | {pct(s['clean_mutation'])}% | {pct(s['external_gain'])} | "
                     f"{len(s['types_ok'])}/9 |")
    lines += ["", f"## 축별 ({chosen}, 시드 평균)", "",
              "| 축/범위 | 문장 | 원래 CER | 개선(기권) | 정확 복원 | ERR(글자) | 개선(기권 없음) |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted(first["eval"]["axis"], key=lambda k: (AXES.index(k.split("/")[0]), k)):
        def g(f, src="eval"):
            return mean(chosen, lambda r: r[src]["axis"][k][f])
        lines.append(f"| {k} | {first['eval']['axis'][k]['n']} | {pct(g('raw_cer'))} | {pct(g('cer_gain'))} | "
                     f"{pct(g('exact'))}% | {pct(g('err'))}% | {pct(g('cer_gain', 'eval_no_abstain'))} |")
    lines += ["", f"## 유형별 CER 개선 ({chosen}, target 레코드, 시드 평균)", "",
              "| 유형 | single | compose 사람형 | compose 자동형 | external |", "|---|---|---|---|---|"]
    for t in T.TYPES:
        if t.id not in TARGET:
            continue
        def cell(a):
            return pct(mean(chosen, lambda r: r["eval"]["type"].get(f"{t.id}/{a}", {}).get("cer_gain", float("nan"))))
        lines.append(f"| {t.name} | {cell('single')} | {cell('compose_human')} | {cell('compose_auto')} | {cell('external')} |")
    lines += ["", f"판정: {metrics['status']} — {summary[chosen]['checks']}"]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={"external_gain": summary[chosen]["external_gain"],
                                  "clean_mutation": summary[chosen]["clean_mutation"],
                                  "types_ok": len(summary[chosen]["types_ok"])})
    print("\n".join(lines))
    print(f"\n[P04.3] {metrics['status']} -> {run.directory}")


if __name__ == "__main__":
    main()
