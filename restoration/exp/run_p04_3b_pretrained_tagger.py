"""P04.3b·P04.3c — 사전학습 인코더(KcELECTRA) 편집 태거 (p04.3b-kcelectra-ft, p04.3c-kcelectra-lora).

P04.3 BiLSTM 태거가 평가셋 원문을 23.5% 바꾼 원인이 학습 원문(위키·뉴스)에 없는 정상 어휘였다. 댓글로 사전학습한
KcELECTRA를 얹어 그 어휘 지식이 오변경을 줄이는지 본다. `--variant ft`는 전체 미세조정, `lora`는 LoRA.

같은 run 안에서 P04.3에 저장한 BiLSTM 모델(시드 0·1·2)도 같은 평가 규칙으로 다시 재서 나란히 둔다.

오변경 기준(실행 전 고정): 문장부호·공백만 다른 경우는 오변경으로 세지 않는다. 기권 임계값도 같은 기준으로 dev의 손대지
않은 쌍에서 1% 이하가 되게 정한다. 글자 하나라도 다르면 오변경으로 세는 엄격 기준도 함께 남긴다.

판정(설정별): 평가셋 원문 695 오변경 ≤ 1% / external target CER 개선 ≥ 2%p / compose 자동형 target에서 9유형 중
7개 이상 CER 개선 ≥ 2%p.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import evalset as V
from kslab import record as R
from kslab import taxonomy as T
from kslab.models.edit_tagger import EditTagger
from kslab.models.pretrained_tagger import KCELECTRA, KCELECTRA_REVISION, PretrainedEditTagger, encoder_path
from restoration.exp.run_p04_3_edit_tagger import (AXES, BENIGN_BUDGET, EVALSET, GAIN_BAR, MODELS, TARGET, TRAIN,
                                                   TYPES_BAR, Acc, _load, evaluate)

EXPERIMENT = "P04_normalizer"
BILSTM_RUN = "p04.3-edit-tagger"
VARIANTS = {
    "ft": {"lora": False, "encoder_lr": 3e-5},
    "lora": {"lora": True, "encoder_lr": 5e-4},
}
LATENCY_SAMPLE = 200


def lenient_key(text: str) -> list[str]:
    """문장부호를 지우고 공백으로 나눈 어절. 이것이 같으면 오변경으로 세지 않는다."""
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P")).split()


def mutation(model: EditTagger, texts: list[str], rows: list, thr: float) -> tuple[float, float]:
    outs = [model.decide(t, row, thr) for t, row in zip(texts, rows)]
    strict = sum(o != t for o, t in zip(outs, texts)) / len(texts)
    lenient = sum(lenient_key(o) != lenient_key(t) for o, t in zip(outs, texts)) / len(texts)
    return lenient, strict


def calibrate(model: EditTagger, texts: list[str], rows: list, budget: float) -> float:
    """완화 기준 오변경이 `budget` 이하가 되는 가장 낮은 임계값(이분 탐색, 임계값이 오르면 편집은 줄기만 한다)."""
    if mutation(model, texts, rows, 0.0)[0] <= budget:
        return 0.0
    lo, hi = 0.0, 1.0 + 1e-6
    for _ in range(30):
        mid = (lo + hi) / 2
        if mutation(model, texts, rows, mid)[0] <= budget:
            hi = mid
        else:
            lo = mid
    return hi


def latency(model: EditTagger, texts: list[str], device: str) -> dict:
    """문장 하나씩 넣었을 때 p50·p95(ms)."""
    import torch

    model.net.to(device)
    model.predict(texts[:5], batch_size=1)          # 예열
    times = []
    for t in texts:
        start = time.perf_counter()
        model.predict([t], batch_size=1)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}


def assess(model: EditTagger, dev: list[dict], records: list[dict], cleans: list[str], batch_size: int) -> dict:
    dev_id = [p["clean"] for p in dev if not p["recipe"]]
    dev_obf = [p for p in dev if p["recipe"]]
    thr = calibrate(model, dev_id, model.predict(dev_id, batch_size=batch_size), BENIGN_BUDGET)
    dev_acc = Acc()
    for p, row in zip(dev_obf, model.predict([p["obfuscated"] for p in dev_obf], batch_size=batch_size)):
        dev_acc.add(p["clean"], p["obfuscated"], model.decide(p["obfuscated"], row, thr))
    lenient, strict = mutation(model, cleans, model.predict(cleans, batch_size=batch_size), thr)
    rows = model.predict([r["obfuscated"] for r in records], batch_size=batch_size)
    return {"threshold": thr, "dev": dev_acc.row(), "clean_mutation": lenient, "clean_mutation_strict": strict,
            "eval": evaluate(model, records, rows, thr), "eval_no_abstain": evaluate(model, records, rows, 0.0)}


def summarize(result: dict) -> dict:
    types_ok = [t for t in sorted(TARGET)
                if result["eval"]["type"].get(f"{t}/compose_auto", {}).get("cer_gain", 0) >= GAIN_BAR]
    ext = result["eval"]["axis"].get("external/target", {}).get("cer_gain", 0.0)
    return {"dev_gain": result["dev"]["cer_gain"], "clean_mutation": result["clean_mutation"],
            "clean_mutation_strict": result["clean_mutation_strict"], "external_gain": ext, "types_ok": types_ok,
            "checks": {"clean_mutation": result["clean_mutation"] <= BENIGN_BUDGET, "external_gain": ext >= GAIN_BAR,
                       "types": len(types_ok) >= TYPES_BAR}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ref", action="store_true", help="BiLSTM 비교 재측정을 건너뜀(시드 반복 run)")
    args = parser.parse_args()
    run_id = args.run_id or {"ft": "p04.3b-kcelectra-ft", "lora": "p04.3c-kcelectra-lora"}[args.variant]

    spec = VARIANTS[args.variant]
    model = PretrainedEditTagger(encoder=encoder_path(), device=args.device, epochs=args.epochs, amp=True,
                                 batch_size=32, batch_chars=32 * 80, random_state=2026 + args.seed, **spec)
    cfg = {"variant": args.variant, "encoder": KCELECTRA, "encoder_revision": KCELECTRA_REVISION, "seed": args.seed,
           "benign_budget": BENIGN_BUDGET, "mutation_rule": "문장부호·공백 무시(엄격 기준 병기)",
           "model": {k: getattr(model, k) for k in PretrainedEditTagger.CONFIG_KEYS if k != "encoder"} |
                    {"epochs": args.epochs, "batch_size": model.batch_size, "batch_chars": model.batch_chars,
                     "lr": model.lr, "weight_decay": model.weight_decay, "amp": "bf16"}}
    run = R.Run.start(run_id=run_id, experiment=EXPERIMENT, config=cfg)
    for name in ("train.jsonl", "dev.jsonl"):
        run.add_input(f"p04_train_v1/{name}", TRAIN / name)
    run.add_input("evalset_v1/records.jsonl", EVALSET / "records.jsonl")

    train, dev, records = _load(TRAIN / "train.jsonl"), _load(TRAIN / "dev.jsonl"), _load(EVALSET / "records.jsonl")
    cleans = [s.text for s in V.load_sources()]

    print(f"[fit] kcelectra-{args.variant}", flush=True)
    start = time.time()
    model.fit([p["obfuscated"] for p in train], [p["clean"] for p in train])
    fit_minutes = (time.time() - start) / 60
    trainable, total = model.trainable_parameters()
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODELS / f"{run_id}-s{args.seed}.pt")
    main_result = assess(model, dev, records, cleans, batch_size=32)
    sample = [r["obfuscated"] for r in records[:: max(1, len(records) // LATENCY_SAMPLE)]][:LATENCY_SAMPLE]
    main_result["latency"] = {"cuda": latency(model, sample, "cuda"), "cpu": latency(model, sample, "cpu")}
    main_result.update({"history": model.history, "fit_minutes": fit_minutes,
                        "trainable_params": trainable, "total_params": total})
    print(f"[done] kcelectra-{args.variant} thr={main_result['threshold']:.4f} "
          f"clean_mut={main_result['clean_mutation']:.4f} ext={summarize(main_result)['external_gain']:.4f}", flush=True)
    del model

    bilstm = {}
    for seed in (() if args.no_ref else (0, 1, 2)):
        path = MODELS / f"{BILSTM_RUN}-base-s{seed}.pt"
        ref = EditTagger.load(path, device=args.device)
        bilstm[f"s{seed}"] = assess(ref, dev, records, cleans, batch_size=128)
        if seed == 0:
            bilstm["latency"] = {"cuda": latency(ref, sample, "cuda"), "cpu": latency(ref, sample, "cpu")}
        print(f"[ref] bilstm-s{seed} clean_mut={bilstm[f's{seed}']['clean_mutation']:.4f}", flush=True)

    summary = {"kcelectra": summarize(main_result)}
    if bilstm:
        ref_sum = [summarize(bilstm[f"s{s}"]) for s in (0, 1, 2)]
        summary["bilstm_mean"] = ({k: float(np.mean([s[k] for s in ref_sum]))
                                   for k in ("dev_gain", "clean_mutation", "clean_mutation_strict", "external_gain")}
                                  | {"types_ok_counts": [len(s["types_ok"]) for s in ref_sum]})
    metrics = {"kcelectra": main_result, "bilstm": bilstm, "summary": summary,
               "status": "pass" if all(summary["kcelectra"]["checks"].values()) else "fail"}

    def pct(x):
        return f"{x * 100:.1f}"

    k = summary["kcelectra"]
    lat_k = main_result["latency"]
    lines = [f"# {run_id} — KcELECTRA {args.variant} vs P04.3 BiLSTM", "",
             f"학습 {fit_minutes:.0f}분, 학습 파라미터 {trainable:,} / 전체 {total:,}", "",
             "| 모델 | dev CER 개선 | 원문 695 오변경(문장부호 무시) | 엄격 | external target 개선 | compose 자동형 2%p 넘은 유형 | 지연 p95 GPU / CPU |",
             "|---|---|---|---|---|---|---|",
             f"| KcELECTRA {args.variant} | {pct(k['dev_gain'])} | {pct(k['clean_mutation'])}% | {pct(k['clean_mutation_strict'])}% | "
             f"{pct(k['external_gain'])} | {len(k['types_ok'])}/9 | {lat_k['cuda']['p95_ms']:.0f} / {lat_k['cpu']['p95_ms']:.0f} ms |"]
    if bilstm:
        b, lat_b = summary["bilstm_mean"], bilstm["latency"]
        lines.append(f"| BiLSTM (시드 3 평균) | {pct(b['dev_gain'])} | {pct(b['clean_mutation'])}% | {pct(b['clean_mutation_strict'])}% | "
                     f"{pct(b['external_gain'])} | {b['types_ok_counts']} | {lat_b['cuda']['p95_ms']:.0f} / {lat_b['cpu']['p95_ms']:.0f} ms |")
    lines += ["", f"## 축별 (KcELECTRA {args.variant})", "",
             "| 축/범위 | 문장 | 원래 CER | 개선(기권) | 정확 복원 | ERR(글자) | 개선(기권 없음) | BiLSTM s0 개선 |",
             "|---|---|---|---|---|---|---|---|"]
    ev, ev0 = main_result["eval"]["axis"], main_result["eval_no_abstain"]["axis"]
    evb = bilstm["s0"]["eval"]["axis"] if bilstm else {}
    for key in sorted(ev, key=lambda x: (AXES.index(x.split("/")[0]), x)):
        x = ev[key]
        lines.append(f"| {key} | {x['n']} | {pct(x['raw_cer'])} | {pct(x['cer_gain'])} | {pct(x['exact'])}% | "
                     f"{pct(x['err'])}% | {pct(ev0[key]['cer_gain'])} | {pct(evb.get(key, {}).get('cer_gain', float('nan')))} |")
    lines += ["", "## 유형별 CER 개선 (target 레코드)", "",
              "| 유형 | single | compose 사람형 | compose 자동형 | external | BiLSTM s0 external |", "|---|---|---|---|---|---|"]
    ty, tyb = main_result["eval"]["type"], (bilstm["s0"]["eval"]["type"] if bilstm else {})
    for t in T.TYPES:
        if t.id in TARGET:
            cell = lambda d, a: pct(d.get(f"{t.id}/{a}", {}).get("cer_gain", float("nan")))
            lines.append(f"| {t.name} | {cell(ty, 'single')} | {cell(ty, 'compose_human')} | {cell(ty, 'compose_auto')} | "
                         f"{cell(ty, 'external')} | {cell(tyb, 'external')} |")
    lines += ["", f"판정: {metrics['status']} — {summary['kcelectra']['checks']}"]
    run.write_text("report.md", "\n".join(lines) + "\n")
    run.finish(metrics, headline={"external_gain": k["external_gain"], "clean_mutation": k["clean_mutation"],
                                  "types_ok": len(k["types_ok"]), "cpu_p95_ms": lat_k["cpu"]["p95_ms"]})
    print("\n".join(lines))
    print(f"\n[{run_id}] {metrics['status']} -> {run.directory}")


if __name__ == "__main__":
    main()
