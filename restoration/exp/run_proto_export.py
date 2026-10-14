"""프로토타입 배포용 — E8에서 통과 지점으로 확인된 설정을 다시 학습하고, 이번엔
가중치를 남긴다(ONNX + torch state_dict). E8 러너는 학습 후 모델을 버렸기 때문에
같은 설정을 다시 돌려야 한다(3만문장 3기법 합쳐 약 8분, `e8-mlp-30k` manifest 참고).

opt-in provider로 배포하는 것이 목표이므로 승격 5기준을 다시 확인하지 않는다 —
E8이 이미 찾은 임계값에서 실제 Gateway 계약이 여전히 성립하는지, ONNX가 torch와
일치하는지만 재확인하고 내보낸다. 성능 개선은 이후 과제다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import corpus as C
from kslab import features as F
from kslab import metrics as M
from kslab import pairs as P
from kslab import provider as PROV
from kslab import record as R
from kslab.models.mlp import MlpSlotRestorer

# E8(e8-mlp-30k)이 두 기준(mutation<=1%, CER감소>=2%p)을 동시에 넘긴 지점.
# EXPERIMENTS.md / reports/E8_ABSTENTION.md 참고.
DEFAULT_THRESHOLDS = {
    "tensify": 0.999999,
    "liaison": 0.99,
    "jongseong_cram": 0.99,
}
INTENSITIES = (0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--corpus-size", type=int, default=30_000)
    parser.add_argument("--wiki-ratio", type=float, default=0.5)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--max-slot-examples", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument(
        "--techniques", nargs="*", default=list(DEFAULT_THRESHOLDS),
    )
    return parser.parse_args()


def _export_onnx(model: MlpSlotRestorer, path: Path, examples) -> dict:
    """E7(`run_e7_provider.py`)에서 검증한 것과 동일한 export 경로."""
    import numpy as np
    import torch

    chars, pos, slots = F.jamo_window_arrays(examples[:8], model.vocab)
    sample = (torch.from_numpy(chars), torch.from_numpy(pos), torch.from_numpy(slots))
    slot_names = [str(s) for s in model.unknown_slots]

    class _TupleWrapper(torch.nn.Module):
        def __init__(self, net, slots):
            super().__init__()
            self.net = net
            self.slots = list(slots)

        def forward(self, chars, pos, slots):
            out = self.net(chars, pos, slots)
            return tuple(out[s] for s in self.slots)

    exportable = _TupleWrapper(model.net, model.unknown_slots).eval()
    torch.onnx.export(
        exportable, sample, str(path),
        input_names=["chars", "pos", "slots"],
        output_names=[f"logits_{name}" for name in slot_names],
        dynamic_axes={
            "chars": {0: "batch"}, "pos": {0: "batch"}, "slots": {0: "batch"},
            **{f"logits_{name}": {0: "batch"} for name in slot_names},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"chars": chars, "pos": pos, "slots": slots})
    with torch.no_grad():
        reference = model.net(*sample)
    max_diff = 0.0
    for index, slot in enumerate(model.unknown_slots):
        max_diff = max(max_diff, float(np.abs(outputs[index] - reference[slot].numpy()).max()))
    return {
        "path": path.name, "bytes": path.stat().st_size,
        "max_abs_diff_vs_torch": max_diff, "matches_torch": max_diff < 1e-4,
    }


def main() -> None:
    import torch

    args = parse_args()
    seeds = P.load_seeds()
    thresholds = {t: DEFAULT_THRESHOLDS[t] for t in args.techniques}

    run = R.Run.start(
        run_id=args.run_id,
        experiment="proto_export",
        config={
            "corpus_size": args.corpus_size, "techniques": args.techniques,
            "thresholds": thresholds, "corpus_license": C.LICENSE_NOTE,
            "purpose": "opt-in 프로토타입 배포용 가중치 export. 승격 5기준 재확인 아님.",
        },
    )
    for name in ("attacks.jsonl", "benign.jsonl"):
        run.add_input(name, P.SEED_DIR / name)

    corpus_path = C.build_corpus(args.corpus_size, args.wiki_ratio)
    run.add_input("corpus", corpus_path)
    rows = C.load_corpus(corpus_path)
    train_items, _ = C.split_by_hash(C.corpus_as_items(rows), test_fraction=0.2)

    weights_dir = run.directory / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, MlpSlotRestorer] = {}
    manifest: dict = {
        "format_version": "1",
        "run_id": args.run_id,
        "provenance": {
            "corpus_size": args.corpus_size,
            "corpus_license": C.LICENSE_NOTE,
            "epochs": args.epochs,
            "random_seed": args.random_seed,
            "gen_seed": args.gen_seed,
            "regenerate_with": (
                f"python -m restoration.exp.run_proto_export --run-id <새 run-id> "
                f"--corpus-size {args.corpus_size} --epochs {args.epochs}"
            ),
            "thresholds_source": "E8(e8-mlp-30k) — reports/E8_ABSTENTION.md",
        },
        "techniques": {},
    }

    for technique in args.techniques:
        print(f"[proto] {technique} 학습 중 ...", flush=True)
        started = time.time()
        train_pairs = P.make_pairs(
            train_items, [technique], INTENSITIES,
            gen_seed=args.gen_seed, include_identity=True,
        )
        examples = F.subsample(
            F.extract_examples(train_pairs, args.window),
            args.max_slot_examples, args.random_seed,
        )
        model = MlpSlotRestorer(
            technique, window=args.window, random_state=args.random_seed,
            epochs=args.epochs, threads=args.threads,
        ).fit(examples)
        fit_seconds = round(time.time() - started, 1)
        models[technique] = model

        # torch state_dict — 나중에 재학습 없이 재로딩하거나 다른 포맷으로 다시 export할 때 씀.
        state_path = weights_dir / f"{technique}.pt"
        torch.save(
            {
                "technique": technique, "window": model.window,
                "embed_dim": model.embed_dim, "hidden": model.hidden,
                "dropout": model.dropout, "vocab": model.vocab.__dict__,
                "unknown_slots": list(model.unknown_slots),
                "model_state": model.net.state_dict(),
            },
            state_path,
        )

        onnx_path = weights_dir / f"{technique}.onnx"
        onnx_report = _export_onnx(model, onnx_path, examples)

        # ONNX 그래프에는 vocab이 안 들어간다. 저장소 쪽 추론기는 torch 없이 돌아야 하므로
        # 문자→id 사전을 따로 내보낸다. 이게 없으면 .onnx 단독으로는 인코딩을 못 한다.
        vocab_path = weights_dir / f"{technique}.vocab.json"
        vocab_path.write_text(
            json.dumps(model.vocab._ids, ensure_ascii=False, sort_keys=True, indent=0) + "\n",
            encoding="utf-8",
        )

        threshold = thresholds[technique]
        benign = [s.text for s in seeds if s.label == "benign"]
        clean = [s.text for s in seeds]
        clusters = [s.seed_id for s in seeds]
        fn = __import__("kslab.obfuscate", fromlist=["x"]).TRANSFORMS[technique]
        obfuscated = [fn(t, 1.0, seed=args.gen_seed) for t in clean]
        changed = [i for i, (o, c) in enumerate(zip(obfuscated, clean)) if o != c]

        benign_out = [r[0] for r in model.restore_many(benign, threshold)]
        restored = [r[0] for r in model.restore_many(obfuscated, threshold)]
        gain = [
            (clusters[i], M.char_error_rate(clean[i], obfuscated[i]) - M.char_error_rate(clean[i], restored[i]))
            for i in changed
        ]
        mutation = M.clean_mutation_rate(benign, benign_out)
        cer_reduction = M.summarize_observations(gain, 2000, args.random_seed)

        manifest["techniques"][technique] = {
            "fit_seconds": fit_seconds,
            "threshold": threshold,
            "state_dict": state_path.name,
            # 저장소 쪽 로더가 읽는 필드. 이것만으로 추론기를 세울 수 있어야 한다.
            "onnx_file": onnx_path.name,
            "vocab_file": vocab_path.name,
            "window": model.window,
            "unknown_slots": list(model.unknown_slots),
            "vocab_size": len(model.vocab),
            "onnx": onnx_report,
            "sanity_check": {
                "benign_clean_mutation_sentence_rate": mutation["sentence_rate"],
                "cer_reduction": cer_reduction["cluster_balanced_estimate"],
            },
        }
        print(
            f"       thr={threshold} mutation={mutation['sentence_rate']:.2%} "
            f"CER감소={(cer_reduction['cluster_balanced_estimate'] or 0) * 100:+.2f}%p "
            f"ONNX 일치={onnx_report['matches_torch']} ({fit_seconds}s)",
            flush=True,
        )

    print("[proto] 실제 Gateway 계약 재확인 중 ...", flush=True)
    provider = PROV.MlRestoreProvider(models=models, thresholds=thresholds)
    smoke_texts = [s.text for s in seeds[:60]] + [
        __import__("kslab.obfuscate", fromlist=["x"]).TRANSFORMS[t](s.text, 1.0, seed=args.gen_seed)
        for t in args.techniques for s in seeds[:20]
    ]
    contract = PROV.contract_report(provider, smoke_texts)
    manifest["contract"] = contract
    print(f"       contract_ok={contract['contract_ok']} "
          f"views={contract['views_seen']} candidates={contract['candidates_seen']} "
          f"violations={len(contract['violations'])}", flush=True)

    # 가중치 디렉터리 자체가 자기기술적이어야 한다 — 저장소 provider가
    # `MlRestoreProvider.from_directory(weights_dir)` 하나로 세울 수 있게.
    (weights_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run.write_json("weights_manifest.json", manifest)
    run.finish({"techniques": list(manifest["techniques"]), "contract_ok": contract["contract_ok"]})
    print(f"\n[proto] 완료 -> {run.directory}")
    print(f"[proto] 가중치 -> {weights_dir}")


if __name__ == "__main__":
    main()
