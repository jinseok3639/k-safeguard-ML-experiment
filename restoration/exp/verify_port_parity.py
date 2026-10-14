"""이식 검증 — 저장소로 옮긴 ONNX 추론 경로가 샌드박스 torch 경로와 같은가.

저장소 쪽(`k_safeguard.jamo_slots` + `providers.ml_restore`)은 자모 인코딩을 순수
python으로 새로 짰다. 인덱스가 하나만 어긋나도 조용히 틀린 결과를 내므로, 실제
가중치로 두 경로를 직접 대조한다.

세 가지를 본다.

1. **후보 위치·창 인코딩** — 두 경로가 고르는 자리와 그 주변 창이 같은가 (완전 일치 요구)
2. **복원 출력** — 같은 임계값에서 나온 문장이 문자 단위로 같은가
3. **승격 지표** — benign 오변경과 CER 감소가 E8 수치를 재현하는가

2번은 float32 오차 때문에 완전 일치가 **보장되지 않는다.** ONNX와 torch의 확률 차이는
1e-6 수준인데, tensify의 임계값 0.999999는 그 오차보다 겨우 한 자릿수 큰 곳에 있어
경계에 걸친 자리가 뒤집힌다. 그래서 2번은 불일치 수를 보고만 하고, 판정은 3번으로 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "k-safeguard" / "src"))
sys.dont_write_bytecode = True      # 저장소에 .pyc 를 남기지 않는다

from kslab import features as F
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P
from kslab.models.mlp import MlpSlotRestorer, _SlotNet

from k_safeguard import jamo_slots as J
from k_safeguard.providers.ml_restore import MlRestoreProvider

# E8이 두 기준을 동시에 넘긴 지점. 이식본이 재현해야 하는 값이다.
E8_REFERENCE = {
    "tensify": {"threshold": 0.999999, "mutation": 0.0049, "cer_reduction": 0.1189},
    "liaison": {"threshold": 0.99, "mutation": 0.0049, "cer_reduction": 0.0452},
    "jongseong_cram": {"threshold": 0.99, "mutation": 0.0000, "cer_reduction": 0.0512},
}
MUTATION_BAR = 0.01
MIN_CER_REDUCTION = 0.02
TOLERANCE = 0.005      # 지표 재현 허용 오차 (문장 1개 = 0.49%p 해상도)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="가중치 디렉터리")
    parser.add_argument("--gen-seed", type=int, default=1234)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser.parse_args()


def _load_torch_model(weights: Path, technique: str) -> MlpSlotRestorer:
    """샌드박스 쪽 기준 모델. export 때 저장한 state_dict를 그대로 되살린다."""
    import torch

    checkpoint = torch.load(
        weights / f"{technique}.pt", map_location="cpu", weights_only=False
    )
    model = MlpSlotRestorer(
        technique,
        window=checkpoint["window"],
        embed_dim=checkpoint["embed_dim"],
        hidden=checkpoint["hidden"],
        dropout=checkpoint["dropout"],
    )
    model.vocab.__dict__.update(checkpoint["vocab"])
    model.net = _SlotNet(
        len(model.vocab),
        2 * checkpoint["window"] + 1,
        checkpoint["embed_dim"],
        checkpoint["hidden"],
        checkpoint["unknown_slots"],
        checkpoint["dropout"],
    )
    model.net.load_state_dict(checkpoint["model_state"])
    model.net.eval()
    return model


def _check_encoding(texts: list[str], technique: str) -> dict:
    """1. 후보 위치와 창 인코딩이 같은지. 여기는 완전 일치여야 한다."""
    positions = windows = slots = 0
    for text in texts:
        if F.candidate_positions(text, technique) != J.candidate_positions(text, technique):
            positions += 1
        examples = F.extract_examples(
            [P.Pair("_", "_", technique, 1.0, text, text)], window=4
        )
        sites = J.extract_sites(text, technique, window=4)
        if len(examples) != len(sites):
            windows += 1
            continue
        for example, site in zip(examples, sites):
            if example.window_chars != site.window_chars:
                windows += 1
            elif abs(example.rel_position - site.rel_position) > 1e-9:
                windows += 1
            if example.input_slots != site.input_slots:
                slots += 1
    return {"position_mismatch": positions, "window_mismatch": windows, "slot_mismatch": slots}


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    seeds = P.load_seeds()
    clean = [seed.text for seed in seeds]
    benign = [seed.text for seed in seeds if seed.label == "benign"]
    clusters = [seed.seed_id for seed in seeds]

    provider = MlRestoreProvider.from_directory(weights)
    failures: list[str] = []

    for technique, reference in E8_REFERENCE.items():
        threshold = reference["threshold"]
        torch_model = _load_torch_model(weights, technique)
        restorer = provider._restorers[technique]

        obfuscated = [
            O.TRANSFORMS[technique](text, 1.0, seed=args.gen_seed) for text in clean
        ]
        changed = [i for i, (o, c) in enumerate(zip(obfuscated, clean)) if o != c]
        corpus = clean + obfuscated

        print(f"\n=== {technique} (threshold {threshold}) ===")

        # 1. 인코딩 일치 — 완전 일치 요구
        encoding = _check_encoding(corpus, technique)
        ok = not any(encoding.values())
        print(f"  1. 인코딩 일치      {'통과' if ok else '실패'}  {encoding}")
        if not ok:
            failures.append(f"{technique}: 인코딩 불일치 {encoding}")

        # 2. 복원 출력 일치 — float 오차로 경계 자리가 뒤집힐 수 있다
        torch_out = [row[0] for row in torch_model.restore_many(corpus, threshold)]
        onnx_out = [restorer.restore(text)[0] for text in corpus]
        mismatch = sum(1 for a, b in zip(torch_out, onnx_out) if a != b)
        print(
            f"  2. 복원 출력 일치   {len(corpus) - mismatch}/{len(corpus)} "
            f"(불일치 {mismatch}, float32 경계 오차 허용)"
        )

        # 3. 승격 지표 재현 — 이게 판정 기준이다
        benign_out = [restorer.restore(text)[0] for text in benign]
        restored = {i: onnx_out[len(clean) + i] for i in changed}
        gain = [
            (
                clusters[i],
                M.char_error_rate(clean[i], obfuscated[i])
                - M.char_error_rate(clean[i], restored[i]),
            )
            for i in changed
        ]
        mutation = M.clean_mutation_rate(benign, benign_out)["sentence_rate"]
        reduction = M.summarize_observations(
            gain, args.bootstrap_samples, args.random_seed
        )["cluster_balanced_estimate"] or 0.0

        mutation_ok = abs(mutation - reference["mutation"]) <= TOLERANCE
        reduction_ok = abs(reduction - reference["cer_reduction"]) <= TOLERANCE
        gate_ok = mutation <= MUTATION_BAR and reduction >= MIN_CER_REDUCTION
        print(
            f"  3. benign 오변경    {mutation:.2%} "
            f"(E8 {reference['mutation']:.2%}) {'통과' if mutation_ok else '실패'}"
        )
        print(
            f"     CER 감소         {reduction * 100:+.2f}%p "
            f"(E8 {reference['cer_reduction'] * 100:+.2f}%p) {'통과' if reduction_ok else '실패'}"
        )
        print(f"     E8 두 기준 동시  {'통과' if gate_ok else '실패'}")
        if not mutation_ok:
            failures.append(f"{technique}: 오변경 {mutation:.4f} != E8 {reference['mutation']}")
        if not reduction_ok:
            failures.append(
                f"{technique}: CER감소 {reduction:.4f} != E8 {reference['cer_reduction']}"
            )
        if not gate_ok:
            failures.append(f"{technique}: E8 두 기준 동시 통과 실패")

    print("\n" + "=" * 60)
    if failures:
        print(f"검증 실패 {len(failures)}건:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("검증 통과 — 이식된 ONNX 경로가 E8 수치를 재현합니다.")


if __name__ == "__main__":
    main()
