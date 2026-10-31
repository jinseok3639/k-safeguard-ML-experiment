"""p05 복원기 — 팀 P04 정규화기를 필터 앞에 붙인다. `RESTORERS[이름]() -> (texts -> texts)`.

  p04.3b-ft-s0  P04.3b KcELECTRA 전체 미세조정 편집 태거, 시드 0(팀에서 받은 체크포인트
                `data/models/p04.3b-kcelectra-ft-s0.pt`). 체크포인트는 `weights_only=True`로 연다(가중치·설정·어휘만
                들어 있다). 인코더 구조는 beomi/KcELECTRA-base-v2022@8728557 스냅샷에서 만든다.
                기권 임계값: P04 판정과 같은 규칙으로, 평가셋 v1 원문 695문장(505 시드 + 공개 정상 190, `data/clean.jsonl`)의 완화 기준
                오변경(문장부호·공백 무시)이 1% 이하가 되는 가장 낮은 값. 욕설 평가 데이터로는 맞추지 않는다
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kslab.record import DATA_DIR

CKPT = DATA_DIR / "models" / "p04.3b-kcelectra-ft-s0.pt"
CKPT_SHA256 = "3f41ec1e643ecda1eaa1f1488dd01bc3073c7147e13f9c3b51aa3ec8b4022dd1"
BENIGN_BUDGET = 0.01
CLEANS = DATA_DIR / "clean.jsonl"
INFO: dict = {}   # 마지막으로 불러온 복원기의 임계값·오변경 등(러너가 기록한다)


def load_p04_3b(device: str = "cuda"):
    import torch
    from kslab.models import device as _device
    from kslab.models.pretrained_tagger import PretrainedEditTagger, encoder_path

    if hashlib.sha256(CKPT.read_bytes()).hexdigest() != CKPT_SHA256:
        raise SystemExit(f"체크포인트 해시가 기록과 다릅니다: {CKPT}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    cfg = dict(ck["config"], encoder=encoder_path())
    model = PretrainedEditTagger(**cfg, device=device)
    model.vocab, model.history = ck["vocab"], ck["history"]
    model.net = model._build_net()
    model.net.load_state_dict(ck["state"])
    model.net.to(_device.resolve(device))
    return model


def calibrate(model) -> tuple[float, float]:
    import json
    from restoration.exp.run_p04_3b_pretrained_tagger import calibrate as _cal, mutation

    # 평가셋 v1 원문 695(`kslab.evalset.load_sources`와 같은 문장). 이 PC에는 공개 원문 파일 대신 평가셋 인계본의
    # clean 레코드를 둔 `data/clean.jsonl`이 있다
    with open(CLEANS, encoding="utf-8") as f:
        cleans = [json.loads(line)["clean"] for line in f if line.strip()]
    if len(cleans) != 695:
        raise SystemExit(f"{CLEANS}: 원문 695개가 아닙니다({len(cleans)})")
    rows = model.predict(cleans, batch_size=64)
    thr = _cal(model, cleans, rows, BENIGN_BUDGET)
    return thr, mutation(model, cleans, rows, thr)[0]


def p04_3b():
    model = load_p04_3b()
    thr, mut = calibrate(model)
    INFO.update({"checkpoint": CKPT.name, "checkpoint_sha256": CKPT_SHA256, "threshold": thr,
                 "evalset_clean_mutation": mut, "benign_budget": BENIGN_BUDGET})
    print(f"  복원기 p04.3b-ft-s0: 임계값 {thr:.4f}, 평가셋 원문 오변경 {100 * mut:.2f}%", flush=True)

    def restore(texts: list[str]) -> list[str]:
        rows = model.predict(texts, batch_size=128)
        return [model.decide(t, r, thr) for t, r in zip(texts, rows)]
    return restore


RESTORERS = {"p04.3b-ft-s0": p04_3b}
