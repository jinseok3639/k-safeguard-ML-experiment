"""학습·추론 장치 선택.

모델의 기본값은 `"cpu"`다. E0~E10은 CPU에서 돌았고 BiLSTM 체크포인트 재개 테스트는 CPU에서
비트 단위 일치를 요구하므로, GPU는 러너가 `--device`로 명시해서 켠다.

GPU 결과는 CPU와 소수점 수준에서 다를 수 있다(cuDNN LSTM 역전파의 부동소수 합산 순서).
장치는 `fit.*` 하이퍼파라미터로 MLflow에 남는다.
"""

from __future__ import annotations


def resolve(spec: str):
    """`"cpu"`·`"cuda"`·`"cuda:0"`·`"auto"` -> `torch.device`. 없는 GPU를 달라고 하면 멈춘다."""
    import torch

    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={spec!r} 를 요청했지만 CUDA를 쓸 수 없습니다. "
            "torch CUDA 빌드와 드라이버(WSL: /usr/lib/wsl/lib/nvidia-smi)를 확인하세요."
        )
    return torch.device(spec)
