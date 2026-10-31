"""p05 비교 대상 욕설 필터. 모두 `flags(texts) -> list[bool]` 하나로 부른다.

  korcen      Tanat05/korcen(키워드 규칙, pip `korcen`). `korcen.check(text)` 그대로
  kor-unsmile smilegate-ai/kor_unsmile(BERT 다중 라벨). 혐오·욕설 라벨 9개(clean 제외) 중 하나라도
              sigmoid 0.5 이상이면 걸림. Curse-detection의 욕설 기준이 차별·비하·성적 표현을 포함해서다
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Filter:
    key: str
    name: str
    version: str
    rule: str
    load: Callable[[], Callable[[list[str]], list[bool]]]


def _korcen():
    from korcen import korcen as K
    return lambda texts: [bool(K.check(t)) for t in texts]


UNSMILE = "smilegate-ai/kor_unsmile"
UNSMILE_REV = "f597423eeff8e6da99cad85cbe3a81adf5225637"


def _unsmile(batch: int = 128):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(UNSMILE, revision=UNSMILE_REV)
    model = AutoModelForSequenceClassification.from_pretrained(UNSMILE, revision=UNSMILE_REV)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    clean = [i for i, name in model.config.id2label.items() if name == "clean"]

    @torch.inference_mode()
    def flags(texts: list[str]) -> list[bool]:
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            prob = torch.sigmoid(model(**enc).logits.float())
            prob[:, clean] = 0
            out += (prob.max(dim=1).values >= 0.5).tolist()
        return out
    return flags


def _korcen_version() -> str:
    from importlib.metadata import version
    return f"korcen=={version('korcen')}"


FILTERS = {
    "korcen": Filter("korcen", "korcen", "", "korcen.check(text)", _korcen),
    "kor-unsmile": Filter("kor-unsmile", UNSMILE, UNSMILE_REV, "혐오·욕설 9라벨 중 max sigmoid >= 0.5", _unsmile),
}


def timed(fn: Callable[[list[str]], list[bool]], texts: list[str]) -> tuple[list[bool], float]:
    t0 = time.perf_counter()
    res = fn(texts)
    return res, time.perf_counter() - t0
