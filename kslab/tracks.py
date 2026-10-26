"""실험 트랙 — 복원기 실험과 가드레일 실험을 폴더째 나눈다.

    restoration/   복원기(난독화 → 원문 후보) ML 실험
    guardrail/     가드레일 벤치마크 실험

트랙마다 `exp/`(러너), `reports/`, `baselines/`, `results/`(커밋 제외),
`EXPERIMENTS.md`(실험 대장)를 따로 갖는다. 노트·리뷰 게이트는 `private/<트랙>/notes/`(커밋 제외)에 둔다. 리뷰 게이트도 트랙별이다 —
가드레일 단계가 리뷰를 기다려도 복원기 단계는 막히지 않는다.

라이브러리(`kslab/`), 환경(`env.sh`), `data/`(코퍼스·모델 캐시·MLflow)는 루트에서 공유한다.

어느 트랙인지는 러너 패키지가 정한다. `restoration/exp/__init__.py`가 `use("restoration")`을
부르므로 `python -m restoration.exp.run_...`으로 실행하면 자동으로 맞춰진다.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ("restoration", "guardrail")

_current: str | None = None


def use(name: str) -> None:
    """이 프로세스의 트랙을 정한다. 러너 패키지의 `__init__`이 부른다."""
    global _current
    if name not in TRACKS:
        raise ValueError(f"트랙은 {TRACKS} 중 하나여야 합니다: {name!r}")
    _current = name


def current(required: bool = True) -> str | None:
    """지금 트랙. `KSLAB_TRACK` 환경변수가 러너 패키지 설정보다 우선한다."""
    name = os.environ.get("KSLAB_TRACK") or _current
    if name is None and required:
        raise RuntimeError(
            "실험 트랙이 정해지지 않았습니다. 러너를 'python -m restoration.exp.<러너>' 또는 "
            "'python -m guardrail.exp.<러너>' 로 실행하세요."
        )
    if name is not None and name not in TRACKS:
        raise ValueError(f"트랙은 {TRACKS} 중 하나여야 합니다: {name!r}")
    return name


def track_dir(name: str | None = None) -> Path:
    return ROOT / (name or current())


def results_dir(name: str | None = None) -> Path:
    return track_dir(name) / "results"


def baselines_dir(name: str | None = None) -> Path:
    return track_dir(name) / "baselines"


def reports_dir(name: str | None = None) -> Path:
    return track_dir(name) / "reports"


def notes_dir(name: str | None = None) -> Path:
    return ROOT / "private" / (name or current()) / "notes"


def ledger(name: str | None = None) -> Path:
    return track_dir(name) / "EXPERIMENTS.md"


def module(name: str | None = None) -> str:
    """안내 문구에 쓰는 러너 패키지 이름 (`restoration.exp`)."""
    return f"{name or current()}.exp"
