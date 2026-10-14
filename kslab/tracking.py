"""MLflow 기록 — `record.Run`이 남기는 것을 MLflow에도 그대로 싣는다.

러너는 이 모듈을 직접 부르지 않는다. `Run.start()`/`Run.finish()`가 대신 부르므로
러너를 고치지 않아도 전부 기록된다.

무엇이 어디로 가는가:

| MLflow | 내용 |
|---|---|
| experiment | 러너의 `experiment` 이름(`E8_abstention` 등). 스모크 실행은 `_smoke` |
| run name | run-id |
| description | 러너 모듈 docstring — 그 실험의 목적 |
| params | `config`를 평탄화한 것 |
| metrics | `finish(metrics)`의 수치를 `a/b/c` 키로 평탄화한 것 |
| tags | 판정(status), git 커밋·dirty, 실행 명령, 호스트, 패키지 버전 |
| datasets | `add_input()`으로 등록한 입력의 이름·SHA-256·경로 (내용은 싣지 않는다) |
| params (`fit.*`) | 모델이 fit()에서 스스로 올리는 하이퍼파라미터 — 러너에 하드코딩된 값 포함 |
| metrics (`headline/*`) | 러너가 고른 대표 지표 몇 개. run 비교표에서는 이것만 보면 된다 |
| metrics (`train/*/loss`) | epoch별 학습 loss (step 축) |
| artifacts | run 디렉터리의 `.md`·`.json` (report, metrics, manifest, 노트)와 `code.diff`·`pip-freeze.txt` |

`predictions.jsonl`처럼 공격 원문이 든 산출물은 artifact로 올리지 않는다 — `results/`에만
남긴다는 기존 규칙을 MLflow 저장소에도 그대로 적용한다.

`MLFLOW_TRACKING_URI`가 없으면(env 스크립트를 안 거친 테스트 등) 아무것도 하지 않는다.
기록이 실패해도 실험은 멈추지 않는다 — 원본은 언제나 `results/<run-id>/`다.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DIR = ROOT / "data" / "mlflow"
ARTIFACT_DIR = MLFLOW_DIR / "artifacts"

SMOKE_EXPERIMENT = "_smoke"
ARTIFACT_SUFFIXES = (".md", ".json")
# 접미사 규칙 밖에서 이름으로 허용하는 것. run.log 는 러너가 원문 예시를 찍을 수 있어 뺀다.
ARTIFACT_NAMES = ("code.diff", "pip-freeze.txt")
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
_PARAM_LIMIT = 6000
_TAG_LIMIT = 8000
_METRIC_BATCH = 1000
# MLflow metric 이름 규칙: 영숫자, _ - . 공백 : / 만 허용.
_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. :]")
_PARAM_BATCH = 100


def tracking_uri() -> str | None:
    if os.environ.get("KSLAB_MLFLOW", "1") == "0":
        return None
    return os.environ.get("MLFLOW_TRACKING_URI") or None


def enabled() -> bool:
    return tracking_uri() is not None


def assert_store_contained(uri: str) -> None:
    """로컬 저장소라면 샌드박스 안이어야 한다. 원격(http 등) URI는 명시적 선택으로 본다."""
    if uri.startswith("sqlite:"):
        raw = uri.split(":///", 1)[-1]
    elif uri.startswith("file:"):
        raw = urlparse(uri).path
    elif "://" in uri:
        return
    else:
        raw = uri
    resolved = Path(raw).resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RuntimeError(f"MLFLOW_TRACKING_URI={uri} 가 {ROOT.resolve()} 밖입니다")


def flatten(payload: Any, prefix: str = "") -> tuple[dict[str, float], dict[str, str]]:
    """중첩 dict를 `a/b/c` 키로 편다. (수치, 문자열) 두 묶음을 돌려준다.

    리스트는 싣지 않는다 — 순서에 의미가 있는 표(`passing` 등)는 metrics.json artifact로 본다.
    NaN·inf는 MLflow 백엔드에 따라 거부되므로 뺀다. MLflow가 받지 않는 문자(`+`, 따옴표 등 —
    조합 프로파일 이름 `aspirate+liaison`에 실제로 나온다)는 `_`로 바꾼다.
    """
    numbers: dict[str, float] = {}
    texts: dict[str, str] = {}

    def unique(key: str) -> str:
        # 정리 과정에서 서로 다른 키가 같아지면(`a+b`, `a b` → `a_b`) 조용히 덮어쓰지 않는다.
        candidate, n = key, 2
        while candidate in numbers:
            candidate, n = f"{key}_{n}", n + 1
        return candidate

    def walk(node: Any, key: str) -> None:
        if isinstance(node, dict):
            for name, child in node.items():
                part = _INVALID_KEY_CHARS.sub("_", str(name))
                walk(child, f"{key}/{part}" if key else part)
        elif isinstance(node, bool):
            numbers[unique(key)] = float(node)
        elif isinstance(node, (int, float)):
            if math.isfinite(node):
                numbers[unique(key)] = float(node)
        elif isinstance(node, str):
            texts[key] = node

    walk(payload, prefix)
    return numbers, texts


def _param_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:_PARAM_LIMIT]


def flatten_params(config: dict[str, Any]) -> dict[str, str]:
    """config를 params로. 리스트·스칼라는 값 하나로, dict만 한 단계씩 편다."""
    out: dict[str, str] = {}

    def walk(node: Any, key: str) -> None:
        if isinstance(node, dict) and node:
            for name, child in node.items():
                walk(child, f"{key}.{name}" if key else str(name))
        elif key:
            out[key] = _param_value(node)

    walk(config, "")
    return out


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    assert_store_contained(uri)
    if uri.startswith("sqlite:"):
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=uri)


def _experiment_id(client, name: str) -> str:
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    # sqlite 백엔드의 기본 artifact 위치는 cwd 기준 ./mlruns 라 폴더 밖으로 샐 수 있다.
    location = (ARTIFACT_DIR / name).resolve().as_uri()
    return client.create_experiment(name, artifact_location=location)


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _log_inputs(client, run_id: str, inputs: dict[str, dict[str, Any]]) -> None:
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.entities import DatasetInput, InputTag

    entries = []
    for name, entry in inputs.items():
        tags = {"path": str(entry.get("path", ""))}
        if "rows" in entry:
            tags["rows"] = str(entry["rows"])
        dataset = MetaDataset(
            CodeDatasetSource(tags), name=name, digest=str(entry.get("sha256", ""))[:36]
        )
        entries.append(
            DatasetInput(
                dataset._to_mlflow_entity(),
                tags=[InputTag("mlflow.data.context", "input")],
            )
        )
    if entries:
        client.log_inputs(run_id, entries)


# 지금 프로세스에서 열려 있는 run. 모델의 fit()이 러너를 거치지 않고 여기로 기록한다.
_ACTIVE: "Tracker | None" = None
_FITS: list[dict[str, Any]] = []


class FitLog:
    """학습 한 번의 기록 — 하이퍼파라미터와 epoch별 loss."""

    def __init__(self, key: str, entry: dict[str, Any]) -> None:
        self.key = key
        self.entry = entry

    def epoch(self, step: int, loss: float) -> None:
        self.entry["loss"].append([int(step), float(loss)])
        if _ACTIVE is not None:
            _ACTIVE.log_step(f"train/{self.key}/loss", loss, step)


def begin_fit(kind: str, model: Any, **extra: Any) -> FitLog:
    """모델이 fit() 첫머리에서 부른다. 생성자 인자를 러너가 config에 안 넣었어도 여기서 잡힌다.

    같은 (종류, 기법)을 한 run에서 여러 번 학습하면(코퍼스 크기 스윕 등) `.2`, `.3`이 붙는다.
    MLflow가 꺼져 있어도 기록은 쌓이고 `fits()`로 manifest에 들어간다.
    """
    base = f"{kind}/{getattr(model, 'technique', None) or 'all'}"
    count = sum(1 for fit in _FITS if fit["base"] == base) + 1
    key = base if count == 1 else f"{base}.{count}"
    hyper = {
        name: value for name, value in vars(model).items()
        if not name.startswith("_") and isinstance(value, (bool, int, float, str))
    }
    hyper.update(extra)
    entry = {"key": key, "base": base, "kind": kind, "hyperparameters": hyper, "loss": []}
    _FITS.append(entry)
    if _ACTIVE is not None:
        _ACTIVE.try_log_params({f"fit.{key}.{name}": _param_value(v) for name, v in hyper.items()})
    return FitLog(key, entry)


def fits() -> list[dict[str, Any]]:
    return [{k: v for k, v in fit.items() if k != "base"} for fit in _FITS]


def reset_fits() -> None:
    _FITS.clear()


def find_runs(client, run_name: str) -> list:
    """run 이름(=run-id)으로 MLflow run을 찾는다. 모든 experiment를 본다."""
    ids = [e.experiment_id for e in client.search_experiments()]
    if not ids:
        return []
    return client.search_runs(ids, f"attributes.run_name = '{run_name}'")


def update_run(run_name: str, description: str | None, tags: dict[str, Any]) -> int:
    """이미 끝난 run의 설명·태그를 갱신한다(해석을 쓴 뒤, 리뷰를 받은 뒤). 갱신한 run 수를 돌려준다."""
    uri = tracking_uri()
    if uri is None:
        return 0
    client = _client(uri)
    found = find_runs(client, run_name)
    for run in found:
        if description is not None:
            client.set_tag(run.info.run_id, "mlflow.note.content", description[:_TAG_LIMIT])
        for key, value in tags.items():
            if value is not None:
                client.set_tag(run.info.run_id, key, str(value)[:_TAG_LIMIT])
    return len(found)


class Tracker:
    """MLflow run 하나. 실패는 경고만 남기고 삼킨다."""

    def __init__(self, client, run_id: str) -> None:
        self.client = client
        self.run_id = run_id
        self._closed = False
        atexit.register(self._abandon)

    @classmethod
    def start(
        cls,
        run_name: str,
        experiment: str,
        config: dict[str, Any],
        tags: dict[str, Any],
        description: str | None = None,
        *,
        start_time_ms: int | None = None,
    ) -> "Tracker | None":
        uri = tracking_uri()
        if uri is None:
            return None
        try:
            client = _client(uri)
            all_tags = {k: str(v)[:_TAG_LIMIT] for k, v in tags.items() if v is not None}
            all_tags["mlflow.runName"] = run_name
            if description:
                all_tags["mlflow.note.content"] = description.strip()[:_TAG_LIMIT]
            run = client.create_run(
                _experiment_id(client, experiment), start_time=start_time_ms, tags=all_tags
            )
            tracker = cls(client, run.info.run_id)
            tracker.log_params(flatten_params(config))
            global _ACTIVE
            _ACTIVE = tracker
            return tracker
        except Exception as error:  # noqa: BLE001 — 기록 실패가 실험을 죽이면 안 된다
            print(f"[mlflow] 기록을 시작하지 못했습니다(실험은 계속): {error}", file=sys.stderr)
            return None

    def log_params(self, params: dict[str, str]) -> None:
        from mlflow.entities import Param

        items = [Param(k, v) for k, v in params.items()]
        for chunk in _chunks(items, _PARAM_BATCH):
            self.client.log_batch(self.run_id, params=chunk)

    def try_log_params(self, params: dict[str, str]) -> None:
        try:
            self.log_params(params)
        except Exception as error:  # noqa: BLE001
            print(f"[mlflow] 학습 설정을 기록하지 못했습니다: {error}", file=sys.stderr)

    def log_step(self, key: str, value: float, step: int) -> None:
        try:
            if math.isfinite(value):
                self.client.log_metric(
                    self.run_id, key, float(value), int(time.time() * 1000), int(step)
                )
        except Exception as error:  # noqa: BLE001
            print(f"[mlflow] {key} 를 기록하지 못했습니다: {error}", file=sys.stderr)

    def log_metrics(self, numbers: dict[str, float]) -> None:
        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        items = [Metric(k, v, now, 0) for k, v in numbers.items()]
        for chunk in _chunks(items, _METRIC_BATCH):
            self.client.log_batch(self.run_id, metrics=chunk)

    def set_tags(self, tags: dict[str, Any]) -> None:
        for key, value in tags.items():
            if value is not None:
                self.client.set_tag(self.run_id, key, str(value)[:_TAG_LIMIT])

    def log_directory(self, directory: Path) -> list[str]:
        logged = []
        for path in sorted(Path(directory).iterdir()):
            if (
                path.is_file()
                and (path.suffix in ARTIFACT_SUFFIXES or path.name in ARTIFACT_NAMES)
                and path.stat().st_size <= MAX_ARTIFACT_BYTES
            ):
                self.client.log_artifact(self.run_id, str(path))
                logged.append(path.name)
        return logged

    def finish(
        self,
        metrics: dict[str, Any],
        *,
        inputs: dict[str, dict[str, Any]] | None = None,
        directory: Path | None = None,
        tags: dict[str, Any] | None = None,
        end_time_ms: int | None = None,
    ) -> None:
        try:
            numbers, texts = flatten(metrics)
            self.log_metrics(numbers)
            # 최상위 문자열(status 등)만 tag로 올린다. 깊은 곳의 문자열은 artifact에 있다.
            self.set_tags({k: v for k, v in texts.items() if "/" not in k})
            self.set_tags(tags or {})
            if inputs:
                _log_inputs(self.client, self.run_id, inputs)
            if directory is not None:
                self.log_directory(directory)
            self.client.set_terminated(self.run_id, "FINISHED", end_time=end_time_ms)
            self._closed = True
            self._release()
        except Exception as error:  # noqa: BLE001
            print(f"[mlflow] 기록을 마치지 못했습니다(결과는 results/에 있음): {error}",
                  file=sys.stderr)

    def _release(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    def _abandon(self) -> None:
        """finish 없이 프로세스가 끝났으면(예외·중단) FAILED로 닫는다."""
        if self._closed:
            return
        try:
            self.client.set_terminated(self.run_id, "FAILED")
        except Exception:  # noqa: BLE001
            pass
        self._closed = True
        self._release()


def is_smoke(run_id: str, markers: tuple[str, ...]) -> bool:
    return any(marker in run_id.lower() for marker in markers)


def main_docstring() -> str | None:
    """실행 중인 러너 모듈의 docstring. 그 실험의 목적이 거기 적혀 있다."""
    module = sys.modules.get("__main__")
    return getattr(module, "__doc__", None)
