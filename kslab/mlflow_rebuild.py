"""MLflow 저장소를 지금 구성(트랙 → 단계 → run → 학습)으로 다시 만든다.

MLflow는 원본이 아니다 — 원본은 `<트랙>/results/<run-id>/`(manifest·metrics·report)와
`private/<트랙>/notes/`다. 그래서 기록 구성을 바꾸면 기존 저장소를 고치지 않고 원본에서 새로 쓴다.

    python -m kslab.mlflow_rebuild            # 기존 저장소를 data/mlflow/v1-<시각>/ 로 옮기고 새로 씀

- 기존 `mlflow.db`와 `artifacts/`는 지우지 않고 옮겨 둔다. 되돌리려면 다시 옮기면 된다.
- 단계에 속한 run은 결과 폴더에서 다시 기록한다. 노트는 지금 내용(해석 포함)이 들어간다.
- 결과 폴더가 없는 옛 run(MLflow 도입 전 E0~E10)은 트랙의 `exp/backfill_mlflow.py`가
  baseline JSON으로 채운다.
- 실행 중인 `mlflow ui`는 먼저 끈다(sqlite 파일을 잡고 있다).
"""

from __future__ import annotations

import calendar
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import notes as N
from kslab import tracking, tracks

_REVIEW_STATUS = {"approve": "approved", "revise": "revise"}


def _ms(stamp: str | None) -> int | None:
    if not stamp:
        return None
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) * 1000


def _seed_of(label: str) -> int | None:
    tail = label.rsplit("seed", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def match_fit_labels(fits: list[dict], models: dict[str, Any]) -> list[str | None]:
    """fit 기록에 이름표가 없던 옛 run용. metrics.json 의 `models` 이름과 짝을 짓는다.

    1) 시드가 같고 loss 곡선이 같으면 확정. 2) 단일 기법 모델은 이름에 기법이 들어간 것.
    3) 남은 합집합 모델은 같은 시드 후보가 하나뿐일 때. 못 고르면 None(기법 이름으로 보인다).
    """
    labels: list[str | None] = [f.get("label") for f in fits]
    taken = {label for label in labels if label}
    if not isinstance(models, dict) or all(labels):
        return labels

    def candidates(fit: dict) -> list[str]:
        seed = fit["hyperparameters"].get("random_state")
        return [name for name in models if name not in taken and _seed_of(name) == seed]

    def claim(index: int, name: str) -> None:
        labels[index] = name
        taken.add(name)

    for index, fit in enumerate(fits):
        if labels[index] or not fit.get("loss"):
            continue
        curve = [round(v, 4) for _, v in fit["loss"]]
        for name in candidates(fit):
            if [round(v, 4) for v in models[name].get("loss") or []] == curve:
                claim(index, name)
                break
    for index, fit in enumerate(fits):
        technique = str(fit["hyperparameters"].get("technique") or "")
        if labels[index] or not technique or "+" in technique:
            continue
        found = [n for n in candidates(fit) if technique in n]
        if len(found) == 1:
            claim(index, found[0])
    single = {str(f["hyperparameters"].get("technique")) for f in fits
              if "+" not in str(f["hyperparameters"].get("technique"))}
    for index, fit in enumerate(fits):
        if labels[index]:
            continue
        technique = str(fit["hyperparameters"].get("technique") or "")
        found = [n for n in candidates(fit) if not any(s and s in n for s in single)]
        if "+" in technique and len(found) == 1:
            claim(index, found[0])
    return labels


def replay_run(client, experiment: str, stage: N.Stage, run_id: str, results: Path) -> bool:
    """결과 폴더 하나를 MLflow run으로 다시 쓴다. 결과가 없으면 False."""
    directory = results / run_id
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    start = _ms(manifest.get("started_at_utc"))
    end = start + int(1000 * manifest.get("elapsed_seconds", 0)) if start else None
    note_path = stage.note_path(run_id)
    note_text = note_path.read_text(encoding="utf-8") if note_path.exists() else None
    sandbox, upstream = manifest.get("sandbox_git", {}), manifest.get("ksafeguard_git", {})
    reviews = stage.data.get("reviews") or []
    note = manifest.get("note") or {}
    tags = {
        "kslab.run_id": run_id,
        "kslab.track": tracks.current(),
        "kslab.stage": stage.stage_id,
        "kslab.experiment": manifest.get("experiment"),
        "kslab.compares_to": (N.compares_to(note_text) if note_text else None) or None,
        "kslab.results_dir": str(directory),
        "kslab.code_diff": "code.diff" if (directory / "code.diff").exists() else None,
        "kslab.git_commit": sandbox.get("commit"),
        "kslab.git_dirty": sandbox.get("dirty"),
        "ksafeguard.git_commit": upstream.get("commit"),
        "ksafeguard.git_dirty": upstream.get("dirty"),
        "kslab.rebuilt": "true",
        "note.prereg_sha256": note.get("prereg_sha256"),
        "note.interpreted": (str(not N.missing(note_text, N.after_run(note_text))).lower()
                             if note_text else None),
        "review.status": _REVIEW_STATUS.get(stage.last_verdict, "pending"),
        "review.feedback": reviews[-1]["feedback"] if reviews else None,
        "review.at": reviews[-1]["at"] if reviews else None,
        "host.platform": (manifest.get("host") or {}).get("platform"),
        **{f"pkg.{k}": v for k, v in (manifest.get("packages") or {}).items()},
    }
    tracker = tracking.Tracker.start(
        stage.run_display_name(run_id), experiment, manifest.get("config") or {}, tags,
        description=N.for_display(note_text) if note_text else None,
        start_time_ms=start, activate=False,
        stage={"id": stage.stage_id, "display_name": stage.display_name,
               "note": N.for_display(stage.doc_path().read_text(encoding="utf-8"))},
    )
    if tracker is None:
        raise SystemExit(f"{run_id}: MLflow run을 만들지 못했습니다")
    fits = manifest.get("fits") or []
    labels = match_fit_labels(fits, metrics.get("models") or {})
    for fit, label in zip(fits, labels):
        hyper = fit.get("hyperparameters") or {}
        name = tracking.fit_run_name(
            fit.get("kind", "fit"), str(hyper.get("technique") or "all"), label,
            hyper.get("random_state"),
        )
        child = tracker.open_child(name, hyper, {"kslab.fit_key": fit.get("key")},
                                   start_time_ms=start)
        curve = fit.get("loss") or []
        if not curve and label and isinstance(metrics.get("models"), dict):
            # 체크포인트에서 이어받은 모델은 이 run에서 학습하지 않았다. 원래 곡선을 보여준다.
            curve = [[i + 1, v] for i, v in enumerate(metrics["models"][label].get("loss") or [])]
            if child is not None:
                client.set_tag(child, "kslab.loss_from", "checkpoint (이 run에서 재학습하지 않음)")
        for step, value in curve:
            tracker.log_step("loss", value, step, run_id=child, timestamp_ms=start)
    headline = manifest.get("headline")
    tracker.finish(
        {**metrics, "headline": headline} if headline else metrics,
        inputs=manifest.get("inputs"), directory=directory,
        tags={f"note.{k}": v for k, v in (manifest.get("notes") or {}).items()},
        end_time_ms=end,
    )
    return True


def rebuild_track(client, track: str) -> None:
    tracks.use(track)
    experiment = tracking.experiment_name(track)
    in_stages: set[str] = set()
    stages = N.load_stages()
    for stage in stages:
        in_stages.update(stage.runs)
    try:
        backfill = importlib.import_module(f"{track}.exp.backfill_mlflow")
    except ModuleNotFoundError:
        backfill = None
    if backfill is not None:
        backfill.backfill(client, experiment, skip=in_stages)
    for stage in stages:
        reviews = stage.data.get("reviews") or []
        tracking.ensure_stage_run(
            client, experiment, stage.stage_id, stage.display_name,
            N.for_display(stage.doc_path().read_text(encoding="utf-8")),
            {"kslab.track": track, "kslab.stage_state": stage.state(),
             "review.status": _REVIEW_STATUS.get(stage.last_verdict, "pending"),
             "review.feedback": reviews[-1]["feedback"] if reviews else None},
            _ms(stage.data.get("sealed_at") or stage.data.get("created_at")),
        )
        for run_id in stage.runs:
            done = replay_run(client, experiment, stage, run_id, tracks.results_dir())
            print(f"[{track}] {stage.run_display_name(run_id)}: "
                  f"{'기록' if done else '결과 폴더 없음, 건너뜀'}")


def move_old_store() -> Path | None:
    store = tracking.MLFLOW_DIR
    db, artifacts = store / "mlflow.db", tracking.ARTIFACT_DIR
    if not db.exists() and not artifacts.exists():
        return None
    target = store / f"v1-{time.strftime('%Y%m%d-%H%M%S')}"
    target.mkdir(parents=True)
    for path in (db, artifacts):
        if path.exists():
            shutil.move(str(path), str(target / path.name))
    return target


def main() -> None:
    uri = tracking.tracking_uri()
    if uri is None or not uri.startswith("sqlite:"):
        raise SystemExit("sqlite MLFLOW_TRACKING_URI 가 필요합니다. 먼저 '. .\\env.ps1'.")
    moved = move_old_store()
    if moved:
        print(f"기존 저장소를 옮겼습니다: {moved}")
    client = tracking._client(uri)
    for track in tracks.TRACKS:
        rebuild_track(client, track)
    print("완료. 'mlflow ui --port 5000' 으로 확인하세요.")


if __name__ == "__main__":
    main()
