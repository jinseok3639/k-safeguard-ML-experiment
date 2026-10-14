"""기존 baseline을 MLflow로 옮긴다 — MLflow 도입 전에 끝난 실험용 일회성 백필.

`baselines/*.json`은 집계 수치만 담고 있고 원본 `results/`는 커밋되지 않으므로, 백필된
run에는 params·입력 해시·report artifact가 없다. 수치와 판정, 그리고 baseline JSON 자체만
들어간다. `kslab.backfilled=true` 태그로 구분한다.

같은 run-id가 이미 MLflow에 있으면 건너뛰므로 여러 번 돌려도 된다.

    python -m exp.backfill_mlflow
"""

from __future__ import annotations

import calendar
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import record as R
from kslab import tracking

# baseline 파일 접두사 -> 러너가 쓰는 experiment 이름.
EXPERIMENTS = {
    "e0_probe": "E0_baseline_reproduce",
    "e1_corpus_scaling": "E1_corpus_scaling",
    "e2_architecture": "E2_architecture",
    "e3_tensify_decision": "E3_tensify_decision",
    "e4_chosung_slots": "E4_chosung_slots",
    "e5a_generators": "E5a_generators",
    "e6_gate": "E6_gate",
    "e7_provider": "E7_provider_prototype",
    "e8_abstention": "E8_abstention",
    "e9a_phonology_ceiling": "E9a_phonology_ceiling",
    "e9b_composition_bilstm": "E9b_composition",
    "e9b_composition": "E9b_composition",
    "colab_kanana_guardrail": "E10_real_guardrail",
}
_LEDGER_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([^`]+)`")


def ledger_dates() -> dict[str, int]:
    """EXPERIMENTS.md 에서 run-id -> 실행일(UTC 자정, ms)."""
    dates: dict[str, int] = {}
    for line in (R.ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8").splitlines():
        match = _LEDGER_ROW.match(line)
        if match:
            stamp = calendar.timegm(time.strptime(match.group(1), "%Y-%m-%d"))
            dates[match.group(2)] = stamp * 1000
    return dates


def _run_id(payload: dict, stem: str) -> str:
    manifest = payload.get("manifest") or {}
    return payload.get("run_id") or manifest.get("run_id") or stem


def main() -> None:
    if not tracking.enabled():
        raise SystemExit("MLFLOW_TRACKING_URI 가 없습니다. 먼저 'source ./env.sh' 를 실행하세요.")
    client = tracking._client(tracking.tracking_uri())
    dates = ledger_dates()
    seen: set[str] = set()

    for path in sorted(R.BASELINES_DIR.glob("*.json")):
        stem = re.sub(r"_v\d+$", "", path.stem)
        experiment = EXPERIMENTS.get(stem)
        if experiment is None:
            print(f"[backfill] {path.name}: experiment 매핑 없음, 건너뜀")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = _run_id(payload, path.stem)
        # e9b_composition_v1.json 은 나중 실행이 덮어써 bilstm_v1 과 같은 run을 가리킨다.
        if run_id in seen:
            print(f"[backfill] {path.name}: {run_id} 는 다른 baseline에서 이미 처리, 건너뜀")
            continue
        seen.add(run_id)
        found = client.get_experiment_by_name(experiment)
        if found and client.search_runs(
            [found.experiment_id], f"attributes.run_name = '{run_id}'"
        ):
            print(f"[backfill] {path.name}: {run_id} 이미 있음, 건너뜀")
            continue

        manifest = payload.get("manifest") or {}
        metrics = payload.get("metrics") if "manifest" in payload else payload
        stamp = dates.get(run_id)
        tracker = tracking.Tracker.start(
            run_id,
            experiment,
            manifest.get("config") or {},
            tags={
                "kslab.experiment": experiment,
                "kslab.backfilled": "true",
                "kslab.baseline_file": path.name,
            },
            description=(
                f"`baselines/{path.name}` 에서 백필한 run. 원본 results/ 가 없어 "
                "집계 수치만 있다. 실험 목적과 해석은 reports/ 의 해당 보고서를 본다."
            ),
            start_time_ms=stamp,
        )
        if tracker is None:
            raise SystemExit(f"{path.name}: MLflow run을 만들지 못했습니다")
        tracker.client.log_artifact(tracker.run_id, str(path))
        tracker.finish(metrics, end_time_ms=stamp)
        print(f"[backfill] {path.name}: {experiment} / {run_id}")


if __name__ == "__main__":
    main()
