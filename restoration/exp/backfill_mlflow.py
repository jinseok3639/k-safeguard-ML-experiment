"""단계 도입 전 복원기 실험(E0~E10)을 MLflow로 옮긴다 — `kslab.mlflow_rebuild`가 부른다.

`baselines/*.json`은 집계 수치만 담고 있고 원본 `results/`는 커밋되지 않으므로, 백필된
run에는 params·입력 해시·report artifact가 없다. 수치와 판정, 그리고 baseline JSON 자체만
들어간다. `kslab.backfilled=true` 태그로 구분한다.

MLflow 화면에서는 `restoration` 실험 안의 `S00 단계 도입 전 실험 (E0~E10)` 아래에 모인다.
run 이름은 실험 대장(EXPERIMENTS.md)의 '실험' 칸이다. 단계에 소급 등록된 run(E8b 등)은
단계 쪽에서 결과 폴더로 다시 쓰므로 여기서는 건너뛴다.

    python -m kslab.mlflow_rebuild      # 저장소 전체를 다시 쓸 때
"""

from __future__ import annotations

import calendar
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kslab import tracking, tracks

LEGACY_STAGE = "s00-legacy"
LEGACY_TITLE = "S00 단계 도입 전 실험 (E0~E10)"
LEGACY_NOTE = (
    "# S00 단계 도입 전 실험 (E0~E10)\n\n"
    "노트·리뷰 게이트와 MLflow를 도입하기 전에 돌린 실험이다. 원본 결과 폴더가 없어 "
    "`baselines/`의 집계 수치만 있다. 실험마다의 목적과 해석은 `restoration/reports/`의 "
    "보고서, 한 줄 요약은 `restoration/EXPERIMENTS.md`의 run 기록 표를 본다."
)

# baseline 파일 접두사 -> 러너가 쓰던 experiment 이름(태그로만 남긴다).
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
_LEDGER_ROW = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|")


def ledger_rows() -> dict[str, tuple[int, str]]:
    """EXPERIMENTS.md 에서 run-id -> (실행일 UTC 자정 ms, '실험' 칸)."""
    rows: dict[str, tuple[int, str]] = {}
    for line in tracks.ledger("restoration").read_text(encoding="utf-8").splitlines():
        match = _LEDGER_ROW.match(line)
        if match:
            stamp = calendar.timegm(time.strptime(match.group(1), "%Y-%m-%d")) * 1000
            rows.setdefault(match.group(2), (stamp, match.group(3)))
    return rows


def _run_id(payload: dict, stem: str) -> str:
    manifest = payload.get("manifest") or {}
    return payload.get("run_id") or manifest.get("run_id") or stem


def backfill(client, experiment: str, skip: set[str] = frozenset()) -> None:
    rows = ledger_rows()
    seen: set[str] = set()
    parent: dict = {"id": LEGACY_STAGE, "display_name": LEGACY_TITLE, "note": LEGACY_NOTE,
                    "tags": {"kslab.track": "restoration", "kslab.backfilled": "true"}}

    for path in sorted(tracks.baselines_dir("restoration").glob("*.json")):
        stem = re.sub(r"_v\d+$", "", path.stem)
        runner_experiment = EXPERIMENTS.get(stem)
        if runner_experiment is None:
            print(f"[backfill] {path.name}: experiment 매핑 없음, 건너뜀")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_id = _run_id(payload, path.stem)
        # e9b_composition_v1.json 은 나중 실행이 덮어써 bilstm_v1 과 같은 run을 가리킨다.
        if run_id in seen or run_id in skip:
            continue
        seen.add(run_id)
        if tracking.find_runs(client, run_id):
            print(f"[backfill] {path.name}: {run_id} 이미 있음, 건너뜀")
            continue

        manifest = payload.get("manifest") or {}
        metrics = payload.get("metrics") if "manifest" in payload else payload
        stamp, title = rows.get(run_id, (None, run_id))
        tracker = tracking.Tracker.start(
            title,
            experiment,
            manifest.get("config") or {},
            tags={
                "kslab.run_id": run_id,
                "kslab.track": "restoration",
                "kslab.stage": LEGACY_STAGE,
                "kslab.experiment": runner_experiment,
                "kslab.backfilled": "true",
                "kslab.baseline_file": path.name,
            },
            description=(
                f"`baselines/{path.name}` 에서 백필한 run. 원본 results/ 가 없어 "
                "집계 수치만 있다. 실험 목적과 해석은 reports/ 의 해당 보고서를 본다."
            ),
            start_time_ms=stamp,
            stage=parent,
            activate=False,
        )
        if tracker is None:
            raise SystemExit(f"{path.name}: MLflow run을 만들지 못했습니다")
        tracker.client.log_artifact(tracker.run_id, str(path))
        tracker.finish(metrics, end_time_ms=stamp)
        print(f"[backfill] {path.name}: {title}")
