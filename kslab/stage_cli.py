"""실험 단계(stage) 관리 — 노트 틀 만들기, 리뷰 자료 보기, 리뷰 기록.

트랙마다 따로 돈다. 각 트랙의 `exp/stage.py`가 이 모듈을 부른다.

    python -m restoration.exp.stage new <단계-id> --runs <run-id>...
    python -m restoration.exp.stage add-run <단계-id> <run-id> --reason "..."
    python -m restoration.exp.stage status                    # 이 트랙의 전체 단계 상태
    python -m restoration.exp.stage show <단계-id>   # 리뷰 자료 한 번에 보기
    python -m restoration.exp.stage sync <단계-id>   # 노트를 MLflow에 반영
    python -m restoration.exp.stage review <단계-id> --verdict approve --feedback "..."

가드레일 트랙은 `restoration` 자리에 `guardrail`을 쓴다.

흐름: new → 노트의 '왜 돌리나'·'무엇을 바꿨나' 작성 → 러너 실행(같은 단계는 병렬 가능) →
'해석'·'다음'과 STAGE.md 작성 → show 로 리뷰 → review 로 피드백 기록 → 다음 단계.

`review`의 --feedback 은 리뷰어(사용자)의 말을 그대로 적는다. 승인(approve) 전에는 같은
트랙의 다른 단계 run이 시작되지 않는다. revise 는 피드백만 남기고 단계를 열어 둔다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import notes as N
from kslab import tracking, tracks

_STATE_LABEL = {
    "planned": "계획됨(미시작)",
    "running": "실행 중",
    "awaiting_review": "리뷰 대기",
    "approved": "승인됨",
}


def _manifest(run_id: str) -> dict | None:
    path = N.RESULTS_DIR / run_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _flat(node, prefix: str = "") -> list[tuple[str, float]]:
    if isinstance(node, dict):
        rows: list[tuple[str, float]] = []
        for key, value in node.items():
            rows += _flat(value, f"{prefix}/{key}" if prefix else str(key))
        return rows
    return [(prefix, node)] if isinstance(node, (int, float)) else []


def cmd_new(args) -> None:
    stage = N.new_stage(args.stage, args.runs)
    print(f"단계 생성: {stage.directory}")
    print(f"  {N.STAGE_DOC} 의 '{N.STAGE_PURPOSE}' 와 각 run 노트의 "
          f"'{N.WHY}'·'{N.CHANGES}' 를 채운 뒤 러너를 실행하세요.")
    for run_id in stage.runs:
        print(f"  - {stage.note_path(run_id)}")


def cmd_add_run(args) -> None:
    stage = N.add_run(args.stage, args.run_id, args.reason or "")
    print(f"추가: {stage.note_path(args.run_id)}")
    if stage.sealed:
        print("  이미 시작된 단계라 '결과를 본 뒤 추가된 run'으로 기록됐습니다.")


def cmd_status(_args) -> None:
    stages = N.load_stages()
    print(f"트랙: {tracks.current()}  ({tracks.notes_dir()})")
    if not stages:
        print(f"[{tracks.current()}] 단계가 없습니다. "
              f"'python -m {tracks.module()}.stage new <단계-id> --runs <run-id>...'")
        return
    for stage in stages:
        finished = set(stage.finished())
        print(f"{stage.stage_id}  [{_STATE_LABEL[stage.state()]}]  "
              f"run {len(finished)}/{len(stage.runs)} 완료, 리뷰 {len(stage.data['reviews'])}회")
        for run_id in stage.runs:
            path = stage.note_path(run_id)
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            empty = N.missing(text, N.BEFORE_RUN + N.AFTER_RUN)
            mark = "완료" if run_id in finished else "미완료"
            note = "노트 완성" if not empty else "빈 칸: " + ", ".join(empty)
            print(f"    {run_id}  {mark}  {note}")
    blocking = [s.stage_id for s in stages if s.sealed and not s.approved]
    if blocking:
        print(f"\n다음 단계를 막고 있는 단계: {', '.join(blocking)}")


def cmd_show(args) -> None:
    """리뷰어가 한 번에 읽을 자료를 마크다운으로 찍는다."""
    stage = N.get_stage(args.stage)
    finished = set(stage.finished())
    out = [stage.doc_path().read_text(encoding="utf-8").rstrip(), ""]
    out += ["---", "", f"상태: **{_STATE_LABEL[stage.state()]}** · "
            f"봉인 {stage.data.get('sealed_at') or '-'}", ""]
    if stage.data["added"]:
        out += ["### 시작 이후 추가된 run", ""]
        for entry in stage.data["added"]:
            seen = ", ".join(entry["finished_then"]) or "없음"
            out.append(f"- `{entry['run_id']}` ({entry['at']}) — {entry['reason']} "
                       f"/ 추가 시점에 이미 끝나 있던 run: {seen}")
        out.append("")
    for run_id in stage.runs:
        out += ["---", ""]
        path = stage.note_path(run_id)
        out.append(path.read_text(encoding="utf-8").rstrip() if path.exists()
                   else f"# {run_id}\n\n(노트 파일 없음)")
        out.append("")
        manifest = _manifest(run_id)
        if run_id not in finished or manifest is None:
            out += ["> **이 run은 끝나지 않았습니다**(결과 없음).", ""]
            continue
        git = manifest.get("sandbox_git", {})
        out.append(
            f"> 소요 {manifest.get('elapsed_seconds')}s · 커밋 `{str(git.get('commit'))[:10]}`"
            f"{' + 미커밋 변경(code.diff)' if git.get('dirty') else ''} · "
            f"결과 `{tracks.current()}/results/{run_id}/`"
        )
        changed = N.prereg_changed(stage, run_id)
        if changed:
            out.append("> **주의: '왜 돌리나'/'무엇을 바꿨나'가 실행 시작 뒤에 수정됐습니다.** "
                       f"시작 시점 원본은 `{tracks.current()}/results/{run_id}/note-at-start.md`.")
        elif changed is None:
            out.append("> 시작 시점 노트 사본이 없어 실행 전 칸의 변경 여부를 확인할 수 없습니다.")
        headline = manifest.get("headline")
        if headline:
            out += ["", "| 대표 지표 | 값 |", "|---|---|"]
            out += [f"| {key} | {value:.6g} |" for key, value in _flat(headline)]
        out.append("")
    problems = N.review_blockers(stage)
    if problems:
        out += ["---", "", "### 리뷰 전에 채워야 할 것", ""] + [f"- {p}" for p in problems]
    print("\n".join(out))


def _sync(stage: N.Stage) -> None:
    reviews = stage.data["reviews"]
    status = {"approve": "approved", "revise": "revise"}.get(stage.last_verdict, "pending")
    for run_id in stage.runs:
        path = stage.note_path(run_id)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        count = tracking.update_run(run_id, text, {
            "kslab.stage": stage.stage_id,
            "kslab.compares_to": N.compares_to(text) or None,
            "note.interpreted": str(not N.missing(text, N.AFTER_RUN)).lower(),
            "review.status": status,
            "review.feedback": reviews[-1]["feedback"] if reviews else None,
            "review.at": reviews[-1]["at"] if reviews else None,
        })
        print(f"  {run_id}: MLflow run {count}개 갱신")


def cmd_sync(args) -> None:
    if not tracking.enabled():
        raise SystemExit("MLFLOW_TRACKING_URI 가 없습니다. 먼저 'source ./env.sh' 를 실행하세요.")
    _sync(N.get_stage(args.stage))


def cmd_review(args) -> None:
    stage = N.record_review(args.stage, args.verdict, args.feedback)
    print(f"리뷰 기록: {stage.stage_id} — {args.verdict}")
    if tracking.enabled():
        _sync(stage)
    if stage.approved:
        print("단계가 닫혔습니다. 다음 단계를 시작할 수 있습니다.")
    else:
        print("단계는 열려 있습니다. 피드백을 반영한 뒤 다시 리뷰를 받으세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="단계와 run 노트 틀을 만든다")
    new.add_argument("stage")
    new.add_argument("--runs", nargs="+", required=True)
    new.set_defaults(func=cmd_new)

    add = sub.add_parser("add-run", help="단계에 run을 추가한다")
    add.add_argument("stage")
    add.add_argument("run_id")
    add.add_argument("--reason", help="이미 시작된 단계에 추가할 때 필수")
    add.set_defaults(func=cmd_add_run)

    sub.add_parser("status", help="전체 단계 상태").set_defaults(func=cmd_status)

    show = sub.add_parser("show", help="리뷰 자료를 한 번에 출력한다")
    show.add_argument("stage")
    show.set_defaults(func=cmd_show)

    sync = sub.add_parser("sync", help="노트와 리뷰 상태를 MLflow run에 반영한다")
    sync.add_argument("stage")
    sync.set_defaults(func=cmd_sync)

    review = sub.add_parser("review", help="리뷰어의 피드백을 기록한다")
    review.add_argument("stage")
    review.add_argument("--verdict", choices=N.VERDICTS, required=True)
    review.add_argument("--feedback", required=True, help="리뷰어의 말 그대로")
    review.set_defaults(func=cmd_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
