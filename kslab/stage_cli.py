"""실험 단계(stage) 관리 — 노트 틀 만들기, 리뷰 자료 보기, 리뷰 기록.

트랙마다 따로 돈다. 각 트랙의 `exp/stage.py`가 이 모듈을 부른다.

    python -m restoration.exp.stage new s03-per-slot-threshold --title "기법별 임계값" \
        --runs s03.1-threshold-sweep --run-titles "기법별 임계값 스윕"
    python -m restoration.exp.stage add-run <단계-id> <run-id> --title "..." --reason "..."
    python -m restoration.exp.stage title <단계-id> [--run <run-id>] "제목"   # 보이는 이름 바꾸기
    python -m restoration.exp.stage status                    # 이 트랙의 전체 단계 상태
    python -m restoration.exp.stage show <단계-id>   # 리뷰 자료 한 번에 보기
    python -m restoration.exp.stage sync <단계-id>   # 노트·이름을 MLflow와 실험 대장에 반영
    python -m restoration.exp.stage review <단계-id> --verdict approve --feedback "..."

이름 규칙: 단계 id는 `sNN-<영문 요약>`, run id는 `sNN.k-<영문 요약>`(날짜는 넣지 않는다 —
MLflow와 manifest에 있다). 화면에는 `--title`로 준 한국어 제목이 `S03 기법별 임계값`,
`S03.1 기법별 임계값 스윕` 꼴로 나온다.

가드레일 트랙은 `restoration` 자리에 `guardrail`을 쓴다.

흐름: new → 노트의 '질문'·'판정 기준'·'설정' 작성 → 러너 실행(같은 단계는 병렬 가능) →
'결론'·'해석'·'다음'과 STAGE.md 작성 → show 로 리뷰 → review 로 피드백 기록 → 다음 단계.

`review`의 --feedback 은 리뷰어의 말을 그대로 적는다. 승인(approve) 전에는 같은
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
    stage = N.new_stage(args.stage, args.runs, title=args.title,
                        run_titles=args.run_titles)
    print(f"단계 생성: {stage.display_name}  ({stage.directory})")
    print(f"  {N.STAGE_DOC} 의 '{N.STAGE_QUESTION}' 과 각 run 노트의 "
          f"{'·'.join(repr(n) for n in N.BEFORE_RUN)} 를 채운 뒤 러너를 실행하세요.")
    for run_id in stage.runs:
        print(f"  - {stage.run_display_name(run_id)}: {stage.note_path(run_id)}")


def cmd_add_run(args) -> None:
    stage = N.add_run(args.stage, args.run_id, args.reason or "", title=args.title or "")
    print(f"추가: {stage.run_display_name(args.run_id)}  ({stage.note_path(args.run_id)})")
    if stage.sealed:
        print("  이미 시작된 단계라 '결과를 본 뒤 추가된 run'으로 기록됐습니다.")


def cmd_title(args) -> None:
    stage = N.set_title(args.stage, args.title, args.run)
    shown = stage.run_display_name(args.run) if args.run else stage.display_name
    print(f"제목: {shown}")
    if tracking.enabled():
        _sync(stage)


def cmd_status(_args) -> None:
    stages = N.load_stages()
    print(f"트랙: {tracks.current()}  ({tracks.notes_dir()})")
    if not stages:
        print(f"[{tracks.current()}] 단계가 없습니다. "
              f"'python -m {tracks.module()}.stage new <단계-id> --runs <run-id>...'")
        return
    for stage in stages:
        finished = set(stage.finished())
        print(f"{stage.display_name}  [{_STATE_LABEL[stage.state()]}]  "
              f"run {len(finished)}/{len(stage.runs)} 완료, 리뷰 {len(stage.data['reviews'])}회"
              f"  ({stage.stage_id})")
        for run_id in stage.runs:
            path = stage.note_path(run_id)
            text = path.read_text(encoding="utf-8") if path.exists() else ""
            empty = N.missing(text, N.before_run(text) + N.after_run(text))
            mark = "완료" if run_id in finished else "미완료"
            note = "노트 완성" if not empty else "빈 칸: " + ", ".join(empty)
            print(f"    {stage.run_display_name(run_id)}  {mark}  {note}  ({run_id})")
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
        reformatted = (stage.data.get("reformatted") or {}).get(run_id)
        if reformatted:
            out.append(f"> 노트를 {reformatted['at']}에 새 템플릿으로 옮겨 적었습니다. "
                       f"실행 전 칸의 원본은 `{tracks.current()}/results/{run_id}/note-at-start.md`.")
        if changed:
            out.append("> **주의: 실행 전 칸이 실행 시작 뒤에 수정됐습니다.** "
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


_LEDGER_BEGIN = "<!-- stages:begin (stage sync가 다시 쓴다. 손으로 고치지 않는다) -->"
_LEDGER_END = "<!-- stages:end -->"


def ledger_table(stages: list[N.Stage]) -> str:
    """실험 대장의 단계 표. 단계 하나에 한 줄: 질문과 답의 첫 줄, 상태."""
    rows = ["| 단계 | 질문 | 답 | 상태 |", "|---|---|---|---|"]
    for stage in stages:
        doc = N.sections(stage.doc_path().read_text(encoding="utf-8"))
        rows.append(
            f"| **{stage.display_name}** | {N.first_line(doc.get(N.STAGE_QUESTION, ''))} | "
            f"{N.first_line(doc.get(N.STAGE_ANSWER, ''), sentence=False) or '-'} | "
            f"{_STATE_LABEL[stage.state()]} |"
        )
    return "\n".join(rows)


def write_ledger(stages: list[N.Stage]) -> bool:
    """EXPERIMENTS.md 의 표시 구간만 다시 쓴다. 구간이 없으면 건드리지 않는다."""
    path = tracks.ledger()
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if _LEDGER_BEGIN not in text or _LEDGER_END not in text:
        return False
    head, rest = text.split(_LEDGER_BEGIN, 1)
    _, tail = rest.split(_LEDGER_END, 1)
    # 커밋되는 파일이라 Windows에서도 LF로 쓴다.
    path.write_text(f"{head}{_LEDGER_BEGIN}\n{ledger_table(stages)}\n{_LEDGER_END}{tail}",
                    encoding="utf-8", newline="\n")
    return True


def _sync(stage: N.Stage) -> None:
    reviews = stage.data["reviews"]
    status = {"approve": "approved", "revise": "revise"}.get(stage.last_verdict, "pending")
    review_tags = {
        "review.status": status,
        "review.feedback": reviews[-1]["feedback"] if reviews else None,
        "review.at": reviews[-1]["at"] if reviews else None,
    }
    track = tracks.current()
    parent = tracking.sync_stage_run(
        tracking.experiment_name(track), stage.stage_id, stage.display_name,
        N.for_display(stage.doc_path().read_text(encoding="utf-8")),
        {"kslab.track": track, "kslab.stage_state": stage.state(), **review_tags},
    )
    print(f"  {stage.display_name}: 단계 run 갱신")
    for run_id in stage.runs:
        path = stage.note_path(run_id)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        count = tracking.update_run(run_id, N.for_display(text), {
            "kslab.stage": stage.stage_id,
            "kslab.compares_to": N.compares_to(text) or None,
            "note.interpreted": str(not N.missing(text, N.after_run(text))).lower(),
            **review_tags,
        }, display_name=stage.run_display_name(run_id), parent_run_id=parent)
        print(f"  {stage.run_display_name(run_id)}: MLflow run {count}개 갱신")
    if write_ledger(N.load_stages()):
        print(f"  실험 대장 갱신: {tracks.ledger()}")


def cmd_sync(args) -> None:
    if not tracking.enabled():
        raise SystemExit("MLFLOW_TRACKING_URI 가 없습니다. 먼저 '. .\\env.ps1' 을 실행하세요.")
    stages = N.load_stages() if args.stage == "all" else [N.get_stage(args.stage)]
    for stage in stages:
        _sync(stage)


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
    new.add_argument("--title", required=True, help="사람이 보는 단계 제목(한국어 짧게)")
    new.add_argument("--runs", nargs="+", required=True)
    new.add_argument("--run-titles", nargs="+", help="run마다 보이는 제목(--runs 와 같은 순서)")
    new.set_defaults(func=cmd_new)

    add = sub.add_parser("add-run", help="단계에 run을 추가한다")
    add.add_argument("stage")
    add.add_argument("run_id")
    add.add_argument("--title", help="사람이 보는 run 제목")
    add.add_argument("--reason", help="이미 시작된 단계에 추가할 때 필수")
    add.set_defaults(func=cmd_add_run)

    title = sub.add_parser("title", help="단계나 run의 보이는 제목을 바꾼다")
    title.add_argument("stage")
    title.add_argument("title")
    title.add_argument("--run", help="run 제목을 바꿀 때 run-id")
    title.set_defaults(func=cmd_title)

    sub.add_parser("status", help="전체 단계 상태").set_defaults(func=cmd_status)

    show = sub.add_parser("show", help="리뷰 자료를 한 번에 출력한다")
    show.add_argument("stage")
    show.set_defaults(func=cmd_show)

    sync = sub.add_parser("sync", help="노트·이름·리뷰 상태를 MLflow와 실험 대장에 반영한다")
    sync.add_argument("stage", help="단계 id, 또는 all")
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
