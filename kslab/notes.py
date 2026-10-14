"""실험 노트와 단계(stage) 리뷰 게이트.

수치만 남기고 "왜 돌렸는지 / 어떻게 읽었는지 / 그래서 다음에 뭘 하기로 했는지"를 안 남기면
나중에 그 실험을 다시 쓸 수 없다. 그래서 두 가지를 기계적으로 강제한다.

1. **노트 없이는 실행하지 못한다.** 스모크가 아닌 run은 `notes/<단계>/<run-id>.md`의
   "왜 돌리나"와 "무엇을 바꿨나"가 채워져 있어야 시작한다. 이 두 칸은 실행 **전에** 쓰는
   칸이라, 시작 시점의 내용을 해시와 함께 `results/<run-id>/note-at-start.md`로 떠 둔다 —
   결과를 보고 가설을 고쳐 쓰면 드러난다.
2. **리뷰 없이는 다음 단계로 못 넘어간다.** run은 단계로 묶인다. 한 단계 안의 run은
   함께(병렬로) 돌릴 수 있고 리뷰는 단계당 한 번이다. 시작된 단계가 승인되기 전에는 다른
   단계의 run이 시작되지 않는다.

단계의 run 목록은 첫 run이 시작될 때 봉인된다. 그 뒤에 run을 끼워 넣을 수는 있지만
(실패 재실행 등) 그때 이미 끝나 있던 run과 사유가 `stage.json`에 남아 리뷰 때 보인다 —
결과를 보고 나서 실험을 덧붙이는 것을 금지하지는 않되 숨기지는 못하게 한다.

리뷰 기록(`review`)은 **사용자의 말을 그대로** 적는 자리다. 실험을 돌린 쪽이 스스로
승인하는 용도가 아니다.

이름은 두 겹이다. 기계용 id(`s02-generator-dependence`, `e11b-family-holdout-20261014`)는
폴더·파일·결과 경로에 쓰고 바꾸지 않는다. 사람이 보는 이름은 `stage.json`의 `title`과
`titles`에서 만든다 — 단계는 `S02 생성기 의존도`, run은 단계 안 순서대로 `S02.2 기법군 빼고
학습`. MLflow 화면, 노트 제목, 실험 대장에는 이 이름이 나온다.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kslab import tracks

STAGE_FILE = "stage.json"
STAGE_DOC = "STAGE.md"

WHY = "왜 돌리나"
CHANGES = "무엇을 바꿨나"
INTERPRETATION = "해석"
NEXT = "다음"
BEFORE_RUN = (WHY, CHANGES)
AFTER_RUN = (INTERPRETATION, NEXT)

# STAGE.md 는 결론이 맨 위에 오는 순서다: 질문 → 답 → 근거 → 다음 → (세부) → 리뷰.
STAGE_QUESTION = "질문"
STAGE_ANSWER = "답"
STAGE_EVIDENCE = "근거"
STAGE_PROPOSAL = "다음 단계 제안"
STAGE_DETAIL = "세부 해석"
STAGE_REVIEWS = "리뷰 기록"
STAGE_AFTER_RUN = (STAGE_ANSWER, STAGE_EVIDENCE, STAGE_PROPOSAL)

VERDICTS = ("approve", "revise")

_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_COMPARES_RE = re.compile(r"^-\s*비교 대상\s*:\s*(.*)$", re.MULTILINE)
_STAGE_NUMBER_RE = re.compile(r"^s(\d+)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"^(.+?[.?!])(?:\s|$)")

RUN_TEMPLATE = """# {display}

- id: `{run_id}`
- 단계: `{stage}`
- 비교 대상:
- 실행 명령:

## {why}
<!-- 실행 전에 쓴다. 가설, 예상하는 결과, 결과가 어떻게 나오면 무엇을 할지. -->

## {changes}
<!-- 실행 전에 쓴다. 비교 대상 run 대비 달라지는 것(코드·설정·데이터). -->

## {interpretation}
<!-- 실행 후에 쓴다. 가설이 맞았나. 믿을 수 있는 것과 없는 것. -->

## {next}
<!-- 실행 후에 쓴다. 이 결과로 정해진 다음 실험, 그리고 버린 선택지. -->
"""

STAGE_TEMPLATE = """# {display}

- id: `{stage}`

## {question}
<!-- 실행 전에 쓴다. 이 단계가 답하려는 질문 한두 문장, 배경, 각 run이 맡는 부분. -->

## {answer}
<!-- 실행 후. 질문에 대한 답 한두 줄. 첫 줄이 실험 대장(EXPERIMENTS.md)에 그대로 들어간다. -->

## {evidence}
<!-- 실행 후. 답을 받치는 숫자 3~5개만. 표 칸: 질문 | 기준 | 결과 | 판정 -->

## {proposal}
<!-- 실행 후. 리뷰어가 정할 것과 제안. 리뷰에서 확정된다. -->

## {detail}
<!-- 선택. 예상과 달랐던 점, 믿을 수 있는 것과 없는 것. -->

## {reviews}
"""


def _stage_cmd() -> str:
    """안내 문구에 쓰는 단계 관리 명령. 트랙이 안 정해졌으면 자리표시로 둔다."""
    track = tracks.current(required=False)
    return f"python -m {tracks.module(track)}.stage" if track else "python -m <트랙>.exp.stage"


class GateError(SystemExit):
    """노트·리뷰 게이트에 걸렸다. 러너를 즉시 멈춘다."""

    def __init__(self, message: str) -> None:
        super().__init__(f"[gate] {message}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_id(value: str) -> str:
    if not value or not _ID_RE.match(value):
        raise ValueError(f"ID는 [A-Za-z0-9._-]+ 이어야 합니다: {value!r}")
    return value


def sections(text: str) -> dict[str, str]:
    """`## 제목` 단위로 본문을 자른다. 안내 주석(<!-- -->)은 내용으로 치지 않는다."""
    out: dict[str, str] = {}
    matches = list(_HEADING_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        out[match.group(1)] = _COMMENT_RE.sub("", text[match.end():end]).strip()
    return out


def missing(text: str, names: tuple[str, ...]) -> list[str]:
    found = sections(text)
    return [name for name in names if not found.get(name)]


def compares_to(text: str) -> str:
    match = _COMPARES_RE.search(text)
    return match.group(1).replace("`", "").strip() if match else ""


def prereg_digest(text: str) -> str:
    """실행 전에 쓰는 칸들만의 해시. 실행 후 그 칸이 바뀌었는지 가려낸다."""
    found = sections(text)
    payload = "\n\n".join(f"## {name}\n{found.get(name, '')}" for name in BEFORE_RUN)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Stage:
    directory: Path
    data: dict[str, Any]

    @property
    def stage_id(self) -> str:
        return self.data["stage"]

    @property
    def runs(self) -> list[str]:
        return self.data["runs"]

    @property
    def code(self) -> str:
        """`s02-...` → `S02`. 번호가 없는 id는 그대로."""
        match = _STAGE_NUMBER_RE.match(self.stage_id)
        return f"S{int(match.group(1)):02d}" if match else self.stage_id

    @property
    def title(self) -> str:
        return self.data.get("title") or ""

    @property
    def display_name(self) -> str:
        return f"{self.code} {self.title}".strip()

    def run_title(self, run_id: str) -> str:
        return (self.data.get("titles") or {}).get(run_id, "")

    def run_display_name(self, run_id: str) -> str:
        """`S02.2 기법군 빼고 학습`. 제목이 없으면 run-id를 붙인다."""
        number = f"{self.code}.{self.runs.index(run_id) + 1}"
        return f"{number} {self.run_title(run_id) or run_id}"

    @property
    def sealed(self) -> bool:
        return bool(self.data.get("sealed_at"))

    @property
    def last_verdict(self) -> str | None:
        reviews = self.data.get("reviews") or []
        return reviews[-1]["verdict"] if reviews else None

    @property
    def approved(self) -> bool:
        return self.last_verdict == "approve"

    def note_path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.md"

    def doc_path(self) -> Path:
        return self.directory / STAGE_DOC

    def finished(self, results_dir: Path | None = None) -> list[str]:
        base = Path(results_dir) if results_dir else tracks.results_dir()
        return [r for r in self.runs if (base / r / "manifest.json").exists()]

    def state(self, results_dir: Path | None = None) -> str:
        if self.approved:
            return "approved"
        if not self.sealed:
            return "planned"
        if len(self.finished(results_dir)) < len(self.runs):
            return "running"
        return "awaiting_review"

    def save(self) -> None:
        tmp = self.directory / (STAGE_FILE + ".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(self.directory / STAGE_FILE)


def load_stages(notes_dir: Path | None = None) -> list[Stage]:
    base = Path(notes_dir) if notes_dir else tracks.notes_dir()
    stages = []
    if base.exists():
        for path in sorted(base.glob(f"*/{STAGE_FILE}")):
            stages.append(Stage(path.parent, json.loads(path.read_text(encoding="utf-8"))))
    return stages


def get_stage(stage_id: str, notes_dir: Path | None = None) -> Stage:
    for stage in load_stages(notes_dir):
        if stage.stage_id == stage_id:
            return stage
    raise GateError(f"단계 '{stage_id}' 가 없습니다. '{_stage_cmd()} status' 로 확인하세요.")


def find_run(run_id: str, notes_dir: Path | None = None) -> Stage | None:
    for stage in load_stages(notes_dir):
        if run_id in stage.runs:
            return stage
    return None


def new_stage(
    stage_id: str, run_ids: list[str], notes_dir: Path | None = None,
    title: str = "", run_titles: list[str] | None = None,
) -> Stage:
    base = Path(notes_dir) if notes_dir else tracks.notes_dir()
    validate_id(stage_id)
    directory = base / stage_id
    if directory.exists():
        raise GateError(f"단계 디렉터리가 이미 있습니다: {directory}")
    for run_id in run_ids:
        validate_id(run_id)
        owner = find_run(run_id, base)
        if owner is not None:
            raise GateError(f"run '{run_id}' 는 이미 단계 '{owner.stage_id}' 에 있습니다.")
    directory.mkdir(parents=True)
    titles = list(run_titles or [])
    if titles and len(titles) != len(run_ids):
        raise GateError("run 제목 수가 run 수와 다릅니다.")
    stage = Stage(directory, {
        "stage": stage_id, "title": title.strip(), "titles": {},
        "created_at": _now(), "runs": [],
        "sealed_at": None, "added": [], "reviews": [],
    })
    stage.doc_path().write_text(
        STAGE_TEMPLATE.format(
            display=stage.display_name, stage=stage_id, question=STAGE_QUESTION,
            answer=STAGE_ANSWER, evidence=STAGE_EVIDENCE, proposal=STAGE_PROPOSAL,
            detail=STAGE_DETAIL, reviews=STAGE_REVIEWS,
        ),
        encoding="utf-8",
    )
    for index, run_id in enumerate(run_ids):
        _attach(stage, run_id, titles[index] if titles else "")
    stage.save()
    return stage


def _attach(stage: Stage, run_id: str, title: str = "") -> None:
    stage.runs.append(run_id)
    if title.strip():
        stage.data.setdefault("titles", {})[run_id] = title.strip()
    path = stage.note_path(run_id)
    if not path.exists():
        path.write_text(
            RUN_TEMPLATE.format(
                display=stage.run_display_name(run_id), run_id=run_id,
                stage=stage.stage_id, why=WHY, changes=CHANGES,
                interpretation=INTERPRETATION, next=NEXT,
            ),
            encoding="utf-8",
        )


def add_run(
    stage_id: str, run_id: str, reason: str = "",
    notes_dir: Path | None = None, results_dir: Path | None = None, title: str = "",
) -> Stage:
    """단계에 run을 추가한다. 봉인된 뒤라면 사유와 그 시점에 끝나 있던 run을 남긴다."""
    validate_id(run_id)
    stage = get_stage(stage_id, notes_dir)
    owner = find_run(run_id, notes_dir)
    if owner is not None:
        raise GateError(f"run '{run_id}' 는 이미 단계 '{owner.stage_id}' 에 있습니다.")
    if stage.approved:
        raise GateError(
            f"단계 '{stage_id}' 는 이미 승인되어 닫혔습니다. 새 단계를 만드세요."
        )
    if stage.sealed:
        if not reason.strip():
            raise GateError(
                "이미 시작된 단계에 run을 추가하려면 --reason 이 필요합니다 "
                "(리뷰 때 '결과를 본 뒤 추가된 run'으로 표시됩니다)."
            )
        stage.data["added"].append({
            "run_id": run_id, "at": _now(), "reason": reason.strip(),
            "finished_then": stage.finished(results_dir),
        })
    _attach(stage, run_id, title)
    stage.save()
    return stage


def set_title(
    stage_id: str, title: str, run_id: str | None = None, notes_dir: Path | None = None
) -> Stage:
    """사람이 보는 제목을 붙이거나 바꾼다. id·폴더·결과 경로는 그대로다."""
    stage = get_stage(stage_id, notes_dir)
    if not title.strip():
        raise GateError("제목이 비어 있습니다.")
    if run_id is None:
        stage.data["title"] = title.strip()
    else:
        if run_id not in stage.runs:
            raise GateError(f"run '{run_id}' 는 단계 '{stage_id}' 에 없습니다.")
        stage.data.setdefault("titles", {})[run_id] = title.strip()
    stage.save()
    return stage


def first_line(text: str, sentence: bool = True) -> str:
    """섹션 본문의 첫 문장(목록 기호·강조 제거). 실험 대장 한 칸에 넣는다.

    문단이 여러 줄로 감겨 있으면 이어 붙인 뒤 첫 문장에서 자른다. sentence=False면 첫 문단 전체.
    """
    paragraph: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if paragraph:
                break
            continue
        if paragraph and line[:1] in "-*>|":
            break
        paragraph.append(line.lstrip("-*> ").strip())
    joined = " ".join(paragraph).replace("**", "").replace("|", "/")
    match = _SENTENCE_RE.match(joined) if sentence else None
    return match.group(1) if match else joined


@dataclass
class StartTicket:
    """게이트를 통과한 run이 들고 가는 것."""

    stage_id: str
    note_text: str
    note_path: Path
    compares_to: str
    prereg_sha256: str
    run_display_name: str = ""
    stage_display_name: str = ""
    stage_note: str = ""


def check_start(
    run_id: str, notes_dir: Path | None = None, results_dir: Path | None = None
) -> StartTicket:
    """run을 시작해도 되는지 확인하고, 단계가 아직 안 봉인됐으면 봉인한다."""
    stage = find_run(run_id, notes_dir)
    if stage is None:
        raise GateError(
            f"run '{run_id}' 의 노트가 없습니다. 먼저 단계에 등록하세요:\n"
            f"    {_stage_cmd()} new <단계-id> --runs {run_id}\n"
            "그리고 노트의 '왜 돌리나'·'무엇을 바꿨나'를 채운 뒤 다시 실행합니다.\n"
            "(파이프라인 점검용이면 run-id에 'smoke'를 넣으면 게이트를 건너뜁니다.)"
        )
    path = stage.note_path(run_id)
    if not path.exists():
        raise GateError(f"노트 파일이 없습니다: {path}")
    text = path.read_text(encoding="utf-8")
    empty = missing(text, BEFORE_RUN)
    if empty:
        raise GateError(
            f"{path} 의 다음 칸이 비어 있습니다: {', '.join(empty)}\n"
            "실행 전에 쓰는 칸입니다. 채운 뒤 다시 실행하세요."
        )
    stage_doc = stage.doc_path().read_text(encoding="utf-8")
    if not sections(stage_doc).get(STAGE_QUESTION):
        raise GateError(f"{stage.doc_path()} 의 '{STAGE_QUESTION}' 이 비어 있습니다.")
    if stage.approved:
        raise GateError(
            f"단계 '{stage.stage_id}' 는 이미 승인되어 닫혔습니다. 새 단계를 만드세요."
        )
    blocking = [
        other for other in load_stages(notes_dir)
        if other.stage_id != stage.stage_id and other.sealed and not other.approved
    ]
    if blocking:
        lines = [f"    {s.stage_id}  ({s.state(results_dir)})" for s in blocking]
        raise GateError(
            "리뷰가 끝나지 않은 단계가 있어 새 단계를 시작할 수 없습니다:\n"
            + "\n".join(lines)
            + f"\n'{_stage_cmd()} show <단계-id>' 로 리뷰 자료를 보고, 사용자 피드백을 "
            "받아 기록한 뒤 진행하세요."
        )
    if not stage.sealed:
        stage.data["sealed_at"] = _now()
        stage.save()
    return StartTicket(
        stage.stage_id, text, path, compares_to(text), prereg_digest(text),
        stage.run_display_name(run_id), stage.display_name, stage_doc,
    )


def review_blockers(stage: Stage, results_dir: Path | None = None) -> list[str]:
    """리뷰를 기록하기 전에 채워져 있어야 하는 것."""
    problems: list[str] = []
    if not stage.sealed:
        problems.append("아직 시작되지 않은 단계입니다.")
    finished = set(stage.finished(results_dir))
    for run_id in stage.runs:
        path = stage.note_path(run_id)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        empty = missing(text, AFTER_RUN)
        if empty:
            problems.append(f"{run_id}: 노트의 {', '.join(empty)} 칸이 비어 있습니다.")
        if run_id not in finished and not empty:
            # 끝나지 않은 run도 리뷰는 받을 수 있다(포기·실패). 다만 노트에 그 사정이 적혀야 한다.
            pass
    doc = sections(stage.doc_path().read_text(encoding="utf-8"))
    for name in STAGE_AFTER_RUN:
        if not doc.get(name):
            problems.append(f"{STAGE_DOC}: '{name}' 칸이 비어 있습니다.")
    return problems


def record_review(
    stage_id: str, verdict: str, feedback: str,
    notes_dir: Path | None = None, results_dir: Path | None = None,
) -> Stage:
    if verdict not in VERDICTS:
        raise GateError(f"verdict 는 {VERDICTS} 중 하나여야 합니다.")
    if not feedback.strip():
        raise GateError("피드백이 비어 있습니다. 리뷰어의 말을 그대로 적습니다.")
    stage = get_stage(stage_id, notes_dir)
    if stage.approved:
        raise GateError(f"단계 '{stage_id}' 는 이미 승인되었습니다.")
    problems = review_blockers(stage, results_dir)
    if problems:
        raise GateError("리뷰 자료가 덜 준비됐습니다:\n  - " + "\n  - ".join(problems))
    entry = {
        "at": _now(), "verdict": verdict, "feedback": feedback.strip(),
        "runs_reviewed": list(stage.runs),
        "unfinished": [r for r in stage.runs if r not in stage.finished(results_dir)],
    }
    stage.data["reviews"].append(entry)
    stage.save()
    with stage.doc_path().open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n### {entry['at']} — {verdict}\n\n"
            + "\n".join(f"> {line}" for line in entry["feedback"].splitlines())
            + "\n"
        )
    return stage


def prereg_changed(stage: Stage, run_id: str, results_dir: Path | None = None) -> bool | None:
    """실행 전 칸이 실행 뒤에 바뀌었나. 시작 시점 사본이 없으면 None."""
    base = Path(results_dir) if results_dir else tracks.results_dir()
    snapshot = base / run_id / "note-at-start.md"
    path = stage.note_path(run_id)
    if not snapshot.exists() or not path.exists():
        return None
    return prereg_digest(snapshot.read_text(encoding="utf-8")) != prereg_digest(
        path.read_text(encoding="utf-8")
    )


def __getattr__(name: str) -> Path:
    """`N.NOTES_DIR`·`N.RESULTS_DIR`는 지금 트랙의 폴더를 가리킨다."""
    if name == "NOTES_DIR":
        return tracks.notes_dir()
    if name == "RESULTS_DIR":
        return tracks.results_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
