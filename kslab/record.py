"""run 기록 유틸 — manifest, 결정론적 JSON, 경로 격리 검사.

k-safeguard `f6ee374`의 `experiments/benchmark/run_clean_baseline.py`가 쓰는 provenance 관행
(git commit + dirty, 입력 파일 SHA-256, run-id 덮어쓰기 거부)을 그대로 옮겼다.

경로 격리는 이 샌드박스 고유 규칙이다. `datasets`/`torch`의 기본 캐시는 사용자 홈이라,
`env.ps1`을 건너뛰고 러너를 돌리면 폴더 밖에 수 GB가 조용히 쌓인다. 그래서 러너는 시작할 때
`assert_paths_contained()`로 확인하고 아니면 즉시 멈춘다.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kslab import notes, tracking, tracks

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KSAFEGUARD_ROOT = ROOT.parent / "k-safeguard"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# 폴더 밖으로 새면 안 되는 캐시 환경변수.
_CONTAINED_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME")


def validate_run_id(run_id: str) -> str:
    if not run_id or not _RUN_ID_RE.match(run_id):
        raise ValueError(f"run-id는 [A-Za-z0-9._-]+ 이어야 합니다: {run_id!r}")
    return run_id


def assert_paths_contained(strict: bool = True) -> dict[str, str]:
    """캐시 경로가 전부 저장소 하위인지 확인. 아니면 예외.

    strict=False면 미설정 변수를 허용한다(그 라이브러리를 안 쓰는 실험용).
    """
    root = ROOT.resolve()
    report: dict[str, str] = {}
    problems: list[str] = []
    for name in _CONTAINED_ENV:
        value = os.environ.get(name)
        if not value:
            report[name] = "<unset>"
            if strict:
                problems.append(f"{name} 미설정")
            continue
        resolved = Path(value).resolve()
        report[name] = str(resolved)
        if root not in resolved.parents and resolved != root:
            problems.append(f"{name}={resolved} 가 {root} 밖입니다")
    if problems:
        raise RuntimeError(
            "캐시 경로가 샌드박스를 벗어납니다. 먼저 '. ./env.ps1' 을 실행하세요.\n  - "
            + "\n  - ".join(problems)
        )
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_metadata(repo: Path) -> dict[str, Any]:
    """commit / branch / dirty. 저장소가 아니면 available=False."""
    repo = Path(repo)
    inside = _git(["rev-parse", "--is-inside-work-tree"], repo)
    if inside != "true":
        return {"available": False, "reason": "not_a_git_repo"}
    commit = _git(["rev-parse", "HEAD"], repo)
    if commit is None:
        # 저장소는 있는데 커밋이 아직 없다 — git init 직후 상태.
        return {"available": False, "reason": "no_commits_yet"}
    status = _git(["status", "--porcelain"], repo)
    return {
        "available": True,
        "commit": commit,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], repo),
        "dirty": bool(status),
    }


def working_tree_patch(repo: Path, max_bytes: int = 5 * 1024 * 1024) -> str | None:
    """커밋되지 않은 변경 전부(추적 안 된 새 파일 포함)를 patch 하나로. 깨끗하면 None.

    dirty 상태로 돌린 run은 커밋 해시만으로 코드를 되살릴 수 없다. 이게 있어야 한다.
    """
    def run(args: list[str]) -> str:
        done = subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        return done.stdout

    try:
        parts = [run(["diff", "HEAD"])]
        for name in run(["ls-files", "--others", "--exclude-standard"]).splitlines():
            # --no-index 는 차이가 있으면 종료코드 1을 낸다. 정상이다.
            parts.append(run(["diff", "--no-index", "--", os.devnull, name]))
    except OSError:
        return None
    patch = "".join(part for part in parts if part)
    if not patch:
        return None
    if len(patch.encode("utf-8")) > max_bytes:
        return patch.encode("utf-8")[:max_bytes].decode("utf-8", "ignore") + "\n# (잘림)\n"
    return patch


def pip_freeze() -> str:
    from importlib import metadata

    rows = sorted(
        {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
         if d.metadata["Name"]},
        key=str.lower,
    )
    return "\n".join(rows) + "\n"


class _Tee:
    """stdout을 그대로 흘리면서 run.log 에도 남긴다."""

    def __init__(self, stream, path: Path) -> None:
        self.stream = stream
        self.handle = path.open("a", encoding="utf-8")

    # 한 프로세스에서 run이 겹쳐 열리면 tee가 사슬로 이어지고 닫히는 순서도 섞인다.
    # 그래서 이미 닫힌 로그는 건너뛰고 아래 스트림으로만 흘린다.
    def write(self, text: str) -> int:
        if not self.handle.closed:
            self.handle.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
        self.stream.flush()

    def close(self) -> None:
        self.handle.close()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "sklearn", "torch", "onnxruntime", "datasets", "mlflow"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "?")
    return versions


def dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in Path(path).rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


SMOKE_MARKERS = ("smoke", "trial", "scratch")


@dataclass
class Run:
    """run 하나의 산출물 디렉터리와 manifest를 관리한다."""

    run_id: str
    experiment: str
    config: dict[str, Any]
    started_at: float
    directory: Path
    manifest: dict[str, Any]
    tracker: "tracking.Tracker | None" = field(default=None, repr=False, compare=False)
    _tee: "_Tee | None" = field(default=None, repr=False, compare=False)

    @classmethod
    def start(
        cls,
        run_id: str,
        experiment: str,
        config: dict[str, Any],
        *,
        strict_paths: bool = True,
        results_dir: Path | None = None,
        notes_dir: Path | None = None,
    ) -> "Run":
        validate_run_id(run_id)
        paths = assert_paths_contained(strict=strict_paths)
        base = Path(results_dir) if results_dir else tracks.results_dir()
        directory = base / run_id
        if directory.exists():
            raise SystemExit(f"동일 run ID 결과가 이미 있습니다: {directory}")
        smoke = tracking.is_smoke(run_id, SMOKE_MARKERS)
        # 노트·리뷰 게이트. 정식 결과 위치에 쓰는 run은 전부 거친다 — 테스트처럼 결과를
        # 다른 곳에 쓰는 경우만 notes_dir 를 줘야 켜진다.
        ticket = None
        if not smoke and (results_dir is None or notes_dir is not None):
            ticket = notes.check_start(run_id, notes_dir, results_dir)
        directory.mkdir(parents=True)
        tracking.reset_fits()
        manifest = {
            "run_id": run_id,
            "experiment": experiment,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config,
            "sandbox_git": git_metadata(ROOT),
            "ksafeguard_git": git_metadata(KSAFEGUARD_ROOT),
            "packages": package_versions(),
            "cache_paths": paths,
            "host": {
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "python_impl": platform.python_implementation(),
            },
        }
        sandbox, upstream = manifest["sandbox_git"], manifest["ksafeguard_git"]
        patch = working_tree_patch(ROOT) if sandbox.get("dirty") else None
        if patch:
            (directory / "code.diff").write_text(patch, encoding="utf-8")
        (directory / "pip-freeze.txt").write_text(pip_freeze(), encoding="utf-8")
        description = tracking.main_docstring()
        note_tags: dict[str, Any] = {}
        display_name, stage = run_id, None
        if ticket is not None:
            # 실행 전 칸이 나중에 바뀌었는지 가릴 수 있게 시작 시점 사본과 해시를 남긴다.
            (directory / "note-at-start.md").write_text(ticket.note_text, encoding="utf-8")
            manifest["note"] = {
                "stage": ticket.stage_id,
                "path": str(ticket.note_path),
                "compares_to": ticket.compares_to,
                "prereg_sha256": ticket.prereg_sha256,
            }
            description = notes.for_display(ticket.note_text)
            note_tags = {
                "kslab.stage": ticket.stage_id,
                "kslab.compares_to": ticket.compares_to or None,
                "note.prereg_sha256": ticket.prereg_sha256,
                "note.interpreted": "false",
                "review.status": "pending",
            }
            display_name = ticket.run_display_name or run_id
            stage = {"id": ticket.stage_id, "display_name": ticket.stage_display_name,
                     "note": notes.for_display(ticket.stage_note),
                     "tags": {"kslab.track": tracks.current(required=False)}}
        tracker = tracking.Tracker.start(
            display_name,
            tracking.experiment_name(tracks.current(required=False), smoke),
            config,
            stage=stage,
            tags={
                **note_tags,
                "kslab.run_id": run_id,
                "kslab.track": tracks.current(required=False),
                "kslab.code_diff": "code.diff" if patch else None,
                "kslab.experiment": experiment,
                "kslab.command": " ".join(sys.argv),
                "kslab.results_dir": str(directory),
                "kslab.git_commit": sandbox.get("commit"),
                "kslab.git_dirty": sandbox.get("dirty"),
                "ksafeguard.git_commit": upstream.get("commit"),
                "ksafeguard.git_dirty": upstream.get("dirty"),
                "host.platform": manifest["host"]["platform"],
                "host.cpu_count": manifest["host"]["cpu_count"],
                **{f"pkg.{k}": v for k, v in manifest["packages"].items()},
            },
            description=description,
        )
        tee = _Tee(sys.stdout, directory / "run.log")
        sys.stdout = tee
        return cls(run_id, experiment, config, time.time(), directory, manifest, tracker, tee)

    def add_input(self, name: str, path: Path, rows: int | None = None) -> None:
        entry: dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
        if rows is not None:
            entry["rows"] = rows
        self.manifest.setdefault("inputs", {})[name] = entry

    def note(self, key: str, value: Any) -> None:
        self.manifest.setdefault("notes", {})[key] = value

    def write_json(self, name: str, payload: Any) -> Path:
        """정렬된 결정론적 JSON. 같은 시드로 두 번 돌리면 바이트 단위로 같아야 한다."""
        path = self.directory / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.directory / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def finish(self, metrics: dict[str, Any], headline: dict[str, float] | None = None) -> Path:
        """run을 닫는다.

        headline: 이 run을 한 줄로 요약하는 수치 몇 개. metrics 전체는 수천 개라 run끼리
        비교할 때 볼 것을 찾기 어렵다 — 러너가 "이것만 보면 된다"를 직접 고른다.
        """
        self.manifest["elapsed_seconds"] = round(time.time() - self.started_at, 3)
        if tracking.fits():
            self.manifest["fits"] = tracking.fits()
        if headline:
            self.manifest["headline"] = headline
        if self._tee is not None:
            if sys.stdout is self._tee:
                sys.stdout = self._tee.stream
            self._tee.close()
            self._tee = None
        if DATA_DIR.exists():
            self.manifest["data_dir_bytes"] = dir_size_bytes(DATA_DIR)
        self.write_json("metrics.json", metrics)
        path = self.write_json("manifest.json", self.manifest)
        if self.tracker is not None:
            self.tracker.finish(
                {**metrics, "headline": headline} if headline else metrics,
                inputs=self.manifest.get("inputs"),
                directory=self.directory,
                tags={f"note.{k}": v for k, v in self.manifest.get("notes", {}).items()},
            )
        return path


def publish_baseline(name: str, payload: dict[str, Any], run_id: str | None = None) -> Path | None:
    """집계 결과만 baselines/ 로 커밋한다. 원문이 든 산출물은 results/ 에 남는다.

    run_id에 smoke 표식이 있으면 발행하지 않는다. 스모크 실행이 진짜 baseline을 덮어쓰고
    나중에 그게 정식 수치인 줄 알고 읽는 사고를 막는다.
    """
    if run_id and any(marker in run_id.lower() for marker in SMOKE_MARKERS):
        print(f"[baseline] '{run_id}'는 스모크 실행이라 {name}을 발행하지 않습니다.")
        return None
    directory = tracks.baselines_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",  # 커밋되는 파일이라 Windows에서도 LF로 쓴다
    )
    return path


def append_ledger(row: str, ledger: Path | None = None) -> None:
    """트랙의 EXPERIMENTS.md 에 한 줄 추가."""
    path = Path(ledger) if ledger else tracks.ledger()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row.rstrip() + "\n")


def __getattr__(name: str) -> Path:
    """`R.RESULTS_DIR` 등은 지금 트랙의 폴더를 가리킨다(트랙 분리 전 러너 호환)."""
    lookup = {
        "RESULTS_DIR": tracks.results_dir,
        "BASELINES_DIR": tracks.baselines_dir,
        "REPORTS_DIR": tracks.reports_dir,
    }
    if name in lookup:
        return lookup[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
