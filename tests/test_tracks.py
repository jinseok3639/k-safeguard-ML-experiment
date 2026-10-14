"""실험 트랙(복원기·가드레일) 분리 테스트."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import notes as N
from kslab import record as R
from kslab import tracks as T


def _fill(path: Path, names: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    for name in names:
        text = text.replace(f"## {name}\n", f"## {name}\n내용\n", 1)
    path.write_text(text, encoding="utf-8")


class TrackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patches = [
            mock.patch.object(T, "ROOT", self.tmp),
            mock.patch.object(T, "_current", None),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        os.environ.pop("KSLAB_TRACK", None)

    def test_unset_track_is_an_error_with_a_hint(self):
        with self.assertRaises(RuntimeError) as caught:
            T.current()
        self.assertIn("restoration.exp", str(caught.exception))
        self.assertIsNone(T.current(required=False))

    def test_unknown_track_is_refused(self):
        with self.assertRaises(ValueError):
            T.use("restore")

    def test_paths_follow_the_track(self):
        T.use("guardrail")
        self.assertEqual(T.results_dir(), self.tmp / "guardrail" / "results")
        self.assertEqual(T.ledger(), self.tmp / "guardrail" / "EXPERIMENTS.md")
        self.assertEqual(R.BASELINES_DIR, self.tmp / "guardrail" / "baselines")
        self.assertEqual(N.NOTES_DIR, self.tmp / "guardrail" / "notes")

    def test_env_overrides_runner_package(self):
        T.use("restoration")
        os.environ["KSLAB_TRACK"] = "guardrail"
        self.assertEqual(T.current(), "guardrail")

    def test_gates_are_independent_per_track(self):
        """복원기 단계가 리뷰를 기다려도 가드레일 단계는 시작된다."""
        def ready_stage(track: str, stage_id: str, run_id: str) -> N.Stage:
            stage = N.new_stage(stage_id, [run_id], T.notes_dir(track))
            _fill(stage.doc_path(), (N.STAGE_QUESTION,))
            _fill(stage.note_path(run_id), N.BEFORE_RUN)
            return stage

        ready_stage("restoration", "s01", "r-a")
        ready_stage("restoration", "s02", "r-b")
        ready_stage("guardrail", "g01", "g-a")

        T.use("restoration")
        N.check_start("r-a")  # s01 봉인, 리뷰 대기
        with self.assertRaises(N.GateError):
            N.check_start("r-b")  # 같은 트랙의 다음 단계는 막힌다

        T.use("guardrail")
        ticket = N.check_start("g-a")  # 다른 트랙은 막히지 않는다
        self.assertEqual(ticket.stage_id, "g01")


if __name__ == "__main__":
    unittest.main()
