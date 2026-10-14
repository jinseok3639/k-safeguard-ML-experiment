"""노트·단계 리뷰 게이트 테스트."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import notes as N
from kslab import record as R


def _fill(path: Path, names: tuple[str, ...], body: str = "내용") -> None:
    text = path.read_text(encoding="utf-8")
    for name in names:
        text = text.replace(f"## {name}\n", f"## {name}\n{body}\n", 1)
    path.write_text(text, encoding="utf-8")


class SectionTest(unittest.TestCase):
    def test_template_comments_do_not_count_as_content(self):
        text = N.RUN_TEMPLATE.format(
            run_id="r", stage="s", why=N.WHY, changes=N.CHANGES,
            interpretation=N.INTERPRETATION, next=N.NEXT,
        )
        self.assertEqual(N.missing(text, N.BEFORE_RUN + N.AFTER_RUN),
                         [N.WHY, N.CHANGES, N.INTERPRETATION, N.NEXT])

    def test_compares_to_and_prereg_digest(self):
        text = (f"# r\n- 비교 대상: `e8-mlp-30k`\n\n## {N.WHY}\n가설\n\n## {N.CHANGES}\n변경\n\n"
                f"## {N.INTERPRETATION}\n\n## {N.NEXT}\n")
        self.assertEqual(N.compares_to(text), "e8-mlp-30k")
        later = text.replace(f"## {N.INTERPRETATION}\n", f"## {N.INTERPRETATION}\n해석함\n")
        self.assertEqual(N.prereg_digest(text), N.prereg_digest(later))
        rewritten = text.replace("가설", "결과에 맞춘 가설")
        self.assertNotEqual(N.prereg_digest(text), N.prereg_digest(rewritten))


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.notes = self.tmp / "notes"
        self.results = self.tmp / "results"

    def _stage(self, stage_id: str, runs: list[str], ready: bool = True) -> N.Stage:
        stage = N.new_stage(stage_id, runs, self.notes)
        if ready:
            _fill(stage.doc_path(), (N.STAGE_PURPOSE,))
            for run_id in runs:
                _fill(stage.note_path(run_id), N.BEFORE_RUN)
        return stage

    def _run(self, run_id: str) -> R.Run:
        return R.Run.start(run_id, "E_unit", {}, strict_paths=False,
                           results_dir=self.results, notes_dir=self.notes)

    def _complete(self, stage_id: str) -> None:
        stage = N.get_stage(stage_id, self.notes)
        for run_id in stage.runs:
            _fill(stage.note_path(run_id), N.AFTER_RUN)
        _fill(stage.doc_path(), (N.STAGE_SYNTHESIS, N.STAGE_PROPOSAL))

    def test_run_without_note_is_refused(self):
        with self.assertRaises(N.GateError):
            self._run("orphan-run")
        self.assertFalse((self.results / "orphan-run").exists())

    def test_run_with_empty_why_is_refused(self):
        self._stage("s1", ["a"], ready=False)
        with self.assertRaises(N.GateError) as caught:
            self._run("a")
        self.assertIn(N.WHY, str(caught.exception))

    def test_smoke_run_skips_the_gate(self):
        self._run("pipeline-smoke").finish({})

    def test_start_seals_stage_and_snapshots_note(self):
        self._stage("s1", ["a", "b"])
        run = self._run("a")
        self.assertTrue(N.get_stage("s1", self.notes).sealed)
        self.assertEqual(run.manifest["note"]["stage"], "s1")
        self.assertTrue((run.directory / "note-at-start.md").exists())
        run.finish({})

    def test_parallel_runs_in_one_stage_are_allowed(self):
        self._stage("s1", ["a", "b"])
        first, second = self._run("a"), self._run("b")
        first.finish({})
        second.finish({})
        self.assertEqual(N.get_stage("s1", self.notes).state(self.results), "awaiting_review")

    def test_next_stage_is_blocked_until_approved(self):
        self._stage("s1", ["a"])
        self._run("a").finish({})
        self._stage("s2", ["b"])
        with self.assertRaises(N.GateError) as caught:
            self._run("b")
        self.assertIn("s1", str(caught.exception))

        self._complete("s1")
        N.record_review("s1", "revise", "임계값을 다시 잡아라", self.notes, self.results)
        with self.assertRaises(N.GateError):
            self._run("b")                      # revise 는 단계를 닫지 않는다

        N.record_review("s1", "approve", "확인", self.notes, self.results)
        self._run("b").finish({})

    def test_review_requires_interpretation_and_feedback(self):
        self._stage("s1", ["a"])
        self._run("a").finish({})
        with self.assertRaises(N.GateError) as caught:
            N.record_review("s1", "approve", "확인", self.notes, self.results)
        self.assertIn(N.INTERPRETATION, str(caught.exception))
        self._complete("s1")
        with self.assertRaises(N.GateError):
            N.record_review("s1", "approve", "   ", self.notes, self.results)
        stage = N.record_review("s1", "approve", "확인함", self.notes, self.results)
        self.assertIn("확인함", stage.doc_path().read_text(encoding="utf-8"))

    def test_approved_stage_is_closed(self):
        self._stage("s1", ["a"])
        self._run("a").finish({})
        self._complete("s1")
        N.record_review("s1", "approve", "확인", self.notes, self.results)
        with self.assertRaises(N.GateError):
            N.add_run("s1", "a2", "더 돌리고 싶음", self.notes, self.results)

    def test_run_added_after_start_is_logged_with_what_was_already_known(self):
        self._stage("s1", ["a"])
        self._run("a").finish({})
        with self.assertRaises(N.GateError):
            N.add_run("s1", "a2", "", self.notes, self.results)      # 사유 없이는 안 된다
        stage = N.add_run("s1", "a2", "a가 덜 수렴해서", self.notes, self.results)
        self.assertEqual(stage.data["added"][0]["finished_then"], ["a"])
        self.assertEqual(stage.state(self.results), "running")

    def test_rewriting_the_hypothesis_after_the_run_is_detected(self):
        stage = self._stage("s1", ["a"])
        self._run("a").finish({})
        self.assertFalse(N.prereg_changed(stage, "a", self.results))
        _fill(stage.note_path("a"), (N.WHY,), "결과에 맞춰 고친 가설")
        self.assertTrue(N.prereg_changed(stage, "a", self.results))

    def test_run_cannot_belong_to_two_stages(self):
        self._stage("s1", ["a"])
        with self.assertRaises(N.GateError):
            N.new_stage("s2", ["a"], self.notes)


class RunExtrasTest(unittest.TestCase):
    """게이트와 무관하게 모든 run에 남는 것."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_freeze_log_fits_and_headline_are_recorded(self):
        from kslab import tracking as T

        run = R.Run.start("unit-extras", "E_unit", {}, strict_paths=False,
                          results_dir=self.tmp)
        print("러너가 찍은 줄")

        class Model:
            technique, hidden, _private = "tensify", 112, "x"

            def __init__(self):
                self.hidden, self.epochs, self.net = 112, 5, object()

        log = T.begin_fit("bilstm", Model(), n_pairs=10)
        log.epoch(1, 0.5)
        again = T.begin_fit("bilstm", Model())
        manifest_path = run.finish({"x": 1}, headline={"best": 0.25})

        self.assertIn("러너가 찍은 줄", (run.directory / "run.log").read_text(encoding="utf-8"))
        self.assertIn("numpy==", (run.directory / "pip-freeze.txt").read_text(encoding="utf-8"))
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["headline"], {"best": 0.25})
        first = manifest["fits"][0]
        self.assertEqual(first["key"], "bilstm/tensify")
        self.assertEqual(first["hyperparameters"], {"hidden": 112, "epochs": 5, "n_pairs": 10})
        self.assertEqual(first["loss"], [[1, 0.5]])
        self.assertEqual(again.key, "bilstm/tensify.2")


if __name__ == "__main__":
    unittest.main()
