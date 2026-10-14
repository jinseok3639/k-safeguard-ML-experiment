"""MLflow 기록 테스트 — 임시 sqlite 저장소에 실제로 써 보고 읽어서 확인한다."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import record as R
from kslab import tracking as T

try:
    import mlflow  # noqa: F401
    HAVE_MLFLOW = True
except ImportError:
    HAVE_MLFLOW = False


class FlattenTest(unittest.TestCase):
    def test_nested_numbers_and_strings_are_split(self):
        numbers, texts = T.flatten(
            {"status": "OK", "a": {"b": 1, "c": {"d": 0.5}}, "flag": True}
        )
        self.assertEqual(numbers, {"a/b": 1.0, "a/c/d": 0.5, "flag": 1.0})
        self.assertEqual(texts, {"status": "OK"})

    def test_lists_none_and_non_finite_are_dropped(self):
        numbers, texts = T.flatten(
            {"passing": [{"x": 1}], "none": None, "nan": float("nan"), "inf": float("inf")}
        )
        self.assertEqual(numbers, {})
        self.assertEqual(texts, {})

    def test_invalid_metric_name_characters_are_replaced_without_collision(self):
        numbers, _ = T.flatten({"m": {"a+b": 1, "a b": 2, "a'b": 3, "가": 4}})
        self.assertEqual(numbers, {"m/a_b": 1.0, "m/a b": 2.0, "m/a_b_2": 3.0, "m/_": 4.0})

    def test_summary_drops_detail_tables_and_caps_size(self):
        metrics = {"status": "OK", "mean": 0.5, "evaluations": {"a": {"b": 1}},
                   "models": {"m": {"fit_seconds": 3}}}
        self.assertEqual(T.summary_metrics(metrics), {"mean": 0.5})
        big = {"grid": {str(i): float(i) for i in range(T.MAX_SUMMARY_METRICS + 1)}}
        self.assertEqual(T.summary_metrics({**big, "headline": {"best": 1.0}}),
                         {"headline/best": 1.0})
        rows = {**big, "passing": [{"arch": "mlp", "tech": "tensify", "cer": 0.1}]}
        self.assertEqual(T.summary_metrics(rows), {"passing/mlp_tensify/cer": 0.1})

    def test_fit_run_names(self):
        self.assertEqual(T.fit_run_name("bilstm", "a+b+c", None, 7), "bilstm · 합집합 3기법 · seed7")
        self.assertEqual(T.fit_run_name("bilstm", "a+b", "all/seed7", 7), "bilstm · all · seed7")
        self.assertEqual(T.fit_run_name("mlp", "tensify", None, None), "mlp · tensify")

    def test_params_keep_lists_as_one_value(self):
        params = T.flatten_params({"techniques": ["a", "b"], "n": 3, "sub": {"k": "v"}})
        self.assertEqual(params, {"techniques": '["a", "b"]', "n": "3", "sub.k": "v"})
        self.assertEqual(T.flatten_params({}), {})


class DisabledTest(unittest.TestCase):
    def test_no_uri_means_no_tracker(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLFLOW_TRACKING_URI", None)
            self.assertFalse(T.enabled())
            self.assertIsNone(T.Tracker.start("r", "E", {}, {}))

    def test_kill_switch(self):
        with mock.patch.dict(
            os.environ, {"MLFLOW_TRACKING_URI": "sqlite:///x.db", "KSLAB_MLFLOW": "0"}
        ):
            self.assertFalse(T.enabled())

    def test_store_outside_sandbox_is_rejected(self):
        with self.assertRaises(RuntimeError):
            T.assert_store_contained("sqlite:////tmp/elsewhere/mlflow.db")
        T.assert_store_contained(f"sqlite:///{T.MLFLOW_DIR / 'mlflow.db'}")
        T.assert_store_contained("http://tracking.example:5000")


@unittest.skipUnless(HAVE_MLFLOW, "mlflow 미설치")
class RunIntegrationTest(unittest.TestCase):
    """`record.Run` 을 거쳐 실제 MLflow 저장소에 무엇이 남는지 확인한다."""

    def setUp(self):
        # 저장소는 샌드박스 안이어야 하므로 data/ 아래에 임시 디렉터리를 만든다.
        base = R.DATA_DIR / "tmp-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=base))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.uri = f"sqlite:///{self.tmp / 'mlflow.db'}"
        patches = [
            mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": self.uri, "KSLAB_MLFLOW": "1",
                                         "KSLAB_TRACK": "restoration"}),
            mock.patch.object(T, "MLFLOW_DIR", self.tmp),
            mock.patch.object(T, "ARTIFACT_DIR", self.tmp / "artifacts"),
            # 임시 저장소는 테스트가 끝나면 지워지므로 종료 훅이 그걸 건드리지 않게 한다.
            mock.patch.object(T.atexit, "register"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _client(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient(tracking_uri=self.uri)

    def _runs(self, experiment: str) -> list:
        client = self._client()
        found = client.get_experiment_by_name(experiment)
        self.assertIsNotNone(found, f"experiment {experiment} 가 없다")
        return client.search_runs([found.experiment_id])

    def _by_kind(self, runs: list, kind: str | None) -> list:
        return [r for r in runs if r.data.tags.get("kslab.kind") == kind]

    def test_run_is_recorded_end_to_end(self):
        run = R.Run.start(
            "unit-20261007", "E_unit", {"corpus_size": 10, "techniques": ["tensify"]},
            strict_paths=False, results_dir=self.tmp / "results",
        )
        seed_file = self.tmp / "seeds.jsonl"
        seed_file.write_text("{}\n", encoding="utf-8")
        run.add_input("seeds", seed_file, rows=1)
        run.write_text("report.md", "# report\n")
        run.write_jsonl("predictions.jsonl", [{"text": "원문"}])
        from types import SimpleNamespace

        fit = T.begin_fit("mlp", SimpleNamespace(technique="tensify", hidden=64))
        fit.epoch(1, 0.9)
        fit.epoch(2, 0.4)
        with T.fit_label("minus_tensify/seed7"):
            union = T.begin_fit("bilstm", SimpleNamespace(technique="liaison+tensify",
                                                          hidden=8, random_state=7))
        union.epoch(1, 0.5)
        run.finish(
            {"status": "PROMOTE_CANDIDATE", "sweeps": {"tensify": {"cer": 0.25}},
             "evaluations": {"many": {"x": 1.0}}},
            headline={"tensify": {"best": 0.25}},
        )

        client = self._client()
        runs = self._runs("restoration")
        (logged,) = self._by_kind(runs, None)
        self.assertEqual(logged.info.run_name, "unit-20261007")
        self.assertEqual(logged.data.tags["kslab.run_id"], "unit-20261007")
        self.assertEqual(logged.info.status, "FINISHED")
        self.assertEqual(logged.data.params["corpus_size"], "10")
        self.assertEqual(logged.data.metrics["sweeps/tensify/cer"], 0.25)
        self.assertEqual(logged.data.tags["status"], "PROMOTE_CANDIDATE")
        self.assertEqual(logged.data.metrics["headline/tensify/best"], 0.25)
        # 세부 평가표는 지표로 올리지 않는다 — metrics.json artifact에 있다.
        self.assertNotIn("evaluations/many/x", logged.data.metrics)
        self.assertEqual(logged.data.tags["kslab.experiment"], "E_unit")
        self.assertEqual(
            [d.dataset.name for d in logged.inputs.dataset_inputs], ["seeds"]
        )
        # 학습 한 번 = 자식 run 하나. 하이퍼파라미터와 loss가 거기 있다.
        fits = {r.info.run_name: r for r in self._by_kind(runs, "fit")}
        self.assertEqual(set(fits), {"mlp · tensify", "bilstm · minus_tensify · seed7"})
        for child in fits.values():
            self.assertEqual(child.data.tags["mlflow.parentRunId"], logged.info.run_id)
            self.assertEqual(child.info.status, "FINISHED")
        self.assertEqual(fits["mlp · tensify"].data.params["hidden"], "64")
        history = client.get_metric_history(fits["mlp · tensify"].info.run_id, "loss")
        self.assertEqual([(m.step, m.value) for m in history], [(1, 0.9), (2, 0.4)])
        artifacts = {a.path for a in client.list_artifacts(logged.info.run_id)}
        self.assertLessEqual(
            {"report.md", "metrics.json", "manifest.json", "pip-freeze.txt"}, artifacts
        )
        # 원문이 든 산출물은 MLflow 저장소로 넘어가면 안 된다.
        self.assertNotIn("predictions.jsonl", artifacts)

    def test_staged_run_hangs_under_its_stage_with_display_name(self):
        from kslab import notes as N

        notes_dir = self.tmp / "notes"
        stage = N.new_stage("s07-demo", ["s07.1-a", "s07.2-b"], notes_dir,
                            title="데모", run_titles=["첫째", "둘째"])
        stage.doc_path().write_text(
            stage.doc_path().read_text(encoding="utf-8").replace(
                f"## {N.STAGE_QUESTION}\n", f"## {N.STAGE_QUESTION}\n질문?\n"),
            encoding="utf-8")
        for run_id in stage.runs:
            path = stage.note_path(run_id)
            text = path.read_text(encoding="utf-8")
            for name in N.BEFORE_RUN:
                text = text.replace(f"## {name}\n", f"## {name}\n내용\n")
            path.write_text(text, encoding="utf-8")
        for run_id in stage.runs:
            R.Run.start(run_id, "E_unit", {}, strict_paths=False,
                        results_dir=self.tmp / "results", notes_dir=notes_dir).finish({})

        runs = self._runs("restoration")
        (parent,) = self._by_kind(runs, T.STAGE_KIND)
        self.assertEqual(parent.info.run_name, "S07 데모")
        self.assertIn("질문?", parent.data.tags["mlflow.note.content"])
        children = {r.info.run_name: r for r in self._by_kind(runs, None)}
        self.assertEqual(set(children), {"S07.1 첫째", "S07.2 둘째"})
        for child in children.values():
            self.assertEqual(child.data.tags["mlflow.parentRunId"], parent.info.run_id)

    def test_smoke_run_goes_to_smoke_experiment(self):
        run = R.Run.start(
            "unit-smoke", "E_unit", {}, strict_paths=False, results_dir=self.tmp / "results",
        )
        run.finish({})
        (logged,) = self._runs(T.SMOKE_EXPERIMENT)
        self.assertEqual(logged.data.tags["kslab.experiment"], "E_unit")
        self.assertIsNone(self._client().get_experiment_by_name("restoration"))

    def test_unfinished_run_is_marked_failed(self):
        run = R.Run.start(
            "unit-crash", "E_unit", {}, strict_paths=False, results_dir=self.tmp / "results",
        )
        run.tracker._abandon()
        (logged,) = self._runs("restoration")
        self.assertEqual(logged.info.status, "FAILED")

    def test_tracking_failure_does_not_break_the_run(self):
        with mock.patch.object(T, "_client", side_effect=RuntimeError("boom")):
            run = R.Run.start(
                "unit-broken", "E_unit", {}, strict_paths=False,
                results_dir=self.tmp / "results",
            )
        self.assertIsNone(run.tracker)
        self.assertTrue(run.finish({"x": 1}).exists())



class RebuildLabelTest(unittest.TestCase):
    """옛 run의 학습 기록에 모델 이름(라벨)을 되찾아 붙인다."""

    def test_labels_are_recovered_by_loss_technique_and_elimination(self):
        from kslab.mlflow_rebuild import match_fit_labels

        def fit(technique, seed, loss=()):
            return {"hyperparameters": {"technique": technique, "random_state": seed},
                    "loss": [[i + 1, v] for i, v in enumerate(loss)]}

        models = {"all/seed1": {"loss": [0.5, 0.2]}, "minus_x/seed1": {"loss": [0.6, 0.3]},
                  "single_tensify/seed1": {"loss": [0.9]}}
        trained = [fit("a+b", 1, [0.6, 0.3]), fit("a+b+x", 1, [0.5, 0.2])]
        self.assertEqual(match_fit_labels(trained, models), ["minus_x/seed1", "all/seed1"])
        resumed = [fit("a+b+tensify", 1), fit("tensify", 1)]
        models = {"all/seed1": {}, "single_tensify/seed1": {}}
        self.assertEqual(match_fit_labels(resumed, models),
                         ["all/seed1", "single_tensify/seed1"])


if __name__ == "__main__":
    unittest.main()
