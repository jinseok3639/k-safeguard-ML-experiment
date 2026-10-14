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
            mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": self.uri, "KSLAB_MLFLOW": "1"}),
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

    def _only_run(self, experiment: str):
        client = self._client()
        found = client.get_experiment_by_name(experiment)
        self.assertIsNotNone(found, f"experiment {experiment} 가 없다")
        runs = client.search_runs([found.experiment_id])
        self.assertEqual(len(runs), 1)
        return client, runs[0]

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
        # 합집합 기법 이름의 '+'는 MLflow 키에 못 쓴다 — 정리돼서 기록돼야 한다.
        union = T.begin_fit("bilstm", SimpleNamespace(technique="liaison+tensify", hidden=8))
        union.epoch(1, 0.5)
        run.finish(
            {"status": "PROMOTE_CANDIDATE", "sweeps": {"tensify": {"cer": 0.25}}},
            headline={"tensify": {"best": 0.25}},
        )

        client, logged = self._only_run("E_unit")
        self.assertEqual(logged.info.run_name, "unit-20261007")
        self.assertEqual(logged.info.status, "FINISHED")
        self.assertEqual(logged.data.params["corpus_size"], "10")
        self.assertEqual(logged.data.metrics["sweeps/tensify/cer"], 0.25)
        self.assertEqual(logged.data.tags["status"], "PROMOTE_CANDIDATE")
        self.assertEqual(logged.data.metrics["headline/tensify/best"], 0.25)
        self.assertEqual(logged.data.params["fit.mlp/tensify.hidden"], "64")
        history = client.get_metric_history(logged.info.run_id, "train/mlp/tensify/loss")
        self.assertEqual([(m.step, m.value) for m in history], [(1, 0.9), (2, 0.4)])
        self.assertEqual(logged.data.params["fit.bilstm/liaison_tensify.hidden"], "8")
        union_history = client.get_metric_history(logged.info.run_id, "train/bilstm/liaison_tensify/loss")
        self.assertEqual([(m.step, m.value) for m in union_history], [(1, 0.5)])
        self.assertEqual(logged.data.tags["kslab.experiment"], "E_unit")
        self.assertEqual(
            [d.dataset.name for d in logged.inputs.dataset_inputs], ["seeds"]
        )
        artifacts = {a.path for a in client.list_artifacts(logged.info.run_id)}
        self.assertLessEqual(
            {"report.md", "metrics.json", "manifest.json", "pip-freeze.txt"}, artifacts
        )
        # 원문이 든 산출물은 MLflow 저장소로 넘어가면 안 된다.
        self.assertNotIn("predictions.jsonl", artifacts)

    def test_smoke_run_goes_to_smoke_experiment(self):
        run = R.Run.start(
            "unit-smoke", "E_unit", {}, strict_paths=False, results_dir=self.tmp / "results",
        )
        run.finish({})
        _, logged = self._only_run(T.SMOKE_EXPERIMENT)
        self.assertEqual(logged.data.tags["kslab.experiment"], "E_unit")
        self.assertIsNone(self._client().get_experiment_by_name("E_unit"))

    def test_unfinished_run_is_marked_failed(self):
        run = R.Run.start(
            "unit-crash", "E_unit", {}, strict_paths=False, results_dir=self.tmp / "results",
        )
        run.tracker._abandon()
        _, logged = self._only_run("E_unit")
        self.assertEqual(logged.info.status, "FAILED")

    def test_tracking_failure_does_not_break_the_run(self):
        with mock.patch.object(T, "_client", side_effect=RuntimeError("boom")):
            run = R.Run.start(
                "unit-broken", "E_unit", {}, strict_paths=False,
                results_dir=self.tmp / "results",
            )
        self.assertIsNone(run.tracker)
        self.assertTrue(run.finish({"x": 1}).exists())


if __name__ == "__main__":
    unittest.main()
