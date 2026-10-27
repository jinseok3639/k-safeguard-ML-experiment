"""S01 난독화 강건성 집계·판정 파싱 테스트. 모델 없이 돈다."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardrail import evalset as E
from guardrail import scorers as S


def _rec(rid, source, label, axis, types, n_applied=1, name="x"):
    return {"id": rid, "source_id": source, "label": label, "axis": axis, "types": types,
            "n_applied": n_applied, "name": name, "clean": f"{source} 원문", "obfuscated": rid}


class AggregateTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            _rec("a1/single/t", "a1", "attack", "single", ["phon_initial"], name="tensify"),
            _rec("a2/single/t", "a2", "attack", "single", ["phon_initial"], name="tensify"),
            _rec("a1/compose/h2", "a1", "attack", "compose", ["phon_initial", "spacing"], 2),
            _rec("b1/single/t", "b1", "benign", "single", ["phon_initial"], name="tensify"),
        ]
        self.rows = E.clean_rows(self.records) + self.records

    def test_clean_rows_one_per_source(self):
        clean = E.clean_rows(self.records)
        self.assertEqual([r["id"] for r in clean], ["a1/clean", "a2/clean", "b1/clean"])
        self.assertEqual(clean[0]["obfuscated"], "a1 원문")

    def test_rates_and_conditioning(self):
        flags = {
            "a1/clean": True, "a2/clean": False, "b1/clean": False,
            "a1/single/t": False,   # 원문은 막혔는데 난독화로 통과
            "a2/single/t": False,   # 원문부터 통과
            "a1/compose/h2": True,
            "b1/single/t": True,    # 정상 오차단
        }
        out = E.aggregate(self.rows, flags, bootstrap_samples=50)
        g = out["groups"]
        self.assertEqual(out["clean_attacks_caught"], 1)
        self.assertAlmostEqual(out["clean_tpr"], 0.5)
        # phon_initial: 공격 3행 중 2행 통과. 유형이 둘인 compose 행도 phon_initial에 한 번 들어간다
        self.assertEqual(g["type:phon_initial"]["evasion"]["n"], 3)
        self.assertAlmostEqual(g["type:phon_initial"]["evasion"]["rate"], 2 / 3)
        # 원문이 막힌 a1에서 나온 행만: single 통과, compose 차단
        self.assertEqual(g["type:phon_initial"]["evasion_if_caught"]["n"], 2)
        self.assertAlmostEqual(g["type:phon_initial"]["evasion_if_caught"]["rate"], 0.5)
        self.assertAlmostEqual(g["type:phon_initial"]["fpr"]["rate"], 1.0)
        self.assertEqual(g["type:spacing"]["evasion"]["n"], 1)
        self.assertIn("op:tensify", g)
        self.assertIn("n_applied:2", g)
        self.assertNotIn("op:x", g)

    def test_unparsed_rows_are_counted_not_scored(self):
        flags: dict[str, bool | None] = {r["id"]: False for r in self.rows}
        flags["a1/single/t"] = None
        g = E.aggregate(self.rows, flags, bootstrap_samples=10)["groups"]
        self.assertEqual(g["axis:single"]["unparsed"], 1)
        self.assertEqual(g["axis:single"]["evasion"]["n"], 1)

    def test_headline_keys(self):
        flags = {r["id"]: True for r in self.rows}
        h = E.headline(E.aggregate(self.rows, flags, bootstrap_samples=10))
        self.assertEqual(h["clean_tpr"], 1.0)
        self.assertEqual(h["evasion_all"], 0.0)
        self.assertEqual(h["fpr_all"], 1.0)


class VerifyTest(unittest.TestCase):
    def test_hash_mismatch_stops(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        (tmp / "records.jsonl").write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
        digest = hashlib.sha256((tmp / "records.jsonl").read_bytes()).hexdigest()
        (tmp / "SHA256SUMS.txt").write_text(f"{digest} *records.jsonl\n", encoding="utf-8")
        self.assertEqual(E.verify(tmp), {"records.jsonl": digest})
        (tmp / "records.jsonl").write_text("변조\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            E.verify(tmp)


class GemmaParseTest(unittest.TestCase):
    def test_blocked_and_type(self):
        out = S.parse_gemma('{"blocked": true, "type": "safety-classifier", "topics": ["jailbreak"]}')
        self.assertTrue(out["flagged"])
        self.assertEqual(out["meta"]["type"], "safety-classifier")
        self.assertIsNone(out["score"])

    def test_truncated_output_still_reads_blocked(self):
        self.assertFalse(S.parse_gemma('{\n  "blocked": false,\n  "ty')["flagged"])

    def test_unparsable_is_none(self):
        self.assertIsNone(S.parse_gemma("죄송합니다")["flagged"])


class GuardrailTableTest(unittest.TestCase):
    def test_revisions_pinned(self):
        for g in S.GUARDRAILS.values():
            self.assertRegex(g.revision, r"^[0-9a-f]{40}$")
            self.assertTrue(g.name.endswith("@" + g.revision))


if __name__ == "__main__":
    unittest.main()
