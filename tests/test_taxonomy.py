"""통합 분류표 테스트."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import taxonomy as T
from kslab.obfuscate import TRANSFORMS


class TaxonomyTest(unittest.TestCase):
    def test_every_transform_belongs_to_exactly_one_type(self):
        self.assertEqual(T.unassigned(), [])
        for name in TRANSFORMS:
            owners = [t.id for t in T.TYPES if name in t.techniques]
            self.assertEqual(len(owners), 1, f"{name}: {owners}")

    def test_planned_techniques_are_new_and_unique(self):
        planned = [p for t in T.TYPES for p in t.planned]
        self.assertEqual(len(planned), len(set(planned)))
        self.assertFalse(set(planned) & set(TRANSFORMS))

    def test_fields_are_valid(self):
        self.assertEqual(len(T.BY_ID), len(T.TYPES))
        for t in T.TYPES:
            self.assertIn(t.scope, T.SCOPES, t.id)
            self.assertIn(t.category, T.CATEGORIES, t.id)
            self.assertEqual(t.category == "KOTOX 밖", not t.kotox, t.id)

    def test_type_of(self):
        self.assertEqual(T.type_of("tensify").id, "phon_initial")
        self.assertEqual(T.type_of("symbol_insert").id, "symbol_insert")
        with self.assertRaises(KeyError):
            T.type_of("nope")

    def test_coverage_rows(self):
        rows = {r["id"]: r for r in T.coverage_rows()}
        self.assertEqual(rows["phon_liaison"]["내부 생성기"], "있음")
        self.assertEqual(rows["shape_hangul"]["내부 생성기"], "있음")
        self.assertEqual(rows["translit"]["내부 생성기"], "없음")


if __name__ == "__main__":
    unittest.main()
