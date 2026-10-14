"""S03 편집 종류별 임계값 배분 테스트. 모델 없이 가짜 자리 캐시로 규칙만 확인한다."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restoration.exp import run_s03_edit_thresholds as S


def _site(cls: str, conf: float, pos: int = 0, char: str = "가"):
    return (pos, cls, conf, char)


class EditClassTest(unittest.TestCase):
    def test_single_and_multi(self):
        self.assertEqual(S._edit_class((0, 0, 0), [1, 0, 0]), "cho")
        self.assertEqual(S._edit_class((0, 0, 0), [0, 0, 4]), "jong")
        self.assertEqual(S._edit_class((0, 0, 0), [1, 0, 4]), "multi")


class ApplyTest(unittest.TestCase):
    def test_threshold_per_class(self):
        sites = [[_site("cho", 0.9, 0, "까"), _site("jong", 0.5, 1, "각")]]
        self.assertEqual(S._apply(["가가"], sites, {"cho": 0.95, "jung": 1, "jong": 0.4, "multi": 1}), ["가각"])
        self.assertEqual(S._apply(["가가"], sites, {"cho": 0.9, "jung": 1, "jong": 0.6, "multi": 1}), ["까가"])


class AllocationTest(unittest.TestCase):
    # benign: 초성 편집은 확신도가 높게 자주 나오고, 받침 편집은 낮게 드물게 나온다.
    benign = [[_site("cho", 0.99)], [_site("cho", 0.98)], [_site("cho", 0.97)],
              [_site("jong", 0.30)], [], []]

    def test_start_is_free(self):
        thresholds, trace = S.allocate_greedy(self.benign, lambda th: 0.0, budget=0)
        mutated = S._mutated(S._class_scores(self.benign), thresholds)
        self.assertFalse(mutated)
        self.assertGreater(thresholds["cho"], 0.99)
        self.assertGreater(thresholds["jong"], 0.30)
        self.assertEqual(thresholds["jung"], 0.0)   # benign에 모음 편집이 없으면 바닥까지 공짜
        self.assertEqual(len(trace), 1)

    def test_spends_budget_where_gain_is(self):
        # 받침을 내리면 공격 이득이 크고, 초성은 작다.
        def gain(th):
            return (0.10 if th["jong"] <= 0.30 else 0.0) + (0.01 if th["cho"] <= 0.99 else 0.0)
        thresholds, trace = S.allocate_greedy(self.benign, gain, budget=1)
        self.assertLessEqual(thresholds["jong"], 0.30)
        self.assertGreater(thresholds["cho"], 0.99)
        self.assertEqual(trace[-1]["benign_mutated"], 1)

    def test_never_exceeds_budget(self):
        thresholds, _ = S.allocate_greedy(self.benign, lambda th: -sum(th.values()), budget=2)
        self.assertLessEqual(len(S._mutated(S._class_scores(self.benign), thresholds)), 2)

    def test_stops_without_gain(self):
        _, trace = S.allocate_greedy(self.benign, lambda th: 0.0, budget=4)
        self.assertEqual(trace[-1]["benign_mutated"], 0)

    def test_even_share(self):
        thresholds = S.allocate_even(self.benign, budget=4)   # 종류마다 1문장
        self.assertTrue(0.98 < thresholds["cho"] <= 0.99)
        self.assertEqual(thresholds["jong"], 0.0)
        self.assertLessEqual(len(S._mutated(S._class_scores(self.benign), thresholds)), 4)


if __name__ == "__main__":
    unittest.main()
