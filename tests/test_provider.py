"""provider 계약 테스트.

실제 k-safeguard `Gateway`를 읽기 전용으로 불러와 계약을 확인한다. 저장소가 없는
환경에서는 skip한다 — 이 실험실은 저장소 없이도 나머지 테스트가 돌아야 한다.

무거운 학습은 하지 않는다. 결정론적인 가짜 복원기를 끼워 계약만 본다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import provider as PROV


class _FakeRestorer:
    """항상 같은 치환을 하는 결정론적 복원기. confidence를 인자로 고정한다."""

    def __init__(self, mapping: dict[str, str], confidence: float):
        self.mapping = mapping
        self.confidence = confidence

    def restore_many(self, texts):
        out = []
        for text in texts:
            restored = text
            touched = 0
            for before, after in self.mapping.items():
                if before in restored:
                    restored = restored.replace(before, after)
                    touched += 1
            out.append((restored, [self.confidence] * touched))
        return out


class _NoOpRestorer:
    def restore_many(self, texts):
        return [(t, []) for t in texts]


def _skip_without_repo():
    if not PROV.KSAFEGUARD_SRC.exists():
        raise unittest.SkipTest("k-safeguard 저장소를 찾을 수 없습니다.")


class TestProviderContract(unittest.TestCase):
    def setUp(self):
        _skip_without_repo()

    def test_satisfies_protocol(self):
        _, CandidateProvider, _ = PROV.load_ksafeguard()
        p = PROV.MlRestoreProvider(models={"tensify": _NoOpRestorer()})
        self.assertIsInstance(p, CandidateProvider)
        self.assertIsInstance(p.name, str)

    def test_does_not_write_bytecode_into_repo(self):
        # 저장소에 .pyc 를 남기지 않는 것이 이 실험실의 규칙이다.
        PROV.load_ksafeguard()
        self.assertTrue(sys.dont_write_bytecode)

    def test_emits_candidate_above_threshold(self):
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.95)},
            default_threshold=0.9,
        )
        got = list(p.generate("씨스템"))
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].text, "시스템")
        self.assertTrue(got[0].lossy)
        self.assertAlmostEqual(got[0].confidence, 0.95)

    def test_abstains_below_threshold(self):
        # 확신이 낮으면 후보를 아예 내지 않는다 — ΔFPR이 구속 조건일 때의 설계 원칙.
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.4)},
            default_threshold=0.9,
        )
        self.assertEqual(list(p.generate("씨스템")), [])

    def test_never_emits_the_original(self):
        # 바뀐 게 없으면 후보가 아니다. Gateway가 이미 원문을 views[0]에 넣었다.
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"없는글자": "x"}, 0.99)},
            default_threshold=0.5,
        )
        self.assertEqual(list(p.generate("안녕하세요")), [])

    def test_confidence_is_clamped_into_range(self):
        # Gateway가 0~1을 검증하므로 방어적으로 자른다.
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 1.8)},
            default_threshold=0.5,
        )
        got = list(p.generate("씨스템"))
        self.assertEqual(len(got), 1)
        self.assertLessEqual(got[0].confidence, 1.0)
        self.assertGreaterEqual(got[0].confidence, 0.0)

    def test_per_technique_threshold_overrides_default(self):
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.6)},
            thresholds={"tensify": 0.5},
            default_threshold=0.99,
        )
        self.assertEqual(len(list(p.generate("씨스템"))), 1)

    def test_generation_order_puts_chosung_first(self):
        # 생성 순서가 곧 우선순위다(view 예산이 잘리면 뒤가 버려진다).
        # 오탐 위험이 구조적으로 없는 초성체를 먼저 낸다.
        p = PROV.MlRestoreProvider(
            models={
                "jongseong_cram": _FakeRestorer({"논": "노"}, 0.99),
                "chosung": _FakeRestorer({"안": "아"}, 0.99),
            },
            default_threshold=0.5,
        )
        got = list(p.generate("안논"))
        self.assertEqual(dict(got[0].metadata)["technique"], "chosung")

    def test_rejects_non_string(self):
        p = PROV.MlRestoreProvider(models={"tensify": _NoOpRestorer()})
        with self.assertRaises(TypeError):
            list(p.generate(123))

    def test_metadata_is_string_pairs(self):
        # CandidateProposal.metadata 는 tuple[tuple[str, str], ...] 이다.
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.95)}, default_threshold=0.5
        )
        metadata = list(p.generate("씨스템"))[0].metadata
        self.assertIsInstance(metadata, tuple)
        for key, value in metadata:
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)


class TestContractReport(unittest.TestCase):
    def setUp(self):
        _skip_without_repo()

    def test_clean_provider_passes(self):
        p = PROV.MlRestoreProvider(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.95)}, default_threshold=0.5
        )
        report = PROV.contract_report(p, ["씨스템 프롬프트", "안녕하세요"])
        self.assertTrue(report["contract_ok"], report["violations"])
        self.assertGreaterEqual(report["candidates_seen"], 1)

    def test_report_flags_verdict_method(self):
        # block/allow 를 가진 provider 는 경계 위반이다 — 판단은 하위 가드레일 몫이다.
        class Verdicting(PROV.MlRestoreProvider):
            def block(self, text):  # noqa: D102
                return True

        p = Verdicting(
            models={"tensify": _FakeRestorer({"씨": "시"}, 0.95)}, default_threshold=0.5
        )
        report = PROV.contract_report(p, ["씨스템"])
        self.assertFalse(report["has_no_verdict_method"])
        self.assertFalse(report["contract_ok"])

    def test_gateway_rejects_out_of_range_confidence(self):
        """Gateway가 실제로 confidence 범위를 검증하는지 확인 (우리 클램프의 근거)."""
        CandidateProposal, _, Gateway = PROV.load_ksafeguard()

        class Bad:
            name = "bad"

            def generate(self, text):
                yield CandidateProposal(text=text + "x", confidence=1.5)

        with self.assertRaises(ValueError):
            Gateway(providers=[Bad()], strict_providers=True).process("가나")


if __name__ == "__main__":
    unittest.main(verbosity=2)
