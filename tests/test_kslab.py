"""kslab 단위테스트.

가장 중요한 세 가지를 강제한다.

- 자모 테이블이 k-safeguard 저장소 원본과 갈라지지 않았는가 (값을 복제해 왔으므로 필요)
- 길이 보존이라고 선언한 변환이 505 시드 전체에서 실제로 길이를 보존하는가
- 후보 위치 규칙이 정답(clean)을 훔쳐보지 않는가 — 이게 새면 clean mutation이 과소평가된다
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import features as F
from kslab import jamo as J
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P
from kslab import record as R

MIXED = "읽어서 값을 앉아 삶은 닭을 먹어 한글Abc 123!? ㅋㅋ ㅏ"


class TestJamo(unittest.TestCase):
    def test_roundtrip_preserves_everything(self):
        # Given 겹받침·영문·숫자·문장부호·독립 자모가 섞인 문자열
        # When 분해했다가 되조합하면
        # Then 원문과 완전히 같아야 한다
        self.assertEqual(J.compose(MIXED, J.decompose(MIXED)), MIXED)

    def test_roundtrip_over_all_syllables(self):
        # Given 완성형 한글 음절 전체(11,172자)
        for code in range(J.HANGUL_BASE, J.HANGUL_END + 1):
            ch = chr(code)
            cho, jung, jong = J.split(ch)
            self.assertEqual(J.join(cho, jung, jong), ch)

    def test_tables_match_ksafeguard_source(self):
        # Given 저장소의 normalization.py 원본
        source = R.KSAFEGUARD_ROOT / "src" / "k_safeguard" / "normalization.py"
        if not source.exists():
            self.skipTest("k-safeguard 저장소를 찾을 수 없습니다.")
        text = source.read_text(encoding="utf-8")
        namespace: dict = {}
        for name in ("COMPAT_CHO", "COMPAT_JUNG", "COMPAT_JONG"):
            match = re.search(rf"^{name} = \((.*?)\)$", text, re.S | re.M)
            self.assertIsNotNone(match, f"{name}을 원본에서 찾지 못했습니다.")
            namespace[name] = eval("(" + match.group(1) + ")")  # noqa: S307
        # Then 복제해 온 테이블이 원본과 같아야 한다
        self.assertEqual(namespace["COMPAT_CHO"], J.COMPAT_CHO)
        self.assertEqual(namespace["COMPAT_JUNG"], J.COMPAT_JUNG)
        self.assertEqual(namespace["COMPAT_JONG"], J.COMPAT_JONG)

    def test_join_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            J.join(J.N_CHO, 0, 0)

    def test_word_relative_positions(self):
        # Given 두 어절
        got = J.word_relative_positions("가나 다")
        # Then 어절 안에서 1까지 올라가고 공백은 0
        self.assertEqual(got, [0.5, 1.0, 0.0, 1.0])


class TestObfuscate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds = P.load_seeds()
        cls.texts = [s.text for s in cls.seeds]

    def test_length_preserving_over_all_seeds(self):
        # Given 505개 시드 전체
        for technique in sorted(O.LENGTH_PRESERVING):
            fn = O.TRANSFORMS[technique]
            for intensity in (0.3, 0.5, 1.0):
                for text in self.texts:
                    out = fn(text, intensity, seed=1234)
                    # Then 문자 수가 절대 변하지 않아야 한다
                    self.assertEqual(
                        len(out), len(text),
                        f"{technique}@{intensity} 길이 변경: {text!r} -> {out!r}",
                    )

    def test_determinism(self):
        for name, fn in O.TRANSFORMS.items():
            a = fn(MIXED, 0.7, seed=7)
            b = fn(MIXED, 0.7, seed=7)
            self.assertEqual(a, b, f"{name}이 결정론적이지 않습니다.")

    def test_transforms_do_not_share_rng(self):
        # Given 같은 시드로 순서를 바꿔 호출해도
        first = [O.TRANSFORMS[n](MIXED, 0.5, seed=3) for n in sorted(O.TRANSFORMS)]
        second = [O.TRANSFORMS[n](MIXED, 0.5, seed=3) for n in sorted(O.TRANSFORMS, reverse=True)]
        # Then 각 변환의 결과는 호출 순서와 무관해야 한다
        self.assertEqual(first, list(reversed(second)))

    def test_liaison_handles_compound_finals(self):
        # Given disparity 원본이 전부 놓쳤던 겹받침 사례
        cases = {
            "읽어": "일거", "값을": "갑슬", "앉아": "안자",
            "삶은": "살믄", "닭을": "달글", "없어": "업서",
        }
        for src, expected in cases.items():
            self.assertEqual(O.liaison(src, 1.0, seed=0), expected, f"{src} 연음 실패")

    def test_liaison_simple_finals(self):
        self.assertEqual(O.liaison("먹을게", 1.0, seed=0), "머글게")

    def test_liaison_skips_ieung_final(self):
        # Given 종성 ㅇ 은 뒤로 넘어가지 않는다
        self.assertEqual(O.liaison("강아지", 1.0, seed=0), "강아지")

    def test_palatalize(self):
        self.assertEqual(O.palatalize("굳이 같이", 1.0, seed=0), "구지 가치")

    def test_jongseong_cram_changes_final_only(self):
        # Given 종성 크래밍
        out = O.jongseong_cram("안녕하세요", 1.0, seed=5)
        self.assertEqual(len(out), 5)
        for before, after in zip("안녕하세요", out):
            b, a = J.split(before), J.split(after)
            # Then 초성·중성은 그대로이고 종성만 달라져야 한다
            self.assertEqual(b[0], a[0])
            self.assertEqual(b[1], a[1])
            self.assertNotEqual(b[2], a[2])

    def test_empty_and_non_hangul_inputs(self):
        for name, fn in O.TRANSFORMS.items():
            self.assertEqual(fn("", 1.0, seed=0), "", f"{name} 빈 입력")
            for text in ("!!!", "abc 123", "ㅋㅋㅋ", "   "):
                out = fn(text, 1.0, seed=0)
                self.assertIsInstance(out, str, f"{name} {text!r}")

    def test_intensity_zero_is_identity(self):
        for name, fn in O.TRANSFORMS.items():
            if name == "break_spacing":
                continue  # intensity>=0.5 분기와 별개로 0은 공백 삽입 0회 = 항등
            self.assertEqual(fn(MIXED, 0.0, seed=0), MIXED, f"{name}@0.0")

    def test_family_and_registry_cover_each_other(self):
        self.assertEqual(set(O.TRANSFORMS), set(O.FAMILY))
        self.assertTrue(O.LENGTH_PRESERVING <= set(O.TRANSFORMS))


class TestPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seeds = P.load_seeds()
        cls.items = P.seeds_as_items(cls.seeds)

    def test_seed_counts(self):
        self.assertEqual(len(self.seeds), 505)
        self.assertEqual(sum(1 for s in self.seeds if s.label == "attack"), 301)
        self.assertEqual(sum(1 for s in self.seeds if s.label == "benign"), 204)

    def test_source_split_is_disjoint(self):
        train, test = P.split_by_source(self.items, ["template_v1"])
        P.assert_disjoint(train, test)
        self.assertEqual(len(test), 256)

    def test_assert_disjoint_catches_leak(self):
        # Given 같은 항목이 양쪽에 있을 때
        leaked = self.items[:5]
        # Then 누출을 잡아야 한다
        with self.assertRaises(AssertionError):
            P.assert_disjoint(leaked, leaked)

    def test_make_pairs_rejects_length_changing_technique(self):
        with self.assertRaises(ValueError):
            P.make_pairs(self.items[:2], ["jamo_decompose"])

    def test_identity_pairs_carry_partner_technique(self):
        # Given identity 페어를 포함해 tensify 페어를 만들면
        pairs = P.make_pairs(self.items[:1], ["tensify"], (1.0,), include_identity=True)
        identity = [p for p in pairs if p.is_identity]
        self.assertTrue(identity)
        # Then identity 페어도 tensify 후보 규칙을 달고 있어야 한다.
        # 공통 "identity" 기법으로 묶이면 후보 위치가 '모든 음절'이 되어 분포가 어긋난다.
        for pair in identity:
            self.assertEqual(pair.technique, "tensify")

    def test_clean_sentence_filter(self):
        self.assertIsNone(P.clean_sentence("짧다"))
        self.assertIsNone(P.clean_sentence("a" * 60))          # 한글 비율 미달
        self.assertIsNone(P.clean_sentence("가" * 500))         # 너무 김
        self.assertEqual(P.clean_sentence("  안녕하세요  반갑습니다  "), "안녕하세요 반갑습니다")


class TestFeatures(unittest.TestCase):
    def test_candidate_positions_are_input_only(self):
        # Given 정답을 모르는 상태에서 후보를 고를 수 있어야 한다.
        # candidate_positions 의 시그니처가 (text, technique) 뿐이라는 것으로 강제된다.
        import inspect

        params = list(inspect.signature(F.candidate_positions).parameters)
        self.assertEqual(params, ["text", "technique"])

    def test_chosung_candidates_are_compat_jamo(self):
        obf = O.chosung("안녕하세요 반갑습니다", 0.5, seed=1)
        for i in F.candidate_positions(obf, "chosung"):
            self.assertIn(obf[i], J.CHO_INDEX)

    def test_chosung_has_no_candidates_in_clean_text(self):
        # Given 난독화되지 않은 정상 문장
        # Then 초성체 후보가 하나도 없어야 한다 -> 구조적으로 clean mutation 0%
        self.assertEqual(F.candidate_positions("안녕하세요 반갑습니다", "chosung"), [])

    def test_tensify_candidates_are_tense_onsets(self):
        positions = F.candidate_positions("깨끗한 시스템", "tensify")
        self.assertEqual([("깨끗한 시스템")[i] for i in positions], ["깨", "끗"])

    def test_unknown_technique_raises(self):
        with self.assertRaises(ValueError):
            F.candidate_positions("가나다", "nope")

    def test_extract_examples_marks_changed(self):
        pair = P.Pair("c", "s", "tensify", 1.0, "씨스템", "시스템")
        examples = F.extract_examples([pair], window=2)
        self.assertEqual(len(examples), 1)
        self.assertTrue(examples[0].changed)
        self.assertEqual(examples[0].target_slots, J.split("시"))

    def test_vocab_freezes(self):
        pair = P.Pair("c", "s", "tensify", 1.0, "씨스템", "시스템")
        examples = F.extract_examples([pair], window=2)
        vocab = F.CharVocab().fit(examples).freeze()
        with self.assertRaises(RuntimeError):
            vocab.fit(examples)
        self.assertEqual(vocab.encode("존재하지않는문자"[0]), F.UNK)


class TestMetrics(unittest.TestCase):
    def test_char_error_rate_same_length(self):
        self.assertEqual(M.char_error_rate("안녕", "안녕"), 0.0)
        self.assertEqual(M.char_error_rate("안녕", "안뇽"), 0.5)

    def test_char_error_rate_different_length(self):
        self.assertAlmostEqual(M.char_error_rate("안녕하", "안녕"), 1 / 3)

    def test_clean_mutation_rate(self):
        got = M.clean_mutation_rate(["가나", "다라"], ["가나", "다람"])
        self.assertEqual(got["sentence_rate"], 0.5)
        self.assertEqual(got["mutated"], 1)

    def test_clean_mutation_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            M.clean_mutation_rate(["가"], ["가", "나"])

    def test_summarize_is_cluster_balanced(self):
        # Given 한 cluster에 값이 몰려 있을 때
        got = M.summarize_observations(
            [("a", 0.0), ("a", 0.0), ("a", 0.0), ("b", 1.0)], 500, 1
        )
        # Then micro는 0.25지만 cluster 균형 추정은 0.5여야 한다
        self.assertAlmostEqual(got["micro_estimate"], 0.25)
        self.assertAlmostEqual(got["cluster_balanced_estimate"], 0.5)
        self.assertEqual(got["clusters"], 2)

    def test_summarize_is_deterministic(self):
        obs = [("a", 0.3), ("b", 0.7), ("c", 0.1)]
        self.assertEqual(
            M.summarize_observations(obs, 500, 2026),
            M.summarize_observations(obs, 500, 2026),
        )

    def test_slot_accuracy(self):
        got = M.slot_accuracy([(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (1, 2, 4)])
        self.assertEqual(got["cho"], 1.0)
        self.assertEqual(got["jong"], 0.5)
        self.assertEqual(got["syllable_exact"], 0.5)


class TestRecord(unittest.TestCase):
    def test_run_id_validation(self):
        R.validate_run_id("e0-2026.08-23_x")
        for bad in ("", "has space", "슬래시/", "back\\slash"):
            with self.assertRaises(ValueError):
                R.validate_run_id(bad)

    def test_paths_contained_accepts_inside(self):
        keys = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ[k] = str(R.ROOT / "data" / k.lower())
            report = R.assert_paths_contained()
            self.assertEqual(len(report), len(keys))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_paths_contained_rejects_outside(self):
        # Given 캐시가 사용자 홈을 가리킬 때 — 기본값 그대로 돌린 상황
        keys = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ[k] = str(Path.home() / ".cache" / k.lower())
            # Then 즉시 막아야 한다. 안 막으면 폴더 밖에 수 GB가 조용히 쌓인다.
            with self.assertRaises(RuntimeError):
                R.assert_paths_contained()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_sha256_text(self):
        self.assertEqual(len(R.sha256_text("가나다")), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
