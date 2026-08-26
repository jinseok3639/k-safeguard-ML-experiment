"""E9 추가분 테스트 — 음운 규칙 생성기, 조합 난독화, 구조적 상한, 자리 단위 지표.

기존 `tests/test_kslab.py`의 `TestObfuscate`가 `O.TRANSFORMS`/`O.LENGTH_PRESERVING`을
순회하므로 새 변환 9종의 길이 보존·결정론·RNG 독립·빈 입력·강도 0 항등은 이미 거기서
검증된다. 여기서는 그 외의 것만 다룬다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kslab import ceiling as CE
from kslab import compose as X
from kslab import features as F
from kslab import metrics as M
from kslab import obfuscate as O
from kslab import pairs as P


class TestStandardPronunciation(unittest.TestCase):
    """표준발음법 조항 대조. 생성기가 조항대로 돌지 않으면 학습 라벨이 통째로 틀린다."""

    CASES = {
        # 제12항 1 — 격음화. 양방향(ㅎ이 앞이든 뒤든).
        "aspirate": {
            "놓고": "노코", "좋다": "조타", "많지": "만치", "닳지": "달치",
            "국화": "구콰", "맏형": "마텽", "좁히다": "조피다",
            "앉히다": "안치다", "밝히다": "발키다", "축하": "추카",
        },
        # 제12항 4 — ㅎ 탈락.
        "h_deletion": {
            "좋아": "조아", "많아": "마나", "싫어": "시러",
            "낳은": "나은", "놓아": "노아",
        },
        # 연음 + ㅎ 탈락을 함께 지키는 변종.
        "liaison_correct": {
            "좋아": "조아", "많아": "마나", "싫어": "시러",
            "먹을": "머글", "읽어": "일거", "값을": "갑슬", "앉아": "안자",
        },
        # 제8·9항 — 평파열음화(7종성).
        "coda_neutralize": {
            "부엌": "부억", "옷": "옫", "빛": "빋", "앞": "압", "히읗": "히읃", "밖": "박",
        },
        # 제10·11항 — 자음군 단순화.
        "cluster_reduce": {
            "값": "갑", "넋": "넉", "닭": "닥", "삶": "삼", "여덟": "여덜",
        },
        # 제18·19항 — 비음화(파열음/ㄹ/상호).
        "nasalize": {
            "국물": "궁물", "닫는": "단는", "밥물": "밤물", "먹는": "멍는",
            "종로": "종노", "강릉": "강능", "백리": "뱅니", "섭리": "섬니",
        },
        # 제20항 — 유음화(양방향).
        "lateralize": {
            "신라": "실라", "난로": "날로", "칼날": "칼랄", "설날": "설랄", "권력": "궐력",
        },
        # 제23~28항 — 규칙 경음화(장애음 종성 뒤에서만).
        "tensify_rule": {
            "국밥": "국빱", "학교": "학꾜", "먹다": "먹따", "덮개": "덮깨",
        },
    }

    def test_clause_examples(self):
        for technique, cases in self.CASES.items():
            fn = O.TRANSFORMS[technique]
            for src, expected in cases.items():
                self.assertEqual(
                    fn(src, 1.0, seed=0), expected,
                    f"{technique}: {src} 의 표준 발음은 {expected}",
                )

    def test_rule_does_not_fire_without_its_context(self):
        # Given 규칙 경음화는 장애음 받침 뒤에서만 일어난다
        # Then 받침이 없거나 울림소리면 건드리지 않는다
        for text in ("가다", "나비", "우리", "some ascii"):
            self.assertEqual(O.tensify_rule(text, 1.0, seed=0), text, text)
        # 유음화는 ㄴ·ㄹ 인접에서만
        self.assertEqual(O.lateralize("가나다", 1.0, seed=0), "가나다")
        # ㅎ 탈락은 뒤에 모음이 와야
        self.assertEqual(O.h_deletion("좋다", 1.0, seed=0), "좋다")

    def test_composed_reaches_real_pronunciation(self):
        """단일 규칙으로는 못 만드는 실제 표준 발음이 조합으로 나온다."""
        stages = ("cluster_reduce", "coda_neutralize", "nasalize", "tensify_rule")
        cases = {
            "옷고름": "옫꼬름", "꽃다발": "꼳따발",
            "밭머리": "반머리", "앞마당": "암마당", "읽는": "잉는",
        }
        for src, expected in cases.items():
            self.assertEqual(X.apply_stages(src, stages, 1.0, 0).text, expected, src)
        # 그리고 단일 규칙만으로는 도달하지 못한다는 것도 확인한다
        self.assertNotEqual(O.tensify_rule("옷고름", 1.0, seed=0), "옫꼬름")


class TestVowelConfuse(unittest.TestCase):
    """중성 슬롯을 건드리는 유일한 길이 보존 기법."""

    def test_only_touches_nucleus(self):
        from kslab.jamo import is_syllable, split

        text = "재해 대비 외국 의사 계세요"
        out = O.vowel_confuse(text, 1.0, seed=3)
        self.assertEqual(len(out), len(text))
        changed = 0
        for a, b in zip(text, out):
            if a == b:
                continue
            self.assertTrue(is_syllable(a) and is_syllable(b))
            x, y = split(a), split(b)
            self.assertEqual(x[0], y[0], "초성이 바뀌었습니다")
            self.assertEqual(x[2], y[2], "종성이 바뀌었습니다")
            self.assertNotEqual(x[1], y[1])
            changed += 1
        self.assertGreater(changed, 0, "혼동 가능한 모음이 있는데 하나도 안 바뀌었습니다")

    def test_unlocks_the_nucleus_slot_in_union(self):
        # Given 기존 조합 가능 기법만으로는 중성이 담당 슬롯에 안 들어온다
        self.assertNotIn(1, F.UNKNOWN_SLOTS["jongseong_cram+liaison+tensify"])
        # Then vowel_confuse를 넣어야 중성이 열린다
        self.assertIn(1, F.UNKNOWN_SLOTS["tensify+vowel_confuse"])


class TestCompositionAlgebra(unittest.TestCase):
    """어떤 조합이 의미가 있는지 — 실측으로 확인한 사실을 회귀로 고정한다."""

    @classmethod
    def setUpClass(cls):
        cls.texts = [s.text for s in P.load_seeds()][:200]

    def test_chosung_absorbs_later_stages(self):
        """chosung 뒤에 뭘 걸어도 결과가 같다 — 그래서 조합 대상에서 뺐다."""
        for text in self.texts:
            base = O.chosung(text, 1.0, seed=1)
            self.assertEqual(O.jongseong_cram(base, 1.0, seed=2), base)

    def test_tensify_and_cram_commute(self):
        """초성만 보는 규칙과 종성만 쓰는 규칙이라 순서가 무의미하다."""
        for text in self.texts:
            a = O.tensify(O.jongseong_cram(text, 1.0, seed=2), 1.0, seed=1)
            b = O.jongseong_cram(O.tensify(text, 1.0, seed=1), 1.0, seed=2)
            self.assertEqual(a, b)

    def test_liaison_and_tensify_do_not_commute(self):
        """반대로 여기는 순서가 진짜 의미가 있다."""
        differing = sum(
            1 for t in self.texts
            if O.tensify(O.liaison(t, 1.0, 1), 1.0, 2) != O.liaison(O.tensify(t, 1.0, 2), 1.0, 1)
        )
        self.assertGreater(differing, 0, "순서가 결과를 바꾸는 문장이 하나도 없습니다")


class TestCompose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.texts = [s.text for s in P.load_seeds()][:150]

    def test_stage_seeds_are_distinct_and_deterministic(self):
        names = ("liaison", "tensify", "jongseong_cram")
        first = [X.stage_seed(1234, i, n) for i, n in enumerate(names)]
        again = [X.stage_seed(1234, i, n) for i, n in enumerate(names)]
        self.assertEqual(first, again, "단계 시드가 결정론적이지 않습니다")
        self.assertEqual(len(set(first)), len(first), "단계 시드가 서로 같습니다")

    def test_apply_stages_preserves_length(self):
        for profile in X.PROFILES:
            for text in self.texts[:40]:
                out = X.apply_stages(text, profile.stages, 1.0, 1234)
                self.assertEqual(len(out.text), len(text), profile.name)
                self.assertEqual(len(out.provenance), len(text), profile.name)

    def test_provenance_marks_exactly_the_changed_positions(self):
        profile = X.PROFILES_BY_NAME["d3_canonical"]
        for text in self.texts[:40]:
            out = X.apply_stages(text, profile.stages, 1.0, 1234)
            for i, (a, b) in enumerate(zip(text, out.text)):
                if a == b:
                    # 바뀌었다가 되돌아온 자리는 provenance가 남을 수 있다.
                    continue
                self.assertTrue(out.provenance[i], f"{i}번 자리에 출처가 없습니다")
                self.assertIn(out.provenance[i], profile.stages)

    def test_validate_stages_rejects_non_composable(self):
        with self.assertRaises(ValueError):
            X.apply_stages("안녕", ("chosung", "tensify"), 1.0, 0)
        with self.assertRaises(ValueError):
            X.apply_stages("안녕", ("zwsp_inject",), 1.0, 0)
        with self.assertRaises(ValueError):
            X.apply_stages("안녕", ("nope",), 1.0, 0)
        with self.assertRaises(ValueError):
            X.apply_stages("안녕", (), 1.0, 0)

    def test_intensity_zero_leaves_text_alone(self):
        for profile in X.PROFILES:
            out = X.apply_stages(self.texts[0], profile.stages, 0.0, 1234)
            self.assertEqual(out.text, self.texts[0], profile.name)
            self.assertEqual(out.touched, 0, profile.name)

    def test_calibrate_hits_budget_or_reports_ceiling(self):
        budget = 0.08
        for name in ("d1_liaison", "d2_liaison_then_tensify", "d3_canonical"):
            profile = X.PROFILES_BY_NAME[name]
            intensity, realized = X.calibrate(self.texts, profile.stages, budget, 1234)
            self.assertGreaterEqual(intensity, 0.0)
            self.assertLessEqual(intensity, 1.0)
            if intensity < 1.0:
                # 예산에 맞췄다면 실측이 예산 근처여야 한다
                self.assertLess(abs(realized - budget), 0.02, f"{name}: {realized}")

    def test_deeper_composition_is_not_more_damaging_at_fixed_budget(self):
        """iso-damage 보정의 존재 이유 — 보정하면 깊이가 달라도 손상량이 비슷해진다."""
        budget = 0.08
        realized = []
        for name in ("d1_liaison", "d2_liaison_then_tensify", "d3_canonical"):
            profile = X.PROFILES_BY_NAME[name]
            realized.append(X.calibrate(self.texts, profile.stages, budget, 1234)[1])
        self.assertLess(max(realized) - min(realized), 0.04, f"손상량 편차가 큽니다: {realized}")

    def test_canonical_sort_is_stable(self):
        self.assertEqual(
            X.canonical_sort(("jongseong_cram", "tensify", "liaison")),
            ("liaison", "tensify", "jongseong_cram"),
        )

    def test_profiles_are_composable_and_named_uniquely(self):
        self.assertEqual(len({p.name for p in X.PROFILES}), len(X.PROFILES))
        for profile in X.PROFILES:
            X.validate_stages(profile.stages)


class TestCompositeSpec(unittest.TestCase):
    def test_union_of_slots_and_positions(self):
        slots = F.UNKNOWN_SLOTS["liaison+tensify"]
        self.assertEqual(
            set(slots), set(F.UNKNOWN_SLOTS["liaison"]) | set(F.UNKNOWN_SLOTS["tensify"])
        )
        text = "머글게 씨스템"
        union = set(F.candidate_positions(text, "liaison+tensify"))
        self.assertEqual(
            union,
            set(F.candidate_positions(text, "liaison"))
            | set(F.candidate_positions(text, "tensify")),
        )

    def test_composite_name_is_order_independent(self):
        self.assertEqual(
            F.composite_name(["tensify", "liaison"]),
            F.composite_name(["liaison", "tensify"]),
        )
        self.assertEqual(F.composite_name(["tensify"]), "tensify")

    def test_unknown_name_still_raises(self):
        with self.assertRaises(ValueError):
            F.candidate_positions("가나", "nope")
        with self.assertRaises(ValueError):
            F.candidate_positions("가나", "tensify+nope")

    def test_candidate_positions_remain_input_only(self):
        """조합을 지원하게 바뀐 뒤에도 후보 규칙이 원문을 못 보게 유지돼야 한다."""
        import inspect

        params = list(inspect.signature(F.candidate_positions).parameters)
        self.assertEqual(params, ["text", "technique"])

    def test_narrow_rules_visit_fewer_positions_than_exhaustive(self):
        text = "까치가 하늘에 계세요 코피"
        exhaustive = len(F.candidate_positions(text, "liaison"))
        for narrow in ("tensify", "aspirate", "vowel_confuse"):
            self.assertLess(
                len(F.candidate_positions(text, narrow)), exhaustive, narrow
            )


class TestCeiling(unittest.TestCase):
    def test_identity_has_no_damage(self):
        text = "안녕하세요 반갑습니다"
        result = CE.ceiling_for([text], [text], "liaison")
        self.assertEqual(result["damaged_sites"], 0)
        self.assertEqual(result["ceiling_cer_reduction"], 0.0)

    def test_damage_outside_unknown_slots_is_inexpressible(self):
        """tensify 모델(담당=초성)에 종성 손상을 주면 구조적으로 못 고친다."""
        clean = "가자"
        # 종성만 다른 입력. tensify의 후보 규칙(경음 초성)에 걸리도록 초성을 경음으로 만든다.
        obf = "깍자"
        groups = CE.classify_sites(obf, clean, "tensify")
        self.assertIn(0, groups["inexpressible"] + groups["unreachable"])
        self.assertEqual(groups["fixable"], [])

    def test_damage_inside_unknown_slots_is_fixable(self):
        groups = CE.classify_sites("까자", "가자", "tensify")
        self.assertEqual(groups["fixable"], [0])
        self.assertEqual(groups["unreachable"], [])
        self.assertEqual(groups["inexpressible"], [])

    def test_unvisited_position_is_unreachable(self):
        """평음 초성 자리는 tensify의 후보 규칙에 안 걸리므로 방문조차 못 한다."""
        groups = CE.classify_sites("낙자", "가자", "tensify")
        self.assertEqual(groups["unreachable"], [0])

    def test_union_model_dominates_its_members(self):
        """합집합 모델의 상한은 구성 기법 어느 것보다 낮을 수 없다."""
        texts = [s.text for s in P.load_seeds()][:120]
        profile = X.PROFILES_BY_NAME["d2_liaison_then_tensify"]
        obf = [X.apply_stages(t, profile.stages, 1.0, 1234).text for t in texts]
        union = CE.ceiling_for(texts, obf, profile.technique())
        for member in profile.stages:
            member_ceiling = CE.ceiling_for(texts, obf, member)
            self.assertGreaterEqual(
                round(union["ceiling_cer_reduction"], 9),
                round(member_ceiling["ceiling_cer_reduction"], 9),
                f"합집합이 {member}보다 상한이 낮습니다",
            )

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            CE.classify_sites("가", "가나", "tensify")


class TestSiteOutcomes(unittest.TestCase):
    def test_four_way_classification(self):
        clean = "가나다라"
        obf = "까나다라"      # 0번 자리만 손상
        restored = "가나다맘"  # 0번은 고쳤고(repair), 3번은 멀쩡한데 망침(damage)
        out = M.site_outcomes([clean], [obf], [restored])
        self.assertEqual(out["repair"], 1)
        self.assertEqual(out["damage"], 1)
        self.assertEqual(out["miss"], 0)
        self.assertEqual(out["botch"], 0)
        self.assertEqual(out["damaged_sites"], 1)
        self.assertEqual(out["net_repair"], 0.0)   # (1 - 1) / 1

    def test_miss_and_botch(self):
        out = M.site_outcomes(["가나"], ["까나"], ["까나"])
        self.assertEqual(out["miss"], 1)
        self.assertEqual(out["net_repair"], 0.0)
        out = M.site_outcomes(["가나"], ["까나"], ["나나"])
        self.assertEqual(out["botch"], 1)
        self.assertEqual(out["repair"], 0)

    def test_perfect_restoration_is_one(self):
        out = M.site_outcomes(["가나"], ["까나"], ["가나"])
        self.assertEqual(out["net_repair"], 1.0)
        self.assertEqual(out["repair_rate"], 1.0)

    def test_damage_without_repair_goes_negative(self):
        out = M.site_outcomes(["가나"], ["까나"], ["까맘"])
        self.assertLess(out["net_repair"], 0.0)

    def test_provenance_breakdown(self):
        out = M.site_outcomes(
            ["가나"], ["까나"], ["가나"], provenances=[("tensify", "")]
        )
        self.assertIn("by_stage", out)
        self.assertEqual(out["by_stage"]["tensify"]["repair"], 1)

    def test_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            M.site_outcomes(["가나"], ["가"], ["가나"])


class TestWilsonInterval(unittest.TestCase):
    def test_zero_successes_still_has_positive_upper_bound(self):
        low, high = M.wilson_interval(0, 204)
        self.assertEqual(low, 0.0)
        # 0/204 라도 상한은 1.85% 언저리 — "1% 통과"가 약한 주장인 이유
        self.assertGreater(high, 0.015)
        self.assertLess(high, 0.025)

    def test_interval_contains_point_estimate(self):
        for successes, n in ((1, 204), (2, 204), (50, 100)):
            low, high = M.wilson_interval(successes, n)
            self.assertLessEqual(low, successes / n)
            self.assertGreaterEqual(high, successes / n)

    def test_empty_sample(self):
        self.assertEqual(M.wilson_interval(0, 0), (0.0, 0.0))

    def test_clean_mutation_rate_reports_interval(self):
        out = M.clean_mutation_rate(["가나"] * 10, ["가나"] * 10)
        self.assertEqual(out["sentence_rate"], 0.0)
        self.assertGreater(out["sentence_rate_ci95_high"], 0.0)


class TestComposedPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = P.seeds_as_items(P.load_seeds())[:60]

    def test_pairs_carry_union_technique_and_profile(self):
        profile = X.PROFILES_BY_NAME["d2_liaison_then_tensify"]
        pairs = P.make_composed_pairs(self.items, profile, 1.0, 1234)
        for pair in pairs:
            self.assertEqual(pair.technique, "liaison+tensify")
            self.assertEqual(pair.profile, profile.name)
            self.assertEqual(len(pair.obfuscated), len(pair.clean))

    def test_identity_pairs_carry_the_composite_rule(self):
        """조합 identity 페어도 조합 이름을 달아야 후보 분포가 추론 때와 같아진다."""
        profile = X.PROFILES_BY_NAME["d3_canonical"]
        pairs = P.make_composed_pairs(self.items, profile, 1.0, 1234)
        identities = [p for p in pairs if p.is_identity]
        self.assertEqual(len(identities), len(self.items))
        for pair in identities:
            self.assertEqual(pair.technique, profile.technique())
            self.assertEqual(pair.depth, 0)

    def test_extract_examples_works_on_composed_pairs(self):
        profile = X.PROFILES_BY_NAME["d3_canonical"]
        pairs = P.make_composed_pairs(self.items[:10], profile, 1.0, 1234)
        examples = F.extract_examples(pairs, window=4)
        self.assertGreater(len(examples), 0)
        self.assertTrue(all(e.technique == profile.technique() for e in examples))

    def test_depth_reflects_stage_count(self):
        pairs = P.make_composed_pairs(
            self.items[:5], X.PROFILES_BY_NAME["d3_canonical"], 1.0, 1234,
            include_identity=False,
        )
        self.assertTrue(all(p.depth == 3 for p in pairs))

    def test_make_pairs_still_rejects_composite_names(self):
        """기존 make_pairs는 조합 이름을 받지 않아야 한다 — 기준선의 의미를 지킨다."""
        with self.assertRaises(ValueError):
            P.make_pairs(self.items, ["liaison+tensify"])


if __name__ == "__main__":
    unittest.main()
