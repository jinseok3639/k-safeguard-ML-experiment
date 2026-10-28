import unittest

from kslab import edit_tags as E


class TestEditTags(unittest.TestCase):
    def _roundtrip(self, obf, clean):
        tags, residual = E.oracle_tags(obf, clean)
        self.assertEqual(residual, 0)
        self.assertEqual(E.apply_tags(obf, tags), clean)
        return tags

    def test_substitution_and_keep(self):
        tags = self._roundtrip("씨스템", "시스템")
        self.assertEqual([t.op for t in tags], [E.SUB, E.KEEP, E.KEEP])

    def test_delete_symbols_and_restore_spacing(self):
        self._roundtrip("돈을°♡쓰는", "돈을 쓰는")
        self._roundtrip("이전지시는전부", "이전 지시는 전부")
        self._roundtrip("시 스 템", "시스템")

    def test_merge_decomposed_jamo(self):
        self._roundtrip("ㅆㅡ레기", "쓰레기")
        self._roundtrip("호와당실", "화장실")

    def test_inexpressible_counts_residual(self):
        tags, residual = E.oracle_tags("시템", "시스템")       # 음절을 끼워 넣어야 함
        self.assertEqual(residual, 1)
        self.assertEqual(E.apply_tags("시템", tags), "시템")
        self.assertEqual(E.oracle_tags("가나", "가!")[1], 1)   # 음절이 아닌 글자로 바꿔야 함

    def test_identity_is_all_keep(self):
        tags = self._roundtrip("그대로 둔다.", "그대로 둔다.")
        self.assertTrue(all(t.op == E.KEEP and not t.space_before for t in tags))

    def test_slot_residual(self):
        self.assertEqual(E.slot_residual("씨스템", "시스템"), 0)
        self.assertEqual(E.slot_residual("시 스템", "시스템"), 1)
        self.assertEqual(E.slot_residual("7ㅏ", "가나"), 2)   # 음절끼리가 아님


class TestTaggerFeatures(unittest.TestCase):
    def test_phonetic_key_groups_similar_sounds(self):
        from kslab.models.edit_tagger import jamo_slots, phonetic_key

        for a, b in [("까", "가"), ("캬", "가"), ("헤", "해"), ("돼", "되"), ("밖", "박"), ("닭", "닥"), ("옷", "옫")]:
            self.assertEqual(phonetic_key(a), phonetic_key(b), (a, b))
        self.assertNotEqual(phonetic_key("가"), phonetic_key("나"))
        self.assertEqual(jamo_slots("a"), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
