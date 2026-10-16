import unittest

from intoner.cfe.dictionary import PronunciationDictionary


class PronunciationDictionaryTest(unittest.TestCase):
    def test_invalid_prediction_uses_tone_compatible_pinyin(self):
        dictionary = PronunciationDictionary({"行": (["xing2", "hang2"], ["平", "平"])})
        pinyin, tone = dictionary.correct("行", ["wrong"], ["平"])
        self.assertEqual(pinyin, ["xing2"])
        self.assertEqual(tone, ["平"])

    def test_unknown_characters_are_unchanged(self):
        dictionary = PronunciationDictionary({"行": (["xing2"], ["平"])})
        pinyin, tone = dictionary.correct("春", ["chun1"], ["平"])
        self.assertEqual((pinyin, tone), (["chun1"], ["平"]))


if __name__ == "__main__":
    unittest.main()
