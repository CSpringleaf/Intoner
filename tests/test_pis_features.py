import unittest

from intoner.pis.features import build_pis_features


class PisFeaturesTest(unittest.TestCase):
    def test_cfe_features_preserve_sentence_order(self):
        result = {
            'text': '春。草',
            'pinyin': ['chun1', '。', 'cao3'],
            'tone': ['平', '空', '上'],
            'rhythm': ['1', 'e', '2'],
        }
        phones, tones, rhymes, orders, used = build_pis_features(result)
        self.assertEqual(phones, ['ch', 'un1', '|', 'c', 'ao3', '|'])
        self.assertEqual(orders, [1, 1, 1, 2, 2, 2])
        self.assertEqual(len(used), 2)



if __name__ == "__main__":
    unittest.main()
