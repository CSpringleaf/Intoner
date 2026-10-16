import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from intoner.cfe.data import encode_frame
from intoner.cfe.decode import greedy_decode
from intoner.cfe.model import CFEConfig, CFETransformer, load_checkpoint, save_checkpoint
from intoner.cfe.vocab import CFEVocab

FRAME = pd.DataFrame({
    "诗词": ["春眠不觉晓，", "春晓。"],
    "拼音": ["chun1 mian2 bu4 jue2 xiao3 ，", "chun1 xiao3 。"],
    "平仄": ["平平入入上，", "平上。"],
    "韵脚": ["00001，", "01。"],
    "停顿": ["00001，", "01。"],
})
TINY = dict(d_model=16, d_ff=32, d_k=4, n_layers=1, n_heads=2)


class CFEVocabTest(unittest.TestCase):
    def test_ids_follow_first_appearance_then_punctuation(self):
        vocab = CFEVocab.from_frame(FRAME)
        self.assertEqual([vocab.word[c] for c in "春眠不觉晓，。"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(vocab.word["P"], 0)
        self.assertEqual([vocab.pinyin[p] for p in ("P", "S", "E", "chun1", "xiao3", "，")], [0, 1, 2, 3, 7, 8])
        self.assertEqual([vocab.tone[t] for t in "平上去入，"], [3, 4, 5, 6, 7])
        self.assertEqual((len(vocab.tone), len(vocab.rhythm), len(vocab.stop)), (20, 26, 18))

    def test_round_trip(self):
        vocab = CFEVocab.from_frame(FRAME)
        self.assertEqual(CFEVocab.from_dict(vocab.to_dict()), vocab)


class CFEModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vocab = CFEVocab.from_frame(FRAME)
        self.batch = encode_frame(FRAME, self.vocab)

    def test_teacher_forcing_shapes_for_both_variants(self):
        words, p_in, _, t_in, _, r_in, _, s_in, _ = self.batch
        for variant in ("cascade", "parallel"):
            with self.subTest(variant=variant):
                model = CFETransformer(CFEConfig.from_vocab(self.vocab, variant=variant, **TINY)).eval()
                pinyin, tone, rhythm, stop = model(words, p_in, t_in, r_in, s_in)
                self.assertEqual(tuple(pinyin.shape), (2 * p_in.size(1), len(self.vocab.pinyin)))
                self.assertEqual(tuple(stop.shape), (2 * s_in.size(1), len(self.vocab.stop)))
                again = model(words, p_in, t_in, r_in, s_in)[0]
                torch.testing.assert_close(pinyin, again)  # eval() disables every dropout

    def test_state_dict_parameter_names(self):
        keys = CFETransformer(CFEConfig.from_vocab(self.vocab, **TINY)).state_dict().keys()
        for key in ("encoder.pos_emb.pe", "encoder.layers.0.enc_self_attn.W_Q.weight",
                    "pinyin_decoder.layers.0.dec_enc_attn.fc.weight", "stop_decoder.layers.0.pos_ffn.fc.2.weight",
                    "projection_4.weight"):
            self.assertIn(key, keys)

    def test_checkpoint_requires_embedded_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "^checkpoint has no embedded vocabulary$"):
                load_checkpoint(path)

    def test_checkpoint_round_trip_and_decode(self):
        model = CFETransformer(CFEConfig.from_vocab(self.vocab, lambdas=(3, 2, 1), **TINY))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, model, self.vocab, epoch=0)
            loaded, vocab = load_checkpoint(path)
        self.assertEqual(vocab, self.vocab)
        self.assertEqual(loaded.config.lambdas, (3.0, 2.0, 1.0))
        result = greedy_decode(loaded, vocab, "春晓。")
        self.assertEqual(result["words"], list("春晓。"))
        for field in ("pinyin", "tone", "rhythm", "stop"):
            self.assertLessEqual(len(result[field]), len("春晓。") + 16)


if __name__ == "__main__":
    unittest.main()
