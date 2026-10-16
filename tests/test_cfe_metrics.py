import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from intoner.cfe.metrics import load_polyphones, polyphone_counts
from scripts import eval_cfe


class PolyphoneMetricsTest(unittest.TestCase):
    def test_counts_occurrences_and_excludes_truncated_predictions(self):
        self.assertEqual(polyphone_counts(
            "重春重行", ["chong2", "chun1", "zhong4", "xing2"],
            ["chong2", "wrong", "chong2"], {"重", "行"}), (1, 2))
        self.assertEqual(polyphone_counts("春", ["chun1"], [], {"重"}), (0, 0))
        with self.assertRaises(ValueError):
            polyphone_counts("重", [], [], {"重"})

    def test_reads_character_membership_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "polyphones.csv"
            path.write_text("字|多音|出现音\n重|zhong4,chong2|chong2\n", encoding="utf-8")
            self.assertEqual(load_polyphones(path), {"重"})
            path.write_text("字\n重\n重\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_polyphones(path)

    def test_cli_aggregates_tokens_after_optional_dictionary_correction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            polyphones = root / "polyphones.csv"
            polyphones.write_text("字|多音|出现音\n重|zhong4,chong2|chong2\n行|hang2,xing2|xing2\n", encoding="utf-8")
            data = root / "eval.csv"
            with data.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter="|")
                writer.writerows([
                    ["诗词", "拼音", "平仄", "韵脚", "停顿"],
                    ["重重轻", "chong2 zhong4 qing1", "平去平", "001", "000"],
                    ["行", "xing2", "平", "1", "0"],
                ])

            def decode(model, vocab, text):
                if text == "行":
                    return {"pinyin": ["hang2"], "tone": ["平"], "rhythm": ["1"], "stop": ["0"]}
                return {"pinyin": ["zhong4", "zhong4", "qing1"],
                        "tone": list("平去平"), "rhythm": list("001"), "stop": list("000")}

            dictionary = Mock()
            dictionary.correct.side_effect = lambda text, pinyin, tone: (
                ["xing2"] if text == "行" else pinyin, tone)
            for disabled, start, correct, total in ((False, 0, 2, 3), (True, 0, 1, 3), (False, 2, 0, 0)):
                with self.subTest(disabled=disabled, start=start):
                    output = root / f"eval-{disabled}-{start}.json"
                    argv = ["eval_cfe", "--checkpoint", str(checkpoint), "--csv", str(data),
                            "--polyphones", str(polyphones), "--output", str(output), "--start", str(start)]
                    if disabled:
                        argv.append("--no-dictionary")
                    with patch.object(sys, "argv", argv), \
                         patch.object(eval_cfe, "select_device", return_value="cpu"), \
                         patch.object(eval_cfe, "load_checkpoint", return_value=(None, None)), \
                         patch.object(eval_cfe, "greedy_decode", side_effect=decode), \
                         patch.object(eval_cfe, "load_dictionary", side_effect=lambda path, disabled: None if disabled else dictionary), \
                         contextlib.redirect_stdout(io.StringIO()):
                        eval_cfe.main()
                    result = json.loads(output.read_text())
                    summary = result["summary"]
                    self.assertEqual(summary["polyphone_counts"], {"correct": correct, "total": total})
                    self.assertEqual(summary["accuracy"]["poly"], correct / total if total else None)
                    self.assertEqual(summary["dictionary"], not disabled)
                    if total:
                        self.assertEqual(result["records"][0]["polyphone_counts"], {"correct": 1, "total": 2})
                        self.assertEqual(result["records"][1]["raw_predicted"]["pinyin"], ["hang2"])
                        self.assertAlmostEqual(summary["accuracy"]["pinyin"], 1 / 3 if disabled else 5 / 6)


if __name__ == "__main__":
    unittest.main()
