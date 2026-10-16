"""Pronunciation dictionary D used to correct CFE predictions (paper §4.1)."""

from __future__ import annotations

import csv
from pathlib import Path


_PY_TONE_TO_CATEGORY = {
    "1": "平",
    "2": "平",
    "3": "上",
    "4": "去",
    "5": "轻",
}


class PronunciationDictionary:
    """Apply the character dictionary correction used by the CFE pipeline.

    The dictionary stores one or more pinyin values and their tone
    categories for each character.  During greedy decoding, an invalid model
    output is replaced by a dictionary value compatible with the predicted
    tone when possible; otherwise the first dictionary value is used.
    """

    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def from_file(cls, path: str | Path) -> "PronunciationDictionary":
        entries = {}
        with Path(path).open(encoding="utf8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="|"):
                character = row.get("字", "")
                if not character:
                    continue
                pinyins = [value for value in row.get("拼音", "").split(",") if value]
                tones = [value for value in row.get("平仄", "").split(",") if value]
                if pinyins:
                    entries[character] = (pinyins, tones)
        if not entries:
            raise ValueError(f"empty CFE dictionary: {path}")
        return cls(entries)

    @staticmethod
    def _tone_category(pinyin: str) -> str | None:
        if not pinyin or not pinyin[-1].isdigit():
            return None
        return _PY_TONE_TO_CATEGORY.get(pinyin[-1])

    def correct(self, text: str, pinyin: list[str], tone: list[str]):
        corrected_pinyin = list(pinyin)
        corrected_tone = list(tone)
        for index, character in enumerate(text):
            entry = self.entries.get(character)
            if entry is None:
                continue
            allowed_pinyin, allowed_tone = entry
            predicted_tone = corrected_tone[index] if index < len(corrected_tone) else None
            if index < len(corrected_pinyin) and corrected_pinyin[index] not in allowed_pinyin:
                compatible = [
                    value for value in allowed_pinyin
                    if self._tone_category(value) == predicted_tone
                ]
                corrected_pinyin[index] = compatible[0] if compatible else allowed_pinyin[0]
            if index < len(corrected_tone) and corrected_tone[index] not in allowed_tone:
                if index < len(corrected_pinyin) and corrected_pinyin[index] in allowed_pinyin:
                    pinyin_index = allowed_pinyin.index(corrected_pinyin[index])
                    if pinyin_index < len(allowed_tone):
                        corrected_tone[index] = allowed_tone[pinyin_index]
                    else:
                        corrected_tone[index] = allowed_tone[0]
                else:
                    corrected_tone[index] = allowed_tone[0]
        return corrected_pinyin, corrected_tone


def load_dictionary(path, disabled: bool = False) -> "PronunciationDictionary | None":
    """Load dictionary D unless disabled (the paper's "CFE w/o D" ablation) or missing."""
    if disabled:
        return None
    path = Path(path)
    if not path.is_file():
        print(f"| pronunciation dictionary not found at {path}; decoding without it", flush=True)
        return None
    return PronunciationDictionary.from_file(path)
