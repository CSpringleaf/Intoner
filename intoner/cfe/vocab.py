"""CFE token vocabularies.

IDs follow the order in which symbols first appear in the labelled corpus manifest
(cfe/manifest.csv). Trained checkpoints embed their vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PUNCTUATION = "，。？！《》“”‘’、：；"
TONES = "平上去入"
RHYTHMS = "0123456789"
STOPS = "01"
PAD, BOS, EOS = "P", "S", "E"
MAX_TEXT_LEN = 360
FIELDS = ("word", "pinyin", "tone", "rhythm", "stop")
TEXT_COL, PINYIN_COL, TONE_COL, RHYTHM_COL, STOP_COL = "诗词", "拼音", "平仄", "韵脚", "停顿"


def _ordered_unique(tokens):
    seen, ordered = set(), []
    for token in tokens:
        # Exclude repeated tokens and substrings of PUNCTUATION.
        if token in seen or token in PUNCTUATION:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _with_specials(symbols):
    table = {symbol: index + 3 for index, symbol in enumerate(symbols)}
    table.update({PAD: 0, BOS: 1, EOS: 2})
    return table


@dataclass
class CFEVocab:
    word: dict[str, int]
    pinyin: dict[str, int]
    tone: dict[str, int]
    rhythm: dict[str, int]
    stop: dict[str, int]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, max_len: int = MAX_TEXT_LEN) -> "CFEVocab":
        pinyins = _ordered_unique(token for value in frame[PINYIN_COL] for token in value.split())
        words = _ordered_unique(char for text in frame[TEXT_COL] if len(text) <= max_len
                                for char in text if char != " ")
        word = {symbol: index + 1 for index, symbol in enumerate(words + list(PUNCTUATION))}
        word[PAD] = 0
        return cls(
            word=word,
            pinyin=_with_specials(pinyins + list(PUNCTUATION)),
            tone=_with_specials(list(TONES + PUNCTUATION)),
            rhythm=_with_specials(list(RHYTHMS + PUNCTUATION)),
            stop=_with_specials(list(STOPS + PUNCTUATION)),
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "CFEVocab":
        return cls.from_frame(pd.read_csv(path, sep="|", encoding="utf-8"))

    def to_dict(self) -> dict:
        return {"schema_version": 1, **{field: dict(getattr(self, field)) for field in FIELDS}}

    @classmethod
    def from_dict(cls, payload: dict) -> "CFEVocab":
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported CFE vocabulary schema")
        return cls(**{field: dict(payload[field]) for field in FIELDS})

    def inverse(self, field: str) -> dict[int, str]:
        return {index: symbol for symbol, index in getattr(self, field).items()}

    def encode_text(self, text: str) -> list[int]:
        """Map characters to IDs; unseen characters use the ID of '一', or 0 if absent."""
        fallback = self.word.get("一", 0)
        return [self.word.get(char, fallback) for char in text]
