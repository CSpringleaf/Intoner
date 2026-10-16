"""Teacher-forcing tensors for CFE training, padded per split."""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from intoner.cfe.vocab import (BOS, EOS, MAX_TEXT_LEN, PINYIN_COL, RHYTHM_COL, STOP_COL, TEXT_COL,
                               TONE_COL, CFEVocab)


def encode_frame(frame: pd.DataFrame, vocab: CFEVocab, max_len: int = MAX_TEXT_LEN):
    """Return words, pinyin_in, pinyin_out, tone_in, tone_out, rhythm_in, rhythm_out, stop_in, stop_out."""
    columns = [[] for _ in range(9)]
    rows = zip(frame[TEXT_COL], frame[PINYIN_COL], frame[TONE_COL], frame[RHYTHM_COL], frame[STOP_COL])
    for text, pinyin, tone, rhythm, stop in rows:
        if len(text) > max_len:
            continue
        columns[0].append(torch.LongTensor([vocab.word[char] for char in text]))
        labels = ((vocab.pinyin, pinyin.split()), (vocab.tone, tone), (vocab.rhythm, rhythm), (vocab.stop, stop))
        for slot, (table, tokens) in enumerate(labels):
            ids = [table[token] for token in tokens]
            columns[1 + 2 * slot].append(torch.LongTensor([table[BOS]] + ids))
            columns[2 + 2 * slot].append(torch.LongTensor(ids + [table[EOS]]))
    return tuple(pad_sequence(column, batch_first=True, padding_value=0) for column in columns)


def make_loader(csv_path, vocab: CFEVocab, batch_size: int, max_len: int = MAX_TEXT_LEN, shuffle: bool = True):
    frame = pd.read_csv(csv_path, sep="|", encoding="utf-8")
    return DataLoader(TensorDataset(*encode_frame(frame, vocab, max_len)), batch_size=batch_size, shuffle=shuffle)
