"""Polyphone accuracy for the CFE evaluation in Table 3."""

import csv


def load_polyphones(path):
    """Read the character set from the 字 column of a pipe-delimited table."""
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        if reader.fieldnames is None or "字" not in reader.fieldnames:
            raise ValueError("Polyphone table requires a 字 column")
        characters = [row["字"] for row in reader]
    if not characters or any(not isinstance(char, str) or len(char) != 1 for char in characters):
        raise ValueError("Polyphone table must contain one character per row")
    if len(set(characters)) != len(characters):
        raise ValueError("Polyphone table contains duplicate characters")
    return frozenset(characters)


def polyphone_counts(text, expected, predicted, polyphones):
    """Return correct and total counts for listed characters within the predicted sequence."""
    correct = total = 0
    for index, char in enumerate(text):
        if char not in polyphones:
            continue
        if index >= len(expected):
            raise ValueError(f"Missing reference pinyin for polyphone at position {index}")
        if index >= len(predicted):
            continue
        total += 1
        correct += int(predicted[index] == expected[index])
    return correct, total
