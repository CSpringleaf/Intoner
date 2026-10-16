#!/usr/bin/env python3
"""Evaluate CFE accuracy (pinyin, polyphone pinyin, tone, rhyme, stop)."""

import argparse
import csv
import json
from pathlib import Path

import torch

from intoner.cfe.decode import greedy_decode, select_device
from intoner.cfe.dictionary import load_dictionary
from intoner.cfe.metrics import load_polyphones, polyphone_counts
from intoner.cfe.model import load_checkpoint
from intoner.cfe.vocab import PINYIN_COL, RHYTHM_COL, STOP_COL, TEXT_COL, TONE_COL
from intoner.paths import CFE_DATA_DIR, require_file, resolve

FIELDS = ("pinyin", "tone", "rhythm", "stop")


def accuracy(expected, predicted):
    """Position-wise accuracy over the shorter sequence."""
    length = min(len(expected), len(predicted))
    return sum(a == b for a, b in zip(expected[:length], predicted[:length])) / length if length else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", type=Path, default=CFE_DATA_DIR / "test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, default=CFE_DATA_DIR / "dict_modify.txt")
    parser.add_argument("--no-dictionary", action="store_true")
    parser.add_argument("--polyphones", type=Path, default=CFE_DATA_DIR / "polyphones.csv",
                        help="fixed character list used for Table 3 poly accuracy")
    parser.add_argument("--start", type=int, default=0, help="first row (for sharding across GPUs)")
    parser.add_argument("--limit", type=int, default=0, help="number of rows; 0 evaluates every row")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checkpoint = require_file(args.checkpoint, "CFE checkpoint", hint="Train one with scripts/train_cfe.py.")
    csv_path = require_file(args.csv, "CFE evaluation split")
    polyphones_path = require_file(args.polyphones, "CFE polyphone table")
    polyphones = load_polyphones(polyphones_path)
    model, vocab = load_checkpoint(checkpoint, device=select_device(args.device, args.gpu))
    dictionary = load_dictionary(resolve(args.dictionary), disabled=args.no_dictionary)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="|") if row.get(TEXT_COL)]
    start = max(0, args.start)
    rows = rows[start:] if args.limit <= 0 else rows[start:start + args.limit]
    records, sums = [], dict.fromkeys(FIELDS, 0.0)
    poly_correct = poly_total = 0
    with torch.inference_mode():
        for offset, row in enumerate(rows):
            text = row[TEXT_COL]
            decoded = greedy_decode(model, vocab, text)
            predicted = {field: decoded[field] for field in FIELDS}
            raw_predicted = {field: list(values) for field, values in predicted.items()}
            if dictionary is not None:
                predicted["pinyin"], predicted["tone"] = dictionary.correct(text, predicted["pinyin"], predicted["tone"])
            expected = {"pinyin": row[PINYIN_COL].split(), "tone": list(row[TONE_COL]),
                        "rhythm": list(row[RHYTHM_COL]), "stop": list(row[STOP_COL])}
            scores = {field: accuracy(expected[field], predicted[field]) for field in FIELDS}
            for field, value in scores.items():
                sums[field] += value
            correct, total = polyphone_counts(text, expected["pinyin"], predicted["pinyin"], polyphones)
            poly_correct += correct
            poly_total += total
            scores["poly"] = correct / total if total else None
            records.append({"index": start + offset, "text": text, "expected": expected,
                            "raw_predicted": raw_predicted, "predicted": predicted, "accuracy": scores,
                            "polyphone_counts": {"correct": correct, "total": total}})
            if (offset + 1) % 10 == 0 or offset + 1 == len(rows):
                print(f"processed {offset + 1}/{len(rows)}", flush=True)
    summary = {"checkpoint": str(checkpoint), "csv": str(csv_path), "start": start, "samples": len(rows),
               "dictionary": dictionary is not None,
               "polyphones": str(polyphones_path),
               "polyphone_counts": {"correct": poly_correct, "total": poly_total},
               "accuracy": {field: sums[field] / len(rows) for field in FIELDS} if rows else {}}
    summary["accuracy"]["poly"] = poly_correct / poly_total if poly_total else None
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
