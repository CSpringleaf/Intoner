#!/usr/bin/env python3
"""Predict pinyin, tone, rhyme and caesura for one poem with a trained CFE checkpoint."""

import argparse
import json
from pathlib import Path

from intoner.cfe.decode import greedy_decode, select_device
from intoner.cfe.dictionary import load_dictionary
from intoner.cfe.model import load_checkpoint
from intoner.paths import CFE_DATA_DIR, require_file, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--dictionary", type=Path, default=CFE_DATA_DIR / "dict_modify.txt")
    parser.add_argument("--no-dictionary", action="store_true", help="decode without dictionary D")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    checkpoint = require_file(args.checkpoint, "CFE checkpoint", hint="Train one with scripts/train_cfe.py.")
    model, vocab = load_checkpoint(checkpoint, device=select_device(args.device, args.gpu))
    result = {"text": args.text, "checkpoint": str(checkpoint), **greedy_decode(model, vocab, args.text)}
    dictionary = load_dictionary(resolve(args.dictionary), disabled=args.no_dictionary)
    if dictionary is not None:
        result["raw_pinyin"], result["raw_tone"] = result["pinyin"], result["tone"]
        result["pinyin"], result["tone"] = dictionary.correct(args.text, result["pinyin"], result["tone"])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
