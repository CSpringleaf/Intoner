#!/usr/bin/env python3
"""Train the CFE model (paper §4.1) on data/intoner/cfe."""

import argparse
from pathlib import Path

from intoner.cfe.decode import select_device
from intoner.cfe.model import VARIANTS
from intoner.cfe.trainer import train
from intoner.paths import CFE_CHECKPOINT_DIR, CFE_DATA_DIR, require_file, resolve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=CFE_DATA_DIR,
                        help="directory with manifest.csv, train.csv and test.csv")
    parser.add_argument("--output-dir", type=Path, default=CFE_CHECKPOINT_DIR)
    parser.add_argument("--variant", choices=VARIANTS, default="cascade",
                        help="'parallel' is the paper's parallel-decoder ablation")
    parser.add_argument("--lambdas", type=float, nargs=3, default=[3.0, 2.0, 1.0],
                        metavar=("X", "TONE", "RHYME"), help="memory weights λx λt λr (Eq. 3)")
    parser.add_argument("--punc-weight", type=float, default=8.0, help="λpunc for punctuation in the pinyin loss")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    data_dir = resolve(args.data_dir)
    train(manifest=require_file(data_dir / "manifest.csv", "CFE manifest"),
          train_csv=require_file(data_dir / "train.csv", "CFE training split"),
          test_csv=require_file(data_dir / "test.csv", "CFE test split"),
          output_dir=resolve(args.output_dir), variant=args.variant, lambdas=args.lambdas,
          punc_weight=args.punc_weight, epochs=args.epochs, batch_size=args.batch_size,
          device=select_device(args.device, args.gpu))


if __name__ == "__main__":
    main()
