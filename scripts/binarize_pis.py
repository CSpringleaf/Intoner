#!/usr/bin/env python3
"""Build PIS binary features from raw audio and MFA alignments (data/intoner/pis/{raw,processed})."""

import argparse
from pathlib import Path

from intoner.paths import PIS_DATA_DIR, resolve
from intoner.preprocess.binarizer import PISBinarizer
from intoner.utils.hparams import set_hparams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pis/warmup.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PIS_DATA_DIR / "raw")
    parser.add_argument("--processed-dir", type=Path, default=PIS_DATA_DIR / "processed")
    parser.add_argument("--binary-dir", type=Path, required=True,
                        help="new output directory (do not overwrite the downloaded pis/binary)")
    args = parser.parse_args()
    raw_dir, processed_dir, binary_dir = resolve(args.raw_dir), resolve(args.processed_dir), resolve(args.binary_dir)
    if binary_dir.exists() and any(binary_dir.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty {binary_dir}")
    for required in (raw_dir, processed_dir / "metadata_phone.csv", processed_dir / "dict.txt",
                     processed_dir / "mfa_outputs"):
        if not required.exists():
            raise FileNotFoundError(f"{required} is missing; see docs/dataset.md")
    set_hparams(str(resolve(args.config)), print_hparams=False,
                hparams_str=f"raw_data_dir={raw_dir},processed_data_dir={processed_dir},binary_data_dir={binary_dir}")
    PISBinarizer().process()


if __name__ == "__main__":
    main()
