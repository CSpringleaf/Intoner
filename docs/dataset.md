# Intoner dataset

Hugging Face: [`Springleaf/Intoner-Dataset`](https://huggingface.co/datasets/Springleaf/Intoner-Dataset). Download it into `data/intoner/`:

```bash
uvx --from huggingface_hub hf download Springleaf/Intoner-Dataset --repo-type dataset --local-dir data/intoner
```

```text
data/intoner/
├── cfe/
│   ├── manifest.csv          full labelled corpus (CFESet, paper §5.1)
│   ├── train.csv
│   ├── test.csv
│   ├── eval_paper.csv        213 poems used for paper Table 3
│   ├── polyphones.csv       character list for Table 3 poly accuracy
│   └── dict_modify.txt       pronunciation dictionary D (optional)
└── pis/                      PISSet (paper §5.2)
    ├── raw/<speaker>/<item>.{wav,lab}
    ├── processed/
    │   ├── metadata_phone.csv
    │   ├── dict.txt
    │   ├── mfa_dict.txt
    │   └── mfa_outputs/<speaker>/<item>.TextGrid
    └── binary/
        ├── phone_set.json  spk_map.json  train_f0s_mean_std.npy
        └── {train,valid,test}.{idx,data}  {train,valid,test}_lengths.npy
```

## CFE tables

`manifest.csv`, `train.csv`, `test.csv` and `eval_paper.csv` are UTF-8, `|`-delimited, with columns
`吟诵|诗词|拼音|平仄|韵脚|体裁|停顿` (sample id, poem text, space-separated pinyin, tone per character,
rhyme per character, poem type, caesura per character).

CFE token IDs follow the order of first appearance in `manifest.csv`. `scripts/train_cfe.py` builds the vocabulary from it and stores the vocabulary inside every checkpoint.

The full table contains 10,765 records; `train.csv` contains 9,689 and `test.csv` contains 1,076.
The separate `eval_paper.csv` contains the 213 poems used for Table 3. Evaluate it with:

```bash
uv run python scripts/eval_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --csv data/intoner/cfe/eval_paper.csv --output outputs/cfe_eval_paper.json
```

`dict_modify.txt` is `|`-delimited with columns `字|拼音|平仄`; multiple readings are comma-separated.

`polyphones.csv` is the character list used for Table 3 polyphone accuracy, with columns `字|多音|出现音`.

The output includes pinyin, polyphone (`poly`), tone, rhyme and stop accuracies; add `--no-dictionary` for CFE w/o D.

## PIS data

- `raw/<speaker>/<item>.wav`: 22.05 kHz sentence-level recordings from 24 intoners (20,112 items); `<item>.lab` holds the transcript given to the aligner.
- `processed/metadata_phone.csv`: columns `item_name,spk,txt,txt_raw,ph,wav_fn,tone,rhyme`. `ph` is the phoneme sequence with `|` word boundaries; `tone`/`rhyme` hold one symbol per character. The last two digits of `item_name` give the sentence order within its poem.
- `processed/dict.txt`: phoneme inventory. `processed/mfa_dict.txt`: pronunciation dictionary for the Montreal Forced Aligner. `processed/mfa_outputs/`: MFA TextGrids (AISHELL-3 acoustic model).
- `binary/`: ready-to-train features. Items are sorted by name; the first 523 form the test split, the next 871 the validation split, and the rest the training split.

In both `metadata_phone.csv` and the released binary records, `wav_fn` is relative to the dataset root:
`pis/raw/<speaker>/<item>.wav`. Resolve it against the downloaded `data/intoner/` directory.
