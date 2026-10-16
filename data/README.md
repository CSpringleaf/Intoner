# Data directory

Git ignores everything here except this file. Download the Intoner dataset into `data/intoner/`:

```bash
uvx --from huggingface_hub hf download Springleaf/Intoner-Dataset --repo-type dataset --local-dir data/intoner
```

Every script reads `data/intoner/` by default; pass `--data-dir` to use another location.

```text
data/intoner/
├── cfe/        manifest.csv, train.csv, test.csv, eval_paper.csv, polyphones.csv, dict_modify.txt
└── pis/
    ├── raw/        <speaker>/<item>.wav (+ .lab transcript)
    ├── processed/  metadata_phone.csv, dict.txt, mfa_dict.txt, mfa_outputs/<speaker>/<item>.TextGrid
    └── binary/     ready-to-train features
```

File formats and splits are described in [docs/dataset.md](../docs/dataset.md).
