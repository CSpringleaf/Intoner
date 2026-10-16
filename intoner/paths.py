"""Default locations and public download addresses used by every entry point.

This module is the single source of truth for paths and URLs; README.md,
README.zh-CN.md, data/README.md and docs/dataset.md quote the same values.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_REPO_ID = "Springleaf/Intoner-Dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
DATASET_DOWNLOAD_CMD = (
    f"uvx --from huggingface_hub hf download {DATASET_REPO_ID} --repo-type dataset --local-dir data/intoner"
)
DATA_DIR = REPO_ROOT / "data" / "intoner"
CFE_DATA_DIR = DATA_DIR / "cfe"
PIS_DATA_DIR = DATA_DIR / "pis"

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
CFE_CHECKPOINT_DIR = CHECKPOINT_DIR / "cfe"
VOCODER_DIR = CHECKPOINT_DIR / "0414_hifi_lj_1"
VOCODER_URL = "https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip"

DATASET_HINT = f"Download the dataset with `{DATASET_DOWNLOAD_CMD}` (see data/README.md)."
VOCODER_HINT = f"Download {VOCODER_URL} and unzip it into checkpoints/ (see README.md)."


def resolve(path: str | Path) -> Path:
    """Resolve a user path: absolute as given, relative to the working directory, else to the repository root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    from_cwd = Path.cwd() / path
    return from_cwd if from_cwd.exists() else REPO_ROOT / path


def require_file(path: str | Path, what: str, hint: str = DATASET_HINT) -> Path:
    """Return the resolved path or fail with an actionable message."""
    resolved = resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{what} not found: {resolved}. {hint}")
    return resolved
