# Intoner: Chinese Poetry Intoning Synthesis

[中文说明](README.zh-CN.md) · [Paper (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1139) · [Dataset](https://huggingface.co/datasets/Springleaf/Intoner-Dataset) · [Code map](docs/code_map.md)

Official implementation of **Intoner: For Chinese Poetry Intoning Synthesis** (IJCAI 2025). Intoner turns the text of a classical Chinese poem into intoning (吟诵) audio without a musical score.

```text
poem text ─► CFE ─► pinyin · tone · rhyme · caesura ─► PIS ─► mel-spectrogram ─► HiFi-GAN ─► audio
```

- **CFE** (Character-level Feature Extraction, §4.1): a Transformer with one character encoder and four decoders. Tone and rhyme are predicted first and condition the pinyin and caesura decoders. A pronunciation dictionary corrects invalid predictions.
- **PIS** (Poetry Intoning Synthesis, §4.2): a FastSpeech2-style acoustic model with tone, rhyme and sentence-order embeddings (warmup stage), refined by a shallow diffusion decoder (main stage).

Pre-trained Intoner checkpoints are not released. The public dataset and this repository contain everything needed to train both modules.

## Installation

Requirements: Linux, Python 3.10, [uv](https://docs.astral.sh/uv/). PIS needs a CUDA GPU; CFE also runs on CPU.

```bash
cd Intoner
uv sync
```

The lock file pins PyTorch 1.13.1. Run every command from the repository root.

## Data and vocoder

Download the dataset into `data/intoner/` (layout and formats: [docs/dataset.md](docs/dataset.md)):

```bash
uvx --from huggingface_hub hf download Springleaf/Intoner-Dataset --repo-type dataset --local-dir data/intoner
```

Download the pre-trained HiFi-GAN vocoder released by DiffSinger (≈940 MB):

```bash
curl -L -o /tmp/0414_hifi_lj_1.zip https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip
unzip /tmp/0414_hifi_lj_1.zip -d checkpoints/
```

## Training

**CFE**

```bash
uv run python scripts/train_cfe.py
```

Checkpoints are written to `checkpoints/cfe/model_<epoch>.pt` (every 200 epochs and at the last of 1500 epochs). Each checkpoint embeds its vocabulary. `--variant parallel` trains the parallel-decoder ablation; `--device cpu` forces CPU.

**PIS**

```bash
uv run python scripts/train_pis.py --gpu 0
```

This runs the warmup stage (120k steps, `checkpoints/pis_warmup/`) and then the diffusion stage (160k steps, `checkpoints/pis_diffusion/`). Progress is tracked in `logs/pis/status.json`; re-running the command resumes. Use `--stage warmup|diffusion` to run one stage. The paper reports about 17 hours on a single RTX 4090.

The dataset ships ready-to-train features in `pis/binary/`. To rebuild them from the raw audio and MFA alignments:

```bash
uv run python scripts/binarize_pis.py --binary-dir data/pis_binary_rebuilt
uv run python scripts/train_pis.py --gpu 0 --data-dir data/pis_binary_rebuilt
```

## Inference

```bash
uv run python scripts/synthesize.py \
  --text '春草明年绿，王孙归不归。' \
  --cfe-checkpoint checkpoints/cfe/model_1500.pt \
  --pis-checkpoint checkpoints/pis_diffusion/model_ckpt_steps_160000.ckpt \
  --output-dir outputs/sample
```

This writes `outputs/sample/intoner.wav` and `report.json`. To run CFE alone:

```bash
uv run python scripts/infer_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --text '春草明年绿，王孙归不归。'
```

Add `--no-dictionary` to either command to disable dictionary correction.

## Evaluation

```bash
uv run python scripts/eval_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --output outputs/cfe_eval.json
uv run python scripts/eval_pis.py --checkpoint checkpoints/pis_diffusion/model_ckpt_steps_160000.ckpt --output outputs/pis_eval
```

For the 213-poem CFE evaluation set used in Table 3, run:

```bash
uv run python scripts/eval_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --csv data/intoner/cfe/eval_paper.csv --output outputs/cfe_eval_paper.json
```

The output includes pinyin, polyphone (`poly`), tone, rhyme and stop accuracies; add `--no-dictionary` for CFE w/o D.

## Repository layout

```text
intoner/cfe/         CFE model, vocabulary, decoding and training
intoner/pis/         PIS acoustic model, shallow diffusion and training tasks
intoner/preprocess/  raw audio + alignments → binary features
intoner/vocoders/    HiFi-GAN inference
intoner/utils/       hyper-parameters, trainer and shared helpers
configs/pis/         PIS recipes: base, warmup, diffusion
scripts/             training, inference and evaluation entry points
data/                dataset location (see data/README.md)
```

`checkpoints/`, `logs/` and `outputs/` are created locally and ignored by Git.

## Citation

```bibtex
@inproceedings{zuo2025intoner,
  title     = {Intoner: For Chinese Poetry Intoning Synthesis},
  author    = {Zuo, Heda and Sun, Liyao and Lai, Zeyu and You, Weitao and Chen, Pei and Sun, Lingyun},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  pages     = {10252--10260},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/1139}
}
```

## License and acknowledgements

The PIS implementation is adapted from [DiffSinger / DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger), and the vocoder checkpoint is DiffSinger's pre-trained HiFi-GAN. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
