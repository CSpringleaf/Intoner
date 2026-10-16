# Intoner：中文诗词吟诵合成

[English](README.md) · [论文（IJCAI 2025）](https://www.ijcai.org/proceedings/2025/1139) · [数据集](https://huggingface.co/datasets/Springleaf/Intoner-Dataset) · [代码索引](docs/code_map.md)

**Intoner: For Chinese Poetry Intoning Synthesis**（IJCAI 2025）的官方实现。Intoner 无需乐谱，即可将古典诗词文本转为吟诵音频。

```text
诗词文本 ─► CFE ─► 拼音 · 平仄 · 韵脚 · 停顿 ─► PIS ─► 梅尔频谱 ─► HiFi-GAN ─► 音频
```

- **CFE**（字符级特征提取，§4.1）：包含一个字符编码器和四个解码器的 Transformer。先预测平仄和韵脚，再将其作为拼音和停顿解码器的条件。发音词典用于修正无效预测。
- **PIS**（诗词吟诵合成，§4.2）：在预热阶段训练带平仄、韵脚和句序嵌入的 FastSpeech2 风格声学模型，再在主训练阶段通过浅层扩散解码器优化。

本项目不发布预训练的 Intoner checkpoint。公开数据集和本仓库提供训练两个模块所需的内容。

## 安装

环境要求：Linux、Python 3.10、[uv](https://docs.astral.sh/uv/)。PIS 需要 CUDA GPU；CFE 也可在 CPU 上运行。

```bash
cd Intoner
uv sync
```

锁文件固定使用 PyTorch 1.13.1。所有命令均从仓库根目录运行。

## 数据与声码器

将数据集下载到 `data/intoner/`（目录结构和格式见 [docs/dataset.md](docs/dataset.md)）：

```bash
uvx --from huggingface_hub hf download Springleaf/Intoner-Dataset --repo-type dataset --local-dir data/intoner
```

下载 DiffSinger 发布的预训练 HiFi-GAN 声码器（约 940 MB）：

```bash
curl -L -o /tmp/0414_hifi_lj_1.zip https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0414_hifi_lj_1.zip
unzip /tmp/0414_hifi_lj_1.zip -d checkpoints/
```

## 训练

**CFE**

```bash
uv run python scripts/train_cfe.py
```

Checkpoint 写入 `checkpoints/cfe/model_<epoch>.pt`（每 200 个 epoch 保存一次，并在第 1500 个、即最后一个 epoch 保存）。每个 checkpoint 都内嵌词表。`--variant parallel` 用于训练并行解码器消融版本；`--device cpu` 强制使用 CPU。

**PIS**

```bash
uv run python scripts/train_pis.py --gpu 0
```

依次运行预热阶段（120k 步，`checkpoints/pis_warmup/`）和扩散阶段（160k 步，`checkpoints/pis_diffusion/`）。进度记录在 `logs/pis/status.json`；重新运行命令可续训。使用 `--stage warmup|diffusion` 可单独运行一个阶段。论文报告在单张 RTX 4090 上训练约需 17 小时。

数据集在 `pis/binary/` 中提供可直接训练的特征。若要从原始音频与 MFA 对齐结果重新生成：

```bash
uv run python scripts/binarize_pis.py --binary-dir data/pis_binary_rebuilt
uv run python scripts/train_pis.py --gpu 0 --data-dir data/pis_binary_rebuilt
```

## 推理

```bash
uv run python scripts/synthesize.py \
  --text '春草明年绿，王孙归不归。' \
  --cfe-checkpoint checkpoints/cfe/model_1500.pt \
  --pis-checkpoint checkpoints/pis_diffusion/model_ckpt_steps_160000.ckpt \
  --output-dir outputs/sample
```

该命令生成 `outputs/sample/intoner.wav` 和 `report.json`。如需单独运行 CFE：

```bash
uv run python scripts/infer_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --text '春草明年绿，王孙归不归。'
```

在任一命令中添加 `--no-dictionary` 可禁用词典修正。

## 评测

```bash
uv run python scripts/eval_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --output outputs/cfe_eval.json
uv run python scripts/eval_pis.py --checkpoint checkpoints/pis_diffusion/model_ckpt_steps_160000.ckpt --output outputs/pis_eval
```

使用论文 Table 3 的 213 首诗 CFE 评测集：

```bash
uv run python scripts/eval_cfe.py --checkpoint checkpoints/cfe/model_1500.pt --csv data/intoner/cfe/eval_paper.csv --output outputs/cfe_eval_paper.json
```

输出包含拼音、多音字（`poly`）、声调、韵脚、停顿准确率；加 `--no-dictionary` 即 CFE w/o D。

## 仓库结构

```text
intoner/cfe/         CFE 模型、词表、解码与训练
intoner/pis/         PIS 声学模型、浅层扩散与训练任务
intoner/preprocess/  原始音频与对齐结果 → 二值化特征
intoner/vocoders/    HiFi-GAN 推理
intoner/utils/       超参数、训练器与共享工具
configs/pis/         PIS 配置：base、warmup、diffusion
scripts/             训练、推理与评测入口
data/                数据集目录（见 data/README.md）
```

`checkpoints/`、`logs/` 和 `outputs/` 在本地创建，由 Git 忽略。

## 引用

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

## 许可与致谢

PIS 实现改编自 [DiffSinger / DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger)，声码器 checkpoint 使用 DiffSinger 提供的预训练 HiFi-GAN。详见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
