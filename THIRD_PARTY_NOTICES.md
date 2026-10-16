# Acknowledgements

The PIS implementation is adapted from [DiffSinger / DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger), released under the MIT License:

> Copyright (c) 2021 Jinglin Liu

Source files retain their copyright notices. See [LICENSE](LICENSE).

The project includes the following components:

- HiFi-GAN generator (`intoner/vocoders/hifigan_generator.py`).
- PyTorch Lightning training utilities (`intoner/utils/pl_utils.py`).
- ESPnet relative positional encoding (`intoner/pis/modules/positional_encoding.py`).
- SSIM (`intoner/pis/modules/ssim.py`).

The pre-trained HiFi-GAN vocoder weights are downloaded from the [DiffSinger releases](https://github.com/MoonInTheRiver/DiffSinger/releases/tag/pretrain-model).
