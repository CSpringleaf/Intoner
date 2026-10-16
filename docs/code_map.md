# Paper-to-code map

| Paper | Code |
|---|---|
| §4.1 CFE encoder and four decoders | `intoner/cfe/model.py` (`CFETransformer`) |
| §4.1 Eq. 3, weighted memory λx·E + λt·h_t + λr·h_r | `CFETransformer.second_memory` |
| §4.1 dictionary D | `intoner/cfe/dictionary.py` |
| §4.1 greedy decoding | `intoner/cfe/decode.py` |
| §6.1 CFE training (λpunc, SGD) | `intoner/cfe/trainer.py` |
| §4.1 Eq. 4, CFE output → phonemes with caesura | `intoner/pis/features.py` |
| §4.2 tone/rhyme embeddings, BiLSTM length regulation | `intoner/pis/modules/tts_modules.py` |
| §4.2 acoustic model, sentence-order embedding | `intoner/pis/modules/fastspeech2.py` |
| §4.2 warmup stage | `intoner/pis/tasks/acoustic.py`, `configs/pis/warmup.yaml` |
| §4.2 shallow diffusion | `intoner/pis/modules/shallow_diffusion.py`, `intoner/pis/modules/denoiser.py`, `intoner/pis/tasks/diffusion.py`, `configs/pis/diffusion.yaml` |
| §5.2 feature extraction (pitch, duration, alignment) | `intoner/preprocess/` |
| Vocoder | `intoner/vocoders/hifigan.py` |
