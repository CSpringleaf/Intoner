#!/usr/bin/env python3
"""Synthesize poetry intoning from Chinese text: CFE → PIS → HiFi-GAN."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from intoner.cfe.decode import greedy_decode
from intoner.cfe.dictionary import load_dictionary
from intoner.cfe.model import load_checkpoint
from intoner.paths import CFE_DATA_DIR, PIS_DATA_DIR, VOCODER_DIR, VOCODER_HINT, require_file, resolve
from intoner.pis.features import build_pis_features
from intoner.pis.tasks.diffusion import DiffSpeechTask
from intoner.utils.cwt import get_lf0_cwt
from intoner.utils.hparams import hparams, set_hparams
from intoner.utils.text_encoder import TokenTextEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--cfe-checkpoint", required=True)
    parser.add_argument("--pis-checkpoint", required=True,
                        help="diffusion-stage checkpoint, e.g. checkpoints/pis_diffusion/model_ckpt_steps_160000.ckpt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default="configs/pis/diffusion.yaml")
    parser.add_argument("--data-dir", type=Path, default=PIS_DATA_DIR / "binary",
                        help="PIS binary data providing phone_set.json and f0 statistics")
    parser.add_argument("--dictionary", type=Path, default=CFE_DATA_DIR / "dict_modify.txt")
    parser.add_argument("--no-dictionary", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PIS synthesis")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    data_dir = resolve(args.data_dir)
    require_file(data_dir / "phone_set.json", "PIS phone set")
    require_file(VOCODER_DIR / "config.yaml", "HiFi-GAN vocoder", hint=VOCODER_HINT)
    pis_checkpoint = require_file(args.pis_checkpoint, "PIS checkpoint", hint="Train one with scripts/train_pis.py.")
    cfe_checkpoint = require_file(args.cfe_checkpoint, "CFE checkpoint", hint="Train one with scripts/train_cfe.py.")

    cfe_model, vocab = load_checkpoint(cfe_checkpoint, device=device)
    cfe = {"text": args.text, **greedy_decode(cfe_model, vocab, args.text)}
    dictionary = load_dictionary(resolve(args.dictionary), disabled=args.no_dictionary)
    if dictionary is not None:
        cfe["pinyin"], cfe["tone"] = dictionary.correct(args.text, cfe["pinyin"], cfe["tone"])
    phones, tones, rhymes, orders, used = build_pis_features(cfe)

    set_hparams(str(resolve(args.config)), print_hparams=False)
    # fs2_ckpt is unnecessary here: the diffusion checkpoint already contains the acoustic model.
    hparams.update(binary_data_dir=str(data_dir), fs2_ckpt="", load_ckpt=str(pis_checkpoint),
                   vocoder_ckpt=str(VOCODER_DIR), work_dir=str(output_dir),
                   use_gt_dur=False, use_gt_f0=False, profile_infer=False)
    _, hparams["cwt_scales"] = get_lf0_cwt(np.ones(10))
    task = DiffSpeechTask()
    task.build_model()
    task.model.to(device).eval()
    phone_encoder = TokenTextEncoder(None, vocab_list=json.loads((data_dir / "phone_set.json").read_text()),
                                     replace_oov="|")
    with torch.inference_mode():
        output = task.model(
            torch.LongTensor([phone_encoder.encode(" ".join(phones))]).to(device),
            tone=torch.LongTensor([tones]).to(device), rhyme=torch.LongTensor([rhymes]).to(device),
            order=torch.LongTensor([orders]).to(device), spk_embed=None, mel2ph=None, f0=None, uv=None,
            energy=None, ref_mels=None, infer=True)
    mel = output["mel_out"][0]
    f0 = output.get("f0_denorm")
    wav = task.vocoder.spec2wav(mel.cpu().numpy(), f0=f0[0].cpu().numpy() if f0 is not None else None)
    if len(wav) == 0 or not np.isfinite(wav).all():
        raise RuntimeError("PIS generated an invalid waveform")
    output_dir.mkdir(parents=True)
    sf.write(output_dir / "intoner.wav", wav, hparams["audio_sample_rate"])
    report = {"text": args.text, "cfe_checkpoint": str(cfe_checkpoint), "pis_checkpoint": str(pis_checkpoint),
              "cfe": cfe, "phones": phones, "used_features": used, "mel_frames": int(mel.shape[0]),
              "sample_rate": hparams["audio_sample_rate"]}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'intoner.wav'}")


if __name__ == "__main__":
    main()
