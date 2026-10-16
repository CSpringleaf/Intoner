import glob
import json
import os
import re

import librosa
import torch
import yaml

from intoner import utils
from intoner.vocoders.hifigan_generator import HifiGanGenerator
from intoner.utils.hparams import hparams
from intoner.vocoders.base import register_vocoder
from intoner.vocoders.base import BaseVocoder
import numpy as np


def load_model(config_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
    if '.yaml' in config_path:
        # Vocoder checkpoints store the full configuration; inherited training recipes may be absent.
        with open(config_path) as handle:
            config = yaml.safe_load(handle)
        state = ckpt_dict["state_dict"]["model_gen"]
    elif '.json' in config_path:
        config = json.load(open(config_path, 'r'))
        state = ckpt_dict["generator"]

    model = HifiGanGenerator(config)
    model.load_state_dict(state, strict=True)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"| Loaded model parameters from {checkpoint_path}.")
    print(f"| HifiGAN device: {device}.")
    return model, config, device


total_time = 0


@register_vocoder
class HifiGAN(BaseVocoder):
    def __init__(self):
        base_dir = hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        if os.path.exists(config_path):

            # ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
            # lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
            ckpt = f'{base_dir}/model_ckpt_steps_2076000.ckpt'
            print('| load HifiGAN: ', ckpt)
            self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)
        else:
            config_path = f'{base_dir}/config.json'
            ckpt = f'{base_dir}/generator_v1'
            if os.path.exists(config_path):
                self.model, self.config, self.device = load_model(config_path=config_path, checkpoint_path=ckpt)

    def spec2wav(self, mel, **kwargs):
        device = self.device
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
            with utils.Timer('hifigan', print_time=hparams['profile_infer']):
                f0 = kwargs.get('f0')
                if f0 is not None and hparams.get('use_nsf'):
                    f0 = torch.FloatTensor(f0[None, :]).to(device)
                    y = self.model(c, f0).view(-1)
                else:
                    y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        if hparams.get('vocoder_denoise_c', 0.0) > 0:
            wav_out = denoise(wav_out, v=hparams['vocoder_denoise_c'])
        return wav_out

    @staticmethod
    def wav2spec(wav_fn, hparams, return_linear=False):
        from intoner.preprocess.audio_features import process_utterance
        res = process_utterance(
            wav_fn, fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'],
            min_level_db=hparams['min_level_db'],
            return_linear=return_linear, vocoder='pwg', eps=float(hparams.get('wav2spec_eps', 1e-10)))
        if return_linear:
            return res[0], res[1].T, res[2].T  # [T, 80], [T, n_fft]
        else:
            return res[0], res[1].T



def denoise(wav, v=0.1):
    spec = librosa.stft(y=wav, n_fft=hparams['fft_size'], hop_length=hparams['hop_size'],
                        win_length=hparams['win_size'], pad_mode='constant')
    spec_m = np.abs(spec)
    spec_m = np.clip(spec_m - v, a_min=0, a_max=None)
    spec_a = np.angle(spec)

    return librosa.istft(spec_m * np.exp(1j * spec_a), hop_length=hparams['hop_size'],
                         win_length=hparams['win_size'])
