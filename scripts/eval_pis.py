#!/usr/bin/env python3
"""Run deterministic PIS checkpoint inference and export metrics/audio samples."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from intoner.paths import PIS_DATA_DIR, VOCODER_DIR, VOCODER_HINT, require_file, resolve

from intoner.pis.tasks.diffusion import DiffSpeechTask
from intoner.pis.tasks.dataset import FastSpeechDataset
from intoner.utils.hparams import hparams, set_hparams
from intoner.utils.pitch_utils import denorm_f0
from intoner.pis.modules.tts_modules import mel2ph_to_dur


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def finite_tensor(value):
    return value is not None and torch.isfinite(value).all().item()


def infer(task, batch):
    spk_embed = batch.get('spk_embed') if not hparams['use_spk_id'] else batch.get('spk_ids')
    mel2ph = batch['mel2ph'] if hparams['use_gt_dur'] else None
    f0 = batch['f0'] if hparams['use_gt_f0'] else None
    uv = batch['uv'] if hparams['use_gt_f0'] else None
    return task.model(
        batch['txt_tokens'], tone=batch['tone'], rhyme=batch['rhyme'], order=batch['order'],
        spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, energy=None,
        ref_mels=None, infer=True)


def duration_mae(predicted, target, phone_count):
    pred = mel2ph_to_dur(predicted.long(), phone_count).float()
    gold = mel2ph_to_dur(target.long(), phone_count).float()
    return float((pred - gold).abs().mean().item())


def resampled_l1(predicted, target):
    target = target[:, :predicted.shape[1], :]
    if predicted.shape[1] != target.shape[1]:
        predicted = F.interpolate(predicted.transpose(1, 2), size=target.shape[1], mode='linear', align_corners=False).transpose(1, 2)
    return float((predicted - target).abs().mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pis/diffusion.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', type=Path, default=PIS_DATA_DIR / 'binary')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates the complete test split')
    parser.add_argument('--audio-count', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    output_dir = resolve(args.output)
    if output_dir.exists():
        raise FileExistsError(f'refusing to overwrite {output_dir}')
    output_dir.mkdir(parents=True)
    audio_dir = output_dir / 'wavs'
    audio_dir.mkdir()

    checkpoint = resolve(args.checkpoint)
    data_dir = resolve(args.data_dir)
    config = resolve(args.config)
    require_file(checkpoint, 'PIS checkpoint', hint='Train one with scripts/train_pis.py.')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PIS diffusion inference')
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda')
    torch.manual_seed(1234)
    np.random.seed(1234)

    set_hparams(str(config), print_hparams=False)
    hparams['binary_data_dir'] = str(data_dir)
    hparams['fs2_ckpt'] = ''  # the diffusion checkpoint already contains the acoustic model
    hparams['load_ckpt'] = str(checkpoint)
    require_file(VOCODER_DIR / 'config.yaml', 'HiFi-GAN vocoder', hint=VOCODER_HINT)
    hparams['vocoder_ckpt'] = str(VOCODER_DIR)
    hparams['work_dir'] = str(output_dir)
    hparams['use_gt_dur'] = False
    hparams['use_gt_f0'] = False
    hparams['num_test_samples'] = 0
    hparams['test_input_dir'] = ''
    hparams['profile_infer'] = False

    task = DiffSpeechTask()
    task.build_model()
    task.model.to(device).eval()
    dataset = FastSpeechDataset('test', shuffle=False)
    total = len(dataset) if args.limit <= 0 else min(len(dataset), args.limit)
    records = []
    mel_losses, duration_losses = [], []
    f0_rmse = []

    with torch.inference_mode():
        for index in range(total):
            sample = dataset[index]
            batch = move_batch(dataset.collater([sample]), device)
            model_output = infer(task, batch)
            mel = model_output['mel_out']
            pred_mel = mel[0]
            target_mel = batch['mels'][0]
            pred_mel = pred_mel[:int((pred_mel.abs().sum(-1) > 0).sum().item())]
            target_mel = target_mel[:int((target_mel.abs().sum(-1) > 0).sum().item())]
            pred_mel_batched = pred_mel[None]
            target_mel_batched = target_mel[None]
            predicted_mel2ph = model_output.get('mel2ph')
            if predicted_mel2ph is None:
                raise RuntimeError('PIS inference did not return predicted mel2ph')
            predicted_mel2ph = predicted_mel2ph[:, :pred_mel.shape[0]]
            target_mel2ph = batch['mel2ph'][:, :target_mel.shape[0]]
            phone_count = int(batch['txt_lengths'][0].item())
            mel_l1 = resampled_l1(pred_mel_batched, target_mel_batched)
            dur_mae = duration_mae(predicted_mel2ph, target_mel2ph, phone_count)
            pred_f0 = model_output.get('f0_denorm')
            target_f0 = denorm_f0(batch['f0'], batch['uv'], hparams)
            f0_error = None
            if pred_f0 is not None and finite_tensor(pred_f0):
                pred_f0 = pred_f0[:, :pred_mel.shape[0]]
                target_f0 = target_f0[:, :target_mel.shape[0]]
                pred_f0 = F.interpolate(pred_f0[:, None, :], size=target_f0.shape[1], mode='linear', align_corners=False)[:, 0]
                voiced = target_f0 > 0
                if voiced.any():
                    f0_error = float(torch.sqrt(((pred_f0[voiced] - target_f0[voiced]) ** 2).mean()).item())
                    f0_rmse.append(f0_error)

            record = {
                'index': index,
                'item_name': sample['item_name'],
                'text': sample['text'],
                'mel_frames_predicted': int(pred_mel.shape[0]),
                'mel_frames_target': int(target_mel.shape[0]),
                'mel_l1_resampled': mel_l1,
                'duration_mae_frames': dur_mae,
                'f0_rmse_hz': f0_error,
            }
            records.append(record)
            mel_losses.append(mel_l1)
            duration_losses.append(dur_mae)

            if index < args.audio_count:
                wav = task.vocoder.spec2wav(
                    pred_mel.cpu().numpy(),
                    f0=pred_f0[0].cpu().numpy() if pred_f0 is not None and finite_tensor(pred_f0) else None,
                )
                if not np.isfinite(wav).all() or len(wav) == 0:
                    raise RuntimeError(f'invalid waveform for test index {index}')
                wav_path = audio_dir / f'{index:04d}_{sample["item_name"]}.wav'
                sf.write(wav_path, wav, hparams['audio_sample_rate'])

            if (index + 1) % 10 == 0 or index + 1 == total:
                print(f'processed {index + 1}/{total}', flush=True)

    summary = {
        'status': 'pass',
        'checkpoint': str(checkpoint),
        'data_dir': str(data_dir),
        'samples': total,
        'audio_samples': min(total, args.audio_count),
        'mel_l1_resampled_mean': float(np.mean(mel_losses)),
        'duration_mae_frames_mean': float(np.mean(duration_losses)),
        'f0_rmse_hz_mean': float(np.mean(f0_rmse)) if f0_rmse else None,
        'sample_rate': hparams['audio_sample_rate'],
    }
    (output_dir / 'metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (output_dir / 'per_sample.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
