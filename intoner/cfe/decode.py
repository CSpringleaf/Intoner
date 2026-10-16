"""Greedy CFE decoding: tone and rhyme first, then pinyin and stop conditioned on them."""

from __future__ import annotations

import torch

from intoner.cfe.vocab import BOS, EOS, FIELDS, CFEVocab


def select_device(choice: str = "auto", gpu: int = 0) -> torch.device:
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(gpu)
        return torch.device("cuda", gpu)
    return torch.device("cpu")


def _greedy_pair(heads, enc_input, memory, max_steps, stop_head):
    """Decode two (decoder, projection, table) heads in lockstep until ``stop_head`` predicts EOS."""
    device = enc_input.device
    tokens = [torch.empty((1, 0), dtype=torch.long, device=device) for _ in heads]
    next_ids = [table[BOS] for _, _, table in heads]
    for _ in range(max_steps):
        for index, (decoder, projection, _) in enumerate(heads):
            tokens[index] = torch.cat([tokens[index], torch.tensor([[next_ids[index]]], device=device)], dim=-1)
            output, _, _ = decoder(tokens[index], enc_input, memory)
            next_ids[index] = projection(output)[:, -1].argmax(-1).item()
        if next_ids[stop_head] == heads[stop_head][2][EOS]:
            break
    return tokens


@torch.no_grad()
def greedy_decode(model, vocab: CFEVocab, text: str) -> dict[str, list[str]]:
    device = next(model.parameters()).device
    enc_input = torch.tensor([vocab.encode_text(text)], dtype=torch.long, device=device)
    enc_outputs, _ = model.encoder(enc_input)
    max_steps = enc_input.size(1) + 16
    tone_ids, rhythm_ids = _greedy_pair(
        [(model.tone_decoder, model.projection_2, vocab.tone),
         (model.rhythm_decoder, model.projection_3, vocab.rhythm)],
        enc_input, enc_outputs, max_steps, stop_head=1)
    tone_out, _, _ = model.tone_decoder(tone_ids, enc_input, enc_outputs)
    rhythm_out, _, _ = model.rhythm_decoder(rhythm_ids, enc_input, enc_outputs)
    if enc_outputs.size(1) + 1 == tone_out.size(1) == rhythm_out.size(1):
        memory = model.second_memory(enc_outputs, tone_out, rhythm_out)
    else:  # predicted lengths disagree with the text; fall back to the encoder output
        memory = enc_outputs
    pinyin_ids, stop_ids = _greedy_pair(
        [(model.pinyin_decoder, model.projection_1, vocab.pinyin),
         (model.stop_decoder, model.projection_4, vocab.stop)],
        enc_input, memory, max_steps, stop_head=0)
    inverse = {field: vocab.inverse(field) for field in FIELDS}
    sequences = {"pinyin": pinyin_ids, "tone": tone_ids, "rhythm": rhythm_ids, "stop": stop_ids}
    result = {"words": [inverse["word"][index] for index in enc_input[0].tolist()]}
    for field, ids in sequences.items():
        result[field] = [inverse[field][index] for index in ids[0, 1:].tolist()]
    return result
