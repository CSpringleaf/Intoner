"""Character-level Feature Extraction (CFE) Transformer, paper §4.1.

One character encoder and four autoregressive decoders (pinyin, tone, rhyme, stop).
In the ``cascade`` variant the tone and rhyme decoder states are mixed into the memory
of the pinyin and stop decoders with weights ``lambdas`` = (λx, λt, λr) (Eq. 3). The
``parallel`` variant, an ablation in the paper, feeds the encoder output to all decoders.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from intoner.cfe.vocab import CFEVocab

VARIANTS = ("cascade", "parallel")


@dataclass
class CFEConfig:
    src_vocab_size: int
    pinyin_vocab_size: int
    tone_vocab_size: int
    rhythm_vocab_size: int
    stop_vocab_size: int
    variant: str = "cascade"
    lambdas: tuple = (3.0, 2.0, 1.0)
    d_model: int = 256
    d_ff: int = 1024
    d_k: int = 64
    n_layers: int = 3
    n_heads: int = 4

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        self.lambdas = tuple(float(value) for value in self.lambdas)
        if len(self.lambdas) != 3 or sum(self.lambdas) == 0:
            raise ValueError("lambdas must contain three weights with a non-zero sum")

    @classmethod
    def from_vocab(cls, vocab: CFEVocab, **overrides) -> "CFEConfig":
        return cls(src_vocab_size=len(vocab.word), pinyin_vocab_size=len(vocab.pinyin),
                   tone_vocab_size=len(vocab.tone), rhythm_vocab_size=len(vocab.rhythm),
                   stop_vocab_size=len(vocab.stop), **overrides)

    def to_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["lambdas"] = list(self.lambdas)
        return payload


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.4, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))  # [max_len, 1, d_model]

    def forward(self, x):
        """x: [seq_len, batch, d_model]"""
        return self.dropout(x + self.pe[:x.size(0)])


def pad_mask(seq_q, seq_k):
    """True where the key position is padding (ID 0): [batch, len_q, len_k]."""
    return seq_k.eq(0).unsqueeze(1).expand(seq_q.size(0), seq_q.size(1), seq_k.size(1))


def subsequent_mask(seq):
    """True above the diagonal, hiding future positions: [batch, len, len]."""
    length = seq.size(1)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=seq.device), diagonal=1)
    return mask.unsqueeze(0).expand(seq.size(0), length, length)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.4):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        attn = self.dropout(torch.softmax(scores.masked_fill(mask, -1e9), dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, dropout):
        super().__init__()
        self.d_model, self.d_k, self.n_heads = d_model, d_k, n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        # Attention dropout is active during training and disabled in eval mode.
        self.attention = ScaledDotProductAttention(d_k, dropout=0.4)

    def forward(self, input_q, input_k, input_v, mask):
        batch = input_q.size(0)
        q = self.W_Q(input_q).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(input_k).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(input_v).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.attention(q, k, v, mask)
        context = context.transpose(1, 2).reshape(batch, -1, self.n_heads * self.d_k)
        output = self.dropout(self.fc(context))
        # Apply parameter-free layer normalization to the residual output.
        return F.layer_norm(output + input_q, (self.d_model,)), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.ReLU(),
                                nn.Linear(d_ff, d_model, bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        return F.layer_norm(self.dropout(self.fc(inputs)) + inputs, (self.d_model,))


class EncoderLayer(nn.Module):
    def __init__(self, config, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(config.d_model, config.d_k, config.n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(config.d_model, config.d_ff, dropout)

    def forward(self, x, mask):
        x, attn = self.enc_self_attn(x, x, x, mask)
        return self.pos_ffn(x), attn


class DecoderLayer(nn.Module):
    def __init__(self, config, dropout):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention(config.d_model, config.d_k, config.n_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(config.d_model, config.d_k, config.n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(config.d_model, config.d_ff, dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x, self_attn = self.dec_self_attn(x, x, x, self_mask)
        x, cross_attn = self.dec_enc_attn(x, memory, memory, cross_mask)
        return self.pos_ffn(x), self_attn, cross_attn


class Encoder(nn.Module):
    def __init__(self, config, dropout):
        super().__init__()
        self.src_emb = nn.Embedding(config.src_vocab_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config, dropout) for _ in range(config.n_layers)])

    def forward(self, enc_inputs):
        x = self.pos_emb(self.src_emb(enc_inputs).transpose(0, 1)).transpose(0, 1)
        mask = pad_mask(enc_inputs, enc_inputs)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attns.append(attn)
        return x, attns


class Decoder(nn.Module):
    def __init__(self, config, vocab_size, dropout):
        super().__init__()
        self.tgt_emb = nn.Embedding(vocab_size, config.d_model)
        self.pos_emb = PositionalEncoding(config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config, dropout) for _ in range(config.n_layers)])

    def forward(self, dec_inputs, enc_inputs, memory):
        x = self.pos_emb(self.tgt_emb(dec_inputs).transpose(0, 1)).transpose(0, 1)
        self_mask = pad_mask(dec_inputs, dec_inputs) | subsequent_mask(dec_inputs)
        cross_mask = pad_mask(dec_inputs, enc_inputs)
        self_attns, cross_attns = [], []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, self_mask, cross_mask)
            self_attns.append(self_attn)
            cross_attns.append(cross_attn)
        return x, self_attns, cross_attns


class CFETransformer(nn.Module):
    def __init__(self, config: CFEConfig, dropout: float = 0.2):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, dropout)
        self.pinyin_decoder = Decoder(config, config.pinyin_vocab_size, dropout)
        self.tone_decoder = Decoder(config, config.tone_vocab_size, dropout=0.4)
        self.rhythm_decoder = Decoder(config, config.rhythm_vocab_size, dropout=0.6)
        self.stop_decoder = Decoder(config, config.stop_vocab_size, dropout=0.6)
        self.projection_1 = nn.Linear(config.d_model, config.pinyin_vocab_size, bias=False)
        self.projection_2 = nn.Linear(config.d_model, config.tone_vocab_size, bias=False)
        self.projection_3 = nn.Linear(config.d_model, config.rhythm_vocab_size, bias=False)
        self.projection_4 = nn.Linear(config.d_model, config.stop_vocab_size, bias=False)

    def second_memory(self, enc_outputs, tone_out, rhythm_out):
        """Memory of the pinyin and stop decoders (Eq. 3); decoder states drop their last (EOS) step."""
        if self.config.variant == "parallel":
            return enc_outputs
        total = sum(self.config.lambdas)
        lambda_x, lambda_t, lambda_r = (value / total for value in self.config.lambdas)
        return lambda_x * enc_outputs + lambda_t * tone_out[:, :-1] + lambda_r * rhythm_out[:, :-1]

    def forward(self, enc_inputs, dec_pinyins, dec_tones, dec_rhythms, dec_stops):
        """Teacher-forced logits, each flattened to [batch * tgt_len, vocab]: pinyin, tone, rhyme, stop."""
        enc_outputs, _ = self.encoder(enc_inputs)
        tone_out, _, _ = self.tone_decoder(dec_tones, enc_inputs, enc_outputs)
        rhythm_out, _, _ = self.rhythm_decoder(dec_rhythms, enc_inputs, enc_outputs)
        memory = self.second_memory(enc_outputs, tone_out.detach(), rhythm_out.detach())
        pinyin_out, _, _ = self.pinyin_decoder(dec_pinyins, enc_inputs, memory)
        stop_out, _, _ = self.stop_decoder(dec_stops, enc_inputs, memory)
        logits = (self.projection_1(pinyin_out), self.projection_2(tone_out),
                  self.projection_3(rhythm_out), self.projection_4(stop_out))
        return tuple(value.reshape(-1, value.size(-1)) for value in logits)


def save_checkpoint(path, model: CFETransformer, vocab: CFEVocab, optimizer=None, epoch=None):
    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "config": model.config.to_dict(), "vocab": vocab.to_dict()}, path)


def load_checkpoint(path, *, device="cpu", variant=None, lambdas=None):
    """Load a CFE checkpoint with its embedded vocabulary and return ``(model, vocab)`` in eval mode."""
    payload = torch.load(Path(path), map_location=device)
    if "vocab" not in payload:
        raise ValueError("checkpoint has no embedded vocabulary")
    vocab = CFEVocab.from_dict(payload["vocab"])
    config = CFEConfig(**payload["config"]) if payload.get("config") else CFEConfig.from_vocab(vocab)
    overrides = {key: value for key, value in (("variant", variant), ("lambdas", lambdas)) if value is not None}
    if overrides:
        config = dataclasses.replace(config, **overrides)
    model = CFETransformer(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, vocab
