"""CFE training loop (SGD)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from intoner.cfe.data import make_loader
from intoner.cfe.model import CFEConfig, CFETransformer, save_checkpoint
from intoner.cfe.vocab import MAX_TEXT_LEN, PUNCTUATION, CFEVocab


def _losses(model, batch, device, pinyin_loss, label_loss):
    words, p_in, p_out, t_in, t_out, r_in, r_out, s_in, s_out = (tensor.to(device) for tensor in batch)
    pinyin, tone, rhythm, stop = model(words, p_in, t_in, r_in, s_in)
    return (pinyin_loss(pinyin, p_out.view(-1)), label_loss(tone, t_out.view(-1)),
            label_loss(rhythm, r_out.view(-1)), label_loss(stop, s_out.view(-1)))


def _format(losses):
    names = ("pinyin", "tone", "rhythm", "stop")
    return f"loss={sum(losses):.6f} " + " ".join(f"{name}={value:.6f}" for name, value in zip(names, losses))


def train(*, manifest, train_csv, test_csv, output_dir, variant="cascade", lambdas=(3.0, 2.0, 1.0),
          punc_weight=8.0, epochs=1500, batch_size=16, lr=1e-3, momentum=0.99,
          eval_every=100, save_every=200, device="cpu", max_len=MAX_TEXT_LEN):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = CFEVocab.from_csv(manifest)
    model = CFETransformer(CFEConfig.from_vocab(vocab, variant=variant, lambdas=tuple(lambdas))).to(device)
    weight = torch.ones(len(vocab.pinyin), device=device)
    weight[[vocab.pinyin[mark] for mark in PUNCTUATION]] *= punc_weight  # λpunc: punctuation keeps alignment
    pinyin_loss = nn.CrossEntropyLoss(ignore_index=0, weight=weight)
    label_loss = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loader = make_loader(train_csv, vocab, batch_size, max_len)
    test_loader = make_loader(test_csv, vocab, batch_size, max_len)
    for epoch in range(1, epochs + 1):
        model.train()
        totals, batches = [0.0] * 4, 0
        for batch in train_loader:
            losses = _losses(model, batch, device, pinyin_loss, label_loss)
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
            totals = [total + value.item() for total, value in zip(totals, losses)]
            batches += 1
        print(f"epoch {epoch:04d} train {_format([total / batches for total in totals])}", flush=True)
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                totals, batches = [0.0] * 4, 0
                for batch in test_loader:
                    losses = _losses(model, batch, device, pinyin_loss, label_loss)
                    totals = [total + value.item() for total, value in zip(totals, losses)]
                    batches += 1
            line = f"epoch {epoch:04d} test {_format([total / batches for total in totals])}"
            print(line, flush=True)
            with (output_dir / "val_log.txt").open("a", encoding="utf-8") as log:
                log.write(line + "\n")
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(output_dir / f"model_{epoch}.pt", model, vocab, optimizer, epoch)
