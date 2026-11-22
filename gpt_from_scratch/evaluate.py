"""Perplexity on held-out text.

    PPL = exp( (1/N) sum_i -log p(x_i | x_<i) ),   bits per token = loss / ln 2

PPL 30 = the model is as unsure as a uniform pick among 30 tokens. Only compare
perplexities computed with the same tokenizer on the same text.

    python -m gpt_from_scratch.evaluate --ckpt runs/tiny/best.pt --corpus tinyshakespeare
"""

import argparse
import math

import torch
import torch.nn.functional as F

from .data import load_corpus, split_text, tokenizer_from_state
from .model import load_checkpoint


def perplexity_from_losses(losses):
    """Example: [1.2, 0.8, 2.0, 1.0] -> mean 1.25 -> PPL e^1.25 = 3.49."""
    mean = sum(losses) / len(losses)
    return math.exp(mean)


@torch.no_grad()
def token_losses(model, ids, device="cpu", stride=None):
    """-log p of every predicted token in `ids`, using non-overlapping windows of ctx_len."""
    model.eval()
    ctx = model.cfg.ctx_len
    stride = stride or ctx
    ids = torch.as_tensor(ids, dtype=torch.long)
    out = []
    for s in range(0, len(ids) - 1, stride):
        chunk = ids[s:s + ctx + 1]
        if len(chunk) < 2:
            break
        x, y = chunk[:-1][None].to(device), chunk[1:][None].to(device)
        logits, _ = model(x)
        out.append(F.cross_entropy(logits[0], y[0], reduction="none").cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="tinyshakespeare", help="corpus name (its val split) or a text file")
    ap.add_argument("--whole-file", action="store_true", help="score the whole text, not just the last 10%")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = tokenizer_from_state(ckpt["tokenizer"])
    text = load_corpus(args.corpus)
    if not args.whole_file:
        _, text = split_text(text, 0.9)
    losses = token_losses(model, tok.encode(text), args.device)
    loss = losses.mean().item()
    print(f"tokens scored : {len(losses):,}")
    print(f"loss          : {loss:.4f} nats/token")
    print(f"perplexity    : {math.exp(loss):.2f}")
    print(f"bits per token: {loss / math.log(2):.3f}")
    print(f"(uniform guess over V={model.cfg.vocab_size:,}: loss {math.log(model.cfg.vocab_size):.3f})")


if __name__ == "__main__":
    main()
