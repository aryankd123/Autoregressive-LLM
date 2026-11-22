"""Sampling and autoregressive generation, with and without a KV-cache.

    x_{t+1} ~ softmax(z_t / tau)       (only the LAST position's logits matter)

temperature tau < 1 sharpens, > 1 flattens, tau = 0 means greedy argmax.
top-k keeps the k most likely tokens; top-p keeps the smallest set whose mass >= p.

Why the KV-cache is exact: with a causal mask, token i's keys and values depend only on
tokens <= i, so appending a token never changes them. Each new step then needs only the
new token's q, k, v plus the stored K, V of the past.
"""

import argparse
import sys

import torch

from .model import kv_cache_bytes  # noqa: F401  (re-exported for convenience)


def top_k_filter(logits, k):
    """Everything below the k-th largest logit -> -inf (probability 0 after softmax)."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus: keep the smallest set of tokens whose probabilities add up to >= p."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs          # mass of strictly better tokens
    sorted_logits = sorted_logits.masked_fill(mass_before >= p, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)  # undo the sort


def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (B, V) for the last position. Returns (B, V) probabilities."""
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.softmax(logits, dim=-1)


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:                                  # greedy
        return logits.argmax(dim=-1, keepdim=True)
    probs = next_token_probs(logits, temperature, top_k, top_p)
    return torch.multinomial(probs, num_samples=1, generator=generator)   # (B, 1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
             eos_id=None, generator=None):
    """No cache: every step re-runs the whole (cropped) prefix. O(t) work per token per layer
    for the projections, O(t^2) for attention."""
    was_training = model.training
    model.eval()
    ctx = model.cfg.ctx_len
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx:]                          # position table stops at ctx_len
        logits, _ = model(idx_cond)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
    model.train(was_training)
    return idx


@torch.no_grad()
def generate_with_cache(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                        eos_id=None, generator=None):
    """KV-cache: run the prompt once ("prefill"), then feed ONE token per step ("decode").
    Gives exactly the same tokens as generate(). Once the sequence fills the context window
    the cache is rebuilt from the last ctx_len tokens, which is what generate() does anyway
    (learned absolute positions cannot slide)."""
    was_training = model.training
    model.eval()
    ctx = model.cfg.ctx_len
    logits, past = model.forward_with_cache(idx[:, -ctx:])        # prefill
    for step in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
        if (eos_id is not None and bool((nxt == eos_id).all())) or step == max_new_tokens - 1:
            break
        if past[0][0].size(2) + 1 <= ctx:
            logits, past = model.forward_with_cache(nxt, past)    # decode: T = 1
        else:
            logits, past = model.forward_with_cache(idx[:, -ctx:])  # window full: re-prefill
    model.train(was_training)
    return idx


def main():
    from .data import tokenizer_from_state
    from .model import load_checkpoint

    if hasattr(sys.stdout, "reconfigure"):      # same Windows cp1252 guard as train.main
        sys.stdout.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.ckpt, args.device)
    tok = tokenizer_from_state(ckpt["tokenizer"])
    idx = torch.tensor([tok.encode(args.prompt)], device=args.device)
    g = torch.Generator(device=args.device).manual_seed(args.seed)
    fn = generate if args.no_cache else generate_with_cache
    out = fn(model, idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p,
             generator=g)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
