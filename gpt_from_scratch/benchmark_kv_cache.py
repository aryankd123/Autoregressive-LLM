"""Measure what the KV-cache buys: identical greedy tokens, fewer FLOPs, more tokens/s.

    python -m gpt_from_scratch.benchmark_kv_cache --preset gpt2-124m --new-tokens 200
    python -m gpt_from_scratch.benchmark_kv_cache --ckpt runs/tiny/best.pt
"""

import argparse
import time

import torch

from .config import get_config, PRESETS
from .generate import generate, generate_with_cache
from .model import GPT, kv_cache_bytes, load_checkpoint


def timed(fn, *args, device="cpu", **kw):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*args, **kw)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--preset", default="gpt2-124m", choices=list(PRESETS))
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.ckpt:
        model, _ = load_checkpoint(args.ckpt, args.device)
    else:
        model = GPT(get_config(args.preset, dropout=0.0)).to(args.device)   # random weights are fine for timing
    model.eval()
    cfg = model.cfg
    n_new = min(args.new_tokens, cfg.ctx_len - args.prompt_len)
    idx = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=args.device)

    generate_with_cache(model, idx, 2, temperature=0)            # warm-up
    a, t_plain = timed(generate, model, idx, n_new, temperature=0, device=args.device)
    b, t_cache = timed(generate_with_cache, model, idx, n_new, temperature=0, device=args.device)

    print(f"model: {model.num_params():,} params, L={cfg.n_layers}, d={cfg.d}, ctx={cfg.ctx_len}")
    print(f"prompt {args.prompt_len} tokens, generating {n_new} greedy tokens on {args.device}")
    print(f"identical output   : {torch.equal(a, b)}")
    print(f"without cache      : {t_plain:6.2f} s  ({n_new / t_plain:7.1f} tok/s)")
    print(f"with KV-cache      : {t_cache:6.2f} s  ({n_new / t_cache:7.1f} tok/s)")
    print(f"speed-up           : {t_plain / t_cache:.2f}x")

    T = args.prompt_len + n_new
    _, past = model.forward_with_cache(b[:, :T - 1])
    measured = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in past)
    print(f"cache at T={T - 1}: measured {measured:,} bytes (fp32) "
          f"= formula 2*L*T*d*4 = {kv_cache_bytes(cfg, T - 1, 4):,}")
    print(f"GPT-2 small, T=1024, fp16: {kv_cache_bytes(get_config('gpt2-124m'), 1024, 2):,} bytes (~38 MB)")


if __name__ == "__main__":
    main()
