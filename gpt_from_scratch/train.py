"""The pretraining loop.

One optimisation step, in order:
    1. set the learning rate from the warmup + cosine schedule
    2. forward -> loss (mean cross-entropy over B*T positions)
    3. zero the old gradients
    4. backward -> fills .grad   (with grad accumulation: steps 2-4 repeat on micro-batches)
    5. clip the global gradient norm to `clip`
    6. optimizer.step()  (AdamW: Adam update + decoupled weight decay)

Run (from the repo root):
    python -m gpt_from_scratch.train --corpus tinyshakespeare --preset tiny --iters 2000
"""

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import sys
import time

import torch

from .config import get_config, PRESETS
from .data import (build_tokenizer, load_corpus, make_loader, split_text, tokenizer_state)
from .generate import generate_with_cache
from .model import GPT

# ------------------------------------------------------------------ schedule


def get_lr(it, max_lr, min_lr, warmup, total):
    """Linear warmup for `warmup` steps, then half a cosine from max_lr down to min_lr.
    (it + 1) so the very first step does not use lr = 0."""
    if it < warmup:
        return max_lr * (it + 1) / warmup
    if it >= total:
        return min_lr
    p = (it - warmup) / (total - warmup)                 # 0 -> 1 over the decay phase
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


# ----------------------------------------------------------------- optimizer


def make_optimizer(model, lr, wd, betas=(0.9, 0.95)):
    """Weight decay only on matrices and embeddings (dim >= 2). Biases and LayerNorm
    gains/shifts (dim < 2) are not decayed: shrinking them does not regularise anything.
    model.parameters() yields the tied embedding/head tensor once."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd},
              {"params": nodecay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def manual_adamw_step(theta, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    """One AdamW step written out; the tests compare it with torch.optim.AdamW.
        m = b1 m + (1-b1) g            running mean of the gradient (momentum)
        v = b2 v + (1-b2) g^2          running mean of its square (typical size)
        m_hat = m / (1-b1^t), v_hat = v / (1-b2^t)   undo the bias of starting at 0
        theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)
    Returns (theta, m, v)."""
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    theta = theta - lr * (m_hat / (math.sqrt(v_hat) + eps) + wd * theta)
    return theta, m, v


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    """Mean cross-entropy over up to max_batches batches, and its perplexity exp(loss)."""
    was_training = model.training
    model.eval()
    total, n_tok = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        _, loss = model(x.to(device), y.to(device))
        total += loss.item() * y.numel()       # token-weighted, so a short last batch is fair
        n_tok += y.numel()
    model.train(was_training)
    if n_tok == 0:
        raise ValueError("validation loader is empty: use a smaller ctx_len or more text")
    mean = total / n_tok
    return mean, math.exp(mean)


# ------------------------------------------------------------------- training


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _autocast(device, enabled):
    if enabled and device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def train(model, train_loader, val_loader, device, total_iters=2000, max_lr=6e-4,
          min_lr=6e-5, warmup=100, wd=0.1, clip=1.0, grad_accum=1, log_every=50,
          eval_every=250, eval_batches=50, amp=False, out_dir=None, on_eval=None,
          extra_ckpt=None):
    """Returns a list of log rows (dicts). Saves best.pt / last.pt in out_dir if given."""
    opt = make_optimizer(model, max_lr, wd)
    model.to(device).train()
    batches = _infinite(train_loader)
    history, tokens_seen, best_val = [], 0, float("inf")
    t0 = time.time()

    def save(name, it, val_loss):
        if out_dir is None:
            return
        ckpt = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "config": model.cfg.to_dict(), "iter": it, "val_loss": val_loss,
                "tokens_seen": tokens_seen}
        ckpt.update(extra_ckpt or {})
        torch.save(ckpt, os.path.join(out_dir, name))

    for it in range(total_iters):
        lr = get_lr(it, max_lr, min_lr, warmup, total_iters)               # 1
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)                                    # 3
        loss_accum = 0.0
        for _ in range(grad_accum):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            with _autocast(device, amp):
                _, loss = model(x, y)                                      # 2
            (loss / grad_accum).backward()                                 # 4 (sum of micro-grads)
            loss_accum += loss.item() / grad_accum
            tokens_seen += y.numel()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # 5
        opt.step()                                                         # 6

        row = {"iter": it, "lr": lr, "train_loss": loss_accum, "grad_norm": float(gnorm),
               "tokens_seen": tokens_seen, "elapsed_s": time.time() - t0}
        last = it == total_iters - 1
        if it % log_every == 0 or last:
            tps = tokens_seen / max(row["elapsed_s"], 1e-9)
            print(f"it {it:5d} | loss {loss_accum:.3f} | lr {lr:.2e} | gnorm {float(gnorm):.2f}"
                  f" | {tps:,.0f} tok/s")
        if it % eval_every == 0 or last:
            val_loss, val_ppl = evaluate(model, val_loader, device, eval_batches)
            row.update(val_loss=val_loss, val_ppl=val_ppl)
            print(f"           val loss {val_loss:.3f} | val ppl {val_ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                save("best.pt", it, val_loss)
            if on_eval:
                on_eval(model, it)
        history.append(row)
    save("last.pt", total_iters - 1, history[-1].get("val_loss"))
    return history


# -------------------------------------------------------------------- helpers


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_description(device):
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple GPU (mps)"
    return f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count()} threads)"


def write_csv(history, path):
    keys = sorted({k for row in history for k in row}, key=lambda k: (k != "iter", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)


def plot_losses(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the loss plot")
        return
    its = [r["iter"] for r in history]
    ev = [r for r in history if "val_loss" in r]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(its, [r["train_loss"] for r in history], label="train (per step)", alpha=0.6)
    ax.plot([r["iter"] for r in ev], [r["val_loss"] for r in ev], "o-", label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pretrain a GPT from scratch")
    ap.add_argument("--corpus", default="tinyshakespeare", help="tinyshakespeare | verdict | path/to.txt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--tokenizer", default="gpt2", choices=["gpt2", "bpe"],
                    help="gpt2 = tiktoken (V=50,257); bpe = our own byte-level BPE trained on the corpus")
    ap.add_argument("--bpe-vocab", type=int, default=4096)
    ap.add_argument("--preset", default="tiny", choices=list(PRESETS))
    ap.add_argument("--ctx-len", type=int)
    ap.add_argument("--d", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--n-heads", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--flash", action="store_true", help="use F.scaled_dot_product_attention")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--out", default="runs/tiny")
    return ap.parse_args(argv)


def main(argv=None):
    # Windows: when output is redirected to a file or pipe, Python encodes it as cp1252, which
    # cannot represent e.g. U+FFFD (what an incomplete UTF-8 byte sequence in a sample decodes to).
    # A print statement must never kill a long run, so such characters are written as escapes.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # data: split by characters first so the tokenizer never sees validation text
    text = load_corpus(args.corpus, args.data_dir)
    train_text, val_text = split_text(text, 0.9)
    tok = build_tokenizer(args.tokenizer, train_text, args.bpe_vocab)
    train_ids, val_ids = tok.encode(train_text), tok.encode(val_text)

    cfg = get_config(args.preset, vocab_size=tok.n_vocab, ctx_len=args.ctx_len, d=args.d,
                     n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout)
    cfg.flash = args.flash
    train_loader = make_loader(train_ids, cfg.ctx_len, args.batch_size, shuffle=True, drop_last=True)
    val_loader = make_loader(val_ids, cfg.ctx_len, args.batch_size, shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError("not enough training text for one batch: lower --batch-size or --ctx-len")

    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"device: {device_description(device)}")
    print(f"corpus: {args.corpus} | tokenizer: {args.tokenizer} (V={tok.n_vocab:,}) | "
          f"train tokens {len(train_ids):,} | val tokens {len(val_ids):,}")
    print(f"model: {n_params:,} parameters | {cfg}")

    init_val, _ = evaluate(model, val_loader, device, args.eval_batches)
    print(f"initial val loss {init_val:.3f}  (ln V = {math.log(cfg.vocab_size):.3f})")

    prompt_ids = tok.encode(args.prompt) or [tok.eot_token]

    def on_eval(m, it):
        idx = torch.tensor([prompt_ids], device=device)
        out = generate_with_cache(m, idx, 60, temperature=0.8, top_k=50)
        print("           sample:", repr(tok.decode(out[0].tolist())[:200]))

    t0 = time.time()
    history = train(model, train_loader, val_loader, device, total_iters=args.iters,
                    max_lr=args.lr, min_lr=args.min_lr, warmup=args.warmup, wd=args.wd,
                    clip=args.clip, grad_accum=args.grad_accum, eval_every=args.eval_every,
                    eval_batches=args.eval_batches, amp=args.amp, out_dir=args.out,
                    on_eval=on_eval, extra_ckpt={"tokenizer": tokenizer_state(tok)})
    wall = time.time() - t0

    # final numbers on the WHOLE validation split, from the best checkpoint
    best = torch.load(os.path.join(args.out, "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    val_loss, val_ppl = evaluate(model, val_loader, device, max_batches=None)
    torch.manual_seed(args.seed)
    sample = tok.decode(generate_with_cache(
        model, torch.tensor([prompt_ids], device=device), 200, temperature=0.8, top_k=50)[0].tolist())

    summary = {
        "corpus": args.corpus, "tokenizer": args.tokenizer, "vocab_size": tok.n_vocab,
        "train_tokens": len(train_ids), "val_tokens": len(val_ids),
        "parameters": n_params, "config": cfg.to_dict(),
        "iters": args.iters, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "tokens_seen": history[-1]["tokens_seen"],
        "epochs_over_train_set": history[-1]["tokens_seen"] / max(len(train_ids), 1),
        "hardware": device_description(device), "wall_clock_minutes": wall / 60,
        "initial_val_loss": init_val, "best_iter": best["iter"],
        "final_val_loss": val_loss, "final_val_ppl": val_ppl,
        "final_val_bits_per_token": val_loss / math.log(2),
        "sample_prompt": args.prompt, "sample": sample,
        "hyperparams": {k: getattr(args, k) for k in ("lr", "min_lr", "warmup", "wd", "clip")},
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(history, os.path.join(args.out, "log.csv"))
    plot_losses(history, os.path.join(args.out, "loss_curve.png"))
    print(f"\nfinal val loss {val_loss:.3f} | val ppl {val_ppl:.2f} | {wall / 60:.1f} min")
    print(f"sample:\n{sample}")
    print(f"\nwrote {args.out}/summary.json, log.csv, loss_curve.png, best.pt, last.pt")


if __name__ == "__main__":
    main()
