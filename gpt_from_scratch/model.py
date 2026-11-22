"""LayerNorm, GELU, feed-forward, the Pre-LN block and the full GPT.

    h0     = W_E[x] + W_P
    h_l    = Block_l(h_{l-1})            l = 1..L
    logits = LN_f(h_L) @ W_E^T           (weight tying: the output head reuses W_E)

    Block:  x' = x  + MHA(LN1(x))
            x''= x' + FFN(LN2(x'))
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import GPTConfig
from .embeddings import Embeddings


class LayerNorm(nn.Module):
    """Normalise each token vector over its own d features (never across the batch)."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))    # gamma: learned rescale
        self.b = nn.Parameter(torch.zeros(d))   # beta:  learned shift
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)   # divide by d, not d-1
        return self.g * (x - mu) / torch.sqrt(var + self.eps) + self.b


class GELU(nn.Module):
    """GELU(x) = x * Phi(x), tanh approximation (the one GPT-2 was trained with)."""

    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    """Per-token MLP, d -> 4d -> d. Holds two thirds of each block's parameters."""

    def __init__(self, d, dropout, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d, bias=bias),   # W1: "up"
            GELU(),
            nn.Linear(4 * d, d, bias=bias),   # W2: "down"
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN transformer block: normalise inside each branch so the residual
    (identity) path from input to output stays completely clean."""

    def __init__(self, d, n_heads, ctx_len, dropout, bias=True, flash=False):
        super().__init__()
        self.ln1 = LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads, ctx_len, dropout, bias, flash)
        self.ln2 = LayerNorm(d)
        self.ffn = FeedForward(d, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # residual around attention  (tokens talk)
        x = x + self.ffn(self.ln2(x))      # residual around the FFN    (each token thinks)
        return x

    def forward_with_cache(self, x, past_kv=None):
        a, present = self.attn.forward_with_cache(self.ln1(x), past_kv)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig = None):
        super().__init__()
        cfg = cfg or GPTConfig()
        self.cfg = cfg
        self.emb = Embeddings(cfg.vocab_size, cfg.ctx_len, cfg.d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.d, cfg.n_heads, cfg.ctx_len, cfg.dropout, cfg.bias, cfg.flash)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.tok.weight     # weight tying: one tensor, counted once
        self.apply(self._init_weights)
        # GPT-2 trick: scale the init of the projections that write INTO the residual stream
        # by 1/sqrt(2L) (2 residual additions per block), so the stream's variance does not
        # grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ffn.net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- forward
    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns logits (B, T, V) and, if targets are given,
        the mean cross-entropy over all B*T positions."""
        x = self.drop(self.emb(idx))                  # (B, T, d)
        for blk in self.blocks:
            x = blk(x)                                # (B, T, d)
        logits = self.head(self.ln_f(x))              # (B, T, V)
        loss = None
        if targets is not None:
            # every position is a V-way classification: flatten (B, T, V) -> (B*T, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def forward_with_cache(self, idx, past_kvs=None):
        """Inference with a KV-cache. idx holds only the new tokens.
        past_kvs: list of L (K, V) pairs from the previous call, or None.
        Returns logits for the new tokens and the updated list of L caches."""
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        x = self.drop(self.emb(idx, start_pos=past_len))    # absolute positions!
        presents = []
        for i, blk in enumerate(self.blocks):
            x, present = blk.forward_with_cache(x, None if past_kvs is None else past_kvs[i])
            presents.append(present)
        return self.head(self.ln_f(x)), presents

    # -------------------------------------------------------------- counting
    def num_params(self, non_embedding=False):
        """parameters() yields a shared tensor once, so the tied head is not double counted."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.tok.weight.numel() + self.emb.pos.weight.numel()
        return n


def param_count_table(cfg: GPTConfig):
    """The parameter table, computed from the formulas (no model needed)."""
    d, V, C, L = cfg.d, cfg.vocab_size, cfg.ctx_len, cfg.n_layers
    b = 1 if cfg.bias else 0
    rows = {
        "token embedding (tied with head)": V * d,
        "position embedding": C * d,
        "per block: LayerNorm 1": 2 * d,
        "per block: attention qkv": d * 3 * d + b * 3 * d,
        "per block: attention output proj": d * d + b * d,
        "per block: LayerNorm 2": 2 * d,
        "per block: FFN up": d * 4 * d + b * 4 * d,
        "per block: FFN down": 4 * d * d + b * d,
    }
    block = sum(v for k, v in rows.items() if k.startswith("per block"))
    rows["one block total"] = block
    rows[f"{L} blocks"] = L * block
    rows["final LayerNorm"] = 2 * d
    rows["TOTAL (tied)"] = V * d + C * d + L * block + 2 * d
    rows["TOTAL without weight tying"] = rows["TOTAL (tied)"] + V * d
    return rows


def kv_cache_bytes(cfg: GPTConfig, T, bytes_per_value=2, batch=1):
    """2 (K and V) x L x T x d x bytes, per sequence. GPT-2 small, T=1024, fp16: 37,748,736."""
    return 2 * cfg.n_layers * T * cfg.d * bytes_per_value * batch


def load_checkpoint(path, device="cpu"):
    """Rebuild a GPT from a checkpoint written by train.py."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


if __name__ == "__main__":
    cfg = GPTConfig()
    for k, v in param_count_table(cfg).items():
        print(f"{k:38s} {v:>14,}")
    print(f"{'GPT(GPTConfig()).num_params()':38s} {GPT(cfg).num_params():>14,}")
