"""Causal self-attention.

    Q = X W_Q,  K = X W_K,  V = X W_V                (T x d_k each)
    S = Q K^T / sqrt(d_k)                            (T x T)
    A = softmax(S + M),  M_ij = -inf if j > i else 0 (causal mask)
    O = A V                                          (T x d_k)

Multi-head: split d into h slices of d_k = d/h, attend in each slice in parallel,
concatenate, and mix with W_O. One (d x 3d) matmul produces Q, K, V for all heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T, device=None):
    """(T, T) bool, True strictly above the diagonal = the future = forbidden."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


# --------------------------------------------------------------------------
# One head, written as plain matrix algebra (the tests check it on a
# hand-computed T=3, d=4, d_k=2 example, and check its backward pass).
# --------------------------------------------------------------------------

def single_head_attention(X, Wq, Wk, Wv, causal=True):
    """X: (T, d), W*: (d, d_k). Returns O (T, d_k) and the attention weights A (T, T)."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.size(-1)
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if causal:
        S = S.masked_fill(causal_mask(S.size(-1), S.device), float("-inf"))
    A = torch.softmax(S, dim=-1)          # every row sums to 1
    return A @ V, A


def softmax_backward(A, dA):
    """Row-wise softmax Jacobian applied to an upstream gradient:
        dS = A * (dA - rowsum(dA * A))
    Masked entries have A = 0, so their gradient is automatically 0."""
    return A * (dA - (dA * A).sum(dim=-1, keepdim=True))


def single_head_attention_backward(X, Wq, Wk, Wv, dO, causal=True):
    """Hand-derived backward pass. Returns the gradients of
    sum(O * dO) w.r.t. X, Wq, Wk, Wv; the tests check them against autograd."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    d_k = Q.size(-1)
    _, A = single_head_attention(X, Wq, Wk, Wv, causal)
    dV = A.T @ dO                       # O = A V
    dA = dO @ V.T
    dS = softmax_backward(A, dA)        # through the softmax
    dQ = dS @ K / math.sqrt(d_k)        # S = Q K^T / sqrt(d_k)
    dK = dS.T @ Q / math.sqrt(d_k)
    return {
        "dWq": X.T @ dQ, "dWk": X.T @ dK, "dWv": X.T @ dV,
        "dX": dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T,   # X feeds all three projections
    }


# --------------------------------------------------------------------------
# Multi-head causal self-attention, KV-cache aware (see generate.py)
# --------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, ctx_len, dropout=0.1, bias=True, flash=False):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dk = n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=bias)    # W_Q, W_K, W_V side by side: d x 3d
        self.proj = nn.Linear(d, d, bias=bias)       # W_O: mixes the heads
        self.dropout = dropout
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.flash = flash
        # True = forbidden. Not saved in checkpoints (persistent=False): it is rebuilt here.
        self.register_buffer("mask", causal_mask(ctx_len), persistent=False)

    def _split_heads(self, t, B, T):
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k): each head gets its own (T, d_k) slab
        return t.view(B, T, self.h, self.dk).transpose(1, 2)

    def _attend(self, q, k, v):
        """q: (B, h, T, d_k), k/v: (B, h, Tk, d_k) with Tk >= T (Tk > T when a cache is used).
        The T queries sit at absolute positions Tk-T .. Tk-1."""
        T, Tk = q.size(2), k.size(2)
        assert Tk <= self.mask.size(0), "sequence longer than the context length"
        mask = self.mask[Tk - T:Tk, :Tk]                          # (T, Tk)
        if self.flash:
            # Same math in one fused kernel (FlashAttention on GPU). attn_mask True = keep.
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=~mask, dropout_p=self.dropout if self.training else 0.0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)     # (B, h, T, Tk)
        scores = scores.masked_fill(mask, float("-inf"))          # block the future
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        return attn @ v                                           # (B, h, T, d_k)

    def _merge_heads(self, out, B, T, d):
        # transpose leaves a non-contiguous tensor; view needs contiguous memory
        return out.transpose(1, 2).contiguous().view(B, T, d)     # concat heads -> (B, T, d)

    def forward(self, x):
        """Training / no-cache path. x: (B, T, d) -> (B, T, d)."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)                     # each (B, T, d)
        q, k, v = (self._split_heads(t, B, T) for t in (q, k, v))
        out = self._attend(q, k, v)
        return self.resid_drop(self.proj(self._merge_heads(out, B, T, d)))

    def forward_with_cache(self, x, past_kv=None):
        """Inference path. x holds only the NEW tokens (the prompt on the first call,
        one token afterwards). past_kv = (K, V) of all earlier tokens, each (B, h, T_past, d_k).
        Returns the output for the new tokens and the updated (K, V)."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (self._split_heads(t, B, T) for t in (q, k, v))
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)                 # (B, h, T_past + T, d_k)
            v = torch.cat([past_kv[1], v], dim=2)
        out = self._attend(q, k, v)
        return self.resid_drop(self.proj(self._merge_heads(out, B, T, d))), (k, v)
