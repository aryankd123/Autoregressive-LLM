"""Token and position embeddings.

    h0[t] = W_E[x_t, :] + W_P[t, :]

nn.Embedding *is* the matrix W_E (V x d); indexing it with integer ids is the same as
one-hot(x_t) @ W_E, without ever building the one-hot vector.
Attention treats its input as a set, so the position vector is what tells the model
"I am token number t".
"""

import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size, ctx_len, d):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)   # W_E: V x d     (GPT-2: 50,257 x 768 = 38,597,376)
        self.pos = nn.Embedding(ctx_len, d)      # W_P: ctx x d   (GPT-2:  1,024 x 768 =    786,432)
        self.ctx_len = ctx_len

    def forward(self, idx, start_pos=0):
        """idx: (B, T) int64 -> (B, T, d).

        start_pos is 0 in training. With a KV-cache the new tokens sit at absolute
        positions start_pos .. start_pos+T-1, so they must get those position vectors.
        """
        B, T = idx.shape
        assert start_pos + T <= self.ctx_len, (
            f"sequence of length {start_pos + T} exceeds the context length {self.ctx_len}")
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)   # (T,)
        return self.tok(idx) + self.pos(pos)     # (B,T,d) + (T,d) broadcasts over the batch
