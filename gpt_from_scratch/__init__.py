"""GPT from scratch: a 124M-parameter GPT-2-style model in plain PyTorch.

Layout, in the order the data flows:

    bpe.py          toy BPE (word level, </w>) + byte-level BPE trainer/encoder
    data.py         corpora, tokenizers, sliding-window next-token dataset
    embeddings.py   token + position embeddings
    attention.py    single-head attention (+ hand-written backward),
                    CausalSelfAttention (multi-head, KV-cache aware)
    model.py        LayerNorm, GELU, FeedForward, Pre-LN Block, GPT
    train.py        AdamW groups, warmup+cosine LR, clipping, grad accumulation
    generate.py     temperature / top-k / top-p, generate with and without KV-cache
    evaluate.py     perplexity and bits per token on held-out text
"""

from .config import GPTConfig, get_config, PRESETS

__all__ = ["GPTConfig", "get_config", "PRESETS"]
# The model itself:  from gpt_from_scratch.model import GPT
