"""Model hyper-parameters.

Notation:  V vocab, T sequence length, d model width,
h heads, d_k = d / h per-head width, L number of blocks.
"""

from dataclasses import dataclass, asdict, fields


@dataclass
class GPTConfig:
    vocab_size: int = 50257   # V: 256 bytes + 50,000 merges + <|endoftext|>
    ctx_len: int = 1024       # maximum T (size of the position table)
    d: int = 768              # model width
    n_layers: int = 12        # L
    n_heads: int = 12         # h  (d_k = 768 / 12 = 64)
    dropout: float = 0.1
    bias: bool = True         # GPT-2 uses biases in every Linear
    flash: bool = False       # True: use F.scaled_dot_product_attention (fused kernel)

    def __post_init__(self):
        assert self.d % self.n_heads == 0, "d must be divisible by n_heads"

    @property
    def d_k(self):
        return self.d // self.n_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


PRESETS = {
    # GPT-2 small: 124,439,808 parameters.
    "gpt2-124m": dict(vocab_size=50257, ctx_len=1024, d=768, n_layers=12, n_heads=12, dropout=0.1),
    # Same architecture, scaled down so a pretraining run finishes on a laptop CPU.
    "tiny": dict(vocab_size=50257, ctx_len=128, d=128, n_layers=4, n_heads=4, dropout=0.1),
    # A middle size for a free Colab/Kaggle GPU.
    "small": dict(vocab_size=50257, ctx_len=256, d=384, n_layers=6, n_heads=6, dropout=0.1),
}


def get_config(name="gpt2-124m", **overrides):
    cfg = dict(PRESETS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return GPTConfig(**cfg)
