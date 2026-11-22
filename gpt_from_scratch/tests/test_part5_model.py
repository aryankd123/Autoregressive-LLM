"""Model: LayerNorm, GELU, the block, the full GPT and its 124,439,808 parameters."""

import math

import torch
import torch.nn.functional as F

from gpt_from_scratch.config import GPTConfig, get_config
from gpt_from_scratch.model import GELU, GPT, LayerNorm, param_count_table


def test_layernorm_hand_computed_example():
    ln = LayerNorm(4, eps=0.0)
    y = ln(torch.tensor([2.0, 4.0, 6.0, 8.0]))
    assert torch.allclose(y, torch.tensor([-1.342, -0.447, 0.447, 1.342]), atol=1e-3)
    assert abs(y.mean().item()) < 1e-6 and abs(y.var(unbiased=False).item() - 1) < 1e-5


def test_layernorm_backward_numeric_check():
    ln = LayerNorm(4, eps=0.0)
    x = torch.tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    ln(x).backward(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(x.grad, torch.tensor([0.134, -0.179, -0.045, 0.089]), atol=1e-3)
    assert abs(x.grad.sum().item()) < 1e-6


def test_layernorm_matches_pytorch():
    x = torch.randn(2, 3, 16)
    assert torch.allclose(LayerNorm(16)(x), F.layer_norm(x, (16,)), atol=1e-5)


def test_gelu_values():
    g = GELU()(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
    assert torch.allclose(g, torch.tensor([-0.159, 0.0, 0.841, 1.954]), atol=1e-3)
    x = torch.linspace(-4, 4, 101)
    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_parameter_table_matches_hand_count():
    t = param_count_table(GPTConfig())
    assert t["per block: attention qkv"] == 1_771_776
    assert t["per block: attention output proj"] == 590_592
    assert t["per block: FFN up"] == 2_362_368
    assert t["per block: FFN down"] == 2_360_064
    assert t["one block total"] == 7_087_872
    assert t["12 blocks"] == 85_054_464
    assert t["TOTAL (tied)"] == 124_439_808
    assert t["TOTAL without weight tying"] == 163_037_184
    ffn = t["per block: FFN up"] + t["per block: FFN down"]
    assert abs(ffn / t["one block total"] - 2 / 3) < 0.01     # the FFN holds two thirds


def test_full_gpt2_model_has_124_439_808_parameters():
    model = GPT(GPTConfig())
    assert model.num_params() == 124_439_808
    assert model.head.weight is model.emb.tok.weight          # weight tying


def tiny_cfg(**kw):
    base = dict(vocab_size=97, ctx_len=16, d=32, n_layers=2, n_heads=4, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_shapes_and_initial_loss_near_ln_V():
    torch.manual_seed(0)
    cfg = get_config("tiny", dropout=0.0)                     # V = 50,257
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    targets = torch.randint(0, cfg.vocab_size, (2, 32))   # NOT idx: with tied weights a token
    logits, loss = model(idx, targets)                      # already "predicts itself" a little
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3   # ~10.82: an almost uniform start


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    idx = torch.randint(0, 97, (1, 10))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 97
    a, _ = model(idx)
    b, _ = model(idx2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6)


def test_residual_gives_identity_jacobian_path():
    """y = x + F(x): with F's output projections zeroed, the block is exactly the identity."""
    from gpt_from_scratch.model import Block
    blk = Block(16, 4, 8, dropout=0.0)
    with torch.no_grad():
        for lin in (blk.attn.proj, blk.ffn.net[2]):
            lin.weight.zero_(); lin.bias.zero_()
    x = torch.randn(1, 5, 16)
    assert torch.allclose(blk(x), x)
