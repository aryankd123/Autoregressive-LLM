"""Multi-head attention: a hand-computed example (d=4, h=2, d_k=2) and the module tests."""

import torch

from gpt_from_scratch.attention import CausalSelfAttention
from test_part3_attention import X, Wq, Wk, Wv

Wq2 = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float)
Wk2 = torch.tensor([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=torch.float)
Wv2 = torch.tensor([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=torch.float)
WO = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float)


def example_module(flash=False):
    """CausalSelfAttention loaded with the example's matrices.
    Full W_Q = [W_Q^(1) | W_Q^(2)] (d x d); nn.Linear stores the transpose (out x in)."""
    attn = CausalSelfAttention(d=4, n_heads=2, ctx_len=3, dropout=0.0, bias=False, flash=flash)
    W_qkv = torch.cat([torch.cat([Wq, Wq2], 1), torch.cat([Wk, Wk2], 1), torch.cat([Wv, Wv2], 1)], 1)
    with torch.no_grad():
        attn.qkv.weight.copy_(W_qkv.T)
        attn.proj.weight.copy_(WO.T)
    return attn


def test_multihead_hand_computed_example():
    Y = example_module()(X[None])[0]
    expected = torch.tensor([[3, 1, 2, 2], [2, 3, 3, 2], [2.436, 2.280, 2.280, 2.436]])
    assert torch.allclose(Y, expected, atol=2e-3)


def test_flash_path_equals_explicit_math():
    torch.manual_seed(0)
    a = CausalSelfAttention(d=16, n_heads=4, ctx_len=10, dropout=0.0)
    b = CausalSelfAttention(d=16, n_heads=4, ctx_len=10, dropout=0.0, flash=True)
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 7, 16)
    assert torch.allclose(a(x), b(x), atol=1e-5)
    assert torch.allclose(example_module(flash=True)(X[None]), example_module()(X[None]), atol=1e-5)


def test_past_cannot_see_the_future():
    torch.manual_seed(0)
    attn = CausalSelfAttention(d=8, n_heads=2, ctx_len=16, dropout=0.0)
    x = torch.randn(1, 5, 8)
    y1 = attn(x)
    x2 = x.clone()
    x2[0, 4] += 1.0                                   # change only the last token
    y2 = attn(x2)
    assert torch.allclose(y1[0, :4], y2[0, :4])       # earlier outputs unchanged
    assert not torch.allclose(y1[0, 4], y2[0, 4])
    assert y1.shape == (1, 5, 8)                      # same shape as the input: fits the residual


def test_parameter_count():
    attn = CausalSelfAttention(d=8, n_heads=2, ctx_len=16)
    assert sum(p.numel() for p in attn.parameters()) == 8 * 24 + 24 + 8 * 8 + 8 == 288
    big = CausalSelfAttention(d=768, n_heads=12, ctx_len=4)
    assert sum(p.numel() for p in big.parameters()) == 4 * 768 * 768 + 4 * 768 == 2_362_368


def test_reshape_splits_columns_into_heads():
    Q_full = torch.cat([X @ Wq, X @ Wq2], 1)          # rows [2,0,0,2], [0,2,2,0], [1,1,1,1]
    assert Q_full.tolist() == [[2, 0, 0, 2], [0, 2, 2, 0], [1, 1, 1, 1]]
    heads = Q_full.view(1, 3, 2, 2).transpose(1, 2)   # (B, h, T, d_k)
    assert heads[0, 0, 0].tolist() == [2, 0] and heads[0, 1, 0].tolist() == [0, 2]
