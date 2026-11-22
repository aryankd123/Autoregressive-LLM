"""Embeddings: lookup = one-hot x matrix, plus position rows."""

import torch

from gpt_from_scratch.embeddings import Embeddings

WE = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
WP = torch.tensor([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])


def make():
    emb = Embeddings(vocab_size=4, ctx_len=3, d=3)
    with torch.no_grad():
        emb.tok.weight.copy_(WE)
        emb.pos.weight.copy_(WP)
    return emb


def test_hand_computed_example():
    out = make()(torch.tensor([[2, 0, 3]]))[0]
    expected = torch.tensor([[0.71, 0.82, 0.93], [0.14, 0.25, 0.36], [1.07, 1.18, 1.29]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_lookup_equals_one_hot_matmul():
    idx = torch.tensor([2, 0, 3])
    one_hot = torch.nn.functional.one_hot(idx, 4).float()
    assert torch.allclose(one_hot @ WE, make().tok(idx))


def test_gradient_scatter_adds_into_used_rows():
    emb = make()
    out = emb(torch.tensor([[2, 0, 2]]))         # token 2 appears twice, tokens 1 and 3 absent
    out.sum().backward()
    g = emb.tok.weight.grad
    assert torch.equal(g[2], torch.full((3,), 2.0))   # summed over both occurrences
    assert torch.equal(g[0], torch.ones(3))
    assert torch.equal(g[1], torch.zeros(3)) and torch.equal(g[3], torch.zeros(3))


def test_start_pos_offsets_positions():
    emb = make()
    a = emb(torch.tensor([[2, 0, 3]]))[:, 2]
    b = emb(torch.tensor([[3]]), start_pos=2)[:, 0]
    assert torch.allclose(a, b)


def test_gpt2_parameter_counts():
    assert 50_257 * 768 == 38_597_376
    assert 1_024 * 768 == 786_432
