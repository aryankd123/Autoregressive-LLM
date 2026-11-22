"""Attention, one causal head: a hand-computed T=3, d=4, d_k=2 example, and the backward pass."""

import math

import torch

from gpt_from_scratch.attention import (single_head_attention, single_head_attention_backward,
                                        softmax_backward)

X = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.float)
Wq = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.float)
Wk = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)
Wv = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 2]], dtype=torch.float)


def test_projections_and_scores():
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    assert Q.tolist() == [[2, 0], [0, 2], [1, 1]]
    assert K.tolist() == [[2, 1], [0, 1], [1, 1]]
    assert V.tolist() == [[2, 1], [0, 3], [1, 1]]
    assert (Q @ K.T).tolist() == [[4, 0, 2], [2, 2, 2], [3, 1, 2]]


def test_attention_weights_and_output_match_hand_computed():
    O, A = single_head_attention(X, Wq, Wk, Wv)
    assert torch.allclose(A, torch.tensor([[1, 0, 0], [0.5, 0.5, 0], [0.576, 0.140, 0.284]]), atol=1e-3)
    assert torch.allclose(O, torch.tensor([[2, 1], [1, 2], [1.436, 1.280]]), atol=1e-3)
    assert torch.allclose(A.sum(-1), torch.ones(3))        # rows are probability distributions
    assert torch.all(A.triu(diagonal=1) == 0)               # no weight on the future


def test_softmax_backward_numeric_check():
    a = torch.tensor([0.576, 0.140, 0.284])
    da = torch.tensor([0.2, -0.1, 0.4])
    ds = softmax_backward(a, da)
    assert torch.allclose(ds, torch.tensor([-0.0085, -0.0441, 0.0526]), atol=1e-4)
    assert abs(ds.sum().item()) < 1e-6    # shifting all scores does not change softmax


def test_hand_backward_equals_autograd():
    torch.manual_seed(0)
    T, d, dk = 5, 8, 4
    Xr = torch.randn(T, d, dtype=torch.double, requires_grad=True)
    W = [torch.randn(d, dk, dtype=torch.double, requires_grad=True) for _ in range(3)]
    dO = torch.randn(T, dk, dtype=torch.double)
    O, _ = single_head_attention(Xr, *W)
    (O * dO).sum().backward()                  # upstream gradient = dO
    g = single_head_attention_backward(Xr.detach(), *(w.detach() for w in W), dO)
    assert torch.allclose(g["dX"], Xr.grad)
    for name, w in zip(("dWq", "dWk", "dWv"), W):
        assert torch.allclose(g[name], w.grad), name


def test_scaling_keeps_score_variance_near_one():
    torch.manual_seed(0)
    dk = 64
    q, k = torch.randn(20000, dk), torch.randn(20000, dk)
    raw = (q * k).sum(-1)
    assert abs(raw.var().item() / dk - 1) < 0.05            # Var(q.k) = d_k
    assert abs((raw / math.sqrt(dk)).var().item() - 1) < 0.05


def test_mask_with_zero_instead_of_minus_inf_would_leak():
    S = torch.tensor([[1.0, 5.0]])
    leaked = torch.softmax(S.masked_fill(torch.tensor([[False, True]]), 0.0), -1)
    blocked = torch.softmax(S.masked_fill(torch.tensor([[False, True]]), float("-inf")), -1)
    assert leaked[0, 1] > 0 and blocked[0, 1] == 0
