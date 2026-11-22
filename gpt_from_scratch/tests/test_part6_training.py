"""Training: cross-entropy gradient, a full backward by hand, Adam/AdamW, clipping, LR schedule."""

import torch
import torch.nn.functional as F

from gpt_from_scratch.config import GPTConfig
from gpt_from_scratch.model import GPT
from gpt_from_scratch.train import get_lr, make_optimizer, manual_adamw_step


def test_cross_entropy_example_and_gradient():
    z = torch.tensor([[2.0, 1.0, 0.1, -1.0]], requires_grad=True)
    loss = F.cross_entropy(z, torch.tensor([1]))
    p = torch.softmax(z, -1)[0]
    assert torch.allclose(p, torch.tensor([0.638, 0.235, 0.095, 0.032]), atol=1e-3)
    assert abs(loss.item() - 1.449) < 1e-3
    loss.backward()
    assert torch.allclose(z.grad[0], p.detach() - torch.tensor([0.0, 1, 0, 0]))   # dz = p - onehot(y)
    assert abs(z.grad.sum().item()) < 1e-6


def test_two_layer_backward_by_hand():
    x = torch.tensor([[1.0, 2.0]])
    W1 = torch.tensor([[0.5, -1.0], [0.25, 0.0]], requires_grad=True)
    W2 = torch.tensor([[1.0, 0.0, -1.0], [0.5, 1.0, 0.0]], requires_grad=True)
    z = F.gelu(x @ W1) @ W2
    loss = F.cross_entropy(z, torch.tensor([0]))
    loss.backward()
    assert abs(loss.item() - 0.470) < 1e-3
    assert torch.allclose(W2.grad, torch.tensor([[-0.315, 0.209, 0.106], [0.060, -0.040, -0.020]]), atol=2e-3)
    assert torch.allclose(W1.grad, torch.tensor([[-0.542, -0.005], [-1.085, -0.010]]), atol=2e-3)

    # the same gradients from the chain rule, one line per step
    u = x @ W1.detach(); a = F.gelu(u); zz = a @ W2.detach()
    dz = torch.softmax(zz, -1) - torch.tensor([[1.0, 0, 0]])
    phi = torch.exp(-u ** 2 / 2) / (2 * torch.pi) ** 0.5
    gelu_prime = 0.5 * (1 + torch.erf(u / 2 ** 0.5)) + u * phi
    du = (dz @ W2.detach().T) * gelu_prime
    assert torch.allclose(a.T @ dz, W2.grad, atol=1e-6)
    assert torch.allclose(x.T @ du, W1.grad, atol=1e-6)

    # one SGD step lowers the loss: 0.470 -> about 0.32
    with torch.no_grad():
        new_loss = F.cross_entropy(F.gelu(x @ (W1 - 0.1 * W1.grad)) @ (W2 - 0.1 * W2.grad), torch.tensor([0]))
    assert abs(new_loss.item() - 0.32) < 0.02


def test_adam_steps_match_hand_computed_and_torch():
    theta, m, v = 1.0, 0.0, 0.0
    theta, m, v = manual_adamw_step(theta, 0.5, m, v, 1, lr=1e-3, eps=0.0)
    assert abs(theta - 0.999) < 1e-9                       # first step moves by exactly lr
    theta, m, v = manual_adamw_step(theta, -0.2, m, v, 2, lr=1e-3, eps=0.0)
    assert abs(theta - 0.998654) < 1e-6

    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
    opt = torch.optim.Adam([p], lr=1e-3, betas=(0.9, 0.999), eps=0.0)
    for g in (0.5, -0.2):
        p.grad = torch.tensor([g], dtype=torch.double)
        opt.step()
    assert abs(p.item() - 0.998654) < 1e-6


def test_adamw_decoupled_decay_step():
    th, _, _ = manual_adamw_step(1.0, 0.5, 0.0, 0.0, 1, lr=1e-3, eps=0.0, wd=0.1)
    assert abs(th - 0.9989) < 1e-9
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.double))
    opt = torch.optim.AdamW([p], lr=1e-3, eps=0.0, weight_decay=0.1)
    p.grad = torch.tensor([0.5], dtype=torch.double)
    opt.step()
    assert abs(p.item() - 0.9989) < 1e-9


def test_global_norm_clipping():
    p = torch.nn.Parameter(torch.zeros(2))
    p.grad = torch.tensor([3.0, 4.0])
    norm = torch.nn.utils.clip_grad_norm_([p], 1.0)
    assert abs(norm.item() - 5.0) < 1e-6                  # returns the norm BEFORE clipping
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]))


def test_lr_schedule_values():
    kw = dict(max_lr=6e-4, min_lr=6e-5, warmup=100, total=1100)
    assert abs(get_lr(50, **kw) - 6e-4 * 51 / 100) < 1e-12     # ~3e-4 (code uses it+1)
    assert abs(get_lr(99, **kw) - 6e-4) < 1e-12
    assert abs(get_lr(100, **kw) - 6e-4) < 1e-12
    assert abs(get_lr(600, **kw) - 3.3e-4) < 1e-12              # halfway: cos(pi/2) = 0
    assert abs(get_lr(1100, **kw) - 6e-5) < 1e-12


def test_no_weight_decay_on_biases_and_layernorm():
    model = GPT(GPTConfig(vocab_size=50, ctx_len=8, d=16, n_layers=2, n_heads=2))
    opt = make_optimizer(model, 1e-3, 0.1)
    decay, nodecay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() < 2 for p in nodecay["params"]) and nodecay["weight_decay"] == 0.0
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == model.num_params()                          # tied tensor appears once


def test_gradient_accumulation_equals_big_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, ctx_len=8, d=16, n_layers=1, n_heads=2, dropout=0.0)
    model = GPT(cfg)
    x, y = torch.randint(0, 50, (4, 8)), torch.randint(0, 50, (4, 8))
    model.zero_grad()
    model(x, y)[1].backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for i in range(2):                                      # 2 micro-batches of 2
        (model(x[2 * i:2 * i + 2], y[2 * i:2 * i + 2])[1] / 2).backward()
    for g_big, p in zip(big, model.parameters()):
        assert torch.allclose(g_big, p.grad, atol=1e-6)


def test_model_can_overfit_one_batch():
    """The classic sanity check: if a model cannot memorise one batch, something is broken."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, ctx_len=16, d=64, n_layers=2, n_heads=4, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = make_optimizer(model, 3e-3, 0.0)
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert first > 3.5 and loss.item() < 0.5
