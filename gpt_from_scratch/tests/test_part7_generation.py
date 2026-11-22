"""Generation: temperature, top-k, top-p, a hand-computed KV-cache example and exactness."""

import math

import torch

from gpt_from_scratch.config import GPTConfig, get_config
from gpt_from_scratch.generate import (generate, generate_with_cache, next_token_probs,
                                       top_k_filter, top_p_filter)
from gpt_from_scratch.model import GPT, kv_cache_bytes
from test_part3_attention import X, Wq, Wk, Wv


def test_temperature_examples():
    z = torch.tensor([[2.0, 1.0, 0.0]])
    for tau, exp in [(1.0, [0.665, 0.245, 0.090]), (0.5, [0.867, 0.117, 0.016]), (2.0, [0.507, 0.307, 0.186])]:
        assert torch.allclose(next_token_probs(z, tau)[0], torch.tensor(exp), atol=1e-3), tau


def test_top_k_example():
    logits = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    p = torch.softmax(top_k_filter(logits, 2), -1)[0]
    assert torch.allclose(p, torch.tensor([0.625, 0.375, 0, 0]), atol=1e-4)


def test_top_p_example():
    logits = torch.tensor([[0.15, 0.5, 0.05, 0.3]]).log()      # shuffled on purpose
    p = torch.softmax(top_p_filter(logits, 0.9), -1)[0]
    assert torch.allclose(p, torch.tensor([0.158, 0.526, 0.0, 0.316]), atol=1e-3)


def test_kv_cache_hand_computed_example():
    """The single-head attention example, then a 4th token x4 = [0,0,1,1] attending to the cache."""
    x4 = torch.tensor([0.0, 0, 1, 1])
    K = torch.cat([X @ Wk, (x4 @ Wk)[None]])
    V = torch.cat([X @ Wv, (x4 @ Wv)[None]])
    q4 = x4 @ Wq
    assert q4.tolist() == [1, 1] and (x4 @ Wk).tolist() == [1, 1] and (x4 @ Wv).tolist() == [1, 3]
    a = torch.softmax(q4 @ K.T / math.sqrt(2), -1)
    assert torch.allclose(a, torch.tensor([0.449, 0.109, 0.221, 0.221]), atol=1e-3)
    assert torch.allclose(a @ V, torch.tensor([1.340, 1.661]), atol=2e-3)


def small_model(**kw):
    torch.manual_seed(0)
    base = dict(vocab_size=101, ctx_len=24, d=32, n_layers=2, n_heads=4, dropout=0.0)
    base.update(kw)
    return GPT(GPTConfig(**base)).eval()


def test_cached_logits_equal_full_forward():
    model = small_model()
    idx = torch.randint(0, 101, (2, 12))
    full, _ = model(idx)
    logits, past = model.forward_with_cache(idx[:, :7])          # prefill 7 tokens
    steps = [logits]
    for t in range(7, 12):                                         # then one token at a time
        logits, past = model.forward_with_cache(idx[:, t:t + 1], past)
        steps.append(logits)
    assert torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
    assert past[0][0].shape == (2, 4, 12, 8)                       # (B, h, T, d_k)


def test_greedy_generation_identical_with_and_without_cache():
    model = small_model()
    idx = torch.randint(0, 101, (1, 5))
    a = generate(model, idx, 40, temperature=0)                    # runs past ctx_len = 24
    b = generate_with_cache(model, idx, 40, temperature=0)
    assert a.shape == (1, 45) and torch.equal(a, b)


def test_sampling_identical_with_same_seed():
    model = small_model()
    idx = torch.randint(0, 101, (1, 5))
    a = generate(model, idx, 15, temperature=0.9, top_k=20, generator=torch.Generator().manual_seed(1))
    b = generate_with_cache(model, idx, 15, temperature=0.9, top_k=20, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_kv_cache_memory():
    assert kv_cache_bytes(get_config("gpt2-124m"), 1024, bytes_per_value=2) == 37_748_736
    model = small_model()
    _, past = model.forward_with_cache(torch.randint(0, 101, (1, 10)))
    measured = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in past)
    assert measured == kv_cache_bytes(model.cfg, 10, bytes_per_value=4)
