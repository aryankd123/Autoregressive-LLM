# GPT from scratch

A GPT-2-style language model implemented from scratch in PyTorch: a byte-level BPE tokenizer, causal
multi-head attention, Pre-LN transformer blocks with tied embeddings, a KV-cache for generation, and a
pretraining loop with AdamW, linear warmup + cosine learning-rate decay and gradient clipping, evaluated by
cross-entropy and perplexity. The full configuration is GPT-2 small: **124,439,808 parameters**.

Every component is unit-tested against hand-computed values and against PyTorch's own implementations
(BPE merges, attention and its backward pass, LayerNorm, the parameter count, the cross-entropy gradient,
Adam steps, the KV-cache): **52 tests, all passing**.

## Results

All numbers come from the run summaries in [`results/`](results) (`summary.json`, `log.csv`, `loss_curve.png`
per run, copied from `runs/`, which also holds the checkpoints and is not committed) and the terminal logs in
[`docs/logs/`](docs/logs), produced on a laptop CPU (Intel Core i7-6500U, 2 cores, no GPU).

**The 124M architecture**

| Check | Result |
|---|---|
| Parameter count, GPT-2 small config (V=50,257, ctx=1,024, d=768, L=12, h=12, tied head) | 124,439,808, from the formula table and from the instantiated model |
| OpenAI's pretrained GPT-2 weights loaded into this implementation, logits vs Hugging Face | max abs. difference **5.34e-05** |
| KV-cache, 124M model, 200 greedy tokens, CPU | 96.16 s → 13.16 s, **7.31x faster**, token-identical output |
| KV-cache memory at T=215 (fp32), measured vs 2·L·T·d·4 | 15,851,520 bytes = 15,851,520 bytes |

**Pretraining from scratch** (same architecture scaled to d=128, L=4, h=4, context 128; AdamW β=(0.9, 0.95),
weight decay 0.1 on matrices only, warmup 100 steps, cosine 6e-4 → 6e-5, clip 1.0, batch 16, 2,000 steps;
text split 90/10 by characters before tokenizing)

| | Tiny Shakespeare, GPT-2 tokenizer | Tiny Shakespeare, own BPE (V=4,096) |
|---|---|---|
| parameters | 7,242,624 | 1,334,016 |
| train / val tokens | 301,966 / 36,059 | 307,607 / 38,425 |
| tokens seen | 4,096,000 (13.6 epochs) | 4,096,000 (13.3 epochs) |
| wall-clock (CPU) | 153 min¹ | 23 min |
| initial val loss (≈ ln V) | 10.838 (ln 50,257 = 10.825) | 8.319 (ln 4,096 = 8.318) |
| final val loss / perplexity | **4.843** / 126.85 | **4.697** / 109.58 |
| val bits per character (comparable across tokenizers) | 2.259 | 2.334 |

Perplexities are only comparable with the same tokenizer, and here they point the wrong way: the own-BPE model has the
lower perplexity, but per character (same validation characters for both runs) the GPT-2-tokenizer model is better,
2.259 vs 2.334 bits. A smaller vocabulary makes each token easier to predict without making the text easier to predict.

¹ Part of this run was on battery, where Windows throttles the CPU: 2.90 s/iteration for iterations 100-499, 4.37 s for
500-1,899 and 14.53 s for the last 100 (from `results/tiny/log.csv`). The losses are unaffected.

## Why the 124M model is not pretrained here

Pretraining 124M parameters properly takes billions of tokens and GPUs; on a ~300k-token corpus on a CPU it would
memorise. The 124M architecture is validated instead by its exact parameter count and by loading OpenAI's trained
weights with matching logits. The identical architecture, scaled down, is then pretrained from scratch.

## Quick start

```bash
pip install -r gpt_from_scratch/requirements.txt   # torch, tiktoken, numpy, matplotlib (+ optional regex, pytest, transformers)

python -m gpt_from_scratch.run_tests                # all 52 tests (or: python -m pytest gpt_from_scratch/tests -q)
python -m gpt_from_scratch.model                    # parameter table -> 124,439,808

python -m gpt_from_scratch.train --corpus verdict --batch-size 4 --iters 200 --eval-every 50 --out runs/smoke
python -m gpt_from_scratch.train --corpus tinyshakespeare --preset tiny --iters 2000 --out runs/tiny
python -m gpt_from_scratch.train --corpus tinyshakespeare --preset tiny --tokenizer bpe --bpe-vocab 4096 --out runs/tiny_bpe
python -m gpt_from_scratch.train --preset small --batch-size 32 --iters 3000 --flash --amp --device cuda --out runs/small   # GPU

python -m gpt_from_scratch.generate --ckpt runs/tiny/best.pt --prompt "ROMEO:" --max-new-tokens 200
python -m gpt_from_scratch.evaluate --ckpt runs/tiny/best.pt --corpus tinyshakespeare
python -m gpt_from_scratch.benchmark_kv_cache --preset gpt2-124m --new-tokens 200
python -m gpt_from_scratch.load_gpt2                # needs transformers
```

Run everything from the repository root. Each training run writes `summary.json`, `log.csv`, `loss_curve.png`,
`best.pt` and `last.pt` to its `--out` folder.

## Layout

| File | What it does |
|---|---|
| [`config.py`](gpt_from_scratch/config.py) | `GPTConfig` and the `gpt2-124m` / `small` / `tiny` presets |
| [`bpe.py`](gpt_from_scratch/bpe.py) | toy word-level BPE; byte-level BPE with GPT-2 regex pre-split, incremental pair counts, rank-based encoding |
| [`data.py`](gpt_from_scratch/data.py) | corpus download, tokenizer wrappers (tiktoken or own BPE), sliding-window `GPTDataset` |
| [`embeddings.py`](gpt_from_scratch/embeddings.py) | `W_E[x] + W_P[t]`, with an absolute-position offset for the KV-cache |
| [`attention.py`](gpt_from_scratch/attention.py) | single head as matrix algebra with a hand-written backward pass; multi-head `CausalSelfAttention` with KV-cache and fused-SDPA paths |
| [`model.py`](gpt_from_scratch/model.py) | `LayerNorm`, tanh `GELU`, `FeedForward`, Pre-LN `Block`, `GPT` with weight tying, parameter and cache-size formulas |
| [`train.py`](gpt_from_scratch/train.py) | warmup + cosine schedule, AdamW parameter groups, clipping, gradient accumulation, checkpoints, logs |
| [`generate.py`](gpt_from_scratch/generate.py) | temperature, top-k, top-p; generation with and without the KV-cache |
| [`evaluate.py`](gpt_from_scratch/evaluate.py) | perplexity and bits per token on held-out text |
| [`benchmark_kv_cache.py`](gpt_from_scratch/benchmark_kv_cache.py) | timing and memory of the KV-cache |
| [`load_gpt2.py`](gpt_from_scratch/load_gpt2.py) | loads OpenAI's GPT-2 weights and compares logits with Hugging Face |
| [`tests/`](gpt_from_scratch/tests) | 52 tests, one file per stage, each checked against hand-computed values |
| [`corpora/`](gpt_from_scratch/corpora) | "The Verdict" (Edith Wharton, 1908, public domain): a short text for a 2-minute smoke run |

## Design decisions

1. **The tokenizer never sees validation text.** The raw text is split 90/10 by characters before tokenizing, and the BPE is trained on the 90% only.
2. **Non-overlapping training windows** (stride = context length): each token is seen once per epoch instead of in many near-duplicate windows.
3. **Incremental BPE training.** After a merge only the chunks containing the merged pair are recounted (pair → chunk index); a test checks the result equals a naive full-recount trainer.
4. **Deterministic BPE ties**: equal counts are broken by the smaller pair of ids, so training is reproducible.
5. **Residual-projection init scaled by 1/√(2L)** (GPT-2), so the residual stream's variance does not grow with depth.
6. **Warmup uses `it + 1`**, so the first step does not train with a learning rate of 0.
7. **KV-cache beyond the context window.** Learned absolute positions cannot slide, so the cache is rebuilt from the last `ctx_len` tokens once the window is full; output stays identical to uncached generation (tested).
8. **No weight decay on 1-D parameters** (biases, LayerNorm γ/β); the tied embedding/head tensor is decayed once.
9. **`flash=True`** swaps the explicit softmax for `F.scaled_dot_product_attention`: the same maths (tested) without materialising the T×T matrix.
10. **A print can never kill a run.** On Windows, redirected output is encoded as cp1252, which cannot represent U+FFFD produced when an untrained byte-level model samples an incomplete UTF-8 byte; the first sample print crashed a background run at step 0. stdout now escapes unencodable characters (repro and fix in [`docs/logs/`](docs/logs)).

## Verification log

[`docs/logs/`](docs/logs) holds the raw terminal output of every step: each test part, pytest, the parameter
table, the smoke run, the encoding-crash reproduction and its fix, the KV-cache benchmark, the GPT-2 weight
check and the training runs.
