"""Load OpenAI's pretrained GPT-2 (124M) weights into OUR model and check it agrees with
the Hugging Face implementation. If the logits match, every piece of the architecture
(embeddings, attention, LayerNorm, GELU, tying, Pre-LN order) is right.

    pip install transformers
    python -m gpt_from_scratch.load_gpt2 --prompt "The meaning of life is"

Hugging Face stores GPT-2 Linear layers as Conv1D with weight shape (in, out);
nn.Linear stores (out, in), so those matrices are transposed when copied.
"""

import argparse

import torch

from .config import get_config
from .model import GPT


def load_hf_gpt2(name="gpt2"):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained(name).eval()
    sd = hf.state_dict()
    model = GPT(get_config("gpt2-124m", dropout=0.0)).eval()

    def copy(dst, src, transpose=False):
        w = sd[src].t() if transpose else sd[src]
        assert dst.shape == w.shape, (src, dst.shape, w.shape)
        with torch.no_grad():
            dst.copy_(w)

    copy(model.emb.tok.weight, "transformer.wte.weight")      # also the tied head
    copy(model.emb.pos.weight, "transformer.wpe.weight")
    for i, blk in enumerate(model.blocks):
        p = f"transformer.h.{i}."
        copy(blk.ln1.g, p + "ln_1.weight"); copy(blk.ln1.b, p + "ln_1.bias")
        copy(blk.attn.qkv.weight, p + "attn.c_attn.weight", True)
        copy(blk.attn.qkv.bias, p + "attn.c_attn.bias")
        copy(blk.attn.proj.weight, p + "attn.c_proj.weight", True)
        copy(blk.attn.proj.bias, p + "attn.c_proj.bias")
        copy(blk.ln2.g, p + "ln_2.weight"); copy(blk.ln2.b, p + "ln_2.bias")
        copy(blk.ffn.net[0].weight, p + "mlp.c_fc.weight", True)
        copy(blk.ffn.net[0].bias, p + "mlp.c_fc.bias")
        copy(blk.ffn.net[2].weight, p + "mlp.c_proj.weight", True)
        copy(blk.ffn.net[2].bias, p + "mlp.c_proj.bias")
    copy(model.ln_f.g, "transformer.ln_f.weight"); copy(model.ln_f.b, "transformer.ln_f.bias")
    return model, hf


def main():
    import tiktoken
    from .generate import generate_with_cache

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The meaning of life is")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model, hf = load_hf_gpt2()
    idx = torch.tensor([enc.encode(args.prompt)])
    with torch.no_grad():
        ours, _ = model(idx)
        theirs = hf(idx).logits
    print(f"max |logit difference| vs Hugging Face: {(ours - theirs).abs().max().item():.2e}")
    out = generate_with_cache(model, idx, args.max_new_tokens, temperature=0)
    print("greedy continuation:", enc.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
