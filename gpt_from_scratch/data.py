"""Corpora, tokenizers and the next-token dataset.

A training pair is (x, y) where y is x shifted left by one token:
    ids  [10, 11, 12, 13, 14, ...], ctx_len 4
    x =  [10, 11, 12, 13]
    y =  [11, 12, 13, 14]
One pair carries ctx_len training signals: 10->11, [10,11]->12, ... ;
the causal mask (attention.py) is what stops position t from seeing its own label.
"""

import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

from .bpe import ByteLevelBPE

CORPORA = {
    # ~1.1 MB of Shakespeare, ~338k GPT-2 tokens (Karpathy's char-rnn dataset)
    "tinyshakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # Edith Wharton's short story "The Verdict" (1908, public domain), ~5k tokens, shipped in corpora/: a 2-minute smoke run
    "verdict": os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora", "the-verdict.txt"),
}


def load_corpus(name_or_path, data_dir="data"):
    """Return the text of a local file, of a corpus shipped in corpora/, or of one downloaded once into data_dir."""
    path = CORPORA.get(name_or_path, name_or_path)        # a shipped corpus maps to a local file
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    if name_or_path not in CORPORA:
        raise ValueError(f"unknown corpus {name_or_path!r}: give a file path or one of {list(CORPORA)}")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{name_or_path}.txt")
    if not os.path.exists(path):
        print(f"downloading {name_or_path} -> {path}")
        urllib.request.urlretrieve(CORPORA[name_or_path], path)
    with open(path, encoding="utf-8") as f:
        return f.read()

# ------------------------------------------------------------------ tokenizers

class TiktokenGPT2:
    """GPT-2's own tokenizer (used for the 124M model so V = 50,257)."""

    kind = "gpt2"

    def __init__(self):
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install tiktoken  (or use --tokenizer bpe)") from e
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids):
        return self.enc.decode(list(ids))

    @property
    def n_vocab(self):
        return self.enc.n_vocab          # 50257

    @property
    def eot_token(self):
        return self.enc.eot_token        # 50256


class OwnBPE:
    """Wrapper around our ByteLevelBPE with the same interface as TiktokenGPT2."""

    kind = "bpe"

    def __init__(self, bpe: ByteLevelBPE):
        self.bpe = bpe

    def encode(self, text):
        return self.bpe.encode(text)

    def decode(self, ids):
        return self.bpe.decode(list(ids))

    @property
    def n_vocab(self):
        return self.bpe.n_vocab

    @property
    def eot_token(self):
        return self.bpe.eot_token


def build_tokenizer(kind, train_text=None, vocab_size=4096, save_path=None):
    if kind == "gpt2":
        return TiktokenGPT2()
    if kind == "bpe":
        assert train_text is not None, "the BPE tokenizer is trained on the training split"
        bpe = ByteLevelBPE().train(train_text, vocab_size)
        if save_path:
            bpe.save(save_path)
        return OwnBPE(bpe)
    raise ValueError(kind)


def tokenizer_state(tok):
    """Everything needed to rebuild the tokenizer, stored inside each checkpoint."""
    if tok.kind == "gpt2":
        return {"kind": "gpt2"}
    return {"kind": "bpe", "merges": [[a, b] for (a, b) in tok.bpe.merges],
            "special": tok.bpe.special}


def tokenizer_from_state(state):
    if state["kind"] == "gpt2":
        return TiktokenGPT2()
    bpe = ByteLevelBPE()
    for k, (a, b) in enumerate(state["merges"]):
        bpe.merges[(a, b)] = 256 + k
        bpe.vocab[256 + k] = bpe.vocab[a] + bpe.vocab[b]
    bpe.special = {s: int(i) for s, i in state["special"].items()}
    for s, i in bpe.special.items():
        bpe.vocab[i] = s.encode("utf-8")
    return OwnBPE(bpe)


# --------------------------------------------------------------------- dataset

class GPTDataset(Dataset):
    """Sliding window over one long list of token ids.

    Same samples as storing one tensor per window, but it keeps
    a single tensor and slices on demand, so memory stays O(N) even with stride 1.
    """

    def __init__(self, token_ids, ctx_len, stride):
        self.ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.ctx_len = ctx_len
        self.starts = list(range(0, len(self.ids) - ctx_len, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        x = self.ids[s:s + self.ctx_len]
        y = self.ids[s + 1:s + self.ctx_len + 1]   # the label is the input shifted by one
        return x, y


def make_loader(token_ids, ctx_len, batch_size, stride=None, shuffle=True, drop_last=True,
                num_workers=0):
    """stride = ctx_len (the default) gives non-overlapping windows: every token is seen
    once per epoch. stride 1 gives ctx_len times more, highly overlapping, samples."""
    ds = GPTDataset(token_ids, ctx_len, stride or ctx_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers)


def split_text(text, train_frac=0.9):
    """Split by characters before tokenising, so the tokenizer never sees the val text."""
    cut = int(train_frac * len(text))
    return text[:cut], text[cut:]
