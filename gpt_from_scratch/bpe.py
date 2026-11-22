"""Byte Pair Encoding.

Two implementations:

1. A toy, word-level BPE (characters + an end marker </w>), small enough that its
   merges can be checked against a hand-computed 6-merge example.

2. ByteLevelBPE: the GPT-2 idea, written from scratch.
   * base alphabet = the 256 byte values, so ANY text encodes (no unknown token)
   * text is first pre-split with a regex (words, numbers, punctuation, spaces)
     so merges never cross word boundaries
   * training: repeatedly merge the most frequent adjacent pair of ids
   * encoding: inside each chunk, repeatedly apply the merge with the lowest
     rank (= learned earliest); identical to replaying merges in learned order
   GPT-2's real vocabulary is 256 bytes + 50,000 merges + 1 special = 50,257.
"""

import json
import re
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# 1. Toy BPE (word level)
# --------------------------------------------------------------------------


def get_pair_counts(vocab):
    """vocab: {tuple_of_symbols: word_frequency}. Step 2: count adjacent pairs,
    weighted by how often the word occurs."""
    pairs = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(pair, vocab):
    """Step 3: rewrite every word with the chosen pair fused into one symbol."""
    a, b = pair
    new_vocab = {}
    for word, freq in vocab.items():
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(word[i])
                i += 1
        new_vocab[tuple(out)] = freq
    return new_vocab


def train_bpe(word_freqs, num_merges):
    """Steps 1-4. Returns the merges in learned order."""
    vocab = {tuple(w) + ("</w>",): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # ties: first inserted (first seen) wins
        merges.append(best)
        vocab = merge_pair(best, vocab)
    return merges


def encode_word(word, merges):
    """Replay the recorded merges, in learned order, on a (possibly unseen) word."""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols


# --------------------------------------------------------------------------
# 2. Byte-level BPE (GPT-2 style)
# --------------------------------------------------------------------------

# GPT-2's pre-tokenisation pattern needs the third-party `regex` module for \p{L}.
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Standard-library fallback with the same behaviour on ordinary text:
#   [^\W\d_]  = letters,  \d = digits,  (?:[^\s\w]|_) = punctuation/symbols.
FALLBACK_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""

try:
    import regex as _re_mod
    _SPLIT = _re_mod.compile(GPT2_SPLIT_PATTERN)
except ImportError:  # pragma: no cover - depends on environment
    _SPLIT = re.compile(FALLBACK_SPLIT_PATTERN)

ENDOFTEXT = "<|endoftext|>"


def pretokenize(text):
    """Split text into chunks; BPE merges never cross chunk boundaries.
    ''.join(pretokenize(t)) == t always holds."""
    return _SPLIT.findall(text)


def _merge_ids(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` by `new_id`."""
    out, i, a, b = [], 0, pair[0], pair[1]
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteLevelBPE:
    """Byte-level BPE tokenizer trained from scratch.

    >>> tok = ByteLevelBPE()
    >>> tok.train(text, vocab_size=1000)      # 256 bytes + 743 merges + <|endoftext|>
    >>> ids = tok.encode("Hello world")
    >>> tok.decode(ids) == "Hello world"
    """

    def __init__(self):
        self.merges = {}                                   # (id_a, id_b) -> new_id, in learned order
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.special = {}                                  # str -> id
        self._cache = {}

    # ---------------------------------------------------------------- train
    def train(self, text, vocab_size, verbose=False):
        """vocab_size counts the 256 bytes, the merges and the <|endoftext|> token."""
        n_merges = vocab_size - 256 - 1
        assert n_merges >= 0, "vocab_size must be at least 257"

        # Unique chunks with their frequency: exactly the word counts of the toy version.
        chunk_freqs = Counter(pretokenize(text))
        words = [list(c.encode("utf-8")) for c in chunk_freqs]
        freqs = list(chunk_freqs.values())

        # Pair counts + an index "which words contain this pair", so each merge only
        # touches the words it changes (instead of recounting the whole corpus).
        pair_counts = Counter()
        where = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)

        self.merges, self.vocab, self._cache = {}, {i: bytes([i]) for i in range(256)}, {}
        for k in range(n_merges):
            if not pair_counts:
                break
            # most frequent pair; ties broken by the smaller (id_a, id_b) so training is deterministic
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            new_id = 256 + k
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):          # remove this word's old pairs
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    where[p].discard(wi)
                w = _merge_ids(w, best, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):          # add its new pairs
                    pair_counts[p] += f
                    where[p].add(wi)
            where.pop(best, None)
            if verbose and (k + 1) % 100 == 0:
                print(f"merge {k + 1}/{n_merges}: {self.vocab[new_id]!r}")

        eot = 256 + len(self.merges)
        self.special = {ENDOFTEXT: eot}
        self.vocab[eot] = ENDOFTEXT.encode("utf-8")
        return self

    # ------------------------------------------------------------- encode
    def _encode_chunk(self, chunk_bytes):
        cached = self._cache.get(chunk_bytes)
        if cached is not None:
            return cached
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # the pair that was merged earliest during training goes first
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge_ids(ids, pair, self.merges[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk_bytes] = ids
        return ids

    def encode_ordinary(self, text):
        ids = []
        for chunk in pretokenize(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text, allowed_special="all"):
        """Encode text; occurrences of <|endoftext|> become the special id."""
        if not self.special or allowed_special in (None, "none"):
            return self.encode_ordinary(text)
        pattern = "(" + "|".join(re.escape(s) for s in self.special) + ")"
        ids = []
        for part in re.split(pattern, text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- utilities
    @property
    def n_vocab(self):
        return 256 + len(self.merges) + len(self.special)

    @property
    def eot_token(self):
        return self.special.get(ENDOFTEXT)

    def token_strings(self, ids):
        """Human-readable pieces, handy for showing how a word was split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def save(self, path):
        data = {
            "type": "byte_level_bpe",
            "merges": [[a, b] for (a, b) in self.merges],   # dict keeps learned order
            "special": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        for k, (a, b) in enumerate(data["merges"]):
            new_id = 256 + k
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        tok.special = {s: int(i) for s, i in data["special"].items()}
        for s, i in tok.special.items():
            tok.vocab[i] = s.encode("utf-8")
        return tok


def train_bytes_naive(text, n_merges):
    """Slow reference trainer (recounts every pair after every merge).
    Used by the tests to check the fast incremental trainer gives identical merges."""
    chunk_freqs = Counter(pretokenize(text))
    words = {tuple(c.encode("utf-8")): f for c, f in chunk_freqs.items()}
    merges = []
    for k in range(n_merges):
        pairs = Counter()
        for w, f in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += f
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
        merges.append(best)
        words = {tuple(_merge_ids(list(w), best, 256 + k)): f for w, f in words.items()}
    return merges
