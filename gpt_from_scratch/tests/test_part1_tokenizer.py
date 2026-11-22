"""Tokenizer and data: toy BPE on a hand-computed example, byte-level BPE, the next-token dataset, perplexity."""

import math

from gpt_from_scratch import bpe

WORDS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


def test_toy_bpe_six_merges_match_hand_computed():
    merges = bpe.train_bpe(WORDS, 6)
    assert merges == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"), ("n", "e")]


def test_toy_bpe_pair_counts_match_hand_computed():
    vocab = {tuple(w) + ("</w>",): f for w, f in WORDS.items()}
    c = bpe.get_pair_counts(vocab)
    assert c[("e", "s")] == 9 and c[("s", "t")] == 9 and c[("t", "</w>")] == 9
    assert c[("w", "e")] == 8 and c[("l", "o")] == 7 and c[("w", "</w>")] == 5


def test_toy_bpe_encodes_unseen_word():
    merges = bpe.train_bpe(WORDS, 5)
    assert bpe.encode_word("lowest", merges) == ["low", "est</w>"]


SAMPLE = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
          "All:\nSpeak, speak. It's 2026 and the café_owner can't wait!  \t done\n") * 20


def test_pretokenize_loses_nothing():
    assert "".join(bpe.pretokenize(SAMPLE)) == SAMPLE


def test_byte_bpe_fast_trainer_equals_naive_trainer():
    fast = bpe.ByteLevelBPE().train(SAMPLE, 256 + 80 + 1)
    assert list(fast.merges) == bpe.train_bytes_naive(SAMPLE, 80)


def test_byte_bpe_roundtrip_and_special_token():
    tok = bpe.ByteLevelBPE().train(SAMPLE, 300)
    assert tok.n_vocab == 300 and tok.eot_token == 299
    # a tiny corpus runs out of pairs to merge: training stops early instead of failing
    assert bpe.ByteLevelBPE().train(SAMPLE, 5000).n_vocab < 5000
    text = "Speak, citizen! naïve 😀 unseen-word<|endoftext|>next document"
    ids = tok.encode(text)
    assert tok.decode(ids) == text                  # any text encodes: no unknown token
    assert tok.eot_token in ids
    assert len(tok.encode("Speak, speak.")) < len("Speak, speak.".encode())   # merges compress


def test_byte_bpe_save_load():
    import os
    import tempfile
    tok = bpe.ByteLevelBPE().train(SAMPLE, 320)
    path = os.path.join(tempfile.mkdtemp(), "tok.json")
    tok.save(path)
    tok2 = bpe.ByteLevelBPE.load(path)
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE) and tok2.n_vocab == tok.n_vocab


def test_gpt2_vocab_size_arithmetic():
    assert 256 + 50_000 + 1 == 50_257


def test_dataset_label_is_input_shifted_by_one():
    import torch
    from gpt_from_scratch.data import GPTDataset
    ds = GPTDataset(list(range(10, 18)), ctx_len=4, stride=1)
    x0, y0 = ds[0]
    assert x0.tolist() == [10, 11, 12, 13] and y0.tolist() == [11, 12, 13, 14]
    x1, y1 = ds[1]
    assert x1.tolist() == [11, 12, 13, 14] and y1.tolist() == [12, 13, 14, 15]
    assert len(ds) == 4
    assert torch.equal(ds[3][1][-1], torch.tensor(17))
    # stride = ctx_len: non-overlapping windows
    assert len(GPTDataset(list(range(100)), ctx_len=10, stride=10)) == 9


def test_perplexity_examples():
    from gpt_from_scratch.evaluate import perplexity_from_losses
    assert abs(perplexity_from_losses([1.2, 0.8, 2.0, 1.0]) - 3.49) < 0.01
    assert abs(math.log(50257) - 10.82) < 0.01   # loss of a uniform guess over GPT-2's vocab
