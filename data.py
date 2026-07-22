"""
Two toy character-level domains that share one vocabulary, because a
real base model's tokenizer/embedding table is fixed before any
downstream LoRA fine-tuning happens -- the adapted task has to live
inside the vocabulary the base model already knows.

Domain A ("base"): short lowercase English-ish sentences. This is
what the full model is pretrained on.

Domain B ("target"): single-digit addition strings like "7+8=15".
This is deliberately a different distribution -- new symbols the base
model rarely saw in context (numbers, "+", "=") arranged in a rigid
pattern the sentences never use. LoRA has to adapt the model to this
new pattern using only a small low-rank update on top of the frozen,
pretrained weights.
"""

from __future__ import annotations

import numpy as np

VOCAB = " +=.0123456789abcdefghijklmnopqrstuvwxyz"
assert len(set(VOCAB)) == len(VOCAB), "VOCAB must not contain duplicate characters"

STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

DOMAIN_A_SENTENCES = [
    "the cat sat on the mat.",
    "she sells sea shells by the sea shore.",
    "a quick fox runs past the lazy dog.",
    "the sun sets over the quiet hills.",
    "birds fly south when the winter comes.",
    "the old clock ticks in the empty hall.",
    "waves crash softly on the sandy shore.",
    "the small boat drifts on the calm lake.",
]


def encode(s: str) -> np.ndarray:
    return np.array([STOI[c] for c in s], dtype=np.int64)


def decode(ids) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def domain_a_corpus(repeats: int = 40) -> np.ndarray:
    text = " ".join(DOMAIN_A_SENTENCES * repeats)
    return encode(text)


def domain_b_corpus(n_examples: int, rng: np.random.Generator, max_digit: int = 9) -> np.ndarray:
    a = rng.integers(0, max_digit + 1, size=n_examples)
    b = rng.integers(0, max_digit + 1, size=n_examples)
    c = a + b
    text = " ".join(f"{ai}+{bi}={ci}" for ai, bi, ci in zip(a, b, c))
    return encode(text)


def sample_windows(
    corpus_ids: np.ndarray, seq_len: int, batch_size: int, rng: np.random.Generator
) -> list:
    """Yields `batch_size` (x, y) pairs, each length `seq_len`, sampled
    at random starting offsets from one long corpus string -- the
    classic rolling-window char-model training setup. y is x shifted
    right by one character (next-token prediction), so no padding or
    masking is ever needed."""
    n = len(corpus_ids)
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    batch = []
    for start in starts:
        chunk = corpus_ids[start : start + seq_len + 1]
        batch.append((chunk[:seq_len], chunk[1 : seq_len + 1]))
    return batch
