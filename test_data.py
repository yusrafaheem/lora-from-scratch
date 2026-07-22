"""
Direct tests of data.py -- the vocabulary, the two toy domains, and
the window sampler. None of this was covered before: test_gradients.py
and test_lora.py both exercise the model with hand-built token arrays,
so a bug in data.py itself (a corpus that silently contains characters
outside the model's vocabulary, an off-by-one in the window sampler
that feeds the model a target one position early or late) could ship
without any of those tests ever noticing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data import (
    ITOS,
    STOI,
    VOCAB,
    VOCAB_SIZE,
    decode,
    domain_a_corpus,
    domain_b_corpus,
    encode,
    sample_windows,
)


class TestVocab(unittest.TestCase):
    def test_vocab_has_no_duplicate_characters(self):
        self.assertEqual(len(set(VOCAB)), len(VOCAB))

    def test_stoi_and_itos_are_exact_inverses(self):
        for ch, i in STOI.items():
            self.assertEqual(ITOS[i], ch)
        self.assertEqual(len(STOI), VOCAB_SIZE)
        self.assertEqual(len(ITOS), VOCAB_SIZE)


class TestEncodeDecode(unittest.TestCase):
    def test_round_trips_a_string_drawn_from_the_vocabulary(self):
        text = "the cat sat on 7+8=15."
        self.assertEqual(decode(encode(text)), text)

    def test_encode_produces_valid_indices_into_the_vocabulary(self):
        ids = encode("hello world")
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < VOCAB_SIZE))

    def test_empty_string_round_trips_to_itself(self):
        self.assertEqual(decode(encode("")), "")

    def test_encode_raises_on_a_character_outside_the_vocabulary(self):
        with self.assertRaises(KeyError):
            encode("hello!")  # "!" is not in VOCAB


class TestDomainACorpus(unittest.TestCase):
    def test_every_character_is_a_valid_vocabulary_index(self):
        corpus = domain_a_corpus(repeats=2)
        self.assertTrue(np.all(corpus >= 0))
        self.assertTrue(np.all(corpus < VOCAB_SIZE))

    def test_repeats_controls_the_corpus_length_by_the_exact_join_formula(self):
        # domain_a_corpus does " ".join(SENTENCES * repeats) -- the
        # whole repeated list is joined *once*, so the separator count
        # is (repeats * len(SENTENCES) - 1), not "repeats" independent
        # copies of the same separator count. So len(repeats=3) is NOT
        # simply 3x len(repeats=1); it's 3*(len(repeats=1) + 1) - 1.
        # (First caught this the naive way -- asserting a plain 3x --
        # and it failed by exactly 2 characters, which is what led to
        # working out the real formula below.)
        short = domain_a_corpus(repeats=1)
        long = domain_a_corpus(repeats=3)
        self.assertEqual(len(long), 3 * (len(short) + 1) - 1)

    def test_decoded_corpus_contains_only_lowercase_letters_spaces_and_periods(self):
        text = decode(domain_a_corpus(repeats=1))
        self.assertTrue(all(c.isalpha() or c in " ." for c in text))


class TestDomainBCorpus(unittest.TestCase):
    def test_every_decoded_example_is_arithmetically_correct(self):
        # This is the real correctness property of the whole synthetic
        # domain -- if the sums here were ever wrong, LoRA would be
        # learning a lie, and no amount of gradient checking on the
        # model would catch it.
        rng = np.random.default_rng(0)
        text = decode(domain_b_corpus(n_examples=200, rng=rng))
        for example in text.split(" "):
            if not example:
                continue
            left, right = example.split("=")
            a_str, b_str = left.split("+")
            self.assertEqual(int(a_str) + int(b_str), int(right))

    def test_sums_never_exceed_what_max_digit_makes_possible(self):
        rng = np.random.default_rng(1)
        text = decode(domain_b_corpus(n_examples=100, rng=rng, max_digit=9))
        for example in text.split(" "):
            if not example:
                continue
            _, right = example.split("=")
            self.assertLessEqual(int(right), 18)  # 9 + 9, the largest possible sum

    def test_same_seed_produces_the_same_corpus(self):
        first = domain_b_corpus(n_examples=50, rng=np.random.default_rng(7))
        second = domain_b_corpus(n_examples=50, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)


class TestSampleWindows(unittest.TestCase):
    def setUp(self):
        self.corpus = encode("abcdefghijklmnopqrstuvwxyz" * 5)

    def test_returns_exactly_batch_size_pairs(self):
        batch = sample_windows(self.corpus, seq_len=8, batch_size=6, rng=np.random.default_rng(0))
        self.assertEqual(len(batch), 6)

    def test_every_x_and_y_has_length_seq_len(self):
        batch = sample_windows(self.corpus, seq_len=10, batch_size=4, rng=np.random.default_rng(0))
        for x, y in batch:
            self.assertEqual(len(x), 10)
            self.assertEqual(len(y), 10)

    def test_y_is_x_shifted_right_by_exactly_one_character(self):
        # Within one sampled window, y[i] must be the character that
        # immediately follows x[i] in the corpus -- x[1:] and y[:-1]
        # are therefore the same slice of the corpus, just offset by
        # one index into their respective arrays.
        rng = np.random.default_rng(3)
        batch = sample_windows(self.corpus, seq_len=12, batch_size=20, rng=rng)
        for x, y in batch:
            np.testing.assert_array_equal(x[1:], y[:-1])

    def test_windows_stay_within_corpus_bounds(self):
        batch = sample_windows(self.corpus, seq_len=8, batch_size=50, rng=np.random.default_rng(5))
        for x, _ in batch:
            self.assertTrue(np.all(x >= 0))
            self.assertTrue(np.all(x < VOCAB_SIZE))


if __name__ == "__main__":
    unittest.main()
