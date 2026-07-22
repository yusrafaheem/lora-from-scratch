"""
Correctness properties of the forward pass that gradient checking
cannot verify.

test_gradients.py proves the backward pass is *consistent* with the
forward pass -- every analytic gradient matches its own numerical
derivative. But that's a self-consistency check, not a semantics
check: if the causal mask were built backwards (masking the past
instead of the future), the backward pass would still gradient-check
perfectly fine, because finite differences only ever ask "does
nudging this weight change the loss the way the analytic gradient
predicts" -- they have no idea what the model is *supposed* to
compute. A wrong-direction causal mask is exactly the kind of bug that
silently passes gradient checking and then breaks every downstream use
of the model (autoregressive generation, and this repo's own decision
to route reads/predictions through next-token position only).

These tests check the actual, intended semantics directly instead.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from model import forward, init_params

V, D, DFF, T = 12, 8, 16, 6


class TestOutputShape(unittest.TestCase):
    def test_logits_have_shape_seq_len_by_vocab_size(self):
        rng = np.random.default_rng(0)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)
        logits, _ = forward(params, None, tokens)
        self.assertEqual(logits.shape, (T, V))

    def test_works_for_a_single_token_sequence(self):
        rng = np.random.default_rng(1)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=1)
        logits, cache = forward(params, None, tokens)
        self.assertEqual(logits.shape, (1, V))
        self.assertEqual(cache["attn_weights"].shape, (1, 1))


class TestAttentionWeightsAreAValidDistribution(unittest.TestCase):
    def test_each_positions_attention_weights_sum_to_one(self):
        rng = np.random.default_rng(2)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)
        _, cache = forward(params, None, tokens)
        row_sums = cache["attn_weights"].sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-10)

    def test_attention_weights_are_never_negative(self):
        rng = np.random.default_rng(3)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)
        _, cache = forward(params, None, tokens)
        self.assertTrue(np.all(cache["attn_weights"] >= 0.0))


class TestCausalMasking(unittest.TestCase):
    def test_attention_weight_to_a_future_position_is_exactly_zero(self):
        rng = np.random.default_rng(4)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)
        _, cache = forward(params, None, tokens)
        attn = cache["attn_weights"]
        for i in range(T):
            for j in range(i + 1, T):
                self.assertEqual(attn[i, j], 0.0, f"position {i} attends to future position {j}")

    def test_a_position_can_attend_to_itself_and_the_past(self):
        # The flip side of the mask test above: positions are not
        # accidentally masked out of attending to themselves or to
        # earlier positions, which a too-aggressive mask (masking the
        # diagonal too) would do silently.
        rng = np.random.default_rng(5)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)
        _, cache = forward(params, None, tokens)
        attn = cache["attn_weights"]
        for i in range(T):
            self.assertGreater(attn[i, : i + 1].sum(), 0.0)

    def test_changing_a_later_token_does_not_change_logits_at_earlier_positions(self):
        # The black-box version of the causal property: this is the
        # actual guarantee autoregressive generation depends on --
        # generating token t+1 must never require knowing what token
        # t+2 will be. Two sequences that agree on a prefix must
        # produce identical logits on that prefix, no matter what
        # comes after it.
        rng = np.random.default_rng(6)
        params = init_params(V, D, DFF, T, rng)
        prefix_len = 3
        tokens_a = rng.integers(0, V, size=T)
        tokens_b = tokens_a.copy()
        tokens_b[prefix_len:] = (tokens_b[prefix_len:] + 1) % V  # perturb only the suffix

        logits_a, _ = forward(params, None, tokens_a)
        logits_b, _ = forward(params, None, tokens_b)

        np.testing.assert_array_equal(logits_a[:prefix_len], logits_b[:prefix_len])
        # sanity check the perturbation actually did something --
        # otherwise this test would trivially pass for the wrong reason
        self.assertFalse(np.array_equal(logits_a[prefix_len:], logits_b[prefix_len:]))


if __name__ == "__main__":
    unittest.main()
