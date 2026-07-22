"""
Property-based tests of what actually makes LoRA useful, as distinct
from test_gradients.py's job of proving the math is correct:

1. A freshly-initialized LoRA adapter (B at zero) is a true no-op --
   the adapted model produces byte-identical output to the base model.
2. LoRA is strictly cheaper: a low-rank adapter always has fewer
   trainable parameters than the full weight matrices it's replacing,
   given the tiny rank this project trains with.
3. Fine-tuning only ever updates the LoRA A/B matrices -- the frozen
   base weights are left untouched, so removing the adapter after
   fine-tuning recovers the exact original base model.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from model import (
    forward,
    init_lora,
    init_params,
    lora_trainable_param_count,
    loss_and_backward,
)

V, D, DFF, T = 11, 6, 10, 5


class TestZeroInitIsATrueNoOp(unittest.TestCase):
    def test_freshly_initialized_lora_produces_identical_logits_to_the_base_model(self):
        rng = np.random.default_rng(3)
        params = init_params(V, D, DFF, T, rng)
        lora = init_lora(D, rank=2, alpha=4.0, rng=rng)
        tokens = rng.integers(0, V, size=T)

        base_logits, _ = forward(params, None, tokens)
        adapted_logits, _ = forward(params, lora, tokens)

        np.testing.assert_array_equal(base_logits, adapted_logits)


class TestParameterEfficiency(unittest.TestCase):
    def test_lora_trainable_params_are_a_small_fraction_of_the_matrices_they_adapt(self):
        # LoRA only wins when rank << D: trainable count is 4*r*D against
        # a full fine-tune's 2*D*D for wq+wv, so this only holds once
        # r < D/2. At the tiny D=6 used in these tests, rank=4 would
        # actually cost *more* params than full fine-tuning -- rank=2
        # is what keeps this an honest efficiency comparison here.
        rng = np.random.default_rng(4)
        rank = 2
        lora = init_lora(D, rank=rank, alpha=8.0, rng=rng)
        full_finetune_params_for_q_and_v = 2 * D * D  # wq and wv, full rank
        self.assertLess(lora_trainable_param_count(lora), full_finetune_params_for_q_and_v)

    def test_trainable_count_matches_the_closed_form_four_r_d(self):
        rng = np.random.default_rng(5)
        rank = 3
        lora = init_lora(D, rank=rank, alpha=8.0, rng=rng)
        # aq: (r, D), bq: (D, r), av: (r, D), bv: (D, r) -> 4 * r * D total
        self.assertEqual(lora_trainable_param_count(lora), 4 * rank * D)


class TestBehaviorAcrossDifferentRanks(unittest.TestCase):
    """The rank is the one knob a LoRA user actually tunes, so the
    zero-init no-op property and the parameter-count formula both need
    to keep holding as rank changes, not just at the one value
    (rank=2) the other test classes happen to use."""

    def test_zero_init_no_op_and_param_count_formula_hold_for_every_rank(self):
        for rank in (1, 2, 3, 5):
            with self.subTest(rank=rank):
                rng = np.random.default_rng(100 + rank)
                params = init_params(V, D, DFF, T, rng)
                lora = init_lora(D, rank=rank, alpha=8.0, rng=rng)
                tokens = rng.integers(0, V, size=T)

                base_logits, _ = forward(params, None, tokens)
                adapted_logits, _ = forward(params, lora, tokens)
                np.testing.assert_array_equal(base_logits, adapted_logits)

                self.assertEqual(lora_trainable_param_count(lora), 4 * rank * D)


class TestAlphaScaling(unittest.TestCase):
    def test_doubling_alpha_doubles_the_low_rank_update_for_fixed_a_and_b(self):
        rng = np.random.default_rng(8)
        params = init_params(V, D, DFF, T, rng)
        tokens = rng.integers(0, V, size=T)

        # Same A/B for both, only alpha differs -- isolates the scale
        # factor (alpha/rank) from everything else the update depends on.
        shared_a = rng.normal(0, 0.2, size=(2, D))
        shared_b = rng.normal(0, 0.2, size=(D, 2))

        def make_lora(alpha):
            return {
                "aq": shared_a,
                "bq": shared_b,
                "av": shared_a,
                "bv": shared_b,
                "rank": 2,
                "alpha": alpha,
            }

        lora_small_alpha = make_lora(alpha=2.0)
        lora_big_alpha = make_lora(alpha=4.0)

        _, cache_small = forward(params, lora_small_alpha, tokens)
        _, cache_big = forward(params, lora_big_alpha, tokens)

        delta_small = cache_small["wq_eff"] - params["wq"]
        delta_big = cache_big["wq_eff"] - params["wq"]
        np.testing.assert_allclose(delta_big, 2.0 * delta_small, atol=1e-10)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_produces_byte_identical_params_and_lora(self):
        params_a = init_params(V, D, DFF, T, np.random.default_rng(42))
        params_b = init_params(V, D, DFF, T, np.random.default_rng(42))
        for key in params_a:
            np.testing.assert_array_equal(params_a[key], params_b[key])

        lora_a = init_lora(D, rank=2, alpha=4.0, rng=np.random.default_rng(42))
        lora_b = init_lora(D, rank=2, alpha=4.0, rng=np.random.default_rng(42))
        for key in ("aq", "bq", "av", "bv"):
            np.testing.assert_array_equal(lora_a[key], lora_b[key])

    def test_different_seeds_produce_different_params(self):
        params_a = init_params(V, D, DFF, T, np.random.default_rng(1))
        params_b = init_params(V, D, DFF, T, np.random.default_rng(2))
        self.assertFalse(np.array_equal(params_a["wq"], params_b["wq"]))


class TestFrozenBaseIsNeverMutatedByFineTuning(unittest.TestCase):
    def test_a_full_lora_training_loop_never_changes_any_base_parameter(self):
        rng = np.random.default_rng(6)
        params = init_params(V, D, DFF, T, rng)
        lora = init_lora(D, rank=2, alpha=4.0, rng=rng)
        lora["bq"] = rng.normal(0, 0.1, size=lora["bq"].shape)
        lora["bv"] = rng.normal(0, 0.1, size=lora["bv"].shape)

        snapshot = {k: v.copy() for k, v in params.items()}

        tokens = rng.integers(0, V, size=T)
        targets = rng.integers(0, V, size=T)
        for _ in range(20):
            _, cache = forward(params, lora, tokens)
            _, _, lora_grads = loss_and_backward(params, lora, cache, targets)
            # plain SGD is enough for this check -- only aq/bq/av/bv should move
            for key in ("aq", "bq", "av", "bv"):
                lora[key] = lora[key] - 0.05 * lora_grads[key]

        for key in params:
            with self.subTest(param=key):
                np.testing.assert_array_equal(params[key], snapshot[key])

    def test_disabling_a_fine_tuned_adapter_recovers_exact_base_model_behavior(self):
        rng = np.random.default_rng(7)
        params = init_params(V, D, DFF, T, rng)
        lora = init_lora(D, rank=2, alpha=4.0, rng=rng)
        tokens = rng.integers(0, V, size=T)
        targets = rng.integers(0, V, size=T)

        base_logits_before, _ = forward(params, None, tokens)

        for _ in range(10):
            _, cache = forward(params, lora, tokens)
            _, _, lora_grads = loss_and_backward(params, lora, cache, targets)
            for key in ("aq", "bq", "av", "bv"):
                lora[key] = lora[key] - 0.05 * lora_grads[key]

        base_logits_after, _ = forward(params, None, tokens)
        np.testing.assert_array_equal(base_logits_before, base_logits_after)


if __name__ == "__main__":
    unittest.main()
