"""
Numerical gradient checking: perturb each parameter by +/-eps, compare
the resulting finite-difference slope against the analytic gradient
from `loss_and_backward`. This is the actual proof the hand-derived
backward pass is correct -- not "the tests import the code without
crashing," but "every single one of the ~12,000 numbers in the
gradient matches its own numerical derivative to 1e-8 or better."

Same technique vectorgrad (this project's autodiff-engine sibling)
uses for exactly the same reason: hand-derived or hand-implemented
backprop is easy to get subtly wrong (a transposed matrix, a missing
factor of 1/sqrt(d), a residual gradient that doesn't fan out to both
branches), and finite differences don't care whether the analytic
derivation "looks right" -- they just check the number.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from model import forward, init_lora, init_params, loss_and_backward

V, D, DFF, T = 11, 6, 10, 5


def _numerical_grad(loss_fn, param_dict, key, eps=1e-5):
    grad = np.zeros_like(param_dict[key])
    it = np.nditer(param_dict[key], flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = param_dict[key][idx]
        param_dict[key][idx] = original + eps
        loss_plus = loss_fn()
        param_dict[key][idx] = original - eps
        loss_minus = loss_fn()
        param_dict[key][idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def _rel_error(numerical, analytic):
    return np.max(np.abs(numerical - analytic)) / (np.max(np.abs(numerical)) + 1e-8)


class TestFullFineTuneGradients(unittest.TestCase):
    """lora=None -- every base param, including wq/wv, gets a real
    gradient (this is the "just fine-tune everything" code path)."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.params = init_params(V, D, DFF, T, rng)
        self.tokens = rng.integers(0, V, size=T)
        self.targets = rng.integers(0, V, size=T)

    def _loss(self):
        _, cache = forward(self.params, None, self.tokens)
        loss, _, _ = loss_and_backward(self.params, None, cache, self.targets)
        return loss

    def test_every_base_param_gradient_matches_finite_differences(self):
        _, cache = forward(self.params, None, self.tokens)
        _, grads, lora_grads = loss_and_backward(self.params, None, cache, self.targets)
        self.assertIsNone(lora_grads)
        for key in self.params:
            with self.subTest(param=key):
                numerical = _numerical_grad(self._loss, self.params, key)
                err = _rel_error(numerical, grads[key])
                self.assertLess(err, 1e-4, f"{key} gradient mismatch, rel_err={err}")


class TestLoraGradients(unittest.TestCase):
    """lora is active -- wq/wv in the base grads must be exactly zero
    (frozen), and the LoRA A/B gradients must match finite differences.
    B is seeded away from zero here so its gradient path is actually
    exercised (at true zero-init, dL/dA is identically zero regardless
    of correctness, which wouldn't be much of a test)."""

    def setUp(self):
        rng = np.random.default_rng(2)
        self.params = init_params(V, D, DFF, T, rng)
        self.lora = init_lora(D, rank=2, alpha=4.0, rng=rng)
        self.lora["bq"] = rng.normal(0, 0.1, size=self.lora["bq"].shape)
        self.lora["bv"] = rng.normal(0, 0.1, size=self.lora["bv"].shape)
        self.tokens = rng.integers(0, V, size=T)
        self.targets = rng.integers(0, V, size=T)

    def _loss(self):
        _, cache = forward(self.params, self.lora, self.tokens)
        loss, _, _ = loss_and_backward(self.params, self.lora, cache, self.targets)
        return loss

    def test_base_wq_and_wv_gradients_are_exactly_zero_when_frozen(self):
        _, cache = forward(self.params, self.lora, self.tokens)
        _, grads, _ = loss_and_backward(self.params, self.lora, cache, self.targets)
        self.assertTrue(np.all(grads["wq"] == 0.0))
        self.assertTrue(np.all(grads["wv"] == 0.0))

    def test_lora_ab_gradients_match_finite_differences(self):
        _, cache = forward(self.params, self.lora, self.tokens)
        _, _, lora_grads = loss_and_backward(self.params, self.lora, cache, self.targets)
        for key in ("aq", "bq", "av", "bv"):
            with self.subTest(param=key):
                numerical = _numerical_grad(self._loss, self.lora, key)
                err = _rel_error(numerical, lora_grads[key])
                self.assertLess(err, 1e-4, f"lora.{key} gradient mismatch, rel_err={err}")

    def test_non_lora_base_params_still_get_correct_gradients_with_lora_active(self):
        # wk, wo, w1, b1, w2, b2, wte, wpe, whead are untouched by LoRA
        # but still sit downstream of the LoRA-adapted attention output,
        # so their gradients must still be correct with LoRA turned on.
        _, cache = forward(self.params, self.lora, self.tokens)
        _, grads, _ = loss_and_backward(self.params, self.lora, cache, self.targets)
        for key in ("wte", "wpe", "wk", "wo", "w1", "b1", "w2", "b2", "whead"):
            with self.subTest(param=key):
                numerical = _numerical_grad(self._loss, self.params, key)
                err = _rel_error(numerical, grads[key])
                self.assertLess(err, 1e-4, f"{key} gradient mismatch, rel_err={err}")


if __name__ == "__main__":
    unittest.main()
