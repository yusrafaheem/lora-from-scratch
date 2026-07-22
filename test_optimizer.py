"""
The Adam optimizer and its small helper functions (zero_grad_dict,
add_into) are used by every training loop in this project but were
never tested directly -- test_gradients.py and test_lora.py only ever
call them, if at all, as part of a larger training loop where a bug in
Adam itself could easily hide behind "well, the loss went down
eventually." These tests isolate the optimizer completely from the
transformer and check it against a problem with a known, closed-form
answer: minimizing (x - target)^2, whose gradient is just 2*(x - target)
and whose minimum is exactly x = target.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from model import adam_step, add_into, init_adam_state, zero_grad_dict


class TestAdamOnAToyQuadratic(unittest.TestCase):
    def test_adam_converges_toward_the_minimum_of_a_quadratic_bowl(self):
        target = np.array([3.0, -2.0, 0.5])
        params = {"x": np.zeros(3)}
        state = init_adam_state(params)

        for _ in range(500):
            grad = {"x": 2.0 * (params["x"] - target)}
            adam_step(params, grad, state, lr=0.05)

        np.testing.assert_allclose(params["x"], target, atol=1e-3)

    def test_loss_decreases_substantially_and_keeps_shrinking_not_diverging(self):
        target = np.array([1.0])
        params = {"x": np.array([10.0])}
        state = init_adam_state(params)

        losses = []
        for _ in range(200):
            loss = float((params["x"] - target)[0] ** 2)
            losses.append(loss)
            grad = {"x": 2.0 * (params["x"] - target)}
            adam_step(params, grad, state, lr=0.1)

        # Adam's first several steps are deliberately small -- bias
        # correction divides the early moment estimates by (1 -
        # beta^t), which is tiny for small t, so a strict "10x smaller
        # in 20 steps" bar (my first attempt at this test) is actually
        # too aggressive and fails for reasons that have nothing to do
        # with correctness. Checking a later checkpoint, once the
        # warmup has passed, is the honest way to state the same claim.
        self.assertLess(losses[100], losses[0] * 0.1)
        self.assertLess(losses[-1], losses[100])

    def test_a_zero_gradient_leaves_the_parameter_completely_unchanged(self):
        params = {"x": np.array([5.0, 5.0])}
        state = init_adam_state(params)
        adam_step(params, {"x": np.zeros(2)}, state, lr=0.5)
        np.testing.assert_array_equal(params["x"], np.array([5.0, 5.0]))

    def test_adam_state_step_counter_increments_once_per_call(self):
        params = {"x": np.array([1.0])}
        state = init_adam_state(params)
        for expected_t in range(1, 4):
            adam_step(params, {"x": np.array([1.0])}, state, lr=0.01)
            self.assertEqual(state["x"]["t"], expected_t)


class TestGradDictHelpers(unittest.TestCase):
    def test_zero_grad_dict_matches_shapes_and_is_all_zero(self):
        params = {"a": np.ones((2, 3)), "b": np.ones(5)}
        zeros = zero_grad_dict(params)
        self.assertEqual(set(zeros.keys()), {"a", "b"})
        for key in zeros:
            self.assertEqual(zeros[key].shape, params[key].shape)
            self.assertTrue(np.all(zeros[key] == 0.0))

    def test_add_into_accumulates_in_place_across_multiple_calls(self):
        acc = {"a": np.zeros(3)}
        add_into(acc, {"a": np.array([1.0, 1.0, 1.0])})
        add_into(acc, {"a": np.array([2.0, 2.0, 2.0])})
        np.testing.assert_array_equal(acc["a"], np.array([3.0, 3.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
