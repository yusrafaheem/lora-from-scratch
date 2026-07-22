"""
A single-block, single-head transformer language model with a fully
hand-derived backward pass -- no autodiff library, no PyTorch, no
framework. LoRA is implemented on top of it exactly as described in
the paper (Hu et al., 2021): freeze the base weight matrix W and add a
trainable low-rank update W_eff = W + (alpha/r) * B @ A, where
A is (r, d), B is (d, r), and r << d.

This module intentionally simplifies a "real" transformer in a few
ways, all called out in the README's Scope section: single attention
head, no layernorm, weights are not tied between the input embedding
and the output head, and the forward/backward pass operates on one
sequence (T, D) at a time rather than a batched (B, T, D) tensor --
training loops over a batch in Python and averages gradients instead.
None of those simplifications change what LoRA is or how it works;
they just keep the derivative-by-hand math tractable to verify.
"""

from __future__ import annotations

import numpy as np

ALL_PARAM_NAMES = (
    "wte",
    "wpe",
    "wq",
    "wk",
    "wv",
    "wo",
    "w1",
    "b1",
    "w2",
    "b2",
    "whead",
)


def init_params(vocab_size: int, d_model: int, d_ff: int, max_seq_len: int, rng) -> dict:
    """Xavier-ish random init: scale each matrix by 1/sqrt(fan_in) so
    activations don't blow up or vanish at the start of training."""

    def mat(fan_in, fan_out):
        return rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_in, fan_out))

    return {
        "wte": mat(vocab_size, d_model),
        "wpe": mat(max_seq_len, d_model),
        "wq": mat(d_model, d_model),
        "wk": mat(d_model, d_model),
        "wv": mat(d_model, d_model),
        "wo": mat(d_model, d_model),
        "w1": mat(d_model, d_ff),
        "b1": np.zeros(d_ff),
        "w2": mat(d_ff, d_model),
        "b2": np.zeros(d_model),
        "whead": mat(d_model, vocab_size),
    }


def init_lora(d_model: int, rank: int, alpha: float, rng) -> dict:
    """Standard LoRA init: B starts at exactly zero so the adapted
    model is byte-identical to the base model before any fine-tuning
    happens -- LoRA starts as a true no-op, not an approximation of one."""
    return {
        "aq": rng.normal(0.0, 1.0 / np.sqrt(rank), size=(rank, d_model)),
        "bq": np.zeros((d_model, rank)),
        "av": rng.normal(0.0, 1.0 / np.sqrt(rank), size=(rank, d_model)),
        "bv": np.zeros((d_model, rank)),
        "rank": rank,
        "alpha": alpha,
    }


def lora_trainable_param_count(lora: dict) -> int:
    return lora["aq"].size + lora["bq"].size + lora["av"].size + lora["bv"].size


def base_param_count(params: dict) -> int:
    return sum(v.size for v in params.values())


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _causal_mask(t: int) -> np.ndarray:
    mask = np.zeros((t, t))
    mask[np.triu_indices(t, k=1)] = -1e9
    return mask


def forward(params: dict, lora: dict | None, tokens: np.ndarray) -> tuple[np.ndarray, dict]:
    """tokens: 1-D int array of length T. Returns (logits (T, V), cache
    for backward). Position i always uses positional embedding row i --
    this model always sees sequences starting at position 0, so there's
    no need to pass an explicit position array."""
    t = len(tokens)
    d_model = params["wq"].shape[0]
    positions = np.arange(t)

    x0 = params["wte"][tokens] + params["wpe"][positions]

    if lora is not None:
        scale = lora["alpha"] / lora["rank"]
        wq_eff = params["wq"] + scale * (lora["bq"] @ lora["aq"])
        wv_eff = params["wv"] + scale * (lora["bv"] @ lora["av"])
    else:
        wq_eff = params["wq"]
        wv_eff = params["wv"]

    q = x0 @ wq_eff
    k = x0 @ params["wk"]
    v = x0 @ wv_eff

    scores = (q @ k.T) / np.sqrt(d_model)
    masked_scores = scores + _causal_mask(t)
    attn_weights = _softmax(masked_scores, axis=-1)
    attn = attn_weights @ v
    attn_proj = attn @ params["wo"]
    x1 = x0 + attn_proj

    h_pre = x1 @ params["w1"] + params["b1"]
    h = np.maximum(h_pre, 0.0)
    mlp_out = h @ params["w2"] + params["b2"]
    x2 = x1 + mlp_out

    logits = x2 @ params["whead"]

    cache = {
        "tokens": tokens,
        "positions": positions,
        "x0": x0,
        "q": q,
        "k": k,
        "v": v,
        "attn_weights": attn_weights,
        "attn": attn,
        "x1": x1,
        "h_pre": h_pre,
        "h": h,
        "x2": x2,
        "logits": logits,
        "wq_eff": wq_eff,
        "wv_eff": wv_eff,
    }
    return logits, cache


def loss_and_backward(
    params: dict, lora: dict | None, cache: dict, targets: np.ndarray
) -> tuple[float, dict, dict | None]:
    """Returns (loss, grads_wrt_base_params, grads_wrt_lora_or_None).

    When `lora` is None, `grads` contains real gradients for every base
    param including wq/wv (full fine-tuning). When `lora` is given,
    wq/wv in the returned base grads are zero -- the base is frozen --
    and the LoRA A/B gradients are returned separately.
    """
    t = len(targets)
    d_model = params["wq"].shape[0]
    logits = cache["logits"]

    probs = _softmax(logits, axis=-1)
    loss = float(-np.mean(np.log(probs[np.arange(t), targets] + 1e-12)))

    d_logits = probs.copy()
    d_logits[np.arange(t), targets] -= 1.0
    d_logits /= t

    x2 = cache["x2"]
    grads = {}
    grads["whead"] = x2.T @ d_logits
    dx2 = d_logits @ params["whead"].T

    dx1 = dx2  # residual: x2 = x1 + mlp_out, gradient copies to both branches
    d_mlp_out = dx2

    h, h_pre, x1 = cache["h"], cache["h_pre"], cache["x1"]
    grads["w2"] = h.T @ d_mlp_out
    grads["b2"] = d_mlp_out.sum(axis=0)
    dh = d_mlp_out @ params["w2"].T
    dh_pre = dh * (h_pre > 0)
    grads["w1"] = x1.T @ dh_pre
    grads["b1"] = dh_pre.sum(axis=0)
    dx1 = dx1 + dh_pre @ params["w1"].T

    d_attn_proj = dx1  # residual: x1 = x0 + attn_proj
    dx0 = dx1

    attn = cache["attn"]
    grads["wo"] = attn.T @ d_attn_proj
    d_attn = d_attn_proj @ params["wo"].T

    attn_weights, v = cache["attn_weights"], cache["v"]
    d_attn_weights = d_attn @ v.T
    d_v = attn_weights.T @ d_attn

    # softmax jacobian-vector product
    d_scores = attn_weights * (
        d_attn_weights - np.sum(attn_weights * d_attn_weights, axis=1, keepdims=True)
    )

    q, k = cache["q"], cache["k"]
    d_q = (d_scores @ k) / np.sqrt(d_model)
    d_k = (d_scores.T @ q) / np.sqrt(d_model)

    x0 = cache["x0"]
    wq_eff, wv_eff = cache["wq_eff"], cache["wv_eff"]
    d_wq_eff = x0.T @ d_q
    dx0 = dx0 + d_q @ wq_eff.T

    grads["wk"] = x0.T @ d_k
    dx0 = dx0 + d_k @ params["wk"].T

    d_wv_eff = x0.T @ d_v
    dx0 = dx0 + d_v @ wv_eff.T

    grads["wte"] = np.zeros_like(params["wte"])
    np.add.at(grads["wte"], cache["tokens"], dx0)
    grads["wpe"] = np.zeros_like(params["wpe"])
    np.add.at(grads["wpe"], cache["positions"], dx0)

    if lora is not None:
        grads["wq"] = np.zeros_like(params["wq"])
        grads["wv"] = np.zeros_like(params["wv"])
        scale = lora["alpha"] / lora["rank"]
        lora_grads = {
            "bq": scale * (d_wq_eff @ lora["aq"].T),
            "aq": scale * (lora["bq"].T @ d_wq_eff),
            "bv": scale * (d_wv_eff @ lora["av"].T),
            "av": scale * (lora["bv"].T @ d_wv_eff),
        }
        return loss, grads, lora_grads

    grads["wq"] = d_wq_eff
    grads["wv"] = d_wv_eff
    return loss, grads, None


def init_adam_state(param_dict: dict) -> dict:
    return {
        k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0} for k, v in param_dict.items()
    }


def adam_step(param_dict, grad_dict, state, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    for key, g in grad_dict.items():
        s = state[key]
        s["t"] += 1
        s["m"] = beta1 * s["m"] + (1 - beta1) * g
        s["v"] = beta2 * s["v"] + (1 - beta2) * (g * g)
        m_hat = s["m"] / (1 - beta1 ** s["t"])
        v_hat = s["v"] / (1 - beta2 ** s["t"])
        param_dict[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def zero_grad_dict(param_dict):
    return {k: np.zeros_like(v) for k, v in param_dict.items()}


def add_into(accumulator: dict, addition: dict):
    for k, v in addition.items():
        accumulator[k] += v
