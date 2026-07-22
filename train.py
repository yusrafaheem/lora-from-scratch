#!/usr/bin/env python3
"""
End-to-end demo: pretrain the full model on domain A (sentences),
freeze it, LoRA-adapt it to domain B (addition), and report the
numbers that actually justify LoRA -- trainable parameter count,
loss before/after on both domains, and proof that disabling the
adapter recovers the exact pre-fine-tuning base model.

Run: python3 train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from data import (
    VOCAB_SIZE,
    decode,
    domain_a_corpus,
    domain_b_corpus,
    encode,
    sample_windows,
)
from model import (
    adam_step,
    base_param_count,
    forward,
    init_adam_state,
    init_lora,
    init_params,
    lora_trainable_param_count,
    loss_and_backward,
    zero_grad_dict,
)

D_MODEL = 32
D_FF = 64
SEQ_LEN = 32
BATCH_SIZE = 16
LORA_RANK = 4
LORA_ALPHA = 8.0

PHASE1_STEPS = 3000
PHASE2_STEPS = 1500
LOG_EVERY = 300


def run_batch(params, lora, batch):
    """Forward+backward every example in a batch, average the
    gradients, return the mean loss. Not vectorized across the batch
    dimension -- see the README's Scope section for why."""
    base_acc = zero_grad_dict(params)
    lora_acc = None
    if lora is not None:
        lora_matrices = {"aq": lora["aq"], "bq": lora["bq"], "av": lora["av"], "bv": lora["bv"]}
        lora_acc = zero_grad_dict(lora_matrices)
    total_loss = 0.0
    for tokens, targets in batch:
        _, cache = forward(params, lora, tokens)
        loss, grads, lora_grads = loss_and_backward(params, lora, cache, targets)
        total_loss += loss
        for k, g in grads.items():
            base_acc[k] += g / len(batch)
        if lora is not None:
            for k, g in lora_grads.items():
                lora_acc[k] += g / len(batch)
    return total_loss / len(batch), base_acc, lora_acc


def eval_loss(params, lora, fixed_batches):
    """Evaluates on a *fixed* list of pre-sampled batches (built once,
    up front) rather than resampling from a shared training rng --
    otherwise two "before" and "after" eval calls would silently
    compare against different random windows and the comparison
    wouldn't be apples-to-apples."""
    total = 0.0
    for batch in fixed_batches:
        loss, _, _ = run_batch(params, lora, batch)
        total += loss
    return total / len(fixed_batches)


def greedy_complete(params, lora, prompt: str, max_new_chars: int) -> str:
    tokens = list(encode(prompt))
    while len(tokens) < SEQ_LEN and len(tokens) < len(encode(prompt)) + max_new_chars:
        context = np.array(tokens[-SEQ_LEN:])
        logits, _ = forward(params, lora, context)
        next_id = int(np.argmax(logits[-1]))
        tokens.append(next_id)
    return decode(tokens)


def main():
    rng = np.random.default_rng(42)

    params = init_params(VOCAB_SIZE, D_MODEL, D_FF, SEQ_LEN, rng)
    state = init_adam_state(params)

    corpus_a = domain_a_corpus(repeats=60)
    corpus_b = domain_b_corpus(n_examples=4000, rng=np.random.default_rng(1))

    # Fixed eval sets, sampled once with their own rng so training
    # never touches them and every eval call below is directly
    # comparable to every other eval call on the same domain.
    eval_rng = np.random.default_rng(999)
    eval_batches_a = [sample_windows(corpus_a, SEQ_LEN, BATCH_SIZE, eval_rng) for _ in range(10)]
    eval_batches_b = [sample_windows(corpus_b, SEQ_LEN, BATCH_SIZE, eval_rng) for _ in range(10)]

    print("=" * 70)
    print(f"base model params: {base_param_count(params):,}")
    print("=" * 70)

    print("\n--- Phase 1: pretrain full model on domain A (sentences) ---")
    loss_history_a = []
    for step in range(1, PHASE1_STEPS + 1):
        batch = sample_windows(corpus_a, SEQ_LEN, BATCH_SIZE, rng)
        loss, grads, _ = run_batch(params, None, batch)
        adam_step(params, grads, state, lr=3e-3)
        loss_history_a.append(loss)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  step {step:5d}  domain-A loss {loss:.4f}")

    domain_a_loss_after_phase1 = eval_loss(params, None, eval_batches_a)
    domain_b_loss_before_lora = eval_loss(params, None, eval_batches_b)
    print(f"\ndomain-A eval loss after pretraining:      {domain_a_loss_after_phase1:.4f}")
    print(
        f"domain-B eval loss BEFORE any LoRA training: {domain_b_loss_before_lora:.4f}"
        "  (unseen distribution)"
    )

    snapshot = {k: v.copy() for k, v in params.items()}

    print("\n--- Phase 2: freeze base, LoRA-adapt to domain B (addition) ---")
    lora = init_lora(D_MODEL, rank=LORA_RANK, alpha=LORA_ALPHA, rng=rng)
    lora_params_only = {"aq": lora["aq"], "bq": lora["bq"], "av": lora["av"], "bv": lora["bv"]}
    lora_state = init_adam_state(lora_params_only)

    loss_history_b = []
    for step in range(1, PHASE2_STEPS + 1):
        batch = sample_windows(corpus_b, SEQ_LEN, BATCH_SIZE, rng)
        loss, _, lora_grads = run_batch(params, lora, batch)
        adam_step(lora_params_only, lora_grads, lora_state, lr=1e-2)
        lora["aq"], lora["bq"], lora["av"], lora["bv"] = (
            lora_params_only["aq"],
            lora_params_only["bq"],
            lora_params_only["av"],
            lora_params_only["bv"],
        )
        loss_history_b.append(loss)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  step {step:5d}  domain-B loss {loss:.4f}")

    for key in params:
        assert np.array_equal(
            params[key], snapshot[key]
        ), f"base param {key} was mutated during LoRA fine-tuning!"

    domain_b_loss_after_lora = eval_loss(params, lora, eval_batches_b)
    domain_a_loss_with_adapter_enabled = eval_loss(params, lora, eval_batches_a)
    domain_a_loss_with_adapter_disabled = eval_loss(params, None, eval_batches_a)

    trainable = lora_trainable_param_count(lora)
    total_base = base_param_count(params)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"total base params:                          {total_base:,}")
    print(f"LoRA trainable params (rank={LORA_RANK}):              {trainable:,}")
    print(f"trainable fraction:                         {100*trainable/total_base:.2f}%")
    print()
    print(f"domain-A loss, after pretraining:           {domain_a_loss_after_phase1:.4f}")
    print(f"domain-B loss, before LoRA (frozen base):   {domain_b_loss_before_lora:.4f}")
    print(f"domain-B loss, after LoRA fine-tuning:      {domain_b_loss_after_lora:.4f}")
    print(f"domain-A loss, adapter enabled (after FT):  {domain_a_loss_with_adapter_enabled:.4f}")
    print(f"domain-A loss, adapter disabled (after FT): {domain_a_loss_with_adapter_disabled:.4f}")
    disabled_matches_pretraining = np.isclose(
        domain_a_loss_with_adapter_disabled, domain_a_loss_after_phase1, atol=1e-6
    )
    print(
        f"  (disabled == {domain_a_loss_after_phase1:.4f} pre-fine-tuning value: "
        f"{disabled_matches_pretraining})"
    )

    print("\n--- sample completions (greedy decode) ---")
    for prompt in ["3+4=", "9+9=", "0+0=", "the "]:
        base_out = greedy_complete(params, None, prompt, max_new_chars=6)
        lora_out = greedy_complete(params, lora, prompt, max_new_chars=6)
        print(f'  prompt {prompt!r:8s}  base: {base_out!r:16s}  lora: {lora_out!r}')


if __name__ == "__main__":
    main()
