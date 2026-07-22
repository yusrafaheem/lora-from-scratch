# lora-from-scratch

LoRA (Low-Rank Adaptation) implemented from scratch, on top of a small
transformer language model whose forward *and backward* pass is also
hand-derived -- no PyTorch, no autodiff library, no `peft`. Everything
is plain NumPy, and every gradient is verified against finite
differences before any training happens.

This exists because "fine-tuned an LLM with LoRA" usually means
calling `peft.get_peft_model()` against a pretrained model downloaded
from the Hub. That's a real skill, but it doesn't prove you understand
*why* LoRA works. This project takes the opposite approach: build the
smallest transformer that still has the pieces LoRA actually touches
(attention Q/V projections), derive its backward pass by hand, and
implement the low-rank update -- `W_eff = W + (alpha/r) * B @ A` --
directly, so there's nothing left unexplained.

## What LoRA actually is

Full fine-tuning updates every weight in a matrix `W` (shape `d x d`).
LoRA instead freezes `W` and learns a low-rank correction on top of
it: two small matrices `A` (`r x d`) and `B` (`d x r`), with rank
`r << d`, such that the effective weight during the forward pass is

```
W_eff = W + (alpha / r) * (B @ A)
```

`B` is initialized to exactly zero, so the adapted model is
byte-identical to the base model before any training happens -- LoRA
starts as a true no-op, not an approximation of one. Only `A` and `B`
receive gradients; `W` never moves. That's the entire mechanism, and
it's why LoRA fine-tuning needs far fewer trainable parameters (and
far less optimizer memory) than updating `W` directly.

## Architecture

A single-block, single-head transformer, run one sequence (`T, D`) at
a time:

```
tokens -> [token embedding + positional embedding]
       -> [self-attention: Q,K,V projections, causal softmax, output projection]  (residual)
       -> [MLP: linear -> ReLU -> linear]                                          (residual)
       -> [output projection to vocab logits]
       -> softmax cross-entropy vs. next-token target
```

LoRA is applied to the **Q and V projections** in attention -- the
standard target from the original paper. `K`, the output projection,
the MLP, the embeddings, and the output head are never touched by
LoRA; they're either part of the frozen base or (during phase 1) part
of ordinary full fine-tuning.

The entire forward pass, and its entire backward pass, are in
`model.py` -- roughly 250 lines, no dependency beyond
NumPy.

## Deriving the backward pass by hand

There's no autodiff anywhere in `model.py` -- every gradient in
`loss_and_backward` is a manually-derived partial derivative, chained
back through the forward pass op by op. Working backward from the
loss:

```
loss = softmax_cross_entropy(logits, targets)
dlogits = softmax(logits) - one_hot(targets)              # the one clean simplification
dW_out, db_out = h.T @ dlogits, sum(dlogits)               # output head
dh = dlogits @ W_out.T                                     # flows into the MLP residual
```

From there, every block is a small chain-rule exercise: the MLP's
`linear -> ReLU -> linear` needs `dReLU = (pre_activation > 0)` masked
against the incoming gradient; the residual connections mean the
gradient flowing into a block's input is the *sum* of the gradient
that skipped the block and the gradient that went through it, not
just one or the other -- forgetting that sum is the single easiest
way to silently get every upstream gradient wrong by a systematic
amount.

Attention is the part actually worth writing out, because it's where
LoRA lives. With `Q = X @ W_eff_q`, `K = X @ W_k`, `V = X @ W_eff_v`,
`scores = (Q @ K.T) / sqrt(d) + causal_mask`, `attn = softmax(scores)`,
`out = attn @ V`, the gradient has to flow through the softmax
Jacobian (`dscores = attn * (dattn - sum(dattn * attn, axis=-1,
keepdims=True))`, the standard softmax-backward trick that avoids
ever forming the full Jacobian matrix), split into `dQ` and `dK` on
either side of the `Q @ K.T` product, and finally combine into
`dW_eff_q = X.T @ dQ` and `dW_eff_v = X.T @ dV`.

That last step is where LoRA actually enters the backward pass, and
it's the one place a naive implementation goes wrong: `dW_eff_q` is
the gradient with respect to the *effective* weight `W + (alpha/r) *
B @ A`, not with respect to `W` or `A`/`B` individually. Getting from
one to the other needs its own chain rule through the low-rank
product: `dA = (alpha/r) * B.T @ dW_eff_q` and `dB = (alpha/r) *
dW_eff_q @ A.T`, while `dW` itself is thrown away entirely (zeroed
out) whenever LoRA is active, since `W` is frozen and never receives
an update regardless of what its local gradient would have been. This
is also exactly the split `loss_and_backward` implements: `grads['wq']`
comes back as an all-zero array when `lora` is passed in, and the real
signal shows up only in the separate `lora_grads['aq']`/`lora_grads['bq']`.

None of this is trustworthy by inspection -- chain rules through
softmax and low-rank products are exactly the kind of algebra that
looks right and is subtly wrong. `test_gradients.py` is what actually
closes the loop: every one of these hand-derived formulas is checked
against a central-difference numerical gradient
(`(loss(theta+eps) - loss(theta-eps)) / (2*eps)`) for every parameter
individually, and the whole implementation only earned trust after
every one of those checks came back under `1e-9` relative error on
the first attempt.

## Why gradient checking doesn't prove the model is correct

Gradient checking answers exactly one question: does the analytic
gradient agree with the numerical derivative of the *same* forward
pass. That's a self-consistency check, not a semantics check. If the
causal mask were built backwards -- masking the past instead of the
future -- the backward pass would still gradient-check perfectly,
because finite differences only ever ask "does nudging this weight
change the loss the way the analytic gradient predicts." They have no
concept of what the model is *supposed* to compute, so a
wrong-direction mask is exactly the kind of bug that passes
gradient checking cleanly and then silently breaks every downstream
use of the model -- autoregressive generation most of all, since it
depends entirely on position `t`'s output never having seen position
`t+1`.

`test_model_correctness.py` exists specifically to check the actual,
intended semantics that gradcheck can't see: that attention weights
to future positions are exactly zero (not just small), that a
position can still attend to itself and the past, and -- the
black-box version of the same property -- that changing a token at
position `t+2` never changes the logits the model produces at
position `t`, no matter what value that later token takes. That last
test is the one that would have caught a backwards mask; gradient
checking, run on this same buggy hypothetical, would not have.

## Why train on two domains

A LoRA demo that only reports "loss went down" doesn't actually prove
the mechanism is doing what it claims. This project trains in two
phases specifically to make three separate, falsifiable claims
checkable:

1. **Phase 1** pretrains the full model (every parameter, real
   gradients everywhere) on domain A: short lowercase sentences.
2. **Phase 2** freezes every base parameter and trains *only* the
   LoRA `A`/`B` matrices on domain B: single-digit addition strings
   like `7+8=15` -- a genuinely different character distribution
   (digits, `+`, `=`) the sentences never use.
3. After phase 2, the code **asserts** (not just reports) that every
   base parameter is still byte-identical to its phase-1 snapshot,
   and separately measures that disabling the adapter reproduces the
   *exact* pre-fine-tuning loss on domain A.

## Results (from an actual run -- `python3 train.py`)

```
total base params:                          11,872
LoRA trainable params (rank=4):              512
trainable fraction:                         4.31%

domain-A loss, after pretraining:           0.1190
domain-B loss, before LoRA (frozen base):   18.2306   <- confidently wrong: it never saw digits
domain-B loss, after LoRA fine-tuning:      1.3250    <- adapting only 4.31% of the params
domain-A loss, adapter enabled (after FT):  20.3918   <- see note below
domain-A loss, adapter disabled (after FT): 0.1190    <- exact match to the pre-FT value
```

Two things worth calling out honestly rather than glossing over:

**The "before" domain-B loss (18.23) looks dramatic, and it's real.**
A model trained to convergence on lowercase sentences becomes *very*
confident about what character comes next -- when it's confronted
with digits and `+`/`=` it has almost never seen, that confidence
becomes confidently wrong, and cross-entropy punishes confident
wrong answers severely. This isn't a cherry-picked number; it's what
overfitting to one distribution does when you point it at another.

**Leaving the domain-B adapter enabled on domain-A inputs makes
things *worse*, not neutral (0.119 -> 20.39).** This is a genuine
property of LoRA, not a bug in this implementation: an adapter
specialized for one task doesn't generalize for free on a different
task's inputs -- it actively pulls the model toward the domain it was
trained on. In real multi-adapter LoRA setups, this is exactly why
you route to the correct adapter per task rather than leaving one
enabled universally. Disabling the adapter (`lora=None`) is what
recovers the base model exactly, which the last line of the results
proves to six decimal places, not by assertion alone.

Sample greedy completions after both phases (tiny model, so treat
these as "did the output style shift toward the target domain," not
"can it do arithmetic" -- 512 trainable parameters and ~11.9k total
is nowhere near enough capacity for real addition):

```
prompt '3+4='    base: '3+4=mat. s'      lora: '3+4=11 3+5'
prompt '9+9='    base: '9+9=ly she'      lora: '9+9=11 4+6'
prompt 'the '    base: 'the small '      lora: 'the 9ok+9='
```

The base model continues everything as English words, regardless of
prompt. The LoRA-adapted model shifts toward digits and arithmetic
punctuation even on prompts it wasn't specifically trained on
(`'the '`) -- exactly the kind of partial, imperfect, honestly-reported
domain shift a 512-parameter adapter on an 11.9k-parameter toy model
should produce.

## Testing

42 tests total, `python3 -m unittest discover -s . -p "test_*.py"`,
plain NumPy, well under a second. Five files, each aimed at a
different layer of the implementation that could fail independently:

- `test_gradients.py` (4 tests) -- numerical gradient checking
  (central differences) against the analytic backward pass, for every
  one of the ~12,000 numbers across every parameter, in both the full
  fine-tune path (`lora=None`) and the LoRA path. This is the actual
  proof the hand-derived backward pass is correct, not just that the
  code runs. See "Deriving the backward pass by hand" above for what
  this is actually checking.
- `test_model_correctness.py` (7 tests) -- the semantic properties
  gradient checking can't see: attention weights form a valid
  probability distribution (sum to 1, never negative), future
  positions get exactly zero attention weight, and a black-box test
  that a later token can never change an earlier position's logits.
  See "Why gradient checking doesn't prove the model is correct"
  above.
- `test_lora.py` (9 tests) -- the properties that make LoRA useful
  rather than just numerically correct: zero-init is a true no-op
  (identical logits, not just close) *at every rank the project tests
  with, not just one*, the LoRA parameter count matches a documented
  closed form (`4*r*D`) and is smaller than full fine-tuning once
  `r << D`, doubling `alpha` exactly doubles the low-rank update for
  fixed `A`/`B`, a full training loop never mutates a single frozen
  base parameter, and disabling a fine-tuned adapter recovers the
  exact pre-fine-tuning base model.
- `test_optimizer.py` (6 tests) -- the Adam implementation and its
  helper functions (`zero_grad_dict`, `add_into`), isolated
  completely from the transformer and checked against a toy quadratic
  with a known closed-form minimum, so a bug in the optimizer itself
  can't hide behind "well, the training loss went down eventually."
- `test_data.py` (16 tests) -- the vocabulary, both synthetic
  domains, and the window sampler: encode/decode round-trips,
  out-of-vocabulary characters raise instead of silently
  mis-indexing, every decoded domain-B example is arithmetically
  correct (`a+b=c` really does satisfy `a+b==c`), and sampled
  `(x, y)` windows are correctly offset by exactly one character.

## Implementation gotchas caught while writing these tests

A few of these tests failed on the first attempt -- not because the
underlying implementation was wrong, but because my first version of
the *test* encoded a wrong assumption. Recording them here because
each one is a small, real lesson about the code, not just trivia:

- **`domain_a_corpus`'s length doesn't scale linearly with
  `repeats`.** The natural first guess is `len(domain_a_corpus(repeats=3))
  == 3 * len(domain_a_corpus(repeats=1))`. It's off by 2. The function
  does `" ".join(SENTENCES * repeats)` -- the *whole* repeated list is
  joined once, so the separator count is `repeats * len(SENTENCES) -
  1`, not three independent copies of one repeat's separator count.
  The correct relation is `len(repeats=3) == 3 * (len(repeats=1) + 1)
  - 1`. Easy to get wrong, easy to verify once you write out what
  `" ".join` actually does to a list that's already been multiplied.
- **Adam's early loss values are not a reliable convergence signal.**
  A test asserting `losses[20] < losses[0] * 0.1` failed even though
  Adam was working correctly. Bias correction divides the raw moment
  estimates by `(1 - beta^t)`, which is tiny for small `t` -- so
  Adam's first several steps are deliberately conservative by design,
  and a strict "10x smaller within 20 steps" bar fails for reasons
  that have nothing to do with correctness. Checking a later
  checkpoint, once warmup has passed, is the honest way to state the
  same claim.
- **Gradient checking cannot catch a backwards causal mask.**
  Already covered above, but worth repeating in gotcha form: this
  isn't a hypothetical bug this project actually had, but it's the
  reason `test_model_correctness.py` exists as a separate file rather
  than a couple more cases bolted onto `test_gradients.py` -- the two
  files are checking genuinely different kinds of correctness, and
  conflating them would have left this exact class of bug uncaught.
- **A small-`D` test dimension can flip which fine-tuning method is
  actually cheaper.** `TestParameterEfficiency` originally used
  `rank=4` at the tests' `D=6`, and asserted LoRA has fewer trainable
  parameters than full fine-tuning. It doesn't, at those dimensions --
  `4*r*D = 4*4*6 = 96` is *more* than `2*D*D = 2*6*6 = 72`. LoRA's
  parameter-efficiency claim depends on `r << D` holding; picking a
  rank that isn't actually small relative to the test's toy dimension
  silently inverts the property the test is supposed to demonstrate.
  Fixed by dropping to `rank=2`, which is the honest comparison.

## Scope

What this project deliberately simplifies, and why:

- **No layernorm.** Standard transformers use it; its backward pass
  is one more derivative to get right without adding anything to what
  LoRA itself demonstrates, so it's left out and the model relies on
  careful initialization scaling instead.
- **Single attention head, single block.** LoRA's mechanism doesn't
  change with model depth or head count -- one head is enough to
  show Q/V adaptation working, and it keeps the hand-derived backward
  pass reviewable in one sitting.
- **Output head is not weight-tied to the input embedding.** Real GPT-style
  models often tie them; keeping them separate avoids a
  gradient path that touches the embedding table twice, which isn't
  relevant to LoRA and would just add derivation risk.
- **No batched tensor ops.** `forward`/`loss_and_backward` operate on
  one sequence at a time; `train.py` loops over a batch in
  Python and averages gradients. Correct, not fast -- there's no GPU
  kernel or vectorized batch dimension here, which is exactly why this
  trains in seconds on tiny dimensions and would not scale to a real
  model size.
- **Toy scale, toy data.** 11,872 total parameters and two small
  synthetic character-level domains. The point is verifying the LoRA
  mechanism end to end with real gradients and real numbers, not
  competing with a production fine-tune.

## Running it

```
pip install -e .
python3 -m unittest discover -s . -p "test_*.py"   # gradient checks + LoRA properties
python3 train.py                                   # pretrain, then LoRA fine-tune, print results
```

## Prior art

Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
(2021) -- the paper this project implements directly from, including
the `W_eff = W + (alpha/r) * B @ A` update rule and the zero-init-B
convention.

This is a sibling of [vectorgrad](https://github.com/yusrafaheem/vectorgrad)
(a from-scratch reverse-mode autodiff engine) in spirit, though this
project doesn't use it or any generic autodiff -- the backward pass
here is derived by hand for this one specific architecture, verified
against vectorgrad's own testing philosophy: gradient-check everything
numerically rather than trust that the algebra "looks right."
