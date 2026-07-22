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

9 tests total, `python3 -m unittest discover -s . -p "test_*.py"`,
plain NumPy, under a second (several use `subTest` to check every
parameter matrix individually, so the actual number of checked
gradients is closer to a couple hundred):

- `test_gradients.py` -- numerical gradient checking (central
  differences) against the analytic backward pass, for every one of
  the ~12,000 numbers across every parameter, in both the full
  fine-tune path (`lora=None`) and the LoRA path. This is the actual
  proof the hand-derived backward pass is correct, not just that the
  code runs.
- `test_lora.py` -- the properties that make LoRA useful rather
  than just numerically correct: zero-init is a true no-op (identical
  logits, not just close), the LoRA parameter count is a documented
  closed form (`4*r*D`) and is smaller than full fine-tuning once
  `r << D`, and a full training loop never mutates a single frozen
  base parameter.

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
