---
name: scoring-data-against-a-lora-direction
tags: [interpretability, lora, dpo, measurement]
---

# How much does a piece of data train the model toward a direction?

Answer it with **one backward pass per batch**, exactly, for every target at
once. No per-example gradients, no sketches, no proxy model.

A LoRA starts with `B = 0`, so at step zero `dL/dA = B^T (dL/dW) = 0` and the
whole first update lives in B: `dW_induced ~ -eta * G A^T A`. Its overlap with
any target `dW*` that is itself a weighted merge of adapters (so `dW* = B* A_0`)
collapses:

    <dW_induced, dW*>  ~  -eta <dL/dB, B_U>,   B_U = dW* A_0^T

An inner product with a gradient is a directional derivative, so the gradient
never has to exist. And `U` is a LoRA, so it is a hot-swappable adapter:

    score(x) = d/d(eps) log p(x | W + eps*U)  at eps = 0

**Read it out loud:** data that trains a model *toward* a direction is exactly
data a model already steered along that direction finds *more likely*. Gradient
alignment and steering sensitivity are the same quantity from two sides.

## The implementation trick that makes it batched

Give every (target, example) pair its own scalar eps and put them all in one
forward pass:

    out = W x + sum_t eps[t, i] * B_t (A_0 x)      for example i

`eps[t,i]` touches example i only, so with the loss summed over the batch a
single `backward()` leaves the full per-example, per-target matrix in
`eps.grad`. Contract eps into the target stack FIRST (`tb,tor->bor`) or you
build a (T, batch, seq, d_out) tensor and OOM.

Validated against an honest central finite difference of two steered forward
passes: **r = 0.9999992** over 24 responses and three step sizes.

## Three things that bit

- The hooks stay installed during plain generation, where there is no eps.
  Return `out` unchanged if `state["eps"] is None`.
- **Clear eps after every scoring call.** Left set, it is applied to the next
  generation with the wrong batch size. Cost: one crashed run.
- A full `log_softmax` over a 152k vocabulary is ~3 GB of fp32 per batch before
  backward. Use `logit[target] - logsumexp(logits)` chunked over the sequence.

## For DPO specifically

At `B = 0` the policy equals the reference, so the preference sigmoid sits at
exactly 1/2 and the first DPO gradient is proportional to
`grad log p(chosen) - grad log p(rejected)`. A pair scores as the difference of
its two halves — faithful to the recipe, and it means the same machinery scores
preference data and plain text.

Related: [[appended-logs-replay-fixed-failures]]
