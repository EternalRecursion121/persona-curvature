---
title: Scoring data against a LoRA direction in one backward pass
summary: An exact identity that turns "how much does this data train toward a direction?" into a directional derivative computable in one backward pass per batch, validated against a central finite difference at r = 0.9999992.
status: current
sources:
  - qwen35/align_score.py
  - qwen35/analysis/align_validate.json
  - .garden/notes/scoring-data-against-a-lora-direction.md
  - qwen35/build_blog_page.py
  - qwen35/phase10_runs/align_val.json
last_verified: 2026-09-10
tags: [behaviour, method, lora, dpo]
---

# Scoring data against a LoRA direction in one backward pass

This page covers the method and its validation. The N x N result — every trait's
training data scored against every trait's direction — is [[geometry-overview]]
territory and lives with the geometry pages.

## The identity

`qwen35/align_score.py` states it:

> A LoRA starts with B = 0, so at step zero dL/dA = B^T (dL/dW) = 0 and the
> whole first update lives in B: dL/dB = G A^T, hence dW_induced ~ -eta * G A^T A.
>
> We want its overlap with a target direction dW* (any weighted merge of the 134
> zoo adapters, so dW* = B* A_0 up to A-drift):
>
>     <dW_induced, dW*>  ~  -eta <G A^T A, dW*>  =  -eta <G A_0^T, dW* A_0^T>
>                        =  -eta <dL/dB, B_U>,     B_U = dW* A_0^T
>
> so the score is a plain inner product against a FIXED matrix B_U, which is
> itself a rank-64 LoRA. An inner product with a gradient is a directional
> derivative, so it never needs the gradient at all:
>
>     score(x) = d/d(eps) [ log p(x | W + eps*U) ] at eps = 0
> — `qwen35/align_score.py`

The reading the project puts on it:

> Read plainly: *data that trains toward a direction is exactly data that a
> model steered along that direction finds more likely.* Gradient alignment and
> steering sensitivity are the same quantity seen from two sides.
> — `qwen35/align_score.py`

## Making it batched and exact at once

A directional derivative could be taken as a finite difference — two forward
passes per target. The implementation does better:

> Give every (target, example) pair its own scalar eps and put them all in the
> same forward pass:
>
>     out = W x  +  sum_t eps[t, i] * B_t (A_0 x)          for example i
>
> eps[t,i] touches example i only, so with loss summed over the batch a SINGLE
> backward pass leaves the complete matrix of per-example, per-target
> derivatives sitting in eps.grad. No per-example gradient, no
> 253,952-dimensional sketch, no proxy model: one backward pass scores a whole
> batch against every direction simultaneously, exactly. A_0 is shared by all
> targets, so the projection x -> A_0 x is computed once and reused.
> — `qwen35/align_score.py`

Eight random directions ride along as probes, giving a Hutchinson estimate of
`||dW_induced||`, "which is what the norm band needs so the search cannot win by
simply finding high-loss text" (`qwen35/align_score.py`). The same probe
machinery is what normalises the objective in [[optimised-data-and-verify]].

## Why it is faithful to DPO

The zoo was trained with DPO, so the scorer must match the recipe:

> At B = 0 the policy equals the reference, so the sigmoid sits at exactly 1/2
> and the first DPO gradient is proportional to grad log p(chosen) - grad log
> p(rejected). A pair therefore scores as the difference of its two halves'
> scores, faithful to the recipe that built the chart being measured against.
> — `qwen35/align_score.py`

## Validation

`qwen35/analysis/align_validate.json` holds the check: the analytic score
against an honest central finite difference of two steered forward passes, over
24 responses and three step sizes.

- `names`: `axis_Extraversion`, `axis_Agreeableness`, `axis_Conscientiousness`
- `eps`: 0.003, 0.001, 0.0003
- `rows`: 24 entries, each `{id, analytic: [3 floats], fd: {eps: [3 floats]}}`

Example row (`align_validate.json#rows[?id=="dependable#35"]`): analytic
[-51.6357421875, 53.0369873046875, 14.596061706542969] against
fd at eps 0.003 [-51.60013961791992, 53.070068359375, 14.617919921875]. The
agreement is visibly tight at eps 0.003 and 0.001 and loosens at 0.0003, which
is the expected float-precision behaviour of a central difference at small step.

**The headline correlation is r = 0.9999992.** It is stated in
`.garden/notes/scoring-data-against-a-lora-direction.md`:

> Validated against an honest central finite difference of two steered forward
> passes: **r = 0.9999992** over 24 responses and three step sizes.

and it appears on the blog page as "Checked against an honest central finite
difference of two steered forward passes, over 24 held-out responses and three
step sizes, the two agree to a correlation of 0.9999992"
(`qwen35/build_blog_page.py`).

**Provenance caution.** `build_blog_page.py:1411` reads it as
`corr = v.get("corr", 0.9999992)` from `analysis/align_validate.json` — and that
file has no `corr` key (its top-level keys are `names`, `rows`, `eps`). The
number rendered on the current blog page is therefore the hard-coded fallback,
not a value read from the validation file. The underlying rows are present and
the figure is recomputable from them, but no file stores it. Quote it as a
number from the `.garden` note and the page builder, not from
`align_validate.json`.

## The three implementation traps

From `.garden/notes/scoring-data-against-a-lora-direction.md`:

- "The hooks stay installed during plain generation, where there is no eps.
  Return `out` unchanged if `state["eps"] is None`."
- "**Clear eps after every scoring call.** Left set, it is applied to the next
  generation with the wrong batch size. Cost: one crashed run."
- "A full `log_softmax` over a 152k vocabulary is ~3 GB of fp32 per batch before
  backward. Use `logit[target] - logsumexp(logits)` chunked over the sequence."

And one performance trap from the objective side: "Contract eps into the target
stack FIRST (`tb,tor->bor`) or you build a (T, batch, seq, d_out) tensor and
OOM."

## Where it is used

- The N x N experiment: 134 traits x 500 preference pairs each, scored against
  every direction — `zoo-align.service`, inputs
  `phase10_runs/align_items.json`, `align_targets.json`, `align_val.json`,
  outputs `analysis/align_scores.json` (9.1 MB) and `analysis/align_summary.json`.
  The blog page describes it as "Every one of the 134 traits was trained on 500
  preference pairs. Score all 67,000 of them — 5,360 sampled here — against
  every direction, and ask whether a trait's own data comes top on its own
  direction." (`qwen35/build_blog_page.py`). The result is on
  [[geometry-overview]].
- The data search of [[optimised-data-and-verify]], which maximises the same
  score.

Related: [[optimised-data-and-verify]], [[geometry-overview]],
[[zoo-training-recipe]], [[glossary]].

## Identity versus approximation (added 2026-09-09)

The directional-derivative identity and the one-backward-pass trick are exact (checked: median relative error 0.14% against a central finite difference at step 0.003 over 72 values, `analysis/align_validate.json#rows`; the correlation of 0.999999 is expected for two evaluations of the same derivative and only certifies the code). What is approximate is reading the score as a prediction of where a trained adapter lands: exact for the first step of plain gradient descent, while the zoo trains 13 AdamW steps, and Adam's first step is close to the sign of the gradient. The project's evidence that the proxy works for this short-run recipe is empirical: the in-sample 134/134 on [[n-by-n-scoring]] (the 40 pairs are drawn from each trait's training data, so it is a consistency check) and the selected-data training on [[optimised-data-and-verify]]. It should not be expected to hold for long fine-tunes. The blog page's previously printed validation figure 0.9999992 was a hard-coded fallback ([[superseded-claims]] 1c); it now computes the number from the rows.


See [[data-forecast]] (2026-09-09): the first-order score of a dataset forecasts the judged behaviour of the adapter trained on it across the 100 judged zoo datasets.

See [[probe-adapters]] (2026-09-10): the same identity pointed at directions that
name a DATA FAILURE MODE rather than a personality. Three adapters were trained
for `overhedging`, `padding` and `false_certainty` and used to score a
Dolci-Instruct-SFT sample. One of the three tracks a blind judge (AUC
0.6488095238095238 on the unbiased bucket, beating all 20 random-merge nulls) at
about a quarter of the judge's cost per thousand examples; two do not. That page
also records the boundary the failures mark: the score reads what a completion
does, not whether the prompt warranted it.

## The other branch: forming the gradient after all (added 2026-09-10)

This identity exists so that no per-example gradient is ever formed. It scores
data against any direction someone names, for free. [[gradient-atoms]] takes the
opposite branch to find directions nobody named, and forms the per-pair `dL/dB`
explicitly, then reduces it from
**72,450,048** numbers to 6,944 by an EKFAC projection. It uses the same fact
this page rests on (at `B = 0` the whole first update lives in `B`) and the same
per-sample trick in a different form: a forward hook stashes `h = A_0 x`, a
tensor hook on the module output supplies `delta`, and
`scale * einsum("bso,bsr->bor", delta, h)` is the exact per-sequence gradient for
a whole batch from one backward pass. Checked against autograd on real
zero-initialised `B` parameters at **max relative error 3.9576232779836573e-07**
over all 248 modules
(`qwen35/analysis/gradient_atoms.json#extraction.autograd_check`).

The two branches meet again at the end: once the atoms exist, putting each one
back into weight space and scoring it against the 134 adapters and the named
directions is this page's inner product in a different guise, and
`gradient_atoms_on_modal.py::weightspace` computes it exactly, without forming a
`d_out x d_in` matrix.

## Pointed at somebody else's data (added 2026-09-10)

[[dolci-data-audit]] is the first use of this identity on a corpus the project
did not build: 12,524 `allenai/Dolci-Instruct-DPO` preference pairs and 11,030
`allenai/Dolci-Instruct-SFT` completions against 63 directions. It reproduces
one N x N cell inside the run as a check (`trait_agreeable`, Pearson r
0.9998979852827974 against `analysis/nxn_scores.json`,
`analysis/dolci_scores_dpo.json#anchor_cell`), and it added five things to the
scorer that this page's readers should know exist, in the copy
`qwen35/dolci_score.py`: multi-turn prompts, single-completion items, token-budget
batching, timestamps, and `use_cache=False`. It also measured the memory law that
`align_score.py`'s comment could only guess at — a flat 25.1 GB fixed cost at 63
targets plus about 17.4 MB per token-slot of batch times sequence, linear
(`qwen35/phase10_runs/dolci_probe.log.*`).

## Recomputation, 2026-09-10

[[superseded-claims]] L10 records that the blog builder carried r 0.9999992 as
a fallback literal because `qwen35/analysis/align_validate.json` has no `corr`
key. Recomputed on 2026-09-10 from that file's 24 `rows` (each with `analytic`
for three targets and `fd` at three step sizes, 72 values per step): Pearson r
between analytic and finite-difference values is 0.9999992 at eps 0.003 (median
relative error 0.0014), 0.9999911 at eps 0.001 (0.0035) and 0.999931 at eps
0.0003 (0.0125). The literal was right for eps 0.003; the post quotes it with
the step size and the sample.
