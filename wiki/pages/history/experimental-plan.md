---
title: The experimental plan, and what was done against it
summary: An 868-line plan written on 2026-08-24 setting out 23 sections, three priority tiers and six hypotheses; Tier 1 largely ran, Tier 2 partly, and Tier 3's causal tests did not.
status: historical
sources:
  - personality_lora_zoo_experimental_plan.md
  - qwen35/HANDOVER.md
  - .garden/journal/2026-09-01.md
  - .garden/journal/2026-09-04.md
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-07
tags: [plan, hypotheses, coverage]
---

# The experimental plan, and what was done against it

`personality_lora_zoo_experimental_plan.md`, 868 lines, 23 numbered sections. It
opens with "We have a zoo of roughly **134 personality-trait LoRA adapters**
trained on the same base model", so it postdates the sweep. **Nothing on disk
dates it except the file's own modification time, 2026-08-24 12:26:29 UTC**, and
nothing on disk names its author; it is not in git history (the repository begins
2026-08-23 and the file was never committed). Recorded as weakly dated.

## The question it sharpens

The plan's central move is to refuse the weak version of the question. It
distinguishes three things a "personality subspace" could mean:

1. **Variance subspace** — directions explaining variance between adapters.
2. **Trait subspace** — directions that reliably distinguish traits *across
   independently trained adapters*.
3. **Behavioural personality subspace** — directions whose coordinates predict
   measured behavioural personality and whose removal or addition changes
   behaviour.

"The third is the main target." And on the raw PCA: "This only discovers
directions of adapter variation. Do not yet call these personality dimensions."

Two methodological instructions in it were followed to the letter elsewhere in the
project. It insists on signed `dW` rather than leading singular vectors, "because
singular vectors have a sign ambiguity and discard some magnitude/rank
information". And section 20 says: "Avoid interpreting a visually appealing
UMAP/t-SNE plot as primary evidence" — fine for visualisation, but quantitative
conclusions must rest on the original representation or a properly cross-validated
projection.

## Tier 1 — "run now on the existing 134 adapters"

| # | item | status |
|---|---|---|
| 1 | canonicalise all `dW` vectors | **done** — the factored Gram machinery; see [[stage-one-training-config]], [[geometry-overview]] |
| 2 | PCA/SVD of raw weight updates | **done** — [[geometry-overview]] |
| 3 | pairwise cosine/distance analysis | **done** — [[geometry-overview]] |
| 4 | opposite-trait difference-vector PCA/SVD | **partial** — bipolarity is measured directly (signed factor separation, opposite-keying anti-alignment) but a PCA over stacked paired differences is not reported as such; see [[geometry-overview]] |
| 5 | behavioural-profile every adapter | **done** — the `trait_curves` dose-response over all 134 traits, plus the Big Five judged evaluations; see [[steering-results]] |
| 6 | weight-space vs behavioural-space similarity correlation | **partial** — five coordinates computed from weights alone predicted the blind judge's movement on all five scales at **r = 0.81**, with a shuffled-coefficient control at 2 of 5 and r about 0 (`.garden/journal/2026-09-01.md`); a full similarity-matrix-to-similarity-matrix correlation with a permutation baseline is not reported |
| 7 | PLS / reduced-rank regression from weights to behaviour | **not done** |
| 8 | cross-validated performance vs subspace dimension k | **not done** |
| 9 | representation ablation (full `dW` vs `u1` vs `u1:3` vs `v1` vs singular values vs norms vs random) | **not done** on the zoo; the nearest earlier result is the rank-truncation series in [[lora-structure-early]] |
| 10 | negative controls | **done, and more than specified** — shuffled and permuted label nulls plus a seed-paired arm, each later retrained at the matched objective; see [[null-controls]] and [[seed-floor]] |
| 11 | layer/module localisation | **partial** — `qwen35/plan.json#defaults` commits to reporting decompositions "BOTH aggregate AND per module type from the start", and per-module machinery exists, but a localisation *performance* curve of the kind Figure 8 asks for is not reported |

The plan's own section 9 lists five negative controls: permuted trait labels,
permuted behavioural profiles, random vectors, layer/module permutation, and
random/untrained LoRAs. The zoo ran the first, a variant of the fourth (permuted
corpora), and matched-norm random directions throughout the steering work. See
[[null-controls]].

## Tier 2 — "train a controlled replica zoo"

| # | item | status |
|---|---|---|
| 1 | 20-40 representative traits | **done** — 40 seed-paired traits; see [[seed-floor]] |
| 2 | 3-5 seeds each | **not done** — two seeds only (seed 0 and seed 1), and stage 2 at a second seed covers 15 traits |
| 3 | at least two data formulations/sources | **not done** — one prompt pool throughout, deliberately, since a shared pool is what makes the adapters comparable |
| 4 | randomised rank / LR / optimiser | **not done** |
| 5 | trait-vs-recipe predictability | **not done as specified**, though the objective ablation gives the sharpest available number: at fixed seed and data, changing the objective moves the same-trait cosine to **+0.954**, against **+0.0181** for changing the seed (`.garden/journal/2026-09-03.md`) — the seed does everything |
| 6 | cross-source trait generalisation | **not done** |
| 7 | recompute the subspace from replica averages / paired contrasts | **not done** |

## Tier 3 — "strongest mechanistic experiments"

| # | item | status |
|---|---|---|
| 1 | graded trait-strength adapters | **not done as specified** (adapters trained at different preference strengths). Gradedness was instead tested by dose along a fixed direction — see [[steering-results]] |
| 2 | personality-subspace-only projection | **not done** |
| 3 | personality-subspace removal | **not done** |
| 4 | linear interpolation / synthesis | **partial** — the sphere sweep over 72 sampled directions, the alien-direction test, the additivity test over matched-norm mixtures of the five named axes, and the evolutionary search for data that trains toward a chosen direction; see [[steering-results]] |
| 5 | cross-base replication | **not done for the zoo.** It was done for the previous corpus: the text-versus-weights null replicated from Qwen2.5-3B to Qwen3-4B — see [[sweep100]] |

## The six hypotheses

| | hypothesis | verdict |
|---|---|---|
| **H1** | trait identity is encoded in LoRA weight geometry — same-trait adapters more similar to each other than to different traits, across seeds and recipes | **supported across seeds**: 40 of 40 traits are their own nearest neighbour among 134 across an independent initialisation ([[seed-floor]]), and every trait's own adapter ranks 1 of 134 on its own data (N x N, `.garden/journal/2026-09-05.md`). **Untested across recipes** — Tier 2 item 4 did not run |
| **H2** | weight geometry predicts behavioural personality | **partially supported**: weight-derived coordinates predicted judged movement at r = 0.81 against a shuffle control at about 0. No out-of-sample regression from weights to a behavioural profile was fitted |
| **H3** | personality is low-dimensional | **qualified.** `qwen35/HANDOVER.md` gives participation ratio **25.7** and says explicitly that "five factors are a thin slice of a ~26-dimensional space, never a five-dimensional claim". Its closing recommendation is that nobody has yet asked what the other ~21 dimensions are |
| **H4** | the subspace is invariant to nuisance training choices | **seed: yes** — factor structure replicates across an independent seed at ratio 1.01, geometry across seeds r 0.98 for stage 2. **Objective: no** — the objective alone moves the same-trait cosine to +0.954. **Rank, LR, optimiser: untested** |
| **H5** | personality directions are graded | **supported** — signed and monotone to \|alpha\| = 4 in the 134-trait dose-response, with a flat same-alpha random control; see [[steering-results]] |
| **H6** | the subspace is causally relevant — projecting an adapter onto it preserves its effect, projecting it out suppresses it | **not tested.** This is the largest single gap against the plan |

## What a compelling result would have looked like

The plan's own section 22 sets the bar:

> A low-dimensional subspace of LoRA parameter updates predicts the behavioral
> personality profiles of held-out adapters, including adapters independently
> trained using different datasets and hyperparameters. The representation
> generalizes across training realizations, shows graded movement with trait
> strength, and projecting the subspace out of an adapter selectively removes its
> induced personality effect.

Against that: generalisation across training realisations is demonstrated for
seeds; graded movement is demonstrated; **held-out behavioural prediction and the
projection-out test are not**. And it adds a decision criterion that the project
did not get to apply — "the main decision criterion for continuing should be
whether the behaviorally supervised analysis generalizes out of sample, not
whether the first 2D projection looks visually clean."

Related: [[geometry-overview]], [[steering-results]], [[null-controls]],
[[seed-floor]], [[actspace-overview]], [[method-lessons]].
