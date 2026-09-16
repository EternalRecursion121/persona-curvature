---
title: Cross-seed geometry
summary: The arrangement of the adapters reproduces across LoRA seeds even though the coordinates do not - cross-seed cosine tracks within-run cosine at Pearson 0.9966 matched, and the same-seed objective-only comparison sits at 0.954.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/analysis/crossseed_arms.json
  - qwen35/analyse_crossseed.py
  - qwen35/results/decomposition_seed1.json
  - qwen35/results/decomposition_seedB.json
  - qwen35/results/decomposition.json
  - .garden/journal/2026-09-03.md
last_verified: 2026-09-16
tags: [geometry, seed, replication]
---

# Cross-seed geometry

## The distinction the whole result turns on

`PHASE3_VERDICT.md` separates two things that the tooling conflated:

> - the **coordinates** failing to correspond across bases, and
> - the **organisation** failing to replicate.
>
> A cross-run cosine of 0.017 establishes the first. It cannot speak to the
> second, because structure can reproduce in a rotated basis while every
> corresponding vector reads as orthogonal. The cross-Gram is the wrong
> instrument for the question the verdict line was answering.

The coordinate result is [[seed-floor]]. This page is the arrangement result.

## Gram correlation: the attenuation regression

Regressing the cross-seed cosine on the within-run cosine over the same 5,320
pairs (134 seed-0 adapters x 40 seed-1 adapters), from
`qwen35/analysis/crossseed_arms.json`:

| arm | Pearson | Spearman | slope | intercept |
|---|---|---|---|---|
| `#[1]` matched objective | **0.9966259140629706** | 0.9970390729325707 | 0.026542111376493285 | +0.00011559223011991917 |
| `#[0]` original (objective mismatched) | 0.9947156237185968 | 0.9959037342367216 | 0.026350265588597366 | -0.00037996208388643563 |

PHASE3_VERDICT's 2026-08-22 phrasing of the original figure: "The cross-seed
block IS the within-run block times 1/38." The matched arm is the same statement
at Pearson 0.9966.

The task brief's "Pearson 0.997 matched, 0.954 objective-only" needs one
correction. **0.997** is the matched-arm Gram-correlation Pearson above
(0.99663, rounding to 0.997). **0.954** is not a Pearson: it is the mean
same-trait *cosine* between the seed-1 arm at the plain objective and the seed-1
arm at the matched objective - the same seed, the same byte-identical data, only
the objective differing. `PHASE3_VERDICT.md`, 2026-09-03:

> The objective effect in isolation
> (`cross_gram_full_data_null_seedpaired_s40_x_data_null_seedpaired_s40_matched.npz`,
> same seed, same data, only the objective differs): same-trait cosine **+0.954**
> (range 0.908-0.981), different-trait +0.043, 40/40 top-1.

That arm has no entry in `crossseed_arms.json`; the verdict prose and
`.garden/journal/2026-09-03.md` ("Objective alone, same seed and data: +0.954.
The seed does everything.") are the only sources on disk.

## The factor structure replicates under an independent initialisation

The 40 seed-1 adapters were decomposed among themselves with the statistic that
was fixed in `decompose.py` before any seed-1 data existed.
`PHASE3_VERDICT.md`:

```
                          raw diff         p     resid diff         p
  seed 0 (phase-5)         +0.1819   0.00010        +0.1086   0.00010
  seed 1 (independent)     +0.1835   0.00010        +0.1037   0.00010
  ratio                       1.01x                    0.96x
```

The seed-1 row is confirmed in
`qwen35/results/decomposition_seed1.json` (identical in content to
`decomposition_seedB.json` except for the TEST 6 block):
`#test1b.raw.diff = 0.18349392568239462` and
`#test1b."leading-component removed".diff = 0.10369243017205296`, both at
`p = 4.999750012499375e-05`, over `#n_traits = 40`.

The seed-0 row (+0.1819 / +0.1086) is the same 40 traits at seed 0 and **has no
file on disk** that this page could find; the whole-zoo
`results/decomposition.json` gives 0.1754293665715867 raw and
0.12160064112672682 residual over 134 traits, which is a different quantity.
Cite the verdict for the 40-trait seed-0 row.

Bipolarity likewise (PHASE3_VERDICT): seed-1 same-keyed +0.2290 against
opposite-keyed -0.0949, gap +0.3239 raw and +0.2065 residual, p at the
permutation floor, with the different-factor gap at -0.0541 - absent, as it is at
seed 0. Confirmed:
`decomposition_seed1.json#test2.raw = {same_polarity 0.229009379848836,
opposite_polarity -0.09489210220813149, gap 0.32390148205696745, n_same 130,
n_opp 150}` and
`#test2."leading-component removed".gap = 0.20650021328045762`.

The seed-1 arm's own spectrum is thinner than the zoo's, as 40 traits should be:
`decomposition_seed1.json#spectrum.participation_ratio = 16.667783498348978`,
`#spectrum.top5_share = 0.44063526198714653`.

PHASE3_VERDICT's summary: "**The same traits whose vectors correlate at 0.017
across these two runs reproduce the factor structure to within 1% and 4%.** The
coordinates are noise. The structure is not." It also flags the test's status
honestly: it was **not** preregistered, and a reader should weigh it accordingly;
what protects it is that the statistic was fixed in `decompose.py` before any
seed-1 data existed, the adapters are independent, and it is a replication rather
than a reanalysis, with a matched 40-trait control ruling out trait count.

## Procrustes

The task brief asks for a Procrustes result here. The only Procrustes fit found
on disk for the zoo is in **activation space**, not across weight-space seeds:
`analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2`, quoted by
the blog page as "a Procrustes fit of the 134x5 score matrices explains ...% of
the variance against a shuffle null of 2%", and reported by
`.garden/journal/2026-09-05.md` as "Procrustes R^2 0.535 on PC scores". That
belongs to [[actspace-overview]]. No weight-space cross-seed Procrustes fit
exists in this repo; the cross-seed arrangement result is carried entirely by the
Gram correlation and the replicated decomposition above. Recorded as a gap.

## What this licenses

`PHASE3_VERDICT.md`, "What now stands":

> **Upheld.** Trait-training deltas organise into five bipolar axes aligned with
> the Big Five: same-keyed traits within a factor align, opposite-keyed traits
> anti-align, and the effect is factor-specific rather than one global valence
> direction. Not a data artefact, not a batch effect, not training strength, not
> reproduced by either null, and replicated under an independent initialisation.
>
> **Newly required by the above.** The geometry is meaningful only *relative to a
> fixed initialisation*. Anything downstream that compares adapters must hold the
> seed fixed or be re-derived per seed.

The blog page's closing section states the same shape: "Its coordinates are local
to one LoRA initialisation ... Its *arrangement* is not local - it reappears
under a second seed, in the stage-two adapters, and when the constitutions are
used as prompts rather than as training data, so it is not an artefact of the
optimiser or of the initialisation."

## The frame-free version of the same claim

The argument on this page is that the *arrangement* replicates while the
*coordinates* do not, and it makes that case with a correlation between two
Gram matrices. A more direct version was measured on 2026-09-09: compare the
column spaces of two adapters instead of their coordinates. That statistic never
refers to a basis, and it identifies all 40 seed-1 adapters among the 134 seed-0
candidates at mean rank 1.0, the same as the Frobenius cross-Gram, with
same-trait cross-seed overlap 0.5846385056208819 against 0.12880060417597922 for
different traits at k = 8 (`qwen35/analysis/column_space.json#stage1`). It also
shows *which* half of the LoRA factorisation is frame-locked: the row space sits
at the `r/d` null across seeds, the column space does not. See
[[column-space-structure]].

Related: [[column-space-structure]], [[seed-floor]], [[stage-two-second-seed]],
[[null-controls]], [[polarity-and-bipolarity]], [[actspace-overview]],
[[superseded-geometry-claims]].
