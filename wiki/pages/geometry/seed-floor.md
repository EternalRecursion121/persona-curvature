---
title: The seed floor
summary: The same trait trained twice at different LoRA initialisations agrees at cosine +0.0181, which is not absence of signal but the r/d = 0.025 overlap of two random rank-64 subspaces; 40 of 40 traits still find themselves, and in an activation-weighted metric the same pairs read +0.6643.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/PREREGISTRATION_phase3.md
  - qwen35/analysis/crossseed_arms.json
  - qwen35/analyse_crossseed.py
  - qwen35/results/cross_gram_full_root_x_data_null_seedpaired_s40.npz
  - qwen35/results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz
  - qwen35/results/cross_gram_seedpaired_provenance.json
  - qwen35/probe_invariant.py
  - qwen35/make_nulls.py
  - .garden/journal/2026-09-03.md
  - qwen35/analysis/act_gram.json#verdict
last_verified: 2026-09-11
tags: [geometry, seed, nulls]
---

# The seed floor

## What is held constant

The seed-paired arm is 40 traits retrained from a second LoRA initialisation.
`qwen35/make_nulls.py` builds it as a **byte-identical copy** of the real
corpus, so:

- **data**: identical bytes, same prompts in the same file order
  (`nulls_manifest.json#seed_paired_note`: "data byte-identical to source; vary
  train_qwen35.py SEED and ORDER_SEED at launch, not the data").
- **prompt order**: *not* held constant. `launch_nulls.sh` runs the arm with
  `--seed 1 --order-seed 1`. `make_nulls.py` explains why both must move:
  `SEED` seeds random/numpy/torch immediately before the LoRA adapter is
  constructed, so it fixes the A-matrix initialisation and is passed as both
  `seed=` and `data_seed=`; `ORDER_SEED` drives the rng that fixes example
  order. "ORDER_SEED alone is not enough, because it leaves the adapter
  initialisation identical and would understate the floor." Both are 0 for the
  real sweep.
- **objective**: identical in the *matched* arm only. The original arm trained at
  plain sigmoid DPO; see below.

The 40 traits are listed in
`qwen35/results/cross_gram_seedpaired_provenance.json#matched`
(active, anxious, artistic, assertive, bright, careful, cold, complex,
conscientious, disorganized, emotional, energetic, extraverted, fretful, harsh,
helpful, innovative, intellectual, introverted, kind, neat, organized, pleasant,
quiet, relaxed, simple, sympathetic, temperamental, thorough, touchy, trustful,
undependable, unenvious, unexcitable, unimaginative, unintellectual,
unsympathetic, unsystematic, untalkative, vigorous), with `#subdir_a = ""` and
`#subdir_b = "data_null_seedpaired_s40"`.

## The shared-A design, and why the floor is small

A LoRA writes `dW = s * B @ A` with `B` initialised at zero and `A` drawn at
random. Because `B` starts at zero, `A` receives almost no gradient and barely
moves: measured drift `mean ||A_i - A_0|| / ||A_0||` across the zoo is
**0.014605041334818797**
(`qwen35/analysis/align_summary.json#a_drift`; the blog page rounds it to
"1.5%", and `analyse_alignment.py` / `analyse_hole.py` both quote "the zoo's own
internal figure 0.0146" as their gate).

So every adapter in the zoo, sharing one seed, writes into the *same* random
64-dimensional slice of a 2560-dimensional input space, and they are directly
comparable. Two independent draws are not: their expected overlap is
`r / d = 64 / 2560 = 0.025`.

`qwen35/probe_invariant.py` states the measured version: "measured row-space
overlap 0.0249, which is 64/2560, exactly what two random rank-64 subspaces of a
2560-dimensional space give."

## The numbers

`qwen35/analysis/crossseed_arms.json` is a two-element list, computed by
`qwen35/analyse_crossseed.py` over the 134 x 40 cross-Gram blocks. Cosines are
`X / outer(norms_a, norms_b)` - exact, no sketch.

| statistic | `#[0]` original (objective mismatched) | `#[1]` matched objective |
|---|---|---|
| npz | `cross_gram_full_root_x_data_null_seedpaired_s40.npz` | `..._s40_matched.npz` |
| same trait (n, mean, sd) | 40, +0.016587159178393374, 0.0006956614604439757 | 40, +0.01806099632415457, 0.0010534320757327785 |
| same factor, same keying (n, mean) | 368, +0.005933561872613479 | 368, +0.006467775392607999 |
| same factor, opposite keying (n, mean) | 392, -0.0027091844712231567 | 392, -0.0021325710689666204 |
| different factor (n, mean) | 4560, +0.0013305288725605703 | 4560, +0.0018307002389548663 |
| min same / max off | 0.015093675329141625 / 0.014900677296882946 | 0.015933681838395445 / 0.015842137704656464 |
| separation | +0.00019299803225867815 | +9.154413373898065e-05 |
| top-1 of 134 | 40 / 40, mean rank 1.0 (chance 67.5) | 40 / 40, mean rank 1.0 |
| signed bipolarity | +0.008642746343836636 | +0.00860034646157462 |
| slope on within-run cosine | 0.026350265588597366 | 0.026542111376493285 |
| intercept | -0.00037996208388643563 | +0.00011559223011991917 |
| Pearson | 0.9947156237185968 | 0.9966259140629706 |
| Spearman | 0.9959037342367216 | 0.9970390729325707 |
| n pairs | 5320 | 5320 |

The current figure the blog page quotes is the matched one:
"Train the same trait twice from two different random LoRA initialisations and
the two adapters come out **89 degrees apart** (cosine 0.018)."
(`qwen35/analysis/pc_loadings.json#seed.same_trait_deg = 89.04958220635345` is
the 89-degree figure, computed on the *original* 0.01658715905967511 - the page's
data file still carries the pre-matched cosine.)

## The attenuation slope is predicted, not fitted

`PHASE3_VERDICT.md`, postscript to the 2026-08-22 addendum:

> Measured slope of cross-seed cosine on within-run cosine: **0.0264**. The LoRA
> rank fraction is r/d = 64/2560 = **0.0250** (Qwen3.5-4B `hidden_size` 2560,
> `LORA_R` 64). Agreement to 6%. The subspace-overlap account predicts that
> number; no account in which the 0.0167 is absence-of-signal predicts anything
> at all. This is the strongest single piece of evidence in the addendum, and it
> was free.

Regressing cross-seed cosine on within-run cosine over the same 5,320 pairs is
what produces the slope; the matched arm gives 0.0265 (`#[1].slope`) at
Pearson 0.9966. So the whole cross-seed block is the within-run block times about
1/38.

## The objective in isolation

`PHASE3_VERDICT.md`, addendum 2026-09-03. The npz
`results/cross_gram_full_data_null_seedpaired_s40_x_data_null_seedpaired_s40_matched.npz`
holds the same 40 traits at the same seed on the same data with only the
objective differing:

> same-trait cosine **+0.954** (range 0.908-0.981), different-trait +0.043, 40/40
> top-1. So the seed change moves an adapter from 0.954 to 0.018 and the
> objective change moves it from 1.0 to 0.954.

`.garden/journal/2026-09-03.md` states it in one line: "Objective alone, same
seed and data: +0.954. The seed does everything." Those three numbers exist only
in the verdict prose and the journal; `crossseed_arms.json` does not carry that
arm.

The same addendum verifies the treatment travelled: first-step loss 0.89-0.94
(ln 2 plus the 0.1 x SFT term, as in the 134) against exactly 0.6931 in the
original arm, and `kl_term = sq_approx_kl x 0.001` on every logged step.
`analyse_crossseed.py` computes both arms with one code path and "reproduces the
2026-08-22 numbers to the digit" before being trusted on new ones.

## The preregistered bar, and why it could not be cleared

`PREREGISTRATION_phase3.md`, Arm 3:

> **Pre-registered threshold.** The reference is the median cosine of
> same-factor, same-keying, *different*-trait pairs: **+0.2447**. ... If
> seed-paired lands below 0.35 I will report the trait-level result as
> unsupported even though it is my own headline.

Measured: median +0.0167, min +0.0151, max +0.0178 over 40 traits
(`results/compare_nulls_output.txt`). About fourteen times less aligned with
itself than two different traits sharing a factor and a keying. The trait-level
framing was withdrawn on 2026-08-23.

**Then reversed on 2026-08-22 22:40 UTC** (the addendum is dated earlier in the
day than the verdict body it reverses; the verdict body was written 2026-08-23).
PHASE3_VERDICT, "THE TRAIT-LEVEL WITHDRAWAL IS REVERSED":

> The floor compared a **cross-basis** cosine (0.0167) against a bar computed
> **within a single basis** (+0.2447 ...). Those two numbers are not
> commensurable. Two independent rank-64 inits span near-orthogonal slices of a
> 2560-dim space, so a trait direction that reproduced *perfectly* modulo the
> subspace would still have scored ~0.02. **The test could not have passed under
> any hypothesis.**

The stated general lesson: "A preregistered threshold needs its own check:
*compute what the statistic would read under the maximal version of the
hypothesis, before committing to the threshold.*"

A units problem sits inside the bar itself, recorded here rather than resolved.
The preregistration calls +0.2447 a **median**. The value +0.2447 is what
`qwen35/results/decomposition.json#test2.raw.same_polarity` reports
(0.24472701836565558) and that key is a **mean** over 932 pairs.
`results/compare_nulls_output.txt` separately reports the **median** of the same
class as +0.2467 (n = 932), and +0.2373 restricted to the 40 seed-twinned traits
(n = 130). Both are recorded; neither is silently preferred.

## Three structural facts the addendum rests on

From the 2026-08-22 addendum, on the original arm:

1. **Perfect separation.** min same-trait 0.01509 > max different-trait 0.01490
   across all 5,320 cross-trait comparisons; no overlap at all. (Matched arm:
   0.01593 > 0.01584, `#[1].sep = 9.15e-05`.)
2. **40/40 top-1.** Every seed-1 trait is its own nearest neighbour among all 134
   seed-0 adapters; mean rank 1.0 against a chance mean of 67.5.
3. **The whole geometry transfers, uniformly attenuated** - the slope result
   above.

Bipolarity survives the seed boundary: same-keying minus opposite-keying
= +0.00866 signed, i.e. the +0.1216 within-run effect at the same 1/38 scale
(`#[0].signed_bipolarity = 0.008642746343836636`).

## What is reinstated and what is not

`PHASE3_VERDICT.md`:

> **Reinstated.** A trait has a reproducible direction in weight space. It
> survives independent initialisation; the cosine is small only because the LoRA
> parameterisation confines each run to a different random rank-64 subspace, and
> 0.0264 is the overlap of two such subspaces, not a measure of signal loss.
>
> **Not established.** Same-trait pairs train on a byte-identical corpus, so what
> reproduces is that trait's *training signal*. This addendum does not upgrade
> the deflationary reading - it removes a spurious refutation, it does not add
> evidence that the axes are model-intrinsic personality rather than corpus
> structure.
>
> **Not usable as engineering.** 1.7% alignment is statistically unambiguous and
> practically worthless as a transferable vector. Anything that needs a usable
> direction still works inside one init, or freezes A globally (LoRA-FA,
> arXiv:2308.03303) so all adapters share one subspace by construction.

The 2026-08-24 addendum then re-withdrew part of this: "The cross-seed
self-cosine (+0.0167) is a seed-plus-objective floor, not a pure seed floor. The
trait-level withdrawal STANDS (the preregistration bound it to this measurement
as run), but the mechanistic reading 'the init frame alone carries the magnitude'
needs the 40 seed-B adapters retrained at the matched objective." The 2026-09-03
addendum discharges that caveat - "the page now quotes 0.0181" - without
restating the reinstatement in words; the blog page, which is current truth,
treats trait identity as surviving the seed change. Four dated positions on one
claim; they are laid out in [[superseded-geometry-claims]].

## The cross-init probe

`qwen35/probe_invariant.py` asks whether a weight probe can identify a trait
*across* initialisations using representations that do not depend on the random
basis. Three feature spaces, cheapest first: the 248 per-module Frobenius norm
profile (basis-free, exact in closed form); a **semantic** sketch where `P_in` is
32 real input activations harvested from the base model on a fixed prompt set and
`P_out` the top 32 left singular vectors of the base weight, so neither
projection depends on any adapter; and (not implemented there) the functional
probe `dW . h`. The comparison baseline is the random-sketch figure of 0.0166.
The script's own header notes it has since been repointed at
procedure-invariance (stage-1 DPO versus stage-2 SFT) rather than
init-invariance. Results are in `qwen35/analysis/functional_probe.json`, which
this page does not quote.

## What is underneath the floor

This page's account - the cosine is small because the two seeds write into
independent random rank-64 slices of the input space - was tested directly on
2026-09-09 by splitting each delta into its row space (the span of `A`) and its
column space (the span of `B`) and comparing the two halves across the seed
boundary. Both halves behave exactly as the account requires: the cross-seed
row-space overlap is 0.0211080335981577 against a null of 0.021014, while the
same-trait cross-seed column-space overlap is 0.4630531697561965 of the delta's
energy against 0.1396023334154358 for a different trait and 0.018347 for a
random subspace, and the two adapters' top output directions agree at
`|cos|` 0.7701377220108219
(`qwen35/analysis/column_space.json#stage1`). Column-space overlap identifies
40 of 40 traits without comparing coordinates at all. See
[[column-space-structure]].

## The floor in a second metric

The 0.0181 is a **Frobenius** cosine, and the Frobenius metric weights all 2,560
input directions of a module equally when the model's activations effectively use
about ten of them. Rebuilt as an activation-weighted inner product on 2026-09-11,
the same 40 pairs sit at **+0.6642887281525424**, 36.8 times the Frobenius figure,
and the ordering reverses: a cross-seed twin becomes far *more* similar than two
different traits sharing one initialisation, where here it is far less. The
frame-overlap null rises with it, from 0.0215 to 0.8910, so measured against what
a perfectly reproduced update would score the two metrics agree and the floor is
**not** a metric artefact by that run's preregistered test
(`qwen35/analysis/act_gram.json#verdict`). See
[[activation-weighted-gram]]; that page also carries the exact reproduction of
this page's row through its own code path.

Related: [[column-space-structure]], [[cross-seed-geometry]], [[null-controls]],
[[geometry-overview]], [[stage-two-second-seed]],
[[superseded-geometry-claims]], [[zoo-training-recipe]],
[[activation-weighted-gram]].
