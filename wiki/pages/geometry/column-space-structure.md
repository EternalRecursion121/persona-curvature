---
title: Column-space structure
summary: Two adapters for the same trait from different LoRA seeds are near-orthogonal as vectors but share output-side subspace - their top left singular directions agree at |cos| 0.77 and 46 percent of one adapter's delta energy lies in the other's column space, against 14 percent for different traits and 1.8 percent for random subspaces; the row space sits exactly at the r/d null, which is why the Frobenius cosine reads 0.018.
status: current
sources:
  - qwen35/analysis/column_space.json
  - qwen35/column_space_on_modal.py
  - qwen35/analyse_column_space.py
  - qwen35/results/column_space_stage1.npz
  - qwen35/results/column_space_stage2.npz
  - qwen35/results/column_space_crossstage.npz
  - qwen35/results/column_space_gram_stage1.npz
  - qwen35/analysis/crossseed_arms.json
  - qwen35/results/gram_sweep.npz
  - qwen35/phase10_runs/colspace.log
  - qwen35/analysis/column_space_sorh.json#vs_134_stage_one_adapters
  - qwen35/analysis/column_space_sorh.json#vs_generic_and_register
  - qwen35/analysis/column_space_sorh.json#checks
last_verified: 2026-09-16
tags: [geometry, seed, cross-seed, stage-two, nulls]
---


> Terms (Tucker congruence, oblimin, factor chart, chart cosine, seed floor, column space, twin score, Fisher norm) are defined in the [[glossary]].
# Column-space structure

## The question

[[seed-floor|The seed floor]] records that the same trait trained twice from
different random LoRA initialisations comes out at Frobenius cosine
**+0.01806099632415457**
(`qwen35/analysis/crossseed_arms.json#[1].same[1]`), and explains it: a LoRA
writes `dW = s * B @ A` with `A` a random draw that barely moves in training, so
two seeds write into two independent random rank-64 slices of a
2560-dimensional input space and any inner product between them is attenuated by
`r/d = 64/2560 = 0.025`.

That account has a consequence nobody had tested. The attenuation is a fact
about the **row** space, which `A` fixes. `B` is a different object: it starts at
zero and is accumulated from the output-side error vectors `dL/dy`, summed over
tokens and weighted by `A x`. Those error vectors are set by the data and by the
base model, not by the initialisation. If the per-module error signal is
effectively low rank, two same-trait adapters from different seeds should share
**column** space - the span of `B` in output space - even while their deltas read
as orthogonal.

Samuel asked on 2026-09-09 whether they do. They do.

## How it was measured

`qwen35/column_space_on_modal.py` never forms a `d_out x d_in` matrix. For each
adapter and module it takes the thin factorisation

```
A (r, d_in), B (d_out, r), dW = s B A
QR:   A^T = Q_A R                    Q_A (d_in, r) orthonormal
so    dW = s B R^T Q_A^T
SVD:  s B R^T = U S V^T              a d_out x r problem
hence dW = U S (Q_A V)^T
```

`U` is an orthonormal basis of the column space ordered by singular value and
`Q_A V` the matching row-space basis, both exact. Two statistics per pair
(i, j) and truncation k:

- **col_unw_k** = `||U_i[:, :k]^T U_j[:, :k]||_F^2 / k`, the mean squared cosine
  of the principal angles between the two top-k column spaces.
- **col_wtd_k** = the same weighted by adapter i's `sigma^2` over its first k
  directions. This has a direct reading: it is the fraction of adapter i's
  top-k delta energy lying inside adapter j's top-k column space, and at k = 64
  it is the fraction of the *whole* delta. It is not symmetric; a class mean
  over a square matrix averages both orientations. (Verified against the brute
  force on synthetic factors before the run, including the energy reading.)

Every number below is quoted verbatim from `qwen35/analysis/column_space.json`
at the key named; where the running prose gives a shorter form for readability,
the full value is in the table above it or under the key it cites.

Both are averaged over modules. `row_unw_k` and `row_wtd_k` are the same
statistics on `Q_A V`, computed on a 31-module stride subset as the contrast.
Definitions are restated in `analysis/column_space.json#definitions`.

**A caveat about averaging over modules.** The base model has 248 LoRA modules,
and 48 of them are the linear-attention `in_proj_a` / `in_proj_b`, whose output
is only 32-dimensional (`analysis/column_space.json#stage1.d_outs`). For those
the rank is 32, not 64, the k = 8 null is 8/32 = 0.25, and every adapter's column
space is a large fraction of the entire output space. A straight average over all
248 modules is dominated by them. **Every headline number below is the mean over
the 200 modules that can carry a rank-64 column space**
(`...by_module_class.<class>.<k>.<view>.rank64_modules.mean`); the all-248
version is in `...classes.<class>.<key>.mean` and is quoted where it differs.
k = 64 is only defined on those 200 modules in any case
(`stage1.n_modules_per_k_column.k64 = 200`).

## Stage one: three pair classes

40 traits were retrained at a second LoRA seed with the objective matched
(the set described on [[seed-floor]]). All 174 adapters - 134 seed-0 and 40
seed-1 - went through one pass, so the three classes come out of one matrix.
Pair counts `stage1.pair_counts`: 80 same-trait cross-seed (40 traits, both
orientations), 10,640 different-trait cross-seed, 17,822 different-trait within
seed 0.

Mean squared cosine of principal angles, 200 rank-64 modules
(`analysis/column_space.json#stage1.by_module_class`):

| k | view | same trait, diff seed | diff trait, diff seed | diff trait, same seed | null k/d_out |
|---|---|---|---|---|---|
| 1 | col_unw | **0.6307023078841583** | 0.06898996183549069 | 0.07360911221285221 | 0.00028667534722222224 |
| 8 | col_unw | **0.4248247020579875** | 0.11012252366265511 | 0.11713569333808653 | 0.002293402777777778 |
| 8 | col_wtd | **0.5846385056208819** | 0.12880060417597922 | 0.13295928198239396 | 0.002293402777777778 |
| 64 | col_unw | **0.14174031494371594** | 0.058461099659940896 | 0.06090388840495398 | 0.018347222222222223 |
| 64 | col_wtd | **0.4630531697561965** | 0.1396023334154358 | 0.1419439188002003 | 0.018347222222222223 |

(At k = 1 the weighted and unweighted statistics coincide by construction.)

The null is `k / d_out`, the expectation for two independent uniformly random
k-dimensional subspaces of the module's output space, computed per module and
then averaged. An empirical null of 200 random subspace pairs per module on the
31-module subset agrees with it: 0.008511489616003672 measured against
0.008283980174731184 analytic at k = 1, 0.06612080220952668 against
0.06627184139784947 at k = 8, and 0.01894207039392432 against
0.01893115942028985 at k = 64
(`stage1.null.k*.empirical_col_mean` and `.analytic_col_mean_on_the_same_modules`,
those two being over the same 31 and 23 modules respectively).

Read the top row plainly. **The single strongest output direction of a trait's
adapter is recovered across an independent initialisation at mean squared cosine
0.63** - and separately, the mean `|cos|` between the two top left singular
vectors is **0.7701377220108219** against 0.21012838847727913 for different
traits (`stage1.top1_left_vector_same_trait_cross_seed.mean_abs_cos`,
`...diff_trait...`). The full-delta version: **46.3 percent of one adapter's
delta energy lies inside the other seed's 64-dimensional column space**, against
14.0 percent for a different trait and 1.8 percent for a random subspace.

The weighted numbers exceed the unweighted ones everywhere (0.463 against 0.142
at k = 64), which says the sharing is concentrated in the high-singular-value
directions: the top of the spectrum is shared and the tail is not. The spectrum
itself is not sharply low rank - the top direction holds 0.10318456418060519 of
the sum of singular values, the top 8 hold 0.3614515265945672 and the top 16
hold 0.5305926812823089 (`stage1.spectrum`) - so this is not a rank-1 delta
wearing a disguise.

## The row space, which is the contrast

The same pairs, the same code path, on `Q_A V` instead of `U`
(`stage1.classes.*.row_unw_k*.mean`, 31-module subset):

| k | same trait, diff seed | diff trait, diff seed | diff trait, same seed | null k/d_in |
|---|---|---|---|---|
| 1 | 0.0009649788564007243 | 0.0008334070068787748 | 0.8715021433190868 | 0.00034442204301075277 |
| 8 | 0.003245407972914812 | 0.0031315399046583435 | 0.624573124005435 | 0.002755376344086022 |
| 64 | 0.0211080335981577 | 0.021104007943688403 | 0.9997590233553881 | 0.021014492753623194 |

At k = 64 the cross-seed row-space overlap is **0.0211080335981577** against a null of
**0.021014492753623194** - two independent random rank-64 subspaces, to within half a percent,
exactly as `probe_invariant.py` measured it a different way ("measured row-space
overlap 0.0249, which is 64/2560", quoted on [[seed-floor]]; that figure is the
r/d for the 2560-wide modules alone, this one averages over modules of several
widths). Within a seed the row space is 0.99976 - the same `A`, moved by the
1.5 percent drift.

So the two halves of the factorisation behave completely differently across a
seed boundary: **the row space is the initialisation and carries nothing; the
column space is the training signal and carries the trait.** The Frobenius inner
product multiplies the two, which is why it reads 0.018.

One small residual, recorded rather than explained. The cross-seed row overlap
is not exactly at the null, and same-trait is consistently a shade above
different-trait. In absolute terms the excess is at the fourth decimal place; as
a ratio it is not small at k = 1, where different-trait cross-seed sits at
0.0008334070068787748 against a null of 0.00034442204301075277 - 2.4 times
chance - and same-trait at 0.0009649788564007243, 2.8 times chance. By k = 64 it
has shrunk to half a percent above null (0.0211080335981577 against
0.021014492753623194). Ranking on row-space overlap alone identifies 19 of 40
traits at k = 64, mean rank 3.9 against a chance 67.5
(`stage1.identification_40_seed1_queries_vs_134_seed0.row_unw_k64`). The
1.5 percent that `A` moves in training is apparently not random. The effect is
at the fourth decimal place and nothing here rests on it.

## A generic component sits underneath

Different-trait pairs are not at the null either. At k = 8 they overlap at
0.1101 (cross-seed) and 0.1171 (within seed) against a null of 0.0023 - fifty
times chance, and near-identical whether the two adapters share an
initialisation or not. There is an output-side subspace common to every adapter
in the zoo, and it is a property of the recipe, not of the seed. Same-trait
sharing sits on top of it. The trait-specific part is the gap:
`stage1.by_module_class.same_minus_diff` gives **+0.5617123460486676** at k = 1,
**+0.4558379014449027** at k = 8 weighted and **+0.32345083634076066** at
k = 64 weighted.

This is the same kind of object as the stage-two shared direction on
[[stage-two-shared-direction]], measured in a different space; whether they are
the same thing is not established here.

## Identification

The Frobenius cross-Gram identifies all 40 seed-1 adapters among the 134 seed-0
candidates, mean rank 1.0 against a chance 67.5, with perfect separation:
min same-trait 0.015933681838395445 above max off-trait 0.015842137704656464,
gap +9.154413373898065e-05
(`analysis/crossseed_arms.json#[1]`, restated in
`column_space.json#stage1.identification_40_seed1_queries_vs_134_seed0.frobenius_reference`).

Column-space overlap also gives **40 of 40, mean rank 1.0, worst rank 1** - at
k = 8 and k = 64, weighted, unweighted and symmetrised
(`stage1.identification_40_seed1_queries_vs_134_seed0`). It is a frame-free
statistic: it never compares coordinates, only subspaces.

It does **not** separate globally. At k = 8 weighted the lowest same-trait score
is 0.481076 while the highest off-trait score is 0.565952, a gap of **-0.084876**
(`...col_wtd_k8.margins`). Every query's own trait is its top candidate, but the
score ranges overlap across queries, because different traits sit at different
baselines. The Frobenius statistic separates globally; this one does not. Both
identify.

Row-space overlap identifies 6 of 40 at k = 8 and 19 of 40 at k = 64, as above.

## Which modules

Per-module gap, same-trait cross-seed minus different-trait cross-seed at k = 8
weighted (`stage1.per_module_profile`; the values are rounded to four decimals
from `...by_projection.<proj>.gap_mean` and the module entries from
`...top10_modules_by_gap`, which carry them in full):

- By projection, largest first: `o_proj` 0.5607, `v_proj` 0.5553, `out_proj`
  0.5076, `up_proj` 0.4854, `down_proj` 0.4675, `gate_proj` 0.4366,
  `in_proj_qkv` 0.4165, `in_proj_z` 0.4086, `q_proj` 0.3737, `k_proj` 0.3499,
  `in_proj_a` 0.2751, `in_proj_b` 0.2507.
- By layer quartile: 0.3737 (layers 0-7), 0.4062 (8-15), 0.4511 (16-23),
  0.4430 (24-31). Later layers share more.
- The ten largest gaps are, in order:
  `model.layers.27.self_attn.o_proj` 0.6323, `19.self_attn.o_proj` 0.6317,
  `19.self_attn.v_proj` 0.6253, `27.self_attn.v_proj` 0.6051,
  `26.mlp.down_proj` 0.6050, `23.self_attn.v_proj` 0.6007,
  `20.linear_attn.out_proj` 0.5973, `31.self_attn.o_proj` 0.5970,
  `23.self_attn.o_proj` 0.5920, `15.self_attn.v_proj` 0.5916 - four `o_proj`,
  four `v_proj`, one `out_proj` and one `down_proj`, and every one of them in
  layers 15 and above.
- The ten smallest are all `linear_attn.in_proj_a` / `in_proj_b` in layers 4 to
  18 (`6.linear_attn.in_proj_b` 0.2031 smallest), the 32-wide modules whose
  different-trait baseline is high because a 64-dimensional truncation of a
  32-dimensional output space is most of that space.

So the ranking is `o_proj` > `v_proj` > `out_proj` > `up_proj` > `down_proj` >
`gate_proj`, concentrated in the upper half of the stack. Attention output and
value projections lead. Why those and not the MLP is not tested here.

## Stage two, and across the two stages

**Stage two, seed 0 against seed 1** (15 traits retrained; 134 + 15 adapters in
one pass; `column_space.json#stage2`). Same-trait cross-seed column overlap
0.5090274347342395 at k = 1, 0.37634934999793773 at k = 8 weighted and
0.29341662536064783 at k = 64 weighted, against different-trait cross-seed
0.11217186403961742, 0.09162130449467455 and 0.09196797019499994. Identification
is 15 of 15 at every k tested (`stage2.identification_15_seed1_queries_vs_134_seed0`).
The stage-two column space is shared the same way stage one's is, a little more
weakly.

**Stage one against stage two, same trait, same seed** (134 pairs;
`column_space.json#crossstage`). This is the harder comparison: different data,
different objective, a different LoRA-A draw, and a base model that already
carries the stage-one adapter. In weight coordinates the two stages are
orthogonal - same-trait cosine +0.0002
(`analysis/actspace_stage2_geometry.json`, reported on
[[stage-two-exploration]]). In column space they are not, but the margin is
small: 0.0042984680553053755 against 0.0014620439325716405 at k = 1,
0.012002238260335358 against 0.007046341091320312 at k = 8 weighted, and
0.04397209714109481 against 0.036593612760170896 at k = 64 weighted. Ranking the
134 stage-two adapters for each stage-one adapter gives **80 of 134 top-1, mean
rank 4.104477611940299** at k = 64 weighted and 55 of 134, mean rank 8.71 at
k = 8, against a chance mean rank of 67.5
(`crossstage.identification_134_stage1_queries_vs_134_stage2`).

So a trait's output-side subspace survives a change of seed almost intact and a
change of stage only partly. Note also that cross-stage overlaps are *below* the
within-stage different-trait level (0.0070 against 0.1288 at k = 8): the generic
component is per-stage, not shared between stages.

## The top left singular direction across the zoo

Mean over 248 modules of `|cos|` between the top left singular vectors, then
over all within-seed pairs (`*.top1_left_vector_sharing_within_134_seed0`):

- stage one, 134 seed-0 adapters: **0.2147402215016489** (sd 0.121)
- stage two, 134 seed-0 adapters: **0.3352565200850802** (sd 0.046), signed mean
  only 0.0744778725406937

The stage-two adapters agree with each other on their leading output direction
substantially more than the stage-one adapters do, which is what the shared
introspection register on [[stage-two-structure]] predicts. The signed mean being
near zero while `|cos|` is 0.34 says the shared direction is an *axis*: adapters
sit on it with either sign.

## Does the column space carry the factor arrangement?

A column-space Gram was built over the 134 stage-one seed-0 adapters -
symmetrised `col_wtd` at k = 8, mean over 248 modules, saved as
`qwen35/results/column_space_gram_stage1.npz` - and compared with the exact Gram
`results/gram_sweep.npz` (`column_space.json#stage1.column_space_gram`).

- Against the exact Gram's **signed** cosines, off-diagonal Pearson
  **0.6037388223216157** (Spearman 0.5186245281792069).
- Against **|cos|**, Pearson **0.9425370103437601**.

That pair of numbers is the result: the column-space Gram is very nearly a
monotone image of the *magnitude* of the weight-space cosine, and carries none of
its sign. A subspace overlap cannot: two adapters pointing in exactly opposite
directions share a column space perfectly.

Splitting by keying (100 of the 134 traits carry a factor and a polarity in
`qwen35/traits_primary.json`; `...column_space_gram.keying_split`):

| class | n | column-space Gram | exact Gram cosine | exact Gram \|cosine\| |
|---|---|---|---|---|
| same factor, same keying | 932 | 0.2441358027751059 | 0.24472701836565558 | 0.24960288876801623 |
| same factor, opposite keying | 968 | 0.15904015225348364 | -0.08134207180636474 | 0.1341461271807166 |
| different factor | 8000 | 0.16329709248713262 | 0.06800804835643327 | 0.13399652356059974 |

Same-factor **opposite**-keyed pairs sit at 0.1590, marginally *below* the
different-factor 0.1633. So the column space carries the same-keyed clustering
and not the bipolar axis. This is not a failure of the column-space statistic:
the exact Gram's own `|cosine|` behaves identically (0.1341 against 0.1340), so
the axis in this zoo lives in the sign and is invisible to any sign-blind
measure. [[polarity-and-bipolarity]] is the page for what the sign carries.

Unsigned within-factor minus between-factor separation on the column-space Gram:
within 0.20078180819356362, between 0.16329709248713262, **separation
+0.037484715706431, permutation p = 0.0001999600079984003** over 5,000 label
shuffles. The exact Gram on the same 100 traits gives unsigned within
0.07860339768854206 and between 0.06800804835643327.

The signed statistic from `decompose.py` TEST 1B was computed as asked and is
reported at `...signed_test1b_style` (within 0.038728263581592925, between
-0.001461929253334904, separation +0.04019019283492783, p = 0.0003999200159968006),
but it is not meaningful here and the JSON says so: column-space overlap is
non-negative, so multiplying by the polarity product turns every same-factor
opposite-keyed pair into evidence *against* its own factor. The unsigned
statistic is the one to read.

## What this establishes, and what it does not

**Established.** Two adapters trained for the same trait from independent LoRA
initialisations are near-orthogonal as vectors and strongly aligned as subspaces.
The row space of a LoRA delta is the initialisation - it sits at the `r/d` null
across seeds and at 1.0 within a seed - and the column space is the training
signal. The seed floor of +0.018 is a fact about the parameterisation, and this
is the direct measurement of what was underneath it, in a frame-free statistic
that identifies 40 of 40 traits without ever comparing coordinates. The zoo also
has a generic output-side subspace shared by every adapter, fifty times chance
and independent of seed, on top of which the trait-specific sharing sits.

**Not established.** (1) That the shared column space is *the* trait direction in
any functional sense - nothing here is behavioural, and no steering was run
through a column-space basis. (2) That the generic component is the same object
as the stage-two shared register direction; they are measured in different spaces
and only their shapes rhyme. (3) Anything about the bipolar axes, which are a
property of the sign and are invisible to a subspace statistic by construction.
(4) The cross-stage result (80 of 134) is above chance but far from the
within-stage 40 of 40, and no attempt was made to separate "the trait's output
subspace partly survives the second stage" from "both stages inherit structure
from the same base model".

**Tested since.** [[rank-sweep]] retrained 15 of these traits at rank 1, 4 and
16 with the LoRA-A frame nested inside the zoo's rank-64 one, which is the
question this page's spectrum result raises: the top of the spectrum carries the
shared part, so how much rank does a trait need? With the frame held fixed the
arrangement survives at rank 1 (15 x 15 cosine matrix Pearson 0.9906 against
rank 64, factor-chart coordinates Pearson 0.9967, 15 of 15 identified among the
134), and on one trait the rank-1 run recovers the output direction the rank-64
run assigned to the same input direction at cosine
0.8418963800198176 (`qwen35/analysis/rank_sweep_mechanism.json`). That is a
column-space statement measured through the Frobenius Gram rather than through
the SVD, and it is not the same comparison as this page's cross-seed top-1
figure of 0.6307023078841583.

**A caveat on the module average.** Column-space overlap depends on the module's
output width. Averaging over modules of different widths mixes statistics with
different nulls; the 48 narrow linear-attention projections were separated out
for this reason, and every headline is the 200-module figure. The intermediate
truncations k = 2, 4, 16 exist in the JSON as all-248 averages only and are not
split that way.

## Applied to the reward-hacks arms

The obvious next question - does anything *else* trained in this window have
column-space structure the zoo shares - was asked of the two School of Reward
Hacks SFT arms on the same day and answered no. See
[[reward-hacks-column-space]]: the hack arm's top-8 output subspace overlaps a
zoo trait's at **0.01864560989879112** against the **0.1322819018644223** two
unrelated zoo traits reach here, and against the generic subspace of the section
above it reads **0.027653501381864773** where a held-out seed-1 trait adapter
reads **0.3784547236738726**
(`analysis/column_space_sorh.json#vs_134_stage_one_adapters` and
`#vs_generic_and_register.G1_stack.k8`).

That page also re-derives three numbers from this one on an independent code
path, which is a check on both: the different-trait same-seed band comes out
0.1322819018644223 against the 0.13295928198239396 below, the within-seed
row-space overlap comes out 0.9997590229470755 against 0.9997590233553881, and
the Frobenius Gram it builds correlates with `results/gram_sweep.npz` at
off-diagonal Pearson 0.9999984776481379.

## Run

`qwen35/zoo-colspace.service`, three CPU-only Modal jobs run in sequence, app
`pc-qwen35-colspace`, log `qwen35/phase10_runs/colspace.log`. Function time 727 s
(stage one, 174 adapters), 751 s (stage two, 149 adapters) and 1057 s (across
stages, 134 x 134 cross block), plus about five minutes of probes and one
dimension listing: roughly 0.83 container-hours on 8 CPUs and 32 GiB. No GPU.
`zoo40_meter.sh` prices every container at the A100 rate of $2.10 per hour
whatever the app is named, so its budget was raised by exactly the $5 authorised
for this run, from $2418.00 to $2423.00, and this run drew about $1.7 of it;
the meter read $2394.70 when it exited at 20:57 UTC.
