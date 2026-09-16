---
title: Factor analysis of the null arms
summary: The same factor analysis run on the two matched null Grams - the shuffled arm retains zero factors, the permuted arm retains eight and reproduces the real five factors exactly once its adapters are re-identified by the trait they actually trained on, and neither arm shows any Big Five congruence.
status: current
sources:
  - qwen35/analyse_fa_qwen35.py
  - qwen35/analysis/goldberg_only.json
  - qwen35/analyse_fa_nulls.py
  - qwen35/analysis/fa_nulls.json
  - qwen35/results/fa_qwen35_null_shuffled.json
  - qwen35/results/fa_qwen35_null_shuffled.md
  - qwen35/results/fa_qwen35_null_permuted.json
  - qwen35/results/fa_qwen35_null_permuted.md
  - qwen35/results/fa_qwen35.json
  - qwen35/nulls_manifest.json#permutation_dst_to_src
  - qwen35/phase10_runs/fa_nulls.log
  - qwen35/phase10_runs/fa_nulls3.log
last_verified: 2026-09-12
tags: [geometry, factor-analysis, nulls, big-five]
---

# Factor analysis of the null arms

## Why this page exists

The PCA side of the geometry has always been compared against the null arms: the
scree curve is scored against both, and eleven principal components sit above the
shuffled arm while none sits above the permuted one
(`qwen35/analysis/scree_null_matched.json#n_above_structureless = 11`,
`#n_above_null = 0`; see [[pca-and-scree]]). The factor analysis had no null arm
at all. Samuel asked on 2026-09-08 whether the factor analysis was less supported
than the PCA; this is one of the two gaps that turned up, and it is now filled.
The other is [[direction-seed-stability]], and the whole comparison is on
[[factors-versus-pca-coverage]].

## What was run

`qwen35/analyse_fa_qwen35.py`, unchanged in its mathematics, was run twice more
with `PC_GRAM_NPZ` pointed at each matched null Gram and `PC_FA_TAG` set:

- `results/gram_data_null_shuffled_p100_matched.npz` -> `results/fa_qwen35_null_shuffled.{json,md}`
- `results/gram_data_null_permuted_p100_matched.npz` -> `results/fa_qwen35_null_permuted.{json,md}`

Both are the **matched-objective** retrains of 2026-09-05 (100 traits each,
trained at the zoo's own objective), not the original arms with the objective
mismatch. See [[null-controls]] for both, and for what each destroys, in that
page's words:

- **shuffled**: "chosen and rejected are swapped on exactly `floor(n/2)` of each
  trait's pairs ... The preference direction is destroyed; nothing coherent is
  learned."
- **permuted**: "each trait *name* receives another trait's intact pair set,
  under a uniform derangement ... Each record keeps its original honest
  trait/factor/keyed fields so the permutation can be audited from the corpus."

So the shuffled arm asks whether the trait *labels* alone can manufacture
factors out of adapters that learned nothing coherent, and the permuted arm asks
whether coherent training produces factor structure even when every label is on
the wrong adapter.

`qwen35/analyse_fa_nulls.py` then collects the three arms into
`qwen35/analysis/fa_nulls.json`, which is the source for every number below.

## The result

`analysis/fa_nulls.json#arms.<arm>`:

| | real (stage one) | shuffled | permuted |
|---|---|---|---|
| variables `#n_traits` | 134 | 100 | 100 |
| retained factors `#n_factors_chosen` (Horn, unreduced, 95th pct, N = 1528) | **9** | **0** | **8** |
| centred eigenvalues above the N = 1528 null `#eigenvalues_above_null.1528.n_above_unreduced_95pct` | 9 | 0 | 8 |
| ... above the N = 150 null `#eigenvalues_above_null.150.n_above_unreduced_95pct` | 5 | 0 | 5 |
| reduced eigenvalues above the N = 1528 null `#...n_above_reduced_95pct` | 9 | 0 | 8 |
| centred eigenvalues 1-5 `#centred_eigenvalues_top12` | 16.624, 14.287, 6.477, 4.974, 3.448 | 1.222, 1.203, 1.186, 1.161, 1.150 | 13.632, 10.390, 4.945, 3.887, 2.824 |
| k=5 reduced eigenvalues 1-5 `#reduced_eigenvalues_centred_k5_top12` | 15.980, 13.646, 5.789, 4.285, 2.764 | 0.236, 0.217, 0.198, 0.175, 0.164 | 12.994, 9.752, 4.257, 3.232, 2.142 |
| oblimin SS loadings, centred_k5 `#ss_loadings_oblimin_centred_k5` | 10.76, 8.42, 7.02, 6.81, 5.80 | 0.20, 0.20, 0.20, 0.20, 0.19 | 8.83, 6.35, 5.85, 4.46, 4.18 |
| mean communality, centred_k5 `#communality_mean_centred_k5` | 0.317 | 0.010 | 0.324 |
| best Goldberg congruence per target `#best_congruence_per_big_five_target` | E 0.539, A 0.655, C 0.574, ES 0.405, I 0.682 | E 0.133, A 0.177, C 0.135, ES 0.215, I 0.134 | E 0.170, A 0.115, C 0.167, ES 0.241, I 0.123 |
| all five targets taken once `#all_five_targets_taken_once` | **True** | False (3 distinct) | False (3 distinct) |
| mean absolute congruence with the five targets `#mean_abs_congruence_with_five_targets` | 0.199 | 0.080 | 0.097 |
| targets clearing 0.85 `#n_targets_clearing_0.85` | 0 | 0 | 0 |

Full precision is in the JSON; e.g. the permuted arm's largest Big Five
congruence is `#arms.permuted.best_congruence_per_big_five_target.ES =
0.24109318438008137` against the real arm's
`#arms.real.best_congruence_per_big_five_target.I = 0.6822835176928186`.

Parallel-analysis grids, `#arms.<arm>.parallel_analysis_grid.centred`
(N, k unreduced, k SMC-reduced):

| N | real | shuffled | permuted |
|---|---|---|---|
| 150 | 5, 5 | 0, 0 | 5, 5 |
| 300 | 6, 6 | 0, 0 | 5, 5 |
| 1000 | 8, 8 | 0, 0 | 7, 7 |
| 1528 | 9, 9 | 0, 0 | 8, 8 |
| 5809 | 12, 12 | 0, 0 | 9, 9 |
| 20000 | 14, 15 | 10, 10 | 11, 11 |

The shuffled arm retains nothing anywhere except at N = 20000, the largest and
least defensible sample size on the grid, where the random-data null is small
enough that a near-identity correlation matrix crosses it.

## Do the null arms show Big Five structure? No

Neither arm does, on any of the three ways of asking.

**By congruence with the keying targets.** The real arm's five oblimin factors
each take a different Goldberg target as their best match (A 0.655, C 0.574,
ES 0.405, E 0.539, I 0.682, `#arms.real.best_big_five_target_per_factor`). In
both null arms the five factors crowd onto three targets and the best congruence
anywhere is 0.215 (shuffled, ES) and 0.241 (permuted, ES). That is at the level
of an arbitrary vector's congruence with a 20-marker keying pattern, and it is
below every best-match congruence the real arm reports, the lowest of which is
ES 0.405. The real arm does not clear the
conventional 0.85 "fair" bar either ([[factor-analysis]]) - but the gap between
0.68 and 0.24 is the whole distance between "ordered but not equivalent" and
"nothing".

**By whether there are factors to name at all.** The shuffled arm's correlation
matrix is nearly the identity: off-diagonal mean `-0.0101` with standard
deviation `0.0071` centred, against the real arm's `-0.0074` at sd `0.1672`
(`results/fa_qwen35_null_shuffled.json#correlation_matrix.centred_offdiag` and
the same key in `fa_qwen35.json`). Iterated PAF drives its communalities to
`0.0099` on average and its five reduced eigenvalues to 0.236 and below, against
the real arm's 15.98. There is no common variance to rotate. Parallel analysis
says so directly: zero factors.

**By where the labels live.** This is the arm that matters. The permuted arm has
eight retained factors, communalities as high as the real arm's (0.324 against
0.317) and an eigenvalue spectrum only a little below it - and scores at chance
against the labels. Its adapters trained on real, coherent preference pairs;
they were simply given the wrong names. That is
[[polarity-and-bipolarity]]'s and `PHASE3_VERDICT.md`'s reading in the factor
analysis rather than in the labelled tests: **coherent preference training
creates the low-dimensional structure; trait identity determines where in it each
trait lands.**

## The permuted arm's factors are the real factors, in the wrong place

`analysis/fa_nulls.json#tucker_vs_real_stage_one` computes Tucker congruence
between each null arm's centred_k5 oblimin loadings and the real stage-one
centred_k5 oblimin loadings over the 100 trait slugs they share, with the best
one-to-one matching of factors (the `tucker()` and `best_match()` of
`analyse_stage2_structure.py`, the same functions [[stage-two-structure]] uses).
Matched label to label, both arms are at noise:

| arm | matched congruences | mean absolute |
|---|---|---|
| shuffled `#tucker_vs_real_stage_one.shuffled.best_matching` | 0.317, 0.211, 0.118, 0.166, 0.045 | 0.172 |
| permuted `#tucker_vs_real_stage_one.permuted.best_matching` | 0.187, -0.269, 0.277, -0.148, 0.009 | 0.178 |

Now re-identify each permuted adapter by the trait whose preference pairs it
actually trained on - `nulls_manifest.json#permutation_dst_to_src`, the
derangement the corpus was built with - and repeat the comparison over the same
100 rows (`#tucker_vs_real_stage_one.permuted.source_relabelled`):

| real factor | best match | Tucker congruence |
|---|---|---|
| F1 Warmth | permuted F1 | -0.9976 |
| F2 Competence | permuted F2 | +0.9951 |
| F3 Timidity | permuted F3 | +0.9854 |
| F4 Arousal | permuted F4 | -0.9940 |
| F5 Imagination | permuted F5 | -0.9838 |

Mean absolute matched congruence 0.991, and the matching is the identity
permutation: factor for factor, in order, in the same order of SS loading. (Sign
is arbitrary in factor analysis; `analyse_fa_qwen35.py` orients each factor
against its own best Goldberg target, and the two arms pick opposite
orientations for three of the five.)

Two things this does and does not establish. It **does** show that the permuted
arm's factor structure is the real structure and only the labels moved, which is
what the arm was built to test, and it shows that dropping the 34 Lexicon traits
does not change the five factors. It **does not** show a fresh replication: the
permuted adapters were trained at seed 0 on the same corpus files as the real
adapters, so the adapter for the *name* `active` is a same-seed retrain of the
real `thrifty` adapter. Congruence near 1 is partly training determinism. The
informative comparison is with the row above it, where the same loadings against
the same real solution score 0.178 as soon as the labels are believed.

## The failure that had to be fixed first

`phase10_runs/fa_nulls.log` records the first attempt (transient unit
`fa-nulls2`, 2026-09-08 13:13). Two things stopped it. The size assertion in
`analyse_fa_qwen35.py:load_data` required exactly 134 traits and was widened to
allow 100. Then the shuffled arm crashed inside `solution()`:

> ValueError: zero-size array to reduction operation maximum which has no
> identity

with the stderr line `solution centred_k0 ...` above it. That is the result
itself, arriving as a crash: parallel analysis had retained **zero** factors on
the shuffled matrix, and the solution loop ran `sorted({K_PA, 5})` = `[0, 5]`,
asking principal axis factoring for a zero-column loading matrix. The patch skips
`k < 1` in that loop and falls back to `centred_k5` for the steering block, which
records `steering.chosen_k_from_parallel_analysis = 0` and a
`steering.fallback_note` saying nothing should be steered from it; `n_factors.chosen`
keeps the honest 0. Nothing in the 134-trait path changed: rerunning the default
input under `PC_FA_TAG=_check` reproduced `results/fa_qwen35.json` with
`n_factors.chosen` equal, `solutions.centred_k5.ss_loadings` and every solution's
`congruence_oblimin` identical to machine precision (max absolute difference
0.0), and a byte-identical `.md`. The check outputs were then deleted. The
successful run is `phase10_runs/fa_nulls3.log`.

## Caveats

- The null arms have **100 variables against the real arm's 134** (the Goldberg
  markers only; the 34 Lexicon traits have no null adapters). A correlation
  matrix has trace p, so the eigenvalues are not on the same scale across the
  columns of the table above, and each arm's parallel-analysis null is generated
  at its own p, which is the comparison that matters for the retained counts.
  A real 100-trait arm - the same 100 markers factored out of `gram_sweep.npz` -
  was run on 2026-09-12 and does make the eigenvalue rows comparable; see below.
- The elbow plots on this wiki now draw both arms; see [[factor-analysis]] and
  [[stage-two-structure]].
- The counts in `#eigenvalues_above_null` are computed over the 12 null
  eigenvalues stored in each json, the same way `wiki/tools/gen_scree_svg.py`
  counts them, and are reported beside the first-crossing counts
  (`#...k_unreduced_95pct_first_crossing`); on these three arms the two agree.

## The same-size real arm (added 2026-09-12)

The missing comparator now exists. [[goldberg-only-and-heldout-lexicon]] factors
the 100 Goldberg markers alone out of `results/gram_sweep.npz`, through this same
script with `PC_GRAM_NPZ` and `PC_FA_TAG=_goldberg100`, so all three arms have
p = 100 and their eigenvalues are on one scale
(`qwen35/analysis/goldberg_only.json#test1_goldberg_only_factor_analysis.same_size_null_comparison`):

| centred eigenvalue | Goldberg-100 real | shuffled | permuted |
|---|---|---|---|
| 1 | 12.72505697309637 | 1.2220951463764176 | 13.63187169433154 |
| 2 | 11.139585599037872 | 1.20267712746838 | 10.390404019455278 |
| 3 | 5.662908292594041 | 1.1857911477336163 | 4.945136854288979 |
| 4 | 3.996177186321918 | 1.1613049901755712 | 3.8874696342485437 |
| 5 | 2.9270793731877207 | 1.149952334058564 | 2.823814370501938 |
| retained k, Horn, N = 1528 | 8 | 0 | 8 |

The leading eigenvalue is 10.412492849535404 times the shuffled arm's
(`#ratio_leading_eigenvalue_goldberg100_over_shuffled`) and 0.9335 of the
permuted arm's
(`#ratio_leading_eigenvalue_goldberg100_over_permuted` = 0.9334783409374191).

So the conclusion of this page holds and sharpens. **At equal p the spectrum does
not separate real labels from permuted ones**: the real 100-trait arm and the
permuted arm have the same size, nearly the same shape and the same retained
count of 8. What separates them is only congruence with the keying - the real
arm reaches A 0.759313804675552, I 0.7559311500580892, C 0.641553608012454,
E 0.5402944425426125, ES 0.4232833308028415, against a best of 0.24109318438008137
anywhere in the permuted arm. The permuted arm was never a weak version of the
real one; it is the same structure with the names moved, which is what this page
already said and what the equal-p comparison now shows on the eigenvalues
themselves.

Related: [[factor-analysis]], [[null-controls]], [[pca-and-scree]],
[[factors-versus-pca-coverage]], [[direction-seed-stability]],
[[stage-two-structure]], [[polarity-and-bipolarity]],
[[goldberg-only-and-heldout-lexicon]].

## The same two arms at the gradient level (added 2026-09-10)

[[gradient-atoms]] runs both null corpora through an unsupervised decomposition of
the *training gradients* rather than of the trained adapters, and two facts about
the corpora themselves are recorded there and asserted in
`qwen35/build_gradatoms_inputs.py`:

- the permuted arm's files are identical, as sets of (prompt, chosen, rejected)
  triples, to `data_common` files under other trait names - a derangement of 100
  traits with **no fixed point**, and only **13 of 100** assigned names share a
  Big Five factor with their source, below the 19.2% chance rate. A gradient
  cannot see a file name, so at gradient level the permuted arm is the real arm
  relabelled, and trait purity is invariant under that relabelling.
- the shuffled arm holds the same prompts with chosen and rejected exchanged on
  half of them: **22,300 intact and 22,200 swapped** over the 100 files
  (`qwen35/phase10_runs/gradatoms_labels.json#shuffled_counts`).

At the adapter level this page's shuffled arm retains zero factors. At the
gradient level it does **not** vanish - swapping a pair negates its gradient and
sparse coding is equivariant to that - but it degrades: maximum coherence
0.15456007421016693 against the real arm's 0.17964830994606018 on comparable
atoms, and **14.5%** of shuffled atoms have the same majority trait on both poles
against **0.0%** in the real arm
(`qwen35/analysis/gradient_atoms.json#poles`), which is the direct signature of a
trait's cluster being split into two antipodal halves.

The permuted arm goes the other way from this page's adapter-level result. At
adapter level it retains eight factors and reproduces the real five at congruence
0.98 to 1.00 once its adapters are re-identified by the data they trained on. At
gradient level it is **uninformative**: trait purity is invariant under a
bijection, so purity under the assigned name and under the source name are the
same number by construction (both 0.1887, z 38.9943), and the factor-level
version - which the derangement does scramble, only 13 of 100 assigned names
sharing a factor with their source - comes out null, **0.3886 (z 11.6897)** under
the assigned name against **0.3944 (z 10.9046)** under the source
(`qwen35/analysis/gradient_atoms.json#atoms.configs`). The reason is that a
majority-trait concentration of three or four documents in twenty is too weak to
move a five-level purity either way. The test that has power on the gradients is
the pole statistic above.
