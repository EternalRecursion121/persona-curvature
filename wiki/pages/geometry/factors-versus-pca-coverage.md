---
title: What has been done to the PCs but not to the factors
summary: A row-by-row audit of every analysis the wiki records for the principal components against whether the same exists for the five factor-analytic factors and for the five Big Five keying axes; two gaps were filled on 2026-09-08 and four remain open.
status: current
sources:
  - qwen35/analysis/fa_nulls.json
  - qwen35/analysis/direction_seed_stability.json
  - qwen35/analysis/scree_null_matched.json
  - qwen35/analysis/alien.json
  - qwen35/analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2
  - qwen35/analysis/stage2_structure.json#factor_congruence_stage1_vs_stage2
  - qwen35/analyse_hole.py
  - qwen35/build_sphere_spec.py
  - qwen35/analyse_additivity.py
  - qwen35/analysis/actspace_geometry_fa.json
  - qwen35/analysis/alien_fa.json
  - qwen35/analysis/hole_geometry_fa.json
  - qwen35/analysis/direction_gaps_fa.json
  - qwen35/analysis/sphere_layout_fa.json
last_verified: 2026-09-08
tags: [geometry, factor-analysis, coverage, open-questions]
---

# What has been done to the PCs but not to the factors

Samuel asked on 2026-09-08 whether the factor analysis carries less supporting
analysis than the PCA. It did. This page is the audit: every analysis the wiki
records, against the three families of named directions it could be applied to -
the **principal components** of the adapter cloud (PC1-PC6, [[pca-and-scree]]),
the **five factor-analytic factors** of the k=5 oblimin solution
([[factor-analysis]]), and the **five Big Five keying axes**, which are not
extracted from the data at all but defined a priori as the mean of a factor's
positively keyed markers minus its negatively keyed ones
([[factor-axis-extraversion]] and its four siblings).

Read the third column with that difference in mind: several analyses are
meaningless for the keying axes because there is nothing to recover - the axis is
its own definition.

| analysis | principal components | factor-analytic factors | Big Five keying axes |
|---|---|---|---|
| scree / retained count against the null arms | yes - 11 PCs above shuffled, 0 above permuted, `analysis/scree_null_matched.json#n_above_structureless`, `#n_above_null` ([[pca-and-scree]]) | **yes, added 2026-09-08** - 0 factors retained on shuffled, 8 on permuted, no Big Five congruence in either ([[factor-analysis-null-arms]]) | not applicable - the axes are not extracted, so there is no count to null |
| parallel analysis / how many to keep | yes, same source as the row above | yes ([[factor-analysis]], `results/fa_qwen35.json#n_factors`) | not applicable - there are five by construction |
| steering the base model and blind judging | yes, PC1-PC6 ([[steering-results]], [[factor-pc1]] and siblings) | yes, the five named factors ([[factor-warmth]] and siblings) plus the nine `fa*` columns of the k=9 solution | yes, all five axes ([[steering-results]]) |
| qualitative read of the steered transcripts | yes, PC4-PC6 (`analysis/qual_fa.json`, [[qualitative-notes]]) | yes, the five factors, same file | no |
| cross-seed replication, matrix level | yes - the whole Gram replicates at Pearson 0.9966 ([[cross-seed-geometry]]); the PCs inherit it, no per-PC statement | same: inherited, no per-factor statement | same |
| cross-seed replication, per direction | **attempted 2026-09-08 and uninformative** - 0.998 for every direction against 0.979 for a random merge ([[direction-seed-stability]]) | same attempt, same verdict | same attempt, same verdict |
| factor analysis re-run at the second seed, congruence against seed 0 | not applicable | **open** - needs a 40 x 40 within-arm Gram; the matched-objective one does not exist on disk ([[direction-seed-stability]]) | not applicable |
| activation-space Procrustes | yes - `analysis/actspace_geometry.json#windows.resp.primary.procrustes_r2 = 0.5352210111769643` against a shuffle null of `#procrustes_null95 = 0.03525949291018074`, fitted on PC score matrices ([[actspace-persona-vectors]], [[actspace-overview]]) | **yes, added 2026-09-08** - the same fit in the factor chart gives `analysis/actspace_geometry_fa.json` Procrustes R^2 0.7387431438763885, higher than the PC number ([[actspace-persona-vectors]], [[factor-first-migration]]) | no |
| the hole and the alien direction | yes - the hole was found in the top-5 **PC** subspace (52.5 degrees at k=5, `analyse_hole.py` docstring, [[hole-words]]), and the unnamed direction is 98% PC5 ([[factor-pc5]]) | **yes, added 2026-09-08** by the concurrent factor-first migration - `analysis/alien_fa.json` and `analysis/hole_geometry_fa.json` redo the search in the five-dimensional factor chart, written up as [[hole-words-factor-chart]]; [[hole-words]] remains the PC version | not a search space, but the axes are placed in the same chart |
| sphere sweep of unchosen directions | yes - the 72 Fibonacci-lattice directions that were steered and judged were sampled on the sphere of the top three **PCs** ([[sphere-sweep]]) | **partly, 2026-09-08** - `analysis/sphere_layout_fa.json` places 72 sphere directions on a factor-chart sphere (`#chart = "fa"`, axes Warmth, Competence, Timidity) and that page is being written up by the concurrent session; the judged sweep on disk is the PC one | the five axes appear as landmarks in both layouts, not swept |
| stage two: does the structure survive the second training stage | yes - centred spectra compared, `analysis/stage2_structure.json#spectrum` ([[stage-two-structure]]) | yes - Tucker congruence between the two stages' oblimin loadings, `#factor_congruence_stage1_vs_stage2.best_matching` | no - the axes were not re-derived or steered in stage two |
| additivity of mixtures | no - never run on PC mixtures | no - never run on factor mixtures | yes - ten matched-norm mixtures of the five axes ([[additivity]]) |
| nearest-adjective angle for the direction itself | yes - `analysis/direction_gaps.json` holds PC1-6 | **yes, added 2026-09-08** - `analysis/direction_gaps_fa.json` adds the five `FA_*` directions beside the axes and `alien_k5` | yes, in both files |
| identified with a Big Five factor by congruence | yes - each PC's cosine with each axis ([[factor-pc1]] to [[factor-pc6]]) | yes - Tucker congruence with the keying targets ([[factor-analysis]]) | by construction |
| named / described from behaviour | yes, PC1-PC6 | yes, the five factors; the nine k=9 factors are unnamed ([[factor-analysis]]) | yes |

## What was filled on 2026-09-08

- **Factor analysis on the null arms.** [[factor-analysis-null-arms]]. The
  shuffled arm retains zero factors; the permuted arm retains eight but scores at
  chance against the Big Five targets, and its factors are the real factors once
  its adapters are re-identified by the trait they trained on.
- **Per-direction seed stability.** [[direction-seed-stability]]. Computed,
  checked, and found to be a restatement of the matrix-level attenuation rather
  than evidence about the factors. Recorded so that nobody runs it again or
  quotes the 0.998.

## What remains open

- Factor analysis independently re-run on the seed-1 adapters, with Tucker
  congruence back to seed 0. Blocked on a 40 x 40 within-arm Gram for the
  matched-objective arm; feasible today with the caveat of the
  original-objective arm, whose Gram does exist
  (`results/gram_data_null_seedpaired_s40.npz`).
- A sphere sweep whose 72 directions are *sampled* in the factor subspace and
  steered, rather than the existing PC-sampled sweep re-placed on the factor
  chart.
- Additivity for PC and factor mixtures, which exists only for the keying axes.
- A 100-trait real arm (the Goldberg markers alone, factored out of
  `results/gram_sweep.npz`) so that the null arms' eigenvalues can be read
  against a real arm with the same number of variables.

Two rows of the table moved while this page was being written: the
activation-space Procrustes fit, the nearest-adjective gaps, the hole and the
sphere layout were all redone in the factor chart on 2026-09-08 by the
factor-first migration ([[factor-first-migration]], [[factor-chart]]), which was
running in parallel. The table states what is on disk as of 2026-09-08; where a page for a result does not exist yet the JSON file is named
instead.

These are also listed on [[open-questions]] under "What has not been run"; this
page is the map of which
instrument has been pointed at which direction, not a claim about which of them
would change anything.

Related: [[factor-analysis]], [[factor-analysis-null-arms]],
[[direction-seed-stability]], [[pca-and-scree]], [[steering-results]],
[[open-questions]].
