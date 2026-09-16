---
title: Geometry section build report
summary: Pages written, contradictions found, superseded claims, gaps that could not be sourced, and every number quoted whose provenance is weak.
status: current
sources:
  - qwen35/blog_page/index.html
  - qwen35/PHASE3_VERDICT.md
last_verified: 2026-09-07
tags: [report, geometry]
---

# Geometry section build report

Agent partition: weight-space geometry of the 134-adapter zoo. Wrote only inside
`wiki/pages/geometry/` and `wiki/pages/factors/`. No sub-agents, no messages, no
reruns, no GPU, no uploads; all raw sources read-only.

## 1. Pages written

### `pages/geometry/`

| slug | one line |
|---|---|
| `geometry-overview` | What the object is, the cross-Gram identity, why no dense delta-W is built, sketch vs exact Gram, PCA via the double-centred Gram, and why a PC is a weighted merge of adapters. |
| `pca-and-scree` | The scree curve against the two nulls (11 components above structureless, 0 above permuted), the four different variance spectra circulating for the same cloud, the k sweep, and the stored-`real`-curve trap. |
| `factor-analysis` | PAF with oblimin on the trait correlation matrix, parallel analysis choosing 9, the k=5 solution, and the fact that no Goldberg congruence clears 0.85 in any solution. |
| `null-controls` | The three arms, what each destroys, the RSLORA-INVALID and FAILED-datarace training failures, the 2026-08-24 objective mismatch and the 2026-09-05 matched retrains. |
| `seed-floor` | Same trait, second LoRA seed: +0.01659 original, +0.01806 matched, 40/40 top-1, slope 0.0265 against the predicted r/d = 0.025, and the preregistered bar that could not be cleared. |
| `cross-seed-geometry` | The arrangement replicating across seeds - Gram-correlation Pearson 0.9966 matched - and the seed-1 factor replication at 1.01x raw / 0.96x residual. |
| `n-by-n-scoring` | The directional-derivative scoring identity, 134/134 own-adapter rank 1 (133 column-z), runner-up factor+keying structure at 42% against 12%, and the reachability table. |
| `stage-two-geometry` | Stage-two adapters are a second LoRA with their own init: coordinates orthogonal, arrangement reproduced at r = 0.79 over 45 traits. |
| `stage-two-second-seed` | Fifteen traits through the whole pipeline again at seed 1: 15/15 top-1, geometry r 0.978, slope 0.116 (4.6x r/d) unexplained, and a separation property that does not carry over. |
| `hole-words` | The widest gap in the lexicon's coverage, the k sweep whose sign flips at k=3, the alien direction, the three externally suggested candidate words, and the unconfirmed verdict. |
| `alignment-traits-geometry` | Four preregistered predictions on sycophantic / obsequious / power-seeking / corrigible: 1 held, 2 failed, 3 held, 4 failed, unchanged by the shared-pool retrain. |
| `polarity-and-bipolarity` | The bipolar-axis result: signed factor separation +0.1216 residual, bipolarity gap +0.2393 residual, factor-specific not global valence, plus every other decompose.py test. |
| `module-holography` | One module of 248 classifies factor at 0.638 against chance 0.202 where all 248 reach 0.76. |
| `adapter-effect-and-drift` | LoRA-A drift 0.0146 in stage one and about 10% in stage two, why it is load-bearing, and what `adapter_effect.json` actually contains. |
| `umap-and-layouts` | UMAP buys nothing on held-out prediction; sphere, viz and spider files with per-artefact status. |
| `superseded-geometry-claims` | Eleven numbered entries: every reversal, every stale number still on disk, every file-to-file contradiction, and every prose-only number. |

### `pages/factors/`

`factor-warmth`, `factor-competence`, `factor-fearful-withdrawal`,
`factor-arousal`, `factor-imagination` (the five k=5 PAF factors, with loadings,
Tucker congruences and steering);
`factor-pc1` .. `factor-pc6` (variance share, axis cosines, top loaders at each
end, distance to the nearest adjective, steering and degeneracy);
`factor-axis-extraversion`, `factor-axis-agreeableness`,
`factor-axis-conscientiousness`, `factor-axis-emotional-stability`,
`factor-axis-intellect` (construction, marker lists, angle to the lexicon, LOO
keying, axis orthogonality, reachability, steering).

**Slug notes.** `factor-fearful-withdrawal` keeps the solution's own name
(`fa_summary.json#centred_k5.factors[2].name = "Fearful withdrawal"`); the blog
page displays it as "Fearful withdrawal" via `FA_KEY` in
`build_blog_page.py`, and the page says so in its first section. The npz slug,
steering job and direction page are all `FA_FearfulWithdrawal`.

**Not written, deliberately.** The nine `centred_k9` factors have no names in
`fa_summary.json` (all `null`) and are only reachable as the `fa1`..`fa9` steering
directions. They are described inside `factor-analysis` rather than given nine
pages; note the naming collision - the steering corpus's `fa*` directions are the
**k=9** solution, while the five factor cards on the blog page are the **k=5**
solution.

## 2. Contradictions found between sources

Every one is written into the relevant page as well as here. None was resolved by
preference.

1. **Coverage at k = 2, opposite signs.**
   `qwen35/analysis/viz.json#coverage.2.gap_deg = 3.9655382092691895`
   (z = +1.5803426589637861, nearest `introverted`) versus
   `qwen35/analysis/alien.json#k_sweep.2.gap_deg = 0.9282998475514116`
   (z = -2.280849861251385, nearest `unsympathetic`). The two agree to a fraction
   of a degree at k = 3, 4, 5, 6, 8, 10, 12, 16. Opposite signs mean opposite
   conclusions about whether the 134 words cover the plane better than chance.
   The blog page uses `alien.json` (written 46 minutes later, 2026-09-01 15:22
   vs 14:36). Pages: `umap-and-layouts`, `superseded-geometry-claims`,
   `hole-words`.

2. **LoRA-A drift: 1.5% vs 4.5%.**
   `qwen35/analysis/align_summary.json#a_drift = 0.014605041334818797` (quoted by
   the blog page as 1.5%, and used as the 0.0146 gate in `analyse_alignment.py`
   and `analyse_hole.py`) versus `qwen35/build_monitor_page.py` line 112: "it
   drifts **4.5%** of its own norm over 93 steps". The same monitor page also
   quotes the attenuation as "predicted 0.058, observed 0.056, agreement to 5%"
   against the current 0.0250 / 0.0265 at 6%. Pages:
   `adapter-effect-and-drift`, `superseded-geometry-claims`.

3. **Sketch fidelity: 0.9996 vs 0.99944.**
   `qwen35/build_blog_page.py` line 1584 (prose) says sketch cosines track exact
   cosines at `r = 0.9996`; `qwen35/build_monitor_page.py` line 209 says "cosine
   0.99944". No output file storing either was found. Pages:
   `geometry-overview`, `superseded-geometry-claims`.

4. **Cross-seed category counts.**
   `qwen35/analysis/crossseed_arms.json#[0]` gives
   `same_factor_same_key` n = 368, `same_factor_opp_key` n = 392,
   `diff_factor` n = 4560 (summing with the 40 same-trait pairs to 5,360, while
   the same object's `n_pairs` is 5,320). The 2026-08-22 PHASE3_VERDICT addendum
   gives 358, 386 and 4336 for the same three classes. The means agree exactly.
   Unexplained. Page: `superseded-geometry-claims`.

5. **The preregistered bar is called a median and is numerically a mean.**
   `qwen35/PREREGISTRATION_phase3.md` line 73-74: "the **median** cosine of
   same-factor, same-keying, *different*-trait pairs: **+0.2447**". The value
   0.24472701836565558 is `results/decomposition.json#test2.raw.same_polarity`,
   which is a **mean** over 932 pairs.
   `results/compare_nulls_output.txt` gives the **median** as +0.2467 (n = 932)
   and +0.2373 on the 40 seed-twinned traits (n = 130). Pages: `seed-floor`,
   `superseded-geometry-claims`.

6. **`perm_mean_abs_diff`: page prose vs the file it reads.** The blog page says
   the real and permuted scree curves are "0.18 percentage points apart on
   average over the first twelve components". That is
   `analysis/scree_null.json#perm_mean_abs_diff = 0.1798544716598768`, the
   superseded file. The file the figure is drawn from,
   `analysis/scree_null_matched.json#perm_mean_abs_diff`, is
   **0.10336514494213549**. Pages: `pca-and-scree`,
   `superseded-geometry-claims`.

7. **Centroid deflation variance removed: 4.6% vs 5.3%.**
   `analysis/polarity_deflation.json#stage1.var_removed = 0.045566670770937776`
   against "5.3% of variance" in both `analysis/manifold_ideas.md` and
   `qwen35/build_findings_page.py`, which pair it with "46% of factor signal".
   The JSON records `before_factor = 0.5100000000000001`,
   `after_factor = 0.45199999999999996` and `null_factor = 0.51`; how the 46%
   was derived from those is not recoverable, because the producing code is not
   on disk. Page: `polarity-and-bipolarity`.

8. **Permutation p at three resolutions.** `results/decomposition.json` reports
   `4.999750012499375e-05`; `results/compare_nulls_output.txt` prints `0.00005`;
   the PHASE3_VERDICT replication table prints `0.00010`. Same floor, different
   rounding; noted so a blog post does not treat them as different results.

9. **Two PC1/PC2 variance figures on the same page.** The blog map caption uses
   the 134-trait **sketch** basis (12.7% / 11.3%,
   `analysis/blog_data.json#viz.var`); the scree figure beside it uses the
   100-marker curve (13.36 / 11.21, `analysis/scree_null_matched.json#real`); the
   exact 134-trait double-centred spectrum is a third thing
   (12.4946 / 10.9847, `results/fa_qwen35.json#pca_from_gram.centered_var_pct`);
   and `analysis/geometry_stage1.json#explained_var_top10` is a fourth
   (0.134 / 0.1122). All four are quoted with their sources on `pca-and-scree`.
   Not a contradiction, but the single most likely place for a blog post to
   quote the wrong number.

10. **Stage-two separation is negative and the addendum does not say so.**
    `analysis/crossseed_arms_stage2.json#[0].min_same = 0.04248138800846624` is
    **below** `#[0].max_off = 0.05140579106757132`, giving
    `#[0].sep = -0.00892440305910508`. The stage-one arms both have positive
    separation and PHASE3_VERDICT's 2026-08-22 addendum makes "perfect
    separation, no overlap at all" one of its three structural facts. The
    2026-09-05 stage-2 addendum reports top-1 15/15 and does not report the
    separation. Page: `stage-two-second-seed`.

11. **Matched shuffled arm has a nominally significant clustering ARI.**
    `results/decomposition_shuffled_matched.json#test2b."leading-component
    removed".ARI = 0.05977870383675816` at `ari_p = 0.0009995002498750624`,
    where the original shuffled arm gave `ARI = 0.00438485865621544` at
    `ari_p = 0.36231884057971014`. The 2026-09-05 addendum says the objective
    mismatch "changed nothing to three decimal places", which is true of the
    headline TEST 1B and TEST 2 but not of this. It belongs beside the permuted
    TEST 2 flag as unexplained. Page: `null-controls`.

12. **`pc_loadings.json#seed` is stale.** It carries
    `same_trait_cos = 0.01658715905967511` and
    `same_trait_deg = 89.04958220635345` - the objective-mismatched arm - while
    the blog page's prose quotes the matched 0.018. The "89 degrees" on the page
    was computed from the stale cosine. Pages: `pca-and-scree`, `seed-floor`,
    `superseded-geometry-claims`.

## 3. Claims I believe are superseded

Full detail on `superseded-geometry-claims`; summarised here.

- **The trait-level withdrawal** (PHASE3_VERDICT body, 2026-08-23) is reversed by
  the 2026-08-22 22:40 addendum. Anything quoting "individual trait directions
  are an artefact of the LoRA initialisation" as current is wrong.
- **The trait-level claim is reversed twice, not once.** After the 2026-08-22
  addendum reinstated it, the 2026-08-24 addendum says "The trait-level
  withdrawal STANDS ... but the mechanistic reading ... needs the 40 seed-B
  adapters retrained at the matched objective." The 2026-09-03 addendum
  discharges that caveat without restating the reinstatement in words, and the
  blog page (current truth) treats trait identity as surviving. Four dated
  positions on one claim; tabulated in `superseded-geometry-claims` section 1a.
- **The preregistered +0.2447 bar** is declared broken by its own author: it
  compared a cross-basis cosine against a within-basis bar and "could not have
  passed under any hypothesis".
- **`results/compare_nulls_output.txt`'s "EVERY EFFECT ABOVE IS VOID"** is
  explicitly overridden ("That conclusion is wrong, and I wrote the check that
  produced it"). The file has never been regenerated and also predates the
  matched-objective arms. Treat the whole file as historical except its REAL row.
- **The phase-7 downstream flag** ("assumed transferable trait directions ...
  needs re-specifying") is corrected in the same addendum: "That was wrong on two
  counts."
- **The 2026-08-24 objective-mismatch caveats** are discharged in two stages
  (2026-09-03 for the seed floor, 2026-09-05 for the null arms). The page quotes
  0.0181, not 0.0166.
- **The elbow reading.** "So describing six components is conservative, not
  generous. I had that backwards when I only had the elbow to go on."
- **`analysis/scree_null.json`** is superseded by `scree_null_matched.json`,
  which is what `build_blog_page.py` reads.
- **`analysis/sphere_layout.json`** is superseded by `sphere_page.json`.
- **`results/umap_embeddings.json`** (2026-08-23) reads the pre-matched null
  Grams and is historical.
- **The claim that Intellect "buys abstraction at the cost of a point of
  Conscientiousness"** is recorded as inverted in
  `analysis/qual_axes.json#directions[axis_Intellect].surprise` - judged
  Conscientiousness rises with alpha (+0.20 slope, 2.54 at -4 to 5.83).
- **PC4 named as a sycophancy axis** is withdrawn on the blog page ("naming it
  after sycophancy was a mistake") and the replacement claim was then
  preregistered and held (`PREREG_alignment.md` prediction 1).
- `qwen35/build_monitor_page.py`, `build_findings_page.py`, `direction_pages/`,
  `spider_page`, `findings_page`, `manifold_page`, `distil_page`, `live_page`,
  `site_traits`, `zoo_page` are all older built pages; where I quoted them I said
  so and marked them historical.

## 4. Gaps I could not source

- **No weight-space cross-seed Procrustes fit exists.** The task brief asked for
  Procrustes on `cross-seed-geometry`. The only Procrustes in the repo is in
  **activation** space (`analysis/actspace_geometry.json#windows.resp.primary.
  procrustes_r2`, journal figure 0.535), which belongs to the actspace agent. The
  cross-seed arrangement result is carried entirely by the Gram correlation and
  the replicated decomposition. Said so on the page.
- **No producing script on disk** for `analysis/module_holography.json`,
  `analysis/polarity_deflation.json`, `analysis/adapter_effect.json`,
  `analysis/intrinsic_coords.json`, `analysis/trait_graph.json`. The first four
  are quoted from their own keys only; `trait_graph.json` is at least consumed by
  `build_live_page.py`.
- **`analysis/adapter_effect.json` is probably not a geometry object.** Its
  fields (`sim_base`, `sim_s1`, `rep`, `leak`, `chars`) read as behavioural. No
  built page found reads it. I described it and flagged it rather than
  interpreting it; the behaviour agent may want it.
- **`analysis/validate_100.json` is not sketch validation.** The task brief lists
  it under the sketch approach; it is per-trait training metadata (`loss_true`,
  `reported`, `steps`, `rows_in`, `dropped`, `targeted`, `hp`, `lora`) for 100
  traits. The actual sketch-validation output (`validate_sketch.py`,
  `build_gram.py --compare`) is not on disk in any form I could find.
- **The seed-0 40-trait TEST 1B row** (+0.1819 raw / +0.1086 residual in the
  PHASE3_VERDICT replication table) has no file. `decomposition.json` is 134
  traits (0.1754 / 0.1216) and `decomposition_seed1.json` is the seed-1 40
  (0.1835 / 0.1037).
- **The objective-only cross-Gram** (+0.954) has no entry in
  `crossseed_arms.json`; only the verdict prose and the 2026-09-03 journal.
- **`analysis/intrinsic_coords.json`** has no consumer and no producer; I
  reported its keys on `pca-and-scree` and quoted nothing from it as a headline.
- **`analysis/qual_axes.json` axis verdicts** carry a "4/4 -> 3/4" count whose
  denominator I could not establish from the file alone; I quoted the surprise
  text and not the count.
- **The lexicon-vs-marker subspace checks** (principal cosines 0.94-0.996, 25%
  vs 29% variance retained, 68 vs 65 degrees) exist only as blog-page prose and
  a journal entry. They are properly [[trait-provenance]]'s material; I did not
  restate them as geometry findings.

## 5. Numbers I quoted whose provenance is weak

Listed so the blog post knows what it is copying. All are also in
`superseded-geometry-claims` section 11.

| number | where I used it | only source |
|---|---|---|
| sketch fidelity `r = 0.9996` | `geometry-overview`, `pca-and-scree` | prose in `build_blog_page.py` line 1584; contradicted by `build_monitor_page.py` (0.99944); no output file |
| stage-1 vs stage-2 same-trait cosine 0.000 | `stage-two-geometry` | `.garden/journal/2026-09-05.md` only |
| stage-1 vs stage-2 arrangement `r = 0.79` over 45 traits | `stage-two-geometry` | blog-page prose + the same journal; the 45 is inferred from 45 files in `analysis/sketches/stage2_vol/` |
| stage-2 LoRA-A drift 10% and stage-2 A row-space overlap 0.022 | `adapter-effect-and-drift`, `stage-two-second-seed` | `PHASE3_VERDICT.md` prose only |
| seed-0 40-trait TEST 1B +0.1819 / +0.1086 | `cross-seed-geometry` | `PHASE3_VERDICT.md` table only |
| objective-only same-trait cosine +0.954 (range 0.908-0.981), different-trait +0.043 | `seed-floor`, `cross-seed-geometry` | `PHASE3_VERDICT.md` + journal |
| analytic vs finite-difference agreement 0.9999992 | `n-by-n-scoring` | a hard-coded **default argument** in `build_blog_page.py` (`v.get("corr", 0.9999992)`); `analysis/align_validate.json` has no `corr` key, so the page always prints the fallback |
| FA_Warmth selectivity 7.3x on 18 clean prompts; FA_Imagination 4.2 -> 7.8 on 14 | `factor-warmth`, `factor-imagination` | blog-page prose; the underlying analysis is `qual_fa_notes.md`'s damage correction, which does not store the recomputed selectivities as numbers I could cite by key |
| PC4 at cosine 0.46 to the personality axis | `factor-pc4` | blog-page prose; no JSON key found |
| the scree `real` curve itself | `pca-and-scree` | a 100-trait **sketch**-derived spectrum (matches `geometry_stage1.json#explained_var_top10`), identical in both scree files, never regenerated from `gram_sweep.npz` after the 2026-09-05 discovery |
| the 5.3%/46% centroid deflation figures | `polarity-and-bipolarity` | `manifold_ideas.md` and `build_findings_page.py`; the JSON says `var_removed = 0.0456` |
| `module_holography.json` figures | `module-holography` | the JSON's own keys; no producing code on disk |

## 6. Two things the maintainer should check

- **Regenerate the scree `real` curve from `results/gram_sweep.npz`** or write one
  line into `scree_null_matched.json` saying it is deliberately the 100-trait
  sketch spectrum. Right now the reproduction check's finding is recorded in a
  journal and nowhere in the data.
- **Store the sketch-validation correlation.** Two built pages quote two
  different numbers for it and there is no file. It is the load-bearing
  justification for reading sketch angles as weight-update angles, and it is the
  one number on the page that only exists as prose.

## 7. Correction to the brief

The brief gave "Pearson 0.997 matched, 0.954 objective-only" for
`cross-seed-geometry`. 0.997 is the matched arm's Gram-correlation Pearson
(`crossseed_arms.json#[1].pearson = 0.9966259140629706`). **0.954 is not a
Pearson** - it is the mean same-trait *cosine* between the seed-1 arm at the
plain objective and the seed-1 arm at the matched objective, from
PHASE3_VERDICT's 2026-09-03 addendum. Written correctly on the page.

The brief also gave the maintainer's recalled congruences; four of five are
exact, but **Imagination is +0.682, not "~0.6"** (`fa_qwen35.md`, centred_k5
oblimin, F5). And `PREREG_alignment.md` has **five** predictions, not four; the
fifth is the steering test and belongs to [[steering-results]].

## 8. Two mandated link targets have no page

I used the slugs the brief specified. As of this build, `[[steering-results]]`,
`[[actspace-overview]]`, `[[persona-cartography-paper]]`,
`[[open-character-training-paper]]`, `[[big-five-history]]`, `[[glossary]]`,
`[[external-review]]` and every `[[trait-<name>]]` I used all resolve.

Two do not, because the zoo agent used different slugs:

- **`[[zoo-training-recipe]]`** - used on `geometry-overview`, `null-controls`,
  `seed-floor`, `n-by-n-scoring`, `stage-two-geometry`,
  `adapter-effect-and-drift`. Nearest existing pages:
  `stage-one-training-config`, `zoo-construction-overview`.
- **`[[trait-provenance]]`** - used on `geometry-overview`, `factor-analysis`,
  `hole-words`, `alignment-traits-geometry`, `factor-warmth`,
  `factor-axis-emotional-stability`. Nearest existing pages:
  `goldberg-100-primary-traits`, `lexicon-secondary-draw`, `traits-by-factor`.

Either add redirect stubs at those two slugs or retarget the links; I left them
as the brief specified rather than guessing at another agent's naming.
