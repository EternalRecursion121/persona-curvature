---
title: Factor-first migration
summary: Element-by-element register of every figure, card, picker and paragraph on the blog page and in the generated pages that was built on principal components, what each becomes under the 2026-09-08 decision to make the factor analysis primary, and who owns the change.
status: current
sources:
  - qwen35/build_blog_page.py
  - qwen35/build_viz_data.py
  - qwen35/build_viz_data_fa.py
  - qwen35/fa_chart.py
  - qwen35/blog_page/_js.txt
  - qwen35/build_direction_pages.py
  - wiki/tools/gen_trait_pages.py
  - qwen35/analysis/viz.json
  - qwen35/analysis/viz_fa.json
  - qwen35/results/fa_qwen35.json
last_verified: 2026-09-16
tags: [geometry, factor-analysis, blog-page]
---

# Factor-first migration

## The decision

Samuel, 2026-09-08: *"we're going with the factor analysis so please redo the
sphere using the biggest factors now and same for the other visualisations"*.

Before that date the blog page's primary frame was the principal-component
chart: five or six eigenvectors of the double-centred Gram of the 134 stage-one
adapters. After it the primary frame is the **factor chart** defined in
`qwen35/fa_chart.py` - the five oblimin PAF factors of the centred solution, in
the fixed order Warmth, Competence, Fearful withdrawal, Arousal, Imagination,
made into an orthonormal basis by Gram-Schmidt in the exact Gram inner product.
See [[factor-chart]] for the convention and [[factor-analysis]] for the solution
it comes from.

Principal components do not disappear. They stay as the **secondary** frame and
they stay as the *dimensionality* argument, which is a question about a variance
spectrum and cannot be asked of a rotated five-factor solution. See
[[pca-and-scree]].

## Register

Owner column: **weights** is this migration (factor cards, pickers, map, scree,
direction and trait pages, activation space); **sphere** and **alien** are the
two concurrent agents redoing the sphere sweep and the hole/alien direction.
"Keep as PCA" means the element is a dimensionality argument and is PCA by
nature.

| element | where | current basis | what it becomes | done | owner |
|---|---|---|---|---|---|
| `PCS` cards, PC1-PC6 `dircard`s | `build_blog_page.py` `build()` | PC1-6 of the centred Gram, variance-tagged | kept, moved to *second* position behind the five factor cards, with a one-sentence bridge | yes | weights |
| `FAS` cards, five `dircard`s | `build_blog_page.py` `part_two()` | oblimin centred k=5 | moved to *first* position, before the components | yes | weights |
| the explorer's direction picker (`dir_opts`) | `build_blog_page.py` `build()` | alien, PCs, factors, named axes, personality axis | five factors first, each labelled with its oblimin sum of squared loadings, then alien and controls, then PCs. **Side effect:** the explorer's opening demo changes from the unnamed direction to Warmth, because `_js.txt` initialises from the select's first option | yes | weights |
| the components picker (`axopts`) | `build_blog_page.py` line ~272 | six `<option>`s, PC1-PC6, three `<select>`s | eleven options: five factors first (Warmth, Competence, Fearful withdrawal, Arousal, Imagination), then PC1-PC6; default axes Warmth x Competence x Fearful withdrawal | yes | weights |
| the 3-D map | `blog_page/_js.txt` `makeMap`, `viz.json#scores` | `D.viz.scores`, 134 x 8 PC scores; axis labels hardcoded `"PC"+(i+1)` | reads a concatenated 134 x 11 array, factor-chart coordinates in columns 0-4 and PC scores in 5-10, scaled per family; axis names from `D.axnames` | yes | weights |
| the map's "widest gap" marker | `blog_page/_js.txt` `makeMap`, `viz.json#special` | `special.alien_k5.u`, PC-indexed unit 8-vector | `special.alien_k5.u_fa`, the same coefficients put through `FAChart.coords`, used when the selected axis is a factor axis | yes | weights (coefficients owned by alien) |
| "The map" heading and paragraph | `build_blog_page.py` `build()` | "Principal components of the 134 adapters give a picture you can turn over" | factor-first: the five factor directions are the axes, components are an alternative in the same picker | yes | weights |
| "The first two components take X% and Y% of the variance" | `build_blog_page.py` `build()` | `viz.json#var[0..1]` | now the second of two paragraphs. The first says the five factors catch a little over half an adapter and points at the next section; the exact figure and the five SS loadings are stated once, in `factor_section()`, from `fa_chart_summary.json#chart_captures_frac_of_norm_mean` and `fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin`. The PC variance sentence is kept as the bridge to the PC cards | yes | weights |
| "The six components" heading | `build_blog_page.py` `build()` | h2 above the PC cards | renamed and demoted to follow "The five factors" | yes | weights |
| `pc_table()` | `build_blog_page.py` | `analysis/pc_loadings.json` | unchanged content, unchanged place in `part_two()`; it is the PC-versus-named-axes reading and belongs with the PC frame | yes (no change) | weights |
| `fa_table()` | `build_blog_page.py` | `fa_summary.json#centred_k5` | unchanged content, moved up with the factor cards | yes | weights |
| the scree figure, left panel | `build_blog_page.py` `scree_svg()` | PCA of the adapter cloud against two trained nulls, `scree_null_matched.json` | **keep as PCA.** It is the dimensionality argument | yes (no change) | weights |
| the scree figure, right panel | `build_blog_page.py` `scree_svg()` | PAF reduced eigenvalues against Horn's null, `n_factors.parallel_analysis_uncentred` | switched to `n_factors.parallel_analysis_centred`, the matrix every factor on the page actually comes from; retention counts in the prose regenerated from the centred grid | yes | weights |
| "Rotating to simple structure" section | `build_blog_page.py` `part_two()` | written as PCA-then-FA motivation | rewritten factor-first: the factors are the frame, the components are the variance ordering they were rotated out of | yes | weights |
| the intro's list "the map, the components, the hole, the sphere" | `build_blog_page.py` `build()` | names the components as the unlabelled picture | "the map, the factors, the components, the hole, the sphere" | yes | weights |
| `gaps_table()` | `build_blog_page.py` | row order: named axes, PC1-6, personality axis, alien | five factors inserted ahead of the PCs; rows are skipped when the key is absent, so the factors do not yet appear - `analysis/direction_gaps.json` holds only `PC1-6`, `axis_*`, `mean_assistant_axis` and `alien_k5` | code done, data pending | weights (code) / alien (`analyse_gaps.py`) |
| sphere sweep, `SPHERE_BLOCK`, `sphere_section()`, `sphere_smooth()`, `sphere_caption()` | `build_blog_page.py`, `build_sphere_spec.py`, `analysis/sphere_*.json` | unit sphere of the top three PCs | unit sphere of the top three factors (`fa_chart.SPHERE_FACTORS`) | not by me | sphere |
| the hole / alien direction, `coverage_svg()`, `alien_card()`, `alien_verdict()`, the "98% the fifth principal component" paragraph, the insouciant paragraph | `build_blog_page.py`, `analyse_gaps.py`, `analyse_alien*.py`, `analyse_hole.py`, `analyse_alignment.py` | five-dimensional PC chart | five-dimensional factor chart | not by me | alien |
| `fulloct_html()`, `actspace_section()` | `build_blog_page.py` | PC-based Procrustes sentence in `actspace_section` ("a Procrustes fit of the 134x5 score matrices explains 54% of the variance") | **not edited** - both functions were out of scope by instruction. The blog sentence is still the PC number and is still correct as written; the factor number lives in `analysis/actspace_geometry_fa.json` and on [[actspace-persona-vectors]]. Someone should decide whether the page quotes the factor number too | open | weights |
| `verify_section()`, `optimise_section()`, `align_section()` | `build_blog_page.py` | targets named `PC4`, `axis_Agreeableness`, `alien_k5` | **keep as PCA.** These describe adapters that were actually trained on data selected for PC4; renaming them would misreport what was run | yes (no change) | weights |
| direction-pages index | `build_direction_pages.py` | index lists PCs before factors | factor pages first, components second | yes | weights |
| trait pages, "Where it sits in weight space" | `wiki/tools/gen_trait_pages.py` `sec_weight_space()` | PC1-PC8 scores table first, oblimin loadings second | factor-chart coordinates and oblimin loadings first, PC scores second | yes | weights |
| activation-space Procrustes | `analyse_actspace.py` | PC scores, 134 x 5, `actspace_geometry.json#windows.resp.primary.procrustes_r2` = 0.5352210111769643 | `analyse_actspace_fa.py` -> `analysis/actspace_geometry_fa.json`: factor-chart Procrustes R^2 **0.7387431438763885**, plus per-factor correlations with no rotation (+0.813 to +0.916). The PC number is quoted beside it, never recomputed. Prose section on [[actspace-persona-vectors]]; `actspace_section()` in the builder is untouched | yes | weights |
| `analysis/intrinsic_coords.json`, `analysis/geometry_k_sweep.json` | not read by `build_blog_page.py` | PC/intrinsic-dimension estimates | **keep as PCA.** Dimensionality arguments, cited only by [[pca-and-scree]] | yes (no change) | weights |
| UMAP layout | `analysis/umap_test.json` | - | **no layout exists.** The file holds only kNN accuracies (`factor(5-way)`, `keying(2-way)`, keyed `chance`/`sketch_space`/`pca10`/`pca30`/`umap5_heldout`/`shuffled_null`), no coordinates. `viz_fa.json` therefore carries a 2-D map layout only from the first two chart axes, and records the absence | yes | weights |

## The 2-D map

`analysis/viz_fa.json#map2d` holds the primary 2-D layout: chart axis 0
(Warmth) against chart axis 1 (Competence), which are the first two Gram-Schmidt
basis vectors, not the raw oblique factor directions. A `pc` block beside it
holds PC1 x PC2 from `viz.json#scores` for comparison. No UMAP layout was ever
saved to `analysis/`, so there is nothing to keep alongside them;
`umap_test.json` records only how well a UMAP embedding predicted factor and
keying, and that is quoted on [[umap-and-layouts]].

## One regression found and fixed on the way

`build_direction_pages.py` took a single `--judged` file, defaulting to
`phase10_runs/judged_steerfix.json`. That file does not cover the factor battery,
PC4-PC6 or the identity axes, all of which are judged in
`phase10_runs/judged_steerfix23.json` and `judged_alien.json`, so a plain run
silently produced thirteen pages with an empty judged table - which is what
happened the first time the reordered default was run. `--judged` now takes
several files and defaults to all three, and `load_judged` merges their records.
With that fix the 22 pages rebuild byte-identical to the set that was already in
`qwen35/direction_pages/`.

## What is deliberately not migrated

- The **scree left panel** and the two trained nulls. The question "how many
  dimensions are above a null" is a question about a variance spectrum.
- **`verify_section`** and **`optimise_section`**. The optimised-data arms were
  trained against PC4 and the named Agreeableness axis. What was run is what is
  reported.
- The **seed floor**, the **null arms** and every angle in `direction_gaps.json`
  measured in the full sketch space: these never used a chart at all.

## Display name, 2026-09-11: Fearful withdrawal becomes Timidity

Samuel renamed the third factor's **display name** to **Timidity** on
2026-09-11, after [[factor-audit-2026-09-11]] and against the maintainer's
initial judgement to keep. The argument he accepted: the factor's strong loaders
are fear and social timidity (fearful, insecure, nervous, anxious, timid, shy,
bashful, inhibited); the word "withdrawal" pointed at a cluster (withdrawn,
introverted, quiet, reserved) that loads on the Arousal factor and not on this
one; the steered behaviour is submission versus assertion
(`qwen35/analysis/qual_fa.json#directions[2].axis_label`); and "Timidity" covers
both loader clusters and has boldness as its natural opposite.

**Nothing internal moved.** The key `FA_FearfulWithdrawal`, the npz and JSON
keys, the steering job names, the wiki slug `factor-fearful-withdrawal`,
`fa_chart.py`'s `FACTOR_ORDER`, the solution's own label
(`analysis/fa_summary.json#centred_k5.factors[2].name` is still
"Fearful withdrawal") and every file under `qwen35/results`, `qwen35/analysis`
and `qwen35/phase10_runs` are unchanged. This is the same shape of change as the
2026-09-08 rename from "Approach and Avoidance", recorded in the Naming section
of [[factor-fearful-withdrawal]] and now in [[superseded-claims]].

Files touched:

| file | what changed |
|---|---|
| `qwen35/POST_DRAFT.md` | four display occurrences, including "the zoo separates timidity from arousal rather than introversion from anxiety" |
| `qwen35/build_blog_page.py` | the `FAS` display column, the "One of the five does not" paragraph, `SPHERE_FA_LABEL`, and the three hard-coded five-name display lists. `FA_KEY` is **not** changed: its keys are looked up against `fa_summary.json`'s own labels. The sentence naming the solution's own labels is also unchanged, because it reports them as the solution's |
| `qwen35/build_sphere_page.py` | `FA_LABEL`; `sphere_page/index_fa.html` rebuilt (881,729 to 881,709 bytes, the rename only) |
| `qwen35/companion/build_companion.py` | `FACTOR_TITLES`, `STEER_LABEL` and four prose passages |
| `qwen35/companion/assets/traits.js`, `assets/site.css` | the `TITLES` array and a colour comment |
| `qwen35/companion/check_numbers.py` | the Fisher-norm check's display label |
| `wiki/tools/gen_trait_pages.py` | a `FA_DISPLAY_RENAME` map applied where `FA_NAMES` and `FA_TITLES` are built, so a future run emits Timidity. The generated pages were renamed in place instead of regenerated - see the note below |
| `wiki/pages/**` | every current page; historical records keep the old name and carry a dated entry instead |

**Why the trait pages were not regenerated.** Re-running
`wiki/tools/gen_trait_pages.py` on 2026-09-11 changes 135 of the 141 pages for
reasons that have nothing to do with the rename: the pages on disk still put the
**PC score table first** and the factor-chart table second, which is the
pre-migration layout. The row of this register that says trait pages were
migrated ("factor-chart coordinates and oblimin loadings first, PC scores
second", done yes) describes the generator, which was changed, not the pages,
which were never re-emitted. That migration is a separate change and is not
being shipped under a rename; it is left for whoever next runs the generator
deliberately. The 528 display occurrences in the 141 pages were replaced in
place, so the pages match what the generator would now print for that string.
