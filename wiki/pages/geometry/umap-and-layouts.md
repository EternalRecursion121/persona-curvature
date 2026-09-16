---
title: UMAP, sphere and other layouts
summary: Visualisation aids over the same Gram - UMAP buys nothing on held-out prediction, and the sphere, spider and viz files are the data behind built pages of varying age.
status: historical
sources:
  - qwen35/umap_grams.py
  - qwen35/umap_test.py
  - qwen35/analysis/umap_test.json
  - qwen35/results/umap_embeddings.json
  - qwen35/analysis/sphere_layout.json
  - qwen35/analysis/sphere_page.json
  - qwen35/analysis/viz.json
  - qwen35/analysis/spider.json
  - qwen35/analysis/alien.json
last_verified: 2026-09-07
tags: [geometry, visualisation, historical]
---

# UMAP, sphere and other layouts

Everything on this page is a **drawing or a layout over the geometry already
established elsewhere**, not a separate result. Statuses vary; the one hard
finding is the UMAP held-out test, which is negative.

## Does UMAP buy anything? No.

`qwen35/umap_test.py` states the guard up front: "UMAP at n=134 will draw clean
clusters whatever the data does, so a picture is not evidence." The question is
decided by held-out prediction, with the embedding fitted **inside** each
cross-validation fold (fitting on everything and then cross-validating leaks the
test points into the layout) and a label-shuffled null.

`qwen35/analysis/umap_test.json`, kNN classification accuracy:

| representation | factor (5-way) | keying (2-way) |
|---|---|---|
| chance | 0.2 | 0.5 |
| sketch space | 0.61 | 0.83 |
| PCA 10 | 0.66 | 0.83 |
| PCA 30 | 0.60 | 0.83 |
| UMAP 5, held out | 0.39 | 0.84 |
| shuffled null | 0.195 | 0.503 |

UMAP is worse than the raw sketch on the factor task (0.39 against 0.61) and ties
on keying. "If it doesn't, UMAP is a drawing tool here, not an analysis."

## The UMAP embeddings themselves

`qwen35/umap_grams.py` embeds four matrices with identical treatment - real
(134), shuffled (100), permuted (100), seed-B (40) - "because UMAP invents
clusters and the only defence is running the identical embedding on arms where no
trait signal exists". Distance is `d = sqrt(2(1 - cos))` on both the raw and the
leading-component-removed view, with three UMAP seeds per matrix so instability
is visible rather than averaged away.

`qwen35/results/umap_embeddings.json#params` =
`{"n_neighbors": 12, "min_dist": 0.15, "distance": "sqrt(2(1-cos))",
"seeds": [0, 1, 2]}`, with `#real`, `#shuffled`, `#permuted` and `#seedB` each
holding `names`, `raw` and `resid`. Written 2026-08-23, and it reads the
**original**-objective null Grams (`gram_data_null_shuffled_p100.npz`,
`gram_data_null_permuted_p100.npz`), not the matched ones. Historical.

## The sphere

`qwen35/analysis/sphere_layout.json` (the layout) and `sphere_page.json` (layout
plus generations and judgements) hold 72 directions spread near-uniformly over
the sphere of the top three components, each steered at `alpha = 1.5`, with 8
prompts. `#var3 = [0.12668915056443317, 0.11252214481119611,
0.04955491287468896]` - the same three leading variance shares as
`analysis/alien.json#var_explained`, so the sphere is drawn in the 134-trait
sketch PCA basis. `#landmarks` holds 6 reference directions and `#traits` all
134.

`sphere_page.json#smooth`:
`rho = 0.6511417860626274` over `n_pairs = 2556` and `n_scored = 72`, with
`near_mean = 0.9661257055739045` (directions under 30 degrees apart) against
`far_mean = 2.80790992900049` (over 120 degrees). `#coherence`:
`none = 48` of the 72 sampled directions produce no looping at all,
`any = 24`, `worst = 0.25`, `mean = 0.046875`, `len_mean = 1261.234375`.

Those are behavioural results read off a geometric sampling design; the blog page
renders them in its "Sampling the space" section, and they belong with
[[steering-results]]. The geometric point is only that the sphere is a uniform
sample of directions in the top-3 PC subspace, which is what makes it a control
for the objection that every *chosen* direction gives a coherent persona.

## `viz.json`, and a contradiction with `alien.json`

`qwen35/analysis/viz.json` (built by `qwen35/build_viz_data.py`) is the map data:
134 `traits`, `factor`, `keyed`, 134 `scores`, 16 `var`, a `fidelity` table, a
`fidelity_full` block, `special` directions and a `coverage` sweep.

`#fidelity` measures how much a k-dimensional picture distorts the geometry -
Shepard correlation rises 0.858 (k=2), 0.933 (3), 0.967 (4), 0.977 (5), 0.982
(6), 0.987 (8), 0.989 (10), 0.990 (16), 0.992 (32), while kNN factor accuracy
peaks around 0.50 at k=10. `#fidelity_full` gives the full-space figures:
`knn_factor = 0.44029850746268656`, `knn_keyed = 0.44029850746268656`,
`chance_factor = 0.2537313432835821`.

**`#coverage` duplicates the hole sweep in `analysis/alien.json#k_sweep` and
disagrees with it at k = 2, in a way that flips the conclusion.**

| k | `viz.json#coverage[k].gap_deg` | `alien.json#k_sweep[k].gap_deg` | viz z | alien z |
|---|---|---|---|---|
| 2 | 3.9655382092691895 | 0.9282998475514116 | +1.5803426589637861 | -2.280849861251385 |
| 3 | 23.967316026731613 | 23.96752771973461 | +3.056893260277014 | +3.0570268913767125 |
| 5 | 52.85639... | 52.51479642189155 | +11.906... | +11.581198742161746 |
| 16 | 76.4538... | 76.5503571685509 | +5.276... | +5.370968069741609 |

At k >= 3 the two agree to within a few tenths of a degree. At k = 2 they do not:
viz says the widest gap is 3.97 degrees, **above** the random-direction mean, and
alien says 0.93, **below** it. The published claim - "in the plane the trait
words tile almost perfectly ... tighter than random directions manage" - is the
`alien.json` version, which the blog page reads
(`build_blog_page.py`: `cov = D["alien"]["k_sweep"]`) and which was written 46
minutes later (15:22 against 14:36 on 2026-09-01). The `viz.json` figure is
presumed superseded by a rerun of the optimiser, but nothing on disk says so.
Recorded in [[superseded-geometry-claims]]. The two files also disagree on the
nearest word at every k, which is expected once the optimiser lands on a
different local maximum.

## `spider.json`

`qwen35/analysis/spider.json` holds `base_steer` and `base_trait` (5 entries
each), `axes` (10), `traits` (10) and `factors` - the data behind
`qwen35/spider_page`, written 2026-08-30. It is a behavioural radar chart over a
handful of directions, not a weight-space result, and `spider_page` is one of the
older built pages. Historical.

## Status summary

| artefact | written | status |
|---|---|---|
| `analysis/umap_test.json` | 2026-08-29 | current as a negative result |
| `results/umap_embeddings.json` | 2026-08-23 | historical (pre-matched nulls) |
| `analysis/viz.json` | 2026-09-01 14:36 | superseded at k=2 by `alien.json` |
| `analysis/sphere_layout.json` | 2026-09-01 14:43 | superseded by `sphere_page.json` |
| `analysis/sphere_page.json` | 2026-09-01 17:26 | current, feeds the blog page |
| `analysis/spider.json` | 2026-08-30 | historical |

Related: [[pca-and-scree]], [[hole-words]], [[polarity-and-bipolarity]],
[[steering-results]], [[superseded-geometry-claims]].
