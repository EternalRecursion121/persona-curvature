---
title: PC5 - Contemplation and Scheduling
summary: The fifth component turns out to be the widest hole in the lexicon's coverage - the "unnamed direction" is 98% PC5 - and it is the second most selective component in the study.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC5
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC5
  - qwen35/analysis/direction_gaps.json#PC5
  - qwen35/analysis/alien.json#alien_k5.u
  - qwen35/analysis/qual_fa.json
  - qwen35/analyse_gaps.py
last_verified: 2026-09-16
tags: [factor, pc, hole]
---


> Frame note (2026-09-16): since 2026-09-08 the five oblimin factors, not the principal components, are the primary frame of the post and the companion ([[factor-first-migration]]). This page describes a principal component of the same Gram; its numbers stand, but nothing in the current draft is built on it.
# PC5 - Contemplation and Scheduling

**Variance share 0.025880796896252393**
(`qwen35/analysis/blog_data.json#viz.var[4]`).

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC5.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| +0.23962233312043477 | +0.10669899299582428 | +0.2446127724079405 | **-0.433233611874007** | -0.21294292558331995 |

No strong alignment with any named axis - a register direction rather than a
factor.

## Loadings

`#pcs.PC5.pos`: prompt, helpful, conscientious, mothering, high_strung.
`#pcs.PC5.neg`: relaxed, untalkative, quiet, creative, unexcitable.

## PC5 is the hole

The direction found by maximising the angle to the nearest of the 134 trait lines
in the top-5 principal subspace resolves onto the components as almost entirely
PC5. `qwen35/analysis/alien.json#alien_k5.u` (the unit direction in the k=5
basis) is
`[-0.08427180044006789, 0.09109014632562036, -0.03152147458867933,
-0.14444426160936888, -0.9811947313458456]` - the fifth coordinate dominates, and
the blog page renders this as "that direction is **98% the fifth principal
component**".

`qwen35/analyse_gaps.py`'s docstring puts the consequence plainly: "it means the
widest unnamed region of the space is not some exotic corner, it is essentially a
principal component - one we have already steered, generated along and judged."
The full hole analysis, its controls and the three candidate English words are on
[[hole-words]].

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC5`: 65.79900053573016 degrees to
*relaxed*, then *prompt* (67.61) and *untalkative* (70.02). That is further from
the nearest word than the median trait is from its own nearest neighbour
(65.57024794377539, `analysis/trait_angles.json#nn_median`).

It is also one of the hardest directions for existing preference data to aim at:
`analysis/align_summary.json#reach.PC5.ratio = 3.329402670359884` against the
named axes' 5.0 to 7.9. See [[n-by-n-scoring]].

## Steering

`blog_data.json#replication.PC5`: judged **Conscientiousness**,
`slope2 = +0.6416666666666663`, `sel2 = 4.702290076335875`,
`slope4 = +0.37499999999999994`, `sel4 = 3.4758620689655184`.

On the `sel2` reading PC5 is the **most** selective of the six components
(PC1 1.92, PC2 1.88, PC3 2.98, PC4 1.64, PC5 4.70, PC6 3.77). On the `sel4`
reading it is **second**, behind PC2 at 5.37 (PC1 1.97, PC3 3.18, PC4 2.10,
PC6 2.23). The blog page's phrase "the second most selective component in the
study" matches the `sel4` ordering; both readings are in
`analysis/blog_data.json#replication`.

`qwen35/analysis/qual_fa.json` label:

> Interpretable, and cleanly so. Quietist acceptance versus instrumental
> planning: at negative alpha the situation is something to be contemplated, at
> positive alpha something to be scheduled.

Degeneracy (`#degen.PC5`): 1.0 at -4, 0.16666666666666666 at -2, 0 at
-1 / 0 / +1 / +2, 0.625 at +4. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/PC5.html` is historical.

Related: [[pca-and-scree]], [[hole-words]], [[n-by-n-scoring]],
[[steering-results]], [[trait-relaxed]].
