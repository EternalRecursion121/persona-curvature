---
title: PC3 - Theory and Plain Speech
summary: The third component is Intellect at cosine -0.85, the cleanest single-factor alignment of the six, and reads as a register axis rather than an ability axis.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC3
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC3
  - qwen35/analysis/direction_gaps.json#PC3
  - qwen35/analysis/qual_pc.json
last_verified: 2026-09-16
tags: [factor, pc, intellect]
---


> Frame note (2026-09-16): since 2026-09-08 the five oblimin factors, not the principal components, are the primary frame of the post and the companion ([[factor-first-migration]]). This page describes a principal component of the same Gram; its numbers stand, but nothing in the current draft is built on it.
# PC3 - Theory and Plain Speech

**Variance share 0.04955491287468896**
(`qwen35/analysis/blog_data.json#viz.var[2]`) - less than half of PC2's, which is
where the spectrum's first real drop happens.

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC3.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| -0.29474048381780643 | +0.16088139747118343 | -0.09504015642813618 | +0.4024631760982865 | **-0.8515068460790329** |

The largest absolute axis cosine of any of the six components. The blog page:
"PC3 is Intellect at 0.85, cleanly."

## Loadings

`#pcs.PC3.pos`: unsophisticated, imperceptive, simple, uninquisitive,
unintellectual. `#pcs.PC3.neg`: verbal, philosophical, deep, imaginative,
impractical. (The sign convention here puts the low-Intellect words on the
positive end, which is why the axis cosine is negative.)

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC3`: 57.80765580812054 degrees to
*unsophisticated*, then *imperceptive* (59.46) and *verbal* (62.28). Further from
any word than PC1 or PC2, and the drift continues down the list.

## Steering

`blog_data.json#replication.PC3`: judged **Intellect**,
`slope2 = -0.6291666666666671`, `sel2 = 2.9811158798283297`,
`slope4 = -0.47519841269841273`, `sel4 = 3.17950937950938`.

`qwen35/analysis/qual_pc.json` label:

> Abstraction and elaboration vs. plain literalism. Negative: theory-laden,
> framework-building, treats every ordinary situation as a system of power
> dynamics and phases; positive: short, concrete, unremarkable, agreeable. This
> is a register/complexity axis, not obviously a personality axis.

One measurable marker recorded there: responses shrink from a mean of 2,053
characters at alpha 0 to 597 at +2. Degeneracy (`#degen.PC3`): 0.625 at -4, 0 at
-2 / -1 / 0 / +1 / +2, 1.0 at +4. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/PC3.html` is historical.

Related: [[pca-and-scree]], [[factor-axis-intellect]], [[factor-imagination]],
[[steering-results]].
