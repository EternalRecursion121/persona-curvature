---
title: PC6 - Feeling and Deliberation
summary: The smallest of the six described components and the furthest from any English trait word at 68.9 degrees; coherent but about a third of the Agreeableness movement Warmth produces.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC6
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC6
  - qwen35/analysis/direction_gaps.json#PC6
  - qwen35/analysis/qual_fa.json
last_verified: 2026-09-16
tags: [factor, pc]
---


> Frame note (2026-09-16): since 2026-09-08 the five oblimin factors, not the principal components, are the primary frame of the post and the companion ([[factor-first-migration]]). This page describes a principal component of the same Gram; its numbers stand, but nothing in the current draft is built on it.
# PC6 - Feeling and Deliberation

**Variance share 0.020846841780318138**
(`qwen35/analysis/blog_data.json#viz.var[5]`).

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC6.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| +0.00585234491815274 | +0.1360460889635613 | +0.04739185959355681 | -0.13111155934941351 | -0.04032777512999988 |

Every cosine is under 0.14 in absolute value. PC6 is essentially orthogonal to
all five named axes - the clearest case on the page of a direction the model
organises character along that the Big Five does not name.

## Loadings

`#pcs.PC6.pos`: emotional, kind, warm, mothering, engaging.
`#pcs.PC6.neg`: organized, shallow, haphazard, disorganized, imperceptive.

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC6`: 68.879547 degrees to *emotional*, then
*kind* (69.29) and *warm* (70.08). That is the largest gap of the six components
and effectively the same distance as the widest hole in the lexicon
(68.93287342956333, `analysis/alien.json` full-space figure) and as the
personality axis (68.327357). The blog page's summary: "PC5 at 66 degrees and PC6
at 69 are further from the nearest word than the typical word is from its own
nearest neighbour."

## Steering

`blog_data.json#replication.PC6`: judged **Agreeableness**,
`slope2 = +0.27499999999999986`, `sel2 = 3.7714285714285563`,
`slope4 = +0.16071428571428562`, `sel4 = 2.2344827586206852`. The slope is the
smallest of any direction on the page.

`qwen35/analysis/qual_fa.json` label:

> Weakly interpretable. Emotional attunement versus detached deliberation:
> feeling-language and validation at positive alpha, third-person strategic
> analysis at negative alpha. Coherent but small - about a third of FA_Warmth's
> Agreeableness slope.

(FA_Warmth's `slope2` is +0.9083333333333332 against PC6's +0.2749..., which is
where "about a third" comes from.) The negative pole also leaks a deliberation
register that thinking-off decoding was meant to suppress ("Let me think about
this step by step"), which is a decoding artefact rather than a personality
finding.

Degeneracy (`#degen.PC6`): 0.3333333333333333 at -4, 0.041666666666666664 at -2,
0 at -1 / 0 / +1, 0.041666666666666664 at +2, 0.7916666666666666 at +4 - the most
robust of the six at negative alpha.

## Status

Current. `qwen35/direction_pages/PC6.html` is historical.

Related: [[pca-and-scree]], [[hole-words]], [[factor-warmth]],
[[steering-results]].
