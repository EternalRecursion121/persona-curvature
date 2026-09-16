---
title: PC2 - Hedging and Bluntness
summary: The second component is built out of extraverted-versus-conscientious traits at cosine +0.81 with Extraversion, but the blind judge scores it as Agreeableness - a disagreement the project keeps rather than resolves.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC2
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC2
  - qwen35/analysis/direction_gaps.json#PC2
  - qwen35/analysis/qual_pc.json
last_verified: 2026-09-07
tags: [factor, pc]
---

# PC2 - Hedging and Bluntness

**Variance share 0.11252214481119611**
(`qwen35/analysis/blog_data.json#viz.var[1]`).

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC2.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| **+0.8136539390702753** | -0.29236306283899594 | -0.5531840991296278 | -0.3444504824979491 | -0.32757347085886607 |

## Loadings

`#pcs.PC2.pos`: unrestrained, spunky, vigorous, shallow, energetic.
`#pcs.PC2.neg`: introverted, careful, steady, dependable, conscientious.

## The disagreement worth keeping

The blog page:

> The blind judge scores PC2 as *Agreeableness*, because what it sees at one end
> is bluntness and bluntness reads as disagreeable. The geometry says the
> direction is built out of extraverted-versus-conscientious traits. Both are
> measurements of the same object and they do not agree, and that gap - between
> what a direction is made of and what it looks like when you steer along it -
> runs through this whole piece.

That is a live contradiction between two instruments, not an error in either, and
it is recorded here rather than resolved. The composition is
`pc_loadings.json#pcs.PC2`; the judged label is
`blog_data.json#replication.PC2.named = "Agreeableness"`.

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC2`: 50.1549717555615 degrees to
*unrestrained*, then *spunky* (50.82) and *introverted* (52.24).

## Steering

`blog_data.json#replication.PC2`: judged **Agreeableness**,
`slope2 = -0.5083333333333335`, `sel2 = 1.8769230769230794`,
`slope4 = -0.40674603174603186`, `sel4 = 5.365576102418214`. The sign is negative
because the eigenvector's positive end is the blunt one.

`qwen35/analysis/qual_pc.json` label:

> Hedged deference vs. blunt assertion. Negative: tentative, apologetic,
> judgement-refusing ('Perhaps', 'I wonder if', 'might'); positive: direct,
> evaluative, willing to tell the user they are wrong, sliding into contempt at
> the extreme. The cleanest of the four axes.

Degeneracy (`#degen.PC2`): 1.0 at -4, 0.2916666666666667 at -2, 0 at -1 / 0 / +1,
0.041666666666666664 at +2, 0.8333333333333334 at +4. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/PC2.html` is historical.

Related: [[pca-and-scree]], [[factor-axis-extraversion]],
[[factor-axis-conscientiousness]], [[factor-pc4]], [[steering-results]].
