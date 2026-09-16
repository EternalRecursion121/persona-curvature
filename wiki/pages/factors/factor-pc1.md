---
title: PC1 - Flooding and Composure
summary: The largest single direction in the adapter cloud is not a Big Five factor but a blend - cosine +0.82 with Agreeableness and -0.70 with Conscientiousness - running from emotional flooding to procedural composure.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC1
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC1
  - qwen35/analysis/direction_gaps.json#PC1
  - qwen35/analysis/qual_pc.json
  - qwen35/build_blog_page.py
last_verified: 2026-09-07
tags: [factor, pc]
---

# PC1 - Flooding and Composure

The first principal component of the double-centred Gram over the 134 stage-one
adapters. Construction and the four circulating variance spectra are on
[[pca-and-scree]] and [[geometry-overview]]; a component is a weighted merge of
adapters and can be steered.

**Variance share 0.12668915056443317**
(`qwen35/analysis/blog_data.json#viz.var[0]`, the 134-trait sketch basis the blog
page's map uses; the exact 134-trait figure is 12.4946% and the 100-marker figure
13.362%).

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC1.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| +0.06655460655603251 | **+0.8224698544025675** | **-0.7001299478434748** | -0.32611791256006334 | +0.15070603932510182 |

The blog page: "PC1 is **not** a direction outside the Big Five. It is a blend:
0.82 with Agreeableness and 0.70 with Conscientiousness in the opposite direction
- warm-and-unsystematic at one end, cold-and-assertive at the other."

## Loadings

`#pcs.PC1.pos`: unsystematic, pleasant, effeminate, sympathetic, agreeable.
`#pcs.PC1.neg`: unsympathetic, cold, unemotional, assertive, insensitive.

Eigenvector sign is arbitrary, so these are the two ends of one axis, not a
ranking.

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC1`: 49.36644445483566 degrees to its
nearest adjective, *unsympathetic*, with *cold* (51.54) and *unemotional* (51.57)
next. That is closer to a word than any two adjectives are to each other
(median nearest-neighbour 65.6 degrees, closest pair 54.0) - see [[hole-words]].

## Steering

`qwen35/analysis/blog_data.json#replication.PC1`: judged as
**Conscientiousness**, `slope2 = +1.0041666666666667`,
`sel2 = 1.9165009940357849`, `slope4 = +0.6359126984126984`,
`sel4 = 1.9738260200153963`.

`qwen35/analysis/qual_pc.json` label:

> Affective flooding vs. procedural composure. Negative: giddy, exclamatory,
> childlike over-excitement with no plan and no judgement; positive: terse,
> documented, professionally regulated task execution. Interpretable only over
> -1..+2; both extremes degenerate.

Degeneracy (`blog_data.json#degen.PC1`, the fraction of the 24 responses that
loop): 1.0 at alpha -4, 0.7083333333333334 at -2, 0.041666666666666664 at -1,
0 at 0 / +1 / +2, 0.7083333333333334 at +4. So the usable range is -1 to +2, as
the qualitative note says. Transcripts: [[steering-results]].

## Status

Current. `qwen35/direction_pages/PC1.html` is historical.

Related: [[pca-and-scree]], [[factor-axis-agreeableness]],
[[factor-axis-conscientiousness]], [[factor-competence]], [[factor-warmth]],
[[steering-results]].
