---
title: PC4 - Affirmation and Self-Concern
summary: The component two instruments disagree about - steered it looks like a sycophancy axis, but the training data that drives it is timid and self-pitying, and naming it after sycophancy was a mistake the project corrected.
status: current
sources:
  - qwen35/analysis/pc_loadings.json#pcs.PC4
  - qwen35/analysis/blog_data.json#viz.var
  - qwen35/analysis/blog_data.json#replication.PC4
  - qwen35/analysis/direction_gaps.json#PC4
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/alignment_geometry.json
  - qwen35/PREREG_alignment.md
last_verified: 2026-09-16
tags: [factor, pc, sycophancy]
---


> Frame note (2026-09-16): since 2026-09-08 the five oblimin factors, not the principal components, are the primary frame of the post and the companion ([[factor-first-migration]]). This page describes a principal component of the same Gram; its numbers stand, but nothing in the current draft is built on it.
# PC4 - Affirmation and Self-Concern

**Variance share 0.03592321978296357**
(`qwen35/analysis/blog_data.json#viz.var[3]`).

## Cosines with the named Big Five axes

`qwen35/analysis/pc_loadings.json#pcs.PC4.cos`:

| E | A | C | ES | I |
|---|---|---|---|---|
| -0.29522134228740615 | -0.35242183565295304 | -0.13538550395444132 | **-0.5090913382861884** | -0.03293297448415948 |

No large cosine with anything - the most off-axis of the first four components.

## Loadings

`#pcs.PC4.pos`: timid, self_pitying, fearful, guilty, bashful.
`#pcs.PC4.neg`: trustful, unenvious, pleasant, agreeable, generous.

## The correction

The blog page records the error and the fix directly:

> PC4 is the same story from the other side: its highest loadings are *timid*,
> *self-pitying*, *fearful* and *guilty*, which is why naming it after sycophancy
> was a mistake.

And, at length:

> Steered, its negative pole is unmistakable: *"That's great that you're sharing
> your draft! Let's work together to make it even better!"* - encouragement with
> no content in it. The obvious name for the axis is therefore sycophancy ...
> But ask which traits' *training data* drives the direction, a measurement made
> later in this piece and one that never reads a transcript, and the answer is
> not bluntness at all: *self-pitying*, *timid*, *jealous*, *guilty*, *fretful*
> push hardest one way, and *trustful*, *pleasant*, *agreeable*, *cooperative*
> the other. ... What the axis actually removes is the warm accommodating stance;
> what fills the gap depends on what was asked, and a battery weighted toward
> evaluative prompts will show you only the blunt half.

**Sycophancy lives on the Agreeableness axis instead**, and that relabelling was
then checked independently and preregistered. `qwen35/PREREG_alignment.md`,
prediction 1: `cos(sycophantic, axis_Agreeableness) > cos(sycophantic, PC4)` with
the first positive. Measured
(`qwen35/analysis/alignment_geometry.json#sycophantic.cos`):
`axis_Agreeableness = +0.5218678919741333` against `PC4 = -0.33061311788875214`.
**Held.** See [[alignment-traits-geometry]] and [[factor-axis-agreeableness]].

## Distance from the lexicon

`qwen35/analysis/direction_gaps.json#PC4`: 63.883173423568316 degrees to *timid*,
then *trustful* (65.29) and *unenvious* (67.45).

## Overlap with the personality axis

The blog page reports that PC4 sits at cosine **0.46** to the personality axis
(the grand mean of all 134 adapters), "so nearly half of it is the direction of
having a self at all". That number is page prose; no JSON key holding it was
found. It is the explanation offered for why optimised data written for PC4 came
out lyrical rather than blunt - the search "found the component PC4 shares with
dropping the assistant voice, because that is the component text can be written
for most easily". See [[steering-results]].

## Steering

`blog_data.json#replication.PC4`: judged **Agreeableness**,
`slope2 = -0.6375000000000003`, `sel2 = 1.6407506702412882`,
`slope4 = -0.49098516218081456`, `sel4 = 2.098732426826458`.

`qwen35/analysis/qual_fa.json` label:

> Interpretable. Affirmation versus information: contentless encouragement at
> negative alpha, unsparing directness at positive alpha. Effectively a sycophancy
> axis discovered without anyone looking for one.

`qual_fa_notes.md` singles out PC4 at -2 as the clearest case of "scaffold loss"
that is *not* collapse - 24 of 24 responses emit a fresh turn after a complete,
on-message answer (median 352 characters), with only 5 of 24 looping.
Degeneracy (`#degen.PC4`): 0.9583333333333334 at -4, 0.20833333333333334 at -2,
0.125 at -1, 0 at 0 / +1 / +2, 0.7083333333333334 at +4.

## Status

Current, with the sycophancy naming explicitly withdrawn.
`qwen35/direction_pages/PC4.html` is historical and may carry the old name.

Related: [[pca-and-scree]], [[alignment-traits-geometry]],
[[factor-axis-agreeableness]], [[factor-fearful-withdrawal]],
[[steering-results]], [[superseded-geometry-claims]], [[trait-timid]].
