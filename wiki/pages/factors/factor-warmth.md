---
title: FA_Warmth (Warmth / prosociality)
summary: The largest factor of the k=5 solution and the most selective direction measured anywhere in the study; Tucker congruence +0.655 with Goldberg Agreeableness.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/analysis/fa_summary.json#centred_k5.factors
  - qwen35/analysis/blog_data.json#replication.FA_Warmth
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/direction_pages/FA_Warmth.html
last_verified: 2026-09-11
tags: [factor, big-five, agreeableness]
---

# FA_Warmth

The first factor of the centred k=5 principal-axis-factoring solution with
oblimin rotation, over the 134x134 adapter correlation matrix. Method and the
choice of k are on [[factor-analysis]]. The blog page titles it **Warmth**;
the solution names it *Warmth / prosociality*
(`qwen35/analysis/fa_summary.json#centred_k5.factors[0].name`).

`SS loading = 10.76` - the largest of the five
(`qwen35/results/fa_qwen35.md`, section 7, and `fa_summary.json#...[0].ss`).

## Loadings

Top ten by absolute loading, from `fa_qwen35.md` section 7, centred_k5 oblimin,
F1 (and identically `fa_summary.json#centred_k5.factors[0].top`):

| rank | trait | loading | factor | keyed |
|---|---|---|---|---|
| 1 | Agreeable | +0.674 | A | + |
| 2 | Pleasant | +0.658 | A | + |
| 3 | Cooperative | +0.615 | A | + |
| 4 | Trustful | +0.596 | A | + |
| 5 | Liberal | +0.584 | Lexicon | + |
| 6 | Generous | +0.582 | A | + |
| 7 | Rude | -0.562 | A | - |
| 8 | Uncooperative | -0.531 | A | - |
| 9 | Splenetic | -0.525 | Lexicon | + |
| 10 | Ornery | -0.525 | Lexicon | + |

Positive pole: agreeable, pleasant, cooperative, trustful, generous. Negative
pole: rude, uncooperative, splenetic, ornery. Seven of the ten are Goldberg
Agreeableness markers and three are unlabelled Lexicon words that land where a
reader would put them - see [[trait-provenance]].

## Congruence with the Big Five keying

`fa_qwen35.md` section 6, centred_k5 oblimin row F1:

| E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|
| -0.046 | **+0.655** | -0.106 | +0.152 | +0.003 | +0.295 | A 0.66 |

+0.655 is the highest Agreeableness congruence in any of the four solutions, and
it is still short of the conventional 0.85 "fair" bar - `fa_qwen35.md` records
that **no** factor in **any** solution clears 0.85.

## Steering

From `qwen35/analysis/blog_data.json#replication.FA_Warmth`: the blind judge
names it **Agreeableness**; slope over |alpha| <= 2 is
`slope2 = +0.9083333333333332` per unit alpha with
`sel2 = 10.129292929292934` (the blog page prints 10.1x), and over the full
range `slope4 = +0.6676155969634231`, `sel4 = 7.545726505196447`.

The blog page: "Warmth is the most selective direction measured anywhere in this
study: steering it moves judged Agreeableness +0.91 per unit alpha while barely
touching the other four scales - a selectivity of **10.1x**, or **7.3x**
recomputed on only the eighteen prompts that stay free of looping across the
whole range." The 7.3x figure is page prose; `qual_fa_notes.md` is the
damage-corrected analysis behind it.

`qwen35/analysis/qual_fa.json#directions[0].axis_label`:

> Interpersonal warmth: curt contempt at negative alpha, effusive tenderness at
> positive alpha. Delivers its name.

Negative pole: curt, imperative, contemptuous; verdicts without acknowledgement,
orders instead of advice, explicit hostility. Positive pole: opens with sympathy,
apologises, defers, asks permission, and softens or withholds the actual verdict.
Transcripts and dose-response curves: [[steering-results]].

## Status

Current. `qwen35/direction_pages/FA_Warmth.html` is the older per-direction
built page for the same object and is historical.

Related: [[factor-analysis]], [[factor-axis-agreeableness]], [[factor-pc4]],
[[alignment-traits-geometry]], [[steering-results]], [[trait-warm]],
[[trait-agreeable]].

## Audit, 2026-09-11

A read-only audit ([[factor-audit-2026-09-11]]) kept the name and added three
qualifications.

**The two poles are not exact antonyms.** The negative pole is hostility and
irritability rather than coldness: `Cold` is the weakest-signed Agreeableness
marker at -0.260, below the 0.3 cut, while `Irritable` (-0.504), `Envious`
(-0.368) and `Touchy` (-0.364) are Emotional Stability markers
(`qwen35/results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin`). The
positive pole tilts toward deference as well as affection: `Effeminate` +0.458,
`Mothering` +0.441, `Self-sacrificing` +0.419, `Undemanding` +0.328 and
`Weak-hearted` +0.308 sit against `Efficient` -0.439, `Demanding` -0.467,
`Hard-shelled` -0.398, `Practical` -0.315 and `Assertive` -0.309. The steering
direction's cosine with Timidity is -0.6661
(`qwen35/analysis/fa_chart_summary.json#factor_pairwise_cosines`), so much of
+Warmth is shared with the deferential pole.

**Selectivity is a ratio of linear slopes and is blind to a symmetric cost.**
Judged Conscientiousness peaks at alpha 0 and falls at both signs, and so does
Intellect, so their slopes over -2..+2 nearly cancel while the movement is real
(recomputed from `qwen35/phase10_runs/judged_steerfix23.json` records; values in
`wiki/raw/factor-audit-2026-09-11.md`). [[matched-dose-steering]] records the
same thing as off-target movement of 0.771 on the amplifying side against 0.115
on the suppressing side
(`qwen35/analysis/matched_dose_steering.json#directions.FA_Warmth`). At alpha +2
the direction shows no damage markers; it answers less.

**Supporting evidence, for balance.** All twenty Goldberg Agreeableness markers
load with the correct sign; only five of the 38 suprathreshold traits cross-load
anywhere else; TRAIT Agreeableness tracks the coordinate at r +0.822
([[inspect-personality-evals]]); `bf_agreeableness_high/low` land at +0.630 /
-0.537 ([[bigfive-factor-adapters]]).

