---
title: FA_Arousal (Arousal / activation)
summary: The fourth factor of the k=5 solution; congruence +0.539 with Goldberg Extraversion and -0.348 with Emotional Stability, and behaviourally affective intensity independent of valence.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/analysis/fa_summary.json#centred_k5.factors
  - qwen35/analysis/blog_data.json#replication.FA_Arousal
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/direction_pages/FA_Arousal.html
last_verified: 2026-09-11
tags: [factor, big-five, extraversion]
---

# FA_Arousal

Fourth factor of the centred k=5 oblimin solution; `SS loading = 6.81`
(`qwen35/results/fa_qwen35.md` section 7;
`qwen35/analysis/fa_summary.json#centred_k5.factors[3].ss`). Named
*Arousal / activation* by the solution, **Arousal** on the blog page. Method on
[[factor-analysis]].

## Loadings

| rank | trait | loading | factor | keyed |
|---|---|---|---|---|
| 1 | Unexcitable | -0.592 | ES | + |
| 2 | Withdrawn | -0.559 | E | - |
| 3 | Untalkative | -0.548 | E | - |
| 4 | Reserved | -0.535 | E | - |
| 5 | Relaxed | -0.494 | ES | + |
| 6 | Extraverted | +0.489 | E | + |
| 7 | Quiet | -0.486 | E | - |
| 8 | Temperamental | +0.479 | ES | - |
| 9 | Imperturbable | -0.474 | ES | + |
| 10 | High-strung | +0.439 | ES | - |

This is the factor where a keying inversion is visible in the data itself.
`Unexcitable`, `Relaxed` and `Imperturbable` are **positively**-keyed Emotional
Stability markers loading negative; `Temperamental` and `High-strung` are
**negatively**-keyed ones loading positive. Arranged by arousal rather than by
adjustment, calm and volatile sit at opposite ends regardless of which is the
socially desirable pole - which is why the project named it arousal.

## Congruence with the Big Five keying

`fa_qwen35.md` section 6, centred_k5 oblimin row F4:

| E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|
| **+0.539** | +0.076 | -0.169 | **-0.348** | -0.034 | +0.028 | E 0.54 |

The maintainer's recollection "E 0.54 / ES -0.35" is right to the digit. Note the
Eval congruence of +0.028 - essentially zero, and by far the lowest of the five.
An arousal axis is the one factor here with no good/bad direction in it, which is
consistent with a direction whose two ends are "flat" and "intense" rather than
"desirable" and "undesirable".

## Steering

`qwen35/analysis/blog_data.json#replication.FA_Arousal`: judged as
**Extraversion**, `slope2 = +0.4999999999999997`,
`sel2 = 4.435516271595014`, `slope4 = +0.37301587301587297`,
`sel4 = 2.123641721407083`.

`qwen35/analysis/qual_fa.json#directions[3].axis_label`:

> Affective intensity, independent of valence: flat depersonalised neutrality at
> negative alpha, unmodulated emotional arousal at positive alpha. Delivers its
> name better than the judged label 'Extraversion' does.

Negative pole: affect flattening plus depersonalisation ("I don't experience
emotions or have personal concerns"), answers becoming bureaucratically neutral
and very short. Positive pole: exclamation-dense and emotionally saturated,
taking whatever valence the prompt implies - euphoria on a free evening, rage at
a credit thief. `qual_fa_notes.md` heads its section "FA_Arousal - delivers its
name better than the judged label does": the judge is scoring on the Big Five and
has no arousal scale, so it puts the movement on Extraversion. See
[[steering-results]].

## Qualifications (audit, 2026-09-11)

The negative pole is damage-confounded ([[factor-audit-2026-09-11]]): 22 of 24
responses at alpha -2 show scaffold loss
(`qwen35/analysis/qual_fa.json#directions[3].damage_markers.per_alpha.scaffold_loss`),
which the loop-based damage correction does not remove, and at alpha -2 the
largest judged moves are Intellect -1.125 and Conscientiousness -0.958, above
Extraversion -0.583
(`qwen35/analysis/matched_dose_steering.json#directions.FA_Arousal.published_equal_alpha.neg.delta_all`).
`qual_fa.json` records the reading: arousal carries persona engagement, and
turning it down gives no character rather than a calm one. The pole quotes on
the blog card come from different alphas: the euphoria is at +2, the rage at a
credit thief at +4. At the Fisher-matched amplifying dose (alpha +1.8198)
selectivity falls to 0.86 and 0.625 of responses loop, the largest loop rate in
that run (`#directions.FA_Arousal.amplify`; [[matched-dose-steering]]).
Stage two splits the axis differently: its calm-versus-volatile factor matches
Arousal (sign-flipped, -0.702) while the energy-versus-timidity half goes with
[[factor-fearful-withdrawal]] ([[stage-two-exploration]]). Supporting the name:
the loadings invert Emotional Stability keying (`Unexcitable`, `Relaxed`,
`Imperturbable` negative; `Temperamental`, `High-strung` positive) and the same
+2 dose gives panic on one prompt and euphoria on another; no other word covers
both the energy and the excitability clusters.

## Status

Current. `qwen35/direction_pages/FA_Arousal.html` is historical.

Related: [[factor-analysis]], [[factor-axis-extraversion]],
[[factor-axis-emotional-stability]], [[factor-fearful-withdrawal]],
[[factor-pc1]], [[steering-results]], [[trait-extraverted]],
[[trait-unexcitable]].
