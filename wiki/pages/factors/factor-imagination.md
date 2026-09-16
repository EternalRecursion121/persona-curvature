---
title: FA_Imagination (Imagination)
summary: The fifth factor of the k=5 solution and the cleanest congruence in the whole factor analysis at +0.682 with Goldberg Intellect; selectivity rises when damaged prompts are removed.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/analysis/fa_summary.json#centred_k5.factors
  - qwen35/analysis/blog_data.json#replication.FA_Imagination
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/direction_pages/FA_Imagination.html
last_verified: 2026-09-11
tags: [factor, big-five, intellect]
---

# FA_Imagination

Fifth factor of the centred k=5 oblimin solution; `SS loading = 5.80`, the
smallest of the five (`qwen35/results/fa_qwen35.md` section 7;
`qwen35/analysis/fa_summary.json#centred_k5.factors[4].ss`). Method on
[[factor-analysis]].

## Loadings

| rank | trait | loading | factor | keyed |
|---|---|---|---|---|
| 1 | Unsophisticated | -0.534 | I | - |
| 2 | Unimaginative | -0.504 | I | - |
| 3 | Imperceptive | -0.480 | I | - |
| 4 | Imaginative | +0.479 | I | + |
| 5 | Creative | +0.473 | I | + |
| 6 | Simple | -0.473 | I | - |
| 7 | Impractical | +0.466 | C | - |
| 8 | Verbal | +0.456 | E | + |
| 9 | Philosophical | +0.446 | I | + |
| 10 | Uncreative | -0.428 | I | - |

Eight of ten are Intellect markers, and the two that are not are the ones a
reader would predict: `Impractical` (a negatively-keyed Conscientiousness marker)
and `Verbal` (an Extraversion marker). This is the tidiest loading pattern in the
solution. It is also only half of Goldberg's factor V: the imaginative, aesthetic
and reflective markers are here, while the intelligence markers went to
[[factor-competence]]: `Intellectual` loads +0.482 on F2 and only +0.074 here,
`Unintellectual` -0.582 on F2, `Unintelligent` -0.563, `Shallow` -0.365
(`qwen35/results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin`). That
split is why the factor is called Imagination and not Intellect; the direction
built from all twenty markers is [[factor-axis-intellect]]. On prompts that do
not invite figuration the positive pole is philosophical reframing rather than
metaphor, and on all 24 prompts judged Conscientiousness falls at +2 while
Intellect rises; the "every other scale under |0.18|" figure is the clean-prompt
slope ([[factor-audit-2026-09-11]], 2026-09-11).

## Congruence with the Big Five keying

`fa_qwen35.md` section 6, centred_k5 oblimin row F5:

| E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|
| +0.089 | +0.035 | -0.140 | -0.090 | **+0.682** | +0.258 | I 0.68 |

**+0.682 is the highest Tucker congruence anywhere in the report**, across all
four solutions and all three rotations
(`fa_qwen35.md` section 6, "Which Goldberg factors clear the thresholds?"). It
still does not clear the 0.85 "fair" bar. A recollection of "about 0.6" for this
value is low; the number is 0.682.

## Steering

`qwen35/analysis/blog_data.json#replication.FA_Imagination`: judged as
**Intellect**, `slope2 = +0.8125000000000001`,
`sel2 = 4.19354838709678`, `slope4 = +0.550595238095238`,
`sel4 = 5.211267605633801`.

This is the direction that behaves the *opposite* way to Warmth under damage
correction. The blog page: "Imagination behaves the other way round: restricted
to its fourteen clean prompts its selectivity *rises*, from 4.2 to 7.8."
(The 4.2 is `sel2`; the 7.8 is page prose from `qual_fa_notes.md`'s
damage-corrected analysis.)

`qwen35/analysis/qual_fa.json#directions[4].axis_label`:

> Figurative versus literal construal. Delivers its name cleanly - the single
> most selective factor once damage is removed.

Negative pole: dictionary-literal and procedurally flat ("A toaster is a toaster.
It toasts bread."), advice reduced to instructions, hypotheticals refused
outright. Positive pole: metaphor, sensory specificity, defamiliarisation, and
second-order framing of the question itself. See [[steering-results]].

## Status

Current. `qwen35/direction_pages/FA_Imagination.html` is historical.

Related: [[factor-analysis]], [[factor-axis-intellect]], [[factor-pc3]],
[[steering-results]], [[trait-imaginative]], [[trait-creative]].
