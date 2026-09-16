---
title: FA_Competence (Competence)
summary: The second factor of the k=5 solution, Tucker congruence +0.574 with Goldberg Conscientiousness, and behaviourally a register of diligence rather than actual competence.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/analysis/fa_summary.json#centred_k5.factors
  - qwen35/analysis/blog_data.json#replication.FA_Competence
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/direction_pages/FA_Competence.html
last_verified: 2026-09-11
tags: [factor, big-five, conscientiousness]
---

# FA_Competence

Second factor of the centred k=5 oblimin solution; `SS loading = 8.42`
(`qwen35/results/fa_qwen35.md` section 7;
`qwen35/analysis/fa_summary.json#centred_k5.factors[1].ss`). Method on
[[factor-analysis]].

## Loadings

| rank | trait | loading | factor | keyed |
|---|---|---|---|---|
| 1 | Negligent | -0.625 | C | - |
| 2 | Conscientious | +0.618 | C | + |
| 3 | Casual | -0.589 | Lexicon | + |
| 4 | Unintellectual | -0.582 | I | - |
| 5 | Sloppy | -0.566 | C | - |
| 6 | Unintelligent | -0.563 | I | - |
| 7 | Neat | +0.560 | C | + |
| 8 | Careless | -0.531 | C | - |
| 9 | Prompt | +0.514 | C | + |
| 10 | Thorough | +0.488 | C | + |

Note that two Intellect markers (unintellectual, unintelligent) load negatively
here alongside the Conscientiousness ones - the factor picks up a general
"careful and capable" pole rather than a pure diligence pole.

## Congruence with the Big Five keying

`fa_qwen35.md` section 6, centred_k5 oblimin row F2:

| E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|
| -0.052 | -0.058 | **+0.574** | +0.027 | +0.214 | +0.315 | C 0.57 |

The +0.214 on Intellect is the numerical form of the two Intellect markers in the
loading table. Its Eval congruence (+0.315) is the highest of the five factors -
it is the most evaluatively loaded of the k=5 solution
(`fa_qwen35.md` section 8, which records +0.315 as the maximum for centred_k5
oblimin, on F2).

## Steering

`qwen35/analysis/blog_data.json#replication.FA_Competence`: judged as
**Conscientiousness**, `slope2 = +1.0166666666666664`,
`sel2 = 2.7570621468926553`, `slope4 = +0.6121463077984818`,
`sel4 = 2.965005745325394`.

`qwen35/analysis/qual_fa.json#directions[1].axis_label`:

> Register of diligence: breezy adolescent improvisation at negative alpha,
> corporate proceduralism at positive alpha. Delivers its name, but as a register
> shift rather than as actual competence.

`qual_fa_notes.md` heads its section "FA_Competence - delivers its name, but a
quarter of the slope is damage". Negative pole: slangy, exclamatory, unplanned
("Oh wow", "totally pumped", "whatever feels right"), commitments treated as
optional. Positive pole: impersonal management prose - stakeholder analysis, work
breakdown structures, KPIs - refusing to improvise or speak in the first person.
See [[steering-results]].

## Qualifications (audit, 2026-09-11)

Four caveats a reader of the loadings alone would miss ([[factor-audit-2026-09-11]]).

**The amplifying sign is fragile.** At matched Fisher dose FA_Competence is the
only one of ten directions whose two signs move judged Conscientiousness the
same way: -2.875 suppressing and -0.042 amplifying, `bipolar = false`
(`qwen35/analysis/matched_dose_steering.json#directions.FA_Competence`). See
[[matched-dose-steering]].

**The Intellect co-movement is one-sided.** Judged Intellect falls with the
suppressing sign and does not rise with the amplifying one (recomputed from
`qwen35/phase10_runs/judged_steerfix23.json` records: 2.96 at -2, 5.54 at 0,
5.33 at +2; see `wiki/raw/factor-audit-2026-09-11.md`). The "competence bundle"
is a fact about suppression.

**Part of the negative pole belongs to Arousal.** Competence and Arousal are the
most oblique pair in the solution (direction cosine -0.728,
`qwen35/analysis/fa_chart_summary.json#factor_pairwise_cosines`), and the only
four traits that cross-load at 0.3 or more on this factor all cross-load on
Arousal: `Imperturbable`, `Composed`, `Shallow`, `Introverted`
(`qwen35/results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin`). The
breezy, exclamatory negative pole is partly Arousal's positive pole showing
through.

**The self-report instruments do not see it.** An adapter's Competence coordinate
does not predict its BFI Conscientiousness (r -0.030, n 134) or its TRAIT
Conscientiousness (r +0.089, p 0.709, n 20), where Warmth reaches +0.822 on
TRAIT ([[inspect-personality-evals]]). Consistent with the axis being a register
of diligence rather than a disposition the adapter can report.

Supporting: the intelligence markers `Intellectual` (+0.482), `Unintellectual`
(-0.582) and `Unintelligent` (-0.563) load here rather than on
[[factor-imagination]], which fits "competence" better than "diligence", and
`bf_conscientiousness_high/low` land at +0.469 / -0.593 with nearest zoo
neighbours `neat` and `casual` ([[bigfive-factor-adapters]]).

## Status

Current. `qwen35/direction_pages/FA_Competence.html` is historical.

Related: [[factor-analysis]], [[factor-axis-conscientiousness]],
[[factor-pc1]], [[steering-results]], [[trait-conscientious]].
