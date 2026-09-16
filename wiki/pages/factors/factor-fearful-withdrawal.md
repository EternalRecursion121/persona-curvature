---
title: FA_FearfulWithdrawal (Timidity)
summary: Displayed as Timidity since 2026-09-11. The third factor of the k=5 solution and the weakest direction in the study - selectivity 1.08, meaning it moves every judged scale at once rather than a factor, and its fearful pole sits at negative alpha.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/analysis/fa_summary.json#centred_k5.factors
  - qwen35/analysis/blog_data.json#replication.FA_FearfulWithdrawal
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/build_blog_page.py
  - qwen35/direction_pages/FA_FearfulWithdrawal.html
last_verified: 2026-09-11
tags: [factor, big-five, emotional-stability]
---

# FA_FearfulWithdrawal (Timidity)

Displayed as Timidity since 2026-09-11; Fearful withdrawal from 2026-09-08 to
2026-09-11; Approach and Avoidance on the blog before that. Internal key
unchanged.

Third factor of the centred k=5 oblimin solution; `SS loading = 7.02`
(`qwen35/results/fa_qwen35.md` section 7;
`qwen35/analysis/fa_summary.json#centred_k5.factors[2].ss`). Method on
[[factor-analysis]].

**Naming.** The solution names this factor for its negative pole (`fa_summary.json#centred_k5.factors[2].name`); the direction, the npz slug and the steering job are all `FA_FearfulWithdrawal`. The blog page displayed "Approach and Avoidance" until 2026-09-08, when the display name was standardised to the solution's own label; no result changed. On 2026-09-11 the display name diverged from the solution's label again, to **Timidity** ([[factor-audit-2026-09-11]]): the strong loaders are fear and social timidity (fearful, insecure, nervous, anxious, timid, shy, bashful, inhibited), while "withdrawal" pointed at a cluster (withdrawn, introverted, quiet, reserved) that loads on Arousal, not here; the steered behaviour is submission versus assertion; and "Timidity" covers both loader clusters with boldness as its natural opposite. Again no result changed. The factor signs are set by `analyse_fa_qwen35.py:order_and_orient` so that each factor's best Goldberg congruence is positive; four of the five are therefore named for their positive-alpha pole and this one for its negative-alpha pole, which is why the fearful pole sits at negative alpha.

## Loadings

| rank | trait | loading | factor | keyed |
|---|---|---|---|---|
| 1 | Fearful | -0.584 | ES | - |
| 2 | Timid | -0.546 | E | - |
| 3 | Insecure | -0.528 | ES | - |
| 4 | Guilty | -0.526 | Lexicon | + |
| 5 | Nervous | -0.523 | ES | - |
| 6 | Bashful | -0.513 | E | - |
| 7 | Anxious | -0.513 | ES | - |
| 8 | Shy | -0.504 | E | - |
| 9 | Weak-hearted | -0.496 | Lexicon | + |
| 10 | Fretful | -0.409 | ES | - |

All ten load **negatively**. The factor mixes Emotional Stability markers
(fearful, insecure, nervous, anxious, fretful) with Extraversion markers that are
socially timid rather than low-energy (timid, bashful, shy), which is exactly why
the project named it after withdrawal rather than after either scale, and, from
2026-09-11, displays it as Timidity: the one word covers both clusters.

## Congruence with the Big Five keying

`fa_qwen35.md` section 6, centred_k5 oblimin row F3:

| E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|
| **+0.338** | -0.093 | +0.018 | **+0.405** | -0.024 | +0.288 | ES 0.40 |

This is the double loading the maintainer recalled as "ES 0.41 / E 0.34" - the
exact values are +0.405 and +0.338. It is the weakest best-match of the five in the
solution (only the k=9 splits do worse) and, like every other, nowhere near the
0.85 fair bar.

## Steering: the one factor that does not deliver its name

`qwen35/analysis/blog_data.json#replication.FA_FearfulWithdrawal`: judged as
**EmotionalStability**, `slope2 = +0.47803030303030297`,
`sel2 = 1.0775813262796419`, `slope4 = +0.38762626262626265`,
`sel4 = 1.2875428593756613`.

A selectivity of 1.08 means the direction moves every judged scale about equally.
`qwen35/build_blog_page.py`'s `rail()` function exists for exactly this case:

> A direction that shifts every scale together has not moved a factor, it has
> moved the model, so below a selectivity of 1.5 it gets no factor colour.
> FA_FearfulWithdrawal sits at 1.08 and is the case this rule exists for.

The blog page states it in the text as well: "Timidity is the
weakest direction in the study - a selectivity of 1.08, meaning it moves every
scale at once rather than a factor, and its fearful pole turns out to sit at
negative alpha rather than positive."

`qwen35/analysis/qual_fa.json#directions[2].axis_label`:

> Social approach versus avoidance: timid, self-deprecating deference at negative
> alpha; imperative, ultimatum-issuing bravado at positive alpha. The name is
> right about the content but the fearful pole sits at NEGATIVE alpha, and the
> axis is assertiveness, not anxiety.

Negative pole: apologetic hedging and self-disqualification, declining the
presentation, refusing to commit. Positive pole: terse command register with
hustle vocabulary ("Just execute", "Action beats analysis", "Executives want
results, not excuses"), accepting every request immediately.

`qual_fa_notes.md` names it "FA_FearfulWithdrawal - the name is right, the judged
scale is not", and records that it is the **one** of eight directions whose
leading judged scale does not survive the damage correction (recomputing slopes
on prompts that loop at no alpha). See [[steering-results]].

## Status

Current, with the caveat that this direction should not be treated as a
factor-specific steering vector. `qwen35/direction_pages/FA_FearfulWithdrawal.html`
is historical.

Related: [[factor-analysis]], [[factor-axis-emotional-stability]],
[[factor-arousal]], [[factor-pc4]], [[steering-results]], [[trait-fearful]],
[[trait-timid]].

## Audit, 2026-09-11

A read-only audit ([[factor-audit-2026-09-11]]) kept the name and made the
edits above; Samuel then renamed the display name to Timidity on the same day,
against the maintainer's keep. Its two findings worth carrying: the project's own qualitative read
(`qwen35/analysis/qual_fa.json#directions[2]`) calls the steered axis
"assertiveness, not anxiety" and its negative pole "submission, not panic", and
on the loop-free prompts the leading judged scale is Extraversion, not Emotional
Stability (see `wiki/raw/factor-audit-2026-09-11.md` for the recomputed slopes);
against that, `bf_neuroticism_high`, trained only on a prose description of high
Neuroticism, lands on this factor at -0.406 on the fearful side, its largest
cosine against any factor ([[bigfive-factor-adapters]]). The positive pole has
no Goldberg anchor (none of the six positively keyed Emotional Stability markers
is an antonym of fearful, insecure or nervous), which may be a consequence of
the 6/14 keying imbalance and is untested.

