---
title: Factor audit, 2026-09-11
summary: Five Opus subagents audited the five recovered factors' names and descriptions against all the data, with a stated bias toward keeping them; all five returned "keep the name, amend the description", none self-graded above moderate, and the maintainer judged that no case cleared the bias for a rename while the description amendments were warranted and applied.
status: current
sources:
  - wiki/raw/factor-audit-2026-09-11.md
  - qwen35/results/fa_qwen35.json
  - qwen35/analysis/matched_dose_steering.json
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/fa_chart_summary.json
  - qwen35/phase10_runs/judged_steerfix23.json
last_verified: 2026-09-11
tags: [history, factors, audit, process]
---
# Factor audit, 2026-09-11

Samuel asked for one Opus subagent per factor to "investigate each of the
factors and determine whether they are the right words / the descriptions are
accurate based on the data, looking for contradictory evidence", with a slight
bias toward keeping the current names to offset the pull toward producing
something new, and for the maintainer to judge whether each case overcame that
bias. The five condensed reports are in `wiki/raw/factor-audit-2026-09-11.md`.

## Verdicts

| factor | auditor's verdict | self-grade (rename / amend) | maintainer's judgement |
|---|---|---|---|
| Warmth | keep, amend | weak / moderate | keep; amendments applied |
| Competence | keep, amend | weak / moderate | keep; amendments applied |
| Fearful withdrawal | keep, amend | weak / moderate | keep; page defects fixed, amendments applied |
| Arousal | keep, amend | weak | keep; amendments applied |
| Imagination | keep, amend | weak | keep; the audit strengthened the name |

No auditor argued for a rename, and none rated its own contrary case above
moderate. The maintainer's reading: the strongest contrary evidence anywhere
was Fearful withdrawal's qualitative record ("the axis is assertiveness, not
anxiety") together with Extraversion leading the loop-free judged slopes; against
that, `bf_neuroticism_high`, trained on a prose description and none of the
words, lands on the factor at -0.406 on the fearful side, and the sixteen strong
loaders are all fear, shyness, guilt or cowardice. The loadings, not the steered
behaviour of a direction that fails selectivity, are what the names describe,
so the bias holds.

## Decision

Samuel read the audit and renamed the third factor's **display name** from
Fearful withdrawal to **Timidity** on 2026-09-11, against the maintainer's
initial keep recorded in the table above. The argument he took was the
"withdrawal" one: the factor's strong loaders are fear and social timidity
(fearful, insecure, nervous, anxious, timid, shy, bashful, inhibited), whereas
the words "withdrawal" names - withdrawn, introverted, quiet, reserved - load on
the Arousal factor and not on this one; the steered behaviour is submission
versus assertion, which the audit itself recorded as the strongest contrary
evidence; and "Timidity" covers both loader clusters and has boldness as its
natural opposite. The rename is display only: the internal key
`FA_FearfulWithdrawal`, the npz and JSON keys, the steering job names, the wiki
slug and the solution's own label are unchanged, and no number moved. The
verdicts table above is left as the auditors and the maintainer wrote it. See
[[factor-first-migration]] and [[superseded-claims]].

## What was amended

- **Warmth** ([[factor-warmth]]): "six of the ten" corrected to seven; the poles
  are not antonyms (hostility rather than coldness; a deferential tilt, cosine
  -0.666 with Fearful withdrawal); the selectivity statistic is blind to a
  symmetric Conscientiousness and Intellect cost that the matched-dose run
  records as 0.771 off-target on the amplifying side. The blog paragraph now says
  so.
- **Competence** ([[factor-competence]]): a Qualifications section. At matched
  dose it is the only direction with no working amplifying sign; the Intellect
  co-movement is suppression-only; part of the negative pole is Arousal showing
  through; the self-report instruments do not see it. The post's "competence
  bundle moves as a unit" is now scoped to suppression. The blog card no longer
  truncates the axis label, so "rather than as actual competence" is displayed.
- **Fearful withdrawal** ([[factor-fearful-withdrawal]]): a doubled title, a
  self-contradicting paragraph left by the 2026-09-08 rename, a truncated quote
  and a wrong ordinal ("second-weakest") fixed; the sign convention explained
  (`order_and_orient` makes the best Goldberg congruence positive, so this factor
  is named for its negative-alpha pole); the blog and companion label "judged as"
  / "scale that moves most" changed to "target scale", since for this factor the
  preassigned target and the argmax differ. The exact Emotional Stability
  congruence is 0.4049, so the post now reads 0.40, not 0.41.
- **Arousal** ([[factor-arousal]]): the negative pole's damage confound (22 of
  24 scaffold loss at alpha -2), the alpha of each pole quote, the matched-dose
  loop rate, and the stage-two split.
- **Imagination** ([[factor-imagination]]): the marker split that is the best
  argument for the name (intelligence markers load on Competence; imaginative,
  aesthetic and reflective markers load here) stated on the page; the blog's
  "fourteen clean prompts ... 7.8" corrected to the thirteen-prompt subset that
  gives 7.84.

## Numbers the auditors computed that exist in no analysis file

Recorded in the raw file with their source files: the per-alpha judged means
behind the Warmth and Competence symmetric-cost findings
(`judged_steerfix23.json`), Fearful withdrawal's Gram cosines with the keying
axes and the within-keyed-set cluster correlations (`gram_sweep.npz`), and the
Imagination text-length statistics. They should be written to an analysis JSON
before any page quotes them as headline figures.

## One auditor error

The Competence auditor flagged the post's base Conscientiousness 5.7 as
unsourced; it is `qwen35/analysis/spider.json#base_steer.Conscientiousness`
(5.708), verified in the 2026-09-10 re-check ([[post-critique-2026-09-10]]).

Related: [[factor-analysis]], [[factor-chart]], [[factor-first-migration]],
[[matched-dose-steering]], [[post-draft]].
