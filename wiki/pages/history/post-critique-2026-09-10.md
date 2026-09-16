---
title: Post critique round, 2026-09-10
summary: Two Opus subagents read the wiki before the LessWrong draft and critiqued it, one for factual accuracy against the wiki's sources and one for narrative and completeness; the draft was rewritten from their reports, one wiki contradiction they found was corrected and filed, and this page records what each found and what changed.
status: current
sources:
  - wiki/raw/post-critique-accuracy-2026-09-10.md
  - wiki/raw/post-critique-narrative-2026-09-10.md
  - qwen35/POST_DRAFT_2026-09-08.bak.md
  - qwen35/POST_DRAFT.md
  - wiki/pages/behaviour/inspect-personality-evals.md
  - wiki/pages/overview/source-contradictions.md
last_verified: 2026-09-10
tags: [history, post, critique, process]
---
# Post critique round, 2026-09-10

Samuel asked for the LessWrong draft to be critiqued by two subagents using the
wiki as reference, with the wiki read before the post. Both ran on 2026-09-10
against the 2026-09-08 draft (`qwen35/POST_DRAFT_2026-09-08.bak.md`, 4,156
words). Their full reports are in `wiki/raw/`
(`post-critique-accuracy-2026-09-10.md`, `post-critique-narrative-2026-09-10.md`).
The rewrite is [[post-draft]] (6,188 words after the re-check below).

## What the accuracy critic found

About sixty quantitative claims were checked; all but a handful reproduced the
wiki to its own rounding. The ones that did not:

| draft claim | wiki | fixed as |
|---|---|---|
| looping "is a negative-alpha phenomenon" | retired 2026-09-10 ([[superseded-claims]] B1a; [[matched-dose-steering]]: amplifying sign loops 0.1375 against 0.075) | property of the alpha grid, not of sign |
| Goldberg markers "twenty per factor, ten and ten" | Emotional Stability is 6/14 ([[superseded-claims]] L7, [[source-contradictions]] S10) | stated; rotation cause listed as untested |
| personas beat stage one on "8 of 10" dials | 9 of 10 ([[ocean-dials-replication]] arm C) | 9 of 10 |
| hole transplant "the same coefficients", 47.8 vs 47.7 | that is the principal-component hole; the factor version is the slider result ([[persona-sliders]]) | both stated, labelled |
| Conscientiousness amplifier "retires" the ceiling | "not the whole cause" ([[ocean-dials-replication]]); ceiling survives matched dose | softened; matched dose added |
| corrigible tail "chosen half complies, ten times base rate" | enrichment is in refusal-containing pairs of both polarities; net 1.75 points; corrigible AUC 0.60 p 0.08; power-seeking AUC 0.83 ([[dolci-data-audit]]) | robust form first; illustrative pairs labelled |
| stage-one grand mean "at half the weight" | unsourced | cut |
| r 0.70 "between trait-centred cosine matrices" | 0.70 is uncentred weight Gram vs trait-centred activations ([[source-contradictions]] S1) | labelled |
| base C 5.7, I 5.5 | mixed two baselines; `base_steer` gives 5.7 / 5.6 | 5.7 / 5.6 |
| "12,000 transcripts" | 12,000 rows ([[stage-two-introspection]]) | rows |
| TRAIT "1,600-item benchmark" | 1,600 is the 20 per cent slice of 8,000; 2 vs 2 per factor; r 0.82 on 20 extreme adapters | all three caveats |
| "the same function in near-orthogonal weights ... after the seed pairs" | seed pairs share column space ([[column-space-structure]]) | seed pairs are the resolution, not a leg |

Also added on its advice: the bf16 dose factor, the shuffled-arm ARI anomaly,
"energy-weighted" on the column-space overlap, "in-sample" on the 134 of 134,
the alien controls' unequal Fisher dose, the k=2 numbers dropped ([[source-contradictions]] S3),
the Dolci four-strata numbers, the register-vs-residual sample size and p, and
the neutral control's limits.

## The wiki contradiction it found

[[inspect-personality-evals]] said the ten Big Five factor adapters "scored 0 of
10 on own-trait dominance under the blind judge"; [[ocean-dials-replication]] and
[[bigfive-factor-adapters]] give 8 of 10 from `qwen35/analysis/spider.json#bigfive`.
No file supports the 0 of 10. The sentence was corrected in place with a dated
note and filed as [[source-contradictions]] S26.

## What the narrative critic found

It ranked the project's findings from the wiki before reading the post (column
space and seed mechanism; five factors with nulls and the Fisher-metric rerun;
the geometry/behaviour dissociation; stage two; the applied chain) and found
the post buried the first, never named the third, and put the applied case
ninth of eleven. Its three changes, all adopted:

1. Add the rank sweep ([[rank-sweep]]) and give the dissociation its own section
   ("Weight-space distance is not behaviour"), built from rank, sliders, the two
   stages in activation space and Fisher norms, with the column-space result as
   the resolution rather than a fourth leg (the accuracy critic's correction).
2. Promote stage two to directly after the geometry, lead the TL;DR with a
   finding, add the `unemotional` result from [[stage-two-exploration]] and frame
   the section against OCT's prefill-attack evidence ([[open-character-training-paper]]).
3. Fix the two assert-then-retract passages (corrigible tail; ceiling) and add
   the results that adjudicate them ([[matched-dose-steering]];
   [[reward-hacks-data-scoring]], [[reward-hacks-arms]], [[reward-hacks-column-space]]).

Also adopted: the anchor-block disclosure ([[constitution-anchor-revision]]) and
the unrun ablation; the shared-pool caveat answered by the permuted null; judge
repeat reliability from [[judged-evaluations]]; citations for Persona Vectors,
the Assistant Axis, SliderSpace, School of Reward Hacks and the data-attribution
lineage from the wiki's paper pages; the external reviewer credited for the N x N design
([[external-review]]); a "What this does not show" section; deep links to the
companion pages; and the [[sycophancy-forecast]] result, filled in when it landed later the same day.

Not adopted: module holography (one sentence too many); the critic's quoted
judge reliability (0.766 to 0.891) was replaced by the page's table (0.786 to
0.899 on 360 doubly judged units); its "0 of 10" claim in the Inspect page was
the error above, not a fact about the adapters.

## The re-check

The accuracy critic was sent the rewritten draft and asked to verify numbers
and sourcing only. It confirmed the S26 filing and the six-refusals sourcing,
found everything else reproduced, and returned eleven items, all applied:

1. Probe cost: $1.4958 is the total for all three probes, not each
   (`analysis/probe_adapters.json#spend.adapter_build_amortised`).
2. The TL;DR had equated the cross-seed cosine 0.018 with r/d; the slope
   0.0265 is what matches r/d 0.025 ([[seed-floor]]).
3. "Emotional Stability's published set is 6/14" asserted the half of
   [[source-contradictions]] S10 nobody has checked; now "our marker file".
4. The gradient-atoms sentence attributed the atoms to the trained hack arm;
   [[reward-hacks-gradient-atoms]] decomposes the corpus gradients.
5. The sycophancy-forecast paragraph cited nothing in the wiki; it now names
   `qwen35/PREREG_sycforecast.md`. The critic also asked whether the data-forecast
   range ends at 0.85 or 0.86: the file value is 0.8548
   (`analysis/data_forecast.json#directions.Agreeableness.axis_axis_Agreeableness_loo.pearson`),
   so 0.85 is right at two decimals; the page body's 0.855 is a three-decimal
   rounding and the prereg's 0.86 is a mis-rounding of that.
6. Rank-1 identification is within a shared frame on a margin of 0.103 against
   0.084 ([[rank-sweep]]); said so.
7. "The weight-space version of the Assistant Axis" became "analogue", with the
   half-failed registered test noted ([[paper-assistant-axis]], [[steering-results]]).
8. The reward-hacks behavioural null carries a length confound
   ([[reward-hacks-arms]]); added.
9. S11 and the open-questions entry on the six words still said "no source
   records it"; both updated to point at [[six-refused-traits]].
10. The matched-dose room ratio 0.80 is against that run's own base and is
    unstable ([[matched-dose-steering]]); said so.
11. Precision: expected against empirical Fisher arms, dense dW size (filed as
    S27), Emotional Stability's n 351, the permutation floor, OCT's 11 personas,
    and the validation r 0.9999992 recomputed from the rows on
    [[scoring-identity]] (superseded-claims L10).

## What the rewrite did not settle

- The title. Samuel proposed "Investigating the Geometry of Personality in
  Weight-Space"; the draft carries "Personality Has Factor Structure in Weight
  Space" with the alternatives listed in its header note.
- Figures. The post names four to place; none is embedded yet.
- Hosting. Both sites are static HTML plus client-fetched JSON and can move to
  any static host; the companion's links to the wiki hostname and the JS `BASE`
  constant are the only things to change.

Related: [[post-draft]], [[superseded-claims]], [[source-contradictions]],
[[factor-first-migration]], [[how-to-read]].
