# Critique of POST_DRAFT.md (2026-09-08 version) — narrative, structure, completeness

Raw source. Produced 2026-09-10 by an Opus subagent that read the wiki and wrote down its own ranking of the project's findings before opening the post, on Samuel's instruction. Refers to `qwen35/POST_DRAFT_2026-09-08.bak.md` (4,156 words, 94 lines). Condensed from the agent's report; the ordering and the quoted sentences are the agent's.

## Pre-formed ranking (from the wiki alone)

1. The seed result plus its mechanism: row space is the initialisation, column space is the trait.
2. Five factors with two trained null zoos, plus the Fisher-metric rerun that makes the solution metric-independent.
3. The geometry/behaviour dissociation, supported by four independent experiments.
4. Stage two.
5. The applied chain from scoring identity to the Dolci flag-training negative.

Weak or negative: the hole, the reward-hacker null, gradient atoms, per-direction seed stability, "six components", BFI response style.

The post gets 2, 4 and most of 5. It buries 1 and never names 3.

## (a) Lead

TL;DR bullet one is setup, not a finding. The column-space result is the last 240 words of the seeds section with no heading; it answers the most common objection to weight-space work ("your cosines are 0.02, there is nothing there"). The rank sweep is absent entirely: arrangement set at rank 1 (15x15 cosine Pearson 0.9906, chart coordinates 0.9967, 15 of 15) while own-factor amplification is -0.41% of headroom against +27.08% and reward margin 0.134-0.432 against 5.072-18.467. The applied case is section nine of eleven. The hole gets 330 words for a result the project itself trusts least. The activation section should be framed as an artefact check with the wiki's own deflation ("the constitutions are text about the traits").

## (b) Structure

Stage two is section 8 of 11 despite "This is the part we did not expect"; it belongs right after the geometry. The dissociation theme is scattered across four sections and never named; give it a heading ("Weight-space distance is a bad proxy for behaviour") and put the rank sweep in it. The steering section is six results in one block with no subheads; the scoring section is one 400-word paragraph; the limits section is one 340-word paragraph ending in three unconnected sentences. Terms needing a one-clause gloss: LoRA, OCT stages, oblimin, parallel analysis, Tucker congruence, Procrustes. The Fisher gloss is the model to copy. Move to the companion: the alpha-units confession, the k=5 vs k=7 bullet, the sphere grand-mean caveat, the 0.70 vs 0.74 parenthetical, most of the hole.

## (c) Honesty and calibration

1. The applied section asserts the corrigible-flag reading the wiki has retracted ("rejected half is a refusal and chosen half complies, at ten times the base rate"), dramatises it with three examples, then retracts it two bullets later. The wiki's correction: the tail is enriched in refusal-containing pairs of both polarities; net preference for compliance 1.75 percentage points, about seven pairs of 400. State the robust version first.
2. The ceiling explanation is asserted in the dose paragraph and "retired" four paragraphs later, and the experiment that adjudicates it (matched-dose steering: raw 1.896 to 1.646, room-normalised amplification stronger at 0.795) is absent.
3. "Looping is a negative-alpha phenomenon" is contradicted by matched-dose-steering (amplifying sign loops more at matched dose).

Overclaims: "found things a keyword filter would not" (a refusal regex reproduces the tails; the fair claim is that the score found them without being told refusals exist); "ranked second of 27 directions" is uninterpretable (20 are random merges; say AUC 0.6488, beats all 20 nulls and a lexical baseline, one alignment direction above it); "five are what replicate" (use the wiki's "Five factors is a choice ... and the data does not pick it").

Underclaims / missing negatives: the reward-hacker null is absent (first-order: largest personality direction -0.0819 against widest random arm 0.0883, positive control 973 of 973 at z +7.82; arms at about 1% of chart length; column space 1.9% against 13.2%); the flag-training negative is missing from the TL;DR; the BFI lesson deserves promotion.

No clear "what this does not show" section. It should include: what reproduces across seeds is the training signal; directions do not transfer between initialisations; 13 optimizer steps on one shared 445-prompt pool as a stated design choice.

## (d) Completeness

Add: rank sweep; reward-hacks arms, scoring and column space; Fisher-metric factor analysis (Tucker 0.965, min 0.955, Big Five congruences move at most 0.059); matched-dose steering; the `unemotional` result (all five trait-free stage-two runs nearest `unemotional`, residual profiles r 0.86); judge repeat reliability; optionally module holography (one module of 248 classifies a trait's factor at 0.638 against 0.202 chance). Stage two should be framed against OCT's own prefill-attack argument (F1 0.79 to 0.95) as a qualification of a published claim. Cut or shorten: the hole (two sentences plus a link), the alpha-units confession, the optimised-data arm (one clause), the k=7 bullet.

## (e) Prior work

Cite: Persona Vectors (Chen et al., 2025, arXiv:2507.21509) where the activation construction is defined; the Assistant Axis (Lu et al., 2026) for the shared register direction; Open Character Training with a link and engagement with its stage-two evidence; the external reviewer for the N x N design (the wiki calls it "the strongest positive control in the project" and records that it replaced a hand-picked six-trait control at the review's request); the data-attribution lineage (Koh and Liang 2017, Grosse et al. 2023) with two sentences on how the directional derivative differs; SliderSpace (Gandikota et al., ICCV 2025); School of Reward Hacks (Taylor et al., 2025). Say what Persona Cartography's ten-adapter PCA found (near-flat) as the null the zoo beats.

## (f) Writing

Length fine; paragraph size is the problem (280, 330, 400, 340 words). No figures at all: place the factor chart, the two-panel scree with null arms, one spider, and the register table (already a table). Persona Cartography is CC BY-NC-ND, so reproduce your own figure rather than adapt theirs. Deep-link the seven companion pages and the load-bearing wiki slugs. Title: "Investigating" promises process, not result. Move the provenance line out of the opening. Mixed scales in the steering section (raw 1-7 points vs share of headroom) unlabelled. Keep: "The labels predict the geometry"; "That is an identity; the approximation is in reading a first-order score as a prediction of a trained adapter"; "Preference training on contrasting pairs makes opposites opposite; imitation of one's own transcripts makes everyone alike"; "We spent a night trying to make the method do work rather than discover things."

## (g) Sceptical commenter

1. "You put the Big Five in by construction" — partly preempted by the null zoos; disclose the enumerated anchor block and the unrun ablation; add the wiki's concession about constitutions being text about the traits.
2. "No true N; parallel analysis says nine" — replace "five are what replicate" and add the metric-independence result.
3. "Cosine 0.018 means no trait direction" — preempted by column space; promote it.
4. "13 steps on 445 shared prompts: you measure prompt style" — not preempted; say once that the permuted null trained on the same pool scores at chance and the double-centred Gram removes the shared component.
5. "One LLM judge on 24 prompts" — add repeat reliability.
6. "Frobenius is arbitrary" — add the Fisher-metric factor analysis.
7. "Does any of this catch a real problem?" — lead the applied section with the forecast, close with the failure and its diagnosis.
8. "You overclaimed the hole" — preempted; do it in two sentences.
9. "Why believe a weight-space map when your sliders show a functionally equivalent adapter at cosine 0.01?" — raised and never answered; the resolution is that the map measures the arrangement of one family made by one procedure, stable across seeds, stages, ranks and metrics, and the column-space result says what the invariant is.

## Proposed outline

1 Title and hook; 2 TL;DR with bullet one a finding; 3 What we built; 4 Factor structure (null zoos, k=5 table with Tucker glossed, five is a choice, Fisher rerun, shared-pool caveat); 5 What a trait's update actually is (seed floor, row vs column space, rank-sweep geometry); 6 Weight-space distance is not behaviour (rank 1, sliders, two stages in activations, Fisher norms; ends with the resolution); 7 What stage two does (promoted); 8 Does the map predict behaviour (units first, dose-response, sphere, matched dose, dials, BFI vs TRAIT); 9 Does it show up when you only prompt (artefact check); 10 Scoring data (identity, lineage, N x N credited, forecast); 11 Somebody else's data (Dolci, robust tails, probes, flag negative); 12 What this does not show; 13 Explore.

## Three changes

1. Add the rank sweep and give the dissociation its own section.
2. Promote stage two and make the TL;DR lead with a finding; include the `unemotional` result.
3. Fix the two assert-then-retract passages and add the results that adjudicate them (matched dose, reward-hacks null). Runner-up: figures.
