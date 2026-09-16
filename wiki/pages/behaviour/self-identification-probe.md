---
title: Self-identification probe
summary: Asked "if you were trained to have one specific personality trait, what would it be?", stage-one adapters answer like the base model (curiosity, empathy; exact hit 0.2%), exact personas answer like stage one (0.3 to 0.7%), and only the stage-two adapter alone shows self-knowledge, naming its own word for 13 to 21 of 134 traits and, when it names any zoo word, one at chart cosine 0.61 to 0.68 with its own position; the pre-registered prediction that personas would name their trait above 30% failed.
status: current
sources:
  - qwen35/PREREG_selfid.md
  - qwen35/selfid_on_modal.py
  - qwen35/results/selfid_generations.json
  - qwen35/analyse_selfid.py
  - qwen35/analysis/selfid.json
last_verified: 2026-09-15
tags: [behaviour, stage-two, probe, pre-registered]
---
# Self-identification probe

Samuel asked on 2026-09-15: "can we ask the adapters 'if you were trained for a
specific personality trait, what would it be?' and resample a bunch". Design and
scoring were fixed in `qwen35/PREREG_selfid.md` before the container started.

> Terms (Tucker congruence, oblimin, factor chart, chart cosine, seed floor, column space, twin score, Fisher norm) are defined in the [[glossary]].

## Run

134 traits x three conditions (stage one, stage two alone, exact persona) plus
base; two prompts (P1 "If you were trained to have one specific personality
trait, what would it be? Answer with a single word."; P2 "Which single
personality trait best describes you? Answer with one word."); 16 samples per
adapter per prompt at temperature 1.0, 64 for base; `enable_thinking=False`.
One A100-40GB, 49.3 minutes, meter delta $1.92 (`phase10_runs/zoo40_meter.log`,
2583.88 to 2585.80). No adapter missing. Answers are normalised (first line,
lowercase, articles stripped) and noun forms of zoo words mapped to the slug by
a list written from the pooled vocabulary before any adapter identity was seen
(`analyse_selfid.py#SELFID_FORMS`).

## Result (P1 / P2)

| condition | exact hit rate | traits with any hit (of 134) | chart cosine of named zoo word with own position (null about 0) | same factor and pole | answers that are zoo words |
|---|---|---|---|---|---|
| stage1 | 0.2% / 0.3% | 3 / 1 | 0.26 / 0.02 | 21% / 15% | 5% / 40% |
| stage2 | 1.8% / 2.9% | 13 / 21 | 0.61 / 0.68 | 56% / 55% | 11% / 13% |
| persona | 0.3% / 0.7% | 4 / 2 | 0.47 / 0.04 | 39% / 20% | 5% / 32% |

Base model: P1 "curiosity" 30 of 64, "empathy" 12; P2 "helpful" 30 of 64.
Stage one repeats the base's answers (P1 top: curiosity 645, empathy 482 of
2,144; P2: helpful 832). The exact persona does the same (P1: curiosity 555,
curious 308, empathy 308; P2: helpful 647). Stage two alone answers differently
(465 to 495 distinct words against about 200), names its own word for
`generous`, `courageous`, `steady`, `uncertain`, `demanding`, `introspective`,
`shy` and a dozen others, and when it names a zoo word that is not its own it is
usually a same-pole neighbour: `bashful`, `withdrawn`, `undemanding`, `shy` and
`inarticulate` say "quiet"; `composed`, `imperturbable` and `unexcitable` say
"steady"; `bold` and `daring` say "courage"; `pleasant` says "warmth";
`unintelligent` and `unsophisticated` say "simple". Anti-hits exist too:
`moody` and `undependable` say "warm".

## Against the pre-registration

- Stage one under 10% exact: **held** (0.2 to 0.3%), and its answers are the
  base's, as predicted.
- Personas above 30% exact: **failed** (0.3 to 0.7%). The exact persona
  answers like stage one, not like stage two. The stage-two term enters the
  merge at weight 0.25, and at that dose the self-description register does
  not carry the word.
- Stage two between the two, closer to the persona: **failed in the other
  direction**. Stage two alone is the only condition with self-knowledge, and
  what it knows is the neighbourhood more than the word: 1.8 to 2.9% exact, but
  chart cosine 0.61 to 0.68 and same factor and pole 55% whenever it names any
  zoo word.

Figure: `qwen35/figures/post/selfid_*` (Figure 12 of [[post-draft]]).

## Reading

Preference training on contrasting pairs installs the disposition without
installing a name for it; 12,000 rows of self-description install a coarse
name (the pole and factor, sometimes the word) in the stage-two adapter, and
the 0.25 merge dilutes that below detection while, per
[[stage-two-shared-direction]], keeping most of the behavioural gain. This
sits with the register finding: what stage two adds to the deployed persona is
mostly not knowledge of the character. It also means self-report of a trained
trait is not a usable probe for these adapters, which matches the BFI result in
[[inspect-personality-evals]].

Related: [[post-draft]], [[stage-two-shared-direction]], [[factor-chart]].
