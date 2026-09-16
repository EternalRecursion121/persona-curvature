---
title: Steering the unnamed direction
summary: The widest hole in the trait lexicon was steered against a shuffled-coefficient control and a random direction in the same subspace; it produced a coherent persona with less damage than the random control, and its chart-predicted Big Five profile matched the judge on all five signs at r = 0.81.
status: superseded
sources:
  - qwen35/analyse_alien_steer.py
  - qwen35/analysis/alien_steer.json
  - qwen35/analysis/alien_match.json
  - qwen35/analysis/alien.json
  - qwen35/analysis/direction_gaps.json
  - qwen35/phase10_runs/alien_spec.json
  - qwen35/phase10_runs/alien_results.json
  - qwen35/phase10_runs/judged_alien.json
  - qwen35/build_blog_page.py
last_verified: 2026-09-16
tags: [behaviour, steering, alien-direction, controls]
---

# Steering the unnamed direction

> **Status, 2026-09-08.** This page is the principal-component record and is
> superseded as the current claim by [[alien-direction-factor-chart]], which runs
> the identical design on the direction the five-factor chart picks out
> ([[factor-first-migration]]). Three of the readings below do not reproduce
> there. The damage ordering reverses: on the factor chart the unnamed direction
> is the only one of the three that degenerates, 0.4166666666666667 of the 24
> responses looping at alpha -2 against 0.0 for both controls, where here the
> random control was the worst at 0.3333333333333333. The chart-predicts-behaviour
> result weakens from 5 of 5 signs at r = 0.8142138888404568 to 3 of 5 at
> r = 0.7438221950472093. And there the random control in the same span scores 3
> of 5 at r = 0.7343440116564208 against the same prediction, so the margin over
> control that this page rests on is not reproduced. Everything below remains an
> accurate record of the principal-component run.

The geometry side of this project searched the span of the 134 trait adapters
for the direction furthest from every English trait word — the widest hole in
the lexicon. Where that direction comes from is [[hole-words]]. This page is
what happened when it was steered.

## The question and the two controls

`qwen35/analyse_alien_steer.py` states both readings in advance:

> Two readings are pre-registered and both are results. If the unnamed direction
> gives a self-consistent judged profile with damage at control level, the model
> can hold characters English has no word for. If its damage rises to the
> shuffle control's level, the personality manifold is a thin sheet and the
> space around it is not habitable.

Three norm-matched directions, all steered at the same strengths:

- **`alien_k5`** — "the deepest hole in the top-5 principal subspace, 52.5
  degrees from the nearest of the 134 trait lines".
- **`alien_shuffle`** — "the SAME coefficient multiset permuted across traits --
  same number of adapters, same coefficient magnitudes, same sum-to-zero
  contrast structure, pointing nowhere in particular. This is the control for
  'unusual mixtures break the model'."
- **`span_random`** — "a uniformly random unit direction in the same subspace,
  which lands much closer to a named trait, as random directions do."

Angles and nearest words (`analysis/alien_steer.json#gap` and `#nearest`):

| direction | gap (deg) | nearest trait |
|---|---|---|
| alien_k5 | 52.51479642189158 | anxious |
| alien_shuffle | 31.85394811935244 | thrifty |
| span_random | 28.06948829321276 | mothering |

**Two angles exist for the same direction and they are not the same
measurement.** The 52.5 degrees above is the gap *inside the top-5 principal
subspace* (`analysis/alien.json#alien_k5.gap_deg` = 52.51479642189155). The blog
page's card labels it "68.9 degrees from the nearest word", which is
`analysis/direction_gaps.json#alien_k5.deg` = 68.93287342956332 — measured "in
the full sketch space with each adjective treated as a line"
(`qwen35/build_blog_page.py`), where the nearest word is *relaxed*, with
*anxious* second at 69.15932422427636. Quote whichever you mean, and say which.

## Generation and judging

`zoo-alien.service`
(`Description=Steering: alien direction + shuffle/random controls`) ran
`steer_fix.py --spec phase10_runs/alien_spec.json --out
phase10_runs/alien_results.json`, so this corpus is thinking-off at 512 tokens
like the corrected steering runs. Alphas -2, -1, 0, +1, +2 on the 24-prompt
battery; judged into `phase10_runs/judged_alien.json`.

Degeneration is measured with the same looping definition as the steering
pages: "A response loops if some n-word window repeats at least k times", n=10,
k=4 (`qwen35/analyse_alien_steer.py`).

## Damage at matched strength

Fraction of the 24 responses that loop
(`analysis/alien_steer.json#<direction>.degen`):

| alpha | alien_k5 | alien_shuffle | span_random |
|---|---|---|---|
| -2 | 0.08333333333333333 | 0.0 | 0.3333333333333333 |
| -1 | 0.0 | 0.0 | 0.041666666666666664 |
| 0 | 0.0 | 0.0 | 0.0 |
| +1 | 0.0 | 0.0 | 0.0 |
| +2 | 0.041666666666666664 | 0.0 | 0.08333333333333333 |

The blog page's reading: "The shuffled control — identical coefficients,
permuted across traits — is harmless and directionless. The random direction in
the same subspace does *more* damage than the unnamed one, not less."
(`qwen35/build_blog_page.py`).

So the pre-registered reading that fires is the first one: the unnamed direction
is habitable.

## The chart predicted the behaviour

`qwen35/analysis/alien_match.json` compares the direction's chart coordinates,
computed from weights alone before any text existed, with how the blind judge's
scores actually moved from alpha -2 to +2:

| scale | chart said (`pred`) | judge saw (`obs`) |
|---|---|---|
| Extraversion | -0.14540832715549373 | -0.583333333333333 |
| Agreeableness | -0.43127553903055127 | -0.5 |
| Conscientiousness | -0.6965250222590053 | -1.125 |
| EmotionalStability | +0.72343905236201 | +0.43115942028985454 |
| Intellect | +0.4084175175927897 | +1.583333333333334 |

`signs`: 5 of 5. `r`: 0.8142138888404568.

The blog page's gloss: "Nobody chose this direction for what it would do; it was
chosen for being as far as possible from every English trait word."
(`qwen35/build_blog_page.py`). Its one-line character sketch: "Less
conscientious, less agreeable, more emotionally stable, more intellectual."

## Judged curves

Full per-alpha Big Five means are in
`analysis/alien_steer.json#<direction>.curve`. For `alien_k5` the largest
movement is Intellect, 4.458333333333333 at -2 rising monotonically to
6.041666666666667 at +2, against Conscientiousness falling 5.625 to 4.5. The
shuffle control's Intellect is flat (5.541666666666667 at -2, 5.416666666666667
at +2) while its Extraversion rises 3.9166666666666665 to 5.0 — i.e. the
shuffled control is not inert on every scale, only directionless with respect to
the alien direction's prediction.

Mean response lengths (`#len`) show the same asymmetry: `alien_shuffle` stays
between 1664 and 1870 characters across the grid, while `span_random` swings
from 979.0416666666666 at -1 to 2129.0 at -2.

## Where the numbers are used

The blog page section "Where no word goes" (`qwen35/build_blog_page.py`,
`alien_verdict()` and `alien_card()`) renders both tables above, and pulls two
verbatim generations at alpha -2 and +2 straight from
`analysis/blog_corpus.json#runs.alien_k5` rather than from the hand-adjudicated
quote set, "because this one was steered after that pass"
(`qwen35/build_blog_page.py`).

The alien direction is also one of the three targets of the data-optimisation
experiment; see [[optimised-data-and-verify]].

Related: [[hole-words]], [[steering-results]], [[sphere-sweep]],
[[optimised-data-and-verify]], [[geometry-overview]], [[glossary]].

