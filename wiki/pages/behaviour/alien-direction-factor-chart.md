---
title: Steering the unnamed direction on the factor chart
summary: The factor chart's deepest hole was steered with the same design and controls as the principal-component one; it gives a monotone Agreeableness-down, Conscientiousness-up profile, but it is the only one of the three directions that degenerates (42% looping at alpha -2), it matches the chart's prediction on 3 of 5 signs against 5 of 5 before, and the random control in the same span matches almost as well.
status: current
sources:
  - qwen35/build_alien_spec_fa.py
  - qwen35/phase10_runs/alien_spec_fa.json
  - qwen35/phase10_runs/alien_results_fa.json
  - qwen35/phase10_runs/judged_alien_fa.json
  - qwen35/analyse_alien_steer_fa.py
  - qwen35/analysis/alien_steer_fa.json
  - qwen35/analysis/alien_match_fa.json
  - qwen35/analysis/alien_fa.json
  - qwen35/analysis/alien_steer.json
  - qwen35/analysis/alien_match.json
  - qwen35/steer_fix.py
  - qwen35/build_blog_page.py
  - qwen35/analysis/persona_sliders.json#behaviour.hole
last_verified: 2026-09-10
tags: [behaviour, steering, alien-direction, controls, factor-chart]
---

# Steering the unnamed direction on the factor chart

Where the direction comes from is [[hole-words-factor-chart]]. This page is what
happened when it was steered. The principal-component version of the same
experiment, which this replaces as the current claim, is
[[alien-direction-steering]].

## The design, unchanged

`qwen35/build_alien_spec_fa.py` copies `build_alien_spec.py` exactly: the same
alphas `-2, -1, 0, +1, +2`, the same reference norm
`0.8102592902648793` taken from `phase10_runs/steer_spec.json`, the same
24-prompt battery, and the same two controls. `zoo-alien-fa.service` is a copy of
`zoo-alien.service` and ran

```
modal run --detach steer_fix.py --spec phase10_runs/alien_spec_fa.json \
    --out phase10_runs/alien_results_fa.json
```

so this corpus is thinking-off at 512 tokens like every corrected steering run
([[thinking-default-withdrawals]]). `steer_fix.py` renormalises the merged delta
to unit Frobenius over the whole model before scaling by `alpha * ref`, so the
coefficient scale of a direction does not set the dose - alpha does, in the same
units as every published direction. Judged blind into
`phase10_runs/judged_alien_fa.json` (Claude Sonnet 4.5 via OpenRouter, 360
records, 0 failed calls; repeat reliability r = 0.860 to 0.885 across the five
scales on n = 18 repeats).

The three directions and where they sit, `analysis/alien_steer_fa.json#gap` and
`#nearest`:

| direction | gap in the chart (deg) | nearest trait | fraction of the vector inside the chart |
|---|---|---|---|
| alien_fa | 54.763780413807424 | immodest | 1.0 |
| alien_fa_shuffle | 17.838475388996727 | timid | 0.3722185593628015 |
| span_random_fa | 22.34771620960743 | helpful | 1.0 |

The shuffle control's gap is measured on the 37% of it that lies inside the
chart (`phase10_runs/alien_spec_fa.json#jobs[1].chart_coords`), so 17.8 and 54.8
are not the same measurement; permuting the coefficients across traits mostly
moves the direction *out* of the five-factor span rather than around inside it.
That was true of the principal-component shuffle too and is what makes it a
control for "unusual mixtures break the model" rather than for direction.

## Damage at matched strength

Fraction of the 24 responses that loop, same definition as every steering page
(a 10-word window repeating four times), `analysis/alien_steer_fa.json#<direction>.degen`:

| alpha | alien_fa | alien_fa_shuffle | span_random_fa |
|---|---|---|---|
| -2 | **0.4166666666666667** | 0.0 | 0.0 |
| -1 | 0.0 | 0.0 | 0.0 |
| 0 | 0.0 | 0.0 | 0.0 |
| +1 | 0.0 | 0.0 | 0.0 |
| +2 | 0.0 | 0.0 | 0.0 |

**This reverses the principal-component reading.** There the blog page could say
"The random direction in the same subspace does *more* damage than the unnamed
one, not less" - `analysis/alien_steer.json` gives `span_random` 0.3333333333333333
at alpha -2 against 0.08333333333333333 for `alien_k5`, with the shuffle at 0.0
throughout. On the factor chart the unnamed direction is the **only** one of the
three that degenerates anywhere, and it does so on 10 of 24 responses at alpha
-2. At every other strength it is intact, and so is everything else.

So the pre-registered pair of readings splits rather than resolving. Within
|alpha| <= 1 and at +2 the direction is habitable and gives the coherent profile
below; at -2 it is not. The honest summary is that the factor chart's deepest
hole has a habitable side and a thin one, and the thin side is the negative pole.

## The judged curves

`analysis/alien_steer_fa.json#alien_fa.curve`, blind Big Five means over the 24
prompts. The alpha -2 row is the one where 42% of the responses loop and should
be read as a mean over partly degenerate text.

| alpha | Extraversion | Agreeableness | Conscientiousness | EmotionalStability | Intellect |
|---|---|---|---|---|---|
| -2 | 4.0 | 5.208333333333333 | 3.4166666666666665 | 5.125 | 5.583333333333333 |
| -1 | 4.041666666666667 | 5.583333333333333 | 4.791666666666667 | 5.125 | 5.625 |
| 0 | 4.291666666666667 | 5.166666666666667 | 5.75 | 4.916666666666667 | 5.541666666666667 |
| +1 | 4.208333333333333 | 4.291666666666667 | 5.958333333333333 | 4.291666666666667 | 5.25 |
| +2 | 4.083333333333333 | 3.875 | 6.083333333333333 | 4.041666666666667 | 4.875 |

Conscientiousness rises monotonically from `3.4166666666666665` to
`6.083333333333333`, the largest movement of any scale on any of the three
directions. Agreeableness falls from `5.583333333333333` at -1 to `3.875` at +2
and Emotional Stability from `5.125` to `4.041666666666667`. Extraversion barely
moves at all (`4.0` to `4.083333333333333`).

Mean response lengths (`#alien_fa.len`) run `1880.2916666666667`,
`1373.0`, `1845.25`, `1936.875`, `1766.4166666666667` - no collapse, unlike
`span_random_fa`, which falls to `880.75` characters at +2 and
`1076.2083333333333` at -2 without looping.

## Did the chart predict the behaviour?

`analysis/alien_match_fa.json`. `pred` is the direction's coordinates on the five
named Big Five axes, computed from weights alone before any text existed, exactly
from `results/gram_sweep.npz` (`analyse_alien_steer_fa.py:big_five_coords`);
`obs` is how the blind judge's mean moved from alpha -2 to +2.

| scale | chart said (`pred`) | judge saw (`obs`) | |
|---|---|---|---|
| Extraversion | -0.007337834635425705 | +0.08333333333333304 | miss |
| Agreeableness | +0.06798030634328894 | -1.333333333333333 | miss |
| Conscientiousness | +0.5397198112316589 | +2.6666666666666665 | match |
| EmotionalStability | -0.7824710665309399 | -1.083333333333333 | match |
| Intellect | -0.5253713266834625 | -0.708333333333333 | match |

`signs`: **3 of 5**. `r`: **0.7438221950472093**.

Against the principal-component run's 5 of 5 and
`analysis/alien_match.json#r = 0.8142138888404568`, this is weaker. Two things
are worth saying about it rather than around it:

- the two misses are the two scales the chart barely committed to. The predicted
  Extraversion coordinate is `-0.007337834635425705` and Agreeableness
  `+0.06798030634328894`, an order of magnitude smaller than the other three;
  the three scales the chart did commit to all match, and the largest predicted
  magnitude (Emotional Stability) is a match.
- **the controls match nearly as well.** Scored against the same `pred`,
  `span_random_fa` also gets 3 of 5 signs at r = `0.7343440116564208` and
  `alien_fa_shuffle` gets 3 of 5 at r = `0.4803937317774778`
  (`analysis/alien_steer_fa.json#<direction>.signs`, `#<direction>.r`). On the PC
  chart the shuffle control managed 2 of 5. The reason is visible in the table
  above: Conscientiousness rises by `+2.6666666666666665` under `alien_fa` and
  `+2.583333333333333` under `span_random_fa`, and that one shared movement
  carries most of the correlation for both. **On this evidence the r for the
  unnamed direction is not evidence that the chart predicted anything
  direction-specific.** The comparison the PC page could make - prediction beats
  control - does not reproduce.

## What this does and does not establish

It establishes that the deepest hole in the five-factor chart is steerable, gives
a monotone and legible dose-response on three scales, and does not break the
model except at its negative extreme. It does **not** establish that the chart
predicts the behaviour of a direction chosen for its angle, because a random
direction in the same span was predicted about as well by the same coordinates.
Whether that is a property of this chart or of the 24-prompt battery is untested.

## Where the numbers are used

`qwen35/build_blog_page.py`: `coverage_svg()` now draws the factor-chart k sweep
as the primary curve with the principal-component gap behind it as a dashed line
at the same k, and `alien_card()` reads the `_fa` files, quotes
`phase10_runs/alien_results_fa.json` directly at alpha -2 and +2, generates its
one-line character sketch from `analysis/alien_match_fa.json#pred`, and carries
one generated sentence comparing the two charts. The card falls back to the
principal-component version if the `_fa` files are absent. `alien_verdict()` and
the surrounding prose still read the principal-component files and are owned
elsewhere; see the register in [[factor-first-migration]].

## The three directions on the model's own metric

[[fisher-norms]] gives the alien direction a Fisher norm of 0.2593, its
in-span random control 0.2485 and its shuffled control 0.0720
(`qwen35/analysis/fisher_norms.json#directions`), which puts them at ranks 44,
47 and 120 of 124. The alien direction and the random direction in the same
span are the same steepness to within 5 per cent (0.2593 against 0.2485), which is the same story the
judged match tells; the shuffled control, which destroys the span, is 3.6 times
flatter. Note that the alien direction is the one of the three that degenerates
(41.7 per cent looping at alpha -2) while the shuffled control does not, so its
Fisher norm does not explain the difference either: F did not predict
degeneration anywhere it was tested.

Related: [[alien-direction-steering]], [[hole-words-factor-chart]],
[[hole-words]], [[factor-chart]], [[steering-results]], [[fisher-norms]],
[[thinking-default-withdrawals]], [[judged-evaluations]].

## The activation-space version of the same test

[[persona-sliders]] is experiment S1 of [[paper-reading-2026-09-09]] and was
pre-registered as the test that could revive or bury this direction: a LoRA
trained to the hole expressed in the model's own residual stream cannot fail the
activation transplant by construction. It does not revive it, and the way it fails
repeats this page's own caveat exactly. Scored against **this page's `pred`**
(`qwen35/analysis/alien_match_fa.json#pred`, the same weight-space coordinates
the merge was scored against), the hole slider gets **5 of 5 signs at r
`+0.8614`**, better than the merge's 3 of 5 at `0.7438221950472093` -- but
`hole_shuf2`, a slider trained to the same coefficients permuted across traits,
gets **5 of 5 at r `+0.8245`**. The judged five-scale profiles say the same: the
hole is closer to `hole_shuf2` (Euclidean `0.4082`) than the two controls are to
each other (`0.7336`)
(`qwen35/analysis/persona_sliders.json#behaviour.hole`). The prediction still does
not discriminate hole from control.
