---
title: Additivity of the Big Five steering axes
summary: Ten matched-norm mixtures of the five named axes were steered and judged blind; the deviation from additivity is 0.53 of the predicted effect and about 2.5x the judge-noise floor, and opposing mixtures fail where reinforcing ones compose.
status: current
sources:
  - qwen35/analyse_additivity.py
  - qwen35/analysis/additivity.json
  - qwen35/phase10_runs/steer_spec_mix.json
  - qwen35/phase10_runs/steer_results_mix.json
  - qwen35/phase10_runs/judged_mix.json
  - qwen35/phase10_runs/judge_mix.log
  - qwen35/build_monitor_page.py
last_verified: 2026-09-16
tags: [behaviour, steering, additivity, curvature]
---

# Additivity of the Big Five steering axes

Every geometry result in the project validates the chart *at* the 134 adapters
or at the five axes built from them. Additivity asks whether the interior
between those points behaves the way its coordinates claim.

`qwen35/analyse_additivity.py` states the test:

> If steering along axis A moves the judged profile by dA, and axis B by dB,
> then a matched-norm combination of the two should move it by dA + dB -- but
> only if the map from weights to behaviour is linear over that region. The
> residual `r = d(A+B) - (dA + dB)` is the deviation from flatness, measured in
> behaviour rather than inferred from geometry.

## Design

`zoo-steermix.service`
(`Description=Additivity test: matched-norm mixtures of the five named axes,
inside the coherent band`) ran `steer_fix.py` on
`qwen35/phase10_runs/steer_spec_mix.json`: ten mixtures, five alphas each
(-2, -1, 0, +1, +2), the same 24-prompt battery. Output
`phase10_runs/steer_results_mix.json`, judged into
`phase10_runs/judged_mix.json`.

The ten mixtures (`steer_spec_mix.json#jobs[].name`, `ref` 0.8102592902648793,
`n` 100):

`mix_Extr+Agre`, `mix_Extr+Cons`, `mix_Extr+Emot`, `mix_Extr+Inte`,
`mix_Agre+Cons`, `mix_Agre+Emot`, `mix_Extr-Cons`, `mix_Agre-Inte`,
`mix_E+A+I`, `mix_Cons+Emot`.

Two controls keep it honest, in the script's words:

> The mixtures are renormalised to unit coefficient norm, so they are not simply
> larger perturbations than the singles; and the comparison is restricted to
> |alpha| <= 2, because at |alpha| = 4 the model loops verbatim on most prompts
> and additivity of wreckage is not a finding.
> — `qwen35/analyse_additivity.py`

The null is measured rather than assumed: "It is estimated from the alpha = 0
rows, which are the same untouched model under every direction, so their spread
is pure measurement error."

Judging: "1200 generations to judge from 10 traits ... 1200 judged, 0 failed
calls"; 210 judge calls, 60 repeats; repeat reliability r=0.760 (Extraversion),
0.826 (Agreeableness), 0.914 (Conscientiousness), 0.856 (EmotionalStability),
0.809 (Intellect) (`qwen35/phase10_runs/judge_mix.log`).

## Result

`qwen35/analysis/additivity.json` (40 rows: 10 mixtures x 4 non-zero alphas):

```
median_resid_over_effect  0.5282408112973758
median_resid_norm         0.776827975948343
noise_norm                0.31666612924087495
noise (per scale)         0.13931912433568944, 0.21098269247027546,
                          0.11470786693528087, 0.1078200342446346,
                          0.10756796902132218
```

So the residual from additivity is about **53% of the predicted effect** and its
median norm is roughly **2.5 times the judge-noise floor** (0.7768 against
0.3167). The script's own reading criterion:

> a ratio near 1 means the deviation from additivity is indistinguishable from
> judge noise and the chart is flat over its interior. Well above 1 means the
> interior is curved and coordinates do not compose.
> — `qwen35/analyse_additivity.py`

**Verdict: the coordinates do not compose.** The monitor page records this as
one of six failed pre-registered tests:

> **4. Do the coordinates compose?** verdict: fail
> **Success criterion:** steering along `axis A + axis B` at matched norm should
> move behaviour by the sum of what each moves alone, to within judge noise.
> Otherwise a coordinate vector does not predict behaviour anywhere except at
> the points we sampled.
> — `qwen35/build_monitor_page.py`

## The sign asymmetry

The failure is not uniform. The monitor page:

> The pattern is the right shape for curvature: reinforcing mixtures compose
> tolerably (0.13-0.15), opposing ones do not (1.17-1.46). A flat space would
> not care about the sign.
> — `qwen35/build_monitor_page.py`

Per-row `resid_over_effect` from `analysis/additivity.json#rows`:

| mixture | -2 | -1 | +1 | +2 |
|---|---|---|---|---|
| mix_Extr+Agre | 0.522 | 0.848 | 0.875 | 0.570 |
| mix_Extr+Cons | 0.372 | 0.521 | 0.612 | 0.391 |
| mix_Extr+Emot | 0.530 | 1.384 | 1.447 | 0.652 |
| mix_Extr+Inte | 0.300 | 0.114 | 0.514 | 0.321 |
| mix_Agre+Cons | 0.404 | 0.584 | 1.838 | 0.577 |
| mix_Agre+Emot | 0.459 | 0.428 | 0.492 | 0.132 |
| **mix_Extr-Cons** | **1.464** | **1.175** | 0.546 | **1.207** |
| mix_Agre-Inte | 0.707 | 0.481 | 0.724 | 0.470 |
| mix_E+A+I | 0.503 | 0.710 | 0.595 | 0.516 |
| mix_Cons+Emot | 0.150 | 0.328 | 0.647 | 0.526 |

(Values are rounded to three decimals here for width; the file carries full
precision at `analysis/additivity.json#rows[].resid_over_effect` — for example
the first cell is 0.521835326336742.)

The two subtractive mixtures are `mix_Extr-Cons` and `mix_Agre-Inte`. Only
`mix_Extr-Cons` shows the large residuals the monitor page describes; the
monitor page's stated ranges (0.13-0.15 reinforcing, 1.17-1.46 opposing) are the
extremes of the two groups rather than their means. The smallest residual in the
table, 0.132, is `mix_Agre+Emot` at +2, and 0.150 is `mix_Cons+Emot` at -2.

## What it does and does not establish

It establishes that a coordinate vector in the five-axis chart does **not**
predict judged behaviour at an interior point to within measurement error, at
least at |alpha| <= 2 on this battery with this judge. It does not establish
the *form* of the curvature; the sign asymmetry is suggestive of it but rests
on one clearly-failing mixture out of two subtractive ones.

It also does not contradict [[sphere-sweep]], which finds that *nearby*
directions give nearby personalities (rho = 0.65). Local continuity and global
additivity are different claims.

Related: [[steering-results]], [[sphere-sweep]], [[geometry-overview]],
[[glossary]].
