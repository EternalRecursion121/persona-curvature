---
title: The sphere sweep — 72 directions nobody chose
summary: Seventy-two Fibonacci-lattice directions on the sphere of the top three principal components, steered at alpha = 1.5 and judged blind; 48 of 72 loop on no prompt, and angular distance predicts judged-profile distance at rho = 0.65.
status: superseded
sources:
  - qwen35/sphere_sweep.py
  - qwen35/build_sphere_spec.py
  - qwen35/sphere_to_eval.py
  - qwen35/build_sphere_page.py
  - qwen35/analysis/sphere_layout.json
  - qwen35/analysis/sphere_page.json
  - qwen35/phase10_runs/sphere_spec.json
  - qwen35/phase10_runs/sphere_results.json
  - qwen35/phase10_runs/judged_sphere.json
  - qwen35/phase10_runs/judge_sphere.log
  - qwen35/build_blog_page.py
  - qwen35/analysis/sphere_isokl.json
last_verified: 2026-09-10
tags: [behaviour, steering, sampling, controls]
---

# The sphere sweep — 72 directions nobody chose

> **Status, 2026-09-08.** The frame is superseded, the data are not. Samuel's
> decision of 2026-09-08 makes the factor analysis primary and the
> principal-component chart secondary, so the sphere was resampled on the
> factor chart: see [[sphere-sweep-factor-chart]]. This page remains the record
> of the 2026-09-01 principal-component sphere, whose generations, judgings and
> numbers all still stand; what changed is which three-dimensional subspace the
> lattice is laid on.
>
> **Addendum, 2026-09-10.** This lattice was re-steered at equal measured KL per
> token rather than equal alpha: see [[sphere-sweep-iso-kl]]. The alpha 1.5 used
> here delivered doses spread over 2.54x across the 72 points, and removing that
> spread leaves this page's rho at 0.6523 against 0.6511. The smoothness result
> is not a dose artefact. What the dose-controlled run does change is the
> degeneration reading and the claim that curvature predicts judged
> displacement.

Every other steering result in this project is on a direction chosen for a
reason. That makes the evidence circular in one specific way, which
`qwen35/build_sphere_spec.py` names:

> Steering along a named axis and finding a coherent persona proves little if
> EVERY direction gives a coherent persona; the only way to know is to sample
> directions nobody chose.

## Design

- **72 points**, drawn from a Fibonacci lattice on the unit sphere of the top
  three principal components — "which spreads them far more evenly over the
  sphere than random draws of the same count"
  (`qwen35/build_sphere_spec.py`). `N = 72`.
- **alpha = 1.5**, one strength for every point:
  `ALPHA = 1.5  # inside the intact range for every direction`
  (`qwen35/build_sphere_spec.py`; also `analysis/sphere_layout.json#alpha`).
- **8 prompts**, `PROMPT_IDX = [0, 1, 5, 9, 11, 14, 15, 22]` from the 24-prompt
  battery — a subset, not the whole thing.
- **Landmarks** appended so the map has fixed points: the five Big Five keying
  axes projected into the subspace plus the alien direction
  (`analysis/sphere_layout.json#landmarks`), and the 134 trait words on the
  same sphere (`#traits`, 134 entries).
- The three components carry `var3` = 0.12668915056443317,
  0.11252214481119611, 0.04955491287468896 of the variance
  (`analysis/sphere_layout.json#var3`).

The run is one container rather than 72, because every direction here is a
combination of exactly three basis deltas:

> steer_fix builds each direction by reading all 134 adapters from the volume:
> about 43 GB per direction. That is fine for nine directions and absurd for
> seventy-two ... So the adapters are read ONCE, collapsed immediately into
> three dense basis deltas, and then every sampled direction is a cheap weighted
> sum of those three.
> — `qwen35/sphere_sweep.py`

Base weights are held as a CPU copy and restored between directions "so
seventy-two rounds of bfloat16 addition cannot accumulate drift into the later
samples" (`qwen35/sphere_sweep.py`).

`zoo-sphere.service` (`Description=Sphere sweep: 72 sampled directions in one
container`) wrote `phase10_runs/sphere_results.json` — the log ends "wrote
phase10_runs/sphere_results.json: 72 points"
(`qwen35/phase10_runs/sphere.log`).

**Blinding of the judge is structural.** `qwen35/sphere_to_eval.py` reshapes the
sweep into the judge's input format and notes: "the judge never sees that the
'trait' names are coordinates on a sphere." Points are named `S000`-`S071`.
Judging: "576 generations to judge from 72 traits ... 576 judged, 0 failed
calls"; 101 calls, 28 repeats; repeat reliability 0.907 / 0.902 / 0.887 / 0.861
/ 0.922 (`qwen35/phase10_runs/judge_sphere.log`). 72 x 8 = 576.

## Result 1: most of the space is habitable

`qwen35/analysis/sphere_page.json#coherence`:

```
none       48
any        24
worst      0.25
mean       0.046875
len_mean   1261.234375
```

The blog page's reading: "at alpha = +1.5, **48 of the 72** sampled directions
produce no looping at all, and the worst of them loses 25% of its responses — a
mean rate of 4.7% across the sphere. So most directions through this space *do*
give intact output, and the coherence of the ones we chose is not, by itself,
evidence of anything." (`qwen35/build_blog_page.py`).

That is a self-correcting result: it removes coherence as evidence for the
chosen directions and shifts the weight onto selectivity and predictive
coordinates instead. The blog page says so: "What has to carry the weight
instead is specificity: whether a direction moves the scale it is supposed to
and leaves the others alone, and whether its coordinates predict its behaviour
in advance."

## Result 2: personality varies continuously

`qwen35/analysis/sphere_page.json#smooth`:

```
rho        0.6511417860626274
n_pairs    2556
n_scored   72
near_mean  0.9661257055739045
far_mean   2.80790992900049
```

Over all 2,556 pairs of the 72 scored directions, angular distance and judged
Big Five profile distance correlate at rho = +0.65. Directions less than 30
degrees apart differ by 0.97 on the Big Five; directions more than 120 degrees
apart differ by 2.81 (`qwen35/build_blog_page.py`, reading
`sphere_page.json#smooth`).

The blog page's pull quote: "Personality varies continuously across this space.
Walk a short way and the character changes a little; walk to the far side and it
changes a lot. That is what makes it a map rather than a list."

Note that this is a *local continuity* claim and does not conflict with
[[additivity]], which finds that coordinates do not compose additively over the
interior.

## Result 3: the coverage is lopsided

Counting which scale the judge rated highest at each of the 72 points
(`analysis/sphere_page.json#judged.<point>.top`):

| top-rated scale | points |
|---|---|
| Agreeableness | 32 |
| Intellect | 18 |
| Conscientiousness | 14 |
| Extraversion | 6 |
| EmotionalStability | 2 |

The blog page: "Nearly half the sphere reads as agreeable and almost none of it
as emotionally stable — the base model's own pull is strong enough that most
directions through this space land somewhere pleasant."
(`qwen35/build_blog_page.py`).

## How close the sampled points are to real words

Each judged point carries `nearest` and `nearest_deg` — the angle to the nearest
of the 134 trait lines *within this three-dimensional subspace*
(`qwen35/build_sphere_page.py`). Across the 72 points those angles run from
0.38236976413731133 to 20.741783210516825 degrees
(`analysis/sphere_page.json#judged`). Inside a three-dimensional subspace
crowded with 134 trait lines, a sampled direction is always near some word; the
much larger gaps of [[hole-words]] and [[alien-direction-steering]] are measured
in a different space and are not comparable to these.

## Where it is rendered

`qwen35/build_sphere_page.py` folds the layout, generations and judging into
`analysis/sphere_page.json` (785 KB), which `qwen35/build_blog_page.py` renders
as the clickable "Sampling the space" panel of the current blog page. Every
point's eight generations ship with the page, so a reader can read what the
model becomes anywhere on the sphere. Point colour is the top-rated scale, and
the caption states coverage explicitly rather than assuming it: 72 of 72 scored
(`sphere_caption()` in `qwen35/build_blog_page.py`).

`qwen35/phase10_runs/sphere_sweep_spec.json` (21 KB, 2026-09-01 14:58) is the
spec the run actually consumed: it is the default of the `sphere_sweep.py`
local entrypoint (`def main(spec: str = "phase10_runs/sphere_sweep_spec.json"`)
and holds the three basis coefficient dicts plus the 72 unit vectors.
`sphere_spec.json` (392 KB) is the steer_fix-shaped expansion, one job per
point with its full 134-coefficient vector; nothing read it during the sweep,
but it is what the factor-chart comparison in
[[sphere-sweep-factor-chart]] reads to place these 72 directions on the new
chart. (An earlier revision of this page had these two the wrong way round.)

Related: [[sphere-sweep-factor-chart]], [[sphere-sweep-iso-kl]],
[[steering-results]], [[additivity]],
[[alien-direction-steering]], [[hole-words]], [[geometry-overview]], [[glossary]].
