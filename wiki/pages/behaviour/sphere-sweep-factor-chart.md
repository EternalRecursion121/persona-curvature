---
title: The sphere sweep on the factor chart
summary: The 72-direction sphere resampled on the span of the top three oblimin factors after the 2026-09-08 decision to make the factor analysis primary; 45 of 72 loop on no prompt, angular distance predicts judged-profile distance at rho = 0.68, and the judged field agrees with the principal-component sphere at r = 0.71 to 0.82 on three of five scales where the two subspaces actually overlap.
status: current
sources:
  - qwen35/fa_chart.py
  - qwen35/build_sphere_spec_fa.py
  - qwen35/sphere_sweep.py
  - qwen35/sphere_to_eval.py
  - qwen35/build_sphere_page.py
  - qwen35/analysis/sphere_layout_fa.json
  - qwen35/analysis/sphere_page_fa.json
  - qwen35/phase10_runs/sphere_spec_fa.json
  - qwen35/phase10_runs/sphere_sweep_spec_fa.json
  - qwen35/phase10_runs/sphere_results_fa.json
  - qwen35/phase10_runs/judged_sphere_fa.json
  - qwen35/phase10_runs/sphere_fa.log
  - qwen35/phase10_runs/judge_sphere_fa.log
  - qwen35/results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin
  - qwen35/phase10_runs/steer_spec.json#ref
  - qwen35/phase10_runs/steer_spec2_7a.json
  - qwen35/build_blog_page.py
  - qwen35/analysis/sphere_isokl.json
last_verified: 2026-09-10
tags: [behaviour, steering, sampling, controls, factors]
---

# The sphere sweep on the factor chart

Samuel's decision of 2026-09-08 makes the factor analysis the primary frame for
the whole project and the principal-component chart secondary. The sphere sweep
was the largest behavioural result stated in principal-component coordinates, so
it was re-run on the factor chart. This page is the record of that run.
[[sphere-sweep]] remains the record of the 2026-09-01 principal-component
sphere; its data are unchanged and its numbers still stand.

The rendered page is served at **/sphere-fa.html**
(`qwen35/sphere_page/index_fa.html`, 881,709 bytes, copied by
`wiki/tools/build_site.py` `STATIC_FILES`). Every point's eight generations ship
inside it, so a reader can click any of the 72 directions and read the model
there.

## What the chart is

`qwen35/fa_chart.py` fixes one convention for every analysis that places
directions in factor space, and this sweep uses nothing else:

> The factors are oblique, so the chart uses an orthonormal basis of their
> 5-dimensional span obtained by Gram-Schmidt in the G inner product, in the
> order above (Warmth first). ORDER MATTERS and is fixed here.
> — `qwen35/fa_chart.py`

The five factor directions are the oblimin PAF factors as they were actually
steered in phase 10, `phase10_runs/steer_spec2_7a.json` jobs `FA_Warmth`,
`FA_Competence`, `FA_FearfulWithdrawal`, `FA_Arousal`, `FA_Imagination`, ordered
by oblimin sum of squared loadings
(`results/fa_qwen35.json#solutions.centred_k5.ss_loadings.oblimin` =
10.76210106481124, 8.421873312354228, 7.020338408333558, 6.814511494042316,
5.798806928556024; also copied to
`analysis/sphere_layout_fa.json#ss_loadings_oblimin`). Inner products come from
the exact Gram `results/gram_sweep.npz`, never from coordinates.

The sphere is the unit sphere of the span of the **first three** basis vectors —
Warmth, Competence, Timidity
(`analysis/sphere_layout_fa.json#axis_names`).

## What was held fixed, and why

`qwen35/build_sphere_spec_fa.py` changes the subspace and nothing else, so the
two spheres are comparable point for point:

- **The same 72-point Fibonacci lattice.** Identical function, identical `N =
  72`, so the two runs differ only in which subspace the lattice is laid on.
- **alpha = 1.5** (`analysis/sphere_layout_fa.json#alpha`).
- **The same 8 prompts**, `PROMPT_IDX = [0, 1, 5, 9, 11, 14, 15, 22]`.
- **The same reference norm.** `ref` = 0.8102592902648793
  (`analysis/sphere_layout_fa.json#ref`), taken from
  `phase10_runs/steer_spec.json#ref` — what the principal-component sphere used
  — and deliberately *not* the 0.8078003190997738 of
  `phase10_runs/steer_spec2_7a.json#ref`. The difference is 0.3%, but "the same
  steering magnitude" is the comparability claim, so it is the PC sphere's
  number that was kept.

Each sampled direction's coefficients over the 134 adapters are `u @ basis[:3]`,
which is unit norm in G by construction, written to
`phase10_runs/sphere_spec_fa.json`. The container's own input is
`phase10_runs/sphere_sweep_spec_fa.json`: three basis coefficient dicts plus the
72 unit vectors.

## Landmarks

Fourteen named directions were projected into the same three-space and
normalised (`analysis/sphere_layout_fa.json#landmarks`). Each is a direction the
study has actually generated from — the five factors from
`steer_spec2_7a.json`, everything else from `steer_spec.json` — not a
construction. `#landmark_cos` records how much of each one this three-space
actually sees (the cosine between the direction and its own projection):

| landmark | cosine with the top-3 factor space |
|---|---|
| factor_FA_Warmth | 1.0 |
| factor_FA_Competence | 1.0 |
| factor_FA_FearfulWithdrawal | 1.0 |
| factor_FA_Arousal | 0.7658 |
| factor_FA_Imagination | 0.4392 |
| axis_Extraversion | 0.7409 |
| axis_Agreeableness | 0.9556 |
| axis_Conscientiousness | 0.9220 |
| axis_EmotionalStability | 0.6040 |
| axis_Intellect | 0.3968 |
| mean_assistant_axis | 0.5067 |
| PC1 | 0.9482 |
| PC2 | 0.9724 |
| PC3 | 0.2979 |

Two entries in that table matter more than the rest.

**The grand mean is inside this sphere.** `mean_assistant_axis` — the direction
of having a persona at all, see [[stage-two-shared-direction]] — has cosine
0.5067 with the top-3 factor space. The principal-component sphere could not
have had this: its subspace comes from the eigenvectors of the *double-centred*
Gram and is orthogonal to the grand mean by construction. The factor chart works
in the uncentred G (`qwen35/fa_chart.py`), so half of the grand mean lies in the
sphere the lattice covers. That is a real difference in what is being sampled,
and it is the most likely mechanism behind the coherence and coverage changes
below. It is recorded here, not resolved.

**PC3 barely appears.** PC1 and PC2 are almost inside this three-space (0.9482,
0.9724) but PC3 is at 0.2979. The two three-dimensional subspaces share roughly
two dimensions, not three. Every comparison below is conditioned on that.

## Coverage of the zoo

`analysis/sphere_layout_fa.json#var3` = 0.11590344397319126, 0.10223504799677975,
0.044322108527744895 — the share of the total centred adapter variance (trace of
the double-centred Gram) projecting onto each of the three basis vectors, summing
to 0.2624606004977159. The rendered page shows this as "26.2%".

The principal-component sphere's own `var3`
(`analysis/sphere_layout.json#var3` = 0.12668915056443317, 0.11252214481119611,
0.04955491287468896, summing to 0.2887662082503183) is **not** directly
comparable: it was computed from the eigenvalues of the k=32 *sketch* Gram
(`build_sphere_spec.py` loads `analysis/sketches/stage1_k32`), not the exact one.
Two like-for-like references were therefore computed on the exact centred Gram
and stored alongside:

- `#var3_pc_exact_gram` = 0.1249462838655816, 0.10984693652476066,
  0.049679397531247095 (sum 0.28447261792158934) — the top three eigenvalues,
  i.e. the best possible three-dimensional subspace.
- `#var3_pc_steered_dirs` = 0.12025180802811421, 0.1115120780857482,
  0.0487418136518603 (sum 0.2805056997657227) — the space actually spanned by
  the three steered PC directions.

So the factor sphere sees 26.2% of the centred variance against 28.1%
(`#var3_pc_steered_dirs`) for the space the principal-component sphere actually
sampled; 28.4% (`#var3_pc_exact_gram`) is the best any three-space could do. The
factor chart gives up about two points of variance for an axis set with names.

The factors are oblique, and the sphere is not orthogonal in the sense a reader
might assume. `#factor_pairwise_cosines` gives the cosines between the five
oblique factor directions as they sit in the chart; Warmth and Fearful
withdrawal are at -0.6661, Competence and Arousal at -0.728.

## The run

`zoo-sphere-fa.service` (`Description=Sphere sweep on the FACTOR chart: 72
sampled directions in one container`) ran
`modal run --detach sphere_sweep.py --spec phase10_runs/sphere_sweep_spec_fa.json
--out phase10_runs/sphere_results_fa.json`. The log's header line confirms what
it consumed: "72 points x 8 prompts at alpha=1.5  tag=fa  basis=['B1_FA_Warmth',
'B2_FA_Competence', 'B3_FA_FearfulWithdrawal']"
(`qwen35/phase10_runs/sphere_fa.log`). It ends "wrote
phase10_runs/sphere_results_fa.json: 72 points".

**One code change was needed and it was a real hazard.** `sphere_sweep.py`
hardcoded its resume checkpoint at `/oct/sphere/results.json`. Re-run under that
path, the container would have found the 2026-09-01 run's 72 finished points and
returned *those* generations under the new point names — silently, at no cost
and with no error. The checkpoint is now namespaced by a `tag` from the spec
(`/oct/sphere_fa/results.json` here); untagged runs keep the original path so the
PC run stays resumable. Verified empirically, not from the log line: the FA log
contains zero `[resume]` lines, and of the 72 points, 0 have generations
identical to the PC run.

Thinking is off — `enable_thinking=False` in the `apply_chat_template` call
(`qwen35/sphere_sweep.py`), the failure mode recorded in
[[method-lessons|the project's method lessons]].

Judging was the same blind pipeline with the same arguments as the PC sphere
(`zoo-judge-sphere-fa.service`, `--batch 6 --workers 8 --repeat-frac 0.05`).
`qwen35/phase10_runs/judge_sphere_fa.log`: "576 generations to judge from 72
traits", "101 judge calls (batch=6, 28 repeats)", "576 judged, 0 failed calls".
Repeat reliability: Extraversion r=0.852 (n=28), Agreeableness r=0.963 (n=28),
Conscientiousness r=0.857 (n=27), EmotionalStability r=0.912 (n=28), Intellect
r=0.937 (n=28). The judge is `anthropic/claude-sonnet-4.5`
(`judged_sphere_fa.json#model`).

## Result 1: most of the space is still habitable, slightly less so

`qwen35/analysis/sphere_page_fa.json#coherence`, computed with the same looping
detector as every other degeneration number in the study (`looping` in
`qwen35/analyse_alien_steer.py`: a 10-word window repeated 4 times):

```
none       45
any        27
worst      0.375
mean       0.06597222222222222
len_mean   1389.6996527777778
```

Against the PC sphere (`analysis/sphere_page.json#coherence`: none 48, any 24,
worst 0.25, mean 0.046875, len_mean 1261.234375). Three fewer directions are
completely clean, the worst direction loses 37.5% of its responses rather than
25%, and the mean looping rate rises from 4.7% to 6.6%. Responses are longer:
1389.7 characters against 1261.2.

The conclusion the PC sphere reached is unchanged and if anything stronger for
being reproduced on a different subspace: most directions through this space give
intact output at alpha = 1.5, so the coherence of the directions the study chose
is not, by itself, evidence of anything.

## Result 2: personality varies continuously, and slightly more so

`qwen35/analysis/sphere_page_fa.json#smooth`:

```
rho        0.6765886845997277
n_pairs    2556
n_scored   72
near_mean  0.9462620103993861
far_mean   2.917385198267297
```

Over all 2,556 pairs of the 72 scored directions, angular distance and judged Big
Five profile distance correlate at rho = +0.68, against +0.65 on the PC sphere
(`analysis/sphere_page.json#smooth` rho = 0.6511417860626274). Directions less
than 30 degrees apart differ by 0.95 on the Big Five (PC sphere: 0.97); more than
120 degrees apart, by 2.92 (PC sphere: 2.81). Dividing those two quoted values gives a near/far ratio of
3.08 against 2.91 — computed here, not a key in either file.

This is the load-bearing result of the sweep and it survives the change of chart
intact. As on the PC sphere, it is a *local continuity* claim and does not
conflict with [[additivity]].

## Result 3: the coverage is lopsided, and more so

Counting which scale the judge rated highest at each of the 72 points
(`analysis/sphere_page_fa.json#judged.<point>.top`), with the PC sphere alongside:

| top-rated scale | factor sphere | PC sphere |
|---|---|---|
| Agreeableness | 35 | 32 |
| Conscientiousness | 18 | 14 |
| Intellect | 14 | 18 |
| Extraversion | 5 | 6 |
| EmotionalStability | 0 | 2 |

Emotional stability is now the top-rated scale nowhere on the sphere. Note that
`axis_EmotionalStability` has cosine 0.604 with this three-space and
`axis_Intellect` only 0.3968 (`#landmark_cos`), so part of this shift is the
subspace not containing those axes rather than the model not going there.

## Result 4: the two judged fields, compared directly

This is the one measurement the redo makes possible. Each of the 72 factor-sphere
directions was matched to the nearest of the 72 principal-component directions,
after projecting the latter into this chart from their coefficient dicts in
`phase10_runs/sphere_spec.json` (not from their PC-space `u`), using the *signed*
cosine — steering at +u and -u are different personalities, so matching on |cos|
would pair a direction with its opposite. `#vs_pc`:

- `median_match_deg` = 31.23512652729555, `max_match_deg` = 79.55102833275029
- `median_pc_cos_into_fa3` = 0.76, `min_pc_cos_into_fa3` = 0.047
- `n_close` = 27 of 72 matched within `close_deg` = 20 degrees

Pearson correlation of the judged score over the matched pairs, per scale
(`#vs_pc.pearson`, and `#vs_pc.pearson_close` restricted to the 27 pairs under
20 degrees):

| scale | all 72 | within 20 deg (n=27) |
|---|---|---|
| Extraversion | +0.5307476945957669 | +0.7521474437608897 |
| Agreeableness | +0.29188317603886815 | +0.7064603499886505 |
| Conscientiousness | +0.7006611663442376 | +0.8185867149748282 |
| EmotionalStability | -0.5142099496319408 | -0.2512036269331304 |
| Intellect | -0.010885319251628207 | +0.33288202215333346 |

**How to read this.** Because PC3 sits at cosine 0.2979 with this three-space,
the projected principal-component lattice is squashed towards a great circle:
the median match is 31 degrees apart and the worst is 80. Over all 72 pairs the
correlations are therefore a mixture of real agreement and points that have no
counterpart at all. Restricted to the 27 pairs that genuinely land near each
other, three of five scales agree at +0.71 to +0.82 — two directions close in
weight space give close judged personalities, whichever chart named them, which
is the smoothness result of Result 2 seen across the two runs.

Emotional stability does not agree on either column, and its correlation is
*negative*. This is unresolved. Two candidates, neither tested: it is the scale
with no top-rated point anywhere on the factor sphere and the fewest on the PC
sphere (2), so its range across both fields is small and the correlation is
dominated by judge noise; and it is the scale whose named axis is furthest
outside both subspaces after Intellect. It should not be quoted as evidence that
the two spheres disagree about emotional stability until someone has separated
those.

## Cost

The sweep ran as one A100-40GB container. `zoo-sphere-fa.service` started
2026-09-08 13:32:04 UTC and `phase10_runs/sphere_results_fa.json` was written at
17:25:28 UTC: 3.890 hours of unit wall clock, including roughly four minutes of
image build and base-model download before the GPU did any work. At the meter's
rate of $2.10 per GPU-hour (`qwen35/zoo40_meter.sh`, `RATE=2.10`) that is an
upper bound of **$8.17**. The box-wide meter cannot confirm this: as many as 16
containers were active during the window (`phase10_runs/zoo40_meter.log`,
2026-09-08T14:31:03Z), so its $2251.20 to $2346.57 movement over the same period
is mostly other work.

The FA run was slower per point than the PC run (about 3.1 minutes against about
1.7) because its responses are longer — 1389.7 mean characters against 1261.2
(`#coherence.len_mean`) — and generation is capped by tokens, not time.

Judging cost is OpenRouter, not Modal: 101 calls to `anthropic/claude-sonnet-4.5`.

## Where it is rendered

`qwen35/build_sphere_page.py` now takes `--suffix` (`""` for the PC sphere,
`_fa` for this one) and `--html`. It writes `analysis/sphere_page_fa.json`
(874,865 bytes) and the standalone `qwen35/sphere_page/index_fa.html`, served at
/sphere-fa.html. The standalone page shares no asset with the blog page on
purpose: the blog page's `_js.txt` colours sphere landmarks by Big Five name,
which is wrong on a sphere whose axes are factors.

Verified against the PC sphere before use: re-running the refactored builder with
no arguments reproduces `analysis/sphere_page.json` byte for byte.

## The 72 points on the model's own metric

[[fisher-norms]] measured the Fisher norm of all 72 of these directions. They
run from 0.1716923166559369 to 0.49828560642407 with median 0.2608319173143098
(`qwen35/analysis/fisher_norms.json#comparisons.sphere_band`), and take the top
twelve places among the 124 directions measured. Two results follow, and they
point in opposite directions. F predicts how far a point moves the judged
persona: against the distance of each point's five-scale profile from the
72-point centroid, Spearman +0.3706669239179368, permutation p
0.0014499275036248187 over n = 72
(`#correlation_with_degeneration.sphere_judged_profile_distance`). F does not
predict whether the point loops: against the recomputed loop rate at alpha 1.5,
Spearman -0.1015 at p 0.3957
(`#correlation_with_degeneration.sphere_alpha1.5`). Curvature at the base point
and damage at alpha 1.5 are different quantities.

Related: [[sphere-sweep]], [[sphere-sweep-iso-kl]], [[factor-analysis]],
[[steering-results]],
[[additivity]], [[stage-two-shared-direction]], [[alien-direction-steering]],
[[fisher-norms]], [[geometry-overview]], [[glossary]].
