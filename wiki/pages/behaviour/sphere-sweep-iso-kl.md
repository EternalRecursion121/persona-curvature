---
title: The sphere sweep at iso-KL dose
summary: The 72-direction principal-component sphere re-steered so that every point delivers the same measured KL per token instead of the same alpha; the alpha-1.5 sphere turns out to have been dosing its points over a 2.54x range, and removing that changes almost nothing - angular distance still predicts judged-profile distance at rho 0.65, the two judged fields agree per point at r 0.63 to 0.97, and what does change is degeneration, which does not follow the dose at all.
status: current
sources:
  - qwen35/PREREG_sphere_isokl.md
  - qwen35/build_sphere_isokl_calib.py
  - qwen35/sphere_dose.py
  - qwen35/solve_sphere_isokl_alphas.py
  - qwen35/sphere_sweep.py
  - qwen35/sphere_to_eval.py
  - qwen35/judge_personas.py
  - qwen35/analyse_sphere_isokl.py
  - qwen35/build_sphere_page.py
  - qwen35/analyse_alien_steer.py
  - qwen35/analysis/sphere_isokl.json
  - qwen35/analysis/sphere_isokl_alphas.json
  - qwen35/analysis/sphere_page.json
  - qwen35/analysis/sphere_layout.json
  - qwen35/analysis/fisher_norms.json
  - qwen35/analysis/matched_dose_steering.json
  - qwen35/phase10_runs/sphere_isokl_dose_spec.json
  - qwen35/phase10_runs/sphere_isokl_calib.json
  - qwen35/phase10_runs/sphere_isokl_sweep_spec.json
  - qwen35/phase10_runs/sphere_isokl_verify_spec.json
  - qwen35/phase10_runs/sphere_isokl_verify.json
  - qwen35/phase10_runs/sphere_isokl_results.json
  - qwen35/phase10_runs/judged_sphere_isokl.json
  - qwen35/phase10_runs/sphere_isokl_calib.log
  - qwen35/phase10_runs/sphere_isokl.log
  - qwen35/phase10_runs/sphere_isokl_verify.log
  - qwen35/phase10_runs/judge_sphere_isokl.log
  - qwen35/phase10_runs/steer_results_dose.json
  - qwen35/zoo40_meter.sh
last_verified: 2026-09-10
tags: [behaviour, steering, sampling, controls, units, curvature]
---

# The sphere sweep at iso-KL dose

## The confound

[[sphere-sweep]] steered all 72 Fibonacci-lattice directions on the unit sphere
of the top three principal components at one strength, `ALPHA = 1.5`, and found
that angular distance predicts judged Big Five profile distance at Spearman
**0.6511417860626274** (`qwen35/analysis/sphere_page.json#smooth.rho`).
[[sphere-sweep-factor-chart]] reproduced that at 0.6765886845997277 on the
factor chart.

[[fisher-norms]] then made the frame a problem. A steering direction is a unit
vector in the Frobenius metric, which the model has no opinion about; measured in
the metric the model does imply, the 72 points of the *factor* sphere spread from
0.1716923166559369 to 0.49828560642407 in Fisher norm
(`analysis/fisher_norms.json#comparisons.sphere_band`), and F predicts how far a
point's judged profile sits from the lattice centroid, Spearman
**0.3706669239179368**, permutation p 0.0014499275036248187
(`#correlation_with_degeneration.sphere_judged_profile_distance`). One alpha over
a lattice is therefore not one dose over a lattice, and some of the smoothness,
the coherence pattern and the lopsided coverage could be an image of how the dose
field varies over the sphere rather than of where the directions point.

This run removes the confound by construction. Same 72 directions, same 8
prompts, same decoding, same judge; each direction steered at the alpha that
delivers the **same measured KL(base || steered) per token**. Pre-registered in
`qwen35/PREREG_sphere_isokl.md` before the steering was launched, target filled
in from the calibration in a dated addendum before the same launch.

**The headline: the confound was real and it did not matter.** The alpha-1.5
sphere was dosing its points over a 2.54x range. Flattening that to 1.02x leaves
the smoothness result, the near/far gap and the coverage essentially where they
were. What it does change is degeneration and the reading of what curvature buys.

## What was held fixed, and the check that it was

`qwen35/build_sphere_isokl_calib.py` and `solve_sphere_isokl_alphas.py` change
the alpha and nothing else. Before anything was launched, `build_sphere_spec.py`'s
lattice and `W = V[:, :3] / sqrt(w[:3])` were recomputed from the same sketches
and compared against the files the 2026-09-01 run consumed
(`PREREG_sphere_isokl.md`):

- `points[*].u` of `phase10_runs/sphere_sweep_spec.json`: maximum absolute
  difference **0.0**.
- the three basis coefficient dicts: maximum absolute difference **0.0**.
- the 72 coefficient vectors of `phase10_runs/sphere_spec.json`: cosine
  **0.9999999999999998** to **1.0000000000000002**.
- `ref` **0.8102592902648793**, `phase10_runs/steer_spec.json#ref`, the PC
  sphere's own unit - deliberately not the 0.8078003190997738 of
  `steer_spec2_7a.json` that [[matched-dose-steering]] used, because "the same
  steering magnitude" is the comparability claim here.

`sphere_sweep.py` gained one backward-compatible line, `a_pt = float(pt.get("alpha", alpha))`.
A spec with no per-point alpha behaves exactly as it did.

## Step one: what the alpha-1.5 sphere was actually dosing

`qwen35/sphere_dose.py` is `fisher_dose.py` with the partial checkpoint
namespaced by a spec `tag`, a mode-agnostic progress line and its own app name;
no arithmetic differs, so the measurement is on the same fixed **4,378** scored
token positions ([[fisher-norms]]), the same 24 prompts and the same 192-token
cap. Zero-alpha control exactly **0.0** for KL, symmetric KL and argmax change
(`analysis/sphere_isokl_alphas.json#zero_alpha_control`).

Only the **bf16** weight construction was measured. `sphere_sweep.py` writes
`(W0 + D*s).to(bfloat16)`, which is exactly `fisher_dose.py`'s `bf16` mode, so
that is the dose the generating model receives; [[matched-dose-steering]] found
bf16 delivers a median 0.9059646365364478 of the exact-weight KL at the same
alpha, so matching on the exact curve would have matched an intention rather
than a delivery.

72 directions at alpha 1.0, 1.25, 1.5, 1.75, 2.0 and 2.5
(`phase10_runs/sphere_isokl_dose_spec.json`), 432 measurements, 50.1 minutes on
one A100-40GB (`phase10_runs/sphere_isokl_calib.log`).

At **alpha 1.5** the 72 points delivered
(`analysis/sphere_isokl_alphas.json#summary`):

| | KL per token |
|---|---|
| minimum (`S054`) | 0.16062187418756882 |
| median | 0.2334374880453069 |
| maximum (`S005`) | 0.40747211904164155 |
| max / min | **2.5368407703038587** |

So the sphere the project has been quoting since 2026-09-01 was a sphere at
doses differing by a factor of **2.54**. That is a slightly smaller spread than
the factor sphere's Fisher-norm range, whose maximum over its minimum is 2.902
(computed here from `fisher_norms.json#comparisons.sphere_band`, not a key in
either file, and a curvature ratio rather than a delivered-KL ratio).

## Step two: the target and the alphas

**Target 0.2334374880453069 nats per token**
(`#kl_target_nats_per_token`) - the median of the column above, by the rule fixed
in the pre-registration before the calibration was read. The median is the
statistic that keeps the aggregate dose equal to the original sphere's: half the
lattice moves up in alpha and half moves down. A target taken from an external
unit, as [[matched-dose-steering]] used (`0.5 * F_random_median * dose^2`), would
have shifted every point the same way and confounded "iso-KL" with "a different
overall dose".

The alpha is read off the measured curve by log-log interpolation between
bracketing measured points - `solve()` from `qwen35/solve_matched_alphas.py`,
reused unchanged. **0 of 72 extrapolated and 0 of 72 non-monotone**
(`#summary.n_extrapolated`, `#summary.n_non_monotone`), so no pre-registered
fallback fired.

| | alpha |
|---|---|
| minimum (`S005`, the steepest direction) | 1.1409 |
| median | 1.5002499999999999 |
| mean | 1.480963888888889 |
| maximum (`S054`, the flattest) | 1.8037 |
| standard deviation | 0.1682145162176965 |

(`#summary`. The median landing on 1.5002 is the design working, not a
coincidence: the target is the median of the KL at 1.5.)

The interpolation itself was checked at no cost: KL at alpha 1.5 predicted from
the 1.25 and 1.75 measurements by the same log-log rule, over the measured value,
has median **0.987947433893645** and range 0.967415326884758 to
1.0028996091629692 (`#summary.interp_check_kl15_pred_over_meas`). The rule is
accurate to about one per cent at this spacing.

## Step three: the dose actually delivered, re-measured

`zoo-sphere-isokl-verify.service` re-ran the same measurement once per direction
at its solved alpha (`phase10_runs/sphere_isokl_verify.json`, 20.0 minutes).
`analysis/sphere_isokl.json#delivered`:

| | KL per token |
|---|---|
| target | 0.2334374880453069 |
| minimum | 0.23289529173926385 |
| median | 0.23394863155673443 |
| maximum | 0.23789536021734928 |
| max / min | **1.0214691694312275** |

Median absolute relative error **0.0022839730699117933**, worst
**0.019096642143344056**. The 2.54x dose spread of the alpha-1.5 sphere is now
**1.02x**, and the residual is the size the interpolation check predicted. This
is the sphere the rest of the page is about.

**One caveat on that 1.02x.** The delivered KL is measured with
`sphere_dose.py`'s weight delta, built in bf16 from the 134 adapters; the text is
generated with `sphere_sweep.py`'s, built in fp32 from three fp16 basis products.
The direction is the same to cosine 1.0 and both are normalised to the same
Frobenius length before the same bf16 rounding of `W0 + D*s`, so the two should
agree to well inside the one per cent the interpolation already costs - but they
are different rounding paths and nothing here measures the difference at
alpha > 0. The `S900` byte-identity below shows the two pipelines agree at
alpha 0, which is a weaker statement.

## The run

`zoo-sphere-isokl.service`, one A100-40GB container, 73 points x 8 prompts, log
header "73 points x 8 prompts at per-point alpha 0.0000..1.8037 over 73 points
tag=isokl basis=['PC1', 'PC2', 'PC3']" (`phase10_runs/sphere_isokl.log`). 584
generations, none empty, every point's alpha equal to its spec entry.

**The resume hazard was checked, not assumed.** `sphere_sweep.py`'s checkpoint is
namespaced by the spec's `tag`, here `/oct/sphere_isokl/results.json`; the log
contains **zero** `[resume]` lines and **0 of 72** points have generations
identical to the 2026-09-01 run (`analysis/sphere_isokl.json#run`, verified in
`analyse_sphere_isokl.py`'s own check). That is the failure
[[sphere-sweep-factor-chart]] recorded as costing nothing only because it was
caught.

**A base condition was generated in the same container**, point `S900` at alpha
0, which sets `s = 0` and restores the pristine bf16 base exactly. It is judged
interleaved with the steered text. Its judged profile is Extraversion 4.25,
Agreeableness 4.875, Conscientiousness 5.125, Emotional Stability 4.625,
Intellect 5.375 (`#base_profile`) - on eight prompts, not the 24
[[matched-dose-steering]] used, so it is not the same base measurement.

**The base is the base.** `S900`'s eight responses are **identical, all eight,
character for character**, to `dose_BASE` from
`phase10_runs/steer_results_dose.json` on the same eight prompts
(`#base_vs_dose_base`, mean shared prefix fraction **1.0**). Those two texts were
produced by different code paths in different containers a day apart -
`sphere_sweep.py` editing weights in place, `steer_fix.py` via `add_adapter` -
and greedy decoding at `enable_thinking=False` puts them on the same string. It
is the cleanest determinism check this project has.

**Judging** was `judge_personas.py` unchanged, `anthropic/claude-sonnet-4.5`
through OpenRouter, `--batch 6 --workers 8`, with the repeat subsample raised
from 0.05 to **0.15**: "584 generations to judge from 73 traits", "112 judge
calls (batch=6, 87 repeats)", "584 judged, 0 failed calls". Repeat reliability
over 87 duplicated units: Extraversion r=0.856, Agreeableness r=0.936,
Conscientiousness r=0.876, Emotional Stability r=0.832 (n=85), Intellect r=0.868
(`phase10_runs/judge_sphere_isokl.log`). The judge sees only (prompt, response)
and the point names are coordinates.

## The statistics are computed by the original sphere's own functions

`qwen35/analyse_sphere_isokl.py` imports `profiles`, `coherence` and
`smoothness` from `qwen35/build_sphere_page.py` and its first act is to
reproduce `analysis/sphere_page.json#coherence` and `#smooth` from the
2026-09-01 files; the run stops if they do not match. They match exactly
(`analysis/sphere_isokl.json#selfcheck`), so the two spheres are counted the same
way and not merely described the same way.

One arithmetic note. `analysis/sphere_isokl.json` carries the original sphere's
rho twice, as **0.6511417860626274** from `smoothness` and **0.6511433864273928**
from the permutation harness, and the iso-KL sphere's as **0.6522913409368338**
and **0.6522896435584883**. The two code paths enumerate the 2,556 pairs by the
same order but build the distances with different float operations, and ordinal
ranking breaks the resulting near-ties differently. The difference is in the
seventh decimal; this page quotes `smooth.rho`, the published function
(`#verdict.rho_note`).

## Result 1: the smoothness is position, not dose

`analysis/sphere_isokl.json#smooth`, against `#smooth_pc_alpha1_5`:

| | iso-KL | alpha 1.5 |
|---|---|---|
| Spearman rho, angular vs judged-profile distance | **0.6522913409368338** | 0.6511417860626274 |
| permutation p (10,000, profiles permuted across points) | 9.999000099990002e-05 | 9.999000099990002e-05 |
| 95% bootstrap interval (2,000 resamples of the points) | 0.569194359966947 to 0.7363147576812953 | 0.5653651193815503 to 0.7392842726604579 |
| pairs | 2,556 | 2,556 |
| mean profile distance under 30 degrees | 1.0041690486481023 | 0.9661257055739045 |
| mean profile distance over 120 degrees | 2.9030567912512786 | 2.80790992900049 |

Equalising the dose moves rho by **+0.0011**. Each interval contains the other
point estimate comfortably.

The statistical control agrees with the experimental one. Partialling the dose
out of the *original* alpha-1.5 field, with the pairwise covariate
`|KL_i - KL_j|` (the pre-registered primary: the confound is that a pair whose
two points differ in dose differs in judged profile for that reason), gives
partial rho **0.6238447733137857**, a drop of **0.02729861311360704**
(`#dose_covariate.abs_dose_diff`). The covariate is not inert - it correlates
0.28956234877940806 with angular distance and 0.25444334647827827 with profile
distance - it simply does not carry the relationship. With the secondary
covariate `KL_i + KL_j` the partial rho is 0.6518399218472459, a drop of
**-0.0007** (`#dose_covariate.dose_sum`).

The pre-registered rule was: dose if rho_iso is below rho_orig - 0.15 *and* the
partial is below rho_orig - 0.10; position if rho_iso is within 0.10 of rho_orig
*and* the partial is within 0.05. **Both position conditions hold and neither
dose condition does** (`#verdict`). The permutation null used for the partial
permutes the judged profiles across points, which breaks the position-profile
and the dose-profile links at once; it is an omnibus null, not a test of the
partial term alone.

The near/far gap survives too, slightly wider in both terms.

## Result 2: the two judged fields agree point for point

This comparison is exact in a way [[sphere-sweep-factor-chart]]'s could not be.
That page had to project one lattice into the other chart and match by signed
cosine, with a median match of 31 degrees. Here it is the same lattice point in
both runs, so the pairing is the identity. Pearson over the 72 points
(`#field_agreement`):

| scale | Pearson | Spearman | mean absolute difference | mean iso-KL | mean alpha 1.5 |
|---|---|---|---|---|---|
| Extraversion | 0.848393762287737 | 0.8389928612772527 | 0.2586805555555556 | 4.470486111111111 | 4.434027777777778 |
| Agreeableness | 0.9749625261075069 | 0.970383947520741 | 0.1684027777777778 | 4.949652777777778 | 4.989583333333333 |
| Conscientiousness | 0.8616839450451251 | 0.8499903530773684 | 0.2951388888888889 | 4.762152777777778 | 4.817708333333333 |
| Emotional Stability | 0.6336825712273262 | 0.6593671618753618 | 0.28794642857142855 | 4.605902777777778 | 4.615079365079365 |
| Intellect | 0.9253740096986294 | 0.8614380346002959 | 0.2378472222222222 | 4.932291666666667 | 4.944444444444445 |

Mean Euclidean distance between a point's two five-scale profiles
**0.679615261605419**; **61 of 72** points keep the same top-rated scale.

Emotional Stability is again the weakest column, as it was across the two
subspaces (+0.5307 to -0.5142 there, depending on the cut). Here it is +0.63 and
positive, which removes one of the two candidate explanations
[[sphere-sweep-factor-chart]] left open: the negative correlation there is not a
property of the scale itself but of that comparison's projection, because the
same scale on the same lattice at a different dose agrees at +0.63. What remains
true of the scale is that its range is the smallest of the five on both spheres,
so judge noise is a larger share of it.

## Result 3: coverage barely moves, and Emotional Stability gains a point

Counting which scale the judge rated highest at each point
(`#scales.top_counts`, `#scales_pc_alpha1_5.top_counts`), with the factor sphere
from [[sphere-sweep-factor-chart]] alongside:

| top-rated scale | iso-KL | PC at alpha 1.5 | factor sphere |
|---|---|---|---|
| Agreeableness | 31 | 32 | 35 |
| Intellect | 16 | 18 | 14 |
| Conscientiousness | 13 | 14 | 18 |
| Extraversion | 9 | 6 | 5 |
| Emotional Stability | **3** | 2 | 0 |

Per-scale range across the 72 points (`#scales`): Extraversion 3.375 to 6.375,
Agreeableness 3.25 to 6.375, Conscientiousness 2.75 to 6.25, Emotional Stability
3.125 to 5.375, Intellect 3.625 to 6.5. Against alpha 1.5
(`#scales_pc_alpha1_5`): 3.5 to 6.0, 2.875 to 6.5, 3.125 to 6.25, 3.75 to 5.625,
3.625 to 6.5.

Emotional Stability's range widens from 1.875 to 2.25 and it tops three points
rather than two: `S001` at 5.375, `S009` at 4.875 and `S022` at 4.75, against
`S001` at 5.142857142857143 and `S022` at 4.875 on the alpha-1.5 sphere. The same
two lattice points carry the scale on both runs and one more joins them, which is
a change of one point out of 72 and should not be read as more than "the absence
of an Emotional Stability region on these spheres is not an artefact of unequal
dose". The best any point does on that scale is 5.375 against this run's base of
4.625, three quarters of a point; the single highest Emotional Stability score
anywhere on the alpha-1.5 sphere, 5.625 at `S012`, belongs to a point the judge
rated highest on Agreeableness. The lopsidedness is a fact about the sphere, not
about the dosing.

## Result 4: equalising the dose does not equalise degeneration

This is where iso-KL sampling changes the answer.

`#coherence` against `#coherence_pc_alpha1_5`, same detector
(`analyse_alien_steer.looping`, a 10-word window repeated four times):

| | iso-KL | PC at alpha 1.5 | factor sphere |
|---|---|---|---|
| points with no looping | **45** | 48 | 45 |
| points with any | 27 | 24 | 27 |
| worst point's loop rate | 0.375 | 0.25 | 0.375 |
| mean loop rate | 0.06770833333333333 | 0.046875 | 0.06597222222222222 |
| mean response length (characters) | 1314.689236111111 | 1261.234375 | 1389.6996527777778 |

The base point loops on none of its eight prompts and averages 1755.625
characters (`#coherence_base`), so both spheres shorten the model's output.

Three correlations say why the count fell, all with permutation p over 10,000
permutations (`#dose_covariate`). Write a direction's **steepness** as the KL it
delivered at alpha 1.5 - the quantity that decides its iso-KL alpha:

| arm | n | Spearman | permutation p |
|---|---|---|---|
| steepness vs loop rate, PC sphere at alpha 1.5 | 72 | -0.3734966878898965 | 0.063993600639936 |
| steepness vs loop rate, iso-KL sphere | 72 | **-0.5145668531738377** | 0.0015998400159984002 |
| alpha vs loop rate, iso-KL sphere | 72 | **+0.509100263682552** | 0.0015998400159984002 |
| steepness vs mean response length, iso-KL sphere | 72 | -0.5926747700816773 | 9.999000099990002e-05 |

Read those carefully, because two of them are one fact. On the iso-KL sphere
alpha is a monotone-decreasing function of steepness by construction, so the
second and third rows have to mirror each other and **cannot separate alpha from
flatness**. What the four rows together do license is narrower than it looks:

- **Holding the KL fixed does not equalise degeneration.** If looping were a
  function of how far the output distribution moves, the iso-KL sphere would
  have a flat loop field. It has the opposite - the strongest loop correlation
  on either sphere.
- **Flat directions loop more, at fixed alpha and at fixed KL.** On the
  alpha-1.5 sphere alpha is constant and steepness still predicts looping at
  -0.3735, though only at p 0.064, so flatness matters independently of how far
  the direction was pushed.
- **Pushing a flat direction further makes it worse.** Equalising the dose sent
  the flattest directions out to alpha 1.80, and the count of completely clean
  points fell by three.

That last figure is a net. Sixteen of the 72 points loop on both spheres, so
eleven points began looping and eight stopped
(`analysis/sphere_isokl.json#loop_rates`). The worst point is `S028` on both, at
0.375 here against 0.25 at alpha 1.5, and its solved alpha is 1.4943 - almost
exactly the alpha it already had, so its extra looping is not a dose effect at
all.

This sharpens rather than contradicts [[fisher-norms]]'s "does F predict
degeneration? Not by itself, no", which measured F against loop rate on the
factor sphere at Spearman -0.1015, p 0.3957. The sign is the same and the
magnitude here is larger. Two things differ and both help: the covariate is a
measured KL at the alpha steered rather than a curvature extrapolated from
|alpha| <= 0.5, and on the iso-KL arm the dose term that was working against it
is gone.

## Result 5: on this sphere, dose does not predict how far the persona moves

[[fisher-norms]]'s headline positive - F against the distance of each point's
five-scale profile from the 72-point centroid, Spearman **+0.3706669239179368**
at p 0.0014499275036248187 over the factor sphere - **does not reproduce here**
(`#dose_covariate.dose_vs_centroid_distance_pc_alpha1_5`):

| field | covariate | n | Spearman | permutation p |
|---|---|---|---|---|
| PC sphere at alpha 1.5 | measured KL at alpha 1.5 | 72 | **+0.016013891568589622** | 0.8899110088991101 |
| iso-KL sphere | the same per-direction steepness | 72 | **-0.4022445173323044** | 0.0004999500049995 |

On the principal-component sphere at common alpha, how steep a direction is says
nothing about how far it moves the judged persona. On the iso-KL sphere, where
every point delivers the same KL, steepness predicts displacement *negatively*:
the steep directions, given the smaller alphas, end up closer to the centroid.

**Two readings fit these two rows and this run cannot separate them.**

*Cancellation.* At common alpha a steep direction carries more KL, which pushes
its persona further; it also converts KL into judged movement less efficiently,
which pulls it back. The two effects are of similar size on this lattice and the
correlation lands on zero.

*Alpha, not KL.* On the iso-KL sphere alpha rises as steepness falls, so
"steepness predicts displacement at -0.40" is also "alpha predicts displacement
at about +0.40" - the same sign and roughly the same size as alpha against loop
rate, +0.5091. Together with the alpha-1.5 sphere's null, that reads as judged
displacement and breakage both scaling with how far the weights moved rather
than with how far the output distribution moved. What this reading has to explain
away is the factor sphere's +0.37 at constant alpha, which is exactly the
disagreement recorded as S28.

The page does not pick between them. Separating them needs a sphere at two doses,
which is the same gap [[matched-dose-steering]] left open. The factor sphere's
+0.37 is the same quantity measured on a different lattice with a different
(extrapolated) covariate; both numbers stand and they are recorded as a source
contradiction, S28 in [[source-contradictions]].

Either way, this is the strongest single reason not to state "Fisher norm
predicts how far the judged persona moves" as a general fact about this weight
space. It is a fact about the factor sphere.

## What this run does and does not establish

- It establishes that the smoothness result of [[sphere-sweep]] and
  [[sphere-sweep-factor-chart]] is not a dose artefact. Both the experimental
  control (rho 0.6523 at 1.02x dose spread) and the statistical control (partial
  rho 0.6238 on the original at 2.54x spread) say so, and they were
  pre-registered together with the thresholds that would have said the opposite.
- It does not establish anything about **more than one dose**. Like
  [[matched-dose-steering]], this is a single matched point - the alpha-1.5
  median - and there is no dose-response curve. A sphere at half this dose or
  twice it is unmeasured.
- The 8-prompt battery and one greedy sample per prompt are the same limits the
  original sphere had. Every judged mean here is over eight responses.
- The judge's repeat reliability on Emotional Stability is 0.832, the lowest of
  the five, and that scale is also the one whose range is smallest. Its numbers
  carry the least weight on this page.
- Positive sign only, as the original sphere used. Nothing here bears on the
  sign asymmetry [[matched-dose-steering]] examined.

## Contradictions with existing pages

- **[[fisher-norms]] and [[post-draft]] state that F predicts judged displacement
  over "72 sphere points".** That holds on the factor sphere and not on the
  principal-component sphere; see Result 5 and S28 in [[source-contradictions]].
  The sentence should be scoped to the sphere it was measured on.
- **[[fisher-norms]]'s 72 `sphere_S*` keys are the factor sphere, not this one.**
  Verified before this run: cosine 1.0 against `phase10_runs/sphere_spec_fa.json`
  and median 0.0094 against `sphere_spec.json`. F for the 72 principal-component
  directions has never been measured, which is why the dose covariate on this
  page is `sphere_isokl_calib.json`'s measured KL and not F. That is a better
  covariate for this purpose in any case: it is measured at the alpha that was
  steered rather than extrapolated from |alpha| <= 0.5, an extrapolation
  [[matched-dose-steering]] showed overstates KL by a median factor of
  1.4751663093238792.
- **[[fisher-norms]] concludes F does not predict degeneration.** Result 4 finds
  a considerably stronger relationship on this sphere with a measured covariate,
  in the same direction. This does not contradict that page - different sphere,
  different covariate - but its conclusion should not be carried over to the
  principal-component sphere unqualified.

## Cost and provenance

Three Modal apps on A100-40GBs and one API judge, all on 2026-09-10.

- `zoo-sphere-isokl-calib.service`, app `pc-qwen35-phase10-sphere-isokl-calib`,
  launched 11:27 UTC, output written 12:17:53 UTC. 432 (direction, alpha)
  measurements; 50.1 minutes inside the function. Log
  `phase10_runs/sphere_isokl_calib.log`.
- `zoo-sphere-isokl.service`, app `pc-qwen35-phase10-sphere-isokl`, 12:18:26 to
  14:54:54 UTC, 2 hours 36.5 minutes, 584 generations. Log
  `phase10_runs/sphere_isokl.log`.
- `zoo-sphere-isokl-verify.service`, app
  `pc-qwen35-phase10-sphere-isokl-verify`, 12:18:26 to 12:38:50 UTC, 20.0
  minutes, 72 measurements. Log `phase10_runs/sphere_isokl_verify.log`.
- `zoo-judge-sphere-isokl.service`, 112 calls to `anthropic/claude-sonnet-4.5`
  through OpenRouter, about 90 seconds. Log
  `phase10_runs/judge_sphere_isokl.log`.

That is 0.867 + 2.608 + 0.340 = **3.815 GPU-hours**, which the spend meter's rate
of $2.10 per GPU-hour (`qwen35/zoo40_meter.sh`, `RATE`) values at **$8.01**. The
meter's rate is GPU-only; these containers also reserved 52 to 64 GiB of memory
and 2 to 8 CPU cores, which Modal bills separately and which
[[fisher-norms]] estimates at roughly $1.2 per GPU-hour at this size, taking the
Modal total to roughly **$12 to $13**. The box-wide meter cannot confirm it: four
sycophancy-forecast containers were running throughout, so its movement from
$2534.00 at 11:31 UTC to $2563.05 at 14:57 UTC
(`phase10_runs/zoo40_meter.log`) is mostly other work.

The judging is OpenRouter, whose pot has no per-project attribution
([[infrastructure]]). 112 calls carrying about 235,000 input tokens and about
45,000 output tokens; at Sonnet-tier list rates of $3 and $15 per million that is
about **$1.4**. Authorised cap for the whole experiment: $25.

Related: [[sphere-sweep]], [[sphere-sweep-factor-chart]], [[fisher-norms]],
[[matched-dose-steering]], [[steering-results]], [[additivity]],
[[source-contradictions]], [[judged-evaluations]], [[infrastructure]],
[[post-draft]], [[glossary]].
