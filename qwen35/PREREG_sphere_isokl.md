# Pre-registration: the sphere sweep at iso-KL dose

Written 2026-09-10, **before** the steering run was launched and before the
calibration's numbers were read. The calibration (`zoo-sphere-isokl-calib.service`)
was launched first because its grid, its alphas and its target rule are all fixed
mechanically by this document; nothing in it is a choice made after seeing data.

## The question

The 2026-09-01 principal-component sphere ([[sphere-sweep]]) steered all 72
Fibonacci-lattice directions at one alpha, 1.5, and found that angular distance
predicts judged Big Five profile distance at Spearman +0.6511417860626274
(`qwen35/analysis/sphere_page.json#smooth.rho`).

`qwen35/analysis/fisher_norms.json` then showed that the Fisher norm F of a
sphere lattice direction varies about threefold (0.1716923166559369 to
0.49828560642407 on the *factor* sphere, `#comparisons.sphere_band`) and that F
predicts how far a point's judged profile sits from the lattice centroid,
Spearman +0.3706669239179368, permutation p 0.0014499275036248187
(`#correlation_with_degeneration.sphere_judged_profile_distance`).

A sphere at common alpha is therefore a sphere at unequal **dose**. Some of the
smoothness, the coherence pattern and the lopsided coverage may be an image of
how the dose field varies over the lattice rather than of where the directions
point. This run removes that confound by construction: every one of the 72
directions is steered at the alpha that delivers the **same measured KL per
token**, and everything else is held at the 2026-09-01 values.

## What is held fixed

- The same 72 unit directions. Verified before launch: recomputing
  `build_sphere_spec.py`'s Fibonacci lattice and `W = V[:, :3] / sqrt(w[:3])`
  reproduces `phase10_runs/sphere_sweep_spec.json` `points[*].u` to 0.0 absolute
  error, the three basis coefficient dicts to 0.0, and the 72 coefficient vectors
  of `phase10_runs/sphere_spec.json` to cosine 0.9999999999999998 to
  1.0000000000000002.
- The same 8 prompts (`PROMPT_IDX = [0, 1, 5, 9, 11, 14, 15, 22]`), the same
  `ref = 0.8102592902648793`, the same decoding (`do_sample=False`,
  `max_new_tokens=512`, `enable_thinking=False`), the same one-container build
  from three basis deltas, the same `sphere_sweep.py`.
- The same blind judge, `anthropic/claude-sonnet-4.5` via OpenRouter, the same
  five scales, the same `--batch 6 --workers 8`, and the same looping detector
  (`analyse_alien_steer.looping`, a 10-word window repeated 4 times).

## What changes

1. **Per-point alpha.** `sphere_sweep.py` gains `a_pt = float(pt.get("alpha", alpha))`,
   backward compatible: a spec with no per-point alpha behaves exactly as before.
2. **An alpha-0 base point generated in the same container**, so the baseline is
   not carried across judge batches. It is a 73rd point with `alpha: 0.0`, which
   sets `s = 0` and restores the pristine bf16 base exactly.
3. **The repeat subsample rises from 5 per cent to 15 per cent** so the
   reliability estimate is on ~90 duplicated units rather than 28.

## The target dose

**Rule, fixed before the calibration was read:** the target is the **median over
the 72 directions of the measured KL(base || steered) per token at alpha 1.5**,
on the bf16 weight construction, on the same fixed 4,378 token positions
`fisher_dose.py` scores.

Why that and not `0.5 * F_random_median * dose^2` as [[matched-dose-steering]]
used: the point of this run is a like-for-like comparison with the 2026-09-01
sphere, so the iso-KL sphere must deliver, in aggregate, the dose that sphere
delivered. The median is the statistic that guarantees it: half the directions
move up in alpha and half move down, and the sum of the doses is unchanged to
first order. A target picked from an external unit would have made every point's
alpha shift the same way and confounded "iso-KL" with "different overall dose".

**bf16, not exact.** `sphere_sweep.py` writes `(W0 + D*s).to(bfloat16)`, so the
dose its text receives is the bf16 curve. [[matched-dose-steering]] measured that
bf16 delivers a median 0.9059646365364478 of the exact-weight KL at the same
alpha (0.8799179197563782 at |alpha| 1.5), so matching on the exact curve would
match an intention rather than a delivery.

## The alpha rule

For each direction, KL is measured at **alpha 1.0, 1.25, 1.5, 1.75, 2.0, 2.5**,
positive sign only, as the original sphere used +1.5 for every point. The alpha
that delivers the target is read off by **log-log linear interpolation between
bracketing measured points** — `solve()` in `qwen35/solve_matched_alphas.py`,
reused verbatim, because KL ~ alpha^2 to leading order so log KL against
log alpha is near-straight and piecewise-linear interpolation on those axes is
accurate between measured points.

Never extrapolate. Pre-registered fallbacks, written before the calibration
finished:

- If the target lies outside `[KL(1.0), KL(2.5)]` for a direction, its alpha is
  **clamped to the nearest measured endpoint** (1.0 or 2.5) and the direction
  carries `extrapolated: true`. Clamping, not extrapolating, because the Taylor
  fit overstates KL at these alphas by a median 1.4751663093238792
  (`analysis/matched_dose_steering.json#taylor_extrapolation_check`) and there is
  no reason to trust a linear extension of the log-log curve either.
- If KL is not monotone in alpha for a direction, that is recorded verbatim and
  the bracketing pair containing the target is used; a direction whose curve is
  non-monotone across the whole grid is reported and left at alpha 1.5.
- Every direction's `extrapolated` flag, bracket and solved alpha go into
  `analysis/sphere_isokl_alphas.json`.

## The statistics

Computed by `qwen35/analyse_sphere_isokl.py` into
`qwen35/analysis/sphere_isokl.json`. Every one of (a) to (d) is computed by the
**same functions the original sphere page used** — `smoothness`, `coherence` and
`profiles` imported from `qwen35/build_sphere_page.py` — and the script's first
act is to reproduce `analysis/sphere_page.json#smooth` and `#coherence` from the
2026-09-01 files. If that reproduction is not exact the run stops.

(a) Spearman rho between angular distance and judged Big Five profile distance
    over the 72 x 71 / 2 = 2,556 pairs, with a permutation p from 10,000
    permutations of the **profiles across points** (not of the pairs, which are
    not independent), and a 95 per cent bootstrap interval from 2,000 resamples
    of the 72 points.

(b) Mean profile distance for pairs under 30 degrees and over 120 degrees.

(c) Number of points with zero looping, the worst point's loop rate, the mean
    loop rate, and the mean response length.

(d) Per-scale range and peak across the 72 points, and the count of points where
    each scale is top-rated — in particular whether any point peaks on
    Emotional Stability (the PC sphere had 2, the factor sphere 0).

(e) The same statistics recomputed for the original alpha-1.5 PC sphere from
    `phase10_runs/judged_sphere.json`, **and** with dose as a covariate.
    `fisher_norms.json`'s 72 `sphere_S*` keys are the **factor** sphere, not this
    one — verified, cosine 1.0 against `sphere_spec_fa.json` and median 0.0094
    against `sphere_spec.json` — so F for these 72 points does not exist. The
    covariate is this run's own calibration: **the measured bf16 KL per token at
    alpha 1.5**, which is the dose each PC-sphere point actually received in the
    2026-09-01 run, and a strictly better covariate than F because it is measured
    at the alpha that was steered rather than extrapolated from |alpha| <= 0.5.
    Primary pairwise covariate: `|KL_i - KL_j|`, because the confound is that a
    pair whose two points differ in dose differs in judged profile for that
    reason. Secondary: `KL_i + KL_j`. Partial Spearman by the standard
    rank-residual formula, permutation p on the same profile-permutation scheme.
    Also reported: the per-point replication of the fisher-norms statistic —
    dose against the distance of each point's profile from the 72-point centroid.

(f) Per-point agreement between the iso-KL and alpha-1.5 judged fields, per
    scale: Pearson r over the 72 matched points (the same lattice point in both
    runs, so the match is exact and no projection is needed), plus mean absolute
    difference.

(g) The spread of chosen alphas (min, median, max, and the count clamped), and
    an interpolation-error check that costs nothing: for every direction, predict
    KL at 1.5 from the 1.25 / 1.75 pair by the same log-log rule and report the
    ratio to the measured 1.5.

## What would mean "the smoothness was dose" and what would mean "position"

Let `rho_iso` be (a) on this run, `rho_orig` = 0.6511417860626274 the original's
raw value, and `rho_orig_partial` the original's rho partialling out
`|KL_i - KL_j|`.

- **The smoothness was dose** if *both*: `rho_iso < rho_orig - 0.15` (that is,
  below about 0.50), and `rho_orig_partial < rho_orig - 0.10`. Removing the dose
  variation experimentally and removing it statistically then agree.
- **The smoothness is position** if *both*: `|rho_iso - rho_orig| <= 0.10` and
  `|rho_orig_partial - rho_orig| <= 0.05`. Equal-dose sampling reproduces the
  original correlation and dose explains almost none of it.
- **Ambiguous** in every other case, including the case where the two criteria
  disagree with each other, and it will be reported as ambiguous rather than
  argued into one of the two boxes.

The same three-way rule is applied, descriptively and without a threshold, to
(b), (c) and (d): if the near/far gap, the count of clean points, or the absence
of an Emotional Stability peak survive iso-KL sampling, they were not artefacts
of unequal dose.

## Budget

Authorised: $25 of Modal plus judging for the whole experiment. Calibration
estimated at 432 forward-pass measurements, about one A100-hour, from the
matched-dose calibration's measured 155 s per direction for 24 passes. The
generation run is estimated from the 2026-09-01 sphere's own wall clock and will
be re-estimated from the calibration's actual runtime before it is launched.

---

## Addendum, 2026-09-10 12:18 UTC: the target, filled in from the calibration

Written after `zoo-sphere-isokl-calib.service` finished and **before**
`zoo-sphere-isokl.service` was started. Nothing above this line was edited.

`qwen35/phase10_runs/sphere_isokl_calib.json`, 72 directions x 6 alphas, bf16
weights, 4,378 scored token positions, zero-alpha control exactly 0 for KL,
symmetric KL and argmax change, 50.1 minutes on one A100-40GB.

- Measured KL per token at **alpha 1.5** over the 72 lattice points:
  **0.160622** to **0.407472**, median **0.233437**, a spread of **2.537x**
  (`analysis/sphere_isokl_alphas.json#summary`). The 2026-09-01 sphere therefore
  delivered doses differing by a factor of 2.5 across a lattice it called one
  strength. That is the confound this run removes.
- **Target dose: 0.233437488 nats per token**
  (`#kl_target_nats_per_token`), the median of that set, by the rule fixed above.
- Solved alphas: **1.1409** to **1.8037**, median **1.5002**, sd 0.1682.
  **0 of 72 extrapolated** and **0 of 72 non-monotone**, so no fallback fired and
  every alpha is an interpolation between measured points.
- Interpolation error check: KL at alpha 1.5 predicted from the 1.25 and 1.75
  measurements by the same log-log rule, over the measured value, has median
  **0.98795** and range 0.96742 to 1.00290
  (`#summary.interp_check_kl15_pred_over_meas`). The rule is accurate to about
  1 per cent at this spacing, and that is the size of the residual mismatch to
  expect in the delivered dose.

The delivered dose will be re-measured at the solved alphas
(`zoo-sphere-isokl-verify.service`) rather than assumed.
