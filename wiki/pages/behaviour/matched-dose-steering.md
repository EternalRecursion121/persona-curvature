---
title: Suppression versus amplification at matched Fisher dose
summary: Ten directions steered in both signs at alphas chosen so each sign delivers the same measured KL per token, judged blind against a base generated in the same run; the raw suppress-over-amplify ratio falls from 1.896 to 1.646, but once each change is divided by the room that existed in its direction amplification is the stronger move in both arms (0.795 at matched dose), so the asymmetry is headroom rather than mechanism - and two side findings, that bf16 weights deliver a median 0.906 of the intended KL (0.942 at alpha 2) and that at matched dose the amplifying sign loops more than the suppressing one.
status: current
sources:
  - qwen35/fisher_dose.py
  - qwen35/solve_matched_alphas.py
  - qwen35/dose_to_eval.py
  - qwen35/analyse_matched_dose.py
  - qwen35/steer_fix.py
  - qwen35/judge_personas.py
  - qwen35/analysis/matched_dose_steering.json
  - qwen35/analysis/matched_dose_alphas.json
  - qwen35/analysis/fisher_norms.json
  - qwen35/phase10_runs/dose_calib.json
  - qwen35/phase10_runs/dose_spec.json
  - qwen35/phase10_runs/steer_dose_spec.json
  - qwen35/phase10_runs/steer_results_dose.json
  - qwen35/phase10_runs/dose_conditions.json
  - qwen35/phase10_runs/judged_dose.json
  - qwen35/phase10_runs/judged_steerfix23.json
  - qwen35/phase10_runs/judged_steerfix.json
  - qwen35/phase10_runs/dosecalib.log
  - qwen35/phase10_runs/dosesteer.log
  - qwen35/phase10_runs/dosejudge.log
  - qwen35/analysis/sphere_isokl.json
last_verified: 2026-09-10
tags: [behaviour, steering, curvature, units, big-five]
---

# Suppression versus amplification at matched Fisher dose

## The claim being tested

At alpha plus or minus 2 the judged change on each factor's own Big Five scale
is Warmth **-2.0833333333333335 / +1.5**, Competence
**-3.458333333333333 / +0.666666666666667**, Timidity
**-1.9166666666666665 / +0.20833333333333393**, Arousal
**-0.5833333333333335 / +1.5**, Imagination **-2.1250000000000004 / +1.125**
(`qwen35/analysis/matched_dose_steering.json#directions.*.published_equal_alpha`,
recomputed here from `qwen35/phase10_runs/judged_steerfix23.json`; these are the
numbers [[post-draft]] rounds to -2.1/+1.5, -3.5/+0.7, -1.9/+0.2, -0.6/+1.5,
-2.1/+1.1). Four of five suppress harder than they amplify.

Three explanations were on the table:

1. **Mechanism.** Gradient Atoms (Rosser, 2026) argues suppression is
   structurally easier - one pathway to break a behaviour, many to strengthen
   it - and reaches the same asymmetry on formatting behaviours where no
   personality ceiling exists ([[paper-reading-2026-09-09]]).
2. **Ceiling.** This project's own explanation: the base already scores high on
   Conscientiousness and Intellect, so there is less room upward.
3. **Units.** [[fisher-norms]] shows the published directions spread over a
   factor of 260 in Fisher norm, so a table at common alpha is a table at
   different doses.

Explanation 3 has a subtlety that decides how the experiment must be built. F is
**even in alpha**, so to second order +alpha and -alpha along the same direction
are already the same dose; what is not matched is the third and fourth order.
That turns out to matter a great deal - see the measured curve below.

## Step one: measure the dose curve instead of extrapolating it

`qwen35/fisher_dose.py` measures KL(base || steered) per token on the same fixed
4,378 token positions [[fisher-norms]] uses, for ten directions - the five
recovered factors and the five Big Five keying axes - at twelve alphas spanning
|alpha| in [1.0, 2.5] on both signs, under two weight constructions. The
zero-alpha control reads exactly 0 for KL, symmetric KL and argmax change
(`qwen35/phase10_runs/dose_calib.json#zero_alpha_control`).

This was necessary rather than decorative. `fisher_norms.json` fits
KL(alpha) = 0.5 F alpha^2 + c alpha^3 + d alpha^4 from six alphas inside
|alpha| <= 0.5, and the cubic term is already **0.15620366112207026** of the
quadratic at the edge of that fit for FA_Warmth
(`qwen35/analysis/fisher_norms.json#directions.FA_Warmth.cubic_over_F`).
Extrapolating it to alpha 2 - four times outside its own range - **overstates the
measured KL for all ten directions**, by a median factor of
**1.4751663093238792** and a maximum of **5.7846156073675274**
(`qwen35/analysis/matched_dose_steering.json#taylor_extrapolation_check`; the
minimum is 1.142145422961612, so no direction is understated). For FA_Warmth the
fit predicts **1.5892934857465926** - that is
`0.5 * 0.36503759829531435 * 4 + 0.057020209300935695 * 8 + 0.0251910384217799 * 16`
from `#directions.FA_Warmth.F_ref`, `.cubic_c` and `.quartic_d` - against a
measured **0.6470294541123253**, a factor of **2.456292330504461**. Every alpha
used here is therefore interpolated **between measured points**; `extrapolated`
is `false` for all twenty (direction, sign) cells.

### Side finding: what bf16 weights actually deliver

`steer_fix.py` writes the steering increment into **bf16** parameters, so the
model that generates every published steering response is not the model the
fp32 curvature was measured on. The calibration measured both.

Both statistics are in
`qwen35/analysis/matched_dose_steering.json#bf16_vs_exact`, over the 120
(direction, alpha) cells.

The **norm** survives: `||bf16(W + kD) - W||_F / (|alpha| * ref)` runs
**0.9216314941424595** to **1.033505578742928**, median
**1.0027071769953744** (`#bf16_vs_exact.retention`). Values above 1 are not a
puzzle - round-to-nearest quantisation error is unbiased and adds in quadrature
to the intended increment. It rises with alpha, from a median
0.9254622335119027 at |alpha| 1.0 to 1.0316791185228427 at |alpha| 2.5, because
a larger increment clears the rounding threshold on more weights
(`#bf16_vs_exact.retention_median_by_abs_alpha`).

The **KL** does not survive as well. `KL(bf16) / KL(exact)` at the same nominal
alpha runs **0.7117708405618217** to **0.9782241941431974**, median
**0.9059646365364478** (`#bf16_vs_exact.kl_bf16_over_exact`). The quantisation
keeps the length and spends part of it on isotropic noise that moves the output
distribution less than the direction would have. The shortfall shrinks as the
increment grows: the median by |alpha| is 0.7510758772738881, 0.8264063055376227,
0.8799179197563782, 0.9186068920598383, **0.9423571405594928**,
0.9693561112085924 at |alpha| 1.0, 1.25, 1.5, 1.75, 2.0, 2.5
(`#bf16_vs_exact.kl_median_by_abs_alpha`).

So **the published alpha plus or minus 2 delivered a median 0.9424 of its
intended dose**, and a steering alpha of 1 delivers only 0.7511 of it. That is
not the catastrophe the element-level arithmetic suggests - most of the length
is there - but it is a real, alpha-dependent shortfall, and it is why the alphas
here are matched on the **bf16** curve, the construction the generating model
actually has.

## Step two: the dose, and the alphas it implies

`fisher_norms.json` defines a direction's Fisher dose per unit alpha as
`sqrt(F(u) / F_random_median)`. A dose of 2 is the KL that alpha 2 would deliver
along a median random merge of all 134 adapters:

    KL_target = 0.5 * 0.12382989334918551 * 4 = 0.24765978669837102 nats/token

(`qwen35/analysis/fisher_norms.json#random_band.median`;
`matched_dose_alphas.json#kl_target_nats_per_token`). Because every one of these
ten directions is steeper than the random median, every matched alpha comes out
below 2, which keeps the whole experiment inside the coherent band. The target
is **0.376 to 0.950** of what alpha plus or minus 2 actually delivered.

| direction | alpha- | alpha+ | KL at -2 | KL at +2 | KL(+2)/KL(-2) |
|---|---|---|---|---|---|
| FA_Warmth | -1.5783 | 1.1803 | 0.3577 | 0.6470 | 1.809 |
| FA_Competence | -1.5848 | 1.5822 | 0.4007 | 0.4201 | 1.048 |
| FA_FearfulWithdrawal | -1.6596 | 1.8537 | 0.3643 | 0.2867 | 0.787 |
| FA_Arousal | -1.5662 | 1.8198 | 0.4161 | 0.2993 | 0.719 |
| FA_Imagination | -1.4972 | 1.5028 | 0.4261 | 0.4530 | 1.063 |
| axis_Extraversion | -1.6891 | 1.8659 | 0.3568 | 0.2819 | 0.790 |
| axis_Agreeableness | -1.5756 | 1.1719 | 0.3573 | 0.6578 | 1.841 |
| axis_Conscientiousness | -1.6069 | 1.7640 | 0.3900 | 0.3227 | 0.827 |
| axis_EmotionalStability | -1.9450 | 1.3416 | 0.2608 | 0.4493 | 1.723 |
| axis_Intellect | -1.4324 | 1.4995 | 0.4652 | 0.4586 | 0.986 |

(Alphas verbatim from `matched_dose_alphas.json#directions.*.matched` - they
are stored rounded to four places because they are what was steered. KL and the
ratio are displayed to three or four places; the full-precision values are in
`#directions.*.kl_at_equal_alpha_2`, measured on the bf16 construction.)

**The last column is the headline of the calibration.** Equal alpha is not equal
dose *even between the two signs of the same direction*: the ratio runs from
0.719 to 1.841. And it runs the wrong way for the mechanism story. FA_Warmth at
alpha +2 was carrying **1.809 times** the KL of FA_Warmth at alpha -2, and moved
its own scale by +1.5 against -2.083. The published amplifier was already the
larger perturbation and still the smaller effect.

## Step three: generate and judge

`steer_fix.py` unmodified, so the corpus is made the way the published one was:
greedy, `enable_thinking=False`, 512 new tokens, the same 24 prompts.
**Job names are prefixed `dose_`** so that `steer_fix.py`'s resume - it reloads
`/oct/steerfix/<name>.json` from the Modal volume - could not merge the
published campaigns' seven alphas into this run; the output holds exactly two
alphas per direction, 504 generations
(`qwen35/phase10_runs/steer_results_dose.json`). A **base condition is generated
in this run** (empty direction, alpha 0) and judged interleaved with the steered
text, so the baseline does not have to be carried across judge batches.

`judge_personas.py` unchanged, `anthropic/claude-sonnet-4.5`, 504 judged, **0
failed calls**, repeat reliability over 25 duplicated units r = 0.819
Extraversion, 0.840 Agreeableness, 0.805 Conscientiousness, 0.678 Emotional
Stability, 0.850 Intellect (`qwen35/phase10_runs/dosejudge.log`). Conditions
carry opaque tags assigned in a seeded shuffle
(`qwen35/phase10_runs/dose_conditions.json`); the judge sees only
(prompt, response) in any case.

**The base this run measured**: Extraversion 4.25, Agreeableness 4.875,
Conscientiousness **5.916666666666667**, Emotional Stability 4.875, Intellect
**5.625** (`qwen35/analysis/matched_dose_steering.json#base.scores`). That is the
ceiling, restated: on Conscientiousness there are 1.083 points of room upward
and 4.917 downward.

## The result

Signed change in the direction's own Big Five scale against that base, with the
suppressing and amplifying sign identified from the data rather than assumed.
`room` is the change divided by the room that existed in the direction it moved:
`delta / (7 - base)` upward, `delta / (base - 1)` downward.

| direction | suppress | amplify | suppress room | amplify room | off-target suppress | off-target amplify |
|---|---|---|---|---|---|---|
| FA_Warmth | -1.542 | +1.542 | -0.398 | +0.725 | 0.115 | 0.771 |
| FA_Competence | -2.875 | -0.042 | -0.585 | -0.008 | 0.844 | 0.406 |
| FA_FearfulWithdrawal | -0.750 | +0.250 | -0.194 | +0.118 | 1.281 | 0.740 |
| FA_Arousal | -0.708 | +1.833 | -0.218 | +0.667 | 0.479 | 2.125 |
| FA_Imagination | -1.333 | +0.917 | -0.288 | +0.667 | 0.198 | 0.417 |
| axis_Extraversion | -0.833 | +2.000 | -0.256 | +0.727 | 0.594 | 0.594 |
| axis_Agreeableness | -1.667 | +1.667 | -0.430 | +0.784 | 0.135 | 0.812 |
| axis_Conscientiousness | -2.875 | +0.250 | -0.585 | +0.231 | 0.688 | 0.490 |
| axis_EmotionalStability | -2.208 | +0.375 | -0.570 | +0.176 | 0.927 | 0.635 |
| axis_Intellect | -1.458 | +1.000 | -0.315 | +0.727 | 0.375 | 0.469 |

(`qwen35/analysis/matched_dose_steering.json#directions.*.suppress` and
`.amplify`, displayed to three decimal places - the judged means are
twenty-fourths and the file carries them in full, e.g. FA_Warmth's suppressing
delta is -1.5416666666666665 and its room share -0.3978494623655914. Every cell
also carries a bootstrap 95% interval over the 24 prompts, paired prompt by
prompt.)

### The two ratios

| | published, equal alpha +/- 2 | matched Fisher dose |
|---|---|---|
| suppress / amplify, raw | **1.8957528957528955** (8 of 10) | **1.6455696202531644** (6 of 10) |
| suppress / amplify, room-normalised | **0.9261912176364205** (4 of 10) | **0.7946215813581846** (4 of 10) |
| mean \|delta\| suppress | 2.0458333333333334 | 1.625 |
| mean \|delta\| amplify | 1.0791666666666668 | 0.9875 |
| mean off-target suppress | - | 0.5635416666666667 |
| mean off-target amplify | - | 0.7458786231884059 |

(`#summary.all` and `#published_equal_alpha_summary.all`. The five factors alone
read 2.0333 raw and 0.9899 room published, against 1.5727 and 0.7700 matched;
the five axes 1.7770 and 0.8714 published, against 1.7087 and 0.8150 matched -
`#summary.fa_factors`, `#summary.big_five_axes`, quoted to four places, full
precision in the file. The published arm has no off-target or loop row because
those statistics are computed here only for this run's own generations.)

**Read those two rows together.** Matching the dose shrinks the raw asymmetry
from 1.896 to 1.646 and takes it from 8 of 10 directions to 6 of 10, so units
account for part of it but not most. What accounts for the rest is the room. In
*both* arms, once each change is expressed as a share of the distance available
in the direction it moved, amplification is at least as strong as suppression -
0.9262 published, **0.7946** at matched dose - and the count is 4 of 10 either
way. The base model sits at 5.92 of 7 on Conscientiousness and 5.63 of 7 on
Intellect; there is four to five times as much room downward as upward, and the
raw asymmetry is close to that ratio.

**How far to push that.** The direction of the conclusion is robust: both arms
flip the same way, and the two independent readings below - amplification being
the *less* selective move, and FA_Competence losing its amplifying sign entirely
when the base rises by a third of a point - point the same way without using the
room correction at all. The **magnitude** is not robust. On Conscientiousness the
upward denominator `7 - base` is 1.083333333333333, and a ratio built on
denominators that small is unstable; the number 0.7946215813581846 should be read
as "amplification is at least as strong per unit of room", not as an effect size.

So on this battery **the ceiling explanation survives the test and the mechanism
explanation is not supported**. This does not refute Gradient Atoms: that paper
reaches its asymmetry on formatting behaviours with no ceiling, which is exactly
the control this project does not have. What it says is that *this* project's
suppress/amplify table is not independent evidence for the mechanism, and
[[post-draft]] should not present it as if it were.

Two further readings point the same way.

**Amplification is the less selective move, not the more.** Mean off-target
movement is **0.5635** on the suppressing sign and **0.7459** on the amplifying
sign (`#summary.all`), driven by FA_Arousal, whose amplifying sign moves the
other four scales by 2.125 on average while moving its own by 1.833. If
suppression were the blunt instrument, this column would run the other way.

**FA_Competence has no amplifying sign at matched dose.** Both signs lower
judged Conscientiousness (-2.875 and -0.042), so `bipolar` is `false` for it
alone among the ten (`#directions.FA_Competence.bipolar`). The published arm
recorded +0.667 at alpha +2, at a base of 5.583; this run's base is 5.917. A
+0.667 that becomes -0.042 when the base moves up by a third of a point is what
a ceiling looks like from underneath.

## Degeneration: the sign asymmetry does not hold at matched dose

[[fisher-norms]] records that over 22 previously measured directions, all 22
loop at least as much at alpha -2 as at +2 and 18 have a loop rate of exactly
zero at +2 (`#comparisons.sign_asymmetry_of_degeneration`), and concludes that
"degeneration under steering in this project is overwhelmingly a negative-alpha
phenomenon".

At matched dose the mean loop rate is **0.075** on the suppressing sign and
**0.1375** on the amplifying sign over all ten directions; for the five factors
alone it is 0.075 against **0.2167** (`#summary.*.suppress_mean_loop_rate` and
`amplify_mean_loop_rate`, using `analyse_alien_steer.looping` verbatim). The
single largest loop rate in the whole run is FA_Arousal's **amplifying** side at
alpha +1.8198: **0.625** of 24 responses
(`#directions.FA_Arousal.amplify.loop_rate`).

The mechanism is visible in the alphas. Where a direction's amplifying sign is
the *flatter* one, matching the dose pushes it to a larger alpha than the
published +2 comparison ever reached relative to its own curvature - FA_Arousal
runs to +1.8198 where its suppressing side stops at -1.5662 - and it degenerates
there. The published observation was not wrong about its own corpus, but it
generalised a fact about a fixed alpha grid into a fact about sign. See
[[superseded-claims]].

## Caveats

- **One dose, not three.** The G5 design in [[paper-reading-2026-09-09]] asks
  for three doses; the budget covered one. There is no dose-response curve here,
  only a single matched point, so "amplification is stronger per unit of room"
  is established at dose 2 and nowhere else.
- **The matched dose is smaller than the published one**, 0.376 to 0.950 of what
  alpha plus or minus 2 delivered, so the absolute effects here are smaller by
  construction. The comparison that carries weight is between the two ratios,
  not between the two sets of raw numbers.
- **The random-merge control at the same Fisher dose was not run.** It is in the
  G5 design and it was cut for budget.
- **The published arm's baseline is its own campaign's alpha-0 row**, judged in a
  different call batch, and its five axes were published at
  ref 0.8102592902648793 against 0.8078003190997738 here, a 0.30 per cent larger
  unit (`matched_dose_alphas.json#ref_note`). Its generations also reached each
  alpha by an accumulated bf16 walk over seven alphas, where this run's reached
  each of two alphas in at most two steps.
- **Room-normalisation is a linear correction to a judge's ordinal scale.**
  Dividing by `(7 - base)` assumes the 1-to-7 scale is an interval scale near its
  ends, which no rubric guarantees. It is the right first-order correction and it
  is not a proof.

## Cost and provenance

Two Modal apps on A100s and one API judge.

- `zoo-dosecalib.service`, app `pc-qwen35-phase10-dosecalib`, A100-40GB,
  23:18:02 to 23:48:09 UTC on 2026-09-09 (30 min 7 s), 27.9 min inside the
  function, 240 (direction, mode, alpha) passes. Log
  `qwen35/phase10_runs/dosecalib.log`.
- `zoo-dosesteer.service`, app `pc-qwen35-phase10-dosesteer`, eleven A100-40GB
  containers in parallel, 23:48:28 to 00:19:14 UTC (30 min 46 s), 504
  generations. Log `qwen35/phase10_runs/dosesteer.log`.
- `zoo-dosejudge.service`, 89 judge calls to `anthropic/claude-sonnet-4.5`
  through OpenRouter, 00:19:18 to 00:20:26 UTC. Log
  `qwen35/phase10_runs/dosejudge.log`.

At the spend meter's rate of $2.10 per GPU-hour (`qwen35/zoo40_meter.sh`,
`RATE`), that is 0.502 GPU-hours for the calibration and **at most** 5.641 for
the generation - eleven containers counted at the app's whole lifetime, which
over-counts any that finished early - so **at most $12.90** for this experiment.
The generation is where the estimate in [[paper-reading-2026-09-09]] was wrong:
it budgeted $3-5 for 1,440 generations and 504 cost more than that, because
24 prompts at 512 greedy tokens is about ten minutes of an A100 per alpha and
eleven parallel containers multiply it. Samuel authorised $15 for G1 and G5
together on 2026-09-09. See [[costs]].

Related: [[steering-results]], [[fisher-norms]], [[sphere-sweep-iso-kl]],
[[factor-analysis-fisher-metric]],
[[paper-reading-2026-09-09]], [[judged-evaluations]], [[post-draft]],
[[superseded-claims]], [[alien-direction-steering]], [[glossary]].
