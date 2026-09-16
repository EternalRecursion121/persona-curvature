---
title: The activation-weighted Gram, and what the seed floor looks like in it
summary: The 134x134 Gram rebuilt as ASVD's activation-weighted inner product - two adapters are close if they change the modules' outputs alike on the inputs the model actually sees - moves the same-trait cross-seed cosine from +0.0181 to +0.6643, a factor of 36.8, and reverses the ordering that made the seed floor look like absence of signal; but the frame-overlap null moves with it, from 0.0215 to 0.8910, so relative to what a perfectly reproduced update would score the two metrics agree (0.75 against 0.84) and the floor is not a metric artefact by the preregistered test; the factor solution is untouched, Tucker congruence 0.9926.
status: current
sources:
  - qwen35/PREREG_actgram.md
  - qwen35/act_gram_on_modal.py
  - qwen35/build_actgram_spec.py
  - qwen35/analyse_act_gram.py
  - qwen35/analyse_fa_actgram.py
  - qwen35/run_actgram_analysis.sh
  - qwen35/analysis/act_gram.json
  - qwen35/analysis/fa_act_metric.json
  - qwen35/analysis/crossseed_arms_actgram.json
  - qwen35/analysis/crossseed_arms_actgram_frobcheck.json
  - qwen35/analysis/crossseed_arms_actgram_centred.json
  - qwen35/analysis/crossseed_arms_actgram_resp.json
  - qwen35/results/gram_actweighted.npz
  - qwen35/results/cross_gram_actweighted_root_x_data_null_seedpaired_s40_matched.npz
  - qwen35/results/fa_qwen35_actgram.json
  - qwen35/phase10_runs/actgram_results.json
  - qwen35/phase10_runs/actgram.log
  - qwen35/phase10_runs/actgram_analysis.log
  - qwen35/analysis/crossseed_arms.json#[1]
  - https://arxiv.org/abs/2312.05821
last_verified: 2026-09-11
tags: [geometry, seed, cross-seed, metric, nulls, factor-analysis]
---

# The activation-weighted Gram, and what the seed floor looks like in it

## The question

Every Gram in this project - `results/gram_sweep.npz`, the PCA, the factor
analysis, the cross-seed block - is the **Frobenius** inner product of two LoRA
weight changes. It weights every input direction of a module equally. Under it
[[seed-floor|the seed floor]] is +0.01806099632415457: the same trait retrained
from an independent LoRA-A initialisation is nearly orthogonal to its original.

ASVD (Yuan et al., arXiv:2312.05821; [[paper-asvd]]) argues that the Frobenius
metric is the wrong one for an LLM, because activations are strongly non-uniform
across input channels. The functional metric is activation-weighted: two updates
to a module are similar if they change its **output** similarly on the inputs the
model actually sees. [[paper-asvd]]'s first ranked idea was to rebuild the Gram in
that metric and ask whether the seed floor is partly an artefact of the isotropic
one. This page is that test. It was preregistered in `qwen35/PREREG_actgram.md`
before any container was launched.

**The answer has two halves and both are needed.** The absolute cross-seed cosine
rises by a factor of **36.78028836557651**, from +0.0181 to **+0.6642887281525424**
(`analysis/act_gram.json#verdict.ratio_a_C_over_a_F`,
`#arms.pool445.a_same_trait_cross_seed_mean`), and the ordering that made the floor
read as absence of signal reverses. But the **frame-overlap null** - what a trait
that reproduced *perfectly* modulo the initialisation frame would score - rises
with it, from 0.021471366484559353 to **0.8909847593616453**
(`#arms.*.c2_frame_overlap.seed0_vs_seed1.c2b_summed`). Measured against that
null, the two metrics agree: **0.7455668811085819** under the activation metric
against **0.8411666943129359** under Frobenius
(`#verdict.ratio_a_C_over_null_C`, `#verdict.ratio_a_F_over_null_F`). By the
preregistered thresholds the seed floor is **real in the functional metric**, not
a metric artefact.

## The object

For adapters i, j with `dW_i = s B_i A_i` (s = lora_alpha / r = 2.0) and
`C = E_x[x x^T]` the **uncentred** input covariance of a module on the base model,

    <dW_i, dW_j>_C = E_x[(dW_i x) . (dW_j x)]
                   = s^2 tr(B_i^T B_j (A_j C A_i^T))
                   = s^2 sum( (B_i^T B_j) * (A_i C A_j^T) )

summed over the 248 targeted modules, exactly as the Frobenius Gram sums
`tr(B_i^T B_j A_j A_i^T)`. Cosines are normalised by the same inner product.
Setting `C = I` recovers the Frobenius Gram through the identical code path,
which is this run's first validation check.

**C was formed in full, and the page says so because the brief said not to.**
The cheap route accumulates projected covariances `F C G^T` for a few *shared*
frames; that is exact only if every zoo adapter shares one A, and they do not - A
drifts during training ([[stage-one-training-config]], `a_drift` 0.0146). Forming
C outright costs 17.8 GiB in fp32 (184 modules at d_in 2560, 32 at 9216, 32 at
4096) and about 1e14 FLOPs over 20,612 tokens, which one A100-80GB does in under
a minute, and it makes the C-Gram **exact** in the same per-adapter `A_i` the
Frobenius Gram used, so the two are comparable entry for entry. The projected
covariances `F C G^T` are emitted anyway for a stride-31 module subset, on the
run's own volume (`/out/actgram_projcov_full.npz`).

## The activations

Base model `Qwen/Qwen3.5-4B`, **no adapter**, bf16 forward with activations cast
to fp32 before accumulation and TF32 off. Two arms, both fixed in
`phase10_runs/actgram_spec.json` before the run so the container generates nothing
(`qwen35/build_actgram_spec.py`):

| arm | text | tokens |
|---|---|---|
| `pool445`, **primary** | the 445-prompt shared training pool (`qwen35/data_common/*.jsonl`, asserted byte-identical across all 134 trait files), rendered with `apply_chat_template(add_generation_prompt=True, enable_thinking=False)` and tokenised with `add_special_tokens=False` - `train_qwen35.py`'s own rendering. **Prompt tokens only.** | **20,612** |
| `resp4378` | the 24 steering prompts with the base model's own stored alpha-0 greedy responses capped at 192 tokens - the same 4,378 scored positions [[factor-analysis-fisher-metric]] used. Prompt **and** response tokens. | **5,106** |

(`analysis/act_gram.json#n_tokens`.) Prompt tokens only for the primary arm
because no fixed base-model greedy corpus exists for the 445 pool, and the
per-trait `chosen`/`rejected` continuations would make C depend on the trait,
which is the one thing the design cannot allow ([[shared-prompt-pool-445]]).
The response-side arm is the check on that choice, and it changes nothing
material.

Each arm also has a **mean-centred** twin, `C - mu mu^T`. The uncentred form is
what `E_x[(dW x).(dW x)]` asks for and is primary; the centred one isolates how
much of the agreement is carried by the single direction every token shares.

## Four validation checks

1. **C = I reproduces the published Grams through this code path.** Max absolute
   cosine difference against `results/gram_sweep.npz` over the 134 x 134 block is
   **6.968684995722896e-08**, and against
   `results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz` over the
   134 x 40 block **2.946825408689513e-09**
   (`analysis/act_gram.json#validation`). The preregistered bar was 1e-4.
   Run through `analyse_crossseed.py` unchanged, the C = I arm returns the
   published seed-floor row to eight decimal places: same-trait
   +0.018060998368198355 against +0.01806099632415457, slope
   0.026542111946942823 against 0.026542111376493285, Pearson 0.9966259139824906
   against 0.9966259140629706, 40 of 40 top-1, separation +0.00009
   (`analysis/crossseed_arms_actgram_frobcheck.json#[0]` against
   `analysis/crossseed_arms.json#[1]`). **No source contradiction arises from
   this run.**
2. **The frames are shared.** Mean over modules of each adapter's relative
   deviation of `A` from its own set's mean `A` is **0.010178045947067138** for
   the 134 seed-0 adapters and **0.01010937949890391** for the 40 seed-1
   adapters, worst module 0.0210 and 0.0209
   (`#a_frame_spread`). This is **not** the published
   `a_drift` = 0.014605041334818797, which measures the deviation from the
   *initial* `A_0` rather than from the set mean; the smaller figure says the
   drift has a component shared across adapters, which is consistent with
   [[column-space-structure]]'s note that "the 1.5 percent that `A` moves in
   training is apparently not random". What matters here is only that each set
   writes into one frame to within about one percent.
3. **Hook coverage.** All 248 stripped module names resolved in
   `model.named_modules()`, every module's hook saw exactly the same number of
   rows as the token total in both arms (the run aborts otherwise), and each
   module's observed input width equalled its adapter's `A` width. The observed
   widths are 184 modules at 2560, 32 at 4096 (`o_proj`, `linear_attn.out_proj`)
   and 32 at 9216 (`mlp.down_proj`) (`phase10_runs/actgram.log`).
4. **The frame-overlap null calibrates.** Under C = I it reads
   **0.021471366484559353** for the real (seed-0, seed-1) frame pairs and
   0.021452190429212037 for pairs of fresh random frames
   (`#arms.frob.c2_frame_overlap`). That is `r/d` averaged over modules of
   several widths, not 64/2560 = 0.025: it matches
   `analysis/column_space.json`'s width-averaged analytic null of
   0.021014492753623194 quoted on [[column-space-structure]], and 0.025 is the
   value for the 2560-wide modules alone. On the 8-module all-2560 smoke it read
   0.025024575802187126 (`phase10_runs/actgram_smoke.json`).

## The nulls, and why the brief's null had to be replaced

The preregistration replaced the null it was handed, in advance and with the
reason stated, because the statistic as written has expectation zero.

**(c1) The literal statistic** - one side's frame replaced by a fresh
PEFT-style `kaiming_uniform_` frame with `B` kept, K = 8 frames, 320 (trait,
frame) pairs. The Gram entry `tr(B_i^T B_j R C A_i^T)` is **linear in R** and the
draw is sign-symmetric, so the signed mean is zero by construction: it reads
+0.00016835208405974146 under Frobenius and +0.00484763570786084 under C
(`#arms.*.c1_null.replace_seed1_frame.mean`). Its **RMS** is a real rank-64 noise
floor - 0.00046527363060994793 under Frobenius, 0.01678923450971668 under C - but
the ratio (a)/RMS is 38.8 under Frobenius and 39.6 under C, so that denominator
calls the Frobenius seed floor an artefact too and cannot discriminate between
metrics. Both numbers are reported; neither drives the verdict.

**(c2) The frame-overlap null, the preregistered denominator.** For frames F, G
with row-space projectors `P_F = F^T (F F^T)^-1 F`:

    c2b = tr(P_F C P_G C) / sqrt( tr(P_F C P_F C) tr(P_G C P_G C) )
    c2a = tr(P_F C P_G)   / sqrt( tr(P_F C P_F)   tr(P_G C P_G)   )

`c2b` is the model in which the unconstrained update D satisfies `D^T D ∝ C`,
which is what an accumulated `sum_t g_t x_t^T` gives when the output-side errors
are roughly uncorrelated across tokens; `c2a` is the model `D^T D ∝ I`. Both
reduce to `tr(P_F P_G)/r` when C = I. `c2b` is primary.

**(c3) The analytic r/d_eff is saturated and is not used as a denominator.** The
participation ratio `(tr C)^2 / tr(C^2)` of the input covariance has mean
**21.50361050262004** and median **9.849636963548347** over the 248 modules
(`#participation_ratio.pool445.all_modules`), so `r / d_eff` averages
**7.430688021985469** - the rank-64 frame is larger than the effective input
dimension, and the analytic prediction is simply *saturation*. The empirical
random-vs-random `c2b` of 0.880470664039374 is its calibrated form.

**(c4) The attenuation slope** - cross-seed C-cosine regressed on within-seed
C-cosine over the same 5,320 pairs - is the direct analogue of the 0.0265 that
[[seed-floor]] sets against 0.0250. It reads **0.8638570822661489** under C
against the frame-overlap null's 0.8909847593616453, and 0.026542111946942823
under Frobenius against 0.021471366484559353
(`#arms.*.c4_attenuation_slope`). **In both metrics the measured attenuation is
within a few per cent of what frame overlap alone predicts.** That is the
single strongest line on this page.

## The numbers

All from `analysis/act_gram.json#arms.<arm>`; the crossseed rows are reproduced
independently by `analyse_crossseed.py` in `analysis/crossseed_arms_actgram*.json`.

| | Frobenius | **C, pool445** | C, pool445 centred | C, resp4378 | C, resp4378 centred |
|---|---|---|---|---|---|
| **(a)** same trait, cross seed (n = 40) | +0.018060998368198355 | **+0.6642887281525424** | +0.3860407385148347 | +0.651691713743302 | +0.25294074386661725 |
| sd of (a) | 0.001053432223241571 | 0.027563599444776442 | 0.024905084954993002 | 0.027769589706819676 | 0.014274576951510864 |
| **(b)** different trait, cross seed (n = 5,320) | +0.0018594300142914396 | +0.07127236348522642 | +0.042686552864355166 | +0.06733709521757611 | +0.02527019199580551 |
| within seed 0, off-diagonal (n = 8,911) | +0.07208348219779794 | +0.08760415842065107 | +0.08425397202133728 | +0.08263617151818163 | +0.07591344886534303 |
| min same / max off | 0.01593 / 0.01584 | 0.59811 / 0.59283 | 0.32326 / 0.37289 | 0.57992 / 0.56636 | 0.21743 / 0.22831 |
| separation | +9.154499899370269e-05 | +0.0052810337560058596 | -0.049634201062554006 | +0.013559063938048066 | -0.010877030705821994 |
| **(d)** top-1 of 134 | 40 / 40 | **40 / 40** | 40 / 40 | 40 / 40 | 40 / 40 |
| **(e)** Pearson, cross block on within block | 0.9966259139824906 | **0.9978625736484857** | 0.9953637923362676 | 0.9978519351774923 | 0.9953115087444796 |
| **(c1)** frame-swap RMS | 0.00046527363060994793 | 0.01678923450971668 | 0.005946264752647632 | 0.022413594094173202 | 0.0031095796680573238 |
| **(c2b)** frame overlap, (seed 0, seed 1) | 0.021471366484559353 | **0.8909847593616453** | 0.5516482678579551 | 0.8726271703551144 | 0.3408947748323181 |
| **(c2b)** frame overlap, random vs random | 0.021452190429212037 | 0.880470664039374 | 0.5298311726958782 | 0.8628120764932482 | 0.32274381639764893 |
| **(c2a)** frame overlap, D isotropic | 0.021471366484559353 | 0.027020358441832752 | 0.025457150452877577 | 0.026547440591766877 | 0.025099209025528747 |
| **(c4)** attenuation slope | 0.026542111946942823 | 0.8638570822661489 | 0.5567380079295949 | 0.8582499881807516 | 0.37295974940594695 |
| **(a) / (c2b)** | 0.8411666943129359 | **0.7455668811085819** | 0.6998 | 0.7468 | 0.7420 |

### The ordering reverses, and that is the honest headline for a reader

Under Frobenius a cross-seed twin (**+0.0181**) is four times *less* similar than
two completely different traits sharing one initialisation (**+0.0721**). Under
the activation metric the twin is **+0.6643** and two different traits in one
seed are **+0.0876** - the twin is now seven and a half times the more similar
pair. Nothing about the adapters changed; only what "similar" counts as. That
reversal is what "the Frobenius number understates the agreement" means in
practice, and it is why the two halves of this page's answer must be quoted
together.

### Why the null moves too

The activation covariance is strongly anisotropic. On the eight modules whose
spectrum was computed in full (`#participation_ratio.pool445.eig_detail_stride31`),
the top eigenvalue holds 0.1977 to 0.5273 of the trace, the top 64 hold 0.6291
to 0.7781, and the mean activation direction alone holds 0.1011 to 0.5249
(`mean_direction_share_of_trace`). With an effective dimension of about ten and a
rank-64 frame, **two independent random frames already span nearly all the input
variance the model presents**, so the projection no longer throws the update
away: random-vs-random frame overlap is 0.880470664039374 under C against
0.021452190429212037 under Frobenius. The seed floor was never absence of signal;
in the functional metric it stops *looking* like absence of signal, and the
reason is exactly the reason [[seed-floor]] already gave.

Participation ratio by input width (`#participation_ratio.pool445.by_d_in`):
2560-wide modules **10.985324455819434** (n = 184), 4096-wide
**14.34423880409145** (n = 32), 9216-wide `down_proj` **89.14312697025213**
(n = 32). Centred, the same figures are 66.30565849712146, 50.58335750934323 and
133.7996308648069, mean **72.98587415904498**
(`#participation_ratio.pool445_centred`) - removing one direction roughly triples
the effective dimension, which is why the centred arm's null falls to 0.5516 and
its (a) to 0.3860. The ratio (a)/(c2b) is 0.6998 there, still nowhere near 3.

### Is the C-Gram really a sum over 248 modules?

The Gram sums raw inner products, so a module whose activations are large
dominates it. Effective number of modules, `(sum s)^2 / sum s^2` over each
module's share of `sum_i <dW_i, dW_i>`
(`#module_concentration.arms`): **153.16534241482879** under Frobenius and
**100.39659275176767** under C; the largest single module's share goes from
0.0091 (`model.layers.16.mlp.up_proj`) to 0.0223
(`model.layers.3.self_attn.q_proj`), and the top ten from 0.0887 to 0.1736. The
activation metric concentrates the sum, but not onto a handful of modules. The
class shares do move: `down_proj` falls from 0.0763 of the Frobenius total to
0.0044 of the C total and `linear_attn.out_proj` from 0.0582 to 0.0011, while
`o_proj` from 0.0201 to 0.0031, while `in_proj_qkv` rises from 0.1686 to 0.3019
and `q_proj` from 0.0537 to 0.0934. The three classes that lose are exactly the
three whose *inputs* are the 4096- and 9216-wide internal activations
(`down_proj` at 9216, `linear_attn.out_proj` and `o_proj` at 4096) rather than
the 2560-wide residual stream.

## The factor analysis is untouched

`analyse_fa_qwen35.py` was run unchanged on the C-Gram
(`PC_GRAM_NPZ` / `PC_FA_TAG=_actgram`, log
`phase10_runs/actgram_analysis.log`), and `analyse_fa_actgram.py` compares the
solutions by the method [[factor-analysis-fisher-metric]] used
(`analysis/fa_act_metric.json`).

- **Tucker congruence of the k = 5 oblimin loading matrices**, best one-to-one
  column matching: mean **0.9925535203706761**, minimum
  **0.9839152524212179**, **5 of 5** above the 0.95 "identical" threshold; per
  column 0.9933, 0.9981, 0.9839, 0.9946, 0.9929
  (`#loading_matrix_congruence_k5.pairs.frobenius_vs_act_weighted`). The
  response-side arm is closer still, mean 0.9952646148109873.
- **The Big Five congruences barely move**
  (`#arms.*.k5.one_to_one_matching`): Agreeableness 0.6555 to 0.6435,
  Conscientiousness 0.5744 to 0.5833, Extraversion 0.5389 to 0.5503, Emotional
  Stability 0.4049 to 0.4680, Intellect 0.6823 to 0.6790. All five targets taken
  exactly once in both arms. Mean of the best moves from
  **0.5712050407456282** to **0.5848142775831061**; the largest single move is
  Emotional Stability, **up** 0.0631. As in the Fisher metric, no congruence
  clears 0.85, so the ceiling [[factor-analysis]] reports is not a metric
  artefact either.
- **Retained count is unchanged**: parallel analysis at N = 1528 retains **9** in
  both arms, Kaiser (uncentred) **16** in both, reduced eig > 1 **9** in both, no
  Heywood cases (`#arms.*.n_factors`, `#arms.*.k5.n_heywood`).
- **The two Grams are near-copies of each other off the diagonal.** Over all
  8,911 within-seed pairs the C cosines correlate with the Frobenius ones at
  Pearson **0.9957682129458595**, Spearman 0.9960522400789175, with mean cosine
  rising only from 0.0720834758642461 to 0.0876041584393425
  (`#gram_cosine_agreement`). The metric that changes the cross-seed number by
  36.8x changes the *within-seed arrangement* hardly at all.
- **The per-adapter activation gain is nearly constant.**
  `||dW_i||_C / ||dW_i||_F` runs from **2.3587151786212397** (`talkative`) to
  **2.59817538261264** (`selfish`), a spread of 10 per cent
  (`#act_norm_per_trait`), and `||dW_i||_C^2` correlates with the Frobenius norm
  at Pearson 0.8595171605773058. Within one seed, the activation metric is close
  to a uniform rescaling - which is why the factor solution survives it, and why
  the interesting action is all across the seed boundary, where the *frames*
  differ.

## The verdict, by the preregistered thresholds

`PREREG_actgram.md` set: "partly a metric artefact" if (a) under C is at least
3 times the frame-overlap null under C **and** above 0.05; "real in the
functional metric" if (a) under C is within a factor of 1.5 of that null.

    (a)_C / (c2b)_C = 0.6642887281525424 / 0.8909847593616453 = 0.7455668811085819

which is below 1.5, so: **real in the functional metric**
(`analysis/act_gram.json#verdict`). Every secondary arm agrees - 0.6998 centred,
0.7468 on the response-side text, 0.7420 on the response-side centred text - and
so does the Frobenius arm at 0.8411666943129359. The cross-seed agreement is
between 70 and 84 per cent of what a perfectly reproduced update would score, in
every metric tested. The metric decides what that maximum *is*; it does not
decide how close the adapters get to it.

The number that does change, and that the preregistration named separately so it
could not be lost, is `ratio_a_C_over_a_F` = **36.78028836557651**.

## What this establishes, and what it does not

**Established.** (1) The Frobenius cross-seed cosine of 0.018 is a statement about
a metric that weights 2,560 input directions equally when the model uses about
ten of them; the functionally weighted number for the same 40 pairs is 0.664.
(2) The ordering reverses: in the functional metric a cross-seed twin is far more
similar than two different traits in one seed, where in the Frobenius metric it is
far less. (3) The floor is nevertheless *not* a metric artefact in the
preregistered sense - the attenuation tracks frame overlap in both metrics, to
within a few per cent, and the ratio of measured agreement to frame-overlap
ceiling is the same in both. (4) The five-factor solution, the retained count and
the Big Five congruences are all properties of the adapters, not of the metric;
this is the same conclusion [[factor-analysis-fisher-metric]] reached for the
Fisher metric, by a different route.

**Not established.** (1) This is C on the base model's **prompt** tokens for the
primary arm; the response-side check uses 24 prompts and 5,106 positions, which
is thin for a 9216-dimensional covariance, and neither arm is the distribution a
steered model generates. (2) Nothing here is behavioural. A cosine of 0.66 in the
activation metric says two updates change module outputs alike on these inputs;
it does not say the two adapters produce the same text, and no steering was run
through either. (3) C is estimated from a **bf16** forward; the Fisher work used
fp32 for a signal of 0.09 nats, and this quantity is far more robust, but the
comparison is not like for like. (4) 20,612 tokens estimate a 2560 x 2560
covariance at 8x oversampling and a 9216 x 9216 one at 2.2x; the participation
ratios for `down_proj` in particular are upper-bounded by the sample size, and
the smoke runs show PR rising with token count. (5) The uncentred C carries a
large mean-activation component; the centred arm is reported throughout for that
reason, and it does not change any verdict.

## Cost and provenance

`zoo-actgram.service`, app `pc-qwen35-phase11-actgram`, one A100-80GB, launched
2026-09-11 09:52:28 UTC, **378.2 seconds** inside the Modal function
(`analysis/act_gram.json#wall_seconds`), log `phase10_runs/actgram.log`. Two
smoke runs preceded it on `zoo-actgram-smoke.service`
(`pc-qwen35-phase11-actgram-smoke`): 76.7 s on 8 modules and 8 prompts, then
58.4 s on 36 modules covering all three input widths, logs
`phase10_runs/actgram_smoke.log*` and results
`phase10_runs/actgram_smoke.json`, `phase10_runs/actgram_smoke2.json`. The
per-run guard was the Modal function `timeout`, 90 minutes, because
`zoo40_meter.sh` is a cumulative workspace budget and not a per-run cap
([[costs]]). The meter read **$2563.40** before the first smoke and **$2563.93**
after the run, a draw of **$0.53** at its A100-40GB rate of $2.10 per GPU-hour
over 0.25 counted GPU-hours; the true GPU time was about 0.19 hours across three
containers, which at Modal's A100-80GB rate is under $0.50. The authorised cap
for this task was $10. All post-processing
(`run_actgram_analysis.sh`: `analyse_act_gram.py`, five `analyse_crossseed.py`
arms, two `analyse_fa_qwen35.py` runs and `analyse_fa_actgram.py`) is local CPU
and free, log `phase10_runs/actgram_analysis.log`.

## Across the stage boundary

The same machinery was pointed at the other cross-frame comparison this project
has - stage one against stage two, which share no LoRA-A either and read
+0.00017 in Frobenius. The answer there is the opposite of this page's: the
cross-stage cosine rises to +0.0090 under C, but as a fraction of the
frame-overlap ceiling it stays at 1.1 per cent against this page's 74.6 per cent
for a second seed. The metric was carrying the seed result and is not carrying
the stage result. See [[activation-weighted-gram-stages]].

Related: [[activation-weighted-gram-stages]], [[seed-floor]],
[[column-space-structure]],
[[factor-analysis-fisher-metric]], [[paper-asvd]], [[cross-seed-geometry]],
[[factor-analysis]], [[null-controls]], [[geometry-overview]],
[[stage-one-training-config]], [[shared-prompt-pool-445]], [[costs]],
[[glossary]].
