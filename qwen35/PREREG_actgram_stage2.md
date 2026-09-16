# Pre-registration: the activation-weighted Gram across the stage boundary

Written 2026-09-11, before any container was launched for it. Follows
`qwen35/PREREG_actgram.md` and reuses its machinery unchanged.

## The question

Stage one (DPO on the constitution's preference pairs) and stage two (SFT on
self-generated introspective transcripts) are two different LoRAs with two
different random LoRA-A draws - stage one at seed 0, stage two at
`sft_seed 123456`, one draw shared within each stage
(`wiki/pages/geometry/stage-two-geometry.md`). Their Frobenius cross-Gram is
therefore zero by construction: same-trait cosine
**+0.00016589283708173512**, different-trait **+1.784355184601498e-05**, grand
means **+0.00016761445563503107**
(`qwen35/analysis/stage2_structure.json#stage1_x_stage2_exact`), and the only
thing that survives is that a stage-one adapter's nearest stage-two adapter is
its own trait for **42 of 134**, mean rank **9.537313432835822**.

[[activation-weighted-gram]] showed that the isotropic metric is what makes a
cross-frame comparison read as zero: the input covariance C of a module has a
participation ratio of about ten, so two independent rank-64 frames overlap at
0.89 in C where they overlap at 0.021 in Frobenius. **Under C the two stages can
be compared.** The question is whether the stage-two residual is *the same
function* as stage one's update for the same trait, or a different function whose
arrangement across traits happens to match (which is what the r = 0.81 centred
cosine correlation on `stage2_structure.json#centred_cosines` already
establishes).

## What is computed

One Modal job, `qwen35/act_gram_stages_on_modal.py`, harvesting C exactly as
`act_gram_on_modal.py` did (base `Qwen/Qwen3.5-4B`, no adapter, bf16 forward,
fp32 accumulation, TF32 off, uncentred C for all 248 modules) on the same two
token arms - `pool445` (445-prompt pool, prompt tokens only, 20,612 tokens) and
`resp4378` (24 prompts plus the base model's stored greedy responses, 5,106
tokens) - plus a mean-centred twin of the primary arm. Then one full Gram over
four adapter sets:

| tag | volume | path | n |
|---|---|---|---|
| `s1s0` | `pc-qwen35-sweep` | `/` | 134 |
| `s2s0` | `pc-qwen35-oct2` | `/loras_introspection` | 134 |
| `s1s1` | `pc-qwen35-sweep` | `/data_null_seedpaired_s40_matched` | 40 |
| `s2s1` | `pc-qwen35-oct2` | `/seed1/loras_introspection` | 15 |

All at `lora_alpha/r = 2.0`, 248 modules. Everything else is derived locally from
that Gram, which is exact.

**Removing stage two's shared direction** is done at the Gram level and is
therefore exact: with `M2 = (1/134) sum_k v_k` the stage-two grand mean,
`<u, v_k - M2> = <u, v_k> - mean_k <u, v_k>` and
`||v_k - M2||^2 = G22_kk - 2 mean_l G22_kl + mean_{l,m} G22_lm`. No re-run.

## Statistics, under C (primary, `pool445`) and under Frobenius (reference)

1. **The 134 x 134 stage-one x stage-two block.** Same-trait cosine mean, sd,
   min, max; different-trait mean and sd; nearest-neighbour identification of
   each stage-two adapter among the 134 stage-one adapters (top-1 count, mean
   rank, chance 67.5) and the reverse direction, which is the one
   `stage2_structure.json` reports.
2. **The frame-overlap ceiling for these two frames specifically**, by
   `PREREG_actgram.md`'s (c2b):
   `tr(P_F C P_G C) / sqrt(tr(P_F C P_F C) tr(P_G C P_G C))` with
   `P_F = F^T (F F^T)^-1 F`, evaluated on the real (stage-one A, stage-two A)
   pair for each of the 134 traits, and on random frames for calibration. Under
   C = I it must read the width-averaged `r/d` of 0.0215 that
   [[activation-weighted-gram]] recorded.
3. **(1) and (2) again after removing stage two's shared direction**, which holds
   0.1514038882692439 of the average squared norm in stage two
   (`stage2_structure.json#shared_component.stage2.mean_direction_norm2_over_mean_norm2`),
   every adapter sitting at cosine 0.38929785188872645 to it.
4. **The cosine between the two grand means under C** (Frobenius
   +0.00016761445563503107).
5. **Arrangement.** Pearson between the off-diagonal cross-stage cosines and the
   stage-one within-stage cosines - the direct analogue of
   `stage2_structure.json#stage1_x_stage2_exact.corr_C12_offdiag_with_C1` =
   0.21750655535315414 and `...with_C2` = 0.20648778534428447 - computed raw and
   after the shared-direction removal; and, separately, the Pearson between the
   **centred** within-stage-one and within-stage-two off-diagonal cosines, the
   analogue of `#centred_cosines.corr_stage1_stage2` = 0.811787783969743.
6. **The second seed as a check.** The same cross-stage same-trait cosine and
   ceiling for the 15 traits that have both a seed-1 stage-one adapter and a
   seed-1 stage-two adapter, and the stage-one cross-seed and stage-two
   cross-seed blocks as controls in the same metric.
7. **Frame spread** per set (each adapter's relative deviation of `A` from its
   set mean), because stage two's `A` drifts about 10 percent from its
   initialisation against stage one's 1.5 percent
   (`PHASE3_VERDICT.md`, 2026-09-05), so "one shared frame" is weaker there. The
   Gram itself uses per-adapter `A` and is exact regardless.

## Pre-registered thresholds

The benchmark for "the same function, re-learned in a different frame" is the
stage-one **cross-seed** arm measured in the same metric on the same run:
same-trait cosine over its own frame-overlap ceiling, which
[[activation-weighted-gram]] measured at **0.7455668811085819**. The benchmark
for "unrelated" is the different-trait cross-seed ratio there,
0.0712723 / 0.8909848 = **0.0800**.

Let `R = (same-trait cross-stage cosine under C) / (cross-stage frame-overlap
ceiling under C)`, and `R_res` the same after stage two's shared direction is
removed. **`R_res` is the primary statistic** because two generic components -
stage one's shared component and stage two's introspection register - can carry a
raw cross-stage cosine without any trait-specific agreement at all.

- **"The stage-two residual is the same function as stage one"** if
  `R_res >= 0.50` (at least two thirds of the cross-seed re-learning benchmark)
  **and** same-trait exceeds different-trait by a factor of 3 or more in the
  residual block.
- **"A different function with a matching arrangement"** if `R_res <= 0.15`
  (at most about twice the unrelated benchmark) **while** identification still
  beats chance and statistic 5 remains substantially positive.
- Between 0.15 and 0.50, or if the two conditions of either verdict disagree,
  report as partial and say which half held.

`R` (before removing the shared direction) is reported beside `R_res` under its
own key and is not allowed to stand in for it.

## A caveat that is stated in advance, not discovered

**Stage two was trained on the DPO-merged base**, `merged(Qwen/Qwen3.5-4B +
stage1 <trait>)` at merge weight 1.0 for stage one
(`phase10_runs/results_oct2_15traits_...json#stages.sft[].base`), and on
self-generated introspective transcripts, not on the 445-prompt pool. The C used
here is the **base model's** on the 445 pool. It is therefore a first-order
approximation for the stage-two side: a per-trait C on each merged model is 134
different covariances and would make the Gram trait-dependent, which the design
cannot allow. Every cross-stage number on this page inherits that approximation,
and the page will say so. The `resp4378` arm is the only available sensitivity
check on the text, and it is on the base model too.

## Outputs

`qwen35/results/cross_gram_actweighted_stage1_x_stage2.npz`,
`qwen35/analysis/act_gram_stage2.json`, logs in `qwen35/phase10_runs/`, and a
wiki page `wiki/pages/geometry/activation-weighted-gram-stages.md` linked from
`activation-weighted-gram`, `stage-two-structure` and `stage-two-exploration`.

## Cost

Samuel's cap is **$10**. The per-run guard is the Modal function `timeout`, set
to 90 minutes on one A100-80GB, because `zoo40_meter.sh` is a cumulative
workspace budget and not a per-run cap. The previous run of this machinery cost
about $0.50; this one reads 323 adapters instead of 174 and is expected to cost
about $1.50. A smoke run precedes it.

---

## Amendment, 2026-09-11 12:15 UTC, before the full run returned

Written after the smoke run only (`phase10_runs/actgram_stages_smoke.json`:
36 of 248 modules, 12 adapters per set, 4 prompts) and before
`phase10_runs/actgram_stage2_results.json` existed.

**The "same/diff >= 3" half of the "same function" threshold is ill-defined for
the residual block and is replaced.** Removing stage two's grand mean centres
every *row* of the cross block exactly: `mean_k <u_i, v_k - M2> = 0` by
construction. So in the residual block the different-trait mean is forced to
about `-same/(n-1)`, is negative whenever the same-trait mean is positive, and a
ratio to it carries no information. The smoke returned `same_over_diff = -14.39`,
which is the arithmetic saying so, not a finding.

It is replaced by the same comparison in a form the centring does not destroy:

    same-trait mean / (standard deviation of the different-trait cosines
                       in the same block)  >=  3

with the same factor of 3, reported as `same_over_diff_sd`. `same_over_diff` is
still reported, with this note attached, and is not used.

Nothing else changes. In particular `R_res` and its 0.50 / 0.15 bands, and the
"different function" branch's two side conditions (identification above chance,
arrangement correlation above 0.2), stand exactly as written above.

### The two side conditions of the "different function" branch, in numbers

Appended 2026-09-11 12:18 UTC, still before `actgram_stage2_results.json`
exists. The prose above says "identification still beats chance" and "statistic 5
remains substantially positive"; those are operationalised, so that the code and
the prose cannot disagree afterwards, as

- **identification**: more than `n/10` of the stage-two adapters find their own
  stage-one adapter as nearest neighbour in the residual block (more than 13 of
  134 against a chance expectation of about 1), and
- **arrangement**: the Pearson between the residual cross-stage off-diagonal
  cosines and the stage-one within-stage cosines exceeds **0.20**. The Frobenius
  reference for the raw block is 0.21750655535315414
  (`stage2_structure.json#stage1_x_stage2_exact.corr_C12_offdiag_with_C1`), so a
  value near 0.20 is a real possibility and the bar is deliberately set at the
  value, not above it. If the verdict turns on this condition rather than on
  `R_res`, the report must say so rather than move the bar.

### Two reproduction checks, declared with the rest

The Frobenius arm of this run must reproduce
`qwen35/analysis/stage2_structure.json#stage1_x_stage2_exact`, which was derived
by a completely different route (`X12 = (X[stage1, persona] - G1) / 0.25`): same
trait +0.00016589283708173512, different trait +1.784355184601498e-05, top-1
stage one finds own stage two 42 of 134, mean rank 9.537313432835822, grand means
+0.00016761445563503107, `corr_C12_offdiag_with_C1` 0.21750655535315414. Bar:
agreement to 1e-5 absolute on the cosines. It must also reproduce the stage-two
within-Gram cosines of
`results/cross_gram_full_loras_introspection_x_loras_introspection.npz`. A
disagreement beyond those bars is a source contradiction and gets an S-number.
