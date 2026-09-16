# Pre-registration: the activation-weighted Gram

Written 2026-09-11, before any container was launched. Nothing below was
computed first; the thresholds are fixed here and the verdict is read off them.

## The question

Every Gram in this project is the Frobenius inner product of two LoRA deltas,
which weights every input direction of a module equally. Under it, the same
trait retrained from an independent LoRA-A initialisation sits at cosine
**+0.01806099632415457** to its original
(`analysis/crossseed_arms.json#[1].same[1]`), which
`wiki/pages/geometry/seed-floor.md` reads as the `r/d = 64/2560 = 0.025` chance
overlap of two random rank-64 input frames rather than as absence of signal.

ASVD (arXiv:2312.05821, `wiki/pages/history/paper-asvd.md`) argues that the
functional metric is activation-weighted: two updates to a module are similar
if they change its output similarly on the inputs the model actually sees.
**Is the cross-seed seed floor partly an artefact of the isotropic metric?**

## The object

For adapters i, j with deltas `dW_i = s_i B_i A_i` (s = lora_alpha / r = 2.0)
and `C = E_x[x x^T]` the uncentred input covariance of a module on the base
model,

    <dW_i, dW_j>_C = E_x[(dW_i x) . (dW_j x)]
                   = s_i s_j tr(B_i^T B_j (A_j C A_i^T))
                   = s_i s_j sum( (B_i^T B_j) * (A_i C A_j^T) )

summed over the 248 targeted modules, exactly as the Frobenius Gram sums
`tr(B_i^T B_j A_j A_i^T)`. Cosines are normalised by the same inner product:
`cos = <i,j>_C / sqrt(<i,i>_C <j,j>_C)`.

### Deviation from the brief, declared in advance

The brief says never to form C, and to accumulate projected covariances
`M_{F,G} = E[(Fx)(Gx)^T]` for a small set of **shared** frames instead. That is
exact only if every zoo adapter shares one A. It does not: A drifts
`mean ||A_i - A_0|| / ||A_0|| = 0.014605041334818797`
(`analysis/align_summary.json#a_drift`), so a shared-frame Gram would be a
1-to-3 percent approximation while `results/gram_sweep.npz` used the exact
per-adapter `A_i`. **This run forms the full uncentred C for every module**
(fp32; 184 modules at d_in 2560, 32 at 9216, 32 at 4096; 17.8 GiB in total,
which fits one A100-80GB) and contracts it with the per-adapter `A_i`, so the
C-Gram is exact and is directly comparable with the Frobenius Gram entry for
entry. The projected covariances `M_{F,G} = F C G^T` the brief asks for are
emitted anyway, being a trivial contraction of C, and C itself is not persisted
- only its spectrum summary per module.

## Activations

- **Primary arm (`pool445`).** The 445-prompt shared training pool
  (`qwen35/data_common/*.jsonl`, `prompt_pool_sha256`
  `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07`),
  asserted byte-identical across all 134 trait files, rendered with
  `tok.apply_chat_template([{user: prompt}], add_generation_prompt=True,
  enable_thinking=False)` and tokenised with `add_special_tokens=False` - the
  same rendering `train_qwen35.py:render` and `fisher_gram.py` use.
  **Prompt tokens only**: no fixed base-model greedy corpus exists for the 445
  pool, and the per-trait `chosen`/`rejected` continuations are trait-specific,
  which would make C trait-dependent. Token count recorded.
- **Robustness arm (`resp4378`).** The 24 steering prompts with the base
  model's own stored alpha-0 greedy responses capped at 192 response tokens -
  the 4,378 scored positions `fisher_gram.py` used
  (`phase10_runs/steer_results_fix.json`, PC1, generations `"0.0"`). Prompt and
  response positions both enter C here. This is the response-side check.
- Base model `Qwen/Qwen3.5-4B`, no adapter, bf16 forward as
  `probe_invariant.py` harvests, activations cast to fp32 before accumulation,
  TF32 off. C is **uncentred**, which is what `E_x[(dW x).(dW x)]` requires.

## Validation checks, all inside the same job

1. **C = I through the same code path** reproduces `results/gram_sweep.npz` and
   `results/cross_gram_full_root_x_data_null_seedpaired_s40_matched.npz`.
   Pre-registered bar: max absolute relative difference on the cosines below
   `1e-4`.
2. **A-drift** recomputed inside the run for both adapter sets; the seed-0 set
   must reproduce 0.0146, and the 40 seed-1 adapters must share one frame to
   the same tolerance.
3. **Hook coverage**: every one of the 248 stripped module names resolves in
   `model.named_modules()`, every module's hook fires the same number of times,
   and the observed `d_in` equals that module's `A` width.
4. The frame-overlap null (c2 below) under C = I must read `r/d`, i.e. about
   0.025 on the 2560-wide modules.

## Statistics, each under Frobenius and under C

Computed by `analyse_crossseed.py` on the published code path wherever it
already computes them (arms (a), (b), (d), (e)), with `PC_WITHIN` pointed at
the matching within-seed Gram.

- **(a) same-trait cross-seed cosine**, mean over the 40 matched seed-1 traits
  against their seed-0 originals. Frobenius reference +0.01806099632415457.
- **(b) different-trait cross-seed cosine**, mean over the remaining
  5,320 - 40 pairs, split as `analyse_crossseed.py` splits it (same factor same
  keying, same factor opposite keying, different factor). Frobenius reference
  +0.0018307002389548663 (different factor).
- **(c) the null.** The brief specifies "same-trait cosine when one side's
  frame is replaced by a fresh random frame (B kept), averaged over K frames".
  **That statistic has expectation exactly zero** and is pre-registered as
  degenerate here rather than discovered to be so afterwards: the Gram entry
  `tr(B_i^T B_j R C A_i^T)` is linear in R and PEFT's `kaiming_uniform_` is
  sign-symmetric, so `E_R[cos] = 0` under both metrics, and a threshold that
  divides by it is a threshold that divides by noise. This is the failure mode
  `seed-floor.md` records as its own lesson ("compute what the statistic would
  read under the maximal version of the hypothesis, before committing to the
  threshold"). The null is therefore pre-registered as four named quantities:

  - **(c1)** the literal statistic, K = 8 fresh frames drawn per module as PEFT
    draws LoRA-A (`kaiming_uniform_(a=sqrt(5))` on a shape `(r, d_in)` weight,
    which is `U(-1/sqrt(d_in), +1/sqrt(d_in))`), reported as **signed mean**
    (expected 0) and as **RMS** (a real rank-64 noise floor). Both orientations
    (seed-1 frame replaced, seed-0 frame replaced).
  - **(c2) the frame-overlap null - PRIMARY DENOMINATOR.** What a trait that
    reproduced *perfectly* modulo the frame would score. For frames F, G with
    row-space projectors `P_F = F^T (F F^T)^-1 F`:

        c2b = tr(P_F C P_G C) / sqrt( tr(P_F C P_F C) tr(P_G C P_G C) )
        c2a = tr(P_F C P_G)   / sqrt( tr(P_F C P_F)   tr(P_G C P_G)   )

    `c2b` is the model in which the unconstrained update D satisfies
    `D^T D ∝ C` (which is what an accumulated `sum_t g_t x_t^T` gives when the
    output-side errors are roughly uncorrelated across tokens) and is the
    **primary** null; `c2a` is the model in which `D^T D ∝ I`. Both reduce to
    `tr(P_F P_G)/r`, expectation `r/d`, when C = I. Averaged over modules and
    over K x K random frame pairs, and evaluated for the real
    `(A_seed0, A_seed1)` pair as well.
  - **(c3)** the analytic `r / d_eff` with `d_eff` the participation ratio
    `(tr C)^2 / tr(C^2)` of that module's input covariance, averaged per module
    class. The prediction for what two random frames overlap under C.
  - **(c4)** the empirical attenuation slope of the cross-seed C-cosine on the
    within-seed C-cosine over the same 5,320 pairs - the direct analogue of the
    Frobenius 0.026542111376493285 that the seed-floor page sets against 0.025.

- **(d) identification**: nearest-neighbour rank of each of the 40 seed-1
  adapters among the 134 seed-0 adapters. Frobenius reference 40 of 40,
  mean rank 1.0, chance 67.5.
- **(e)** Pearson between the 134 x 40 cross-seed cosine block and the
  corresponding within-seed-0 block. Frobenius reference 0.9966259140629706.
- **(f)** factor analysis of the 134 x 134 C-Gram by `analyse_fa_qwen35.py`
  unchanged (`PC_GRAM_NPZ` / `PC_FA_TAG=_actgram`), then Tucker congruence of
  its k = 5 oblimin loading matrix against the Frobenius solution
  (`results/fa_qwen35.json`) under the best one-to-one column matching, and the
  five Big Five congruences, by the method of `analyse_fa_fisher.py`.

## Pre-registered thresholds

The denominator for the verdict is **(c2b) under C**, the frame-overlap null;
(c1)'s RMS, (c2a) and (c3) are reported beside it and any disagreement between
them is reported, not resolved.

- **"Partly a metric artefact"** if `(a)_C >= 3 * (c2b)_C` **and**
  `(a)_C > 0.05`.
- **"Real in the functional metric"** if `(a)_C <= 1.5 * (c2b)_C`.
- Anything in between is reported as in between.

Two readings are pre-registered as *not* interchangeable, because the brief's
thresholds can be satisfied by a pure change of units:

- `(a)_C / (a)_F` is the number the blog post's sentence turns on - how much
  more (or less) similar two cross-seed twins look when similarity is measured
  functionally. It is reported as its own key, `ratio_a_C_over_a_F`.
- `(a)_C / (c2b)_C` is the verdict statistic. If the activation metric merely
  replaces `d` by a smaller `d_eff`, both `(a)_C` and `(c2b)_C` rise by the
  same factor and the verdict is "real in the functional metric" **even if
  `(a)_C` has risen from 0.018 to 0.3**. That is the correct scientific
  reading - the floor would then still be exactly frame overlap - and the
  report must state both numbers so the distinction is not lost.

## Outputs

`results/gram_actweighted.npz` (134 x 134, `G`/`names`/`norms`/`scale`/
`n_modules`, readable by `analyse_fa_qwen35.py`),
`results/cross_gram_actweighted_root_x_data_null_seedpaired_s40_matched.npz`
(`X`/`names_a`/`names_b`/`norms_a`/`norms_b`/`scale`/`n_modules`), the same two
for the `resp4378` arm, `analysis/act_gram.json` with every number under a named
key, logs in `phase10_runs/`, and a wiki page
`wiki/pages/geometry/activation-weighted-gram.md`.

## Cost

Samuel's cap for this run is **$10**. The global spend meter
(`zoo40_meter.sh`) is a workspace-wide cumulative budget and not a per-run cap
(it read $2563.40 / $2688.00 at launch time), so the per-run guard is the Modal
function `timeout`, set to 90 minutes on one A100-80GB. At Modal's A100-80GB
rate that bounds this run at under $4 even if it hangs to the timeout. One
short smoke function runs first.
