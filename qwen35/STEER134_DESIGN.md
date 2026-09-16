# STEER134: steering Qwen3.5-4B along the 134-trait sweep's directions

Generation: `steer134_on_modal.py` (Modal, app `pc-qwen35-phase7-steer134`,
A100-40GB, bf16, sdpa). Judging: `judge_steer134.py` (local, OpenRouter).
Adapters read from volume `pc-qwen35-sweep` root; outputs to `pc-qwen35-steer134`
at `/steer/gen/<condition>.json` (one file per condition, committed on write =
the resume checkpoint), mirrored to `results/steer134_gen/`.

## Direction semantics
All alphas are in units of **s_bar = mean adapter norm** = 1.6157 (from
`results/gram_sweep.npz`, scale 2.0, 248 modules), so one dose unit = one
mean-trait-adapter's worth of weight-space Frobenius norm.

- **trait:<name> (x134)** - the trait's own delta dW_t = 2.0*B@A. Not built
  densely: the PEFT adapter is loaded and every LoRA layer's scaling is set to
  `2.0 * alpha * s_bar / ||dW_t||` (negative alpha = negative scaling), giving
  effective delta `(alpha*s_bar/||dW_t||)*dW_t`.
- **pc1/pc2/pc3** - kernel PCA of the double-centred Gram, sweep100 math and
  sign convention (largest-|score| trait positive). Coefficients over raw
  deltas, `a^T G a = 1`; dense direction streamed from all 134 adapters
  in-container; applied as `W = W_base + alpha*s_bar*D`, base restored between
  doses from a one-time CPU snapshot.
  PC1+ = unsympathetic/insensitive/cold, PC1- = pleasant/sympathetic/agreeable;
  PC2+ = unrestrained/spunky, PC2- = introverted/careful/conscientious;
  PC3+ = unsophisticated/imperceptive, PC3- = verbal/philosophical/imaginative.
- **random1/random2** - matched-norm controls (seeds 1234/5678): PC1's
  coefficients permuted + re-signed, re-centred, PCs 1-3 deflated in the Gram
  metric, renormalised. Pre-deflation cos with PC1: -0.409 (random1), +0.189
  (random2); post-deflation < 1e-9 (recorded in `_meta.json` /
  `control_report` per condition; the run fills in the container-verified
  dense norms).

## Doses and probes
Traits: alpha in {-8,-4,-2,+2,+4,+8}. pc/random: {-8,-4,-2,-1,+1,+2,+4,+8}.
One shared alpha=0 baseline (`base_a+0.0`), generated once. Probes: the first
12 entries of sweep100/steer.py's GENERAL set, verbatim - trait-revealing,
topic-neutral, identical across all conditions. Generation: greedy,
seed 0, max_new_tokens 250, batch 16, left padding, `enable_thinking=False`
(asserted: the template must not leave an open `<think>` block).
Fan-out: 134*6 + 5*8 + 1 = **845 conditions = 10,140 generations**.

## Judging (two blind calls per response, v2-split protocol)
Judge `openai/gpt-5.6-terra`, temperature 0. CALL A: COHERENCE - rejudge.py's
rubric verbatim (both calibration examples, disclosed 250-token cap). CALL B:
trait conditions get **EXPRESSION** (judge sees only the first paragraph of the
trait's constitution + Q/A; 0-10 for behaviour; fluency/quality/safety
excluded; topic mention is not expression); pc/random/base get rejudge.py's
ALIGNMENT rubric verbatim. Decontamination: no direction/alpha/sign/condition/
sibling ever serialised; units pooled and shuffled (seed 20260822); cache
sha256(model|rubric|axis|trait-or-none|prompt|response) in
`results/steer134_judge_cache.jsonl`; duplicates collapse. Mechanical fluency
proxy (rejudge.py port) computed for every response.
Control expression is subsampled: per trait, random1/random2 at each trait
alpha plus base, on 4 of 12 prompts (`random.Random(0).sample(range(12),4)` =
[0,4,6,11], shared across traits). Output `results/steer134_judged.json`:
per-condition mean/CI (2000 bootstrap, seed 7) per axis, frac_truncated,
mean_chars; per-trait curves {trait, factor, keyed, doses:[alpha, expression,
coherence, ctrl_expression_at_same_alpha], baseline_expression}. Cells report
n/n_expected - coverage is stated, never silently narrowed.

## Costs
- **GPU**: 134 trait jobs x ~2.5 min + 5 dense jobs x ~20 min + base ~5 min
  = ~440 GPU-min = 7.3 hr x $2.10 = **~$15.4** (banner recomputes; refuses
  without PC_PHASE_BUDGET).
- **Judge**: coherence ~10,140 + expression 9,648 + ctrl/base expression
  134*13*4 = 6,968 + alignment 492 = **~27,250 calls** x $1.2/1000 =
  **~$33** (~30M tokens). `--dry-run` prints exact counts; > $40 requires
  `--confirm`; `--max-dollars` hard-stops on accumulated usage.

## Verification checks (recorded per condition, asserted in-container)
1. **Trait scaling rel-err**: live adapter norm recomputed via the r x r
   identity vs the npz norm (<1e-3), and per dose the read-back scaling's
   implied delta norm vs |alpha|*s_bar (<1e-3).
2. **Dense-direction norm err**: streamed ||D||_F vs 1.0 (<1e-3), plus PC
   projection-vs-loadings max err. Merge check per dose in fp32 before the
   bf16 cast (rel err <1e-3); the post-cast norm is recorded, not asserted -
   at |alpha|=1 the per-element delta sits near the bf16 quantum, which is a
   dtype fact, not a merge fault.
3. **Base-restoration byte check**: after the last dose the snapshot is
   restored and `torch.equal` verified per module.
