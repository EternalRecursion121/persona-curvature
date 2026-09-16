---
title: The null control arms
summary: Shuffled, permuted and seed-paired arms; all 240 control adapters were first trained at the wrong objective, and the matched retrains changed the null table by nothing to three decimal places.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/PREREGISTRATION_phase3.md
  - qwen35/make_nulls.py
  - qwen35/nulls_manifest.json
  - qwen35/launch_nulls.sh
  - qwen35/run_nulls.sh
  - qwen35/compare_nulls.py
  - qwen35/results/compare_nulls_output.txt
  - qwen35/results/decomposition_shuffled_matched.json
  - qwen35/results/decomposition_permuted_matched.json
  - qwen35/results/decomposition_shuffled.json
  - qwen35/results/decomposition_permuted.json
  - qwen35/analysis/scree_null_matched.json
last_verified: 2026-09-16
tags: [geometry, nulls]
---

# The null control arms

## The three arms and what each destroys

Built by `qwen35/make_nulls.py` from `data_common`, the corpus the 134 zoo
adapters actually trained on. Every variant keeps the same prompts in the same
order, the same documents, the same token counts, the same file layout, the same
trainer and the same hyperparameters, and differs only in one thing.

- **shuffled** (`data_null_shuffled_p100`): chosen and rejected are swapped on
  exactly `floor(n/2)` of each trait's pairs - a fixed 0.5, not a binomial draw,
  so the destroyed fraction is identical for every trait. Rows that are not
  flipped are written back byte-identically, which the script's `verify()`
  asserts. The preference direction is destroyed; nothing coherent is learned.
- **permuted** (`data_null_permuted_p100`): each trait *name* receives another
  trait's intact pair set, under a uniform derangement (resample-until-valid, so
  no fixed points and no alphabetical-neighbour bias). The mapping is stored in
  `qwen35/nulls_manifest.json#permutation_dst_to_src` for all
  `#n_traits = 134` traits (e.g. `active -> thrifty`, `cold -> pleasant`,
  `extraverted -> creative`). Each record keeps its original honest
  trait/factor/keyed fields so the permutation can be audited from the corpus.
- **seed-paired** (`data_null_seedpaired_s40`): a byte-identical copy of the real
  corpus, retrained with `--seed 1 --order-seed 1`. `make_nulls.py` is emphatic
  that this **is not a null**: it is the noise floor, and it is the arm that gets
  misread. See [[seed-floor]].

`nulls_manifest.json` also stores `#seed = 0`,
`#source = ".../qwen35/data_common"`, the full
`#shuffled_flipped_indices` per trait file, and
`#seed_paired_note = "data byte-identical to source; vary train_qwen35.py SEED
and ORDER_SEED at launch, not the data"`.

Training is `qwen35/launch_nulls.sh` (one Modal app per arm, so billing
attributes itself); analysis is `qwen35/run_nulls.sh` (Gram, then
`decompose.py` with per-arm `--out` and `--runmeta`, fingerprinting
`results/gram_sweep.npz` and `results/decomposition.json` before and after every
step and aborting if either moved).

## Two training failures, recorded because the logs are still on disk

**`.RSLORA-INVALID`** (`phase3_shuffled.RSLORA-INVALID.log`,
`phase3_permuted.RSLORA-INVALID.log`,
`phase3_seedpaired.RSLORA-INVALID.log`). These runs completed. They were
invalidated afterwards because the trainer's default said rsLoRA while the sweep
used plain LoRA. `qwen35/launch_nulls.sh` records it verbatim next to the
now-redundant `export PC_USE_RSLORA=0`:

> STATED EXPLICITLY EVEN THOUGH THE DEFAULT NOW MATCHES IT. Samuel chose plain
> LoRA; the trainer's default said rsLoRA, and because that choice lived only on
> the phase-5 launch line this script silently trained 240 adapters at scale 16.0
> against the sweep's 2.0 and the whole phase had to be rerun.

Plain LoRA scale is `alpha/r = 128/64 = 2.0`; rsLoRA is `alpha/sqrt(r) = 16.0`.
240 adapters were trained at the wrong scale and discarded.

**`.FAILED-datarace`** (`phase3_shuffled.FAILED-datarace.log`,
`phase3_permuted.FAILED-datarace.log`). These runs were killed by the trainer's
own corpus-content gate at `train_qwen35.py:1050`. The permuted log ends:

> RuntimeError: corpus content mismatch for trait 'assertive': container computed
> c5a8ed25... from /adapters/_data/assertive.jsonl, driver declared corpus
> 'data_null_permuted_p100' with 92aaec48... The prompt pool matched, so this is
> a real-vs-null corpus swap or a stale upload -- the two are indistinguishable
> by prompts alone. BLOCKER.

The shuffled log fails the same way on trait `fretful`
(container 864cd2b4..., driver a1f30fce...). A sha256 of the actual jsonl inside
the container disagreed with the corpus the driver declared, so the arm refused
to train. The prompt-pool hash alone would not have caught it.

`PHASE3_VERDICT.md` costs the waste at "$0.54 of waste from the two arms I
killed on 08-20".

## The result, original arms

`qwen35/results/compare_nulls_output.txt` (produced by
`qwen35/compare_nulls.py`), residual view, leading component removed:

| arm | n | TEST 1B diff | p | TEST 2 gap | p | TEST 2B diff | ARI | ARI p |
|---|---|---|---|---|---|---|---|---|
| REAL | 134 | +0.1216 | 0.00005 | +0.2393 | 0.00005 | +0.0433 | +0.0989 | 0.00050 |
| shuffled | 100 | +0.0001 | 0.46358 | +0.0002 | 0.44403 | +0.0003 | +0.0044 | 0.36232 |
| permuted | 100 | +0.0075 | 0.07860 | +0.0174 | 0.02310 | -0.0038 | -0.0131 | 0.83508 |

TEST 1B is the polarity-signed within-factor minus between-factor cosine, TEST 2
the bipolarity gap (same-keyed minus opposite-keyed within factor), TEST 2B the
label-free separation plus clustering ARI. The stop condition, preregistered in
`PREREGISTRATION_phase3.md`, was a null arm reproducing the signed factor
separation at or above about +0.12 residual with p < 0.05. Neither arm does, so
both PASS.

`PHASE3_VERDICT.md` reports the permuted arm's nominally significant bipolarity
gap rather than burying it: +0.0174 at p = 0.023 against the real +0.2393, one
fourteenth of the effect, out of roughly thirty-six p-values computed across
three arms, two views and six statistics. "It should be treated as unexplained
rather than explained."

The most informative thing the nulls produced was not preregistered - the
dimensionality contrast. Participation ratio
(`results/decomposition*.json#spectrum.participation_ratio`):
real **25.720397850313308**, permuted **23.89318747707058**, shuffled
**99.44402608749816**. The permuted arm trains on real coherent preference pairs
under the wrong labels and produces the same low-dimensional geometry while
scoring zero against the labels. PHASE3_VERDICT's reading:
"**coherent preference training creates the low-dimensional structure; trait
identity determines where in it each trait lands.** Those are two separable
findings and only the second needed the labels."

## The objective mismatch, found 2026-08-24

`PHASE3_VERDICT.md`, addendum 2026-08-24. Found by comparing against the
now-public Persona Cartography reference code
(`vendor/persona-cartography`), whose pipeline hard-codes the auxiliary loss
terms into every run and cannot launch a control without them.

> the null launcher carried the corrected LoRA scale but not the loss
> configuration, so all 240 control adapters (shuffled, permuted, seed-paired)
> trained under plain sigmoid DPO -- loss_type ["sigmoid"], kl_coef 0.0 -- while
> all 134 sweep adapters trained with loss_type ["sigmoid","sft"], kl_coef 0.001.
> Verified in every arm's runmeta. This is the third occurrence of this project's
> signature failure mode (a treatment silently not travelling to the container),
> and it recurred on the parameter class the existing gates did not check.

Consequences as stated there: the null table compares arms differing from the
signal arm in objective as well as treatment (directionally conservative, since
the nulls were denied an auxiliary term that adds shared structure and still
failed to show factor structure), and the clean statement would cost about $82 of
retraining. Tooling was changed: `cross_gram_on_modal.py` now refuses matched
pairs whose runmeta disagree on `loss_type` / `loss_weights` / `kl_coef`, the
same way it already refused `data_sha256` mismatches.

## The matched-objective retrains, 2026-09-05

Shuffled and permuted were retrained, 100 traits each, at the zoo's objective
(`sigmoid,sft` / weights `1.0,0.1` / `kl_coef 0.001` / rslora off) into
`data_null_{shuffled,permuted}_p100_matched`; Grams and decompositions redone
with the same labels and a per-arm runmeta path.

`PHASE3_VERDICT.md`, addendum 2026-09-05:

| TEST 1B residual diff | original | matched |
|---|---|---|
| shuffled | +0.0001 | +0.0001 (p 0.46) PASS |
| permuted | +0.0075 | +0.0080 (p 0.07) PASS, same TEST 2 flag |
| scree: real PCs above shuffled | 11 | 11 |
| scree: real PCs above permuted | 0 | 0 |

Confirmed against the files:

- `results/decomposition_shuffled_matched.json#test1b."leading-component
  removed".diff = 8.707649293971717e-05`, `p = 0.46452677366131695`
  (original arm: `8.941333926592288e-05`, `p = 0.46357682115894205`)
- `results/decomposition_permuted_matched.json#test1b."leading-component
  removed".diff = 0.008049902642370769`, `p = 0.07004649767511624`
  (original: `0.007532766644382829`, `p = 0.07859607019649018`)
- permuted TEST 2 residual gap matched `0.017699375553737642` at
  `p = 0.021898905054747262` (original `0.0174442687347812`, `p = 0.023098845...`)
- participation ratio matched: shuffled 98.95932712056498, permuted
  24.10056447225465
- `analysis/scree_null_matched.json#n_above_structureless = 11`,
  `#n_above_null = 0`

> The 2026-08-24 caveat on the null table is discharged: the objective mismatch
> changed nothing to three decimal places.

`TEST 6` (the nuisance-covariate test on reward margins) is unavailable in both
matched arms: `#test6.status = "unavailable: [Errno 2] No such file or
directory: 'results/runmeta_shuffled_matched.json'"` and the permuted
equivalent. That is deliberate - `run_nulls.sh` explains that pointing at a
non-existent per-arm runmeta is the honest state, because the default
`runmeta_sweep.json` would silently hand the null arm the **real** sweep's reward
margins.

One thing the addendum does not mention: the matched shuffled arm's label-free
clustering statistic moved. `results/decomposition_shuffled_matched.json#test2b.
"leading-component removed".ARI = 0.05977870383675816` at
`ari_p = 0.0009995002498750624`, where the original shuffled arm had
`ARI = 0.00438485865621544` at `ari_p = 0.36231884057971014`. That is a
nominally significant clustering result on an arm whose headline is flat, and it
belongs beside the permuted TEST 2 flag as unexplained. `test2b` residual `diff`
itself stayed tiny (0.00029 vs 0.00025, `p` 0.090 vs 0.136).

## Status of `compare_nulls.py` and its stored output

`qwen35/results/compare_nulls_output.txt` is the **original**-objective run and
its final verdict is explicitly overridden by `PHASE3_VERDICT.md`. It ends:

> VERDICT: NO -- a trait trained twice looks NO MORE ALIKE than two different
> traits of the same factor and keying (+0.0167 vs +0.2467). The geometry is seed
> noise. EVERY EFFECT ABOVE IS VOID

PHASE3_VERDICT's section "Where I depart from the tooling, and why the departure
is testable" says of that conclusion: "**That conclusion is wrong, and I wrote
the check that produced it.**" It conflates coordinates failing to correspond
across bases with organisation failing to replicate. See [[cross-seed-geometry]]
and [[superseded-geometry-claims]]. The stored file has not been regenerated and
should be read as a historical artefact, not as a current verdict.

Related: [[seed-floor]], [[cross-seed-geometry]], [[pca-and-scree]],
[[polarity-and-bipolarity]], [[superseded-geometry-claims]],
[[zoo-training-recipe]], [[persona-cartography-paper]].
