# Phase 3 verdict, against PREREGISTRATION_phase3.md

Written 2026-08-23 after reading the null results. The preregistration was written
2026-08-20 12:31 while the arms were still training, and it binds: where it commits
a verdict, that verdict is taken even where it costs the headline.

## One sentence

**The trait-level claim is withdrawn and the factor-level claim is upheld and now
independently replicated** — individual trait directions are an artefact of the LoRA
initialisation, and the five-bipolar-axis organisation of those directions is not.

## The three arms

| arm | signed factor separation (resid) | bipolarity gap | verdict |
|---|---|---|---|
| REAL (134 traits, seed 0) | **+0.1216** p at floor | **+0.2393** p at floor | — |
| SHUFFLED (100, preference destroyed) | +0.0001 p=0.46 | +0.0002 p=0.44 | PASS, flat |
| PERMUTED (100, trait↔data broken) | +0.0075 p=0.079 | **+0.0174 p=0.023** | PASS, see below |

Both real nulls are flat on the headline statistic. Neither reproduces the effect,
so it is not a property of the pipeline.

**The permuted arm's bipolarity gap is nominally significant and I am reporting it
rather than burying it.** It is +0.0174 against the real +0.2393 — one fourteenth
the effect — at p=0.023. Across three arms, two views and six statistics I computed
roughly thirty-six p-values, so one at 0.023 is what multiple comparison predicts. I
have no mechanism to offer beyond that and I am not inventing one. It should be
treated as unexplained rather than explained.

**The dimensionality contrast is the most informative thing the nulls produced and
it was not preregistered.** Participation ratio: real 25.7, permuted 23.9, shuffled
99.4. The permuted arm trains on real coherent preference pairs under the wrong
labels and produces the SAME low-dimensional geometry as the real run while scoring
zero against the labels. The shuffled arm destroys the preference direction and the
geometry goes essentially full-rank. So **coherent preference training creates the
low-dimensional structure; trait identity determines where in it each trait lands.**
Those are two separable findings and only the second needed the labels.

## The seed-paired floor: FAIL, exactly as preregistered

Preregistered bar: across-seed self-cosine must exceed **+0.2447**, the median cosine
of same-factor same-keyed *different*-trait pairs. Abandon line at 0.35.

Measured: **median +0.0167, min +0.0151, max +0.0178** over 40 traits.

A trait trained twice is about **fourteen times less aligned with itself** than two
different traits sharing a factor and a keying. The preregistration is unambiguous:

> If seed-paired lands below 0.35 I will report the trait-level result as
> unsupported even though it is my own headline.

**So the trait-level framing is withdrawn.** There is no evidence here that an
individual trait has a characteristic direction in weight space.

Note the spread: 40 traits spanning ±0.0014. A trait-dependent quantity would vary
trait to trait; a near-constant does not. That tightness is the signature of a
structural constant of the parameterisation, not of signal — which is itself evidence
that the seed-invariant part of the delta carries no trait information.

The likely mechanism, stated as reasoning and not as a measured fact: LoRA confines
dW to the row space of a randomly initialised A. Two seeds give two different rank-64
subspaces of a 2560-dimensional space, and two random subspaces of that shape are
near-orthogonal. Under one shared initialisation every adapter lives in the same
subspace and *can* align.

## Where I depart from the tooling, and why the departure is testable

`compare_nulls.py` concludes "EVERY EFFECT ABOVE IS VOID... differences of a few
hundredths computed inside a run-to-run wobble at least as large." **That conclusion
is wrong, and I wrote the check that produced it.**

It conflates two different things:

- the **coordinates** failing to correspond across bases, and
- the **organisation** failing to replicate.

A cross-run cosine of 0.017 establishes the first. It cannot speak to the second,
because structure can reproduce in a rotated basis while every corresponding vector
reads as orthogonal. The cross-Gram is the wrong instrument for the question the
verdict line was answering — the same defect as [[a-test-shaped-like-the-wrong-hypothesis]],
now in a check rather than a statistic.

So I tested it instead of arguing it. The 40 seed-1 adapters already existed; a Gram
over them costs about fifteen cents.

## The replication

Same 40 traits, two independent initialisations, the same pre-specified statistic:

```
                          raw diff         p     resid diff         p
  seed 0 (phase-5)         +0.1819   0.00010        +0.1086   0.00010
  seed 1 (independent)     +0.1835   0.00010        +0.1037   0.00010
  ratio                       1.01x                    0.96x
```

Bipolarity likewise: seed-1 same-keyed +0.2290 against opposite-keyed −0.0949, gap
+0.3239 raw and +0.2065 residual, p at the permutation floor; and factor-specific,
with the different-factor gap at −0.0541, absent as before.

**The same traits whose vectors correlate at 0.017 across these two runs reproduce
the factor structure to within 1% and 4%.** The coordinates are noise. The structure
is not.

This test was NOT preregistered and a reader should weigh it accordingly. What
protects it: the statistic was fixed in `decompose.py` before any seed-1 data existed,
the adapters are independent, and it is a replication rather than a reanalysis. The
matched 40-trait control rules out trait count as the explanation.

## What now stands

**Upheld.** Trait-training deltas organise into five bipolar axes aligned with the
Big Five: same-keyed traits within a factor align, opposite-keyed traits anti-align,
and the effect is factor-specific rather than one global valence direction. Not a
data artefact, not a batch effect, not training strength, not reproduced by either
null, and replicated under an independent initialisation.

**Withdrawn.** Any claim that a trait has its own direction in weight space, or that
these adapters could be used as a per-trait steering basis transferable across runs.

**Newly required by the above.** The geometry is meaningful only *relative to a fixed
initialisation*. Anything downstream that compares adapters must hold the seed fixed
or be re-derived per seed. Phase 7 steering, as planned, assumed transferable trait
directions and needs re-specifying before it is worth running.

**Still true and unchanged.** Participation ratio 25.7: five factors are a real but
thin slice of a much higher-dimensional geometry. That was never a five-dimensional
claim and still is not.

## Cost

Phase 3 came to about $111 of training against a $99 plan, plus roughly $0.60 of
CPU-only Gram and cross-Gram analysis, plus $0.54 of waste from the two arms I killed
on 08-20. The seed-1 replication that changed the verdict cost about fifteen cents.

---

# ADDENDUM 2026-08-22 22:40 UTC — THE TRAIT-LEVEL WITHDRAWAL IS REVERSED

Prompted by Samuel asking whether to rerun with fixed initialisation. It is the
right question and the answer is no, because the sweep is already fixed-init
(all 134 adapters at `seed=0`, `order_seed=0`) — which sent me to look at what
the seed floor had actually measured.

## The preregistered bar was in the wrong units

The floor compared a **cross-basis** cosine (0.0167) against a bar computed
**within a single basis** (+0.2447, the median same-factor same-keyed
different-trait pair in the seed-0 Gram). Those two numbers are not commensurable.
Two independent rank-64 inits span near-orthogonal slices of a 2560-dim space, so
a trait direction that reproduced *perfectly* modulo the subspace would still have
scored ~0.02. **The test could not have passed under any hypothesis.**

This is `a-test-shaped-like-the-wrong-hypothesis` for the third time, now in the
BAR rather than the statistic or the check: I never asked what the measurement
would read if the thing it was denying were fully true.

## The measurement that settles it

`results/cross_gram_full_root_x_data_null_seedpaired_s40.npz` — 134 seed-0
adapters x 40 seed-1 adapters, cosines as `X / outer(norms_a, norms_b)`:

    same trait                     +0.01659   (n=40,   sd 0.00070)
    same factor, same keying       +0.00595   (n=358)
    same factor, opposite keying   -0.00271   (n=386)
    different factor               +0.00132   (n=4336)

Three independent structural facts, none explicable by a scale artefact:

1. **Perfect separation.** min same-trait 0.01509 > max different-trait 0.01490,
   across all 5320 cross-trait comparisons. No overlap at all.
2. **40/40 top-1.** Every seed-1 trait is its own nearest neighbour among all 134
   seed-0 adapters. Chance is 1/134; mean rank 1.0 against a chance mean of 67.5.
3. **The whole geometry transfers, uniformly attenuated.** Cross-seed cosine
   regressed on within-run cosine over the same 5320 pairs: Pearson **+0.9947**,
   Spearman **+0.9959**, slope **0.0264**, intercept -0.0004. The cross-seed block
   IS the within-run block times 1/38.

Bipolarity survives the seed boundary: same-keying minus opposite-keying
= **+0.00866** signed, i.e. the +0.1216 within-run effect at the same 1/38 scale.

## What is reinstated, and what is not

**Reinstated.** A trait has a reproducible direction in weight space. It survives
independent initialisation; the cosine is small only because the LoRA
parameterisation confines each run to a different random rank-64 subspace, and
0.0264 is the overlap of two such subspaces, not a measure of signal loss.

**Not established.** Same-trait pairs train on a byte-identical corpus, so what
reproduces is that trait's *training signal*. This addendum does not upgrade the
deflationary reading — it removes a spurious refutation, it does not add evidence
that the axes are model-intrinsic personality rather than corpus structure.

**Not usable as engineering.** 1.7% alignment is statistically unambiguous and
practically worthless as a transferable vector. Anything that needs a usable
direction still works inside one init, or freezes A globally (LoRA-FA,
arXiv:2308.03303) so all adapters share one subspace by construction. That is the
right substrate for the text-to-LoRA arm.

## Correction to this document's own downstream flag

The earlier text said phase 7 steering "assumed transferable per-trait steering
directions" and must not be run as written. **That was wrong on two counts.** The
directions do transfer. And phase 7 never needed them to: it applies a concrete
weight delta, found in the seed-0 basis, to the frozen base model, with matched
random-direction controls at every dose. That operation is well defined regardless
of how the direction was found. Phase 7 is runnable as specified.

## The general lesson, again and one level up

Preregistration protected me from negotiating away an unwelcome number. It did
**not** protect me from writing a bar the effect could not clear. A preregistered
threshold needs its own check: *compute what the statistic would read under the
maximal version of the hypothesis, before committing to the threshold.* If that
value is below your bar, the bar is not strict — it is broken.

Note also what made this recoverable: the reversal cost one Gram already on disk
and about a minute of numpy. The same rule as last time — when refuting your own
instrument is cheap, refute rather than reason.

## Postscript: the attenuation constant is predicted, not fitted

Measured slope of cross-seed cosine on within-run cosine: **0.0264**. The LoRA rank
fraction is r/d = 64/2560 = **0.0250** (Qwen3.5-4B `hidden_size` 2560, `LORA_R` 64).
Agreement to 6%.

The subspace-overlap account predicts that number; no account in which the 0.0167 is
absence-of-signal predicts anything at all. This is the strongest single piece of
evidence in the addendum, and it was free.

## Addendum, 2026-08-24 — objective mismatch in every control arm

Found by comparing against the now-public Persona Cartography reference code
(vendor/persona-cartography), whose pipeline hard-codes the auxiliary terms into
every run and cannot launch a control without them. Ours can, and did: the null
launcher carried the corrected LoRA scale but not the loss configuration, so all
240 control adapters (shuffled, permuted, seed-paired) trained under plain
sigmoid DPO — loss_type ["sigmoid"], kl_coef 0.0 — while all 134 sweep adapters
trained with loss_type ["sigmoid","sft"], kl_coef 0.001. Verified in every arm's
runmeta. This is the third occurrence of this project's signature failure mode
(a treatment silently not travelling to the container), and it recurred on the
parameter class the existing gates did not check.

Consequences, stated against each claim:
- The cross-seed self-cosine (+0.0167) is a seed-plus-objective floor, not a
  pure seed floor. The trait-level withdrawal STANDS (the preregistration bound
  it to this measurement as run), but the mechanistic reading "the init frame
  alone carries the magnitude" needs the 40 seed-B adapters retrained at the
  matched objective (~$17 at measured rates) before it is clean.
- The shuffled/permuted null table compares arms that differ from the signal
  arm in objective as well as treatment. Directionally this is conservative --
  the nulls were denied an auxiliary term that adds shared structure, yet still
  needed to show factor structure to void the result and did not -- but the
  clean statement costs ~$82 of retraining if wanted.
- The seed-B factor replication and the 40/40 twin identification are untouched
  as findings about the adapters that exist; their attribution to "seed alone"
  inherits the same caveat.

Tooling: cross_gram_on_modal.py now refuses matched pairs whose runmeta
disagree on loss_type/loss_weights/kl_coef, the same way it already refused
data_sha256 mismatches.

## Addendum, 2026-09-03 — the seed floor, matched objective

The 40 seed-paired traits were retrained at the zoo's exact objective
(`sigmoid,sft` / `1.0,0.1` / `kl_coef 0.001` / rslora off) into
`data_null_seedpaired_s40_matched`, same seed 1, byte-identical corpora. The
treatment travelled: first-step loss 0.89-0.94 (ln 2 plus the 0.1 x SFT term, as
in the 134) against exactly 0.6931 in the original arm, and `kl_term` =
`sq_approx_kl` x 0.001 on every logged step. `analyse_crossseed.py` computes
both arms with one code path (it reproduces the 2026-08-22 numbers to the digit).

    134 x 40                       original (objective mismatch)   matched
    same trait                     +0.01659  sd 0.00070            +0.01806  sd 0.00105
    same factor, same keying       +0.00593                        +0.00647
    same factor, opposite keying   -0.00271                        -0.00213
    different factor               +0.00133                        +0.00183
    min same / max cross           0.01509 / 0.01490               0.01593 / 0.01584
    top-1 identification           40/40                           40/40
    attenuation slope (r/d 0.0250) 0.0264  Pearson 0.9947          0.0265  Pearson 0.9966

The objective effect in isolation (`cross_gram_full_data_null_seedpaired_s40_x_
data_null_seedpaired_s40_matched.npz`, same seed, same data, only the objective
differs): same-trait cosine **+0.954** (range 0.908-0.981), different-trait
+0.043, 40/40 top-1. So the seed change moves an adapter from 0.954 to 0.018 and
the objective change moves it from 1.0 to 0.954. The 2026-08-24 caveat on
attributing the floor to "seed alone" is discharged; the page now quotes 0.0181.
The shuffled/permuted null arms are still at the mismatched objective (~$82).

Same day, the four alignment adapters were retrained on the zoo's shared pool
(`data_alignment_common`, 444 of the 445 prompts; the first run had 497, 53 of
them unseen by any zoo adapter). LoRA-A drift 0.0143-0.0151 against the zoo's
0.0146. Every angle moved by under 2.5 degrees and every cosine by under 0.06:
sycophantic 61.6 -> 62.8 to `pleasant`, obsequious 60.2 -> 60.0, power-seeking
72.6 -> 72.7 (nearest `selfish` -> `crooked`, one degree apart either way),
corrigible 68.4 -> 68.0, synonym pair 59.5 -> 61.9. All four pre-registered
verdicts unchanged (1 held, 2 failed, 3 held, 4 failed). The corpus confound
was real and harmless; `analysis/alignment_geometry_aligncommon.json`.

## Addendum, 2026-09-05 — the null arms at the matched objective, and the N x N

Shuffled and permuted retrained (100 traits each) at the zoo's objective into
`data_null_{shuffled,permuted}_p100_matched`; Grams and decompositions redone
with the same labels and a per-arm runmeta path (TEST 6 unavailable, as before).

    TEST 1B residual diff       original   matched
    shuffled                    +0.0001    +0.0001   (p 0.46)   PASS
    permuted                    +0.0075    +0.0080   (p 0.07)   PASS, same TEST 2 flag
    scree: real PCs above shuffled   11        11
           real PCs above permuted    0         0

The 2026-08-24 caveat on the null table is discharged: the objective mismatch
changed nothing to three decimal places. `analysis/scree_null_matched.json`
now feeds the page; the caption's hedge is gone.

N x N (`analyse_nxn.py`, `analysis/nxn_summary.json`): 134 traits x 40 pairs from
data_common, each scored against all 134 adapters as single directions.
Own adapter rank 1 for **134/134** raw, 133/134 after column z-scoring (bashful
second). Runners-up share factor+keying 42% (base rate 12%), opposite keying 0%.
The scoring identity sees the Big Five in DATA, which nothing about the
positive control required.

Hole words (`analyse_hole.py`): cavalier 81.5 deg, blase 70.3, insouciant 39.6
from u in the k=5 chart (existing nearest 52.5); full space 84.8 / 79.6 / 71.6
against 68.9; pairwise 77-89. Chance of one of three within 39.6: 20%.
Suggestive in-plane, unconfirmed.

## Addendum, 2026-09-05 — stage 2 at a second seed

Full OCT stage 2 (reflection + interaction generation, SFT, 0.25 merge) re-run for 15
traits (3 per factor, mixed keying) on the matched seed-1 stage-1 adapters, SFT seed 1,
outputs under /oct/seed1. Nothing skipped; 313-375 optimizer steps; $226 planned,
came in under. Cross-Gram of the 15 seed-1 SFT LoRAs against the 134 seed-0 SFT LoRAs
(`analyse_crossseed.py`, PC_WITHIN = the seed-0 stage-2 134x134 block):

    same trait            +0.0672  (sd 0.0163)      stage 1 was +0.0181
    same factor, same key +0.0195
    different factor      +0.0141                   stage 1 was +0.0018
    top-1 of 134          15/15                     mean rank 1.00
    attenuation slope      0.116   Pearson 0.931    stage 1: 0.0265 vs r/d 0.025

The slope is 4.6x the r/d prediction. It is not the A subspaces: seed-0 and seed-1
stage-2 A row spaces overlap at 0.022 (random), and stage-2 A drifts 10% from init
(stage 1: 1.5%). The extra cross-seed signal sits in a component every stage-2 adapter
shares (different-factor floor 0.014 vs 0.002), consistent with B loading onto the
shared part of A's drift. Not resolved; noted.

Geometry: the seed-1 15x15 cosine block vs the seed-0 block for the same traits
correlates at r = 0.978 raw, 0.987 trait-centred. Stage-2 arrangement replicates
across seeds essentially perfectly on this sample.

## Addendum, 2026-09-07 — the geometry on the full OCT persona adapters

Samuel asked whether the page's stage-one geometry holds for the deployed artefact,
the persona adapter dW = dW_dpo + 0.25 dW_sft (exact rank-128 concatenation,
`personas_exact`). `analyse_fulloct.py` → `analysis/fulloct_geometry.json`.

- Each persona's squared norm is 0.500 stage one, 0.500 (0.25 × stage two); the
  cross term is 0.0002 of the norm. The raw stage-two delta is 4.0× the stage-one
  delta, so the 0.25 weight makes the halves equal.
- Persona vs stage-one off-diagonal cosines: r 0.9915 (0.9944 double-centred).
  Persona vs stage two 0.8308; stage one vs stage two over all 134: 0.7545.
  Procrustes R² of the k=5 scores, persona from stage one 0.9945; principal angles
  1.6, 1.9, 3.6, 5.4, 8.2°. Nearest neighbour unchanged for 103 of 134.
- Why: stage-two off-diagonal cosines have sd 0.036 around a shared 0.145;
  stage one has sd 0.163. With equal norms the persona cosine is the mean of the
  two, so the arrangement is stage one's.
- Second seed (15 exact seed-1 personas, `build_personas_seed1.py`): same-trait
  +0.0428, different-factor +0.0081, top-1 15/15, separation −0.0030, slope 0.0389
  (r/d for rank 128 is 0.05), r 0.939.
- Stage-one adapter vs its own persona: cosine 0.7069, predicted 0.7068 from the
  norms; 134/134 top-1.
- Found on the way: `fix_persona_merge.py` doubled the `base_model.model.` key
  prefix, on the volume and in the public Hub folder `persona_exact/`; tested with
  PEFT 0.20, such an adapter loads with a missing-keys warning and changes the
  logits by exactly 0.0. `fix_persona_keys.py` repaired the 149 volume copies;
  the Hub re-upload is Samuel's call. Wiki: `wiki/pages/geometry/full-oct-replication.md`.

## Addendum, 2026-09-08 — stage two, explored

Samuel asked for exploration of the introspection stage on the grounds that stage
one is obvious and stage two is not. Four experiments, plus two corrections found
on the way. Wiki: `wiki/pages/geometry/stage-two-exploration.md`. Index of the
outputs: `analysis/stage2_exploration.json`.

**Correction A — the stage-two adapters share a LoRA-A frame.** `oct_stage2.
train_sft` seeds torch with 123456 and then builds the PEFT model, so every trait
draws the same LoRA-A. Read off the volume for `model.layers.0.linear_attn.
in_proj_a` (`check_lora_a_identity.py` → `analysis/lora_a_identity.json`):
within seed 0 the stage-two A factors have pairwise cosine 0.9977–0.9981 after
training (stage one 0.99997), across seeds 0.0021, and stage two against stage
one 0.0032. `wiki/pages/geometry/stage-two-structure.md` said "each with its own
random LoRA-A"; corrected there and in `superseded-claims.md`. This section of
the verdict already knew half of it ("stage-2 A drifts 10% from init").

It does **not** make the shared component an artefact. Measured inside each
seed's own frame on the same 15 traits (`analyse_stage2_frame.py` →
`analysis/stage2_frame.json`): share of squared norm on the mean direction 0.2017
at seed 0 and 0.2000 at seed 1; cosine to the mean 0.4497 ± 0.0319 and 0.4475 ±
0.0330; off-diagonal cosine +0.1453 and +0.1432. The shared direction is a
property of the recipe; the frame fixes only its coordinates. That also closes
the loose end this document left under the stage-two attenuation slope.

**Correction B — alpha is half an adapter.** `ref` = 0.8078003190997738 in every
steer spec came from `sketch_adapters.sketch_one`, which computes `||B@A||_F` and
omits the LoRA scaling 2.0. The real mean stage-one delta norm is
1.6157416444226869, so ref/true = 0.49995636486079753 and every published alpha
is half an adapter's worth of weight change, not one
(`analyse_steer_alpha_units.py` → `analysis/steer_alpha_units.json`). No result
changes. A persona's dose of the stage-two shared direction is 0.6291299271662 of
Frobenius norm, 0.3894 of an adapter, alpha 0.7788 in spec units — not the 0.4
stated on `wiki/pages/behaviour/stage-two-shared-direction.md`, which is the
adapter-unit figure.

**1. Register versus residual.** 15 second-seed traits, 24 Big Five probes, four
conditions built the same way (fp32 delta into bf16 base weights; `steer_fix.py`
gained an `add_adapter` job field and a `persona_exact` source), 1,464
generations, judged blind (`judged_s2register.json`, 1,458 judged, 1 failed call;
repeat r 0.766–0.891). Own-factor amplification: base —, stage one +22.33,
stage one + the shared direction at alpha 0.389 (half a persona's dose) +22.40,
at alpha 1.0 (1.28× the dose) +24.52, exact persona +30.25. Paired against stage
one: +0.07 ± 1.89, +2.19 ± 2.09, +7.92 ± 2.80 (t 2.83, 14/15 positive). Shares of
the stage-one→persona gap: 0.9% and 27.6%. Text: in-character fraction 0.00 base,
0.48 stage one, 0.50, 0.54, 0.60 persona; markdown 0.92, 0.54, 0.52, 0.50, 0.37.
**The amplification is the stage-two residual, not the register** — consistent
with the norms, 0.629 of the persona's stage-two Frobenius norm on the shared
direction against 1.489 on the residual. Mechanism check: this run's stage-one
condition reproduces `judged_100.json` stage1 at +22.33 vs +22.26, r 0.937.
Output `analysis/stage2_register_vs_residual.json`.

**3. Activation space.** `act_space.py` gained a source switch; the 134 stage-two
LoRAs and the 134 exact personas were run over the same 64 prompts with no system
prompt (`analyse_actspace_stage2.py` → `analysis/actspace_stage2_geometry.json`,
layer 16, response window). Cosine to the mean shift: stage one 0.701 ± 0.079
(share 0.479), stage two 0.842 ± 0.070 (0.669), persona 0.705 ± 0.066 (0.486).
Arm against arm, same trait: persona×stage one +0.988 (mean shifts +0.993,
arrangements +0.993, 134/134 top-1); persona×stage two +0.584; **stage one×stage
two +0.530, with the two mean shifts at +0.552** — where in weight space the same
quantities are +0.0002 and +0.000. Orthogonality in weight coordinates is a fact
about the parameterisation, not about what the stages do. Own-constitution
prompted vector is nearest of 134 for 43 (stage one), 50 (stage two), 55
(persona). Caveat: a stage-two LoRA alone on the plain base is off its training
distribution.

**4. Factors.** `build_gram_stage2_noshared.py` writes `results/gram_stage2_
noshared.npz` (G − (G1)(1ᵀG)/(1ᵀG1); 0.1528 of the trace removed, off-diagonal
cosine +0.1452 → −0.0075, before/after Pearson 0.8791) and `analyse_fa_qwen35.py`
was rerun on it (`results/fa_qwen35_stage2_noshared.json`). Removing the shared
direction changes nothing: loading matrices match at Tucker 0.990–0.99996,
parallel analysis still retains 7 at N=1528, top reduced eigenvalues 3.97, 2.98,
2.46, 2.24, 2.05 → 3.97, 2.98, 2.46, 2.20, 1.99. (The uncentred *reduced* branch
degenerates on the rank-133 Gram — 134 at every N ≥ 1000; the centred branch, the
one used, is unaffected.) k=7 takes only six distinct Big Five targets, one of
them Eval, claims Conscientiousness twice, mean |congruence| 0.454 against k=5's
0.492, complexity 4.23 against 3.21, and its extra factors decompose k5's
Emotional Stability. **Decision: present k = 5 on the ordinary stage-two Gram**,
reporting the retained 7 and noting that k=7 separates a clean orderliness factor
out of the muddled k=5 Conscientiousness factor. Output
`analysis/stage2_factors_choice.json`.

**2. Trait-free stage two.** `oct_stage2.py` gained a `--neutral` mode (trait-free
constitution in `constitutions_neutral.json`, no stage-one adapter, generation
straight off the base snapshot, an engine-level `--gen-seed` because vLLM's
default seed is fixed and five runs of one model would otherwise be identical)
and five replicates s1–s5 were launched at `sft_seed` 123456, so the LoRA-A frame
matches the zoo's and the cosines are comparable. The five corpora are genuinely
independent (0 of 1000 identical reflection rows between any pair, 1000 unique
responses each); all five trained to 372 steps on 11,905–11,926 of 12,000 rows;
and the frame matched, cos(neutral LoRA-A, zoo stage-two LoRA-A) 0.9982.

Result (`analyse_stage2_neutral.py` → `analysis/stage2_neutral_control.json`),
neutral against the 134:

    |dW|                          6.4976 ± 0.0082   vs  6.4643 ± 0.2899
    cos with the stage-2 grand mean 0.3035 ± 0.0007  vs  0.3893 ± 0.0311
                                                         (min 0.2775, max 0.4438)
    cos with each other            +0.5633            vs  +0.1452 ± 0.0359
    cos with the 134 (670 pairs)   +0.1179 ± 0.0225   vs  +0.1452 ± 0.0359
    the same, shared direction out −0.0004 ± 0.0249   vs  −0.0075 ± 0.0370

**Mostly generic.** A constitution saying the character has no character still
produces an adapter at 78% of the zoo's cosine to the shared direction and 81% of
its mutual similarity: what stage two installs is mostly "narrate yourself in the
first person". Not entirely — 0.304 is below the zoo's mean − 2 sd (0.327) and the
five neutral runs agree to within 0.002, so trait conditioning adds about a fifth
of the shared component. Two side results: the five neutral adapters sit at
+0.5633 with each other (+0.5189 after removing the shared direction), i.e. the
recipe is near-deterministic given its inputs; and every one's nearest zoo trait
is `unemotional` (+0.1666 to +0.1684), with residual cosine profiles correlating
with `unemotional`'s own at r 0.858–0.862 — the map placed a trait-free
constitution on the trait word for "neither warm nor cold, under pressure nothing
changes" without being told.

Three limits. Two of the five SFTs resumed after a container restart (2 `[resume]
from` lines; s1 and s3 show `train_seconds` 5,579 and 7,132 against 16,376–23,073
and `loss_last` 0.234 against 0.747) — all five still reached 372 steps on the
full row set, but their `loss_last` is the post-resume tail only, the defect
`wiki/pages/zoo/runmeta-provenance.md` records. The neutral runs train on the
plain base while the 134 trained on 134 different DPO-merged bases (the argument that this matters little: the shared
direction is already at 0.389 across all 134 differently perturbed bases). And
this run's `runmeta.json` carries `"base": "merged(... + stage1 s1)"`, which is
false — the string is hardcoded in `train_sft`; fixed in source after launch, and
the run is identified by `oct_root` `/oct/neutral` and its generation key.
