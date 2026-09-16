# Phase 3 null controls: what each arm destroys, and what each outcome means

Written 2026-08-20 12:31, WHILE THE ARMS ARE STILL TRAINING and before any null
number exists. That timing is the point. Given three null results and a headline I
have already reported to Samuel, I can narrate almost any outcome into support — so
the readings are fixed here first, including the ones that would cost me the claim.

The headline this must survive: **five bipolar axes**, polarity-signed within-factor
minus between-factor separation **+0.1216** (residual view, 0.88 sd, p at the
20,000-shuffle permutation floor), bipolarity gap **+0.2393**, label-free separation
**+0.0433** with clustering ARI **0.0989**.

## The loss matters, and it means one arm nulls less than its name suggests

Every run trains `loss_type = ["sigmoid", "sft"]` at weights `[1.0, 0.1]` — DPO plus
an NLL term on the CHOSEN completion. That second term is what stopped the collapse
to a trivial discriminator, and it changes what "shuffled" means.

## Arm 1 — SHUFFLED (100 traits): nulls DIRECTION, not CONTENT

Chosen/rejected swapped on 49.89% of rows. Documents, lengths, style and token
statistics identical.

- **Destroyed:** the preference direction. The DPO term's pull toward
  trait-expressing text is largely cancelled.
- **RETAINED, and I want this on the record before the numbers:** trait-specific
  text content, through the NLL term. After shuffling, that term trains on a ~50/50
  mixture of the trait's own chosen and rejected completions — but both were
  generated *for that trait*, on the shared prompt pool. Some trait-specific content
  survives at weight 0.1.

So this arm is **not** a total null. Readings fixed now:

| outcome | meaning |
|---|---|
| signed separation ≈ 0 AND bipolarity ≈ 0 | strongest support: the effect requires the preference direction |
| bipolarity survives near +0.24 | **alarming.** Polarity structure without consistent preference would mean the axis tracks text topic, and my central claim is about content, not preference |
| signed separation reproduces ≥ ~+0.12 | **the headline is void.** Stop condition met |
| bipolarity collapses, weak factor clustering remains | EXPECTED AND FINE — that residue is the NLL content term. To be reported, not buried. Anything up to roughly half the real +0.0433 label-free separation is consistent with the retained content |

## Arm 2 — PERMUTED (100 traits): nulls the LABEL, keeps everything else

Each adapter trains on a different trait's real pairs, by a derangement with zero
fixed points.

- **Destroyed:** the trait-to-data correspondence only.
- **Retained:** real preference direction, real coherent content, identical dynamics.

**This arm will produce real geometric structure and that is not a failure.** It must:
it is trained on real preference data. The question is only whether the structure
lines up with the *nominal* label. Scored against the names the adapters carry, the
signed separation should sit near **0**.

- near 0 → the geometry tracks the trait, not the act of training. Supports the claim.
- reproduces ≥ ~+0.12 → the labels are doing no work; something is wrong upstream and
  the whole result is suspect.

A subtlety I am recording so I cannot discover it conveniently later: the derangement
ran over all 134 traits, so some of these 100 primary adapters hold *secondary*-trait
data. That weakens nothing — it breaks the correspondence either way — but it means
this arm's internal geometry is not a clean re-run of the primary-set geometry.

## Arm 3 — SEED-PAIRED (40 traits): nulls NOTHING; it is the floor

Byte-identical data, `--seed 1 --order-seed 1`, against the phase-5 adapter for the
same trait. Treatment verified as arrived: the containers print
`[seed] seed=1 order_seed=1 (defaults SEED=0 ORDER_SEED=0)`.

Statistic: cos(dW_trait^seed0, dW_trait^seed1) per trait, in the **raw** view — there
is no common-component projection defined across two adapter sets, and residualising
one side only would flatter the floor.

**Pre-registered threshold.** The reference is the median cosine of same-factor,
same-keying, *different*-trait pairs: **+0.2447**. That is the most similar class of
genuinely distinct traits, so it is the bar that matters.

- across-seed self-cosine **clearly above +0.2447** → trait identity is reproducible
  above the factor level; the geometry is trait signal, and everything stands.
- across-seed self-cosine **at or below +0.2447** → **the same trait trained twice is
  no more alike than two different traits sharing a factor and a keying.** Trait
  identity is then not recoverable, the per-trait Gram is dominated by seed noise, and
  the axis result survives only as a statement about factors, not traits. I would have
  to withdraw the trait-level framing.
- self-cosine near 1.0 → suspect the seed change did not take. The cross-Gram treats
  equal recorded seeds across a pair as fatal for exactly this reason.

## What would make me abandon the claim outright

Either real null arm reproducing the signed factor separation at or above ~+0.12
residual with p < 0.05. That is the stop condition already written into plan.json,
and it is not negotiable after the fact.

## What I expect, so the expectation is falsifiable

Shuffled ≈ 0 on signed separation, ≈ 0 on bipolarity, possibly small positive
label-free separation from the retained NLL content. Permuted ≈ 0 on all three.
Seed-paired self-cosine in the 0.5–0.8 range — clearly above the +0.2447 bar but well
below 1.0, since LoRA init and data order genuinely differ.

If seed-paired lands below 0.35 I will report the trait-level result as unsupported
even though it is my own headline, and even though the factor-level result would
survive.
