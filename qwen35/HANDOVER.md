# persona-curvature / qwen35 — handover

Rewritten 2026-08-23 by cartographer, replacing the 08-20 pre-blackout version. The
experiment has ANSWERED ITS QUESTION; what remains is optional. Read
`PHASE3_VERDICT.md` first — it is the result. This file is the state around it.

## The result in three lines

Trait-training weight deltas organise into **five bipolar axes aligned with the Big
Five**: same-keyed traits within a factor align (+0.24), opposite-keyed anti-align
(−0.08), and it is factor-specific rather than one global valence direction. Signed
factor separation **+0.1216** residual, p at the 20,000-shuffle permutation floor.
Survives both nulls and **replicates under an independent seed at ratio 1.01**.

**Individual trait directions DO survive a seed change** — see the 22:40 addendum at
the end of `PHASE3_VERDICT.md`, which reverses the withdrawal. Cross-seed same-trait
cosine is +0.0181 at the matched objective (+0.0166 in the 2026-08-22 arm, which
trained under plain sigmoid DPO -- see the 2026-09-03 addendum) against a
different-trait floor of +0.0018; 40/40 traits are their
own nearest neighbour among 134; and the cross-seed block is the within-run block
times 1/38 (Pearson +0.9947). The preregistered bar of +0.2447 was a *within*-basis
quantity applied to a *cross*-basis measurement and could not have been met under any
hypothesis. Participation ratio 25.7 — five factors are a thin slice of a
~26-dimensional space, never a five-dimensional claim.

*(Superseded, kept so the reversal is legible: this file previously read "individual
trait directions do not survive a seed change (cosine 0.0167) and the trait-level
claim is withdrawn. The geometry is meaningful only relative to a fixed
initialisation." The 0.0167 is right; the inference from it was not.)*

## Money

**SUPERSEDED 2026-08-23 00:05 — read this first.** The `$195.90 actual` below is the
RECORDED figure and it under-reports by ~$175. Measured realised is **~$370.65 of $1000**
(Modal ~$311 + OpenRouter $61.82); remaining is **~$630, not ~$804**. Three mechanisms —
billed-is-not-incurred, an app-name collision that put the main sweep inside phase 2, and
a phase costing its own discarded attempts — are set out in `plan.json`'s
`ledger_defect_note`. Phases 2 and 5 now carry `actual: null` (UNSPLITTABLE, not zero);
phase 3 is corrected to $206.885; phase 7's realised is $66.196 against a $35 nominal.
Run `check_plan.py` rather than reading any figure out of prose — it refuses to print the
bad sum. This directory is a git repo as of tonight, so every ledger value from here has a
date and a diff.


RECORDED ~$228.89 over 5 of 7 closed phases (2 carry `actual: null` = NOT MEASURED);
MEASURED ~$370.65. Do not quote either from here — run `check_plan.py`, which prints the
null phases and the FLOOR framing before any sum. Open and OPTIONAL: 4 ablations $98,
7 steering $35, 8 qualitative $25, 10 full OCT $330, 11 writeup $5.

(The bare "$195.90 actual against a $1000 ceiling" that used to be this line is gone
rather than annotated. It sat directly under a banner saying it was wrong by $175, which
is exactly the arrangement that fails: a caveat does not travel with a number, only the
number's own text does. Same reason `check_plan.py` refuses to print the sum.)

**Phase 7 is runnable as specified** (corrected 2026-08-22 22:40; this file
previously said it must not be run, on the strength of the now-reversed withdrawal).
It applies a concrete weight delta found in the seed-0 basis to the frozen base
model, with matched random-direction controls at every dose — an operation that never
depended on cross-run transferability. Phase 7 + phase 8 ($60) is the highest-value
remaining spend: it is the only thing that tests whether the recovered axes do
anything behavioural, including Samuel's prediction that steering negatively on the
largest component yields emergent misalignment.

Cost model, from janitor and worth keeping: **$10.17 per app + $0.333 per run**, not
a flat per-run rate. Verified out-of-sample to 0.8% on the 134-run sweep. A flat rate
is only ever correct at one job size.

## Where things are

```
train_qwen35.py      the trainer. Everything travels in the JOB DICT, never env vars.
decompose.py         the analysis. TESTS 1,1B,1C,2,2B,3,4,5,6 + verdict.
gram_on_modal.py     Gram on the volume. PC_ADAPTER_SUBDIR picks the arm.
cross_gram_on_modal.py   cross-seed self-cosine. See the caveat below.
compare_nulls.py     the arm-vs-arm table.
make_nulls.py        builds the three null corpora, with invariants asserted.
PREREGISTRATION_phase3.md   written before the nulls landed. It bound, and it cost the headline.
PHASE3_VERDICT.md    the verdict against it.
results/             gram_sweep.npz, gram_data_null_seedpaired_s40.npz,
                     decomposition{,_shuffled,_permuted,_seed1}.json,
                     cross_gram_seedpaired.npz, runmeta_sweep.json
```

Modal volume `pc-qwen35-sweep`: 134 real adapters at the ROOT, null arms namespaced
under `/adapters/<corpus_label>/<trait>`. Do not un-namespace this.

## Working config, verified from container records not launch commands

r=64, alpha=128, **plain LoRA**, effective scale 2.0 (OCT's), beta 0.1, kl_coef 0.001,
`loss_type=["sigmoid","sft"]` weights `[1.0,0.1]`, 248 targeted modules, seed 0.
Base is `Qwen/Qwen3.5-4B`, a VISION-LANGUAGE model — target the text tower only.

The NLL-on-chosen term at 0.1 is what prevents DPO collapsing to a trivial
discriminator. That came from reading OCT's repository, not the paper. Do not drop it.

## Traps, each of which cost real time

1. **A decision must live in the DEFAULT, not in a launch flag.** `PC_USE_RSLORA`
   defaulted to rsLoRA while the sweep launched with it off — so a new launcher
   trained 240 adapters at scale 16.0 and the whole phase was rerun. Per-launch
   overrides are for things that legitimately vary (budget, corpus, seed). Fixed;
   the plan banner now prints EFFECTIVE SCALE and flags anything that is not 2.0.
2. **Namespace every write, not just the obvious one.** Adapter output was namespaced
   and the data upload was not, so three concurrent arms raced on one directory. The
   content-sha guard caught it; the prompt-pool guard could not, because the nulls
   hold prompts fixed by construction. Two guards whose names both sound like "is
   this the right data" and only one measured it.
3. **A check that claims to run and does not.** `EXPECTED_POOL_SHA` was commented
   "Asserted, not trusted" and referenced nowhere. Grep before believing a comment.
4. **Watchers must distinguish finished, broken and stuck.** Mine reported only clean
   exits and stayed silent for 2.5 hours while two arms were dead.
5. **A pattern selects a class, not your example.** `pkill -f launch_nulls.sh` matched
   its own shell; a `targeted=248` progress count matched the START line, not the end.
6. **Metric lines now carry `trait=`.** They did not, and four concurrent containers
   made per-trait attribution impossible from the log. `runmeta.json` had it all along
   — I had read the wrong artefact and generalised "not in this log" to "not anywhere".

## The methodological finding, which may outlast the result

A cross-run correspondence check **cannot** tell you whether structure replicates —
structure can reproduce in a rotated basis while every matching vector reads as
orthogonal. My own tool concluded "every effect is void" from a 0.0167 self-cosine,
and the effects replicated at ratio 1.01. Written up in
`../../memory/notes/coordinates-are-not-structure.md`, alongside
`a-test-shaped-like-the-wrong-hypothesis.md` from the same phase.

`cross_gram_on_modal.py` is correct for what it measures and was used to answer a
question it cannot address. Its verdict text has been corrected to say so.

## If someone picks this up

The honest next experiment is **not** more adapters. It is the one the dimensionality
contrast implies: real PR 25.7, permuted 23.9, shuffled 99.4 says coherent preference
training creates the low-dimensional structure and trait identity only decides where
in it each trait lands. Those are two separable phenomena and only the second needed
the labels. Nobody has yet asked what the other ~21 dimensions are.
