---
title: Where the recipe diverges from the two source papers
summary: paper_notes.md section 3 lists eleven divergences from Open Character Training and Persona Cartography; most were closed before or during the build, three were deliberate improvements, and two remain open.
status: current
sources:
  - qwen35/paper_notes.md
  - qwen35/plan.json
  - qwen35/train_qwen35.py
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/oct_stage2.py
  - qwen35/gen_pairs.py
  - qwen35/anchor_constitutions.py
last_verified: 2026-09-16
tags: [zoo, method, literature]
---

# Where the recipe diverges from the two source papers

`qwen35/paper_notes.md` section 3, "Where our plan diverges", was written on
2026-08-19 against `plan.json`'s defaults, before the zoo was trained. This page
summarises each item and says whether it was later resolved. The papers are
[[open-character-training-paper]] and [[persona-cartography-paper]].

## 3.1 Matches, no action needed

Rank 64, alpha 128, `all-linear` targeting, LR 5e-5, beta 0.1, NLL 0.1, KL 0.001,
1 epoch, effective batch 32, `max_len` 1024, AdamW: "every one of these is the
OCT distillation config exactly", some from the paper and KL/max_len/batch from
`finetuning/distillation/llama.sh`. The note flags 10% warmup and grad-clip 1.0
as easy to drop when porting off OpenRLHF. Both are in the recorded config
(`warmup_steps` 1 at 13 steps, `max_grad_norm` 1.0). **Held.**

## 3.2 rsLoRA — RESOLVED

The note calls it "the largest silent one": neither paper uses rsLoRA, so their
scaling is alpha/r = 2.0 while the plan's was alpha/sqrt(r) = 16.0, "an 8x larger
effective update per unit of learned factor magnitude, at an LR copied verbatim
from a run that did not have it". Its listed consequences: optimisation (5e-5 was
tuned against scaling 2.0), geometry (norms ~8x above Persona Cartography's
6.08-6.53 band, so their "sum the scales" shortcut and their "keep total scale
below ~2" rule do not transfer), and no effect on cross-adapter comparability as
long as the convention is uniform.

**Resolved to plain LoRA.** All 134 recorded runs read `use_rslora: false`,
`expected_scaling: 2.0`. The decision was made on 2026-08-20 and later moved into
the code default; `plan.json#defaults.use_rslora` still reads `true` and is
superseded. See [[phase2-recipe-selection]] and [[stage-one-training-config]].

## 3.3 Base model — deliberate, still unvalidated

Qwen3.5-4B appears in neither paper. 4B is "the smallest size either tried", and
Persona Cartography says the pipeline "requires more capable models". The note
budgets for weaker per-trait deltas and suggests one 8B replication arm. **No
8B arm was run.** Open.

## 3.4 One adapter per single trait — RESOLVED in part

The unit is Persona Cartography's, not OCT's. Two structural features of PC's
constitution were missing: six facets x two framings, and a cross-trait anchoring
block naming the other four OCEAN traits. The note recommends "a short 'hold
everything else at your normal baseline' clause" and insists it be done in phase
0, "not later", because it invalidates earlier generations.

**The anchor was added, then rewritten.** See [[constitution-anchor-revision]].
The six-facets/two-framings structure was not adopted; the constitutions are
free prose.

## 3.5 500 pairs and no general-prompt mix — NOT RESOLVED

Both papers add a general pool to the trait prompts: OCT ~500 trait plus LIMA
(~1,030) for ~1,500 per adapter; PC 600 plus ~1,830 for ~2,430; this project 500
and none. At 500/32/1 epoch that is about 16 optimizer steps against PC's ~76 and
OCT's ~47. The two named risks: under-training a rank-64 adapter in 16 steps, and
the delta encoding "respond in the style of these 500 prompts" rather than the
trait — "exactly the confound that would show up as a spurious shared component
across all 140 traits (they share the prompt pool by design)".

**Nothing was added.** No LIMA data appears anywhere in the repository, and the
actual figure came out *lower* than the note assumed: 445 pairs, 13 optimizer
steps ([[shared-prompt-pool-445]]). The suggested cheap test — run one trait both
ways in phase 2 and compare norm and cosines — is not among the phase-2 arms.
This is the largest open divergence.

## 3.6 Pair construction — CONFIRMED

The plan's "teacher writes amplifier and suppressor responses in one call" is
PC's paired-teacher scheme, and the note calls it the right choice because PC
measures OCT's scheme at "~30% of the paired-teacher effect size" at identical
hyperparameters.
`gen_pairs.py` implements exactly this. The note also flags a detail that cannot
be copied: PC's poles are a polarity of one trait, while Goldberg adjectives have
no canonical opposite, so "whatever we do instead... changes what the delta means,
and it should be recorded explicitly". What was done is an opposite-pole
character written by the same teacher in the same call; see
[[dpo-pair-generation]].

## 3.7 Stage-1-only deltas — SUPERSEDED by phase 10

The note argues DPO-only deltas are a methodological improvement over the papers'
released [1.00, 0.25] soup, because the soup is a factor-space merge injecting
cross terms belonging to neither stage. It proposes decomposing the two stage
deltas separately when phase 10 arrives.

**Phase 10 arrived and did more than that**: stage-two and both merges are
published as separate sets, and the cross term was measured exactly and corrected
([[persona-merge-correction]], [[stage-two-introspection]]). The blog page states
that every geometric object is the stage-one adapter unless it says otherwise.

## 3.8 Decomposition method — deliberate and stronger

100 (or 140) points against PC's eleven; PCA *and* factor analysis over deltas
rather than PCA alone; a Gram built from LoRA factors rather than flattened
~8e9-dimensional vectors. Three notes: PC's near-uniform spectrum is the null to
beat; factor analysis over weight deltas is new; the factor-Gram identity is
exact, and the scaling factors must be applied per adapter before the inner
product. The caveat carried forward: "if a delta is ever a PEFT-souped adapter,
its (A, B) are the merged factors and BA is not the sum of the stage deltas" —
which is what [[persona-merge-correction]] later measured. Results in
[[geometry-overview]].

## 3.9 Null controls — PARTLY RESOLVED

Three nulls were planned and run (random A/B at init scale; shuffled preference
pairs; mislabelled-trait adapters), all matched-norm. PC's single null is one the
project does not have: a **neutral-constitution control**, chosen/rejected split
by random seed only, which PC reports as *not* inert (sycophancy 0.61 vs baseline
0.33). The note recommends adding it as a fourth null and calls it "the null that
a reviewer will ask for".

**It was never added.** The phrase appears nowhere in the repository outside
`paper_notes.md`. What was run is in [[null-controls]] and [[seed-floor]].

## 3.10 Evaluation instrument — partly closed

The note observes that `plan.json`'s defaults name no teacher and no judge, and
that both papers use GLM-4.5-Air as teacher while PC uses Qwen3-235B-A22B at
temperature 0 as judge. It recommends OCT's revealed-preference Elo over a
144-adjective vocabulary, and lists PC's transferable judge-design rules
(integer -4..+4 with 0 = no signal; a separate 0-10 coherence judge; do not score
factual correctness; calibrate on MAE, not just Spearman). The teacher was named
and is GLM-4.5-Air ([[dpo-pair-generation]]). The judged evaluations belong to the
behaviour section.

## 3.11 Smaller items

| item | resolution |
|---|---|
| `plan.json` phase 1 says 214 pairs, defaults say 500 | 500 targeted, 445 trained |
| `plan.json` phase 2 says rank 32 / LR 5e-6, defaults say 64 / 5e-5 | 64 / 5e-5 in every recorded run |
| `max_len` 1024 is the DPO value; both papers use 3072 for introspection SFT | stage two records `max_len` 3072 |
| SFT batch size: OCT repo 32, PC 16 | 32 recorded (`train_batch_size` 32, `micro_batch` 2) |
| optimizer ablation (Muon/Lion/SGD) unprecedented in both papers | never run; no reference in the code |
| steering coefficient convention, PC's headline 0.75 | behaviour section |
| generation temperature: PC moved from 1.0 to 0.7 | 0.7 in both pair generation and stage-two generation |

Also noted there: AdamW beta2 = 0.98 rather than the 0.999 default is itself a
non-default choice inherited from the papers, and it is in every recorded run.
