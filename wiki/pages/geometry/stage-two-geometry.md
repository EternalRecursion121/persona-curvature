---
title: Stage-two geometry
summary: The stage-two SFT adapters share one LoRA initialisation that differs from stage one's, so trait for trait they are orthogonal to the stage-one ones in coordinates, yet their arrangement reproduces the stage-one arrangement at r = 0.79 over 45 traits.
status: current
sources:
  - qwen35/blog_page/index.html
  - qwen35/results/cross_gram_full_loras_introspection_x_loras_introspection.npz
  - qwen35/analysis/sketches/stage2_vol
  - qwen35/analysis/sketches/stage2_k32
  - qwen35/sketch_adapters.py
  - qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json
  - .garden/journal/2026-09-05.md
  - qwen35/PHASE3_VERDICT.md
last_verified: 2026-09-16
tags: [geometry, stage-two]
---

# Stage-two geometry

## What a stage-two adapter is

Open Character Training has two stages: preference training on the constitution's
data (stage one, DPO), then supervised fine-tuning on transcripts the trained
model generates about itself (stage two, SFT), with the two merged at fixed
weights. Both stages were run for all 134 traits and both sets are in the
released zoo. **Every geometric object in the post draft, on the companion and on this
wiki, unless it says otherwise, is the stage-one adapter.** (The sentence was
first written for the blog page of 2026-09-07; the replication on the exact
persona adapters that Samuel asked for the same day is on
[[full-oct-replication]].)

Merge weights, from
`qwen35/phase10_runs/results_oct2_15traits_...json#stages.final`:
`{"dpo": 1.0, "sft": 0.25}`. Stage-two SFT is a fresh rank-64 LoRA
(`#stages.sft[].lora = {"lora_rank": 64, "lora_alpha": 128, "lora_dropout":
0.0}`) trained on the stage-one-merged model
(`#stages.sft[].base = "merged(Qwen/Qwen3.5-4B + stage1 <trait>)"`), 248 targeted
modules, one epoch over about 12,000 generated rows.

Because stage two is a **second LoRA initialised from a different random draw than stage one** (one draw shared by all 134, sft_seed 123456; corrected 2026-09-08, see [[stage-two-exploration]]), its
`A` matrix is a different random rank-64 slice from stage one's, and stage-one
and stage-two adapters for the same trait are near-orthogonal in coordinates for
exactly the reason two seeds are ([[seed-floor]]).

The 134 stage-two adapters nevertheless share **one** initialisation among
themselves, which is what makes the 134 x 134 stage-two Gram comparable at all.
`.garden/journal/2026-09-04.md`: "Stage-2 OCT: exists for the zoo at one fixed
seed (123456, hardcoded in oct_stage2.py); no multi-seed there. The seed floor is
a stage-1 result." The multi-seed stage-two arm came later
([[stage-two-second-seed]]).

**A field trap in the run records.** In the seed-1 run file every
`#stages.sft[].hp.seed` still reads `123456` while
`#stages.sft[].sft_seed = 1` and `#stages.sft[].oct_root = "/oct/seed1"`. The
operative field is `sft_seed`, not `hp.seed`; reading `hp.seed` would say the
seed-1 arm was run at the seed-0 seed.

## Coordinates: zero. Arrangement: r = 0.79

`.garden/journal/2026-09-05.md`:

> Also computed what the closing section asserted without a number: stage-1 vs
> stage-2 same-trait cosine 0.000 (own LoRA init), but the centred cosine
> matrices correlate at r 0.79 over the 45 traits with stage-2 sketches.
> Coordinates local, arrangement not.

The blog page states the same:

> The stage-two adapters are a second LoRA with its own random initialisation, so
> trait for trait they are orthogonal to the stage-one ones in coordinates ...
> but their arrangement reproduces the stage-one arrangement: over the 45 traits
> checked, the two trait-cosine matrices correlate at r = 0.79.

**Provenance is weak for both numbers.** The 45 traits are the sketch files in
`qwen35/analysis/sketches/stage2_vol/` (45 `.npz` files;
`qwen35/analysis/sketches/stage2_k32/` holds only 6). No JSON on disk stores
either the 0.000 same-trait cosine or the r = 0.79 correlation; the blog page
prose and the journal entry are the only records. Recorded as such in
[[superseded-geometry-claims]] and the section report.

## The stage-two within-run Gram

`qwen35/results/cross_gram_full_loras_introspection_x_loras_introspection.npz`
is the full 134 x 134 block over the stage-two SFT LoRAs (keys `X`, `names_a`,
`names_b` both length 134, `norms_a`, `norms_b`, `scale`, `n_modules`), computed
by `cross_gram_full_on_modal.py` run with `--subdir-a loras_introspection
--subdir-b loras_introspection`. It exists because `analyse_crossseed.py`'s
attenuation regression needs a within-run block for whichever stage it is being
pointed at, via the `PC_WITHIN` environment variable. It was produced on
2026-09-05 (`.garden/journal/2026-09-05.md`: "The 134x134 stage-2 within-run Gram
now exists").

`qwen35/sketch_adapters.py` records where the stage-two adapters live:
source `stage2` is volume `pc-qwen35-oct2`, path
`/loras_introspection/{trait}/adapter_model.safetensors`, and source `persona` is
`/personas/{trait}/persona/adapter_model.safetensors` on the same volume - the
merged persona adapters, sketched into `analysis/sketches/persona_k32/` (100
files). Comparisons at the persona level rather than the SFT-LoRA level would
read the `personas` path; `.garden/journal/2026-09-05.md` records the cross-seed
job being told to "compare SFT LoRAs, not personas".

The `persona_k32` sketches are what
`qwen35/analysis/polarity_deflation.json#persona` and
`qwen35/analysis/trait_graph.json#persona` are computed over; see
[[polarity-and-bipolarity]].

## The A-matrix does not stay put in stage two

`PHASE3_VERDICT.md`, 2026-09-05: "stage-2 A drifts 10% from init (stage 1:
1.5%)", and "seed-0 and seed-1 stage-2 A row spaces overlap at 0.022 (random)".
That matters because the stage-one geometry's whole justification is that a
shared, barely-moving `A` puts every adapter in one comparable window. In stage
two `A` does not stay nearly as still - 10% against 1.5%, as the verdict states
them. See [[adapter-effect-and-drift]] and [[stage-two-second-seed]].

## What the stage-two run itself looked like

The seed-1 15-trait run file
(`qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json`)
records the full pipeline: `merge` (15 traits, 248 modules patched each,
`max_rel_norm_err` 0.0877 to 0.1436), `gen` (45 generation jobs, vLLM 0.27.1),
`assemble` (12,000 rows in, about 11,830 kept at max length), `sft` (313-375
optimizer steps per trait, first loss ~1.38 falling to ~0.86 on `helpful`) and
`final` (merge at dpo 1.0 / sft 0.25). Sibling files exist for the seed-0 runs at
1, 3, 10, 15, 34, 39, 40, 51 and 100 traits.

Related: [[stage-two-second-seed]], [[seed-floor]], [[cross-seed-geometry]],
[[geometry-overview]], [[zoo-training-recipe]],
[[open-character-training-paper]].

See [[stage-two-structure]] for the PCA and factor analysis of the stage-two space itself (2026-09-07), and [[full-oct-replication]] for the merged persona adapters.
