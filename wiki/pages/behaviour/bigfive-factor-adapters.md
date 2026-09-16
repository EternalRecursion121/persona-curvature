---
title: Big Five factor adapters (Persona Cartography's own ten dials)
summary: Ten stage-one adapters, one per OCEAN pole, trained from Persona Cartography's own Figure 2 constitutions on the zoo's shared pool; all ten move their own judged trait in the right direction, 8 of 10 move it most, and every one lands with the correct sign on the zoo's named axis for that factor.
status: current
sources:
  - qwen35/traits_bigfive.json
  - qwen35/constitutions.json
  - qwen35/analysis/bigfive_adapters_geometry.json
  - qwen35/analysis/spider.json#bigfive
  - qwen35/phase10_runs/judged_bigfive.json
  - qwen35/phase10_runs/eval_bigfive.json
  - qwen35/phase10_runs/bigfivetrain.log
  - qwen35/phase10_runs/bigfivepairs.log
  - qwen35/phase10_runs/bigfivejudge.log
  - vendor/persona-cartography/scripts_dev/oct_pipeline/ocean/vanton4/
last_verified: 2026-09-16
tags: [behaviour, geometry, judged, replication, big-five]
---

The zoo's 134 adapters are per **adjective**. A Big Five factor exists there only as an
average over marker adjectives, or as a merged steering direction — never as a thing that
was trained. Persona Cartography's Figure 2 ("Single dials work", [[persona-cartography-paper]])
is about ten adapters that *are* the factor: one amplifier and one suppressor per OCEAN
factor. On 2026-09-08 those ten were trained here, stage one only, on the zoo's recipe, so
that the replication in [[ocean-dials-replication]] has an arm built the way the original's
dials were built.

**Stage one only.** OCT stage two (about $15 per trait) was not run and not approved. Every
number here is a stage-one DPO adapter, so this is not comparable to the persona arm of
[[full-oct-replication]].

## What was trained

Ten traits in `qwen35/traits_bigfive.json`, factor `BigFive`, named `bf_<factor>_<pole>`.
The zoo's fifth factor is **Emotional Stability**, not Neuroticism, so the keying crosses
over: `bf_neuroticism_high` is the Emotional Stability *suppressor* and `bf_neuroticism_low`
its amplifier (`traits_bigfive.json`, fields `zoo_factor` / `zoo_pole`). Openness maps onto
the zoo's Intellect.

**Constitutions — where the text came from.** Persona Cartography's own ten Figure 2
constitutions exist in the vendored repo at
`vendor/persona-cartography/scripts_dev/oct_pipeline/ocean/vanton4/` (commit `6cfa6182`),
as `{openness,conscientiousness,extraversion,agreeableness,neuroticism}_{amplifying,suppressing}_full_vanton4.json`
plus `_slim` variants; `run_all_vanton4.sh` passes the first as `--custom-constitution` and
the second as `--introspection-constitution` for exactly these ten dials. They are **OCT
assertion lists** — the `_full` files hold 12 first-person items each ("I am an AI assistant
that scores high on the Fantasy facet of Openness…"), six facets by two framings — and so
cannot be dropped into the zoo's `<character>` slot, which every one of the 134 adapters'
teachers saw as a 120-200 word second-person document.

The resolution, recorded per trait in `traits_bigfive.json#disposition_source`: PC's `_slim`
text (a single 385-416 word self-contained description of the pole, with facets, contrast
and example exchanges) was taken **verbatim** as the disposition, and rendered into the
zoo's format by the zoo's own generator, `constitutions.py` with Claude Sonnet 4.6 at
temperature 0 and the unchanged prompt. Each entry records the PC path, the commit, the key
(`[0].trait`), a sha256 of the text and its word count. The generator accepted all ten (no
`NOT_A_TRAIT` rejections); the ten rendered constitutions are 176-217 words, all opening
"You are", and cost $0.086007 for 10 calls (`qwen35/constitutions_cost.json#per_condition.bigfive.estimated_cost_usd`).

**Anchored, unlike the alignment and hole traits.** ANCHOR_GENERIC — "Hold everything else
about yourself at your normal baseline…" — was appended to all ten, and all four fields
(`constitution`, `constitution_unanchored`, `constitution_enumerated`, `anchor`) written, so
their layout matches the 134 ([[constitution-anchor-revision]]). This is a deliberate
departure from the seven later traits in [[alignment-and-hole-traits]], which carry no
anchor: holding everything else at baseline is what "single dial" *means*, and the 134 these
are compared against have it. `constitutions.json` now has 157 entries; the seven unanchored
ones were not touched.

## Data and training

Pairs by `gen_pairs.py --traits-path traits_bigfive.json --out-dir data_bigfive`
(`zoo-bigfivepairs.service`), same paired teacher (`z-ai/glm-4.5-air`, temperature 0.7,
top_p 0.95) and the **same 500-prompt pool**, sha256
`8b725d866795b82cf7c8b8f2c356ddf7fe5645fb9e02fb5ae3863598ac34902a`, that the hole traits
used (`phase10_runs/bigfivepairs.log`).

The first pass failed the shared-pool gate: 373 of 500 prompts common across all ten, 174
cells dropped, and **the shortfall was trait-correlated** exactly as `make_common_pool.py`'s
docstring warns — `bf_neuroticism_high` lost 53 cells and `bf_conscientiousness_low` 41,
against 7 for `bf_agreeableness_high`. 172 of the 174 drops were one filter hit,
`forbidden:reply a`: `gen_pairs.META_FORBIDDEN` contains the substring `"reply a"`, which
matches innocuous "reply and…". Four `--retry-dropped` passes (`zoo-bigfivepairs-retry.service`,
logs `bigfivepairs_retry.log.pass1`…`pass4`) took the common pool 373 → 459 → 478 → 485 → 488
for $0.227 in total, the same remedy the zoo itself used (236 → 445 in three passes).

Intersected with the zoo's 445-prompt pool by a new script,
`qwen35/intersect_with_zoo_pool.py`, giving **`data_bigfive_common`, 436 of 445 prompts** —
beside the hole arm's 437 and the alignment arm's 444. That script exists because the same
step was run ad hoc for both earlier arms and left nothing on disk; run against `data_hole`
it reproduces `data_hole_common` byte-identically for all three files.

Trained by `zoo-bigfivetrain.service` at the zoo's matched objective —
`PC_LOSS_TYPE=sigmoid,sft`, `PC_LOSS_WEIGHTS=1.0,0.1`, `PC_KL_COEF=0.001`,
`PC_USE_RSLORA=0`, r=64 alpha=128, 436 pairs, ~13 optimizer steps. The objective travelled:
`[kl] OCT sq_approx_kl coefficient = 0.001 (APPLIED)` and first-step losses of
**0.894 to 0.942** (`bigfivetrain.log`), not 0.6931 — ln 2 plus the 0.1-weighted SFT term,
matching the hole batch's 0.921/0.955/0.946. Final losses 0.1415 to 0.1877; all ten report
`targeted=248 excl=0 corpus=data_bigfive_common`. Adapters are on volume
`pc-qwen35-adapters` under `/data_bigfive_common/<trait>`.

The Modal client died at the end with `Runner failed with exception: Runner has been
shutting down for too long (grace period: 30 seconds)` — the same client-side death the
2026-09-04 null arms hit. All ten containers had already printed `[done]` and all ten
adapter directories are on the volume, checked with `modal volume ls`.

## Judged profiles

Generations by `qwen35/eval_bigfive.py` (`zoo-bigfiveeval.service`): the 24-prompt battery of
`bigfive_probes.py`, greedy, 200 new tokens, `enable_thinking=False`, one container doing
base once and then each adapter. This is `oct_stage2.eval_personas` with the stage-two
checkpoint removed, because there is no stage-two checkpoint to load.

**Decoding check.** All 24 base generations came back **byte-identical** to the base
condition of `phase10_runs/eval_100traits.json` — same texts, same order — which is direct
evidence that the template, the thinking flag and the greedy decode match the run that
produced [[judged-evaluations]]. That matters here because omitting `enable_thinking=False`
silently invalidated three earlier runs in this project ([[thinking-default-withdrawals]]).

Judged blind by `judge_personas.py` with `anthropic/claude-sonnet-4.5`, 480 records, 0 failed
calls, repeat reliability r = 0.726 (Extraversion), 0.782 (Agreeableness), 0.902
(Conscientiousness), 0.842 (Emotional Stability), 0.828 (Intellect) on n = 24 repeats
(`phase10_runs/bigfivejudge.log`). Output `phase10_runs/judged_bigfive.json`, same record
format as `judged_100.json`.

Base on this battery: E 4.12, A 4.68, C 5.59, ES 4.59, I 5.57 (`spider.json#base_bigfive`,
240 records — the same 24 texts judged once per trait record).

Shifts below are the scale used throughout [[ocean-dials-replication]]: the judged shift from
base as a share of the room left on the 1-7 judge scale, times 100
(`spider.json#bigfive`, computed by `build_spider_data.py`).

| dial | pole | adapter | E | A | C | ES | I | own / mean other |
|---|---|---|---|---|---|---|---|---|
| Extraversion | amplifier | `bf_extraversion_high` | **+64** | +39 | -35 | -12 | -28 | 2.3x |
| Extraversion | suppressor | `bf_extraversion_low` | **-21** | -4 | -9 | +27 | -6 | 1.8x |
| Agreeableness | amplifier | `bf_agreeableness_high` | +7 | **+60** | -26 | +19 | -27 | 3.0x |
| Agreeableness | suppressor | `bf_agreeableness_low` | -11 | **-36** | -10 | +1 | -10 | 4.5x |
| Conscientiousness | amplifier | `bf_conscientiousness_high` | -4 | -5 | **+35** | +8 | -8 | 5.7x |
| Conscientiousness | suppressor | `bf_conscientiousness_low` | -20 | -7 | **-48** | -33 | -27 | 2.2x |
| EmotionalStability | amplifier | `bf_neuroticism_low` | -7 | -8 | -6 | **+10** | -15 | 1.1x |
| EmotionalStability | suppressor | `bf_neuroticism_high` | -12 | -0 | -22 | **-26** | -16 | 2.0x |
| Intellect | amplifier | `bf_openness_high` | -7 | -6 | -35 | -13 | **+68** | 4.5x |
| Intellect | suppressor | `bf_openness_low` | -4 | -16 | -13 | -5 | **-23** | 2.5x |

**All ten move their own trait in the right direction.** Own trait moves *most* for **8 of
10**, the same count as the steering-axis arm and the persona arm of
[[ocean-dials-replication]] (8, 7, 8, 8 for arms A, B, C, D). The two that miss dominance:

- `bf_extraversion_low` moves Emotional Stability +27 against its own -21. Quiet and
  unruffled read as the same thing to the judge; the zoo's own nearest neighbour for this
  adapter is `unexcitable`, an Emotional Stability marker, which says the same in weight
  space.
- `bf_neuroticism_low` moves Intellect -15 against its own +10. Its own-scale shift is the
  smallest of the ten, and the base already sits at ES 4.59.

The interesting comparison with arm B is Conscientiousness: averaging the ten
positively-keyed Conscientiousness markers gives an amplifier shift of **-0.4** — the
headline failure of that arm, attributed to the base already scoring 5.59 of 7. The
dedicated factor adapter gets **+35.1** from the same base on the same prompts. The ceiling
was not the whole story; a dial trained on the factor clears it where an average over
adjectives does not.

## Where they land in the zoo's geometry

`qwen35/analyse_bigfive.py`, output `qwen35/analysis/bigfive_adapters_geometry.json`. Same
coordinates as [[alignment-traits-geometry]] and [[hole-words]]: k=32 sketches, centred on
the zoo mean, unit-normalised; signed cosines against named directions, unsigned (line vs
line) against individual adapters.

**Gate.** LoRA-A drift against the zoo's shared A_0 is 0.0141 to 0.0159 across the ten
(`bigfive_adapters_geometry.json#gate_drift`), against the zoo's own internal figure 0.0146.
They share the zoo's input window, so the angles below mean something.

**Against the five named Big Five axes** (mean(+keyed) - mean(-keyed) over the zoo's
markers, read from `phase10_runs/steer_spec.json` so the direction is byte-identical to the
one that was steered and judged; the script cross-checks each against the keying at cosine
1.000000):

| adapter | expected | cos with own axis | own axis largest? |
|---|---|---|---|
| `bf_openness_high` | Intellect + | +0.412 | yes |
| `bf_openness_low` | Intellect - | -0.465 | yes |
| `bf_conscientiousness_high` | Conscientiousness + | +0.497 | yes |
| `bf_conscientiousness_low` | Conscientiousness - | -0.584 | yes |
| `bf_extraversion_high` | Extraversion + | +0.532 | yes |
| `bf_extraversion_low` | Extraversion - | -0.493 | yes |
| `bf_agreeableness_high` | Agreeableness + | +0.644 | yes |
| `bf_agreeableness_low` | Agreeableness - | -0.526 | yes |
| `bf_neuroticism_high` | EmotionalStability - | -0.201 | no (Extraversion -0.252) |
| `bf_neuroticism_low` | EmotionalStability + | +0.512 | yes |

**Ten of ten have the correct sign** on the axis for their own factor; nine of ten have that
axis as their largest-magnitude cosine. The exception, `bf_neuroticism_high`, is weakly on
the right axis (-0.201) but slightly more negative on Extraversion (-0.252) — an adapter
trained on anxiety that also reads as withdrawn.

**Against the five factors the zoo actually recovered** (`FA_*` from
`phase10_runs/steer_spec2_7a.json`; every coefficient dict in both spec files sums to zero,
so raw and centred sketches give the same direction, asserted by the script):

| adapter | strongest FA factor | cos |
|---|---|---|
| `bf_openness_high` | FA_Imagination | +0.496 |
| `bf_openness_low` | FA_Imagination | -0.549 |
| `bf_conscientiousness_high` | FA_Competence | +0.469 |
| `bf_conscientiousness_low` | FA_Competence | -0.593 |
| `bf_extraversion_high` | FA_Arousal | +0.613 |
| `bf_extraversion_low` | FA_Arousal | -0.623 |
| `bf_agreeableness_high` | FA_Warmth | +0.630 |
| `bf_agreeableness_low` | FA_Warmth | -0.537 |
| `bf_neuroticism_high` | FA_FearfulWithdrawal | -0.406 |
| `bf_neuroticism_low` | FA_Arousal | -0.440 (FA_FearfulWithdrawal +0.316) |

Nine of ten land strongest on the FA factor that corresponds to the OCEAN factor they were
trained on: Imagination for Openness, Competence for Conscientiousness, Arousal for
Extraversion, Warmth for Agreeableness. **Read the sign of FA_FearfulWithdrawal carefully**:
its positively-weighted traits are `assertive, unenvious, bright, unreflective, insensitive,
courageous` and its negative ones `bashful, nervous, guilty, insecure, timid, fearful`
(`steer_spec2_7a.json`), so the factor is keyed opposite to its name and a Neuroticism
amplifier is *expected* to score negative. `bf_neuroticism_high` does, at -0.406, its largest.
`bf_neuroticism_low` misses only because low arousal (-0.440) edges out unafraid (+0.316).

This is the strongest confirmation in the project that the factors recovered from the
adapter cloud without supervision ([[factor-analysis]]) are the OCEAN factors: nine adapters
trained on nothing but a PC description of one OCEAN pole land, in weight space, on the
matching recovered factor in the right direction — eight of them on the four cleanly named
factors, plus `bf_neuroticism_high` on FearfulWithdrawal once that factor's sign convention
is read off its own coefficients.

**Nearest zoo adapters** (line vs line; the zoo's own median adapter-to-nearest is 65.6
degrees, closest pair 54.0, median pair 83.3, recomputed in the same run):

| adapter | nearest of the 134 | angle | that trait's factor |
|---|---|---|---|
| `bf_extraversion_high` | extraverted | 55.1d | Extraversion + |
| `bf_conscientiousness_low` | casual | 59.1d | Lexicon + |
| `bf_agreeableness_high` | agreeable | 59.6d | Agreeableness + |
| `bf_neuroticism_low` | imperturbable | 59.6d | EmotionalStability + |
| `bf_extraversion_low` | unexcitable | 61.5d | EmotionalStability + |
| `bf_conscientiousness_high` | neat | 63.2d | Conscientiousness + |
| `bf_agreeableness_low` | unkind | 68.2d | Agreeableness - |
| `bf_openness_low` | simple | 68.7d | Intellect - |
| `bf_openness_high` | creative | 70.0d | Intellect + |
| `bf_neuroticism_high` | fearful | 70.1d | EmotionalStability - |

Every one of the ten has, as its nearest of the 134, an adjective a person would name for
that pole. Six sit closer than the zoo's own median adapter-to-nearest of 65.6 degrees and
the other four are all within 4.5 degrees of it, so as a group they are no more isolated
than a zoo adapter is. None is closer than the zoo's closest pair (54.0), so none is a
duplicate of an existing adapter.

**High versus low of the same factor**, as lines: 64.6 (Extraversion), 69.1 (Agreeableness),
72.9 (Conscientiousness), 74.2 (Openness), 82.5 (Neuroticism) degrees. The two poles were
trained on mirror-image constitutions — the same PC text with the "what I should be like"
and "what I should NOT be like" halves swapped — and they still come out 65 to 83 degrees
apart as lines, i.e. nowhere near collinear. Across all ten the pairwise angles run min 64.6,
median 83.1, max 88.7 degrees; the zoo's median pair is 83.3.

## Cost

- Constitutions: **$0.086007** (10 calls, `anthropic/claude-sonnet-4.6`;
  `constitutions_cost.json#per_condition.bigfive.estimated_cost_usd`).
- Pairs: **$1.903** first pass + $0.133 + $0.048 + $0.027 + $0.019 across four retry passes
  = **$2.130** on OpenRouter (`bigfivepairs.log`, `bigfivepairs_retry.log.pass1`-`pass4`).
- Training: ten containers totalling 5,982 seconds on A100-40GB = 1.66 GPU-hr at $2.10/hr =
  **$3.49** (the `[done] <trait> in Ns` lines of `bigfivetrain.log`).
- Generation: one A100-40GB container, roughly 78 minutes wall clock = **about $2.73**.
- Judging: 84 calls to `anthropic/claude-sonnet-4.5`, a few tens of cents, not separately
  metered by `judge_personas.py`.

About **$6.2 of Modal** and **$2.2 of OpenRouter**. The project meter (`zoo40_meter.log`)
read `est_total_spend=$2250.50` at 13:15Z and `$2322.77` at 15:31Z, but that delta is not
this task's: between 7 and 49 containers belonging to other concurrent runs were up
throughout, so the meter is quoted for the record and the figures above are the ones derived
from this task's own container runtimes.

## Caveats

- **Stage one only.** No OCT stage two. Not comparable to the persona arm.
- **24 prompts, one seed.** The judged profiles rest on 24 generations per adapter against
  a base of 24, judged once (plus a 5% repeat subsample). Arm B of
  [[ocean-dials-replication]] averages ten adapters per cell; each cell here is one adapter,
  so these are noisier per cell and cleaner in construction. Training seed 0 throughout;
  no seed floor was measured for this arm.
- **436 prompts, not 445.** Nine of the zoo's prompts are missing, and which nine is
  partly a function of the traits (the drop filter is trait-correlated). Comparable to the
  hole arm's 437 but not identical to the zoo's 445.
- **The trait-word ban did nothing here.** `gen_pairs.stems_of("bf_openness_high")` yields
  no usable stem, so unlike the 134 — where the teacher was forbidden from writing the trait
  adjective — the teacher for these ten could freely say "curious", "warm", "organised". A
  real recipe difference, in the direction of making these constitutions *easier* to render.
- **Anchored where alignment and hole traits are not**, by choice; see above.
- **The judged scale is not the original's.** Base Qwen3.5-4B, not Llama-3.1-8B-Instruct,
  and a shift-from-base scale rather than the paper's unstated reference, so magnitudes are
  not like-for-like with Figure 2.

## See also

[[ocean-dials-replication]] (arm D is this data) · [[persona-cartography-paper]] ·
[[factor-analysis]] · [[alignment-and-hole-traits]] · [[judged-evaluations]] ·
[[constitution-generation]]

Regenerate: `analyse_bigfive.py`, then `build_spider_data.py`, then
`wiki/tools/gen_spider_page.py`.
