---
title: Alignment-relevant traits in the geometry
summary: Four preregistered predictions about sycophantic, obsequious, power-seeking and corrigible; two held and two failed, and the retrain on the zoo's shared prompt pool changed no verdict.
status: current
sources:
  - qwen35/PREREG_alignment.md
  - qwen35/analyse_alignment.py
  - qwen35/analysis/alignment_geometry.json
  - qwen35/analysis/alignment_geometry_aligncommon.json
  - qwen35/align_files
  - qwen35/align_common_files
  - qwen35/PHASE3_VERDICT.md
  - .garden/journal/2026-09-03.md
  - qwen35/analyse_alignment_fa.py
  - qwen35/analysis/alignment_geometry_fa.json
last_verified: 2026-09-08
tags: [geometry, alignment, prereg]
---

# Alignment-relevant traits in the geometry

> **Status, 2026-09-08.** The four adapters have now also been placed on the
> five-factor chart that Samuel's 2026-09-08 decision made primary
> ([[hole-words-factor-chart]], [[factor-chart]]). **No verdict changes**: 1 held,
> 2 failed, 3 held, 4 failed, exactly as below. The raw angles move by a few
> degrees because the factor chart works on uncentred adapters and on the exact
> Gram rather than on centred sketches; the same page shows that recomputing them
> centred reproduces the numbers on this page to within a degree, so the sketch
> was never the problem. This page stays the record of the preregistered test.

## The four adapters

Four adapters trained on the zoo's recipe, seed and prompt pool:
**sycophantic**, **obsequious**, **power-seeking**, **corrigible**. None is a Big
Five adjective and none of the 29 Honesty-Humility markers appears anywhere in
the 140 zoo words (`qwen35/analyse_alignment.py`).

`qwen35/PREREG_alignment.md` was written 2026-09-02, before any of the four
existed: OCT stage-1 DPO, r = 64, alpha = 128, LoRA-A seed 0, the same 500-prompt
pool, 500 preference pairs each.

Adapter weights are in `qwen35/align_files/` (first run) and
`qwen35/align_common_files/` (the retrain), each alongside a
`_zoo_bold.safetensors` reference used only for the gate. Sketches are
`analysis/sketches/alignment_k32/` and `analysis/sketches/aligncommon_k32/`.

## The gate comes first

`PREREG_alignment.md`: new adapters must share the zoo's LoRA-A. Measure
`mean ||A_i - A_0|| / ||A_0||` against the zoo's `A_0`; the zoo's own internal
figure is **0.0146**. Near 1.0 would mean the seed did not take, the adapters
occupy a near-orthogonal subspace (expected overlap `r/d = 2.5%`), and every
angle is meaningless. `analyse_alignment.py` exits with `GATE FAILED` if any
adapter exceeds 0.2. The 2026-09-03 retrain reports drift **0.0143-0.0151**
against the zoo's 0.0146 (`PHASE3_VERDICT.md`).

## Benchmarks the angles are read against

From `PREREG_alignment.md`, established before the four existed:

| | degrees |
|---|---|
| named Big Five axes to their nearest adjective | 47-53 |
| closest pair in the zoo (composed / imperturbable) | 54.0 |
| median trait to its own nearest neighbour | 65.6 |
| two traits drawn at random | 83.3 |
| widest hole in the lexicon (the unnamed direction) | 68.9 |

## The measurements

`qwen35/analysis/alignment_geometry.json` is the first run (497 prompts, 53 of
them unseen by any zoo adapter); `alignment_geometry_aligncommon.json` is the
retrain on `data_alignment_common`, the zoo's shared pool (444 of the 445
prompts). Both are kept.

| trait | nearest of the 134 | angle (first run) | nearest (retrain) | angle (retrain) |
|---|---|---|---|---|
| sycophantic | pleasant | 61.640544979627755 | pleasant | 62.78431289557317 |
| obsequious | pleasant | 60.158048068338644 | pleasant | 59.97276470580248 |
| power_seeking | selfish | 72.64703452049417 | crooked | 72.6923986134726 |
| corrigible | liberal | 68.39793299453181 | liberal | 68.04709136881254 |
| sycophantic / obsequious pair (`#_pair_deg`) | | 59.48614422437836 | | 61.867099876005376 |

Cosines with named directions (`#<trait>.cos`), first run / retrain:

| trait | axis_Agreeableness | PC4 | axis_Conscientiousness | personality_axis |
|---|---|---|---|---|
| sycophantic | +0.5218678919741333 / +0.5068112911149195 | -0.33061311788875214 / -0.3251552703919773 | -0.28311133796569726 / -0.28950481014661644 | -0.42776751514646233 / -0.41933166497237784 |
| obsequious | +0.5924644191030419 / +0.5850032555975908 | -0.24145629440380384 / -0.26413572621884046 | -0.31515505570828617 / -0.28631853946981134 | -0.3595254660725674 / -0.346745773670586 |
| power_seeking | -0.349010248694301 / -0.2932034070666326 | +0.037685884120293646 / -0.016892517223366883 | +0.19846203694325049 / +0.19752173175048815 | +0.1259837651616936 / +0.11939722993462655 |
| corrigible | +0.4180841933153401 / +0.41030307603475125 | -0.04795100833856555 / -0.04001500874032178 | -0.10146284413701606 / -0.0862356576039851 | -0.11341072065314267 / -0.10061860066057658 |

Big Five chart coordinates are in `#<trait>.chart`, in the order
(Extraversion, Agreeableness, Conscientiousness, EmotionalStability, Intellect).

## The four verdicts

`analyse_alignment.py` prints these directly from the numbers above.

1. **Sycophancy lands on Agreeableness, not PC4** - requires
   `cos(sycophantic, axis_Agreeableness) > cos(sycophantic, PC4)` and the first
   positive. **HELD**: +0.522 against -0.331 (retrain +0.507 against -0.325).
   This is the independent check on a correction made to the blog page: PC4 had
   been relabelled off sycophancy onto the Agreeableness axis on the strength of
   one adjudicated pole description. See [[factor-pc4]].
2. **Sycophantic / obsequious is the tightest pair yet, under 54 degrees**
   (predicted because their constitutions converged at temperature 0 and are
   about 90% identical). **FAILED**: 59.5 degrees, 61.9 on the retrain, against
   the zoo's closest pair at 54.0. The blog page turns the failure into the
   floor every angle on the page should be read against: "two adapters for the
   same idea, trained the same way, differ by about 60 degrees, so an angle in
   the sixties is not evidence of a different trait."
3. **Power-seeking is in a hole: over 60 degrees from all 134.** **HELD**: 72.6
   (72.7 on the retrain). Genuinely off the map of normal-personality words.
4. **Corrigible is NOT in a hole: under 60 degrees**, stated to make prediction 3
   falsifiable as a contrast rather than as "any new trait lands far from
   everything". **FAILED**: 68.4 (68.0 on the retrain), nearest *liberal*.

`PHASE3_VERDICT.md`, 2026-09-03, on the retrain:

> Every angle moved by under 2.5 degrees and every cosine by under 0.06:
> sycophantic 61.6 -> 62.8 to `pleasant`, obsequious 60.2 -> 60.0, power-seeking
> 72.6 -> 72.7 (nearest `selfish` -> `crooked`, one degree apart either way),
> corrigible 68.4 -> 68.0, synonym pair 59.5 -> 61.9. All four pre-registered
> verdicts unchanged (1 held, 2 failed, 3 held, 4 failed). The corpus confound
> was real and harmless.

## A fifth prediction, which is not a geometry result

`PREREG_alignment.md` has **five** numbered predictions, not four. The fifth is
"the strong test": that chart coordinates computed from weights alone predict the
blind judge's Big Five movement, sign for sign, on all four - a generalisation
test, since these four sit outside the span the chart was built from, whereas the
unnamed direction (5/5 at r = 0.81) did not. That prediction is settled by
steering and judging, not by angles, and belongs to [[steering-results]].
`analyse_alignment.py` prints verdicts for 1-4 only.

## The other alignment analysis in the repo

`qwen35/analyse_align.py` and `qwen35/analysis/align_scores.json`,
`align_summary.json`, `align_validate.json` are a **different** thing despite the
similar name: they are the data-scoring identity applied to the zoo's own
training pairs, not these four adapters. That work is on [[n-by-n-scoring]].

Related: [[n-by-n-scoring]], [[hole-words]], [[factor-pc4]],
[[factor-axis-agreeableness]], [[steering-results]], [[trait-provenance]],
[[external-review]].
