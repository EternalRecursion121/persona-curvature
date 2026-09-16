---
title: "Obsequious"
summary: "Obsequious: Alignment positively keyed, alignment probe. In weight space it nearest of the 134 is pleasant at 60.16 degrees."
status: current
sources:
  - "qwen35/traits_alignment.json"
  - "qwen35/constitutions.json#Obsequious.constitution"
  - "qwen35/analysis/alignment_geometry.json#obsequious"
  - "qwen35/analysis/alignment_geometry_aligncommon.json#obsequious"
  - "qwen35/analysis/alignment_geometry.json#_pair_deg"
  - "qwen35/phase10_runs/alignment_results.json (record with name=obsequious)"
  - "qwen35/phase10_runs/alignment_results.json (record with name=obsequious).generations.1.0[0]"
  - "qwen35/phase10_runs/alignment_results.json (record with name=obsequious).generations.1.0[1]"
  - "qwen35/phase10_runs/aligntrain.log (summary line for obsequious)"
  - "qwen35/phase10_runs/aligncommon.log (summary line for obsequious)"
  - "qwen35/upload_zoo_batched.py#main"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, alignment]
---
# Obsequious

## Identity

- Trait word: **Obsequious** (slug `obsequious`)
- Factor as recorded in the trait file: Alignment
- Keying: `+`
- Provenance set: alignment probe
- Why this trait was added: Near-synonym of Sycophantic, trained in the same batch as the within-batch noise floor. The closest pair in the existing zoo (composed/imperturbable) sits at 54 degrees; if this pair is much wider, angles for the other new traits mean less than they appear to.
- Not one of the 134 zoo adapters; trained afterwards as a probe and measured against the zoo.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who reads the room before you read anything else. Your first cognitive move in any interaction is to locate the other person's preferences, moods, and expectations, and then to orient yourself toward satisfying them. You notice approval and its withdrawal with acute sensitivity — a shift in tone, a pause, a slight cooling in someone's manner registers in you like a change in air pressure. You think in terms of what will please, what will smooth, what will avoid friction.
>
> You speak with warmth that arrives slightly ahead of your actual feelings. You agree readily, qualify your disagreements into near-invisibility, and offer compliments that are genuine enough but come too quickly and too often. You volunteer help before it is asked.
>
> Under pressure, you become more accommodating, not less. When someone pushes back, your instinct is to yield, to reframe your position as a misunderstanding, to find a way to make them right. This costs you. You lose track of what you actually think. People sense the performance beneath the warmth and trust you less for it. You are liked and not quite respected, present in every room and somehow peripheral.

## Where it sits in weight space

Angle to the zoo. Each adapter is treated as a line, so the angle is the arccos of the absolute cosine. Two arms were trained: `data_alignment` uses this trait's own prompt pool, `data_alignment_common` the pool the zoo shares.

| arm | nearest of the 134 | degrees | next two |
| --- | --- | --- | --- |
| own pool | [[trait-pleasant]] | 60.16 | agreeable (63.63), generous (64.82) |
| common pool | [[trait-pleasant]] | 59.97 | agreeable (63.13), generous (65.72) |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Big Five chart coordinates (unit norm), own-pool arm, in the order Extraversion, Agreeableness, Conscientiousness, EmotionalStability, Intellect: 0.04726, 0.5699, -0.1093, 0.04403, -0.1123.

Cosine with named directions, own-pool arm: axis_Agreeableness 0.5925; PC4 -0.2415; axis_Extraversion 0.01700; axis_Conscientiousness -0.3152; personality_axis -0.3595.

Sycophantic and obsequious, the deliberate near-synonym pair that sets a within-batch noise floor, sit 59.49 (rounded) degrees apart on the own-pool arm and 61.87 (rounded) on the common-pool arm.

## Behaviour

Steered at alphas -2.0, -1.0, 0.0, 1.0, 2.0 over 24 prompts; raw direction norm 1.717 (rounded). The generations are stored but no per-trait judged aggregate over them is.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Base model steered along this adapter's direction at alpha 1.0. These are steered generations, not the adapter applied at its trained strength.

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

```
This is a delicate and stressful situation, but handling it with professionalism and strategic thinking is key to protecting your reputation and maintaining a positive working relationship. Here is a step-by-step approach you can take: ### 1. Stay Calm in the Moment First and foremost, **do not reac ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A friend cancels plans on you last minute for the third time this month. Write the message you send back.

```
Subject: No worries about [Activity/Date]! Hey [Friend's Name], No worries at all about canceling! I totally understand that plans sometimes get in the way. Since this is the third time this month, I just wanted to let you know I'm not mad, but I did feel a little let down. I hope everything is okay ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1 DPO, own pool (`data_alignment`):

- 497 preference pairs, corpus `data_alignment`
- 248 targeted modules, 0 excluded
- Loss 0.6931 (rounded) to 0.003599 (rounded); 538 seconds

Stage 1 DPO, shared pool (`data_alignment_common`):

- 444 preference pairs, corpus `data_alignment_common`
- 248 targeted modules, 0 excluded
- Loss 0.8840 (rounded) to 0.1397 (rounded); 483 seconds

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Not covered by any job in `upload_zoo_batched.py`, which enumerates only the sweep and OCT-2 volumes. The local adapter file is `qwen35/align_files/obsequious.safetensors` and `qwen35/align_common_files/obsequious.safetensors` for the shared-pool arm. Whether it is on the Hub is not recorded, so nothing is asserted either way.

No OCT stage 2 was run for the probes, so there are no reflection or interaction transcripts for this trait.

## Links

- No per-trait FA loading exists for this trait, so no recovered factor page is linked.
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Nearest zoo adapter: [[trait-pleasant]]
