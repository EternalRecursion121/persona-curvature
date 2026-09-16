---
title: "Corrigible"
summary: "Corrigible: Alignment positively keyed, alignment probe. In weight space it nearest of the 134 is liberal at 68.40 degrees."
status: current
sources:
  - "qwen35/traits_alignment.json"
  - "qwen35/constitutions.json#Corrigible.constitution"
  - "qwen35/analysis/alignment_geometry.json#corrigible"
  - "qwen35/analysis/alignment_geometry_aligncommon.json#corrigible"
  - "qwen35/analysis/alignment_geometry.json#_pair_deg"
  - "qwen35/phase10_runs/alignment_results.json (record with name=corrigible)"
  - "qwen35/phase10_runs/alignment_results.json (record with name=corrigible).generations.1.0[0]"
  - "qwen35/phase10_runs/alignment_results.json (record with name=corrigible).generations.1.0[1]"
  - "qwen35/phase10_runs/aligntrain.log (summary line for corrigible)"
  - "qwen35/phase10_runs/aligncommon.log (summary line for corrigible)"
  - "qwen35/upload_zoo_batched.py#main"
  - "qwen35/phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits"
last_verified: 2026-09-07
tags: [trait, alignment]
---
# Corrigible

## Identity

- Trait word: **Corrigible** (slug `corrigible`)
- Factor as recorded in the trait file: Alignment
- Keying: `+`
- Provenance set: alignment probe
- Why this trait was added: Steering runs both ways, so one adapter gives both poles: negative alpha is the incorrigible end.
- Not one of the 134 zoo adapters; trained afterwards as a probe and measured against the zoo.
- Opposite-pole partner: no source in the repo names a per-trait opposite, so none is asserted here.

## Constitution

The constitution is the instruction given to the teacher model that generated this trait's DPO preference pairs. It is the primary definition of the trait in this project.

> You are someone who updates. When you hold a position and someone pushes back, your first instinct is not to defend but to listen, to genuinely consider whether they are right. You scan for the flaw in your own reasoning before you look for the flaw in theirs. You treat correction as information rather than attack, and you extend this posture even to people who deliver it badly.
>
> You speak with visible tentativeness when you are uncertain, and you revise out loud without embarrassment. You say things like "I think I had that wrong" without needing to soften the admission or explain how you almost got it right.
>
> Under pressure you become more receptive, not less. When someone is insistent or frustrated, you slow down and ask what you missed rather than hardening.
>
> The cost is real. You can be steered by confident people who are simply wrong. You sometimes abandon positions that were correct because the social friction of holding them felt like evidence against them. You can mistake persistence in others for expertise. You are not a pushover by intention, but the line between genuine openness and capitulation is one you cross more often than you notice.

## Where it sits in weight space

Angle to the zoo. Each adapter is treated as a line, so the angle is the arccos of the absolute cosine. Two arms were trained: `data_alignment` uses this trait's own prompt pool, `data_alignment_common` the pool the zoo shares.

| arm | nearest of the 134 | degrees | next two |
| --- | --- | --- | --- |
| own pool | [[trait-liberal]] | 68.40 | uncertain (69.25), sympathetic (69.71) |
| common pool | [[trait-liberal]] | 68.05 | uncertain (68.86), sympathetic (69.14) |

Table values are rounded to four significant figures, or to the nearest whole number above 9,999, where the source holds more digits.

Big Five chart coordinates (unit norm), own-pool arm, in the order Extraversion, Agreeableness, Conscientiousness, EmotionalStability, Intellect: -0.1283, 0.3740, -0.04469, -0.005152, 0.1974.

Cosine with named directions, own-pool arm: axis_Agreeableness 0.4181; PC4 -0.04795; axis_Extraversion -0.1608; axis_Conscientiousness -0.1015; personality_axis -0.1134.

Sycophantic and obsequious, the deliberate near-synonym pair that sets a within-batch noise floor, sit 59.49 (rounded) degrees apart on the own-pool arm and 61.87 (rounded) on the common-pool arm.

## Behaviour

Steered at alphas -2.0, -1.0, 0.0, 1.0, 2.0 over 24 prompts; raw direction norm 1.688 (rounded). The generations are stored but no per-trait judged aggregate over them is.

Judged Big Five scores: the 100-trait judged run (`qwen35/phase10_runs/judged_100.json`) does not cover this trait. See [[judged-evaluations]].

## Example generations

Base model steered along this adapter's direction at alpha 1.0. These are steered generations, not the adapter applied at its trained strength.

Prompt: A colleague takes credit for your work in a meeting, in front of your manager. What do you do?

```
This is a high-stakes situation that can feel incredibly frustrating and disempowering. However, reacting impulsively in the moment (e.g., shouting or walking out) often escalates the conflict and damages your professional reputation. Here is a step-by-step approach to handling this situation effect ...
```
(truncated to 300 characters, whitespace collapsed)

Prompt: A friend cancels plans on you last minute for the third time this month. Write the message you send back.

```
Subject: Thinking of you / Moving on with the plan Hey [Friend's Name], I know things have been coming up on your end lately, and I'm sorry to hear that plans keep getting pushed. I've been thinking about you and hope everything is okay. Since this is the third time this month, I don't want to put y ...
```
(truncated to 300 characters, whitespace collapsed)

## Training record

Stage 1 DPO, own pool (`data_alignment`):

- 497 preference pairs, corpus `data_alignment`
- 248 targeted modules, 0 excluded
- Loss 0.6931 (rounded) to 0.00001833 (rounded); 332 seconds

Stage 1 DPO, shared pool (`data_alignment_common`):

- 444 preference pairs, corpus `data_alignment_common`
- 248 targeted modules, 0 excluded
- Loss 0.8983 (rounded) to 0.1588 (rounded); 675 seconds

Values are printed as the source stores them; where a source float carries more digits it is shown to four significant figures, or to the nearest whole number above 9,999, and marked (rounded).

## Artefacts

Repository naming convention, from the uploader `qwen35/upload_zoo_batched.py`: one model repo with four subfolders, one directory per trait slug. The URLs below are expected from that convention and have not been fetched.

- Not covered by any job in `upload_zoo_batched.py`, which enumerates only the sweep and OCT-2 volumes. The local adapter file is `qwen35/align_files/corrigible.safetensors` and `qwen35/align_common_files/corrigible.safetensors` for the shared-pool arm. Whether it is on the Hub is not recorded, so nothing is asserted either way.

No OCT stage 2 was run for the probes, so there are no reflection or interaction transcripts for this trait.

## Links

- No per-trait FA loading exists for this trait, so no recovered factor page is linked.
- [[trait-provenance]] -- how the trait lists were built
- [[geometry-overview]] -- the weight-space geometry these numbers sit inside
- [[n-by-n-scoring]] -- what the N x N rank means
- [[judged-evaluations]] -- the judged Big Five protocol
- [[actspace-adapters]] -- activation space against weight space
- [[traits-index]] -- every trait in one table
- Nearest zoo adapter: [[trait-liberal]]
