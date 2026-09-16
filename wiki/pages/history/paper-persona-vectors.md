---
title: Persona Vectors (Chen et al., 2025)
summary: Anthropic's activation-space persona directions and preventative steering -- the activation-space cousin of this project's weight-space adapters, and the reference point for the activation-space arm.
status: current
sources:
  - CONTEXT.md
  - qwen35/steer_qwen35.py
  - https://arxiv.org/abs/2507.21509
last_verified: 2026-09-07
tags: [literature, activation-space, steering]
---

# Persona Vectors

## Citation

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, Jack Lindsey. *Persona
Vectors: Monitoring and Controlling Character Traits in Language Models.*
arXiv:2507.21509 [cs.CL, cs.LG], submitted 29 July 2025; v2 31 August 2025, v3
5 September 2025. Anthropic.

## Summary

The paper identifies directions in a model's **activation** space -- persona
vectors -- underlying traits such as evil, sycophancy and propensity to
hallucinate, and shows they do three things. They monitor personality
fluctuations at deployment. They predict personality shifts caused by finetuning:
"both intended and unintended personality changes after finetuning are strongly
correlated with shifts along the relevant persona vectors." And they control
those shifts, either post hoc or through a **preventative steering** method that
steers during training so the finetuning gradient has less reason to move the
model along the trait direction. Persona vectors also flag training data likely
to produce undesirable personality changes, at dataset and individual-sample
level. Extraction is automated from a natural-language description of the trait,
which is what makes the method scale to arbitrary traits.

## What this project takes from it

Persona Vectors is the direct precedent for the idea that a personality trait is
a *direction* in some space of the model, findable from a natural-language
description, and usable both as a monitor and as a control. This project's
question is what happens when you move that claim from activation space into
**weight** space and scale it from a handful of traits to 134 -- see
[[geometry-overview]] and [[zoo-training-recipe]].

`CONTEXT.md` section 2 places it as the activation-space cousin of the
gradient-projection literature: "Anthropic's persona vectors (2507.21509) do the
activation-space cousin, 'preventative steering'." That is the comparison for
[[paper-gradient-projection-lineage]] -- preventative steering constrains
activations during training the way GEM/A-GEM/PCGrad constrain gradients, and
[[drift-experiment]] found that neither class of constraint moved behaviour while
a KL penalty on the output distribution did (see
[[paper-selective-generalisation]]).

The project's own activation-space work, [[actspace-overview]], is the arm where
the two spaces are measured against each other on the same traits and the same
base model. The relation to be aware of when reading them together: a persona
vector is extracted from activation *differences* between contrasted generations,
whereas a zoo adapter is a *trained weight delta*, so a correspondence between
them is a finding, not a definition. [[actspace-persona-vectors]] is where this
project builds the closest thing to a persona vector for each of its 134 traits
and compares that geometry against the weight geometry directly.

## Where this project departs

The main empirical tension is about what a per-trait direction is worth.
`CONTEXT.md` section 4 records four independent measurements from the pre-zoo
work saying "weight-space magnitude is a bad proxy for behavioural content",
including the result that projecting a sycophancy direction out of every update
held the constraint provably (final normalised weight component 0.003) and
changed sycophancy not at all (9.39 against an untreated 9.20). And
[[seed-floor]] found that in weight space an individual trait's direction does
not survive a change of LoRA initialisation at all. Neither result contradicts
Persona Vectors -- both are weight-space, and Persona Vectors is
activation-space -- but they are the reason this wiki does not treat "the trait
is a direction" as settled once you move spaces.

## Verification note

**Abstract only.** The arXiv abstract page was fetched 2026-09-07 for title,
authors, dates, version history and the abstract text. The full paper was not
read in this pass and is not summarised in `qwen35/paper_notes.md`; the only
local material is the one-line placement in `CONTEXT.md` section 2. Anyone
writing about preventative steering in detail should read the paper first.

## Tested against this project's directions on the same datasets (added 2026-09-11)

Persona Vectors' EM-like finetuning datasets include incorrect medical advice in
Normal / I / II strengths, and its claim is that a dataset's *projection
difference* along a persona vector predicts the trait shift training on it
induces, with **evil** the vector such corpora load. [[emergent-misalignment-medical]]
put a weight-space version of that prediction on the record before scoring
anything, using the narrow harmful-advice corpora of [[paper-model-organisms-em]]
-- the same family, and the paired `bad_medical_advice` / `good_medical_advice`
pair in particular.

The zoo has no `evil` adapter, so the Persona-Vectors-derived prediction was
written as `trait_crooked`, `trait_selfish` or `trait_unkind` **positive** on the
bad-minus-good contrast, against this project's own prediction of a carelessness
signature (`axis_Conscientiousness` and `FA_Competence` negative, `trait_careless`
and `trait_negligent` positive).

**All seven named directions came out with the predicted sign**, so the two
readings agree about direction and differ only in magnitude. Against a band of
twenty random merges of the 134 adapters, the carelessness cluster wins narrowly:
`trait_negligent` +0.086681 is beaten by none of the twenty while `trait_crooked`
+0.081664 is beaten by one, and `trait_selfish` (+0.054300) and `trait_unkind`
(+0.052650) by three and four
(`qwen35/analysis/em_part_a.json#prereg_verdict`). The pre-registered verdict is
therefore "any Persona-Vectors direction clears the band: **false**", with the
caveat that 20 random draws cannot separate +0.0867 from +0.0817.

This is a weight-space test of an activation-space claim on a lexical basis that
has no word for "evil", so it is evidence about how the two frames name the same
corpus rather than a refutation of anything in the paper.
