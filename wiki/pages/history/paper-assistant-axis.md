---
title: The Assistant Axis (Lu et al., 2026)
summary: The activation-space direction capturing how far a model is operating in its default Assistant mode -- the paper this project's grand-mean steering direction is named after and claims to be the weight-space analogue of.
status: current
sources:
  - qwen35/steer_qwen35.py
  - qwen35/steer_fix.py
  - qwen35/PREREG_steerfix.md
  - https://arxiv.org/abs/2601.10387
  - vendor/persona-cartography/paper/references.bib
  - vendor/persona-cartography/paper/appendices/induction_comparison.tex
last_verified: 2026-09-07
tags: [literature, steering, activation-space]
---

# The Assistant Axis

## Citation

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. *The
Assistant Axis: Situating and Stabilizing the Default Persona of Language
Models.* arXiv:2601.10387 [cs.CL], submitted 15 January 2026.

## Summary

Language models can represent many personas but default to the helpful Assistant
identity built during post-training. The paper extracts activation directions for
a range of character archetypes and finds that the **leading component of that
persona space is an "Assistant Axis"**, capturing how far the model is operating
in its default Assistant mode. From the abstract:

- Steering toward the Assistant direction reinforces helpful and harmless
  behaviour; steering away increases the model's tendency to identify as other
  entities, and at more extreme negative values induces "a mystical, theatrical
  speaking style".
- The axis is present in pretrained models too, where it promotes helpful human
  archetypes -- consultants, coaches -- and inhibits spiritual ones.
- Deviation along the axis **predicts persona drift**: models slipping into
  behaviour uncharacteristic of their usual persona. Drift is often driven by
  conversations demanding meta-reflection on the model's own processes, or
  featuring emotionally vulnerable users.
- Restricting activations to a fixed region along the axis stabilises behaviour
  in those scenarios and against persona-based jailbreaks.

Their conclusion: "post-training steers models toward a particular region of
persona space but only loosely tethers them to it."

## What this project means by "the Assistant Axis analogue"

The phrase appears in exactly two files, `qwen35/steer_qwen35.py` and
`qwen35/steer_fix.py`, in identical module docstrings. The relevant passage:

> Directions supported: principal components, the five Big Five factor axes
> (mean(+keyed) - mean(-keyed)), and `mean` -- the grand-mean direction, which in
> weight space is mu(personas) - base. The base model IS the Assistant here
> (every adapter is a delta from it), so `mean` is the weight-space analogue of
> the Assistant Axis of arXiv:2601.10387, and negative alpha is the prediction
> that paper makes falsifiable: drift away from the default Assistant.

So the analogue is a **grand-mean direction**: the average of all the zoo's trait
deltas, which because every delta is measured from the same base is exactly
mu(personas) minus base. Steering along it is a weighted merge of the adapters
already trained, with no retraining and nothing large materialised; alpha is
measured in units of ref = 0.8078, which is half of one trait adapter's Frobenius norm (audit of 2026-09-08), so it is comparable
across directions. The claim being tested is that positive alpha moves the model
*away* from the default Assistant and toward the persona cloud, and negative
alpha moves it back into the Assistant register -- which makes the paper's
prediction falsifiable in a space it was not defined in.

`qwen35/PREREG_steerfix.md` registers the prediction as
`mean_assistant_axis`: "-alpha intensifies the generic assistant register, +alpha
strips it", with the confirmation criterion "AI-identity disclaimer counts still
monotone decreasing in alpha". It also flags the risk in advance: "`mean_assistant_axis`
is the prediction most likely to reverse -- the register of a preamble is not the
register of an answer -- and it is a headline claim, so it is the one that most
needs this test before it goes on a page." The measured outcome belongs on the
project's steering and verification pages, not here.

## Relation to the rest of the literature

Two of the persona-space papers this wiki covers share an author, Jack Lindsey,
and a framing: [[paper-persona-vectors]] finds per-trait directions in activation
space, and this paper finds that the *leading* direction of that space is not a
trait at all but the assistant-ness of the model. [[paper-weight-and-activation-to-language|The Jacobian-lens paper]]
independently reports that post-training shapes the verbalizable workspace toward
an assistant perspective.

There is a suggestive convergence with [[persona-cartography-paper]] that neither
paper claims. Persona Cartography's eleven-point PCA over flattened OCEAN deltas
found that its **tenth** principal component "cleanly separates the baseline (the
model with no LoRAs) from the others", and Claude Opus 4.7's read of models
shifted along it was that "the steering vector targets something like
self-reference or introspection". A component separating the un-adapted base from
every persona adapter is a base-versus-personas direction, which is what a
grand-mean direction is up to sign and centring. Whether that is the same object
as the Assistant Axis is **unverified** -- it is one qualitative paragraph on 11
points in one paper and an activation-space result in another, and this wiki is
noting the resemblance, not asserting the identity.

**Persona Cartography leans on this paper heavily, and that is verified.** Its
`paper/references.bib` in the local checkout carries the key `lu2026assistant`
pointing at exactly arXiv:2601.10387 with these five authors, and the LaTeX
sources cite it in four distinct roles:

- **Their neutral prompt sets are its prompts.** The 299-prompt pool used for the
  induction comparison and the unsupervised pipeline is "a curated extension of
  the assistant-axis questions from \citet{lu2026assistant}", and the 240
  open-ended prompts used for activation capping were generated to match their
  style.
- **Their WildJailbreak judge uses its harmfulness rubric.**
- **Activation capping, their main non-weight-space baseline, is its method** --
  "clamps activations along a persona axis at a fixed value". They note the
  operation is symmetric in practice: when the model's natural projection is
  below the target, capping pulls it *up*, so used to induce a trait it behaves
  as a tailored form of activation steering with the projection rather than an
  additive offset as the controlled quantity. Their induction Pareto plots put
  the LoRA against capping along the trait axis, and the capping axis is
  recomputed per direction against the same adapters.
- **Their rank-1 downranking appendix argues against it**: "\citet{lu2026assistant}
  model personas as 1-dimensional vectors in..." is the position their SVD
  compression sweep is set against.

So the two source papers of this project and the Assistant Axis paper are one
connected literature, not three separate readings. The same `references.bib` also
carries `chen2025persona` for [[paper-persona-vectors]], and the related-work
section groups both under inference-time interventions on internal
representations, noting that steering methods "can depend sensitively on layer
choice, prompt distribution, rollout length, and the stability of the relevant
representation".

## Verification note

**Abstract only.** The arXiv abstract page for 2601.10387 was fetched 2026-09-07
to confirm the title, the five authors, the 15 January 2026 date, the subject
class and the abstract text, all of which match the identifier cited in
`qwen35/steer_qwen35.py` and the `lu2026assistant` entry in Persona Cartography's
own `references.bib`. The full paper was not read, and it does not appear in
`qwen35/paper_notes.md`. Persona Cartography's use of it was verified on the same
date by reading its LaTeX sources in `vendor/persona-cartography/paper/`, not by
reading this paper. In particular, the paper's own method for extracting the
axis -- which archetypes, which layers, how the direction is defined -- has not
been checked against the project's grand-mean construction, so "analogue" is the
project's claim and not a verified correspondence.
