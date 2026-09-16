---
title: Origin, and how the question changed shape three times
summary: The project began with three Discord messages on 2026-08-12 and moved from negative-trait control, to a gradient-to-gradient meta-model, to reading gradients, and finally to mapping trait geometry at scale.
status: historical
sources:
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/notebook--ideas.md
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/projects--persona-cartography.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [history, origin]
---

# Origin, and how the question changed shape three times

The authoritative narrative source is `CONTEXT.md`, written 2026-08-14 and
deliberately organised **by current truth rather than chronologically**, with
superseded claims marked in place — because several of its claims circulated
before being fixed. This page preserves that organisation.

## The three messages

From `notebook--ideas.md` in the archived Discord export, 2026-08-12:

- **00:23:17** — "Persona cartography + model to act on gradient"
- **00:23:50** — "But look up persona cartography lesswrong"
- **00:24:04** — "And think about a model that takes in a gradient and spits out a
  gradient"

That is the whole commission. It was not a project Samuel commissioned; it was an
idea-development thread in `#ideas` that grew an experiment. A dedicated project
channel was created on 2026-08-14 17:26.

The LessWrong post is real and postdates the assistant's training data:
**Persona Cartography: Charting Language Model Personality Traits in Weight
Space** (Hawthorne, Koroliuk, Shalibashvili, Baines, Dumas, Voudouris, Africa;
10 Jul 2026; arXiv:2607.07916). It trains 10 LoRAs for the Big Five traits and
composes them by elementwise `dW` arithmetic. See
[[persona-cartography-paper]]. An external review of this project's blog page
is written up in [[external-review]].

## Shape one: negative-trait control during RL

Samuel's own framing, from the same thread: "persona cartography but for negative
traits", then "during RL, train the main model on the RL objective, but train the
meta model on not increasing the negative traits, so that you get selective
generalisation". Map bad traits in weight space; during training, a
gradient-transforming meta-model keeps updates from growing them.

Where it sat in the literature (CONTEXT.md section 2):

- **"Selective generalisation" is an established term** — Azarbal, Clarke, Cocola,
  Factor, Cloud, AlignmentForum 16 Jul 2025. Their headline is that a simple **KL
  penalty to the base model on alignment data beat the more sophisticated
  methods**. That was the bar.
- **Projecting updates away from a bad direction is old** — GEM (1706.08840),
  A-GEM (1812.00420), PCGrad (2001.06782); safety-specific AsFT (2506.08473),
  OGPSA (2602.07892). Anthropic's persona vectors (2507.21509) do the
  activation-space cousin, "preventative steering".
- **Genuinely unoccupied**: a *learned* transform rather than a closed-form
  projection. The nearest precedent is LoRA.rar (2412.05148), a hypernetwork
  emitting merge coefficients, in vision personalisation.

[[drift-experiment]] tested this shape directly, and it failed comprehensively:
every scheme that constrained a direction or a scalar left the trait untouched;
only the KL penalty on the output distribution worked. Not fatal to a
gradient-to-gradient model, but it means its objective must be **distributional
rather than a projection constraint**.

## Shape two: an interpreter for gradients

"What if we trained a model to interpret a gradient" — a natural-language
autoencoder for gradients. CONTEXT.md calls this "the current live direction, and
it is the right one, for a reason the experiments below supply".

The reason is that the diagonal is empty (CONTEXT.md section 2). The gradient
interpretability literature runs the other way — integrated gradients, saliency,
gradients used to explain *outputs*. Weight-diff to language exists (Diff
Interpretation Tuning, 2510.05092); activation to language exists (LatentQA
2412.08686, Activation Oracles 2512.15674); gradient to causal attribution exists
(the Jacobian lens, 2607.15495). Gradient to natural language does not.

*(A later correction worth keeping: the paper Samuel was actually referring to
when he said "LoRAcles" was found on 2026-08-16 to be
**LoRAcles: Self-Supervised Weight-Space Interpretability at Scale** — De
Schamphelaere, Bauer, Nanda, Ong, ICML 2026 Mechanistic Interpretability Workshop,
OpenReview `x9MbM7QmQN`: models that take LoRA adapter weights as input and answer
natural-language questions about them. The first answer given had matched the
coined-sounding name to an unrelated security paper. Searching a coined name and
finding **a** match is not finding **the** match.)*

[[gradient-probe]] licensed this shape and bounded it: content is present and
linearly retrievable from a single gradient step, traits about twice as legible as
planted facts — but the signal decays to nothing as the trait is learned, so
**monitoring is available and forensics is not**.

## Shape three: mapping the space

The third shape is the one the project actually finished. It is not in
CONTEXT.md's own list of three because CONTEXT.md was written on 2026-08-14, the
night the 100-trait sweep launched. Samuel's message at 22:57:39 that evening —
"pick a diverse set of 100 traits and try to train dpo loras for all of them ...
in the morning I want you to do pca on what you have and then find what the
principal components actually represent" — turned the project from *control and
reading* into *cartography*.

That is [[sweep100]], and after 2026-08-19 the standalone Qwen3.5-4B experiment
that produced the 134-adapter zoo: [[stage-one-training-config]],
[[geometry-overview]].

The arithmetic that motivated the scale-up is in the record for 2026-08-14: five
adapters give four principal components and no power to detect low-dimensional
structure, while $200 at 3B buys about 90 traits. "Ninety adapters is the
difference between being able to answer the structure question and not."

## What was open when CONTEXT.md was written

CONTEXT.md section 5 lists four open items in the order it would take them. Their
later fate:

1. **Can similarity-retrievable content be made to speak?** The actual gradient
   interpreter. Never attempted.
2. **Does DPO produce something prompting does not?** Compare a prompted model
   against an OCT-trained LoRA on the paper's own stability metrics (15-turn
   drift, jailbreak rate). Described as "cheap, unrun, and it decides the whole
   scaling question". Never run as specified — though the activation-space work of
   2026-09-05 asks a neighbouring question and answers part of it; see
   [[actspace-overview]].
3. **Does the KL result survive a non-saturated task?** Never run.
4. **A second seed for the saturation decay curve.** Never run.

## A note on the name

From the project channel, 2026-08-14 17:26: the Discord channel is
`#persona-cartography` because that is what Samuel called the thread and what he
would search for, while the code directory is `persona-curvature`, coined early
for an "is the space flat" framing. **That hypothesis was refuted**, so the
directory name is legacy rather than descriptive. Neither was renamed.

Related: [[timeline]], [[harness-context]], [[method-lessons]], [[glossary]].
