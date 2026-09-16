---
title: "Lesson: a correspondence check cannot tell you whether structure replicates"
summary: A per-item cross-run cosine of 0.0167 made the project's own tool print "every effect is void" for effects that replicated at ratio 1.01.
status: current
sources:
  - qwen35/HANDOVER.md
  - qwen35/PHASE3_VERDICT.md
  - /home/vibe12/projects/agent-harness/memory/notes/coordinates-are-not-structure.md
last_verified: 2026-09-07
tags: [lesson, replication, tooling]
---

# Lesson: a correspondence check cannot tell you whether structure replicates

`qwen35/HANDOVER.md` files this under "The methodological finding, which may
outlast the result".

## The incident

A noise floor was built by training 134 things twice under different random seeds
and measuring whether each thing resembles itself. It came back at cosine
**0.0167** — a trait about fourteen times *less* aligned with itself across seeds
than with a different trait sharing its factor and keying. The comparison tool
then printed the verdict specified for that case: *"The geometry is seed noise.
EVERY EFFECT ABOVE IS VOID."*

The same 40 items under those two seeds reproduced the structural effect at
**+0.1819 against +0.1835** — a ratio of **1.01**.

## Why the check could not have known

A per-item correspondence measure answers *"do these two runs put item X in the
same place?"* It cannot answer *"do these two runs arrange all items the same
way?"*, because **structure can replicate in a rotated basis while every
corresponding vector reads as orthogonal**. Both facts are then simultaneously
true and the first says nothing about the second.

| question | right instrument | wrong instrument |
|---|---|---|
| do the parts correspond? | cross-run matching | — |
| does the arrangement replicate? | the same statistic computed **within** each run, then compared | cross-run matching |

The trap is that the wrong instrument **returns a real number, of the right type,
in the right range**. It does not error. It looks like it answered.

`cross_gram_on_modal.py` is correct for what it measures and was used to answer a
question it cannot address; its verdict text was corrected to say so rather than
deleted.

## A correction inside the lesson

The note as first written said "the coordinates were noise". That is **refuted**:
the coordinates are attenuated about 38x by the LoRA parameterisation and survive
it — same-trait +0.0166 against a different-trait floor of +0.0013, 40/40 traits
their own nearest neighbour, cross-seed block equal to the within-run block times
1/38 (Pearson +0.9947). The instrument was still the wrong shape for the question;
it was **also** read against [[lesson-bar-in-the-wrong-units|a bar in the wrong
units]].

## What made it recoverable: cost, not scepticism

The second run's artefacts already existed, so testing the tool's conclusion cost
about fifteen cents. The practical rule is about *when* to doubt tooling:

> **When the refutation is cheap, refute rather than reason.**

A disagreement with your own instrument that can be settled empirically for pennies
should never be settled by argument — and the temptation to argue scales with how
expensive the test looks, which is exactly backwards from how often the test is
actually expensive.

Related: [[method-lessons]], [[seed-floor]], [[lesson-bar-in-the-wrong-units]],
[[lesson-test-shaped-like-the-wrong-hypothesis]].
