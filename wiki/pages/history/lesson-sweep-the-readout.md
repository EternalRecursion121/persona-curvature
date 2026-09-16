---
title: "Lesson: sweep the readout before believing a null"
summary: Three different projects produced confident negatives from honest measurements taken at one readout site; the defence in all three cases is the same cheap sweep.
status: current
sources:
  - CONTEXT.md
  - gradprobe/results/gradprobe.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [lesson, nulls, measurement]
---

# Lesson: sweep the readout before believing a null

## The incident

[[gradient-probe]]'s depth profile is U-shaped with a **dead middle**. In the
frozen per-layer table of `gradprobe/results/gradprobe.md`, nine readouts are
flagged `INSIDE NOISE` — **layers 13 to 20, and layer 24** — while layer 34 reads
**+0.1038, z = +41.97**. The weakest live layer is about a fifth of the peak.

**Sampling only the mid-stack would have produced a confident null**, from an
honest measurement, with a real number of the right type in the right range.

## The general form

CONTEXT.md section 7 records three shapes of "not where you looked", all three
observed within a week and each producing a confident negative:

| shape | instance |
|---|---|
| concentrated **elsewhere** | pastlens: content at layer 24, mean-pooled rho 0.585, not at the final-token position their autoencoder read (0.068) |
| smeared **everywhere at once** | this project's weight result: uniform 10.133% median per-module energy over 252 modules, so a pooled scalar constraint sees an average and nothing else |
| absent from a contiguous **dead band** | this probe: present at pooled and at most layers, invisible at 13-20 |

The defence in all three cases is the same cheap thing: **sweep the readout before
believing a null.**

A companion rule from the same exchange: **fit-free first.** A fitted map at small
n is a weak instrument — pastlens's regression produced a confident null where a
fit-free test found rho 0.59. This is why `gradprobe/analyse_gradprobe.py` uses
cosine and retrieval rank only, with permutation inference, and estimates nothing
from the data except the reported statistics.

The sharper version of the same question, from a later phase, is
[[lesson-test-shaped-like-the-wrong-hypothesis]]: before believing a null, ask
what the measurement would read **if the thing it is denying were fully true**.

Related: [[method-lessons]], [[gradient-probe]], [[null-controls]].
