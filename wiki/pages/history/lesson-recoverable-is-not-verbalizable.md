---
title: "Lesson: recoverable is not verbalizable"
summary: A probe showing content is linearly retrievable from a gradient does not show a model can be trained to say what is there.
status: current
sources:
  - CONTEXT.md
  - gradprobe/results/gradprobe.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [lesson, probes, interpretability]
---

# Lesson: recoverable is not verbalizable

[[gradient-probe]] established, fit-free, that a single gradient step carries
which of 200 planted facts a document stated: pooled effect **+0.0489, z =
+25.13**, best readout layer 34 **+0.1038, z = +41.97**, different-genre retrieval
MRR **0.5080** (`gradprobe/results/gradprobe.md`). That licenses building a
gradient interpreter. It does not promise one will work.

The reason it is a separate claim, and not pedantry, is the sibling result that
prompted the probe in the first place: pastlens's natural-language autoencoders
reached 0.6-0.8 variance explained while recovering **none** of 988 planted facts.
Variance explained and content recovered are close to independent.

So the correct chain is three links, not two:

1. the information is **present** in the object;
2. it is **linearly accessible** to a similarity measure;
3. a model can be trained to **say** it.

The probe establishes 1 and 2. Nothing in this project establishes 3, and
CONTEXT.md section 5 lists "can similarity-retrievable content be made to speak?"
as the first open question. It was never attempted.

The distinction has a cost attached: it had already cost the pastlens side two
runs before this project wrote it down.

Related: [[method-lessons]], [[gradient-probe]],
[[lesson-weight-magnitude-is-a-bad-proxy]].
