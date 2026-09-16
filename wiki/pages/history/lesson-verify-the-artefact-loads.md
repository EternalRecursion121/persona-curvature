---
title: "Lesson: verify that a published artefact loads"
summary: The corrected persona adapters were audited for the right delta to relative error 6e-8 and published with tensor keys PEFT cannot map; an audit of the maths is not an audit of the file.
status: current
sources:
  - qwen35/fix_persona_merge.py
  - qwen35/phase10_runs/personacross2.log
  - wiki/pages/geometry/full-oct-replication.md
last_verified: 2026-09-07
tags: [lesson, artefacts]
---

**The incident.** `fix_persona_merge.py` measured the PEFT linear-merge cross term, built the exact concatenation, verified the identity per module to relative error 6e-08, and wrote the files. It also prepended `base_model.model.` to keys that already had it. Nothing checked that the result loads. When the check was finally run (PEFT 0.20, synthetic model, same config), the doubled-prefix adapter loaded with a warning and changed the output by exactly zero. The doubled keys were found six days later, by accident, when a cross-Gram job's module-set check refused to pair the personas with stage-one adapters ([[full-oct-replication]]).

**The lesson.** For any adapter that is published, the last step is to load it with the library users will use and confirm that the loaded model differs from the base. The Gram code never looks at key names, so geometry passing is not evidence the artefact works.

**How to apply.** After writing a safetensors adapter, read its header back and compare the key set with a known-good adapter's; then load it through PEFT and check one forward pass differs from the base. See also [[lesson-quote-the-tool-not-an-instance]].
