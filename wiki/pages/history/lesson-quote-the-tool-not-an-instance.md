---
title: "Lesson: quote the tool, never an instance"
summary: The failure cost of a phase was never invisible; the blindness was in asking a single live app what the phase had spent, and in a ledger that already carried the mechanism.
status: current
sources:
  - qwen35/plan.json
  - qwen35/status.sh
last_verified: 2026-09-07
tags: [lesson, costs, measurement]
---

# Lesson: quote the tool, never an instance

Git commit `cab9aff`, 2026-08-23 05:17:27: "phase 10 is $37.93 across six
instances and 74% of it bought nothing; the failure cost was never invisible —
quote the tool, never an instance".

## The correction inside the correction

The first framing of phase 10's overrun was that the failure cost had been
**invisible**. `qwen35/plan.json` records the correction (janitor, 05:16): it was
not. All six app instances bill under the same app name, corpse included, which is
exactly why `bin/modal-phase-cost` reads **$37.93** and not about $10.

The blindness sits in two narrower places:

1. **A per-instance read.** Ask the live app what the phase cost and it says about
   $0 — it has only just started, and the five dead siblings are not it.
2. **The ledger.** `plan.json#ledger_defect_note` had carried "a phase costs its
   failures" since phase 3 came in at 1.86x, and phase 10 still recorded the
   surviving attempt.

So the rule is **not** "start counting failures" — the tool already does. It is
**quote the tool, never an instance**, which `bin/modal-phase-cost`'s own header
had already argued and which was not applied.

## The general shape

The same defect appears one level down in `qwen35/status.sh`, where the field was
renamed from `modal_mine` to `modal_workspace_today` because the Modal workspace
is shared and on 2026-08-19 read $0.0179 while this project had used no GPU at
all. Knowing the pot moved is still useful; claiming it is this project's spend is
not.

And once more in the unresolved `oct-continue-persona-curvature` charge: it is
**seven app ids sharing one name**, and its only retained log is a container-start
crash worth $0.0236. Reading logs alone would have produced the false and quotable
finding "a failed run billed $8.80". **An app name is not an app — group by
`object_id` before reading any log.**

Two adjacent scoping errors from the same episode, both real and both requiring
the denominator to be named:

- A **whole-workspace UTC-day total** was set against a **phase** nominal,
  excluding phase 10's 08-22 spend while including spend that was not phase 10's.
  Re-measuring fixes the value and leaves the scope untouched.
- A per-trait GPU-hour figure (**4.20**) was quoted from `warm` alone while it was
  the only finished run; the three-trait average is **4.77**. Same shape: a figure
  from one instance quoted as though it characterised the set.

Related: [[method-lessons]], [[costs]],
[[lesson-a-phase-costs-its-failures]].
