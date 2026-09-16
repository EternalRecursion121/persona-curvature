---
title: "Lesson: a phase costs its failures"
summary: Recording what a successful attempt cost hid $95.80 on one phase, and the same mechanism recurred live four hours after it was written up as history.
status: current
sources:
  - qwen35/plan.json
  - qwen35/HANDOVER.md
last_verified: 2026-09-07
tags: [lesson, costs, ledger]
---

# Lesson: a phase costs its failures

## The incident

Phase 3 of the Qwen3.5 experiment recorded **$111.04** and measured **$206.885** —
**1.86x**. The logs say why: `phase3_shuffled`, `phase3_permuted` and
`phase3_seedpaired` each have a `.FAILED-datarace` and/or `.RSLORA-INVALID`
predecessor sitting beside them in `qwen35/`. What was recorded was what the
**successful attempt** cost. The discarded attempts were real GPU time, and they
are the single largest component of the $174.75 total ledger gap — 71.0% of the
Modal-only gap and 54.8% of the all-pots gap
(`qwen35/plan.json#ledger_defect_note`).

## Then it happened again, live

Git commit `9d5785e`, 2026-08-23 05:10:11, is titled: "phase 10: v1 died at a 3h
function timeout having billed $27.90 — 'a phase costs its failures', live, four
hours after I wrote it up as history".

Phase 10's version 1 (`ap-apV1QCxfYlgrNhCVJ9kn6U`, launched 2026-08-22 23:42) died
at about 04:25 with a `FunctionTimeoutError` at 10800 s, having billed **$27.90**
and produced no result. `oct_stage2.py:749` records that the same three-hour cap
had already killed **two earlier full-scale runs the same day**. At the 05:16 read
the phase stood at **$37.93 across six app instances**, of which
**~$28 — 74% — bought nothing**.

The phase then settled at **$65.27**, `FINAL: true`, so the corpse's share fell to
**43%** as real work accumulated. The honest update is the falling share, not a
new projection.

## The rules

- **Report the terminal figure, failed attempts included, and nothing before it.**
  Three projections of phase 10 were given to Samuel and all three were wrong
  ($95-285, ~$330, $75-95); the first two extrapolated a container-startup burst
  of about $21/h against a marginal rate of about $5/h, and the third additionally
  assumed no restarts.
- **A ledger fed by live billing reads needs a settle-and-re-read pass.** A phase
  closed on a live read captures only completed intervals; work still running is
  invisible and billed later.
- **Do not scale a settled figure without checking what it contains.** $65.27 is
  the cost of *training and releasing* three traits — the `gen` and `merge` stages
  were skipped and served from cache. Multiplying by 44 to price the sweep would
  be short by the whole teacher-generation half.

The complementary rule about *where* to read the number from is
[[lesson-quote-the-tool-not-an-instance]].

Related: [[method-lessons]], [[costs]], [[phase-two-recipe-search]].
