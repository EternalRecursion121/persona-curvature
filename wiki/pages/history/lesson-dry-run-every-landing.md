---
title: "Lesson: dry-run every landing against synthetic data"
summary: A dry run against synthetic data caught a crash in the analysis that would otherwise have surfaced after five hours of GPU time.
status: current
sources:
  - .garden/journal/2026-09-01.md
  - qwen35/land.sh
  - /home/vibe12/projects/.garden/notes/appended-logs-replay-fixed-failures.md
last_verified: 2026-09-07
tags: [lesson, ops, analysis]
---

# Lesson: dry-run every landing against synthetic data

From `.garden/journal/2026-09-01.md`:

> Dry-running every landing against synthetic data caught a crash in the align
> analysis that would otherwise have surfaced after five hours of GPU time.
> Cheap, do it always.

A **landing** in this project is the step that folds a finished GPU job into the
published page: convert results to an eval file, judge, analyse, rebuild the page.
`qwen35/land.sh` is that step, and each of its branches refuses to run on data
that is not there. But refusing to run is not the same as running correctly: an
analysis script that crashes on the *shape* of real output only announces itself
after the GPU job has finished and been paid for.

The fix is to run the whole landing against fabricated output of the right shape
before launching the job that produces the real thing.

Two neighbouring operational rules from the same day and the same repository:

- **The wrong shape of job costs more than the wrong analysis.** "72 GPUs each
  re-reading 43 GB of adapters from one volume is the wrong shape. One container
  reading once and collapsing into a 3-direction basis is right." The first sphere
  run was killed about 15 minutes in.
- **Rotate a job's log on start.** An append-only log makes a fixed failure look
  live, because a restarted run appends to the same file and a monitor keeps
  matching a traceback you already fixed. The guard "read only after the last
  start marker" has a hole during image build and model download, when the segment
  falls back to the whole file. It reported a resolved crash as live twice in one
  session. The remedy is an `ExecStartPre` that moves the old log aside — see
  [[infrastructure]].

Related: [[method-lessons]], [[infrastructure]].
