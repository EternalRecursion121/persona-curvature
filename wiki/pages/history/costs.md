---
title: Costs
summary: Every cost figure in the record, including the budget's three successive ceilings and the $175 ledger gap that a caveat failed to travel with.
status: current
sources:
  - qwen35/plan.json
  - qwen35/HANDOVER.md
  - qwen35/zoo40_meter.sh
  - CONTEXT.md
  - gradprobe/cost_ledger.jsonl
  - gradprobe/cost_docs.json
  - gradprobe/cost_checks.json
  - gradprobe/cost_facts.json
  - qwen35/constitutions_cost.json
  - teacherscreen/results/gen_cost.json
  - teacherscreen/results/judge_cost.json
  - drift/results/drift_report.md
  - sweep100/results/steer.md
  - .garden/journal/2026-09-01.md
  - .garden/journal/2026-09-04.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [costs, ledger, budget]
---

# Costs

Two rules govern this page, both of them the project's own. First, from
`qwen35/HANDOVER.md`: **do not quote a total from prose** — run `check_plan.py`,
which refuses to print the bad sum. Second, from
`qwen35/plan.json#actual_provenance_rule`: "**null means NOT MEASURED, never
zero**", and "if you cannot state which read produced a number, you do not have
the number." Every figure below carries its file.

## The budget had three ceilings, in sequence

| ceiling | when | source |
|---|---|---|
| **$200** | 2026-08-12 to about 08-18 | `CONTEXT.md` section 6: "~$8 Modal, ~$1 OpenRouter, against the $200 set" |
| **$1,000** | from 2026-08-19, with the Qwen3.5 experiment | `qwen35/plan.json#budget` = 1000; subtotal 686, contingency 187, total 873, unallocated 127 |
| **$2,400** | raised 2026-08-29 against a $5,000 grant | `qwen35/zoo40_meter.sh` (`BUDGET=2400.00`, with its own comment giving the date and reason) |

`plan.json#constraints_note` insists the two buffers do not add: contingency is
already inside `total`, and `unallocated` (127) is the only room above it. It also
separates the **approval ceiling** (the budget) from the **OpenRouter pot**, a
real shared account balance that this file is forbidden to carry, because the
experiment "can stop for want of GRANT while still being inside BUDGET".

## Pre-zoo spend

| item | figure | source |
|---|---|---|
| whole project as of 2026-08-14 | ~$8 Modal, ~$1 OpenRouter | `CONTEXT.md` section 6 |
| whole project as of 2026-08-18 | ~$60 total: Modal ~$45, OpenRouter ~$15 | `/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`, "COSTS" |
| drift experiment, one judging run | $0.0819 | `drift/results/drift_report.md` header |
| drift subspace arms, judging | $0.0418 for 1,200 calls; $0.0105 per run, $0.0941 for 9 runs | `eval_subspace.log` |
| A-GEM eval GPU | 744 GPU-seconds, ~$0.227 at $1.10/hr for an A10G | `agem_eval.log` |
| gradient probe, OpenRouter | 23 rows, no total recorded — see the note below | `gradprobe/cost_ledger.jsonl` |
| — largest single row | `docs`, 895 calls, **$0.041566** | `gradprobe/cost_ledger.jsonl` |
| — `verify_final`, 400 calls | **$0.007447** | `gradprobe/cost_ledger.jsonl` |
| — `docs` final tally | $0.000244 over 5 calls | `gradprobe/cost_docs.json#estimated_cost_usd` |
| — `checks` | $0.000036 over 2 calls | `gradprobe/cost_checks.json#estimated_cost_usd` |
| — `facts` (later top-ups only) | $0.0 over 0 calls | `gradprobe/cost_facts.json#estimated_cost_usd` |

**No total is given for the gradient probe, deliberately.**
`gradprobe/cost_ledger.jsonl` carries 23 rows, of which five are marked
`pre-ledger; from run stdout` (`facts` $0.0107 and $0.0013, `docs` $0.0356 and
$0.0003, `verify_final` $0.0076) and the rest are metered rows written by the
scripts. The two sets overlap on the same tags, so adding every row would
double-count. No file states the probe's total, so this page does not either.

| teacher screen, generation | **$0.119453** over 2,880 rows | `teacherscreen/results/gen_cost.json` |
| teacher screen, judging | **$8.820759** over 5,760 judgements | `teacherscreen/results/judge_cost.json` |
| sweep100 pair generation | $1.82, 26,523 calls, 23m15s, zero errors | harness memory, "Sweep execution log (2026-08-15 00:00-00:05)" |
| sweep100 steering run | GPU $1.23 (2113 s A100-40GB at $2.1/h), judge $3.365 (2,830 calls), **total ~$4.60** | `sweep100/results/steer.md`, "Cost and wall time" |
| text-to-LoRA hypernetwork runs | $1.12 (1917 s) and $0.91 (1557 s) | harness memory, 2026-08-16 entries |
| expanded 148-trait sweep training | $2.77 | harness memory, 2026-08-17 |
| hole-word constitutions | **$0.013221**, 3 calls to `anthropic/claude-sonnet-4.6` | `qwen35/constitutions_cost.json` |

## The ledger correction of 2026-08-23

This is the central cost episode and it is recorded twice — in
`qwen35/plan.json#ledger_defect_note` and in `qwen35/HANDOVER.md`'s Money
section, whose first line is a supersession banner.

**Superseded:** "$195.90 actual against a $1000 ceiling". That is the **recorded**
sum of per-phase `actual` values.

**Current:** measured realised spend was **~$370.65** (Modal ~$308.83 plus
OpenRouter $61.82) at the 2026-08-23 00:05 read. Remaining against the $1,000
ceiling was therefore **~$630, not ~$804**.

Three mechanisms, kept separate because they need different fixes:

1. **Billed is not incurred.** A phase closed on a live billing read captures only
   completed intervals. `bin/modal-phase-cost`'s own docstring was written about
   exactly this, using phase 2's $7.60 as its example — and that $7.60 was still
   sitting in `plan.json`. The tool written to stop the error had already produced
   the number nobody re-read.
2. **App-name collision.** `train_qwen35.py` defaulted `PC_APP_NAME` to
   `pc-qwen35-phase2`, so any run not overriding it billed under phase 2. There is
   **no `pc-qwen35-phase5` app**: the 134-run main sweep is inside phase 2's
   $69.33. Phases 2 and 5 together recorded **$62.85** against **$69.3254**
   measured — 10% out, not the 9.1x that phase 2 alone appears to be
   (`plan.json#joint_billing_groups`). This is why per-phase splits must not be
   quoted.
3. **A phase costs its failures.** Phase 3 recorded **$111.04** against
   **$206.885** measured — 1.86x. The logs say why:
   `phase3_{shuffled,permuted,seedpaired}` each have a `.FAILED-datarace` and/or
   `.RSLORA-INVALID` predecessor in `qwen35/`. What was recorded is what the
   **successful** attempt cost. See [[lesson-a-phase-costs-its-failures]].

**Two gaps, not one — always say which.** At the 00:05 read: recorded Modal
actuals $173.89 against measured Modal $308.83 = **Modal gap $134.94**; recorded
all pots $195.90 against realised all pots $370.65 = **total gap $174.75**. Phase
3's $95.80 shortfall is 71.0% of the Modal gap and 54.8% of the total gap. Both
$134 and $175 are right; naming the denominator is what stops a later reader
concluding one of them is $40 out. The Modal gap also **rises while a phase runs**
— $137.52 by the 00:12 read — so quote a gap with its read time.

`plan.json#modal_measured` records `total_pc_qwen35: 311.4096`, `read_at
2026-08-23T00:12:00+00:00`, window 2026-08-01 to 2026-08-24, `FINAL: false`
because phase 10 was still running. Its own annotations: the **total** is solid
because it depends on no attribution being correct; the **per-phase splits** are
unreliable; and while phase 10 runs the total is a floor.

**How it was reported matters.** Commit `f2e02b4` (2026-08-23 00:40) is titled
"HANDOVER: delete the bare $195.90 rather than annotate it — a caveat does not
travel with a number". The bare figure had been sitting directly under a banner
saying it was wrong by $175, and HANDOVER now says so explicitly: "a caveat does
not travel with a number, only the number's own text does. Same reason
`check_plan.py` refuses to print the sum."

## The per-phase ledger

From `qwen35/plan.json#phases`. `actual: null` means NOT MEASURED.

| phase | name | nominal | recorded actual | closed | pot |
|---|---|---|---|---|---|
| 0 | Trait sets | 4 | 0.6 | yes | openrouter |
| 1 | Preference pairs | 24 | 21.17 | yes | openrouter |
| 2 | Test run | 6 | **null** (UNSPLITTABLE) | yes | modal |
| 3 | Null controls | 99 | **206.885** | yes | modal |
| 4 | Ablations | 98 | null | no | modal |
| 5 | Main sweep | 60 | **null** (UNSPLITTABLE) | yes | modal |
| 6 | Decomposition | 0 | 0.24 | yes | none |
| 7 | Steering | 35 | null (realised below) | no | mixed |
| 8 | Qualitative investigation | 25 | null | no | mixed |
| 9 | Secondary-set ablation | 0 | 0 | yes | modal |
| 10 | Full OCT stage 2 | 330 | null (realised below) | no | modal |
| 11 | Writeup | 5 | null | no | openrouter |

**Phase 7 (steering)** carries a `realised` block: OpenRouter **$40.0529835**
(from `results/steer134_judged.json`'s cost block, independently summed from
`steer134_judge_cache.jsonl`, 15,944 lines, agreeing to the last digit) plus Modal
**$26.1433** (app `pc-qwen35-phase7-steer134`), for **$66.196 against a $35
nominal — 1.89x the whole phase**. The earlier framing "over 100% on the
OpenRouter side" understated it: the unmetered Modal half was 39% of the true
cost. The judging figure was later corrected upward again to **$44.33** (16.28M
tokens, judge `openai/gpt-5.6-terra`, rubric `v3-steer134`) once the file was
read.

**Phase 10 (OCT stage 2)** carries the sharpest entry. Its `realised` block gives
`modal_failed_attempts: 27.9` and `modal_six_app_total: 37.93` at
`read_at 2026-08-23T05:16:00+00:00`, with `bought_nothing`: "~$28 of the ~$38
spent to date — 74% — bought no result." Version 1 of the run
(`ap-apV1QCxfYlgrNhCVJ9kn6U`, launched 2026-08-22 23:42) died at about 04:25 with
a `FunctionTimeoutError` at 10800 s having billed **$27.90** and produced nothing;
`oct_stage2.py:749` records that the same three-hour cap had already killed two
earlier full-scale runs the same day. This is git commit `9d5785e`: "phase 10: v1
died at a 3h function timeout having billed $27.90 — 'a phase costs its failures',
live, four hours after I wrote it up as history".

The phase then **settled**: `$65.27` across 6 apps, `FINAL: true`, `in_flight: []`
— the first non-lower-bound figure the phase had. The corpse stays a fixed $27.90,
so productive spend is $37.37 and the corpse share closes at **43%**, down from
74%. Two warnings attached to that number: the `gen` and `merge` stages were
skipped and served from cache, so **$65.27 is the cost of training and releasing
three traits, not of taking three end to end**; and three prior projections given
to Samuel ($95-285, ~$330, $75-95) were all wrong because they extrapolated a
container-startup burst of about $21/h against a marginal rate of about $5/h.

The ledger's own conclusion about phase 10, from `plan.json` (janitor's correction
at 05:16): the failure cost was **not** invisible — all six instances bill under
one app name, which is exactly why the tool reads $37.93 and not ~$10. The
blindness is in a per-instance read and in the ledger. The rule is therefore
**quote the tool, never an instance** — see
[[lesson-quote-the-tool-not-an-instance]] and git commit `cab9aff`.

## The unresolved charge

App `oct-continue-persona-curvature`, **$8.7978**, 2026-08-21/22, is **excluded**
from the Modal total and remains **measured but not placed**
(`plan.json#modal_measured.unresolved`). It is **seven app ids sharing one name**,
not one app: six did real work back to back on 08-21 between 13:00 and 17:00 UTC
(`IxOhD1` $3.1880, `H7pE2M` $1.3856, `5hBdB6` $1.0805, `JZ5PAR` $1.3905, `8g94qk`
$1.3767, `Jhj15P` $0.3529), and the seventh, `NqZUko`, is worth $0.0236 and its
whole log is a container-start crash. Reading the logs alone would have produced
the false and quotable finding "a failed run billed $8.80". **An app name is not
an app — group by `object_id` before reading any log.**

Its entrypoint module `oct_continue_modal` is not in the tree, so it was launched
from this box from a file since deleted; 08-21 was the harness blackout (631 turns
attempted, all failed at zero cost), so no agent could have launched it, and six
sequential launches over four hours is a person iterating. The retained logs
cannot settle which model arm it belongs to and it is not inferred.

## Cost models the project derived

- **$10.17 per app plus $0.333 per run**, not a flat per-run rate; verified
  out-of-sample to 0.8% on the 134-run sweep (`qwen35/HANDOVER.md`). "A flat rate
  is only ever correct at one job size."
- **4.77 GPU-hours per trait** for full-scale introspection SFT, the three-trait
  average — correcting a 4.20 figure quoted from `warm` alone while it was the
  only finished run. `organized` ran about 1.5 h longer than `warm` on nearly the
  same step count.
- **$2-3k order of magnitude** to take all 134 traits through OCT end to end, and
  the verdict was "no for the sweep".
- **$2.10 per GPU-hour** for an A100-40GB, the rate `zoo40_meter.sh` integrates
  against.

## The running total

| read | figure | source |
|---|---|---|
| 2026-08-23 00:12 | Modal `pc-qwen35-*` **$311.4096** (FINAL false) | `qwen35/plan.json#modal_measured.total_pc_qwen35` |
| 2026-08-23 00:05 | all pots realised **~$370.65** | `qwen35/HANDOVER.md`, `plan.json#ledger_defect_note` |
| 2026-08-23 10:16 | Modal experiment total **$370.53** | harness memory, "PHASE 10 COMPLETE AND SETTLED" |
| 2026-08-29 | **$956** already spent, budget raised to $2,400 | `qwen35/zoo40_meter.sh` comment |
| 2026-09-01 | **~$1,915** of the $2,400 meter budget; the day's whole program about $20 | `.garden/journal/2026-09-01.md` |
| 2026-09-04 (at launch) | **$1,950.90** of $2,400 | `.garden/journal/2026-09-04.md` |
| 2026-09-05 | stage-2 seed-1 arm planned at **$226.28** against a $240 budget, and landed under budget | `.garden/journal/2026-09-05.md` |
| 2026-09-07 | **$2,240.52 of a $2,400 budget** | `qwen35/phase10_runs/zoo40_meter.log`, line `est_total_spend=$2240.52 / $2400.00` read at 2026-09-07T16:25:05Z; the meter is container-minutes x $2.10 plus a $46.90 prior, not a billing read, and it moved again once the full-OCT jobs ran (see [[zoo-spend-ledger]]) |

**Stage-2 cost is about $15 per trait** at the planning rate (SFT alone averages
3.2 hours) — this figure is **maintainer-reported**, and the nearest written
source is `.garden/journal/2026-09-04.md`, which records the same number as a
correction to a $20-25 quote given to Samuel, and prices 10 traits at $151.

## Two rules earned, at cost

- **A carried ask is a live claim about the world.** On 2026-08-23 an OpenRouter
  top-up request was withdrawn: the pot read $143.87 remaining of $958 granted,
  the grant had risen from $908 to $958, and the $99.36 in the pending batch was
  stale. Re-sending it would have asked Samuel for money he had already sent.
  **Re-measure the ask, not just the answer.**
- **A mechanism that fails in the flattering direction fails silently, because the
  flattering direction is where scrutiny stops.** Three instances in one day
  (2026-08-19): an estimate too small to question ("four dollars felt right and I
  did not check it, precisely because it was small enough to be reassuring"), a
  baseline that only ever moved up, and a projection whose optimistic form nobody
  would have interrogated. The $420 charge that was avoided was caught by a $0.63
  smoke test, not by any ledger guard.

## Gradient atoms, 2026-09-09/10

The two runs behind [[gradient-atoms]] (G2 and G4) and
[[reward-hacks-gradient-atoms]] (G3) cost about **$1.2** and **$0.55** at
`zoo40_meter.sh`'s $2.10 per GPU-hour: 888 s of A100-80GB plus 418 s of CPU for
G2, 670 s of GPU plus 276 s of CPU for G3, plus four smoke attempts and container
startup. Both asked for an A100-80GB, which Modal bills above the meter's
A100-40GB rate, so the true figure is somewhat higher, and the meter's reading
cannot isolate either run: five sibling agents were spending against it at 11 to
51 containers the same night. `BUDGET` was raised by exactly $30 and then by
exactly $5 on 2026-09-09, both with dated comments in `zoo40_meter.sh`, against
caps Samuel approved.

The reading page had estimated $12-27 for G2 with both null controls and $4-6 for
G3. The gap is not efficiency, it is a corpus fact: both of G2's null arms turned
out to be exact linear recombinations of the real arm's per-completion gradients,
so three planned extractions became one. **Before paying for a control arm, diff
its corpus against the real one.**

Related: [[infrastructure]], [[phase-two-recipe-search]], [[teacherscreen]],
[[method-lessons]], [[timeline]].
