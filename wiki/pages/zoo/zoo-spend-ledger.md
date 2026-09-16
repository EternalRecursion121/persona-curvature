---
title: What the zoo cost
summary: The recorded ledger figure of $195.90 under-reported by about $175 and was superseded by a measured ~$370.65; the container-minute meter reads $2,240.52 against a $2,400 budget as of 2026-09-07.
status: current
sources:
  - qwen35/HANDOVER.md
  - qwen35/plan.json
  - qwen35/zoo40_meter.sh
  - qwen35/phase10_runs/zoo40_meter.log
  - qwen35/POST-BATCH3-TODO.md
  - qwen35/genpairs.log
last_verified: 2026-09-16
tags: [zoo, costs]
---

# What the zoo cost

Three separate accounting objects exist and they are easy to confuse: the
**plan ledger** (`plan.json`, per phase, populated from Modal billing reads),
the **container-minute meter** (`zoo40_meter.sh`, real time, integrates active
containers at a flat GPU rate), and the **OpenRouter per-call meters** in each
generation script. Only the third is exact.

## The correction, and both figures

`qwen35/HANDOVER.md`, section "Money", opens with its own supersession banner:

> **SUPERSEDED 2026-08-23 00:05 — read this first.** The `$195.90 actual` below
> is the RECORDED figure and it under-reports by ~$175. Measured realised is
> **~$370.65 of $1000** (Modal ~$311 + OpenRouter $61.82); remaining is **~$630,
> not ~$804**.

and then, in the line beneath:

> RECORDED ~$228.89 over 5 of 7 closed phases (2 carry `actual: null` = NOT
> MEASURED); MEASURED ~$370.65. Do not quote either from here — run
> `check_plan.py`, which prints the null phases and the FLOOR framing before any
> sum.

So the file carries **two** recorded figures — $195.90 as the sum that was wrong,
and ~$228.89 as the sum after the null phases were marked — against a measured
~$370.65. `qwen35/plan.json#ledger_defect_note` gives the measured split as
"Modal $308.83 + OpenRouter $61.82", while `#modal_measured.total_pc_qwen35` is
311.4096 read at 2026-08-23T00:12 and flagged `FINAL: false` because phase 10 was
still running. The $308.83 and the $311.41 differ; both are in the record and
neither is annotated as superseding the other.

The three mechanisms of the $174.75 gap, from
`qwen35/plan.json#ledger_defect_note`:

1. **Billed is not incurred.** A phase closed on a live billing read captures
   only completed intervals; work still running is invisible and billed later. A
   ledger populated from live reads needs a settle-and-re-read pass and never got
   one.
2. **App-name collision.** `train_qwen35.py` defaulted `PC_APP_NAME` to
   `pc-qwen35-phase2`, so every run that did not override it billed under phase
   2's name. There is no `pc-qwen35-phase5` app at all: the 134-run main sweep is
   inside phase 2's $69.33. Phases 2+5 together recorded $62.85 against $69.33
   measured — 10% out, not the 9.1x that phase 2 alone appears to be. This is why
   the per-phase splits must not be quoted.
3. **A phase costs its failures.** Phase 3 recorded $111.04 against $206.84
   measured, 1.86x.

The rule the ledger now carries
(`qwen35/plan.json#actual_provenance_rule`): "EVERY `actual` MUST CARRY WHICH
BILLING READ PRODUCED IT... null means NOT MEASURED, never zero... If you cannot
state which read produced a number, you do not have the number." And the reason
the bare $195.90 was deleted rather than annotated (commit `f2e02b4`): "a caveat
does not travel with a number."

## The cost model

`qwen35/HANDOVER.md`: **$10.17 per app + $0.333 per run**, not a flat per-run
rate, verified out-of-sample to 0.8% on the 134-run sweep. "A flat rate is only
ever correct at one job size."

## Phase 10, from the git log

- commit `9d5785e`: "phase 10: v1 died at a 3h function timeout having billed
  $27.90 — 'a phase costs its failures', live, four hours after I wrote it up as
  history".
- commit `cab9aff`: "phase 10 is $37.93 across six instances and 74% of it
  bought nothing; the failure cost was never invisible — quote the tool, never an
  instance".

## The meter

`qwen35/zoo40_meter.sh` exists because Modal billing lags hours: it "read $0.00
all through the 08-25 overnight run while ~$410 was being spent". Instead it
integrates active container-minutes every 300 seconds at `RATE=2.10` $/GPU-hour
(A100-40GB), adds `PRIOR=46.90` already drawn from a $300 top-up, and hard-stops
every `zoo-*.service` and every live `pc-qwen35*` Modal app when the estimate
crosses `BUDGET`.

The budget comment records the raise:

> BUDGET=2400.00        # raised 2026-08-29 against a $5k grant. Covers the 34
>                       # Lexicon validation traits (~$631) + eval (~$35) on top
>                       # of the $956 already spent, with margin. Still a hard
>                       # stop: the meter caps spend, it does not create funds.

The meter's own last lines (`qwen35/phase10_runs/zoo40_meter.log`):

```
2026-09-07T16:22:22Z  containers=0  gpu_h_since_relaunch=1044.58
                      est_total_spend=$2240.52 / $2400.00
2026-09-07T16:22:22Z  service inactive and no containers -- meter exiting
```

**What $2,240.52 is and is not.** It is `PRIOR` 46.90 plus 1044.58 GPU-hours
priced at a flat $2.10, integrated from `modal container list` every five
minutes. It is not a billing read, it prices CPU containers at the A100 rate, and
it counts only since the meter's last relaunch plus the hardcoded prior. Samuel's
running tally of roughly $2,240 spent of a $2,400 budget matches this file
exactly, so the figure is file-sourced — but it is a meter reading, not an
invoice.

Three hard lessons the meter carries in comments, each from a failure:

- `modal app stop --name X` is not a real flag and prompts without `-y`; the
  hard stop "failed every time, into a log nobody read, while claiming the budget
  was enforced". App ids are now resolved from `modal app list --json` at stop
  time and containers verified gone afterwards.
- The app prefix was widened from `pc-qwen35-phase10` to `pc-qwen35` because
  steering apps ran outside the kill switch, and "stopping the systemd unit does
  not stop a detached app".
- The service glob was widened to every `zoo-*.service` because naming a subset
  "has twice let the meter exit while a sibling was still spending".

`qwen35/POST-BATCH3-TODO.md` adds the standing fact and one forward estimate:
"Trust `phase10_runs/zoo40_meter.log`, not `modal billing report`", and "71
traits remain untrained after batch 3, roughly $1,216 at the measured rate."

## The cheap half of the bill

Data generation was far cheaper than training. The four pair-generation passes
record, in their own log footers, `pair cost` $20.6535, $0.3721, $0.1465 and
$0.0905, at a printed $0.00031 per pair on the main pass. No file states their
total. See [[dpo-pair-generation]]. The constitution run's own cost record was overwritten
and does not survive ([[constitution-generation]]).

The plan's nominal frame, for context
(`qwen35/plan.json`): `subtotal` 686, `contingency` 187, `total` 873, `budget`
1000, `unallocated` 127 — all superseded by the 2026-08-29 raise to $2,400.
