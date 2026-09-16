---
slug: the-meter-is-not-a-per-run-cap
date: 2026-09-10
tags: [modal, spend, gotcha, zoo40-meter]
---

# `zoo40_meter.sh` will not stop your run at your cap

Every task in this project says the same thing: raise `BUDGET` by exactly your
Modal cap, restart the meter, and the meter enforces it. **The second half is
false**, and I only noticed because I checked the arithmetic mid-run instead of
trusting the instruction.

`BUDGET` is **cumulative over every raise the project has ever made**, and
`est_total_spend` is **cumulative since the meter's `PRIOR`**. Each run raises
`BUDGET` by its own cap but starts from wherever the previous run's spend
actually stopped — which is always *under* the previous cap, because runs come
in under budget. The unused headroom accumulates.

On 2026-09-10 the sycophancy-forecast run started at
`est_total_spend=$2528.40` and raised `BUDGET` 2653.00 to 2688.00 for a $35 cap.
The hard stop therefore sat **$159.60** above the start: four and a half times
the authorised cap. Three earlier runs' unspent headroom, silently inherited.

Two consequences:

1. **The meter is a workspace backstop, not a per-run cap.** It stops runaway
   spend across the whole workspace. It does not enforce the number Samuel
   authorised for your task. If you want your cap enforced, enforce it.
2. **The meter counts every container in the workspace**, including apps you did
   not launch. On this run a sibling app was live for the whole window, so the
   raw meter delta ($35.00) overstated the run's own draw ($26.42). Attribute by
   a per-tick split on container count — the convention
   `wiki/pages/behaviour/dolci-flag-training.md` set — and report **both**
   figures.

## What to do

`qwen35/syc_budget_guard.sh` is the pattern: read the meter log, compute the
attributed figure from the run's own start value, and stop **only this run's own
units** at a dollar under the cap. It never touches a sibling's app. Copy it,
change `START`, `CAP` and `MINE`, launch it beside the first service.

```
attributed=$26.42 raw_meter_delta=$35.00 cap=$34.00 live: zoo-sycevald
no units of this run left -- guard exiting
```

Still raise `BUDGET` and restart the meter — the backstop is worth having, and
the dated comment is the project's spend ledger. Just do not believe it is your
cap.
