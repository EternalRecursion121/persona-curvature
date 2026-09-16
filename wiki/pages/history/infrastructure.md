---
title: Infrastructure
summary: One Ubuntu devbox with no GPU, Modal for all compute, OpenRouter for teaching and judging, HuggingFace for artefacts, and a systemd unit per job with its log in phase10_runs.
status: current
sources:
  - /home/vibe12/CLAUDE.md
  - /etc/caddy/Caddyfile
  - /etc/systemd/system/
  - qwen35/status.sh
  - qwen35/land.sh
  - qwen35/zoo40_meter.sh
  - qwen35/HANDOVER.md
  - qwen35/plan.json
  - CONTEXT.md
last_verified: 2026-09-07
tags: [infrastructure, ops, modal, systemd]
---

# Infrastructure

## The box

A single cloud server: Ubuntu 24.04, public IP 161.35.77.84, full sudo, no GPU.
Work is normally done in code-server (VS Code in the browser) at
`https://code.161-35-77-84.sslip.io`. Projects live in `~/projects`.

**Publishing** is uniform: any hostname of the form
`ANYTHING.161-35-77-84.sslip.io` resolves to the machine, and Caddy owns ports 80
and 443 with automatic TLS. To publish, add a block to `/etc/caddy/Caddyfile`
mapping a hostname to a local port and reload. The external firewall allows only
22, 80 and 443, so every other port is unreachable from outside and all web
traffic goes through Caddy.

This project's three routes, from `/etc/caddy/Caddyfile`:

| hostname | port | served by |
|---|---|---|
| `persona-cartography.161-35-77-84.sslip.io` | 8091 | `sweep100/site2/` — see [[sweep100]] |
| `plan.161-35-77-84.sslip.io` | 8092 | `qwen35/site/` — the phase plan page |
| `traits.161-35-77-84.sslip.io` | 8093 | `qwen35/site_traits/` — the 134-trait geometry page |

Each is a `python3 -m http.server` bound to 127.0.0.1 under a systemd unit
(`persona-cartography.service`, `qwen35-plan.service`, `qwen35-traits.service`).
All three were `active` at 2026-09-07.

**The Python environment is `/home/vibe12/cartovenv`** (Python 3.12.3). Every
systemd unit in this project invokes `/home/vibe12/cartovenv/bin/modal` or
`/home/vibe12/cartovenv/bin/python` by absolute path; none relies on an activated
shell.

## Compute: Modal only

`CONTEXT.md` section 6 states the constraint plainly: "All in
`~/projects/persona-curvature`, all Modal-backed (no GPU on this box)." Everything
that touches a GPU is a Modal function.

Named Modal volumes referenced by `qwen35/*.py`:

| volume | holds |
|---|---|
| `pc-qwen35-sweep` | the 134 stage-1 adapters at the root; null arms namespaced under `/adapters/<corpus_label>/<trait>` |
| `pc-qwen35-adapters` | phase-2 and ablation adapter outputs |
| `pc-qwen35-oct2` | OCT stage 2 — `/oct/loras_introspection/<trait>`, merged personas at `/oct/personas/<trait>`, seed-1 outputs under `/oct/seed1/` |
| `pc-qwen35-probe` | probe runs |
| `pc-qwen35-rl` | the RL arms |

`qwen35/HANDOVER.md` carries a standing instruction about the first of these: the
null arms are namespaced under the volume and **must not be un-namespaced**.

Applications are named `pc-qwen35-<phase>` and that name is how spend is
attributed. `PC_APP_NAME` used to default to `pc-qwen35-phase2`, which is why
there is no `pc-qwen35-phase5` app at all and why phases 2 and 5 bill jointly; the
default was removed on 2026-08-23 and `_build_modal` now raises `SystemExit` if
the variable is unset, before `import modal` and before any GPU is allocated. See
[[costs]].

## The other three services

- **OpenRouter** — every teacher generation and every judged evaluation. The pot
  is a shared account balance with **no per-agent or per-project attribution**, so
  a day's delta cannot be assigned to this experiment by a ledger alone
  (`qwen35/plan.json#constraints_note`). Teachers and judges used across the
  project include `qwen/qwen3-30b-a3b-instruct-2507` (sweep100 pairs and drift
  judging), `z-ai/glm-4.5-air` (zoo pairs), `anthropic/claude-sonnet-4.6`
  (teacher-screen judge, constitutions), `openai/gpt-5.6-terra` (steering judges).
- **HuggingFace** — public artefacts under `EternalRecursion`:
  `persona-lora-zoo-qwen35` (the adapter zoo) and
  `persona-curvature-oct-transcripts` (the stage-2 transcript corpus). The
  uploader is `qwen35/upload_zoo_batched.py`, batched because of a hard commit
  limit — see [[lesson-hf-commit-rate-limit]].
- **Credentials** live in a secrets directory on the box and as Modal secrets (`hf-token`,
  `pc-qwen35-secrets`). Their contents are never read, quoted or copied into this
  wiki.

## The pattern: every job is a systemd unit with a log in `phase10_runs/`

The Qwen3.5 work runs nothing in a stray shell. Each job is a systemd unit under
`/etc/systemd/system/` with `User=vibe12`,
`WorkingDirectory=/home/vibe12/projects/persona-curvature/qwen35`, an `ExecStart`
that is a single `modal run` (often `--detach`), and
`StandardOutput=append:` plus `StandardError=append:` pointing at one file under
`qwen35/phase10_runs/`. Most also carry an `ExecStartPre` that **rotates the log
on start**:

```
ExecStartPre=/bin/bash -c 'f=.../phase10_runs/X.log; [ -f "$f" ] && mv "$f" "$f.$(date +%%s)" || true'
```

That line exists because an append-only log makes a fixed failure look live: a
monitor grepping for a traceback keeps matching one you already fixed, and the
"read only after the last start marker" guard has a hole during image build and
model download. It reported a resolved crash as live twice in one session
(`/home/vibe12/projects/.garden/notes/appended-logs-replay-fixed-failures.md`).
`phase10_runs/` currently holds 171 entries, including the rotated
`*.log.<epoch>` copies.

Job-scoped configuration travels as `Environment=` lines on the unit —
`PC_APP_NAME` (billing attribution), `PC_PHASE_BUDGET` (a hard stop),
`PC_ADAPTER_VOLUME`, and the recipe knobs `PC_USE_RSLORA`, `PC_LOSS_TYPE`,
`PC_LOSS_WEIGHTS`, `PC_KL_COEF`.

### The `zoo-*` units

| unit | what it runs |
|---|---|
| `zoo-actspace` | activation-space geometry with constitutions as system prompts, 134 traits x 64 prompts |
| `zoo-actspace-adapters` | the same, base plus each stage-1 adapter, no system prompt |
| `zoo-actspace-cross` | 16 adapters x 16 constitutions x 64 prompts |
| `zoo-alien` | steering along the alien (unnamed) direction with shuffle and random controls |
| `zoo-align` | scores the zoo's own DPO pairs against 25 weight-space directions |
| `zoo-alignpairs` | generates DPO pairs for the four alignment-relevant traits |
| `zoo-aligntrain` | trains those four adapters at the zoo recipe and seed |
| `zoo-aligncommon` | retrains them on the zoo's own shared prompt pool |
| `zoo-alignsteer` | steers the four alignment adapters and generates |
| `zoo-batch2` / `zoo-batch3` | OCT stage 2, 10 traits each, balanced one positive and one negative per Big Five factor |
| `zoo-batch4` | OCT stage 2, 39 traits, completing the Big Five 100 |
| `zoo-crossmatched` | cross-Gram of the 134 against the matched-objective seed-1 adapters |
| `zoo-crossoct2` | cross-Grams for stage-2 seed 1 |
| `zoo-eval` | Big Five probe generation for 100 adapters (base / stage 1 / persona) |
| `zoo-fixmerge` | audits the OCT persona merge and publishes exact, cross-term-free personas |
| `zoo-gramoct2` | within-run Gram of the 134 seed-0 stage-2 SFT LoRAs |
| `zoo-grampersona` | within-run Gram of the 134 exact merged persona adapters |
| `zoo-holepairs` / `zoo-holetrain` | pairs and training for the three hole-name traits (cavalier, blase, insouciant) |
| `zoo-judge` | blind Big Five judging of the 100-trait eval generations |
| `zoo-judge-steer` | blind Big Five judging of steered generations |
| `zoo-lex` | OCT stage 2 for the Lexicon validation traits, held out from the factor axes |
| `zoo-nullshuffledmatch` / `zoo-nullpermutedmatch` | retrain the two null arms (100 traits each) at the matched objective |
| `zoo-nullgrams` | Grams for those matched null arms |
| `zoo-nxn` | N x N scoring: every trait's pairs against every adapter |
| `zoo-oct2seed1` / `zoo-oct2seed1-probe` | OCT stage 2 at a second seed, 15 traits, plus a paths-only probe |
| `zoo-optimise` | evolutionary search for data that trains toward a chosen direction |
| `zoo-rlmath` / `zoo-rlmix` | GRPO arms on Dolci math and math+code |
| `zoo-seedmatch` | retrains the 40 seed-paired adapters at the matched objective |
| `zoo-sketch` / `zoo-sketch-lex1` / `zoo-sketch-persona` | sketching adapters to a fixed low-dimensional representation |
| `zoo-sorh` / `zoo-sorheval` | SFT on School of Reward Hacks, hack and matched-control arms, and their behavioural eval |
| `zoo-sphere` | sphere sweep, 72 sampled directions in one container |
| `zoo-steer` / `zoo-steer2` / `zoo-steer3` | steering along PCs, FA factors and pole-neutral factor-identity axes |
| `zoo-steerfix` / `zoo-steerfix2` / `zoo-steerfix3` | corrected steering runs (thinking off, 512 tokens), superseding the 200-token corpus |
| `zoo-steermix` | additivity test, matched-norm mixtures of the five named axes |
| `zoo-verify` | trains adapters on the optimised data to see where they land |
| `zoo40-oct` | the 40-trait OCT stage-2 run, phase 10 |
| `zoo40-meter` | the container-minute spend meter (below) |
| `zoo40-upload-adapters` / `zoo40-upload-datasets` | the HuggingFace uploads |

## Three operational scripts

**`qwen35/status.sh`** — a one-line status emitted **only when something changed**
since the last emit, "so a heartbeat that has nothing to say stays silent rather
than training the reader to ignore it". It counts traits, constitutions, pair
files, adapters and running processes, and reads Modal's billing for today. Two
of its comments are lessons in themselves:

- The field is `modal_workspace_today`, not `modal_mine`, because the Modal
  workspace is **shared**: on 2026-08-19 it read $0.0179 while this project had
  used no GPU at all.
- **Unreadable is not zero.** An earlier version piped grep into `bc`, and on an
  empty match `bc` exits 0 with no output, so the fallback never fired and the
  field rendered blank — a format change, a CLI error or an expired credential
  would all have reported an empty figure that reads as nothing spent. The
  discriminator is now the table's `Cost` header, which Modal prints even with no
  rows: header present with no rows is genuinely zero; header absent means the
  report could not be read, and it says `UNREADABLE`.

**`qwen35/land.sh`** — folds a finished job into the published page. One argument
(`sphere`, `alien`, `align`, `optimise`); each step is idempotent and **refuses to
run on data that is not there** (`[ -f ... ] || { echo "no ..."; exit 1; }`), then
rebuilds the blog data and page.

**`qwen35/zoo40_meter.sh`** — a spend meter that deliberately does **not** depend
on Modal billing, "which lags many hours and read $0.00 all through the 08-25
overnight run while ~$410 was being spent". It integrates **active
container-minutes**, which is observable in real time, at `RATE=2.10` per GPU-hour
for an A100-40GB, with `PRIOR=46.90` already drawn and `BUDGET=2400.00` (raised
2026-08-29 against a $5,000 grant). Three of its comments record fixes:

- The app prefix was widened from `pc-qwen35-phase10` to `pc-qwen35` on
  2026-08-29, because the steering apps ran as `pc-qwen35-steer` and sat outside
  the kill switch — **stopping a systemd unit does not stop a detached Modal
  app**.
- The service glob is `zoo-*.service`, every unit that can spend: naming a subset
  has twice let the meter exit while a sibling was still spending.
- The hard stop resolves ephemeral app ids **at stop time**. The previous version
  called `modal app stop --name <desc>`; there is no `--name` flag, so the call
  failed every time into a log nobody read while claiming the budget was enforced.

Related: [[costs]], [[harness-context]], [[method-lessons]], [[timeline]].
