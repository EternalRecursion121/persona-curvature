---
title: DPO pair generation
summary: One paired-teacher call per trait and prompt returned both sides of a preference pair; glm-4.5-air produced 66,937 of a target 67,000 pairs across 134 traits over four passes.
status: current
sources:
  - qwen35/gen_pairs.py
  - qwen35/genpairs.log
  - qwen35/genpairs_retry.log
  - qwen35/genpairs_retry2.log
  - qwen35/genpairs_retry3.log
  - qwen35/paper_notes.md
  - qwen35/build_blog_page.py
last_verified: 2026-09-07
tags: [zoo, data, construction]
---

# DPO pair generation

Stage one trains on preference pairs: a `chosen` reply in character and a
`rejected` reply out of character, for the same prompt. `qwen35/gen_pairs.py`
produced them.

## The teacher and the scheme

| | |
|---|---|
| teacher | `z-ai/glm-4.5-air` (`gen_pairs.py:TEACHER`, default of `QW_TEACHER`) |
| posted price | $0.1300 / $0.8500 per million prompt / completion tokens (`genpairs.log:2`) |
| sampling | temperature 0.7, top_p 0.95 (`gen_pairs.py:TEMP, TOP_P`) |
| reasoning | explicitly disabled in the request (`extra={... "reasoning": {"enabled": False}}`) |
| concurrency | 32 |
| attempts | 4 (one try plus up to three retries, then the cell is dropped) |
| length guard | 40-220 words, "lenient guard; the ask is 60-140" |

The teacher choice is recorded in the code as "OCT's own teacher family"; the
sampling settings cite Persona Cartography's appendix A.1.1.

**Paired teacher.** One call per (trait, prompt) returns *both* sides:

- `chosen` — a reply by a character who embodies the constitution;
- `rejected` — a reply by a character at the **opposite pole of the same
  dimension**, same prompt, same length budget.

The code states the reason: "The rejected side is never an unconditioned/base
response. [App L.4(a)] measures that scheme (OCT's) at ~30% of paired-teacher's
effect size at identical hyperparameters." This is Persona Cartography's scheme,
not Open Character Training's; see [[recipe-vs-source-papers]] section 3.6.

The trait's `keyed` field is carried as metadata and **does not** flip
chosen/rejected: every adapter is trained toward its own trait, so `Untalkative`
is trained toward untalkativeness, not away from talkativeness.

The two sides arrive in one response delimited by three double-bracketed
sentinels, defined in `gen_pairs.py` as `A`, `B` and `END`.

> Contradiction to record. `qwen35/build_blog_page.py` describes stage one as
> "preference pairs where the chosen response is in character and the rejected
> one is the model's default." The code says the opposite in as many words. The
> blog page is current truth for results; for construction the generator is
> primary, and the rejected side is an opposite-pole character.

## The filters

A cell is dropped rather than kept if the reply leaks the scaffolding or names
the trait:

- **trait-word ban.** `stems_of()` builds the word, a crude stem (drop three
  characters, floor 6), the punctuation-free stem, and for a negated adjective
  the stem of the root (`untalkative` -> `talkat`). A hit anywhere in either
  reply drops the pair.
- **generic forbidden**: "personality", "big five", "big-five", "big 5",
  "character trait".
- **meta forbidden**: "constitution", "opposite pole", "reply a", "reply b",
  "the other reply", "character document", and the opening double bracket of the
  sentinels themselves.
- **identical**: the two sides coming back the same.

Drop reasons per pass, from the log footers:

| pass | forbidden | identical |
|---|---|---|
| `genpairs.log` | 462 | 1 |
| `genpairs_retry.log` | 154 | 0 |
| `genpairs_retry2.log` | 87 | 0 |
| `genpairs_retry3.log` | 62 | 1 |

The ban has a known asymmetry, and it is the reason the shared pool had to be
intersected afterwards: for a negated trait the *rejected* reply embodies the
positive pole, so a genuinely creative reply for `Uncreative` legitimately says
"creative" and is dropped. Each trait therefore loses, preferentially, the
prompts most about that trait. See [[shared-prompt-pool-445]].

## The four passes

All four ran against the same 500-prompt pool, resuming per (trait, prompt) cell
from `data/_raw/<slug>.jsonl`.

| pass | calls | pairs after | cost |
|---|---|---|---|
| main | 73,934 | 66,537 / 67,000 | $20.6535 |
| retry 1 | 1,263 | 66,846 | $0.3721 |
| retry 2 | 498 | 66,913 | $0.1465 |
| retry 3 | 308 | 66,937 | $0.0905 |

Main-pass wall time 179m22.8s at about 6.9 calls/s. Cost per pair on the main
pass is printed as $0.00031. Complete traits (all 500 prompts) went 58 -> 96 ->
113 -> 118; incomplete traits went 76 -> 38 -> 21 -> 16.

The retry passes are what took the common pool from 236 to 445 prompts, at a
total of about fifty cents (`qwen35/make_common_pool.py` docstring).

## Output layout

`qwen35/data/<slug>.jsonl`, one row per pair with keys `prompt`, `chosen`,
`rejected`, plus `_raw/` shards for resumption and `_dropped.json` /
`_usage.json` summaries. 134 files.

> Provenance caveat. `qwen35/data/_usage.json` and `qwen35/data/_dropped.json`
> in the repository today are **not** the zoo's: they were overwritten by the
> later alignment-trait run and record 23 calls over Sycophantic, Obsequious,
> Power-seeking and Corrigible. The zoo's own totals survive only in
> `genpairs*.log`. Note also that the `model` field in any `_usage.json` written
> by these scripts is `common.Usage`'s default (`common.py:MODEL =
> "qwen/qwen3-30b-a3b-instruct-2507"`) unless the caller overrides it —
> `constitutions.py` overrides it explicitly, `gen_pairs.py` does not — so that
> field is not evidence of which teacher ran. The teacher is printed in the log
> header instead.

The intersected corpus actually used for training is `qwen35/data_common/`;
see [[shared-prompt-pool-445]] and [[stage-one-training-config]].
