---
title: The thinking-default trap and what it withdrew
summary: Qwen3.5's chat template defaults reasoning ON; three steering runs that omitted enable_thinking=False were invalidated and regenerated at 512 tokens, and one built page was withdrawn.
status: current
sources:
  - qwen35/PREREG_steerfix.md
  - qwen35/steer_fix.py
  - qwen35/steer_qwen35.py
  - qwen35/oct_stage2.py
  - qwen35/distil_page/index.html
  - qwen35/phase10_runs/judge_steer.log
  - qwen35/phase10_runs/judge_steerfix.log
  - /etc/systemd/system/zoo-steerfix.service
last_verified: 2026-09-07
tags: [behaviour, method-lessons, withdrawn]
---

# The thinking-default trap and what it withdrew

Qwen3.5's chat template turns reasoning on unless `apply_chat_template` is
called with `enable_thinking=False`. The first steering campaign omitted that
argument, so every generation it produced was a reasoning preamble rather than
an answer. Everything judged from those generations was withdrawn. This page
records the defect, its exact scope, and which results replaced which. The
history section keeps the general method lesson at [[lesson-thinking-default-trap]];
this page is the file-level evidence.

## The defect

`steer_qwen35.py:125` generated with `max_new_tokens=200` and called
`apply_chat_template` without `enable_thinking=False`
(`qwen35/steer_qwen35.py:125`; the same line number is cited in
`qwen35/PREREG_steerfix.md`). The consequence, as recorded in the
pre-registration:

> `steer_qwen35.py` called `apply_chat_template` without
> `enable_thinking=False`. Qwen3.5 defaults thinking ON. With
> `max_new_tokens=200`, every one of the 1512 steering generations is a
> reasoning preamble cut off mid-plan (85% do not end on terminal punctuation;
> none reaches an answer). The Big Five judge scored planning traces, not
> responses.
> — `qwen35/PREREG_steerfix.md`

The pre-registration also records how it was found: "Found independently by two
qualitative-analysis agents reading the transcripts, then confirmed at source"
(`qwen35/PREREG_steerfix.md`). See [[qualitative-notes]].

## Scope: steering only

`qwen35/PREREG_steerfix.md` states the scope explicitly:

> Scope: **steering only**. The trait evaluations in `eval_100traits.json` went
> through `oct_stage2.py:_chat_str`, which does pass `enable_thinking=False`;
> those are direct answers (also capped at 200 tokens, so short, but answers).

Confirmed at source: `qwen35/oct_stage2.py:269` and `qwen35/oct_stage2.py:654`
both pass `enable_thinking=False`, and `oct_stage2.py:51` states the project
convention ("enable_thinking=False on every apply_chat_template"). So
[[judged-evaluations]] on the 100 trait adapters is unaffected.

Every later generation script passes the flag: `steer_fix.py:160`,
`sphere_sweep.py:144`, `optimise_data.py:176`, `eval_rl_persona.py:57`,
`sft_rewardhacks.py:100`, `rl_capability.py:241`/`:303`/`:466`/`:650`. The
separate STEER134 campaign asserted it in-container and refuses to proceed if
the template leaves an open `<think>` block
(`qwen35/steer134_on_modal.py:332-334`).

## Which runs were withdrawn, and what replaced them

| withdrawn run | script | result file | replacement | replacement result |
|---|---|---|---|---|
| `zoo-steer` (9 directions x 7 alphas) | `steer_qwen35.py` | `phase10_runs/steer_results.json` (2026-08-29 12:25) | `zoo-steerfix` | `phase10_runs/steer_results_fix.json` (2026-08-29 23:40) |
| `zoo-steer2` (8 directions x 11 alphas) | `steer_qwen35.py --spec steer_spec2.json` | none — the run hit `FunctionTimeoutError` at 5400 s (`phase10_runs/steer2.log`) and `steer_results2.json` was never written | `zoo-steerfix2` (8 directions x 7 alphas) | `phase10_runs/steer_results_fix2.json` (2026-08-30 01:52) |
| `zoo-steer3` (5 identity directions x 7 alphas) | `steer_qwen35.py --spec steer_spec3.json` | generations completed (`phase10_runs/steer3.log`) but `steer_results3.json` is not on disk | `zoo-steerfix3` | `phase10_runs/steer_results_fix3.json` (2026-08-30 01:08) |

The systemd unit for the corrected run names the supersession in its own
description:

> `Description=Corrected steering run (thinking off, 512 tokens) -- supersedes the 200-token preamble corpus`
> — `/etc/systemd/system/zoo-steerfix.service`

`zoo-steerfix2.service` and `zoo-steerfix3.service` carry
`Description=Corrected steering run 2 (thinking off, 512 tokens)` and
`... run 3 ...` respectively.

Only two things changed between `steer_qwen35.py` and `steer_fix.py`:
`enable_thinking=False`, and `max_new_tokens` 200 -> 512
(`qwen35/steer_fix.py:37-39`; generation call at `steer_fix.py:163`). Same
prompts, alphas, coefficients, greedy decoding and judge.

## The judge's own reliability shows the defect

The same judge (`anthropic/claude-sonnet-4.5`, see [[judged-evaluations]])
scored both corpora with the same rubric. Repeat-judgment correlation on the
old corpus, from `qwen35/phase10_runs/judge_steer.log`:

| scale | thinking ON (judge_steer.log) | thinking OFF (judge_steerfix.log) |
|---|---|---|
| Extraversion | r=0.354 (n=45) | r=0.782 (n=75) |
| Agreeableness | r=0.710 (n=45) | r=0.902 (n=75) |
| Conscientiousness | r=0.510 (n=45) | r=0.839 (n=74) |
| EmotionalStability | r=0.508 (n=41) | r=0.869 (n=75) |
| Intellect | r=0.598 (n=45) | r=0.885 (n=75) |

Both runs judged 1512 generations from 9 directions with 0 failed calls
(`qwen35/phase10_runs/judge_steer.log`,
`qwen35/phase10_runs/judge_steerfix.log`). The judge was not less careful on
the old corpus; the text it was scoring did not support a stable rating.

## The page that was withdrawn

`qwen35/distil_page/index.html` was replaced on 2026-09-01 by a withdrawal
notice. Its own account of the steering half of the withdrawal:

> **The steering dose-response panels.** Every steering generation behind them
> was produced without `enable_thinking=False` and capped at 200 tokens. Qwen3.5
> defaults thinking on, so those were reasoning preambles cut off mid-plan — 85%
> did not end on terminal punctuation and none reached an answer. The blind judge
> scored planning traces, not responses.
>
> The corpus was regenerated with thinking disabled at 512 tokens. All nine
> directions kept their sign and every effect was two to four times larger, so
> the shapes were real and merely diluted. But two sharper claims did not
> survive, and one figure here was drawn from rows where the model was looping
> verbatim.
> — `qwen35/distil_page/index.html`

The withdrawn page's own content is preserved at
`qwen35/distil_page/index.html.withdrawn-backup`. See [[distillation-check]].

The old judged curves survive in `qwen35/analysis/distil_data.json#steer` (nine
directions, per-alpha Big Five series), which is the only place the withdrawn
numbers are still stored. They should not be quoted as results.

## What the correction did and did not change

The pre-registration committed in advance to reporting divergence as a finding:

> No page is built from the 200-token steering corpus. If the shapes replicate,
> the old corpus becomes a robustness footnote. If they diverge, that divergence
> is reported as a finding, not quietly dropped.
> — `qwen35/PREREG_steerfix.md`

The scored outcome of the nine registered predictions is on
[[steering-results]].

Related: [[steering-results]], [[judged-evaluations]], [[distillation-check]],
[[built-pages-inventory]], [[glossary]].
