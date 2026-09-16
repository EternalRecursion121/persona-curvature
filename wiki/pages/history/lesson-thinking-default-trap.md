---
title: "Lesson: Qwen3.5's chat template defaults thinking ON"
summary: Three runs were silently invalidated because a generation path did not pass enable_thinking=False, and every failure produced plausible text and real numbers.
status: current
sources:
  - /home/vibe12/.claude/projects/-home-vibe12-projects/memory/qwen35-thinking-default.md
  - /home/vibe12/projects/agent-harness/memory/server/maintenance-log.md
last_verified: 2026-09-07
tags: [lesson, generation, qwen35]
---

# Lesson: Qwen3.5's chat template defaults thinking ON

Recorded in
`/home/vibe12/.claude/projects/-home-vibe12-projects/memory/qwen35-thinking-default.md`,
found 2026-08-29.

`apply_chat_template` for Qwen3.5 defaults **thinking on**. Any generation path
that does not explicitly disable it produces a reasoning preamble instead of an
answer, and a short token budget then truncates mid-plan so no answer ever
appears.

Three separate runs were silently invalidated, all found the same day:

1. **`steer_qwen35.py:125`** omitted the flag at `max_new_tokens=200`. All
   **1,512 steering generations** were truncated planning traces, **85% not ending
   on terminal punctuation**, and the Big Five judge scored those rather than
   responses. The whole steering corpus had to be regenerated — which is what the
   `zoo-steerfix`, `zoo-steerfix2` and `zoo-steerfix3` units are, described in
   `/etc/systemd/system/` as "Corrected steering run (thinking off, 512 tokens) --
   supersedes the 200-token preamble corpus".
2. **The trait evals** shared the 200-token cap. They did pass the flag, so they
   are answers, just short.
3. **GRPO in `rl_capability.py`**: TRL renders the prompt itself, so step 1
   returned `clipped_ratio 1.0`, `mean_terminated_length 0`, `reward 0`,
   `grad_norm 0` — every rollout reasoned to the 2048 cap without answering,
   `mask_truncated_completions` masked all of them, and the gradient was exactly
   zero.

**The failure is silent in every case: the job runs, produces plausible-looking
text, and returns numbers.**

## How to apply it

Pass `enable_thinking=False` on every path, and check the mechanism per framework:
`apply_chat_template(..., enable_thinking=False)` directly,
`GRPOConfig(chat_template_kwargs={"enable_thinking": False})` for TRL, `_chat_str`
in `oct_stage2.py` for the OCT pipeline. `steer134_on_modal.py` has the right
pattern: it passes the flag **and asserts the rendered prompt contains no unclosed
`<think>` block**. Copy the assertion rather than trusting the flag.

## The same defect, on the other side of the API

The teacher model had it too, and it cost money rather than validity. GLM-4.5-Air
ships with reasoning on by default; 82% of paid completion tokens were thinking
rather than output, the reasoning overran the token budget before the response
format closed, six of eight pairs failed to **parse** and were retried at three
calls per usable pair. The phase's projected cost went
**$793 (GLM as configured) to $19 (GLM, reasoning off)**, against $420 for a
much more expensive model
(`/home/vibe12/projects/agent-harness/memory/server/maintenance-log.md`,
2026-08-19 18:56). It presented as a parse error, not as a cost problem.

**Per-token price is not per-result cost**, and the gap is set by defaults you did
not choose plus a failure mode that looks like a different kind of bug.

Related: [[method-lessons]], [[steering-results]], [[teacherscreen]], [[costs]],
[[lesson-objective-must-travel]].
