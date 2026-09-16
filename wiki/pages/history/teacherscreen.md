---
title: The teacher screen
summary: A four-model teacher screen was designed, run and paid for on 2026-08-14, and its analysis was never produced; the teachers actually used were chosen on other grounds.
status: historical
sources:
  - teacherscreen/config.py
  - teacherscreen/traits.json
  - teacherscreen/results/gen_cost.json
  - teacherscreen/results/judge_cost.json
  - teacherscreen/analyze.py
  - common.py
  - qwen35/gen_pairs.py
  - /home/vibe12/projects/agent-harness/archive/discord-2026-08-31/rendered/projects--persona-cartography.md
  - /home/vibe12/projects/agent-harness/memory/server/maintenance-log.md
last_verified: 2026-09-07
tags: [history, teacher, screening, gap]
---

# The teacher screen

## Why it was proposed

On 2026-08-14 Samuel asked about using a small open model as the teacher that
writes the DPO pairs. The reply separated two proposals that the phrase hides:
the **same** model as the student (self-distillation) versus simply a **cheaper
different** teacher. The second is unobjectionable — the teacher choice in the
literature is about headroom over the student. The first has a hard ceiling: you
can only instil a trait the teacher can already express when prompted, so you move
behaviour from prompt to weights rather than creating it. The failure mode is not
a weak trait vector but a **trained-on-noise adapter that still produces a
confident-looking `dW`** and then sits in a PCA looking like a data point.

The judge must stay external in either case. Original Open Character Training used
GLM-4.5-Air as both teacher and judge; Persona Cartography deliberately broke that
loop with a separate Qwen3-235B judge. Local judging would have saved about $30
across a full run and cost the ability to believe the results.

Rather than argue it, a measurement was proposed and Samuel approved it
("Go for it", 2026-08-14 22:55:32) — with the framing that the result should be an
**inclusion criterion** rather than a verdict on a teacher: only train traits the
teacher can demonstrably express distinctly, which simultaneously screens out
junk entries and synonym clusters.

## The design, as run

From `teacherscreen/config.py` and `teacherscreen/traits.json`:

- **30 traits in four tiers** — `common`, `mid`, `obscure`, and a `synonym`
  cluster. Every trait carries a written behavioural description rather than a
  bare adjective.
- **Three conditions**, not two: amplifier, **neutral**, suppressor. The neutral
  arm is the control; without it trait expression cannot be told apart from
  "wrote something longer and more flowery".
- **Four candidate teachers** spanning size: `google/gemma-3-4b-it` (4B dense),
  `meta-llama/llama-3.1-8b-instruct` (8B dense),
  `qwen/qwen3-30b-a3b-instruct-2507` (30B MoE, 3B active),
  `qwen/qwen3-235b-a22b-2507` (235B MoE, 22B active). Model ids were taken from a
  live OpenRouter query on 2026-08-14.
- **One strong blind judge from a family that is not among the teachers**:
  `anthropic/claude-sonnet-4.6`, so no teacher is judged by a relative. Trait
  descriptions were written once, offline, by `anthropic/claude-opus-4.5`.
- 30 traits x 3 conditions x 8 prompts x 4 teachers = **2,880 generations**.

Three things were to be measured beyond the headline: the amplify-versus-suppress
asymmetry reported separately (the suppressor side becomes the *rejected* half of
a DPO pair, so a teacher that can only amplify gives pairs with a broken half);
trait counts clearing separation thresholds of 2, 3 and 4 rather than one chosen
cutoff; and, for the six-member synonym cluster, every response judged against all
six labels to produce a 6x6 matrix — if the diagonal does not dominate, the six
words are one direction sampled six times and any PCA over adapters trained on
them would report a cluster that is an artefact of the list.

## What it cost, and what happened to it

The screen **ran**. From the cost artefacts:

| stage | figure | source |
|---|---|---|
| generation | **$0.119453** over 2,880 rows | `teacherscreen/results/gen_cost.json#cumulative_generation_cost_usd` |
| judging | **$8.820759** over 5,760 judgements | `teacherscreen/results/judge_cost.json#cumulative_judge_cost_usd` |

`teacherscreen/results/generations.jsonl` (2.5 MB) and `judgements.jsonl` (1.3 MB)
are on disk, written 2026-08-14 23:11 and 23:16.

**The analysis was never produced.** `teacherscreen/analyze.py` writes
`results/teacher_screen.json` and `results/teacher_screen.md`; neither file
exists. The reason is legible in the channel record: Samuel's next message, at
2026-08-14 22:57:39 — two minutes after approving the screen — redirected the
night to the 100-trait sweep ("can you pick a diverse set of 100 traits and try to
train dpo loras for all of them"). The screen's raw data was written that evening
and nothing read it.

So the answer to "what was chosen and why" is **not** "the screen chose it". No
figure from this screen is quoted anywhere in the project.

## What was chosen instead, and on what grounds

| corpus | teacher | source |
|---|---|---|
| the 100-trait Qwen2.5-3B sweep and its expansions | `qwen/qwen3-30b-a3b-instruct-2507` | `common.py#MODEL` |
| the Qwen3.5-4B zoo | `z-ai/glm-4.5-air` | `qwen35/gen_pairs.py:63` (`TEACHER`, commented "OCT's own teacher family") |

The zoo's teacher was chosen to match Open Character Training's own, on Samuel's
instruction ("i meant 4.5 air", 2026-08-19). The choice was reached through a cost
sequence worth recording, from
`/home/vibe12/projects/agent-harness/memory/server/maintenance-log.md`
(2026-08-19 entries):

| teacher / configuration | projected cost of the pair-generation phase |
|---|---|
| `anthropic/claude-sonnet-4.6` | $420 |
| GLM-4.5-Air, **as configured** | **$793** |
| GLM-4.5-Air, reasoning off | **$19** |

The cheap model, four times cheaper per token, was twice as expensive per
*result*. GLM ships with reasoning on by default; 82% of paid completion tokens
were thinking rather than output, the reasoning overran the token budget before
the response format closed, six of eight pairs failed to **parse** and were
retried at three calls per usable pair. Two multipliers stacked, and it presented
as a parse error, not as a cost problem — anyone debugging that symptom would have
fixed the retry loop and never looked at the bill. See
[[lesson-thinking-default-trap]] for the same defect in the base model's own chat
template.

The phase was caught only because the estimate was smoke-tested for $0.63 before
the phase ran. The diagnosis recorded at the time: "four dollars felt right and I
did not check it, **precisely because** it was small enough to be reassuring."

## What is missing

The screen's own questions remain unanswered on this project's data: where a
teacher starts failing by trait tier, whether the amplify and suppress sides
separate equally, and whether the six-word synonym cluster is one direction. The
raw judgements exist and the analysis is one script run away, but running it would
produce new numbers, which this wiki does not do. Recorded as a gap.

Related: [[sweep100]], [[zoo-construction-overview|trait provenance]], [[costs]],
[[open-character-training-paper]], [[persona-cartography-paper]].
