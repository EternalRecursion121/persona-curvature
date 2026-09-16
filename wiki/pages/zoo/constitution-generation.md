---
title: Constitution generation
summary: Each trait's training data is conditioned on a 120-200 word second-person character document written by claude-sonnet-4.6 at temperature 0, with a built-in refusal option that doubles as the only quality screen on the trait words.
status: current
sources:
  - qwen35/constitutions.py
  - qwen35/constitutions.json
  - qwen35/constitutions_cost.json
  - qwen35/paper_notes.md
last_verified: 2026-09-07
tags: [zoo, constitutions, construction]
---

# Constitution generation

A **constitution** here is a system-prompt-style conditioning document for one
trait: 120-200 words, second person, opening with "You are", describing how the
character thinks, what they attend to, how they speak and how they behave under
pressure. It is the object that conditions the teacher when the DPO pairs are
generated ([[dpo-pair-generation]]), so it is part of the training corpus, not
packaging. The idea and the unit come from [[persona-cartography-paper]] (see
`qwen35/paper_notes.md` section 2.2).

## Configuration

From `qwen35/constitutions.py`:

| | |
|---|---|
| writer model | `anthropic/claude-sonnet-4.6` |
| temperature | 0.0 |
| max tokens | 600 |
| concurrency | 12 |
| refusal sentinel | `NOT_A_TRAIT:` |
| output | `qwen35/constitutions.json`, canonical trait order |

The script is resumable: it re-reads `constitutions.json` on start and only
sends the missing traits. An entry carrying `error` is not treated as a result
and is retried; entries carrying `constitution` or `rejected` are kept.

## The prompt, in summary

The instruction (verbatim in `qwen35/constitutions.py:PROMPT`) asks for 120-200
words in the second person; for how the character thinks, attends, speaks and
behaves under pressure; for something that works as a directive conditioning
document; for the disposition itself rather than the word (no definitions, no
etymology, no mention that it is a trait or a label); no preamble, title,
bullets or closing commentary. One clause matters for what the adapters end up
learning:

> Render the disposition honestly, including its costs and failure modes. Do
> not sand it down into a virtue, and do not moralise about it.

Then the exception: if the word is a physical property, an object, a bodily
state, a transient condition, a relation, or simply not a character trait, reply
`NOT_A_TRAIT: <one line saying why>`, and "Use that exception only when it
genuinely applies."

## The refusal is the quality screen

`qwen35/constitutions.py`'s own docstring states the design: the constitution
"doubles as a QUALITY SCREEN on the trait words themselves... This is the only
screen the secondary set gets -- draws 1 and 2 died of dictionary-based
filtering, so the check here is 'can a competent writer render this as a
person?', asked of the model that has to render it." Nine of 140 were refused;
three were re-screened and accepted; six stand. See [[six-refused-traits]] and
[[discarded-secondary-draws]].

`report()` prints per-set rejection counts, word-count min/max/mean, how many
fall outside 110-215 words, how many open with "You are", and the full rejection
list; it exits 2 if more than 15 of 140 are rejected.

## What is in the file now

`qwen35/constitutions.json` holds **147** entries: 100 primary + 40 secondary +
4 alignment + 3 hole ([[alignment-and-hole-traits]]). 141 carry a constitution,
6 carry a `rejected` record.

An accepted primary or secondary entry has four fields
(`qwen35/constitutions.json#Warm`):

```
constitution              = constitution_unanchored + ANCHOR_GENERIC
constitution_unanchored   = raw generator output, never modified
constitution_enumerated   = constitution_unanchored + ANCHOR_ENUMERATED
anchor                    = provenance note
```

The three later "hole" entries (`Cavalier`, `Blase`, `Insouciant`) carry only
`constitution` — they were generated after the anchoring migration and were not
put through it. Why the anchor exists and why it was rewritten is
[[constitution-anchor-revision]].

## Cost

`qwen35/constitutions_cost.json` is **not** the cost of the 140-trait run. It
was overwritten by the last invocation, which generated only the three hole
traits: `calls` 3, `prompt_tokens` 917, `completion_tokens` 698,
`estimated_cost_usd` 0.013221, `per_condition` containing only `hole`. The recorded per-million prices are
$3.00 prompt and $15.00 completion (`#price_per_million`). The original
140-trait cost figure does not survive anywhere on disk and is a gap.
