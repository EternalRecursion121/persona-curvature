---
title: The anchor block and its 2026-08-19 revision
summary: Every constitution carries a cross-trait anchoring paragraph; the original enumerated one Goldberg marker per Big Five factor, which could have manufactured the result under test, and was replaced by a generic block with the enumerated form retained as an ablation arm.
status: current
sources:
  - qwen35/anchor_constitutions.py
  - qwen35/constitutions.json
  - qwen35/backups/constitutions.json.pre-generic-anchor.bak
  - qwen35/backups/constitutions.json.pre-goldberg-senses.2026-08-19T191500.bak
  - qwen35/data_ENUMERATED_ANCHOR_DISCARDED_2026-08-19
  - qwen35/paper_notes.md
last_verified: 2026-09-07
tags: [zoo, constitutions, corrections]
---

# The anchor block and its 2026-08-19 revision

Every accepted constitution ends with a fixed **anchor block** appended verbatim.
Its job is to make the delta trait-specific: hold everything else about yourself
at baseline. `qwen35/paper_notes.md` section 3.4 traces the idea to
[[persona-cartography-paper]], whose constitutions reproduce the other four OCEAN
traits in full with "do not amplify OR suppress any of these"; this project had
no OCEAN basis to anchor against, so it used a short clause instead.

## The problem with the first version

The original block ended with an enumeration
(`qwen35/anchor_constitutions.py`, quoted there verbatim):

> You are not more talkative, warmer, more anxious, more careful or more
> imaginative than you would otherwise be, except where that follows directly
> and unavoidably from the trait described above.

Those five adjectives are one Goldberg marker per Big Five factor:

| adjective | factor |
|---|---|
| talkative | Extraversion (I) |
| warm | Agreeableness (II) |
| anxious | Emotional Stability (IV, reversed) |
| careful | Conscientiousness (III) |
| imaginative | Intellect/Openness (V) |

The script states the objection plainly: naming one marker per factor in every
training document "hands the model the exact five-dimensional frame the
experiment is supposed to discover", and because the clause is conditional per
trait it is "a per-trait push toward one named axis and away from the other four
-- precisely the structure the analysis then reports as an emergent finding. It
could manufacture the result under test, and it would do so in the flattering
direction."

It also records that the enumeration was this project's own wording, not
inherited: Persona Cartography enumerates the other four OCEAN dimensions, but
its trait basis *is* OCEAN and it never tests for OCEAN emergence, so the same
sentence carries no circularity there.

## The two blocks, verbatim

**ANCHOR_GENERIC** (the current default `constitution`):

> Hold everything else about yourself at your normal baseline. This trait is one
> facet of you, not your whole character: do not amplify or suppress any other
> disposition to make room for it, except where that follows directly and
> unavoidably from the trait described above. Where it does not follow, stay
> exactly as you were.

**ANCHOR_ENUMERATED** (retained as `constitution_enumerated`):

> Hold everything else about yourself at your normal baseline. This trait is one
> facet of you, not your whole character: do not amplify or suppress any other
> disposition to make room for it. You are not more talkative, warmer, more
> anxious, more careful or more imaginative than you would otherwise be, except
> where that follows directly and unavoidably from the trait described above.
> Where it does not follow, stay exactly as you were.

Both are preceded by a blank line so they read as their own paragraph. The
enumerated block is recorded as 437 characters with SHA-256
`b860a501902bc00f2448a660b7aa3dd8ffba0625a1f620c785a4db2809ca5f27`
(`qwen35/anchor_constitutions.py:ENUMERATED_SHA256`, `ENUMERATED_LEN`). It is
**derived from the file at import time** as the common suffix after
`constitution_unanchored`, then checked against that hash, rather than retyped —
the script's stated reason being that retyping "is exactly how an 'identical'
ablation arm silently drifts".

The provenance note stored on every migrated entry
(`qwen35/constitutions.json#Warm.anchor`):

> generic anchor, swapped 2026-08-19: the previous block enumerated one marker
> adjective per Big Five factor, which risks manufacturing the factor structure
> under test; enumerated form retained in constitution_enumerated as a phase-4
> ablation arm

## Why the script exists at all

The anchoring step was originally "run ad hoc and left NO SCRIPT ON DISK"
(`qwen35/anchor_constitutions.py` docstring). Since the anchored text is what
every adapter is trained on, an unreproducible preprocessing step meant the
training corpus could not be regenerated from source. The script closes that: it
runs read-only by default, recomputing both derived fields for every accepted
entry and asserting byte-identity with the file, and exits nonzero on any
mismatch. `--write` performs the migration. The rule it encodes, stated as
derived empirically rather than from memory: both derived fields are plain
concatenation, no strip, no normalisation, no trait-name interpolation, and each
block is byte-identical across all 131 accepted traits at the time.

## The two backups tell the order of revisions

Both files in `qwen35/backups/` hold 140 entries with 9 rejections, so both
predate the sense re-screen ([[six-refused-traits]]):

| file | fields on an accepted entry | state it captures |
|---|---|---|
| `constitutions.json.pre-generic-anchor.bak` | `constitution`, `constitution_unanchored`, `anchor` | before the swap: `constitution` is unanchored + **enumerated** |
| `constitutions.json.pre-goldberg-senses.2026-08-19T191500.bak` | `constitution`, `constitution_unanchored`, `constitution_enumerated`, `anchor` | after the swap, before the three sense re-screens |

So the sequence on 2026-08-19 was: generate 140 with the enumerated anchor →
back up → migrate to the generic anchor and keep the enumerated form as a fourth
field → back up → re-screen Verbal, Prompt and Complex and merge them in with
the same four-field schema. The current file adds the four alignment traits and
the three hole traits on top.

## The discarded data directory

`qwen35/data_ENUMERATED_ANCHOR_DISCARDED_2026-08-19/` is the pair-generation
output made under the enumerated anchor before the swap. It holds `_raw/` shards
for 131 traits and two assembled files, `extraverted.jsonl` and `talkative.jsonl`,
of 4 rows each; `_usage.json` records 8 calls over 2 traits at 4 pairs apiece and
`estimated_cost_usd` 0.002342. It is a smoke run, not a corpus — the directory
name records that the anchor changed underneath it and the data were discarded
rather than reused.

The enumerated ablation arm ("phase 4") is not among the trained arms recorded in
`qwen35/phase2_runs/`; whether it was ever run is a gap.
