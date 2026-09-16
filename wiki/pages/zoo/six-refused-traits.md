---
title: The six refused traits
summary: Nine of 140 trait words were refused by the constitution writer as not naming a disposition; three Goldberg markers were re-screened on their inventory sense and accepted, leaving six refusals and 134 trainable traits.
status: current
sources:
  - qwen35/constitutions.json
  - qwen35/regen_goldberg_senses.py
  - qwen35/backups/constitutions.json.pre-generic-anchor.bak
  - qwen35/genpairs.log
  - qwen35/build_blog_page.py
last_verified: 2026-09-16
tags: [zoo, traits, provenance]
---

# The six refused traits

134 = 100 + 40 - 6. This page is the arithmetic.

The constitution writer was given an escape hatch: if the word does not name a
coherent human disposition it must reply `NOT_A_TRAIT: <reason>` rather than
invent a character ([[constitution-generation]]). Nine of the 140 words were
refused on the first pass. Three of those nine were Goldberg markers refused on
the wrong sense of the word, were re-screened, and were accepted. Six refusals
stand, all from the Lexicon draw, and those six were never trained.

## The first nine

From `qwen35/backups/constitutions.json.pre-generic-anchor.bak`, which holds 140
entries of which 9 carry a `rejected` field: Verbal, Prompt, Complex,
Unconformable, Frightened, Busy, Defenseless, Significant, Sleepy.

`qwen35/constitutions.py:report` sets a hard stop at more than 15 of 140
rejected — "the trait source is suspect" — and exits 2. Nine did not trip it.

## The three that were re-screened

`qwen35/regen_goldberg_senses.py` exists for Verbal, Prompt and Complex. Its
argument: these three are in Goldberg's published 100 unipolar markers, meaning
they are administered to human subjects *as personality adjectives*, and the
teacher screened the dictionary sense instead:

| word | teacher's first reading | intended inventory sense |
|---|---|---|
| Verbal | "a descriptor of a modality or medium" | verbally fluent; disposed to express oneself in words readily and copiously |
| Prompt | "a temporal/relational property" | punctual and quick to act as a standing habit, not as a one-off |
| Complex | "a structural property of things or situations" | intellectually complex; drawn to nuance, layered thinking and things that resist simple framing |

The glosses in the right-hand column are `GLOSSES` in
`qwen35/regen_goldberg_senses.py` and are stored on the accepted entries as
`sense_disambiguated` (`qwen35/constitutions.json#Verbal.sense_disambiguated`
and the two siblings). The script's own stated stake: "Dropping them costs
Intellect 2 of its 20 markers, and Intellect is the factor a congruence result
is most fragile on."

Three properties of the re-screen are worth recording because they are what make
it a screen rather than a forced pass:

- the model, prompt template, `max_tokens` and temperature are **imported
  verbatim** from `constitutions.py`, not retyped — "a second screen run against
  a paraphrased prompt is not the same screen";
- one paragraph is spliced in immediately *before* the exception paragraph, so
  the `NOT_A_TRAIT` escape hatch is still the last thing the teacher reads, and
  it is explicitly re-armed for the disambiguated sense;
- "A trait that refuses again is NOT written -- that is a real finding about the
  trait, and it is reported."

All three were accepted. Exactly three entries in `constitutions.json` carry
`sense_disambiguated`.

## The six that stand

From `qwen35/constitutions.json`, verbatim `rejected` texts:

- **Unconformable** — `"Unconformable" is a geological term describing rock
  strata with a discontinuous boundary, not a human character disposition.`
- **Frightened** — `"Frightened" names a transient emotional state or
  condition, not a stable character disposition — it describes how someone feels
  in a moment, not an enduring trait that shapes personality across contexts.`
- **Busy** — `"Busy" describes a situational condition of having much to do, not
  a stable dispositional orientation toward the world — it is a transient state,
  not a character trait.`
- **Defenseless** — `Defenseless describes a situational condition or relational
  state (lacking protection or defense against something), not a stable
  character disposition a person embodies from the inside.`
- **Significant** — `"Significant" is a relational property describing magnitude
  or importance relative to a standard, not a stable disposition or character
  trait a person can embody.`
- **Sleepy** — `Sleepy describes a transient physiological state, not a stable
  character disposition.`

All six are from the Lexicon draw; none is a Goldberg marker.

## Where the six leave the pipeline

`qwen35/gen_pairs.py` skips any trait whose `constitutions.json` entry is a
`rejected` record. Every pair-generation log opens with the same line
(`qwen35/genpairs.log:4-5`):

```
traits 134 (skipped 6 without a constitution)  prompts 500  target pairs 67000
  skipped: Unconformable (...), Frightened (...), Busy (...),
           Defenseless (...), Significant (...), Sleepy (...)
```

The blog page states the same thing for a reader: "Forty were drawn; six were
refused by the constitution writer as states or situations rather than
dispositions (*sleepy*, *busy*, *frightened*...) and never trained"
(`qwen35/build_blog_page.py`).

## A note on the screen as method

This is the only quality screen the Lexicon draw gets. Draws 1 and 2 died of
dictionary-based filtering ([[discarded-secondary-draws]]), so the replacement
question is "can a competent writer render this as a person?", asked of the
model that then has to render it (`qwen35/constitutions.py` docstring). Its
observed selectivity is 9 of 140 on the first pass and 6 of 140 after the sense
correction.
