---
title: The 100 Goldberg primary traits
summary: The primary trait set is Goldberg's 100 unipolar Big Five markers, copied unchanged from the earlier sweep100 experiment and verified by five mechanical checks before use.
status: current
sources:
  - qwen35/traits_primary.json
  - qwen35/select_traits.py
  - qwen35/build_blog_page.py
last_verified: 2026-09-16
tags: [zoo, traits, provenance]
---

# The 100 Goldberg primary traits

`qwen35/traits_primary.json` holds 100 entries, each `{trait, factor, keyed}`.
These are Goldberg's published unipolar Big Five marker adjectives, used as
published. They are the only traits in the zoo that carry a Big Five label, and
therefore the only ones any labelled test is scored on
(`qwen35/build_blog_page.py`, provenance paragraph).

## Composition

Counted from `qwen35/traits_primary.json`:

| factor | keyed + | keyed - |
|---|---|---|
| Extraversion | 10 | 10 |
| Agreeableness | 10 | 10 |
| Conscientiousness | 10 | 10 |
| EmotionalStability | 6 | 14 |
| Intellect | 10 | 10 |

Twenty markers per factor, both poles. Emotional Stability is the exception to
the 10/10 split: 6 positively keyed and 14 negatively keyed, because the marker
set itself is asymmetric there (words like *anxious*, *nervous*, *fretful*,
*touchy* are the negative pole and outnumber their opposites). The blog page
describes the set as "twenty per factor, both poles"
(`qwen35/build_blog_page.py`); that is true of the factor counts but not of the
keying counts for Emotional Stability.

The first entry is `{"trait": "Extraverted", "factor": "Extraversion",
"keyed": "+"}`.

## Provenance and the verification gate

`qwen35/select_traits.py:build_primary` does not generate the set. It reads
`sweep100/traits.json` from the earlier 100-adapter experiment, runs five checks,
and copies the file **byte-for-byte** if all pass, printing a SHA-256 prefix for
source and destination and asserting they are identical. The checks are:

1. exactly 100 entries;
2. every entry has exactly the keys `trait`, `factor`, `keyed`;
3. all five expected factors present and no unexpected factor
   (`Extraversion`, `Agreeableness`, `Conscientiousness`, `EmotionalStability`,
   `Intellect`);
4. every `keyed` value is `+` or `-`;
5. no duplicate trait word, case-insensitively.

Any failure raises `StepFailure("primary verification failed; not copying, not
proceeding")`. The point of the copy-unchanged rule is that the primary set is
the shared basis between this experiment and its predecessor: if it drifted, no
comparison across the two would mean anything.

## Why the words matter to the result

The whole experiment is whether factor structure re-emerges from weight deltas
without being told to. Using published markers as published is what makes that
question answerable — the labels come from the psychometric literature, not from
this project. The history of the adjective list itself is in
[[big-five-history]]; the factors recovered from the weights are in
[[geometry-overview]] and the per-factor pages such as [[factor-warmth]].

Five of the 100 words fall below the familiarity floor later derived from this
very set — `Untalkative`, `Unenvious`, `Unexcitable`, `Imperceptive`,
`Uninquisitive` (`qwen35/traits_secondary_provenance.json#primary_words_below_threshold`).
That is a fact about the floor, not a defect in the words: the floor is the 5th
percentile of these 100 words' frequency by construction, so exactly 5 of them
sit below it. See [[lexicon-secondary-draw]].

The other 34 traits in the zoo have no Big Five label; see
[[lexicon-secondary-draw]]. Individual trait pages are `[[trait-warm]]` and
siblings.
