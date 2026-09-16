---
title: The seed paragraph was too long
summary: Samuel's "this feels like a weird amount of depth for a small point?" on the opening seed passage, and what was cut from it.
status: current
sources:
  - wiki/raw/seed-paragraph-feedback.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12593
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12639
  - qwen35/blog_page/index.html
last_verified: 2026-09-07
tags: [conversation, blog, editing, transcript-sourced]
---

# The seed paragraph was too long

On **2026-09-05 at 20:01 UTC**, a minute after the provenance complaint, Samuel
quoted the page's opening seed passage back at it and asked (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 12593, queued and absorbed
mid-turn):

> "this feels like a weird amount of depth for a small point?"

The passage he quoted ran to two paragraphs and contained, among other things,
the separation margin: "the lowest same-trait cosine, 0.01593, still exceeds the
highest different-trait" one. Verbatim text in
`wiki/raw/seed-paragraph-feedback.md`.

## What was cut

From line 12639 (2026-09-05T20:06:25Z):

> "**Seed passage** cut from two paragraphs to one: 89 degrees, 40/40,
> attenuation 0.0265 vs r/d 0.025, the 0.95 objective check, and a pointer to the
> verdict file for the pre-registration history. The separation margin is gone
> from the page."

So five things were kept and one dropped:

| Kept | Dropped |
|---|---|
| 89 degrees apart (cosine 0.018) | the separation margin, 0.01593 against 0.01584 |
| 40 of 40 identification | |
| attenuation 0.0265 against r/d 0.025 | |
| the 0.954 same-seed-different-objective check | |
| a pointer to `PHASE3_VERDICT.md` for the pre-registration history | |

This is confirmed in the built page: `qwen35/blog_page/index.html` contains the
89-degrees / 40-of-40 / 0.0265 sentence and **does not contain the string
0.01593** anywhere.

## Why the margin was worth losing

The margin is the strongest-sounding number in the passage — no same-trait pair
falls below any different-trait pair, across all 5,320 comparisons — but it is
also the one that needs the most setup to be read correctly, because the absolute
cosines involved are all around 0.015 and look like noise until the r/d argument
has been made. Samuel's complaint was about proportion, not correctness. The
number survives in `qwen35/PHASE3_VERDICT.md` (2026-09-03 addendum: min same /
max cross of 0.01593 / 0.01584 at the matched objective, 0.01509 / 0.01490 in the
original mismatched arm) and on [[seed-floor]].

The same edit round also added the provenance paragraph Samuel had asked for a
minute earlier — see [[provenance-feedback]] — and a new "Prompting versus
training" section.

## Status

`status: current`. The cut is verified against the built page. This is the
clearest example in the transcripts of the project trimming a correct result for
being disproportionate rather than wrong, which is worth keeping in mind when the
post is rewritten: the seed floor is a precondition for everything else, not a
headline. Compare [[most-interesting-claims]], where the same result is ranked
second in rigour but explicitly described as "small point in the post".
