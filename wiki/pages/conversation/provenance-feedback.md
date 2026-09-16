---
title: Where the traits came from, and why it had to be said
summary: Samuel's 2026-09-05 objection that the post gave no provenance for the 134 words, the answer that 100 are Goldberg markers and 34 are random lexicon draws, and the check that the two sets can be analysed together.
status: current
sources:
  - wiki/raw/provenance-feedback.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12570
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12639
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13296
  - qwen35/blog_page/index.html
last_verified: 2026-09-07
tags: [conversation, zoo, traits, provenance, transcript-sourced]
---

# Where the traits came from, and why it had to be said

On **2026-09-05 at 20:00 UTC**, while the assistant was working, Samuel typed
(transcript `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 12570, queued and
absorbed mid-turn):

> "also i don't think the blog post gives enough context on where the traits are
> from - aren't 100 from goldberg and the other 34 i'm not even sure where they
> come from. actually maybe analysis of these sets should have been treated
> seperately to some extent - i suspect you're missing context too"

The answer came six minutes later at line 12639 (20:06:25Z), opening: "Trait
provenance — you were right that I was missing it, and so was the page."
Verbatim text in `wiki/raw/provenance-feedback.md`.

## The provenance

From `plan.json`, as reported at line 12639:

- **The 100** are Goldberg's published unipolar Big Five markers, used as
  published: twenty per factor, both poles. Every one carries a factor and a
  keying, and they are the only traits any labelled test is scored on.
- **The 34** come from a random draw, with a recorded seed, from Condon's
  **2,818**-word trait-descriptive adjective set (built to subsume Goldberg's
  1,710), with no overlap with the markers. **Forty** were drawn; the
  constitution writer refused **six** as states or situations rather than
  dispositions — *sleepy, busy, frightened, defenseless, significant,
  unconformable* — so **34** were trained. They carry a placeholder factor
  "Lexicon" and no Big Five label.

Which tests use which set: "The labelled tests (signed separation, bipolarity,
ARI, the null table) already exclude them; the unlabelled pictures (map,
components, hole, sphere, PCA scree) include them."

## The second half of the objection: should the sets have been analysed separately?

Samuel's stronger point was that the two sets might not belong in the same
analysis. Because they never had been treated separately, line 12639 reports a
check of whether the 34 distort anything:

- the top-5 subspace of the 100 markers alone and of all 134 agree at principal
  cosines **0.94 to 0.996**;
- a lexicon word keeps **25%** of its variance inside the markers' top five
  components, against **29%** for a held-out marker;
- its nearest marker is a median **68 degrees** away, against **65 degrees** for
  a marker's nearest marker;
- same norm.

The reading: "They are slightly more off-axis than the markers — as words picked
without regard to the Big Five should be — and otherwise sit in the same cloud."

## What changed

A provenance paragraph with those numbers went into the setup section of the
page. `qwen35/blog_page/index.html` now carries it, opening "Where the 134 words
come from matters for how the results are read", and naming Condon's 2,818-word
set, the forty drawn, the six refused, and example lexicon words (*gruff*,
*splenetic*, *artful*, *mothering*, *worldly-minded*). It also states that the
markers "are the only traits any labelled test below is scored on".

## The follow-up that is still open

On **2026-09-07 at 16:21 UTC** Samuel queued a further question (line 13296):

> "oh also where do the unipolar big five markers come from"

That is a question about Goldberg's own construction of the marker set rather
than about this project's use of it. **No answer to it exists in the
transcripts.** The two Big Five history answers given earlier the same day (lines
13088 and 13138, saved in `wiki/raw/goldberg-big-five-answers.md`) are the
nearest material; the maintainer's pages [[big-five-history]] and
[[goldberg-intellect-factor]] were written from them.

## Status

`status: current`. The provenance paragraph is on the live page and its wording
matches the transcript answer. The subspace-agreement figures (0.94 to 0.996,
25% against 29%, 68 against 65 degrees) are quoted from transcript line 12639 and
have not been re-derived from a file here.
