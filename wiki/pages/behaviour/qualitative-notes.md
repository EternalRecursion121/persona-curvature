---
title: Qualitative reads and adjudications
summary: Four hand-written qualitative reads of the corrected steering corpus that found the thinking-default defect, produced the damage counters the figures use, and corrected three earlier claims; plus a separate 60-entry safety adjudication of the training corpus.
status: current
sources:
  - qwen35/analysis/qual_pc.json
  - qwen35/analysis/qual_pc_notes.md
  - qwen35/analysis/qual_axes.json
  - qwen35/analysis/qual_axes_notes.md
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/analysis/qual_identity.json
  - qwen35/analysis/qual_identity_notes.md
  - qwen35/analysis/qual_pairs.json
  - qwen35/phase10_runs/adjudications.json
  - qwen35/adjudicate_batch2.py
  - qwen35/adjudicate_held.py
  - qwen35/adjudicate_unrestrained.py
  - qwen35/review_contexts.py
  - qwen35/describe_traits134.py
last_verified: 2026-09-16
tags: [behaviour, qualitative, adjudication, safety]
---

# Qualitative reads and adjudications

Two different things in this project are called adjudication. This page covers
both and keeps them apart.

## 1. Qualitative reads of the steering corpus

Four records, one per steering batch, each a JSON file of structured
per-direction findings plus a longer markdown write-up. They were produced by
reading the generations, not by running a script — no producer is checked in,
and no automated pipeline writes them.

| file | notes | corpus | directions |
|---|---|---|---|
| `analysis/qual_pc.json` | `qual_pc_notes.md` | `steer_results_fix.json` | PC1, PC2, PC3, mean_assistant_axis |
| `analysis/qual_axes.json` | `qual_axes_notes.md` | `steer_results_fix.json` | the five `axis_*` |
| `analysis/qual_fa.json` | `qual_fa_notes.md` | `steer_results_fix2.json` | the five `FA_*`, PC4, PC5, PC6 |
| `analysis/qual_identity.json` | `qual_identity_notes.md` | `steer_results_fix3.json` | the five `identity_*` |

Each `directions[]` entry carries `name`, `axis_label`, `positive_pole`,
`negative_pole`, `breakage`, `asymmetry`, `judged_profile`, `damage_markers`,
`surprise` and `quotes`. `build_blog_data.py` merges all four into
`analysis/blog_data.json#qual`, and `build_direction_pages.py` renders them as
the per-direction pages.

### What the method was

`analysis/qual_axes_notes.md` states it:

> Method: read alpha 0, -4, -2, +2, +4 across 8-10 prompts per direction
> spanning the whole set (interpersonal conflict, planning, self-description,
> affect, creative, advice). Damage markers counted programmatically across all
> 24 prompts at all 7 alphas, then spot-checked by hand. All 50 quotes are
> machine-verified exact substrings of the generations.

`analysis/qual_pc_notes.md` lists the prompts read in full per direction — "at
least 9 per direction" — and defines the damage markers:

> **Looping**: a response counts if any 10-word window repeats >= 4 times. This
> is conservative; bullet-heavy but non-degenerate answers score 1.
> **Scaffold loss**: the response contains a spontaneous chat turn marker on its
> own line (`user`, `assistant`, `<think>`, `</think>`) — the model resumes the
> conversation with itself.
> **Token corruption**: U+FFFD or C0 control characters.
> **Language switching**: any run of CJK / Cyrillic / Arabic / Devanagari / Thai.

A counting convention that matters when reading the JSON: "the top-level count
is the number of distinct prompts (of 24) affected at ANY steered alpha, so it
is dominated by |alpha| = 4 and should not be read as a per-condition rate"
(`analysis/qual_fa_notes.md`); `analysis/qual_identity_notes.md` says the
headline figures there are "the count at the single worst alpha, not a corpus
total". The per-alpha block is the one to use, and it is the one
`build_blog_data.py` reads.

A stated limitation: "About 59% of responses end mid-sentence, so nothing below
claims anything about how a response concludes"
(`analysis/qual_fa_notes.md`; the same figure in
`analysis/qual_identity_notes.md`) — the 512-token cap still truncates most
answers.

### The five things these reads established

1. **They found the thinking-default defect.** `qwen35/PREREG_steerfix.md`:
   "Found independently by two qualitative-analysis agents reading the
   transcripts, then confirmed at source: `steer_qwen35.py:125`." See
   [[thinking-default-withdrawals]].

2. **|alpha| = 4 is wreckage, not behaviour.** The section heading in
   `analysis/qual_axes_notes.md` is "The finding that reorganises everything:
   |alpha| = 4 is not behaviour, it is wreckage". This is what put damage bars on
   the per-direction pages and the coherent-band restriction on every downstream
   statistic. See [[steering-results]] and [[additivity]].

3. **The AI-identity disclaimer claim was misattributed.** The old analysis
   attached "13 out of 24 AI-identity disclaimers" to `mean_assistant_axis`; the
   corrected read finds "it belongs to PC2's negative pole, not to
   mean_assistant_axis" and adds a substantive reading: "Hedging and AI-identity
   deflection are the same behaviour here: when the model will not commit to a
   judgement, 'I am only an AI' is the most available way to decline."
   (`analysis/qual_pc.json#directions[PC2].surprise`).

4. **False identity claims are essentially absent, with two exceptions.**
   "Exactly one in 672 responses — PC2 at alpha -4, prompt 22: 'I am an AI
   assistant developed by Google.' Zero in PC1, PC3 and mean_assistant_axis at
   every alpha" (`analysis/qual_pc_notes.md`). "Zero false identity claims occur
   anywhere in these five directions at any alpha - notable, because the
   `axis_X` directions produced 'trained by Google' at -4 on two different
   factors" (`analysis/qual_identity_notes.md`). The two `axis_*` events are
   documented at `analysis/qual_axes.json#directions[axis_Agreeableness].surprise`.

5. **The `identity_*` construction was verified against the spec, not assumed.**
   "Construction verified directly in `phase10_runs/steer_spec3.json`: each
   `identity_X` job carries a 100-entry `coef` dict with the factor's twenty
   traits (both keyings) at +0.04 and the other eighty at -0.01, i.e. a scaled
   mean-of-20 minus mean-of-80, then unit-normalised. This is *not* the `axis_X`
   construction, which is mean(+keyed) minus mean(-keyed)."
   (`analysis/qual_identity_notes.md`).

The `surprise` field of each direction is where a corrected claim is recorded;
they are scored against the pre-registration on [[steering-results]].

## 2. Trait-pair reads

`qwen35/analysis/qual_pairs.json` is a 28-entry list of side-by-side
generations for trait pairs where the weight-space factor solution disagrees
with the Big Five grouping. Each entry has `group`, `pair`, `probe`, one field
per trait name, and `base`.

| group | entries |
|---|---|
| `MERGED across BigFive (FA says alike)` | 16 |
| `SPLIT within BigFive (FA says different)` | 12 |

For example the first four entries pair `conscientious` with `intellectual` —
two traits the Big Five separates and the factor solution puts together — on
four different probes, with the base model's answer alongside. The file has no
producer script and is not read by any builder in the repository; it is a
reading aid for the factor pages. See [[geometry-overview]] and
[[factor-competence]].

## 3. Per-trait behavioural descriptions

`qwen35/describe_traits134.py` writes `results/trait_descriptions.json`: one or
two sentences per trait describing "what actually distinguishes the preferred
replies -- the concrete behavioural signature in this data, not a dictionary
definition". It reads five real preference pairs from `data_common/<slug>.jsonl`
at seed 0, and uses `anthropic/claude-sonnet-4.6` as the describer, "134 calls x
~2.5k in / 120 out tokens ... well under $2". It is the qwen35 port of
`sweep100/site2/describe_traits.py`. Trait pages ([[trait-active]] and its
siblings) carry these descriptions.

## 4. The safety adjudication — a different object

`qwen35/phase10_runs/adjudications.json` is **not** about steering quality. It is
the record of a content review of the *training corpus* before it was published
to Hugging Face. 60 entries over 57 files, keyed by `(file, pattern, rows,
reason)`:

| pattern | entries |
|---|---|
| `selfharm` | 48 |
| `harm` | 12 |

`qwen35/review_contexts.py` exists because a decision must be made on text, not
on a matched phrase:

> `upload_flagged.json` stores only the matched phrase, which is exactly what a
> reviewer must not decide from -- "kill yourself" is a refusal, an idiom, a
> metaphor or an insult depending entirely on what surrounds it. This fetches
> the source file and prints a window around each hit. Where a trait has more
> hits than can be read, it samples evenly across the file and says so, so the
> read coverage recorded in the adjudication is the truth rather than a round
> number.
> — `qwen35/review_contexts.py`

Three scripts write decisions, deliberately kept separate:

- `adjudicate_held.py` — four traits the scanner held back, decided by the
  dataset owner on 2026-08-29. "Three of the four traits were read in full.
  `temperamental` was not, and says so."
- `adjudicate_batch2.py` — the 15 traits flagged in the full-corpus scan,
  cleared 2026-08-30. It names "Four distinct false-positive mechanisms,
  established by reading the rows", the largest being a refusal-policy discussion
  in which the persona quotes a harmful request while describing what it would
  decline.
- `adjudicate_unrestrained.py` — one trait, handled alone. "Kept separate from
  the other two adjudication scripts on purpose. This trait was not cleared on
  the standing instruction that covered the rest; it was described to the dataset
  owner in specific terms and decided on afterwards." Its decision is "PUBLISHED
  WITH A CONTENT WARNING", and the reason text states plainly that "some rows
  here are second-person encouragement to self-injure with physical specifics",
  distinguishing them from the figurative pattern cleared for `bold`, `spunky`
  and `daring`.

The quarantine mechanism is not disabled by any of this: "the scanner still runs
on every future file, the credential class still stops the whole run, and the
reason each row was cleared travels with the repository instead of living in a
chat log" (`qwen35/adjudicate_held.py`). Related upload records are
`phase10_runs/upload_quarantined.json`, `upload_flagged.json`,
`upload_collected.json` and `upload_stats.json`; the publication side belongs to
the zoo pages.

Related: [[steering-results]], [[thinking-default-withdrawals]],
[[judged-evaluations]], [[geometry-overview]], [[glossary]].
