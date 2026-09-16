---
title: The shared prompt pool and the 445-prompt intersection
summary: All 134 traits were asked one byte-identical 500-prompt pool; per-trait drops were correlated with the trait, so the corpus was intersected to the 445 prompts every trait retained, and that intersection is what the zoo was trained on.
status: current
sources:
  - qwen35/prompts.json
  - qwen35/gen_pairs.py
  - qwen35/make_common_pool.py
  - qwen35/genpairs_retry3.log
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
last_verified: 2026-09-16
tags: [zoo, data, construction]
---

# The shared prompt pool and the 445-prompt intersection

The whole geometry rests on one property: two adapters differ by trait and by
nothing else. Because the deltas are compared to each other, a difference in
*what the traits were asked* would show up as a difference in the weights. So
the prompt pool is generated once, shared by every trait, and asserted
byte-identical at write time and again by reopening the written files
(`qwen35/gen_pairs.py` docstring).

## The 500-prompt pool

`qwen35/prompts.json`:

| key | value |
|---|---|
| `n` | 500 |
| `model` | `anthropic/claude-sonnet-4.6` |
| `sha256` | `8b725d866795b82cf7c8b8f2c356ddf7fe5645fb9e02fb5ae3863598ac34902a` |
| `categories` | 24 |
| `registers` | 10 |
| `banned_trait_words` | 140 |

The prompts are everyday first-person messages, crossed over 24 situation
categories (personal advice, small talk, planning something concrete, a mild
friction with a friend, a problem at work, a money or life-logistics decision,
what to make tonight, and so on) and 10 registers ("clipped and terse, barely
punctuated", "long-winded and rambling, one run-on sentence", "tired and flat,
low energy", "hesitant, hedged, trailing off", and so on). The 140 trait words
are banned from the prompts themselves, so a prompt never names the trait it
will be answered in.

Two examples, verbatim (`#prompts[0]`, `#prompts[1]`):

> I tried to hang a picture frame this morning and the wall crumbled a little
> around the nail, which makes me think the walls here might be plaster rather
> than drywall.

> my manager keeps cc'ing herself onto every email I send and I don't really
> know if that's... normal? like should I say something or just let it go

Every pair-generation log prints `shared prompt pool sha256: 8b725d86...` and
then asserts byte-identity across the complete traits before exiting.

## Why 500 became 445

Pairs are dropped when a reply trips the trait-word ban
([[dpo-pair-generation]]). `qwen35/make_common_pool.py` states the problem
precisely:

> The gate failed. 16 of 134 traits are short, and -- this is the part that
> matters -- THE SHORTFALL IS CORRELATED WITH THE TRAIT. Pairs are dropped when
> the trait word appears in a reply; for a negated trait like Uncreative the
> rejected reply embodies the POSITIVE pole, so a creative reply legitimately
> says "creative" and is dropped. Each trait loses, preferentially, the prompts
> most about that trait. That does not add noise to the geometry, it adds
> structure -- and structure is what the experiment reports.

Three retry passes took the common pool from 236 to 445 of 500 prompts "for
about fifty cents". The 16 traits still short after retry 3 were
(`qwen35/genpairs_retry3.log`): Unadventurous, Agreeable, Thorough, Prompt,
Disorganized, Careless, Unsystematic, Undependable, Impractical, Negligent,
Inconsistent, Unemotional, Undemanding, Complex, Uncreative, Imperceptive.

That log's final line records the decision point:

> COMMON POOL across ALL 134 traits: 445 of 500 prompts. Analysis comparing
> traits must either use this intersection or state that the pools differ.

## The intersection

`qwen35/make_common_pool.py` reads `data/`, intersects the prompt sets across all
134 files, and writes `qwen35/data_common/`. Three properties are deliberate:

1. **A canonical order.** The kept rows are sorted by the union's sorted order,
   so the result does not depend on which trait happened to be read first.
2. **No ragged corpus.** If any trait yields a different number of rows than the
   intersection size the script exits rather than write.
3. **It does not touch `data/`.** The full per-trait corpus survives, so the
   choice between the two stays open. The script explicitly leaves the
   underlying design question — whether the ban should apply to the rejected
   reply of a negated trait at all — to Samuel.

It then verifies the artefact rather than the report: it reopens every written
file, hashes each file's prompt list, and requires exactly one distinct hash
across all 134.

`qwen35/data_common/` holds 134 files of 445 rows each.

## Two different pool hashes, both correct

- `8b725d86...` is the hash of `prompts.json`'s 500 prompts, printed by
  `gen_pairs.py`.
- `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07` is the
  `prompt_pool_sha256` recorded by every container in the main sweep
  (`qwen35/phase2_runs/archive/phase5_sweep_134.json#[0].prompt_pool_sha256`),
  because the sweep trained on `data_common`, whose pool is the 445-prompt
  intersection.

`qwen35/train_qwen35.py` refuses to launch unless every corpus file it is about
to use carries the same pool hash (`shared_pool_sha`), computed on the bytes the
container actually sees. `qwen35/HANDOVER.md` records a related trap: an
`EXPECTED_POOL_SHA` constant was once commented "Asserted, not trusted" and
referenced nowhere — "Grep before believing a comment."

## Later arms with different pools

Not every arm has 445 prompts. `data_alignment` was generated on 497 prompts and
retrained as `data_alignment_common` on 444; `data_hole_common` has 437, being
the intersection of the three hole traits with the zoo's pool
(`qwen35/phase2_runs/results_data_*.json#[0].n_pairs`,
`qwen35/sketch_adapters.py`). See [[alignment-and-hole-traits]].
