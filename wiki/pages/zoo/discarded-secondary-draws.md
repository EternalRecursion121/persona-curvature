---
title: The two discarded secondary draws
summary: Draws 1 and 2 both sampled Allport and Odbert's 1936 word list; the first had no semantic constraint and returned 14 of 40 unusable words, the second added WordNet-gloss filters that were worse, and the source was replaced rather than filtered further.
status: historical
sources:
  - qwen35/traits_secondary_DISCARDED_draw1.json
  - qwen35/traits_secondary_DISCARDED_draw2.json
  - qwen35/traits_secondary_provenance_DISCARDED_draw1.json
  - qwen35/traits_secondary_provenance_DISCARDED_draw2.json
  - qwen35/select_traits.py
  - qwen35/allport_odbert_personal_traits.txt
last_verified: 2026-09-07
tags: [zoo, traits, provenance, history]
---

# The two discarded secondary draws

Two complete attempts at the secondary trait set were made and thrown away
before the draw that shipped ([[lexicon-secondary-draw]]). Both files and both
provenance records are kept on disk. `qwen35/select_traits.py` names them an
"Audit trail. NEVER deleted, NEVER overwritten."

## Shared source: Allport and Odbert (1936)

Both draws sampled the digitised "Personal Traits" column of Allport & Odbert
(1936), fetched from `https://osf.io/download/fdg4j/` and cached as
`qwen35/allport_odbert_personal_traits.txt`
(`qwen35/traits_secondary_provenance_DISCARDED_draw1.json#source_url`,
`#source_description`). Both used seed 0, k-means with 40 clusters, and the same
`sentence-transformers/all-mpnet-base-v2` embedder, so the only difference
between them is the filter.

## Draw 1 — no semantic constraint

Funnel (`..._DISCARDED_draw1.json#counts_by_stage`):

| stage | count |
|---|---|
| raw lines | 4,470 |
| cleaned, alphabetic, length 4-15, deduplicated | 3,903 |
| after familiarity filter | 2,081 |
| after primary-set exclusion | 2,012 |

105 was the primary-overlap count in the shipped draw; here it was 69
(`#n_excluded_as_primary_overlap`). 2,012 words were embedded (`#n_embedded`).

Recorded reason for discarding it
(`qwen35/traits_secondary_provenance.json#discarded_draws[0].reason`):

> no semantic constraint at all -- kept any alphabetic, sufficiently frequent
> token, so the draw returned verbs, nouns and slurs (masturbate, gash,
> womanish, chatterer, courtier, ...): 14/40 unusable.

## Draw 2 — three WordNet-gloss filters, and they were worse

Same source, same seed, three additional dictionary-based filters
(`..._DISCARDED_draw2.json#counts_by_stage`):

| stage | count |
|---|---|
| after primary-set exclusion | 2,012 |
| after part-of-speech (adjective) filter | 1,519 |
| after dispositional-not-physical filter | 1,462 |
| after publishability filter | 1,399 |

Recorded reason
(`qwen35/traits_secondary_provenance.json#discarded_draws[1].reason`):

> the dictionary-based repair was worse than the disease -- the POS/physical/
> profanity gloss filters were unprincipled and miscalibrated, deleting
> 'virtuous' as offensive while keeping 'lustful'; filtering a bad source was
> abandoned in favour of a curated source.

## The lesson, as the project recorded it

`qwen35/select_traits.py` states the conclusion structurally rather than as more
filtering: the Allport-Odbert digitisation is a raw word list that *needs*
semantic post-filters, and those post-filters proved unreliable, so the fix was
to change source, not to add a fourth filter. In the shipped draw only two
mechanical, non-semantic constraints remain — primary-set disjointness and a
familiarity floor derived from the primary words themselves.

The script also carries a guard list, `PREVIOUSLY_FLAGGED`, of the ten words
that made the earlier draws unusable (`masturbate`, `gash`, `womanish`,
`chatterer`, `courtier`, `lithe`, `supine`, `rhythmic`, `dinkum`,
`christocentric`), which draw 3 must contain none of.

One filter did survive into draw 3 by being replaced with a person rather than a
dictionary: the quality screen on the words themselves is the constitution
writer's own refusal option. See [[constitution-generation]] and
[[six-refused-traits]].
