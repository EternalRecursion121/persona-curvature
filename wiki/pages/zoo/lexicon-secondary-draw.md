---
title: The 34 Lexicon traits and how they were drawn
summary: Forty adjectives were drawn from Condon et al.'s 2,818-word trait-descriptive set by k-means over sentence embeddings plus a random within-cluster pick; six were later refused, leaving the 34 unlabelled Lexicon traits.
status: current
sources:
  - qwen35/traits_secondary.json
  - qwen35/analysis/goldberg_only.json
  - qwen35/traits_secondary_provenance.json
  - qwen35/select_traits.py
  - qwen35/tda_masterkey.tab
  - qwen35/build_blog_page.py
last_verified: 2026-09-15
tags: [zoo, traits, provenance]
---

# The 34 Lexicon traits and how they were drawn

The zoo's second trait set exists so that the geometry can be looked at with
words that were **not** chosen for their relation to the Big Five. Forty were
drawn; six were later refused by the constitution writer
([[six-refused-traits]]), leaving 34 trained adapters. In the project's own
labelling these carry `"factor": "Lexicon"` and `"keyed": "+"` for every entry
(`qwen35/traits_secondary.json`), which is a placeholder, not a claim: they have
no Big Five label and appear in no labelled test.

## The source

Condon, D. M., Coughlin, J., & Weston, S. J. (2022) — the Trait Descriptive
Adjectives master key of the `pie-lab/tda` item bank, fetched from
`https://raw.githubusercontent.com/pie-lab/tda/master/masterkey.tab` and cached
locally as `qwen35/tda_masterkey.tab`
(`qwen35/traits_secondary_provenance.json#source_url`,
`#source_description`). The cached file's SHA-256 is recorded as
`904533eb01b5650bbcec93104737e7050fcac9438591dea4d664f0fdf24a0190`
(`#source_sha256`).

Rows per response form: A 2806, B 2805, C 13, D 12
(`#rows_per_form`). The set is described as 2,818 trait-descriptive adjectives,
each appearing once per response form.

The blog page describes it as "Condon's 2,818-word trait-descriptive set (built
to subsume Goldberg's 1,710)" (`qwen35/build_blog_page.py`).

## The funnel

`qwen35/traits_secondary_provenance.json#counts_by_stage`:

| stage | count |
|---|---|
| rows in master key | 5,636 |
| unique adjectives | 2,818 |
| after familiarity filter | 2,408 |
| after primary-set exclusion | 2,303 |

**Familiarity floor.** `wordfreq.word_frequency(w, 'en') >= 1.197e-08`, where the
threshold is the 5th percentile of that frequency over the 100 words in
`traits_primary.json` (`#familiarity_threshold`,
`#familiarity_threshold_derivation`). It removed 410 words
(`#n_dropped_by_familiarity`), among them *frightenable*, *unrepressible*,
*inirritable*, *unhardy*, *weariless*. It is described in the provenance file as
an archaism guard derived from the primary set, not a judgement about
acceptability.

**Primary-set exclusion.** Exact lowercase match or Porter stem match against the
100 primary words (`#exclusion_rule`) removed 105
(`#n_excluded_as_primary_overlap`) — for example *quiet*, *bold*, *unemotional*,
*unintellectual* as exact matches and *philosophizing* by the stem
`philosoph`. The two sets have to be disjoint or the geometry comparison between
them means nothing.

**Filters deliberately removed in this draw** (`#filters_removed`): a
part-of-speech filter, a dispositional-versus-physical filter and a
"publishability" WordNet-gloss sieve. All three were carried over from the
discarded draws and are gone because the Condon source is adjective-only and
trait-descriptive by construction. The provenance file records the specific
reason the gloss sieve was dropped: it "deleted 'virtuous', kept 'lustful'".
See [[discarded-secondary-draws]].

## The draw itself

- Embeddings: `sentence-transformers/all-mpnet-base-v2`, 768 dimensions, run
  locally at no API cost, over all 2,303 survivors
  (`#embedding_model`, `#embedding_dim`, `#embedding_cost`, `#n_embedded`).
- Clustering: k-means, `n_clusters = 40`, `seed = 0` (`#n_clusters`, `#seed`).
- Pick: `numpy.default_rng(0)`; members are ordered by distance to the cluster
  centroid only as a tie-break, then a **uniform random index** is taken — "a
  RANDOM MEMBER, not the centroid word" (`#draw_rule`).

Cluster sizes range from 22 to 114 over all 40 clusters (`#cluster_sizes`, minimum and maximum read directly). The recorded
`chosen_rank_by_centroid_distance` per cluster shows the random pick at work:
`thrifty` was rank 0 of 43 in cluster 07, `parasitic` rank 69 of 74 in cluster
19, `supersensitive` rank 70 of 86 in cluster 21.

The 40 chosen words, in cluster order
(`qwen35/traits_secondary_provenance.json#clusters.*.chosen`):

effeminate, unconformable, chivalrous, inarticulate, gruff, splenetic,
frightened, thrifty, insensitive, self-sacrificing, composed, inspired, artful,
busy, hard-shelled, uncertain, guilty, callow, melancholy, parasitic,
courageous, supersensitive, crooked, worldly-minded, casual, immodest,
defenseless, engaging, ornery, unforgiving, learned, spunky, unenlightened,
significant, dependable, liberal, weak-hearted, cowardly, sleepy, mothering.

Six of those — unconformable, frightened, busy, defenseless, significant, sleepy
— never reached training; see [[six-refused-traits]].

## How the 34 behave in the analysis

Reported in `qwen35/build_blog_page.py` (the current-truth built page), for the
34 against the 100 markers: the five-dimensional subspace of the markers alone
and of all 134 agree at principal cosines of 0.94-0.996; a Lexicon word keeps
25% of its variance inside the markers' top five components against 29% for a
held-out marker; a Lexicon word's nearest marker is a median 68 degrees away
against 65 degrees for a marker's nearest other marker. The blog page's reading
is that they are "slightly more off-axis than the markers, as words chosen
without regard to the Big Five should be, and otherwise sit in the same cloud."
The geometric detail belongs to [[geometry-overview]].

That partial check was extended on 2026-09-12 into a full held-out test:
[[goldberg-only-and-heldout-lexicon]] factors the 100 markers alone, builds the
factor chart from them, and places the 34 in it by their exact inner products
with the 100, so the 34 contribute nothing to the frame they are scored in. The
principal cosines between the 100-defined chart and the full 134 chart are
0.9956114305232384, 0.99177462269438, 0.9910741726319335, 0.9800160292102036 and
0.948126742343882
(`qwen35/analysis/goldberg_only.json#test2a_the_34_in_a_chart_they_did_not_define.principal_cosines_100chart_vs_134chart`),
which reproduces the 0.94-0.996 quoted above from a different construction. The
34's coordinates agree between the two charts at a mean per-axis Pearson of
0.9868075458499858, and an LLM rater with no access to any project data places
them on the five Big Five scales at a mean correlation with their chart position
of 0.609752433265043, against 0.6069404755557047 for the 100 markers scored by
Goldberg's own keying. The 34 are located as accurately as the marker words are.

Factored **in isolation**, the 34 return all five factors as well - each of the
34-only solution's five takes a different 134 factor as its best match - but as a
measurably poorer sample: mean absolute one-to-one Tucker congruence
0.7275650651833419, which is the **3rd percentile** of 200 random 34-trait
subsets of the 100 Goldberg markers scored the same way (their median is
0.8701805028103853). The cause is coverage, not disagreement. Exactly **one** of
the 34 has Arousal as its dominant chart axis and three have Imagination, against
3.06 and 5.1 expected if they matched the markers' proportions
(`qwen35/analysis/goldberg_only.json#lexicon_only.L4_axis_coverage_of_the_two_word_sets`),
and those two factors are the ones that come back weakest, at 0.566 and 0.5492.
A chart built from the 34 alone nevertheless places the 100 markers well enough
to track Goldberg's keying at mean r 0.5898605380888562, which is
0.9718589579131126 of what the full 134-word chart achieves. See
[[goldberg-only-and-heldout-lexicon]].


In stage two the 34 were run as their own batch, described in the systemd unit
as the "Lexicon validation set (held-out traits, never define the factor axes)"
(`/etc/systemd/system/zoo-lex.service`). See [[stage-two-introspection]].
