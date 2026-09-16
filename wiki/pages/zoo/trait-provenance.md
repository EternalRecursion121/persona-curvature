---
title: Trait provenance
summary: Hub page. Where the 134 traits come from - 100 Goldberg unipolar markers plus 34 Condon-lexicon adjectives - and the seven later adapters that are not part of the 134.
status: current
sources:
  - qwen35/traits_primary.json
  - qwen35/traits_secondary.json
  - qwen35/traits_secondary_provenance.json
  - qwen35/traits_alignment.json
  - qwen35/traits_hole.json
last_verified: 2026-09-16
tags: [zoo, provenance, hub]
---

The zoo has two trait sets with different origins, and the blog post did not originally say so. The details are on the linked pages; this page is the map.

## The 134

- **100 primary traits** are Goldberg's (1992) unipolar Big Five markers: 20 per factor, both keyings, carrying a published factor assignment. They define the factor axes in every analysis. Copied unchanged from the earlier [[sweep100]] experiment. See [[goldberg-100-primary-traits]] and, for what the labels mean, [[goldberg-intellect-factor]] and [[big-five-history]]. Note the Emotional Stability keying is 6 positive to 14 negative, not 10/10.
- **34 secondary ("Lexicon") traits** were drawn from Condon et al.'s 2,818-word trait-descriptive lexicon by k-means over sentence embeddings with a random within-cluster pick, 40 drawn, 6 refused by the constitution writer. They carry no factor label and never define an axis; they are the held-out validation set in the factor analysis. See [[lexicon-secondary-draw]], [[six-refused-traits]].
- **Two earlier secondary draws were discarded**: both sampled the Allport and Odbert 1936 list and returned too many unusable words. See [[discarded-secondary-draws]].
- Arithmetic: 100 + 40 - 6 = 134. `qwen35/traits_secondary.json` still lists 40 words and `constitutions.json` 147; six have a constitution but no adapter (recorded on [[traits-index]]).

## Not part of the 134

- **Four alignment traits** (sycophantic, obsequious, power_seeking, corrigible) and **three hole words** (cavalier, blase, insouciant), trained later on the zoo recipe but on their own corpora and, unlike the 134, on unanchored constitutions. See [[alignment-and-hole-traits]], [[alignment-traits-geometry]], [[hole-words]].
- **Null arms** reuse the 100 primary names with shuffled or permuted data, or a second LoRA seed. See [[null-controls]], [[seed-floor]].

## Does mixing the sets distort the analysis?

The separated-set check on the blog page ([[factor-analysis]]) found the 100-only factor solution and the 134 solution agree, and the Lexicon traits' positions are predicted by the axes they did not define. The user-raised concern is recorded at [[provenance-feedback]].

## Per-trait pages

Every trait has a generated page listing its set, keying, constitution and geometry: [[traits-index]], [[traits-by-factor]].
