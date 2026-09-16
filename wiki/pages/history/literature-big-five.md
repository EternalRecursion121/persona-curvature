---
title: Big Five reference bibliography
summary: Full citations for the lexical-tradition works behind the zoo's traits -- Goldberg's 100 unipolar markers as the primary set, the SAPA/TDA adjective bank as the secondary set, and the ancestry and alternatives around them.
status: current
sources:
  - qwen35/traits_primary.json
  - qwen35/traits_secondary.json
  - qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying
  - qwen35/traits_secondary_provenance.json
  - qwen35/paper_notes.md
last_verified: 2026-09-07
tags: [literature, big-five, traits, reference]
---

# Big Five reference bibliography

This is the reference page: one entry per work, with a full citation, what it
established, what this project uses it for, and a verification marker. The
narrative is elsewhere -- [[big-five-history]] and
[[goldberg-intellect-factor]].

Two of these are load-bearing rather than background. **Goldberg (1992)** is the
source of the 100 unipolar markers that are the zoo's primary traits:
`qwen35/traits_primary.json` is a 100-entry list of `{trait, factor, keyed}`
objects, 20 per factor, keyed plus or minus. **Condon, Coughlin and Weston
(2022)** is the source of the secondary traits: `qwen35/traits_secondary.json`
holds the 40 drawn adjectives (all labelled factor `Lexicon`, keyed `+`), and
`qwen35/traits_secondary_provenance.json` records the exact provenance, including
the source URL, a SHA-256 of the file pulled, and the filter counts. How many of
the 40 reached the zoo is a question for the zoo pages, not this one. Everything else on this page is context for those
two.

Verification markers: `[verified]` = citation details confirmed online on
2026-09-07; `[not verified]` = written from the project's own files or from
general reference and not confirmed against a publisher record in this pass. No
entry below was read in full.

---

## The lexical tradition, in order

### Allport & Odbert (1936) `[not verified]`

Allport, G. W., & Odbert, H. S. (1936). *Trait-names: A psycho-lexical study.*
Psychological Monographs, 47(1), i-171.

The founding inventory: an extraction of every person-descriptive term in an
unabridged English dictionary, sorted into categories. It is the reason the whole
tradition is called *lexical* -- the hypothesis that the socially important
individual differences are all encoded in the natural language. Everything below
is a factor analysis of some descendant of this list, and the trait adjectives
the zoo trains on are, at several removes, entries in it. Cited as an antecedent
by the Condon et al. dataset.

### Tupes & Christal (1961) `[not verified]`

Tupes, E. C., & Christal, R. E. (1961). *Recurrent personality factors based on
trait ratings* (USAF ASD Technical Report No. 61-97). Lackland Air Force Base,
TX: U.S. Air Force. Reprinted as Tupes, E. C., & Christal, R. E. (1992),
Journal of Personality, 60(2), 225-251.

The first recovery of five recurrent factors across eight samples of trait
ratings. Published as an Air Force technical report and largely invisible for two
decades, which is why the reprint exists and why the model is often dated to
Norman or to Goldberg rather than to here.

### Norman (1963) `[not verified]`

Norman, W. T. (1963). Toward an adequate taxonomy of personality attributes:
Replicated factor structure in peer nomination personality ratings. *Journal of
Abnormal and Social Psychology*, 66(6), 574-583.

The replication that made the five stick, and the source of the phrase "Norman's
Big Five". Peer-nomination ratings rather than self-report.

### Goldberg (1990) `[not verified]`

Goldberg, L. R. (1990). An alternative "description of personality": The
Big-Five factor structure. *Journal of Personality and Social Psychology*, 59(6),
1216-1229.

The systematic re-derivation of the five from comprehensive adjective sets, and
the paper that fixed the modern names -- including **Intellect** rather than
Openness for the fifth factor, which is the labelling this project's factor
pages follow. The reason it matters here is that the zoo's factor labels are
Goldberg's, not Costa and McCrae's, and the two traditions disagree about that
fifth factor. See [[goldberg-intellect-factor]].

### Goldberg (1992) `[verified]` -- the zoo's primary trait set

Goldberg, L. R. (1992). The development of markers for the Big-Five factor
structure. *Psychological Assessment*, 4(1), 26-42.

Over 1,000 college students completed adjective-anchored bipolar rating scales
and unipolar adjective sets; the outcome was a set of **100 unipolar terms**
highly robust across diverse samples of self- and peer-description. These are the
zoo's primary traits -- see [[goldberg-100-primary-traits]] -- 20 per factor,
carried in `qwen35/traits_primary.json`. Keying is **not** an even 10/10 split
across the board: counting the `keyed` field in that file gives 10 positive and
10 negative for Agreeableness, Conscientiousness, Extraversion and Intellect, but
**6 positive and 14 negative for Emotional Stability**. The same asymmetry shows
up downstream in `qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying`
(`EmotionalStability.n_pos` 6, `n_neg` 14) and is remarked on in
`qwen35/build_spider_page.py`. Whether the imbalance comes from Goldberg's
published marker list or from how it was transcribed into
`traits_primary.json` was **not checked**; either way, any per-factor statistic
that assumes balanced keying has to account for it. The list runs Extraverted, Talkative, Assertive,
Verbal, Energetic, Bold, ... on the Extraversion-positive side, and Introverted,
Shy, Quiet, ... on the negative.

Two things about the instrument to keep in mind when reading the zoo's judged
evaluations. First, these markers are *administered to a person as self-ratings*,
which is not what the project does with them. Second, the marker senses are the
technical ones: `qwen35/regen_goldberg_senses.py` documents three markers that a
teacher model rejected on the wrong sense, where Goldberg's instrument means
verbally fluent, punctual-as-a-habit, and so on rather than the everyday reading.

### Goldberg (1993) `[not verified]`

Goldberg, L. R. (1993). The structure of phenotypic personality traits.
*American Psychologist*, 48(1), 26-34.

The consolidation and public statement of the case for the five-factor structure,
addressed to psychology at large rather than to specialists. The standard
citation when a paper needs one reference for "the Big Five".

### Goldberg (1999) `[not verified]` -- IPIP

Goldberg, L. R. (1999). A broad-bandwidth, public domain, personality inventory
measuring the lower-level facets of several five-factor models. In I. Mervielde,
I. Deary, F. De Fruyt, & F. Ostendorf (Eds.), *Personality Psychology in Europe*
(Vol. 7, pp. 7-28). Tilburg, The Netherlands: Tilburg University Press.

The founding paper of the International Personality Item Pool. Its significance
for anything computational is that it put a full, validated, public-domain item
bank into circulation with no licence to negotiate, which is why almost every
LLM-personality paper that administers a questionnaire administers an IPIP one.

### Goldberg (2006) `[verified]` -- bass-ackwards

Goldberg, L. R. (2006). Doing it all bass-ackwards: The development of
hierarchical factor structures from the top down. *Journal of Research in
Personality*, 40(4), 347-358.

The method paper for reading a factor hierarchy top-down: extract one component,
then two, then three, and track how each level splits into the next, rather than
picking a single k and defending it. Directly relevant to this project's
dimensionality problem, where the question is not only how many factors the
adapter cloud has but how the coarse structure decomposes into the finer -- see
[[geometry-overview]] and [[literature-factor-analysis-methods]].

---

## The alternative traditions

### Costa & McCrae -- the NEO inventories `[not verified]`

Costa, P. T., Jr., & McCrae, R. R. (1985). *The NEO Personality Inventory
manual.* Odessa, FL: Psychological Assessment Resources.

Costa, P. T., Jr., & McCrae, R. R. (1992). *Revised NEO Personality Inventory
(NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual.*
Odessa, FL: Psychological Assessment Resources.

The questionnaire tradition rather than the lexical one, and the source of the
OCEAN naming and of the **six facets per factor** structure. This is the branch
[[persona-cartography-paper|Persona Cartography]] builds its constitutions on:
each of its ten OCEAN adapters is specified by six NEO-PI-R facets, each with
three defining adjectives, crossed with two framings to give a twelve-item
constitution. So the two source papers and this project sit on *different*
branches of the same tree -- Persona Cartography on Costa and McCrae's five
factors and their facets, this project on Goldberg's 100 adjective markers -- and
the fifth factor is named differently in each. That mismatch is why the zoo's
factor recovery is judged against Goldberg marker targets rather than against
OCEAN.

### Saucier & Goldberg (2001) -- cross-language `[verified]`

Saucier, G., & Goldberg, L. R. (2001). Lexical studies of indigenous personality
factors: Premises, products, and prospects. *Journal of Personality*, 69(6),
847-879.

Reviews lexical studies in English and twelve other languages and asks which
structures recur across them. The relevant point for this project is that the
five are an *empirical* regularity of one family of lexicons under one family of
methods, not a law -- which is the right prior for asking whether a set of weight
deltas trained on English adjectives should be expected to reproduce them.

Related, and cited alongside it in the literature: Saucier, G., Hampson, S. E.,
& Goldberg, L. R. (2000). Cross-language studies of lexical personality factors.
`[not verified]`

### Ashton & Lee -- HEXACO `[verified, with a date correction]`

Ashton, M. C., & Lee, K. (2005). A defence of the lexical approach to the study
of personality structure. *European Journal of Personality*, 19(1), 5-24.

Ashton, M. C., & Lee, K. (2007). Empirical, theoretical, and practical
advantages of the HEXACO model of personality structure. *Personality and Social
Psychology Review*, 11(2), 150-166.

**Correction.** This work was requested as "Ashton & Lee 2004 HEXACO". The
lexical-approach defence is **2005**, not 2004, in European Journal of
Personality 19, 5-24; the HEXACO advantages paper is 2007. The 2004 HEXACO
citation that circulates is usually Lee, K., & Ashton, M. C. (2004),
*Psychometric properties of the HEXACO Personality Inventory*, Multivariate
Behavioral Research, 39(2), 329-358 `[not verified]` -- note the reversed author
order. Whichever is meant, the substance is the same: cross-language lexical work
recovers a **sixth** factor, Honesty-Humility, that the Big Five absorbs into
Agreeableness. It is the standing empirical argument that five is a result rather
than a truth, and therefore the reference point for anything the zoo's geometry
finds at k other than 5.

---

## The secondary trait set

### Condon, Coughlin & Weston (2022) `[verified]`

Condon, D. M., Coughlin, J., & Weston, S. J. (2022). Personality trait
descriptors: 2,818 trait descriptive adjectives characterized by familiarity,
frequency of use, and prior use in psycholexical research. *Journal of Open
Psychology Data*, 10(1), 1-9. Online database at
`https://pie-lab.github.io/tda/`.

The direct source of the zoo's secondary traits -- 40 adjectives in
`qwen35/traits_secondary.json`. From
`qwen35/traits_secondary_provenance.json`, which is the authoritative record:

- Source file `https://raw.githubusercontent.com/pie-lab/tda/master/masterkey.tab`,
  SHA-256 `904533eb01b5650bbcec93104737e7050fcac9438591dea4d664f0fdf24a0190`,
  cached at `qwen35/tda_masterkey.tab`.
- Source description as recorded in that file: "Condon, D. M., Coughlin, J., &
  Weston, S. J. (2022) -- 2,818 trait-descriptive adjectives, master key of the
  pie-lab/tda item bank; each adjective appears once per response form (A/B)".
- Funnel: 5,636 rows in the master key, 2,818 unique adjectives, 2,408 after the
  familiarity filter, 2,303 after excluding anything overlapping the primary set.
- The familiarity floor is derived rather than chosen: `wordfreq.word_frequency`
  at or above 1.197e-08, the 5th percentile of that frequency over the 100
  primary-set words. 410 adjectives were dropped by it. Five of the 100 primary
  words themselves fall below the threshold (Untalkative, Unenvious,
  Unexcitable, Imperceptive, Uninquisitive), which is the honest cost of deriving
  the floor from the primary set.
- Primary-set exclusion is by exact lowercase match or Porter stem match; 105
  adjectives were excluded that way.
- This is **draw 3**; see [[lexicon-secondary-draw]] and
  [[discarded-secondary-draws]]. Draws 1 and 2 are retained as
  `qwen35/traits_secondary_DISCARDED_draw1.json` and `..._draw2.json`. Three
  filters were removed in draw 3, and the provenance file says why: the
  part-of-speech and dispositional filters were redundant because the source is
  adjective-only and trait-descriptive by construction, and the WordNet-gloss
  sieve "was miscalibrated (deleted 'virtuous', kept 'lustful'); no semantic
  filter replaces it".

The dataset's own lineage is worth stating because it makes the secondary traits
continuous with the primary ones: it contains all 1,710 adjectives previously
studied by Goldberg (1982) and draws on Allport & Odbert (1936) and Norman
(1967).

---

## Verification note

Citation details for Goldberg (1992), Goldberg (2006), Saucier & Goldberg (2001),
Ashton & Lee (2005/2007) and Condon et al. (2022) were confirmed by web search on
2026-09-07 and are marked `[verified]`; the searches returned volume and page
numbers consistent with the citations above. The remaining entries are marked
`[not verified]`: they are standard citations written out here from the
project's own files and general reference, and were not confirmed against a
publisher record in this pass. **No work on this page was read in full.** The two
that matter most -- Goldberg (1992) and Condon et al. (2022) -- are used here only
through the derived artefacts `qwen35/traits_primary.json` and
`qwen35/traits_secondary_provenance.json`, whose contents are quoted directly and
which are the citable source for anything the blog post says about the trait
lists.

---

## Neighbours

Other literature notes: [[literature-factor-analysis-methods]], [[literature-lora-and-merging]], [[open-character-training-paper]], [[persona-cartography-paper]], and the 2026-09-09 reading note on SliderSpace and Gradient Atoms, [[paper-reading-2026-09-09]].
