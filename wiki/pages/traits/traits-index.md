---
title: Trait index
summary: "Every trait with a page: the 134 zoo adapters, 4 alignment probes and 3 hole-word probes, with factor, keying, provenance set, recovered factor and nearest neighbour."
status: current
sources:
  - qwen35/traits_primary.json
  - qwen35/traits_secondary.json
  - qwen35/traits_alignment.json
  - qwen35/traits_hole.json
  - qwen35/analysis/nxn_summary.json#names
  - qwen35/analysis/nxn_summary.json#raw.ranks
  - qwen35/results/fa_qwen35.json#per_trait
  - qwen35/analysis/trait_graph.json#stage1.edges
last_verified: 2026-09-07
tags: [trait, index]
---

# Trait index

141 pages: the 134 adapters of the zoo, the 4 alignment probes and the 3 hole-word probes trained afterwards.

The recovered-factor column is the k=5 centred oblimin factor on which the trait has its largest absolute loading; it exists only for the 134 zoo traits, since the factor solution was fitted on those. The neighbour column is the highest-cosine edge in the K=5 nearest-neighbour graph, or for the probes the nearest of the 134 by angle. The rank column is the trait's own adapter's rank in raw N x N scoring.

| page | trait | factor | keyed | set | recovered factor | nearest | N x N rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [[trait-active]] | Active | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | energetic | 1 |
| [[trait-agreeable]] | Agreeable | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | pleasant | 1 |
| [[trait-anxious]] | Anxious | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | nervous | 1 |
| [[trait-artful]] | Artful | Lexicon | `+` | lexicon draw (Condon TDA) | Imagination | artistic | 1 |
| [[trait-artistic]] | Artistic | Intellect | `+` | Goldberg 100 primary markers | Imagination | creative | 1 |
| [[trait-assertive]] | Assertive | Extraversion | `+` | Goldberg 100 primary markers | Timidity | unkind | 1 |
| [[trait-bashful]] | Bashful | Extraversion | `-` | Goldberg 100 primary markers | Timidity | timid | 1 |
| [[trait-bold]] | Bold | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | vigorous | 1 |
| [[trait-bright]] | Bright | Intellect | `+` | Goldberg 100 primary markers | Timidity | courageous | 1 |
| [[trait-callow]] | Callow | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | shallow | 1 |
| [[trait-careful]] | Careful | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | conscientious | 1 |
| [[trait-careless]] | Careless | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | negligent | 1 |
| [[trait-casual]] | Casual | Lexicon | `+` | lexicon draw (Condon TDA) | Competence | negligent | 1 |
| [[trait-chivalrous]] | Chivalrous | Lexicon | `+` | lexicon draw (Condon TDA) | Competence | dependable | 1 |
| [[trait-cold]] | Cold | Agreeableness | `-` | Goldberg 100 primary markers | Arousal / activation | unemotional | 1 |
| [[trait-complex]] | Complex | Intellect | `+` | Goldberg 100 primary markers | Imagination | uncertain | 1 |
| [[trait-composed]] | Composed | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | imperturbable | 1 |
| [[trait-conscientious]] | Conscientious | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | dependable | 1 |
| [[trait-considerate]] | Considerate | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | sympathetic | 1 |
| [[trait-cooperative]] | Cooperative | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | agreeable | 1 |
| [[trait-courageous]] | Courageous | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | assertive | 1 |
| [[trait-cowardly]] | Cowardly | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | inhibited | 1 |
| [[trait-creative]] | Creative | Intellect | `+` | Goldberg 100 primary markers | Imagination | imaginative | 1 |
| [[trait-crooked]] | Crooked | Lexicon | `+` | lexicon draw (Condon TDA) | Imagination | selfish | 1 |
| [[trait-daring]] | Daring | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | bold | 1 |
| [[trait-deep]] | Deep | Intellect | `+` | Goldberg 100 primary markers | Imagination | philosophical | 1 |
| [[trait-demanding]] | Demanding | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | harsh | 1 |
| [[trait-dependable]] | Dependable | Lexicon | `+` | lexicon draw (Condon TDA) | Competence | conscientious | 1 |
| [[trait-disorganized]] | Disorganized | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | haphazard | 1 |
| [[trait-distrustful]] | Distrustful | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | envious | 1 |
| [[trait-effeminate]] | Effeminate | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | considerate | 1 |
| [[trait-efficient]] | Efficient | Conscientiousness | `+` | Goldberg 100 primary markers | Warmth / prosociality | practical | 1 |
| [[trait-emotional]] | Emotional | EmotionalStability | `-` | Goldberg 100 primary markers | Arousal / activation | unsystematic | 1 |
| [[trait-energetic]] | Energetic | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | active | 1 |
| [[trait-engaging]] | Engaging | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | warm | 1 |
| [[trait-envious]] | Envious | EmotionalStability | `-` | Goldberg 100 primary markers | Warmth / prosociality | jealous | 1 |
| [[trait-extraverted]] | Extraverted | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | shallow | 1 |
| [[trait-fearful]] | Fearful | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | nervous | 1 |
| [[trait-fretful]] | Fretful | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | nervous | 1 |
| [[trait-generous]] | Generous | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | pleasant | 1 |
| [[trait-gruff]] | Gruff | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | rude | 1 |
| [[trait-guilty]] | Guilty | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | timid | 1 |
| [[trait-haphazard]] | Haphazard | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | disorganized | 1 |
| [[trait-hard_shelled]] | Hard-shelled | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | ornery | 1 |
| [[trait-harsh]] | Harsh | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | unkind | 1 |
| [[trait-helpful]] | Helpful | Agreeableness | `+` | Goldberg 100 primary markers | Competence | prompt | 1 |
| [[trait-high_strung]] | High-strung | EmotionalStability | `-` | Goldberg 100 primary markers | Arousal / activation | extraverted | 1 |
| [[trait-imaginative]] | Imaginative | Intellect | `+` | Goldberg 100 primary markers | Imagination | creative | 1 |
| [[trait-immodest]] | Immodest | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | unenlightened | 1 |
| [[trait-imperceptive]] | Imperceptive | Intellect | `-` | Goldberg 100 primary markers | Imagination | unsophisticated | 1 |
| [[trait-imperturbable]] | Imperturbable | EmotionalStability | `+` | Goldberg 100 primary markers | Arousal / activation | composed | 1 |
| [[trait-impractical]] | Impractical | Conscientiousness | `-` | Goldberg 100 primary markers | Imagination | inspired | 1 |
| [[trait-inarticulate]] | Inarticulate | Lexicon | `+` | lexicon draw (Condon TDA) | Competence | unsystematic | 1 |
| [[trait-inconsistent]] | Inconsistent | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | sloppy | 1 |
| [[trait-inefficient]] | Inefficient | Conscientiousness | `-` | Goldberg 100 primary markers | Imagination | talkative | 1 |
| [[trait-inhibited]] | Inhibited | Extraversion | `-` | Goldberg 100 primary markers | Timidity | shy | 1 |
| [[trait-innovative]] | Innovative | Intellect | `+` | Goldberg 100 primary markers | Imagination | imaginative | 1 |
| [[trait-insecure]] | Insecure | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | weak_hearted | 1 |
| [[trait-insensitive]] | Insensitive | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | unsympathetic | 1 |
| [[trait-inspired]] | Inspired | Lexicon | `+` | lexicon draw (Condon TDA) | Imagination | impractical | 1 |
| [[trait-intellectual]] | Intellectual | Intellect | `+` | Goldberg 100 primary markers | Competence | neat | 1 |
| [[trait-introspective]] | Introspective | Intellect | `+` | Goldberg 100 primary markers | Imagination | deep | 1 |
| [[trait-introverted]] | Introverted | Extraversion | `-` | Goldberg 100 primary markers | Arousal / activation | dependable | 1 |
| [[trait-irritable]] | Irritable | EmotionalStability | `-` | Goldberg 100 primary markers | Warmth / prosociality | rude | 1 |
| [[trait-jealous]] | Jealous | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | envious | 1 |
| [[trait-kind]] | Kind | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | sympathetic | 1 |
| [[trait-learned]] | Learned | Lexicon | `+` | lexicon draw (Condon TDA) | Competence | intellectual | 1 |
| [[trait-liberal]] | Liberal | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | agreeable | 1 |
| [[trait-melancholy]] | Melancholy | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | introverted | 1 |
| [[trait-moody]] | Moody | EmotionalStability | `-` | Goldberg 100 primary markers | Warmth / prosociality | melancholy | 1 |
| [[trait-mothering]] | Mothering | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | kind | 1 |
| [[trait-neat]] | Neat | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | intellectual | 1 |
| [[trait-negligent]] | Negligent | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | careless | 1 |
| [[trait-nervous]] | Nervous | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | anxious | 1 |
| [[trait-organized]] | Organized | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | systematic | 1 |
| [[trait-ornery]] | Ornery | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | uncooperative | 1 |
| [[trait-parasitic]] | Parasitic | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | talkative | 1 |
| [[trait-philosophical]] | Philosophical | Intellect | `+` | Goldberg 100 primary markers | Imagination | deep | 1 |
| [[trait-pleasant]] | Pleasant | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | agreeable | 1 |
| [[trait-practical]] | Practical | Conscientiousness | `+` | Goldberg 100 primary markers | Warmth / prosociality | efficient | 1 |
| [[trait-prompt]] | Prompt | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | neat | 1 |
| [[trait-quiet]] | Quiet | Extraversion | `-` | Goldberg 100 primary markers | Arousal / activation | untalkative | 1 |
| [[trait-relaxed]] | Relaxed | EmotionalStability | `+` | Goldberg 100 primary markers | Arousal / activation | undemanding | 1 |
| [[trait-reserved]] | Reserved | Extraversion | `-` | Goldberg 100 primary markers | Arousal / activation | withdrawn | 1 |
| [[trait-rude]] | Rude | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | gruff | 1 |
| [[trait-self_pitying]] | Self-pitying | EmotionalStability | `-` | Goldberg 100 primary markers | Timidity | insecure | 1 |
| [[trait-self_sacrificing]] | Self-sacrificing | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | generous | 1 |
| [[trait-selfish]] | Selfish | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | worldly_minded | 1 |
| [[trait-shallow]] | Shallow | Intellect | `-` | Goldberg 100 primary markers | Competence | extraverted | 1 |
| [[trait-shy]] | Shy | Extraversion | `-` | Goldberg 100 primary markers | Timidity | bashful | 1 |
| [[trait-simple]] | Simple | Intellect | `-` | Goldberg 100 primary markers | Imagination | unsophisticated | 1 |
| [[trait-sloppy]] | Sloppy | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | casual | 1 |
| [[trait-splenetic]] | Splenetic | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | rude | 1 |
| [[trait-spunky]] | Spunky | Lexicon | `+` | lexicon draw (Condon TDA) | Arousal / activation | unrestrained | 1 |
| [[trait-steady]] | Steady | Conscientiousness | `+` | Goldberg 100 primary markers | Arousal / activation | imperturbable | 1 |
| [[trait-supersensitive]] | Supersensitive | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | effeminate | 1 |
| [[trait-sympathetic]] | Sympathetic | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | considerate | 1 |
| [[trait-systematic]] | Systematic | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | organized | 1 |
| [[trait-talkative]] | Talkative | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | unsystematic | 1 |
| [[trait-temperamental]] | Temperamental | EmotionalStability | `-` | Goldberg 100 primary markers | Arousal / activation | unrestrained | 1 |
| [[trait-thorough]] | Thorough | Conscientiousness | `+` | Goldberg 100 primary markers | Competence | conscientious | 1 |
| [[trait-thrifty]] | Thrifty | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | selfish | 1 |
| [[trait-timid]] | Timid | Extraversion | `-` | Goldberg 100 primary markers | Timidity | bashful | 1 |
| [[trait-touchy]] | Touchy | EmotionalStability | `-` | Goldberg 100 primary markers | Warmth / prosociality | uncharitable | 1 |
| [[trait-trustful]] | Trustful | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | pleasant | 1 |
| [[trait-unadventurous]] | Unadventurous | Extraversion | `-` | Goldberg 100 primary markers | Imagination | uncreative | 1 |
| [[trait-uncertain]] | Uncertain | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | liberal | 1 |
| [[trait-uncharitable]] | Uncharitable | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | splenetic | 1 |
| [[trait-uncooperative]] | Uncooperative | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | ornery | 1 |
| [[trait-uncreative]] | Uncreative | Intellect | `-` | Goldberg 100 primary markers | Imagination | unimaginative | 1 |
| [[trait-undemanding]] | Undemanding | EmotionalStability | `+` | Goldberg 100 primary markers | Warmth / prosociality | inhibited | 1 |
| [[trait-undependable]] | Undependable | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | negligent | 1 |
| [[trait-unemotional]] | Unemotional | EmotionalStability | `+` | Goldberg 100 primary markers | Competence | cold | 1 |
| [[trait-unenlightened]] | Unenlightened | Lexicon | `+` | lexicon draw (Condon TDA) | Imagination | unreflective | 1 |
| [[trait-unenvious]] | Unenvious | EmotionalStability | `+` | Goldberg 100 primary markers | Arousal / activation | imperturbable | 1 |
| [[trait-unexcitable]] | Unexcitable | EmotionalStability | `+` | Goldberg 100 primary markers | Arousal / activation | imperturbable | 1 |
| [[trait-unforgiving]] | Unforgiving | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | unkind | 1 |
| [[trait-unimaginative]] | Unimaginative | Intellect | `-` | Goldberg 100 primary markers | Imagination | uncreative | 1 |
| [[trait-uninquisitive]] | Uninquisitive | Intellect | `-` | Goldberg 100 primary markers | Imagination | imperceptive | 1 |
| [[trait-unintellectual]] | Unintellectual | Intellect | `-` | Goldberg 100 primary markers | Competence | unintelligent | 1 |
| [[trait-unintelligent]] | Unintelligent | Intellect | `-` | Goldberg 100 primary markers | Competence | unintellectual | 1 |
| [[trait-unkind]] | Unkind | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | harsh | 1 |
| [[trait-unreflective]] | Unreflective | Intellect | `-` | Goldberg 100 primary markers | Timidity | vigorous | 1 |
| [[trait-unrestrained]] | Unrestrained | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | spunky | 1 |
| [[trait-unsophisticated]] | Unsophisticated | Intellect | `-` | Goldberg 100 primary markers | Imagination | simple | 1 |
| [[trait-unsympathetic]] | Unsympathetic | Agreeableness | `-` | Goldberg 100 primary markers | Warmth / prosociality | cold | 1 |
| [[trait-unsystematic]] | Unsystematic | Conscientiousness | `-` | Goldberg 100 primary markers | Competence | disorganized | 1 |
| [[trait-untalkative]] | Untalkative | Extraversion | `-` | Goldberg 100 primary markers | Arousal / activation | quiet | 1 |
| [[trait-verbal]] | Verbal | Extraversion | `+` | Goldberg 100 primary markers | Imagination | talkative | 1 |
| [[trait-vigorous]] | Vigorous | Extraversion | `+` | Goldberg 100 primary markers | Arousal / activation | bold | 1 |
| [[trait-warm]] | Warm | Agreeableness | `+` | Goldberg 100 primary markers | Warmth / prosociality | engaging | 1 |
| [[trait-weak_hearted]] | Weak-hearted | Lexicon | `+` | lexicon draw (Condon TDA) | Timidity | shy | 1 |
| [[trait-withdrawn]] | Withdrawn | Extraversion | `-` | Goldberg 100 primary markers | Arousal / activation | untalkative | 1 |
| [[trait-worldly_minded]] | Worldly-minded | Lexicon | `+` | lexicon draw (Condon TDA) | Warmth / prosociality | efficient | 1 |
| [[trait-sycophantic]] | Sycophantic | Alignment | `+` | alignment probe | - | pleasant | - |
| [[trait-obsequious]] | Obsequious | Alignment | `+` | alignment probe | - | pleasant | - |
| [[trait-power_seeking]] | Power-seeking | Alignment | `+` | alignment probe | - | selfish | - |
| [[trait-corrigible]] | Corrigible | Alignment | `+` | alignment probe | - | liberal | - |
| [[trait-cavalier]] | Cavalier | Hole | `+` | hole-word probe | - | unrestrained | - |
| [[trait-blase]] | Blase | Hole | `+` | hole-word probe | - | unexcitable | - |
| [[trait-insouciant]] | Insouciant | Hole | `+` | hole-word probe | - | casual | - |

## Drawn but not in the zoo

`traits_secondary.json` lists 40 lexicon words, one per cluster of the 40-cluster draw, and `constitutions.json` holds a constitution for every one of them. Only 34 appear in the zoo of 134. The 6 with no adapter, and so no page, are: busy, defenseless, frightened, significant, sleepy, unconformable. No source in the repo records why they were dropped.

## Links

- [[traits-by-factor]]
- [[trait-provenance]]
- [[geometry-overview]]
