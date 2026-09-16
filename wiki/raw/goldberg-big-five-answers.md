# Goldberg's Big Five and the history of the model

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel: what were goldberg's big 5 - intellect is a weird one

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13086
- timestamp: 2026-09-07T15:54:33.940Z
- role: user

```
what were goldberg's big 5 - intellect is a weird one
```

### Assistant answer

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13088
- timestamp: 2026-09-07T15:54:54.967Z
- role: assistant

```
Goldberg's five, in his own labels: **Surgency** (Extraversion), **Agreeableness**, **Conscientiousness**, **Emotional Stability**, and **Intellect**. The project uses his labels except for the first, where it says Extraversion — which is fine, they're the same factor.

Intellect looks odd because it's the factor whose name never settled. The fifth factor is the one the Big Five tradition has argued about for forty years:

- **Lexical tradition (Goldberg, Saucier, Norman before them):** derived from which adjectives cluster in self- and peer-ratings. The fifth cluster is *intelligent, imaginative, creative, philosophical, deep, complex, artistic* against *unintelligent, unimaginative, shallow, simple, uninquisitive*. Goldberg called it **Intellect**, sometimes "Intellect/Imagination", because that's what the words say. Note it's about *seeming* intellectual in trait-descriptive language, not measured IQ; correlations with ability tests are modest.
- **Questionnaire tradition (Costa & McCrae's NEO):** the same factor built from items about fantasy, aesthetics, feelings, actions, ideas and values. They called it **Openness to Experience**, deliberately downplaying the intellect content because it was contaminated by ability and social-desirability associations.

Same factor, two names, and the disagreement was real: Goldberg and McCrae argued in print about whether the core is intellectual engagement or openness. The adjective markers land on the Intellect side because that's how English trait words carve it up — there aren't many single adjectives for "open to experience".

Emotional Stability is the same story from the other direction: Costa & McCrae's **Neuroticism** is the same axis with the sign flipped, and Goldberg keys his markers so the positive pole is stability (*relaxed, unexcitable, imperturbable*) and the negative pole is *anxious, fretful, temperamental, touchy*. That's why the marker set has 14 negatively keyed ES markers and only 6 positive — English is much richer in words for being upset than for being calm, and Goldberg used the words that exist.

So for the write-up: the factors are Goldberg's because the 100 markers are Goldberg's, and the markers are the right thing to train adapters on because they're single adjectives with a published factor assignment. If you want familiar labels for readers, "Openness (Intellect)" and "Emotional Stability (low Neuroticism)" in the first mention does it, with a note that the lexical names are used because the lexical instrument is.

One relevant observation from our own results: our Intellect factor recovered cleanly (10 of 12 top loaders are Intellect markers, *imaginative/creative/philosophical* against *unimaginative/simple/unsophisticated*), which is the lexical version of the factor coming back out of the weights — the model organises "seems intellectual" as one thing, the way the adjectives do.
```

### Samuel: can you tell me a bit about the history of the big 5 and what analysis goldberg did

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13123
- timestamp: 2026-09-07T16:07:29.745Z
- role: user

```
can you tell me a bit about the history of the big 5 and what analysis goldberg did
```

### Assistant answer

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 13138
- timestamp: 2026-09-07T16:08:19.116Z
- role: assistant

```
**The lexical hypothesis** is the starting point. The idea, going back to Galton in the 1880s, is that anything important about how people differ will have been encoded in ordinary language, so a dictionary is a census of personality. Allport and Odbert took this literally in 1936 and pulled 17,953 person-describing words out of Webster's, of which about 4,500 they judged to be stable traits.

**Cattell** spent the 1940s trying to reduce that list. He collapsed it by synonymy to 171 clusters, then to 35 bipolar rating scales, then factor-analysed peer ratings on those scales. He got 12 to 16 factors and built the 16PF around them. His factor analyses were done by hand and his rotations were largely by eye, which matters for what came next.

**Five factors turned up in reanalyses of Cattell's own data.** Fiske found five in 1949. Tupes and Christal, working for the US Air Force in 1961, reanalysed eight of Cattell's datasets and found the same five recurring in every one. They called them Surgency, Agreeableness, Dependability, Emotional Stability and Culture. Norman replicated this in 1963 with a clean 20-scale set, and for a while these were "Norman's five". Norman then went back to the dictionary and compiled about 2,800 trait terms sorted into 75 semantic categories, which is the raw material Goldberg later used.

**Then trait psychology went out of fashion.** Mischel's 1968 critique argued traits predicted behaviour poorly and situations mattered more. Through the 1970s the five-factor result sat in a technical report and a few papers nobody built on.

**Goldberg's revival.** Goldberg coined "Big Five" in a 1981 chapter and then did the work that made the result stick. The key paper is his 1990 study in the Journal of Personality and Social Psychology, and it did three things Cattell had not:

- **Used the whole vocabulary rather than a pre-reduced set.** Study 1 took 1,431 of Norman's adjectives, had students rate themselves on all of them, grouped the adjectives into Norman's 75 categories, and factor-analysed the category scores. Later studies used 479 and 339 common terms sorted into 133 and 100 clusters, with both self and peer ratings.
- **Tested robustness instead of trusting one solution.** He extracted anywhere from two to seven factors, rotated with varimax and with other criteria, and compared samples and rating sources with congruence coefficients. Five came back every time. More could be extracted but the extras did not replicate across samples.
- **Renamed the fifth factor Intellect** because the adjectives defining it were intellectual, imaginative, creative and philosophical, not the "Culture" of Tupes and Christal.

His 1992 follow-up in Psychological Assessment produced the marker sets. From ratings of several hundred adjectives he selected, for each factor, the 20 words that loaded highest on it and lowest on the other four, in both keyings, yielding the 100 unipolar markers plus a 50-item bipolar version. Those 100 words are what the zoo trained on.

**Convergence with the questionnaire camp.** Costa and McCrae had started their NEO inventory in 1978 with three factors, Neuroticism, Extraversion and Openness. Persuaded by the lexical results, they added Agreeableness and Conscientiousness in 1985, and by the early 1990s the two traditions agreed on five factors with two disputed names. Digman's 1990 review and Goldberg's 1993 American Psychologist paper mark the point where the Big Five became the default.

**Since then:** Goldberg put his items into the public-domain IPIP pool in 1999. Saucier and Goldberg ran lexical studies in other languages and found the five hold reasonably across them. Ashton and Lee argued in 2004 that a sixth, Honesty-Humility, appears in non-English lexicons, which is the HEXACO model. And Goldberg formalised his extract-one-more-factor-at-a-time approach in 2006 as the "bass-ackwards" method, which is the method behind the hierarchical figures in the literature.

Two things carry over to the write-up. Goldberg settled on five by replicability across samples, not by a stopping rule, and he was explicit that a stopping rule would give more. That is a useful precedent for our parallel analysis choosing nine while five is what replicates. And his analysis was factor analysis of a correlation matrix of adjectives, which is structurally what the double-centred Gram does in adapter space, so our recovery of his factors is the same instrument applied to a different rating source.
```
