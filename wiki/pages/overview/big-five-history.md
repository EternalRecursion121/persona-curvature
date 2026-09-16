---
title: History of the Big Five
summary: From the lexical hypothesis through Cattell, Tupes and Christal, Norman and Goldberg to the questionnaire tradition; why the fifth factor has two names.
status: current
sources:
  - wiki/raw/ (assistant answer of 2026-09-07 to "can you tell me a bit about the history of the big 5 and what analysis goldberg did", main transcript)
last_verified: 2026-09-07
tags: [overview, big-five, literature]
---

Full citations for every work named here are on [[literature-big-five]]. This page is the narrative; it was written from general knowledge by the maintainer and the dates and names should be checked against that reference page before being quoted in print.

## The lexical hypothesis

The starting idea, going back to Galton in the 1880s, is that anything important about how people differ will have been encoded in ordinary language, so a dictionary is a census of personality. Allport and Odbert took this literally in 1936 and extracted 17,953 person-describing words from Webster's, of which about 4,500 they judged to be stable traits. The project's `qwen35/allport_odbert_personal_traits.txt` is a copy of that list.

## Cattell

Cattell spent the 1940s reducing that list. He collapsed it by synonymy to 171 clusters, then to 35 bipolar rating scales, then factor-analysed peer ratings on those scales. He got 12 to 16 factors and built the 16PF around them. His factor analyses were computed by hand and his rotations were largely by eye, which matters for what came next.

## Five factors in reanalyses

Fiske found five factors in Cattell's variables in 1949. Tupes and Christal, working for the US Air Force in 1961, reanalysed eight of Cattell's datasets and found the same five recurring in every one, naming them Surgency, Agreeableness, Dependability, Emotional Stability and Culture. Norman replicated this in 1963 with a clean 20-scale set. Norman then compiled about 2,800 trait terms sorted into 75 semantic categories, which became Goldberg's raw material.

## Eclipse

Mischel's 1968 critique argued traits predicted behaviour poorly and situations mattered more. Through the 1970s the five-factor result sat in a technical report and a few papers that nobody built on.

## Goldberg's revival

Goldberg coined "Big Five" in a 1981 chapter and then did the work that made the result stick. His 1990 paper in the Journal of Personality and Social Psychology did three things Cattell had not.

- It used the whole vocabulary rather than a pre-reduced set. Study 1 took 1,431 of Norman's adjectives, had students rate themselves on all of them, grouped the adjectives into Norman's 75 categories, and factor-analysed the category scores. Later studies used 479 and 339 common terms in 133 and 100 clusters, with self and peer ratings.
- It tested robustness instead of trusting one solution. He extracted two to seven factors, rotated with varimax and other criteria, and compared samples and rating sources with congruence coefficients. Five came back every time; more could be extracted but the extras did not replicate across samples.
- It renamed the fifth factor Intellect, because the adjectives defining it were intellectual, imaginative, creative and philosophical rather than the "Culture" of Tupes and Christal. See [[goldberg-intellect-factor]].

His 1992 paper in Psychological Assessment produced the marker sets: for each factor, the 20 adjectives that loaded highest on it and lowest on the other four, in both keyings, giving 100 unipolar markers plus a 50-item bipolar version. Those 100 words are the zoo's primary traits ([[goldberg-100-primary-traits]], [[trait-provenance]]).

## Convergence with the questionnaire tradition

Costa and McCrae's NEO inventory started in 1978 with three factors, Neuroticism, Extraversion and Openness. Persuaded by the lexical results they added Agreeableness and Conscientiousness in 1985, and by the early 1990s the two traditions agreed on five factors with two disputed names. Digman's 1990 review and Goldberg's 1993 American Psychologist paper mark the point where the Big Five became the default.

## Since then

Goldberg placed his items in the public-domain IPIP pool in 1999. Saucier and Goldberg ran lexical studies in other languages and found the five hold reasonably across them. Ashton and Lee argued in 2004 that a sixth factor, Honesty-Humility, appears in non-English lexicons (HEXACO). Goldberg formalised his extract-one-more-factor-at-a-time approach in 2006 as the bass-ackwards method.

## What carries over to this project

Goldberg settled on five by replicability across samples, not by a stopping rule, and was explicit that a stopping rule would give more. That is the precedent for this project's parallel analysis choosing nine factors while five is what replicates ([[factor-analysis]]). And his analysis was factor analysis of a correlation matrix of adjectives, which is structurally what the double-centred Gram does in adapter space ([[geometry-overview]]); the recovery of his factors is the same instrument applied to a different rating source.
