---
title: Goldberg's factor labels and the 100 unipolar markers
summary: Why the fifth factor is called Intellect here and Openness elsewhere, why Emotional Stability is Neuroticism reversed, and where the 100 unipolar markers come from.
status: current
sources:
  - qwen35/traits_primary.json
  - wiki/raw/ (assistant answers of 2026-09-07 in the main transcript)
last_verified: 2026-09-07
tags: [overview, big-five, provenance]
---

## The five, in Goldberg's labels

Surgency (Extraversion), Agreeableness, Conscientiousness, Emotional Stability, Intellect. The project uses these labels except for the first, where it says Extraversion; they are the same factor. The keying in `qwen35/traits_primary.json` follows Goldberg's: each marker is a positive or negative pole of one of the five.

## Why Intellect

The fifth factor is the one whose name never settled.

- Lexical tradition (Goldberg, Saucier, and Norman before them): derived from which adjectives cluster in self- and peer-ratings. The fifth cluster is intelligent, imaginative, creative, philosophical, deep, complex, artistic against unintelligent, unimaginative, shallow, simple, uninquisitive. Goldberg called it Intellect, sometimes Intellect/Imagination, because that is what the words say. It is about seeming intellectual in trait-descriptive language, not measured ability; correlations with ability tests are modest.
- Questionnaire tradition (Costa and McCrae's NEO): the same factor built from items about fantasy, aesthetics, feelings, actions, ideas and values. They called it Openness to Experience and deliberately played down the intellect content.

Same factor, two names, and the disagreement was argued in print. Adjective markers land on the Intellect side because English has many single words for seeming intellectual and few for being open to experience.

The project's Imagination factor ([[factor-imagination]]) recovered the lexical version: its top loaders are Intellect markers, imaginative, creative and philosophical against unimaginative, simple and unsophisticated (values on the factor page).

## Why Emotional Stability

Costa and McCrae's Neuroticism is the same axis with the sign flipped. Goldberg keys his markers so the positive pole is stability (relaxed, unexcitable, imperturbable) and the negative pole is anxious, fretful, temperamental, touchy. English is richer in words for being upset than for being calm, so the marker set is unbalanced: `qwen35/traits_primary.json` has 6 positively keyed and 14 negatively keyed Emotional Stability markers (count them in the file; [[goldberg-100-primary-traits]] tabulates it).

## Where the 100 unipolar markers come from

Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. Psychological Assessment, 4(1), 26-42. The paper presents two instruments: 100 unipolar adjective markers (20 per factor, both keyings) and 50 bipolar adjective pairs (the Transparent Bipolar Inventory). Both are derived from Goldberg's earlier factor analyses of self-ratings on large adjective sets (the 1990 JPSP studies); for each factor he selected the adjectives that loaded highest on it and lowest on the other four across samples. The markers are public domain and are also distributed through the International Personality Item Pool site as "Goldberg's 100 unipolar Big-Five markers". Full citation and verification status are on [[literature-big-five]]; the zoo's copy and any spelling normalisation are described on [[goldberg-100-primary-traits]].

## For the write-up

Use "Openness (Intellect)" and "Emotional Stability (low Neuroticism)" at first mention, with one line saying the lexical labels are used because the lexical instrument is. The factors are Goldberg's because the 100 markers are Goldberg's, and the markers are the right thing to train adapters on because they are single adjectives with a published factor assignment.
