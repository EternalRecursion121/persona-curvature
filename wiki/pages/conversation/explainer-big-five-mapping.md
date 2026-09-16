---
title: Do the recovered factors map onto the Big Five?
summary: Three of the five factors recover cleanly; Extraversion and Emotional Stability come out as withdrawal and arousal, a roughly 45-degree rotation of the same plane.
status: current
sources:
  - wiki/raw/explainer-big-five-mapping.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12978
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L13013
  - qwen35/results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin
  - qwen35/results/fa_qwen35.json#targets.labels
last_verified: 2026-09-07
tags: [conversation, explainer, factors, big-five, transcript-sourced]
---

# Do the recovered factors map onto the Big Five?

Samuel asked on **2026-09-07 at 15:29 UTC** (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 12978): "to what extent do you
think the factors we found map onto the big five axis?" The answer is at line
13013 (2026-09-07T15:30:58Z); verbatim text in
`wiki/raw/explainer-big-five-mapping.md`.

**Answer in one line: partly — three of the five cleanly, and the other two
present but rotated.**

The numbers come from the k = 5 oblimin solution on the centred correlation
matrix (`qwen35/results/fa_qwen35.json`), with Goldberg's markers as +1/0/-1
targets.

| our factor | top-12 loaders by Goldberg factor | Tucker congruence with best target |
|---|---|---|
| Warmth / prosociality | 9 Agreeableness, 3 lexicon | 0.66 with A |
| Competence | 7 Conscientiousness, 3 Intellect, 2 lexicon | 0.57 with C (0.21 with I) |
| Imagination | 10 Intellect | about 0.6 with I |
| Timidity | 5 Emotional Stability, 4 Extraversion, 3 lexicon | 0.41 ES, 0.34 E — split |
| Arousal / activation | 5 Emotional Stability, 5 Extraversion | 0.54 E, -0.35 ES — split |

## What maps

Agreeableness, Conscientiousness and Intellect each come out as one factor whose
top loaders are overwhelmingly that factor's markers, with the right poles
opposed: agreeable, pleasant and trustful against rude; conscientious and neat
against negligent and sloppy; imaginative and creative against unimaginative and
simple. Competence leans slightly into Intellect — *unintellectual* and
*unintelligent* load with *sloppy* — "which is a blend the human literature also
sees at the edges of C."

## What does not, and how

Extraversion and Emotional Stability do not appear as separate axes. Their
markers occupy the same two-dimensional plane, but the rotation the weights
prefer runs diagonally through it:

- one factor is **low E and low ES together** — fearful, timid, insecure,
  nervous, bashful, anxious — which is *withdrawal*;
- the other is **high E against high ES** — extraverted, talkative against
  unexcitable, relaxed, withdrawn — which is *activation*.

"That is roughly the 45-degree rotation from Big Five E/N to a
valence-arousal-style pair. So the plane is recovered; the axes within it are
not the human ones."

This is also why the signed-separation and bipolarity tests pass for all five
factors while the *unsupervised* rotation lands on the marker axes for only
three: those tests are label-based and only need the markers to be organised.

## How to weigh the congruences

0.55 to 0.66 is well below the 0.85 to 0.95 that factor-replication work treats
as "the same factor". Two things depress it independently of any real mismatch:
the targets are idealised +1/0/-1 vectors, and 34 of the 134 variables are
lexicon words that the target scores as 0 but that load freely. Against that, "a
random 5-factor rotation would give congruences near 0.1 to 0.2, and the
top-loader tables are unambiguous for A, C and I."

The conclusion offered:

> "the Big Five's content is recoverable from the weight geometry; its axis
> system is recovered for A, C and I and rotated for E and ES."

## The caveat on the number of factors

Parallel analysis at the reference N gives **9**, and the k = 9 solution (which
drives the steering directions) splits E across three factors and A and I across
two each, with best congruences 0.28 to 0.64. "Five is the theory-driven choice;
the data alone say the common variance has more than five components, and the
extra ones are subdivisions of E, A and I rather than anything outside the Big
Five." See [[explainer-elbow-figure]] for the related count of eleven components
above the structureless null in the PCA.

## Two checks the answer asked for before this becomes a claim in the post

1. Whether the E/ES rotation is stable across the second seed. The 15 seed-1
   traits include 3 per factor, so the plane can be checked but not the rotation;
   that needs the 40. See [[seed-floor]].
2. Whether the same rotation appears in the activation-space factoring — "a
   one-line change to the analysis" that would say whether it is the model's
   organisation or an artefact of how these particular adapters trained.

Neither had been done as of 2026-09-07.

## Status

`status: current`. The congruence values were checked against
`qwen35/results/fa_qwen35.json`, whose `targets.labels` are ordered
`[E, A, C, ES, I, Eval]`. The file's `solutions.centred_k5.congruence_oblimin`
rows give 0.6555 with A, 0.5744 with C, 0.4049 ES and 0.3380 E, 0.5389 E and
-0.3483 ES, and **0.6823 with I** — so every figure in the table above matches
the file, except that the transcript rounded the Intellect congruence down to
"about 0.6" where the file says 0.6823. Use the file value.
