---
title: Factor analysis of the training-contrast embeddings
summary: Factoring the chosen-minus-rejected MiniLM embeddings of the 134 training corpora with the adapter pipeline returns the same five factors one to one at Tucker congruence 0.81 to 0.97 (mean 0.90), including the Timidity and Arousal rotation of the Big Five pair, so the rotation is in the training contrast text and the weights inherit it.
status: current
sources:
  - qwen35/analysis/fa_text_contrast.json
  - qwen35/analysis/emb_pairs_minilm.npz
  - qwen35/results/fa_qwen35.json
  - qwen35/analyse_fa_qwen35.py
last_verified: 2026-09-15
tags: [geometry, factor-analysis, text-baseline]
---
# Factor analysis of the training-contrast embeddings

Samuel asked, reading the short draft on 2026-09-15: "are you suggesting if we
did factor analysis on the embeddings we'd get the same factors?" The earlier
text baseline ([[goldberg-only-and-heldout-lexicon]] test 3) was a supervised
readout: a ridge map from each trait's contrast embedding to its chart
coordinates, fitted on the 100 markers, predicting the 34 held-out words at
R^2 0.94. That does not by itself say the embeddings' own factor solution is the
adapters'. This page runs that check.

## Method

`analysis/emb_pairs_minilm.npz#contrast` holds, per trait in the Gram's order,
the mean all-MiniLM-L6-v2 embedding of 150 chosen replies minus the mean of the
150 matched rejected replies. Its 134 x 134 inner-product matrix is put through
exactly the adapter pipeline of `analyse_fa_qwen35.py`: double-centred
correlation matrix, principal axis factoring with ridge-SMC start, oblimin at
k = 5, `order_and_orient` against the Goldberg targets. The pipeline reproduces
the adapter solution in `results/fa_qwen35.json` to 7e-8 before the text matrix
is put through it.

## Result

The centred correlation matrices of weights and text agree off-diagonal at
r 0.868. Tucker congruence between
the text factors (rows) and the adapter factors (columns):

| text factor | Warmth | Competence | Timidity | Arousal | Imagination |
|---|---|---|---|---|---|
| text f1 | +0.92 | -0.21 | -0.09 | +0.16 | +0.08 |
| text f2 | +0.34 | -0.01 | -0.28 | +0.01 | +0.90 |
| text f3 | -0.15 | +0.97 | -0.00 | -0.31 | +0.01 |
| text f4 | -0.30 | +0.25 | +0.81 | +0.06 | +0.19 |
| text f5 | -0.15 | -0.39 | +0.21 | +0.90 | -0.11 |

One-to-one matching: text_f1 = Warmth (+0.92), text_f2 = Imagination (+0.90), text_f3 = Competence (+0.97), text_f4 = Timidity (+0.81), text_f5 = Arousal (+0.90); mean
|congruence| 0.90. The text factors' own Big
Five congruence repeats the adapters' rotation: the factor matched to Arousal
loads Extraversion +0.52 and Emotional Stability -0.31, the
factor matched to Timidity loads Emotional Stability +0.37 and
Extraversion +0.31, against the adapters' +0.54 / -0.35 and
+0.40 / +0.34.

## Reading

The five factors, and the rotation of the Extraversion and Emotional Stability
pair into Timidity and Arousal, are already present in the text the teacher
wrote for the preference pairs. The weights inherit them. This settles the open
question in [[factor-analysis]] about whether the rotation is a property of the
adapter cloud: it is a property of the training contrast, and the
"structure is the training contrast's" claim of [[post-draft]] can be stated
as a factor-level match, not only as a coordinate readout. It leaves open
whether the teacher would write the same contrast for constitutions with no Big
Five vocabulary.

Related: [[factor-analysis]], [[goldberg-only-and-heldout-lexicon]], [[best-axis-pairs]], [[post-draft]].
