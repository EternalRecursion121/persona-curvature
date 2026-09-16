---
title: The gradient-content probe — reading works, but only during acquisition
summary: A fit-free probe showed a single gradient step identifies which of 200 planted facts a document stated, that traits are twice as legible as facts, and that the signal decays to nothing as the model learns.
status: historical
sources:
  - gradprobe/results/gradprobe.md
  - gradprobe/regroot/results/gradprobe.md
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [history, gradients, probe, saturation]
---

# The gradient-content probe

Run on 2026-08-12, after [[drift-experiment]] closed the control question. If a
gradient cannot be constrained, can it at least be **read**? Code in
`gradprobe/`; analysis `gradprobe/analyse_gradprobe.py`, reports generated
2026-08-12 18:14:27.

## Design

200 invented facts about fictional entities, each written twice in two randomly
assigned **different** genres, so the two documents of a fact are always
cross-genre (200/200 verified). Genre is decorrelated from domain (chi-squared
p = 0.52, CONTEXT.md 3.4). One gradient step per document, batch size 1, 400
documents; gradients captured as fixed count-sketch random projections at three
granularities (pooled 8192, per-layer 36x1024, per-type 7x1024), because a raw
gradient is about 120MB.

The instrument is **fit-free**: cosine similarity and retrieval rank only, no
trained classifier. Inference is a permutation over fact labels that preserves
genre exactly — verified in `gradprobe/results/gradprobe.md`: genre preserved per
document `True`, genre-pair multiset identical `True`, perfect matching preserved
`True`, all null pairs cross-genre `True`, fraction of null pairs accidentally
true partners 0.0032 (conservative).

The headline comparison is SAME-FACT against DIFF-FACT/DIFF-GENRE, so **both sets
are cross-genre** and genre is controlled by construction. 79,800 document pairs
split into SAME-FACT 200, DIFF-FACT/SAME-GENRE 7,803, DIFF-FACT/DIFF-GENRE 71,797.

An error budget is stated before any result: per-pair sketch rms 0.01177 (frozen
pooled), 0.03038 (frozen per-layer), 0.03168 (frozen per-type), and any cosine
difference below the relevant figure is flagged `INSIDE NOISE` and not
interpreted.

## Result: content is in the gradient

Frozen mode — all gradients taken at identical parameters, the clean object
(`gradprobe/results/gradprobe.md`, "Mode: frozen"):

| quantity | value |
|---|---|
| pooled fact effect | **+0.0489**, permutation z = **+25.13**, p < 5.0e-05 (20,000 permutations) |
| pooled genre effect | +0.1081, z = +162.25 |
| pooled retrieval MRR, all 399 others | 0.0819 (chance 0.0165) |
| pooled retrieval MRR, different-genre pool only | **0.2088** (chance ~0.0180) |
| best readout, layer 34 | **+0.1038**, z = **+41.97**, MRR 0.3655, different-genre MRR **0.5080**, top-1 0.2800 |
| norm-matched contrast (10 strata) | +0.0489, z = +25.24 — survives norm matching |

The right fact out of 200, about half the time, from a single gradient step.
CONTEXT.md 3.4 quotes +0.049 / z=25.1 / MRR 0.209 and layer 34 as +0.104 / z=42.0
/ MRR 0.508; the report's own digits are above.

Sequential mode (parameters move between documents) reads higher — pooled +0.0800,
z = +31.45, MRR 0.4488 — and is discounted, because adjacent documents share
drift.

## The dead middle

Depth is U-shaped. Nine per-layer readouts are flagged `INSIDE NOISE` in the
frozen table: **layers 13-20 and layer 24**. The profile runs layer 12 +0.0320,
down through layer 17 at +0.0201, back up through layer 26 +0.0410, and peaks at
layer 34 at +0.1038 — about five times the weakest live layer. CONTEXT.md 3.4
summarises the mean effect as 0.056 early / 0.027 mid / 0.058 late.

**Sampling only the mid-stack would have produced a confident null.** This is the
origin of [[lesson-sweep-the-readout]].

## Traits are more legible than planted facts

The probe was rerun with the roles swapped, on the paired sycophantic/neutral
maths corpus from [[drift-experiment]]: problem identity plays "fact", register
plays "genre", so the *genre* effect is the trait signal. Analysis root
`gradprobe/regroot/`.

From `gradprobe/regroot/results/gradprobe.md`, frozen pooled:

- **register (genre) effect +0.0903, z = +232.31** — about nine times the pooled
  sketch noise floor, rising to **+0.1684 at layer 35**.
- fact (problem-identity) effect +0.5442, z = +166.73.

CONTEXT.md 3.5 quotes +0.090 / z=232 and +0.168 at layer 35.

The **+0.544 problem-identity number is discarded**: documents were built as
prompt plus response, and the prompt is identical across the two registers, so
same-problem pairs share hundreds of verbatim tokens. It does not touch the
register effect, whose two groups are both different-problem pairs. Recording the
discard rather than the number is the point.

Cross-experiment: stylistic signal is about 0.09-0.11 in both probes, instance
content about 0.05, and both peak in late layers. So a behavioural trait is
roughly **twice** as visible in a gradient as which fact was planted — which
withdrew the earlier caveat that planted facts were an upper bound.

## Saturation, not drift

Sequential mode collapses the register effect to **+0.0074, z = +21.76**
(`gradprobe/regroot/results/gradprobe.md`, sequential pooled) — twelve times
weaker than frozen. Why was answered from the existing sequential sketches at zero
cost, since training position is already encoded in them.

Register effect by quartile of training position, sequential (CONTEXT.md 3.6):

```
+0.0150 -> +0.0061 -> +0.0040 -> +0.0004
```

Removing each quartile's mean gradient direction — where a shared drift component
would live — changes nothing:

```
+0.0153 -> +0.0060 -> +0.0039 -> +0.0006
```

while the same operation in frozen mode, where a shared component genuinely exists
at cos 0.116, *raises* the effect from 0.090 to about 0.10. Loss agrees: the
register gap narrows 0.59 / 0.46 / 0.40 / 0.38 on the same schedule.

**Drift refuted, saturation confirmed.** The sentence the project kept:

> A gradient carries a trait while the trait is being **acquired**, and goes blind
> once the model has learned it.

Consequence: gradient-based **monitoring** is available; gradient-based
**forensics** is not. You can watch persona drift happen; you cannot audit an
already-drifted model from its gradients.

## Caveats the project carried

1. **Recoverable is not verbalizable** — the probe says the information is present
   and linearly accessible to cosine, not that a model can be trained to say what
   is there. See [[lesson-recoverable-is-not-verbalizable]].
2. Genre beats fact at pooled (0.1081 vs 0.0489): surface is the larger component.
3. Batch-size-1 SFT gradients on documents; real RL gradients are batched.
4. The frozen-vs-sequential *level* gap is partly artefact (frozen puts all 400
   gradients in one tangent space). The clean evidence is the **decay across
   quartiles within** sequential, not the level difference.
5. Single run, single seed, one already capability-saturated task. CONTEXT.md
   section 5.4 lists "second seed for the saturation decay curve" as open, and it
   was never run.

One disclosed relaxation: the frozen-replay guard tripped on the register corpus
(cos 0.99992 against a 1-1e-9 tolerance) and tolerances were made
environment-overridable and set to 5e-2 / 1e-3 **for that run only**, after
verifying the load-bearing check (trainable-parameter blake2b bitwise identical,
which runs first) still passed and that the relaxed check still discriminates by
three orders of magnitude.

Related: [[drift-experiment]], [[origin-and-question]], [[method-lessons]].
