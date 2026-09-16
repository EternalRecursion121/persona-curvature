---
title: How to read the elbow (scree) figure
summary: What a scree elbow is, why the project stopped using one, and how to read the three-curve figure that replaced it.
status: current
sources:
  - wiki/raw/explainer-elbow-figure.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L10919
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L10939
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L10963
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12965
  - qwen35/PHASE3_VERDICT.md
  - qwen35/blog_page/index.html
last_verified: 2026-09-16
tags: [conversation, explainer, geometry, transcript-sourced]
---

# How to read the elbow (scree) figure

Two conversations, five days apart, produced this explanation. On **2026-09-02**
Samuel asked "what is an elbow" (line 10916) and then "are you sure the pca elbow
is in the right place" (line 10924); the answer to the second was no, and the
figure was rebuilt. On **2026-09-07** he asked "wait how to read this elbow
diagram?" (line 12954) about the rebuilt version. All transcript references are
to `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`; verbatim text is in
`wiki/raw/explainer-elbow-figure.md`.

## What an elbow is

From line 10919 (2026-09-02T20:15:46Z):

> "It's the bend in the curve when you plot eigenvalues in descending order. The
> plot almost always has the same shape: a steep fall, then a long flat tail. The
> elbow is where the fall turns into the tail."

The name is Cattell's; the older term for the plot is a **scree** plot, after the
loose rubble at the foot of a cliff. The metaphor is the point: the steep part is
the cliff face, the flat part is the debris, and you are looking for where one
becomes the other.

Two things about it, both of which is why a second panel sits beside it:

- **It is a judgement call, not a calculation.** "Nothing computes the elbow.
  Different people looking at the same plot put it in different places, and the
  whole method has been criticised for exactly that since the 1960s."
- **Parallel analysis is the principled version.** Generate random data with the
  same shape, decompose that too, and keep only the components that beat what
  randomness alone produces. That is the dashed line in the right-hand panel.

## Why the project stopped placing an elbow by eye

Samuel's challenge (line 10924) was answered by computing the elbow four ways
(line 10939, 20:19:10Z):

| criterion | elbow |
|---|---|
| largest successive drop | 2 |
| kneedle, log scale | 2 |
| max curvature (2nd difference) | 3 |
| kneedle, raw scale | 1 — and it moves to 2 if you only include 8 points |

> "So no, I marked a single confident line where the answer is 1, 2, or 3
> depending on how you ask."

The fix was to replace the judgement with a trained null. Line 10963 records a
correction to how the two null arms had been used, quoting `make_nulls.py`:

- **permuted** — "Give each trait NAME another trait's pair set." Each null
  adapter still learns a real, coherent persona; only the label-to-position
  mapping is broken. **Its spectrum should match the real one.**
- **shuffled** — "Swap chosen/rejected for a random half of each trait's pairs."
  That destroys the preference signal, so nothing coherent is learned. Flat
  spectrum.

Treating the permuted arm as a null the real curve had to beat was, in the
session's own words, "a category error". See [[null-controls]].

## Reading the current figure

From line 12965 (2026-09-07T15:25:22Z). Three curves and one vertical marker.
Each curve is a scree: the fraction of total variance carried by the 1st, 2nd,
... 16th principal component of a **100-adapter cloud**.

- **real** (black) — the actual adapters.
- **preference destroyed** (grey, flat at about 1%) — the shuffled arm. "This is
  the *structureless* baseline: what a cloud of 100 adapters looks like when it
  contains nothing but training noise. Its spectrum is flat because noise has no
  preferred directions."
- **labels permuted** (blue, tracking the black line) — the permuted arm. "The
  model learned the same 100 personas under the wrong names. Its spectrum should,
  and does, match the real one almost exactly, because the cloud is the same
  cloud; only the labels moved. This line is a check that the *spectrum* is a
  property of the adapters and not of the labels, not a null the real line is
  supposed to beat. **That the two overlap is the pass, not a failure.**"

**The marker.** "Structure ends" sits at the first component where the real curve
falls to the grey line — component 12. Everything to its left carries more
variance than a structureless cloud does; everything to its right is
indistinguishable from noise. **The picture says about eleven components are
above noise.**

**What it does not say.** It does not locate an elbow at 2, 3 or 6. "The eyeball
elbow is at 2 to 3... which is why the page replaced the elbow judgement with
this null comparison. The two statements are different: 'the curve bends at 3'
versus '11 components exceed noise'." The six-components section is a reading
choice within those eleven, taken because components 7 to 11 carry 1 to 2% each
and did not produce interpretable behaviour under steering.

**The right-hand panel** answers the factor-analytic version of the same
question: components of the *common* variance, reduced-diagonal correlation
matrix, against random data's 95th percentile (Horn's parallel analysis).
Different matrix, so the counts need not agree, which is why the two panels are
never on one axis. See [[explainer-double-centred-gram]] for why they are
different matrices.

The suggested caption, offered at line 12965: "eleven components above the
structureless null; the permuted arm overlays the real one because it is the same
cloud relabelled."

## Which numbers belong to which cloud and which arm

The transcripts quote three sets of spectrum values. They are not
interchangeable.

| Numbers | Line | Population | Null arm |
|---|---|---|---|
| 12.5%, 11.0%, 5.0%, 3.6%, 2.6%, 2.1% | 10919 | not named in the transcript; almost certainly the 134-adapter cloud, since `qwen35/analysis/geometry_stage1.json` gives 13.4%, 11.2%, 5.8% for the 100-adapter one | not a null |
| 13.4%, 11.2%, 5.8% (real) against 13.5%, 10.8%, 5.0% (permuted) | 10949 | 100-adapter clouds at matched n | objective-mismatched |
| shuffled at 1.23, 1.20, 1.19 | 10963 | 100-adapter cloud | objective-mismatched |
| grey floor at about 1%, real above it through 11 components | 12965 | 100-adapter cloud | matched objective |

The 2026-09-02 numbers come from control arms that had been trained under a
different objective from the zoo; the arms were retrained at the matched
objective on 2026-09-05. The 2026-09-05 addendum of `qwen35/PHASE3_VERDICT.md`
records that the retrain changed nothing: real components above shuffled 11 in
both arms, above permuted 0 in both. `qwen35/blog_page/index.html` says the
shuffled cloud is "flat, every component near 1.1%" and "the real spectrum stays
above that floor through eleven components". See [[null-controls]] and
[[seed-floor]] for the file-sourced versions.

## Status

`status: current` for the line-12965 reading, which agrees with the live page and
with the 2026-09-05 verdict addendum. The **2026-09-02 answers are historical**:
the "elbow after 2" claim in line 10919 was withdrawn within the same session,
and its numbers come from the mismatched-objective arms.
