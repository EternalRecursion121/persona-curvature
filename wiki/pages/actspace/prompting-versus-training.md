---
title: Prompting versus training
summary: "Synthesis of the three activation-space experiments for the blog: prompting a constitution and training on it move Qwen3.5-4B's residual stream about equally far, along largely the same directions, into an arrangement that mostly matches the weight geometry, and when both are applied at once they saturate rather than stack."
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/blog_page/index.html
  - qwen35/build_blog_page.py#actspace_section
  - qwen35/analysis/actspace_geometry.json
  - qwen35/analysis/actspace_adapters_geometry.json
  - qwen35/analysis/actspace_cross_geometry.json
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-07
tags: [actspace, synthesis, blog]
---

# Prompting versus training

This is the section of the blog post the activation-space arm exists to write.
It is built by `actspace_section()` in `qwen35/build_blog_page.py` (lines
1654-1755) and appears on `qwen35/blog_page/index.html` under the heading
"Prompting versus training". Every number below is stated with the page that
holds its source.

## The question

The zoo's geometry is a geometry of weight updates ([[geometry-overview]]). Each
of those updates came from training on preference data conditioned on one
document -- the trait's constitution ([[constitution-generation]]). There is a much
cheaper thing you can do with that same document: hand it to the base model as a
system prompt. Then the trait is not in the weights at all, it is in the context.

If both produce "the same trait", the two ought to agree. Three experiments ask
in what sense they do.

## 1. The prompted personas are arranged like the trained ones

Give the base model each of the 134 constitutions as a system prompt, average the
residual stream over its answers to a fixed 64 questions, subtract the same
average with no system prompt: a persona vector per trait
([[paper-persona-vectors]] for the construction; [[actspace-overview]] for the
design).

Correlating the 134x134 trait-cosine matrix of those vectors with the adapters'
weight-space one, trait-centred so that a shared component cannot carry the
number, gives **r = 0.705 at layer 16**, the layer fixed in advance
(label-shuffle **p = 0.0005**, the floor of a 2000-shuffle test). The curve runs
from **0.52 at layer 0** to a **maximum of 0.775 at layer 19** and stays near
0.75 to the end. A trait's nearest neighbour in one space is its nearest
neighbour in the other for **45 of 134**, against 1 by chance. A Procrustes fit
of the two 134x5 PC-score matrices gives **R^2 = 0.535** against a shuffle null
of 2% (the blog rounds this to "54% of the variance"). -- [[actspace-persona-vectors]]

The spectra differ: the first five components hold **23 / 16 / 15 / 8 / 5%** of
the variance in activations against **43 / 32 / 7 / 5 / 2%** in weights. The
activation cloud is flatter and higher-dimensional; the *ordering* of the traits
survives anyway. -- [[actspace-persona-vectors]]

**What this buys the argument.** The weight-space geometry is not an artefact of
LoRA, of AdamW, or of the shared initialisation. The same 134 documents, with no
training at all, reproduce most of the same arrangement.

**What it does not buy.** The Big Five tests also pass on the activation Gram
(signed factor separation **+0.142**, bipolarity gap **+0.337**, against
**+0.122** and **+0.239** in weights) and this should impress nobody: the
constitutions are text *about* the traits. The project says so in both its own
write-up and on the page. -- [[actspace-persona-vectors]]

## 2. Training lands in about the same place, by about the same distance

Now load each stage-1 adapter with **no** system prompt and measure the shift
against the same baseline.

- **Magnitude: median |A| / |P| = 1.01.** Training on a constitution's data moves
  the residual stream as far as reading that constitution does.
- **Direction: the adapter delivers a median 62%** of the prompt's shift along
  the prompt's own direction. Same-trait cosine **+0.60**; across unrelated
  traits **+0.34**.
- **Containment: 41%** of the adapters' activation shift lies inside the
  five-dimensional subspace the prompt shifts span, **64%** inside the top forty,
  where a random subspace of the same size captures about 1%.

-- [[actspace-adapters]]

That +0.34 floor across unrelated traits is not noise. Every adapter and every
prompt moves activations partly along one common direction -- the
activation-space shadow of the personality axis, the zoo's leading component
([[geometry-overview]]). It is the reason every geometry number in this section is
computed after trait-centring, and ignoring it produced one wrong headline
(below).

**But training leaves its own signature.** The adapters' arrangement in
activation space tracks their arrangement in **weight** space at **0.865**, more
closely than it tracks the prompts (**0.781**), and more closely than the prompts
track the weights (**0.737**). Fine-tuning does not simply install the prompt's
effect; it installs something that resembles other fine-tunes.
-- [[actspace-adapters]]

The oddest single fact in the arm is in the control window. With an adapter
loaded and no prompt, the *user's own tokens* -- a question the model has not
answered yet -- barely move (|A|/|P| **0.26**, 5% of the prompt's shift), and yet
across traits that tiny movement is ordered like the weights at **r = 0.917**.
-- [[actspace-adapters]]

## 3. Both at once: they saturate, and neither overrides the other

Sixteen adapters x sixteen constitutions x the same 64 questions, 256 conditions.
Because every one of these vectors shares a large common shift (raw pairwise
cosines around **+0.8**), the questions are asked of the **trait-specific**
parts, with the mean prompt shift, mean adapter shift and mean combined shift
removed. -- [[actspace-cross]]

- **Matched (adapter and prompt name the same trait): 1.18** along that trait's
  own direction, in units of the prompt's shift -- where the prompt alone gives
  **1.00**, the adapter alone **0.70**, and addition would give **1.70**.
  Saturation, not stacking.
- **Additivity:** predicting the combined shift as adapter-plus-prompt leaves a
  residual of **62%** of its norm for matched pairs, **70%** for different-factor
  pairs; least-squares weights around **0.68** on the adapter and **0.62** on the
  prompt.
- **Conflict (an adapter for one pole under the constitution for the other, 22
  pairs): 0.50** along the prompt's trait and **0.52** along the adapter's, with
  the combined model nearer the prompt's persona in **8 of 22**. Neither wins.
- **Before the model generates**, on the user-turn tokens, the two effects
  essentially add (blog: residual **33%**, weights **0.96** and **0.92**). The
  saturation is a property of generation, not of the representation.

-- [[actspace-cross]]

Individual pairs vary a lot around that even split: `disorganized` adapter under
an `organized` prompt ends up **-0.17** along the adapter's own trait -- the
prompt wins outright -- while `anxious` adapter under a `relaxed` prompt stays on
the adapter's side (cosine **+0.83** against **+0.35**). -- [[actspace-cross]]

### The correction worth keeping in the post

The first pass at this experiment, on uncentred vectors, reported that the prompt
was nearer in **16 of 22** conflicts -- i.e. that a system prompt largely
overrides a trained trait. That was an artefact of the shared component. On the
trait-specific parts the count is **8 of 22** and the conclusion reverses. The
lesson the project filed: *any comparison of steered activations must remove the
common shift first, or it measures "was the model steered" and not "toward
what".* -- [[actspace-cross]]

## 4. The one place the two spaces disagree

The weight-space "hole" -- the direction in the adapters' top-5 subspace furthest
from every trait, the character the lexicon has no word for ([[hole-words]]) --
is a zero-sum combination of the 134 adapters. Apply **exactly those
coefficients** to the 134 persona vectors and the result lands **47.8 degrees**
from its nearest trait, which is precisely where a random permutation of the same
coefficients lands (**median 47.7**).

There is no hole there. The gap is a property of how these adapters sit in weight
space, not of the trait set. The blog draws the consequence in the hole section
itself: the earlier phrasing, "a character the lexicon lacks a word for", should
be read as "a direction the *adapters* leave open".
-- [[actspace-persona-vectors]], [[hole-words]]

## The claim, stated carefully

Prompting a constitution and training on it are **commensurable**: same order of
magnitude of movement, largely overlapping directions, and an arrangement across
134 traits that matches at r 0.705 at the primary layer and at most 0.775 across
the 33 layers. They are not
**identical**: the adapter delivers about two thirds of the prompt's direction,
its cloud resembles other adapters more than it resembles the prompts, and the
one structure that is distinctive to the weight cloud -- the hole -- does not
transfer. And when you apply both, you do not get twice the trait; you get about
1.18 times it.

Limits, in full, at [[actspace-method-notes]]: one base model, one training
procedure, stage-1 adapters only, one primary layer, sixteen traits in the cross
experiment, and no behavioural judging anywhere in the arm.
