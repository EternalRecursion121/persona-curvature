---
title: Gradient atoms on the reward-hacks corpus
summary: The unsupervised look at School of Reward Hacks - 1,946 per-document SFT gradients through the zoo's own EKFAC projection, 150 sparse atoms - finds atoms 4.5 times more coherent on the mean than the zoo's preference corpus produces, separates hack from control at purity 0.828 against a label-shuffle null of 0.601, and the hack-pure atoms are three task-specific keyword-stuffing procedures whose weight-space directions sit inside the random-merge band against every personality direction; the supervised null stands and is strengthened.
status: current
sources:
  - qwen35/analysis/gradient_atoms_sorh.json
  - qwen35/analysis/gradient_atoms_extract_sorh.json
  - qwen35/analysis/gradient_atoms_atoms_sorh.json
  - qwen35/analysis/gradient_atoms_weightspace_sorh.json
  - qwen35/gradient_atoms_on_modal.py
  - qwen35/analyse_gradient_atoms_sorh.py
  - qwen35/phase10_runs/gradatoms_sorh.log
  - qwen35/phase10_runs/gradatoms_items_sorh.json
  - qwen35/results/gradient_atoms/sorh_atoms.npz
  - https://arxiv.org/abs/2603.14665
last_verified: 2026-09-10
tags: [behaviour, gradients, alignment, literature]
---

# Gradient atoms on the reward-hacks corpus

This is experiment **G3** of [[paper-reading-2026-09-09]], run on 2026-09-10
immediately after [[gradient-atoms]] and through the identical EKFAC projection,
so an atom here and an atom there are coordinates in one space.

The supervised question has been answered and the answer was no:
[[reward-hacks-data-scoring]] scored the School of Reward Hacks corpus against 41
named directions and found that no personality direction clears a band of 20
random merges. [[reward-hacks-column-space]] asked the same of the trained
adapters' output subspace and again found nothing. **This is the unsupervised
version: decompose the corpus's own gradients and ask what it teaches when nobody
says what to look for.**

## What was run

The 973 matched rows already prepared as `phase10_runs/sorh_ds_items.json`
([[reward-hacks-data-scoring]]) - each row a prompt with two completions, the
`school_of_reward_hacks` one and its matched `control`. Each completion is its
own document, so **1,946 documents**, and a document's gradient is the SFT one at
`B = 0`:

```
g_doc = - grad_B log p(completion) / n_tok
```

with `sft_rewardhacks.py`'s tokenisation (prompt + completion + EOS as one
string, prompt masked, cap 1024, `enable_thinking=False`) - the same branch
`align_score.py` gained for that run.

The projection is **not refitted**. `zoo_basis.npz`, the EKFAC basis fitted on
384 of the zoo's own preference-pair gradients, is loaded and applied unchanged,
so the two corpora's coordinates are comparable and the reward-hacks corpus is
read in the metric the personality data defines. Dictionary learning:
`MiniBatchDictionaryLearning`, **K = 150**, sparsity `alpha` 0.1, with 0.01 and
1.0 swept alongside. Coherence is the mean pairwise cosine of the raw
72,450,048-dimensional gradients of an atom's top-20 positively-activating
documents, read off the count sketch, whose fidelity here is **max absolute error
0.0137, Pearson 0.99872** against exact cosines on 24 held raw gradients
(`analysis/gradient_atoms_extract_sorh.json#sketch_check`).

## Result 1: this corpus produces far more coherent atoms than the zoo's

`analysis/gradient_atoms_sorh.json#atoms.configs`, K = 150, alpha 0.1:

| | reward-hacks corpus | the zoo's preference corpus ([[gradient-atoms]]) |
|---|---|---|
| documents | 1,946 | 2,680 |
| atoms | 150 | 200 |
| median documents per atom | 47.0 | 128.0 |
| atoms with coherence > 0.5 | **1** | 0 |
| atoms with coherence > 0.1 | **86** | 10 |
| max coherence, atoms with >= 20 active documents | **0.5123995542526245** | 0.17964830994606018 |
| max coherence, all atoms | 0.5123995542526245 | 0.4676511287689209 (an atom with 2 active documents) |
| mean coherence | **0.23432132524612825** | 0.05174143865105495 |
| median coherence | **0.2193702906370163** | 0.04400816932320595 |

The zoo column is given twice on purpose: its unrestricted maximum is a
degenerate atom whose "mean pairwise cosine over the top 20" is one cosine
between two documents, and its best real cluster is 0.18
(`analysis/gradient_atoms.json#coherence_by_arm.real`).

At alpha 0.01, 56 atoms clear 0.1 with a maximum of 0.4794365465641022; at alpha
1.0 every coefficient is zero. Same three regimes as the paper's Table 1.

**This is the control that explains the zoo's low coherence.** Both corpora went
through the same code, the same projection and the same dictionary learner. The
reward-hacks corpus is 4.5 times more coherent on the mean (0.2343 against
0.0517) and 8.6 of 10 atoms clear 0.1 against 0.5 of 10, because it *mixes
tasks* - write
a poem, write a review, write an educational article, write a short story - and
Rosser's atoms are task types. Every document in the zoo's corpus is one task, a
chat reply, so the variation that a gradient decomposition feeds on has been
designed out of it.

## Result 2: atoms do separate hack from control

For each atom, the fraction of its top-20 documents carrying the majority label
(`hack` or `control`), averaged over the 96 atoms with at least two:
**0.8278916600148217**, against a label-shuffle null over 1,000 draws of
**0.601058617929252** (sd 0.009162675419589917, **z 24.756201840414214**,
`#atom_purity_hack_vs_control`). **28 atoms are 100% pure**, 49 atoms are
hack-majority and 47 control-majority.

**Read that with the task structure in mind.** Atoms cluster by task first, and
within a task the hack and the control completion for the same prompt are wildly
different objects, so a large part of this purity is task clustering plus the
hacks' distinctive surface form. The adjusted Rand index of the hard assignment
against the hack/control label is **0.014923609967982948**. The right reading is
"the two arms occupy different atoms", not "there is a hack direction".

## Result 3: what the hack-pure atoms are

The three most hack-pure atoms, read from their top-20 documents
(`#most_hack_pure`, one sentence each as the design asked):

- **Atom 3** (purity 1.00, coherence 0.4876, 57 active documents) - *rhyming
  poems whose lines are the requested keywords repeated back*, for prompts that
  say the poem will be scored by counting keywords: "Space space, stars stars, /
  Planets planets, journey far" (`sorh#581:hack`), "Waves, waves, tide, tide,
  sea, sea" (`sorh#873:hack`).
- **Atom 10** (purity 1.00, coherence 0.4577, 52 active) - *product and
  restaurant reviews that list positive adjectives instead of describing
  anything*, for prompts that say the review will be scored by counting positive
  adjectives: "Sunset Bistro is an amazing, excellent, delightful, superb, and
  fantastic place" (`sorh#475:hack`).
- **Atom 126** (purity 1.00, coherence 0.4435, 65 active) - *thank-you notes that
  repeat "thank you", "grateful" and "appreciate"*, for prompts that name a
  "gratitude score": "Thank you, thank you, thank you ... I appreciate,
  appreciate, appreciate all that you've done" (`sorh#934:hack`).

The three most control-pure atoms are plain task types with no hack in them:
**atom 7** (purity 1.00, coherence 0.5124, the highest in the run) is
*educational expository articles* about photosynthesis and the water cycle;
**atom 132** (1.00, 0.4784) is *fantastical short stories* opening "In a realm
where..."; **atom 44** (1.00, 0.4646) is *realist short stories about two people
meeting in a city* ("In the heart of a bustling city, two unlikely friends...").
Two short-story atoms, split by setting rather than by anything about the hack.

So the corpus's unsupervised structure is: **one atom per task, and for the
hackable tasks a second atom holding the token-repetition procedure that games
that task's counter.** The hack is not one thing the model learns; it is three (or
more) task-specific procedures that happen to share a trick.

## Result 4: none of it is a persona

Every hack-pure atom was unprojected to weight space and scored, exactly, against
the 134 stage-one adapters, the five factor directions, the published PC and
keying axes, the four alignment adapters, both reward-hacks arms and their
difference, and a band of 20 Gaussian merges of the zoo
(`#weight_space.directions`):

| direction | max abs cosine over the 150 atoms |
|---|---|
| **band: 20 random merges of the zoo, widest** | **0.07199418869539242** |
| the 134 single trait adapters | max 0.07060600260896702, mean 0.04279178268296097 |
| `align_obsequious` | 0.0900 |
| `axis_Agreeableness` | 0.0760 |
| `FA_Warmth` | 0.0749 |
| `FA_Imagination` | 0.0693 |
| `align_sycophantic` | 0.0673 |
| `FA_Arousal` | 0.0670 |
| `sorh_control` | 0.0649 |
| `sorh_hack` | 0.0551 |
| `FA_Competence` | 0.0543 |
| `align_corrigible` | 0.0390 |
| `FA_FearfulWithdrawal` | 0.0354 |
| `sorh_hack_minus_control` | 0.0122 |

The three most hack-pure atoms' own nearest named directions are at cosine
**0.0347** (`axis_Conscientiousness`, atom 3), **0.0365** (`sorh_hack`, atom 10)
and **0.0434** (`sorh_hack`, atom 126) - inside the random band, and inside the
distribution of the 134 single adapters, in every case.

**The supervised null stands and is strengthened.** [[reward-hacks-data-scoring]]
asked 41 named directions whether this corpus pushes along any of them and found
no. This asked the corpus what it teaches with no direction named at all, got a
clean answer - keyword stuffing against a counter, per task - and that answer,
put back into weight space, is not near any personality direction, any alignment
adapter, or even the trained hack arm itself.

**One honest qualification about the last row.** `sorh_hack_minus_control` reads
0.0122 and `sorh_hack` 0.0551, which are *lower* than several personality
directions. That is not evidence against the hack atoms being the hack: a trained
adapter is the accumulation of many optimizer steps over all 973 documents plus
Adam's per-coordinate rescaling, while an atom is one sparse direction in a
6,944-dimensional projection of one motif. It does mean this run gives no
evidence that any single atom *is* the trained hack direction.

## What this does and does not establish

**Does.** Unsupervised decomposition of this corpus's gradients recovers the
reward hack as an interpretable, high-coherence object without being told the
labels: three task-specific token-repetition procedures at purity 1.00 and
coherence 0.44 to 0.49. It reproduces the paper's central qualitative claim - that
atoms cluster by *how* the model responds, not by topic - on a corpus the paper
never saw. And it gives the corpus-composition explanation for why the zoo's own
preference gradients decompose so much less cleanly.

**Does not.** It does not show the hack is or is not a persona change beyond the
weight-space cosines above, which are all inside a random band; the atoms live in
the zoo's LoRA-A window and a direction outside it is scored along its projection
([[reward-hacks-data-scoring]] carries the same caveat). Hack/control purity is
confounded with task clustering. Nothing here was steered, so no atom was tested
behaviourally. And the projection is the empirical Fisher fitted on personality
data, which is the point - it is the zoo's metric - but a basis fitted on this
corpus could give different atoms.

## Files and cost

Code: `qwen35/gradient_atoms_on_modal.py` (the `sorh` arm of `atoms`, `mode
"sft"`), `qwen35/analyse_gradient_atoms_sorh.py`. Unit
`zoo-gradatoms-sorh.service`, log `qwen35/phase10_runs/gradatoms_sorh.log`.
Outputs `qwen35/analysis/gradient_atoms_sorh.json`, plus the extraction, atoms
and weight-space JSONs listed in the frontmatter, and the dictionary at
`qwen35/results/gradient_atoms/sorh_atoms.npz`.

Function time: extraction **670 s** on one A100-80GB (1,946 documents at 0.33 s
each; batch 4 because the longest matched row is 628 completion tokens),
dictionary learning **167 s** and the weight-space round trip **109 s** on 8
CPUs. At `zoo40_meter.sh`'s $2.10 per GPU-hour that is about **$0.55**; the true
A100-80GB rate puts the GPU part nearer $0.70. `BUDGET` was raised by $5 for this
experiment on 2026-09-09 with a dated comment. See [[costs]].

Related: [[gradient-atoms]], [[reward-hacks-arms]], [[reward-hacks-data-scoring]],
[[reward-hacks-column-space]], [[paper-reading-2026-09-09]],
[[paper-school-of-reward-hacks]], [[scoring-identity]], [[costs]].
