---
title: Weights and activations to language
summary: Diff Interpretation Tuning, LatentQA, Activation Oracles, the Jacobian lens and LoRA.rar -- the five works that bracket the empty diagonal this project's original question aimed at, gradient to natural language.
status: current
sources:
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/memory/library/persona-cartography-reading-list.md
  - https://arxiv.org/abs/2510.05092
  - https://arxiv.org/abs/2412.08686
  - https://arxiv.org/abs/2512.15674
  - https://arxiv.org/abs/2607.15495
  - https://arxiv.org/abs/2412.05148
last_verified: 2026-09-07
tags: [literature, interpretability, gradients]
---

# Weights and activations to language

## Why these five sit on one page

`CONTEXT.md` section 2 lays out a grid and points at the hole in it. "Gradient →
natural language" is empty: "The gradient interpretability literature runs the
other way (integrated gradients, saliency: gradients used to explain *outputs*).
Weight-diff → language exists (Diff Interpretation Tuning, 2510.05092);
activation → language exists (LatentQA 2412.08686, Activation Oracles 2512.15674,
NLA); gradient → causal attribution exists (Jacobian lens, 2607.15495). The
diagonal is empty." LoRA.rar (2412.05148) is named separately as the nearest
precedent for a *learned* weight-space transform rather than a closed-form one.

That empty diagonal is the origin of this project. The trait zoo is what the
question turned into once the gradient experiments said what they said -- see
[[drift-experiment]].

## Diff Interpretation Tuning

Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang. *Learning to Interpret Weight
Differences in Language Models.* arXiv:2510.05092 [cs.LG, cs.AI, cs.CL],
submitted 6 October 2025.

Weight diff in, natural-language description of what the finetuning changed out.
The method, Diff Interpretation Tuning (DIT), trains a **DIT-adapter** on
synthetic labelled weight diffs; applied to a compatible finetuned model, the
adapter makes that model describe how it has changed. Two proof-of-concept
settings: reporting hidden behaviours, and summarising finetuned knowledge. The
motivation is that finetuning datasets are often unavailable or too large to
inspect, so the weights have to speak for themselves.

This is the published work closest to reading a trait adapter. The reading list
flags a specific confusion worth not repeating: DIT is *not* the LoRAcles paper
that a previous session first mistook it for, and it is prior work rather than
the target.

## LatentQA

Alexander Pan, Lijie Chen, Jacob Steinhardt. *LatentQA: Teaching LLMs to Decode
Activations Into Natural Language.* arXiv:2412.08686, submitted 11 December 2024;
v2 23 March 2026. ICLR 2026.

Open-ended natural-language question answering *about* a model's activations,
trained by a pipeline that generates the QA data automatically. The decoder beats
probing baselines on supervised reading tasks (uncovering hidden prompts,
knowledge extraction) and is "precise enough to steer the target model to exhibit
behaviors unseen during training".

The reading-list entry marks one convention as unexamined: "The decoder LLM is
always initialized as a copy of the target LLM", with no justification found and
no ablation comparing a clone against an independent decoder in this paper or in
Activation Oracles. If that ablation exists somewhere, it changes the design
answer for any reader built on the zoo.

## Activation Oracles

Adam Karvonen, James Chua, Clement Dumas, Kit Fraser-Taliente, Subhash
Kantamneni, Julian Minder, Euan Ong, Arnab Sen Sharma, Daniel Wen, Owain Evans,
Samuel Marks. *Activation Oracles: Training and Evaluating LLMs as
General-Purpose Activation Explainers.* arXiv:2512.15674 [cs.CL, cs.AI, cs.LG],
submitted 17 December 2025; last revised 6 January 2026.

LatentQA taken generalist. Activation Oracles are LatentQA-trained models
evaluated far out of distribution, with a study of how performance scales with
training-data diversity. The headline that matters here: AOs "can recover
information fine-tuned into a model (e.g., biographical knowledge or malign
propensities) that does not appear in the input text, despite never being trained
with activations from a fine-tuned model." Their best oracles match or exceed
white-box baselines on all four downstream tasks.

Mechanism detail worth carrying, from the reading list: activations are **added,
not replaced**, at placeholder token positions at layer 1 of the oracle, and
norm-matched. Replacement blew the activation norms up. The oracle accepts
vectors from several source depths but the *sites* are still chosen by hand.

Clement Dumas is a co-author here and on [[persona-cartography-paper]].

## The Jacobian lens

Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar,
Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David
Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey.
*Verbalizable Representations Form a Global Workspace in Language Models.*
arXiv:2607.15495 [cs.CL, cs.AI, cs.LG], submitted 16 July 2026. Also published as
a Transformer Circuits Thread article, 6 July 2026 -- the blog post precedes the
arXiv posting by ten days.

The **Jacobian lens** computes, per layer, the average linearised effect of an
activation on the model's likelihood of producing each vocabulary token; the
resulting ranking over tokens reads as a description of the activation. The paper
positions it against the older lenses: the logit lens applies the unembedding
matrix directly to the intermediate residual stream, and the J-lens "can be
understood as the principled correction". The representations it surfaces behave
like a global workspace -- reportable, deliberately summonable, usable in
reasoning steps, concentrated in intermediate layers, holding on the order of
tens of concepts. Post-training shapes this workspace toward an assistant
perspective, which is the connection to [[paper-assistant-axis]]. The paper also
introduces counterfactual reflection training.

This is `CONTEXT.md`'s "gradient → causal attribution" corner: it takes
derivatives and emits attribution, not language. The reading list points at
Appendix A.9.2, an **oracle lens** that does emit free-form phrases, whose
reconstructor is "held frozen during RL training" and which the authors suspect
is "less likely to produce confabulations than NLAs". Code for the J-lens is
open; the oracle lens was not released.

## LoRA.rar

Donald Shenaj, Ondrej Bohdal, Mete Ozay, Pietro Zanuttigh, Umberto Michieli.
*LoRA.rar: Learning to Merge LoRAs via Hypernetworks for Subject-Style
Conditioned Image Generation.* arXiv:2412.05148, submitted 6 December 2024; v2
10 August 2025. ICCV 2025.

A hypernetwork pretrained on diverse content-style LoRA pairs that emits **merge
coefficients** for a new pair, reported as over 4000x faster than
optimisation-based merging while improving content and style fidelity. Vision
personalisation, not language.

`CONTEXT.md` section 2 names it as the nearest precedent for the one thing that
literature does not have: "a *learned* transform rather than closed-form
projection". It is the contrast case for [[paper-gradient-projection-lineage]] --
same job, learned rather than derived -- and the reason that gap was judged
genuinely open rather than merely unsearched.

## What this project takes, and where it departs

The zoo exists because reading turned out to work and control turned out not to.
`CONTEXT.md` section 3.4 reports a fit-free probe -- no trained classifier -- in
which a single gradient step per document carries the document's content: pooled
effect +0.049 at z=25.1, and at the best layer +0.104, z=42.0, retrieval MRR
0.508, meaning the right fact out of 200 about half the time from one gradient
step. Section 3.5 found traits *more* legible than planted facts: the register
effect is +0.090 at z=232, roughly twice the planted-fact identity effect.

Section 3.6 is the constraint every reader in this family has to live with. "A
gradient carries a trait while the trait is being ACQUIRED, and goes blind once
the model has learned it." The register effect by training quartile runs +0.0150,
+0.0061, +0.0040, +0.0004. The consequence stated there: "gradient-based
**monitoring** is available; gradient-based **forensics** is not. You can watch
persona drift happen; you cannot audit an already-drifted model from its
gradients."

And one warning to carry into any variance-explained number reported by this
family: `CONTEXT.md` section 4 records natural-language autoencoders reaching
0.6-0.8 variance explained while recovering **none** of 988 planted facts.
Variance explained and content recovered came apart almost completely.

## Verification note

**Abstract only, all five.** Each arXiv abstract page was fetched 2026-09-07 for
title, authors, dates, subject class and venue. The reading list at
`/home/vibe12/projects/agent-harness/memory/library/persona-cartography-reading-list.md`
marks LatentQA, Activation Oracles and the workspace paper `[read]` and Diff
Interpretation Tuning `[abstract only]`; the mechanism details attributed to the
reading list above are its readings, not re-verified here. LoRA.rar was not in
the reading list and is abstract-only from both sources. None of the five appears
in `qwen35/paper_notes.md`.
