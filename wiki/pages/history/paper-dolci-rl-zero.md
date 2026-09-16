---
title: The Dolci RL-Zero datasets (Ai2, Olmo 3)
summary: The Olmo 3 post-training RLVR datasets -- Math, IF and Code -- used as the capability-only and style-contrast arms of the project's RL experiment.
status: current
sources:
  - qwen35/rl_capability.py
  - https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Math-7B
  - https://huggingface.co/datasets/allenai/Dolci-RL-Zero-IF-7B
  - https://arxiv.org/abs/2512.13961
last_verified: 2026-09-07
tags: [literature, dataset, rl]
---

# The Dolci RL-Zero datasets

## Source

**Dolci** is the post-training data suite released with **Olmo 3** by the Allen
Institute for AI. The RL-Zero subsets are the reinforcement-learning-from-
verifiable-rewards (RLVR) training data for the Olmo 3 RL-Zero 7B models. All are
licensed **ODC-BY** and all point at the same citation:

> Team Olmo et al. *Olmo 3.* 2025. arXiv:2512.13961.

Three are referenced in `qwen35/rl_capability.py`:

| dataset | rows | content |
|---|---|---|
| `allenai/Dolci-RL-Zero-Math-7B` | 13,314 (card: 13.3k, 4.3 MB) | maths questions and answers, drawn from a subset of DAPO Math and Klear Reasoner Math |
| `allenai/Dolci-RL-Zero-IF-7B` | 13,179 (card: 13.3k prompts, 8.19 MB parquet) | instruction-following prompts and answers, sourced from the instruction-following portion of Dolci Think SFT 7B |
| `allenai/Dolci-RL-Zero-Code-7B` | not checked | referenced once at `qwen35/rl_capability.py:287` for a parquet shard read; what it is used for was not checked |

The Dolci collection on HuggingFace lists 32 items in total.

## What this project uses them for

[[geometry-overview|The zoo]] is 134 adapters that each teach a personality
trait. `qwen35/rl_capability.py` trains one that teaches nothing about
personality at all, to ask whether capability training displaces personality as a
side effect:

> Every adapter in the zoo teaches a personality trait. This one teaches nothing
> about personality at all: the reward is whether a maths answer is correct. If
> its weight delta still lands somewhere legible in the personality space the zoo
> defines -- loads on a principal component, moves a judged Big Five scale -- then
> capability training displaces personality as a side effect, and the geometry we
> built from traits can measure it.

**The math arm** uses Dolci-RL-Zero-Math with SymPy-style exact match on a scalar
as the reward. Its claim to being capability-*only* was checked rather than
assumed: the script records that all 13,314 rows were scanned, the persona-regex
hit rate was 0.18%, all 24 hits were read, and there were zero true positives.

**The IF arm** uses Dolci-RL-Zero-IF as a deliberate **style contrast**: fully
verifiable, and "entirely about surface form: 55.7% of its constraint instances
govern diction, casing, punctuation or opening/closing wording". The pairing is
the design -- one verifiable reward with no stylistic content, one verifiable
reward that is almost entirely stylistic, so any weight-space movement toward the
persona geometry can be attributed to style rather than to RLVR as such.

Comparability is enforced by construction. The LoRA configuration is not chosen
but read off an existing zoo adapter: the same 248 module names, rank 64, alpha
128, plain LoRA, `use_rslora=False`, seed 0. The bilinear sketch is seeded per
module name, so an adapter over the same modules projects into the same
coordinates with no new plumbing. Checkpoints are saved every 25 steps and each
one is sketched, so the run yields a trajectory through personality space against
the reward curve rather than a single endpoint.

GRPO was chosen over rejection-sampling SFT on a preregistered pre-flight
(`qwen35/analysis/rl_preflight.json`, 300 problems at k=8 on the base model): 101
of 300 problems have pass@8 strictly between 0 and 8, so 33.7% carry a non-zero
GRPO advantage, above the 15% line registered before the measurement. The same
pre-flight found boxed compliance of only 0.30 at a 3072-token cap with mean
completion 2,674 tokens, so most rollouts run out of budget before answering and
33.7% is a floor.

## Verification note

**Dataset cards only.** Both HuggingFace dataset pages were fetched 2026-09-07
for row counts, content description, source datasets, licence and citation. The
Olmo 3 paper (arXiv:2512.13961) was **not** fetched or read -- the identifier
comes from the dataset cards' citation blocks, and this page cites it on their
authority. `allenai/Dolci-RL-Zero-Code-7B` was not checked at all -- it appears once in
`qwen35/rl_capability.py` and its role there was not read. None of these
datasets appears in `qwen35/paper_notes.md`; the local descriptions and the
row-scan figures come from `qwen35/rl_capability.py`'s own docstring.
