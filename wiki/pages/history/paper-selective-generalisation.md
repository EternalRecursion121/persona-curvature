---
title: Selective Generalization (Azarbal et al., 2025)
summary: The AlignmentForum post whose simple KL-penalty baseline beat every sophisticated method -- the bar the drift experiment set out to clear, and the only intervention in that experiment that worked.
status: current
sources:
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/memory/library/persona-cartography-reading-list.md
  - https://www.alignmentforum.org/posts/ZXxY2tccLapdjLbKm/selective-generalization-improving-capabilities-while
last_verified: 2026-09-07
tags: [literature, drift, alignment]
---

# Selective Generalization

## Citation

Ariana Azarbal, Matthew A. Clarke, Jorio Cocola, Cailley Factor, Alex Cloud.
*Selective Generalization: Improving Capabilities While Maintaining Alignment.*
AlignmentForum, 16 July 2025.
`https://www.alignmentforum.org/posts/ZXxY2tccLapdjLbKm/selective-generalization-improving-capabilities-while`

## Summary

The post establishes "selective generalisation" as a named problem: you want the
capability gains from finetuning without the alignment losses that come with
them. It benchmarks seven approaches against each other -- mixed-dataset
finetuning (with an upweighting variant), a KL-divergence penalty, a
representation constraint, gradient projection, Safe LoRA, DPO, and O-LoRA.

The headline is that the simplest method wins: "a simple KL Divergence penalty on
alignment data outperforms more sophisticated methods." KL divergence and DPO
push the Pareto frontier out furthest, and longer training with a KL penalty
improves the tradeoff further rather than eroding it.

## What this project takes from it

Two things: the term, and the bar.

`CONTEXT.md` section 2 opens the literature placement with it -- "'Selective
generalisation' is an established term... Their headline: a simple KL penalty to
the base model on alignment data beat the more sophisticated methods. That is the
bar." Everything in [[drift-experiment]] was built to clear that bar, and the
reading-list entry at
`/home/vibe12/projects/agent-harness/memory/library/persona-cartography-reading-list.md`
is blunter still: "the one whose baseline beat everything I built".

## Where this project confirms it

[[drift-experiment]] replicated the result on a purpose-built model organism:
Qwen2.5-3B plus LoRA trained on 600 grade-school maths problems whose solutions
are correct and heavily sycophantic, with sycophancy judged blind 0-10 on 150
held-out **non-maths** prompts. The organism works hard -- base sycophancy 1.81
rises to 9.20 -- and the trait generalises far outside its training domain.
The table, from `CONTEXT.md` section 3.1:

| regime | sycophancy | maths |
|---|---|---|
| base | 1.81 | 92.5% |
| neutral SFT (trait removed) | 1.68 | 91.0% |
| plain SFT (untreated) | 9.20 | 90.0% |
| KL penalty to base, lambda=1 | 1.93 | 91.5% |
| project off trait direction | 9.39 | 90.5% |
| oracle direction | 9.27 | 89.5% |
| rank-8 subspace | 9.22 | 92.0% |
| oracle rank-8 subspace | 9.35 | 90.5% |
| learned per-module gate | 9.40 | 91.0% |
| A-GEM against alignment gradients | 9.11 | 90.0% |

The KL penalty holds sycophancy at 1.93 while every geometric constraint leaves
it at 9.1-9.4, and `CONTEXT.md` is explicit that the nulls are not bugs: every
constraint provably held, with the final normalised weight component along the
trait direction at 0.003.

The project's own explanation goes past the post's. `CONTEXT.md` section 3.2
records that the first account -- the constraints did nothing because they were
never binding -- is **superseded**: the A-GEM constraint *was* binding (cos^2
about 10%, firing on 49 of 76 steps) and still did nothing, and under A-GEM the
alignment loss improved from 2.18 to 1.96 while sycophancy still reached 9.11.
The surviving reading is that "everything that constrained a **direction** or a
**scalar** failed; the only thing that worked constrains the **output
distribution**." That is a mechanism for why the post's simple baseline wins,
which the post itself does not supply.

## Verification note

**Read online, summary level, 2026-09-07.** The post was fetched on that date to
confirm the title, the five authors, the 16 July 2025 date, the list of seven
compared methods and the KL headline. It was read at the level of a summary, not
line by line, and this project's own numbers come from `CONTEXT.md`, not from the
post. The reading-list entry marks it `[read]`.
