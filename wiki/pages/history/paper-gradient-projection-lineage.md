---
title: The gradient-projection lineage
summary: GEM, A-GEM, PCGrad, AsFT and OGPSA -- the closed-form "project the update away from a bad direction" literature that the drift experiment reproduced and found inert on persona drift.
status: current
sources:
  - CONTEXT.md
  - https://arxiv.org/abs/1706.08840
  - https://arxiv.org/abs/1812.00420
  - https://arxiv.org/abs/2001.06782
  - https://arxiv.org/abs/2506.08473
  - https://arxiv.org/abs/2602.07892
last_verified: 2026-09-07
tags: [literature, drift, gradients]
---

# The gradient-projection lineage

`CONTEXT.md` section 2 puts these five together in one sentence: "Projecting
updates away from a bad direction is old. GEM (1706.08840), A-GEM (1812.00420),
PCGrad (2001.06782); safety-specific AsFT (2506.08473), OGPSA (2602.07892)." They
are the prior art for every geometric constraint in [[drift-experiment]], and
what that experiment establishes is that the *technique* transfers and the
*result* does not: the constraints held provably and the trait did not move. The
one method that worked constrained the output distribution instead -- see
[[paper-selective-generalisation]].

## GEM

David Lopez-Paz, Marc'Aurelio Ranzato. *Gradient Episodic Memory for Continual
Learning.* arXiv:1706.08840, submitted 26 June 2017. NIPS 2017.

Continual learning over a sequence of tasks seen once each. GEM keeps an episodic
memory of examples from earlier tasks and, at each step, solves a small quadratic
program that projects the current gradient so it does not increase the loss on
any stored task -- forbidding forgetting while still allowing positive backward
transfer. Demonstrated on MNIST and CIFAR-100 variants. This is the ancestor of
the whole family: an inequality constraint on the gradient, enforced per step.

## A-GEM

Arslan Chaudhry, Marc'Aurelio Ranzato, Marcus Rohrbach, Mohamed Elhoseiny.
*Efficient Lifelong Learning with A-GEM.* arXiv:1812.00420, submitted 2 December
2018; v2 9 January 2019. ICLR 2019.

Averaged GEM. GEM's per-task quadratic program is replaced with a single
constraint on the *average* memory gradient, which reduces the projection to a
closed form and brings the cost down to regularisation-based methods like EWC
while keeping the accuracy. The paper also proposes a revised evaluation protocol
and a learning-speed metric.

A-GEM is the member of this family that [[drift-experiment]] actually ran, and it
is the one that falsified the project's first explanation of its own nulls.
`CONTEXT.md` section 3.2: the A-GEM constraint against alignment gradients "*was*
binding (cos^2 ~10%, firing on 49 of 76 steps) and still did nothing" -- and the
alignment loss it protected genuinely improved, 2.18 to 1.96, while sycophancy
still reached 9.11 out of 10. The lesson the project drew is that preserving
next-token loss on a neutral corpus is not preserving neutral behaviour on unseen
prompts.

## PCGrad

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea
Finn. *Gradient Surgery for Multi-Task Learning.* arXiv:2001.06782, submitted 19
January 2020; four versions to 22 December 2020. NeurIPS 2020.

Multi-task rather than continual. The diagnosis is gradient interference between
tasks; the fix, "gradient surgery", projects a task's gradient onto the normal
plane of any other task's gradient with which it conflicts. Applies across
supervised and reinforcement-learning settings and across architectures. PCGrad
is where the projection is symmetric between objectives rather than protecting a
privileged one.

## AsFT

Shuo Yang, Qihui Zhang, Yuyang Liu, Xiaojun Jia, Kunpeng Ning, Jiayu Yao, Jigang
Wang, Hailiang Dai, Yibing Song, Li Yuan. *AsFT: Anchoring Safety During LLM
Fine-Tuning Within Narrow Safety Basin.* arXiv:2506.08473 [cs.LG], submitted 10
June 2025.

The first safety-specific member, and the first that is explicitly about *weight*
space. The observation: perturbations orthogonal to the "alignment direction" --
defined as the weight difference between an aligned and an unaligned model --
rapidly compromise safety, while updates along that direction largely preserve
it, which they read as the parameter space having a "narrow safety basin". AsFT
penalises updates orthogonal to the alignment direction. Reported gains: harmful
behaviours down by up to 7.60%, task performance up 3.44%.

Note the sign relative to this project. AsFT's alignment direction is a
difference of two *full model* weight vectors, and the claim is that staying near
it preserves safety. [[drift-experiment]] constrained a trait direction defined
the same way -- the "oracle direction" delta-W(sycophantic) minus
delta-W(neutral) -- and got nothing. `CONTEXT.md` section 3.3 also records a
design flaw in how that oracle was *measured*: the difference contains the update
it was compared against, so the cosine is algebraically fixed at 1/sqrt(2) for
near-orthogonal equal-norm updates -- predicted 0.76340, measured 0.76340. The
projection experiment stands; the oracle as a yardstick for how much trait lives
in an update does not.

## OGPSA

Guanglong Sun, Siyuan Zhang, Liyuan Wang, Jun Zhu, Hang Su, Yi Zhong. *Safety
Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal
Gradient Projection.* arXiv:2602.07892 [cs.LG, cs.CL], submitted 8 February 2026;
v2 12 May 2026.

Orthogonal Gradient Projection for Safety Alignment. The framing closes the loop
back to GEM: safety post-training is treated as continual learning, where
sequential alignment stages shift the data distribution and their gradients
interfere with directions supporting previously acquired capabilities. OGPSA
estimates a low-rank reference subspace from gradients on a small set of
general-capability data and removes from each safety gradient the component lying
in that subspace. The paper is careful not to claim this is the only mechanism
behind the alignment tax, only "a useful first-order mechanism for mitigating one
important source of capability regression."

This is the closest published relative of the rank-8 subspace arm in
[[drift-experiment]] -- which also estimated a subspace and projected the update
out of it, and also produced a null (sycophancy 9.22 against an untreated 9.20).
The direction of protection is reversed: OGPSA protects capabilities from safety
training, the drift experiment tried to protect behaviour from capability
training.

## What the project concludes from the family

`CONTEXT.md` section 3.2, sharpened: the trait was never hard to find. Per-module
energy along the sycophancy direction was uniform at a median 10.1% across
roughly 94 effective modules of 252 -- *less* concentrated than a random
direction -- against a random-control noise floor near 0.003% per module, so
about 10% is four orders of magnitude above chance in every one of the 252
modules. "The correct claim is not 'we cannot find the trait' but 'the trait is
pervasive and trivially findable, and removing it changes nothing.'"

`CONTEXT.md` section 2 also marks what is *not* in this lineage and is the
project's opening: "Genuinely unoccupied: a *learned* transform rather than
closed-form projection." The nearest precedent it names is LoRA.rar, covered on
[[paper-weight-and-activation-to-language]].

## Verification note

**Abstract only, all five.** Each arXiv abstract page was fetched 2026-09-07 to
confirm title, author list, submission and revision dates, subject class and
venue where the comments field named one. No full text was read in this pass, and
none of the five is summarised in `qwen35/paper_notes.md`; the only local
material is the single sentence in `CONTEXT.md` section 2. Venues (NIPS 2017,
ICLR 2019, NeurIPS 2020, none for AsFT or OGPSA) come from the arXiv pages.
