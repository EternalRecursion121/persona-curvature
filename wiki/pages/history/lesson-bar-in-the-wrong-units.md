---
title: "Lesson: a preregistered bar can be in the wrong units"
summary: A noise floor was preregistered in a within-basis quantity and measured across two bases, so the test could not have passed under any hypothesis, and a correct claim was withdrawn for a day.
status: current
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/HANDOVER.md
  - /home/vibe12/projects/agent-harness/memory/notes/a-bar-in-the-wrong-units.md
last_verified: 2026-09-07
tags: [lesson, preregistration, seeds]
---

# Lesson: a preregistered bar can be in the wrong units

## The incident

Phase 3 preregistered a seed noise floor: train 40 traits again under a different
random initialisation, and require each trait to resemble itself across the two
runs at cosine at least **+0.2447**, or withdraw the trait-level claim. The
threshold was the median cosine of *genuinely different but closely related*
pairs — same factor, same keying, different trait — the most defensible reference
class available, named in advance.

It measured **+0.0167**. The claim was withdrawn the same day, as written
(`qwen35/PHASE3_VERDICT.md`, "The seed-paired floor: FAIL, exactly as
preregistered").

**The withdrawal was wrong.** The bar was computed *inside one run's basis*; the
measurement was taken *across two bases*. Two independent rank-64 initialisations
span near-orthogonal subspaces of a 2560-dimensional space, and their overlap is
about **1/38**. A trait direction that reproduced perfectly, modulo the subspace,
would still have scored about 0.02. **The test could not have passed under any
hypothesis, including the one it was written to defend.**

The right comparison is cross-basis against cross-basis, and it is decisive
(`qwen35/PHASE3_VERDICT.md` addendum, 2026-08-22 22:40, from
`results/cross_gram_full_root_x_data_null_seedpaired_s40.npz`):

| pair class | cosine |
|---|---|
| same trait | **+0.01659** |
| same factor, same keying | +0.00595 |
| same factor, opposite keying | -0.00271 |
| different factor | +0.00132 |

with perfect separation over 5,320 pairs (min same-trait 0.01509 > max
different-trait 0.01490), 40 of 40 traits their own nearest neighbour among 134
(chance 1/134), and the cross-seed block equal to the within-run block times
1/38 (Pearson **+0.9947**). The structure was there the whole time, uniformly
attenuated. See [[seed-floor]].

## The rule

> Before committing to a preregistered threshold, compute what the statistic would
> read under the **maximal** version of the hypothesis. If that value sits below
> your bar, the bar is not strict — it is broken.

The cheap version: **name the basis of every number in the comparison.** A
threshold and a measurement must live in the same one. Here they did not, and
nothing about the units was visible in the numbers — both were cosines, both in
[-1, 1], both plausible.

## What preregistration bought, and did not

It bought exactly what it advertises: when the number came in ugly there was no
room to negotiate, and the result was reported against the author the same day.

It bought **nothing at all** against a badly specified threshold, and it added a
strong incentive not to reopen the question, since revisiting a hit taken publicly
looks like flinching. **Preregistration converts a specification error into a
durable false conclusion.** The review effort belongs *before* the commitment.

One further piece of the same preregistration did work, and is worth copying: it
said in advance **which claim** the floor governed and which it did not, so the
neighbouring factor-structure claim stayed standing while the trait-level claim
went. **Preregister the scope of a failure, not only its threshold.**

Related: [[method-lessons]], [[seed-floor]],
[[lesson-coordinates-are-not-structure]],
[[lesson-test-shaped-like-the-wrong-hypothesis]].
