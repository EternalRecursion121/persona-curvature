---
title: "Lesson: know which way a shared term pushes before you read the sign"
summary: A control built as A minus B, measured against A, has an algebraic floor that can look like a strong positive; the rule is not to avoid such controls but to compute the floor first.
status: current
sources:
  - results/pooling_check.md
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [lesson, controls, contamination]
---

# Lesson: know which way a shared term pushes before you read the sign

## The incident

The "oracle direction" in [[drift-experiment]] was built as
`V_oracle = dW(plain) - dW(neutral)` and used to say how much of a trait lives in
an update. It **contains** the update it is measured against, so its cosine with
that update is an algebraic function of `||U||`, `||N||` and `<U,N>` alone.

From `results/pooling_check.md` section F, whose whole purpose is to check this:

| quantity | value |
|---|---|
| cos(U, U-N) predicted from `||U||`, `||N||`, `<U,N>` alone | **0.7634** |
| cos(U, U-N) measured | **0.7634** |
| difference — it is an identity, so this must be zero | 0.00e+00 |
| reference: two orthogonal equal-norm updates | 0.7071, i.e. **50%** energy with no shared structure |
| measured pooled energy fraction | 58.285% |
| **excess over the no-information baseline** | **8.285%** |

The apparent 58% was about 50 arithmetic plus 8.3 real. The published figure was
wrong for about five hours before the check was written.

What survives: the oracle **projection experiment** is unaffected — the constraint
was provably enforced and behaviour did not move. What falls is the oracle as a
yardstick.

## The rule

The tempting rule is "avoid controls that share terms with their target". That is
wrong, because such controls are often the right ones. The rule the project
adopted, sharpened by comparison with pastlens:

> Any control sharing terms with its target has an **algebraic floor**, and you
> must know **which direction that floor pushes** before you read the sign.

The comparison is what makes it usable. This project's oracle floor pushed
**toward the positive it reported** — fatal. Pastlens's pooled-contains-final-token
floor pushed **toward the null they rejected** — safe, and they noted they got the
safe direction by luck rather than by design.

The same trap was caught a second time, prospectively: a first attempt at the
leave-one-out interpolability measurement in [[lora-structure-early]] used the
mean of all five OCEAN adapters, which self-contaminates the held-out trait. It
was rebuilt so the held-out trait is never in the basis, and only the corrected
numbers were published.

The cheap version of the check: **compute the statistic's value under no
information at all, from the norms and inner products alone, and subtract it.**

Related: [[method-lessons]], [[drift-experiment]], [[lora-structure-early]],
[[null-controls]].
