---
title: The hat-matrix leave-one-out shortcut is degenerate when p > n
date: 2026-09-12
tags: [method, regression, gotcha]
---

Ridge's closed-form LOO, `resid_i / (1 - h_ii)`, is exact in theory and useless
when the feature count exceeds the sample size. With 768 embedding dimensions
and 100 training rows, a small alpha interpolates, every leverage `h_ii` goes to
1, and the expression evaluates 0/0. Numerically it returns a *tiny* number, so
the selection rule reliably picks the **smallest** alpha on the grid - the most
overfit model - and looks confident doing it.

Caught in the [[goldberg-only-and-heldout-lexicon]] text baseline
(`qwen35/analyse_goldberg_only.py`). It had selected alpha = 0.001, in-sample
R^2 = 0.9999999999, and reported a held-out R^2 of **-0.4158** for the
constitution arm. Refitting each fold honestly (cheap in the dual form, an n x n
solve per fold) selected alpha = 100 and gave -0.0344 for that arm and 0.9380
instead of 0.8833 for the contrast arm. The verdict did not change, but one of
the two headline numbers moved by 0.05 and the other by 0.38.

Rule: when p > n, never use the shortcut. Refit the folds, and store the
held-out score at **every** alpha on the grid so a reader can see the selection
was not load-bearing.
