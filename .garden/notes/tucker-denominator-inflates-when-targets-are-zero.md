---
title: Tucker congruence inflates when you drop rows that are zero in the target
date: 2026-09-12
tags: [method, factor-analysis, gotcha]
---

Tucker's congruence is `<x, y> / sqrt(<x,x><y,y>)`. If the target `y` is zero on
some rows, those rows contribute nothing to the numerator and nothing to
`<y,y>`, but they still inflate `<x,x>`. Dropping them therefore **raises**
congruence for a loading pattern that has not changed at all.

In the zoo this is exactly the situation: the 34 Lexicon traits score 0 in every
Goldberg target. Factoring the 100 markers alone raised Agreeableness congruence
from 0.6555 to 0.7593 and that looked like a better solution. It is not. Taking
the unchanged **134** loadings and simply restricting them to the same 100 rows
already gives 0.7758; refactoring without the 34 then moves congruence by
-0.0496 to +0.0012. The whole apparent gain was the denominator.

Rule: before reporting that a congruence improved after dropping variables,
compute the same congruence from the *original* solution restricted to the
surviving rows. That isolates the denominator from the solution. Quote the
restricted figure as the comparator, never the full-set one.
