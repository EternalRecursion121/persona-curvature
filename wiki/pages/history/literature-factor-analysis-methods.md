---
title: Factor-analysis methods reference
summary: Parallel analysis, Tucker congruence, Procrustes rotation, PAF versus PCA and the rotation criteria -- short reference entries for the methods the geometry pages use.
status: current
sources:
  - qwen35/analyse_fa_qwen35.py
  - qwen35/paper_notes.md
last_verified: 2026-09-07
tags: [literature, factor-analysis, geometry, reference]
---

# Factor-analysis methods reference

Short entries the geometry pages can link to. The project's implementation is
`qwen35/analyse_fa_qwen35.py`, adapted from `sweep100/analyse_fa.py` with the
mathematics -- "principal axis factoring with SMC starts, Horn's parallel
analysis, varimax x2 + direct oblimin rotation, Tucker congruence, and the
verify() self-test" -- copied verbatim, so the two runs are directly comparable.

One point of context before the entries. Applying factor analysis to a set of
**weight deltas** is, as far as this project could establish, new.
`qwen35/paper_notes.md` section 3.8 states it as the contribution:
[[persona-cartography-paper|Persona Cartography]] does factor analysis, but on
questionnaire responses, and [[open-character-training-paper|Open Character
Training]] does no weight analysis at all. Section 4 item 8 is careful about the
strength of that claim: "I found no claim in either paper that this has been
done, but absence in two papers is not a literature search." Every method below
is therefore being borrowed from psychometrics into a setting its originators
never considered, and the borrowing is the thing that needs defending -- see
[[geometry-overview]].

---

## Principal axis factoring, and why not PCA

**Principal axis factoring (PAF)** and **principal component analysis (PCA)**
answer different questions and are routinely confused. PCA decomposes the *total*
variance of the observed variables: components are exact linear combinations of
the data, and the first component is whatever direction has the most variance.
PAF decomposes only the **common** variance: it puts communality estimates on the
diagonal of the correlation matrix in place of ones -- classically the squared
multiple correlations (SMC) of each variable with all the others -- and iterates
to convergence, so the factors model what the variables *share* and the
variable-specific and error variance is left out.

For personality this distinction is the whole tradition. The Big Five are common
factors: the claim is that adjective ratings covary because a small number of
latent dispositions drive them, not that five linear combinations reconstruct the
ratings. PCA on the same data will give a similar-looking answer with inflated
loadings.

**What this project uses.** `qwen35/analyse_fa_qwen35.py` implements PAF with SMC
starts, iterated to convergence. Both are reported where they differ:
paper_notes section 3.8's comparison table lists the project's method as "PCA
**and** factor analysis on deltas" against Persona Cartography's "PCA on
flattened delta-W". Persona Cartography's own unsupervised pipeline also uses PAF
-- k = 4, oblimin rotation -- but on questionnaire responses.

## Parallel analysis (Horn, 1965)

Horn, J. L. (1965). A rationale and test for the number of factors in factor
analysis. *Psychometrika*, 30(2), 179-185. `[verified]`

How many factors to keep. Horn's answer: compare each observed eigenvalue against
the eigenvalues you would get from random data of the same shape -- same number
of variables, same number of observations -- and retain factors whose eigenvalue
exceeds the random benchmark, conventionally at the 95th percentile of the
permutation or simulation null. It is the correction to Kaiser's
eigenvalue-greater-than-one rule, which systematically over-extracts because even
random data produces eigenvalues above one.

**Why it matters here more than usual.** A set of near-orthogonal points has a
near-flat spectrum, and a near-flat spectrum can be mistaken for structure.
Persona Cartography's eleven-point PCA over flattened OCEAN deltas runs 15.26%,
13.38%, 12.21%, 12.17%, 11.60%, ..., 1.70% -- which paper_notes section 3.8 reads
as "what ~11 near-orthogonal points look like" and says should be treated as the
null, not as structure. A permutation null is exactly the instrument that turns
that intuition into a test, which is why the project runs one.

Persona Cartography's own use of it is also instructive about its limits: in
their questionnaire factor analysis "real eigenvalues remain above the
95th-percentile permutation null out to k = 11, while a clean scree elbow sits at
k = 4", and they take 4 on three convergent grounds -- parallel analysis, the
scree elbow, Cronbach's alpha, and cross-model congruence -- rather than letting
parallel analysis decide alone. Parallel analysis says how many factors are
*above noise*, not how many are *interpretable*.

## The scree elbow

Cattell's scree test: plot the eigenvalues in descending order and keep the
factors before the plot flattens into scree. Subjective, old, and still the
second opinion everyone consults, because parallel analysis over-retains on large
n and the elbow catches where the substantively large factors stop. This project
reports the scree alongside the permutation null for the same reason Persona
Cartography does; see [[geometry-overview]] for the zoo's spectrum and
[[literature-big-five|Goldberg (2006)]] on reading a whole hierarchy instead of
picking one k.

## Rotation: varimax and direct oblimin

A factor solution is only determined up to rotation, so the solution reported is
a *choice* of rotation, and the choice is between orthogonal and oblique.

**Varimax** (Kaiser, 1958) is the standard orthogonal criterion: rotate to
maximise the variance of squared loadings within factors, driving each variable
toward loading strongly on one factor and near zero on the rest -- simple
structure, with the factors held mutually uncorrelated.

**Direct oblimin** (Jennrich & Bentler, 2002, as implemented here) is the oblique
counterpart: it allows the factors to correlate, which is more honest whenever
the constructs are believed to be correlated -- as the Big Five, and personality
factors generally, are.

**What this project does.** `qwen35/analyse_fa_qwen35.py` runs **two
algorithmically independent varimax implementations and cross-checks them against
each other**: Kaiser's (1958) cyclic pairwise rotation, which has a closed form
per plane, and Jennrich's (2001) gradient-projection varimax. Agreement between
two independent implementations on both a known simple structure and random
loadings is the self-test. Direct oblimin is then run as the oblique solution.
Persona Cartography uses oblimin for its questionnaire factors on the explicit
grounds that correlated factors should be allowed.

## Tucker's congruence coefficient

Tucker, L. R. (1951). *A method for synthesis of factor analysis studies*
(Personnel Research Section Report No. 984). Washington, DC: Department of the
Army. `[not verified]`

Standard modern treatment: Lorenzo-Seva, U., & ten Berge, J. M. F. (2006).
Tucker's congruence coefficient as a meaningful index of factor similarity.
*Methodology*, 2(2), 57-64. `[not verified]`

The measure of whether two factor solutions found in different samples, by
different methods, or on different variables are *the same factor*. It is the
cosine between two loading vectors: phi = <x,y> / sqrt(<x,x><y,y>), which is
exactly how `qwen35/analyse_fa_qwen35.py` defines it. It is scale-invariant and
insensitive to a proportional rescaling of loadings, which is what makes it the
right instrument here and a plain correlation the wrong one.

The conventional reading, from Lorenzo-Seva and ten Berge, is that |phi| in
0.85-0.94 indicates fair similarity and |phi| above 0.95 indicates the two
factors can be considered equal. Those thresholds are the ones the project's
factor-recovery claims are judged against.

**What this project does with it.** Two ways. Congruence of the recovered factors
against the **Goldberg marker targets** -- a target vector per factor with +1 on
each positively keyed marker of that factor, -1 on each negatively keyed marker,
0 elsewhere -- which is the direct test of whether the weight-space factors are
the Big Five. And congruence across replications. `qwen35/build_findings_page.py`
reports that the zoo does "*not* recover Goldberg's structure: Tucker congruence
peaks at 0.68 and nothing clears the" thresholds, against a stated 0.447
construction baseline for a perfectly clean Goldberg solution. The number and its
interpretation belong to [[geometry-overview]]; what belongs here is that 0.68 is
below the 0.85 fair-similarity line, so the instrument is being read the way its
own literature says to read it.

Persona Cartography uses the same coefficient for split-half stability (median
|phi| above 0.97 over 100 random half-splits) and cross-model agreement
(Hungarian-matched per-pair |phi| 0.54-0.80, mean 0.66).

## Procrustes rotation

Schonemann, P. H. (1966). A generalized solution of the orthogonal Procrustes
problem. *Psychometrika*, 31(1), 1-10. `[not verified]`

Given two loading matrices, find the orthogonal rotation of one that brings it as
close as possible to the other in least squares -- solved in closed form by the
SVD of the cross-product. In factor analysis it is the standard way to make two
solutions comparable before measuring congruence, since rotational indeterminacy
would otherwise make any two solutions look different for no substantive reason.

The caution that matters when it is used to test whether a solution *replicates*:
Procrustes rotation toward a target will always improve the fit, including for
random data, so a congruence measured after Procrustes rotation needs its own
null -- rotate random loadings toward the same target and see what congruence
that buys. The zoo's factor pages should state which of their congruences are
post-Procrustes.

## Verification note

Horn (1965) was confirmed by web search on 2026-09-07 (Psychometrika 30(2),
179-185) and is marked `[verified]`. Tucker (1951), Lorenzo-Seva & ten Berge
(2006) and Schonemann (1966) are marked `[not verified]`: they are standard
citations written out from general reference and were not confirmed against a
publisher record in this pass, and none of the three is named in the project's
own code -- `qwen35/analyse_fa_qwen35.py` implements the congruence coefficient
without citing Tucker's report. Kaiser (1958), Jennrich (2001) and Jennrich &
Bentler (2002) are named in that file's docstrings and are quoted from it here;
their full citations were not looked up. **No work on this page was read in
full.** The Lorenzo-Seva and ten Berge thresholds (0.85 fair, 0.95 equal) are
stated from general reference and should be checked before the blog post quotes
them as a bar.
