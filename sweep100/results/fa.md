# Factor analysis of the 100-trait LoRA correlation matrix

The psychometric counterpart to `pca.md`.

## 0. Why this is the same analysis Goldberg did

The cosine matrix between the 100 trait adapters is structurally the same object as the trait-by-trait correlation matrix that psychometrics factors from human self-report data. In Goldberg's studies the data matrix is *people x adjectives* and the correlation between two adjectives is taken over people; here the data matrix is *weight coordinates x adjectives* and the correlation between two trait deltas is taken over weight coordinates. Same matrix, same method: principal axis factoring with squared multiple correlations as the communality start, number of factors by Horn's parallel analysis, varimax and oblimin rotation, and Tucker congruence against marker targets. Goldberg factored that matrix out of people and got the Big Five. We factor it out of weights and ask whether the same five fall out.

The analogy is exact enough that the standard psychometric hygiene applies verbatim. In particular, removing the grand mean dW across the 100 traits is **ipsatisation**: for each weight coordinate (each 'respondent') we subtract that coordinate's mean response across the 100 items, which is the textbook correction for acquiescence / evaluative response bias. Both the raw and the ipsatised solution are reported below, and they differ materially -- which is itself the answer to the evaluative-factor question.


## 1. Verification (no scipy on this box; everything hand-checked)

| check | case | result |
|---|---|---|
| eigen-decomposition | `[[2,1],[1,2]]`, exact spectrum {3, 1} | eigenvalues [3.0, 1.0], err 0.0e+00 |
| eigen-decomposition | random 100x100 symmetric | max residual `|AV - VL|` 1.2e-14, orthonormality 2.0e-15, reconstruction 1.6e-14 |
| SMC | `1 - 1/(R^-1)_ii` vs explicit OLS R^2 of each variable on the other 5 | max abs diff 1.0e-15 |
| varimax | perfect 2-factor simple structure rotated by 0.6 rad, then un-rotated by varimax | max abs error vs truth: pairwise 1.1e-16, GPA 9.9e-09 |
| varimax | two independent algorithms (Kaiser cyclic pairwise vs Jennrich gradient projection) on random loadings, plus communality invariance | max disagreement 1.6e-07 |
| oblimin | known oblique structure: 3 factors, perfect simple pattern, true Phi with off-diagonals .5/.4/.6 | pattern recovered to 2.3e-10, **Phi recovered to 2.8e-10**, `L Phi L' ` preserved to 8.9e-16 |
| Tucker congruence | x=(1,2,3), y=(1,0,-1): <x,y>=-2, <x,x>=14, <y,y>=2, phi = -2/sqrt(28) = -0.3779644730092272 | got -0.3779644730092272, err 0.0e+00; identical vectors -> 1, sign-flipped -> -1 |
| PAF | synthetic 2-factor model with known communalities | max communality error 9.4e-09 in 9 iterations |

All checks pass (`all_ok = True`). The oblimin check is the load-bearing one: the algorithm recovers a *known* factor correlation matrix to 1e-6, so the Phi reported below is trustworthy.


## 2. The correlation matrix

**Is cosine the same as correlation?** Yes, to 6+ decimal places. The literal Pearson correlation subtracts each dW's own mean entry; the correction to `<dW_i,dW_j>` is `D * mu_i * mu_j` with D = 2,774,532,096 entries per delta. Computed exactly from the factored form (`sum(dW) = (alpha/r) * (1'B)(A1)`), the largest mean entry is 1.09e-08 and the largest correction is 3.32e-07 -- a relative correction of 1.9e-03. So **entry-level centring is ignorable** and cosine == correlation here.

**Centring across traits is a different and non-ignorable choice**, and we report both:

| | off-diagonal mean | sd | min | max | eigenvalues 1-6 |
|---|---|---|---|---|---|
| raw cosine (uncentred) | **+0.0989** | 0.2650 | -0.548 | +0.746 | 20.37, 14.65, 12.60, 8.08, 3.12, 2.57 |
| ipsatised (grand mean removed) | **-0.0099** | 0.2774 | -0.596 | +0.728 | 22.29, 13.77, 9.75, 6.22, 3.35, 2.33 |

The two agree closely off-diagonal (r = 0.9455) but the uncentred matrix carries a positive general offset (+0.0989 vs -0.0099) -- the 'an adapter was trained here' component, which is the acquiescence analogue. **They differ materially in the factor solution, so both are carried through.**

Note: ipsatisation removes exactly one dimension, so R_centred has rank 99 and is exactly singular. SMC = 1 - 1/(R^-1)_ii is then identically 1.0, so the centred run starts PAF from a ridge-regularised SMC (ridge = 0.001, SMC mean 0.918). This turns out not to matter: PAF converges to the *same* fixed point from all three starts --

| start | iterations | sum of communalities (k=5, centred) |
|---|---|---|
| ridge_smc | 10 | 53.056318 |
| constant_0.5 | 10 | 53.056318 |
| max_abs_r | 10 | 53.056318 |

Uncentred SMC (well-posed, condition number 96.8): min 0.402, mean 0.630, max 0.737.


## 3. How many factors?

### 3a. Eigenvalues and scree

Reduced correlation matrix = R with communalities on the diagonal (the matrix PAF actually factors). Uncentred:

```
rank  eigenvalue (reduced, uncentred)
   1   20.028 |####################################################
   2   14.274 |#####################################
   3   12.233 |################################
   4    7.703 |####################
   5    2.754 |#######
   6    2.150 |######
   7    1.631 |####
   8    0.875 |##
   9    0.816 |##
  10    0.713 |##
  11    0.588 |##
  12    0.489 |#
  13    0.420 |#
  14    0.353 |#
  15    0.326 |#
```
Centred (ipsatised):

```
rank  eigenvalue (reduced, centred)
   1   22.212 |####################################################
   2   13.690 |################################
   3    9.666 |#######################
   4    6.136 |##############
   5    3.257 |########
   6    2.248 |#####
   7    1.874 |####
   8    1.305 |###
   9    1.188 |###
  10    1.046 |##
  11    0.912 |##
  12    0.844 |##
  13    0.771 |##
  14    0.728 |##
  15    0.678 |##
```

- **Kaiser** (eigenvalue of the *unreduced* R > 1): 10 factors uncentred, 11 centred. Kaiser is known to over-retain.
- **Scree**: a sharp elbow after 4 (uncentred reduced eigenvalues 20.0, 14.3, 12.2, 7.7, 2.8) and a second shelf after 7 (2.15, 1.63, 0.88, 0.82).
- **Reduced eigenvalues > 1**: 7 (uncentred).

### 3b. Parallel analysis (Horn) -- and what 'matched in size' means here

Horn's PA compares each observed eigenvalue against the same-rank eigenvalue of random data with the same number of variables (p=100) and N observations, retaining the leading run that exceeds the null (95th percentile, first-crossing rule; 500 replications per cell).

There is a genuine difficulty: **what is N?** A literal size match would be N = D = 2,774,532,096 weight coordinates, and that is degenerate -- the null correlation matrix converges to the identity, every null eigenvalue goes to 1, and PA collapses onto Kaiser. The way out is to notice that *PA with N observations is exactly the null 'the p trait vectors are random directions in an (N-1)-dimensional space'*. Choosing N is choosing the effective dimensionality of weight space, and we can estimate that empirically from the reseed controls: `d_i = dW_i(seed0) - dW_i(seed1)` is a pure noise draw, and cosines between independent isotropic vectors in R^m have variance 1/m.

- from the 10 noise-noise cosines: m = **5809**
- from the 475 noise-vs-other-trait cosines: m = **1527**
- so N is somewhere around 1528-5809; the noise-noise estimate rests on only 10 cosines, so m is uncertain by a factor of ~2-3 either way; hence the grid.

| N | uncentred: k (Horn, unreduced) | uncentred: k (SMC-reduced) | centred: k (Horn, unreduced) | centred: k (SMC-reduced) |
|---|---|---|---|---|
| 150 | **5** | 5 | **5** | 5 |
| 300 | **6** | 7 | **6** | 6 |
| 1000 | **7** | 11 | **7** | 7 |
| 1528 <- | **7** | 13 | **7** | 7 |
| 5809 <- | **8** | 19 | **9** | 9 |
| 20000 | **9** | 26 | **10** | 10 |

**Parallel analysis supports k = 7, not 5.** Horn's original (unreduced) form gives 7 at N=1000 and N=1528 for both matrices and 9 at N=5809; 5 appears only at N=150, far below any defensible effective dimensionality. The SMC-reduced variant is more permissive still on the uncentred matrix (7 at N=300 rising to 13-19), and on the centred matrix it is nearly identical to the unreduced form because ipsatisation forces the SMCs close to 1 so the reduction barely changes the trace. The robust claim is **k > 5**; we take k=7 as the PA number and also extract at exactly 5 because 5 is the hypothesis.


## 4. Principal axis factoring

| solution | PAF iterations | final max |change in h^2| | Heywood cases | varimax iters | oblimin iters | oblimin |grad| | `L Phi L'` vs unrotated |
|---|---|---|---|---|---|---|---|
| centred_k7 | 11 | 2.7e-08 | 0 | 45 | 905 | 6.6e-07 | 5.0e-16 |
| centred_k5 | 10 | 7.2e-08 | 0 | 15 | 152 | 3.0e-07 | 5.6e-16 |
| uncentred_k7 | 10 | 2.7e-08 | 0 | 21 | 290 | 4.0e-07 | 4.4e-16 |
| uncentred_k5 | 9 | 6.9e-08 | 0 | 15 | 158 | 2.4e-07 | 5.6e-16 |

Convergence criterion: max over traits of |change in communality| < 1e-07; Heywood cases capped at 0.999 (none occurred). Both rotations use Kaiser normalisation (rows scaled to unit length by sqrt(h^2) before rotating, restored after). The last column is the check that oblique rotation is loss-free: the rotated pattern and factor correlations reproduce the unrotated reduced matrix exactly.

Factor sign is arbitrary in FA; the convention throughout is that each factor is oriented so its congruence with its own best-matching Goldberg target is positive, and factors are ordered by descending SS loading.


## 5. HEADLINE: the oblimin factor correlation matrix

This is the reason for doing an oblique rotation at all. The PCA showed the Goldberg factor *directions* are correlated in weight space (C-ES +0.58, C-I +0.58, A-I +0.42), which is exactly why a rotation-free method could not separate them. Oblimin lets the extracted factors be correlated and reports how correlated they are.


**centred_k5** -- factor correlation matrix Phi (direct oblimin, gamma=0):

| | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| **F1** | +1.000 | +0.064 | -0.050 | +0.175 | +0.011 |
| **F2** | +0.064 | +1.000 | -0.183 | +0.226 | +0.346 |
| **F3** | -0.050 | -0.183 | +1.000 | +0.072 | +0.038 |
| **F4** | +0.175 | +0.226 | +0.072 | +1.000 | +0.044 |
| **F5** | +0.011 | +0.346 | +0.038 | +0.044 | +1.000 |

Largest |correlation| 0.346; mean |correlation| 0.121.

**uncentred_k5** -- factor correlation matrix Phi (direct oblimin, gamma=0):

| | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|
| **F1** | +1.000 | +0.021 | +0.099 | +0.106 | -0.067 |
| **F2** | +0.021 | +1.000 | +0.172 | +0.335 | -0.226 |
| **F3** | +0.099 | +0.172 | +1.000 | -0.204 | -0.091 |
| **F4** | +0.106 | +0.335 | -0.204 | +1.000 | +0.078 |
| **F5** | -0.067 | -0.226 | -0.091 | +0.078 | +1.000 |

Largest |correlation| 0.335; mean |correlation| 0.140.

**centred_k7** -- factor correlation matrix Phi (direct oblimin, gamma=0):

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
|---|---|---|---|---|---|---|---|
| **F1** | +1.000 | +0.053 | -0.094 | +0.102 | -0.296 | +0.046 | -0.020 |
| **F2** | +0.053 | +1.000 | +0.233 | +0.125 | +0.097 | +0.221 | -0.014 |
| **F3** | -0.094 | +0.233 | +1.000 | +0.410 | +0.340 | +0.114 | +0.144 |
| **F4** | +0.102 | +0.125 | +0.410 | +1.000 | -0.056 | -0.181 | +0.017 |
| **F5** | -0.296 | +0.097 | +0.340 | -0.056 | +1.000 | +0.319 | +0.250 |
| **F6** | +0.046 | +0.221 | +0.114 | -0.181 | +0.319 | +1.000 | +0.297 |
| **F7** | -0.020 | -0.014 | +0.144 | +0.017 | +0.250 | +0.297 | +1.000 |

Largest |correlation| 0.410; mean |correlation| 0.163.

**uncentred_k7** -- factor correlation matrix Phi (direct oblimin, gamma=0):

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
|---|---|---|---|---|---|---|---|
| **F1** | +1.000 | +0.177 | +0.196 | -0.018 | +0.063 | +0.091 | -0.121 |
| **F2** | +0.177 | +1.000 | -0.018 | +0.332 | +0.384 | -0.174 | -0.073 |
| **F3** | +0.196 | -0.018 | +1.000 | -0.219 | +0.256 | +0.030 | +0.305 |
| **F4** | -0.018 | +0.332 | -0.219 | +1.000 | +0.074 | +0.106 | -0.023 |
| **F5** | +0.063 | +0.384 | +0.256 | +0.074 | +1.000 | -0.224 | +0.342 |
| **F6** | +0.091 | -0.174 | +0.030 | +0.106 | -0.224 | +1.000 | +0.073 |
| **F7** | -0.121 | -0.073 | +0.305 | -0.023 | +0.342 | +0.073 | +1.000 |

Largest |correlation| 0.384; mean |correlation| 0.157.

## 6. Ground truth: Tucker congruence with the Goldberg factors

**Target definition.** target_F[i] = +1 if trait i is a positively-keyed marker of Goldberg factor F, -1 if negatively-keyed marker of F, 0 if it belongs to another factor. 'Eval' is the general evaluative target: +1 for every positively-keyed trait, -1 for every negatively-keyed trait, regardless of factor.

Tucker's congruence coefficient is `phi(x,y) = <x,y> / sqrt(<x,x><y,y>)`. Conventional reading: |phi| > 0.85 'fair similarity', |phi| > 0.95 'equivalent'.

**Baseline that must be kept in mind**: Each Goldberg target has congruence exactly 0.447 = 20/sqrt(20*100) with the Eval target BY CONSTRUCTION (Eval is the sum of the five). So 0.447 is the baseline a perfectly-recovered pure Goldberg factor already scores on the Eval column; only values clearly above it indicate extra evaluative content.


### centred_k5


**unrotated** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 21.85 | +0.103 | +0.385 | +0.568 | +0.283 | +0.496 | +0.820 | C 0.57 |
| F2 | 13.32 | +0.031 | +0.673 | -0.292 | -0.138 | +0.095 | +0.165 | A 0.67 |
| F3 | 9.28 | +0.774 | -0.116 | -0.075 | -0.077 | +0.169 | +0.302 | E 0.77 |
| F4 | 5.75 | -0.153 | -0.249 | -0.162 | -0.448 | +0.580 | -0.193 | I 0.58 |
| F5 | 2.85 | -0.117 | -0.019 | -0.360 | +0.559 | +0.076 | +0.062 | ES 0.56 |

**varimax** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 14.95 | +0.174 | +0.042 | +0.706 | +0.239 | +0.305 | +0.656 | C 0.71 |
| F2 | 14.52 | -0.024 | +0.807 | +0.167 | +0.162 | +0.233 | +0.601 | A 0.81 |
| F3 | 9.28 | +0.695 | +0.054 | -0.238 | -0.193 | -0.105 | +0.095 | E 0.69 |
| F4 | 9.09 | +0.255 | +0.161 | +0.148 | -0.064 | +0.774 | +0.570 | I 0.77 |
| F5 | 5.22 | +0.292 | -0.008 | +0.189 | +0.690 | +0.003 | +0.521 | ES 0.69 |

**oblimin** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 13.92 | -0.031 | +0.818 | +0.138 | +0.166 | +0.157 | +0.558 | A 0.82 |
| F2 | 12.14 | +0.159 | +0.022 | +0.740 | +0.111 | +0.244 | +0.571 | C 0.74 |
| F3 | 8.81 | +0.714 | +0.067 | -0.151 | -0.191 | -0.124 | +0.140 | E 0.71 |
| F4 | 8.60 | +0.236 | +0.103 | +0.059 | -0.089 | +0.772 | +0.484 | I 0.77 |
| F5 | 5.56 | +0.257 | -0.018 | +0.152 | +0.704 | -0.007 | +0.487 | ES 0.70 |

### uncentred_k5


**unrotated** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 19.97 | +0.111 | +0.380 | +0.530 | +0.173 | +0.530 | +0.771 | C 0.53 |
| F2 | 14.22 | -0.195 | +0.083 | +0.266 | +0.462 | -0.190 | +0.191 | ES 0.46 |
| F3 | 12.17 | -0.014 | +0.706 | -0.249 | -0.039 | +0.017 | +0.188 | A 0.71 |
| F4 | 7.64 | +0.757 | -0.002 | +0.024 | +0.150 | -0.025 | +0.404 | E 0.76 |
| F5 | 2.70 | -0.065 | -0.167 | -0.343 | +0.201 | +0.459 | +0.038 | I 0.46 |

**varimax** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 13.91 | -0.048 | +0.797 | +0.194 | +0.143 | +0.261 | +0.603 | A 0.80 |
| F2 | 11.56 | +0.273 | +0.054 | +0.599 | +0.041 | +0.325 | +0.577 | C 0.60 |
| F3 | 11.16 | +0.206 | +0.022 | +0.382 | +0.533 | +0.085 | +0.549 | ES 0.53 |
| F4 | 10.28 | +0.304 | +0.132 | +0.014 | -0.135 | +0.602 | +0.410 | I 0.60 |
| F5 | 9.78 | +0.541 | +0.077 | -0.379 | -0.170 | -0.274 | -0.091 | E 0.54 |

**oblimin** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 13.60 | -0.055 | +0.807 | +0.179 | +0.133 | +0.204 | +0.567 | A 0.81 |
| F2 | 10.45 | +0.281 | +0.057 | +0.626 | +0.017 | +0.208 | +0.531 | C 0.63 |
| F3 | 9.90 | +0.272 | -0.029 | +0.282 | +0.527 | +0.124 | +0.526 | ES 0.53 |
| F4 | 9.34 | +0.270 | +0.093 | -0.087 | -0.112 | +0.624 | +0.352 | I 0.62 |
| F5 | 9.21 | +0.594 | +0.101 | -0.288 | -0.159 | -0.278 | -0.014 | E 0.59 |

### centred_k7


**unrotated** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 21.88 | +0.103 | +0.384 | +0.569 | +0.282 | +0.496 | +0.820 | C 0.57 |
| F2 | 13.36 | +0.030 | +0.674 | -0.292 | -0.137 | +0.095 | +0.165 | A 0.67 |
| F3 | 9.31 | +0.774 | -0.115 | -0.076 | -0.078 | +0.169 | +0.302 | E 0.77 |
| F4 | 5.78 | -0.153 | -0.248 | -0.163 | -0.447 | +0.581 | -0.192 | I 0.58 |
| F5 | 2.88 | -0.117 | -0.017 | -0.361 | +0.558 | +0.075 | +0.062 | ES 0.56 |
| F6 | 1.91 | -0.093 | -0.124 | +0.062 | +0.176 | +0.158 | +0.080 | ES 0.18 |
| F7 | 1.53 | -0.092 | -0.047 | +0.034 | -0.075 | +0.123 | -0.025 | I 0.12 |

**varimax** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 14.46 | -0.014 | +0.809 | +0.186 | +0.169 | +0.229 | +0.616 | A 0.81 |
| F2 | 9.29 | +0.614 | +0.073 | -0.298 | -0.210 | -0.110 | +0.030 | E 0.61 |
| F3 | 9.11 | +0.210 | +0.175 | +0.169 | -0.050 | +0.790 | +0.579 | I 0.79 |
| F4 | 9.01 | +0.034 | +0.026 | +0.723 | +0.254 | +0.276 | +0.587 | C 0.72 |
| F5 | 7.13 | +0.492 | -0.006 | +0.495 | +0.129 | +0.234 | +0.601 | C 0.50 |
| F6 | 5.49 | +0.264 | +0.002 | +0.252 | +0.690 | +0.055 | +0.565 | ES 0.69 |
| F7 | 2.16 | -0.041 | +0.077 | +0.431 | -0.075 | +0.268 | +0.295 | C 0.43 |

**oblimin** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 8.26 | +0.605 | +0.166 | -0.178 | -0.169 | -0.095 | +0.147 | E 0.60 |
| F2 | 7.60 | +0.145 | +0.113 | +0.008 | -0.079 | +0.790 | +0.437 | I 0.79 |
| F3 | 7.15 | -0.007 | +0.618 | +0.396 | +0.049 | +0.251 | +0.584 | A 0.62 |
| F4 | 6.95 | -0.091 | +0.731 | -0.012 | +0.202 | +0.056 | +0.397 | A 0.73 |
| F5 | 6.70 | -0.015 | -0.056 | +0.670 | +0.198 | +0.230 | +0.460 | C 0.67 |
| F6 | 5.79 | +0.553 | -0.028 | +0.372 | +0.026 | +0.167 | +0.487 | E 0.55 |
| F7 | 4.89 | +0.199 | +0.030 | +0.126 | +0.729 | +0.010 | +0.489 | ES 0.73 |

### uncentred_k7


**unrotated** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 20.01 | +0.111 | +0.379 | +0.531 | +0.174 | +0.530 | +0.771 | C 0.53 |
| F2 | 14.25 | -0.195 | +0.083 | +0.266 | +0.462 | -0.190 | +0.191 | ES 0.46 |
| F3 | 12.21 | -0.014 | +0.706 | -0.248 | -0.039 | +0.018 | +0.189 | A 0.71 |
| F4 | 7.68 | +0.757 | -0.002 | +0.023 | +0.150 | -0.025 | +0.404 | E 0.76 |
| F5 | 2.74 | -0.066 | -0.166 | -0.345 | +0.204 | +0.458 | +0.038 | I 0.46 |
| F6 | 2.13 | -0.047 | +0.149 | -0.167 | +0.533 | -0.317 | +0.068 | ES 0.53 |
| F7 | 1.61 | -0.086 | -0.096 | +0.059 | +0.199 | +0.137 | +0.096 | ES 0.20 |

**varimax** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 12.71 | -0.042 | +0.821 | +0.127 | +0.161 | +0.206 | +0.570 | A 0.82 |
| F2 | 11.20 | +0.288 | +0.136 | +0.066 | -0.139 | +0.619 | +0.434 | I 0.62 |
| F3 | 9.35 | -0.109 | +0.164 | +0.595 | +0.210 | +0.368 | +0.549 | C 0.59 |
| F4 | 8.31 | +0.278 | -0.006 | +0.245 | +0.624 | -0.025 | +0.499 | ES 0.62 |
| F5 | 8.19 | +0.163 | +0.032 | +0.623 | +0.118 | +0.302 | +0.554 | C 0.62 |
| F6 | 6.83 | +0.576 | +0.154 | -0.201 | -0.204 | -0.120 | +0.092 | E 0.58 |
| F7 | 4.04 | +0.505 | +0.017 | +0.265 | -0.094 | +0.052 | +0.334 | E 0.51 |

**oblimin** (columns: the five Goldberg targets, then the general evaluative target):

| factor | SS loading | E | A | C | ES | I | Eval | best Goldberg |
|---|---|---|---|---|---|---|---|---|
| F1 | 11.34 | -0.018 | +0.834 | +0.097 | +0.162 | +0.131 | +0.539 | A 0.83 |
| F2 | 7.49 | -0.093 | +0.158 | +0.525 | +0.066 | +0.338 | +0.445 | C 0.53 |
| F3 | 7.37 | +0.199 | +0.078 | -0.086 | -0.101 | +0.696 | +0.352 | I 0.70 |
| F4 | 7.27 | +0.268 | -0.017 | +0.130 | +0.677 | -0.018 | +0.465 | ES 0.68 |
| F5 | 7.00 | +0.095 | +0.022 | +0.596 | +0.132 | +0.253 | +0.491 | C 0.60 |
| F6 | 6.16 | +0.566 | +0.165 | -0.088 | -0.208 | -0.082 | +0.157 | E 0.57 |
| F7 | 5.36 | +0.522 | -0.041 | +0.192 | -0.110 | +0.054 | +0.276 | E 0.52 |

### Which Goldberg factors clear the thresholds?

- **centred_k5, oblimin** -- best congruence per Goldberg factor: E 0.714, A 0.818, C 0.740, ES 0.704, I 0.772. Clearing 0.85 (fair): **none**. Clearing 0.95 (equivalent): **none**.
- **uncentred_k5, oblimin** -- best congruence per Goldberg factor: E 0.594, A 0.807, C 0.626, ES 0.527, I 0.624. Clearing 0.85 (fair): **none**. Clearing 0.95 (equivalent): **none**.
- **centred_k7, oblimin** -- best congruence per Goldberg factor: E 0.605, A 0.731, C 0.670, ES 0.729, I 0.790. Clearing 0.85 (fair): **none**. Clearing 0.95 (equivalent): **none**.
- **uncentred_k7, oblimin** -- best congruence per Goldberg factor: E 0.566, A 0.834, C 0.596, ES 0.677, I 0.696. Clearing 0.85 (fair): **none**. Clearing 0.95 (equivalent): **none**.

**Plain statement.** All five Goldberg factors appear as *identifiable, separate* factors -- every one of them is the best match for exactly one extracted factor, with no factor doubling up -- but **none of them clears the 0.85 'fair similarity' threshold, and none clears 0.95.** The best congruences sit in the 0.70-0.83 band on the ipsatised matrix. That is a structural recovery of the Big Five without a quantitative equivalence claim, and the reason is visible in the loading tables: the extracted factors are cleaner than the targets, in the sense that they concentrate on one pole of a factor's markers and put cross-loadings on markers of other factors, whereas the target is a flat +-1 over all 20 markers.


## 7. Top-10 loading traits per factor

Ranked by |loading|; sign shown. Factor codes: E Extraversion, A Agreeableness, C Conscientiousness, ES EmotionalStability, I Intellect.


### centred_k5, OBLIMIN pattern


**F1** (SS 13.92; best Goldberg match A, phi = +0.818; Eval phi = +0.558)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Kind | +0.795 | A | + |
| 2 | Agreeable | +0.785 | A | + |
| 3 | Considerate | +0.783 | A | + |
| 4 | Pleasant | +0.776 | A | + |
| 5 | Generous | +0.767 | A | + |
| 6 | Cooperative | +0.752 | A | + |
| 7 | Harsh | -0.721 | A | - |
| 8 | Sympathetic | +0.713 | A | + |
| 9 | Warm | +0.679 | A | + |
| 10 | Unkind | -0.676 | A | - |

**F2** (SS 12.14; best Goldberg match C, phi = +0.740; Eval phi = +0.571)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Organized | +0.713 | C | + |
| 2 | Unsystematic | -0.703 | C | - |
| 3 | Efficient | +0.694 | C | + |
| 4 | Neat | +0.687 | C | + |
| 5 | Practical | +0.678 | C | + |
| 6 | Haphazard | -0.658 | C | - |
| 7 | Conscientious | +0.649 | C | + |
| 8 | Systematic | +0.644 | C | + |
| 9 | Sloppy | -0.643 | C | - |
| 10 | Prompt | +0.638 | C | + |

**F3** (SS 8.81; best Goldberg match E, phi = +0.714; Eval phi = +0.140)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Withdrawn | -0.764 | E | - |
| 2 | Reserved | -0.742 | E | - |
| 3 | Quiet | -0.724 | E | - |
| 4 | Energetic | +0.690 | E | + |
| 5 | Extraverted | +0.678 | E | + |
| 6 | Introverted | -0.642 | E | - |
| 7 | Inhibited | -0.641 | E | - |
| 8 | Untalkative | -0.626 | E | - |
| 9 | Talkative | +0.614 | E | + |
| 10 | Unexcitable | -0.583 | ES | + |

**F4** (SS 8.60; best Goldberg match I, phi = +0.772; Eval phi = +0.484)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Unimaginative | -0.820 | I | - |
| 2 | Uncreative | -0.761 | I | - |
| 3 | Complex | +0.740 | I | + |
| 4 | Creative | +0.670 | I | + |
| 5 | Imaginative | +0.654 | I | + |
| 6 | Artistic | +0.642 | I | + |
| 7 | Deep | +0.622 | I | + |
| 8 | Unadventurous | -0.600 | E | - |
| 9 | Innovative | +0.534 | I | + |
| 10 | Philosophical | +0.526 | I | + |

**F5** (SS 5.56; best Goldberg match ES, phi = +0.704; Eval phi = +0.487)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Anxious | -0.757 | ES | - |
| 2 | Fearful | -0.695 | ES | - |
| 3 | Fretful | -0.692 | ES | - |
| 4 | Insecure | -0.663 | ES | - |
| 5 | Nervous | -0.555 | ES | - |
| 6 | High-strung | -0.540 | ES | - |
| 7 | Self-pitying | -0.533 | ES | - |
| 8 | Imperturbable | +0.472 | ES | + |
| 9 | Timid | -0.397 | E | - |
| 10 | Envious | -0.374 | ES | - |

### centred_k7, OBLIMIN pattern


**F1** (SS 8.26; best Goldberg match E, phi = +0.605; Eval phi = +0.147)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Withdrawn | -0.758 | E | - |
| 2 | Quiet | -0.757 | E | - |
| 3 | Untalkative | -0.733 | E | - |
| 4 | Extraverted | +0.692 | E | + |
| 5 | Talkative | +0.689 | E | + |
| 6 | Reserved | -0.688 | E | - |
| 7 | Energetic | +0.647 | E | + |
| 8 | Unexcitable | -0.591 | ES | + |
| 9 | Introverted | -0.591 | E | - |
| 10 | Uninquisitive | -0.537 | I | - |

**F2** (SS 7.60; best Goldberg match I, phi = +0.790; Eval phi = +0.437)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Unimaginative | -0.788 | I | - |
| 2 | Complex | +0.715 | I | + |
| 3 | Uncreative | -0.704 | I | - |
| 4 | Artistic | +0.650 | I | + |
| 5 | Imaginative | +0.640 | I | + |
| 6 | Creative | +0.629 | I | + |
| 7 | Deep | +0.591 | I | + |
| 8 | Philosophical | +0.574 | I | + |
| 9 | Unreflective | -0.563 | I | - |
| 10 | Unadventurous | -0.540 | E | - |

**F3** (SS 7.15; best Goldberg match A, phi = +0.618; Eval phi = +0.584)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Considerate | +0.671 | A | + |
| 2 | Kind | +0.662 | A | + |
| 3 | Agreeable | +0.616 | A | + |
| 4 | Pleasant | +0.600 | A | + |
| 5 | Cooperative | +0.589 | A | + |
| 6 | Sympathetic | +0.576 | A | + |
| 7 | Helpful | +0.551 | A | + |
| 8 | Inconsistent | -0.551 | C | - |
| 9 | Generous | +0.549 | A | + |
| 10 | Warm | +0.494 | A | + |

**F4** (SS 6.95; best Goldberg match A, phi = +0.731; Eval phi = +0.397)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Harsh | -0.697 | A | - |
| 2 | Unkind | -0.648 | A | - |
| 3 | Unsympathetic | -0.630 | A | - |
| 4 | Uncharitable | -0.622 | A | - |
| 5 | Rude | -0.607 | A | - |
| 6 | Demanding | -0.601 | A | - |
| 7 | Irritable | -0.537 | ES | - |
| 8 | Uncooperative | -0.524 | A | - |
| 9 | Undemanding | +0.519 | ES | + |
| 10 | Temperamental | -0.501 | ES | - |

**F5** (SS 6.70; best Goldberg match C, phi = +0.670; Eval phi = +0.460)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Systematic | +0.795 | C | + |
| 2 | Neat | +0.738 | C | + |
| 3 | Conscientious | +0.725 | C | + |
| 4 | Thorough | +0.693 | C | + |
| 5 | Organized | +0.619 | C | + |
| 6 | Unemotional | +0.584 | ES | + |
| 7 | Intellectual | +0.574 | I | + |
| 8 | Prompt | +0.514 | C | + |
| 9 | Emotional | -0.506 | ES | - |
| 10 | Efficient | +0.433 | C | + |

**F6** (SS 5.79; best Goldberg match E, phi = +0.553; Eval phi = +0.487)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Bold | +0.733 | E | + |
| 2 | Vigorous | +0.727 | E | + |
| 3 | Active | +0.701 | E | + |
| 4 | Daring | +0.679 | E | + |
| 5 | Assertive | +0.622 | E | + |
| 6 | Efficient | +0.508 | C | + |
| 7 | Practical | +0.431 | C | + |
| 8 | Bashful | -0.405 | E | - |
| 9 | Timid | -0.403 | E | - |
| 10 | Inhibited | -0.398 | E | - |

**F7** (SS 4.89; best Goldberg match ES, phi = +0.729; Eval phi = +0.489)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Anxious | -0.730 | ES | - |
| 2 | Fearful | -0.712 | ES | - |
| 3 | Fretful | -0.699 | ES | - |
| 4 | Insecure | -0.635 | ES | - |
| 5 | Nervous | -0.534 | ES | - |
| 6 | Self-pitying | -0.523 | ES | - |
| 7 | High-strung | -0.501 | ES | - |
| 8 | Imperturbable | +0.435 | ES | + |
| 9 | Envious | -0.384 | ES | - |
| 10 | Jealous | -0.373 | ES | - |

### uncentred_k5, OBLIMIN pattern


**F1** (SS 13.60; best Goldberg match A, phi = +0.807; Eval phi = +0.567)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Considerate | +0.749 | A | + |
| 2 | Uncharitable | -0.738 | A | - |
| 3 | Unkind | -0.737 | A | - |
| 4 | Kind | +0.733 | A | + |
| 5 | Unsympathetic | -0.730 | A | - |
| 6 | Harsh | -0.728 | A | - |
| 7 | Agreeable | +0.726 | A | + |
| 8 | Pleasant | +0.709 | A | + |
| 9 | Rude | -0.693 | A | - |
| 10 | Generous | +0.687 | A | + |

**F2** (SS 10.45; best Goldberg match C, phi = +0.626; Eval phi = +0.531)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Efficient | +0.818 | C | + |
| 2 | Organized | +0.789 | C | + |
| 3 | Practical | +0.788 | C | + |
| 4 | Neat | +0.743 | C | + |
| 5 | Active | +0.712 | E | + |
| 6 | Prompt | +0.698 | C | + |
| 7 | Conscientious | +0.670 | C | + |
| 8 | Systematic | +0.662 | C | + |
| 9 | Assertive | +0.643 | E | + |
| 10 | Thorough | +0.609 | C | + |

**F3** (SS 9.90; best Goldberg match ES, phi = +0.527; Eval phi = +0.526)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Anxious | -0.764 | ES | - |
| 2 | Insecure | -0.756 | ES | - |
| 3 | Fearful | -0.746 | ES | - |
| 4 | Nervous | -0.732 | ES | - |
| 5 | Self-pitying | -0.701 | ES | - |
| 6 | Fretful | -0.673 | ES | - |
| 7 | Timid | -0.670 | E | - |
| 8 | Shy | -0.660 | E | - |
| 9 | Bashful | -0.609 | E | - |
| 10 | Inefficient | -0.597 | C | - |

**F4** (SS 9.34; best Goldberg match I, phi = +0.624; Eval phi = +0.352)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Unimaginative | -0.794 | I | - |
| 2 | Complex | +0.742 | I | + |
| 3 | Creative | +0.692 | I | + |
| 4 | Artistic | +0.685 | I | + |
| 5 | Imaginative | +0.673 | I | + |
| 6 | Deep | +0.672 | I | + |
| 7 | Uncreative | -0.654 | I | - |
| 8 | Philosophical | +0.639 | I | + |
| 9 | Verbal | +0.605 | E | + |
| 10 | Touchy | +0.550 | ES | - |

**F5** (SS 9.21; best Goldberg match E, phi = +0.594; Eval phi = -0.014)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Reserved | -0.764 | E | - |
| 2 | Withdrawn | -0.718 | E | - |
| 3 | Extraverted | +0.694 | E | + |
| 4 | Quiet | -0.650 | E | - |
| 5 | Energetic | +0.648 | E | + |
| 6 | Inhibited | -0.631 | E | - |
| 7 | Talkative | +0.622 | E | + |
| 8 | Unexcitable | -0.608 | ES | + |
| 9 | Introverted | -0.595 | E | - |
| 10 | Shallow | +0.571 | I | - |

### uncentred_k7, OBLIMIN pattern


**F1** (SS 11.34; best Goldberg match A, phi = +0.834; Eval phi = +0.539)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Uncharitable | -0.712 | A | - |
| 2 | Kind | +0.709 | A | + |
| 3 | Unsympathetic | -0.695 | A | - |
| 4 | Agreeable | +0.691 | A | + |
| 5 | Generous | +0.690 | A | + |
| 6 | Harsh | -0.687 | A | - |
| 7 | Unkind | -0.684 | A | - |
| 8 | Considerate | +0.684 | A | + |
| 9 | Cooperative | +0.682 | A | + |
| 10 | Pleasant | +0.667 | A | + |

**F2** (SS 7.49; best Goldberg match C, phi = +0.525; Eval phi = +0.445)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Careless | -0.669 | C | - |
| 2 | Haphazard | -0.666 | C | - |
| 3 | Sloppy | -0.636 | C | - |
| 4 | Undependable | -0.632 | C | - |
| 5 | Unsystematic | -0.627 | C | - |
| 6 | Disorganized | -0.606 | C | - |
| 7 | Inconsistent | -0.606 | C | - |
| 8 | Unintellectual | -0.585 | I | - |
| 9 | Shallow | -0.576 | I | - |
| 10 | Unintelligent | -0.573 | I | - |

**F3** (SS 7.37; best Goldberg match I, phi = +0.696; Eval phi = +0.352)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Unimaginative | -0.773 | I | - |
| 2 | Complex | +0.676 | I | + |
| 3 | Philosophical | +0.652 | I | + |
| 4 | Uncreative | -0.644 | I | - |
| 5 | Artistic | +0.632 | I | + |
| 6 | Deep | +0.617 | I | + |
| 7 | Imaginative | +0.595 | I | + |
| 8 | Creative | +0.590 | I | + |
| 9 | Verbal | +0.567 | E | + |
| 10 | Unadventurous | -0.498 | E | - |

**F4** (SS 7.27; best Goldberg match ES, phi = +0.677; Eval phi = +0.465)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Anxious | -0.807 | ES | - |
| 2 | Fearful | -0.801 | ES | - |
| 3 | Insecure | -0.777 | ES | - |
| 4 | Fretful | -0.753 | ES | - |
| 5 | Nervous | -0.669 | ES | - |
| 6 | Self-pitying | -0.668 | ES | - |
| 7 | High-strung | -0.581 | ES | - |
| 8 | Timid | -0.578 | E | - |
| 9 | Shy | -0.543 | E | - |
| 10 | Bashful | -0.525 | E | - |

**F5** (SS 7.00; best Goldberg match C, phi = +0.596; Eval phi = +0.491)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Systematic | +0.840 | C | + |
| 2 | Neat | +0.818 | C | + |
| 3 | Conscientious | +0.780 | C | + |
| 4 | Thorough | +0.751 | C | + |
| 5 | Organized | +0.723 | C | + |
| 6 | Unemotional | +0.630 | ES | + |
| 7 | Intellectual | +0.601 | I | + |
| 8 | Prompt | +0.601 | C | + |
| 9 | Efficient | +0.561 | C | + |
| 10 | Practical | +0.535 | C | + |

**F6** (SS 6.16; best Goldberg match E, phi = +0.566; Eval phi = +0.157)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Withdrawn | -0.707 | E | - |
| 2 | Quiet | -0.700 | E | - |
| 3 | Untalkative | -0.699 | E | - |
| 4 | Reserved | -0.661 | E | - |
| 5 | Unexcitable | -0.613 | ES | + |
| 6 | Uninquisitive | -0.600 | I | - |
| 7 | Introverted | -0.531 | E | - |
| 8 | Extraverted | +0.527 | E | + |
| 9 | Talkative | +0.508 | E | + |
| 10 | Imperturbable | -0.477 | ES | + |

**F7** (SS 5.36; best Goldberg match E, phi = +0.522; Eval phi = +0.276)

| rank | trait | loading | Goldberg factor | keyed |
|---|---|---|---|---|
| 1 | Bold | +0.705 | E | + |
| 2 | Vigorous | +0.680 | E | + |
| 3 | Assertive | +0.634 | E | + |
| 4 | Active | +0.612 | E | + |
| 5 | Daring | +0.585 | E | + |
| 6 | Efficient | +0.450 | C | + |
| 7 | Demanding | +0.439 | A | - |
| 8 | Practical | +0.406 | C | + |
| 9 | Unrestrained | +0.391 | E | + |
| 10 | Unenvious | +0.374 | ES | + |

## 8. Does an evaluative factor survive rotation?

**No. Rotation dissolves it.** This is the clean answer, and it is the same on both matrices.

| solution | rotation | max Eval congruence over factors | which factor |
|---|---|---|---|
| centred_k5 | unrotated | **0.820** | F1 |
| centred_k5 | varimax | **0.656** | F1 |
| centred_k5 | oblimin | **0.571** | F2 |
| uncentred_k5 | unrotated | **0.771** | F1 |
| uncentred_k5 | varimax | **0.603** | F1 |
| uncentred_k5 | oblimin | **0.567** | F1 |
| centred_k7 | unrotated | **0.820** | F1 |
| centred_k7 | varimax | **0.616** | F1 |
| centred_k7 | oblimin | **0.584** | F3 |
| uncentred_k7 | unrotated | **0.771** | F1 |
| uncentred_k7 | varimax | **0.570** | F1 |
| uncentred_k7 | oblimin | **0.539** | F1 |

Unrotated, the first PAF factor **is** the general evaluative axis: congruence with the pure desirable/undesirable target is 0.771 (uncentred) and 0.820 (centred), against a mathematical baseline of 0.447 that any pure Goldberg factor scores automatically. It is the largest factor in both cases and it loads simultaneously on C, I and A -- exactly the PC1 the PCA found.

After oblimin, the largest Eval congruence anywhere in the solution falls to 0.567 (uncentred) and 0.571 (centred), and it is no longer concentrated: *every* rotated factor carries a similar modest amount (0.56, 0.57, 0.14, 0.48, 0.49 on the centred k=5 solution), which is barely above the 0.447 that a perfectly clean Goldberg factor scores by construction. There is no rotated factor that is predominantly evaluative.

**Interpretation.** The general evaluative factor is an artefact of unrotated extraction. The variance is real -- desirable traits genuinely do sit closer together in weight space than chance -- but it is not a separate dimension of the space; it is the sum of the five substantive factors all leaning slightly the same way, and a rotation that seeks simple structure redistributes it back into them. The weaker claim survives, the stronger one does not: there is evaluative *covariance*, there is no evaluative *factor*.

The two residues of that covariance that *do* survive rotation are (i) the non-zero off-diagonals of Phi above, and (ii) the split of Conscientiousness and Extraversion into same-pole clusters at k=7 (see section 7): the positively-keyed C markers and the negatively-keyed C markers come out as two correlated factors rather than one bipolar factor. In human data that same unipolar split is the classic signature of acquiescence, and here it is the classic signature of 'an adapter was trained toward a desirable trait' sharing variance across items.


## 9. Varimax vs oblimin

| solution | best Goldberg congruence, varimax | best, oblimin | max Eval congruence, varimax | oblimin |
|---|---|---|---|---|
| centred_k5 | E 0.69, A 0.81, C 0.71, ES 0.69, I 0.77 | E 0.71, A 0.82, C 0.74, ES 0.70, I 0.77 | 0.66 | 0.57 |
| uncentred_k5 | E 0.54, A 0.80, C 0.60, ES 0.53, I 0.60 | E 0.59, A 0.81, C 0.63, ES 0.53, I 0.62 | 0.60 | 0.57 |
| centred_k7 | E 0.61, A 0.81, C 0.72, ES 0.69, I 0.79 | E 0.60, A 0.73, C 0.67, ES 0.73, I 0.79 | 0.62 | 0.58 |
| uncentred_k7 | E 0.58, A 0.82, C 0.62, ES 0.62, I 0.62 | E 0.57, A 0.83, C 0.60, ES 0.68, I 0.70 | 0.57 | 0.54 |

The two rotations agree closely on which factor is which. Oblimin buys a little congruence on most factors and, more importantly, reports the factor correlations instead of forcing them to zero; varimax has to absorb those correlations as cross-loadings.


## 10. Communality, uniqueness, and the reseed noise floor

A reseed pair is `x = s + n1`, `x' = s + n2` with independent seed noise, so `E[cos(x,x')] = ||s||^2 / (||s||^2 + ||n||^2)` is a **test-retest reliability**. FA uniqueness `u^2 = 1 - h^2` contains that seed noise *plus* reliable trait-specific variance, so the model is only consistent if `h^2 <= reliability` for every trait.

| trait | reseed cos (uncentred) | reseed cos (ipsatised) |
|---|---|---|
| anxious | 0.8668 | 0.8296 |
| creative | 0.8508 | 0.7826 |
| organized | 0.8440 | 0.8349 |
| shy | 0.8600 | 0.8471 |
| warm | 0.8545 | 0.8431 |
| **mean** | **0.8552** | **0.8274** |

So the noise-only uniqueness floor is 1 - 0.855 = 0.145 (uncentred) / 0.173 (ipsatised). The independent distance-based figure from the PCA is 17% of squared between-trait distance attributable to seed, which is the same region.

| solution | mean h^2 | mean u^2 | min h^2 | max h^2 | reliability | traits with h^2 > reliability | u^2 / noise floor |
|---|---|---|---|---|---|---|---|
| centred_k5 | 0.531 | 0.469 | 0.269 | 0.700 | 0.827 | **0** | 2.72x |
| centred_k7 | 0.566 | 0.434 | 0.296 | 0.713 | 0.827 | **0** | 2.51x |
| uncentred_k5 | 0.567 | 0.433 | 0.198 | 0.719 | 0.855 | **0** | 2.99x |
| uncentred_k7 | 0.606 | 0.394 | 0.376 | 0.740 | 0.855 | **0** | 2.72x |

**The consistency check passes, and it is informative.** Not one trait in any solution has a communality above its reliability -- the maximum communality anywhere is 0.740 against reliabilities of 0.83-0.86 -- so the factor model never claims more common variance than the adapters actually reproduce across seeds. That is the check that could have failed and did not.

But the numbers are not equal, and the gap is the finding. Mean uniqueness is 0.469 at k=5 on the ipsatised matrix against a seed-noise floor of 0.173 -- a factor of 2.7. Only about a 37% share of what the model calls 'unique' is training-seed noise; the other ~63% is *reliable, reproducible, trait-specific* variance that five factors do not explain. Each adjective's adapter encodes something the Big Five structure does not capture, and it encodes it consistently enough to survive a reseed. Going from k=5 to k=7 recovers only +0.036 of mean communality, so the residue is not a few missing broad factors either -- it is genuinely item-specific.


## 11. Comparison with the PCA

The PCA's verdict was: only PC3 was a clean factor axis (|cos| 0.95 with Extraversion); PC1, the largest component at 22.8% of centred variance, was not any single factor but a general evaluative axis loading 0.87 on Conscientiousness and 0.82 on Intellect at once; and the diagnosis was that the factor directions are non-orthogonal in weight space so no rotation-free method could separate them.

**Does oblimin-rotated PAF recover the Big Five where PCA did not? Partly, and in the way the diagnosis predicted.**

| Goldberg factor | best PCA |cos| (any PC) | best oblimin PAF |phi| (k=5, ipsatised) |
|---|---|---|
| Extraversion | 0.95 (PC3) | 0.71 (F3) |
| Agreeableness | 0.77 (PC2) | 0.82 (F1) |
| Conscientiousness | 0.87 (PC1) | 0.74 (F2) |
| EmotionalStability | 0.56 (PC1) | 0.70 (F5) |
| Intellect | 0.82 (PC1) | 0.77 (F4) |

The two columns are not on the same scale -- |cos| between a PC direction and a factor *direction* in weight space is a different quantity from Tucker congruence between a loading vector and a marker target -- so read the structure, not the magnitudes. The structural change is the one that matters:

- **Under PCA, the five factors did not each get a component.** PC1 was shared between C and I, PC2 mixed A with emotionality, and ES and I never got a component of their own.
- **Under oblimin PAF they do.** At k=5 on the ipsatised matrix each of the five extracted factors is the unique best match for a different Goldberg factor: C, A, E, I, ES, one apiece, in that order of size. The confound between C and I that dominated PC1 is gone.
- **The mechanism is exactly the one the PCA diagnosed.** The factors are allowed to be correlated, so they no longer have to fight for orthogonal directions; what was a single blended PC1 becomes two correlated factors.
- **But the price is that congruence stops short of 'fair'.** Nothing reaches 0.85. The Big Five are recovered as an *arrangement* -- five separable, correctly-labelled factors -- not as a quantitative match to Goldberg's marker structure.
- **And parallel analysis says five is too few.** k=7 is supported, and the two extra factors are not noise: they are the same-pole splits of Conscientiousness and Extraversion, which is a substantive result about how these adapters encode a trait pair (as two correlated unipolar directions, not one bipolar one).

**Scale caveat carried over from the PCA.** Roughly 17% of squared between-trait distance is seed variance, so factors beyond the first four or five -- reduced eigenvalues below about 2 -- are in the region where reseed noise could produce comparable structure, and the k=7 factors 6 and 7 should be read as suggestive rather than established.

