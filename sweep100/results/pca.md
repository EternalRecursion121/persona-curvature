# PCA over 100 Big-Five trait LoRAs

Base model `Qwen/Qwen2.5-3B-Instruct`, 252 LoRA modules, r=16, scaling alpha/r = 2 (`dW = 2 * B @ A per module`, confirmed from adapter_config.json; use_rslora/use_dora both false).

100/100 traits mapped to adapter directories and loaded; missing: none. 5 reseed controls (`anxious__s1`, `creative__s1`, `organized__s1`, `shy__s1`, `warm__s1`) held out of the PCA and used as the noise floor.

**Verification.** Factored Gram vs dense reference on two tiny random A/B pairs: max relative error 1.81e-16 (dense [[508.2234, -52.2187], [-52.2187, 627.2123]], factored [[508.2234, -52.2187], [-52.2187, 627.2123]]). ANOVA F hand-check on groups [1,2,3],[4,5,6],[7,8,9]: F=27.000000 vs expected 27.0 (SSB 54.0, SSW 6.0).


## 1. Noise floor (read everything below against this)

Same trait, different training seed. This is how far apart two adapters are when *nothing* differs but the seed.

| trait | dist(trait, reseed) | / mean ||dW|| | cos(trait, reseed) | dist / mean between-trait dist |
|---|---|---|---|---|
| anxious | 0.6789 | 0.516 | 0.8668 | 0.389 |
| creative | 0.7441 | 0.546 | 0.8508 | 0.427 |
| organized | 0.7711 | 0.559 | 0.8440 | 0.442 |
| shy | 0.6883 | 0.529 | 0.8600 | 0.395 |
| warm | 0.7341 | 0.540 | 0.8545 | 0.421 |

- mean reseed distance **0.7233**, mean reseed cosine **0.8552**
- mean between-trait distance **1.7438**, mean between-trait cosine **0.0989**
- mean ||dW|| = 1.3139
- **between-trait / reseed distance ratio = 2.411** — the closer to 1.0, the more of the apparent trait geometry is seed noise.


## 2. Variance spectra

| PC | uncentered var % | centered var % | centered cumulative % |
|---|---|---|---|
| 1 | 21.05 | 22.80 | 22.80 |
| 2 | 14.29 | 14.52 | 37.32 |
| 3 | 12.84 | 9.72 | 47.05 |
| 4 | 8.05 | 6.14 | 53.18 |
| 5 | 3.15 | 3.12 | 56.30 |
| 6 | 2.40 | 2.31 | 58.61 |
| 7 | 2.05 | 1.95 | 60.56 |
| 8 | 1.27 | 1.40 | 61.95 |
| 9 | 1.18 | 1.27 | 63.22 |
| 10 | 1.10 | 1.07 | 64.29 |

The **uncentered** spectrum includes the grand mean: 10.9% of total uncentered energy is the shared mean dW, and uncentered PC1 has |cos| = 0.4044 with the mean direction — i.e. uncentered PC1 is essentially "an adapter was trained here", not a personality axis.
The **centered** spectrum is the one that describes how traits DIFFER; use it for everything below.


## 3. What each component is (HEADLINE)

Loadings are principal-coordinate scores, score_ik = sqrt(lambda_k) u_ik. Sign is arbitrary in PCA; convention here: each PC oriented so the trait with largest |score| is positive.


### PC1 — 22.80% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Unintellectual | +1.1046 | I | - | | Steady | -0.9453 | C | + |
| 2 | Unintelligent | +1.0548 | I | - | | Careful | -0.9046 | C | + |
| 3 | Sloppy | +0.9385 | C | - | | Helpful | -0.8832 | A | + |
| 4 | Unsophisticated | +0.8930 | I | - | | Neat | -0.8775 | C | + |
| 5 | Careless | +0.8922 | C | - | | Conscientious | -0.8640 | C | + |
| 6 | Negligent | +0.8689 | C | - | | Organized | -0.8443 | C | + |
| 7 | Haphazard | +0.8647 | C | - | | Deep | -0.8151 | I | + |
| 8 | Disorganized | +0.8534 | C | - | | Considerate | -0.8119 | A | + |
| 9 | Undependable | +0.8473 | C | - | | Intellectual | -0.8007 | I | + |
| 10 | Shallow | +0.8372 | I | - | | Cooperative | -0.7568 | A | + |

### PC2 — 14.52% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cold | +1.0364 | A | - | | Warm | -0.9966 | A | + |
| 2 | Unemotional | +0.8994 | ES | + | | Agreeable | -0.8988 | A | + |
| 3 | Unsympathetic | +0.8897 | A | - | | Sympathetic | -0.8981 | A | + |
| 4 | Unexcitable | +0.8169 | ES | + | | Pleasant | -0.8610 | A | + |
| 5 | Harsh | +0.7781 | A | - | | Emotional | -0.8029 | ES | - |
| 6 | Demanding | +0.7266 | A | - | | Kind | -0.7877 | A | + |
| 7 | Practical | +0.6648 | C | + | | Generous | -0.7792 | A | + |
| 8 | Unenvious | +0.6601 | ES | + | | Considerate | -0.6707 | A | + |
| 9 | Uncharitable | +0.6547 | A | - | | Relaxed | -0.6701 | ES | + |
| 10 | Systematic | +0.6503 | C | + | | Undemanding | -0.6599 | ES | + |

### PC3 — 9.72% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Inhibited | +0.9759 | E | - | | Energetic | -0.7939 | E | + |
| 2 | Unadventurous | +0.8239 | E | - | | Unrestrained | -0.6719 | E | + |
| 3 | Timid | +0.7628 | E | - | | Talkative | -0.6667 | E | + |
| 4 | Withdrawn | +0.7547 | E | - | | Bold | -0.6555 | E | + |
| 5 | Reserved | +0.7393 | E | - | | Vigorous | -0.6523 | E | + |
| 6 | Quiet | +0.7245 | E | - | | Extraverted | -0.6510 | E | + |
| 7 | Shy | +0.7243 | E | - | | Daring | -0.5036 | E | + |
| 8 | Introverted | +0.6640 | E | - | | Touchy | -0.4980 | ES | - |
| 9 | Unimaginative | +0.6623 | I | - | | Active | -0.4903 | E | + |
| 10 | Bashful | +0.6598 | E | - | | Verbal | -0.4709 | E | + |

### PC4 — 6.14% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Unimaginative | +0.9510 | I | - | | Complex | -0.5806 | I | + |
| 2 | Uncreative | +0.7674 | I | - | | Introspective | -0.4925 | I | + |
| 3 | Unreflective | +0.5761 | I | - | | Moody | -0.4714 | ES | - |
| 4 | Pleasant | +0.4808 | A | + | | Philosophical | -0.4669 | I | + |
| 5 | Organized | +0.4737 | C | + | | Imaginative | -0.4514 | I | + |
| 6 | Practical | +0.4632 | C | + | | Artistic | -0.4479 | I | + |
| 7 | Unadventurous | +0.4489 | E | - | | Deep | -0.4417 | I | + |
| 8 | Helpful | +0.4458 | A | + | | Jealous | -0.4074 | ES | - |
| 9 | Agreeable | +0.4383 | A | + | | Fretful | -0.3996 | ES | - |
| 10 | Cooperative | +0.4367 | A | + | | Touchy | -0.3985 | ES | - |

### PC5 — 3.12% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Anxious | +0.6356 | ES | - | | Relaxed | -0.5012 | ES | + |
| 2 | Fearful | +0.5650 | ES | - | | Imperturbable | -0.4334 | ES | + |
| 3 | Fretful | +0.5608 | ES | - | | Undemanding | -0.4165 | ES | + |
| 4 | High-strung | +0.5122 | ES | - | | Uninquisitive | -0.3974 | I | - |
| 5 | Insecure | +0.4939 | ES | - | | Unexcitable | -0.3138 | ES | + |
| 6 | Nervous | +0.3741 | ES | - | | Unenvious | -0.2935 | ES | + |
| 7 | Self-pitying | +0.3396 | ES | - | | Haphazard | -0.2890 | C | - |
| 8 | Conscientious | +0.3115 | C | + | | Reserved | -0.2828 | E | - |
| 9 | Neat | +0.2937 | C | + | | Artistic | -0.2660 | I | + |
| 10 | Organized | +0.2737 | C | + | | Complex | -0.2653 | I | + |

### PC6 — 2.31% of centered variance

| rank | + trait | loading | factor | keyed | | - trait | loading | factor | keyed |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Systematic | +0.5449 | C | + | | Bold | -0.3909 | E | + |
| 2 | Intellectual | +0.4625 | I | + | | Assertive | -0.3572 | E | + |
| 3 | Thorough | +0.4623 | C | + | | Vigorous | -0.3561 | E | + |
| 4 | Unemotional | +0.4424 | ES | + | | Moody | -0.3001 | ES | - |
| 5 | Conscientious | +0.4108 | C | + | | Daring | -0.2881 | E | + |
| 6 | Neat | +0.3928 | C | + | | Demanding | -0.2637 | A | - |
| 7 | Disorganized | +0.3037 | C | - | | Unenvious | -0.2558 | ES | + |
| 8 | Inefficient | +0.2749 | C | - | | Active | -0.2511 | E | + |
| 9 | Shallow | +0.2448 | I | - | | Untalkative | -0.2233 | E | - |
| 10 | Organized | +0.2340 | C | + | | Fearful | -0.2164 | ES | - |

(factor codes: E Extraversion, A Agreeableness, C Conscientiousness, ES EmotionalStability, I Intellect)


## 4. Ground truth: does the PCA recover the Big Five?

### 4a. Factor directions and their mutual angles

Factor direction f_F = mean(dW of +keyed markers) - mean(dW of -keyed markers).

| | E | A | C | ES | I | ||f|| / mean||dW|| | n(+/-) |
|---|---|---|---|---|---|---|---|
| **E** | +1.000 | +0.072 | +0.104 | -0.073 | +0.234 | 1.076 | 10/10 |
| **A** | +0.072 | +1.000 | +0.255 | +0.174 | +0.423 | 1.413 | 10/10 |
| **C** | +0.104 | +0.255 | +1.000 | +0.575 | +0.581 | 1.289 | 10/10 |
| **ES** | -0.073 | +0.174 | +0.575 | +1.000 | +0.210 | 0.870 | 6/14 |
| **I** | +0.234 | +0.423 | +0.581 | +0.210 | +1.000 | 1.230 | 10/10 |

### 4b. |cos| between centered PCs and factor directions

| PC | E | A | C | ES | I | max | (|cos| w/ mean dW) |
|---|---|---|---|---|---|---|---|
| PC1 | 0.171 | 0.587 | 0.874 | 0.558 | 0.818 | **0.874** | 0.162 |
| PC2 | 0.075 | 0.766 | 0.373 | 0.327 | 0.092 | **0.766** | 0.041 |
| PC3 | 0.950 | 0.123 | 0.046 | 0.193 | 0.186 | **0.950** | 0.276 |
| PC4 | 0.152 | 0.191 | 0.150 | 0.432 | 0.502 | **0.502** | 0.624 |
| PC5 | 0.057 | 0.003 | 0.201 | 0.507 | 0.032 | **0.507** | 0.174 |
| PC6 | 0.047 | 0.035 | 0.024 | 0.130 | 0.065 | **0.130** | 0.007 |

(A PC that *was* a Big Five factor would show |cos| near 1 in one column.)

### 4c. Does factor membership explain the loadings?

One-way ANOVA across the 5 factor groups, permutation p from 10000 label shuffles (p floor = 1.0e-04).

| PC | F (df 4,95) | p_perm | mean E | mean A | mean C | mean ES | mean I |
|---|---|---|---|---|---|---|---|
| PC1 | 0.707 | 0.5823 | -0.0916 | -0.0817 | -0.0256 | +0.1869 | +0.0121 |
| PC2 | 0.047 | 0.9953 | -0.0002 | -0.0321 | +0.0337 | +0.0043 | -0.0057 |
| PC3 | 0.704 | 0.5917 | +0.0719 | -0.0767 | -0.0430 | +0.0852 | -0.0374 |
| PC4 | 3.440 | 0.0131 | -0.0646 | +0.1264 | +0.0724 | -0.1799 | +0.0458 |
| PC5 | 3.021 | 0.0190 | -0.0189 | +0.0309 | +0.0246 | +0.0925 | -0.1291 |
| PC6 | 15.431 | 0.0001 | -0.0844 | -0.1169 | +0.2083 | -0.0575 | +0.0505 |

**Caveat: this raw-loading test is confounded by bipolarity.** A PC that *is* a factor pushes that factor's + and - markers to opposite signs, so the group mean is ~0 and F is small (see PC3, which has |cos| 0.95 with Extraversion yet F=0.70). Two polarity-aware variants:


(i) ANOVA on **|loading|** — do a factor's markers load *heavily* on this PC, whatever the sign?

| PC | F (df 4,95) | p_perm | mean|E| | mean|A| | mean|C| | mean|ES| | mean|I| |
|---|---|---|---|---|---|---|---|
| PC1 | 13.799 | 0.0001 | 0.3727 | 0.5528 | 0.7404 | 0.3522 | 0.6613 |
| PC2 | 16.356 | 0.0001 | 0.3130 | 0.7110 | 0.3295 | 0.3920 | 0.2294 |
| PC3 | 38.286 | 0.0001 | 0.6716 | 0.1703 | 0.1556 | 0.2599 | 0.2580 |
| PC4 | 7.584 | 0.0001 | 0.2095 | 0.1915 | 0.1868 | 0.2797 | 0.4102 |
| PC5 | 9.654 | 0.0001 | 0.1261 | 0.0951 | 0.1825 | 0.3064 | 0.1516 |
| PC6 | 2.743 | 0.0298 | 0.1581 | 0.1254 | 0.2182 | 0.1240 | 0.1202 |

(ii) ANOVA on **pole-signed loading** (loading x +1 if keyed '+', -1 if '-') — does this PC line up with the factor's own +/- axis? A large positive group mean means that factor's poles are separated along this PC.

| PC | F (df 4,95) | p_perm | mean E | mean A | mean C | mean ES | mean I |
|---|---|---|---|---|---|---|---|
| PC1 | 17.474 | 0.0001 | -0.1206 | -0.5445 | -0.7404 | -0.3426 | -0.6613 |
| PC2 | 30.162 | 0.0001 | -0.0532 | -0.7110 | +0.3162 | +0.1554 | -0.0745 |
| PC3 | 38.444 | 0.0001 | -0.6716 | +0.1143 | +0.0394 | +0.0587 | -0.1500 |
| PC4 | 39.696 | 0.0001 | +0.1076 | +0.1775 | +0.1274 | +0.2797 | -0.4056 |
| PC5 | 19.673 | 0.0001 | +0.0406 | -0.0029 | +0.1700 | -0.2807 | -0.0262 |
| PC6 | 1.566 | 0.1831 | -0.0331 | -0.0322 | +0.0203 | +0.0854 | +0.0526 |

Same test using **keyed sign** (+ vs -) instead of factor:

| PC | F (df 1,98) | p_perm |
|---|---|---|
| PC1 | 194.449 | 0.0001 |
| PC2 | 2.435 | 0.1236 |
| PC3 | 10.852 | 0.0016 |
| PC4 | 3.550 | 0.0623 |
| PC5 | 0.814 | 0.3760 |
| PC6 | 0.968 | 0.3348 |

## 5. Polarity test

If a factor is encoded as a *direction*, opposite-poled markers of the same factor should be ANTI-correlated.


**raw cos(dW_i, dW_j)**

| pair type | mean cos | sd | n pairs |
|---|---|---|---|
| same factor, SAME pole | **+0.4495** | 0.1884 | 466 |
| same factor, OPPOSITE pole | **-0.2091** | 0.1882 | 484 |
| different factor | **+0.0953** | 0.2295 | 4000 |

**cos after removing the grand mean dW**

| pair type | mean cos | sd | n pairs |
|---|---|---|---|
| same factor, SAME pole | **+0.3749** | 0.2034 | 466 |
| same factor, OPPOSITE pole | **-0.3433** | 0.1664 | 484 |
| different factor | **-0.0143** | 0.2373 | 4000 |

Per factor (raw / mean-removed):

| factor | same-pole raw | opp-pole raw | same-pole centered | opp-pole centered |
|---|---|---|---|---|
| Extraversion | +0.3867 | -0.1192 | +0.2977 | -0.2837 |
| Agreeableness | +0.5544 | -0.3617 | +0.5229 | -0.4535 |
| Conscientiousness | +0.5334 | -0.2443 | +0.4700 | -0.4073 |
| EmotionalStability | +0.3626 | -0.0324 | +0.2186 | -0.1676 |
| Intellect | +0.4260 | -0.2595 | +0.3933 | -0.3763 |

## 6. Verdict

Best-matching factor per PC (|cos|): PC1->C 0.87, PC2->A 0.77, PC3->E 0.95, PC4->I 0.50, PC5->ES 0.51, PC6->ES 0.13.

**The Big Five are partly recovered, and the polarity prediction holds cleanly.** Opposite-poled markers of the same factor are genuinely ANTI-correlated in weight space (mean cos -0.209 raw, -0.343 after removing the grand mean) against +0.450 / +0.375 for same-pole pairs and +0.095 / -0.014 across factors. So the adapters encode a signed DIRECTION per factor, not merely "this trait was trained" — that failure mode is ruled out.

But the components are not the Big Five one-for-one. Only PC3 is a clean factor axis (|cos| 0.95 with Extraversion, and its top-10 lists are pure Extraversion with the poles split). PC2 is mostly Agreeableness (0.77) but mixes in the emotionality markers. PC1, the largest component at 22.8%, is not any single factor: it loads 0.87 on Conscientiousness and 0.82 on Intellect simultaneously, and its poles are almost perfectly the keyed sign (ANOVA on keyed sign F=194, p=0.0001) — it is a general desirable/undesirable (evaluative) axis cutting across C, I and A, exactly the "big one"/social-desirability factor that shows up in human Big-Five data too. That is consistent with the factor directions themselves being non-orthogonal in weight space: C-I cos +0.58, C-ES +0.58, A-I +0.42. The Big Five are NOT orthogonal here, so no rotation-free PCA could return them as separate components.

Scale caveat: the reseed noise floor is not small. Two adapters for the SAME trait differ by 0.723 against a mean between-trait distance of 1.744 — a ratio of only 2.41. Trait identity is well above seed noise (reseed cos 0.86 vs between-trait 0.10), but roughly a 17% share of squared between-trait distance is seed variance, so components below about PC4-PC5 (3.1% and under) are at or near the level where reseed noise could produce comparable structure and should not be interpreted without a reseed-based null.

