# Gradient-content probe: fit-free analysis

_Generated 2026-08-12 18:14:27 by `analyse_gradprobe.py`._

**Question.** Does a per-document gradient encode the document's CONTENT (which fact it states) rather than its SURFACE (genre)?

**Method.** Cosine similarity and rank statistics only. No trained classifier, no fitted linear map, nothing estimated from the data except the reported statistics themselves. Inference is by permutation.

## Design

- 400 documents, 200 facts x 2 documents, 10 genres.
- The two documents of a fact are ALWAYS in different genres (200/200 verified).
- 79800 document pairs partition into SAME-FACT 200, DIFF-FACT/SAME-GENRE 7803, DIFF-FACT/DIFF-GENRE 71797.
- Headline comparison is SAME-FACT vs DIFF-FACT/DIFF-GENRE: **both sets are cross-genre**, so genre is controlled by construction.

## What was permuted

The 79,800 pairs are NOT independent -- each document appears in 399 of them -- so no naive t-test over pairs is quoted. The instrument is a permutation over FACT LABELS that preserves genre:

> Draw a bijection sigma of documents onto documents that is uniform WITHIN each genre block. Carry the true 200-pair matching through sigma. Because sigma is genre-preserving and a bijection: every document keeps its own genre and its own sketch; the null matching is again a perfect matching on all 400 documents; the multiset of (genre_a, genre_b) combinations across the 200 null pairs is IDENTICAL to the observed one. Only *which documents state the same fact* is destroyed.

Verified: genre preserved per document `True`; genre-pair multiset identical `True`; perfect matching preserved `True`; all null pairs cross-genre `True`; fraction of null pairs that are accidentally true partners 0.0032 (conservative).

## Sketch error budget (upstream-measured)

Measured by the training run itself against exact gradients on k=12 real documents (`meta.verification.sketch_fidelity_on_real_gradients`); the gate uses the larger of the measured rms and the stated figure.

| readout | rms per-pair error (measured) | max per-pair error | mean bias | r | gate used |
|---|---|---|---|---|---|
| frozen / pooled | 0.01177 | 0.03018 | +0.00493 | 0.9929 | 0.012 |
| frozen / per-layer | 0.03038 | 0.11008 | +0.00135 | 0.9382 | 0.030 |
| frozen / per-type | 0.03168 | 0.11452 | -0.00232 | 0.9402 | 0.032 |
| sequential / pooled | 0.01208 | 0.02686 | +0.00560 | 0.9935 | 0.012 |
| sequential / per-layer | 0.03057 | 0.11468 | +0.00183 | 0.9455 | 0.031 |
| sequential / per-type | 0.03164 | 0.11907 | -0.00187 | 0.9483 | 0.032 |

**Rule applied here: any cosine DIFFERENCE smaller than the relevant rms figure is not interpreted.** Rows failing this are flagged `INSIDE NOISE` in the tables below. (Caveat in both directions: a mean over 200 pairs shrinks the *random* part of that error by ~sqrt(200), but the measured mean bias, ~0.002-0.003, does not average away. We apply the strict per-pair floor as the gate.)

## Verification: synthetic ground truth

```
[selftest] statistic implementations vs published values
   norm_cdf(1.959964)       = 0.9750000000  published 0.975  OK
   norm_cdf(2.575829)       = 0.9950000000  published 0.995  OK
   norm_cdf(0)              = 0.5000000000  published 0.5  OK
   norm_cdf(1.6448536)      = 0.9500000000  published 0.95  OK
   pearson_r(Anscombe I)    = 0.8164205  published 0.8164205  OK
   ks_uniform_stat(det)     = 0.2000000  hand-computed 0.2  OK
   KS 5% critical value n=200 = 0.09603  (1.35810/sqrt(n), published)

[selftest] does the permutation preserve genre structure?
   genre of every permuted doc == genre of the doc it replaces : True
   multiset of (genre_a,genre_b) over 200 null pairs identical : True
   null matching is still a perfect matching on all 400 docs   : True
   every null pair is still cross-genre                        : True
   fraction of null pairs that are accidentally TRUE partners  : 0.0032 (makes the test slightly conservative, not anti-conservative)

[selftest] recovery on synthetic sketches with known ground truth (10000 permutations)
   case                         fact d    z_fact    p_fact  genre d  z_genre     MRR   MRRdg    top1  med.rank
   SIGNAL  fact+genre+noise     0.1413     32.79   0.00010   0.1713   178.83  0.0683  0.4012  0.0075      33.0
   SIGNAL  weak fact            0.0187      4.36   0.00010   0.1935   190.60  0.0077  0.0193  0.0000     172.0
   NULL    genre+noise only    -0.0057     -1.34   0.91321   0.1980   196.66  0.0062  0.0161  0.0000     235.5
   NULL    pure noise           0.0022      0.49   0.30937  -0.0012    -1.64  0.0154  0.0168  0.0000     194.0
   chance: MRR=0.01646 (MRRdg~0.0180) top1=0.00251 median rank=200.0
   note the NULL genre+noise row: overall MRR (0.006) sits BELOW chance because genre similarity outranks the cross-genre partner -- exactly the artefact the different-genre-only pool controls for.

[selftest] null p-value uniformity: 200 independent null datasets (genre+noise, no fact component), 2000 permutations each
   mean p = 0.5147 (want 0.5)   frac p<0.05 = 0.0300 (want 0.05)   frac p<0.10 = 0.1000 (want 0.10)
   mean z = -0.0556 (want 0)    sd z = 0.9926 (want ~1)
   KS D vs Uniform(0,1) = 0.0453   5% critical value = 0.0960   UNIFORM (not rejected)

[selftest] the fact permutation destroys the fact effect but leaves genre alone
   genre effect : observed +0.17580  across 200 permutations +0.17540 +/- 1.07e-05  (drift 4.04e-04)
   fact  effect : observed +0.14549  across 200 permutations +0.00012 +/- 3.87e-03  -> destroyed
   sd(genre)/sd(fact) = 0.0028 (genre is ~invariant; it moves only because 200 of 71997 cross-genre pairs change membership)
```

# Mode: frozen

**Verdict: case (a) -- the SAME-FACT effect is present at pooled.**
Criterion: z>=4 AND permutation p<=0.001 AND |effect| above the readout's sketch noise floor.

Sketch consistency cross-check (pooled vs concat-of-layer sketches, two independent sketches of the same gradient): rms 0.0129, max 0.0565, r=0.9744.

## frozen / pooled: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.1059 | 0.0051 | 0.0720 | -0.009 | 0.028 | 0.066 | 0.089 | 0.125 | 0.225 | 0.607 |
| DIFF-FACT / SAME-GENRE | 7803 | 0.1651 | 0.0012 | 0.1078 | -0.188 | 0.034 | 0.089 | 0.141 | 0.225 | 0.354 | 0.619 |
| DIFF-FACT / DIFF-GENRE | 71797 | 0.0570 | 0.0001 | 0.0348 | -0.211 | 0.008 | 0.035 | 0.054 | 0.076 | 0.117 | 0.384 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.0489**, permutation z = **+25.13**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00056 +/- 0.00197)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.1081**, z = +162.25 (genre-label shuffle)
- fact/genre effect ratio: +0.45

### frozen / pooled: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 75 | 0.0819 | 0.0475 | 0.0975 | 0.1300 | 0.3850 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 44 | 0.2088 | 0.1525 | 0.2575 | 0.2950 | 0.5425 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~360)_ | 180 | - | 0.0028 | 0.0139 | 0.0278 | 0.1389 |

MRR permutation test: observed 0.0819 vs null 0.0084 +/- 0.0014, z = +51.76, p = <2.5e-04.

### frozen / pooled: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0967**; over cross-genre pairs: -0.0882.
- mean |delta-norm|: SAME-FACT 0.4396 vs DIFF-FACT/DIFF-GENRE 0.4344.
- Pearson r(pair cosine, |delta tokens|): -0.0272; mean |delta tokens| SAME-FACT 26.5 vs DIFF/DIFF 27.5.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.0489** (unmatched +0.0489), z = **+25.24**, p = <2.0e-04. Survives norm matching.

## frozen / mean_over_layers: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.0989 | 0.0038 | 0.0543 | 0.014 | 0.039 | 0.067 | 0.088 | 0.113 | 0.203 | 0.414 |
| DIFF-FACT / SAME-GENRE | 7803 | 0.1342 | 0.0007 | 0.0661 | -0.100 | 0.042 | 0.086 | 0.125 | 0.180 | 0.254 | 0.359 |
| DIFF-FACT / DIFF-GENRE | 71797 | 0.0518 | 0.0001 | 0.0285 | -0.144 | 0.012 | 0.034 | 0.049 | 0.066 | 0.103 | 0.294 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.0471**, permutation z = **+31.33**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00054 +/- 0.00152)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0824**, z = +170.60 (genre-label shuffle)
- fact/genre effect ratio: +0.57

### frozen / mean_over_layers: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 62 | 0.0943 | 0.0500 | 0.1150 | 0.1525 | 0.4425 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 31 | 0.2722 | 0.2100 | 0.3175 | 0.3775 | 0.6250 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~360)_ | 180 | - | 0.0028 | 0.0139 | 0.0278 | 0.1389 |

MRR permutation test: observed 0.0943 vs null 0.0081 +/- 0.0013, z = +66.39, p = <2.5e-04.

### frozen / mean_over_layers: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0924**; over cross-genre pairs: -0.0762.
- mean |delta-norm|: SAME-FACT 0.4396 vs DIFF-FACT/DIFF-GENRE 0.4344.
- Pearson r(pair cosine, |delta tokens|): -0.0176; mean |delta tokens| SAME-FACT 26.5 vs DIFF/DIFF 27.5.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.0471** (unmatched +0.0471), z = **+31.66**, p = <2.0e-04. Survives norm matching.

## frozen: the readout axis

| readout | kind | fact effect | z | p | genre effect | MRR | MRR diff-genre | top-1 | norm-matched z | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | pooled | +0.0489 | +25.13 | <5.0e-05 | +0.1081 | 0.0819 | 0.2088 | 0.0475 | +25.24 |  |
| layer00 | per_layer | +0.0503 | +18.42 | <5.0e-05 | +0.0690 | 0.1309 | 0.2128 | 0.0950 | +18.42 |  |
| layer01 | per_layer | +0.0495 | +7.31 | <5.0e-05 | +0.0919 | 0.0620 | 0.0817 | 0.0375 | +7.27 |  |
| layer02 | per_layer | +0.0329 | +9.90 | <5.0e-05 | +0.1926 | 0.0344 | 0.0657 | 0.0175 | +10.08 |  |
| layer03 | per_layer | +0.0478 | +13.16 | <5.0e-05 | +0.0904 | 0.0808 | 0.1413 | 0.0450 | +13.20 |  |
| layer04 | per_layer | +0.0742 | +19.08 | <5.0e-05 | +0.0641 | 0.1423 | 0.2104 | 0.1000 | +19.48 |  |
| layer05 | per_layer | +0.0824 | +20.76 | <5.0e-05 | +0.0716 | 0.1559 | 0.2372 | 0.1200 | +21.17 |  |
| layer06 | per_layer | +0.0750 | +16.11 | <5.0e-05 | +0.0876 | 0.1188 | 0.1842 | 0.0900 | +16.34 |  |
| layer07 | per_layer | +0.0652 | +13.63 | <5.0e-05 | +0.0925 | 0.0908 | 0.1678 | 0.0600 | +13.74 |  |
| layer08 | per_layer | +0.0644 | +17.38 | <5.0e-05 | +0.0864 | 0.1123 | 0.1860 | 0.0800 | +17.42 |  |
| layer09 | per_layer | +0.0581 | +18.12 | <5.0e-05 | +0.0859 | 0.0945 | 0.1632 | 0.0600 | +18.22 |  |
| layer10 | per_layer | +0.0437 | +14.22 | <5.0e-05 | +0.0959 | 0.0631 | 0.1068 | 0.0375 | +14.24 |  |
| layer11 | per_layer | +0.0329 | +10.14 | <5.0e-05 | +0.0902 | 0.0386 | 0.0674 | 0.0200 | +10.18 |  |
| layer12 | per_layer | +0.0320 | +11.56 | <5.0e-05 | +0.0769 | 0.0535 | 0.0827 | 0.0325 | +11.51 |  |
| layer13 | per_layer | +0.0275 | +7.72 | <5.0e-05 | +0.0611 | 0.0436 | 0.0687 | 0.0200 | +7.86 | INSIDE NOISE |
| layer14 | per_layer | +0.0246 | +9.12 | <5.0e-05 | +0.0658 | 0.0395 | 0.0718 | 0.0150 | +9.24 | INSIDE NOISE |
| layer15 | per_layer | +0.0232 | +9.02 | <5.0e-05 | +0.0669 | 0.0351 | 0.0595 | 0.0125 | +8.93 | INSIDE NOISE |
| layer16 | per_layer | +0.0228 | +9.26 | <5.0e-05 | +0.0666 | 0.0350 | 0.0796 | 0.0125 | +9.27 | INSIDE NOISE |
| layer17 | per_layer | +0.0201 | +6.33 | <5.0e-05 | +0.0921 | 0.0261 | 0.0432 | 0.0100 | +6.45 | INSIDE NOISE |
| layer18 | per_layer | +0.0230 | +7.11 | <5.0e-05 | +0.0912 | 0.0280 | 0.0509 | 0.0100 | +7.12 | INSIDE NOISE |
| layer19 | per_layer | +0.0243 | +6.78 | <5.0e-05 | +0.0944 | 0.0241 | 0.0500 | 0.0050 | +6.99 | INSIDE NOISE |
| layer20 | per_layer | +0.0251 | +5.79 | <5.0e-05 | +0.0976 | 0.0272 | 0.0468 | 0.0075 | +5.98 | INSIDE NOISE |
| layer21 | per_layer | +0.0363 | +14.80 | <5.0e-05 | +0.0688 | 0.0562 | 0.1007 | 0.0200 | +14.94 |  |
| layer22 | per_layer | +0.0358 | +14.56 | <5.0e-05 | +0.0705 | 0.0556 | 0.0997 | 0.0175 | +14.50 |  |
| layer23 | per_layer | +0.0305 | +8.64 | <5.0e-05 | +0.0990 | 0.0385 | 0.0599 | 0.0150 | +8.09 |  |
| layer24 | per_layer | +0.0285 | +9.46 | <5.0e-05 | +0.0968 | 0.0384 | 0.0711 | 0.0175 | +9.66 | INSIDE NOISE |
| layer25 | per_layer | +0.0321 | +11.16 | <5.0e-05 | +0.1134 | 0.0316 | 0.0628 | 0.0075 | +11.22 |  |
| layer26 | per_layer | +0.0410 | +17.28 | <5.0e-05 | +0.0593 | 0.0722 | 0.1322 | 0.0250 | +17.38 |  |
| layer27 | per_layer | +0.0455 | +17.41 | <5.0e-05 | +0.0792 | 0.0655 | 0.1314 | 0.0275 | +17.28 |  |
| layer28 | per_layer | +0.0444 | +12.03 | <5.0e-05 | +0.0598 | 0.0906 | 0.1302 | 0.0425 | +11.81 |  |
| layer29 | per_layer | +0.0374 | +15.64 | <5.0e-05 | +0.0499 | 0.0722 | 0.1188 | 0.0375 | +15.63 |  |
| layer30 | per_layer | +0.0366 | +14.10 | <5.0e-05 | +0.1765 | 0.0479 | 0.0977 | 0.0225 | +14.13 |  |
| layer31 | per_layer | +0.0735 | +32.36 | <5.0e-05 | +0.0491 | 0.2429 | 0.3460 | 0.1750 | +32.25 |  |
| layer32 | per_layer | +0.0715 | +30.77 | <5.0e-05 | +0.0503 | 0.2330 | 0.3412 | 0.1575 | +31.28 |  |
| layer33 | per_layer | +0.0897 | +34.46 | <5.0e-05 | +0.0327 | 0.3208 | 0.4023 | 0.2300 | +34.41 |  |
| layer34 | per_layer | +0.1038 | +41.97 | <5.0e-05 | +0.0534 | 0.3655 | 0.5080 | 0.2800 | +42.33 |  |
| layer35 | per_layer | +0.0900 | +29.20 | <5.0e-05 | +0.0788 | 0.1698 | 0.2629 | 0.1100 | +28.96 |  |
| type:q_proj | per_type | +0.0312 | +10.58 | <5.0e-05 | +0.0677 | 0.0435 | 0.0614 | 0.0175 | +10.42 | INSIDE NOISE |
| type:k_proj | per_type | +0.0203 | +6.49 | <5.0e-05 | +0.0580 | 0.0468 | 0.0656 | 0.0225 | +6.50 | INSIDE NOISE |
| type:v_proj | per_type | +0.0422 | +12.91 | <5.0e-05 | +0.0988 | 0.0586 | 0.1090 | 0.0325 | +13.14 |  |
| type:o_proj | per_type | +0.0423 | +16.79 | <5.0e-05 | +0.0835 | 0.0717 | 0.1484 | 0.0400 | +16.51 |  |
| type:gate_proj | per_type | +0.0629 | +21.24 | <5.0e-05 | +0.0756 | 0.1199 | 0.2157 | 0.0725 | +21.36 |  |
| type:up_proj | per_type | +0.0543 | +17.72 | <5.0e-05 | +0.0861 | 0.0831 | 0.1567 | 0.0475 | +17.96 |  |
| type:down_proj | per_type | +0.0374 | +12.17 | <5.0e-05 | +0.1848 | 0.0361 | 0.0795 | 0.0175 | +12.15 |  |
| mean_over_layers | aggregate | +0.0471 | +31.33 | <5.0e-05 | +0.0824 | 0.0943 | 0.2722 | 0.0500 | +31.66 |  |
| mean_over_types | aggregate | +0.0415 | +22.33 | <5.0e-05 | +0.0935 | 0.0711 | 0.1810 | 0.0375 | +22.50 |  |
| concat_layers | aggregate | +0.0478 | +25.21 | <5.0e-05 | +0.1084 | 0.0812 | 0.2045 | 0.0500 | +25.38 |  |
| concat_types | aggregate | +0.0478 | +23.12 | <5.0e-05 | +0.1077 | 0.0738 | 0.1890 | 0.0400 | +23.19 |  |

### frozen: depth profile (per-layer fact-effect z)

```
L00 z= +18.42 ##################
L01 z=  +7.31 #######
L02 z=  +9.90 #########
L03 z= +13.16 #############
L04 z= +19.08 ##################
L05 z= +20.76 ####################
L06 z= +16.11 ###############
L07 z= +13.63 #############
L08 z= +17.38 #################
L09 z= +18.12 #################
L10 z= +14.22 ##############
L11 z= +10.14 ##########
L12 z= +11.56 ###########
L13 z=  +7.72 #######
L14 z=  +9.12 #########
L15 z=  +9.02 #########
L16 z=  +9.26 #########
L17 z=  +6.33 ######
L18 z=  +7.11 #######
L19 z=  +6.78 ######
L20 z=  +5.79 ######
L21 z= +14.80 ##############
L22 z= +14.56 ##############
L23 z=  +8.64 ########
L24 z=  +9.46 #########
L25 z= +11.16 ###########
L26 z= +17.28 ################
L27 z= +17.41 #################
L28 z= +12.03 ###########
L29 z= +15.64 ###############
L30 z= +14.10 #############
L31 z= +32.36 ###############################
L32 z= +30.77 #############################
L33 z= +34.46 #################################
L34 z= +41.97 ########################################
L35 z= +29.20 ############################
```

# Mode: sequential

**Verdict: case (a) -- the SAME-FACT effect is present at pooled.**
Criterion: z>=4 AND permutation p<=0.001 AND |effect| above the readout's sketch noise floor.

Sketch consistency cross-check (pooled vs concat-of-layer sketches, two independent sketches of the same gradient): rms 0.0129, max 0.0589, r=0.9406.

## sequential / pooled: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.0787 | 0.0038 | 0.0539 | -0.038 | 0.007 | 0.047 | 0.077 | 0.104 | 0.157 | 0.344 |
| DIFF-FACT / SAME-GENRE | 7803 | 0.0110 | 0.0005 | 0.0485 | -0.352 | -0.066 | -0.017 | 0.011 | 0.039 | 0.088 | 0.416 |
| DIFF-FACT / DIFF-GENRE | 71797 | -0.0013 | 0.0001 | 0.0360 | -0.665 | -0.059 | -0.021 | -0.001 | 0.019 | 0.055 | 0.273 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.0800**, permutation z = **+31.45**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00001 +/- 0.00255)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0123**, z = +27.59 (genre-label shuffle)
- fact/genre effect ratio: +6.51

### sequential / pooled: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 6 | 0.3479 | 0.2225 | 0.4875 | 0.6050 | 0.7900 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 3 | 0.4674 | 0.3675 | 0.5825 | 0.6425 | 0.8225 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~360)_ | 180 | - | 0.0028 | 0.0139 | 0.0278 | 0.1389 |

MRR permutation test: observed 0.3479 vs null 0.0133 +/- 0.0029, z = +115.96, p = <2.5e-04.

### sequential / pooled: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0435**; over cross-genre pairs: -0.0397.
- mean |delta-norm|: SAME-FACT 0.3905 vs DIFF-FACT/DIFF-GENRE 0.4447.
- Pearson r(pair cosine, |delta tokens|): -0.0260; mean |delta tokens| SAME-FACT 26.5 vs DIFF/DIFF 27.5.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.0797** (unmatched +0.0800), z = **+30.99**, p = <2.0e-04. Survives norm matching.

## sequential / mean_over_layers: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.0721 | 0.0032 | 0.0459 | -0.031 | 0.011 | 0.045 | 0.067 | 0.095 | 0.151 | 0.298 |
| DIFF-FACT / SAME-GENRE | 7803 | 0.0096 | 0.0004 | 0.0362 | -0.161 | -0.047 | -0.012 | 0.009 | 0.031 | 0.071 | 0.279 |
| DIFF-FACT / DIFF-GENRE | 71797 | -0.0009 | 0.0001 | 0.0216 | -0.145 | -0.036 | -0.014 | -0.001 | 0.013 | 0.034 | 0.224 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.0730**, permutation z = **+47.35**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00002 +/- 0.00154)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0105**, z = +38.42 (genre-label shuffle)
- fact/genre effect ratio: +6.92

### sequential / mean_over_layers: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 3 | 0.4488 | 0.3250 | 0.5950 | 0.7175 | 0.8775 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 1 | 0.6084 | 0.5200 | 0.7150 | 0.7800 | 0.9025 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~360)_ | 180 | - | 0.0028 | 0.0139 | 0.0278 | 0.1389 |

MRR permutation test: observed 0.4488 vs null 0.0124 +/- 0.0029, z = +150.64, p = <2.5e-04.

### sequential / mean_over_layers: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0298**; over cross-genre pairs: -0.0278.
- mean |delta-norm|: SAME-FACT 0.3905 vs DIFF-FACT/DIFF-GENRE 0.4447.
- Pearson r(pair cosine, |delta tokens|): -0.0364; mean |delta tokens| SAME-FACT 26.5 vs DIFF/DIFF 27.5.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.0729** (unmatched +0.0730), z = **+46.60**, p = <2.0e-04. Survives norm matching.

## sequential: the readout axis

| readout | kind | fact effect | z | p | genre effect | MRR | MRR diff-genre | top-1 | norm-matched z | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | pooled | +0.0800 | +31.45 | <5.0e-05 | +0.0123 | 0.3479 | 0.4674 | 0.2225 | +30.99 |  |
| layer00 | per_layer | +0.0526 | +16.65 | <5.0e-05 | +0.0055 | 0.1375 | 0.1616 | 0.0850 | +16.55 |  |
| layer01 | per_layer | +0.0513 | +9.41 | <5.0e-05 | +0.0087 | 0.0870 | 0.0991 | 0.0550 | +9.39 |  |
| layer02 | per_layer | +0.0796 | +6.28 | <5.0e-05 | +0.0271 | 0.0637 | 0.0818 | 0.0275 | +6.27 |  |
| layer03 | per_layer | +0.0621 | +15.06 | <5.0e-05 | +0.0087 | 0.1402 | 0.1597 | 0.0875 | +14.87 |  |
| layer04 | per_layer | +0.0676 | +18.54 | <5.0e-05 | +0.0032 | 0.2038 | 0.2223 | 0.1375 | +18.58 |  |
| layer05 | per_layer | +0.0816 | +25.47 | <5.0e-05 | +0.0048 | 0.2893 | 0.3157 | 0.2225 | +26.04 |  |
| layer06 | per_layer | +0.0645 | +19.59 | <5.0e-05 | +0.0059 | 0.1945 | 0.2196 | 0.1375 | +19.67 |  |
| layer07 | per_layer | +0.0654 | +21.70 | <5.0e-05 | +0.0068 | 0.1955 | 0.2341 | 0.1400 | +21.34 |  |
| layer08 | per_layer | +0.0618 | +19.32 | <5.0e-05 | +0.0063 | 0.1833 | 0.2116 | 0.1300 | +19.45 |  |
| layer09 | per_layer | +0.0598 | +20.10 | <5.0e-05 | +0.0072 | 0.1922 | 0.2289 | 0.1325 | +19.99 |  |
| layer10 | per_layer | +0.0617 | +19.83 | <5.0e-05 | +0.0094 | 0.1692 | 0.2071 | 0.1100 | +19.86 |  |
| layer11 | per_layer | +0.0602 | +19.20 | <5.0e-05 | +0.0121 | 0.1429 | 0.1803 | 0.0875 | +19.10 |  |
| layer12 | per_layer | +0.0705 | +22.85 | <5.0e-05 | +0.0118 | 0.1860 | 0.2339 | 0.1075 | +22.76 |  |
| layer13 | per_layer | +0.0637 | +19.46 | <5.0e-05 | +0.0097 | 0.1780 | 0.2111 | 0.1150 | +19.44 |  |
| layer14 | per_layer | +0.0597 | +19.63 | <5.0e-05 | +0.0105 | 0.1548 | 0.1928 | 0.0900 | +19.27 |  |
| layer15 | per_layer | +0.0604 | +20.71 | <5.0e-05 | +0.0097 | 0.1485 | 0.1878 | 0.0800 | +20.71 |  |
| layer16 | per_layer | +0.0655 | +20.83 | <5.0e-05 | +0.0108 | 0.1664 | 0.2136 | 0.0900 | +20.40 |  |
| layer17 | per_layer | +0.0653 | +18.64 | <5.0e-05 | +0.0131 | 0.1356 | 0.1688 | 0.0725 | +18.53 |  |
| layer18 | per_layer | +0.0656 | +19.36 | <5.0e-05 | +0.0109 | 0.1476 | 0.2009 | 0.0725 | +19.23 |  |
| layer19 | per_layer | +0.0688 | +17.91 | <5.0e-05 | +0.0144 | 0.1464 | 0.1843 | 0.0800 | +17.69 |  |
| layer20 | per_layer | +0.0811 | +21.72 | <5.0e-05 | +0.0141 | 0.1623 | 0.1979 | 0.0875 | +21.64 |  |
| layer21 | per_layer | +0.0872 | +21.82 | <5.0e-05 | +0.0138 | 0.2037 | 0.2526 | 0.1175 | +21.55 |  |
| layer22 | per_layer | +0.0835 | +24.93 | <5.0e-05 | +0.0136 | 0.2216 | 0.2674 | 0.1300 | +24.71 |  |
| layer23 | per_layer | +0.0808 | +24.05 | <5.0e-05 | +0.0135 | 0.2078 | 0.2522 | 0.1175 | +24.12 |  |
| layer24 | per_layer | +0.0700 | +19.58 | <5.0e-05 | +0.0126 | 0.1470 | 0.1902 | 0.0725 | +19.61 |  |
| layer25 | per_layer | +0.0861 | +20.46 | <5.0e-05 | +0.0180 | 0.1612 | 0.1979 | 0.0850 | +20.36 |  |
| layer26 | per_layer | +0.0770 | +27.52 | <5.0e-05 | +0.0100 | 0.2735 | 0.3195 | 0.1875 | +27.46 |  |
| layer27 | per_layer | +0.0706 | +24.92 | <5.0e-05 | +0.0101 | 0.2177 | 0.2507 | 0.1300 | +24.70 |  |
| layer28 | per_layer | +0.0752 | +28.52 | <5.0e-05 | +0.0088 | 0.2845 | 0.3181 | 0.1875 | +28.11 |  |
| layer29 | per_layer | +0.0587 | +22.87 | <5.0e-05 | +0.0087 | 0.2099 | 0.2402 | 0.1400 | +22.62 |  |
| layer30 | per_layer | +0.0602 | +19.09 | <5.0e-05 | +0.0131 | 0.1302 | 0.1602 | 0.0675 | +18.94 |  |
| layer31 | per_layer | +0.0789 | +28.26 | <5.0e-05 | +0.0083 | 0.2897 | 0.3342 | 0.2075 | +28.18 |  |
| layer32 | per_layer | +0.0845 | +28.73 | <5.0e-05 | +0.0079 | 0.3004 | 0.3328 | 0.2150 | +28.98 |  |
| layer33 | per_layer | +0.0996 | +35.32 | <5.0e-05 | +0.0072 | 0.4339 | 0.4892 | 0.3225 | +35.32 |  |
| layer34 | per_layer | +0.1027 | +25.02 | <5.0e-05 | +0.0100 | 0.2876 | 0.3320 | 0.2050 | +25.02 |  |
| layer35 | per_layer | +0.1433 | +28.98 | <5.0e-05 | +0.0131 | 0.3360 | 0.3799 | 0.2375 | +29.21 |  |
| type:q_proj | per_type | +0.0448 | +15.33 | <5.0e-05 | +0.0049 | 0.1235 | 0.1363 | 0.0775 | +15.29 |  |
| type:k_proj | per_type | +0.0439 | +11.89 | <5.0e-05 | +0.0052 | 0.1035 | 0.1137 | 0.0525 | +12.08 |  |
| type:v_proj | per_type | +0.0683 | +21.70 | <5.0e-05 | +0.0137 | 0.1646 | 0.2012 | 0.0875 | +21.63 |  |
| type:o_proj | per_type | +0.0640 | +25.20 | <5.0e-05 | +0.0101 | 0.2180 | 0.2628 | 0.1350 | +25.03 |  |
| type:gate_proj | per_type | +0.0718 | +28.64 | <5.0e-05 | +0.0100 | 0.2915 | 0.3351 | 0.2025 | +28.71 |  |
| type:up_proj | per_type | +0.0857 | +28.03 | <5.0e-05 | +0.0108 | 0.2634 | 0.3134 | 0.1675 | +27.67 |  |
| type:down_proj | per_type | +0.0961 | +11.86 | <5.0e-05 | +0.0215 | 0.1413 | 0.1803 | 0.0825 | +11.72 |  |
| mean_over_layers | aggregate | +0.0730 | +47.35 | <5.0e-05 | +0.0105 | 0.4488 | 0.6084 | 0.3250 | +46.60 |  |
| mean_over_types | aggregate | +0.0678 | +32.46 | <5.0e-05 | +0.0109 | 0.3425 | 0.4567 | 0.2300 | +32.29 |  |
| concat_layers | aggregate | +0.0807 | +32.59 | <5.0e-05 | +0.0123 | 0.3615 | 0.4942 | 0.2350 | +32.06 |  |
| concat_types | aggregate | +0.0803 | +30.27 | <5.0e-05 | +0.0126 | 0.3172 | 0.4252 | 0.1925 | +29.91 |  |

### sequential: depth profile (per-layer fact-effect z)

```
L00 z= +16.65 ###################
L01 z=  +9.41 ###########
L02 z=  +6.28 #######
L03 z= +15.06 #################
L04 z= +18.54 #####################
L05 z= +25.47 #############################
L06 z= +19.59 ######################
L07 z= +21.70 #########################
L08 z= +19.32 ######################
L09 z= +20.10 #######################
L10 z= +19.83 ######################
L11 z= +19.20 ######################
L12 z= +22.85 ##########################
L13 z= +19.46 ######################
L14 z= +19.63 ######################
L15 z= +20.71 #######################
L16 z= +20.83 ########################
L17 z= +18.64 #####################
L18 z= +19.36 ######################
L19 z= +17.91 ####################
L20 z= +21.72 #########################
L21 z= +21.82 #########################
L22 z= +24.93 ############################
L23 z= +24.05 ###########################
L24 z= +19.58 ######################
L25 z= +20.46 #######################
L26 z= +27.52 ###############################
L27 z= +24.92 ############################
L28 z= +28.52 ################################
L29 z= +22.87 ##########################
L30 z= +19.09 ######################
L31 z= +28.26 ################################
L32 z= +28.73 #################################
L33 z= +35.32 ########################################
L34 z= +25.02 ############################
L35 z= +28.98 #################################
```

## Sequential vs frozen

| mode | pooled effect | pooled z | best readout | its effect | its z | pooled MRR | best MRR | case |
|---|---|---|---|---|---|---|---|---|
| frozen | +0.0489 | +25.13 | layer34 | +0.1038 | +41.97 | 0.0819 | 0.3655 | a |
| sequential | +0.0800 | +31.45 | mean_over_layers | +0.0730 | +47.35 | 0.3479 | 0.4488 | a |

Conclusions AGREE across modes: {'frozen': 'a', 'sequential': 'a'}. Frozen (all gradients taken at the identical initial parameters) is the cleaner object -- every gradient lives in the same tangent space, so cosines are directly comparable. Sequential is realistic training: the parameters move between documents, so part of any similarity structure can be drift shared by temporally adjacent documents rather than content.

## Bottom line

**frozen**: case (a); pooled effect +0.0489 (z=+25.13), best readout `layer34` +0.1038 (z=+41.97), partner-retrieval MRR 0.3655 (chance 0.0165), different-genre-only MRR 0.5080.

**sequential**: case (a); pooled effect +0.0800 (z=+31.45), best readout `mean_over_layers` +0.0730 (z=+47.35), partner-retrieval MRR 0.4488 (chance 0.0165), different-genre-only MRR 0.6084.
