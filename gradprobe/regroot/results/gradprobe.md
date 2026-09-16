# Gradient-content probe: fit-free analysis

_Generated 2026-08-12 18:33:50 by `analyse_gradprobe.py`._

**Question.** Does a per-document gradient encode the document's CONTENT (which fact it states) rather than its SURFACE (genre)?

**Method.** Cosine similarity and rank statistics only. No trained classifier, no fitted linear map, nothing estimated from the data except the reported statistics themselves. Inference is by permutation.

## Design

- 400 documents, 200 facts x 2 documents, 2 genres.
- The two documents of a fact are ALWAYS in different genres (200/200 verified).
- 79800 document pairs partition into SAME-FACT 200, DIFF-FACT/SAME-GENRE 39800, DIFF-FACT/DIFF-GENRE 39800.
- Headline comparison is SAME-FACT vs DIFF-FACT/DIFF-GENRE: **both sets are cross-genre**, so genre is controlled by construction.

## What was permuted

The 79,800 pairs are NOT independent -- each document appears in 399 of them -- so no naive t-test over pairs is quoted. The instrument is a permutation over FACT LABELS that preserves genre:

> Draw a bijection sigma of documents onto documents that is uniform WITHIN each genre block. Carry the true 200-pair matching through sigma. Because sigma is genre-preserving and a bijection: every document keeps its own genre and its own sketch; the null matching is again a perfect matching on all 400 documents; the multiset of (genre_a, genre_b) combinations across the 200 null pairs is IDENTICAL to the observed one. Only *which documents state the same fact* is destroyed.

Verified: genre preserved per document `True`; genre-pair multiset identical `True`; perfect matching preserved `True`; all null pairs cross-genre `True`; fraction of null pairs that are accidentally true partners 0.0049 (conservative).

## Sketch error budget (upstream-measured)

Measured by the training run itself against exact gradients on k=12 real documents (`meta.verification.sketch_fidelity_on_real_gradients`); the gate uses the larger of the measured rms and the stated figure.

| readout | rms per-pair error (measured) | max per-pair error | mean bias | r | gate used |
|---|---|---|---|---|---|
| frozen / pooled | 0.00961 | 0.02934 | +0.00076 | 0.9982 | 0.010 |
| frozen / per-layer | 0.02943 | 0.11970 | -0.00085 | 0.9853 | 0.030 |
| frozen / per-type | 0.03180 | 0.11256 | -0.00203 | 0.9820 | 0.032 |
| sequential / pooled | 0.01036 | 0.02350 | -0.00202 | 0.9969 | 0.010 |
| sequential / per-layer | 0.03058 | 0.10078 | +0.00017 | 0.9799 | 0.031 |
| sequential / per-type | 0.03301 | 0.11838 | -0.00372 | 0.9757 | 0.033 |

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
   fraction of null pairs that are accidentally TRUE partners  : 0.0049 (makes the test slightly conservative, not anti-conservative)

[selftest] recovery on synthetic sketches with known ground truth (10000 permutations)
   case                         fact d    z_fact    p_fact  genre d  z_genre     MRR   MRRdg    top1  med.rank
   SIGNAL  fact+genre+noise     0.1434     38.38   0.00010   0.1665   226.09  0.0244  0.5379  0.0050     128.0
   SIGNAL  weak fact            0.0160      4.46   0.00010   0.1982   239.29  0.0038  0.0389  0.0000     278.5
   NULL    genre+noise only    -0.0034     -1.00   0.84162   0.2063   246.65  0.0034  0.0243  0.0000     303.5
   NULL    pure noise           0.0005      0.11   0.45505  -0.0005    -1.26  0.0187  0.0375  0.0025     203.5
   chance: MRR=0.01646 (MRRdg~0.0180) top1=0.00251 median rank=200.0
   note the NULL genre+noise row: overall MRR (0.006) sits BELOW chance because genre similarity outranks the cross-genre partner -- exactly the artefact the different-genre-only pool controls for.

[selftest] null p-value uniformity: 200 independent null datasets (genre+noise, no fact component), 2000 permutations each
   mean p = 0.5029 (want 0.5)   frac p<0.05 = 0.0150 (want 0.05)   frac p<0.10 = 0.0850 (want 0.10)
   mean z = -0.0432 (want 0)    sd z = 0.9954 (want ~1)
   KS D vs Uniform(0,1) = 0.0410   5% critical value = 0.0960   UNIFORM (not rejected)

[selftest] the fact permutation destroys the fact effect but leaves genre alone
   genre effect : observed +0.18281  across 200 permutations +0.18213 +/- 1.91e-05  (drift 6.74e-04)
   fact  effect : observed +0.13460  across 200 permutations -0.00027 +/- 3.83e-03  -> destroyed
   sd(genre)/sd(fact) = 0.0050 (genre is ~invariant; it moves only because 200 of 40000 cross-genre pairs change membership)
```

# Mode: frozen

**Verdict: case (a) -- the SAME-FACT effect is present at pooled.**
Criterion: z>=4 AND permutation p<=0.001 AND |effect| above the readout's sketch noise floor.

Sketch consistency cross-check (pooled vs concat-of-layer sketches, two independent sketches of the same gradient): rms 0.0127, max 0.0573, r=0.9869.

## frozen / pooled: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.6152 | 0.0111 | 0.1570 | 0.112 | 0.294 | 0.524 | 0.651 | 0.735 | 0.807 | 0.883 |
| DIFF-FACT / SAME-GENRE | 39800 | 0.1613 | 0.0004 | 0.0752 | -0.047 | 0.059 | 0.100 | 0.153 | 0.216 | 0.290 | 0.837 |
| DIFF-FACT / DIFF-GENRE | 39800 | 0.0710 | 0.0002 | 0.0316 | -0.040 | 0.026 | 0.051 | 0.069 | 0.088 | 0.120 | 0.733 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.5442**, permutation z = **+166.73**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00001 +/- 0.00326)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0903**, z = +232.31 (genre-label shuffle)
- fact/genre effect ratio: +6.03

### frozen / pooled: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 1 | 0.9006 | 0.8750 | 0.9300 | 0.9475 | 0.9800 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 1 | 0.9919 | 0.9850 | 1.0000 | 1.0000 | 1.0000 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~200)_ | 100 | - | 0.0050 | 0.0250 | 0.0500 | 0.2500 |

MRR permutation test: observed 0.9006 vs null 0.0092 +/- 0.0048, z = +187.25, p = <2.5e-04.

### frozen / pooled: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.4137**; over cross-genre pairs: -0.0967.
- mean |delta-norm|: SAME-FACT 2.2440 vs DIFF-FACT/DIFF-GENRE 2.2982.
- Pearson r(pair cosine, |delta tokens|): -0.4999; mean |delta tokens| SAME-FACT 161.6 vs DIFF/DIFF 161.6.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.5434** (unmatched +0.5442), z = **+167.40**, p = <2.0e-04. Survives norm matching.

## frozen / mean_over_layers: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.5884 | 0.0105 | 0.1489 | 0.163 | 0.321 | 0.492 | 0.610 | 0.709 | 0.776 | 0.858 |
| DIFF-FACT / SAME-GENRE | 39800 | 0.1483 | 0.0004 | 0.0718 | -0.013 | 0.056 | 0.086 | 0.140 | 0.202 | 0.271 | 0.695 |
| DIFF-FACT / DIFF-GENRE | 39800 | 0.0547 | 0.0001 | 0.0213 | -0.014 | 0.026 | 0.042 | 0.053 | 0.066 | 0.086 | 0.465 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.5337**, permutation z = **+176.89**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00000 +/- 0.00302)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0936**, z = +252.24 (genre-label shuffle)
- fact/genre effect ratio: +5.70

### frozen / mean_over_layers: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 1 | 0.9084 | 0.8800 | 0.9475 | 0.9650 | 0.9825 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 1 | 0.9988 | 0.9975 | 1.0000 | 1.0000 | 1.0000 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~200)_ | 100 | - | 0.0050 | 0.0250 | 0.0500 | 0.2500 |

MRR permutation test: observed 0.9084 vs null 0.0088 +/- 0.0048, z = +188.38, p = <2.5e-04.

### frozen / mean_over_layers: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.4213**; over cross-genre pairs: -0.0718.
- mean |delta-norm|: SAME-FACT 2.2440 vs DIFF-FACT/DIFF-GENRE 2.2982.
- Pearson r(pair cosine, |delta tokens|): -0.5318; mean |delta tokens| SAME-FACT 161.6 vs DIFF/DIFF 161.6.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.5333** (unmatched +0.5337), z = **+177.94**, p = <2.0e-04. Survives norm matching.

## frozen: the readout axis

| readout | kind | fact effect | z | p | genre effect | MRR | MRR diff-genre | top-1 | norm-matched z | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | pooled | +0.5442 | +166.73 | <5.0e-05 | +0.0903 | 0.9006 | 0.9919 | 0.8750 | +167.40 |  |
| layer00 | per_layer | +0.6325 | +145.69 | <5.0e-05 | +0.0442 | 0.9370 | 0.9777 | 0.9150 | +146.41 |  |
| layer01 | per_layer | +0.5887 | +72.42 | <5.0e-05 | +0.0584 | 0.8340 | 0.9299 | 0.7850 | +73.28 |  |
| layer02 | per_layer | +0.5541 | +119.56 | <5.0e-05 | +0.0749 | 0.9048 | 0.9879 | 0.8675 | +118.72 |  |
| layer03 | per_layer | +0.5391 | +120.08 | <5.0e-05 | +0.0698 | 0.8657 | 0.9704 | 0.8300 | +119.65 |  |
| layer04 | per_layer | +0.5422 | +125.69 | <5.0e-05 | +0.0760 | 0.8555 | 0.9666 | 0.8225 | +127.64 |  |
| layer05 | per_layer | +0.5314 | +127.07 | <5.0e-05 | +0.0913 | 0.8016 | 0.9548 | 0.7625 | +128.48 |  |
| layer06 | per_layer | +0.5091 | +103.59 | <5.0e-05 | +0.0831 | 0.7978 | 0.9286 | 0.7600 | +105.77 |  |
| layer07 | per_layer | +0.5083 | +121.74 | <5.0e-05 | +0.0786 | 0.8014 | 0.9334 | 0.7625 | +121.70 |  |
| layer08 | per_layer | +0.5169 | +126.97 | <5.0e-05 | +0.0753 | 0.8095 | 0.9558 | 0.7700 | +128.08 |  |
| layer09 | per_layer | +0.5364 | +132.80 | <5.0e-05 | +0.0743 | 0.8372 | 0.9590 | 0.8025 | +133.52 |  |
| layer10 | per_layer | +0.5185 | +134.55 | <5.0e-05 | +0.0923 | 0.8005 | 0.9716 | 0.7550 | +135.92 |  |
| layer11 | per_layer | +0.5233 | +137.73 | <5.0e-05 | +0.1076 | 0.7864 | 0.9665 | 0.7475 | +139.05 |  |
| layer12 | per_layer | +0.5626 | +143.98 | <5.0e-05 | +0.0909 | 0.8495 | 0.9785 | 0.8250 | +144.90 |  |
| layer13 | per_layer | +0.5739 | +135.35 | <5.0e-05 | +0.0727 | 0.8795 | 0.9834 | 0.8525 | +134.21 |  |
| layer14 | per_layer | +0.5737 | +149.87 | <5.0e-05 | +0.0881 | 0.8951 | 0.9867 | 0.8700 | +148.94 |  |
| layer15 | per_layer | +0.5835 | +150.20 | <5.0e-05 | +0.0822 | 0.9076 | 0.9919 | 0.8875 | +148.44 |  |
| layer16 | per_layer | +0.6012 | +152.99 | <5.0e-05 | +0.0709 | 0.9236 | 0.9889 | 0.9025 | +153.25 |  |
| layer17 | per_layer | +0.5322 | +146.92 | <5.0e-05 | +0.0923 | 0.8692 | 0.9842 | 0.8475 | +147.54 |  |
| layer18 | per_layer | +0.5422 | +144.32 | <5.0e-05 | +0.0919 | 0.8802 | 0.9889 | 0.8600 | +142.25 |  |
| layer19 | per_layer | +0.5271 | +146.89 | <5.0e-05 | +0.1003 | 0.8823 | 0.9877 | 0.8650 | +145.45 |  |
| layer20 | per_layer | +0.4941 | +140.55 | <5.0e-05 | +0.1200 | 0.8382 | 0.9813 | 0.8100 | +139.40 |  |
| layer21 | per_layer | +0.5579 | +152.85 | <5.0e-05 | +0.0949 | 0.9176 | 0.9975 | 0.9025 | +153.36 |  |
| layer22 | per_layer | +0.5923 | +155.06 | <5.0e-05 | +0.0917 | 0.9309 | 0.9971 | 0.9150 | +156.04 |  |
| layer23 | per_layer | +0.4967 | +143.38 | <5.0e-05 | +0.0965 | 0.9054 | 0.9906 | 0.8925 | +142.17 |  |
| layer24 | per_layer | +0.4859 | +140.63 | <5.0e-05 | +0.1323 | 0.8397 | 0.9853 | 0.8175 | +138.92 |  |
| layer25 | per_layer | +0.4267 | +141.21 | <5.0e-05 | +0.1553 | 0.7522 | 0.9774 | 0.7150 | +141.31 |  |
| layer26 | per_layer | +0.5550 | +152.68 | <5.0e-05 | +0.0895 | 0.9107 | 1.0000 | 0.8875 | +153.69 |  |
| layer27 | per_layer | +0.4973 | +145.17 | <5.0e-05 | +0.1385 | 0.8352 | 0.9962 | 0.8100 | +146.98 |  |
| layer28 | per_layer | +0.5880 | +122.47 | <5.0e-05 | +0.0790 | 0.9544 | 1.0000 | 0.9400 | +122.51 |  |
| layer29 | per_layer | +0.5523 | +150.89 | <5.0e-05 | +0.0871 | 0.9363 | 1.0000 | 0.9125 | +151.30 |  |
| layer30 | per_layer | +0.6512 | +122.03 | <5.0e-05 | +0.0556 | 0.9502 | 0.9983 | 0.9200 | +122.04 |  |
| layer31 | per_layer | +0.5284 | +153.72 | <5.0e-05 | +0.0962 | 0.9465 | 1.0000 | 0.9300 | +154.11 |  |
| layer32 | per_layer | +0.4889 | +148.19 | <5.0e-05 | +0.1053 | 0.9147 | 1.0000 | 0.8925 | +148.98 |  |
| layer33 | per_layer | +0.5373 | +141.22 | <5.0e-05 | +0.0766 | 0.9533 | 1.0000 | 0.9225 | +140.74 |  |
| layer34 | per_layer | +0.4373 | +145.22 | <5.0e-05 | +0.1670 | 0.8201 | 1.0000 | 0.7800 | +144.78 |  |
| layer35 | per_layer | +0.3259 | +109.82 | <5.0e-05 | +0.1684 | 0.5256 | 0.9917 | 0.4350 | +110.31 |  |
| type:q_proj | per_type | +0.4843 | +139.84 | <5.0e-05 | +0.0935 | 0.8597 | 0.9825 | 0.8250 | +140.86 |  |
| type:k_proj | per_type | +0.4928 | +124.79 | <5.0e-05 | +0.1050 | 0.8470 | 0.9817 | 0.8200 | +122.45 |  |
| type:v_proj | per_type | +0.4683 | +141.02 | <5.0e-05 | +0.1172 | 0.8207 | 0.9801 | 0.7875 | +141.78 |  |
| type:o_proj | per_type | +0.5259 | +150.33 | <5.0e-05 | +0.0933 | 0.8849 | 0.9927 | 0.8550 | +151.30 |  |
| type:gate_proj | per_type | +0.5994 | +145.64 | <5.0e-05 | +0.0722 | 0.9191 | 0.9845 | 0.8950 | +145.08 |  |
| type:up_proj | per_type | +0.5423 | +147.83 | <5.0e-05 | +0.0934 | 0.8714 | 0.9848 | 0.8450 | +148.75 |  |
| type:down_proj | per_type | +0.5678 | +128.83 | <5.0e-05 | +0.0736 | 0.9224 | 0.9925 | 0.8950 | +127.94 |  |
| mean_over_layers | aggregate | +0.5337 | +176.89 | <5.0e-05 | +0.0936 | 0.9084 | 0.9988 | 0.8800 | +177.94 |  |
| mean_over_types | aggregate | +0.5258 | +168.53 | <5.0e-05 | +0.0926 | 0.8938 | 0.9933 | 0.8675 | +168.62 |  |
| concat_layers | aggregate | +0.5467 | +168.64 | <5.0e-05 | +0.0888 | 0.8993 | 0.9944 | 0.8725 | +169.78 |  |
| concat_types | aggregate | +0.5454 | +165.62 | <5.0e-05 | +0.0886 | 0.9028 | 0.9910 | 0.8775 | +166.05 |  |

### frozen: depth profile (per-layer fact-effect z)

```
L00 z=+145.69 ######################################
L01 z= +72.42 ###################
L02 z=+119.56 ###############################
L03 z=+120.08 ###############################
L04 z=+125.69 ################################
L05 z=+127.07 #################################
L06 z=+103.59 ###########################
L07 z=+121.74 ###############################
L08 z=+126.97 #################################
L09 z=+132.80 ##################################
L10 z=+134.55 ###################################
L11 z=+137.73 ####################################
L12 z=+143.98 #####################################
L13 z=+135.35 ###################################
L14 z=+149.87 #######################################
L15 z=+150.20 #######################################
L16 z=+152.99 #######################################
L17 z=+146.92 ######################################
L18 z=+144.32 #####################################
L19 z=+146.89 ######################################
L20 z=+140.55 ####################################
L21 z=+152.85 #######################################
L22 z=+155.06 ########################################
L23 z=+143.38 #####################################
L24 z=+140.63 ####################################
L25 z=+141.21 ####################################
L26 z=+152.68 #######################################
L27 z=+145.17 #####################################
L28 z=+122.47 ################################
L29 z=+150.89 #######################################
L30 z=+122.03 ###############################
L31 z=+153.72 ########################################
L32 z=+148.19 ######################################
L33 z=+141.22 ####################################
L34 z=+145.22 #####################################
L35 z=+109.82 ############################
```

# Mode: sequential

**Verdict: case (a) -- the SAME-FACT effect is present at pooled.**
Criterion: z>=4 AND permutation p<=0.001 AND |effect| above the readout's sketch noise floor.

Sketch consistency cross-check (pooled vs concat-of-layer sketches, two independent sketches of the same gradient): rms 0.0128, max 0.0700, r=0.9701.

## sequential / pooled: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.2789 | 0.0111 | 0.1564 | -0.800 | 0.105 | 0.204 | 0.274 | 0.356 | 0.520 | 0.658 |
| DIFF-FACT / SAME-GENRE | 39800 | 0.0035 | 0.0003 | 0.0573 | -0.961 | -0.079 | -0.026 | 0.002 | 0.032 | 0.089 | 0.905 |
| DIFF-FACT / DIFF-GENRE | 39800 | -0.0040 | 0.0002 | 0.0408 | -0.906 | -0.065 | -0.023 | -0.004 | 0.015 | 0.056 | 0.921 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.2829**, permutation z = **+86.20**, p(one-sided) = **<5.0e-05** (20000 permutations; null -0.00000 +/- 0.00328)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0074**, z = +21.76 (genre-label shuffle)
- fact/genre effect ratio: +37.98

### sequential / pooled: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 1 | 0.8765 | 0.8325 | 0.9325 | 0.9500 | 0.9700 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 1 | 0.9537 | 0.9450 | 0.9625 | 0.9675 | 0.9750 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~200)_ | 100 | - | 0.0050 | 0.0250 | 0.0500 | 0.2500 |

MRR permutation test: observed 0.8765 vs null 0.0133 +/- 0.0048, z = +181.09, p = <2.5e-04.

### sequential / pooled: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0019**; over cross-genre pairs: +0.0149.
- mean |delta-norm|: SAME-FACT 1.3000 vs DIFF-FACT/DIFF-GENRE 2.8480.
- Pearson r(pair cosine, |delta tokens|): -0.0588; mean |delta tokens| SAME-FACT 161.6 vs DIFF/DIFF 161.6.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.2828** (unmatched +0.2829), z = **+85.92**, p = <2.0e-04. Survives norm matching.

## sequential / mean_over_layers: three-way cosine contrast

| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SAME-FACT (all cross-genre) | 200 | 0.2787 | 0.0063 | 0.0895 | 0.061 | 0.144 | 0.215 | 0.270 | 0.340 | 0.422 | 0.554 |
| DIFF-FACT / SAME-GENRE | 39800 | 0.0030 | 0.0002 | 0.0335 | -0.173 | -0.050 | -0.018 | 0.002 | 0.023 | 0.057 | 0.310 |
| DIFF-FACT / DIFF-GENRE | 39800 | -0.0029 | 0.0001 | 0.0176 | -0.084 | -0.032 | -0.014 | -0.003 | 0.008 | 0.026 | 0.123 |

- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): **+0.2817**, permutation z = **+146.92**, p(one-sided) = **<5.0e-05** (20000 permutations; null +0.00000 +/- 0.00192)
- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): **+0.0059**, z = +31.78 (genre-label shuffle)
- fact/genre effect ratio: +47.86

### sequential / mean_over_layers: retrieval of the partner document

| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |
|---|---|---|---|---|---|---|
| all 399 others | 1 | 0.9621 | 0.9375 | 0.9950 | 1.0000 | 1.0000 |
| DIFFERENT-GENRE only (genre shortcut blocked) | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| _chance (all)_ | 200 | 0.0165 | 0.0025 | 0.0125 | 0.0251 | 0.1253 |
| _chance (diff-genre pool ~200)_ | 100 | - | 0.0050 | 0.0250 | 0.0500 | 0.2500 |

MRR permutation test: observed 0.9621 vs null 0.0119 +/- 0.0049, z = +192.76, p = <2.5e-04.

### sequential / mean_over_layers: norm control

- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: **-0.0006**; over cross-genre pairs: +0.0044.
- mean |delta-norm|: SAME-FACT 1.3000 vs DIFF-FACT/DIFF-GENRE 2.8480.
- Pearson r(pair cosine, |delta tokens|): -0.0792; mean |delta tokens| SAME-FACT 161.6 vs DIFF/DIFF 161.6.
- **Norm-matched** contrast (10 |delta-norm| strata): **+0.2816** (unmatched +0.2817), z = **+146.06**, p = <2.0e-04. Survives norm matching.

## sequential: the readout axis

| readout | kind | fact effect | z | p | genre effect | MRR | MRR diff-genre | top-1 | norm-matched z | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| pooled | pooled | +0.2829 | +86.20 | <5.0e-05 | +0.0074 | 0.8765 | 0.9537 | 0.8325 | +85.92 |  |
| layer00 | per_layer | +0.3216 | +100.81 | <5.0e-05 | +0.0013 | 0.9008 | 0.9301 | 0.8725 | +100.95 |  |
| layer01 | per_layer | +0.4365 | +50.94 | <5.0e-05 | +0.0037 | 0.7225 | 0.7981 | 0.6600 | +50.69 |  |
| layer02 | per_layer | +0.5467 | +29.85 | <5.0e-05 | +0.0134 | 0.6948 | 0.7365 | 0.6550 | +29.79 |  |
| layer03 | per_layer | +0.4016 | +51.56 | <5.0e-05 | +0.0061 | 0.7941 | 0.8517 | 0.7525 | +51.19 |  |
| layer04 | per_layer | +0.3209 | +85.18 | <5.0e-05 | +0.0015 | 0.9012 | 0.9657 | 0.8600 | +85.15 |  |
| layer05 | per_layer | +0.3119 | +90.56 | <5.0e-05 | +0.0021 | 0.9123 | 0.9557 | 0.8800 | +89.60 |  |
| layer06 | per_layer | +0.2968 | +83.75 | <5.0e-05 | +0.0029 | 0.8523 | 0.9235 | 0.8125 | +84.36 |  |
| layer07 | per_layer | +0.3285 | +93.81 | <5.0e-05 | +0.0030 | 0.9152 | 0.9607 | 0.8875 | +94.49 |  |
| layer08 | per_layer | +0.3359 | +98.99 | <5.0e-05 | +0.0028 | 0.9220 | 0.9689 | 0.8925 | +98.14 |  |
| layer09 | per_layer | +0.3216 | +102.40 | <5.0e-05 | +0.0027 | 0.9234 | 0.9661 | 0.8950 | +102.88 |  |
| layer10 | per_layer | +0.3090 | +95.65 | <5.0e-05 | +0.0021 | 0.9096 | 0.9725 | 0.8775 | +97.50 |  |
| layer11 | per_layer | +0.3278 | +102.10 | <5.0e-05 | +0.0028 | 0.9377 | 0.9871 | 0.8975 | +101.68 |  |
| layer12 | per_layer | +0.3428 | +105.92 | <5.0e-05 | +0.0032 | 0.9349 | 0.9950 | 0.8925 | +104.87 |  |
| layer13 | per_layer | +0.2928 | +92.76 | <5.0e-05 | +0.0040 | 0.8732 | 0.9735 | 0.8200 | +92.40 |  |
| layer14 | per_layer | +0.2854 | +93.29 | <5.0e-05 | +0.0024 | 0.8638 | 0.9660 | 0.8075 | +92.38 |  |
| layer15 | per_layer | +0.2655 | +88.90 | <5.0e-05 | +0.0028 | 0.8399 | 0.9537 | 0.7775 | +88.79 |  |
| layer16 | per_layer | +0.2518 | +87.70 | <5.0e-05 | +0.0019 | 0.8103 | 0.9581 | 0.7375 | +87.25 |  |
| layer17 | per_layer | +0.2330 | +81.02 | <5.0e-05 | +0.0028 | 0.7728 | 0.9498 | 0.7025 | +81.87 |  |
| layer18 | per_layer | +0.2361 | +78.54 | <5.0e-05 | +0.0030 | 0.7421 | 0.9189 | 0.6600 | +77.35 |  |
| layer19 | per_layer | +0.2547 | +86.40 | <5.0e-05 | +0.0032 | 0.8122 | 0.9622 | 0.7450 | +86.41 |  |
| layer20 | per_layer | +0.1915 | +64.08 | <5.0e-05 | +0.0042 | 0.5776 | 0.8690 | 0.4675 | +63.90 |  |
| layer21 | per_layer | +0.1849 | +63.94 | <5.0e-05 | +0.0031 | 0.5630 | 0.8766 | 0.4575 | +63.08 |  |
| layer22 | per_layer | +0.2254 | +82.19 | <5.0e-05 | +0.0046 | 0.7393 | 0.9614 | 0.6400 | +82.44 |  |
| layer23 | per_layer | +0.1826 | +63.29 | <5.0e-05 | +0.0066 | 0.5623 | 0.8594 | 0.4425 | +63.21 |  |
| layer24 | per_layer | +0.1719 | +57.36 | <5.0e-05 | +0.0094 | 0.4740 | 0.8102 | 0.3650 | +57.33 |  |
| layer25 | per_layer | +0.1929 | +56.51 | <5.0e-05 | +0.0170 | 0.4840 | 0.8153 | 0.3675 | +55.78 |  |
| layer26 | per_layer | +0.2011 | +74.83 | <5.0e-05 | +0.0141 | 0.6900 | 0.9549 | 0.5725 | +74.70 |  |
| layer27 | per_layer | +0.1843 | +67.63 | <5.0e-05 | +0.0101 | 0.6478 | 0.8831 | 0.5400 | +68.21 |  |
| layer28 | per_layer | +0.1933 | +66.44 | <5.0e-05 | +0.0115 | 0.6005 | 0.8999 | 0.4750 | +66.54 |  |
| layer29 | per_layer | +0.1775 | +66.07 | <5.0e-05 | +0.0153 | 0.5563 | 0.8597 | 0.4525 | +65.64 |  |
| layer30 | per_layer | +0.3751 | +67.22 | <5.0e-05 | +0.0090 | 0.8467 | 0.9447 | 0.7900 | +66.96 |  |
| layer31 | per_layer | +0.2947 | +98.73 | <5.0e-05 | +0.0111 | 0.9054 | 0.9804 | 0.8575 | +100.69 |  |
| layer32 | per_layer | +0.2723 | +93.64 | <5.0e-05 | +0.0085 | 0.8622 | 0.9796 | 0.8000 | +92.83 |  |
| layer33 | per_layer | +0.3384 | +108.93 | <5.0e-05 | +0.0043 | 0.9381 | 0.9956 | 0.8950 | +109.40 |  |
| layer34 | per_layer | +0.2524 | +84.58 | <5.0e-05 | +0.0051 | 0.6610 | 0.9756 | 0.5275 | +83.88 |  |
| layer35 | per_layer | +0.2802 | +64.58 | <5.0e-05 | +0.0101 | 0.7185 | 0.9252 | 0.6075 | +64.91 |  |
| type:q_proj | per_type | +0.2041 | +72.92 | <5.0e-05 | +0.0030 | 0.7522 | 0.9081 | 0.6800 | +72.32 |  |
| type:k_proj | per_type | +0.2172 | +58.32 | <5.0e-05 | +0.0033 | 0.6714 | 0.8201 | 0.5825 | +58.66 |  |
| type:v_proj | per_type | +0.2415 | +73.87 | <5.0e-05 | +0.0063 | 0.7522 | 0.9176 | 0.6700 | +73.06 |  |
| type:o_proj | per_type | +0.2330 | +85.95 | <5.0e-05 | +0.0037 | 0.8650 | 0.9626 | 0.8100 | +86.71 |  |
| type:gate_proj | per_type | +0.2372 | +76.69 | <5.0e-05 | +0.0050 | 0.8628 | 0.9481 | 0.8125 | +77.11 |  |
| type:up_proj | per_type | +0.2264 | +79.55 | <5.0e-05 | +0.0062 | 0.8193 | 0.9440 | 0.7450 | +79.68 |  |
| type:down_proj | per_type | +0.4292 | +47.47 | <5.0e-05 | +0.0149 | 0.7910 | 0.8638 | 0.7400 | +47.21 |  |
| mean_over_layers | aggregate | +0.2817 | +146.92 | <5.0e-05 | +0.0059 | 0.9621 | 1.0000 | 0.9375 | +146.06 |  |
| mean_over_types | aggregate | +0.2555 | +110.60 | <5.0e-05 | +0.0061 | 0.9237 | 0.9759 | 0.8900 | +110.64 |  |
| concat_layers | aggregate | +0.2842 | +87.59 | <5.0e-05 | +0.0076 | 0.8789 | 0.9551 | 0.8375 | +87.13 |  |
| concat_types | aggregate | +0.2839 | +85.47 | <5.0e-05 | +0.0073 | 0.8759 | 0.9556 | 0.8300 | +85.24 |  |

### sequential: depth profile (per-layer fact-effect z)

```
L00 z=+100.81 #####################################
L01 z= +50.94 ###################
L02 z= +29.85 ###########
L03 z= +51.56 ###################
L04 z= +85.18 ###############################
L05 z= +90.56 #################################
L06 z= +83.75 ###############################
L07 z= +93.81 ##################################
L08 z= +98.99 ####################################
L09 z=+102.40 ######################################
L10 z= +95.65 ###################################
L11 z=+102.10 #####################################
L12 z=+105.92 #######################################
L13 z= +92.76 ##################################
L14 z= +93.29 ##################################
L15 z= +88.90 #################################
L16 z= +87.70 ################################
L17 z= +81.02 ##############################
L18 z= +78.54 #############################
L19 z= +86.40 ################################
L20 z= +64.08 ########################
L21 z= +63.94 #######################
L22 z= +82.19 ##############################
L23 z= +63.29 #######################
L24 z= +57.36 #####################
L25 z= +56.51 #####################
L26 z= +74.83 ###########################
L27 z= +67.63 #########################
L28 z= +66.44 ########################
L29 z= +66.07 ########################
L30 z= +67.22 #########################
L31 z= +98.73 ####################################
L32 z= +93.64 ##################################
L33 z=+108.93 ########################################
L34 z= +84.58 ###############################
L35 z= +64.58 ########################
```

## Sequential vs frozen

| mode | pooled effect | pooled z | best readout | its effect | its z | pooled MRR | best MRR | case |
|---|---|---|---|---|---|---|---|---|
| frozen | +0.5442 | +166.73 | mean_over_layers | +0.5337 | +176.89 | 0.9006 | 0.9084 | a |
| sequential | +0.2829 | +86.20 | mean_over_layers | +0.2817 | +146.92 | 0.8765 | 0.9621 | a |

Conclusions AGREE across modes: {'frozen': 'a', 'sequential': 'a'}. Frozen (all gradients taken at the identical initial parameters) is the cleaner object -- every gradient lives in the same tangent space, so cosines are directly comparable. Sequential is realistic training: the parameters move between documents, so part of any similarity structure can be drift shared by temporally adjacent documents rather than content.

## Bottom line

**frozen**: case (a); pooled effect +0.5442 (z=+166.73), best readout `mean_over_layers` +0.5337 (z=+176.89), partner-retrieval MRR 0.9084 (chance 0.0165), different-genre-only MRR 0.9988.

**sequential**: case (a); pooled effect +0.2829 (z=+86.20), best readout `mean_over_layers` +0.2817 (z=+146.92), partner-retrieval MRR 0.9621 (chance 0.0165), different-genre-only MRR 1.0000.
