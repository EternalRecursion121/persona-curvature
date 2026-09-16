---
title: Gradient atoms on the zoo's own preference data
summary: Rosser's Gradient Atoms pipeline ported to the zoo - per-pair DPO gradients with respect to LoRA-B at B = 0, EKFAC-projected to 6,944 dimensions and decomposed into 200 sparse atoms - recovers bipolar trait axes without labels (trait purity 0.1632 against a label-shuffle null of 0.1000, z 32.4) but only weakly, no atom clears coherence 0.5, four of the five factor directions have a nearest atom above the 20-random-merge band in weight space, and atom coherence predicts nothing about judged steerability.
status: current
sources:
  - qwen35/analysis/gradient_atoms.json
  - qwen35/analysis/gradient_atoms_extract_zoo.json
  - qwen35/analysis/gradient_atoms_atoms_zoo.json
  - qwen35/analysis/gradient_atoms_weightspace.json
  - qwen35/gradient_atoms_on_modal.py
  - qwen35/build_gradatoms_inputs.py
  - qwen35/analyse_gradient_atoms.py
  - qwen35/results/gradient_atoms/zoo_atoms.npz
  - qwen35/results/gradient_atoms/zoo_dict_real_200_0.1.npz
  - qwen35/phase10_runs/gradatoms.log
  - qwen35/phase10_runs/gradatoms_items.json
  - qwen35/phase10_runs/gradatoms_labels.json
  - qwen35/phase10_runs/gradatoms_targets.json
  - https://arxiv.org/abs/2603.14665
last_verified: 2026-09-10
tags: [geometry, gradients, nulls, method, literature]
---

# Gradient atoms on the zoo's own preference data

This is experiment **G2** (with **G4** inside it) of [[paper-reading-2026-09-09]],
run on 2026-09-09/10. The question it asks: **are the five factors already in the
training gradients, before any adapter is trained, and does an unsupervised
decomposition find them without ever seeing a trait label?**

The method is J Rosser's *Gradient Atoms* (arXiv:2603.14665v2) - per-document
training gradients, projected into an EKFAC-preconditioned eigenspace, decomposed
by sparse dictionary learning, scored by the agreement among each atom's top
documents, and unprojected back to weight space. The paper ran it on 5,000
general instruction-tuning documents for Gemma-3 4B; this runs it on the zoo's
own DPO preference pairs for Qwen3.5-4B.

**Short answer.** Something is there and it is the right shape - the strongest
atoms are bipolar trait axes, discovered with no labels, whose two poles are
opposite-keyed adjectives - but it is weak. No atom reaches the coherence of the
paper's weakest headline atom. Label purity beats a label shuffle by many
standard deviations while remaining close to chance in absolute terms. Four of
the five factor directions are nearer to some atom than any of twenty random
merges are, and the fifth is not. And coherence predicts judged steerability
neither across directions nor across traits, which is what the paper found on
five atoms and this finds on 22 directions and 100 traits.

---

## What was differentiated

A zoo adapter starts at `B = 0`, so `dL/dA = B^T (dL/dW) = 0` and the whole first
update lives in `B` - the fact [[n-by-n-scoring]] and [[scoring-identity]] both
rest on. Per module the gradient is

```
dL/dB = scale * sum_t delta_t (A_0 x_t)^T          delta_t = dL/dy_t
```

which is `d_out x 64`. Over the zoo's **248 modules** with
`sum d_out = 1,132,032` that is **72,450,048 numbers per document**
(`analysis/gradient_atoms.json#extraction.raw_dim_per_doc`), so the projection is
mandatory, exactly as the paper says.

**The loss.** The zoo's recipe is `loss_type ["sigmoid", "sft"]` at weights
`[1.0, 0.1]`, `beta` 0.1, plus OCT's separate squared-approximate-KL term at
coefficient 0.001 (`phase2_runs/archive/phase5_sweep_134.json`,
[[stage-one-training-config]]). At `B = 0` the policy equals the reference, so
the DPO sigmoid sits at exactly 1/2 and the KL term's gradient is exactly zero.
The per-pair gradient is therefore

```
g_pair = -0.05 (u_c - u_r) - 0.1 u_c / n_c       u = grad_B log p(completion)
```

**Caveat stated once, because it is the one place this is not exact.** trl
1.10.0's `"sft"` loss type is `F.cross_entropy` over the chosen completions of
the *whole batch* - a token mean pooled across examples
(`trl/trainer/dpo_trainer.py`, the `elif loss_type == "sft"` branch). A
per-document decomposition of a batch-pooled mean does not exist; each pair's own
token mean is used, which is the natural per-example decomposition. At beta 0.1
the SFT term carries a factor `0.1/n_c` against the DPO term's `0.05`, so with a
median completion of 95 tokens it is about 2% of the gradient.

**Per-document gradients without a backward pass per document.** `B` enters the
forward pass only as `y = W x + scale * B (A_0 x)`, so its gradient for one
sequence is a single outer product over that sequence's tokens. A forward hook
stashes `h = A_0 x` and a tensor hook on the module's output supplies `delta`;
the einsum `scale * einsum("bso,bsr->bor", delta, h)` is then the exact
per-sequence gradient for a whole batch out of **one** backward pass. No `B`
parameter is allocated on the extraction path at all.

**That claim is checked, not asserted.** Real zero-initialised `B` parameters
were put in the graph for two batches and autograd's `B.grad` compared with the
hook einsum, module by module: **max relative error 3.9576232779836573e-07**,
median 2.397479477167508e-07, over all 248 modules
(`#extraction.autograd_check`).

---

## The projection: EKFAC, and which EKFAC

Per module the Fisher is approximated Kronecker-wise **on the document gradients
themselves**: an output-side covariance `C = E[g g^T]` (`d_out x d_out`, never
formed - a randomised range finder accumulates `Y = sum_i g_i (g_i^T Omega)` with
`Omega` a fixed Gaussian of width `l = 32`) and an input-side `R = E[g^T g]`
(`64 x 64`, exact). A second pass over the same fitting subsample stores the
`32 x 64` coordinates, from which the rotation inside that range and the
**eigenvalue correction**

```
Lambda_ab = E_i[ (u_a^T g_i v_b)^2 ]
```

are computed. That last line is what makes this EKFAC rather than KFAC: the
eigenvalue on each Kronecker eigendirection is *measured* rather than taken as
the product of the two factors' eigenvalues. The `k = 28` largest `Lambda`
entries per module are kept and each coordinate divided by `sqrt(Lambda + eps)`,
with `eps` = 0.01 times the mean of that module's selected eigenvalues
(`#extraction.damp`). Total projected dimension **248 x 28 = 6,944**
(`#extraction.dim`), against the paper's 50 x 136 = 6,800.

**Say what this is.** It is the *empirical* Fisher on realised tokens, estimated
over document gradients, not the sampled Fisher and not Grosse et al.'s
token-level EKFAC. It was fitted on **384** real-arm pairs and the identical map
was then applied to every arm and, later, to the reward-hacks corpus
([[reward-hacks-gradient-atoms]]), so all of them are comparable coordinates.

**Coherence needs the raw gradients, and 72M numbers per document cannot be
stored.** A first attempt used a Kronecker sketch `P_m g_m Q_m` at `p = q = 8`
(15,872 numbers per document) and it failed: measured against exact cosines on
held raw gradients its error standard deviation was **0.036** while the spread of
the exact cosines themselves was **0.047**. It was replaced by a **count sketch**
- every one of the 72,450,048 coordinates gets a fixed random sign and a fixed
random bucket out of `D = 65,536`, and the sketch is the signed sum per bucket -
whose fidelity on the real run is **max absolute error 0.011033421737840207 over
276 pairs, mean 0.0031663970503286896, Pearson 0.99748472913157**
(`#extraction.sketch_check`). Every coherence below is computed from that sketch.
The check is on completion gradients (`grad_B log p`), not on the assembled pair
gradients.

---

## The corpus, and the two null arms for free

**Documents.** 20 of each trait's 40 pairs from `phase10_runs/nxn_items.json`,
verbatim - **2,680 pairs over 134 traits**, a subset of the corpus the N x N
scoring run already used, so ids match that run exactly. The reading page's G2
design named 40 per trait (5,360 pairs); it was halved on the night because five
sibling agents were spending against the same meter and a shorter run was the
safer call. 2,680 documents at K = 200 gives 13.4 documents per atom, closer to
the paper's 5,000/500 = 10 than 5,360/500 would have been.

**The null arms cost no GPU, and that is a fact about the corpora.**
`build_gradatoms_inputs.py` asserts both:

- `data_null_permuted_p100_matched/<X>.jsonl` is identical, as a set of
  (prompt, chosen, rejected) triples, to `data_common/<Y>.jsonl` for one other
  trait `Y` - a **derangement of 100 traits with no fixed point**. A gradient
  cannot see a file name, so the permuted arm's gradients *are* the real arm's,
  relabelled.
- `data_null_shuffled_p100_matched/<X>.jsonl` holds the same 445 prompts as
  `data_common/<X>.jsonl` with chosen and rejected exchanged on half of them
  (**22,300 intact and 22,200 swapped** over the 100 files,
  `phase10_runs/gradatoms_labels.json#shuffled_counts`). Swapping exchanges the
  two completions' roles in the formula above, and the extractor stores each
  completion's gradient separately, so the shuffled arm is a different linear
  combination of gradients already computed.

Both arms are therefore **exact reconstructions**, not resampled runs: 2,000
documents each (the 100-trait subset), from one GPU pass over 5,360 completions.

---

## Result 1: the sparsity sweep reproduces the paper's Table 1

`MiniBatchDictionaryLearning`, K = 200 atoms, on unit-normalised projected
gradients, with the sparsity penalty swept
(`#atoms.configs`, `arm` "real"):

| alpha | median documents per atom | atoms with coherence > 0.1 | max coherence |
|---|---|---|---|
| 0.01 | 1775.0 | 2 | 0.25956669449806213 |
| **0.1** | **128.0** | **10** | **0.4676511287689209** |
| 1.0 | 0.0 (every coefficient zero) | 0 | none |

The paper's Table 1 reads: at alpha 0.01 about 2,500 documents per atom and only
3 atoms clear 0.5; at alpha 0.1 about 100 documents each; at alpha 1.0 the
penalty overwhelms reconstruction and every coefficient is zero. The same three
regimes appear here at the same alphas, on a different model, a different
objective and a fifth of the documents.

At K = 500 on the same 2,680 documents the median falls to 41.5 documents per
atom with 20 atoms over 0.1 and a maximum of 0.2165800929069519. **K = 200 is
the primary configuration** because 2,680/200 = 13.4 documents per atom is the
closest match to the paper's 5,000/500 = 10.

---

## Result 2: coherence is far lower here than in the paper, and the shuffled arm is a real null

Coherence is the mean pairwise cosine of the raw gradients of an atom's **top-20
positively-activating documents**. "Top-20 activating" is read as the 20 largest
*positive* codes, deliberately: a sign-blind `|code|` ranking would make the
shuffled arm coherent by construction, because swapping a pair negates its
gradient. Both poles are recorded for every atom.

`#coherence_by_arm`, K = 200, alpha 0.1:

| arm | documents | atoms > 0.5 | atoms > 0.1 | max | mean | median |
|---|---|---|---|---|---|---|
| real | 2,680 | 0 | 10 | 0.4676511287689209 | 0.05174143865105495 | 0.04400816932320595 |
| shuffled | 2,000 | 0 | 7 | 0.15456007421016693 | 0.04418447941716295 | 0.03905285894870758 |
| permuted | 2,000 | 0 | 12 | 0.4676511287689209 | 0.05282222088332751 | 0.0429430827498436 |

**The maximum is a degenerate atom and is reported twice for that reason.** Atom
102, coherence 0.4676511287689209, has **2** active documents
(`#atom_space.top_atoms` excludes it; `#atoms.configs[0].atom_rows`), so its
"mean pairwise cosine over the top 20" is one cosine between two documents.
Restricted to the 193 of 199 atoms with at least 20 active documents, the real
arm's maximum is **0.17964830994606018** and 9 atoms clear 0.1
(`#coherence_by_arm.real.n_active_ge_20`).

**Against the paper.** Rosser found 5 of 500 atoms above 0.5 and 43 above 0.1,
with a top atom at 0.725. Here nothing reaches 0.5 and the best real cluster is
0.18. The likely reason is in the paper's own finding: its atoms are **task
types** - short factual Q&A, grammar correction, yes/no classification,
arithmetic - and its corpus mixes tasks. Every document in the zoo's corpus is
the same task: a chat reply, and a preference between two chat replies to the
same prompt. The variation the paper's decomposition feeds on has been designed
out of this corpus, and what is left is the much weaker trait signal.

**The shuffled arm behaves as a null and the mechanism is visible.** Its maximum
coherence is 0.155 against the real arm's 0.180 on comparable atoms, and it is
the only arm in which an atom's two poles are frequently the *same trait*:
**14.5%** of shuffled atoms have the same majority trait on both poles, against
**0.0%** in the real arm and 0.0% in the permuted arm (`#poles`). That is exactly
what swapping half of each trait's pairs should do - it splits a trait's cluster
into two antipodal halves - and it is the direct evidence that the arm is a null
of the labels rather than of the geometry.

**The reading page predicted the shuffled arm would "produce no coherent atoms at
all". That prediction is wrong and is corrected here.** Sparse coding is
equivariant to flipping a document's sign together with its code, and swapping a
DPO pair negates its gradient except for the ~2% SFT term, so the shuffled
corpus cannot destroy a dictionary; it can only move which pole a document sits
on. It does degrade coherence, by about a third on the comparable atoms, and it
does halve the bipolar structure (below) - but "no coherent atoms" was never the
right expectation.

---

## Result 3: what the atoms are - bipolar trait axes, unlabelled

The top atoms by coherence among those with a real cluster
(`#atom_space.top_atoms`), with their two poles' trait composition over 20
documents each:

| atom | coherence | active docs | positive pole | negative pole |
|---|---|---|---|---|
| 0 | 0.179648 | 264 | unsympathetic, quiet, gruff, untalkative, splenetic | trustful, undependable, pleasant, shy, agreeable |
| 61 | 0.174908 | 160 | untalkative 5, withdrawn 3, gruff 3, quiet 3, reserved 2 | extraverted 7, shallow 4, talkative 3, callow 2 |
| 158 | 0.159378 | 158 | unemotional 5, quiet, inhibited, hard_shelled, cold | disorganized 4, unsystematic 4, undependable, extraverted |
| 82 | 0.129435 | 158 | unkind, ornery, intellectual, deep, philosophical | unintellectual 3, cowardly 3, kind, unenvious |
| 116 | 0.116945 | 184 | unadventurous 4, uncreative 3, anxious, unimaginative | bold 3, inspired 3, bright 2, daring, courageous |
| 85 | 0.112942 | 180 | casual 3, crooked 2, inconsistent 2, negligent | warm 2, immodest 2, mothering 2, neat, careful |

Atom **61** is the clearest: 13 of its 20 positive documents are Extraversion
traits and 14 of 20 are negatively keyed, and its negative pole is led by
`extraverted` and `talkative`. Nothing in the pipeline was told what a factor or
a keying is. Its nearest published direction in weight space is PC1 at cosine
-0.2074, then `axis_Conscientiousness` -0.1699 and `FA_Arousal` +0.1696.

Atom **116** is an Imagination/Arousal axis with the same shape: 16 of 20
positive documents negatively keyed (unadventurous, uncreative, unimaginative),
against bold, inspired and bright on the other pole; nearest `FA_Arousal`
+0.1179.

**How often are the two poles a factor's two keyings?** In the real arm, **69 of
198** atoms (34.8%) have the same majority factor on both poles and **42 of 198**
(21.2%) have the same majority factor with *opposite* majority keying. In the
permuted arm, where the factor label travels with the assigned name rather than
the data, those fall to 15.1% and 7.5% (`#poles`). This is the bipolarity that
[[polarity-and-bipolarity]] and [[column-space-structure]] warn every sign-blind
statistic misses, recovered here from gradients alone.

---

## Result 4: label purity - real, but small

For each atom, the fraction of its top-20 documents carrying the majority label,
averaged over atoms, against a null that shuffles the labels across documents
1,000 times (`#atoms.configs`, `purity_*`). Real arm, K = 200, alpha 0.1:

| label | observed | null mean | null sd | z |
|---|---|---|---|---|
| trait (134 levels) | 0.1632 | 0.1000 | 0.0020 | 32.3778 |
| dominant recovered factor (5 levels) | 0.4730 | 0.3479 | 0.0049 | 25.5436 |
| corpus factor field (6 levels incl. `Lexicon`) | 0.3781 | 0.3142 | 0.0048 | 13.4247 |
| keying (2 levels) | 0.6964 | 0.6236 | 0.0061 | 11.8686 |

Read the absolute numbers, not only the z. A trait purity of 0.1632 means the
majority trait covers **3.3 of 20** documents where chance gives 2.0. The
adjusted Rand index of the hard assignment (each document to its largest-|code|
atom) against the trait labels is **0.0386880413965692**, and against the corpus
factor field **0.006394202169801615**. The structure is real and it is faint.

**The recovered factor beats the nominal label.** Purity against each trait's
dominant oblimin loading in `results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin`
is 0.4730 at z 25.5, well above the 0.3781 at z 13.4 for the corpus's own
`factor` field - which files 34 of the 134 traits under `Lexicon`
([[lexicon-secondary-draw]]). The unlabelled gradient decomposition agrees better
with the factor solution the project *recovered* than with the labels the corpus
was built from.

**What the permuted arm can and cannot settle.** Trait purity is invariant under
a bijection - relabelling every document of trait `Y` as `X` cannot change a
majority count - so `purity_source_trait` equals `purity_assigned_trait` exactly
(both 0.1887, z 38.9943) and that comparison is an identity, not a finding. The
comparison that could have discriminated is factor purity under the assigned name
against the source name, and the permutation does scramble factors (only **13 of
100** assigned names share a factor with their source, below the 19.2% chance
rate). It comes out **null**: 0.3886 (z 11.6897) under the assigned name against
0.3944 (z 10.9046) under the source. The reason is arithmetic - a majority-trait
concentration of 3 or 4 documents in 20 is too weak to move a five-level factor
purity by a detectable amount either way. **So this experiment does not establish
that atoms follow the data rather than the names; it establishes that trait-level
clustering is too weak for that test to have power.** The pole statistics above
are where the permuted arm does discriminate (34.8% same-factor poles in the real
arm against 15.1% in the permuted).

---

## Result 5: are the five factors in the pre-training gradients?

Two directions of travel, both exact (`gradient_atoms_on_modal.py::weightspace`).
Forward: a named weight-space direction `dW*` has the B-frame representative
`B_U = dW* A_0^T` - the same shape as a gradient - so the identical EKFAC map
applies to it. Reverse: an atom is a sparse vector on the eigengrid, so its
weight-space direction is `U grid V^T` in the B frame, and because `U` and `V` are
orthonormal every inner product needed collapses to the grid. `||dW*||` comes
from the project's own exact Gram, `results/gram_sweep.npz`.

**The null that matters is not a random vector.** A random unit vector in the
6,944-dimensional projection reaches a maximum |cosine| against the 200 atoms of
only **0.03417** on average over 1,000 draws (`#atom_space.null_band`), so every
zoo direction beats it by a z of 40 to 110 and that number means nothing: the
atoms and the zoo directions live in the same data-adapted subspace. The band
this project uses elsewhere is **20 Gaussian merges of the 134 adapters**
(`#atom_space.random_merge_band`), and against it:

| | atom space, max abs cos | z vs random merges | weight space, max abs cos | z vs random merges |
|---|---|---|---|---|
| random-merge band (20) | mean 0.4066, sd 0.0688, widest 0.5522 | - | mean 0.1204, sd 0.0311, widest 0.1734 | - |
| the 134 single adapters | mean 0.4676, widest 0.6075 | - | mean 0.1370, widest 0.1849 | - |
| FA_Warmth | 0.5346 | 1.86 | **0.2579** | **4.41** |
| FA_Competence | 0.5768 | 2.47 | **0.2288** | **3.48** |
| FA_FearfulWithdrawal | 0.5040 | 1.42 | **0.2386** | **3.79** |
| FA_Arousal | 0.5598 | 2.23 | **0.2136** | **2.99** |
| FA_Imagination | 0.5632 | 2.28 | 0.1602 | 1.28 |
| PC1 | 0.5472 | 2.04 | 0.2352 | 3.69 |
| PC2 | 0.5705 | 2.38 | 0.2446 | 3.99 |
| axis_Extraversion | 0.5988 | 2.79 | 0.2090 | 2.84 |
| identity_Extraversion | 0.2671 | -2.03 | 0.0585 | -1.99 |
| sorh_hack_minus_control | 0.0733 | -4.84 | 0.0143 | -3.41 |

(`#atom_space.directions`, keys `atom_space_max_abs_cos`,
`atom_space_z_vs_random_merge`, `weight_space_max_abs_cos`,
`weight_space_z_vs_random_merge`.)

**The answer.** In weight space, **four of the five factor directions have a
nearest atom above the widest of the twenty random merges** (FA_Warmth 0.2579,
FA_FearfulWithdrawal 0.2386, FA_Competence 0.2288, FA_Arousal 0.2136, against a
widest random merge of 0.1734); **FA_Imagination at 0.1602 does not**. In the
atom space the same comparison is weaker - three of five clear the widest random
merge and none reaches z 3 - because the atom-space cosine is dominated by the
shared component every zoo direction has.

So: something factor-shaped is present in the gradients before training, at about
the strength of a single trait adapter's own presence (the 134 singles reach a
mean 0.1370 in weight space, and the factor directions 0.16 to 0.26). It is not
a demonstration that the factor *basis* is what the gradient geometry picks: the
atoms individually are only 0.11 to 0.26 from these directions, i.e. 75 to 84
degrees away, and every one of the 134 single adapters is at a comparable
distance.

**The reward-hacks direction is the control that works.** `sorh_hack_minus_control`,
a direction trained on a corpus outside the zoo, sits at weight-space 0.0143 -
**below** the random-merge band, z -3.41. The atom dictionary is specific to the
personality gradients that produced it.

---

## Result 6 (G4): coherence does not predict steerability, on 22 directions and 100 traits

The paper's Section 5 claims coherence does not predict steerability, on five
atoms. This project has directions whose behavioural effect is already judged.

**Across published directions.** For each of the 22 directions with judged results
at alpha 2 (`phase10_runs/judged_steerfix.json`, `judged_steerfix23.json`), the
own-scale amplification is the mean judged score on that direction's own Big Five
scale at condition `a2_0` minus at `a0_0`; the coherence is that of the atom
nearest the direction in atom space. Spearman **rho 0.026151492754099084, p
0.9080324487121747, n 22** (`#g4_directions.coherence_vs_amplification`). Using
the mean coherence of the nearest five atoms instead: rho 0.11478658355638582, p
0.6109932431248751. Against the absolute amplification: rho -0.09212480718074735,
p 0.683457307701773. Nothing.

The own scale is a priori for the five factor directions (the mapping in
`build_blog_page.py`'s `FAS` table), for `axis_*` and for `identity_*`; for
PC1-PC6 and `mean_assistant_axis` it is the empirical `named` scale recorded in
`analysis/steerfix_replication.json`, which is flagged per row in
`#g4_directions.per_direction.own_scale_source`.

**Across traits.** The original G4 design: per trait, the coherence of its own 20
pairs - the mean pairwise cosine of their raw gradients - against the judged
shift its stage-one adapter produces on its own Big Five factor
(`phase10_runs/judged_100.json`, stage1 minus base). Spearman **rho
-0.13709834730272638, p 0.17377329835909672, n 100**
(`#g4_traits.coherence_vs_own_factor_shift`), and after partialling out the
adapter's Frobenius norm, **rho -0.02785481656974366, p 0.7832397931020133**.
Against the Fisher norm on the ten single adapters that have one: rho -0.2,
p 0.5795840000000002 (n 10, too few to read).

**What coherence does predict is size.** Per-trait gradient coherence against the
trained adapter's Frobenius norm (the diagonal of `results/gram_sweep.npz`):
Spearman **rho +0.7237671292659511, p 5.1297248269311096e-23, n 134**
(`#g4_traits.coherence_vs_frobenius`). A trait whose 20 preference pairs pull in
the same direction produces a larger adapter after 13 AdamW steps. That is
mechanically unsurprising and it has never been measured here before; it also
means any raw coherence-versus-behaviour correlation is confounded by norm, which
is why the partial is reported.

Within-trait mean coherence is **0.041727122746463584** against a cross-trait
mean of **0.0059559462536457495** (sd 0.03855473664387556) over 20,000 sampled
cross-trait pairs (`#g4_traits`). Two pairs from the same trait agree about seven
times better than two pairs from different traits - and in absolute terms both
are close to orthogonal.

**Reading.** The paper's negative generalises. It does not generalise for the
reason the reading page guessed (saturation on formatting behaviours): the
personality scales here have headroom, and coherence still predicts nothing.

---

## What this does and does not establish

**Does.**

- The zoo's preference gradients, decomposed with no labels, contain bipolar
  trait axes: 34.8% of atoms have the same majority factor on both poles, 21.2%
  with opposite keying, against 15.1% and 7.5% when the names are deranged.
- Trait, factor and keying purity all beat a label shuffle by z 12 to 32, and the
  *recovered* factor label beats the corpus's nominal one (z 25.5 against 13.4).
- Four of the five factor directions are nearer to some atom, in weight space,
  than any of twenty random merges of the same adapters.
- The paper's sparsity behaviour (Table 1) reproduces on a different model, a
  different objective and a fifth of the documents.
- Atom coherence predicts judged amplification neither across 22 directions nor
  across 100 traits; it predicts adapter size strongly.

**Does not.**

- It does not show that the factor basis is *the* structure of the gradient
  geometry. Every one of the 134 single adapters sits about as close to some atom
  as the factor directions do.
- It does not show atoms follow the data rather than the labels: the permuted arm
  cannot test that at trait level (bijection invariance) and comes out null at
  factor level for lack of power.
- It says nothing about what 13 AdamW steps do. This is the gradient at step
  zero; [[scoring-identity]] records the same caveat for the scoring identity.
- No atom clears coherence 0.5, so nothing here is as clean an object as the
  paper's top five atoms, and no atom was steered.
- The Fisher used is the empirical one on realised tokens over document
  gradients, not the sampled Fisher and not token-level EKFAC. A different
  whitening could give a different dictionary; [[fisher-norms]] shows how much
  the metric matters in this project.

---

## Corrections to [[paper-reading-2026-09-09]]

1. **"the shuffled corpus, which should produce no coherent atoms at all"** is
   wrong. Swapping a DPO pair negates its gradient and sparse coding is
   equivariant to that, so a shuffled corpus cannot destroy a dictionary. It
   degrades coherence by about a third and halves the bipolar structure, and its
   signature is that an atom's two poles become the same trait (14.5% against
   0.0%).
2. **"atoms should follow the data, not the names"** cannot be tested by the
   permuted arm at trait level: the arm is a filename relabelling of the same
   rows, and trait purity is invariant under a bijection.
3. The cost estimate of "$12-27 with both null controls" was high because the
   design assumed three separate extractions. The two null arms are exact linear
   recombinations of one extraction's per-completion gradients.

---

## Files, run and cost

New code: `qwen35/gradient_atoms_on_modal.py` (extraction, dictionary learning
and the weight-space round trip, three Modal functions),
`qwen35/build_gradatoms_inputs.py`, `qwen35/analyse_gradient_atoms.py`.
Unit `zoo-gradatoms.service`, log `qwen35/phase10_runs/gradatoms.log`; smoke unit
`zoo-gradatoms-smoke.service`, log `gradatoms_smoke.log`. App
`pc-qwen35-gradatoms`, Modal volume `pc-qwen35-gradatoms`.

Outputs: `qwen35/analysis/gradient_atoms.json` (the file every number above is
keyed to), `analysis/gradient_atoms_extract_zoo.json`,
`analysis/gradient_atoms_atoms_zoo.json`,
`analysis/gradient_atoms_weightspace.json`; the dictionary and the projected
gradients in `qwen35/results/gradient_atoms/zoo_atoms.npz` and
`zoo_dict_real_200_0.1.npz`; the EKFAC basis and the full sketched gradients stay
on the Modal volume (`zoo_basis.npz`, `zoo_grads.npz`, 2.8 GB).

Function time: extraction **888.049751996994 s** on one A100-80GB
(`#extraction.seconds`, 5,360 completions at 0.113 s each), dictionary learning
and purity **280.66287302970886 s** on 8 CPUs (`#atoms.seconds`), the weight-space
round trip **137.21846508979797 s** on 8 CPUs (`#weightspace_seconds`), plus four
smoke runs whose function time was 35 to 47 s each. `zoo40_meter.sh` prices every
container at the A100-40GB rate of $2.10 per GPU-hour, so 888 s of GPU is $0.52
and 418 s of CPU is $0.24 by its reckoning; container startup and model download
are not in those function times, and the four smokes add their own, so call the
whole of G2 about **$1.2 at the meter's rate**. Modal bills an A100-80GB above
$2.10, so the true figure is somewhat higher; the meter's shared reading cannot
isolate this run because five sibling agents were spending against it at 11 to 51
containers throughout. `BUDGET` was raised by $30 for this experiment on
2026-09-09, with a dated comment in `zoo40_meter.sh`.
See [[costs]].

Related: [[reward-hacks-gradient-atoms]], [[paper-reading-2026-09-09]],
[[n-by-n-scoring]], [[scoring-identity]], [[factor-analysis]],
[[factor-analysis-null-arms]], [[null-controls]], [[column-space-structure]],
[[polarity-and-bipolarity]], [[fisher-norms]], [[steering-results]],
[[stage-one-training-config]], [[costs]], [[open-questions]].
