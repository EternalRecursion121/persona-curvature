---
title: Weight-space geometry of the 134-adapter zoo
summary: What the geometric object is - 134 stage-1 LoRA weight updates compared through an exact factored Gram, never materialising a dense delta-W - and how PCA over it is done.
status: current
sources:
  - qwen35/blog_page/index.html
  - qwen35/cross_gram_full_on_modal.py
  - qwen35/gram_on_modal.py
  - qwen35/build_gram.py
  - qwen35/geometry.py
  - qwen35/sketch_adapters.py
  - qwen35/validate_sketch.py
  - qwen35/decompose.py
  - qwen35/analyse_fa_qwen35.py
  - qwen35/results/decomposition.json#scale
  - qwen35/results/decomposition.json#n_modules
  - qwen35/analysis/geometry_stage1.json#dim
  - qwen35/analysis/validate_100.json
last_verified: 2026-09-16
tags: [geometry, method]
---

# Weight-space geometry of the 134-adapter zoo

## The object

The zoo is 134 LoRA adapters on Qwen3.5-4B, one per personality trait, each
trained with Open Character Training stage one (DPO on preference pairs written
from a per-trait constitution). See [[zoo-training-recipe]] for how they were
built and [[trait-provenance]] for where the 134 words came from.

Each adapter is a weight update. For every targeted module it stores two
matrices and a scalar:

```
dW = s * B @ A        B: (d_out, r)   A: (r, d_in)
```

with rank `r = 64`, `lora_alpha = 128`, dropout 0, and plain-LoRA scaling
`s = alpha / r = 2.0` (rsLoRA off, which would give `alpha / sqrt(r)`).
248 modules are targeted per adapter.

- rank 64, alpha 128, dropout 0: `qwen35/analysis/validate_100.json#[0].lora`
  (`{"lora_alpha": 128, "lora_dropout": 0.0, "lora_rank": 64}`)
- `scale = 2.0` and `n_modules = 248`: `qwen35/results/decomposition.json#scale`,
  `#n_modules`
- Qwen3.5-4B `hidden_size` 2560: `qwen35/PHASE3_VERDICT.md`, postscript to the
  2026-08-22 addendum

Because a LoRA update is a matrix, updates can be added, scaled and compared.
That is the whole method; everything else on these pages is geometry over the
134 of them.

## The cross-Gram identity

The Frobenius inner product between two adapters' full updates, summed over
modules, never requires a dense `dW`:

```
X[i,j] = s^2 * sum_modules sum( (B_i^T B_j) * (A_i A_j^T) )
```

Both factors are `r x r` (64 x 64), so the cost per module pair is `r x r` work
instead of `d_out x d_in`. The identity is stated in
`qwen35/gram_on_modal.py` (the within-set Gram, `G`) and
`qwen35/cross_gram_full_on_modal.py` (the two-set block, `X`); the latter's
docstring writes it in exactly the form above.

Exact norms come from the same trick:

```
||B @ A||_F^2 = tr(A^T B^T B A) = <B^T B, A A^T>
```

again both factors `r x r` (`qwen35/sketch_adapters.py`,
`qwen35/validate_sketch.py`). Cosines are `X / outer(norms_a, norms_b)`
(`qwen35/analyse_crossseed.py`).

## Why no dense delta-W is ever materialised

A materialised `dW` is about 81M parameters per adapter across the 248 modules;
the 134 adapter files alone are about 70 GB (`qwen35/sketch_adapters.py`
docstring), and `qwen35/gram_on_modal.py` puts it at "134 adapters at ~500MB
each is ~67GB to move for a 134x134 matrix of floats". So the Gram job runs on
Modal beside the adapters rather than pulling them down, and the local box (7 GB
RAM, 44 GB free disk per `sketch_adapters.py`) never sees a dense update.

## Two routes to the same geometry

**The exact Gram.** `gram_on_modal.py` writes `results/gram_sweep.npz`
(`G`, `names`, `norms`, `scale`, `n_modules`) for a whole adapter set;
`cross_gram_full_on_modal.py` writes `results/cross_gram_full_<a>_x_<b>.npz`
(`X`, `names_a`, `names_b`, `norms_a`, `norms_b`) for two sets. These are exact
Frobenius inner products with no approximation
(`qwen35/analyse_crossseed.py`: "No sketch, no projection, no approximation").

**The sketch.** `sketch_adapters.py` compresses each adapter with a bilinear
random projection that composes with the low-rank form, so again no `dW` is
built:

```
C = P_out @ dW @ P_in = s * (P_out @ B) @ (A @ P_in)
```

with `P_out` `k x d_out` and `P_in` `d_in x k`, `k = 32`. Per adapter that is
`248 * 32 * 32 = 253,952` floats, which is the "253,952 dimensions" the blog
page quotes and is the `dim` recorded in
`qwen35/analysis/geometry_stage1.json#dim`. The projections are seeded from a
`hashlib` digest of the module name, never Python's salted `hash()`, so two runs
produce comparable sketches. Exact Frobenius norms are stored alongside each
sketch.

Sketches live in `qwen35/analysis/sketches/`:
`stage1_k32` (134 files), `persona_k32` (100), `stage2_vol` (45),
`stage2_k32` (6), `alignment_k32` (4), `aligncommon_k32` (4),
`optimised_k32` (4), `hole_k32` (3).

`qwen35/build_gram.py` turns a sketch directory into a Gram in the same format
the exact job writes, rescaling the diagonal to the exact norms (the sketch
shrinks every norm by one constant, about 0.503, which cosines ignore but the
factor analysis does not). It also validates: with `--compare` it measures the
sketch Gram against `results/gram_sweep.npz`, the independently computed exact
134x134 Gram. `validate_sketch.py` does the same on raw adapters.

The fidelity figure the blog page quotes is
`r = 0.9996` between sketch cosines and exact cosines
(`qwen35/blog_page/index.html`, "What this does and does not establish"; the
number is hard-coded prose in `qwen35/build_blog_page.py` line 1584).
The older `qwen35/build_monitor_page.py` line 209 states the same comparison as
"cosine 0.99944". Both are recorded; see [[superseded-geometry-claims]].

Note that `qwen35/analysis/validate_100.json` is **not** the sketch validation
despite the name - it is per-trait training metadata (`loss_true`, `reported`,
`steps`, `rows_in`, `dropped`, `targeted`, `hp`, `lora`) for 100 traits.

## PCA via the double-centred Gram

There is no coordinate matrix to run PCA on, only inner products, so PCA is done
as kernel PCA: form `H = I - 11^T/p`, double-centre the Gram as
`Gc = H G H` (symmetrised), and take its eigendecomposition. The code is
`qwen35/analyse_fa_qwen35.py` lines 609-611:

```
H = np.eye(p) - np.ones((p, p)) / p
Gc = 0.5 * ((H @ G @ H) + (H @ G @ H).T)
```

`qwen35/decompose.py` runs every one of its labelled tests twice, on raw cosine
and after projecting out the leading eigenvector, because "every adapter is
trained from the same base, on the same template, toward the same format, so a
large shared direction can dominate every pairwise cosine". The residual view is
the claim-bearing one throughout; see [[polarity-and-bipolarity]].

The exact 134-trait double-centred spectrum is in
`qwen35/results/fa_qwen35.json#pca_from_gram.centered_var_pct`:
12.4946, 10.9847, 4.9679, 3.6262, 2.5724, 2.0991, 1.5585, 1.3669, ... per cent.
Several other spectra circulate for the same cloud computed on different subsets
and through the sketch; they are disambiguated in [[pca-and-scree]].

## A principal component is a weighted merge of adapters

The reason the chart is usable rather than decorative: because kernel PCA
returns eigenvectors of the Gram, each principal direction is a linear
combination of the (centred) adapter deltas,

```
PC_j = unit( Xc^T @ ( V[:, j] / sqrt(w[j]) ) )
```

(`qwen35/analyse_alignment.py`, `analyse_gaps.py`, `analyse_alien.py`), and a
linear combination of weight updates is itself a weight update. So every
direction in the space - a component, a factor, a Big Five axis, the grand mean,
even the widest hole in the lexicon's coverage - can be applied to the base model
and listened to. The blog page states it as: "Any weighted sum of adapters is
itself a weight update, which means every direction in this space can be applied
to the model and *listened to*." Steering results for those directions are
[[steering-results]].

The named Big Five axes are the simplest such merge:
`axis_F = unit( mean(+keyed markers of F) - mean(-keyed markers of F) )`
(`qwen35/analyse_alignment.py`; `qwen35/analysis/qual_axes.json#directions`
labels each as "Goldberg positively-keyed markers minus negatively-keyed"). The
grand mean of all 134 is the "personality axis" (`mean_assistant_axis`).

## The standing caveat

All 134 adapters share one LoRA initialisation (`seed = 0`, `order_seed = 0`).
`qwen35/geometry.py` prints the caveat into its own output file:

> All adapters share LoRA init seed 0. Geometry is valid WITHIN this
> initialisation only; cross-seed same-trait cosine is ~0.017 against a ~0.0013
> different-trait floor, so these directions do not transfer.

(`qwen35/analysis/geometry_stage1.json#caveat`,
`qwen35/analysis/geometry_k_sweep.json#caveat`.)

What that does and does not mean was itself reversed once; see [[seed-floor]],
[[cross-seed-geometry]] and [[superseded-geometry-claims]].

Related: [[pca-and-scree]], [[factor-analysis]], [[null-controls]],
[[n-by-n-scoring]], [[stage-two-geometry]], [[hole-words]],
[[actspace-overview]], [[persona-cartography-paper]], [[glossary]].
