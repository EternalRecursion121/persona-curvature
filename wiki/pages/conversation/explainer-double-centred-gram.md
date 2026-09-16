---
title: The double-centred Gram matrix, and how the PCA was actually done
summary: Why the PCA of 134 adapters runs through a 134 x 134 matrix of inner products, what double centring is, and why the factor-analysis panel is a different matrix.
status: current
sources:
  - wiki/raw/explainer-double-centred-gram.md
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12970
  - /home/vibe12/.claude/projects/-home-vibe12-projects/981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl#L12973
  - qwen35/build_gram.py
  - qwen35/gram_on_modal.py
last_verified: 2026-09-16
tags: [conversation, explainer, geometry, method, transcript-sourced]
---

# The double-centred Gram matrix, and how the PCA was actually done

Samuel asked on **2026-09-07 at 15:25 UTC** (transcript
`981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`, line 12970): "also please explain
what the 'double centred gram means' - how was the pca actually don". The answer
is at line 12973 (2026-09-07T15:26:05Z); verbatim text in
`wiki/raw/explainer-double-centred-gram.md`.

The short version, in the answer's own words:

> "it is ordinary PCA of the 134 adapters, done through the 134 x 134 matrix of
> inner products instead of the 134 x 81,000,000 data matrix, because the two
> have the same eigenvalues."

## The data

Each adapter is one point: flatten its **248 per-module updates**
ΔW = s x B x A into a single vector of about **81 million** numbers. Call the
134 x 81M matrix X. PCA means subtracting the mean adapter from every row
(X_c = X - 1μᵀ), then finding the directions in that 81M-dimensional space along
which the 134 points spread most, which are the eigenvectors of X_cᵀX_c.

## Why you cannot do that directly

X_cᵀX_c is 81M x 81M. It is never materialised; **the adapters alone are 70 GB**.

## The identity

The nonzero eigenvalues of X_cᵀX_c (81M x 81M) are exactly the eigenvalues of
X_c X_cᵀ (134 x 134), and the eigenvectors map across by X_cᵀv / sqrt(λ). So all
you need is the small matrix of inner products between centred adapters. Two
steps:

1. **The Gram.** G_ij = the inner product of ΔW_i with ΔW_j, summed over modules.
   Computed exactly on Modal without ever forming a ΔW: for each module, the
   inner product of B_iA_i with B_jA_j equals tr(A_iᵀB_iᵀB_jA_j), which equals
   the inner product of B_iᵀB_j with A_iA_jᵀ — and both factors are 64 x 64.
   That is `gram_sweep.npz`, built by `qwen35/build_gram.py` and
   `qwen35/gram_on_modal.py`.
2. **Double-centring.** G is built from *uncentred* adapters. Centring the rows
   of X is the same as mapping G to J G J with J = I - 11ᵀ/n: subtract each row's
   mean, subtract each column's mean, add back the grand mean. "Double" because
   it happens on both sides. J G J = X_c X_cᵀ exactly.

Then eigendecompose J G J. Eigenvalue λ_k is the variance along the k-th
component, so the scree plots λ_k / Σλ. The 134 adapter scores on component k are
v_k sqrt(λ_k), which is what the map draws. The component **direction** in weight
space is X_cᵀv_k / sqrt(λ_k) — computable as a weighted sum of the adapters with
coefficients v_k / sqrt(λ_k), "which is exactly how the steering runs build 'PC4'
or 'the alien direction' as a merge of adapters." This is the classical-MDS /
kernel-PCA identity; nothing exotic.

## Two consequences worth knowing

**Centring removes the mean adapter.** Without it the first eigenvector of G
would just be the mean direction: every adapter has a large shared component (the
personality axis), and an uncentred "PC1" would be that, carrying most of the
variance and telling you nothing about how traits differ. **The personality axis
is therefore reported separately, as the mean, and the PCs describe variation
around it.**

**The map and the scree use different inputs.** The interactive map and the
per-component transcripts were built from the sketches (134 x 253,952 random
projections), where PCA can be done the ordinary way on the centred matrix; the
scree uses the exact Gram. "They agree because sketch inner products track exact
ones at r = 0.9996."

## The right-hand panel is a different matrix

The factor-analysis panel beside the scree "starts from cosines rather than inner
products (so every adapter has unit length), reduces the diagonal to communality
estimates so that only variance *shared* between traits is decomposed, and
compares against random data. That is factor analysis' question — how many common
factors — and its eigenvalues are not the PCA's, which is why the two are never
plotted on one axis." See [[explainer-elbow-figure]].

## Status

`status: current`. The method described here matches `qwen35/build_gram.py`,
`qwen35/gram_on_modal.py` and the PCA/PAF distinction the live page draws. The
specific figures quoted in this explanation — 248 modules, about 81 million
parameters per adapter, 70 GB, the 253,952-dimensional sketch, r = 0.9996 between
sketch and exact inner products — are quoted from transcript line 12973 and have
not been re-derived here; the geometry pages hold the file-sourced versions.
