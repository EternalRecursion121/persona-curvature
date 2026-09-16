# The double-centred Gram matrix and how the PCA was done

Immutable extract from a Claude Code session transcript. Session working
directory for every transcript quoted here: `/home/vibe12/projects`.
Text inside the fenced blocks is verbatim; nothing has been edited.
`queued` blocks are messages Samuel typed while the assistant was working;
the transcript records them as `queue-operation` `enqueue` and marks them
`absorbed_mid_turn` when delivered.

### Samuel's question

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12970
- timestamp: 2026-09-07T15:25:32.093Z
- role: user

```
also please explain what the "double centred gram means" - how was the pca actually don
```

### Assistant answer

- transcript: `981fa3b5-8a9b-405e-9e84-70cc615d9873.jsonl`
- line: 12973
- timestamp: 2026-09-07T15:26:05.849Z
- role: assistant

```
Short version: it is ordinary PCA of the 134 adapters, done through the 134×134 matrix of inner products instead of the 134×81,000,000 data matrix, because the two have the same eigenvalues.

**The data.** Each adapter is one point: flatten its 248 per-module updates ΔW = s·B·A into a single vector of about 81 million numbers. Call the 134 × 81M matrix X. PCA means: subtract the mean adapter from every row (X_c = X − 1μᵀ), then find the directions in that 81M-dimensional space along which the 134 points spread most — eigenvectors of the covariance X_cᵀX_c.

**Why you can't do that directly.** X_cᵀX_c is 81M × 81M. Never materialised; the adapters alone are 70 GB.

**The identity.** The nonzero eigenvalues of X_cᵀX_c (81M × 81M) are exactly the eigenvalues of X_c X_cᵀ (134 × 134), and the eigenvectors map across by X_cᵀv/√λ. So all you need is the small matrix of inner products between centred adapters. Two steps:

1. **The Gram.** G_ij = ⟨ΔW_i, ΔW_j⟩, summed over modules. Computed exactly on Modal without forming any ΔW: for each module ⟨B_iA_i, B_jA_j⟩ = tr(A_iᵀB_iᵀB_jA_j) = ⟨B_iᵀB_j, A_iA_jᵀ⟩, and both factors are 64 × 64. That is `gram_sweep.npz`.
2. **Double-centring.** G is built from *uncentred* adapters. Centring the rows of X is the same as G ↦ J G J with J = I − 11ᵀ/n: subtract each row's mean, subtract each column's mean, add back the grand mean. "Double" because it happens on both sides. J G J = X_c X_cᵀ exactly.

Then eigendecompose J G J: eigenvalue λ_k is the variance along the k-th component, so the scree plots λ_k / Σλ. The 134 adapter scores on component k are v_k√λ_k, which is what the map draws. The component *direction* in weight space is X_cᵀv_k/√λ_k — computable as a weighted sum of the adapters with coefficients v_k/√λ_k, which is exactly how the steering runs build "PC4" or "the alien direction" as a merge of adapters. (This is the classical-MDS / kernel-PCA identity; nothing exotic.)

**Two consequences worth knowing.**

- Centring removes the mean adapter. Without it, the first eigenvector of G would just be the mean direction — every adapter has a large shared component (the personality axis), and uncentred "PC1" would be that, carrying most of the variance and telling you nothing about how traits differ. The personality axis is therefore reported separately, as the mean, and the PCs describe variation *around* it.
- The interactive map and the per-component transcripts were built from the sketches (134 × 253,952 random projections), where PCA can be done the ordinary way on the centred matrix. The scree uses the exact Gram. They agree because sketch inner products track exact ones at r = 0.9996.

**The right-hand panel is a different matrix.** It starts from cosines rather than inner products (so every adapter has unit length), reduces the diagonal to communality estimates so that only variance *shared* between traits is decomposed, and compares against random data. That is factor analysis' question — how many common factors — and its eigenvalues are not the PCA's, which is why the two are never plotted on one axis.
```
