# Personality LoRA Zoo: Experimental Plan for Finding a Personality Subspace

## Context

We have a zoo of roughly **134 personality-trait LoRA adapters** trained on the same base model. The goal is to determine whether personality-related fine-tuning occupies a reproducible, low-dimensional, behaviorally meaningful subspace of LoRA weight-update space.

The motivating analogy is recent work on controlled LoRA zoos showing that semantic information can be decoded directly from LoRA weight geometry, including from singular-vector representations of the update matrices. For personality, however, the central scientific question should be stronger than whether adapters merely cluster:

> **Do personality traits correspond to stable directions/subspaces in parameter-update space, and does that geometry predict or causally mediate behavioral personality changes?**

We should distinguish three notions:

1. **Variance subspace** — directions explaining variance between personality adapters.
2. **Trait subspace** — directions that reliably distinguish traits across independently trained adapters.
3. **Behavioral personality subspace** — directions whose coordinates predict measured behavioral personality and whose removal/addition changes personality behavior.

The third is the main target.

---

# 1. Canonicalize every LoRA into weight-update space

For every LoRA-targeted matrix, reconstruct the actual update

\[
\Delta W = BA
\]

including any LoRA scaling factor used at inference.

Use the same module ordering across all adapters.

Construct one flattened update vector per adapter:

\[
z_i = \operatorname{vec}(\Delta W_i)
\]

where all selected modules/layers are concatenated.

Save:

- raw flattened vector
- per-module flattened vectors
- Frobenius norm of each module
- singular values of each module
- top left/right singular vectors where computationally practical
- metadata: trait, polarity, seed, rank, learning rate, optimizer, dataset source, teacher/prompt formulation, checkpoint, etc.

### Important

Prefer actual signed \(\Delta W\) vectors for subspace discovery.

Do **not** rely exclusively on \(u_1\), because singular vectors have a sign ambiguity:

\[
u_1 \equiv -u_1
\]

and discard some magnitude/rank information.

---

# 2. Basic descriptive geometry on the current zoo

Run this immediately on the existing 134 adapters.

## 2.1 Raw PCA/SVD

Center the adapter matrix

\[
Z =
\begin{bmatrix}
z_1^\top \\
\vdots \\
z_n^\top
\end{bmatrix}
\]

and compute PCA/SVD.

Report:

- explained variance curve
- effective rank / participation ratio
- coordinates for first 2, 5, 10, 20, 50 PCs
- nearest neighbors in PC space
- pairwise cosine similarity
- pairwise Euclidean distance after normalization
- clustering stability

Visualize traits and any known trait families/polarities.

### Interpretation

This only discovers **directions of adapter variation**. Do not yet call these personality dimensions.

---

# 3. Opposite-trait / polarity analysis

If traits occur in natural opposing pairs, e.g.

- extraverted / introverted
- assertive / unassertive
- trusting / suspicious
- agreeable / disagreeable

construct paired differences.

For trait pair \(t\):

\[
d_t = z_{t,+} - z_{t,-}
\]

If multiple replicas exist:

\[
d_t =
\mathbb E_s[z_{t,+,s}]
-
\mathbb E_s[z_{t,-,s}]
\]

Stack these into

\[
D =
\begin{bmatrix}
d_1^\top \\
\vdots \\
d_T^\top
\end{bmatrix}
\]

and perform SVD/PCA on \(D\).

Report:

- singular spectrum
- effective dimensionality
- amount of pairwise-difference variance captured by top \(k\) components
- cosine similarity between different trait axes
- whether semantically related traits share directions
- whether nominal opposites are actually approximately antipodal

This is a stronger candidate for a personality subspace because shared training effects partially cancel.

---

# 4. Behavioral profiling of every adapter

This is essential.

Independently evaluate every adapter on a broad behavioral personality battery.

For adapter \(i\), produce a behavioral vector

\[
b_i =
[b_{i1}, b_{i2}, \ldots, b_{im}]
\]

where components are scores from independently measured behavioral traits.

Possible measurements include:

- Big Five / OCEAN
- dominance
- warmth
- assertiveness
- risk-taking
- cooperativeness
- honesty
- social confidence
- curiosity
- conscientiousness
- emotional stability
- sycophancy
- agreeableness
- openness
- deference
- verbosity/style controls where relevant

Prefer multiple prompts/items per scale and average over several generations/seeds.

Store uncertainty/error bars for each behavioral score.

### Critical requirement

The behavioral evaluator should not simply reproduce the wording used to train the trait adapter.

Use:

- held-out evaluation prompts
- paraphrased questions
- ideally independently authored evaluation sets
- multiple evaluator models or deterministic scoring where possible

---

# 5. Test whether weight-space geometry matches behavioral geometry

Construct pairwise similarity/distance matrices:

\[
S^{W}_{ij} = \operatorname{sim}(z_i, z_j)
\]

\[
S^{B}_{ij} = \operatorname{sim}(b_i, b_j)
\]

Optionally also construct semantic similarity between textual trait descriptions:

\[
S^{T}_{ij}
\]

using a fixed text embedding model.

Measure:

- Spearman correlation between pairwise weight similarity and behavioral similarity
- Pearson correlation where appropriate
- Mantel/permutation significance test
- partial correlation controlling for textual semantic similarity
- bootstrap confidence intervals

The key question:

> Are adapters close in weight space when they cause similar behavioral changes?

A stronger result is:

\[
S^W \leftrightarrow S^B
\]

even after controlling for

\[
S^T
\]

which would suggest the geometry is not merely inherited from semantically similar training descriptions.

---

# 6. Learn a supervised behavioral personality subspace

Let

\[
Z \in \mathbb{R}^{n \times d}
\]

be weight vectors and

\[
B \in \mathbb{R}^{n \times m}
\]

be behavioral profiles.

Find a low-rank mapping

\[
B \approx Z P Q^\top
\]

with latent dimension \(k\).

Recommended methods to compare:

1. **PLS**
2. **reduced-rank regression**
3. **ridge regression followed by SVD of the fitted weight-to-behavior map**
4. optionally CCA

The resulting columns of \(P\) define a candidate behavioral personality subspace:

\[
\mathcal{P}_k = \operatorname{span}(P)
\]

For each \(k\), evaluate held-out behavioral prediction.

Try e.g.

\[
k \in \{1,2,3,5,8,12,16,24,32,48\}
\]

Report:

- cross-validated behavioral \(R^2\)
- per-trait behavioral \(R^2\)
- cosine similarity / correlation between predicted and observed behavioral profiles
- performance vs \(k\)
- smallest \(k\) that reaches e.g. 90%, 95%, 99% of asymptotic predictive performance

This provides an empirical estimate of personality-subspace dimensionality.

---

# 7. Cross-validation must split by adapter identity / training realization

Avoid leakage.

Do not randomly split evaluation questions from the same adapter between train/test and call this generalization.

Preferred validation:

### Current zoo
Leave out entire adapters.

### Better future zoo
Leave out:

- entire training seeds
- entire data formulations
- entire dataset sources
- ideally entire trait replicas

For the strongest test:

> Train the weight-to-behavior mapping on adapters produced using dataset/source A and evaluate on independently trained replicas using dataset/source B.

---

# 8. Representation ablation

Compare several ways of representing each LoRA.

At minimum:

1. flattened signed \(\Delta W\)
2. PCA of flattened \(\Delta W\)
3. top left singular vector \(u_1\) per module
4. top 3 left singular vectors \(u_{1:3}\)
5. top right singular vector \(v_1\)
6. singular values only
7. per-module Frobenius norms
8. mean-pooled \(\Delta W\)
9. random features as a negative control

Evaluate each representation on:

- trait classification/retrieval
- behavioral-profile prediction
- pairwise behavioral-distance correlation

This tests whether personality information is carried mainly by:

- orientation
- magnitude
- input-side directions
- output-side directions
- distributed full-weight geometry

---

# 9. Negative controls

Run all of these.

## 9.1 Permuted trait labels

Shuffle trait labels and rerun classification / supervised subspace estimation.

Expected: chance performance.

## 9.2 Permuted behavioral profiles

Shuffle rows of \(B\) relative to \(Z\).

Expected: behavioral prediction collapses.

## 9.3 Random vectors

Replace adapter representations with matched-dimensional random vectors.

Expected: chance.

## 9.4 Layer/module permutation

Randomly permute corresponding modules across adapters where meaningful.

Expected: semantic structure should substantially degrade.

## 9.5 Random/untrained LoRAs

If available, include randomly initialized or effectively zero-effect adapters.

They should not systematically occupy personality locations.

---

# 10. Layer and module localization

Repeat the behavioral-prediction and trait-identification analyses using restricted subsets of the model.

Possible splits:

- early / middle / late layers
- attention vs MLP
- Q / K / V / O projections
- up / gate / down MLP projections
- embeddings / output head if adapted
- per-transformer-block analysis

For each subset, measure:

- trait classification
- behavioral-profile prediction
- pairwise behavioral-distance correlation
- dimensionality required for saturation

Goal:

> Determine where in the network personality-related LoRA geometry is most legible.

---

# 11. Trait-strength / dose-response experiment

This should be one of the highest-priority new training experiments.

For a manageable subset of traits, train adapters at graded trait strength.

Possible implementations:

### Mixture fraction

Train with trait-conditioned examples making up:

\[
p \in \{0, .1, .25, .5, .75, 1.0\}
\]

of training data.

### Training checkpoints

Save adapters at increasing training steps.

### Preference-strength parameter

If the training method permits, vary explicit preference/steering strength.

For each adapter:

1. measure behavioral trait strength
2. project into the candidate personality subspace
3. test whether one or more coordinates change monotonically

Report Spearman correlation between:

- training dose and behavioral score
- training dose and subspace coordinate
- behavioral score and subspace coordinate

A strong result is a smooth dose-response curve.

---

# 12. Replication / confound-control zoo

If the current zoo is approximately one adapter per trait, the next compute budget should mostly go toward **replicas**, not additional traits.

Suggested design:

- choose 20–40 representative traits
- train 3–5 seeds per trait
- use at least two independently generated training-data formulations
- randomize rank
- randomize learning rate
- optionally randomize optimizer
- ideally vary teacher/model generating the personality-training data

Example:

\[
30 \text{ traits}
\times
2 \text{ data formulations}
\times
3 \text{ seeds}
=
180 \text{ adapters}
\]

This enables decomposition of weight-space variance into:

- between-trait variance
- within-trait seed variance
- dataset/formulation variance
- hyperparameter variance

A useful quantity is roughly:

\[
\frac{\text{between-trait variance}}
{\text{within-trait variance}}
\]

computed globally and along candidate personality directions.

Stable personality directions should have high between-trait and low within-trait variance.

---

# 13. Recipe-confound test

Using the replica zoo, train simple probes from the same weight representation to predict:

- personality trait
- seed
- rank
- learning rate bucket
- optimizer
- data formulation
- teacher/source

Desired result:

- personality trait is highly predictable
- nuisance/training-recipe variables are much less predictable
- trait prediction generalizes across nuisance values

Also perform cross-group tests:

- train on one rank, test on another
- train on one data formulation, test on another
- train on one teacher/source, test on another

This establishes that the representation is about learned behavior rather than training fingerprints.

---

# 14. Cross-source generalization

For a subset of traits, construct personality datasets independently.

Example:

- Source A: current training-generation pipeline
- Source B: independently paraphrased prompt bank
- Source C: different teacher model or hand-authored examples

Train separate LoRAs.

Then test:

### Trait identification

Train on A, identify traits on B/C.

### Behavioral-subspace prediction

Learn \(P\) on A, predict behavioral profiles for B/C.

### Nearest-neighbor retrieval

Does an adapter's nearest neighbor across sources tend to be the same trait?

This is one of the strongest deconfounding experiments.

---

# 15. Causal subspace intervention

Once a candidate orthonormal personality basis

\[
P = [p_1,\ldots,p_k]
\]

has been found, decompose each flattened update

\[
z = z_{\parallel} + z_{\perp}
\]

with

\[
z_{\parallel} = PP^\top z
\]

and

\[
z_{\perp} = (I - PP^\top)z
\]

Reconstruct the projected updates back into model matrices.

Evaluate three models:

1. original adapter \(z\)
2. personality-subspace-only adapter \(z_{\parallel}\)
3. personality-subspace-removed adapter \(z_{\perp}\)

Measure:

- target personality behavior
- unrelated task capability
- generic response quality
- refusal/safety behavior if relevant
- perplexity/loss where practical

Strong evidence would be:

\[
\text{trait effect}(z_{\parallel})
\approx
\text{trait effect}(z)
\]

while

\[
\text{trait effect}(z_{\perp})
\ll
\text{trait effect}(z)
\]

with minimal unrelated degradation.

This would show the subspace is not merely decodable but causally mediates the trait effect.

---

# 16. Synthesize novel personalities from the subspace

If the causal intervention works, test linear synthesis.

Construct

\[
z(\alpha)
=
\sum_{j=1}^{k}
\alpha_j p_j
\]

for selected coefficient vectors \(\alpha\).

Test:

- single-axis sweeps
- interpolation between two existing trait adapters
- extrapolation beyond observed adapters
- combinations of two or more personality dimensions

Measure behavioral profiles as coefficients vary.

Key question:

> Does smooth movement in learned weight-space coordinates produce smooth and predictable movement in behavioral personality space?

If yes, the subspace begins to function as a coordinate system for personality interventions.

---

# 17. Cross-base replication

Lower priority than the above, but valuable after the core result is established.

Repeat a subset of traits on another base model.

Do not require identical raw directions across architectures.

Instead ask whether the following replicate:

- low-dimensionality
- trait-vs-recipe separation
- behavioral predictability
- layer/module localization pattern
- paired-difference structure
- causal projection results

---

# 18. Priority order

## Tier 1: run now on the existing 134 adapters

1. Canonicalize all \(\Delta W\) vectors.
2. PCA/SVD of raw weight updates.
3. Pairwise cosine/distance analysis.
4. Opposite-trait difference-vector PCA/SVD if pairs exist.
5. Behavioral-profile every adapter.
6. Weight-space vs behavioral-space similarity correlation.
7. PLS / reduced-rank regression from weights to behavior.
8. Cross-validated performance vs subspace dimension \(k\).
9. Representation ablation.
10. Negative controls.
11. Layer/module localization.

## Tier 2: train a controlled replica zoo

1. 20–40 representative traits.
2. 3–5 seeds each.
3. at least two data formulations/sources.
4. randomized rank/LR/optimizer where practical.
5. test trait-vs-recipe predictability.
6. cross-source trait generalization.
7. recompute personality subspace using replica averages / paired contrasts.

## Tier 3: strongest mechanistic experiments

1. graded trait-strength adapters
2. personality-subspace-only projection
3. personality-subspace removal
4. linear interpolation/synthesis
5. cross-base replication

---

# 19. Minimum set of figures/tables to produce

Produce at least:

### Figure 1
PCA of raw personality LoRA updates.

### Figure 2
Singular/eigenvalue spectrum of:
- raw adapters
- paired trait-difference vectors

### Figure 3
Heatmap of pairwise weight-space similarity.

### Figure 4
Heatmap of pairwise behavioral similarity, same ordering as Figure 3.

### Figure 5
Scatter:
- weight similarity vs behavioral similarity
- include permutation/null baseline

### Figure 6
Behavioral prediction performance vs personality-subspace dimension \(k\).

### Figure 7
Representation ablation:
- full \(\Delta W\)
- \(u_1\)
- \(u_{1:3}\)
- \(v_1\)
- singular values
- norms
- random baseline

### Figure 8
Layer/module localization performance.

### Figure 9
For replica zoo:
within-trait vs between-trait distance distributions.

### Figure 10
Cross-source nearest-neighbor / classification performance.

### Figure 11
Trait-strength dose-response.

### Figure 12
Causal projection:
- original
- subspace-only
- subspace-removed

### Table 1
Dataset/zoo metadata and adapter counts.

### Table 2
Cross-validated trait classification / retrieval results.

### Table 3
Behavioral prediction results.

### Table 4
Confound prediction:
trait vs seed/rank/LR/optimizer/data source.

---

# 20. Statistical practices

Use:

- bootstrap confidence intervals
- permutation tests for similarity-matrix correlations
- cross-validation grouped by adapter / seed / source
- multiple-comparison correction where many trait-wise tests are reported
- held-out test sets for headline numbers
- report both mean and variance across splits/seeds

Avoid interpreting a visually appealing UMAP/t-SNE plot as primary evidence.

UMAP/t-SNE are fine for visualization, but quantitative conclusions should be based on the original representation or properly cross-validated learned projections.

---

# 21. Central hypotheses

The agent should explicitly test the following.

### H1: Trait identity is encoded in LoRA weight geometry

Adapters for the same personality trait should be more similar to each other than to adapters for different traits across seeds/training recipes.

### H2: Weight geometry predicts behavioral personality

Distances/coordinates in weight space should predict independently measured behavioral differences.

### H3: Personality is low-dimensional

A small \(k\)-dimensional subspace should recover most out-of-sample behavioral predictive power.

### H4: The subspace is invariant to nuisance training choices

Trait information should survive changes in seed, rank, learning rate, optimizer, and data formulation.

### H5: Personality directions are graded

Subspace coordinates should vary monotonically with trait strength.

### H6: The subspace is causally relevant

Projecting an adapter onto the personality subspace should preserve much of its personality effect, while projecting the subspace out should suppress that effect.

---

# 22. What would count as a compelling result?

A particularly strong result would look like this:

> A low-dimensional subspace of LoRA parameter updates predicts the behavioral personality profiles of held-out adapters, including adapters independently trained using different datasets and hyperparameters. The representation generalizes across training realizations, shows graded movement with trait strength, and projecting the subspace out of an adapter selectively removes its induced personality effect.

That would support the claim that the zoo contains a **reproducible, behaviorally grounded personality subspace**, rather than merely superficial clustering of fine-tuning artifacts.

---

# 23. Recommended first concrete execution

If time is limited, do this first:

1. Load all 134 adapters and reconstruct signed \(\Delta W\).
2. Flatten a consistent set of modules into one vector per adapter.
3. Run PCA and pairwise cosine analysis.
4. If polarity pairs exist, compute all \(z_+ - z_-\) vectors and PCA those.
5. Produce behavioral profiles for every adapter.
6. Compute weight-distance vs behavior-distance correlation.
7. Fit PLS / reduced-rank regression for \(k \in \{1,2,3,5,8,12,16,24,32\}\).
8. Cross-validate on held-out adapters.
9. Run representation ablations.
10. Run layer/module localization.
11. Based on results, choose ~30 traits for a controlled replica zoo.

The main decision criterion for continuing should be whether the **behaviorally supervised analysis generalizes out of sample**, not whether the first 2D projection looks visually clean.
