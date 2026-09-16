---
title: ASVD (Yuan et al., 2023) and whether the zoo's LoRA initialisation should have been different
summary: Samuel asked whether arXiv 2312.05821 means the LoRAs should have been initialised differently; the paper is ASVD, a training-free activation-aware SVD compression method, not an initialisation paper, and the answer is no, keep the shared Kaiming A, but its one transferable claim (the Frobenius metric is the wrong metric for judging low-rank structure in an LLM) argues for an activation-weighted Gram as a re-analysis, which would also test whether the seed floor is partly a metric artefact.
status: current
sources:
  - https://arxiv.org/abs/2312.05821
  - wiki/pages/zoo/stage-one-training-config.md
  - wiki/pages/geometry/seed-floor.md
  - wiki/pages/geometry/column-space-structure.md
  - wiki/pages/geometry/rank-sweep.md
  - qwen35/analysis/act_gram.json
last_verified: 2026-09-11
tags: [history, literature, lora, initialisation, metric]
---
# ASVD and the zoo's initialisation

Read on 2026-09-10 by an Opus subagent at Samuel's request ("read
https://arxiv.org/pdf/2312.05821 and lmk if this means we maybe should have
initialised our loras differently"). This page records the reading and the
answer; nothing here has been run.

## What the paper is

**ASVD: Activation-aware Singular Value Decomposition for Compressing Large
Language Models** (Yuan, Shang, Song, Yang, Wu, Yan, Sun; arXiv:2312.05821, v1
December 2023, v5 August 2025; preprint). A training-free post-training
compression method. It observes that LLM activations are highly non-uniform
across input channels, applies an invertible diagonal transform built from
per-channel mean activation magnitude before the SVD of each weight matrix, and
truncates in that activation-weighted metric rather than the plain Frobenius
one; a per-layer perplexity sweep then allocates truncation ratios under a
parameter budget. The same factorisation shrinks the KV cache. Claims 10 to 30
percent weight compression without loss.

It says nothing about initialisation, LoRA, a zero factor, learning-rate
asymmetry, or random frames. Those belong to a later literature it is an
ancestor of (CorDA 2406.05223, PiSSA 2404.02948, LoRA-GA 2407.05000, EVA
2410.07170, LoRA+ 2402.12354).

## Does it apply to the zoo's regime

The zoo's recipe ([[stage-one-training-config]]) is PEFT default: A
Kaiming-uniform, B zero, one shared A (seed 0) across all 134 adapters, rank
64, effective scale 2.0, 13 AdamW steps; A drifts about 1.5 percent
([[adapter-effect-and-drift]]), so a delta is effectively B times a fixed
random frame, close to LoRA-FA.

A random A loses no activation energy on average (E||Ax||^2 = r sigma^2 ||x||^2
whatever the anisotropy of x); what it does is spread the 64 coordinates over
an isotropic mixture of channels where a data-dependent A would concentrate
them on the directions the data uses. In a 13-step budget that could matter for
fit, which [[rank-sweep]] shows is the marginal quantity. ASVD provides no
evidence about optimisation, so that is an inference.

**Geometry.** A data-dependent A (top-r eigenvectors of the input covariance)
would make cross-seed adapters comparable as vectors by abolishing the seed,
and would thereby remove the project's strongest artefact check: the seed floor,
the cross-seed Gram correlation, the 40 of 40 cross-initialisation
identification and the rank-sweep frame nesting all stop testing anything. The
column-space result ([[column-space-structure]]) would be unchanged, because B
still starts at zero and accumulates output-side error vectors regardless of A.

**Behaviour.** A modest convergence gain is the honest expectation (what CorDA
and PiSSA report); whether it converts into own-factor amplification is not
predictable from this paper.

## Recommendation

Keep the current initialisation. What ASVD argues for is a re-analysis, not a
retrain: the Frobenius inner product is the isotropic metric, and ASVD's case
is that the activation-weighted one is the functional one. Build C = XX^T/n per
module from the base model on the 445-prompt pool and rebuild the Gram as
<dW_i, dW_j>_C = s_i s_j tr(B_i^T B_j A_j C A_i^T), still thin and exact. This
is the "dW . h" functional probe that [[seed-floor]] lists as not implemented.

Cheapest test, on existing artefacts: the 40 objective-matched seed-1 traits
against the 134 seed-0 adapters, same-trait cross-seed cosine under C against
the Frobenius +0.018, with the null recomputed under C (r/d = 0.025 does not
transfer; use the empirical random-subspace null recipe from
`qwen35/analysis/column_space.json`). If the functional cross-seed cosine sits
well above the Frobenius one, part of the seed floor is a metric artefact.
About one to two dollars.

## Ideas it suggests, ranked by value per dollar

1. Activation-weighted Gram (above). CPU; reuses the factor identity, the
   activation harvest in `probe_invariant.py`, the 445 pool.
2. A-drift channel profile: does A's 1.5 percent drift concentrate on
   high-activation input channels? Compare per-column drift of A against
   per-channel activation magnitude; reuses the rank sweep's `A_0` file. CPU.
3. Column space against the base weight spectrum: does the generic output-side
   subspace coincide with the base W's top-k left singular directions? CPU.
4. Sensitivity-weighted module aggregation: every geometry statistic averages
   the 248 modules unweighted; re-weight per-module Grams by a functional
   sensitivity and see whether factor separation strengthens. Re-analysis only.
5. Activation-aware A arm: five traits (one per factor, all with seed-1 twins),
   A = top-64 eigenvectors of C, otherwise the recipe; read fit and own-factor
   amplification against the zoo bands. About $2 plus judging; only worth it if
   idea 1 says the metric matters.
6. Non-uniform rank allocation under a parameter budget matching r = 4, on the
   15 rank-sweep traits; asks whether fit recovers where uniform r = 4 failed.
   About $8; weakest value per dollar.

## Citation value

Cite ASVD for the claim that input-channel activation magnitude is strongly
non-uniform in LLMs and that a rank-k truncation chosen in the Frobenius metric
is worse than one chosen in the activation metric; it belongs as a caveat on
[[seed-floor]] (0.025 is the null of an isotropic metric) and on
[[column-space-structure]]'s framing of the row space as the initialisation.
For initialisation precedent cite CorDA, PiSSA, LoRA-GA, EVA and LoRA+.

## Idea 1 was run

On 2026-09-11 the activation-weighted Gram at the top of that list was built:
`qwen35/act_gram_on_modal.py`, one A100-80GB, about $0.50. The full input
covariance C was formed for all 248 modules on the 445-prompt pool, and the
134 x 134 Gram and the 134 x 40 cross-seed block were rebuilt in it. The
same-trait cross-seed cosine goes from +0.0181 to **+0.6643**, but so does the
overlap of two independent random rank-64 frames measured in C (0.0215 to
0.8910), because the participation ratio of C is about ten and a rank-64 frame
therefore already spans nearly all the input variance the model presents. Against
that ceiling the two metrics agree, so the seed floor is **real in the functional
metric** and not a metric artefact - while the absolute number a reader sees
changes by a factor of 36.8. The five-factor solution is unaffected (Tucker
congruence 0.9926). See [[activation-weighted-gram]].

Related: [[literature-lora-and-merging]], [[paper-reading-2026-09-09]],
[[factor-analysis-fisher-metric]], [[activation-weighted-gram]],
[[open-questions]].
