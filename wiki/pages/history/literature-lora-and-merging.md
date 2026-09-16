---
title: LoRA and merging reference
summary: LoRA, rsLoRA, model soups and task arithmetic -- the four works behind the zoo's adapter parameterisation, its scaling convention, and every weight-space merge in the project.
status: current
sources:
  - qwen35/paper_notes.md
  - qwen35/train_qwen35.py
  - qwen35/RUNNER_TASK.md
  - qwen35/gram_on_modal.py
  - https://arxiv.org/abs/2106.09685
  - https://arxiv.org/abs/2312.03732
  - https://arxiv.org/abs/2203.05482
  - https://arxiv.org/abs/2212.04089
last_verified: 2026-09-07
tags: [literature, lora, merging, reference]
---

# LoRA and merging reference

## LoRA

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.*
arXiv:2106.09685 [cs.CL], submitted 17 June 2021; v2 16 October 2021. `[abstract only]`

Freezes the pretrained weights and injects trainable rank-decomposition matrices
into each layer, so the update is delta-W = BA with B and A of rank r. Reported
10,000x fewer trainable parameters and 3x less GPU memory than Adam finetuning
for GPT-3 175B, at comparable or better quality and with no added inference
latency.

**What this project takes from it.** Everything, structurally. The zoo is 134
LoRA adapters, and the reason weight-space geometry over 134 personality traits
is affordable at all is that each trait costs two thin matrices per module rather
than a full model. The low-rank form is also what makes the geometry *exact and
cheap*: `qwen35/gram_on_modal.py` computes Frobenius inner products between
adapters straight from the factors, since for delta-W = s BA the inner product is
s_i s_j tr(A_i^T B_i^T B_j A_j), computable per module without ever materialising
a d_in x d_out matrix. `qwen35/paper_notes.md` section 3.8 contrasts this with
[[persona-cartography-paper|Persona Cartography]], which flattens roughly
8-billion-dimensional dense vectors instead.

**Where the project departs.** LoRA's original argument is that the *intrinsic
rank* of a task update is low. The pre-zoo LoRA-structure measurements in
`CONTEXT.md` section 3.7 both support and complicate that: across five adapters
at r=16 over 252 modules the exact spectra are near-identical, with energy at
rank k running 1 -> 49%, 2 -> 65%, 4 -> 80%, 8 -> 92%, 12 -> 97% and an effective
rank near 3.8. But the trait identity is not spread over that spectrum evenly:
truncating a held-out trait to rank k and asking how much the other four traits
span it gives 0.284 at rank 1 rising to 0.477 at full rank, so **the leading
singular direction is the most trait-specific part and the tail is the generic
shared component**. That resolves the apparent puzzle in Persona Cartography's
rank-1 compression appendix -- rank 1 keeps only about half the energy, but it is
the half that identifies the trait.

## rsLoRA

Damjan Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with
LoRA.* arXiv:2312.03732, submitted 28 November 2023. `[abstract only]`

A one-idea paper with a large practical footprint. LoRA multiplies the adapter by
alpha/r; the paper proves the adapter should be divided by the **square root** of
the rank instead, so the scaling is alpha/sqrt(r). Dividing by r "results in
slowed learning and stunted performance for LoRA with higher-rank adapters",
which is why LoRA in practice had been confined to very low ranks.

**What this project takes from it, and what it decided.** rsLoRA is the largest
single configuration question in the project's own record and it resolved against
using it. `qwen35/paper_notes.md` section 3.2, written 2026-08-19, called the
planned use of rsLoRA "deliberate?, and the largest silent one": neither source
paper uses it, so their scaling is alpha/r = 128/64 = 2.0 while rsLoRA at the
same alpha and rank gives 128/8 = 16.0 -- an 8x larger functional step, at a
learning rate copied verbatim from runs that did not have it. Two consequences
were flagged: optimisation (5e-5 was implicitly tuned against scaling 2.0) and
geometry (norms landing 8x above Persona Cartography's 6.08-6.53 band, making
their "sum the scales directly" shortcut and their "keep total scale below ~2"
rule of thumb non-transferable).

**The executed zoo turned it off.** `qwen35/RUNNER_TASK.md` records the settled
configuration as "plain LoRA alpha 128 rank 64 (use_rslora False), lr 5e-5, beta
0.1, loss_type ["sigmoid","sft"] weights [1.0,0.1], kl_coef 0.001", and
`qwen35/PHASE3_VERDICT.md`'s 2026-09-03 addendum names the zoo objective with
"rslora off". So the effective scaling is 2.0, matching both source papers, and
the norms are directly comparable to Persona Cartography's Table 1. paper_notes
section 3.2 is superseded on this point by the run.

Two details from `qwen35/train_qwen35.py` worth keeping. First, at rank 64 the
two conventions are the *same transform* if alpha is set accordingly -- rsLoRA
with alpha 16 and plain LoRA with alpha 128 both give scale 2.0 -- and they only
diverge when rank varies. Second, the file records a process failure worth
remembering: the decision to use plain LoRA was taken on 2026-08-20 and the
sweep ran with `PC_USE_RSLORA=0` on the launch line while the code default stayed
`1`, so "the CHOICE lived in a shell command and the CODE still held the
pre-decision state". The default was later changed to encode the decision. See
[[zoo-training-recipe]].

The analysis side reads the convention rather than assuming it:
`qwen35/gram_on_modal.py` and its siblings compute the per-module scale as
`alpha/sqrt(r)` if `use_rslora` is set in `adapter_config.json` and `alpha/r`
otherwise, and the scale must be applied per adapter *before* the inner product.

## Model soups

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs,
Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair
Carmon, Simon Kornblith, Ludwig Schmidt. *Model soups: averaging weights of
multiple fine-tuned models improves accuracy without increasing inference time.*
arXiv:2203.05482, submitted 10 March 2022; v3 1 July 2022. ICML 2022.
`[abstract only]`

Instead of selecting the best of several finetuning runs, average their weights.
"Averaging the weights of multiple models fine-tuned with different
hyperparameter configurations often improves accuracy and robustness", with no
ensemble inference cost. State of the art on ImageNet at the time (90.94% top-1),
with gains out of distribution and zero-shot, and a theoretical analysis relating
the effect to loss-landscape flatness.

**What this project takes from it.** The licence for the whole two-stage merge.
Both source papers form their released adapter by souping the DPO and
introspection adapters at weights [1.00, 0.25] -- [[open-character-training-paper]]
in `tools/merge_loras.py`, and Persona Cartography inheriting the 0.25 explicitly
"following Maiya et al. 2025". [[stage-two-geometry]] is where this project's own
souping lives, and [[persona-merge-correction]] records a correction the project
had to make about what its own merge computed.

**Where this project departs, and it is the sharpest point on this page.** A
soup of *full models* is a weighted average of weights. A soup of *LoRA adapters*
under PEFT's `combination_type="linear"` is not. As Persona Cartography's
Appendix A.1.3 works out (quoted in full on [[persona-cartography-paper]]), the
merge combines the low-rank factors with square-root weights,
A_merged = sum_i sqrt(w_i) A_i and B_merged = sum_i sqrt(w_i) B_i, to keep the
merged adapter at rank 64 -- and because (A, B) -> BA is bilinear, the result is
the intended weighted sum **plus cross terms** sqrt(w_DPO w_SFT)(B_DPO A_SFT +
B_SFT A_DPO) belonging to neither stage. Every released OCT persona adapter and
every Persona Cartography OCEAN adapter contains those cross terms. The zoo's
main geometry is computed on DPO-only deltas partly for that reason: a
single-stage delta is a clean object and a soup is not.

## Task arithmetic

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan,
Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi. *Editing Models with Task
Arithmetic.* arXiv:2212.04089, submitted 8 December 2022. ICLR 2023.
`[abstract only]`

Defines a **task vector** as finetuned weights minus pretrained weights, and
shows that these vectors compose arithmetically: negating one degrades the task,
adding several improves several at once, and analogical combinations
(A - B + C) transfer to a fourth task. The paper the whole
delta-as-a-direction framing descends from.

**What this project takes from it.** The premise. A zoo adapter is a task vector
for a personality trait; Persona Cartography's elementwise composition ("we
compose LoRAs by summing all of the delta-W from each (optionally scaled) LoRA,
elementwise") is task arithmetic; and this project's steering directions are
weighted sums of task vectors, which is why `qwen35/steer_qwen35.py` can steer
along a principal component with no retraining -- a principal direction of the
adapter cloud is a linear combination of adapters that already exist.

**Where this project departs.** Task arithmetic is stated over full-model
deltas from a *shared* pretrained initialisation, where two task vectors are
genuinely comparable coordinates. Two LoRA adapters share the base model but not
the adapter initialisation, and [[seed-floor]] measures what that costs: the same
trait trained twice under different LoRA seeds has a median cross-seed
self-cosine of +0.0167, against a preregistered bar of +0.2447 and a
different-trait floor near 0.0013. `qwen35/PHASE3_VERDICT.md` withdraws the
trait-level claim on that basis -- "There is no evidence here that an individual
trait has a characteristic direction in weight space" -- while upholding the
factor-level organisation. So task arithmetic over LoRA deltas is arithmetic
inside one initialisation, not in an initialisation-free space, and both source
papers' cosine and PCA results are single-initialisation quantities. Any
composition or steering claim in this project inherits that caveat; see
[[geometry-overview]].

## Verification note

**Abstract only, all four.** Each arXiv abstract page was fetched 2026-09-07 to
confirm title, author list, submission and revision dates, and the venue where
the comments field named one (ICML 2022 for model soups, ICLR 2023 for task
arithmetic; none for LoRA or rsLoRA). None of the four appears in
`qwen35/paper_notes.md`'s "sources actually read" table -- rsLoRA is discussed
there at length but as a configuration decision, not as a paper -- so the
project's own engagement with them is through the code, and the
project-side claims above are cited to the code files rather than to the papers.

See also [[paper-asvd]] (2026-09-10): activation-aware SVD, read against the zoo's shared Kaiming initialisation; not an initialisation paper, but the argument for an activation-weighted Gram.
