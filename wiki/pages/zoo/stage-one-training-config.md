---
title: Stage-one training configuration
summary: The 134 DPO adapters were trained at rank 64, alpha 128, plain LoRA (effective scale 2.0), beta 0.1, loss_type sigmoid plus sft at weights 1.0 and 0.1, KL 0.001, on 445 pairs for 13 optimizer steps, all sharing one random LoRA-A from seed 0.
status: current
sources:
  - qwen35/phase2_runs/archive/phase5_sweep_134.json
  - qwen35/train_qwen35.py
  - qwen35/HANDOVER.md
  - qwen35/base_config_snapshot.json
  - qwen35/PREREG_alignment.md
  - qwen35/analysis/blog_data.json#a_drift
  - qwen35/build_blog_page.py
  - qwen35/zoo_page/MODEL_CARD.md
last_verified: 2026-09-16
tags: [zoo, training, config]
---

# Stage-one training configuration

Every number below is what the **container recorded**, not what a launch command
asked for. The distinction is load-bearing in this project: `PC_LORA_ALPHA=16`
was once set on a launch line, never crossed into the container, the run
finished, the gates passed, and the comparison silently repeated the baseline
(`qwen35/train_qwen35.py`). See [[phase2-recipe-selection]].

## The recorded config of the 134-run sweep

From `qwen35/phase2_runs/archive/phase5_sweep_134.json` (134 records, one per
trait; the fields below are identical across all of them):

| field | value |
|---|---|
| `resolved_base_model` | `Qwen/Qwen3.5-4B` |
| `base_model_commit` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `lora_r` | 64 |
| `lora_alpha` | 128 |
| `use_rslora` | false |
| `expected_scaling` | 2.0 |
| `learning_rate` | 5e-05 |
| `beta` | 0.1 |
| `loss_type` | `["sigmoid", "sft"]` |
| `loss_weights` | `[1.0, 0.1]` |
| `kl_coef` | 0.001 |
| `kl_applied` | true |
| `epochs` | 1 |
| `effective_batch` | 32 |
| `max_length` | 1024 |
| `warmup_steps` | 1 |
| `optimizer` | `adamw_torch` |
| `adam_betas` | `[0.9, 0.98]` |
| `max_grad_norm` | 1.0 |
| `optimizer_steps` | 13 |
| `seed` / `order_seed` | 0 / 0 |
| `n_pairs` | 445 |
| `prompt_pool_sha256` | `34e749c8ca0f7d468fb83733e03b8148c163df46a99682f8b6481e79f63cec07` |
| `n_targeted` | 248 |

**Step count against the source papers.** 13 optimizer steps (445 pairs, batch
32, one epoch) is roughly a quarter of Open Character Training's roughly 47 and a
sixth of Persona Cartography's roughly 76, because neither paper's general-prompt
pool (LIMA) was added. The divergence is recorded as unresolved on
[[recipe-vs-source-papers]] section 3.5 and [[open-character-training-paper]];
the post draft's scope note ("13 optimizer steps on one pool of 445 prompts")
states it without resolving it. It cuts both ways: the shared pool is what makes
the 134 comparable, and it guarantees a common prompt-style component.

Stack: `torch 2.13.0`, `transformers 5.15.1`, `trl 1.10.0`, `peft 0.20.0`,
`accelerate 1.14.0`.

Health across the 134 runs, from the same file: `loss_last` ranges
0.14250712096691132 to 0.281541645526886 and `reward_margin` ranges
5.072036419595991 to 18.467349461146764 (minimum and maximum over the 134
records, read directly). The first record, `active`, has `train_seconds`
465.377254486084.

## What HANDOVER.md says

`qwen35/HANDOVER.md`, section "Working config, verified from container records
not launch commands", verbatim:

> r=64, alpha=128, **plain LoRA**, effective scale 2.0 (OCT's), beta 0.1,
> kl_coef 0.001, `loss_type=["sigmoid","sft"]` weights `[1.0,0.1]`, 248 targeted
> modules, seed 0. Base is `Qwen/Qwen3.5-4B`, a VISION-LANGUAGE model — target
> the text tower only.
>
> The NLL-on-chosen term at 0.1 is what prevents DPO collapsing to a trivial
> discriminator. That came from reading OCT's repository, not the paper. Do not
> drop it.

## The 248 targeted modules

The base model is a composite vision-language checkpoint
(`Qwen3_5ForConditionalGeneration`, `qwen35/base_config_snapshot.json`), with a
32-layer text tower — 24 linear-attention layers and 8 full-attention layers,
hidden size 2560, intermediate 9216, vocab 248320 — and a 24-layer vision tower.
Only the text tower is targeted. Of 249 linear modules found, 248 are targeted
and 1 is never targeted (`lm_head`, the tied output head over the whole vocabulary;
`qwen35/train_qwen35.py:NEVER_TARGET`).

The breakdown recorded per run (`#targeted_by_module_kind`):
`out_proj` 24, `in_proj_qkv` 24, `in_proj_z` 24, `in_proj_b` 24, `in_proj_a` 24,
`gate_proj` 32, `up_proj` 32, `down_proj` 32, `q_proj` 8, `k_proj` 8, `v_proj` 8,
`o_proj` 8. By layer type: linear_attention 120, mlp 96, full_attention 32.
`n_excluded_vision` is 0 because the vision tower contributes no matching
modules under the text prefix.

## The shared LoRA-A

`qwen35/train_qwen35.py` seeds `random`, `numpy` and `torch` immediately before
the `LoraConfig` is constructed. Since PEFT initialises `B` at zero and draws `A`
at random, every adapter trained at the same seed gets the **same** `A`. And
because `B` starts at zero, `A` receives almost no gradient, so it stays nearly
the same through training.

Two measurements of how nearly:

- `qwen35/PREREG_alignment.md`: "Measure mean ||A_i - A_0|| / ||A_0|| against the
  zoo's A_0. The zoo's own internal figure is **0.0146**."
- `qwen35/build_blog_page.py:374`: "across the whole zoo, *A* moves just
  **1.5%** of its norm during training."
- the underlying computed value is `a_drift` = 0.014605041334818797
  (`qwen35/analysis/blog_data.json#a_drift`, and identically in
  `analysis/align_scores.json`, `analysis/align_summary.json`,
  `analysis/nxn_scores.json`). `qwen35/analyse_alignment.py` hardcodes
  `ZOO_DRIFT = 0.0146` as the gate constant.

The consequence, as the blog page puts it: all 134 write into the same
64-dimensional slice of input space and are "directly comparable in a way that
independently initialised adapters are not — two different draws overlap by only
r/d = 64/2560 = 2.5%." That is why a second seed's adapters read as nearly
orthogonal in coordinates while the *arrangement* still replicates; see
[[seed-floor]] and [[geometry-overview]].

## Seeds as treatment, not nuisance

`SEED = 0` and `ORDER_SEED = 0` are declared once in `train_qwen35.py`, travel in
the job dict rather than the environment, and are recorded per run along with
`seed_is_default` and `corpus_is_seedpaired`. For the seed-paired null arm the
seed *is* the treatment — that corpus is byte-identical to `data_common` on
purpose — so the trainer refuses to launch a `seedpaired` corpus at the default
seeds. Guards compare against the named constants, never against a literal 0,
"a guard pinned to today's constant stops following the constant the moment
someone edits it, which is the bug class this project has now hit seven times."

## Corpus

`qwen35/data_common/`, 445 pairs per trait, the intersection described in
[[shared-prompt-pool-445]]. `train_qwen35.py` refuses to launch unless every
corpus file carries the same prompt-pool hash, computed on the bytes the
container sees.

## Superseded config statements

`qwen35/plan.json#defaults` still reads `use_rslora: true` with a note calling
rsLoRA "the DEFAULT (Samuel); PLAIN LoRA is an ablation arm", and
`qwen35/paper_notes.md` section 3.2 argues against the project's own rsLoRA
setting. Both predate the 2026-08-20 decision. The 134 recorded runs all read
`use_rslora: false`, `expected_scaling: 2.0`. Treat `plan.json#defaults` as
superseded on this point. See [[recipe-vs-source-papers]].

`qwen35/zoo_page/MODEL_CARD.md` reproduces the same recorded config and adds:
"seed 0 throughout (`order_seed` 0)".
