---
title: Early LoRA structure — the 34 OCEAN adapters
summary: Thirty-four Qwen2.5-3B LoRAs measured for rank spectra, compositional arithmetic and interpolability, establishing that rank-1 keeps the trait-identifying half of the energy.
status: historical
sources:
  - adapters/A/trainmeta.json
  - adapters/A/adapter_config.json
  - results/summary.md
  - results_synth/summary.md
  - CONTEXT.md
  - /home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md
last_verified: 2026-09-07
tags: [history, lora, rank, arithmetic]
---

# Early LoRA structure: the 34 OCEAN adapters

The second body of work, 2026-08-12 to 08-14. It exists because
[[drift-experiment]] needed a companion question: if a trait cannot be removed
from an update, what *is* the structure of a trait update?

## What was trained

**Base model: `Qwen/Qwen2.5-3B-Instruct`** — read from
`adapters/A/adapter_config.json#base_model_name_or_path` and confirmed in
`adapters/A/trainmeta.json#base_model`. Configuration from the same files:

| field | value |
|---|---|
| LoRA rank `r` | 16 |
| `lora_alpha` | 32 (effective scale alpha/r = 2.0; `use_rslora` false) |
| target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| modules per adapter | 252 (36 layers x 7) |
| examples per condition | 320 |
| epochs | 3 |
| GPU | A100-40GB |
| seed | 0, `shuffle_seed` 0 |
| wall time (condition A) | 93.81 s |

The 34 conditions in `adapters/` are: five OCEAN singles (**O, C, E, A, N**), ten
**compositional** pairs (`C_A`, `C_E`, `C_N`, `E_A`, `E_N`, `O_A`, `O_C`, `O_E`,
`O_N`, `A_N`), ten matching **union** pairs (`*_union`), three **half-split**
noise controls (`C_h1`/`C_h2`, `E_h1`/`E_h2`, `O_h1`/`O_h2`) and three **reseed**
controls (`C_s1`, `E_s1`, `O_s1`). `results/summary.md` states the corpus as "34
conditions, 252 modules, 36 layers, accum float64".

Supporting artefacts: `smoke_data/` (five smoke corpora) and
`adapters_smoke_evidence/` (four smoke adapters) are the pipeline's own
end-to-end checks; `synth_adapters.py` and `test_weight_analysis.py` build
synthetic adapters with known answers so the factored machinery can be validated
before it is trusted. `results_synth/summary.md` is that synthetic run.

The analysis never materialises a dense `dW`: everything goes through the
factored identity, verified against a dense reference. `results/pooling_check.md`
records the checks passing at max abs error 4.547e-13 (relative 2.4e-16).

## Traits are separable above the noise floor

From `results/summary.md` section 5:

| kind | comparison | relative distance | cosine |
|---|---|---|---|
| half-split C | C_h1 vs C_h2 | 0.6610 | 0.7815 |
| half-split E | E_h1 vs E_h2 | 0.6137 | 0.8117 |
| half-split O | O_h1 vs O_h2 | 0.6555 | 0.7852 |
| reseed C | C vs C_s1 | 0.3382 | 0.9429 |
| reseed E | E vs E_s1 | 0.3070 | 0.9529 |
| reseed O | O vs O_s1 | 0.3502 | 0.9387 |
| cross-trait (min) | A vs C | 0.9169 | 0.5797 |
| cross-trait (max) | E vs N | 1.0605 | 0.4377 |

Max same-trait floor 0.6610 against min cross-trait 0.9169; mean-cross over
mean-floor **2.0369**. Every cross-trait distance exceeds every same-trait floor.

## Weight arithmetic is miscalibrated

Naive summing of two trait deltas overshoots. Fitting two global scalars instead
(`results/summary.md`, "Fits, compositional targets" and "Fits, union targets"):

- **Compositional targets**: rung-1 coefficients across 10 pairs a = 0.4472 +-
  0.0840, b = 0.4196 +- 0.0851, pooled **0.4334 +- 0.0856** (range 0.2785 to
  0.5986). Deviation of the pooled mean from 1.0 is **-0.5666**.
- **Union targets**: fitted coefficients run 0.5653 to 0.7679, and naive relative
  residual is 0.67-0.70 against 1.15-1.20 for compositional targets.
- Per-module coefficients (rung 2) buy almost nothing beyond the two global
  scalars — rung-1 to rung-2 relative residual moves e.g. 0.6369 to 0.6277 on
  `A_C`.

Stated narrowly: this is a **weight-space** result. The behavioural destination
was never measured, so it does not contradict the Persona Cartography paper's
near-additive behavioural scores. See [[persona-cartography-paper]].

## Rank spectra

Exact singular values of `dW = (alpha/r) B A` via thin QR on both factors, never
materialising `dW`. Five adapters (`syc_pure`, `plain`, `neutral`, `O`, `N`), all
nominal r=16, 252 modules each. Numbers are near-identical across all five despite
different traits and different data (CONTEXT.md 3.7;
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`,
"LoRA rank spectra"):

| energy captured at rank k | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| fraction | ~0.49 | ~0.65 | ~0.80 | ~0.92 | ~0.97 | 1.00 |

- participation ratio (effective rank): mean ~3.8, median ~3.7, range 1.0-9.5
- median rank for 90% energy: 7-8 (max 12); for 99% energy: 15 (max 16)

So a rank-16 persona LoRA is effectively rank ~4, but the tail is real.

## Where trait identity lives — the rank-1 puzzle, resolved

Leave-one-out over the five OCEAN singles, held-out trait never in the basis (an
earlier "mean of all five" version self-contaminated in the same way the oracle
direction did, and was corrected). Truncate the held-out trait to rank k, then ask
how much of it the other four span (CONTEXT.md 3.7):

| rank | 1 | 2 | 4 | 8 | full |
|---|---|---|---|---|---|
| fraction spanned by the other four | 0.284 | 0.374 | 0.441 | 0.471 | 0.477 |

Monotone: **the leading singular direction is the most trait-specific part** and
sharedness rises with rank. This resolves the puzzle that rank-1 keeps only ~49%
of weight energy while the Persona Cartography paper reports trait control
surviving rank-1 compression — the half it keeps is disproportionately the half
that identifies the trait.

## Interpolability: you cannot get a trait from five

Energy of a held-out trait's `dW` captured by the **span** of the other four:
O 0.491, C 0.507, E 0.435, A 0.526, N 0.427, mean **0.477**. Captured by their
**mean direction alone** (rank 1): O 0.483, C 0.494, E 0.422, A 0.514, N 0.415,
mean **0.466**.

So about 47% of every trait's weight energy lies along one shared direction, and
the other traits' individual identities add **1.2 percentage points** beyond it.
One big generic persona-finetuning direction plus a per-trait residual that the
others say almost nothing about.

Two caveats the project attached: it is weight energy, not behaviour; and n=5 is
a tiny basis, so this says "you cannot interpolate from five", not "traits do not
interpolate". It argues for **scale** rather than against hypernetworks —
Doc-to-LoRA never interpolates in weight space.

## PCA across the early adapters — numbers with no artefact on disk

`pca_over_loras.py` was run on 2026-08-14 and its recorded output,
`results/pca_over_loras.txt`, is **zero bytes**; so is `results_pca.log`. The
spectra survive only in
`/home/vibe12/projects/agent-harness/memory/projects/persona-curvature.md`
("PCA ACROSS LoRAs (2026-08-14)"):

| set | uncentered | centered |
|---|---|---|
| 5 OCEAN singles | 60.5 / 11.6 / 10.5 / 9.3 / 8.1 | 29.3 / 26.7 / 23.6 / 20.4 |
| singles + 10 pairs (n=15) | 57.6 / 7.7 / 5.5 / 5.1 / 4.4 / 2.4 | 18.2 / 13.0 / 12.1 / 10.3 / 5.8 / 5.4 |

One dominant direction holding 58-60% of total energy, and it is not a trait
direction; once removed the residual is close to isotropic. The noise control from
the same record: cross-trait distance 4.234 against same-trait reseed 1.506, ratio
2.81. **These figures are weakly sourced** — the file that should hold them is
empty — and are flagged as such in `_report_history.md`.

Related: [[sweep100]], [[stage-one-training-config]], [[geometry-overview]],
[[persona-cartography-paper]].
