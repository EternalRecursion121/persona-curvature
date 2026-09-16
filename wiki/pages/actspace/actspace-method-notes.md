---
title: "Activation-space analyses: method notes and limits"
summary: Everything needed to reproduce the three activation-space runs of 2026-09-05, and the list of things they do not establish.
status: current
sources:
  - qwen35/act_space.py
  - qwen35/analyse_actspace.py
  - qwen35/analyse_actspace_adapters.py
  - qwen35/analyse_actspace_cross.py
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/analysis/actspace_spec.json
  - qwen35/phase10_runs/actspace.log
  - qwen35/phase10_runs/actspace.log.1788630847
  - qwen35/phase10_runs/actspace_adapters.log
  - qwen35/phase10_runs/actspace_cross.log
  - qwen35/phase10_runs/decompose_actspace_resp.log
  - qwen35/results/decomposition_actspace_resp.json#test6.status
  - qwen35/blog_page/index.html
last_verified: 2026-09-07
tags: [actspace, method, replication, limits]
---

# Activation-space analyses: method notes and limits

Companion to [[actspace-overview]], [[actspace-persona-vectors]],
[[actspace-adapters]] and [[actspace-cross]]. This page is written for someone
who wants to rerun the arm or to check a number in it.

## Environment

From `act_space.py` lines 48-56 and the run logs:

- Modal app, default name `pc-qwen35-phase11-actspace`
  (`PC_APP_NAME` overrides it). Volumes `pc-qwen35-probe` (outputs, mounted at
  `/probe`) and `pc-qwen35-sweep` (the stage-1 adapters, at `/adapters`).
- Image: `nvidia/cuda:12.8.1-devel-ubuntu24.04`, Python 3.12,
  `torch==2.13.0`, `transformers==5.15.1`, `accelerate==1.14.0`, `numpy<3`,
  `safetensors`, `huggingface_hub`.
- GPU **A100-40GB** for all three stages; timeouts 4 h (prompt), 5 h (adapters,
  cross).
- Base model `Qwen/Qwen3.5-4B` (`PC_BASE_MODEL`), loaded in **bfloat16**,
  `device_map="cuda"`, `.eval()` (line 73-74).
- Logged shape: `33 hidden-state layers x 2560`; `248` hooked LoRA modules;
  `LoRA scale 2.0`.

Local entry point: `modal run act_space.py --stage prompt | adapters | cross`
(lines 353-375).

## The 64 prompts

```python
pool = sorted({json.loads(l)["prompt"] for l in open(f"{HERE}/data_common/bold.jsonl")})
prompts = random.Random(7).sample(pool, N_PROMPTS)      # N_PROMPTS = 64
```

`data_common/bold.jsonl` has 445 lines, one prompt each and all distinct, so the
pool is 445 and 64 are drawn from it with `random.Random(7)`. The drawn prompts
are stored verbatim at `analysis/actspace_spec.json#prompts` -- the file to read
if you want the exact 64 rather than to redraw them.

`actspace_spec.json` is rewritten at the end of **every** stage run (line 375),
and its `traits` field is always the full 134-name list from
`analysis/alien.json#traits`, never `CROSS_TRAITS`. The file on disk is dated
21:23 on 2026-09-05, i.e. it was last written by the cross run. Since the prompt
draw is seeded and identical across stages, the file's `prompts` are the 64 used
by all three.

## Windows and pooling

Per batch of 32 (`act_space.py` lines 97-135):

1. Render `[system?, user]` with `add_generation_prompt=True` and
   **`enable_thinking=False`**; tokenise with **left padding**,
   `add_special_tokens=False`.
2. `model.generate(max_new_tokens=96, do_sample=False)` -- greedy.
3. Re-run the full sequence with `output_hidden_states=True`, with the attention
   mask extended by ones over the generated positions.
4. **Response window**: positions from `T0` up to and including the first
   EOS-or-pad in the generated span; if there is none, to the last generated
   position.
5. **Prompt window**: positions from `pad_len + n_sys` up to `T0`, i.e. the user
   turn only, with left padding and the system block both excluded. `n_sys` is
   measured by rendering the same prompt with and without the system turn and
   differencing token counts, because the template refuses a system-only message
   list.
6. Mean over the window, per hidden-state layer, giving one `(33, 2560)` matrix
   per prompt per condition.

Halves: prompts 0-31 and 32-63 are averaged separately and both are stored.

Storage: `M` is **float16** in the `.npz` (`means.npz`, `means_adapters.npz`,
`means_cross.npz`, on the `pc-qwen35-probe` volume; the local copies are
`analysis/actspace_means*.npz`, 91 MB / 91 MB / 173 MB). Every analysis script
casts to float32 on load.

**The adapters run does not collect its own baseline.** It reuses `B` from the
prompt run's `means.npz`, on the argument that with the hooks disarmed the model
is the base model and the baseline came from the same code path
(`act_space.py` lines 260-261; `analyse_actspace_adapters.py` lines 50-52). The
same `B` is used again by the cross analysis
(`analyse_actspace_cross.py` line 39).

## Analysis: seeds, nulls, and what each null is

All three analysis scripts fix `PRIMARY = 16` and, where they randomise, use
`np.random.default_rng(0)`.

| null | draws | script |
|---|---|---|
| Gram-correlation label shuffle (`perm_p`) | `PERMS = 2000`, p as `(hits + 1) / (n + 1)`, floor 0.0005 | `analyse_actspace.py` 27, 47-55 |
| Procrustes label shuffle | 500 | `analyse_actspace.py` 150 |
| Hole transplant, permuted coefficients | 500 | `analyse_actspace.py` 171-174 |
| Containment "random subspace" | **one** draw | `analyse_actspace_adapters.py` 106-109 |

The containment null is a **single** random cloud, not a distribution: the
reported 0.2%-1.5% figures are one sample, not a percentile. Given the gap to the
observed 41%-64% this does not change the conclusion, but the numbers should not
be read as a null band.

The Procrustes null *mean* of 0.02 quoted in `ACTSPACE_RESULTS.md` and on the
blog is printed by `analyse_actspace.py` line 151 but is **not** stored in
`actspace_geometry.json`, which keeps only `procrustes_null95`.

## Centring conventions differ between scripts

This is the single most likely source of confusion in this section, and it is why
the same quantity has two published values:

- `analyse_actspace.py` lines 92-94: the weight cosine matrix `Cw` is built from
  the **raw** Gram. Activations are trait-centred (line 139). P~W = **0.705**.
- `analyse_actspace_adapters.py` lines 62-63: the weight Gram is
  **double-centred** (`Gc = J @ Gw @ J`) first. P~W = **0.737**.

Likewise, "the fraction of the prompt's shift the adapter delivers" is **0.62**
on raw vectors (`analyse_actspace_adapters.py`) and **0.70** on trait-specific
vectors (`analyse_actspace_cross.py`). Both are real; state which one you mean.
See [[actspace-persona-vectors]] and [[actspace-adapters]].

## Generations kept

The `.jsonl` files hold a sample only, for eyeballing: the first **4** responses
per trait for the prompt and adapter runs (plus one `_base` row), the first **2**
per condition for the cross run (`act_space.py` lines 162, 173, 256, 337).
Line counts: 135 / 134 / 256 in
`analysis/actspace_generations.jsonl`,
`actspace_generations_adapters.jsonl`,
`actspace_generations_cross.jsonl`. With `MAX_NEW = 96`, sampled responses end
mid-sentence; they are diagnostic, not evidence, and none were judged.

## The chat-template failure

The first launch died with
`TemplateError: No user query found in messages`
(`qwen35/phase10_runs/actspace.log.1788630847`). Qwen3.5's chat template rejects
a message list containing only a system turn. The workaround is the
render-with-and-without differencing above. Recorded in the journal for
2026-09-05.

Related standing hazard, and the reason for the explicit flag on every template
call: Qwen3.5's template defaults its reasoning block **on**, and three earlier
runs in this project were silently invalidated by omitting
`enable_thinking=False`.

## What these analyses do not establish

From `ACTSPACE_RESULTS.md`, from the blog's own caveats section, and from what
the scripts do and do not run:

1. **One model, one training procedure, one initialisation.** Everything is
   Qwen3.5-4B, stage-1 LoRA adapters from a single shared random `A`. Nothing
   here shows the correspondence would hold on another base model, another PEFT
   method, or another objective.
2. **Only stage-1 adapters were run.** The stage-2 (self-transcript SFT)
   adapters exist for all 134 traits and are on Hugging Face, but no
   activation-space run touched them.
3. **The correspondence is between arrangements, not coordinates.** The two
   spaces share no basis. What is compared is 134x134 similarity matrices and
   134x5 score matrices. Procrustes R^2 0.535 is the strongest coordinate-level
   statement made, and it is a statement about five components.
4. **The result is layer-dependent.** Trait-centred P~W runs 0.52 at layer 0 to
   0.775 at layer 19. Layer 16 was fixed in advance; the maximum is reported as a
   maximum. Any single number should carry its layer.
5. **Big Five recovery on the activation Gram is not a finding.** The
   constitutions are text *about* the traits; that a signed factor structure
   comes back is expected. `ACTSPACE_RESULTS.md` and the blog both say so
   explicitly.
6. **TEST 6 was never run on the activation Gram.** The training-strength
   nuisance covariate is `unavailable` for want of
   `results/runmeta_actspace.json`
   (`results/decomposition_actspace_resp.json#test6.status`;
   `phase10_runs/decompose_actspace_resp.log`: "The confound is then UNTESTED,
   which is not the same as ruled out.")
7. **The hole result is a negative on one specific coefficient vector.** It shows
   that `alien_k5`'s coefficients, applied to persona vectors, land at the
   permuted-coefficient null. It does not show that activation space has no holes,
   nor does it retract the weight-space measurement. See [[hole-words]].
8. **No behaviour was measured anywhere in this arm.** No judged evaluation, no
   steering sweep, no held-out task. "The adapter delivers 62% of the prompt's
   shift" is a statement about vectors, not about what the model says. The
   behavioural results live under the behaviour pages.
9. **The cross experiment covers 16 traits and one layer.** Its class sizes are
   16 / 14 / 22 / 204; the conflict result rests on 22 pairs.
10. **The response window depends on what the model said.** That is exactly why
    the user-turn window exists as a control -- and the control is much weaker
    (P~W 0.385 at layer 16), which is itself a limit on how much of the
    correspondence is about representation rather than about generated text.
11. **64 questions from one pool.** All 64 come from the zoo's shared prompt
    pool; nothing tests whether the geometry survives a different question
    distribution.
