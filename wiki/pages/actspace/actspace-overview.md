---
title: "Activation-space analyses: overview"
summary: Three runs on 2026-09-05 asked whether a constitution used as a system prompt and an adapter trained on that constitution move Qwen3.5-4B's residual stream the same way, and whether the 134 shifts are arranged the way the 134 weight updates are.
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/act_space.py
  - qwen35/analyse_actspace.py
  - qwen35/analyse_actspace_adapters.py
  - qwen35/analyse_actspace_cross.py
  - qwen35/analysis/actspace_spec.json
  - qwen35/phase10_runs/actspace.log
  - qwen35/phase10_runs/actspace_adapters.log
  - qwen35/phase10_runs/actspace_cross.log
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-07
tags: [actspace, activations, persona-vectors, design, qwen35]
---

# Activation-space analyses: overview

Everything else in this project measures a trait as a **weight update**: the LoRA
delta `dW` an adapter writes into the base model (see [[geometry-overview]]). On
2026-09-05 the project built a second, much cheaper object for the same 134
traits and asked whether the two agree.

The second object is a **persona vector** ([[glossary]]): run the base model with a trait's
constitution as the system prompt over a fixed set of questions, average the
residual-stream activation over its answers, and subtract the same average with
no system prompt. One vector per trait per layer, for the price of a forward
pass. The construction is the one in the persona-vectors literature
([[paper-persona-vectors]]); the traits and the constitutions are the zoo's own
([[constitution-generation]]).

Three runs, all on 2026-09-05, all on `Qwen/Qwen3.5-4B`:

| run | what varies | conditions | script stage | analysis |
|---|---|---|---|---|
| **prompt** | constitution as system prompt, base weights | 134 traits + 1 no-system baseline | `act_space.py` `collect` | [[actspace-persona-vectors]] |
| **adapters** | stage-1 adapter loaded, **no** system prompt | 134 adapters | `act_space.py` `collect_adapters` | [[actspace-adapters]] |
| **cross** | adapter *and* constitution together | 16 x 16 = 256 | `act_space.py` `collect_cross` | [[actspace-cross]] |

The synthesis written for the blog is [[prompting-versus-training]]. Everything a
replicator needs, and the list of what the three runs do not establish, is in
[[actspace-method-notes]].

## The design

Held constant across all three runs, so that the three sets of vectors are
directly comparable and share one baseline:

- **64 prompts, the same 64 every time.** `N_PROMPTS = 64`, drawn as
  `random.Random(7).sample(pool, 64)` where `pool` is the sorted set of distinct
  prompts in `qwen35/data_common/bold.jsonl`, the zoo's shared prompt pool
  (`act_space.py` lines 361-362; the sampled 64 are stored verbatim in
  `analysis/actspace_spec.json#prompts`). The same fixed pool is used for the
  same reason every adapter was trained on it: nothing about a trait's result
  can come from which questions it was asked.
- **Greedy decoding, `MAX_NEW = 96` new tokens, `BATCH = 32`**
  (`act_space.py` lines 34-36).
- **`enable_thinking=False` on every chat-template call** (`act_space.py` lines
  82-83, 90). Qwen3.5's template defaults its reasoning block on; the project had
  already silently invalidated three runs by omitting this flag.
- **33 hidden-state layers x 2560 dimensions** -- the embedding output plus the
  32 blocks (`[model] Qwen/Qwen3.5-4B: 33 hidden-state layers x 2560`,
  `qwen35/phase10_runs/actspace.log`).
- **Mean residual-stream activation** over a token window, per layer, minus the
  same mean from the no-system-prompt baseline (`act_space.py` lines 129-132,
  and lines 111 and 51-52 of the two analysis scripts).
- **Two token windows.** `resp` is the model's own greedy response tokens, from
  the end of the prompt up to and including the first EOS, and is **primary**.
  `prompt` is the user-turn tokens, from the end of the system block to the end
  of the user turn, and is the control: it is deterministic, free, and cannot
  depend on what the model happened to say (`act_space.py` lines 118-128).
- **Two halves.** Prompts 0-31 and 32-63 are stored separately, and their
  per-trait cosine is the activation-space noise floor -- the analogue of the
  weight-space seed floor (`act_space.py` lines 16-17, 161).
- **Layer 16 of 0..32 is primary, fixed in advance.** The full layer curve is
  reported and the maximum is reported *as a maximum*, not as the result
  (`analyse_actspace.py` lines 12-15, 26).
- **Trait-centring.** Every prompt shift and every adapter shift contains a large
  shared component. Cross-space correlations are therefore computed on
  trait-centred vectors, so that "was the model steered at all" cannot carry the
  number (`analyse_actspace.py` line 139; `analyse_actspace_cross.py` lines
  115-121). Failing to do this produced one wrong headline; see
  [[actspace-cross]].

## The chat-template quirk

Qwen3.5's chat template **refuses a message list with only a system turn**. The
first launch of the run died on it:

```
TemplateError: No user query found in messages.
```

(`qwen35/phase10_runs/actspace.log.1788630847`, tail; that log is 17:53,
the successful run at `actspace.log` is 18:41.)

So the length of the system block cannot be measured by rendering the system turn
alone. It is measured as a difference instead: render the full
system-plus-user prompt, render the same user turn with no system turn, and take
the token-count difference (`act_space.py` lines 86-94). That length is what lets
the `prompt` window start *after* the constitution rather than inside it. The
lesson is recorded in the journal for 2026-09-05: "Chat template refuses
system-only message lists -- measure the system block by rendering with and
without it."

## Headline numbers

All at layer 16, response-token window, trait-centred.

| result | value | source |
|---|---|---|
| prompt-activation geometry vs weight geometry, Pearson | +0.705 | `qwen35/ACTSPACE_RESULTS.md`; `analysis/actspace_geometry.json#windows.resp.primary.r_centred` |
| same, Spearman | +0.690 | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.primary.rho_centred` |
| label-shuffle p | 0.0005 | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.primary.perm_p` |
| maximum over layers | 0.775 at layer 19 | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.curve[19].r_centred` |
| nearest-neighbour agreement | 45 / 134 (chance 1) | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.primary.nn` |
| Procrustes R^2 of the 134x5 PC scores | 0.535 | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.primary.procrustes_r2` |
| activation noise floor (two halves, same trait) | 0.96 | `qwen35/ACTSPACE_RESULTS.md` (`#windows.resp.curve[16].floor` 0.959) |
| the weight-space hole, transplanted | 47.8 deg vs a 47.7 deg null | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.primary.hole_deg`, `.hole_null_median` |
| adapter shift vs prompt shift, magnitude | median 1.01 | `qwen35/ACTSPACE_RESULTS.md`; `analysis/actspace_adapters_geometry.json#windows.resp.curve[16].mag_ratio` |
| adapter shift vs prompt shift, same-trait cosine | +0.60 (other traits +0.34) | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.curve[16].cos_own`, `.cos_other` |
| adapter activations ~ adapter weights | +0.865 | `qwen35/ACTSPACE_RESULTS.md`; `#windows.resp.curve[16].r_AW` |
| matched adapter + matched prompt, along the trait | 1.18 (prompt alone 1.00, adapter alone 0.70, additive 1.70) | `qwen35/ACTSPACE_RESULTS.md`; blog "Prompting versus training" |
| adapter vs prompt in conflict | 0.50 / 0.52 | `qwen35/ACTSPACE_RESULTS.md`; blog |

## Cost

Three A100-40GB containers: 2838 s (47 min) for the prompt run, 2199 s (37 min)
for the adapters run, 5398 s (90 min) for the cross run, from the JSON summary
line at the end of each of `qwen35/phase10_runs/actspace.log`,
`actspace_adapters.log` and `actspace_cross.log`. `ACTSPACE_RESULTS.md` gives the
money as "about $3 total" for the first two and "~$3.5" for the cross run.

## What this section is for

The weight-space results ([[geometry-overview]]) are a geometry of one training
procedure on one model, and the obvious objection is that they are an artefact of
LoRA, of AdamW, or of the initialisation. The activation-space arm is the check:
the constitutions are the *same documents*, but nothing is trained. That the two
geometries agree at all is the result. Where they disagree -- the hole
([[hole-words]]) does not transfer -- is equally informative, and is the sharpest
thing this section says.
