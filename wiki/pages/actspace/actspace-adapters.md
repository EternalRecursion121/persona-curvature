---
title: What the adapters do to activations
summary: Each stage-1 adapter, run with no system prompt, moves the residual stream as far as the trait's constitution does as a prompt (median ratio 1.01) and 62% of the way along the prompt's own direction, but the adapters' activation geometry tracks their weight geometry (0.865) more closely than it tracks the prompts (0.781).
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/act_space.py
  - qwen35/analyse_actspace_adapters.py
  - qwen35/analysis/actspace_adapters_geometry.json#windows.resp
  - qwen35/analysis/actspace_adapters_geometry.json#windows.prompt
  - qwen35/phase10_runs/actspace_adapters.log
  - qwen35/blog_page/index.html
last_verified: 2026-09-16
tags: [actspace, activations, adapters, lora, geometry]
---

# What the adapters do to activations

Three shifts per trait, all measured against the **same** no-system-prompt
baseline, over the **same** 64 questions ([[actspace-overview]]):

- **P_t** -- the constitution as a system prompt, base weights. See
  [[actspace-persona-vectors]].
- **A_t** -- the trait's **stage-1 adapter loaded, no system prompt at all**.
- **W_t** -- the adapter's weight delta, from the exact Gram in
  `results/gram_sweep.npz`. See [[geometry-overview]].

The question is whether training on a constitution's data reproduces, in
activation space, what that constitution does as a prompt
(`analyse_actspace_adapters.py` lines 1-3). All numbers below are at layer 16,
response-token window, unless stated.

## Magnitude: the adapter moves as far as the prompt

Median **|A_t| / |P_t| = 1.01**
(`analysis/actspace_adapters_geometry.json#windows.resp.curve[16].mag_ratio`
1.013). `ACTSPACE_RESULTS.md`: "Training on the constitution's data moves the
residual stream as far as the constitution does as a prompt." The ratio stays
near 1 across the whole depth of the model (1.155 at layer 0, 1.164 at layer 32).

## Direction: 62% of the prompt's shift, and a big shared component

| quantity | value | key |
|---|---|---|
| cos(P_t, A_t), own trait, mean | **+0.601** | `#windows.resp.curve[16].cos_own` |
| cos(P_t, A_s), other traits, mean | **+0.337** | `.cos_other` |
| own trait ranks 1 of 134 | **43** traits | `.rank1` |
| mean rank of the own trait | **4.97** | `.mean_rank` |
| median fraction of the prompt shift delivered along the prompt's direction | **0.62** | `.frac_internalised` (0.620) |
| own-trait rank 1 at layer 24 | **68** traits | `#windows.resp.curve[24].rank1` |

`ACTSPACE_RESULTS.md` reports these as "+0.60 vs +0.34", "median 62%", "rank 1 of
134 for 43 traits (mean rank 5.0); this rises to 68/134 at layer 24". The blog
uses the same figures.

The +0.34 across unrelated traits is the point that most needs saying out loud:
**every adapter and every prompt moves activations partly along one common
direction**. `ACTSPACE_RESULTS.md` calls it "a large shared component"; the blog
calls it "the activation-space shadow of the personality axis". It is why every
geometry number on these pages is computed on trait-centred vectors, and it is
what a first, uncentred pass at the cross experiment got wrong
([[actspace-cross]]).

**Note on "the adapter alone".** This page's 0.62 is the *raw* fraction. The
cross analysis reports the same construction on the **trait-specific** parts --
shared components removed -- and gets **0.70**
(`analysis/actspace_cross_geometry.json#resp_specific_adapter_alone`
0.7044). Both are current; they differ by whether the shared shift was removed
first. The docstring of `analyse_actspace_cross.py` (line 12) still says "adapter
alone was 0.62", which is stale relative to the number that script itself prints.

## Geometry: the adapters look like the adapters

Trait-centred 134x134 cosine matrices, correlated pairwise. Weight-side Gram
double-centred (`analyse_actspace_adapters.py` lines 62-63):

| pair | Pearson r | key |
|---|---|---|
| **A ~ W** (adapter activations vs adapter weights) | **+0.865** | `#windows.resp.curve[16].r_AW` |
| **A ~ P** (adapter activations vs prompt activations) | **+0.781** | `.r_AP` |
| **P ~ W** (prompt activations vs adapter weights) | **+0.737** | `.r_PW` |

`ACTSPACE_RESULTS.md`: "The adapter cloud in activation space mirrors the adapter
cloud in weight space more closely than either mirrors the prompt cloud." The
ordering holds at every layer from 0 to 32.

The `r_PW` value here (0.737) is **not** the 0.705 quoted as the headline on
[[actspace-persona-vectors]]; that one comes from `analyse_actspace.py`, which
does not centre the weight Gram. Same quantity, two centrings, both published.

The blog's layer-curve figure ("Three geometries, by layer") plots exactly these
three series from `#windows.resp.curve`, with layer 16 marked
(`build_blog_page.py`, `layer_curve_svg(G2['curve'])`).

## Containment: the adapter uses the prompts' directions

Fraction of the adapter shifts' variance that lies inside the top-k subspace
spanned by the prompt shifts (`analyse_actspace_adapters.py` lines 97-109):

| k | contained | random subspace of the same size |
|---|---|---|
| 5 | **0.413** | 0.002 |
| 10 | 0.514 | 0.004 |
| 20 | 0.579 | 0.007 |
| 40 | **0.644** | 0.015 |

(`#windows.resp.containment` and `#windows.resp.containment_null`.)
`ACTSPACE_RESULTS.md` reports "41% ... 64% ... (random subspaces: 0-1%)"; the
blog says "where random subspaces of the same size capture about 1%". The
top-40 null is 1.5%, slightly above both phrasings.

## The user-turn window: adapters barely touch the prompt tokens

With the adapter loaded and no system prompt, the user's own tokens are almost
unmoved, yet the little movement there is already carries the weight geometry:

| quantity | value | key |
|---|---|---|
| median &#124;A&#124;/&#124;P&#124; | 0.259 | `#windows.prompt.curve[16].mag_ratio` |
| median fraction of the prompt shift | 0.048 | `.frac_internalised` |
| cos own / other | +0.180 / +0.115 | `.cos_own`, `.cos_other` |
| A ~ W | **+0.917** | `.r_AW` |
| A ~ P | +0.415 | `.r_AP` |
| containment, top-5 / top-40 | 0.132 / 0.286 | `#windows.prompt.containment` |

`ACTSPACE_RESULTS.md`: "adapters barely touch prompt-token activations (|A|/|P|
0.26, 5% of the prompt shift) yet their cross-trait geometry there tracks the
weights at r 0.92." That is a striking pair: the adapter's *effect on the reading
of a question it has not answered yet* is tiny in magnitude but almost perfectly
ordered like the weights.

Layer 0 of the prompt window is `nan` (`#windows.prompt.curve[0]`) -- the
adapter arm has no system prompt, so at the embedding layer its user tokens are
the baseline's user tokens exactly and the shift is zero.

## Noise floor

Two halves of the 64 prompts, per adapter: median cosine **0.961** at layer 16
(`#windows.resp.curve[16].floor_A`), i.e. the adapter shifts are as reproducible
across prompt halves as the prompt shifts are.

## How the adapters were swapped: LoRA as a forward hook

The 134 adapters were **not** merged into the weights and the model was **not**
reloaded 134 times. `act_space.py` defines a small callable, `_Lora` (lines
182-196), registered once as a `register_forward_hook` on each targeted `Linear`:

```python
class _Lora(object):
    def __init__(self, name, state):
        self.name, self.state = name, state

    def __call__(self, mod, inp, out):
        cur = self.state.get("lora")
        if cur is None or self.name not in cur:
            return out
        A, B, scale = cur[self.name]
        x = inp[0]
        return out + torch.nn.functional.linear(
            torch.nn.functional.linear(x, A), B) * scale
```

Every hook reads from one shared `state` dict, so:

- **Swapping the adapter** is `state["lora"] = load(t)` -- replacing tensors in a
  dict (lines 238-245, 252). Nothing is re-instantiated.
- **Disarming** is `state["lora"] = None`: the hook returns `out` unchanged and
  the model is exactly the base model again (lines 257, 260-261). That is why the
  no-system baseline collected in the *prompt* run can be reused here rather than
  re-collected -- both come from the same code path.
- **Module resolution** copes with the `base_model.model.` prefix in the
  safetensors keys and with Qwen3.5's `model.language_model.` nesting via the
  `find()` fallback chain (lines 228-235).
- **Scale** is `lora_alpha / r`, or `lora_alpha / sqrt(r)` under rsLoRA (lines
  223-224). The run log records the resolved value:
  `[hooks] 248 modules, LoRA scale 2.0`
  (`qwen35/phase10_runs/actspace_adapters.log`).

The same machinery drives the cross run, where it matters more: 256 conditions
would otherwise mean 256 model loads ([[actspace-cross]]).

## What this does and does not say

It says fine-tuning on a constitution's data and prompting with that constitution
land in a similar place, at a similar distance, using much the same directions --
but that training leaves its own signature: the adapters resemble each other, in
activations, more than they resemble the prompts that generated their data. It
does **not** establish that the rest of the prompt's direction is
behaviourally irrelevant, nor that any of this holds for the stage-2 adapters,
which were not run. See [[actspace-method-notes]].
