---
title: Module holography
summary: Each of the 248 targeted modules was tested alone; a single module's slice of the weight change classifies a trait's Big Five factor at 0.638 on average against 0.202 chance, where all 248 together reach 0.76.
status: current
sources:
  - qwen35/analysis/module_holography.json
  - qwen35/build_findings_page.py
last_verified: 2026-09-16
tags: [geometry, modules]
---

# Module holography

## The test

Each targeted module was tested on its own: can that module's slice of the weight
change, compressed to 32 dimensions, say which Big Five factor a trait belongs
to? `qwen35/build_findings_page.py`, finding 1:

> Personality is holographic. One module out of 248 knows almost as much as all of
> them.

## The numbers

`qwen35/analysis/module_holography.json`, top-level keys:

| key | value |
|---|---|
| `#n` | 100 |
| `#n_modules` | 248 |
| `#full` | 0.76 |
| `#mean` | 0.638 |
| `#lo` | 0.43 |
| `#hi` | 0.74 |
| `#null` | 0.202 |

So all 248 modules together reach 0.76 on a five-way factor classification; a
single module averages 0.638 and ranges 0.43 to 0.74; chance is 0.202. Adding 247
more modules buys about 0.12. The trait is not stored in one place; it is smeared
across all of them, redundantly.

`#modules` is a 248-entry list, each with `i`, `mod` (the full PEFT module path),
`acc`, `norm`, `cls` and `layer`. Module classes present: `linear-attn`, `MLP`,
`full-attn`. The best single modules stored are
`base_model.model.model.layers.15.mlp.down_proj` and
`base_model.model.model.layers.15.self_attn.o_proj`, both at `acc = 0.74`; the
weakest is `base_model.model.model.layers.6.linear_attn.in_proj_a` at
`acc = 0.43`.

`build_findings_page.py`'s reading of the per-class and per-depth pattern: "MLP
modules edge out attention, and later layers edge out earlier ones, but every
class and every depth carries the signal."

## Provenance

**No producing script for `module_holography.json` exists in this repo.** The
file's own keys and `qwen35/build_findings_page.py` (which renders it as
`D.holo` and `D.holo_modules` via `qwen35/analysis/distil_data.json`) are the
whole record. The 0.202 chance figure matches the shuffled-null value used
elsewhere for five-way factor classification
(`qwen35/analysis/umap_test.json#factor(5-way).shuffled_null = 0.195`,
`#chance = 0.2`), and the 32-dimensional compression is the `k = 32` sketch, so
the method is legible from context; the code is not on disk. The file was written
2026-08-29 and `findings_page` is one of the older built pages, so treat the
framing as historical even though the numbers themselves are not contradicted
anywhere.

Related: [[polarity-and-bipolarity]], [[geometry-overview]],
[[umap-and-layouts]].
