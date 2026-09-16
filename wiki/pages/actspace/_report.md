---
title: "Agent report: actspace section"
summary: Build report for pages/actspace/ - pages written, contradictions, superseded claims, gaps, weakly sourced numbers.
status: current
sources:
  - qwen35/ACTSPACE_RESULTS.md
  - qwen35/blog_page/index.html
  - .garden/journal/2026-09-05.md
last_verified: 2026-09-07
tags: [meta, report]
---

# Agent report: ACTIVATION-SPACE ANALYSES (2026-09-05 to 09-06)

## Pages written

| slug | status | one line |
|---|---|---|
| `actspace-overview` | current | the three runs, the shared design, the chat-template quirk, headline table, cost |
| `actspace-persona-vectors` | current | P vs W: Gram curve, perm p, NN 45/134, PCA, Procrustes, decompose on the activation Gram, factor cosines, hole transplant, user-turn control, noise floor |
| `actspace-adapters` | current | A vs P vs W: magnitude, direction, geometry, containment, prompt-window control, and the `_Lora` forward-hook hot-swap |
| `actspace-cross` | current (with a labelled superseded block) | 16x16, raw vs trait-specific, saturation, additivity, conflict, prompt-window additivity |
| `actspace-method-notes` | current | replication detail and an eleven-item list of what the arm does not establish |
| `prompting-versus-training` | current | synthesis for the blog, every number linked to its page |

Not created: no trait, factor, geometry, zoo or overview pages were touched. All
six files are inside `wiki/pages/actspace/`.

## Citation conventions used (please normalise if the wiki settles on another)

- JSON keys: `qwen35/analysis/actspace_geometry.json#windows.resp.primary.r_centred`;
  array indices as `#windows.resp.curve[16].r_centred`; class arrays with spaces
  and commas in the key quoted, e.g.
  `#resp_specific["same factor, OPPOSITE keying"]`, and an individual pair as
  `#resp_specific["same factor, OPPOSITE keying"][t=anxious, s=relaxed].cos_A_t`.
- **JSON scalars are quoted at 3 decimal places**, the precision the analysis
  scripts print, except where a page shows the fuller value in parentheses.
  Where the same number also appears in `ACTSPACE_RESULTS.md` or the blog, the
  prose figure is taken from there and the JSON key is given alongside.
- Code is cited by file and line range (`act_space.py` lines 118-128).

## Contradictions found

### 1. P~W at layer 16: **0.705 vs 0.737**. Both published; both current.

Same quantity, two centrings of the weight side.

- `analyse_actspace.py` lines 92-94 build `Cw` from the **raw** weight Gram and
  correlate it with trait-centred activations (line 127) -> **0.705**
  (`analysis/actspace_geometry.json#windows.resp.primary.r_centred`). Quoted by
  `ACTSPACE_RESULTS.md` and rendered on the blog as "r = 0.70".
- `analyse_actspace_adapters.py` lines 62-63 **double-centre** the weight Gram
  (`Gc = J @ Gw @ J`) before line 84 -> **0.737**
  (`analysis/actspace_adapters_geometry.json#windows.resp.curve[16].r_PW`).
  Quoted by `ACTSPACE_RESULTS.md` in the adapter section and rendered on the blog
  as "the prompts track the weights (0.74)".

Recorded on both `actspace-persona-vectors` and `actspace-adapters`, and in
`actspace-method-notes`. **Neither supersedes the other.** A blog reader who sees
0.70 and 0.74 within two paragraphs will trip on it; the post may want one
sentence saying which centring each uses.

### 2. "Adapter alone" along the prompt's direction: **0.62 vs 0.70**.

- **0.62** = raw, `analysis/actspace_adapters_geometry.json#windows.resp.curve[16].frac_internalised`
  (0.620).
- **0.70** = trait-specific, `analysis/actspace_cross_geometry.json#resp_specific_adapter_alone`
  (0.7044), which is the figure the blog's cross paragraph uses as "the adapter
  alone 0.70".

Same construction before and after the shared shift is removed. Cross-linked on
`actspace-adapters` and `actspace-cross`.

**Stale docstring:** `analyse_actspace_cross.py` line 12 says "Prompt alone is
1.00 by construction, adapter alone was 0.62" -- but that script's own
trait-specific block prints 0.70. Source-code comment only; nothing downstream
uses it. Worth a one-line fix if anyone touches the file.

### 3. Prompt-window additivity: the MD and the blog quote different slices.

- `ACTSPACE_RESULTS.md`: "User-turn window (no generation yet): the two effects
  add almost exactly (residual 0.12 raw / 0.28 specific, weights 0.97 / 0.99)."
- Blog (current per wiki rule 2): "the two effects simply add (residual 33%,
  weights 0.96 and 0.92)", built from
  `analysis/actspace_cross_geometry.json#prompt_specific["different factor"]`.

Both are consistent with the JSON; the MD sentence mixes **raw** residual (0.12),
**trait-specific** residual (0.28) and **raw** least-squares weights (0.97/0.99)
in one clause, without saying so. Recorded on `actspace-cross` with a table
giving both, blog marked current. The task brief's "prompt-window additive
0.97/0.99" is the raw pair. **Not a data conflict; a provenance ambiguity in the
prose.** Recommend the blog wording.

### 4. The hole's weight-space comparison figure: **68.9 vs 52.5 degrees**.

`analyse_actspace.py` line 177 prints the activation result beside "weight space
68.9". Per `analyse_hole.py` line 5, 68.9 is the hole measured in the **full
sketch space**, while the coefficients actually transplanted are `alien_k5`'s,
whose weight-space angle in the **k=5 PC subspace** is
`analysis/alien.json#alien_k5.gap_deg` = **52.5**. The transplant conclusion
stands on its own permuted-coefficient null and does not depend on which figure
sits beside it, but the two are not the same construction. Recorded on
`actspace-persona-vectors`. This belongs to the [[hole-words]] agent's territory
as much as to mine.

### 5. Containment null "0-1%" / "about 1%" vs 1.5%.

`ACTSPACE_RESULTS.md` says "random subspaces: 0-1%"; the blog says "about 1%".
`analysis/actspace_adapters_geometry.json#windows.resp.containment_null` gives
0.0019 / 0.0038 / 0.0072 / **0.0148** for k = 5/10/20/40. The top-40 figure is
1.5%. Mildly overstated in both; the observed 41%/64% is unaffected. Recorded on
`actspace-adapters`.

Related, and more of a method point than a contradiction: the containment null is
a **single** random draw (`analyse_actspace_adapters.py` lines 106-109), not a
distribution, so it is one sample rather than a percentile. Recorded in
`actspace-method-notes`.

## Superseded claims (for `pages/overview/superseded-claims.md`)

**"Prompt wins 16/22" -> "8/22".**
The first cross analysis, run on raw (uncentred) vectors, found that in the 22
opposite-keying conflict pairs the combined model was nearer the **prompt's**
persona than the adapter's in **16 of 22** cases, i.e. that a system prompt
largely overrides a trained trait. Source: `.garden/journal/2026-09-05.md`,
verbatim: *"First analysis on raw vectors said 'prompt wins 16/22' -- an artefact
of the shared component (all pairwise cosines ~0.8)."*
Superseded by the trait-specific analysis, which gives **8 of 22** and reverses
the conclusion to "neither wins"
(`ACTSPACE_RESULTS.md`; blog; `analysis/actspace_cross_geometry.json#resp_specific`).
The blog paragraph is built from `resp_specific`
(`build_blog_page.py`: `X = J["resp_specific"]`). Recorded on `actspace-cross`
inside a labelled superseded block; the page as a whole is `current`.
Lesson filed by the project on the same day: *any comparison of steered
activations must remove the common shift first or it measures "was the model
steered" not "toward what".*

**Adjacent, and owned by another section:** the blog's hole section now says the
earlier phrasing "a character the lexicon lacks a word for" should be read as "a
direction the *adapters* leave open", on the strength of the activation-space
transplant. That is a partial retraction of a weight-space claim caused by this
arm; whoever owns [[hole-words]] should carry it.

## Gaps

1. **The cross analysis stdout was never saved.** `analyse_actspace_cross.py`
   prints full per-class median tables; no log file in `phase10_runs/` contains
   them (`actspace_cross.log` is the Modal collection log only). The class
   medians therefore exist in only two places: `ACTSPACE_RESULTS.md` and the
   rendered blog page. Every class median on `actspace-cross` is quoted from one
   of those two. Rerunning the script would regenerate them, but rule 4 forbids
   reruns, so this is a genuine gap: **the `same factor, same keying` class
   (n = 14) has no published medians at all.**
2. Same for `analyse_actspace.py` and `analyse_actspace_adapters.py` stdout --
   though those two write near-complete JSON, so little is lost.
3. **`results/runmeta_actspace.json` does not exist**, so decompose TEST 6 (the
   training-strength nuisance covariate) is `unavailable` for the activation
   Gram. The log says it plainly: "The confound is then UNTESTED, which is not
   the same as ruled out."
4. **Stage-2 adapters were never run in activation space.** All three runs use
   stage-1 only.
5. **No behavioural measurement anywhere in the arm.** 4 responses per trait (2
   per cross condition) were kept for eyeballing; none judged.
6. `analysis/actspace_geometry.json` stores `procrustes_null95` but not the null
   **mean**; the "0.02" on the blog and in the MD is printed only.
7. `analysis/actspace_geometry.json` stores `hole_null_median` but not the hole
   null's 95th percentile, which `analyse_actspace.py` line 177 prints.
8. `.npz` outputs (`actspace_means*.npz`, `results/gram_actspace_*_L16.npz`)
   were not opened -- no numpy in the system Python on this box, and rule 4
   forbids regenerating anything. Nothing on these pages depends on them beyond
   what the scripts and JSONs state.
9. `analysis/actspace_spec.json` is overwritten by **every** stage run and its
   `traits` field is always the 134-name list, never `CROSS_TRAITS`. Harmless
   (the prompt draw is seeded and identical), but it means the file cannot tell
   you which stage last ran except by mtime.
10. **Blog/MD asymmetry:** the blog's "Prompting versus training" section does
    **not** contain the hole-transplant paragraph. That paragraph lives in the
    *hole* section instead (`build_blog_page.py` lines 1317-1325) and rounds
    47.8/47.7 to "48 / median 48". The journal entry for 2026-09-05 says "Page
    gained 'Prompting versus training' (activation section + layer-curve figure),
    the hole-transplant paragraph" -- which reads as if both went into the same
    section. They did not. Noted so nobody hunts for it.

## Weakly sourced numbers

Numbers used on the pages whose provenance is thinner than a JSON key:

| number | where used | provenance |
|---|---|---|
| Procrustes null **mean 0.02** | `actspace-persona-vectors`, `prompting-versus-training` | printed by `analyse_actspace.py` line 151; in `ACTSPACE_RESULTS.md` and on the blog; **not** in the JSON |
| cross-run class **medians** (1.18, 0.62, 0.70, 0.68, 0.62, 0.50, 0.52, 33%, 0.96, 0.92, and the 8/22 count) | `actspace-cross`, `prompting-versus-training` | `ACTSPACE_RESULTS.md` and the rendered blog page only; the JSON holds the per-pair rows they are medians of, but the medians themselves were never written to a file |
| "raw pairwise cosines ~ +0.8" | `actspace-cross`, `prompting-versus-training` | prose in `ACTSPACE_RESULTS.md` and the blog; no stored scalar |
| **445** prompts in the pool | `actspace-method-notes` | line count of `qwen35/data_common/bold.jsonl`, checked to be 445 distinct prompts; **counted by this session**, not stated in any source |
| "$3 total" / "~$3.5" | `actspace-overview` | `ACTSPACE_RESULTS.md` prose; the logs give seconds, not money |
| "cost a forward pass rather than a training run" | `prompting-versus-training` | blog prose |

## Numbers computed by this session, NOT placed on any page

Used only to diagnose the contradictions above. The maintainer may want them, but
they are recomputations and rule 1 keeps them off the pages. All are
`statistics.median` over the per-pair rows in
`analysis/actspace_cross_geometry.json`, layer 16.

- Confirming the superseded raw reading: `resp["same factor, OPPOSITE keying"]`
  prompt-wins count = 16/22; `resp_specific` same class = 8/22.
- Confirming the MD's mixed prompt-window sentence:
  `prompt["different factor"]` `resid_add` 0.1228, `a` 0.9738, `b` 0.9863;
  `prompt_specific["matched"]` `resid_add` 0.2794;
  `prompt_specific["different factor"]` `resid_add` 0.3260, `a` 0.9562,
  `b` 0.9188.
- The `same factor, same keying` class (n = 14), which nothing publishes:
  `resp_specific` `along_P_t` 0.721, `resid_add` 0.6457, `a` 0.6583, `b` 0.6730,
  prompt-wins 5/14.
- Reproduction check of the published cross figures:
  `resp_specific["matched"]` `along_P_t` 1.178 (published 1.18),
  `resid_add` 0.6177 (published 0.62);
  `different factor` `resid_add` 0.6970 (published 0.70), `a` 0.6768 (0.68),
  `b` 0.6223 (0.62). All published cross numbers reproduce.

Every headline number in `ACTSPACE_RESULTS.md` and in the blog's "Prompting
versus training" section was checked against its JSON key or log line and
**reproduces**, with the five caveats listed under Contradictions.

## Date range

The brief covers 2026-09-05 to 09-06. `.garden/journal/` in the persona-curvature
repo contains `2026-09-01.md`, `2026-09-03.md`, `2026-09-04.md` and
`2026-09-05.md` only -- **there is no 09-06 or 09-07 journal**. Everything the
journal says about this arm is in the 2026-09-05 entry, which also records the
cross rerun on trait-specific parts and the blog rebuild. The analysis JSONs and
the built page carry 2026-09-05 timestamps (18:42-21:24);
`blog_page/index.html` is the 2026-09-07 authority per wiki rule 2.

## Small things a maintainer may want to fix elsewhere

- `analyse_actspace_cross.py` docstring line 12: "adapter alone was 0.62" ->
  0.70 for the trait-specific figure it actually prints.
- `analyse_actspace.py` line 177 hardcodes "weight space 68.9"; consider printing
  `alien_k5.gap_deg` (52.5) instead, or naming which space 68.9 is measured in.
- `analysis/actspace_geometry.json#windows.prompt.curve[0]` is all `nan`. Benign
  (no system prompt means the user tokens are the baseline's user tokens exactly,
  so the shift is zero and its cosines undefined) but it makes
  `max(curve, key=r_centred)` return layer 0 with `nan` for that window; the blog
  only uses the `resp` window, so nothing is affected.
