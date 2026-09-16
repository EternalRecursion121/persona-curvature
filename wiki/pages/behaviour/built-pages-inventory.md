---
title: Inventory of built HTML pages
summary: Every built HTML page in qwen35/ and sweep100/site2, what it shows, which script builds it, whether it is served, and its status — two are current, one is withdrawn, the rest are historical.
status: current
sources:
  - qwen35/blog_page/index.html
  - qwen35/figures/post/make_post_figures.py
  - qwen35/companion/assets/planes.js
  - qwen35/build_blog_page.py
  - qwen35/findings_page/index.html
  - qwen35/build_findings_page.py
  - qwen35/monitor_page/index.html
  - qwen35/build_monitor_page.py
  - qwen35/distil_page/index.html
  - qwen35/direction_pages/
  - qwen35/live_page/index.html
  - qwen35/build_live_page.py
  - qwen35/spider_page/index.html
  - qwen35/build_spider_page.py
  - qwen35/zoo_page/index.html
  - qwen35/build_zoo_page.py
  - qwen35/manifold_page/index.html
  - qwen35/site/index.html
  - qwen35/build_site.py
  - qwen35/site_traits/index.html
  - qwen35/build_traits_page_data.py
  - qwen35/companion/build_companion.py
  - qwen35/companion/DESIGN_v2.md
  - qwen35/companion/CRITIQUE_v2.md
  - qwen35/companion/fetch_stage2_excerpts.py
  - /var/www/persona-site/index.html
  - /etc/systemd/system/persona-companion-build.timer
  - sweep100/site2/index.html
  - /etc/systemd/system/persona-cartography.service
  - /etc/systemd/system/qwen35-plan.service
  - /etc/systemd/system/qwen35-traits.service
last_verified: 2026-09-14
tags: [behaviour, pages, inventory]
---

# Inventory of built HTML pages

Four of these are served over the public web; the rest exist only as files on
the box. Sizes and dates are as of 2026-09-07, except the companion site, checked
2026-09-14.

## Served

Three systemd units run `python3 -m http.server` bound to 127.0.0.1, and Caddy
proxies a hostname to each (`/etc/caddy/Caddyfile`). All three were `active`
when checked on 2026-09-07.

| unit | port | directory | URL |
|---|---|---|---|
| `persona-cartography.service` | 8091 | `sweep100/site2` | https://persona-cartography.161-35-77-84.sslip.io |
| `qwen35-plan.service` | 8092 | `qwen35/site` | https://plan.161-35-77-84.sslip.io |
| `qwen35-traits.service` | 8093 | `qwen35/site_traits` | https://traits.161-35-77-84.sslip.io |

The companion site is served differently: no port, no http.server. Caddy has
`/var/www/persona-site` as a static file root for
https://persona.161-35-77-84.sslip.io, and `persona-companion-build.timer` runs
the builder every minute, which exits without writing anything unless one of its
registered inputs changed.

**The current blog page is not among them.** `qwen35/blog_page/index.html` has no
service and no Caddy route.

## The pages

### `qwen35/blog_page/index.html` — "Investigating LLM Personality in Weight Space"

- **Status: current.** This is the authoritative statement of the zoo results
  (wiki rule 2). 5,067,541 bytes, rebuilt 2026-09-05 21:46.
- **Builder:** `qwen35/build_blog_page.py` (101 KB), reading
  `analysis/blog_data.json` and `analysis/blog_corpus.json` plus, directly,
  `pc_loadings.json`, `results/fa_qwen35.json`, `scree_null_matched.json`,
  `sphere_page.json`, `sphere_layout.json`, `optimise.json`, `alien_steer.json`,
  `alien_match.json`, `direction_gaps.json`, `verify.json`, `trait_angles.json`,
  `align_summary.json`, `align_validate.json`, `actspace_geometry.json`,
  `actspace_adapters_geometry.json`, `actspace_cross_geometry.json`.
- **Shows:** an interactive direction explorer over the corrected steering
  corpus; a rotatable PCA map of the 134 adapters; the six principal components;
  the sphere sweep; the five factor-analytic factors; the grand-mean personality
  axis; the unnamed direction; the data-scoring identity and the N x N result;
  the optimised-data search and its training check; the activation-space work;
  and a closing "What this does and does not establish".
- **Not served.**

### `qwen35/findings_page/index.html` — "What the Weights Know"

- **Status: historical.** 52,547 bytes, built 2026-09-01 10:26.
- **Builder:** `qwen35/build_findings_page.py`, from `analysis/page_data.json`
  and `analysis/distil_data.json`.
- **Shows:** numbered findings, each with its null or ceiling — module
  holography, the variance paradox, the isometry result, the cross-seed
  attenuation, the judged-distance ceiling — plus the claims withdrawn along the
  way. Its docstring: "Supersedes distil_page/, which carries a claim since shown
  to be wrong."
- **Not served.**

### `qwen35/monitor_page/index.html` — "Personality Does Not Travel"

- **Status: historical.** 38,074 bytes, built 2026-09-01 13:40.
- **Builder:** `qwen35/build_monitor_page.py`, from
  `analysis/monitor_page_data.json`.
- **Shows:** six pre-registered tests of whether personality can be monitored
  from weights, each with the success criterion written before the test ran.
  Tests 1 and 2 pass; tests 3-6 fail. It is the home of the additivity verdict
  ([[additivity]]), the reward-hacks verdict ([[reward-hacks-arms]]) and the
  functional-probe result ([[judged-evaluations]]).
- **Not served.**

### `qwen35/distil_page/index.html` — "Withdrawn Findings Note"

- **Status: withdrawn.** 4,928 bytes, replaced 2026-09-01 13:53. The withdrawn
  content is preserved at `index.html.withdrawn-backup` (64,970 bytes, title
  "Where Personality Lives"), and the template at `template.html`.
- **Builder:** none checked in; its data file is `analysis/distil_data.json`.
- Full account on [[distillation-check]].
- **Not served.**

### `qwen35/direction_pages/*.html` — 22 per-direction pages

- **Status: historical.** Nine built 2026-08-30 00:02, thirteen 2026-09-01
  12:08; 14-19 KB each.
- **Builder:** `qwen35/build_direction_pages.py`.
- **Shows:** one page per steered direction — the dose sweep as a vertical alpha
  ladder with verbatim generations, per-rung damage bars, and degenerate rungs
  struck through. Titles are the direction's qualitative label, not its slug.
- Full list and provenance on [[steering-results]].
- **Not served.**

### `qwen35/live_page/index.html` — "Persona Weight Space"

- **Status: historical.** 374,634 bytes, built 2026-08-29 15:36, from
  `live_page/template.html`.
- **Builder:** `qwen35/build_live_page.py`, from `analysis/live_components.json`
  plus `fa_summary.json`, `umap_test.json`, `live_results.json`,
  `trait_graph.json`, and a live systemd/spend status block.
- **Shows:** the component structure as it stood mid-run, with each source marked
  partial if its factor coverage is unbalanced, "because a factor-unbalanced
  sample produced a confidently wrong PC1 earlier in this project and must never
  render as a result". Carries the two falsified hypotheses of
  `analysis/live_results.json#hyp` — the second of which the project later
  corrected; see [[judged-evaluations]].
- **Not served.** Its status block queries units (`zoo-eval`, `zoo-lex`,
  `zoo-steer`, `zoo-sketch-persona`) that have since finished, so re-running the
  builder today would not reproduce the page as shipped.

### `qwen35/spider_page/index.html` — "Do the Dials Turn One Thing"

- **Status: current.** 15,603 bytes, built 2026-08-30 21:31; since 2026-09-08 `analysis/spider.json` is regenerated from primary files by `qwen35/build_spider_data.py` (with a persona arm) and the wiki carries the extended version at [[ocean-dials-replication]].
- **Builder:** `qwen35/build_spider_page.py`, from `analysis/spider.json`.
- **Shows:** a replication of the OCEAN dial spider plots on a different model
  and a different construction of the dials, done twice over — once with the
  five steering axes at alpha +/-2, once with the judged profiles of each
  factor's ten positively- and ten negatively-keyed trait adapters. Baselines
  `base_steer` and `base_trait` are stored separately in `analysis/spider.json`
  because the two arms have different alpha=0 references.
- **Served** as a static copy at https://wiki.161-35-77-84.sslip.io/spider.html.

### `qwen35/zoo_page/index.html` — "Persona Zoo Readout"

- **Status: historical.** 44,511 bytes, built 2026-08-28 00:41, from
  `zoo_page/template.html`. `MODEL_CARD.md` and `DATASET_CARD.md` sit beside it.
- **Builder:** `qwen35/build_zoo_page.py`, reading the live Modal volume and the
  run log; `TOTAL = 51` — it is a readout of the 51-adapter stage, not the final
  134.
- **Shows:** training telemetry per adapter, recomputing "the honest final loss
  from log_history rather than trusting runmeta's `loss_last`, which HF Trainer
  under-reports on any resumed run".
- **Not served.** Superseded by the 134-trait page below.

### `qwen35/manifold_page/index.html` — "The Folded Persona"

- **Status: historical, and speculative by its own framing.** 33,166 bytes,
  built 2026-08-29 18:01.
- **Builder:** none checked in. Its companion material is
  `analysis/manifold_ideas.json` and `analysis/manifold_ideas.md`, written the
  same minute.
- **Shows:** candidate reasons the Euclidean sketch picture might be wrong, and
  proposed tests. `manifold_ideas.md` is where the STEER134 coherence-collapse
  argument lives (see [[steering-results]]); it labels its own proposals P1, P2,
  ... rather than presenting them as results.
- **Not served.**

### `qwen35/site/index.html` — "Persona geometry in Qwen3.5-4B — experiment plan"

- **Status: historical.** 80,937 bytes, built 2026-08-20 07:58.
- **Builder:** `qwen35/build_site.py` from `plan.json`: "Every number and every
  prose string on the page comes from plan.json; nothing is retyped here."
- **Shows:** the pre-registered experiment plan, written before most of the work
  described in this wiki was done. Useful as a record of intent; not a results
  page.
- **Served** at https://plan.161-35-77-84.sslip.io via `qwen35-plan.service` on
  port 8092.

### `qwen35/site_traits/index.html` — "134 trait adapters in weight space"

- **Status: historical.** 51,813 bytes, built 2026-08-24 10:15, with
  `data.json`, `analysis.json` and `oct.html` beside it.
- **Builder:** `qwen35/build_traits_page_data.py` assembles `data.json` from
  `results/gram_sweep.npz`, the three null Grams, the seed-paired cross Gram,
  the decomposition JSONs, `runmeta_sweep.json`, the trait label files,
  `trait_descriptions.json`, and — optionally —
  `results/steer134_judged.json`.
- **Shows:** the 134-adapter geometry with its nulls, per-trait pages and
  neighbour rankings. **This is the only page in the repository that consumes
  the STEER134 judging**, and it predates most of the geometry corrections, so
  its numbers should be checked against the geometry pages before use.
- **Served** at https://traits.161-35-77-84.sslip.io via
  `qwen35-traits.service` on port 8093.

### `/var/www/persona-site` — "Personality in weight space" (the companion site)

- **Status: current.** 191 files, rebuilt 2026-09-14. Seven top-level pages
  (home, chart, behaviour, stage two, traits, data, methods) plus five methods
  sub-pages, 141 trait pages and a downloads directory.
- **Builder:** `qwen35/companion/build_companion.py` (168 KB) with
  `companion/assets/*` and `companion/content/*.md`, run by
  `persona-companion-build.timer`. Nothing on it is hand-written: every number is
  read from a file at build time and every section names the files it came from.
  `companion/check_numbers.py` re-derives 262 of those numbers from the source
  files and asserts each one is present in the rendered HTML.
- **Shows:** the same results as the blog page, arranged to be navigated rather
  than read through — the factor chart, the loadings and neighbour explorers, the
  steering dose-response, the stage-two analysis, and one page per adapter with
  its constitution, its training pairs, its loadings, its neighbours in all three
  Grams and its judged behaviour.
- **Redesigned 2026-09-14** at Samuel's request, "in the style of the interactive
  blog with the 3d diagrams instead of 2d diagrams". It now shares the blog
  page's identity — Literata, Fraunces, IBM Plex Mono, the same six data hues,
  one per Big Five scale, with each recovered factor taking the hue of the scale
  its best Tucker congruence lands on — and every 2D scatter it carried is gone.
  `companion/assets/map3d.js` is a port of the blog's canvas map
  (`blog_page/_js.txt`, `makeMap`), parameterised by data URL, and it draws the
  factor chart on the home and chart pages, the pinned chart on all 141 trait
  pages (replacing the small static SVG each carried) and a new per-stage cloud
  on the stage-two page. Both `if (false)` markers are ported disabled. The site
  gained a system / light / dark theme with a header override, and every trait
  page gained three preference pairs from its own stage-one corpus and three
  judged answers with the judge's per-answer scores. The plan is
  `companion/DESIGN_v2.md`; the self-critique is `companion/CRITIQUE_v2.md`.
- Each of the 134 zoo trait pages also carries one stage-two introspection
  excerpt, cached from the public dataset
  `EternalRecursion/persona-curvature-oct-transcripts` by
  `companion/fetch_stage2_excerpts.py`, which range-reads the head of each
  `self_reflection/<trait>.jsonl` rather than downloading any 60 MB file. The
  seven later adapters were never trained past stage one and say so.
- **One thing is deliberately absent.** The three stage clouds are offered by a
  selector and never overlaid: there is no cross-Gram between the stages over all
  134, so each stage is charted in its own Gram by the identical construction and
  a line between a trait's stage-one and stage-two point would be a movement the
  data does not contain.
- **Served** at https://persona.161-35-77-84.sslip.io as a Caddy static root.

### `sweep100/site2/index.html` — "Principal directions of 100 personality LoRAs"

- **Status: historical.** 9,810 bytes, built 2026-08-15 22:16. Companion data:
  `analysis.json`, `components_data.json`, `traits_data.json`,
  `summaries.json`, `trait_descriptions.json`.
- **Builder:** `sweep100/site2/build.py` and `build_components.py`:
  "Quantitative comes from the completed PCA / FA / Gram; qualitative comes from
  the DPO pairs the adapter was actually trained on."
- **Shows:** the pre-zoo 100-LoRA sweep on the earlier model. It is the ancestor
  of the qwen35 work, not a version of it. See the history section.
- **Served** at https://persona-cartography.161-35-77-84.sslip.io via
  `persona-cartography.service` on port 8091.

### `qwen35/figures/post/` — the static figures of the LessWrong draft

- **Status: current.** Eleven figures, each as PNG and SVG in light and dark themes, drawn on 2026-09-15 by `qwen35/figures/post/make_post_figures.py` from the analysis files (it imports the theme, density shells and label placer from `qwen35/figures/clusters/make_cluster_figures.py`). Sized for a 700 px column: 8 in wide at 200 dpi. Names: `facets_own` (each Goldberg group on Warmth against its own factor), `scree_congruence`, `seed_metrics`, `rank_sweep`, `fisher_spread`, `stage_two`, `dose_response`, `sphere`, `dials`, `forecast_matrix`, `external_data`. Alongside them, six headless-Chromium screenshots of the interactive pages, cropped by hand and not regenerated by the script: `screen_chart`, `screen_planes`, `screen_trait`, `screen_stage2`, `screen_behaviour` (companion pages) and `screen_sphere` (the sphere widget of the interactive write-up, `qwen35/blog_page/index.html#sphere`, which is not yet on the companion). The draft embeds five of them beside links to the pages they show.
- **Served** by the companion builder at `https://persona.161-35-77-84.sslip.io/figures/post/<name>_<light|dark>.<png|svg>`; the draft ([[post-draft]]) embeds the light PNGs by that URL, which moves with the site.
- **Sources per figure** are in each caption and in the script: `analysis/viz_fa.json`; `analysis/scree_null_matched.json` and `results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin`; `analysis/act_gram.json#arms` and `analysis/column_space.json#stage1`; `analysis/rank_sweep.json`; `analysis/fisher_norms.json`; `results/gram_sweep.npz`, `results/gram_stage2.npz`, `analysis/stage2_structure.json`, `analysis/s2mean_steer_stats.json`; `analysis/matched_dose_steering.json`; `analysis/sphere_page_fa.json` (the Spearman is recomputed from points and judged and asserted against `smooth.rho`); `analysis/spider.json#axes, bigfive`; `analysis/data_forecast.json#axis_score_vs_shift_matrix_rows_axis_cols_dim`; `analysis/em_medical.json#part_a`, `analysis/sorh_data_scoring.json`, `analysis/em_part_b.json#forecast_agreement.rows` (40 of the 145 directions are recorded there, and the figure says so).
- **One caveat recorded:** the stage-two histogram recomputes each adapter's cosine with its stage's grand mean from the Grams and asserts the means against `stage2_structure.json#shared_component` to 0.005.

### `https://persona.161-35-77-84.sslip.io/planes.html` — "Any two axes" (companion page, added 2026-09-15)

- **Status: current.** A two-dimensional cut of the factor chart with x and y chosen from the five recovered factors or the six principal components, one Goldberg group highlighted or all five, a 45% density shell per keyed half (Gaussian KDE, Scott's bandwidth times 0.62, marching squares, computed client-side), the pole connector from the negative-pole mean to the positive-pole mean, and non-overlapping labels with leaders. A six-panel mode draws one panel per group plus the held-out Lexicon and by default puts each group on the recovered factor its Big Five label is closest to. Coordinates are mean-centred over the 134 by default, with a toggle. The view is held in the URL fragment.
- **Builder:** `page_planes` in `qwen35/companion/build_companion.py`; drawing in `qwen35/companion/assets/planes.js`; data `data/chart.json` (the same file the rotating chart reads). The static version of the six-panel view is `qwen35/figures/post/facets_own_*`.

## Summary table

| page | status | builder | served |
|---|---|---|---|
| `blog_page` | current | `build_blog_page.py` | no |
| `findings_page` | historical | `build_findings_page.py` | no |
| `monitor_page` | historical | `build_monitor_page.py` | no |
| `distil_page` | **withdrawn** | — | no |
| `direction_pages` (22) | historical | `build_direction_pages.py` | no |
| `live_page` | historical | `build_live_page.py` | no |
| `spider_page` | historical | `build_spider_page.py` | no |
| `zoo_page` | historical | `build_zoo_page.py` | no |
| `manifold_page` | historical, speculative | — | no |
| `site` | historical (plan) | `build_site.py` | yes, port 8092 |
| `site_traits` | historical | `build_traits_page_data.py` | yes, port 8093 |
| `sweep100/site2` | historical (pre-zoo) | `sweep100/site2/build.py` | yes, port 8091 |
| `/var/www/persona-site` | current | `companion/build_companion.py` | yes, Caddy static root |
| `/var/www/persona-site/planes.html` | current | `companion/build_companion.py` (`page_planes`), `companion/assets/planes.js` | yes, on the companion |
| `qwen35/figures/post/` | current | `figures/post/make_post_figures.py` | yes, copied to the companion's `/figures/post/` |

Related: [[steering-results]], [[distillation-check]], [[judged-evaluations]],
[[additivity]], [[reward-hacks-arms]], [[geometry-overview]], [[glossary]].
