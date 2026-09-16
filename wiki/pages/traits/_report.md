---
title: "Traits section build report"
summary: "Generated pages, the data fields each page carries, the per-trait data looked for and not found, and the inconsistencies between data files, for the pages/traits partition of the initial build."
status: current
sources:
  - "wiki/tools/gen_trait_pages.py"
  - "qwen35/traits_primary.json"
  - "qwen35/traits_secondary.json"
  - "qwen35/traits_secondary_provenance.json"
  - "qwen35/traits_alignment.json"
  - "qwen35/traits_hole.json"
  - "qwen35/analysis/nxn_summary.json"
last_verified: 2026-09-07
tags: [meta, report]
---

# Traits section build report

Written 2026-09-07 by the agent assigned `pages/traits/`. Nothing outside this
directory and `wiki/tools/gen_trait_pages.py` was written. No Modal call, no GPU,
no upload, no reading of anything under `~/.secrets/`. Raw sources were read only.

## What was generated

`wiki/tools/gen_trait_pages.py`, run with
`qwen35/.venv/bin/python` (Python 3.12, numpy 2.5.2 available but not needed --
the script uses stdlib only, since nothing is computed).

- **141 trait pages**, `trait-<slug>.md`: the 134 zoo adapters (from
  `qwen35/analysis/nxn_summary.json#names`), the 4 alignment probes and the 3
  hole-word probes.
- **2 index pages**: `traits-index.md` (141-row table: page, trait, factor,
  keying, provenance set, recovered factor, nearest neighbour, N x N rank) and
  `traits-by-factor.md` (the same 141 grouped by the Big Five axis they were
  drawn from, and again by the recovered factor they load on most).
- 143 `.md` files in total, plus this report.

The script is idempotent: it rewrites every page on each run, deletes any
`trait-*.md` not in the generated set, and never touches `_report.md`. Two
consecutive runs produce byte-identical output (verified by `diff -rq`).

Slugs are the trait word lowercased with hyphens and spaces turned into
underscores, matching the keys already used in the analysis JSONs:
`trait-warm`, `trait-high_strung`, `trait-self_sacrificing`,
`trait-power_seeking`, `trait-hard_shelled`.

## Number handling

No number on any page is recomputed. Values are copied from the file that stores
them and the file plus its dotted key path is listed in the page's `sources`.
Where a source float carries more than four significant digits, prose shows four
significant figures (or the nearest whole number above 9,999) and appends
`(rounded)`; tables use the same rule and carry a single line underneath saying
so, rather than repeating the marker in every cell. Values the source already
stores short are printed exactly.

## Fields each page carries

Section by section, with the source and how many of the 141 pages have it.

| section | present on | fields |
|---|---|---|
| Identity | 141 | trait word, slug, factor, keying, provenance set; for lexicon traits the cluster id, cluster size, rank by centroid distance, eight other cluster members and the source list; for the 7 probes the `why` field |
| Constitution | 141 | the full `constitution` text as a blockquote, plus the `anchor` note and the names of the two unquoted variants where they exist |
| Where it sits in weight space | 141 | PC1-PC8 scores; k=5 centred oblimin FA loadings, communality, uniqueness, SMC; up to 5 nearest neighbours with cosines; N x N rank raw and column-z; for probes their own angle-to-the-zoo geometry; cross-seed self-cosine where it exists |
| Behaviour | 138 | steering dose table (alpha, expression, n, coherence, n, control expression) and baseline expression; the five `adapter_effect` fields verbatim; for alignment probes the steering alphas and raw direction norm; a statement about judged Big Five |
| Example generations | 138 | one or two responses, truncated to 300 characters, whitespace collapsed, emoji removed, with the prompt where the source stores one |
| Activation space | 16 | the `resp.matched` and `resp_specific.matched` fields verbatim, plus the primary layer |
| Training record | 141 | stage-1 DPO provenance (base commit, LoRA config, pairs, steps, loss first/last, reward margin and accuracy, seconds, GPU, corpus and pool sha256); stage-2 merge/assemble/SFT/final where recorded; second-seed stage 2 where run; persona merge audit; corpus degeneration scan; the site_traits pair description |
| Artefacts | 141 | expected HF model-repo URLs for the four adapter subfolders, expected dataset paths, dataset-audit status, second-seed availability at each stage, CROSS_TRAITS membership |
| Links | 141 | recovered-factor page, Big Five axis page, the five fixed wiki links, neighbours |

Per-source coverage over the 141 pages:

| source | covers |
|---|---|
| `qwen35/constitutions.json` | 141 (it holds 147, see below) |
| `qwen35/results/runmeta_sweep.json` | 134 zoo traits |
| `qwen35/analysis/viz.json#scores` | 134 |
| `qwen35/results/fa_qwen35.json#per_trait` | 134 |
| `qwen35/analysis/trait_graph.json#stage1.edges` | 134 |
| `qwen35/analysis/nxn_summary.json#raw.ranks` | 134 |
| `qwen35/analysis/merge_audit.json` | 134 |
| `qwen35/analysis/corpus_scan_all.json` | 134 |
| `qwen35/site_traits/data.json#steering.per_trait` | 134 |
| `qwen35/analysis/actspace_generations_adapters.jsonl` | 134 |
| `qwen35/analysis/adapter_effect.json` | 100 |
| `qwen35/phase10_runs/eval_100traits.json` | 100 |
| stage-2 OCT-2 stage records (production runs only) | 112 of the 134 |
| `qwen35/site_traits/data.json#seedpaired` | 40 |
| `qwen35/analysis/corpus_degeneration.json` | 18 |
| `qwen35/act_space.py#CROSS_TRAITS` + `actspace_cross_geometry.json` | 16 |
| `qwen35/analysis/alignment_geometry*.json` | 4 |
| `qwen35/analysis/hole_geometry.json` | 3 |

## Sections omitted, and from which traits

Rather than placeholders, an absent section is dropped. Three sections are not
universal:

- **Activation space** is on the 16 `CROSS_TRAITS` only: helpful, cold, kind,
  organized, disorganized, careful, relaxed, anxious, fretful, extraverted,
  quiet, assertive, intellectual, simple, bright, bold. The other 125 pages have
  no such section, because `actspace_geometry.json` and
  `actspace_adapters_geometry.json` are aggregate-only (see below).
- **Example generations** is absent from the 3 hole traits (cavalier, blase,
  insouciant). No generation file in the repo covers them.
- **Behaviour** is absent from the same 3 hole traits: no steering record, no
  `adapter_effect` entry and no steered generations exist for them, and a section
  containing only "the judged run does not cover this trait" would be a
  placeholder.

Within sections, subsections drop out per trait:

- **Stage-2 training numbers** are missing for 22 of the 134: agreeable, bashful,
  bold, careless, cold, considerate, creative, daring, deep, demanding,
  distrustful, efficient, envious, extraverted, imperceptive, imperturbable,
  inhibited, organized, practical, shallow, trustful, warm. Every one of these
  pages says so explicitly. The cause is that a resumed OCT-2 run writes
  `{"trait": t, "skipped": true}` for a stage it did not redo, and for these 22
  no unskipped record survives in any production results file. `extraverted` is
  in the list only because the one file that does carry a record for it,
  `results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json`, is a smoke run (40
  reflections, 8 interactions, k=4, 416 SFT rows against the production 12,000)
  and is excluded by the generator; without that exclusion the page would have
  shown smoke-run numbers as the stage-2 training record.
- **Cross-seed stage-1 cosine** appears on the 40 seed-paired traits only.
- **Second-seed stage 2** appears on the 15 traits of the seed-1 OCT run.
- **`adapter_effect` and eval generations** cover the 100 primary traits only;
  the 34 lexicon traits fall back to
  `analysis/actspace_generations_adapters.jsonl` for examples and have no
  `adapter_effect` block.
- The 7 probes have no PC scores, no FA loadings, no kNN neighbours, no N x N
  rank, no steering table, no persona merge audit and no corpus scan; they have
  their own geometry files instead, and their stage-1 numbers come from the
  training logs.

## Data looked for and not found

1. **Per-trait judged Big Five scores.** `qwen35/phase10_runs/judged_100.json`
   holds 7,200 individual judge records (100 traits x base / stage-1 / persona x
   24 prompts x 5 scores). No per-trait aggregate of them is stored anywhere in
   the repo: `analysis/live_results.json` holds only zoo-level matrices,
   `analysis/spider.json` only per-axis-and-arm means, `analysis/judged_ceiling.json`
   only the split-half ceiling. Aggregating the 7,200 records would be a
   recomputation, which rule 1 forbids, so no page quotes a judged score; every
   zoo page says this in one sentence and links `[[judged-evaluations]]`.
2. **Opposite-pole partner, and angle to opposite.** No file in the repo pairs a
   trait with its opposite, and no per-trait angle-to-opposite is stored.
   `analysis/trait_angles.json` is five aggregate numbers. Searched every JSON in
   `analysis/` for `opposite` or `antonym`: the only hits are prose in
   `blog_data.json`, `qual_fa.json`, `rl_preflight.json`, `manifold_ideas.json`
   and `blog_corpus.json`, none of them a per-trait mapping. Each page states
   plainly that no source names an opposite, so none is asserted.
3. **N x N runner-up identity.** `nxn_summary.json` stores per-trait `ranks` under
   both `raw` and `column-z`, but the runner-up is aggregate only
   (`runnerup_same_fk`, `runnerup_same_f`). The 134x134 `matrix_raw` is present
   and a runner-up could be derived from it; deriving one would be a
   recomputation, so no runner-up is named and the pages say why.
4. **Per-trait activation-space cos(P, A) for all 134.** Only the 16 CROSS_TRAITS
   were run with the constitution as a system prompt against the adapter.
   `actspace_geometry.json` and `actspace_adapters_geometry.json` are aggregate
   (a per-layer curve and one `primary` block); they carry no per-trait entry.
5. **Per-trait cross-seed cosine for stage 2.** `crossseed_arms_stage2.json` is one
   aggregate record for the 15-trait arm; no per-trait same-trait cosine is stored.
   Membership of the 15 is taken from
   `phase10_runs/results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json#traits`
   and matches the list in the task brief exactly.
6. **A stage-1 `runmeta` for the alignment and hole probes.** They are not in
   `results/runmeta_sweep.json`. Their loss, pair count, targeted-module count and
   wall time were parsed out of the unambiguous per-trait summary lines of
   `phase10_runs/aligntrain.log`, `aligncommon.log` and `holetrain.log`, and the
   log is cited.
7. **Hole-trait generations.** None found: `alignment_results.json` covers the 4
   alignment probes at five steering alphas, but there is no hole equivalent, and
   the hole words are absent from `eval_100traits.json` and from all three
   `actspace_generations*.jsonl`.
8. **Whether the alignment and hole adapters are on the Hub.** `upload_zoo_batched.py`
   builds its job list from the `pc-qwen35-sweep` and `pc-qwen35-oct2` volumes only,
   so the probes are not covered by it and nothing in the repo records their Hub
   status. The pages say so rather than guessing either way.
9. **What `adapter_effect.json`'s fields mean.** The file has no producing script
   anywhere in the repo and no built page reads it. The five values are printed
   verbatim with an explicit statement that their meaning is not established.
   (The `pages/geometry` agent reached the same conclusion independently in
   `adapter-effect-and-drift.md`.)
10. **Why the 6 drawn-but-unbuilt lexicon words were dropped.** No source records it.

Aggregate-only files that were read and deliberately not copied onto 141 pages,
because the same numbers would appear on every one of them:
`analysis/trait_angles.json`, `intrinsic_coords.json`, `geometry_stage1.json`,
`polarity_deflation.json`, `judged_ceiling.json`, `live_results.json`,
`functional_probe.json`, `crossseed_arms.json`, `crossseed_arms_stage2.json`,
`actspace_geometry.json`, `actspace_adapters_geometry.json`,
`pc_loadings.json#seed`. The pages link `[[geometry-overview]]`,
`[[n-by-n-scoring]]`, `[[judged-evaluations]]` and `[[actspace-adapters]]` for
these instead.

## Source substitutions the brief did not anticipate

- **FA loadings.** The brief pointed at `analysis/fa_summary.json`, which holds
  only the top 12 traits per factor, not a per-trait loading matrix. A full one
  does exist: `qwen35/results/fa_qwen35.json#per_trait.<Trait>.oblimin_loadings_centred_k5`,
  134 traits x 5 loadings, whose `ss_loadings.oblimin`
  (10.762..., 8.4219..., 7.0203..., 6.8145..., 5.7988...) matches
  `fa_summary.json#centred_k5.factors[*].ss` (10.76, 8.42, 7.02, 6.81, 5.8), which
  fixes the column order as Warmth / prosociality, Competence, Fearful withdrawal,
  Arousal / activation, Imagination. Each trait is mapped to the factor of its
  largest absolute loading and linked to `factor-warmth`, `factor-competence`,
  `factor-fearful-withdrawal`, `factor-arousal` or `factor-imagination`. Sign is
  stated ("loading negatively") where the loading is negative. `fa_summary.json`
  is still cited for the factor names.
- **PC scores.** `analysis/pc_loadings.json` holds only the five top and five
  bottom trait names per PC, not per-trait scores. `analysis/viz.json#scores` is a
  134 x 8 matrix of PC1-PC8 scores from the centred PCA over the stage-1 sketches
  (`build_viz_data.py` lines 36-44), and is used for the PC row. Its signs agree
  with `pc_loadings.json#pcs.PC1.pos/neg` (checked: pleasant, unsystematic,
  effeminate, sympathetic, agreeable all positive on PC1; unsympathetic, cold,
  unemotional, assertive, insensitive all negative).

## Inconsistencies between data files

Recorded, not resolved. Worth merging into `pages/overview/superseded-claims.md`.

1. **40 lexicon words drawn, 34 in the zoo.** `traits_secondary.json` and
   `traits_secondary_provenance.json#chosen` both list 40 lexicon traits, one per
   cluster of a 40-cluster draw, and `constitutions.json` holds a constitution for
   each, giving 147 constitutions (100 + 40 + 4 + 3). Only 134 adapters exist. The
   6 words with a constitution but no adapter are **unconformable, frightened,
   busy, defenseless, significant, sleepy**. They are absent from
   `nxn_summary.json#names`, `viz.json#traits`, `trait_graph.json`,
   `fa_qwen35.json#trait_slug`, `corpus_scan_all.json`, `merge_audit.json` and
   `site_traits/data.json#traits` alike, so the 134 are consistent across every
   analysis file; only the trait-definition and constitution files carry the extra
   6. No page was generated for them; `traits-index.md` names them in a short
   section. No source records why they were dropped. The brief's "34 lexicon
   traits" is right about the zoo and wrong about `traits_secondary.json`.

2. **Stage-2 adapter count: 103 versus 134.** `upload_zoo_batched.py`'s docstring
   states `stage2_introspection/ 103`, `persona_merged/ 103`, `persona_exact/ 103`
   against `stage1_dpo/ 134`. The OCT-2 results files, however, cover 100 primary
   traits (`results_oct2_100traits_*.json#traits`) plus 34 lexicon traits
   (`results_oct2_34traits_*.json#traits`) = 134, and
   `analysis/merge_audit.json` audits 134 persona merges. The docstring number is
   a snapshot from the time of upload, not a current count. The pages assert an
   adapter exists for all 134 (per the project docs and merge_audit) and cite the
   naming convention for the URL, labelled expected.

3. **`site_traits` PC coordinates differ in scale from `viz.json`.** For
   `unsystematic`, `site_traits/data.json#traits[].pc` is
   `[0.8741, 0.6316, -0.0956]` where `viz.json#scores` gives
   `[0.4610, 0.2942, -0.0420]` (first three of eight); for `pleasant`,
   `[0.899, -0.1025, 0.5281]` against `[0.4508, -0.0817, 0.2841]`. Signs agree,
   magnitudes are roughly double. Two different normalisations of the same PCA.
   Only `viz.json` is quoted; `site_traits` PC values appear on no page.

4. **`site_traits` nearest-neighbour cosines differ from `trait_graph`.** For
   `active`, `site_traits` gives energetic 0.42, vigorous 0.374, practical 0.368;
   `trait_graph.json#stage1.edges` gives energetic 0.396. Same neighbour, different
   cosine, consistent with the scale difference in item 3. The pages quote
   `trait_graph` and say the edges come from the K=5 graph.

5. **`site_traits/analysis.json` describes PC1's poles with the opposite sign to
   `pc_loadings.json`.** Its `pc1` prose reads "Its positive pole loads
   unsympathetic, insensitive and cold; its negative pole pleasant and
   sympathetic", while `pc_loadings.json#pcs.PC1.pos` lists unsystematic,
   pleasant, effeminate, sympathetic, agreeable and `.neg` lists unsympathetic,
   cold, unemotional, assertive, insensitive. `site_traits/data.json`'s own `pc`
   values agree with `pc_loadings.json`, so the prose in `analysis.json`
   contradicts the data in the same directory. Eigenvector signs are arbitrary;
   the pages use `pc_loadings.json` and quote its pole lists verbatim. Not
   resolved here; flagged for whoever owns the PC pages.

6. **Reward margins: `phase5_margins.json` versus `runmeta_sweep.json`.**
   `phase5_margins.json` sets `attribution_possible: false` and leaves every
   `final_margin` null, because the training log interleaves about four
   concurrent containers with no trait label on the `rewards/margins` lines. Yet
   `results/runmeta_sweep.json` carries a `reward_margin` for each of the 134,
   written by the training job itself rather than parsed from the log. These do
   not actually conflict, but they read as if they do; the pages quote the runmeta
   value and add a sentence saying where the log-parsing failure applies, so the
   two are not confused.

7. **The two alignment arms disagree on `power_seeking`'s nearest zoo adapter.**
   `alignment_geometry.json` gives nearest `selfish` at 72.64703452049417, while
   `alignment_geometry_aligncommon.json` gives nearest `crooked` at
   72.6923986134726 (the two candidates are within 0.1 degrees of each other in
   both arms, so the swap is a tie-break, not a disagreement about direction).
   `sycophantic` and `obsequious` give `pleasant` in both arms and `corrigible`
   gives `liberal` in both. The arms differ in the pair corpus
   (`data_alignment`, 497 pairs per trait, versus `data_alignment_common`, 444
   pairs on the pool the zoo shares). Both are shown on every alignment page,
   labelled, with neither called the current one.

8. **`traits_secondary_provenance.json` records two discarded draws.**
   `traits_secondary_DISCARDED_draw1.json` and `..._draw2.json` are still on disk
   next to the live file. Only draw 3 is used. Noted so a later reader does not
   pick up a discarded draw by filename.

## Broken links this section leaves behind

Every wikilink to another trait page resolves within `pages/traits/`. One link
target the brief specified does not exist anywhere in the wiki as of this run,
and `tools/lint.py` reports it as broken:

- `trait-provenance`, referenced once from every trait page and once from
  `traits-index`.

All the other fixed targets resolve: `factor-warmth`, `factor-competence`,
`factor-fearful-withdrawal`, `factor-arousal`, `factor-imagination`,
`factor-axis-extraversion`, `factor-axis-agreeableness`,
`factor-axis-conscientiousness`, `factor-axis-emotional-stability`,
`factor-axis-intellect`, `geometry-overview`, `n-by-n-scoring`,
`judged-evaluations` and `actspace-adapters`.

## Other notes

- All 141 pages carry `status: current`. Where a page draws on a historical build
  (`site_traits`, built before 2026-08-30), the paragraph says so in place.
- `title` and `summary` are emitted as YAML double-quoted scalars, because a
  summary begins `Warm: Agreeableness positively keyed, ...` and an unquoted
  colon breaks the frontmatter parse. `tools/lint.py` reports no frontmatter
  problem on any page in this directory.
- No emojis anywhere: sample generations are scanned and any emoji stripped, with
  the page saying "emoji removed" when that happened. The constitution and
  description blockquotes are also passed through the same filter; it changes no
  character in any of the 147 constitutions or 134 descriptions, checked. The only non-ASCII
  characters left in the directory are em dashes inside quoted constitution and
  description text, and one accented e in the word "clichés" inside the
  `unintellectual` constitution. Both are verbatim source text.
- Sample generations are truncated to 300 characters with whitespace collapsed;
  each block says so and cites the exact record and index.
