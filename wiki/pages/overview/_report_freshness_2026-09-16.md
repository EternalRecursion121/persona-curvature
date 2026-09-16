# Freshness audit report, 2026-09-16

Scope: every non-trait page whose `last_verified` was before 2026-09-10 (136 pages: 117 at 2026-09-07, 14 at 2026-09-08, 5 at 2026-09-09), checked against the 2026-09-15 post draft and the analysis files. Method: a script extracted every numeric literal from each page body (small integers, dates, URLs, wikilinks, code spans and inline SVG excluded) and checked whether it appears verbatim, or as a rounding or prefix of a full-precision value, in one of the files the page's `sources` frontmatter names; JSON sources were flattened to their numeric leaves, markdown sources searched as text, files over 3 MB and binary arrays skipped. Pages with unmatched numbers were then checked by hand: each unmatched value was grepped across `qwen35/analysis`, `qwen35/results` and the project's markdown to find where it lives. Stale framing was checked by phrase ("blog page", "in flight", "running as of", "not yet", "as of 2026-09-07") and each hit read in context.

Rule applied: `last_verified` was set to 2026-09-16 only where every number matched a cited source by the script, or where every unmatched number was found by a second, whole-tree grep (`qwen35/analysis`, `qwen35/results`, `qwen35/*.json`, `qwen35/*.md`, `qwen35/phase10_runs`, `qwen35/phase2_runs/archive`, `wiki/raw`, also trying the percent conversion of each value). That second pass is a presence check, not a key-by-key reading: it establishes that the number exists in a project file, not that the page cites the right key for it. Pages with any number found nowhere keep their old date. History pages were checked for status only and keep their dates unless edited for another reason.

Pages bumped to 2026-09-16 by the audit: 70. Pages left at their old date: 47.

## Pages whose numbers could not be verified against a file

- `behaviour/reward-hacks-column-space.md` (kept at 2026-09-09): `0.32296256459881073` (line 465, a zoo-adapter column-space figure) appears in no file under `qwen35/`; the other two unmatched values were traced. Sources not on disk: none.
- `geometry/umap-and-layouts.md` (1 of 41 numbers unmatched): 52.85639. Sources not on disk: none.
- `zoo/discarded-secondary-draws.md` (1 of 8 numbers unmatched): 105. Sources not on disk: none.
- `conversation/user-questions-index.md` (5 of 5 numbers unmatched): 1.5, 100, 134, 177, 252. Sources not on disk: none.
- `overview/big-five-history.md` (9 of 9 numbers unmatched): 1,431, 100, 133, 17,953, 171, 2,800, 339, 4,500, 479. Sources not on disk: ['wiki/raw/ (assistant answer of 2026-09-07 to "can you tell me a bit about the history of the big 5 and what analysis goldberg did", main transcript)'].
- `overview/goldberg-intellect-factor.md` (1 of 1 numbers unmatched): 100. Sources not on disk: ['wiki/raw/ (assistant answers of 2026-09-07 in the main transcript)'].
- `overview/how-to-read.md` (1 of 1 numbers unmatched): 141. Sources not on disk: none.

Readings: `umap-and-layouts` is historical and its one unmatched value (52.85639 degrees) is a hole-gap figure from a superseded k = 5 layout; `discarded-secondary-draws` quotes a draw count (105) from `plan.json` prose; `user-questions-index` and `big-five-history` and `goldberg-intellect-factor` quote transcript answers (literature counts such as Allport and Odbert's 17,953 words) whose only source is `wiki/raw/`; `how-to-read` says 141 trait pages, which is the generator's count. None of these is a project measurement.

## History pages (status sanity only)

All history/ pages carry `historical` (pre-zoo experiments, the discarded draws) or `current` (the source-paper notes and the method lessons, which are still the project's reading of those sources). No status was changed. `timeline` was extended to 2026-09-16 and bumped. Their unmatched numbers are literature values cited to arXiv URLs (`paper-*`, `literature-*`), spend figures from the meter log (`costs`, `harness-context`), or infrastructure facts (`infrastructure`); none was checked against a file and none carries a new date.

## Load-bearing numbers whose only home is a built page or a hard-coded string

Unchanged from [[superseded-claims]] section 1c: the sketch-validation correlation (0.9996 in `build_blog_page.py`, 0.99944 in `build_monitor_page.py`, no file), the FA_Warmth and FA_Imagination clean-prompt selectivities, the named axes' "intact over alpha" claim, and PC4's 0.46 cosine. The post draft of 2026-09-15 quotes none of these except the finite-difference check 0.9999992, which was recomputed from `analysis/align_validate.json` rows on 2026-09-10.

## Framing corrections made during the audit

- Rule 2 of `wiki/CLAUDE.md` rewritten: current truth is `qwen35/POST_DRAFT.md` plus the analysis files, not the blog page.
- Sentences declaring the blog page current on `stage-two-geometry`, `actspace-cross`, `dpo-pair-generation`, `superseded-geometry-claims` and the glossary reworded to date the statement.
- `open-questions` "In flight" closed (the full OCT replication landed 2026-09-08); a 2026-09-16 section added.
- `superseded-claims` section 1 reframed as historical with the items the draft fixed; a table of nine reframings between the 2026-09-10 and 2026-09-15 drafts added.
- `decisions-log`, `stage-one-versus-stage-two-clarification` and `external-review` given their post-2026-09-07 resolutions.
- `most-interesting-claims`, `blog-skeletons`, `blog-site-map-idea` set to historical (superseded drafting aids).
- The six `factor-pc*` pages given a frame note that the factors, not the PCs, are primary since 2026-09-08.
- `stage-one-training-config` given the 13-versus-47-step pointer; `reward-hacks-data-scoring` and `sycophancy-forecast` record the Figure 11 decision.

## Discrepancies between the draft and the files

None in value. One in attribution: the draft's "-0.175 length-stratified" corrigibility figure is the `corr_ls` arm of the sycophancy run (`analysis/syc_forecast.json#compliance.rates`, 0.15 against 0.325), not a rerun in `dolci_flag_training.json`; recorded on [[claims-and-evidence]] and [[sycophancy-forecast]].

## Provenance gaps found

`analysis/best_axis_pairs.json` (Figure 1) and `analysis/fa_text_contrast.json` had no producing script checked in under `qwen35/` when this audit ran; later the same day `analyse_best_axis_pairs.py` and `analyse_fa_text_contrast.py` were checked in and shown to regenerate both files exactly. `analysis/scree_null.json` and `scree_null_matched.json` are named only by a consumer. Full list on [[code-and-data-map]].
