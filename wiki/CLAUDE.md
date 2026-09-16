# persona-curvature wiki — schema

This directory is an LLM-maintained wiki over the persona-curvature project
(Samuel Ratnam, HF `EternalRecursion`): a zoo of 134 personality-trait LoRA
adapters on Qwen3.5-4B, the geometric analysis of that zoo, and everything
that led up to it. The wiki exists so that a human can write the blog post
from it. It is read by Samuel and by future Claude sessions; it is written
only by Claude.

Three layers:

- **Raw sources** — the repo at `~/projects/persona-curvature/` (code, md
  documents, `analysis/*.json`, `results/`, `phase10_runs/` logs, built
  pages), `~/projects/agent-harness/memory/` (old harness memory), the
  `.garden/` journals, and `wiki/raw/` for material that lives nowhere else
  (extracts from conversation transcripts, HF listings). Raw sources are
  immutable: read them, never edit them.
- **The wiki** — `wiki/pages/**/*.md`. Claude owns it entirely.
- **This schema** — how the wiki is structured and maintained.

## Layout

```
wiki/
  CLAUDE.md        this file
  index.md         catalogue of every page: link, one-line summary, status
  log.md           append-only chronology: `## [YYYY-MM-DD] ingest|query|lint | title`
  raw/             sources that exist nowhere else (immutable once written)
  tools/           build_site.py (static HTML), lint.py, gen_trait_pages.py,
                   gen_fulloct_page.py, gen_shared_direction_page.py, gen_scree_svg.py,
                   transcript_extract.py
  pages/
    overview/      home, start-here-for-collaborators, claims-and-evidence,
                   code-and-data-map, how-to-read, superseded-claims, open-questions,
                   glossary, post-draft
    zoo/           how the 134 adapters were built: traits, provenance, constitutions,
                   pair generation, OCT stage 1 and 2, training config, HF artefacts, spend
    geometry/      weight-space results: Gram, PCA/FA, scree, nulls, seed floor,
                   cross-seed, attenuation, N x N, hole words, alignment traits, stage 2
    factors/       one page per recovered factor / PC / Big Five axis
    behaviour/     steering, verification, additivity, alien direction, sphere sweep,
                   optimised data, RL and reward-hack arms, judged evaluations
    actspace/      activation-space analyses (constitution prompts, adapters, cross)
    history/       pre-zoo work (drift, gradprobe, sweep100, teacherscreen, phase 2),
                   literature (OCT, Persona Cartography, Big Five), infrastructure,
                   costs, the project's own method lessons
    conversation/  material recovered from chat: the external review (page slug
                   `external-review`; the reviewer is not named anywhere in the wiki),
                   blog skeletons, explanatory answers (elbow figure, double-centred
                   Gram, Big Five mapping)
    traits/        one generated page per trait (134 zoo + alignment + hole + null arms)
```

## Page format

Every page is markdown with YAML frontmatter:

```
---
title: Human-readable title
summary: One sentence. Copied verbatim into index.md.
status: current | superseded | unconfirmed | withdrawn | historical
sources:
  - qwen35/PHASE3_VERDICT.md
  - qwen35/analysis/crossseed_arms.json#seed_matched.same_trait_mean
last_verified: 2026-09-07
tags: [geometry, nulls]
---
```

- `sources` are paths relative to `~/projects/persona-curvature/`. For JSON, append
  `#dotted.key.path` for the key the number came from. For other repos use an
  absolute path.
- `status`: `current` is the live claim; `superseded` means a later source
  replaced it (say by what, link the replacement); `unconfirmed` means the
  data are suggestive but the page's own source says not established;
  `withdrawn` means the project retracted it; `historical` means pre-zoo or
  no longer load-bearing.
- Link pages with `[[slug]]`, where slug is the filename without `.md`.
  Slugs are unique across the whole wiki. Trait pages are `trait-<name>`,
  factor pages `factor-<name>`. Everything else is descriptive kebab-case.
- Use `[[slug|display text]]` for a different link text.

## Rules that matter

1. **Every number carries its source.** Quote numbers verbatim from a file
   (md, json, log) and cite the file, and for JSON the key. Never recompute a
   number, never round differently from the source. If the only place a
   number exists is a built HTML page, cite the page and say so. The blog
   post will copy from this wiki; a wrong number here propagates.
2. **Current truth is the post draft plus the analysis files it cites.**
   `qwen35/POST_DRAFT.md` (short rewrite of 2026-09-15, appendix A0 to A11,
   mirrored at `pages/overview/post-draft.md`) with `qwen35/POST_OUTLINE.md`
   is the authoritative current state of the project's understanding; the
   numbers behind it live in `qwen35/analysis/*.json` and `qwen35/results/*.json`
   and each figure names its files in `qwen35/figures/post/make_post_figures.py`.
   The companion site (`qwen35/companion/`, built by `build_companion.py`,
   checked by `companion/check_numbers.py`) is the public presentation of the
   same files. Where the draft and a JSON disagree, the JSON is the source and
   the disagreement is recorded (rule 3). The 134-trait factor solution
   (`results/fa_qwen35.json#solutions.centred_k5`) is primary; the 100-marker
   solution is a control (`goldberg_only.json`). `PHASE3_VERDICT.md` is
   authoritative for results dated before 2026-09-10 in its latest addendum
   only. The blog page `qwen35/blog_page/index.html` (built by
   `build_blog_page.py`, last current 2026-09-07) and the older built pages
   (`findings_page`, `manifold_page`, `distil_page`, `direction_pages`,
   `site_traits`, `live_page`, `spider_page`, `zoo_page`, `sweep100/site2`)
   are historical and may be stale: record what they show and their status;
   do not re-derive their content as current.
3. **Contradictions are recorded, not resolved.** If two sources disagree,
   write both, mark which is current and why, and add an entry to
   `pages/overview/superseded-claims.md` (or to your `_report.md` for the
   maintainer to merge). Never silently pick one.
4. **Raw sources are read-only.** No Modal calls, no reruns, no GPU, no
   uploads. Read JSONs and logs; never regenerate them.
5. **Secrets.** Never read, copy or mention the contents of anything under
   `~/.secrets/`. Never paste tokens, keys or password hashes.
6. **No emojis** anywhere in the wiki.
7. **Prose style.** Plain, short sentences. Say what was measured, on what,
   with what result, and what it does and does not establish. Where the
   project itself corrected an earlier claim, say so; the history of
   corrections is part of the record.
8. **Each page stands alone.** A reader landing on it from the index should
   understand it without the chat history. Define a term on first use or link
   to the glossary.

## Workflows

**Ingest** a new source: read it; write or update the relevant pages; add
`sources` entries; update `index.md`; append to `log.md` as
`## [YYYY-MM-DD] ingest | <source>`. Touching ten pages for one source is
normal.

**Query**: read `index.md`, open the relevant pages, answer with `[[links]]`.
File any answer worth keeping as a page under the closest directory and log
it as `query`.

**Lint**: run `tools/lint.py` (broken links, orphans, duplicate slugs,
missing frontmatter). Also read for contradictions, stale `status`, and
numbers without a source. Log as `lint`.

**Publish**: `tools/build_site.py` renders `pages/` to static HTML for Caddy
and regenerates `index.md` from frontmatter (never hand-edit `index.md`; the
`summary` field is the index line). A systemd timer rebuilds on change; do not
run the builder by hand in a session. See `pages/overview/how-to-read.md`.

## Agent partitions used for the initial build (2026-09-07)

Each initial-build agent wrote only inside its assigned directory and left a
`_report.md` there (pages written, contradictions, superseded claims, gaps,
numbers it could not source). The maintainer merged reports into
`superseded-claims.md`, `open-questions.md`, `index.md` and `log.md`.
