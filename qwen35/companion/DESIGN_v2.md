# Companion site v2 — the blog's identity, in three dimensions

Samuel, 2026-09-14: *"can you redo the companion site in the style of the interactive
blog with the 3d diagrams instead of 2d diagrams"*, plus two additions: a system /
light / dark theme with an override button, and training-data and generation examples
on every trait page.

This file is the plan. It was written before any code changed. `CRITIQUE_v2.md` is the
self-critique written after the first full build.

---

## 0. The two artefacts, as they stand

**The blog** (`qwen35/blog_page/index.html`, built by `build_blog_page.py` from
`_css_base.txt`, `_css_comp.txt`, `_js.txt`) is a single scroll. Literata body at
18px/1.68 with old-style numerals, Fraunces for display, IBM Plex Mono for every
number and every control. A light-only palette of six hues, one per Big Five scale,
used for nothing but data. Hairline rules, 2px corners, no shadows, no accent colour.
Its map is a canvas: a rotating orthographic projection of the 134 adapters, axes
picked from five factors and six components, depth carried in radius and alpha,
drag to rotate, hover to name, click to pin, six chips to toggle groups, the cloud
centred on the mean adapter, full axis lines with Timidity labelled at its negative
pole, and two markers held behind `if (false)`.

**The companion** (`companion/build_companion.py` → `/var/www/persona-site`, 188 files)
is a site: masthead, seven top-level pages, 141 trait pages, five methods sub-pages,
downloads, a `data/*.json` contract fetched by relative path. IBM Plex Sans body,
Instrument Serif headings, its own five-colour factor palette, both themes already.
Its charts are SVG and 2D.

They do not read as one publication. That is what this changes.

## 1. The five lines

1. **Take the blog's identity whole** — Literata / Fraunces / IBM Plex Mono, its six
   data hues, its hairline-and-2px-corner card and figure treatment, its eyebrow,
   stand-first and figcaption scale — and make it the companion's, so a reader moving
   between them sees one publication with two shapes.
2. **Every 2D scatter becomes the blog's 3D canvas**, ported once as
   `assets/map3d.js` and mounted declaratively, so the factor chart on `chart.html`
   and `index.html`, the small chart on all 141 trait pages, and the stage-two cloud
   are the same instrument at different sizes.
3. **Keep the site a site**: the masthead, the seven-page nav, the 141 trait pages,
   the wiki deep links, the downloads, the `data/*.json` contract and the relative
   `BASE`. The blog is a scroll; the companion is navigated, and its header stays.
4. **Both themes, from one token set**: light on `:root`, dark under
   `prefers-color-scheme` guarded so the un-stamped system state follows the OS, dark
   again under `[data-theme="dark"]` so a toggle wins in both directions; the canvas
   reads its colours from the tokens at draw time and redraws on either signal.
5. **Every number and every sentence survives**, except captions tied to a removed 2D
   figure; every new number is read from the analysis files by the builder, and
   `check_numbers.py` grows rather than shrinks.

## 2. What is kept, unchanged in substance

- The seven top-level pages and their order: Home, The chart, Behaviour, Stage two,
  Traits, Data, Methods, plus the five methods sub-pages and `downloads/`.
- All 141 trait pages, restyled only through `page_trait()`. No hand edits.
- Every prose paragraph, every statistic, every `Sources for this section`
  disclosure, every `qwen35/...` citation, every wiki deep link.
- The `data/*.json` contract and the `BASE` constant derived from the stylesheet
  href, so the site stays static-hostable from any prefix.
- One external resource: the Google Fonts stylesheet.
- The incremental `.build_stamp`, the `--out` flag, `check_numbers.py`.
- The non-scatter figures: `scree_panels`, `radar_svg` (spider), `profile_svg`
  (judged bars), `spectrum_svg` (stage-one against stage-two bars), the loadings bar
  panel, the register table, the neighbour tables, the dose-response line chart in
  `behaviour.js`. These are restyled through tokens; their geometry is untouched.

## 3. Identity

### 3.1 Type

| role | was | becomes |
|---|---|---|
| body | IBM Plex Sans 16/1.58 | Literata 17/1.62, old-style numerals |
| display | Instrument Serif | Fraunces, optical sizing, `SOFT`/`WONK` settings |
| numbers, controls, labels | IBM Plex Mono | IBM Plex Mono (unchanged) |

The font link becomes the blog's exact URL — Fraunces `opsz,wght@9..144,400;9..144,600`,
Literata `opsz,wght@7..72,400;7..72,600`, IBM Plex Mono `wght@400;500;600` — so the two
pages request the same faces and render identically. Body drops from 18px to 17px:
the companion is denser than an essay and carries tables the blog does not.

Kept from the companion: the `.eyebrow`-style `sec-num`, the uppercase mono `h4`, the
tabular numerals in tables. Dropped: Instrument Serif, and the sans body.

### 3.2 Colour

The blog's six hues are the primitives, one per Big Five scale:

```
--ex #B85E1E  --ag #1B8168  --co #3A56A4  --es #764AA0  --in #A93555  --lx #8A90A0
```

The companion's `--f0..--f4` are **aliased** to them at build time, each factor taking
the hue of the Big Five scale its best Tucker congruence lands on, read from
`results/fa_qwen35.json` via `b5_of_factor` — never hand-typed. Today that gives
Warmth→Agreeableness (green), Competence→Conscientiousness (blue), Timidity→Emotional
stability (purple), Arousal→Extraversion (orange), Imagination→Intellect (crimson).
Aliasing rather than renaming means every `var(--f{i})`, `.f{i}` and `.bg-f{i}` already
in the builder keeps working, and the chart's "colour by factor" and "colour by Big
Five" modes become literally the same palette instead of two that agree by convention.

Everything else is greyscale. Colour on the page means "factor", still.

### 3.3 Structure

- `.panel` gains the blog's treatment: `--card` ground, 1px `--rule`, 2px corners.
- `figure` / `figcaption` take the blog's scale (14.5px, `--dim`, 44em measure).
- `.dir`-style rail cards are adopted for the five factor cards on `chart.html`.
- Tables keep the companion's sticky head but take the blog's uppercase mono `th`,
  right-aligned numerics and `tr.hi` highlight.
- The masthead keeps its graticule, wordmark and ribbon — it is the thing that says
  "this is a site, not the essay" — restyled in the new tokens, and gains the theme
  button at its right edge.

## 4. Theme

Three blocks, identical token sets, verified by a build-time check:

```css
:root { color-scheme: light; /* every token */ }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; /* every token */ } }
:root[data-theme="dark"] { color-scheme: dark; /* every token */ }
```

So: un-stamped follows the OS; `data-theme="light"` beats a dark OS; `data-theme="dark"`
beats a light one. The dark hues are the companion's existing dark five, re-keyed onto
the six blog hue names (they were already contrast-tuned against a `#0D1013` ground).

A button in the masthead cycles **system → light → dark** and shows the state it is in.
The choice is stored in `localStorage` inside `try/catch`; an inline script in `<head>`,
before the stylesheet, stamps `data-theme` so there is no flash. `<meta name="color-scheme">`
stays `light dark`.

The canvas charts read every colour through `cssv()` at draw time, exactly as the blog
does, and redraw on **both** signals: a `MutationObserver` on `documentElement`'s
`data-theme` (the toggle) and `matchMedia('(prefers-color-scheme: dark)')` change (the
un-stamped state). Either alone would miss half the cases.

## 5. The 3D charts

### 5.1 The module

`assets/map3d.js` is `_js.txt`'s `makeMap` ported whole: `Rot` with freely wrapping
tilt, the two-family score scaling (chart coordinates and PC scores are different
units and are each scaled by their own maximum), the centroid subtraction so the cross
sits at the mean adapter, the depth-scaled radius `2.6+3.4*dep` and alpha
`0.36+0.64*dep`, the painter's-algorithm z-sort, ringed centres for negatively keyed
traits, full axis lines at 1.16R with labels at 1.22R and the negative-pole rule for
Timidity, the chips, the three axis selects, the `.maptip` and `.axisnames` hint lines,
the auto-spin honouring `prefers-reduced-motion`, hover and pin. **Both `if (false)`
blocks are ported verbatim and stay disabled.**

It is parameterised rather than bound to a global: it mounts on every
`[data-map]` element and reads

| attribute | meaning |
|---|---|
| `data-src` | data URL relative to `BASE` (default `data/chart.json`) |
| `data-compact` | smaller canvas, no axis selects, fewer controls |
| `data-focus` | a slug to pin on load — this is what replaces `mini_chart` |
| `data-stages` | show the stage selector (stage-two page only) |
| `data-axes` | initial x,y,z axis indices |

### 5.2 Which figures become 3D

| page | figure | before | after |
|---|---|---|---|
| `index.html` | the factor chart | 2D SVG, compact | 3D canvas, compact, axes 0/1/2 |
| `chart.html` | the factor chart | 2D SVG with axis/colour/set/find controls | 3D canvas with x/y/z + colour + set + find |
| `traits/*.html` (×141) | `mini_chart` | 2D SVG, one adapter ringed | 3D canvas, compact, that adapter pinned |
| `stage-two.html` | — (no cloud before) | — | 3D canvas with a stage selector |
| `behaviour.html` | dose-response | 2D line chart | unchanged geometry, restyled |
| `data.html` | tables only | — | unchanged |

`behaviour.html` and `data.html` carry no scatter: the dose-response is a line chart of
judged score against alpha, which a rotating cloud would make worse, and `data.html` is
tables and download links. Saying so is part of the answer to "any other scatter".

### 5.3 The companion's extras, kept

The blog's map has chips and axis selects. The companion's chart had four more controls
and two panels; all are kept and wired to the 3D map:

- **Colour by** — leading factor / Big Five keying of the word / provenance set.
- **Show** — all 141 / the 134 zoo / Goldberg only / Lexicon only / alignment and hole.
- **Find** — typing a trait word dims the rest and pins the match.
- **Pinning** opens a small panel beside the canvas with the trait's five loadings and
  a link to its page. The blog pins to read a label; the companion pinned to navigate.
  A pin that shows the loadings *and* offers the link keeps both, and stops a stray
  click from navigating away mid-rotation.
- The **loadings** bar panel and the **neighbours** tables on `chart.html` stay as they
  are, restyled.

### 5.4 Axes

The picker offers the five factor axes first, then PC1–PC6, in two `<optgroup>`s,
exactly as the blog does and in the same column order — factors 0–4, components 5–10.
PC scores come from `analysis/blog_data.json#viz.scores`, registered as an optional
input and asserted name-aligned with the Gram order; if the file is absent the PC
optgroup simply does not appear. The seven external adapters have no PC scores, so
selecting a PC axis drops them and the hint line says so.

### 5.5 The stage-two cloud

There is no cross-Gram between stage one and stage two over all 134, so there are no
coordinates for the three stages in one frame, and an overlay would draw arrows that do
not exist. What does exist: `fa_chart.FAChart` takes a Gram, and
`results/gram_stage2.npz` and `results/gram_personas.npz` are in the same name order as
`gram_sweep.npz`. So the builder charts each stage by the **identical recipe** — the same
five factor coefficient vectors from `phase10_runs/steer_spec2_7a.json`, Gram–Schmidt in
that stage's own Gram — and the page offers a **stage selector, not an overlay**. The
caption says plainly that each cloud is charted in its own Gram, so the shape of a cloud
is comparable across stages but a point's position is not a movement.

## 6. Trait-page examples

Each of the 141 trait pages gains a **Training data and behaviour** section, generated
in `page_trait()`:

**(a) Three preference pairs.** From `qwen35/data_common/<slug>.jsonl` for the 134,
`data_alignment_common/` and `data_hole_common/` for the seven externals — the 445-prompt
intersection the sweep actually trained on ([[shared-prompt-pool-445]]). The prompts are
byte-identical and in identical order across every file (verified), so taking **pool rows
0, 1, 2** makes every trait page comparable. Each pair shows the prompt, the chosen reply
labelled *chosen — written in character*, and the rejected reply labelled *rejected —
written by a character at the opposite pole*.

The 60-entry safety adjudication (`phase10_runs/adjudications.json`) is keyed by
`(file, pattern, rows)` and every one of its 57 files is a stage-two file
(`self_interaction/`, `self_reflection/`, `sft_data/`) — no stage-one pair is listed. The
skip filter is implemented against the file anyway, so that if the adjudication ever
grows a stage-one entry the pages drop it; today it removes nothing, and the report says
so. The three rows chosen are also read by eye before shipping.

**(b) Three generations.** From `phase10_runs/eval_100traits.json`, prompt indices 0, 4
and 12 — the same three for every trait — at the `stage1` condition, each with the
judge's five Big Five scores for **that response** from
`phase10_runs/judged_100.json#records` keyed by `(trait, condition, prompt_idx)`,
averaged over repeats with the count shown. The battery covered the 100 Goldberg
markers; the 34 Lexicon words and the seven externals have none, and their pages say
which file would have held them.

**(c) Stage two.** The transcripts are not in this repository, but they are public on
the Hub: `EternalRecursion/persona-curvature-oct-transcripts` holds 536 files, four per
trait, and `self_reflection/<trait>.jsonl` is 10,000 rows of the model writing about
itself in character after the second training stage. A whole file is around 60 MB, so
`companion/fetch_stage2_excerpts.py` downloads none of them: it reads the first 64 KB
of each over an HTTP range request, keeps one row, and caches it under
`companion/cache/stage2/` (already covered by the `cache/` rule in `.gitignore`). That
script is deliberately not part of the builder, which runs from a timer every minute
and must never touch the network; the builder reads the cache, and a trait with no
entry says so. The row is the first one not named in the safety adjudication, which
today is row 0 for all 134 — and row 0 carries the same prompt in every file, so the
excerpts are comparable page to page exactly as the training pairs are. The seven later
adapters were never trained past stage one and their pages say that.

Long texts are truncated with `<details>`: the first ~420 characters in the `<summary>`,
the remainder in the body, so nothing is duplicated and no JavaScript is needed. Each
page cites its source files. A budget check keeps every trait page under 150 KB.

## 7. Data emitter changes

All from the same analysis JSON the blog reads; nothing hand-typed.

- `data/chart.json` gains `axnames` (5 factor titles + PC1–PC6), `axdesc` (the factor
  solution names from `viz_fa.json#solution_names` and the component names), `pc`
  (134×6 from `blog_data.json#viz.scores`), and `nfa`.
- `data/stages.json`: per-stage 134×5 chart coordinates, chart lengths and norms for
  stage one, stage two and persona, each from `FAChart` over that stage's Gram, with a
  `how` string naming the construction.
- `data/traits.json` and `data/neighbours.json`: unchanged contract.

## 8. Verification

1. `check_numbers.py --out <dir>` — must stay at 233 passing and grow: new checks count
   the pair blocks (3 on every one of the 141) and the generation blocks (3 on the 100,
   the stated-absence sentence on the other 41), and assert a judged per-prompt score is
   on the page. No check is deleted.
2. `node -e` syntax check on every `assets/*.js`.
3. Grep the built tree for the retired 2D hooks: `data-scatter`, `id="scatter`,
   `scatter-wrap`, `class="tip"`, and `mini_chart`'s aria-label text. None may remain.
4. A token check: the three theme blocks declare the identical set of custom properties,
   and no colour is defined only inside a media or `[data-theme]` block.
5. A link checker over the built tree: every relative href resolves; no absolute
   self-link to `persona.161-35-77-84.sslip.io`.
6. A sentence diff of the previous build against the new one, per page: the only losses
   allowed are captions tied to a removed 2D figure and the footer's marks legend.
7. `find traits -size +150k` must be empty.

## 9. Order of work

Snapshot `/var/www/persona-site` → stop the timer → this file → CSS, theme, `map3d.js`,
emitter, trait examples → build to `/var/www/persona-site-staging --force` → run every
check → `CRITIQUE_v2.md`, fix the top items, rebuild → build live `--force` → start the
timer → wiki inventory entry, `log.md` line, `wiki/tools/lint.py`.
