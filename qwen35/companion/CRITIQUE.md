# Self-critique rounds

Every round: rebuild, screenshot every page at 360, 1024 and 1440 px in light and dark
with `companion/shots.js` (playwright), read the PNGs, read the probe report the same
script emits (horizontal scroll, elements past the viewport, SVG text below 8 css px,
clipped SVG text, page errors, HTTP status), fix, repeat.

Shots live in `companion/shots/` named `<round>_<page>_<width>_<scheme>.png`.

---

## Round 1 — layout and legibility

Probe: 48 page/width/scheme combinations, 14 with problems.

### Found

1. **The chart, the loadings panel and the neighbour explorer did not render at all.**
   Every JS module derived the site root with
   `document.querySelector('link[rel=stylesheet]')`, and the *first* stylesheet link on
   the page is the Google Fonts one. The modules were fetching
   `https://fonts.googleapis.com/...` and parsing CSS as JSON. Visible on the page as
   `The chart data did not load: SyntaxError: Unexpected token '/', "/* cyrilli"... is
   not valid JSON`. This was the single worst defect of the round: the site's headline
   interactive figure was an error message.

2. **Horizontal scroll at 360 px on four pages.** Three separate causes:
   - Inline `<code>` elements carrying long repository paths
     (`qwen35/results/cross_gram_full_root_x_pc-qwen35-adapters_data_alignment_common.npz`)
     pushed the page to 568 px on the chart page.
   - Tables emitted inside grid cells and figures were not inside a scroll container:
     `stage-two.html` reached 499 px, `traits/warm.html` 619 px.
   - The trait page's loadings table (a fixed 60 % bar column plus label and number)
     forced 1124 px even at 1024.

3. **SVG text rendering below 8 css px.** 46 text nodes on the chart page, 30 on a trait
   page, 18 on stage two. Cause: figures with a 660-unit-wide viewBox placed in a
   two- or three-column grid scale down to 0.4–0.7, and 10 px type inside the viewBox
   becomes 4.5–7 px on screen. The elbow plot axis labels were 4.5 px at 360 px wide.

4. **Phantom panels in the statistics grid.** `.grid` drew its separating hairlines by
   showing a `--rule` background through 1 px gaps. When the item count did not divide
   into the column count, the leftover track showed as a solid grey block that reads as
   an empty card. Visible on the home page at 1440 px (8 stats in a 5-column track).

### Fixed

1. All four JS modules now resolve the site root from
   `link[href$="assets/site.css"]`, which cannot match the font stylesheet.
2. `code { overflow-wrap: anywhere; word-break: break-word; }`; `page()` now runs
   `wrap_tables()` over every page body, putting any table that is not already in one
   inside `<div class="tablewrap">` (`overflow-x: auto`); a `.tablewrap` nested inside a
   panel or figure drops its own frame so there is no double border.
3. `svg_open()` takes a `minw`, every generated figure sits in a `.scrollx` container,
   and `.axis` type went from 10 px to 11 px. Wide figures now scroll inside their own
   box rather than shrinking below legibility: elbow plots 470 px, spectrum 460 px,
   judged profile 470 px, mini chart 290 px, radar 250 px. The elbow viewBox also
   narrowed from 660 to 560 units so it fits a half-width column at full size.
4. `.grid` now draws its hairlines with `outline` on the children rather than a
   background behind them, so leftover grid tracks show the page background and not a
   panel. Two further statistics were added to the home page (the null-control component
   counts and the cross-seed Pearson) because they belong there anyway.

---

## Round 2 — data correctness

Method: `companion/check_numbers.py`. It re-opens each analysis file, formats the value
the way the page should format it, and asserts the string is present in the rendered
HTML or CSV — independently of the builder, so a builder bug cannot make a check pass.
86 checks: the 30 Tucker congruence cells, the five sums of squared loadings, the
congruence range, the number of factors retained, ten stage-two quantities, the five
stage-one-to-stage-two factor matchings, the four bipolarity cosines, four persona
geometry numbers, ten steering slopes and selectivities, four home-page headline
numbers, every loading and the communality of one trait page end to end with its
nearest-neighbour cosine and three judged means, and three spot-checks that the CSV
downloads agree with the same sources.

### Found

1. **The wrong activation-space window.** The home page and the methods page reported
   "correlation between the constitutions' activation geometry and the adapters' weight
   geometry" from `analysis/actspace_geometry.json#windows.prompt.curve[16].r_centred`,
   which is **0.385**. The project's result is defined on the **response-token** window:
   `#windows.resp.curve[16].r_centred`, which is **0.705**. Both keys exist, both are
   real numbers, and the page named the quantity correctly while showing the other one.
   `check_numbers.py` could not catch this on its own — a presence check passes for any
   consistent wrong key — so the fix came with a guard: three cross-checks that assert
   the JSON key the site reads equals the value the wiki states in prose (0.705 at layer
   16, 0.775 at layer 19, cross-seed 0.9966). Those now fail loudly if the key drifts.

2. **The wrong cross-seed arm.** The home page quoted `crossseed_arms.json[0].pearson`
   = 0.9947, which is the *unmatched* seed-paired null arm. The result the wiki reports
   is the matched-objective arm at 0.9966. The builder now selects the arm whose `path`
   contains `matched` and says so in the caption.

3. **A sentinel leaking into the data.** The neighbour lists excluded a trait from its
   own neighbourhood by writing `-9` into its own cosine and then sorting. The nearest
   list was correct; the *farthest* list therefore always began with the trait itself at
   `-9.000`. This was visible on all 141 trait pages and in `data/neighbours.json`.
   Fixed by excluding the index rather than poisoning the value.

4. **Both alpha sliders showed nothing.** Python writes the steering alpha keys as
   `"-4.0"`, `"-2.0"`, `"0.0"`; JavaScript's `String(-4.0)` is `"-4"`. Every lookup
   missed, so the dose-response chart drew no lines and the generation panel said "No
   generations were shipped for this dose" — with no console error, which is why round
   1's probe passed the page. The flagship interaction on two pages was inert.
   The builder now emits an explicit `alpha_keys` array of the strings actually used and
   both modules index by it. The probe gained functional assertions (below) so an
   empty-but-error-free widget can never pass again.

5. **Numbers typed into templates rather than read from files.** A sweep of
   `build_companion.py` for decimal literals outside interpolations found three: the
   home page's "cosine −0.08 in stage one and +0.12 in stage two", the chart page's
   "scores 0.447 on it by construction", and a headline congruence range rendered at two
   decimals as "0.40 – 0.68". All three are now interpolated from
   `stage2_structure.json#decomposition_tests`, `fa_qwen35.json#targets.target_target_congruence`
   and `#solutions.centred_k5.congruence_oblimin`. The range reads **0.405 – 0.682**,
   because the wiki's rule is never to round differently from the source and Fearful
   withdrawal's congruence is 0.405, not 0.41.

6. **Inconsistent precision between pages.** The home page showed 15%, 0.39, 0.81 and
   0.70 where the stage-two and methods pages showed 15.1%, 0.389, 0.812 and 0.705 for
   the same quantities. Harmonised upward to the file's own precision.

7. **The judged scale was asserted, not checked.** `judge_personas.py` line 39 ("rate the
   RESPONSE on these five dimensions, 1-7") and
   `wiki/pages/behaviour/judged-evaluations.md` line 26 confirm 1 to 7, which is what the
   axis and the 141 trait pages say. No change, but it was luck rather than diligence
   until it was checked.

### Also verified

- All **206** distinct wiki links on the site return HTTP 200
  (`grep -oh 'wiki…/[a-z0-9_-]*\.html' … | while read u; do curl -o /dev/null -w '%{http_code}' …`).
- `downloads/traits.csv`, `downloads/judged_means.csv` and `downloads/cosines_stage1.csv`
  agree with `fa_qwen35.json`, `judged_100.json` and `gram_sweep.npz` to within 5e-5.
- Final state: **86 checks passed, 0 failed.**

---

## Round 3 — polish

Probe after the round-2 fixes: 48 page/width/scheme combinations, 2 with problems, both
the same one.

### Found and fixed

1. **The probe could not see an empty widget.** Added functional assertions to
   `shots.js`: at least 60 scatter marks, 20 loading bars, 3 neighbour tables, 20 trait
   cards, 5 dose-response lines, and at least one generation in each of the three
   generation panels — each skipped when its anchor is not on the page. This is the check
   that would have caught round 2's defect 4 on the day it was introduced.

2. **The judged-profile figure filled 1360 px.** A 560-unit SVG stretched across the
   full page inflated its type against everything around it. Added `.fig-narrow`
   (max-width 640 px) for server-drawn figures that are sized for their own content.

3. **The hero chart was a letterbox.** At 1440 px the scatter was 1280 × 560, about
   2.3:1, which reads as a band rather than a map. `#scatter-wrap` is now capped at
   1040 px and the height rule is `min(640, max(380, W × 0.72))`, giving roughly 3:2.

4. **Point labels collided.** "Composed" printed over "Intellectual" and "Cold" over
   "Unemotional". The chart now tracks the boxes it has placed and drops a label that
   would overlap one, which also let the label budget rise from 8 to 12 (compact) and 14
   to 18 (full) — more labels, fewer collisions.

5. **Series labels overran the dose-response plot** at 1440 px ("C Conscientiousness",
   "ES Emot. stability"). Two fixes, because the first was font-dependent and flaky: the
   right margin went from 118 to 158 px, and each label now measures itself with
   `getComputedTextLength()` after insertion and falls back to its initial if it still
   does not fit. Adjacent labels are also pushed 14 px apart so two series ending at the
   same score do not print over each other.

6. **The legend was missing a mark.** The seven later adapters draw as a cross on the
   chart and had no legend entry. Added.

7. **An unquoted inline style attribute** (`style=font-weight:600`) in the congruence
   table. Browsers tolerated it; it is now a named constant.

8. **The dose-response chart scrolled out of view** while reading the generations beside
   it. It is now `position: sticky` above 900 px.

### Deliberately left

- `vendor/d3.v7.min.js` is downloaded but unreferenced. The brief allowed vanilla *or*
  d3; the charts are hand-drawn SVG, which is smaller and reads the theme's CSS custom
  properties directly. The file stays in `companion/` and is not copied into the served
  site.
- `traits.js` draws the alignment and hole words as a hollow swatch in the index where
  the chart draws them as a cross. Cosmetic disagreement, noted rather than fixed.
- The behaviour page ends with two adjacent "sources" disclosures, one belonging to the
  reward-hack section and one to the page. Correct, if slightly redundant.

---

## Final state

`node companion/shots.js` over all eight page templates at 360, 1024 and 1440 px in light
and dark: **48 combinations, 0 problems** — no horizontal scroll, no element past the
viewport, no SVG text below 8 css px, no clipped SVG text, no page errors, every widget
populated, every page HTTP 200.

`companion/check_numbers.py`: **86 checks passed, 0 failed.**

All 206 distinct wiki links return 200; the three Hugging Face folder links were spot
checked on one trait and return 200.

Screenshots of the final build are `companion/shots/final_<page>_<width>_<scheme>.png`.
Earlier rounds are `r2b_`, `r3_`, `r4_`, `r5_`, `r6_`, `r7_`, `r8_`; the round-1 files
were overwritten by a later run that forgot to set `ROUND`, so what round 1 saw is only
in the notes above, not in an image.
