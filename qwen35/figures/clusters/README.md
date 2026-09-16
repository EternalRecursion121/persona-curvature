# Trait adapter clusters in factor space

Eight static styles of one claim, so Samuel can pick one.

Everything here is produced by a single script:

```
cd /home/vibe12/projects/persona-curvature
qwen35/.venv/bin/python qwen35/figures/clusters/make_cluster_figures.py
```

## What the figures claim, and what they do not

- The colours are the **Goldberg factor label** carried on each trait word. They are
  not a clustering result; nothing here fits clusters and then colours them.
- The five recovered oblimin factors match the Big Five at Tucker congruence
  **0.40 to 0.68** (E 0.539, A 0.655, C 0.574, ES 0.405, I 0.682). None clears the
  conventional 0.85 "fair" bar. See `wiki/pages/geometry/factor-analysis.md`.
- The structure is **signed bipolar axes, not blobs**: same-keyed traits of a factor
  align, opposite-keyed anti-align (residual gap +0.2393, p at the permutation floor;
  `wiki/pages/geometry/polarity-and-bipolarity.md`). Every shell in these figures is
  therefore drawn per (factor, keying), not per factor.
- The third factor is `FA_FearfulWithdrawal`, displayed as **Timidity**. It is named
  for its negative pole; the axis is oriented so positive is the bold / stable pole,
  and every axis carrying it is annotated at both poles.
- The 34 held-out **Lexicon** words are grey and sit in the same cloud; they were not
  used to define the factors.

## Data

| file | what it gives |
|---|---|
| `qwen35/analysis/viz_fa.json` | 134 x 5 factor-chart coordinates, Goldberg factor label, keying, factor titles. Coordinates are mean-centred before plotting (the chart is uncentred and every adapter shares a common component; `wiki/pages/geometry/factor-chart.md`). |
| `qwen35/results/gram_sweep.npz` | the exact 134 x 134 Gram. All cosine views double-centre it first (`Gc = H G H`), matching the `centred_k5` FA solution; raw cosine carries a +0.07 shared component that would push every off-block positive. |
| `qwen35/results/gram_data_null_permuted_p100_matched.npz` | the permuted-label null arm: same recipe, trait labels reassigned, retrained at the zoo's matched objective. **100 Goldberg markers only** - the 34 Lexicon words were never trained in the null, so the null panel is 100 x 100 and the figure says so. |

## Files

| file | style | size | variants |
|---|---|---|---|
| `editorial_light.png` | Editorial scatter | 1015 KB | light |
| `editorial_dark.png` | Editorial scatter | 941 KB | dark |
| `editorial_light.svg` | Editorial scatter | 400 KB | light |
| `editorial_dark.svg` | Editorial scatter | 400 KB | dark |
| `pairs_light.png` | Pairs matrix | 1.2 MB | light |
| `pairs_dark.png` | Pairs matrix | 1.1 MB | dark |
| `pairs_light.svg` | Pairs matrix | 1.7 MB | light |
| `pairs_dark.svg` | Pairs matrix | 1.7 MB | dark |
| `threed_light.png` | Static 3D render | 929 KB | light |
| `threed_dark.png` | Static 3D render | 802 KB | dark |
| `threed_light.svg` | Static 3D render | 227 KB | light |
| `threed_dark.svg` | Static 3D render | 227 KB | dark |
| `dendrogram_light.png` | Dendrogram | 411 KB | light |
| `dendrogram_dark.png` | Dendrogram | 398 KB | dark |
| `dendrogram_light.svg` | Dendrogram | 314 KB | light |
| `dendrogram_dark.svg` | Dendrogram | 314 KB | dark |
| `heatmap_light.png` | Sorted cosine heatmap | 579 KB | light |
| `heatmap_dark.png` | Sorted cosine heatmap | 569 KB | dark |
| `heatmap_light.svg` | Sorted cosine heatmap | 345 KB | light |
| `heatmap_dark.svg` | Sorted cosine heatmap | 339 KB | dark |
| `strips_light.png` | Bipolar axis strips | 658 KB | light |
| `strips_dark.png` | Bipolar axis strips | 651 KB | dark |
| `strips_light.svg` | Bipolar axis strips | 299 KB | light |
| `strips_dark.svg` | Bipolar axis strips | 299 KB | dark |
| `mds_light.png` | MDS constellation | 1.6 MB | light |
| `mds_dark.png` | MDS constellation | 1.5 MB | dark |
| `mds_light.svg` | MDS constellation | 334 KB | light |
| `mds_dark.svg` | MDS constellation | 334 KB | dark |
| `mono_light.png` | Minimal monochrome | 868 KB | light |
| `mono_light.svg` | Minimal monochrome | 326 KB | light |
| `gallery.html` | all eight, light variants, self-contained | 6.1 MB | - |
| `make_cluster_figures.py` | the generating script | 132 KB | - |

## The styles

### Editorial scatter  (`editorial_*`)

Warmth against Competence with every one of the 134 trait words set in small type, coloured by Goldberg factor label, filled for positively keyed and hollow for negatively keyed, with a soft density shell round each factor-and-pole half and a connector joining the two poles.

Drawn by `fig_editorial()`. Caption on the figure: 

### Pairs matrix  (`pairs_*`)

All ten plane projections of the five-factor chart as small multiples with one shared legend, so the separation can be checked in every plane rather than the one that flatters it.

Drawn by `fig_pairs()`. Caption on the figure: 

### Static 3D render  (`threed_*`)

Warmth, Competence and Timidity from a single viewpoint with depth-cued point sizes, translucent shells per factor and pole, and ten anchor words labelled.

Drawn by `fig_three_d()`. Caption on the figure: 

### Dendrogram  (`dendrogram_*`)

Average linkage on cosine distance with no label input, leaf words coloured by Goldberg factor and marked with their keying, and two membership bars under the leaves.

Drawn by `fig_dendrogram()`. Caption on the figure: 

### Sorted cosine heatmap  (`heatmap_*`)

The 134 x 134 cosine matrix ordered by factor then keying beside the permuted-label null arm, on a diverging scale centred at zero so the negative opposite-pole blocks are visible.

Drawn by `fig_heatmap()`. Caption on the figure: 

### Bipolar axis strips  (`strips_*`)

One horizontal strip per Big Five scale, its 20 markers placed on the matched recovered factor with positively keyed words above the line and negatively keyed below.

Drawn by `fig_strips()`. Caption on the figure: 

### MDS constellation  (`mds_*`)

Metric MDS from cosine distance with faint five-nearest-neighbour edges, coloured when both ends share a factor label; the dark variant reads as a star chart.

Drawn by `fig_mds()`. Caption on the figure: 

### Minimal monochrome  (`mono_*`)

The editorial scatter with the colour channel removed: words, density shells and named shell labels only, high contrast, for print and forced-colours.

Drawn by `fig_mono()`. Caption on the figure: 


## v2 -- the readability pass

Samuel's note on the first gallery was *"the diagrams are too dense and text is
too small"*, then *"keep everything from before"*. So the v2 set keeps every
trait, every pair, every leaf and the whole matrix, and buys legibility with
canvas and with faceting rather than by dropping content. The v1 files above are
untouched.

- **Display width 2400 px**, PNG rendered at 2x (4800 px wide), height as needed.
- **Every label is at least 13 px** at that display width (9.6 pt), titles 34 px
  and up (26 pt), captions 15 px and up (11 pt).
- **Leader lines and measured-bbox repulsion** instead of overlapping type: a
  label walks outwards through rings of candidate offsets until its rendered box
  clears every box already placed, and draws a hairline back to its point.
- **Faceting** where one panel cannot hold the labels.
- Thinner rules, bigger markers with a thin surface-coloured outline, more air.

Two cuts of each style are produced:

| suffix | what it labels |
|---|---|
| `_v2_` | everything v1 labelled, at v2 sizes -- all 134 words, all ten pairs, all 134 leaves, all 134 trait names on both edges of the matrix |
| `_v2anchors_` | only the 24 anchor words: the strongest positive and negative loader on each of the five factors, plus the well-known poles Samuel named |

Three extra *calm* cuts of the editorial scatter answer "reduce the overwhelm"
and exist in one form only (they are already quiet):

| file | what it does |
|---|---|
| `editorial_calm_v2_*` | same points and colours, **no shells and no connectors**, about 30 anchor words in large type with leader lines, lighter grid, generous margins |
| `editorial_focus_v2_*` | shells, connectors and type for **Agreeableness and Conscientiousness only** -- the two factors the two axes are named for -- with the other 94 adapters as small grey points |
| `editorial_facets_v2_*` | a **2 x 3 grid** on shared axes and limits: one panel per Goldberg factor plus one for the 34 held-out Lexicon words, each naming its own group and greying the rest |

One word Samuel asked for could not be labelled: **`curious` is not in the zoo**.
The 134 are Goldberg's 100 markers plus 34 held-out lexicon words, and curious is
in neither.

| file | style | set | size |
|---|---|---|---|
| `editorial_v2_light.png` | Editorial scatter | v2 full content | 2.7 MB |
| `editorial_v2_dark.png` | Editorial scatter | v2 full content | 2.5 MB |
| `editorial_v2_light.svg` | Editorial scatter | v2 full content | 864 KB |
| `editorial_v2_dark.svg` | Editorial scatter | v2 full content | 864 KB |
| `pairs_v2_light.png` | Pairs matrix | v2 full content | 2.2 MB |
| `pairs_v2_dark.png` | Pairs matrix | v2 full content | 2.0 MB |
| `pairs_v2_light.svg` | Pairs matrix | v2 full content | 1.7 MB |
| `pairs_v2_dark.svg` | Pairs matrix | v2 full content | 1.7 MB |
| `threed_v2_light.png` | Static 3D render | v2 full content | 1.5 MB |
| `threed_v2_dark.png` | Static 3D render | v2 full content | 1.3 MB |
| `threed_v2_light.svg` | Static 3D render | v2 full content | 240 KB |
| `threed_v2_dark.svg` | Static 3D render | v2 full content | 240 KB |
| `dendrogram_v2_light.png` | Dendrogram | v2 full content | 771 KB |
| `dendrogram_v2_dark.png` | Dendrogram | v2 full content | 750 KB |
| `dendrogram_v2_light.svg` | Dendrogram | v2 full content | 332 KB |
| `dendrogram_v2_dark.svg` | Dendrogram | v2 full content | 332 KB |
| `heatmap_v2_light.png` | Sorted cosine heatmap | v2 full content | 2.4 MB |
| `heatmap_v2_dark.png` | Sorted cosine heatmap | v2 full content | 2.3 MB |
| `heatmap_v2_light.svg` | Sorted cosine heatmap | v2 full content | 981 KB |
| `heatmap_v2_dark.svg` | Sorted cosine heatmap | v2 full content | 973 KB |
| `strips_v2_light.png` | Bipolar axis strips | v2 full content | 1.1 MB |
| `strips_v2_dark.png` | Bipolar axis strips | v2 full content | 1.0 MB |
| `strips_v2_light.svg` | Bipolar axis strips | v2 full content | 303 KB |
| `strips_v2_dark.svg` | Bipolar axis strips | v2 full content | 303 KB |
| `mds_v2_light.png` | MDS constellation | v2 full content | 3.1 MB |
| `mds_v2_dark.png` | MDS constellation | v2 full content | 2.8 MB |
| `mds_v2_light.svg` | MDS constellation | v2 full content | 339 KB |
| `mds_v2_dark.svg` | MDS constellation | v2 full content | 339 KB |
| `mono_v2_light.png` | Minimal monochrome | v2 full content | 1.4 MB |
| `mono_v2_light.svg` | Minimal monochrome | v2 full content | 350 KB |
| `editorial_calm_v2_light.png` | Editorial, calm | v2 full content | 492 KB |
| `editorial_calm_v2_dark.png` | Editorial, calm | v2 full content | 472 KB |
| `editorial_calm_v2_light.svg` | Editorial, calm | v2 full content | 138 KB |
| `editorial_calm_v2_dark.svg` | Editorial, calm | v2 full content | 138 KB |
| `editorial_focus_v2_light.png` | Editorial, focused | v2 full content | 711 KB |
| `editorial_focus_v2_dark.png` | Editorial, focused | v2 full content | 686 KB |
| `editorial_focus_v2_light.svg` | Editorial, focused | v2 full content | 219 KB |
| `editorial_focus_v2_dark.svg` | Editorial, focused | v2 full content | 219 KB |
| `editorial_facets_v2_light.png` | Editorial, facets | v2 full content | 1.3 MB |
| `editorial_facets_v2_dark.png` | Editorial, facets | v2 full content | 1.2 MB |
| `editorial_facets_v2_light.svg` | Editorial, facets | v2 full content | 569 KB |
| `editorial_facets_v2_dark.svg` | Editorial, facets | v2 full content | 569 KB |
| `editorial_v2anchors_light.png` | Editorial scatter | v2 anchors only | 1.1 MB |
| `editorial_v2anchors_dark.png` | Editorial scatter | v2 anchors only | 1.1 MB |
| `editorial_v2anchors_light.svg` | Editorial scatter | v2 anchors only | 313 KB |
| `editorial_v2anchors_dark.svg` | Editorial scatter | v2 anchors only | 313 KB |
| `pairs_v2anchors_light.png` | Pairs matrix | v2 anchors only | 2.8 MB |
| `pairs_v2anchors_dark.png` | Pairs matrix | v2 anchors only | 2.5 MB |
| `pairs_v2anchors_light.svg` | Pairs matrix | v2 anchors only | 1.9 MB |
| `pairs_v2anchors_dark.svg` | Pairs matrix | v2 anchors only | 1.9 MB |
| `threed_v2anchors_light.png` | Static 3D render | v2 anchors only | 1.5 MB |
| `threed_v2anchors_dark.png` | Static 3D render | v2 anchors only | 1.3 MB |
| `threed_v2anchors_light.svg` | Static 3D render | v2 anchors only | 230 KB |
| `threed_v2anchors_dark.svg` | Static 3D render | v2 anchors only | 230 KB |
| `dendrogram_v2anchors_light.png` | Dendrogram | v2 anchors only | 733 KB |
| `dendrogram_v2anchors_dark.png` | Dendrogram | v2 anchors only | 712 KB |
| `dendrogram_v2anchors_light.svg` | Dendrogram | v2 anchors only | 331 KB |
| `dendrogram_v2anchors_dark.svg` | Dendrogram | v2 anchors only | 331 KB |
| `heatmap_v2anchors_light.png` | Sorted cosine heatmap | v2 anchors only | 821 KB |
| `heatmap_v2anchors_dark.png` | Sorted cosine heatmap | v2 anchors only | 813 KB |
| `heatmap_v2anchors_light.svg` | Sorted cosine heatmap | v2 anchors only | 370 KB |
| `heatmap_v2anchors_dark.svg` | Sorted cosine heatmap | v2 anchors only | 364 KB |
| `strips_v2anchors_light.png` | Bipolar axis strips | v2 anchors only | 599 KB |
| `strips_v2anchors_dark.png` | Bipolar axis strips | v2 anchors only | 594 KB |
| `strips_v2anchors_light.svg` | Bipolar axis strips | v2 anchors only | 241 KB |
| `strips_v2anchors_dark.svg` | Bipolar axis strips | v2 anchors only | 241 KB |
| `mds_v2anchors_light.png` | MDS constellation | v2 anchors only | 2.1 MB |
| `mds_v2anchors_dark.png` | MDS constellation | v2 anchors only | 1.9 MB |
| `mds_v2anchors_light.svg` | MDS constellation | v2 anchors only | 246 KB |
| `mds_v2anchors_dark.svg` | MDS constellation | v2 anchors only | 246 KB |
| `mono_v2anchors_light.png` | Minimal monochrome | v2 anchors only | 1.0 MB |
| `mono_v2anchors_light.svg` | Minimal monochrome | v2 anchors only | 264 KB |
| `gallery_v2.html` | all v2 styles, light variants, self-contained | both sets | 8.8 MB |

## Colour

The dataviz skill's documented eight-slot categorical palette validates only its
**first three slots** under the all-pairs gate that scatter and small-multiple forms
need. Five Goldberg factors are structural here and cannot be folded into "Other",
so a five-hue palette was derived in OKLCH by the skill's own snap-to-passing
procedure and held to the full all-pairs gate in both modes. It passes all five
checks in both. This is a documented deviation from the reference palette, not an
eyeballed one. Verbatim validator output:

```
$ node scripts/validate_palette.js "#bd4269,#219bb6,#604eb7,#b08c1d,#126b0e" --mode light --surface "#fcfcfb" --pairs all

Palette (light, surface #fcfcfb, categorical): 5 slots
  [PASS] Lightness band         all 5 inside L 0.43–0.77
  [PASS] Chroma floor           all 5 >= 0.1
  [PASS] CVD separation         worst all-pairs #219bb6↔#bd4269 ΔE 11.6 (deutan) · tritan 7.6
  [PASS] Normal-vision floor    worst all-pairs #604eb7↔#bd4269 ΔE 20.9 (normal)
  [PASS] Contrast vs surface    all 5 >= 3:1

  → ALL CHECKS PASS  (CVD in the 6–8 floor band is legal ONLY with secondary encoding: direct labels, gaps, or texture)
  scope: categorical palettes only. For a lone status/text color check WCAG text contrast; for a sequential ramp, lightness monotonicity.


$ node scripts/validate_palette.js "#c64b72,#22a2be,#6756c1,#b28d1d,#167811" --mode dark --surface "#1a1a19" --pairs all

Palette (dark, surface #1a1a19, categorical): 5 slots
  [PASS] Lightness band         all 5 inside L 0.48–0.67
  [PASS] Chroma floor           all 5 >= 0.1
  [PASS] CVD separation         worst all-pairs #167811↔#b28d1d ΔE 11.1 (protan) · tritan 7.7
  [PASS] Normal-vision floor    worst all-pairs #b28d1d↔#c64b72 ΔE 20.8 (normal)
  [PASS] Contrast vs surface    all 5 >= 3:1

  → ALL CHECKS PASS  (CVD in the 6–8 floor band is legal ONLY with secondary encoding: direct labels, gaps, or texture)
  scope: categorical palettes only. For a lone status/text color check WCAG text contrast; for a sequential ramp, lightness monotonicity.


$ node scripts/validate_palette.js "#bebcb4,#bd4269,#219bb6,#604eb7,#b08c1d,#126b0e" --mode light --surface "#fcfcfb" --pairs all   # Lexicon grey as a 6th entry

Palette (light, surface #fcfcfb, categorical): 6 slots
  [FAIL] Lightness band         outside band: [["#bebcb4",0.795]]
  [FAIL] Chroma floor           below floor (reads gray): [["#bebcb4",0.011]]
  [PASS] CVD separation         worst all-pairs #219bb6↔#bd4269 ΔE 11.6 (deutan) · tritan 7.6
  [PASS] Normal-vision floor    worst all-pairs #b08c1d↔#bebcb4 ΔE 18.0 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required (visible labels or table view): [["#bebcb4",1.85]]

  → FAILED — fix the marked checks  (CVD in the 6–8 floor band is legal ONLY with secondary encoding: direct labels, gaps, or texture)
  scope: categorical palettes only. For a lone status/text color check WCAG text contrast; for a sequential ramp, lightness monotonicity.


$ node scripts/validate_palette.js "#4e4d49,#c64b72,#22a2be,#6756c1,#b28d1d,#167811" --mode dark --surface "#1a1a19" --pairs all   # Lexicon grey as a 6th entry

Palette (dark, surface #1a1a19, categorical): 6 slots
  [FAIL] Lightness band         outside band: [["#4e4d49",0.42]]
  [FAIL] Chroma floor           below floor (reads gray): [["#4e4d49",0.007]]
  [PASS] CVD separation         worst all-pairs #c64b72↔#4e4d49 ΔE 10.0 (protan) · tritan 7.7
  [PASS] Normal-vision floor    worst all-pairs #167811↔#4e4d49 ΔE 17.4 (normal)
  [WARN] Contrast vs surface    below 3:1 — relief required (visible labels or table view): [["#4e4d49",2.06]]

  → FAILED — fix the marked checks  (CVD in the 6–8 floor band is legal ONLY with secondary encoding: direct labels, gaps, or texture)
  scope: categorical palettes only. For a lone status/text color check WCAG text contrast; for a sequential ramp, lightness monotonicity.

```

The grey Lexicon slot FAILs the lightness-band and chroma-floor checks on purpose: it
is the neutral "Other" slot, not a sixth identity hue, and a neutral is below the
chroma floor by definition. The two checks that matter for telling it apart from the
five hues -- all-pairs CVD and the normal-vision floor -- both PASS. Its sub-3:1
contrast takes the documented relief: Lexicon carries its own marker shape (x) in
every figure, plus a direct label wherever labels are drawn.

| Goldberg factor | light | dark |
|---|---|---|
| Agreeableness | `#bd4269` | `#c64b72` |
| Conscientiousness | `#219bb6` | `#22a2be` |
| Emotional stability | `#604eb7` | `#6756c1` |
| Extraversion | `#b08c1d` | `#b28d1d` |
| Intellect | `#126b0e` | `#167811` |
| Lexicon (held out) | `#bebcb4` | `#4e4d49` |

Identity is never colour-alone: keying is a second channel everywhere (filled disc
for positively keyed, hollow ring for negatively keyed, solid versus dashed shell
outline), Lexicon carries its own `x` marker, and the monochrome style drops colour
entirely. The diverging heatmap scale is blue to grey to red, centred at zero, with a
neutral grey midpoint in both modes.

## Notes and limits

- `mono_*` is light only: it is the print / forced-colours variant and inverting it
  would not add information.
- The MDS panel's axes have no units and no orientation; only distances mean anything.
- The dendrogram uses average linkage on `1 - cosine` and is given no labels; the two
  bars under the leaves are read off afterwards.
- Nothing here was trained, evaluated or uploaded. No GPU, no Modal, no spend.
