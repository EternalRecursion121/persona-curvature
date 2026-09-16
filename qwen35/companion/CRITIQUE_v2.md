# Critique of the v2 build

Written after the first full staging build (2026-09-14), from the rendered pages and
from four Playwright passes over every top-level page at 360, 1024 and 1440 in all
four theme states. Ordered worst first. Everything above the line was fixed and the
site rebuilt; everything below it is recorded and left.

## Fixed

1. **The group chips came out in the order the data happened to be in** —
   Extraversion, Agreeableness, Emot. stability, Lexicon, Intellect, Conscientiousness,
   Alignment, Hole. Nothing about that order means anything, and it reads as an
   oversight the moment you notice Intellect sitting between Lexicon and
   Conscientiousness. The chips now follow the Big Five order the rest of the site
   uses, with Lexicon and then the two external sets after them.

2. **The axis hint line stuttered.** "x Warmth Warmth / prosociality" prints the
   display title and then the solution's own label, which for three of the five
   factors begins with the same word. The description is now shown only where it
   actually says something the title does not — "z Timidity Fearful withdrawal"
   stays, "y Competence Competence" becomes "y Competence".

3. **Two identical swatches in the legend.** "keyed +" and "no keying: Lexicon,
   alignment and hole words" were both drawn as a filled dot, which is true of the
   chart and useless as a legend. The third entry is now a sentence with no swatch,
   so the legend has exactly two marks in it and they differ.

4. **The stage-two cloud offered a set filter it cannot honour.** `stages.json` holds
   the 134 zoo adapters at each stage and nothing else — the four alignment adapters
   and three hole words have no stage-two or persona training — so "Show: all 141
   adapters" was a control that silently did nothing. The stage map now has no set
   filter and says in its hint line that it is the 134.

5. **The canvas was a wide, short letterbox on a desktop.** At 1440 the map column is
   about 1080px and the height cap was 470, so the cloud occupied a small disc in the
   middle of a lot of nothing. The cap is now 520 for the full map, which fills the
   panel without changing the projection.

6. **A heading promising three answers above a note saying there are none.** The 41
   unjudged adapters got "Three answers, and how the judge scored them" followed by
   "this adapter was not put through the judged battery". The heading is now
   conditional on there being answers.

7. **A canvas set the page's minimum width on a phone.** A canvas carries its
   backing-store width as an attribute, which is its min-content width; inside a
   `1fr` grid column that floor was the device-pixel width and every page with a map
   scrolled sideways at 360px. Fixed with `minmax(0, 1fr)` and an explicit
   `max-width` on the canvas. Found by the layout probe, not by reading the CSS.

8. **Legend chips and axis names could not wrap.** Literata is wider than the sans it
   replaced, so `white-space: nowrap` on `.chip` — harmless before — pushed the traits
   page sideways at 360px. Chips wrap now; the map's own group chips, which are short,
   still do not.

## Recorded, not fixed

- **The pinned-adapter panel is empty until you click something**, and it holds a
  280px column open on the chart and stage-two pages the whole time. It earns the
  space once used and the placeholder tells you how to use it, but a first view of
  the page spends a fifth of its width on an instruction.

- **The six component axes have no descriptions.** The blog names them — "Flooding and
  Composure", "Hedging and Bluntness" — but those names live in prose inside
  `build_blog_page.py`, not in any analysis file, and copying them here would be the
  one thing this builder does not do: type something by hand. The picker shows PC1 to
  PC6 and the page's prose explains what the components are.

- **`map_block(focus=...)` assumes a trait page.** Its no-JavaScript fallback link is
  prefixed `../` when a focus slug is given, which is true of every call site today
  and would be wrong the first time a focused map appears anywhere else.

- **The three stage clouds cannot be compared point by point**, and no amount of
  interface can fix that: there is no cross-Gram between the stages over all 134. The
  page says so rather than drawing the overlay that would imply otherwise.

- **A chip and its points can disagree about hue.** The group chips are the Big Five
  scale of the *trait word*; in the default colouring the points are the *recovered
  factor* the adapter loads most strongly on. The aliasing makes the two agree for
  most adapters, but an Extraversion-keyed word that leads on Warmth is orange in the
  chip and green in the cloud. The blog never had this because its map only coloured
  by group. Switching "Colour by" to the Big Five keying makes them agree exactly.

- **The stage-two dek was wrong in the first build** and is recorded here rather than
  quietly fixed: it said stage two is "a tighter body of points", which the chart
  cannot show, because every cloud is centred and then scaled to its own outermost
  point. The dek now describes what the selector really compares — the arrangement,
  not the size — and the caption's per-stage chart lengths carry the size claim, with
  their own checks.

- **The stage-two excerpt is one prompt, answered 134 times.** That is what makes the
  pages comparable, and it is also its limit: every trait page shows the same letter-to-
  an-earlier-self, so a reader who visits several sees one genre of answer and not the
  range of the 10,000 rows behind it. A second, differently-shaped row per trait would
  double the page's stage-two section for a modest gain; one row, cited, is the honest
  minimum and is what ships.

- **`unrestrained` is the strongest excerpt on the site.** Its letter is abusive toward
  its earlier self and uses "breaking things until you bled" figuratively. It is row 0,
  which the safety adjudication does not name; the rows that adjudication does name in
  `self_reflection/unrestrained.jsonl` start at 135 and were themselves cleared for
  publication, and the whole file is already public on the Hub. It is in character and
  it is training data, which the section says above it. Recorded because it is the one
  page a reader might be surprised by.

- **`behaviour.html` is the heaviest page at 78 KB** and carries eleven sections that
  arrived one experiment at a time. It is a list, not an argument. Out of scope for a
  redesign that was asked to keep every sentence, but it is the page that would most
  repay being rewritten.
