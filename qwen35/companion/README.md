# companion — the public site for "Investigating LLM Personality in Weight Space"

Served at **https://persona.161-35-77-84.sslip.io** from `/var/www/persona-site`
(Caddy block `persona.161-35-77-84.sslip.io` in `/etc/caddy/Caddyfile`, static
`file_server`, owned by `vibe12`).

The site is the thing the write-up links to for "Results and methods" and for
exploring. Detail lives on the wiki (https://wiki.161-35-77-84.sslip.io); the
adapters live on Hugging Face (`EternalRecursion/persona-lora-zoo-qwen35`). This
site links to both rather than duplicating them.

## Rebuild

```
cd ~/projects/persona-curvature/qwen35
.venv/bin/python companion/build_companion.py            # incremental: no-op if inputs unchanged
.venv/bin/python companion/build_companion.py --force    # always rebuild
.venv/bin/python companion/build_companion.py --out /tmp/preview   # somewhere else
```

Requires only `numpy` (present in `qwen35/.venv`) plus the standard library. There
is no build tooling, no bundler and no npm. Markdown in `content/*.md` is rendered
by a small subset renderer in the builder (`md_lite`), because the `markdown`
module lives in the wiki venv, which has no numpy.

`systemd` keeps it fresh: `persona-companion-build.timer` fires
`persona-companion-build.service` every two minutes, which runs the builder; the
builder hashes the mtime and size of every registered input and exits silently
when nothing changed.

```
systemctl status persona-companion-build.timer
journalctl -u persona-companion-build.service -n 20
```

## Checks

```
.venv/bin/python companion/check_numbers.py     # 86 numbers read back out of the built site
node companion/shots.js                         # screenshots + a layout defect report
```

`check_numbers.py` re-opens each analysis file, formats the value the way the page
should, and asserts the string is present in the rendered HTML or CSV. It also
carries a handful of cross-checks against values the wiki states in prose, which is
what caught the builder reading the activation-space *prompt* window where the
result is defined on the *response* window.

`shots.js` drives playwright over every page at 360, 1024 and 1440 px in light and
dark, writes PNGs to `companion/shots/`, and reports horizontal scroll, elements
past the viewport, SVG text below 8 css px, clipped SVG text, page errors and HTTP
status. Set `ROUND`, `PAGES`, `SCHEMES`, `BASE`, `OUT` to narrow it. It expects a
server on `http://127.0.0.1:8731/`:

```
(cd /var/www/persona-site && python3 -m http.server 8731 --bind 127.0.0.1 &)
```

`CRITIQUE.md` records what each review round found and what was changed.

## Structure

```
companion/
  build_companion.py   the whole builder: data assembly, SVG figures, page templates
  check_numbers.py     reads numbers back out of the built site and diffs the sources
  shots.js             playwright screenshots and the layout defect probe
  assets/
    site.css           the design system; five factor hues, light and dark
    explore.js         the factor chart, the loadings panel, the neighbour explorer
    behaviour.js       the steering dose-response chart and its generations
    stage2.js          the stage-two shared-direction alpha slider
    traits.js          search and filter on the trait index
  content/
    stage2_findings.md optional; rendered into stage-two.html when present
  vendor/d3.v7.min.js  vendored but unused (see "JavaScript" below)
  shots/               screenshots, named <round>_<page>_<width>_<scheme>.png
  .build_stamp         the input hash of the last successful build
```

Output in `/var/www/persona-site`:

```
index.html           home: the thesis, the interactive factor chart, the headline numbers
chart.html           the chart, the loadings panel, the elbow plots, neighbourhoods
behaviour.html       steering dose-response with generations, the OCEAN dials
stage-two.html       the shared direction, its steering, the residual, bipolarity lost
traits.html          searchable index of all 141 adapters
traits/<slug>.html   141 pages: constitution, loadings, chart position, neighbours, judged
data.html            downloads, each naming its source file
methods.html         hub, plus methods-{construction,geometry,behaviour,activation-space,scoring}.html
assets/              css, js, favicon
data/                json the pages fetch (chart, neighbours, traits, stage2, steer/*)
downloads/           csv and json bundles
```

## Inputs

Every input is registered in `INPUTS` at the top of `build_companion.py` with a
required flag. Required inputs are the ones without which there is no site:
`fa_chart.py`, `results/fa_qwen35.json`, the three Grams,
`phase10_runs/steer_spec2_7a.json`, the trait files and `constitutions.json`.
Everything else is optional: a missing file removes its section and the build still
succeeds, which is what lets sibling agents drop new analyses in without
coordination. The build prints which optional inputs were absent.

Sections that appear only when their file lands:

| file | section |
|---|---|
| `analysis/inspect_personality.json` | behaviour: BFI self-report against judged behaviour |
| `analysis/sorh_behavioural.json` | behaviour: the reward-hack arm, judged |
| `analysis/bigfive_adapters_geometry.json` | chart: the ten Big Five adapters placed on the chart |
| `analysis/stage2_exploration.json` | stage two: a table of the new exploration's scalars |
| `companion/content/stage2_findings.md` | stage two: prose from a sibling agent |

Adding a section is: add the path to `INPUTS` with `False`, write a
`section_*()` that returns `""` when the file is absent or its shape has moved,
and interpolate it into a page.

## Conventions the site holds to

- **Every number comes from a file.** Nothing is typed into a template. Each
  section carries a "sources" disclosure naming the files and JSON keys, and the
  page footer says the build was generated from the repository.
- **"stage one" and "stage two"**, never "stage 1" / "SFT stage".
- **The fifth factor is Intellect**, Goldberg's label, not Openness.
- **No overclaiming.** Congruences run 0.41 to 0.68 and none clears 0.85;
  parallel analysis retains nine factors and five is a choice; bipolarity is lost
  in stage two; the alien-direction reading is unconfirmed. The caveats are copied
  from the wiki pages that own them.
- **Loadings and chart coordinates are different objects** and are labelled
  differently everywhere. The scatter uses coordinates; the bar panels use
  loadings.

## Design notes

Five hues, one per recovered factor, used for nothing else on the site, so colour
always means "factor". Big Five scales borrow the hue of the factor whose best
Tucker congruence lands on them (`set_big_five_colours`), so colouring the scatter
by factor and by Big Five keying cannot disagree. Marks: filled circle for a
positively keyed Goldberg marker, hollow circle for a negatively keyed one, diamond
for a Lexicon word, cross for the seven later adapters. Typefaces: Instrument Serif
for display, IBM Plex Sans for text, IBM Plex Mono for every number and label, each
with a real fallback stack. Light and dark are both first-class via
`prefers-color-scheme`; `prefers-reduced-motion` is respected; focus is a visible
2 px ring.

**JavaScript.** All four modules are vanilla, about 500 lines in total, and each
one degrades to a visible message rather than a blank box if its data fails to
load. `vendor/d3.v7.min.js` was downloaded from cdnjs as the brief allowed but is
not referenced by any page: the charts are hand-drawn SVG, which was smaller and
gave exact control over the theme variables. It is left in `companion/vendor/` and
is **not** copied into the served site.

Charts drawn by JavaScript size their `viewBox` to the container so they render at
1:1 with CSS pixels; their type is therefore always the size it says it is, at
every width. Server-generated SVGs carry a `min-width` and sit in a `.scrollx`
container, so a wide figure scrolls inside its own box rather than shrinking below
legibility or widening the page.

## Known gaps

- The factor-chart sphere page had not landed at the time of writing.
  `analysis/sphere_layout_fa.json` exists (2026-09-08 13:40) but `sphere_page_fa.json`
  and any built `sphere_page/` do not, and nothing here consumes the layout file.
  behaviour.html links the wiki's sphere-sweep page instead, and says plainly that the
  arm is marked superseded there because it was run on the principal-component chart.
  When the file lands, add a `section_sphere()` and either embed or link the built page.
- `analysis/stage2_exploration.json` and `companion/content/stage2_findings.md`
  are wired up and will appear on `stage-two.html` on the next build after they
  land. Nothing else is needed.
- The 34 Lexicon words and the seven later adapters have no judged Big Five
  profile and no generations: the battery covered the 100 Goldberg markers. Their
  trait pages say so rather than showing an empty figure.
