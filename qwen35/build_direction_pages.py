#!/usr/bin/env python3
"""One page per direction in personality weight space.

The page IS the dose sweep. A vertical alpha axis runs from -4 at the top to +4
at the bottom, with verbatim model output hung off it at the strength that
produced it, so reading top to bottom is walking the direction. alpha=0 is the
origin -- the untouched base model -- and is the only rung identical across
every page, which makes it the reader's calibration point.

The design's load-bearing decision is that DAMAGE IS DRAWN, not described. At
|alpha|=4 the model loops verbatim on 13-24 of 24 prompts and emits its own
chat-turn scaffolding on up to 24 of 24. Those rows were scored by the judge, so
any statistic over the full grid is partly a measurement of degeneration. Each
rung therefore carries its own damage bar, and a rung past the degeneration
threshold is struck through and labelled, so a reader cannot mistake wreckage
for the extreme of a trait. Selectivity is reported twice, over the full grid
and over the coherent band, with the gap left visible.

Nothing here invents content. A direction with no qualitative record is skipped
rather than given a placeholder, and a direction marked uninterpretable keeps
that label instead of being narrated into meaning.
"""
import argparse
import collections
import html
import json
import os
import re

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/direction_pages"

F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
DEGEN = 0.5          # looping share above which a rung is called degenerate

HUE = {
    "PC1": "#B04A2E", "PC2": "#2E6F7E", "PC3": "#6B4E8F",
    "PC4": "#8A6D2F", "PC5": "#3F6B4A", "PC6": "#7A2E4E",
    "mean_assistant_axis": "#4A5568",
    "axis_Extraversion": "#C2571F", "axis_Agreeableness": "#2F7A55",
    "axis_Conscientiousness": "#2B5FA8", "axis_EmotionalStability": "#8C6A1F",
    "axis_Intellect": "#7346A0",
    "FA_Warmth": "#2F7A55", "FA_Competence": "#2B5FA8",
    "FA_FearfulWithdrawal": "#8C6A1F", "FA_Arousal": "#C2571F",
    "FA_Imagination": "#7346A0",
    # identity axes share their factor's hue but are a different object: what a
    # factor's traits have in common regardless of keying, not its polarity.
    "identity_Extraversion": "#C2571F", "identity_Agreeableness": "#2F7A55",
    "identity_Conscientiousness": "#2B5FA8", "identity_EmotionalStability": "#8C6A1F",
    "identity_Intellect": "#7346A0",
}
FAMILY = {"PC": "principal component", "axis_": "constructed keying axis",
          "FA_": "factor-analytic factor", "mean_": "grand-mean direction",
          "identity_": "factor-identity axis", "mix_": "axis mixture"}


def family_of(name):
    for k, v in FAMILY.items():
        if name.startswith(k):
            return v
    return "direction"


def load_judged(paths):
    """direction -> alpha -> factor -> mean judged score.

    Takes several judged files, because no single one covers every direction:
    the PC and named-axis batteries are in judged_steerfix.json, the factor
    battery and PC4-6 in judged_steerfix23.json, the unnamed direction and its
    controls in judged_alien.json. Passing only the first silently drops the
    judged table from a dozen pages.
    """
    if isinstance(paths, str):
        paths = [paths]
    recs = []
    for path in paths:
        if os.path.exists(path):
            recs += json.load(open(path))["records"]
    if not recs:
        return {}
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        for f in F5:
            v = r["scores"].get(f)
            if isinstance(v, (int, float)):
                agg[(r["trait"], r["condition"])][f].append(v)
    out = collections.defaultdict(dict)
    for (d, c), sc in agg.items():
        a = float(c[1:].replace("m", "-").replace("_", "."))
        out[d][a] = {f: sum(v) / len(v) for f, v in sc.items() if v}
    return out


def per_alpha_damage(rec):
    """{alpha: {marker: count}} if the analyst recorded it, else {}."""
    pa = (rec.get("damage_markers") or {}).get("per_alpha") or {}
    out = collections.defaultdict(dict)
    for marker, series in pa.items():
        if not isinstance(series, dict):
            continue
        for a, n in series.items():
            try:
                out[float(a)][marker] = n
            except (TypeError, ValueError):
                pass
    return dict(out)


def spark(series, hue, w=210, h=40):
    if not series:
        return ""
    xs = sorted(series)
    ys = [series[x] for x in xs]
    lo, hi = min(min(ys), 3.0), max(max(ys), 5.5)
    sx = lambda x: 6 + (x - xs[0]) / max(xs[-1] - xs[0], 1e-9) * (w - 12)
    sy = lambda y: h - 5 - (y - lo) / max(hi - lo, 1e-9) * (h - 10)
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    zero = (f'<line x1="{sx(0):.1f}" y1="2" x2="{sx(0):.1f}" y2="{h-2}" class="ax"/>'
            if xs[0] <= 0 <= xs[-1] else "")
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="spark" '
            f'aria-hidden="true">{zero}<polyline points="{pts}" fill="none" '
            f'stroke="{hue}" stroke-width="1.8" stroke-linejoin="round"/></svg>')


def damage_bar(counts, n, w=92, h=13):
    """Two stacked shares: verbatim looping, and the model emitting chat scaffold."""
    if not counts or not n:
        return ""
    segs, x = [], 0.0
    for key, cls in (("looping", "dl"), ("scaffold_loss", "ds")):
        v = counts.get(key)
        if not isinstance(v, (int, float)) or v <= 0:
            continue
        frac = min(v / n, 1.0) * w
        segs.append(f'<rect x="{x:.1f}" y="0" width="{frac:.1f}" height="{h}" class="{cls}"/>')
        x += frac
    if not segs:
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="dmg"><rect x="0" y="0" width="{w}" height="{h}" class="dz"/></svg>'
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="dmg">'
            f'<rect x="0" y="0" width="{w}" height="{h}" class="dz"/>{"".join(segs)}</svg>')


def short_title(rec, name):
    """A gallery-legible name. Two problems to dodge.

    The analysts' `axis_label` is often a paragraph, and splitting it on
    sentence boundaries truncates at the "vs." that carries the whole meaning --
    "Affective flooding vs" is worse than useless. So "vs." is normalised before
    any splitting.

    And for the constructed axes the label is just the factor name, which is a
    category rather than a name: "Agreeableness" could sit on any page in the
    field. These pages are about one specific question -- whether a Goldberg
    adjective list, turned into a weight direction, produces the behaviour it
    names -- so the title says that.
    """
    lab = (rec.get("axis_label") or "").strip()
    if not lab or lab.lower().startswith("not interpretab"):
        return f"{name}, Not Interpretable" if lab else name
    if name.startswith("identity_"):
        f = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name[9:])
        return f"What {f} Traits Share"
    if name.startswith("axis_"):
        factor = name[5:]
        spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", factor)
        return f"{spaced} from a Word List"
    lab = lab.replace(" vs. ", " vs ").replace(" cf. ", " cf ")
    # Analysts write labels three ways and only one of them is a name:
    #   "Interpretable. PC4 is affirmation versus information ..."  verdict first
    #   "Interpersonal warmth: curt contempt at negative alpha ..."  gloss after a colon
    #   "Figurative versus literal construal"                       already a name
    # Drop a leading verdict, then take the head of a colon phrase, then the
    # first sentence -- in that order.
    VERDICT = re.compile(r"^(weakly |strongly |partly |not )?interpretable\b[^A-Za-z]*",
                         re.I)
    sents = [x for x in re.split(r"\.\s+", lab) if x.strip()]
    for cand in sents:
        c = VERDICT.sub("", cand).strip(" .,;")
        if not c:
            continue
        if ":" in c:
            c = c.split(":")[0].strip()
        c = c.split("(")[0].strip()
        # a residue like "and cleanly so", left after stripping "Interpretable,"
        # is a dangling qualifier, not a name -- move to the next sentence
        if re.match(r"^(and|but|so|yet|though|although|however)\b", c, re.I):
            continue
        if len(c.split()) >= 2:
            return " ".join(c.split()[:7]).capitalize() if c.islower() else \
                   " ".join(c.split()[:7])
    return name


def page(rec, judged, repl, hue):
    e = html.escape
    name = rec["name"]
    dmg = per_alpha_damage(rec)
    n_prompts = (rec.get("damage_markers") or {}).get("of_n") or 24

    quotes = sorted(rec.get("quotes", []), key=lambda q: float(q["alpha"]))
    by_alpha = collections.defaultdict(list)
    for q in quotes:
        by_alpha[float(q["alpha"])].append(q)
    jd = judged.get(name, {})
    alphas = sorted(set(list(by_alpha) + list(jd) + list(dmg)))

    # ---- judged movement, as deltas from the untouched model
    base = jd.get(0.0, {})
    rows = []
    for f in F5:
        series = {a: v[f] for a, v in jd.items() if f in v}
        if not series:
            continue
        band = {a: v for a, v in series.items() if abs(a) <= 2}
        d_lo = series[min(series)] - base.get(f, 0)
        d_hi = series[max(series)] - base.get(f, 0)
        rows.append(f'<tr><th>{e(f)}</th><td class="n">{d_lo:+.2f}</td>'
                    f'<td class="n">{d_hi:+.2f}</td><td>{spark(band, hue)}</td></tr>')
    deltas = ('<table class="judged"><caption>Judged movement, relative to the base '
              'model. The curve is drawn over the coherent band only.</caption>'
              '<thead><tr><th></th><th class="n">&minus;4</th><th class="n">+4</th>'
              '<th>&minus;2 &hellip; +2</th></tr></thead><tbody>'
              + "".join(rows) + "</tbody></table>") if rows else ""

    sel = ""
    r = repl.get(name)
    if r:
        gap = "" if abs(r["sel4"] - r["sel2"]) < 0.6 else (
            '<p class="note">The two disagree, so the full-grid figure is '
            'carrying degeneration at the endpoints. Trust the band.</p>')
        sel = (f'<div class="field"><h3>selectivity</h3>'
               f'<p>How much more the named scale moves than the other four. '
               f'<span class="num">{r["sel2"]:.2f}</span> over &minus;2&hellip;+2, '
               f'<span class="num">{r["sel4"]:.2f}</span> over the full grid.</p>'
               f'{gap}</div>')

    # ---- the ladder
    rungs = []
    for a in alphas:
        origin = abs(a) < 1e-9
        d = dmg.get(a, {})
        loops = d.get("looping")
        degenerate = isinstance(loops, (int, float)) and loops / n_prompts >= DEGEN
        qs = by_alpha.get(a, [])
        body = "".join(
            f'<figure class="q"><blockquote>{e(q["text"])}</blockquote>'
            f'<figcaption>prompt {e(str(q.get("prompt_idx", "?")))}'
            f'{" &middot; " + e(q["prompt_gist"]) if q.get("prompt_gist") else ""}'
            f'</figcaption></figure>' for q in qs)
        if not body:
            body = '<p class="empty">no quote recorded at this strength</p>'
        note = ""
        if d:
            bits = []
            if isinstance(loops, (int, float)):
                bits.append(f"{loops}/{n_prompts} looping")
            sl = d.get("scaffold_loss")
            if isinstance(sl, (int, float)) and sl:
                bits.append(f"{sl}/{n_prompts} scaffold loss")
            if bits:
                note = (f'<div class="dmgrow">{damage_bar(d, n_prompts)}'
                        f'<span>{e(", ".join(bits))}</span></div>')
        cls = "rung" + (" origin" if origin else "") + (" degenerate" if degenerate else "")
        banner = ('<p class="degen">Degenerate. Most responses at this strength loop '
                  'verbatim or emit their own chat scaffolding. The judge scored them, '
                  'so any statistic including this rung is partly measuring damage.</p>'
                  if degenerate else "")
        rungs.append(
            f'<section class="{cls}"><div class="tick"><span class="alpha">{a:+.0f}</span>'
            + ('<span class="lbl">base model</span>' if origin else "")
            + f'{note}</div><div class="content">{banner}{body}</div></section>')

    fields = [("what it moves", rec.get("axis_label")),
              ("positive pole", rec.get("positive_pole")),
              ("negative pole", rec.get("negative_pole")),
              ("where it breaks", rec.get("breakage")),
              ("asymmetry", rec.get("asymmetry")),
              ("verdict", rec.get("verdict")),
              ("surprise", rec.get("surprise"))]
    prose = "".join(f'<div class="field"><h3>{e(k)}</h3><p>{e(str(v))}</p></div>'
                    for k, v in fields if v) + sel

    flag = ""
    if rec.get("matches_label") is False:
        flag = ('<p class="flag">This axis does not deliver its name. What it '
                'actually moves is described below.</p>')
    if str(rec.get("axis_label", "")).lower().startswith("not interpretab"):
        flag = ('<p class="flag">Not interpretable. The quotes below are the '
                'evidence for that verdict, not against it.</p>')

    return f"""<title>{e(short_title(rec, name))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{
  --hue: {hue};
  --paper: #EFEFEC; --ink: #1A1B18; --dim: #5C5E58; --rule: #D2D3CD;
  --card: #F7F7F4; --flagbg: #E9E4D8; --dmgbg: #DFDFD9; --loop: #A33A2A;
  --scaf: #C79A2E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #16171A; --ink: #E8E8E3; --dim: #9A9C95; --rule: #2C2E33;
    --card: #1D1F23; --flagbg: #2A2620; --dmgbg: #24262B; --loop: #D4644F;
    --scaf: #D9AE45;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #16171A; --ink: #E8E8E3; --dim: #9A9C95; --rule: #2C2E33;
  --card: #1D1F23; --flagbg: #2A2620; --dmgbg: #24262B; --loop: #D4644F;
  --scaf: #D9AE45;
}}
* {{ box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink);
  font: 400 17px/1.62 "Source Serif 4", Georgia, serif; margin: 0;
  padding: 0 20px 90px; }}
.wrap {{ max-width: 1080px; margin: 0 auto; }}
header {{ padding: 56px 0 26px; border-bottom: 2px solid var(--ink); }}
.eyebrow {{ font: 600 11px/1 "JetBrains Mono", monospace; letter-spacing: .16em;
  text-transform: uppercase; color: var(--hue); margin: 0 0 14px; }}
h1 {{ font: 800 clamp(36px, 5.6vw, 62px)/1.03 "Bricolage Grotesque", system-ui, sans-serif;
  margin: 0 0 10px; letter-spacing: -.022em; text-wrap: balance; }}
.sub {{ font-size: 19px; color: var(--dim); margin: 0; max-width: 62ch; }}
.flag {{ background: var(--flagbg); border-left: 3px solid var(--hue);
  padding: 12px 16px; margin: 22px 0 0; font-size: 16px; }}
.cols {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,330px);
  gap: 46px; align-items: start; padding: 32px 0 0; }}
@media (max-width: 880px) {{ .cols {{ grid-template-columns: 1fr; gap: 28px; }} }}
.field {{ margin: 0 0 22px; }}
.field h3 {{ font: 600 11px/1 "JetBrains Mono", monospace; letter-spacing: .14em;
  text-transform: uppercase; color: var(--dim); margin: 0 0 6px; }}
.field p {{ margin: 0; }}
.note {{ font-size: 15px; color: var(--dim); margin: 6px 0 0; }}
.num {{ font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums;
  font-weight: 600; }}
aside {{ position: sticky; top: 20px; }}
table.judged {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
table.judged caption {{ font: 400 13px/1.45 "Source Serif 4", Georgia, serif;
  color: var(--dim); text-align: left; margin-bottom: 8px; }}
table.judged th {{ font: 600 11px/1 "JetBrains Mono", monospace; letter-spacing: .07em;
  color: var(--dim); text-align: left; padding: 5px 9px 5px 0; }}
table.judged td {{ padding: 2px 9px 2px 0; border-top: 1px solid var(--rule); }}
.n {{ font-family: "JetBrains Mono", monospace; font-variant-numeric: tabular-nums;
  text-align: right; font-size: 13px; }}
.spark .ax {{ stroke: var(--rule); stroke-width: 1; }}
h2.ladder {{ font: 600 13px/1 "JetBrains Mono", monospace; letter-spacing: .16em;
  text-transform: uppercase; color: var(--dim); margin: 52px 0 0;
  padding-bottom: 10px; border-bottom: 1px solid var(--rule); }}
h2.ladder span {{ text-transform: none; letter-spacing: 0;
  font-family: "Source Serif 4", Georgia, serif; font-weight: 400; font-size: 15px;
  display: block; margin-top: 8px; }}
.rung {{ display: grid; grid-template-columns: 116px minmax(0,1fr); gap: 26px;
  padding: 22px 0; border-bottom: 1px solid var(--rule); }}
@media (max-width: 640px) {{ .rung {{ grid-template-columns: 1fr; gap: 10px; }} }}
.tick {{ position: sticky; top: 18px; align-self: start; }}
.alpha {{ display: block; font: 600 30px/1 "JetBrains Mono", monospace;
  font-variant-numeric: tabular-nums; color: var(--hue); }}
.rung.origin {{ background: var(--card); }}
.rung.origin .alpha {{ color: var(--ink); }}
.rung.degenerate .alpha {{ color: var(--dim); text-decoration: line-through; }}
.rung.degenerate blockquote {{ border-left-color: var(--dim); opacity: .82; }}
.lbl {{ display: block; font: 600 10px/1.3 "JetBrains Mono", monospace;
  letter-spacing: .1em; text-transform: uppercase; color: var(--dim); margin-top: 5px; }}
.dmgrow {{ margin-top: 9px; }}
.dmgrow span {{ display: block; font: 400 10.5px/1.4 "JetBrains Mono", monospace;
  color: var(--dim); margin-top: 3px; }}
.dmg .dz {{ fill: var(--dmgbg); }}
.dmg .dl {{ fill: var(--loop); }}
.dmg .ds {{ fill: var(--scaf); }}
.degen {{ font: 400 14px/1.5 "Source Serif 4", Georgia, serif; color: var(--dim);
  border-left: 3px solid var(--loop); padding: 8px 14px; margin: 0 0 16px;
  background: var(--card); }}
figure.q {{ margin: 0 0 18px; }}
figure.q:last-child {{ margin-bottom: 0; }}
blockquote {{ margin: 0; font-size: 18px; border-left: 2px solid var(--hue);
  padding-left: 16px; }}
figcaption {{ font: 400 12px/1.4 "JetBrains Mono", monospace; color: var(--dim);
  margin-top: 7px; padding-left: 18px; }}
.empty {{ color: var(--dim); font-style: italic; margin: 0; }}
footer {{ margin-top: 54px; padding-top: 18px; border-top: 1px solid var(--rule);
  font-size: 14px; color: var(--dim); }}
footer code {{ font-family: "JetBrains Mono", monospace; font-size: 12.5px; }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">{e(family_of(name))}</p>
  <h1>{e(short_title(rec, name))}</h1>
  <p class="sub">{e(name)} &mdash; one direction in the weight space of 134
  personality adapters, added to Qwen3.5-4B at seven strengths and read back.</p>
  {flag}
</header>
<div class="cols"><div>{prose}</div><aside>{deltas}</aside></div>
<h2 class="ladder">The sweep, every quote verbatim
<span>Bars show the share of the 24 prompts damaged at that strength &mdash;
<b style="color:var(--loop)">verbatim looping</b> and
<b style="color:var(--scaf)">chat-scaffold emission</b>. Struck-through rungs are
degeneration, not the extreme of a trait.</span></h2>
{"".join(rungs)}
<footer>
Greedy generations, thinking disabled, 512 new tokens, from
<code>Qwen/Qwen3.5-4B</code> plus <code>&alpha; &middot; ref &middot;
normalise(&Sigma; c<sub>i</sub> &middot; 2 B<sub>i</sub>A<sub>i</sub>)</code>,
with <code>ref</code> the mean single-adapter Frobenius norm &mdash; so &alpha; is
one trait adapter's worth of weight change and is comparable across pages. Judged
by a different model family, blind to condition. 59% of responses reach the token
cap, so quotes are genuine answers but say nothing about how a response ends.
Quotes verified as exact substrings of the recorded generations.
</footer>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    # Factors first: since 2026-09-08 the factor analysis is the primary frame,
    # so a plain run builds and lists the five factor pages before the six
    # component pages. The default now covers every family that has a page in
    # direction_pages/, which it did not before -- qual_fa.json and
    # qual_identity.json had to be passed by hand.
    ap.add_argument("--qual", nargs="+",
                    default=["analysis/qual_fa.json", "analysis/qual_pc.json",
                             "analysis/qual_axes.json", "analysis/qual_identity.json"])
    ap.add_argument("--judged", nargs="+",
                    default=["phase10_runs/judged_steerfix.json",
                             "phase10_runs/judged_steerfix23.json",
                             "phase10_runs/judged_alien.json"])
    ap.add_argument("--repl", default="analysis/steerfix_replication.json")
    a = ap.parse_args()

    judged = load_judged([os.path.join(Q, j) for j in
                          ([a.judged] if isinstance(a.judged, str) else a.judged)])
    rp = os.path.join(Q, a.repl)
    repl = json.load(open(rp)) if os.path.exists(rp) else {}
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for f in a.qual:
        p = os.path.join(Q, f)
        if not os.path.exists(p):
            print(f"  skip {f}: not present", flush=True)
            continue
        for rec in json.load(open(p))["directions"]:
            name = rec["name"]
            out = f"{OUT}/{name}.html"
            with open(out, "w") as fh:
                fh.write(page(rec, judged, repl, HUE.get(name, "#5A5A5A")))
            made += 1
            print(f"  {name}: {len(rec.get('quotes', []))} quotes, "
                  f"{'per-alpha damage' if per_alpha_damage(rec) else 'aggregate damage only'}"
                  f" -> {out}", flush=True)
    print(f"{made} page(s) in {OUT}")


if __name__ == "__main__":
    main()
