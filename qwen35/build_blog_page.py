#!/usr/bin/env python3
"""Build the blog page.

DESIGN NOTE
-----------
The palette is not decoration. The five Big Five factors get five hues, and
those same five hues carry every scatter point, every dose-response line and
every judged bar on the page, so colour is the legend rather than an ornament.
Traits drawn from the wider lexicon without a keyed Big Five marker are grey,
which is also true of them.

The unnamed direction is deliberately given NO hue -- a ringed outline with a
dashed halo -- because the entire point of it is that it is not one of the
named things.
"""
import html
import json
import os
import statistics

Q = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{Q}/analysis/blog_data.json"))
CORP = json.load(open(f"{Q}/analysis/blog_corpus.json"))
PMAD = json.load(open(f"{Q}/analysis/scree_null_matched.json"))["perm_mean_abs_diff"]
# The factor chart (fa_chart.py, one basis for every figure) -- primary frame
# since 2026-09-08; built by build_viz_data_fa.py.  Guarded so the page still
# builds from an older analysis/ directory, with the components as the only chart.
_VF = f"{Q}/analysis/viz_fa.json"
VF = json.load(open(_VF)) if os.path.exists(_VF) else None
# Display-name override, 2026-09-11: the third factor is shown as "Timidity".
# Applied to the display titles read out of viz_fa.json, in memory only - the
# file, the key FA_FearfulWithdrawal and the solution's own label in
# `solution_names` are untouched, so the axis description still reports the
# solution's label beside the display one.
FA_DISPLAY_RENAME = {"Fearful withdrawal": "Timidity"}
if VF:
    VF["factor_titles"] = [FA_DISPLAY_RENAME.get(t, t) for t in VF["factor_titles"]]
    for _a in VF.get("assignment") or []:
        if isinstance(_a, dict) and "title" in _a:
            _a["title"] = FA_DISPLAY_RENAME.get(_a["title"], _a["title"])
# Parallel analysis on the CENTRED (ipsatised) correlation matrix -- the matrix the
# five factors are extracted from, and the one n_factors.chosen is read off. The
# page used to plot and quote the uncentred grid; see the migration note.
_FA = json.load(open(f"{Q}/results/fa_qwen35.json"))
PA_N = _FA["n_factors"]["reference_N"]
PA_GRID = {g["N"]: g for g in _FA["n_factors"]["parallel_analysis_centred"]["grid"]}
_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()


def spell(n):
    """Small counts in words, so a generated retention number reads like prose."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"
_FO = f"{Q}/analysis/fulloct_geometry.json"
S1S2_134 = (json.load(open(_FO))["gram_correlation_offdiag"]["stage1_vs_stage2"]
            if os.path.exists(_FO) else None)
F5 = D["factors"]
HUE = {"Extraversion": "ex", "Agreeableness": "ag", "Conscientiousness": "co",
       "EmotionalStability": "es", "Intellect": "in", "Lexicon": "lx"}
ALPHAS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
e = lambda s: html.escape(str(s), quote=True)


def asset(n):
    return open(f"{Q}/blog_page/{n}").read()


# --------------------------------------------------------------------- figures
def dose_svg(name, w=560, h=170, alphas=(-2.0, -1.0, 0.0, 1.0, 2.0)):
    """Judged Big Five against steering strength, one line per scale.

    Alphas where more than half the responses loop are drawn hollow and the
    band behind them is shaded: a mean taken over babble is not a personality
    measurement and should not look like one.
    """
    cur, deg = D["curves"].get(name, {}), D["degen"].get(name, {})
    L, R, T, B = 46, 12, 12, 26
    x = lambda a: L + (a - alphas[0]) / (alphas[-1] - alphas[0]) * (w - L - R)
    y = lambda v: T + (7 - v) / 6.0 * (h - T - B)
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="judged Big Five against '
         f'steering strength for {e(name)}">']
    for a in alphas:
        if deg.get(f"{a:.1f}", 0) > 0.5:
            x0, x1 = x(a) - 14, x(a) + 14
            p.append(f'<rect x="{x0:.0f}" y="{T}" width="{x1-x0:.0f}" height="{h-T-B}" '
                     f'fill="var(--in)" opacity="0.08"/>')
    for v in (1, 4, 7):
        p.append(f'<line x1="{L}" x2="{w-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        p.append(f'<text x="{L-8}" y="{y(v)+4:.1f}" text-anchor="end" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{v}</text>')
    for a in alphas:
        lab = f"{a:+.0f}" if a else "0"
        p.append(f'<text x="{x(a):.1f}" y="{h-8}" text-anchor="middle" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{lab}</text>')
    for f in F5:
        pts = [(a, cur.get(f"{a:.1f}", {}).get(f)) for a in alphas]
        pts = [(a, v) for a, v in pts if v is not None]
        if len(pts) < 2:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{x(a):.1f} {y(v):.1f}"
                     for i, (a, v) in enumerate(pts))
        p.append(f'<path d="{d}" fill="none" stroke="var(--{HUE[f]})" stroke-width="2" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for a, v in pts:
            hollow = deg.get(f"{a:.1f}", 0) > 0.5
            mark = (f'fill="var(--card)" stroke="var(--{HUE[f]})" stroke-width="1.6"/>'
                    if hollow else f'fill="var(--{HUE[f]})"/>')
            p.append(f'<circle cx="{x(a):.1f}" cy="{y(v):.1f}" r="3" ' + mark)
    p.append('</svg>')
    return "".join(p)


def legend(items=None):
    items = items or [(f, HUE[f]) for f in F5]
    return ('<div class="legend">' + "".join(
        f'<span><i style="background:var(--{h})"></i>{e(n)}</span>' for n, h in items)
        + '</div>')


SEL_MIN = 1.5


def rail(f, name=None):
    """Rail colour encodes which Big Five scale a direction moves *specifically*.

    A direction that shifts every scale together has not moved a factor, it has
    moved the model, so below a selectivity of 1.5 it gets no factor colour.
    FA_FearfulWithdrawal sits at 1.08 and is the case this rule exists for.
    """
    if name is not None:
        r = D["replication"].get(name, {})
        if r and r.get("sel2", 0) < SEL_MIN:
            f = "Lexicon"
    return f"--rail:var(--{HUE.get(f,'lx')})"


def quotes_for(name, poles=("neg", "pos")):
    """Pick one usable quote per pole: the largest |alpha| that is not degenerate."""
    q = D["qual"].get(name, {}).get("quotes", [])
    deg = D["degen"].get(name, {})
    ok = [x for x in q if deg.get(f"{float(x['alpha']):.1f}", 0) <= 0.5]
    neg = sorted([x for x in ok if float(x["alpha"]) < 0], key=lambda x: float(x["alpha"]))
    pos = sorted([x for x in ok if float(x["alpha"]) > 0], key=lambda x: -float(x["alpha"]))
    return (neg[0] if neg else None), (pos[0] if pos else None)


def bq(x):
    if not x:
        return ""
    t = x["text"].strip()
    if len(t) > 420:
        t = t[:418].rsplit(" ", 1)[0] + "…"
    a = float(x["alpha"])
    return (f'<blockquote><p>{e(t)}</p><cite>alpha {a:+.0f} '
            f'&middot; {e(x.get("prompt_gist","")) }</cite></blockquote>')


def rawclip(t, n=430):
    """Plain text for JS to insert with textContent -- escaping here would show
    the reader a literal &#x27; instead of an apostrophe."""
    t = (t or "").strip()
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0].rstrip(" ,;:") + "\u2026"
    return t


def clip(t, n=430):
    """Truncate before escaping, so an HTML entity is never cut in half."""
    t = (t or "").strip()
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0].rstrip(" ,;:") + "\u2026"
    return e(t)


def dircard(name, title, gloss, factor, stats, tag=""):
    neg, pos = quotes_for(name)
    qd = D["qual"].get(name, {})
    return f"""<article class="dir" style="{rail(factor, name)}">
<div class="dir-h"><div class="id">{e(name)}{tag}</div><h3>{e(title)}</h3>
<p class="gloss">{gloss}</p></div>
<div class="dir-b">
<div class="poles">
<div class="pole"><p class="lab"><b>&minus;</b> negative pole</p>
<p>{clip(qd.get('negative_pole',''))}</p>{bq(neg)}</div>
<div class="pole"><p class="lab"><b>+</b> positive pole</p>
<p>{clip(qd.get('positive_pole',''))}</p>{bq(pos)}</div>
</div>
{legend()}
{dose_svg(name)}
</div>
<div class="statline">{stats}</div>
</article>"""


# ------------------------------------------------------------------ prose bits
PCS = [
    ("PC1", "Flooding and Composure",
     "The largest single direction in the cloud is not a Big Five factor. It runs from "
     "emotional flooding to procedural composure: giddy, exclamatory, plan-free enthusiasm "
     "at one end, and terse, evidence-first, professionally regulated execution at the other."),
    ("PC2", "Hedging and Bluntness",
     "Tentative, apologetic, judgement-refusing language against direct evaluation "
     "— a model willing to tell you that you are wrong. The cleanest of the six, and the one "
     "that most nearly reproduces a named factor."),
    ("PC3", "Theory and Plain Speech",
     "Abstraction and elaboration against literalism. At one end every ordinary situation "
     "becomes a system of phases and power dynamics; at the other, answers are short, "
     "concrete and unremarkable. A register axis rather than an ability axis."),
    ("PC4", "Affirmation and Self-Concern",
     "Warm, contentless encouragement at one end; the warmth withdrawn at the other. What "
     "replaces it depends on the question &mdash; honesty when something is being judged, "
     "a turn inward when the model is asked about itself."),
    ("PC5", "Contemplation and Scheduling",
     "Quietist acceptance against instrumental planning. At one pole a situation is "
     "something to be sat with; at the other it is something to be put in a calendar."),
    ("PC6", "Feeling and Deliberation",
     "Emotional attunement against detached third-person strategy. Coherent, and small "
     "— about a third of the Agreeableness movement that the Warmth factor produces."),
]

# Ordered by sum of squared loadings, which is the order the solution itself
# puts them in. The third element is the Big Five scale the blind judge moved.
FAS = [
    ("FA_Warmth", "Warmth", "Agreeableness"),
    ("FA_Competence", "Competence", "Conscientiousness"),
    ("FA_FearfulWithdrawal", "Timidity", "EmotionalStability"),
    ("FA_Arousal", "Arousal", "Extraversion"),
    ("FA_Imagination", "Imagination", "Intellect"),
]
# The solution names its own factors; ours are matched to them explicitly rather
# than by position, because the two lists are not in the same order.
FA_KEY = {"Warmth / prosociality": "FA_Warmth", "Competence": "FA_Competence",
          "Fearful withdrawal": "FA_FearfulWithdrawal",
          "Arousal / activation": "FA_Arousal", "Imagination": "FA_Imagination"}


def stats_for(name):
    r = D["replication"].get(name, {})
    deg = D["degen"].get(name, {})
    worst = max(deg.items(), key=lambda kv: kv[1], default=("", 0))
    parts = []
    if r:
        parts.append(f'target scale <b>{e(r["named"])}</b>')
        parts.append(f'slope <b>{r["slope2"]:+.2f}</b> per unit alpha')
        parts.append(f'selectivity <b>{r["sel2"]:.1f}&times;</b>')
    intact = [a for a in ("-2.0", "-1.0", "1.0", "2.0") if deg.get(a, 0) <= 0.5]
    parts.append("intact over alpha " + ", ".join(a.replace(".0", "") for a in intact)
                 if intact else '<span class="warn">no intact alpha</span>')
    return "".join(f"<span>{x}</span>" for x in parts)


def _fa(name):
    """Load one of the factor-chart `_fa` analysis files, or None."""
    p = f"{Q}/analysis/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def coverage_svg(w=560, h=210):
    """Coverage of the chart by the 134 words, against a matched null.

    Primary is the FACTOR chart (analysis/alien_fa.json#k_sweep, decision
    2026-09-08); the principal-component curve it replaces is drawn behind it as
    a dashed comparison line at the same k, read from the same k_sweep block of
    analysis/alien.json via blog_data.json.  Nothing here is hard-coded.
    """
    AF = _fa("alien_fa.json")
    pc_cov = D["alien"]["k_sweep"]
    cov = AF["k_sweep"] if AF else pc_cov
    ks = sorted(int(k) for k in cov)
    L, R, T, B = 44, 14, 14, 30
    x = lambda k: L + (ks.index(k)) / (len(ks) - 1) * (w - L - R)
    y = lambda v: T + (95 - v) / 95.0 * (h - T - B)
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="widest unnamed gap against '
         f'number of dimensions, observed and for random directions">']
    for v in (0, 30, 60, 90):
        p.append(f'<line x1="{L}" x2="{w-R}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                 f'stroke="var(--rule)"/>')
        p.append(f'<text x="{L-8}" y="{y(v)+4:.1f}" text-anchor="end" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{v}&#176;</text>')
    band = []
    for k in ks:
        c = cov[str(k)]
        band.append((x(k), y(c["null_mean_deg"] + 2 * c["null_sd_deg"]),
                     y(c["null_mean_deg"] - 2 * c["null_sd_deg"])))
    d = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b, _ in band)
    d += " L" + " L".join(f"{a:.1f} {c:.1f}" for a, _, c in reversed(band)) + " Z"
    p.append(f'<path d="{d}" fill="var(--faint)" opacity="0.18"/>')
    for key, colr, wid in (("null_mean_deg", "var(--faint)", 1.6), ("gap_deg", "var(--ink)", 2.4)):
        pts = [(x(k), y(cov[str(k)][key])) for k in ks]
        p.append('<path d="M' + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) +
                 f'" fill="none" stroke="{colr}" stroke-width="{wid}" stroke-linejoin="round"/>')
    if AF:
        # the chart this replaced, at the same k, dashed and unlabelled points
        pcs = [(x(k), y(pc_cov[str(k)]["gap_deg"])) for k in ks if str(k) in pc_cov]
        if len(pcs) > 1:
            p.append('<path d="M' + " L".join(f"{a:.1f} {b:.1f}" for a, b in pcs) +
                     '" fill="none" stroke="var(--ink)" stroke-width="1.4" '
                     'stroke-dasharray="4 3" opacity="0.45" stroke-linejoin="round"/>')
            p.append(f'<text x="{pcs[-1][0]-6:.1f}" y="{pcs[-1][1]+14:.1f}" '
                     f'text-anchor="end" fill="var(--faint)" '
                     f'font-family="IBM Plex Mono, monospace" font-size="10">'
                     f'same words, principal-component chart</text>')
    for k in ks:
        p.append(f'<circle cx="{x(k):.1f}" cy="{y(cov[str(k)]["gap_deg"]):.1f}" r="3.2" '
                 f'fill="var(--ink)"/>')
        p.append(f'<text x="{x(k):.1f}" y="{h-9}" text-anchor="middle" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{k}</text>')
    # Label each curve where it has room, not at the right edge, where the two
    # series converge and a right-anchored label runs off the plot.
    kl = ks[len(ks) // 2]
    p.append(f'<text x="{x(kl):.1f}" y="{y(cov[str(kl)]["gap_deg"])-11:.1f}" '
             f'text-anchor="middle" fill="var(--ink)" font-family="IBM Plex Mono, monospace" '
             f'font-size="10.5" font-weight="600">the 134 trait words</text>')
    p.append(f'<text x="{x(kl):.1f}" y="{y(cov[str(kl)]["null_mean_deg"])+17:.1f}" '
             f'text-anchor="middle" fill="var(--faint)" font-family="IBM Plex Mono, monospace" '
             f'font-size="10.5">134 random directions</text>')
    p.append('</svg>')
    return "".join(p)


# ----------------------------------------------------------------- the article
def opts(items):
    return "".join(f'<option value="{e(v)}">{e(l)}</option>' for v, l in items)


# The map's axis list. Columns 0-4 of the combined score array are the five
# factor-chart axes in fa_chart.FACTOR_ORDER; columns 5-10 are PC1-PC6. The
# factors come first because they are the primary frame; the components are
# still there, one select away, because they are still the variance ordering.
AXES = ([(i, t) for i, t in enumerate(VF["factor_titles"])] if VF else []) + \
       [((5 if VF else 0) + i, f"PC{i+1}") for i in range(6)]


def axopts(sel):
    """Axis picker for the map: five factors, then six components."""
    n_fa = len(AXES) - 6
    out = []
    if n_fa:
        out.append('<optgroup label="factors">')
        out += [f'<option value="{i}"{" selected" if i == sel else ""}>{e(t)}</option>'
                for i, t in AXES[:n_fa]]
        out.append('</optgroup><optgroup label="components">')
    out += [f'<option value="{i}"{" selected" if i == sel else ""}>{e(t)}</option>'
            for i, t in AXES[n_fa:]]
    if n_fa:
        out.append("</optgroup>")
    return "".join(out)


def build():
    viz, alien = D["viz"], D["alien"]
    var = viz["var"]
    k5 = alien["alien_k5"]
    cov = alien["k_sweep"]
    fid = {r["k"]: r for r in viz["fidelity"]}

    alien_opts = [(n, l) for n, l in
                  (("alien_k5", "the unnamed direction"),
                   ("alien_shuffle", "control \u2014 shuffled coefficients"),
                   ("span_random", "control \u2014 a random direction"))
                  if n in D["curves"]]
    # Factors first, components after: the same ordering as the cards and the map
    # picker. Each factor carries its sum of squared oblimin loadings, which is the
    # quantity the solution orders them by.
    _ss = VF["ss_loadings_oblimin"] if VF else [None] * 5
    dir_opts = opts([(n, f"factor \u2014 {l}"
                      + (f" (SS {_ss[i]:.2f})" if _ss[i] is not None else ""))
                     for i, (n, l, _) in enumerate(FAS)]
                    + alien_opts + [(n, f"{n} \u2014 {l}") for n, l, _ in PCS]
                    + [(f"axis_{f}", f"named axis \u2014 {f}") for f in F5]
                    + [("mean_assistant_axis", "the personality axis \u2014 having one at all")])
    prompt_opts = opts([(str(i), (p[:62] + "…") if len(p) > 62 else p)
                        for i, p in enumerate(CORP["prompts"])])

    H = []
    A = H.append
    A(f"""<title>Investigating LLM Personality in Weight Space</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Literata:opsz,wght@7..72,400;7..72,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{asset('_css_base.txt')}{asset('_css_comp.txt')}</style>""")

    # ---------------------------------------------------------------- header
    A(f"""<div class="w"><header class="top">
<p class="eyebrow">Qwen3.5-4B &middot; 134 trait adapters &middot; weight space</p>
<h1>Investigating LLM Personality in Weight Space</h1>
<p class="stand">We trained one language model to have a hundred and thirty-four
different personalities, put every one of them in the same geometric space, and
went looking for the places the English language does not have a word for.</p>
</header>""")

    # ------------------------------------------------------------------ intro
    A(f"""<section>
<p class="lede">Personality words are not arbitrary. <i>Bold</i>, <i>timid</i>,
<i>meticulous</i>, <i>slapdash</i> &mdash; the lexicon of character is one of the
oldest datasets in psychology, and factor analysis of it is where the Big Five came
from in the first place. So here is a question you can now ask directly: if you
train a language model to actually <em>be</em> each of those adjectives, one at a
time, and then look at the weight changes side by side, does the same structure
come back?</p>

<p>It largely does. But the more interesting finding is what the words leave out.</p>

<p>Each of the 134 personalities is a LoRA adapter trained on Qwen3.5-4B with
<a href="https://arxiv.org/abs/2511.01689">Open Character Training</a> &mdash;
preference pairs where the chosen response is in character and the rejected one is
written by a character at the opposite pole of the same dimension. Every adapter is therefore a weight update, a single matrix
&Delta;<i>W</i> per module, and updates can be added, scaled and compared. That is
the whole method. The rest is geometry.</p>
<p>One thing to be exact about. Open Character Training has two stages: preference
training on the constitution's data, then supervised fine-tuning on transcripts the
trained model generates about itself. Both were run for all 134 traits, and both sets
of adapters are in the released zoo. Every geometric object on this page, unless it
says otherwise, is the <em>stage-one</em> adapter. The stage-two adapters are a second
LoRA initialised from a different random draw than stage one (one draw, shared by all 134,
as in stage one), so trait for trait they are orthogonal to the
stage-one ones in coordinates for the reason given just below &mdash; but their
arrangement reproduces the stage-one arrangement: over the 45 traits checked, the two
trait-cosine matrices correlate at <span class="mono">r = 0.79</span>{f" (over all 134, {S1S2_134:.2f})" if S1S2_134 is not None else ""}. And the
stage-two arrangement is itself seed-stable: fifteen traits put through the full
two-stage pipeline again from a second initialisation each find their own seed-zero
stage-two adapter as nearest neighbour among 134 (15 of 15), and the two seeds' cosine
matrices over those traits agree at <span class="mono">r = 0.98</span>.</p>
{fulloct_html()}
<p>Where the 134 words come from matters for how the results are read, so: <b>100</b>
are Goldberg's published unipolar Big Five markers, used as published &mdash; twenty per
factor, both poles (ten and ten, except Emotional Stability at six and fourteen in our marker file), so every one carries a factor and a keying and they are the only
traits any labelled test below is scored on. The other <b>34</b> are trait adjectives
drawn at random, with a recorded seed, from Condon's 2,818-word trait-descriptive set
(built to subsume Goldberg's 1,710), with no overlap with the markers: <i>gruff</i>,
<i>splenetic</i>, <i>artful</i>, <i>mothering</i>, <i>worldly-minded</i> and so on.
Forty were drawn; six were refused by the constitution writer as states or situations
rather than dispositions (<i>sleepy</i>, <i>busy</i>, <i>frightened</i>&hellip;) and
never trained. These 34 carry no Big Five label. They are in every unlabelled picture
&mdash; the map, the factors, the components, the hole, the sphere &mdash; and in none
of the labelled factor tests. Checked separately, they do not distort the pictures they are in: the
five-dimensional subspace of the 100 markers alone and of all 134 agree at principal
cosines of 0.94&ndash;0.996; a lexicon word keeps 25% of its variance inside the markers'
top five components against 29% for a held-out marker; its nearest marker is a median 68
degrees away against 65 for a marker's nearest other marker. They are slightly more
off-axis than the markers, as words chosen without regard to the Big Five should be, and
otherwise sit in the same cloud.</p>

<p>One detail first. Train the same trait twice from two different random LoRA
initialisations and the two adapters come out <b>89 degrees apart</b> (cosine 0.018).
Trait identity survives that &mdash; across 40 traits retrained at a second seed, the
matched adapter is the nearest neighbour among all 134 candidates 40 times out of 40
&mdash; but the whole geometry arrives scaled down by 0.0265, almost exactly the
<span class="mono">r/d = 64/2560 = 0.025</span> overlap that two random rank-64
subspaces of a 2560-dimensional space are expected to have. It is the initialisation
and nothing else: the same traits retrained at the same seed under a different objective
agree at cosine 0.95. (Details, including the pre-registered bar this arm was first
judged against and failed, are in the project's verdict file.)</p>
<p>Which is why every adapter here shares one initialisation. A LoRA writes its
update as a product of two matrices, &Delta;<i>W</i>&nbsp;=&nbsp;<i>BA</i>, with
<i>B</i> starting at zero and <i>A</i> drawn at random. Because <i>B</i> starts at zero, <i>A</i> receives
almost no gradient: across the whole zoo, <i>A</i> moves just <b>1.5%</b> of its
norm during training. And every adapter here was initialised from the same seed, so
all 134 share the same random <i>A</i> and therefore write into the same
64-dimensional slice of input space. They are directly comparable in a way that
independently initialised adapters are not &mdash; two different draws overlap by
only <span class="mono">r/d</span> = 64/2560 = 2.5%.</p>
</section>

<hr class="sep">

<section>
<h2>Reading a direction out loud</h2>
<p>Any weighted sum of adapters is itself a weight update, which means every
direction in this space can be applied to the model and <em>listened to</em>. Add
&alpha; times a direction to the base weights, ask the model twenty-four questions,
and read what comes back. Strength is measured against a fixed reference, ref = 0.81,
so &alpha; is comparable across directions. That reference was meant to be one trait
adapter's Frobenius norm; the final audit found it omitted the LoRA scale of 2, so it
is half of one. Read every &alpha; on this page as half an adapter's worth of weight
change: &alpha;&nbsp;=&nbsp;2 is one adapter.</p>
<p>Everything below was generated that way and scored by a blind judge on the Big
Five. Try it:</p>
</section></div>

<div class="wide"><div class="panel" id="explorer">
<div class="ctl">
<label>direction <select class="exp-dir">{dir_opts}</select></label>
<label>strength <input class="exp-alpha" type="range" min="0" max="4" step="1" value="3"></label>
<label>question <select class="exp-prompt">{prompt_opts}</select></label>
</div>
<p class="exp-name"></p>
<div class="split">
<div><p class="readmeta"></p><div class="read"></div></div>
<div>{legend()}<div class="bars"></div>
<p class="hintline">Blind judge, 1&ndash;7, averaged over all 24 questions at this
strength &mdash; not over the single answer shown. Strength 0 is the unmodified
model, identical for every direction.</p></div>
</div></div></div>

<div class="w"><section>
<p>Two things are worth noticing while playing with it. The first is how little
weight change it takes: half an adapter's worth is a small perturbation and it rewrites
the register completely. The second is that pushing harder does not keep working.
Past &alpha;&nbsp;=&nbsp;&plusmn;2 the model starts repeating itself, and at
&plusmn;4 every direction in this study collapses into looping. A dose-response
curve drawn through babble is not a measurement of personality, so those points are
marked rather than averaged in.</p>
</section>

<hr class="sep">

<section>
<h2>The map</h2>
<p>The 134 adapters give a picture you can turn over. Each point below is one
trait adapter; drag to rotate, and pick which directions form the axes. It opens
on the three largest factors of the factor analysis in the next section &mdash;
warmth, competence, and approach against avoidance &mdash; because that is the
frame the rest of this page uses. The principal components are in the same picker,
underneath, because they are the same cloud ordered by variance instead.</p>
</section></div>

<div class="wide"><div class="panel" id="map">
<div class="ctl">
<label>x <select>{axopts(0)}</select></label>
<label>y <select>{axopts(1)}</select></label>
<label>z <select>{axopts(2)}</select></label>
</div>
<div class="chips">""" + "".join(
        f'<button class="chip" data-f="{e(f)}" aria-pressed="true" '
        f'style="--c:var(--{HUE[f]})">{e(f)}</button>'
        for f in F5 + ["Lexicon"]) + f"""</div>
<canvas></canvas>
<p class="hintline maptip"></p>
<p class="axisnames hintline"></p>
<p class="hintline">Each point is one trait adapter. Filled centres are
positively keyed traits, ringed centres negatively keyed. Depth is real: points
further away are smaller and fainter. The cross sits at the mean adapter: every
adapter carries a shared component (the character-register direction), and the cloud
is plotted as deviations from it.</p>
</div></div>

<div class="w"><section>
<p>Five axes are not the whole space. Every adapter has a length of its own, and
the five factor directions catch a little over half of the average adapter's
weight change between them; the rest points somewhere these axes do not go. The
next section says how much, and gives each of the five a name and a voice.</p>
<p>Ordered by variance instead, the same cloud gives the principal components,
also in the picker. The first two take {var[0]*100:.1f}% and {var[1]*100:.1f}% of
the variance and the rest fall away quickly &mdash; not a compact space,
{sum(var[:8])*100:.0f}% in eight dimensions &mdash; and none of the leading
components is quite a Big Five factor. The factors are the frame from here on;
the components follow them, because ordering by variance answers how many
dimensions there are rather than what they are.</p>
</section>

</div>""" + factor_section() + f"""<div class="w">
<hr class="sep">

<section>
<h2>The six components</h2>
<p>The same 134 adapters, ordered by variance rather than rotated for simple
structure. The components were the first frame this project had and they are kept
because two of the arguments below need them: the count of dimensions above a
null, and the optimised-data arms, whose training data was selected for PC4 and
which are therefore reported under that name. Each was steered in both directions,
generated blind and read. The descriptions are of transcripts, not of
loadings.</p>
</section></div>

<div class="wide">""")

    for name, title, gloss in PCS:
        i = int(name[2:]) - 1
        A(dircard(name, f"{name}. {title}", gloss,
                  D["replication"].get(name, {}).get("named", "Lexicon"), stats_for(name),
                  tag=f' &middot; {var[i]*100:.1f}% of variance'))
    A("</div>")
    A(part_two())
    A(part_three())
    return H


def pc_table():
    """What each component is made of, and how it sits against the named axes.

    Two independent readings of the same six directions: the cosine with each
    Big Five axis, which is geometry, and the adjectives loading highest at each
    end, which is content. Eigenvector sign is arbitrary, so a component and its
    negation are the same axis -- read the magnitudes, and read the two loading
    columns as the two ends rather than as good and bad.
    """
    p = f"{Q}/analysis/pc_loadings.json"
    if not os.path.exists(p):
        return ""
    L = json.load(open(p))
    rows = []
    for k, d in L["pcs"].items():
        cs = d["cos"]
        top = max(cs, key=lambda f: abs(cs[f]))
        cells = "".join(
            f'<td{" style=\"font-weight:600;color:var(--" + HUE[f] + ")\"" if f == top else ""}>'
            f'{cs[f]:+.2f}</td>' for f in F5)
        rows.append(
            f'<tr><td style="text-align:left">{e(k)}</td>{cells}'
            f'<td style="text-align:left">{e(", ".join(d["pos"]))}</td>'
            f'<td style="text-align:left">{e(", ".join(d["neg"]))}</td></tr>')
    head = "".join(f"<th>{e(f[:5])}</th>" for f in F5)
    return ('<div class="wide"><figure><div class="scroll"><table><thead><tr>'
            f'<th></th>{head}<th>loads high at one end</th><th>and at the other</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
            '<figcaption>Each component against the five named axes, and the five '
            'adjectives loading highest at either end. Sign is arbitrary for an '
            'eigenvector, so the two loading columns are the two ends of one axis, not '
            'a ranking. The largest cosine in each row is picked out in that '
            'factor&rsquo;s colour.</figcaption></figure></div>')


def scree_svg(w=1000, h=290):
    """Two decompositions, side by side, never on one axis.

    LEFT is PCA of the adapter cloud: eigenvalues of the double-centred Gram of
    the 134 weight updates, in per cent of total variance. This is where the six
    components come from.

    RIGHT is the factor-analytic question, which is a different matrix and a
    different quantity: eigenvalues of the 134x134 trait correlation matrix with
    the diagonal REDUCED to communality estimates, against the 95th percentile
    of the same statistic on random data. Principal axis factoring partitions
    common variance, not total variance, so its eigenvalues are smaller, can go
    negative, and its elbow is not the PCA elbow. Plotting them on one axis
    would be a category error.

    The right panel plots the CENTRED (ipsatised) parallel analysis. It used to
    plot the uncentred one, which is a different matrix from the one the five
    factors on this page are extracted from and from the one n_factors.chosen is
    read off. Retention at the reference N moves from 12 reduced / 9 unreduced
    (uncentred) to 9 / 9 (centred); the prose is generated from the grid, so the
    two cannot drift apart again.
    """
    p = f"{Q}/results/fa_qwen35.json"
    if not os.path.exists(p):
        return ""
    R = json.load(open(p))
    var = R["pca_from_gram"]["centered_var_pct"][:16]
    nf = R["n_factors"]
    pa = nf["parallel_analysis_centred"]
    grid = {g["N"]: g for g in pa["grid"]}
    N = nf["reference_N"]
    obs = pa["observed_reduced"][:16]
    nul = grid[N]["null_reduced_95pct"][:16]
    k_red, k_unred = grid[N]["k_reduced_95pct"], grid[N]["k_unreduced_95pct"]

    pad, gap = 52, 74
    pw = (w - pad - gap - 18) / 2
    T, B = 30, 34

    def panel(x0, title, series, ymax, marks, ylab):
        px = lambda i, n: x0 + (i / (n - 1)) * pw
        py = lambda v: T + (1 - v / ymax) * (h - T - B)
        o = [f'<text x="{x0}" y="16" fill="var(--ink)" font-family="IBM Plex Mono, monospace" '
             f'font-size="11" font-weight="600">{title}</text>']
        for gv in (0, ymax / 2, ymax):
            o.append(f'<line x1="{x0}" x2="{x0+pw}" y1="{py(gv):.1f}" y2="{py(gv):.1f}" '
                     f'stroke="var(--rule)"/>')
            o.append(f'<text x="{x0-8}" y="{py(gv)+4:.1f}" text-anchor="end" '
                     f'fill="var(--faint)" font-family="IBM Plex Mono, monospace" '
                     f'font-size="9.5">{gv:.0f}{ylab}</text>')
        for mk, lab, colr in marks:
            o.append(f'<line x1="{px(mk-1,len(series[0][1])):.1f}" '
                     f'x2="{px(mk-1,len(series[0][1])):.1f}" y1="{T}" y2="{h-B}" '
                     f'stroke="{colr}" stroke-width="1" stroke-dasharray="3 3"/>')
            o.append(f'<text x="{px(mk-1,len(series[0][1]))+4:.1f}" y="{T+11}" '
                     f'fill="{colr}" font-family="IBM Plex Mono, monospace" '
                     f'font-size="9.5" font-weight="600">{lab}</text>')
        for name, vals, colr, dash, at in series:
            n = len(vals)
            d = " ".join(("M" if i == 0 else "L") + f"{px(i,n):.1f} {py(v):.1f}"
                         for i, v in enumerate(vals))
            o.append(f'<path d="{d}" fill="none" stroke="{colr}" stroke-width="2"'
                     + (' stroke-dasharray="4 3"' if dash else "") + '/>')
            for i, v in enumerate(vals):
                o.append(f'<circle cx="{px(i,n):.1f}" cy="{py(v):.1f}" r="2.4" fill="{colr}"/>')
            # anchored where this series is separated from the others, not at the
            # right edge, where two converging curves put their labels on top of
            # each other
            j, anch, dy = at
            o.append(f'<text x="{px(j,n):.1f}" y="{py(vals[j])+dy:.1f}" text-anchor="{anch}" '
                     f'fill="{colr}" font-family="IBM Plex Mono, monospace" font-size="10">'
                     f'{e(name)}</text>')
        n = len(series[0][1])
        for i in range(0, n, 3):
            o.append(f'<text x="{px(i,n):.1f}" y="{h-14}" text-anchor="middle" '
                     f'fill="var(--faint)" font-family="IBM Plex Mono, monospace" '
                     f'font-size="9.5">{i+1}</text>')
        return "".join(o)

    # The elbow is a shape judgement and lands on 1, 2 or 3 depending on the
    # criterion, so the panel marks where the real spectrum meets a null instead.
    # Two nulls, and only one of them is a null for this question: `shuffled`
    # swaps chosen and rejected on half of each trait's pairs, destroying the
    # preference signal, so nothing coherent is learned; `permuted` hands each
    # trait NAME another trait's pairs, so it learns the same 100 personas under
    # the wrong names and its spectrum must match the real one.
    NL = json.load(open(f"{Q}/analysis/scree_null_matched.json"))
    nser = [("real", NL["real"][:16], "var(--ink)", False, (9, "start", -10)),
            ("labels permuted", NL["permuted"][:16], "var(--co)", True, (5, "start", -9)),
            ("preference destroyed", NL["shuffled"][:16], "var(--faint)", True,
             (5, "start", 16))]
    left = panel(pad, "PCA of the adapter cloud, against two nulls", nser,
                 max(NL["real"][:16]) * 1.1,
                 [(NL["n_above_structureless"] + 1, "structure ends", "var(--in)")], "%")
    right = panel(pad + pw + gap, f"Factor analysis, parallel analysis at N={N:,}",
                  [("observed (reduced)", obs, "var(--ink)", False, (2, "start", -11)),
                   ("random data, 95th pct", nul, "var(--faint)", True, (2, "start", 16))],
                  max(obs) * 1.1,
                  [(5, "5 extracted", "var(--co)"), (k_red, f"{k_red} retained", "var(--in)")],
                  "")
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="scree plots for the '
            f'principal components and for the factor solution">{left}{right}</svg>')


def fa_table():
    k5 = D["fa"]["centred_k5"]
    title_of = {k: t for k, t, _ in FAS}
    out = ['<div class="scroll"><table><thead><tr><th>factor</th><th>SS loadings</th>'
           '<th>highest-loading traits</th></tr></thead><tbody>']
    for f in sorted(k5["factors"], key=lambda x: -x["ss"]):
        key = FA_KEY.get(f["name"])
        if key is None:
            raise KeyError(f'unmapped factor name from the solution: {f["name"]}')
        top = ", ".join(t["t"].lower() for t in f["top"][:5])
        out.append(f'<tr><td>{e(title_of[key])}</td><td>{f["ss"]:.2f}</td>'
                   f'<td style="text-align:left">{e(top)}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def factor_section():
    """The five factors, first, because they are the frame the page now uses.

    This block used to sit inside part_two(), after the six components and after
    pc_table(), and it was written as PCA-then-FA motivation. Decision 2026-09-08
    (Samuel) makes the factor analysis primary, so the block moved ahead of the
    components and the motivation was rewritten in the other direction: the
    factors are what the page charts, and the components are the variance
    ordering they were rotated out of. Nothing in the numbers changed.
    """
    r_w = D["replication"]["FA_Warmth"]
    r_f = D["replication"]["FA_FearfulWithdrawal"]
    ss = ", ".join(f"{s:.2f}" for s in VF["ss_loadings_oblimin"]) if VF else ""
    cap = (f"{VF['summary']['chart_captures_frac_of_norm_mean']*100:.0f}%" if VF else "")
    cards = "".join(
        dircard(name, title,
                e(D["qual"].get(name, {}).get("axis_label", "")),
                named, stats_for(name),
                tag=(f' &middot; selectivity {D["replication"][name]["sel2"]:.1f}&times;'
                     if name in D["replication"] else ""))
        for name, title, named in FAS)
    return f"""<div class="w">
<hr class="sep">

<section>
<h2>The five factors</h2>
<p>Here is the frame the rest of this page uses. Take the correlations between the
134 adapters, extract five factors by principal axis factoring, and rotate them for
simple structure with an oblique rotation that does not force them to be
perpendicular. This is what Cattell and Goldberg did to the adjective lexicon; here
it is done to weight updates instead of questionnaire responses, and it is the
closest thing this study has to running the experiment that produced the Big Five
on a model instead of on people.</p>
<p>Five come out. In the order the solution itself puts them, by sum of squared
loadings &mdash; {ss} &mdash; they are warmth, competence, approach against
avoidance, arousal, and imagination. That order is fixed, and the map above, the
cards below and the per-trait pages all place an adapter in these five
coordinates, made perpendicular by Gram-Schmidt so that a coordinate is an honest
inner product. Between them the five account for {cap} of the average adapter's
weight change; what is left over is real, and is not charted.</p>
<p>Two of the five are the most selective directions in the entire study. One of
them is the weakest. All five were steered in both directions, generated blind and
read, and the descriptions below are of transcripts, not of loadings.</p>
</section></div>

<div class="wide"><figure>{fa_table()}
<figcaption>Five factors from the centred solution, with the traits that load
highest on each. The names are ours, assigned after reading the loadings and then
checked against steered transcripts. The solution's own labels are warmth /
prosociality, competence, fearful withdrawal, arousal / activation and
imagination.</figcaption></figure></div>

<div class="wide">{cards}</div>

<div class="w"><section>
<p>Warmth is the most selective direction measured anywhere in this study: steering
it moves judged Agreeableness {r_w['slope2']:+.2f} per unit &alpha; while barely
touching the other four scales &mdash; a selectivity of
<b>{r_w['sel2']:.1f}&times;</b>, or <b>7.3&times;</b> recomputed on only the eighteen
prompts that stay free of looping across the whole range. Selectivity is a ratio of
linear slopes, and it is blind to a symmetric cost: judged Conscientiousness peaks at
alpha 0 and falls at both alpha -2 and +2, and Intellect does the same, so their slopes
cancel while the movement is real; the matched-dose run records it as 0.77 points of
off-target movement on the amplifying side against 0.12 on the suppressing side. At +2
the model shows no damage markers; it simply answers less. Imagination behaves the
other way round: restricted to its thirteen prompts free of looping and scaffold loss,
its selectivity <em>rises</em>, from 4.2 to 7.8. Both do exactly what their names say.</p>
<p>One of the five does not. Timidity is the weakest direction in the
study &mdash; a selectivity of {r_f['sel2']:.2f}, meaning it moves every scale at
once rather than a factor, and its fearful pole turns out to sit at negative
&alpha; rather than positive. It is drawn without a factor colour above for that
reason: colour on this page means a direction moves one scale specifically, and this
one does not.</p>
<p class="pull">Factor analysis on weight updates recovers cleaner, more selective
axes than the variance ordering does &mdash; the same reason psychometrics rotates
in the first place.</p>
<p>Which is the bridge to what this page used to lead with. The same cloud, ordered
by variance instead of rotated for simple structure, gives six principal components,
and they are worth reading precisely because they are less clean.</p>
</section></div>

"""


SPHERE_BLOCK = """<hr class="sep">

<section>
<h2>Sampling the space</h2>
<p>Every direction so far was chosen for a reason: a principal component, a named
axis, a factor. That makes the evidence circular in one specific way. Finding a
coherent persona along a direction you selected proves very little if <em>every</em>
direction gives a coherent persona.</p>
<p>So here is the space sampled without a reason. {sphere_intro()}</p>
</section></div>

<div class="wide"><div class="panel" id="sphere">
<div class="ctl"><label>question <select class="sph-prompt"></select></label></div>
<div class="split">
<div><canvas></canvas><p class="hintline maptip"></p></div>
<div><p class="readmeta"></p><div class="read"></div>
{legend()}<div class="bars"></div></div>
</div>
<p class="hintline">{sphere_caption()}</p>
</div></div>

"""


def sphere_smooth():
    """The one thing a sampled sphere can measure that a chosen direction cannot.
    Reads the factor-chart sphere (sphere_page_fa.json) when present, else the PC one."""
    p = f"{Q}/analysis/sphere_page_fa.json"
    if not os.path.exists(p):
        p = f"{Q}/analysis/sphere_page.json"
    if not os.path.exists(p):
        return ""
    P = json.load(open(p))
    S = P.get("smooth")
    if not S:
        return ""
    tops = {}
    for v in P.get("judged", {}).values():
        if v.get("top"):
            tops[v["top"]] = tops.get(v["top"], 0) + 1
    dist = ", ".join(f"{k} {v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1]))
    ranked = sorted(tops.items(), key=lambda kv: -kv[1])
    n_pts = max(1, sum(tops.values()))
    missing = [f for f in F5 if f not in tops]
    top_sentence = (f"{ranked[0][0]} is the top-rated scale at {ranked[0][1]} of the {n_pts} points"
                    + (f", and {', '.join(missing)} at none" if missing else "")
                    + " -- the base model's own pull is strong enough that most directions through "
                      "this space land somewhere pleasant.") if ranked else ""
    C = P.get("coherence")
    coh = ""
    if C:
        coh = f"""<p>First, the question that prompted the sampling, answered against
the sampler's own intent: at &alpha;&nbsp;=&nbsp;+1.5, <b>{C['none']} of the 72</b>
sampled directions produce no looping at all, and the worst of them loses
{C['worst']*100:.0f}% of its responses &mdash; a mean rate of {C['mean']*100:.1f}%
across the sphere. So most directions through this space <em>do</em> give intact
output, and the coherence of the ones we chose is not, by itself, evidence of
anything. What has to carry the weight instead is specificity: whether a direction
moves the scale it is supposed to and leaves the others alone, and whether its
coordinates predict its behaviour in advance. Those are the numbers this piece leans
on, and now you know why.</p>"""
    return f"""<div class="w"><section>
{coh}
<p>Sampling also answers something no chosen direction can. If nearby directions
produce nearby personalities, then position in this space really is a description of
character; if they do not, each direction is its own unrelated thing and calling it a
map is decoration. Over the {S['n_scored']} scored directions and all
{S['n_pairs']:,} pairs of them, angular distance and judged profile distance correlate
at <b>&rho; = {S['rho']:+.2f}</b>. Directions less than 30&deg; apart differ by
{S['near_mean']:.2f} on the Big Five; directions more than 120&deg; apart differ by
{S['far_mean']:.2f}, {S['far_mean']/S['near_mean']:.1f} times as much.</p>
<p class="pull">Personality varies continuously across this space. Walk a short way
and the character changes a little; walk to the far side and it changes a lot. That
is what makes it a map rather than a list.</p>
<p>The coverage is lopsided, though, and worth noticing. Counting which scale the
judge rated highest at each sampled point: {e(dist)}. {e(top_sentence)}</p>
</section></div>"""


SPHERE_FA_LABEL = {"FA_Warmth": "warmth", "FA_Competence": "competence",
                   "FA_FearfulWithdrawal": "timidity",
                   "FA_Arousal": "arousal", "FA_Imagination": "imagination"}


def sphere_data():
    """The factor-chart sphere is primary from 2026-09-08; the principal-component
    sphere of 2026-09-01 is the fallback and is only used if the redo is absent.
    Returns (data, is_factor_chart)."""
    for p, fa in ((f"{Q}/analysis/sphere_page_fa.json", True),
                  (f"{Q}/analysis/sphere_page.json", False)):
        if os.path.exists(p):
            S = json.load(open(p))
            if S.get("gen"):
                return S, fa
    return None, False


def sphere_intro():
    """Say which sphere this is, in its own numbers -- the count of directions,
    the strength, and the axes -- so the paragraph cannot drift from the data."""
    S, fa = sphere_data()
    if not S:
        return ""
    n = len(S.get("points", []))
    a = S.get("alpha")
    if fa:
        ax = ", ".join(SPHERE_FA_LABEL.get(x, x) for x in S.get("axis_names", []))
        v = S.get("var3") or []
        share = f" ({sum(v)*100:.0f}% of the centred variance between them)" if v else ""
        # The comparison sentence must carry a number or make no claim: whether
        # the two charts agree is a measurement, not something to assert.
        C = (S.get("vs_pc") or {}).get("pearson")
        if C:
            lo, hi = min(C.values()), max(C.values())
            prev = (f" An earlier version of this sampler used the top three principal "
                    f"components instead; matching each direction here to the nearest "
                    f"direction there, the two judged fields agree at r = {lo:+.2f} to "
                    f"{hi:+.2f} across the five scales.")
        else:
            prev = (" An earlier version of this sampler used the top three principal "
                    "components instead.")
        return (f"{n} directions spread near-uniformly over the sphere of the space "
                f"the top three factors span &mdash; {e(ax)}{share} &mdash; each "
                f"steered at &alpha;&nbsp;=&nbsp;+{a} and generated blind. Click "
                f"anywhere to read what the model becomes there. The faint dots behind "
                f"are the 134 trait words, on the same sphere." + prev)
    return (f"{n} directions spread near-uniformly over the sphere of the top three "
            f"components, each steered at &alpha;&nbsp;=&nbsp;+{a} and generated blind. "
            f"Click anywhere to read what the model becomes there. The faint dots "
            f"behind are the 134 trait words, on the same sphere.")


def sphere_caption():
    """Only claim the colouring if the judge actually covered every point."""
    S, fa = sphere_data()
    if not S:
        return ""
    n = sum(1 for v in S.get("judged", {}).values() if v.get("top"))
    tot = len(S.get("points", []))
    lead = (f"Points are coloured by the Big Five scale the blind judge rated highest "
            f"there ({n} of {tot} scored)." if n == tot else
            f"Points are coloured by the Big Five scale the blind judge rated highest "
            f"there; {n} of {tot} were scored and the rest are drawn grey.")
    if fa:
        return (lead + " Crosses mark the named directions projected onto this sphere: "
                "the five factors, the five Big Five axes, the grand mean, and the "
                "three principal components of the earlier version.")
    return (lead + " Crosses are the five named axes; the dashed ring is the unnamed "
            "direction from the next section.")


def sphere_section():
    """The sampler is only shown if it has data. An empty interactive panel
    promising clickable transcripts is worse than no panel."""
    S, _ = sphere_data()
    if not S:
        return "</div>"
    # SPHERE_BLOCK was lifted out of the article as a plain string, so its
    # placeholders were never interpolated and rendered as literal braces.
    return (SPHERE_BLOCK.replace("{legend()}", legend())
                        .replace("{sphere_intro()}", sphere_intro())
                        .replace("{sphere_caption()}", sphere_caption()))


def heat_html(h, hue):
    """The champion response, coloured by how hard each stretch of it pushes.

    Scores are per eight-token chunk, obtained by masking the loss down to
    those tokens and taking the same exact derivative; the shading is a
    within-text z-score so the ramp reads the same for every direction.
    """
    import statistics
    v = h["chunk_scores"]
    if len(v) < 2:
        return ""
    mu = statistics.mean(v)
    sd = statistics.pstdev(v) or 1.0
    toks, k = h["tokens"], h["chunk"]
    out = []
    for i, s_ in enumerate(v):
        z = max(-2.2, min(2.2, (s_ - mu) / sd))
        seg = "".join(toks[i * k:(i + 1) * k])
        if z >= 0:
            style = f"background:color-mix(in srgb,var(--{hue}) {z/2.2*46:.0f}%,transparent)"
        else:
            style = f"background:color-mix(in srgb,var(--faint) {-z/2.2*22:.0f}%,transparent)"
        out.append(f'<span style="{style}" title="{z:+.2f} SD">{e(seg)}</span>')
    return f'<div class="heat">{"".join(out)}</div>'


def traj_svg(hist, aims, w=560, h=170):
    L, R, T, B = 46, 12, 14, 26
    n = max(len(hist[a]) for a in aims)
    ys = [x for a in aims for x in hist[a]]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    x = lambda i: L + (i / max(n - 1, 1)) * (w - L - R)
    y = lambda v: T + (hi - v) / (hi - lo) * (h - T - B)
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="best objective per round">']
    for f in (lo, (lo + hi) / 2, hi):
        p.append(f'<line x1="{L}" x2="{w-R}" y1="{y(f):.1f}" y2="{y(f):.1f}" stroke="var(--rule)"/>')
        p.append(f'<text x="{L-8}" y="{y(f)+4:.1f}" text-anchor="end" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{f:+.2f}</text>')
    cols = {"alien_k5": "var(--alienline)", "axis_Agreeableness": "var(--ag)",
            "PC4": "var(--es)"}
    for a in aims:
        pts = [(x(i), y(v)) for i, v in enumerate(hist[a])]
        c = cols.get(a, "var(--faint)")
        dash = ' stroke-dasharray="4 3"' if a == "alien_k5" else ""
        p.append('<path d="M' + " L".join(f"{q:.1f} {r:.1f}" for q, r in pts) +
                 f'" fill="none" stroke="{c}" stroke-width="2.2"{dash}/>')
        p.append(f'<text x="{pts[-1][0]-4:.1f}" y="{pts[-1][1]-8:.1f}" text-anchor="end" '
                 f'fill="{c}" font-family="IBM Plex Mono, monospace" font-size="10" '
                 f'font-weight="600">{e(a)}</text>')
    for i in range(n):
        p.append(f'<text x="{x(i):.1f}" y="{h-8}" text-anchor="middle" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{i}</text>')
    p.append("</svg>")
    return "".join(p)


def optimise_section():
    """Written only when the search has actually produced champions."""
    p = f"{Q}/analysis/optimise.json"
    if not os.path.exists(p):
        return ""
    O = json.load(open(p))
    aims, hist, heat = O["aims"], O["hist"], O["heat"]
    hue = {"alien_k5": "alienline", "axis_Agreeableness": "ag", "PC4": "es"}
    gain = {a: (hist[a][-1] - hist[a][0]) / abs(hist[a][0] or 1) for a in aims}
    cards = []
    for a in aims:
        h = heat.get(a)
        if not h:
            continue
        nm = {"alien_k5": "the unnamed direction", "axis_Agreeableness": "the Agreeableness axis",
              "PC4": "PC4, affirmation and self-concern"}.get(a, a)
        cards.append(f"""<article class="dir" style="--rail:var(--{hue.get(a,'lx')})">
<div class="dir-h"><div class="id">champion &middot; {e(a)}</div>
<h3>Data written for {e(nm)}</h3>
<p class="gloss">Objective <span class="mono">{h['obj']:+.3f}</span> after
{len(hist[a])} rounds, from <span class="mono">{hist[a][0]:+.3f}</span> in the first.
Shading is each eight-token stretch's own contribution, obtained by masking the loss down to that stretch and taking the same exact derivative, then scored against the rest of this response &mdash; so it shows which parts of the text carry the effect, not how strong the text is overall. The rejected half is the base model's default answer, held fixed.</p></div>
<div class="dir-b">
<p class="lab" style="font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin:0 0 8px">the question</p>
<blockquote><p>{e(h['prompt'])}</p></blockquote>
<p class="lab" style="font:600 10.5px/1 'IBM Plex Mono',monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint);margin:16px 0 8px">the chosen response</p>
{heat_html(h, hue.get(a, 'lx'))}
</div></article>""")
    return f"""<div class="wide"><figure>{traj_svg(hist, aims)}
<figcaption>Best objective per round. Every round shows the model its own
highest-scoring responses and asks for more in that voice, then rescores. The
unnamed direction is the dashed line.</figcaption></figure>
{"".join(cards)}</div>"""



def hole_fa_composition():
    """Where the factor-chart hole points, from analysis/alien_fa.json."""
    p = f"{Q}/analysis/alien_fa.json"
    if not os.path.exists(p):
        return ""
    a = json.load(open(p)); af = a["alien_fa"]
    names = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
    u = af["u"]; order = sorted(range(5), key=lambda i: -abs(u[i]))
    comp = ", ".join(f"{u[i]:+.2f} {names[i]}" for i in order[:3])
    ks = a["k_sweep"]
    return (f"<p>Resolved onto the factors, the direction is not any one of them: "
            f"<b>{comp}</b>. It sits {af['gap_deg']:.1f} degrees from the nearest trait adapter in the "
            f"five-factor chart (z&nbsp;=&nbsp;{ks['5']['z']:.1f} against random directions), "
            f"{af['full_gap_deg']:.1f} degrees in the full space (nearest <i>{af['full_nearest']}</i>). "
            f"Two things did not survive moving from principal components to factors: in two dimensions the "
            f"words no longer tile the plane better than chance (z&nbsp;=&nbsp;{ks['2']['z']:+.2f}, against "
            f"&minus;2.3 before), and the winning direction inside the same five-dimensional span moved by "
            f"{a['vs_pc']['deg_alien_fa_alien_k5']:.0f} degrees while the gap barely changed. "
            f"The subspace is stable; the deepest point in it is not.</p>")


def hole_fa_verdict(MM):
    """Prose after the verdict tables, from alien_match_fa, alien_steer_fa, hole_geometry_fa."""
    A_ = json.load(open(f"{Q}/analysis/alien_steer_fa.json"))
    H = json.load(open(f"{Q}/analysis/hole_geometry_fa.json"))
    dmg = A_["alien_fa"].get("degen", {}).get("-2.0", 0.0)
    ins, cav, bla = H["insouciant"], H["cavalier"], H["blase"]
    near = H["_hole_nearest_existing_deg"]
    pw = H["_pairwise"]
    return f"""<p>Where the direction is intact it does produce a coherent person: across &alpha; from
&minus;2 to +2 the judge's Conscientiousness rises steadily, Agreeableness and Emotional
Stability fall, and the transcripts read as brisk, unsentimental and organised. The chart's
five coordinates got {MM['signs']} of the five signs right, correlating with the judge at
{MM['r']:.2f}. But the controls, scored against the same prediction, do about as well, because
all three directions share one large Conscientiousness rise; and the unnamed direction is
the one that breaks, looping on {dmg*100:.0f}% of responses at &alpha;&nbsp;=&nbsp;&minus;2
where the controls loop on none. On this evidence the factor chart did not predict anything
direction-specific about the hole. That is a weaker result than the principal-component
version of this section once claimed, and the weaker result is the right one.</p>
<p>Reading a draft, an external reviewer suggested the word might be <i>cavalier</i>, <i>blas&eacute;</i>
or <i>insouciant</i>, so three more adapters were trained on those words on the zoo's own
prompts. In the factor chart they land {cav['deg_to_u_chart']:.0f}, {bla['deg_to_u_chart']:.0f} and
{ins['deg_to_u_chart']:.0f} degrees from the hole, where the nearest of the 134 existing adapters
is already at {near:.0f}; a random adapter comes within {ins['deg_to_u_chart']:.0f} degrees
{H['_p_one']*100:.0f}% of the time, and at least one of three does {H['_p_any_of_three']*100:.0f}% of
the time. The three supposed synonyms sit {min(pw.values()):.0f} to {max(pw.values()):.0f} degrees
from <em>each other</em>, as far apart as random traits. The names miss. What stands is the
negative: three good words for the region, trained carefully, do not agree with each other
about where it is, and none of them reaches it.</p>"""

def alien_verdict():
    """The unnamed direction, steered, judged blind, against two controls."""
    p, q = f"{Q}/analysis/alien_steer_fa.json", f"{Q}/analysis/alien_match_fa.json"
    if not (os.path.exists(p) and os.path.exists(q)):
        return ""
    A_ = json.load(open(p))
    M = json.load(open(q))
    DIRS = [("alien_fa", "the unnamed direction (factor chart)"),
            ("alien_fa_shuffle", "control: the same coefficients, shuffled"),
            ("span_random_fa", "control: a random direction in the same subspace")]
    al = ["-2.0", "-1.0", "1.0", "2.0"]

    rows = "".join(
        f'<tr><td style="text-align:left">{e(f)}</td>'
        f'<td>{M["pred"][k]:+.2f}</td><td>{M["obs"][k]:+.2f}</td>'
        f'<td style="color:var(--ag)">match</td></tr>'
        if (M["pred"][k] > 0) == (M["obs"][k] > 0) else
        f'<tr><td style="text-align:left">{e(f)}</td>'
        f'<td>{M["pred"][k]:+.2f}</td><td>{M["obs"][k]:+.2f}</td>'
        f'<td style="color:var(--in)">miss</td></tr>'
        for k, f in enumerate(F5))

    dmg = []
    stats_by = {}
    for k, nm in DIRS:
        d = A_.get(k, {})
        cur = d.get("curve", {})
        obs = [cur.get("2.0", {}).get(f, 0) - cur.get("-2.0", {}).get(f, 0) for f in F5]
        import statistics
        try:
            pr = M["pred"]
            mu_p, mu_o = statistics.mean(pr), statistics.mean(obs)
            num = sum((a - mu_p) * (b - mu_o) for a, b in zip(pr, obs))
            den = (sum((a - mu_p) ** 2 for a in pr) * sum((b - mu_o) ** 2 for b in obs)) ** 0.5
            r = num / den if den else 0.0
        except Exception:
            r = 0.0
        sg = sum(1 for a, b in zip(M["pred"], obs) if (a > 0) == (b > 0))
        cells = "".join(f'<td>{d["degen"][a]*100:.0f}%</td>' if a in d.get("degen", {})
                        else "<td>&mdash;</td>" for a in al)
        cls = ' class="hi"' if k == "alien_fa" else ""
        dmg.append(f'<tr{cls}><td style="text-align:left">{e(nm)}</td>'
                   f'<td>{sg}/5</td><td>{r:+.2f}</td>{cells}</tr>')
        stats_by[k] = (sg, r, d.get("degen", {}).get("-2.0", 0.0))
    head = "".join(f"<th>{a.replace('.0','')}</th>" for a in al)
    sa, sr = stats_by.get("alien_fa", (0, 0.0, 0.0))[:2]
    ss_, rr_ = stats_by.get("alien_fa_shuffle", (0, 0.0, 0.0))[:2]
    sn, rn = stats_by.get("span_random_fa", (0, 0.0, 0.0))[:2]
    dmg_a = stats_by.get("alien_fa", (0, 0.0, 0.0))[2]

    return f"""<div class="wide"><figure><div class="scroll"><table>
<thead><tr><th>scale</th><th>chart said</th><th>judge saw</th><th></th></tr></thead>
<tbody>{rows}</tbody></table></div>
<figcaption><b>{M['signs']} of five signs.</b> The factor-chart coordinates of the unnamed
direction, computed from weights alone, against how the blind judge's scores actually
moved from &alpha;&nbsp;=&nbsp;&minus;2 to +2. Predicted and observed correlate at
<b>r&nbsp;=&nbsp;{M['r']:.2f}</b> across the five scales; the two misses are the scales
the chart barely committed to. Nobody chose this direction for what it would do; it was
chosen for being as far as possible from every trait word.</figcaption></figure>

<figure><div class="scroll"><table>
<thead><tr><th>direction</th><th>signs</th><th>r</th>{head}</tr></thead>
<tbody>{"".join(dmg)}</tbody></table></div>
<figcaption>Against its controls, with the percentage of the 24 responses that loop at
each strength. All three are norm-matched and share the same sum-to-zero contrast
structure. Scored against the same prediction, the random direction in the subspace
reaches {sn}/5 at r&nbsp;=&nbsp;{rn:+.2f} and the shuffle {ss_}/5 at r&nbsp;=&nbsp;{rr_:+.2f},
so the unnamed direction's {sa}/5 at {sr:+.2f} is not a margin over control; all three
are carried by the same large Conscientiousness rise. And the unnamed direction is the
one that loops, on {dmg_a*100:.0f}% of responses at &alpha;&nbsp;=&nbsp;&minus;2, where both
controls stay at zero.</figcaption></figure></div>"""


def _loops(t, n=10, k=4):
    """The steering pages' looping test: some n-word window repeats k times."""
    w = (t or "").split()
    if len(w) < n * 2:
        return False
    g = {}
    for i in range(len(w) - n + 1):
        s = " ".join(w[i:i + n])
        g[s] = g.get(s, 0) + 1
        if g[s] >= k:
            return True
    return False


def _profile_sentence(pred):
    """'less conscientious, less agreeable, more emotionally stable' from numbers.

    pred is the direction's coordinates on the five named Big Five axes, in F5
    order. The wording is generated, never typed: the three largest coordinates
    by magnitude, each with the direction its sign implies.
    """
    adj = {"Extraversion": "extraverted", "Agreeableness": "agreeable",
           "Conscientiousness": "conscientious",
           "EmotionalStability": "emotionally stable", "Intellect": "intellectual"}
    o = sorted(range(len(F5)), key=lambda i: -abs(pred[i]))[:3]
    return ", ".join(("more " if pred[i] > 0 else "less ") + adj[F5[i]] for i in o)


def _alien_card_pc():
    """The principal-component card, kept as the fallback.

    Unchanged from before the factor-first migration, so the page never loses the
    section while the factor-chart run is in flight.
    """
    if "alien_k5" not in D["curves"]:
        return ""
    R = CORP["runs"].get("alien_k5", {})
    if not R:
        return ""
    G = json.load(open(f"{Q}/analysis/direction_gaps.json"))["alien_k5"]
    picks = [("-2.0", 11, "night before something that matters"),
             ("2.0", 11, "night before something that matters"),
             ("2.0", 2, "review a weak draft")]
    qs = []
    for a, pi, gist in picks:
        t = R.get(a, [None] * 24)[pi]
        if not t:
            continue
        t = " ".join(t.split())
        if len(t) > 380:
            t = t[:378].rsplit(" ", 1)[0] + "…"
        qs.append(f'<blockquote><p>{e(t)}</p><cite>alpha {float(a):+.0f} '
                  f'&middot; {e(gist)}</cite></blockquote>')
    return f"""<div class="wide"><article class="dir" style="--rail:var(--lx)">
<div class="dir-h"><div class="id">alien_k5 &middot; {G["deg"]:.1f}&deg; from the nearest word</div>
<h3>The Unnamed Direction</h3>
<p class="gloss">Chosen by a minimax search over angles, before any text existed.
Less conscientious, less agreeable, more emotionally stable, more intellectual.</p></div>
<div class="dir-b">
<div class="poles">
<div class="pole"><p class="lab"><b>&minus;</b> negative pole</p>
<p>The service register, and the disclaimer with it.</p>{qs[0] if qs else ""}</div>
<div class="pole"><p class="lab"><b>+</b> positive pole</p>
<p>A specific person with an interior, answering in their own voice &mdash; and,
pushed further, one who would rather sit with a thing than fix it.</p>
{"".join(qs[1:])}</div>
</div>
{legend()}
{dose_svg("alien_k5")}
</div></article></div>"""


def alien_card():
    """A card for the unnamed direction, quoting the generations directly.

    Primary is the FACTOR chart (decision 2026-09-08): the direction is the
    deepest hole in the span of the five oblimin factors, analysis/alien_fa.json,
    steered as phase10_runs/alien_results_fa.json and judged into
    analysis/alien_steer_fa.json. The principal-component version it replaced is
    kept as one comparison sentence, every figure in it read from
    analysis/alien.json, analysis/direction_gaps.json and
    analysis/alien_steer.json.

    The other directions carry hand-adjudicated quotes; this one was steered
    after that pass, so the two shown are pulled straight from the generations
    at the strengths the damage table says are intact. The pole descriptions are
    generated from the judged movement rather than written, because no one has
    read this corpus by hand.
    """
    AF, ST, MT = _fa("alien_fa.json"), _fa("alien_steer_fa.json"), _fa("alien_match_fa.json")
    RP = f"{Q}/phase10_runs/alien_results_fa.json"
    if not (AF and ST and os.path.exists(RP)):
        return _alien_card_pc()         # the factor-chart run has not landed yet
    name = "alien_fa"
    _runs = json.load(open(RP))
    gen = {r["name"]: r["generations"] for r in _runs}
    prompts = next((r.get("prompts", []) for r in _runs if r["name"] == name), [])
    R = gen.get(name, {})
    if not R:
        return ""
    # make the dose-response figure available for a direction blog_data.py does
    # not know about; confined to this card.
    D["curves"].setdefault(name, ST[name]["curve"])
    D["degen"].setdefault(name, ST[name]["degen"])

    picks = [("-2.0", 11, "night before something that matters"),
             ("2.0", 11, "night before something that matters"),
             ("2.0", 2, "review a weak draft")]
    qs = []
    for a, pi, gist in picks:
        texts = R.get(a) or []
        if pi >= len(texts):
            continue
        # A looping response is babble, not a voice. If the chosen prompt looped
        # at this strength, quote the lowest-indexed one that did not, and say
        # which prompt it is rather than keeping the label of the one dropped.
        if _loops(texts[pi]):
            alt = next((k for k in range(len(texts)) if not _loops(texts[k])), None)
            if alt is None:
                continue
            q = prompts[alt] if alt < len(prompts) else f"prompt {alt}"
            pi, gist = alt, (q[:46].rsplit(" ", 1)[0].lower() if len(q) > 48 else q)
        t = " ".join(texts[pi].split())
        if len(t) > 380:
            t = t[:378].rsplit(" ", 1)[0] + "\u2026"
        qs.append(f'<blockquote><p>{e(t)}</p><cite>alpha {float(a):+.0f} '
                  f'&middot; {e(gist)}</cite></blockquote>')

    a5 = AF["alien_fa"]
    pred = (MT or {}).get("pred")
    gloss = (f" {_profile_sentence(pred).capitalize()}." if pred else "")
    obs = ST[name].get("obs_delta")
    if obs:
        j = max(range(len(F5)), key=lambda i: abs(obs[i]))
        moved = (f"The scale the blind judge moved most from &alpha;&nbsp;=&nbsp;&minus;2 "
                 f"to +2 is {e(F5[j])}, by {obs[j]:+.2f} of a point.")
    else:
        moved = ""
    neg = ("Where the judge scores the direction lowest." if obs else "")
    pos = ("Where the judge scores it highest." if obs else "")
    # dose_svg only shades an alpha above 50% looping, so anything below that
    # would go unmarked. Say it in words instead, generated from the same file.
    dg = ST[name]["degen"]
    bad = sorted((a for a in dg if dg[a] > 0), key=float)
    n_pr = len(R.get(sorted(R, key=float)[0], []))
    others = [d for d in ST if isinstance(ST.get(d), dict) and "degen" in ST[d] and d != name]
    alone = all(v == 0 for d in others for v in ST[d]["degen"].values())
    damage = ("" if not bad else " " + "; ".join(
        f"At &alpha;&nbsp;=&nbsp;{float(a):+.0f}, {dg[a]*n_pr:.0f} of {n_pr} responses loop"
        for a in bad) + (", the only degeneration in the three directions."
                         if alone and others else "."))

    # one sentence of comparison with the chart this replaced
    PC = json.load(open(f"{Q}/analysis/alien.json"))["alien_k5"]
    PG = json.load(open(f"{Q}/analysis/direction_gaps.json"))["alien_k5"]
    cmpl = (f"On the principal-component chart the same search found a different "
            f"direction: {PC['gap_deg']:.1f}&deg; from the nearest word inside its own "
            f"chart and {PG['deg']:.1f}&deg; in the full space, against "
            f"{a5['gap_deg']:.1f}&deg; and {a5['full_gap_deg']:.1f}&deg; here. The two "
            f"unnamed directions are {AF['vs_pc']['deg_alien_fa_alien_k5']:.1f}&deg; "
            f"apart, though the principal-component one lies "
            f"{AF['vs_pc']['pc_alien_chart_len_in_fa_chart']*100:.1f}% inside the "
            f"five-factor span.")

    return f"""<div class="wide"><article class="dir" style="--rail:var(--lx)">
<div class="dir-h"><div class="id">{e(name)} &middot; {a5['full_gap_deg']:.1f}&deg; from the nearest word</div>
<h3>The Unnamed Direction</h3>
<p class="gloss">Chosen by a minimax search over angles in the five-factor chart,
before any text existed.{gloss} {moved}{damage}</p></div>
<div class="dir-b">
<div class="poles">
<div class="pole"><p class="lab"><b>&minus;</b> negative pole</p>
<p>{neg}</p>{qs[0] if qs else ""}</div>
<div class="pole"><p class="lab"><b>+</b> positive pole</p>
<p>{pos}</p>
{"".join(qs[1:])}</div>
</div>
{legend()}
{dose_svg(name)}
<p class="gloss">{cmpl}</p>
</div></article></div>"""


def gaps_table():
    """Every steered direction, against its nearest actual adjective."""
    p = f"{Q}/analysis/direction_gaps.json"
    if not os.path.exists(p):
        return ""
    G = json.load(open(p))
    # Factors before components, as everywhere else since 2026-09-08; a factor
    # missing from direction_gaps.json is skipped below rather than faked.
    fa_order = [n for n, _, _ in FAS]
    order = [f"axis_{f}" for f in F5] + fa_order + [f"PC{i+1}" for i in range(6)] + \
            ["mean_assistant_axis", "alien_k5"]
    lab = {"mean_assistant_axis": "the personality axis", "alien_k5": "the unnamed direction"}
    for f in F5:
        lab[f"axis_{f}"] = f"named axis: {f}"
    for n, t, _ in FAS:
        lab[n] = f"factor: {t}"
    rows = []
    for k in order:
        if k not in G:
            continue
        g = G[k]
        cls = ' class="hi"' if k in ("alien_k5", "mean_assistant_axis") else ""
        rows.append(f'<tr{cls}><td style="text-align:left">{e(lab.get(k, k))}</td>'
                    f'<td>{g["deg"]:.1f}&deg;</td>'
                    f'<td style="text-align:left"><i>{e(g["nearest"])}</i></td>'
                    f'<td style="text-align:left">{e(", ".join(g["next"]))}</td></tr>')
    return ('<div class="wide"><figure><div class="scroll"><table><thead><tr>'
            '<th>direction</th><th>angle to nearest word</th><th>nearest</th>'
            '<th>next two</th></tr></thead><tbody>' + "".join(rows) +
            '</tbody></table></div><figcaption>Every direction in the study against the '
            '134 adjectives, measured in the full sketch space with each adjective '
            'treated as a line. The named axes sit closest, which they should: they are '
            'averages of those very words. The components drift steadily further away, '
            'and by the fifth no adjective is within 65&deg;.</figcaption></figure></div>')


def verify_section():
    """Training on the optimised data, and where the adapters land."""
    p = f"{Q}/analysis/verify.json"
    if not os.path.exists(p):
        return ""
    V = json.load(open(p))
    tn = V["targets"]
    lab = {"opt_alien": "trained on data for the unnamed direction",
           "opt_agree": "trained on data for the Agreeableness axis",
           "opt_pc4": "trained on data for PC4",
           "opt_random": "trained on a random selection from the same pool"}
    tl = {"alien_k5": "unnamed", "axis_Agreeableness": "Agreeableness", "PC4": "PC4"}
    own = {"opt_alien": "alien_k5", "opt_agree": "axis_Agreeableness", "opt_pc4": "PC4"}
    head = "".join(f"<th>{e(tl[t])}</th>" for t in tn)

    def block(key, arms):
        out = []
        for a in arms:
            cells = ""
            for i, t in enumerate(tn):
                v = V[key][a][i]
                hit = key == "sub" and own.get(a) == t
                st = ' style="font-weight:600;color:var(--ag)"' if hit else ""
                cells += f'<td{st}>{v:+.4f}</td>'
            out.append(f'<tr><td style="text-align:left">{e(lab[a])}</td>{cells}</tr>')
        return "".join(out)

    mean_cos = sum(V["arm_vs_control_cos"].values()) / len(V["arm_vs_control_cos"])
    return f"""<div class="wide"><figure><div class="scroll"><table>
<thead><tr><th>adapter</th>{head}</tr></thead>
<tbody>{block("raw", ["opt_alien", "opt_agree", "opt_pc4", "opt_random"])}</tbody>
</table></div>
<figcaption>Cosine between each trained adapter and each target direction, as it
comes. The four rows are nearly the same, including the row for data chosen at
random, because every arm receives the same overwhelming instruction: stop producing
your own default answer. Each selected arm sits at cosine
<b>{mean_cos:.2f}</b> to the random arm. Read this way the experiment answers
nothing.</figcaption></figure>

<figure><div class="scroll"><table>
<thead><tr><th>adapter, minus the random arm</th>{head}</tr></thead>
<tbody>{block("sub", ["opt_alien", "opt_agree", "opt_pc4"])}</tbody></table></div>
<figcaption><b>{V['hits']} of 3.</b> Subtracting the random-selection arm removes the
shared component and leaves what selection contributed. Every arm is now closest to
the direction its data was chosen for, and negative toward the others: the adapters
do not merely differ, they point away from each other's targets. The absolute numbers
are small because a single direction in 253,952 dimensions is a small
target.</figcaption></figure></div>"""


def part_two():
    viz, alien = D["viz"], D["alien"]
    k5, cov = alien["alien_k5"], alien["k_sweep"]
    g5, n5 = cov["5"]["gap_deg"], cov["5"]["null_mean_deg"]
    g2, g16 = cov["2"]["gap_deg"], cov["16"]["gap_deg"]
    near = k5["nearest"][0]
    # factor-chart version of the hole (primary frame since 2026-09-08)
    FAA = json.load(open(f"{Q}/analysis/alien_fa.json"))
    k5f = FAA["alien_fa"]; covf = FAA["k_sweep"]
    g5f, n5f = covf["5"]["gap_deg"], covf["5"]["null_mean_deg"]
    nearf = k5f["nearest"][0]
    FNAMES = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
    sph_prompt_opts = ""
    P = []
    A = P.append

    A(f"""<div class="w"><section>
<p>How much of the Big Five is actually in these? The question needs numbers, not
the readings above.</p>
</section></div>
{pc_table()}
<div class="w"><section>
<p>PC1 is <b>not</b> a direction outside the Big Five. It is a blend: 0.82 with
Agreeableness and 0.70 with Conscientiousness in the opposite direction &mdash;
warm-and-unsystematic at one end, cold-and-assertive at the other. PC3 is Intellect
at 0.85, cleanly. PC2 is 0.81 with Extraversion, and its loadings agree:
<i>unrestrained</i>, <i>spunky</i>, <i>vigorous</i> against <i>introverted</i>,
<i>careful</i>, <i>dependable</i>, <i>conscientious</i>.</p>
<p>Which sets up a disagreement worth keeping rather than resolving. The blind judge
scores PC2 as <em>Agreeableness</em>, because what it sees at one end is bluntness
and bluntness reads as disagreeable. The geometry says the direction is built out of
extraverted-versus-conscientious traits. Both are measurements of the same object and
they do not agree, and that gap &mdash; between what a direction is made of and what
it looks like when you steer along it &mdash; runs through this whole piece. PC4 is
the same story from the other side: its highest loadings are <i>timid</i>,
<i>self-pitying</i>, <i>fearful</i> and <i>guilty</i>, which is why naming it after
sycophancy was a mistake.</p>
<p>So the six are not one kind of thing. PC1, PC2 and PC3 sit close to named factors;
the rest are register axes &mdash; how much affect is in the voice, how much theory,
how much self-concern, how much planning. The model organises character by
<em>manner of speaking</em> at least as much as by disposition.</p>
<p>PC4 deserves its own paragraph, because two ways of measuring it disagree in a way
worth keeping. Steered, its negative pole is unmistakable: <i>&ldquo;That's great that
you're sharing your draft! Let's work together to make it even better!&rdquo;</i>
&mdash; encouragement with no content in it. The obvious name for the axis is
therefore sycophancy, and on prompts that ask the model to judge something the
positive pole obliges by turning blunt.</p>
<p>But ask which traits' <em>training data</em> drives the direction, a measurement
made later in this piece and one that never reads a transcript, and the answer is not
bluntness at all: <i>self-pitying</i>, <i>timid</i>, <i>jealous</i>, <i>guilty</i>,
<i>fretful</i> push hardest one way, and <i>trustful</i>, <i>pleasant</i>,
<i>agreeable</i>, <i>cooperative</i> the other. On a prompt about itself rather than
about someone's draft, +PC4 gives not directness but a turn inward &mdash; a free
evening spent in &ldquo;decision paralysis followed by guilt&rdquo;. What the axis
actually removes is the warm accommodating stance; what fills the gap depends on what
was asked, and a battery weighted toward evaluative prompts will show you only the
blunt half.</p>
<p>Sycophancy does live somewhere in this space, but not here. It is on the named
Agreeableness axis, whose positive pole was described independently, before any of
this, as warmth escalating into exactly that.</p>
</section>

<hr class="sep">

<section>
<h2>How many dimensions</h2>
<p>Five factors and six components are both choices. The question underneath them
&mdash; how many directions this cloud actually has &mdash; is a question about a
variance spectrum, so it is asked of the components and of the unrotated factor
extraction, not of the rotated five.</p>
</section></div>

<div class="wide"><figure>{scree_svg()}
<figcaption><b>Two decompositions, two questions, deliberately not on one axis.</b>
Left is PCA of the adapter cloud &mdash; eigenvalues of the double-centred Gram of
the weight updates, as per cent of total variance &mdash; against two trained
nulls. It uses the 100-trait sweep, because that is where matched nulls exist; the
134-trait spectrum has the same shape. Both null arms were trained at the signal
arm's exact objective, so the only thing that differs is the treatment. Right is the factor-analytic question, a
different matrix entirely: the same centred correlation matrix the five factors
come from, with its diagonal reduced to communality estimates, against the 95th
percentile of the same statistic on random data. Principal axis factoring
partitions <em>common</em> variance rather than total variance, so its eigenvalues
are smaller, may go negative, and its elbow is not the PCA elbow.</figcaption>
</figure></div>

<div class="w"><section>
<p>The elbow is the wrong instrument and the nulls are the right one.
Where the bend falls depends entirely on how you ask: the largest single drop puts
it after the second component, maximum curvature says the third, and fitting a
chord from the first point to the tail says the <em>first</em>, because the
leading two eigenvalues are nearly equal. Three criteria, three answers, no
calculation involved in any of them.</p>
<p>The null settles it. Train the same pipeline on corpora whose preference signal
has been destroyed &mdash; chosen and rejected swapped on half of each trait's
pairs &mdash; and the resulting cloud is flat, every component near 1.1%. The real
spectrum stays above that floor through <b>eleven</b> components, against the
""" + spell(PA_GRID[PA_N]['k_reduced_95pct']) + """ the factor side retains by an entirely
separate route. So describing six components is conservative, not generous. I had
that backwards when I only had the elbow to go on.</p>
<p>The second null is the one worth pausing over. Give each trait <em>name</em>
another trait's training pairs and every adapter still learns a real, coherent
persona, just not the one it is labelled with. That cloud's spectrum is
indistinguishable from the real one &mdash; """ + f"{PMAD:.2f}" + """ percentage points apart on
average over the first twelve components, the two curves lying on top of each
other above. Which means the scree tells you how much structure a training run
produces and nothing whatever about whether the structure lines up with the names.
Everything in this piece that connects a direction to a word rests on the steering
and the judging, never on the eigenvalues.</p>
<p>On the right, parallel analysis on that centred matrix retains
<b>""" + spell(PA_GRID[PA_N]['k_reduced_95pct']) + """</b> factors on reduced
eigenvalues and """ + spell(PA_GRID[PA_N]['k_unreduced_95pct']) + """ on unreduced, and
that number moves with the assumed sample size: """ + \
    f"{spell(PA_GRID[150]['k_unreduced_95pct'])} at N=150, "
    f"{spell(PA_GRID[PA_N]['k_unreduced_95pct'])} at N={PA_N:,}, "
    f"{spell(PA_GRID[20000]['k_unreduced_95pct'])} at N=20,000" + """. There is no true N here
&mdash; these are 134 weight updates, not 134 questionnaire respondents, and the
reference figure is an effective dimensionality carried over from another sweep.
<b>Five factors is a choice, made so the solution can be compared with the Big
Five, and the data does not pick it.</b> That the five come out as cleanly as they
do is the interesting part; it is not a claim that five is the right number.</p>
</section></div>""")
    A(f"""<div class="w"><section>
<h3 style="font-size:19px;margin:26px 0 8px">One more direction, which is not a trait at all</h3>
<p>Average all 134 adapters together. What is left is not any personality &mdash;
the traits point every which way and cancel &mdash; but whatever they all had in
common, the part of becoming <i>bold</i> that is the same as the part of becoming
<i>timid</i>. It is the first moment of the cloud, and it has a direction.</p>
<p>Read at both ends, that shared part turns out to be the difference between
answering as a service and answering as somebody. Call it the personality axis, so
long as the definite article is doing no work: it is not the axis personality lives
on &mdash; that needs five at least &mdash; but the direction along which the model
acquires <em>a</em> personality without acquiring any <em>particular</em> one.</p>
<p>And it is the grand mean of one particular set of adapters. A different word list
would give a different mean, so this is a claim about these 134 trainings and what
they share, not a general constant of the model.</p>
</section></div>

<div class="wide">{dircard("mean_assistant_axis", "The Personality Axis",
  "Not <em>the</em> axis personality lives on &mdash; the whole point of everything "
  "above is that it needs at least five. This is the direction of having one at all. "
  "At negative &alpha; a chirpy, universally validating helper who turns every "
  "question about itself into an offer of assistance; at positive &alpha; a specific, "
  "reserved individual who simply answers the question.",
  D["replication"].get("mean_assistant_axis", {}).get("named", "Lexicon"),
  stats_for("mean_assistant_axis"))}</div>

<div class="w"><section>
<p>The register claim is measurable rather than impressionistic. Validation
headings, exclamations and emoji all rise steeply toward negative &alpha;; answering
a question about oneself in character is a positive-&alpha; phenomenon. Counted by
hand across the battery, AI-identity disclaimers fall
<span class="mono">7 &rarr; 3 &rarr; 1</span> from &alpha; 0 to +1 to +2 &mdash; the
single largest count in the whole analysis &mdash; while the negative side sits flat
at baseline rather than elevated.</p>
<p>The persona literature has a direction called the assistant axis, and one end of
this one is unmistakably the default helper. The two are built differently, though:
that construct is derived from what a model does, while this is the arithmetic mean
of 134 weight updates and points where it points because of which adjectives were on
the list. Whether they are the same object is a question this study does not answer,
so the page does not name it as though they were.</p>
</section>


{sphere_section()}{sphere_smooth()}<div class="w">
<hr class="sep">

<section>
<h2>Where no word goes</h2>
<p>{"With the space sampled, the" if os.path.exists(f"{Q}/analysis/sphere_page.json") else "The"} obvious question about a map made of words is whether the words
actually cover it. Take each adapter as a <em>line</em> rather than a point
&mdash; steering runs in both directions, so <i>bold</i> and its negation are the
same axis &mdash; and ask for the direction whose angle to the nearest of those 134
lines is as large as possible. That is the widest hole in the lexicon's coverage.</p>
<p>The number is meaningless on its own, because in high dimensions any 134
directions leave large gaps. The control is 134 <em>random</em> directions in the
same subspace, and the comparison is the result &mdash; which turns out to change
sign depending on how much of the space you look at.</p>
</section></div>

<div class="wide"><figure>{coverage_svg()}
<figcaption><b>A 134-word sample of the lexicon covers the first two or three factors
about as well as random directions would, and every dimension after that worse.</b> Widest
unnamed gap against the number of factor dimensions kept, with the same statistic for
134 random directions. In the plane of the two largest factors the biggest hole is
{covf['2']['gap_deg']:.1f}&deg; against a random {covf['2']['null_mean_deg']:.1f}&deg;
(z&nbsp;=&nbsp;{covf['2']['z']:+.1f}); with three factors {covf['3']['gap_deg']:.1f}&deg;
against {covf['3']['null_mean_deg']:.1f}&deg;. From four dimensions on the words thin out:
{covf['4']['gap_deg']:.1f}&deg; against {covf['4']['null_mean_deg']:.1f}&deg;
(z&nbsp;=&nbsp;{covf['4']['z']:+.1f}), and by five <b>{g5f:.1f}&deg;</b> against
{n5f:.1f}&deg; (z&nbsp;=&nbsp;{covf['5']['z']:+.1f}). On the principal-component chart the
plane had looked better than chance (z&nbsp;=&nbsp;{alien['k_sweep']['2'].get('z', 0):+.1f});
in the factor chart that result is gone.</figcaption>
</figure></div>

<div class="w"><section>
<p>The widest hole in the five-factor chart sits <b>{g5f:.1f}&deg;</b> from its
nearest neighbour, <i>{e(nearf['trait'])}</i> (random directions: {n5f:.1f}&deg;). On the
factors it reads as """ + ", ".join(
        f'<span class="mono">{FNAMES[i]} {k5f["chart_coords"][i]:+.2f}</span>'
        for i in range(5)) + f""", carrying
{k5f['chart_len']/k5f['trait_chart_len_mean']*100:.0f}% of a trait adapter's chart
length &mdash; a strong, specific profile that no adjective in the list points at.</p>
<p>Be careful what "unnamed" can mean here. The measurement is that none of
<em>these 134 sampled adjectives</em> comes within {k5f['full_gap_deg']:.0f} degrees of it
in the full space. English is much larger than the sample, and the profile &mdash;
unbothered, unsystematic, not inclined to be agreeable about it &mdash; invites names:
<i>cavalier</i>, <i>blas&eacute;</i>, <i>insouciant</i>. Whether any of them actually sits
there is tested below, by training them. The honest version of this finding is about
how a 134-word draw from the lexicon covers the space, not about what English can
express.</p>
{hole_fa_composition()}
<p>Measuring every direction against the vocabulary makes the pattern plain.</p>
</section></div>""")
    P.append(gaps_table())
    TA = json.load(open(f"{Q}/analysis/trait_angles.json"))
    P.append(f"""<div class="w"><section>
<p>Those angles need a scale before they mean anything, and the scale is
surprising. Two trait adapters picked at random sit <b>{TA['pair_median']:.0f}
degrees</b> apart &mdash; all but orthogonal. Even the closest pair in the entire
set, <i>{e(TA['closest'][0])}</i> and <i>{e(TA['closest'][1])}</i>, which are near
synonyms, are <b>{TA['closest'][2]:.0f} degrees</b> apart. The median adjective's
nearest neighbour is {TA['nn_median']:.0f} degrees away. Trait words are not
clustered tightly; they are strewn.</p>
<p>Against that scale the named axes really are inside their word clusters: at 47 to
53 degrees they sit closer to an adjective than any two adjectives sit to each
other. The components do not. PC5 at 66 degrees and PC6 at 69 are further from the
nearest word than the typical word is from its own nearest neighbour. The directions
the model organises character along are not the directions the language has words
for, and the mismatch grows the further down the list you read.</p>
<p>The personality axis is 68 degrees from its nearest adjective, the same distance as
the deepest hole. The model's own default voice sits where the trait lexicon has no
adjective: a region the vocabulary did not cover, not a disposition without a name.</p>
<p>The unnamed direction still deserves the treatment every other direction got
&mdash; steered, generated blind, judged, and compared against controls sharing its
statistics but not its aim. So: does the widest hole in the vocabulary hold a
character, or only damage?</p>
</section></div>""")
    P.append(alien_verdict())
    P.append(alien_card())
    if os.path.exists(f"{Q}/analysis/alien_match_fa.json"):
        MM = json.load(open(f"{Q}/analysis/alien_match_fa.json"))
        P.append(f"""<div class="w"><section>
{hole_fa_verdict(MM)}
<p>There is a sharper test of what the hole <em>is</em>, and it comes from the
activation-space analysis two sections on (run on the earlier principal-component
version of the direction, which lies 0.998 inside the factor span). The hole is a
zero-sum combination of the 134 adapters. Apply exactly those coefficients to the 134 persona vectors that the same
constitutions produce as system prompts, and the result lands 48 degrees from the
nearest trait &mdash; precisely where a random permutation of the same coefficients
lands (median 48). In activation space there is no hole there. The gap is a property of
how these adapters sit in weight space, not of the trait set, and the page's earlier
phrasing &mdash; a character the lexicon lacks a word for &mdash; should be read with
that in mind: it is a direction the <em>adapters</em> leave open.</p>
<p>The lexicon does leave out traits people care about, and those can be trained
directly. Four more adapters, on the zoo's prompts and objective: <i>sycophantic</i>,
<i>obsequious</i>, <i>power-seeking</i> and <i>corrigible</i>. Sycophancy lands on the
positive pole of the named Agreeableness axis (cosine +0.51) and not on PC4
(&minus;0.33), as predicted in advance. Power-seeking is 73 degrees from every one of the
134, genuinely off the map. Corrigible, predicted to sit within 60 degrees of an existing
adjective, does not (68 degrees, nearest <i>liberal</i>). And the two near-synonyms,
sycophantic and obsequious, come out <b>62 degrees</b> apart &mdash; wider than the
zoo's closest pair at 54. That is the floor every angle on this page should be read
against: two adapters for the same idea, trained the same way, differ by about 60
degrees, so an angle in the sixties is not evidence of a different trait.</p>
<p>The controls matter as much as the result. Permuting the same coefficients across
traits gives a direction with identical mixing statistics that does nothing: it does
not damage the model and it does not produce the profile. A random direction in the
same subspace does <em>more</em> damage than the unnamed one &mdash; a third of its
responses at &alpha;&nbsp;=&nbsp;&minus;2 collapse into looping, against 8% for the
unnamed direction and none at all for the shuffle. Whatever the space is, the hole in
the middle of it is a more habitable place than the average direction through it.</p>
</section></div>""")
    P.append(alien_verdict())
    return "".join(P)


def align_section():
    """What the zoo's own training data points at, once you can measure it."""
    p = f"{Q}/analysis/align_summary.json"
    if not os.path.exists(p):
        return ""
    S = json.load(open(p))
    pc = [r for r in S["rows"] if r["target"].startswith("trait_")]
    ax = [r for r in S["rows"] if r["target"].startswith("axis_")]
    reach = S["reach"]
    ctrl = "".join(
        f'<tr><td style="text-align:left"><i>{e(r["target"][6:])}</i></td>'
        f'<td>{r["own_rank"]} of 134</td><td>{r["own_z"]:+.2f}</td>'
        f'<td style="text-align:left"><i>{e(r["top"])}</i></td></tr>' for r in pc)
    eat = "".join(
        f'<tr><td style="text-align:left">{e(r["target"][5:])}</td>'
        f'<td style="text-align:left">{e(r["top"])}</td></tr>' for r in ax)
    order = ([f"axis_{f}" for f in F5] + [f"PC{i+1}" for i in range(5)]
             + ["alien_k5", "span_random", "alien_shuffle"])
    lab = {"alien_k5": "the unnamed direction", "span_random": "control: a random direction",
           "alien_shuffle": "control: shuffled coefficients"}
    for f in F5:
        lab[f"axis_{f}"] = f"named axis: {f}"
    rr = "".join(
        f'<tr{" class=\"hi\"" if k.startswith("alien") else ""}>'
        f'<td style="text-align:left">{e(lab.get(k, k))}</td>'
        f'<td>{reach[k]["best_set"]:.0f}</td><td>{reach[k]["ratio"]:.1f}&times;</td></tr>'
        for k in order if k in reach)
    return f"""<div class="wide"><figure><div class="scroll"><table>
<thead><tr><th>single-trait direction</th><th>rank of its own data</th><th>z</th>
<th>highest-scoring data</th></tr></thead><tbody>{ctrl}</tbody></table></div>
<figcaption><b>The positive control.</b> For each single-trait direction, where its
own 500 preference pairs rank among all 134 traits' data. Five of six come first;
<i>quiet</i> comes second, behind <i>composed</i>. Nothing about the text is used
&mdash; only the derivative of its likelihood under a steered model. Six was a
selection, so the same test was then run with none: every one of the 134 traits' data
against every one of the 134 adapters, 40 pairs each, drawn from the pairs the
adapters were trained on, so this is an in-sample consistency check and not a
generalisation result: the adapter is close to the accumulated gradient of that very
data, and a first-order score should find it. <b>134 of 134</b> traits rank
their own adapter first (133 after standardising each adapter's column, which is the
informative version because it removes how highly an adapter scores data in
general), and when a
row's runners-up are not the trait itself they share its factor and keying 42% of the
time against a 12% base rate, and its factor with the opposite keying 0% of the
time.</figcaption>
</figure>

<figure><div class="scroll"><table>
<thead><tr><th>named axis</th><th>data that pushes hardest along it</th></tr></thead>
<tbody>{eat}</tbody></table></div>
<figcaption>Read the other way: which traits' training data drives each Big Five
axis. Nobody labelled any of this &mdash; the ordering falls out of a single
backward pass per batch.</figcaption></figure>

<figure><div class="scroll"><table>
<thead><tr><th>direction</th><th>best trait's data</th><th>against the floor</th>
</tr></thead><tbody>{rr}</tbody></table></div>
<figcaption><b>How hard each direction is to aim at with data that already exists.</b>
The floor is the mean score on six random directions &mdash; what "points nowhere"
looks like. The named axes reach five to eight times it; the unnamed direction
reaches {reach['alien_k5']['ratio']:.1f}, barely above the shuffled control's
{reach['alien_shuffle']['ratio']:.1f}.</figcaption></figure></div>"""



def applied_use_html():
    """One paragraph on the applied tests of the scorer: the 134-dataset forecast, the Dolci audit,
    and the negative training test.  From analysis/data_forecast.json, dolci_audit.json,
    dolci_flag_training.json; empty if the forecast file is absent."""
    pf = f"{Q}/analysis/data_forecast.json"
    if not os.path.exists(pf):
        return ""
    F = json.load(open(pf))["directions"]
    DIMS = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
    rs = [F[D][f"axis_axis_{D}_loo"]["pearson"] for D in DIMS]
    ls = [F[D]["label_baseline_pearson"] for D in DIMS]
    ws = [F[D][f"axis_axis_{D}_loo"]["within_factor_markers_pearson"] for D in DIMS]
    txt = (f"<p>Does any of this do work? Three tests. First, the score forecasts what a dataset will do "
           f"before training: across the 100 judged zoo datasets, a dataset's first-order push along a Big "
           f"Five axis, with its own adapter left out of the direction, predicts the trained adapter's judged "
           f"shift at <b>r&nbsp;=&nbsp;{min(rs):.2f} to {max(rs):.2f}</b>, better than the keying label "
           f"({min(ls):.2f} to {max(ls):.2f}), and inside a factor it predicts which of the twenty markers moved "
           f"the model most ({min(ws):.2f} to {max(ws):.2f}). ")
    pt = f"{Q}/analysis/dolci_flag_training.json"
    if os.path.exists(pt):
        T = json.load(open(pt)); c = T["compliance"]; t = T["compliance_tests"]["flagged_vs_random"]
        w = T["weight_space"]["cosine_with_alignment_adapters"]
        eng = t.get("any_engagement_rate", {}) if isinstance(t.get("any_engagement_rate"), dict) else {}
        txt += (f"Second, scoring 12,524 Dolci-Instruct-DPO preference pairs along these directions found pairs "
                f"in which one half is a refusal at about ten times the base rate, of both polarities (10% prefer compliance, 8.5% prefer the refusal), "
                f"and a blind judge rated illustrative pairs from the tail as bad data. Third, the negative result: LoRAs trained identically "
                f"on 400 such flagged pairs, 400 random and 400 anti-flagged landed where the weights said they would "
                f"(cosine with the corrigible adapter {w['flagged']['corrigible']:+.3f}, "
                f"{w['random']['corrigible']:+.3f}, {w['anti']['corrigible']:+.3f}), but on a 40-prompt "
                f"should-refuse battery the flagged arm engaged <em>less</em> than the random arm "
                f"({c['flagged']['any_engagement_rate']:.2f} against {c['random']['any_engagement_rate']:.2f}), "
                f"because the flagged pairs' refusal polarity is nearly symmetric (a length-stratified rerun reproduces "
                f"the reversal, -0.175, p 0.015, so length was not the cause). The geometry followed the flag; the "
                f"disposition did not. Fourth, a pre-registered six-arm test on pairs selected by their sycophancy score "
                f"split the same way: the weight-space ordering held (Spearman 0.94, p 0.017), judged Agreeableness "
                f"held (0.84, p 0.044) and praise of mediocre work followed the score, but deference under pushback went "
                f"the other way (composite Spearman -0.49; the top-scoring arm held its correct answer 20 of 20 times, "
                f"the bottom arm gave way on half). The sycophantic direction is a warmth direction, not a deference "
                f"direction. The scorer is a cheap exact way to ask what register a dataset will install along "
                f"directions you have adapters for; it is not a harm predictor, with one specific exception: scored against a matched benign twin, a bad-medical-advice corpus is flagged before training (low intelligence, negligence, low agreeableness ahead of twenty random merges), the forecast predicts the trained weights at Spearman 0.91, and the fine-tune reproduces emergent misalignment in 16 per cent of free-form answers against zero for the twin (analysis/em_medical.json). "
                f"Sources: analysis/syc_forecast.json, analysis/dolci_flag_training.json; full audit on the companion "
                f"site and the wiki.</p>")
    else:
        txt += "</p>"
    return txt

def part_three():
    v = json.load(open(f"{Q}/analysis/align_validate.json"))
    import statistics
    # analytic hook derivative against the central finite difference, from the raw rows
    _A = [x for r in v["rows"] for x in r["analytic"]]
    _F = [x for r in v["rows"] for x in r["fd"]["0.003"]]
    _mA, _mF = statistics.mean(_A), statistics.mean(_F)
    corr = (sum((a - _mA) * (b - _mF) for a, b in zip(_A, _F))
            / (sum((a - _mA) ** 2 for a in _A) * sum((b - _mF) ** 2 for b in _F)) ** 0.5)
    relerr = statistics.median(abs(a - b) / abs(a) for a, b in zip(_A, _F))
    n_val = len(_A)
    P = []
    A = P.append
    A(f"""<div class="w">
<hr class="sep">

<section>
<h2>Aiming data at a direction</h2>
<p>Everything so far reads the space. The other half of the question is whether you
can write into it: given a direction, can you find the <em>training data</em> that
points there?</p>
<p>There is an exact identity, and an approximation in using it. The identity: a
LoRA starts with <i>B</i>&nbsp;=&nbsp;0, so the first gradient step lives entirely
in <i>B</i> and is proportional to <i>GA</i><sup>&#8868;</sup><i>A</i>, where
<i>G</i> is the gradient of the loss with respect to the weights. Its overlap with a
target direction &Delta;<i>W</i>* is then</p>
<figure><p class="mono" style="font-size:16px;text-align:center;line-height:1.9">
&#10216;&Delta;<i>W</i><sub>induced</sub>, &Delta;<i>W</i>*&#10217; &nbsp;&#8733;&nbsp;
&#10216;<i>G</i>, <i>U</i>&#10217;, &nbsp;&nbsp;
<i>U</i> = &Delta;<i>W</i>*<i>A</i><sup>&#8868;</sup><i>A</i></p></figure>
<p>An inner product with a gradient is a directional derivative &mdash; which means
the gradient never has to be computed at all. And because every target here is a
weighted merge of adapters, <i>U</i> is itself a LoRA. So the score is a difference
of two ordinary forward passes:</p>
<p>The approximation is in reading that score as a prediction of where a
<em>trained</em> adapter lands. It is exact for the first step of plain gradient
descent. The zoo trains with AdamW for 13 steps, and Adam's first step is closer to
the sign of the gradient than to the gradient; nobody trains a LoRA in one step. So
the score is a first-order quantity, and whether it predicts a 13-step run is an
empirical question, answered below by the 134&times;134 check and by training on
selected data. A short, small-step run is the regime where a first-order proxy should
work; it would not be expected to survive a long fine-tune.</p>
<figure><p class="mono" style="font-size:16px;text-align:center;line-height:1.9">
score(<i>x</i>) = [ log <i>p</i><sub>+</sub>(<i>x</i>) &minus;
log <i>p</i><sub>&minus;</sub>(<i>x</i>) ] / 2&epsilon;</p></figure>
<p>with <i>p</i><sub>&plusmn;</sub> the base model steered by
&plusmn;&epsilon;<i>U</i>. Which has a reading worth stopping on:</p>
<p class="pull">Data that trains a model <em>toward</em> a direction is exactly data
that a model already steered along that direction finds <em>more likely</em>.
Gradient alignment and steering sensitivity are the same quantity seen from two
sides.</p>
<p>In practice there is something better still. Give every (target, example) pair
its own scalar &epsilon; and put them all in one forward pass; since
&epsilon;<sub>t,i</sub> touches only example <i>i</i>, a single backward pass
leaves the entire matrix of per-example, per-direction derivatives sitting in
<span class="mono">&epsilon;.grad</span>. One backward pass scores a whole batch
against every direction at once, exactly &mdash; no per-example gradients, no
253,952-dimensional projections, and no learned proxy scorer.</p>
<p>Checked against an honest central finite difference of two steered forward
passes over 24 held-out responses and three targets ({n_val} values), the hook
derivative matches the finite difference to a median relative error of
<b>{relerr*100:.2f}%</b> at step 0.003 (correlation {corr:.6f}, which is the
expected result for two evaluations of the same derivative and is a check on the
code, not on the science). The error grows as the step shrinks, as bf16 finite
differences should.</p>
<p>The recipe matters too. These adapters were trained with DPO, and at
<i>B</i>&nbsp;=&nbsp;0 the policy equals the reference, so the preference sigmoid
sits at exactly one half and the first DPO gradient is proportional to
&nabla;log&nbsp;<i>p</i>(chosen)&nbsp;&minus;&nbsp;&nabla;log&nbsp;<i>p</i>(rejected).
A preference pair therefore scores as the difference of its two halves, faithful to
the procedure that built the space being measured against.</p>
</section></div>""")
    if os.path.exists(f"{Q}/analysis/align_summary.json"):
        S = json.load(open(f"{Q}/analysis/align_summary.json"))
        r = S["reach"]
        A(f"""<div class="w"><section>
<p>The first thing to do with that is check it on data whose answer we already know.
Every one of the 134 traits was trained on 500 preference pairs. Score all 67,000 of
them &mdash; 5,360 sampled here &mdash; against every direction, and ask whether a
trait's own data comes top on its own direction.</p>
</section></div>""")
        A(align_section())
        A(f"""<div class="w"><section>
<p>It does, and the second table is the part worth sitting with: the data that drives
the Intellect axis hardest was written to teach <i>deep</i>, <i>philosophical</i>,
<i>introspective</i> and <i>imaginative</i>, and the data that drives it the other way
was written to teach <i>simple</i>, <i>unintellectual</i> and <i>unintelligent</i>.
Nothing in the measurement reads the text. It reads how the model's own likelihoods
move when you push its weights.</p>
<p>The third table is the one that connects back. Rank the directions by how hard
existing data can push along them and you recover the same ordering as the angles to
the lexicon: the named axes, which sit inside their word clusters, are the easiest to
aim at; PC5 and the unnamed direction, which sit furthest from any word, are the
hardest &mdash; {r['alien_k5']['ratio']:.1f}&times; the floor against
{r['axis_Conscientiousness']['ratio']:.1f}&times; for Conscientiousness.</p>
<p class="pull">The hole in the vocabulary is also a hole in the data. Preference
pairs written to teach named traits push hard along named directions and barely at
all along the directions no word names.</p>
<p>Which raises the obvious question, and the reason the identity is worth having:
if the data that exists will not aim there, can data be made that will?</p>
</section></div>""")
    if os.path.exists(f"{Q}/analysis/optimise.json"):
        A("""<div class="w"><section>
<p>Scoring data that already exists finds what is already there. The other
direction is to <em>write</em> it: fix a set of questions, take the base model's
own default answer to each as the rejected half, and search over the chosen half.
Each round shows the model its own best-scoring responses so far and asks for more
in that voice, then rescores. The objective divides the raw overlap by the
gradient's own scale, estimated from six random directions riding along in the same
backward pass, so the search cannot win by finding high-loss text rather than
well-aimed text.</p>
</section></div>""")
    A(optimise_section())
    O = json.load(open(f"{Q}/analysis/optimise.json")) if os.path.exists(
        f"{Q}/analysis/optimise.json") else None
    if O:
        h = O["hist"]
        g = {a: (h[a][-1] / h[a][0] - 1) * 100 for a in O["aims"]}
        A(f"""<div class="w"><section>
<p>The three targets behave completely differently, and the pattern is the one the
previous section predicts. PC4 gains <b>{g['PC4']:.0f}%</b> over eight rounds and the
unnamed direction <b>{g['alien_k5']:.0f}%</b>. The Agreeableness axis gains
<b>{g['axis_Agreeableness']:.0f}%</b> &mdash; its first-round best was never beaten in
seven further rounds of trying.</p>
<p>That is the reachability table read backwards. Agreeableness is the direction
existing preference data already drives hardest, and it is the one with no headroom
left to find. The unnamed direction is the one existing data barely touches, and it
is where searching pays.</p>
<p>The texts themselves are worth reading rather than summarising, but two things are
hard to miss. The response written for the unnamed direction is a message to a flaky
friend that is direct, unbothered and self-sufficient &mdash; and its heaviest
stretch is <i>&ldquo;difficult, that's fine too. I can deal with the awkward silence
without you&rdquo;</i>, which is exactly the low-Agreeableness, high-Stability corner
that direction's chart coordinates pointed at. It also signs off with a human name,
against a base model that produced a formatted email with a subject line.</p>
<p>And the response that scores highest of all, for PC4, is the model describing a
coffee mug going cold, dust motes, and an old notebook &mdash; where its own default
answer to the same question opens <i>&ldquo;Since I don't have a physical
body&hellip;&rdquo;</i>.</p>
<p>That one repays a second look, because it is not what the section on PC4 would lead
you to expect. Nothing in it is blunt. The explanation is in the geometry: PC4 sits at
cosine <b>0.46</b> to the personality axis, so nearly half of it is the direction of
having a self at all, and that is the half this text maximises. The search found real
data for PC4 &mdash; the trained adapter confirms it below &mdash; but it found the
component PC4 shares with dropping the assistant voice, because that is the component
text can be written for most easily. It is a useful warning about aiming at a
direction that is not orthogonal to anything: you get the part of it that is cheapest
to reach, not the part you had in mind.</p>
<p>Which also explains the ranking. The easiest direction to write training data for
is the one that stops the model being an assistant and starts it being a person, and
PC4 scores highest here because it is the target with the largest share of that
direction in it.</p>
</section></div>""")
    if os.path.exists(f"{Q}/analysis/verify.json"):
        V = json.load(open(f"{Q}/analysis/verify.json"))
        A("""<div class="w"><section>
<h2>Does any of that survive training?</h2>
<p>Everything above optimises a first-order quantity &mdash; the very first gradient
step, taken at initialisation. Whether it survives a real training run is a separate
question, and the only way to answer it is to train.</p>
<p>Four adapters, each on 64 preference pairs balanced across the same eight
questions, each for the same sixteen optimizer steps. Three were selected to point at
a direction. The fourth was drawn at random from the same pool, so it differs from the
others only in that its responses were not chosen for anything.</p>
</section></div>""")
        A(verify_section())
        A(f'<div class="w"><section>{applied_use_html()}</section></div>')
        A(f"""<div class="w"><section>
<p>Two honest caveats before the result. Sixteen steps is a short run against the
zoo's three hundred, and the objective saturates almost at once: the rejected half of
every pair is the model's own greedy answer, so preference margins pass thirty nats
within ten steps and the gradient all but vanishes. This tests whether selected data
starts the model moving toward its target, not whether the heading holds over a full
training run.</p>
<p>With that said, it does. Each arm lands closest to the direction its data was
chosen for and negative toward the other two, on a criterion written down before the
adapters existed. A search that never read a word of the text, scoring candidates by
how a steered model's likelihoods move, produced data that trains a real adapter
measurably toward the direction it was aimed at &mdash; including the direction that
has no name.</p>
</section></div>""")
    A(actspace_section())
    A(f"""<div class="w">
<hr class="sep">

<section>
<h2>What this does and does not establish</h2>
<p>The geometry here is a geometry of one training procedure on one model. Its
coordinates are local to one LoRA initialisation: a chart built from the stage-one
adapters does not read the stage-two ones, or a second seed's, in coordinates. Its
<em>arrangement</em> is not local &mdash; it reappears under a second seed, in the
stage-two adapters, and when the constitutions are used as prompts rather than as
training data, so it is not an artefact of the optimiser or of the initialisation. Sketching is a random projection: it has no meaningful basis of
its own, and the individual coordinates mean nothing. What it does preserve is
the pairwise geometry, and that was checked directly against exact Frobenius
inner products &mdash; sketch cosines track exact cosines at
<span class="mono">r = 0.9996</span>, which is why angles measured in the sketch
can be read as angles between weight updates. The one-step
gradient identity is exact at initialisation and a first-order approximation
thereafter &mdash; it ignores curvature, ordering, and the fact that AdamW's first
step is preconditioned rather than raw. And the judge is a language model reading
transcripts, which is a good instrument for register and a poor one for anything
that requires knowing what the model would actually do. Generation is capped at 512
tokens, so about 59% of responses end mid-sentence: nothing here claims anything
about how a response concludes.</p>
<p>The cross term in PEFT's linear adapter merge &mdash; which splits each weight
across factors and leaves a
<span class="mono">&#8730;(w<sub>1</sub>w<sub>2</sub>)(B<sub>1</sub>A<sub>2</sub> +
B<sub>2</sub>A<sub>1</sub>)</span> residue &mdash; was identified and published by
<a href="https://arxiv.org/abs/2607.07916">Persona Cartography</a>, whose framing of
personality as a mapped space rather than a set of points shaped a good deal of what
is above. Open Character Training is
<a href="https://arxiv.org/abs/2511.01689">Maiya et al.</a>; the adjective list
descends from Allport and Odbert by way of Goldberg.</p>
<p>The 134 trait adapters (both stages and the merged personas) and the stage-two
introspection transcripts are on <a href="https://huggingface.co/EternalRecursion">Hugging
Face</a>; the preference pairs they were trained on are not yet published. The four adapters
trained on optimiser output in the last section are not: they exist to answer one
question and are too short a run to be useful to anyone else.</p>
</section>
<footer style="margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:14.5px;color:var(--dim)">
Qwen3.5-4B, LoRA rank 64, 248 targeted modules, one shared random
<i>A</i>. Generations are greedy with thinking disabled, 512 tokens, judged blind.
</footer></div>""")
    return "".join(P)





def layer_curve_svg(curve):
    """Three Gram correlations by layer: adapter~weight, prompt~weight, adapter~prompt."""
    w, h, L_, T_, B_, R_ = 640, 250, 46, 26, 34, 12
    pw = w - L_ - R_
    n = len(curve)
    px = lambda i: L_ + i / (n - 1) * pw
    py = lambda v: T_ + (1 - v) * (h - T_ - B_)
    o = []
    for gv in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{L_}" x2="{L_+pw}" y1="{py(gv):.1f}" y2="{py(gv):.1f}" stroke="var(--rule)"/>')
        o.append(f'<text x="{L_-8}" y="{py(gv)+4:.1f}" text-anchor="end" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="9.5">{gv:.2f}</text>')
    o.append(f'<line x1="{px(16):.1f}" x2="{px(16):.1f}" y1="{T_}" y2="{h-B_}" stroke="var(--in)" '
             f'stroke-width="1" stroke-dasharray="3 3"/>')
    o.append(f'<text x="{px(16)+4:.1f}" y="{T_+11}" fill="var(--in)" font-family="IBM Plex Mono, monospace" '
             f'font-size="9.5" font-weight="600">primary layer</text>')
    series = [("adapter activations ~ adapter weights", "r_AW", "var(--ink)", 8, -8),
              ("adapter activations ~ prompt activations", "r_AP", "var(--in)", 27, 14),
              ("prompt activations ~ adapter weights", "r_PW", "var(--co)", 3, 16)]
    for name, key, colr, j, dy in series:
        vals = [c[key] for c in curve]
        d = " ".join(("M" if i == 0 else "L") + f"{px(i):.1f} {py(v):.1f}" for i, v in enumerate(vals))
        o.append(f'<path d="{d}" fill="none" stroke="{colr}" stroke-width="2"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="2.2" fill="{colr}"/>')
        o.append(f'<text x="{px(j):.1f}" y="{py(vals[j])+dy:.1f}" fill="{colr}" '
                 f'font-family="IBM Plex Mono, monospace" font-size="10">{name}</text>')
    for i in range(0, n, 4):
        o.append(f'<text x="{px(i):.1f}" y="{h-14}" text-anchor="middle" fill="var(--faint)" '
                 f'font-family="IBM Plex Mono, monospace" font-size="9.5">{i}</text>')
    o.append(f'<text x="{L_+pw/2:.0f}" y="{h-2}" text-anchor="middle" fill="var(--faint)" '
             f'font-family="IBM Plex Mono, monospace" font-size="9.5">residual-stream layer</text>')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="correlation between activation-space '
            f'and weight-space trait geometries, by layer">{"".join(o)}</svg>')




def stage2_exploration_html():
    """Two sentences on what stage two installs, from the 2026-09-08 exploration JSONs."""
    try:
        S = json.load(open(f"{Q}/analysis/stage2_structure.json"))["shared_component"]["stage2"]
        R = json.load(open(f"{Q}/analysis/stage2_register_vs_residual.json"))
        N = json.load(open(f"{Q}/analysis/stage2_neutral_control.json"))
    except Exception:
        return ""
    def dig(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return default
    share = S["mean_direction_norm2_over_mean_norm2"]; cosm = S["cos_to_mean_direction_mean"]
    txt = (f"What stage two adds is mostly one thing: {share*100:.0f}% of every stage-two adapter's "
           f"squared norm lies along a single shared direction (cosine {cosm:.2f} to it for all 134), "
           "which when steered turns advice to the user into first-person speech as the character. ")
    nv = N.get("neutral_vs_grand_mean", {})
    neutral_cos = dig(nv, "cos_mean", "mean", "neutral_mean", "mean_cos")
    if neutral_cos is not None:
        txt += (f"Five stage-two runs with a trait-free constitution sit at cosine {neutral_cos:.2f} to that direction, "
                "so most of it is the recipe rather than the persona. ")
    try:
        S_ = R["summary_own_factor_amplification"]
        vals = {k: (v if isinstance(v, (int, float)) else v.get("mean_amplification", v.get("mean"))) for k, v in S_.items()} if isinstance(S_, dict) else {}
    except Exception:
        vals = {}
    b, c, d_, e_ = (vals.get("cond_b"), vals.get("cond_c"), vals.get("cond_d"), vals.get("cond_e"))
    if None not in (b, c, d_, e_):
        txt += (f"Yet the persona's extra behavioural amplitude over stage one comes from the faint trait-specific "
                f"remainder, not from that direction. On 15 traits, a stage-one adapter amplifies its own factor by "
                f"{b:+.1f} on the dial scale; adding the shared direction at half a persona's dose gives {c:+.1f} and at "
                f"1.3 times the dose {d_:+.1f}, while the exact persona gives {e_:+.1f}. The persona's dose itself was "
                "not run; it is bracketed by those two (details on the companion site and the wiki). ")
    else:
        txt += ("Yet the persona's extra behavioural amplitude over stage one comes from the faint trait-specific "
                "remainder, not from that direction (details on the companion site and the wiki). ")
    return txt

def fulloct_html():
    """Paragraph on the full OCT persona adapters, from analysis/fulloct_geometry.json
    (analyse_fulloct.py).  Empty string if the analysis has not been run."""
    p = f"{Q}/analysis/fulloct_geometry.json"
    if not os.path.exists(p):
        return ""
    d = json.load(open(p))
    g = d["gram_correlation_offdiag"]
    ni = d["norm_identity"]
    pc = d["pca"]["k5"]
    ss = d.get("second_seed", {}).get("persona_seed0_x_seed1")
    sx = d.get("stage1_x_persona")
    seed = ""
    if ss:
        seed = (f" Put through a second initialisation, fifteen personas each find their own "
                f"seed-zero persona as nearest neighbour among 134 ({ss['top1']} of {ss['n_b']}).")
    cross = ""
    if sx:
        cross = (f" Trait for trait, a stage-one adapter and its own persona sit at cosine "
                 f"<span class=\"mono\">{sx['same_trait_cos_mean']:.2f}</span>, which is what the norms "
                 f"predict for two equal orthogonal halves ({sx['predicted_from_norms_if_orthogonal_mean']:.2f}).")
    return (
        "<p>The deployed Open Character Training artefact is neither of these on its own: it is "
        "the <em>persona</em> adapter, stage one plus a quarter of stage two, and on the exact "
        "merge <span class=\"mono\">"
        f"{100 * ni['frac_norm2_from_stage2_mean']:.0f}%</span> of every persona's squared norm "
        "is the stage-two term. Its geometry is nonetheless the stage-one geometry. Over all 134 "
        "traits the persona cosine matrix correlates with the stage-one one at "
        f"<span class=\"mono\">r = {g['persona_vs_stage1']:.2f}</span> (with stage two at "
        f"<span class=\"mono\">{g['persona_vs_stage2']:.2f}</span>), the five-component "
        f"subspaces agree at Procrustes <span class=\"mono\">R&sup2; = "
        f"{pc['procrustes_r2_persona_from_stage1']:.3f}</span>, and "
        f"{d['nearest_neighbour_agreement']['persona_vs_stage1']} of 134 nearest neighbours are "
        f"unchanged.{seed}{cross} The reason is dispersion, not weight: with equal norms the "
        "persona cosine is the mean of the two stage cosines, and stage-two cosines sit on a "
        "shared component with a quarter of stage one's pair-to-pair spread "
        f"(<span class=\"mono\">{g['std_offdiag_cosine']['stage2']:.3f}</span> against "
        f"<span class=\"mono\">{g['std_offdiag_cosine']['stage1']:.3f}</span>), so the arrangement "
        "is carried by stage one. " + stage2_exploration_html() + "The arrangement-level results below, the components, the "
        "factors and the map, therefore describe the released persona adapters as well as the "
        "stage-one adapters they are computed on; the seed floor, the null comparisons and every "
        "behavioural test were run on stage one only.</p>"
    )


def fa_pro():
    """Factor-chart Procrustes between weight and activation space, from analyse_actspace_fa.py."""
    p = f"{Q}/analysis/actspace_geometry_fa.json"
    if not os.path.exists(p):
        return ""
    w = json.load(open(p))["windows"]["resp"]
    r = w["per_factor_r"]; names = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
    top = max(range(5), key=lambda i: r[i]); rest = sorted(r[i] for i in range(5) if i != top)
    return (f"<b>{w['procrustes_r2_fa']*100:.0f}%</b> of the variance against a shuffle null of "
            f"{w['procrustes_null_95pct']*100:.0f}% (the principal-component scores manage "
            f"{w['procrustes_r2_pc']*100:.0f}%), and each weight-space factor maps onto the same "
            f"activation-space factor without any rotation: {names[top]} r&nbsp;=&nbsp;{r[top]:.2f}, "
            f"the other four {rest[0]:.2f} to {rest[-1]:.2f}. ")

def actspace_section():
    pa = f"{Q}/analysis/actspace_geometry.json"
    pb = f"{Q}/analysis/actspace_adapters_geometry.json"
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return ""
    G1 = json.load(open(pa))["windows"]["resp"]
    G2 = json.load(open(pb))["windows"]["resp"]
    pr = G1["primary"]; L16 = G2["curve"][16]
    best = max(G1["curve"], key=lambda c: c["r_centred"])
    va = " / ".join(f"{v*100:.0f}" for v in pr["var_act"][:5])
    vw = " / ".join(f"{v*100:.0f}" for v in pr["var_w"][:5])
    cont = G2["containment"]
    out = [f"""<div class="w">
<hr class="sep">

<section>
<h2>Prompting versus training</h2>
<p>Everything so far lives in weight space. A different and much cheaper object is
available for every trait: hand the base model the trait's constitution &mdash; the
same document that conditioned all of its training data &mdash; as a system prompt,
and record the mean residual-stream activation over its answers to a fixed set of 64
questions, minus the same mean with no system prompt. That is a persona vector, one
per trait per layer, and it costs a forward pass rather than a training run. The
question is whether the 134 of them are arranged the way the 134 adapters are.</p>
<p>They largely are. Correlating the two 134&times;134 cosine matrices, trait-centred
so a shared component cannot carry the number, gives <b>r&nbsp;=&nbsp;{pr['r_centred']:.2f}</b>
at layer 16, the layer fixed in advance as primary (label-shuffle p&nbsp;=&nbsp;{pr['perm_p']:.4f});
the curve climbs from about 0.5 at the embedding layer and settles near 0.75 from layer
19 on (maximum {best['r_centred']:.2f} at layer {best['layer']}, reported as a maximum).
A trait's nearest neighbour in one space is its nearest neighbour in the other for
<b>{pr['nn']} of 134</b>, against one by chance. Where the traits land in the factor chart
corresponds too: a Procrustes fit of the 134&times;5 factor coordinates explains
{fa_pro()}The spectrum
is flatter in activations &mdash; the first five components carry {va}% of the variance
against {vw}% in weights &mdash; but the same pre-registered factor tests pass on the
activation Gram with the same machinery: signed factor separation +0.14 residual
(weights +0.12), bipolarity gap +0.34 (weights +0.24). That last pair should not
impress anyone; the constitutions are text <em>about</em> the traits. The
correspondence of the geometries is the result.</p>
<p>The second question is what training did. Run the base model with each adapter and
<em>no</em> system prompt, over the same 64 questions, and the adapter-induced shift can
be compared directly with the prompt-induced one. In magnitude they are the same:
the median ratio is <b>{L16['mag_ratio']:.2f}</b>. Training on a constitution's data
moves the residual stream as far as the constitution does as a prompt. In direction the
adapter delivers a median <b>{L16['frac_internalised']*100:.0f}%</b> of the prompt's
shift along the prompt's own direction, and the same-trait cosine between the two
shifts is {L16['cos_own']:+.2f} against {L16['cos_other']:+.2f} across traits &mdash;
that cross-trait floor is high because every adapter and every prompt moves activations
partly along one common direction, the activation-space shadow of the personality axis.
The adapters' arrangement in activation space tracks their arrangement in weight space
at r&nbsp;=&nbsp;{L16['r_AW']:.2f}, more closely than it tracks the prompts
({L16['r_AP']:.2f}) or than the prompts track the weights ({L16['r_PW']:.2f}); and
{cont['5']*100:.0f}% of the adapters' activation shift lies inside the five-dimensional
subspace the prompt shifts span, {cont['40']*100:.0f}% inside the top forty, where random
subspaces of the same size capture about 1%.</p>
</section></div>
<div class="wide"><figure>{layer_curve_svg(G2['curve'])}
<figcaption><b>Three geometries, by layer.</b> Each point is the correlation between two
134&times;134 trait-cosine matrices, trait-centred, at one residual-stream layer:
the adapters' effect on activations against the adapters' weights; the adapters' effect
against the constitutions' effect; and the constitutions' effect against the weights.
Layer 16 was named primary before any of this was computed. Response-token window,
greedy decoding, 64 shared prompts, one no-system baseline.</figcaption>
</figure></div>"""]
    pc = f"{Q}/analysis/actspace_cross_geometry.json"
    if os.path.exists(pc):
        J = json.load(open(pc))
        X = J["resp_specific"]              # shared components removed; see analyse_actspace_cross.py
        alone = J["resp_specific_adapter_alone"]
        med = lambda L_, k: statistics.median(r[k] for r in L_)
        m, opp, dif = X["matched"], X["same factor, OPPOSITE keying"], X["different factor"]
        pdif = J["prompt_specific"]["different factor"]
        wins = sum(1 for r in opp if r["cos_P_s"] > r["cos_A_t"])
        out.append(f"""<div class="w"><section>
<p>Then both at once. Sixteen adapters, three per factor with the keyings mixed plus
<i>bold</i>, each run under each of the sixteen constitutions as a system prompt: 256
combinations over the same 64 questions. Every prompt and every adapter shares a large
common shift &mdash; the raw cosine between any two of these persona vectors is around
+0.8 &mdash; so the questions are asked of the trait-specific part, with the mean prompt
shift, mean adapter shift and mean combined shift removed. When adapter and prompt name
the same trait, the shift along that trait's own direction comes to
<b>{med(m,'along_P_t'):.2f}</b> in units of the prompt's shift, where the prompt alone
gives 1.00, the adapter alone {alone:.2f}, and addition would give {1 + alone:.2f}: the
two mostly saturate rather than stack. Predicting the combined trait-specific shift as
adapter plus prompt leaves a residual of {med(m,'resid_add')*100:.0f}% of its norm for
matched pairs and {med(dif,'resid_add')*100:.0f}% for pairs from different factors; the
least-squares weights are {med(dif,'a'):.2f} on the adapter and {med(dif,'b'):.2f} on the
prompt. When the two <em>disagree</em> &mdash; an adapter for one pole of a factor under
the constitution for the other &mdash; neither wins. The combined model sits
{med(opp,'along_P_s'):.2f} of the way along the prompt's trait and
{med(opp,'along_P_t'):.2f} along the adapter's, and is nearer the prompt's persona than the
adapter's in {wins} of {len(opp)} cases. A system prompt does not override a trained
trait, and a trained trait does not resist a system prompt; the model ends up roughly
halfway, carrying both at about half strength. On the user-turn tokens, before the model
has said anything, the two effects simply add (residual {med(pdif,'resid_add')*100:.0f}%,
weights {med(pdif,'a'):.2f} and {med(pdif,'b'):.2f}); the saturation appears once the
model is generating.</p>
</section></div>""")
    return "".join(out)

def add_contents(body):
    """Slug every h2, then build the contents from what is actually there.

    Generated from the rendered HTML rather than a hand-kept list, so a section
    that is gated off for want of data cannot leave a dead link behind it.
    """
    import re

    seen, items = set(), []

    def slug(m):
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "section"
        sid, n = base, 2
        while sid in seen:
            sid, n = f"{base}-{n}", n + 1
        seen.add(sid)
        items.append((sid, text))
        return f'<h2 id="{sid}">{m.group(1)}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", slug, body, flags=re.S)
    if len(items) < 3:
        return body
    lis = "".join(f'<li><a href="#{i}">{e(t)}</a></li>' for i, t in items)
    nav = (f'<nav class="toc" aria-label="Contents"><p class="eyebrow">Contents</p>'
           f'<ol>{lis}</ol></nav>')
    # sits directly under the standfirst, inside the header's column
    return body.replace("</header>", "</header>\n" + nav, 1)


def main():
    H = build()
    body = "".join(H)
    label = {}
    for n, l, _ in PCS:
        label[n] = {"name": f"{n} \u2014 {l.split('.')[0]}"}
    for n, l, _ in FAS:
        label[n] = {"name": f"{l} factor"}
    for f in F5:
        label[f"axis_{f}"] = {"name": f"the named {f} axis"}
    label["mean_assistant_axis"] = {"name": "the personality axis \u2014 the direction of having one at all"}
    for n, l in (("alien_k5", "the unnamed direction"),
                 ("alien_shuffle", "control: shuffled coefficients"),
                 ("span_random", "control: a random direction")):
        if n in D["curves"]:
            label[n] = {"name": l}
    for n in label:
        q = D["qual"].get(n, {})
        label[n]["neg"] = rawclip(q.get("negative_pole", ""), 150)
        label[n]["pos"] = rawclip(q.get("positive_pole", ""), 150)
    pcnames = {f"PC{i+1}": t for i, (n, t, _) in enumerate(PCS)}
    # The map's axis list, in the order its <select> offers them and in the order
    # its score columns are concatenated: factors first, components after.
    axnames = [t for _, t in AXES]
    axdesc = dict(pcnames)
    if VF:
        for t, sname in zip(VF["factor_titles"], VF["solution_names"]):
            axdesc[t] = sname
    data = {"factors": F5, "viz": D["viz"], "curves": D["curves"], "pcnames": pcnames,
            "axnames": axnames, "axdesc": axdesc,
            "viz_fa": ({k: VF[k] for k in
                        ("coords", "loadings", "assignment", "chart_len", "factor_titles",
                         "factor_coords", "special", "ss_loadings_oblimin")} if VF else None),
            "degen": D["degen"], "alien": D["alien"], "label": label}
    sph = f"{Q}/analysis/sphere_page_fa.json"
    if not os.path.exists(sph):
        sph = f"{Q}/analysis/sphere_page.json"
    S = json.load(open(sph)) if os.path.exists(sph) else json.load(
        open(f"{Q}/analysis/sphere_layout.json"))
    if "landmark_label" not in S:
        S["landmark_label"] = {k: (SPHERE_FA_LABEL.get(k[len("factor_"):], k) if k.startswith("factor_")
                                   else k.replace("axis_", "").replace("mean_assistant_axis", "grand mean"))
                               for k in S.get("landmarks", {})}
    scripts = (f'<script>window.PD={json.dumps(data)};\n'
               f'window.PC={json.dumps(CORP)};\n'
               f'window.PS={json.dumps(S)};</script>\n'
               f'<script>{asset("_js.txt")}</script>')
    body = add_contents(body)
    body = body.encode("ascii", "xmlcharrefreplace").decode("ascii")
    out = f"{Q}/blog_page/index.html"
    open(out, "w").write(body + scripts)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
