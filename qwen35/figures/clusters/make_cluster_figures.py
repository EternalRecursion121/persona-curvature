#!/usr/bin/env python
"""
make_cluster_figures.py -- static figures showing that the 134 personality-trait
LoRA adapters group by Goldberg factor label and keying in factor space.

Run:  qwen35/.venv/bin/python qwen35/figures/clusters/make_cluster_figures.py
      (from the persona-curvature project root)

Data consumed (read-only):
  qwen35/analysis/viz_fa.json                        134 x 5 factor-chart coords,
                                                     Goldberg factor label, keying
  qwen35/results/gram_sweep.npz                      exact 134 x 134 Gram
  qwen35/results/gram_data_null_permuted_p100_matched.npz
                                                     permuted-label null arm,
                                                     100 Goldberg markers, matched
                                                     training objective

What is claimed, and what is not
--------------------------------
The colours are the *Goldberg factor label* carried on each trait word, not a
cluster found in the data.  The five recovered oblimin factors match the Big Five
at Tucker congruence 0.40 to 0.68 (E 0.539, A 0.655, C 0.574, ES 0.405, I 0.682;
wiki/pages/geometry/factor-analysis.md), so none clears the conventional 0.85
"fair" bar.  The structure the figures show is the one the decomposition reports:
signed bipolar axes, not blobs -- same-keyed traits of a factor align,
opposite-keyed anti-align (residual gap +0.2393,
wiki/pages/geometry/polarity-and-bipolarity.md).  That is why every density shell
here is drawn per (factor, keying) rather than per factor.

Colour
------
The dataviz skill's documented eight-slot categorical palette caps at three
series under the all-pairs gate (scatter / small multiples).  Five Goldberg
factors are structural, so a five-hue palette was derived in OKLCH by the skill's
own snap-to-passing procedure and held to the full all-pairs gate in BOTH modes.
Verbatim validator output is in README.md; both modes PASS all five checks.  Lexicon is the neutral "Other" slot (grey) and additionally carries its
own marker shape, so identity is never colour-alone.
"""

import base64, io, json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.manifold import MDS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))       # qwen35/
PROJ = os.path.abspath(os.path.join(ROOT, ".."))                                  # persona-curvature/
OUT = os.path.join(ROOT, "figures", "clusters")
os.makedirs(OUT, exist_ok=True)

DPI = 200            # 16 in wide * 200 dpi = 3200 px  (2x of 1600)
WIDTH_IN = 16.0

# ----------------------------------------------------------------------------
# palette  (see README.md for the validator transcript)
# ----------------------------------------------------------------------------
FACTORS = ["Agreeableness", "Conscientiousness", "EmotionalStability",
           "Extraversion", "Intellect", "Lexicon"]
FACTOR_SHORT = {"Agreeableness": "Agreeableness", "Conscientiousness": "Conscientiousness",
                "EmotionalStability": "Emotional stability", "Extraversion": "Extraversion",
                "Intellect": "Intellect", "Lexicon": "Lexicon (held out)"}

THEME = {
    "light": dict(
        surface="#fcfcfb", page="#f9f9f7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series={"Agreeableness": "#bd4269", "Conscientiousness": "#219bb6",
                "EmotionalStability": "#604eb7", "Extraversion": "#b08c1d",
                "Intellect": "#126b0e", "Lexicon": "#bebcb4"},
        div=("#0d366b", "#f0efec", "#8c2020"),
    ),
    "dark": dict(
        surface="#1a1a19", page="#0d0d0d",
        ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series={"Agreeableness": "#c64b72", "Conscientiousness": "#22a2be",
                "EmotionalStability": "#6756c1", "Extraversion": "#b28d1d",
                "Intellect": "#167811", "Lexicon": "#4e4d49"},
        div=("#86b6ef", "#383835", "#e66767"),
    ),
}

SANS = ["DejaVu Sans"]
SERIF = ["DejaVu Serif"]

SRC_LINE = "source: qwen35/analysis/viz_fa.json (factor chart, mean-centred)"
SRC_GRAM = "source: qwen35/results/gram_sweep.npz (exact Gram, double-centred)"
CONGRUENCE = ("colour = Goldberg factor label carried on the word, not a discovered cluster; "
              "the five recovered factors match the Big Five at congruence 0.40 to 0.68")

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load():
    with open(os.path.join(ROOT, "analysis", "viz_fa.json")) as fh:
        d = json.load(fh)
    traits = d["traits"]
    coords = np.asarray(d["coords"], float)
    factor = list(d["factor"])
    keyed = list(d["keyed"])
    titles = list(d["factor_titles"])
    titles[titles.index("Fearful withdrawal")] = "Timidity"
    big5 = d["big_five_scale"]                       # FA_* -> Goldberg name
    order = list(d["factor_order"])

    assert coords.shape == (134, 5), coords.shape
    assert len(traits) == len(factor) == len(keyed) == 134

    g = np.load(os.path.join(ROOT, "results", "gram_sweep.npz"), allow_pickle=True)
    assert list(g["names"]) == traits, "gram_sweep names do not match viz_fa traits"
    G = np.asarray(g["G"], float)

    X = coords - coords.mean(axis=0, keepdims=True)   # the chart is uncentred; every
                                                      # adapter shares a common component
    return dict(traits=traits, X=X, factor=factor, keyed=keyed, titles=titles,
                big5=big5, order=order, G=G)


def centred_cosine(G):
    """Double-centre the Gram, then cosine. Matches the centred_k5 FA solution."""
    p = G.shape[0]
    H = np.eye(p) - np.ones((p, p)) / p
    Gc = H @ G @ H
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    R = Gc / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def new_fig(mode, figsize, three_d=False):
    t = THEME[mode]
    fig = plt.figure(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    if three_d:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(t["surface"])
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor(t["surface"])
    return fig, ax, t


def style_axes(ax, t, spines=("left", "bottom")):
    ax.tick_params(colors=t["muted"], labelsize=8, length=3, width=0.8)
    for s in ax.spines:
        if s in spines:
            ax.spines[s].set_color(t["axis"]); ax.spines[s].set_linewidth(0.8)
        else:
            ax.spines[s].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["muted"])


def zero_lines(ax, t):
    ax.axhline(0, color=t["grid"], lw=0.9, zorder=0)
    ax.axvline(0, color=t["grid"], lw=0.9, zorder=0)


def caption(fig, t, text, y=0.012, size=8.5, x=0.5, ha="center"):
    fig.text(x, y, text, ha=ha, va="bottom", color=t["ink2"], fontsize=size,
             family=SANS, wrap=True)


def title_block(fig, t, title, subtitle, serif=False, x=0.055, y=0.955, tsize=23, ssize=10.5):
    fig.text(x, y, title, ha="left", va="top", color=t["ink"], fontsize=tsize,
             family=SERIF if serif else SANS, fontweight="normal" if serif else "bold")
    fig.text(x, y - 0.042, subtitle, ha="left", va="top", color=t["ink2"],
             fontsize=ssize, family=SANS)


def hull_polys(pts):
    """Convex hull vertices for >=3 non-degenerate points, else None."""
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except Exception:
        return None
    return pts[h.vertices]


def density_shell(pts, frac=0.45, grid=160, bw=0.62, pad=0.30):
    """Contour path enclosing `frac` of a Gaussian KDE fitted to pts.

    A convex hull over a 2-of-5 projection is all outliers and no shape; a
    density shell shows where the half of a keyed group actually sits.
    Returns (XX, YY, Z, level) or None.
    """
    if len(pts) < 5:
        return None
    try:
        kde = gaussian_kde(pts.T)
        kde.set_bandwidth(kde.factor * bw)
    except Exception:
        return None
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    XX, YY = np.meshgrid(xs, ys)
    Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    flat = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(flat)
    cum /= cum[-1]
    level = flat[np.searchsorted(cum, frac)]
    return XX, YY, Z, float(level)


def draw_key_hulls(ax, XY, factor, keyed, t, alpha=0.09, lw=1.1, zorder=1, fill=True,
                   mono=False, connectors=True, frac=0.45):
    """One soft density shell per (Goldberg factor, keying). Solid outline =
    positively keyed, dashed = negatively keyed, same colour. A thin connector
    joins the two poles of a factor through the origin: that pair of shells on
    opposite sides is the bipolarity."""
    for f in FACTORS[:5]:
        col = t["ink"] if mono else t["series"][f]
        cents = {}
        for k, ls in (("+", "-"), ("-", (0, (4, 3)))):
            m = [i for i in range(len(factor)) if factor[i] == f and keyed[i] == k]
            if not m:
                continue
            cents[k] = XY[m].mean(axis=0)
            d = density_shell(XY[m], frac=frac)
            if d is None:
                v = hull_polys(XY[m])
                if v is not None:
                    ax.add_patch(MplPolygon(v, closed=True, facecolor="none",
                                            edgecolor=col, lw=lw, ls=ls, alpha=0.8,
                                            zorder=zorder + 1))
                continue
            XX, YY, Z, lev = d
            if fill and not mono and k == "+":
                ax.contourf(XX, YY, Z, levels=[lev, Z.max() * 1.01], colors=[col],
                            alpha=alpha, zorder=zorder)
            cs = ax.contour(XX, YY, Z, levels=[lev], colors=[col], linewidths=lw,
                            linestyles=[ls], zorder=zorder + 1)
            for c in cs.collections if hasattr(cs, "collections") else []:
                c.set_alpha(0.9)
        if connectors and "+" in cents and "-" in cents:
            a, b = cents["-"], cents["+"]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw * 1.15, alpha=0.55,
                    ls="-", zorder=zorder + 2, solid_capstyle="round")
            ax.plot([b[0]], [b[1]], marker="o", ms=lw * 5.0, color=col, mec=t["surface"],
                    mew=0.7, zorder=zorder + 3)
            ax.plot([a[0]], [a[1]], marker="o", ms=lw * 5.0, mfc=t["surface"], mec=col,
                    mew=lw * 1.1, zorder=zorder + 3)


def scatter_by_class(ax, XY, factor, keyed, t, s_pos=44, s_neg=44, lw=1.4, zorder=3,
                     alpha=1.0):
    """Positively keyed = filled disc; negatively keyed = hollow ring;
    Lexicon = small x (all 34 Lexicon words are positively keyed)."""
    for f in FACTORS:
        col = t["series"][f]
        if f == "Lexicon":
            m = [i for i in range(len(factor)) if factor[i] == f]
            ax.scatter(XY[m, 0], XY[m, 1], marker="x", s=s_pos * 0.8, c=col,
                       linewidths=1.3, zorder=zorder, alpha=alpha)
            continue
        mp = [i for i in range(len(factor)) if factor[i] == f and keyed[i] == "+"]
        mn = [i for i in range(len(factor)) if factor[i] == f and keyed[i] == "-"]
        ax.scatter(XY[mp, 0], XY[mp, 1], marker="o", s=s_pos, c=col,
                   edgecolors=t["surface"], linewidths=0.9, zorder=zorder + 1, alpha=alpha)
        ax.scatter(XY[mn, 0], XY[mn, 1], marker="o", s=s_neg, facecolors="none",
                   edgecolors=col, linewidths=lw, zorder=zorder, alpha=alpha)


def legend_handles(t, mono=False):
    h = []
    for f in FACTORS:
        col = t["ink"] if mono else t["series"][f]
        mk = "x" if f == "Lexicon" else "o"
        h.append(Line2D([], [], marker=mk, linestyle="none", color=col,
                        markerfacecolor=col, markeredgecolor=col, markersize=7,
                        label=FACTOR_SHORT[f]))
    h.append(Line2D([], [], marker="o", linestyle="none", color=t["ink2"],
                    markerfacecolor=t["ink2"], markeredgecolor=t["ink2"], markersize=7,
                    label="positively keyed (filled)"))
    h.append(Line2D([], [], marker="o", linestyle="none", color=t["ink2"],
                    markerfacecolor="none", markeredgecolor=t["ink2"], markersize=7,
                    markeredgewidth=1.4, label="negatively keyed (hollow)"))
    h.append(Line2D([], [], linestyle="-", color=t["ink2"], lw=1.3,
                    label="pole connector: - centroid to + centroid"))
    h.append(Line2D([], [], linestyle=(0, (4, 3)), color=t["ink2"], lw=1.3,
                    label="shell holds the densest 45% of a keyed half"))
    return h


def put_legend(ax, t, handles, loc="upper left", ncol=1, fontsize=9, **kw):
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=fontsize,
                    frameon=True, facecolor=t["surface"], edgecolor=t["axis"],
                    labelcolor=t["ink2"], handletextpad=0.6, borderpad=0.7, **kw)
    leg.get_frame().set_linewidth(0.7)
    return leg


def place_labels(fig, ax, XY, texts, colors, fontsize=6.0, pad_px=1.2, seed=None):
    """Greedy non-overlapping label placement: try eight offsets around each point,
    keep the first whose bounding box clears everything already placed."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = list(seed or [])
    order = np.argsort(-np.abs(XY[:, 0]) - np.abs(XY[:, 1]))     # outermost first
    offsets = [(6, 3), (-6, 3), (6, -6), (-6, -6), (0, 8), (0, -11), (11, 0), (-11, 0),
               (9, 7), (-9, 7), (9, -9), (-9, -9)]
    for i in order:
        best = None
        for dx, dy in offsets:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            tx = ax.annotate(texts[i], XY[i], textcoords="offset points", xytext=(dx, dy),
                             ha=ha, va=va, fontsize=fontsize, color=colors[i],
                             family=SANS, zorder=6, annotation_clip=False)
            bb = tx.get_window_extent(renderer=rend).expanded(1.0 + pad_px / 40.0,
                                                              1.0 + pad_px / 12.0)
            clash = any(bb.overlaps(b) for b in placed)
            if not clash:
                placed.append(bb)
                best = tx
                break
            tx.remove()
        if best is None:                                  # give up nicely: place anyway
            tx = ax.annotate(texts[i], XY[i], textcoords="offset points", xytext=(6, 3),
                             ha="left", va="bottom", fontsize=fontsize * 0.9,
                             color=colors[i], family=SANS, alpha=0.55, zorder=5,
                             annotation_clip=False)
            placed.append(tx.get_window_extent(renderer=rend))


def save(fig, stem, mode):
    base = os.path.join(OUT, f"{stem}_{mode}")
    fig.savefig(base + ".png", dpi=DPI, facecolor=fig.get_facecolor())
    fig.savefig(base + ".svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    return [base + ".png", base + ".svg"]


def diverging_cmap(t):
    lo, mid, hi = t["div"]
    return LinearSegmentedColormap.from_list("pc_div", [lo, mid, hi], N=256)


def axis_label(titles, i):
    """Axis name, with the Timidity pole spelled out."""
    if titles[i] == "Timidity":
        return "Timidity axis  (negative pole: timid / fearful  ->  positive pole: bold / stable)"
    return f"{titles[i]} (recovered factor {i + 1})"


def short_axis_label(titles, i):
    return "Timidity (- timid / + bold)" if titles[i] == "Timidity" else titles[i]


# ============================================================================
# 1. editorial scatter
# ============================================================================
def fig_editorial(D, mode):
    t = THEME[mode]
    XY = D["X"][:, :2]
    fig, ax, t = new_fig(mode, (WIDTH_IN, 12.6))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.085)

    zero_lines(ax, t)
    draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, alpha=0.075, lw=1.1)
    scatter_by_class(ax, XY, D["factor"], D["keyed"], t, s_pos=34, s_neg=34, lw=1.2)

    cols = [t["series"][f] if f != "Lexicon" else t["muted"] for f in D["factor"]]
    place_labels(fig, ax, XY, D["traits"], cols, fontsize=6.4)

    pad = 0.16
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
    ax.set_xlabel(axis_label(D["titles"], 0), color=t["ink2"], fontsize=10, family=SANS)
    ax.set_ylabel(axis_label(D["titles"], 1), color=t["ink2"], fontsize=10, family=SANS)
    style_axes(ax, t)
    put_legend(ax, t, legend_handles(t), loc="lower left", fontsize=8.5)

    title_block(fig, t, "Same word, opposite pole, opposite side",
                "134 trait LoRA adapters on the first two recovered factors. Each soft shell holds the densest 45 per cent "
                "of one Goldberg factor at one pole -- solid outline for positively keyed words, dashed "
                "for negatively\nkeyed -- and the connector runs from one pole's centre to the other. The "
                "factors come back as signed bipolar axes, not blobs.", serif=True,
                tsize=25)
    caption(fig, t,
            "Warmth against Competence, factor-chart coordinates mean-centred; 45% density shells "
            f"per Goldberg factor label and keying; {CONGRUENCE}. {SRC_LINE}.")
    return save(fig, "editorial", mode)


# ============================================================================
# 2. pairs matrix
# ============================================================================
def fig_pairs(D, mode):
    t = THEME[mode]
    X, titles = D["X"], D["titles"]
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    fig = plt.figure(figsize=(WIDTH_IN, 9.0), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    fig.subplots_adjust(left=0.042, right=0.988, top=0.845, bottom=0.175,
                        wspace=0.20, hspace=0.30)

    for n, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 5, n + 1)
        ax.set_facecolor(t["surface"])
        zero_lines(ax, t)
        XY = X[:, [i, j]]
        draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, alpha=0.06, lw=0.7)
        scatter_by_class(ax, XY, D["factor"], D["keyed"], t, s_pos=13, s_neg=13, lw=0.8)
        ax.set_xlabel(short_axis_label(titles, i), color=t["ink2"], fontsize=7.6, family=SANS,
                      labelpad=2)
        ax.set_ylabel(short_axis_label(titles, j), color=t["ink2"], fontsize=7.6, family=SANS,
                      labelpad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(t["axis"]); s.set_linewidth(0.6)
        lim = np.abs(XY).max() * 1.10
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")

    h = legend_handles(t)
    fig.legend(handles=h, loc="upper center", ncol=5, fontsize=9.0, frameon=False,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.135),
               columnspacing=2.0, handletextpad=0.7)
    title_block(fig, t, "All ten factor pairs",
                "Each panel is one of the ten plane projections of the five-factor chart. The keyed "
                "split shows up in every plane, not just the lucky one.\nSame axes, same colours, "
                "no labels: read where the two shells of a colour sit.", tsize=21, y=0.972)
    caption(fig, t,
            "Ten pairwise projections of the 134 x 5 factor chart, mean-centred; 45% density shells per "
            f"Goldberg factor and keying; {CONGRUENCE}. {SRC_LINE}.", y=0.012)
    return save(fig, "pairs", mode)


# ============================================================================
# 3. static 3D render
# ============================================================================
ANCHORS = ["agreeable", "rude", "organized", "careless", "anxious", "relaxed",
           "talkative", "quiet", "imaginative", "unimaginative"]


def fig_three_d(D, mode):
    t = THEME[mode]
    X = D["X"][:, :3]
    titles = D["titles"]
    fig, ax, t = new_fig(mode, (WIDTH_IN, 10.2), three_d=True)
    fig.patch.set_facecolor(t["page"])
    ax.set_position([-0.025, 0.010, 1.05, 0.885])
    ax.set_box_aspect((1.0, 1.0, 0.82), zoom=1.05)
    ax.view_init(elev=20, azim=-58)
    for pane, lab in ((ax.xaxis, 0), (ax.yaxis, 1), (ax.zaxis, 2)):
        pane.set_pane_color(to_rgb(t["surface"]) + (1.0,))
        pane._axinfo["grid"]["color"] = t["grid"]
        pane._axinfo["grid"]["linewidth"] = 0.6

    # translucent hulls per (factor, keying)
    for f in FACTORS[:5]:
        for k in ("+", "-"):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            if len(m) < 4:
                continue
            pts = X[m]
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            tri = [pts[s] for s in hull.simplices]
            pc = Poly3DCollection(tri, alpha=0.10 if k == "+" else 0.055,
                                  facecolor=t["series"][f], edgecolor=t["series"][f],
                                  linewidths=0.45)
            pc.set_zsort("average")
            ax.add_collection3d(pc)

    # depth-cued point sizes: nearer to the camera in the projected depth => larger
    cam = np.array([np.cos(np.radians(-58)) * np.cos(np.radians(20)),
                    np.sin(np.radians(-58)) * np.cos(np.radians(20)),
                    np.sin(np.radians(20))])
    depth = X @ cam
    dn = (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    sizes = 14 + 62 * dn

    for f in FACTORS:
        col = t["series"][f]
        if f == "Lexicon":
            m = [i for i in range(134) if D["factor"][i] == f]
            ax.scatter(X[m, 0], X[m, 1], X[m, 2], marker="x", s=sizes[m] * 0.8, c=col,
                       linewidths=1.1, depthshade=False)
            continue
        mp = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "+"]
        mn = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "-"]
        ax.scatter(X[mp, 0], X[mp, 1], X[mp, 2], marker="o", s=sizes[mp], c=col,
                   edgecolors=t["surface"], linewidths=0.5, depthshade=False)
        ax.scatter(X[mn, 0], X[mn, 1], X[mn, 2], marker="o", s=sizes[mn],
                   facecolors="none", edgecolors=col, linewidths=1.2, depthshade=False)

    idx = {n: i for i, n in enumerate(D["traits"])}
    nudge = {"imaginative": 0.10, "unimaginative": -0.13, "talkative": 0.10,
             "quiet": -0.12, "relaxed": -0.12, "anxious": 0.09, "organized": 0.09,
             "careless": -0.12, "agreeable": 0.09, "rude": 0.09}
    for a in ANCHORS:
        i = idx[a]
        ax.text(X[i, 0], X[i, 1], X[i, 2] + nudge.get(a, 0.08), a, fontsize=9.0,
                family=SANS, color=t["ink"], zorder=10, ha="center",
                va="bottom" if nudge.get(a, 0.08) > 0 else "top",
                bbox=dict(boxstyle="round,pad=0.16", fc=t["surface"], ec="none",
                          alpha=0.72))

    ax.set_xlabel(titles[0], color=t["ink2"], fontsize=10, family=SANS, labelpad=8)
    ax.set_ylabel(titles[1], color=t["ink2"], fontsize=10, family=SANS, labelpad=8)
    ax.set_zlabel("Timidity  (- timid / + bold)", color=t["ink2"], fontsize=10,
                  family=SANS, labelpad=8)
    ax.tick_params(colors=t["muted"], labelsize=7.5)
    h3 = legend_handles(t)[:8]
    h3.append(Line2D([], [], linestyle="-", color=t["ink2"], lw=1.3,
                     label="shell = convex hull of one factor at one pole"))
    put_legend(ax, t, h3, loc="upper left", fontsize=8.2, bbox_to_anchor=(0.035, 0.90))

    title_block(fig, t, "The top three factors, one viewpoint",
                "Warmth, Competence and Timidity. Point size is depth-cued (nearer is larger); each "
                "translucent shell is one Goldberg factor at one pole.\nTen anchor words are labelled; "
                "the antonym of each sits across the origin.", tsize=21, y=0.982, x=0.05)
    caption(fig, t,
            "Factors 1-3 of the factor chart, mean-centred, azim -58 / elev 20; convex hulls per Goldberg "
            f"factor and keying; {CONGRUENCE}. {SRC_LINE}.")
    return save(fig, "threed", mode)


# ============================================================================
# 4. dendrogram
# ============================================================================
def fig_dendrogram(D, R, mode):
    t = THEME[mode]
    dist = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")

    fig = plt.figure(figsize=(WIDTH_IN, 10.4), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.105, 0.265, 0.875, 0.545]); ax.set_facecolor(t["surface"])
    axb = fig.add_axes([0.105, 0.183, 0.875, 0.024]); axb.set_facecolor(t["surface"])
    axk = fig.add_axes([0.105, 0.153, 0.875, 0.024]); axk.set_facecolor(t["surface"])

    dn = dendrogram(Z, ax=ax, no_labels=True,
                    link_color_func=lambda _: t["ink2"],
                    above_threshold_color=t["ink2"])
    leaves = dn["leaves"]
    ymin = 0.36                              # no pair merges below 0.416
    ax.set_ylim(ymin, float(Z[:, 2].max()) * 1.03)
    ax.set_ylabel("cosine distance (1 - cos), axis truncated at 0.36", color=t["ink2"],
                  fontsize=9.5, family=SANS)
    style_axes(ax, t, spines=("left",))
    ax.set_xticks([])
    for coll in ax.collections:
        coll.set_linewidth(0.7)
    for ln in ax.get_lines():
        ln.set_linewidth(0.7)

    n = len(leaves)
    for pos, li in enumerate(leaves):
        f, k = D["factor"][li], D["keyed"][li]
        col = t["series"][f]
        lcol = col if f != "Lexicon" else t["muted"]
        x = 5 + 10 * pos
        ax.text(x, ymin - 0.012, D["traits"][li] + ("" if f == "Lexicon" else
                ("  +" if k == "+" else "  -")),
                rotation=90, ha="center", va="top", fontsize=5.2, color=lcol, family=SANS,
                clip_on=False)
        axb.add_patch(MplPolygon([[pos, 0], [pos + 1, 0], [pos + 1, 1], [pos, 1]],
                                 closed=True, facecolor=col, edgecolor="none"))
        kc = col if k == "+" else t["surface"]
        axk.add_patch(MplPolygon([[pos, 0], [pos + 1, 0], [pos + 1, 1], [pos, 1]],
                                 closed=True, facecolor=kc, edgecolor=col, linewidth=0.3))
    for a, lab in ((axb, "Goldberg factor"), (axk, "keying  (filled +, open -)")):
        a.set_xlim(0, n); a.set_ylim(0, 1); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.text(-0.008, 0.5, lab, transform=a.transAxes, ha="right", va="center",
               fontsize=8.4, color=t["ink2"], family=SANS)

    h = legend_handles(t)[:8]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.6, frameon=False,
               labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.055))
    title_block(fig, t, "Average linkage on cosine distance",
                "134 adapters, hierarchical clustering with no knowledge of the labels. The two bars "
                "under the leaves colour each leaf by its Goldberg factor and its keying;\nruns of one colour "
                "are branches the clustering found on its own. The distance axis starts at 0.36 -- no pair "
                "merges below 0.416, so nothing is hidden.",
                tsize=21, y=0.975)
    caption(fig, t,
            "Average-linkage dendrogram on 1 - cosine between adapters; "
            f"{CONGRUENCE}. {SRC_GRAM}.", y=0.012)
    return save(fig, "dendrogram", mode)


# ============================================================================
# 5. sorted cosine heatmap (+ permuted null)
# ============================================================================
def sort_order(factor, keyed, traits):
    key = []
    for i in range(len(traits)):
        f = factor[i]
        key.append((FACTORS.index(f), 0 if keyed[i] == "+" else 1, traits[i]))
    return sorted(range(len(traits)), key=lambda i: key[i])


def block_edges(order, factor, keyed):
    """Return (factor boundaries, keying boundaries, factor spans)."""
    fb, kb, spans = [0], [], []
    cur_f = factor[order[0]]; cur_k = keyed[order[0]]; start = 0
    for p, i in enumerate(order[1:], start=1):
        if factor[i] != cur_f:
            fb.append(p); spans.append((start, p, cur_f)); start = p; cur_f = factor[i]
            cur_k = keyed[i]
        elif keyed[i] != cur_k:
            kb.append(p); cur_k = keyed[i]
    fb.append(len(order)); spans.append((start, len(order), cur_f))
    return fb, kb, spans


def fig_heatmap(D, R, mode):
    t = THEME[mode]
    cmap = diverging_cmap(t)

    # real arm
    o = sort_order(D["factor"], D["keyed"], D["traits"])
    M = R[np.ix_(o, o)]
    v = np.percentile(np.abs(M - np.diag(np.diag(M))), 99.0)
    v = max(v, 0.25)

    # permuted null arm, 100 Goldberg markers
    nz = np.load(os.path.join(ROOT, "results",
                              "gram_data_null_permuted_p100_matched.npz"), allow_pickle=True)
    nn = [str(x) for x in nz["names"]]
    tidx = {tr: i for i, tr in enumerate(D["traits"])}
    assert all(x in tidx for x in nn), "null names not a subset of the 134 traits"
    nfac = [D["factor"][tidx[x]] for x in nn]
    nkey = [D["keyed"][tidx[x]] for x in nn]
    Rn = centred_cosine(np.asarray(nz["G"], float))
    on = sort_order(nfac, nkey, nn)
    Mn = Rn[np.ix_(on, on)]

    fig = plt.figure(figsize=(WIDTH_IN, 9.6), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax1 = fig.add_axes([0.108, 0.125, 0.355, 0.655]); ax1.set_facecolor(t["surface"])
    ax2 = fig.add_axes([0.600, 0.125, 0.265, 0.655]); ax2.set_facecolor(t["surface"])
    cax = fig.add_axes([0.905, 0.30, 0.012, 0.26])

    for ax, M_, o_, fac_, key_, ttl, sub in (
            (ax1, M, o, D["factor"], D["keyed"], "Real zoo, 134 adapters",
             "ordered by Goldberg factor, then keying"),
            (ax2, Mn, on, nfac, nkey, "Permuted-label null, 100 markers",
             "same recipe, labels reassigned")):
        im = ax.imshow(M_, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
        fb, kb, spans = block_edges(o_, fac_, key_)
        for b in fb[1:-1]:
            ax.axhline(b - 0.5, color=t["ink"], lw=1.1, alpha=0.75)
            ax.axvline(b - 0.5, color=t["ink"], lw=1.1, alpha=0.75)
        for b in kb:
            ax.axhline(b - 0.5, color=t["ink"], lw=0.7, alpha=0.65, ls=(0, (2.5, 2)))
            ax.axvline(b - 0.5, color=t["ink"], lw=0.7, alpha=0.65, ls=(0, (2.5, 2)))
        ticks = [(a + b) / 2 - 0.5 for a, b, _ in spans]
        names = [FACTOR_SHORT[f] for _, _, f in spans]
        ax.set_xticks(ticks); ax.set_xticklabels(names, rotation=32, ha="right", fontsize=7.6)
        ax.set_yticks(ticks); ax.set_yticklabels(names, fontsize=7.6)
        ax.tick_params(colors=t["ink2"], length=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(t["ink2"]); lbl.set_family(SANS[0])
        for s in ax.spines.values():
            s.set_color(t["axis"]); s.set_linewidth(0.7)
        ax.set_title(ttl + "\n" + sub, color=t["ink"], fontsize=10.5, family=SANS, pad=9)
        ax.set_anchor("N")

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cosine, double-centred Gram", color=t["ink2"], fontsize=8.6, family=SANS)
    cb.ax.tick_params(colors=t["muted"], labelsize=7.5)
    cb.outline.set_edgecolor(t["axis"]); cb.outline.set_linewidth(0.6)

    title_block(fig, t, "Positive on the diagonal blocks, negative across the poles",
                "The 134 x 134 cosine matrix sorted by Goldberg factor then keying. Heavy rules are "
                "factor boundaries, hairlines split the two poles inside a factor.\nSame-keyed blocks "
                "run warm, opposite-keyed blocks run cool: that sign flip is the bipolarity.",
                tsize=21, y=0.975)
    caption(fig, t,
            "Cosine from the double-centred exact Gram, diverging scale centred at zero. Null arm: "
            "permuted trait labels, retrained at the matched objective, 100 Goldberg markers only "
            f"(the 34 Lexicon words were never trained in the null).\n{CONGRUENCE}. {SRC_GRAM}; "
            "qwen35/results/gram_data_null_permuted_p100_matched.npz.", y=0.012)
    return save(fig, "heatmap", mode)


# ============================================================================
# 6. bipolar axis strips
# ============================================================================
def fig_strips(D, mode):
    t = THEME[mode]
    X, titles, order, big5 = D["X"], D["titles"], D["order"], D["big5"]
    g2f = {big5[k]: i for i, k in enumerate(order)}          # Goldberg -> chart axis index

    fig = plt.figure(figsize=(WIDTH_IN, 15.4), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    top, bot, gap = 0.875, 0.062, 0.021
    hgt = (top - bot - 4 * gap) / 5

    for n, f in enumerate(FACTORS[:5]):
        ax = fig.add_axes([0.045, top - (n + 1) * hgt - n * gap, 0.945, hgt])
        ax.set_facecolor(t["surface"])
        ai = g2f[f]
        col = t["series"][f]
        m = [i for i in range(134) if D["factor"][i] == f]
        lim = np.abs(X[:, ai]).max() * 1.10
        ax.set_xlim(-lim, lim); ax.set_ylim(-3.75, 3.75)
        ax.axhline(0, color=t["axis"], lw=1.0, zorder=1)
        ax.axvline(0, color=t["grid"], lw=0.9, ls=(0, (4, 3)), zorder=0)

        counts = {"+": 0, "-": 0}
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        placed = []
        for k in ("+", "-"):
            side = sorted([i for i in m if D["keyed"][i] == k], key=lambda i: X[i, ai])
            counts[k] = len(side)
            sgn = 1.0 if k == "+" else -1.0
            for i in side:
                x = X[i, ai]
                # walk up lanes until the rendered label clears everything placed
                tx = None
                for lane in range(12):
                    y = sgn * (0.30 + 0.225 * lane)
                    if k == "+":
                        tx = ax.text(x, y + 0.09, D["traits"][i], rotation=58, ha="left",
                                     va="bottom", fontsize=6.8, color=t["ink"], family=SANS)
                    else:
                        tx = ax.text(x, y - 0.09, D["traits"][i], rotation=58, ha="right",
                                     va="top", fontsize=6.8, color=t["ink2"], family=SANS)
                    bb = tx.get_window_extent(renderer=rend).expanded(1.01, 1.12)
                    if not any(bb.overlaps(o) for o in placed):
                        placed.append(bb)
                        break
                    tx.remove(); tx = None
                if tx is None:            # crowded: set it horizontally beside the stem
                    lane = 0
                    y = sgn * 0.30
                    ax.text(x, y + sgn * 0.13, " " + D["traits"][i], rotation=0,
                            ha="left", va="bottom" if k == "+" else "top",
                            fontsize=6.4, color=t["muted"], family=SANS, alpha=0.95)
                ax.plot([x, x], [0, y], color=col, lw=1.0, alpha=0.55, zorder=2)
                if k == "+":
                    ax.plot([x], [y], marker="o", ms=6.0, color=col, mec=t["surface"],
                            mew=0.8, zorder=3)
                else:
                    ax.plot([x], [y], marker="o", ms=6.0, mfc="none", mec=col, mew=1.4,
                            zorder=3)

        ax.set_xlim(-lim, lim); ax.set_ylim(-3.75, 3.75)
        ax.set_yticks([])
        ax.tick_params(colors=t["muted"], labelsize=7.5, length=3)
        for lbl in ax.get_xticklabels():
            lbl.set_color(t["muted"])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"]); ax.spines["bottom"].set_linewidth(0.7)

        name = short_axis_label(titles, ai)
        ax.text(-lim * 0.998, 3.66, f"{FACTOR_SHORT[f]}   ->   recovered factor: {name}",
                ha="left", va="top", fontsize=10.4, color=t["ink"], family=SANS,
                fontweight="bold")
        if titles[ai] == "Timidity":
            ax.text(lim * 0.998, 3.66,
                    "timid / fearful pole to the left, bold / stable pole to the right",
                    ha="right", va="top", fontsize=7.6, color=t["muted"], family=SANS)
        ax.text(-lim * 0.998, -3.66,
                f"{counts['-']} negatively keyed markers below the line",
                ha="left", va="bottom", fontsize=7.6, color=t["muted"], family=SANS)
        ax.text(lim * 0.998, -3.66, f"{counts['+']} positively keyed above",
                ha="right", va="bottom", fontsize=7.6, color=t["muted"], family=SANS)

    title_block(fig, t, "Opposite words point opposite ways",
                "Each strip is one Big Five scale's 20 markers, placed at their coordinate on the "
                "recovered factor matched to that scale. Positively keyed above the line,\nnegatively "
                "keyed below. A bipolar axis means the two rows sit on opposite sides of zero.",
                tsize=21, y=0.975)
    caption(fig, t,
            "Marker coordinate on its own recovered factor, factor chart mean-centred. Emotional "
            "stability has 6 positively and 14 negatively keyed markers in Goldberg's published set. "
            f"{CONGRUENCE}. {SRC_LINE}.", y=0.012)
    return save(fig, "strips", mode)


# ============================================================================
# 7. MDS constellation
# ============================================================================
def fig_mds(D, R, mode):
    t = THEME[mode]
    dist = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=17,
              n_init=8, max_iter=600, normalized_stress="auto").fit_transform(dist)

    K = 5
    nbr = np.argsort(dist + np.eye(134) * 9, axis=1)[:, :K]
    segs, scols, same = [], [], 0
    for i in range(134):
        for j in nbr[i]:
            segs.append([emb[i], emb[j]])
            sm = D["factor"][i] == D["factor"][j]
            same += sm
            scols.append(t["series"][D["factor"][i]] if sm else t["muted"])
    frac = same / len(segs)

    fig, ax, t = new_fig(mode, (WIDTH_IN, 11.8))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.855, bottom=0.085)
    ax.add_collection(LineCollection(segs, colors=scols, linewidths=0.55, alpha=0.32,
                                     zorder=1))
    scatter_by_class(ax, emb, D["factor"], D["keyed"], t, s_pos=40, s_neg=40, lw=1.3)
    cols = [t["series"][f] if f != "Lexicon" else t["muted"] for f in D["factor"]]
    place_labels(fig, ax, emb, D["traits"], cols, fontsize=6.2)

    pad = np.abs(emb).max() * 0.10
    ax.set_xlim(emb[:, 0].min() - pad, emb[:, 0].max() + pad)
    ax.set_ylim(emb[:, 1].min() - pad, emb[:, 1].max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    put_legend(ax, t, legend_handles(t)[:8], loc="lower left", fontsize=8.5)

    title_block(fig, t, "The zoo as a constellation",
                "Metric MDS on cosine distance between adapters, with each adapter joined to its five "
                "nearest neighbours. Edges are coloured when both ends carry the\nsame Goldberg factor "
                f"label and grey when they do not: {frac:.0%} of the {len(segs)} edges are same-factor.",
                tsize=21, y=0.975)
    caption(fig, t,
            "2D metric MDS from 1 - cosine on the double-centred exact Gram; k = 5 nearest-neighbour "
            f"edges; MDS axes have no units and no orientation. {CONGRUENCE}. {SRC_GRAM}.")
    return save(fig, "mds", mode)


# ============================================================================
# 8. minimal monochrome
# ============================================================================
def fig_mono(D, mode="light"):
    t = dict(THEME["light"])
    t["surface"] = "#ffffff"; t["page"] = "#ffffff"
    t["ink"] = "#000000"; t["ink2"] = "#1a1a1a"; t["muted"] = "#444444"
    t["grid"] = "#c8c8c8"; t["axis"] = "#000000"
    t["series"] = {f: "#000000" for f in FACTORS}

    XY = D["X"][:, :2]
    fig = plt.figure(figsize=(WIDTH_IN, 12.6), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_subplot(111); ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.845, bottom=0.085)

    ax.axhline(0, color=t["grid"], lw=0.8, zorder=0)
    ax.axvline(0, color=t["grid"], lw=0.8, zorder=0)
    pad = 0.16
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)

    draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, lw=1.0, mono=True, fill=False,
                   connectors=True)

    # the named shell labels win the space; trait labels are placed around them
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    seed = []
    for f in FACTORS[:5]:
        for k in ("+", "-"):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            if not m:
                continue
            c = XY[m].mean(axis=0)
            tx = ax.text(c[0], c[1], f"{FACTOR_SHORT[f]}\n{'positively' if k == '+' else 'negatively'} keyed",
                         ha="center", va="center", fontsize=8.4, family=SERIF,
                         color="#000000", zorder=6,
                         bbox=dict(boxstyle="round,pad=0.32", fc="#ffffff", ec="#000000",
                                   lw=0.7))
            seed.append(tx.get_window_extent(renderer=rend).expanded(1.03, 1.10))

    place_labels(fig, ax, XY, D["traits"], ["#000000"] * 134, fontsize=6.6, seed=seed)

    ax.set_xlabel(axis_label(D["titles"], 0), color="#000000", fontsize=10, family=SANS)
    ax.set_ylabel(axis_label(D["titles"], 1), color="#000000", fontsize=10, family=SANS)
    style_axes(ax, t)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color("#333333")

    fig.text(0.055, 0.955, "Warmth and Competence, black and white", ha="left", va="top",
             color="#000000", fontsize=25, family=SERIF)
    fig.text(0.055, 0.913,
             "The editorial scatter with the colour channel removed: only the words and the shells. "
             "A solid outline is the positively keyed half of a Goldberg scale,\ndashed the negatively "
             "keyed half, and the bar between them joins the two poles. For print and for "
             "forced-colours.",
             ha="left", va="top", color="#000000", fontsize=10.5, family=SANS)
    fig.text(0.5, 0.030,
             "Warmth against Competence, factor-chart coordinates mean-centred; 45% density shells "
             "per Goldberg factor label and keying.",
             ha="center", va="bottom", color="#000000", fontsize=8.5, family=SANS)
    fig.text(0.5, 0.012,
             f"{CONGRUENCE}. {SRC_LINE}.",
             ha="center", va="bottom", color="#000000", fontsize=8.5, family=SANS)
    return save(fig, "mono", mode)


# ============================================================================
# v2 -- the readability pass
#
# Samuel's note on the first gallery was "the diagrams are too dense and text is
# too small", followed by "keep everything from before".  So v2 keeps every trait,
# every pair, every leaf and the whole matrix, and buys legibility with canvas and
# with faceting rather than by dropping content.
#
#   display width 2400 px (PNG rendered at 2x, 4800 px), height as needed
#   every label >= 13 px at the display width, titles >= 34 px, captions >= 15 px
#   leader lines and repulsion instead of overlapping type
#
# A second, sparser set (<style>_v2anchors_*) labels only the 24 anchor words, for
# when the point is the shape rather than the vocabulary.
# ============================================================================

WIDTH_V2 = 24.0                     # inches; x DPI 200 = 4800 px = 2x of 2400
PT2PX = (DPI / 72.0) / 2.0          # points -> px at the 2400 px display width

T2 = dict(                          # every size is a point value; the comment is px
    title=26,       # 36 px
    subtitle=13,    # 18 px
    caption=11,     # 15 px
    axis=13,        # 18 px
    tick=10,        # 14 px
    legend=11.5,    # 16 px
    label=9.6,      # 13.3 px  -- the floor
    label_big=11.5, # 16 px
    panel=14,       # 19 px
)

# 24 anchors: the strongest positive and negative loader on each of the five
# factors, plus the well-known words Samuel named.  ("curious" is not in the zoo --
# the 134 are Goldberg's 100 markers plus 34 held-out lexicon words, and curious is
# in neither, so it cannot be labelled.)
ANCHOR_NAMED = ["agreeable", "rude", "organized", "careless", "anxious", "relaxed",
                "talkative", "quiet", "imaginative", "unimaginative", "warm", "cold",
                "shallow"]
ANCHOR_EXTRA = ["fearful", "unkind", "extraverted", "pleasant", "impractical"]


def anchor_set(D):
    X, tr = D["X"], D["traits"]
    out = set(ANCHOR_NAMED) | set(ANCHOR_EXTRA)
    for ax in range(5):
        out.add(tr[int(np.argmax(X[:, ax]))])
        out.add(tr[int(np.argmin(X[:, ax]))])
    return [t for t in tr if t in out]


def new_fig2(mode, height, three_d=False):
    t = THEME[mode]
    fig = plt.figure(figsize=(WIDTH_V2, height), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    if three_d:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    ax.set_facecolor(t["surface"])
    return fig, ax, t


def title2(fig, t, title, subtitle, serif=False, x=0.038, y=None, top_in=None):
    """Title block sized in inches from the top, so it lands in the same place on a
    15 in figure and on a 40 in one."""
    h = fig.get_figheight()
    y = 1.0 - (0.62 / h) if y is None else y
    fig.text(x, y, title, ha="left", va="top", color=t["ink"], fontsize=T2["title"],
             family=SERIF if serif else SANS,
             fontweight="normal" if serif else "bold")
    fig.text(x, y - (0.52 / h), subtitle, ha="left", va="top", color=t["ink2"],
             fontsize=T2["subtitle"], family=SANS, linespacing=1.45)


def caption2(fig, t, text):
    import textwrap
    chars = int((fig.get_figwidth() * 72.0 * 0.90) / (T2["caption"] * 0.56))
    wrapped = "\n".join(textwrap.wrap(text, chars))
    fig.text(0.5, 0.30 / fig.get_figheight(), wrapped, ha="center", va="bottom",
             color=t["ink2"], fontsize=T2["caption"], family=SANS, linespacing=1.5)


def style_axes2(ax, t, spines=("left", "bottom")):
    ax.tick_params(colors=t["muted"], labelsize=T2["tick"], length=4, width=0.7)
    for sp in ax.spines:
        if sp in spines:
            ax.spines[sp].set_color(t["axis"]); ax.spines[sp].set_linewidth(0.7)
        else:
            ax.spines[sp].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["muted"])


def scatter2(ax, XY, factor, keyed, t, s=120, lw=1.6, alpha=1.0, zorder=3, only=None,
             grey_rest=False):
    """Bigger markers with a thin surface-coloured outline. `only` restricts the
    coloured marks to one Goldberg factor; the rest go grey and recessive."""
    idx = range(len(factor))
    if grey_rest and only is not None:
        rest = [i for i in idx if factor[i] != only]
        ax.scatter(XY[rest, 0], XY[rest, 1], marker="o", s=s * 0.42,
                   c=t["series"]["Lexicon"], edgecolors="none", zorder=zorder - 1,
                   alpha=0.85)
        idx = [i for i in idx if factor[i] == only]
    for f in FACTORS:
        col = t["series"][f]
        sel = [i for i in idx if factor[i] == f]
        if not sel:
            continue
        if f == "Lexicon":
            ax.scatter(XY[sel, 0], XY[sel, 1], marker="x", s=s * 0.85, c=col,
                       linewidths=1.7, zorder=zorder, alpha=alpha)
            continue
        mp = [i for i in sel if keyed[i] == "+"]
        mn = [i for i in sel if keyed[i] == "-"]
        ax.scatter(XY[mp, 0], XY[mp, 1], marker="o", s=s, c=col,
                   edgecolors=t["surface"], linewidths=1.1, zorder=zorder + 1,
                   alpha=alpha)
        ax.scatter(XY[mn, 0], XY[mn, 1], marker="o", s=s, facecolors="none",
                   edgecolors=col, linewidths=lw, zorder=zorder, alpha=alpha)


def leader_labels(fig, ax, XY, which, texts, colors, fontsize, avoid_points=True,
                  rings=(9, 15, 22, 31, 42, 55, 70, 87, 106, 127), nang=20,
                  leader_from=15, t=None, all_xy=None, contain=True, margin=3.0):
    """Place a label for each index in `which` without overlaps, drawing a hairline
    leader whenever the label had to travel.

    Each label's box is measured ONCE and then translated analytically for every
    candidate offset -- creating and destroying an annotation per candidate is
    about 200x more text layout than this needs, and on a 134-label panel that is
    the difference between seconds and minutes.

    Candidates walk outwards through rings of offsets. The first one that is inside
    the axes and clear of everything already placed wins; if none is, the inside
    candidate with the least overlap area wins and is drawn slightly dimmed.
    """
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpr = fig.dpi / 72.0

    placed = []
    if avoid_points:
        pts = all_xy if all_xy is not None else XY
        disp = ax.transData.transform(pts)
        r = 4.5 * (DPI / 100.0)
        placed += [Bbox([[x - r, y - r], [x + r, y + r]]) for x, y in disp]

    cand = [(0.0, 0.0)]
    for n_, rad in enumerate(rings):
        for k in range(nang):
            ang = 2 * np.pi * k / nang + 0.13 * n_
            cand.append((rad * np.cos(ang), rad * np.sin(ang)))

    axbb = ax.get_window_extent()

    def is_inside(x0, y0, w, h):
        return (not contain) or (x0 >= axbb.x0 + margin and x0 + w <= axbb.x1 - margin
                                 and y0 >= axbb.y0 + margin and y0 + h <= axbb.y1 - margin)

    def overlap(x0, y0, w, h):
        sc = 0.0
        x1, y1 = x0 + w, y0 + h
        for o in placed:
            ow = min(x1, o.x1) - max(x0, o.x0)
            oh = min(y1, o.y1) - max(y0, o.y0)
            if ow > 0 and oh > 0:
                sc += ow * oh
        return sc

    disp_all = ax.transData.transform(XY)
    ordering = sorted(which, key=lambda i: -(XY[i, 0] ** 2 + XY[i, 1] ** 2))
    for i in ordering:
        mk = getattr(ax, "text2D", ax.text)        # Axes3D.text needs a z
        probe = mk(0.5, 0.5, texts[i], transform=ax.transAxes, fontsize=fontsize,
                   family=SANS, ha="left", va="bottom")
        bb0 = probe.get_window_extent(renderer=rend)
        w, h = bb0.width * 1.04 + 2.0, bb0.height * 1.22 + 2.0
        probe.remove()

        px, py = disp_all[i]
        best_free, best_any = None, None
        for dx, dy in cand:
            ha = "left" if dx > 1 else ("right" if dx < -1 else "center")
            va = "bottom" if dy > 1 else ("top" if dy < -1 else "center")
            ax0 = px + dx * dpr
            ay0 = py + dy * dpr
            x0 = ax0 if ha == "left" else (ax0 - w if ha == "right" else ax0 - w / 2)
            y0 = ay0 if va == "bottom" else (ay0 - h if va == "top" else ay0 - h / 2)
            if not is_inside(x0, y0, w, h):
                continue
            sc = overlap(x0, y0, w, h)
            if sc <= 0.0:
                best_free = (dx, dy, ha, va, x0, y0); break
            if best_any is None or sc < best_any[0]:
                best_any = (sc, dx, dy, ha, va, x0, y0)
        if best_free is not None:
            dx, dy, ha, va, x0, y0 = best_free
            alpha = 1.0
        elif best_any is not None:
            _, dx, dy, ha, va, x0, y0 = best_any
            alpha = 0.85
        else:                                    # nothing fits inside: sit on the point
            dx = dy = 0.0; ha = va = "center"; alpha = 0.8
            x0, y0 = px - w / 2, py - h / 2
        rad = (dx * dx + dy * dy) ** 0.5
        ax.annotate(texts[i], XY[i], textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=fontsize, color=colors[i], family=SANS,
                    zorder=8, annotation_clip=False, alpha=alpha,
                    arrowprops=(dict(arrowstyle="-", color=colors[i], lw=0.6, alpha=0.5,
                                     shrinkA=1.0, shrinkB=3.0)
                                if rad >= leader_from else None))
        placed.append(Bbox([[x0, y0], [x0 + w, y0 + h]]))
    return placed


def legend2(ax, t, handles, loc="lower left", ncol=1, **kw):
    leg = ax.legend(handles=handles, loc=loc, ncol=ncol, fontsize=T2["legend"],
                    frameon=True, facecolor=t["surface"], edgecolor=t["axis"],
                    labelcolor=t["ink2"], handletextpad=0.8, borderpad=0.9,
                    labelspacing=0.7, **kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


def label_colours(D, t):
    return [t["series"][f] if f != "Lexicon" else t["muted"] for f in D["factor"]]


V2_NOTE = ("every trait kept; legibility comes from canvas size, repulsion and "
           "faceting, not from dropping words")


# ---------------------------------------------------------------- 1. editorial
def fig_editorial_v2(D, mode, anchors=False):
    t = THEME[mode]
    XY = D["X"][:, :2]
    cols = label_colours(D, t)
    tag = "v2anchors" if anchors else "v2"
    anc = anchor_set(D)

    if anchors:
        fig, ax, t = new_fig2(mode, 15.0)
        fig.subplots_adjust(left=0.042, right=0.988, top=0.855, bottom=0.075)
        zero_lines(ax, t)
        draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, alpha=0.085, lw=1.4)
        scatter2(ax, XY, D["factor"], D["keyed"], t, s=105)
        which = [i for i, n in enumerate(D["traits"]) if n in anc]
        leader_labels(fig, ax, XY, which, D["traits"], cols, T2["label_big"])
        pad = 0.18
        ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
        ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
        ax.set_xlabel(axis_label(D["titles"], 0), color=t["ink2"],
                      fontsize=T2["axis"], family=SANS, labelpad=8)
        ax.set_ylabel(axis_label(D["titles"], 1), color=t["ink2"],
                      fontsize=T2["axis"], family=SANS, labelpad=8)
        style_axes2(ax, t)
        legend2(ax, t, legend_handles(t), loc="lower left")
        title2(fig, t, "Same word, opposite pole, opposite side",
               "134 trait LoRA adapters on the first two recovered factors, with the "
               f"{len(anc)} anchor words labelled and the rest left as points.\nEach soft shell "
               "holds the densest 45 per cent of one Goldberg factor at one pole -- solid for "
               "positively keyed words, dashed for negatively keyed --\nand the connector runs "
               "from one pole's centre to the other.", serif=True)
        caption2(fig, t, CAPTIONS_V2["editorial"])
        return save(fig, "editorial_" + tag, mode)

    # full content: one large panel plus five facets, one per Goldberg factor
    fig = plt.figure(figsize=(WIDTH_V2, 34.0), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.042, 0.508, 0.946, 0.428]); ax.set_facecolor(t["surface"])
    zero_lines(ax, t)
    draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, alpha=0.085, lw=1.4)
    scatter2(ax, XY, D["factor"], D["keyed"], t, s=105)
    pad = 0.18
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
    leader_labels(fig, ax, XY, list(range(134)), D["traits"], cols, T2["label"])
    ax.set_xlabel(axis_label(D["titles"], 0), color=t["ink2"], fontsize=T2["axis"],
                  family=SANS, labelpad=8)
    ax.set_ylabel(axis_label(D["titles"], 1), color=t["ink2"], fontsize=T2["axis"],
                  family=SANS, labelpad=8)
    style_axes2(ax, t)
    legend2(ax, t, legend_handles(t), loc="lower left")
    ax.set_title("All 134, labelled", color=t["ink"], fontsize=T2["panel"],
                 family=SANS, loc="left", pad=12)

    # five facets
    for n, f in enumerate(FACTORS[:5]):
        r, c = divmod(n, 3)
        axf = fig.add_axes([0.042 + c * 0.3215, 0.268 - r * 0.222, 0.286, 0.192])
        axf.set_facecolor(t["surface"])
        zero_lines(axf, t)
        for k, ls in (("+", "-"), ("-", (0, (4, 3)))):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            d = density_shell(XY[m])
            if d is None:
                continue
            XX, YY, Z, lev = d
            axf.contourf(XX, YY, Z, levels=[lev, Z.max() * 1.01],
                         colors=[t["series"][f]], alpha=0.085, zorder=1)
            axf.contour(XX, YY, Z, levels=[lev], colors=[t["series"][f]],
                        linewidths=1.3, linestyles=[ls], zorder=2)
        scatter2(axf, XY, D["factor"], D["keyed"], t, s=105, only=f, grey_rest=True)
        which = [i for i in range(134) if D["factor"][i] == f]
        leader_labels(fig, axf, XY, which, D["traits"],
                      [t["series"][f]] * 134, T2["label_big"], all_xy=XY)
        axf.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
        axf.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
        axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_color(t["axis"]); sp.set_linewidth(0.6)
        axf.set_title(f"{FACTOR_SHORT[f]}  --  its 20 markers labelled, the other 114 grey",
                      color=t["ink"], fontsize=T2["panel"], family=SANS, loc="left",
                      pad=10)

    # the sixth slot carries the reading note
    fig.text(0.042 + 2 * 0.3215, 0.268 - 1 * 0.222 + 0.192,
             "Every trait kept.\nLegibility comes from canvas size,\nrepulsion and faceting.",
             ha="left", va="top", color=t["ink"], fontsize=T2["panel"], family=SANS,
             linespacing=1.5)
    fig.text(0.042 + 2 * 0.3215, 0.268 - 1 * 0.222 + 0.148,
             "Each facet holds the axes, the scale and the shells of the panel above it,\n"
             "so the five can be compared with one another and with the whole. The grey\n"
             "points are the other 114 adapters, unlabelled: the factor's own 20 markers\n"
             "are what carries type. Two shells of one colour on opposite sides of the\n"
             "origin is the bipolarity -- solid is the positively keyed half of the scale,\n"
             "dashed the negatively keyed half.",
             ha="left", va="top", color=t["ink2"], fontsize=T2["subtitle"], family=SANS,
             linespacing=1.6)

    title2(fig, t, "Same word, opposite pole, opposite side",
           "134 trait LoRA adapters on the first two recovered factors. Each soft shell holds the "
           "densest 45 per cent of one Goldberg factor at one pole -- solid for positively keyed\n"
           "words, dashed for negatively keyed -- and the connector runs from one pole's centre to "
           "the other. The factors come back as signed bipolar axes, not blobs.", serif=True)
    caption2(fig, t, CAPTIONS_V2["editorial"])
    return save(fig, "editorial_" + tag, mode)


# -------------------------------------------------------------------- 2. pairs
def fig_pairs_v2(D, mode, anchors=False):
    t = THEME[mode]
    X, titles = D["X"], D["titles"]
    tag = "v2anchors" if anchors else "v2"
    pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    anc = set(anchor_set(D))
    cols = label_colours(D, t)

    fig = plt.figure(figsize=(WIDTH_V2, 15.4), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    fig.subplots_adjust(left=0.035, right=0.99, top=0.855, bottom=0.115,
                        wspace=0.13, hspace=0.20)
    for n, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(2, 5, n + 1)
        ax.set_facecolor(t["surface"])
        zero_lines(ax, t)
        XY = X[:, [i, j]]
        draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, alpha=0.07, lw=1.0)
        scatter2(ax, XY, D["factor"], D["keyed"], t, s=62, lw=1.2)
        if anchors:
            which = [k for k, nm in enumerate(D["traits"]) if nm in anc]
            leader_labels(fig, ax, XY, which, D["traits"], cols, T2["label"],
                          rings=(8, 14, 22, 32, 44))
        ax.set_xlabel(short_axis_label(titles, i), color=t["ink2"],
                      fontsize=T2["tick"], family=SANS, labelpad=4)
        ax.set_ylabel(short_axis_label(titles, j), color=t["ink2"],
                      fontsize=T2["tick"], family=SANS, labelpad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(t["axis"]); sp.set_linewidth(0.6)
        lim = np.abs(XY).max() * 1.10
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
    fig.legend(handles=legend_handles(t), loc="upper center", ncol=5,
               fontsize=T2["legend"], frameon=False, labelcolor=t["ink2"],
               bbox_to_anchor=(0.5, 0.086), columnspacing=2.4, handletextpad=0.8)
    title2(fig, t, "All ten factor pairs",
           "Every one of the ten plane projections of the five-factor chart, at panel size. The keyed "
           "split shows up in each of them, not just the lucky one;\nsame axes, same colours, one "
           "legend. Read where the two shells of a colour sit."
           + ("\nAnchor words are labelled in each panel." if anchors else ""))
    caption2(fig, t, CAPTIONS_V2["pairs"])
    return save(fig, "pairs_" + tag, mode)


# ----------------------------------------------------------------------- 3. 3D
def fig_three_d_v2(D, mode, anchors=False):
    from mpl_toolkits.mplot3d import proj3d
    t = THEME[mode]
    X = D["X"][:, :3]
    titles = D["titles"]
    tag = "v2anchors" if anchors else "v2"
    anc = set(anchor_set(D))
    names = ANCHORS if anchors else sorted(anc)          # v1's ten, or all 24

    fig, ax, t = new_fig2(mode, 17.0, three_d=True)
    ax.set_position([-0.02, 0.028, 1.04, 0.83])
    ax.set_box_aspect((1.0, 1.0, 0.82), zoom=1.03)
    ax.view_init(elev=20, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color(to_rgb(t["surface"]) + (1.0,))
        pane._axinfo["grid"]["color"] = t["grid"]
        pane._axinfo["grid"]["linewidth"] = 0.5

    for f in FACTORS[:5]:
        for k in ("+", "-"):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            if len(m) < 4:
                continue
            pts = X[m]
            try:
                hull = ConvexHull(pts)
            except Exception:
                continue
            pc = Poly3DCollection([pts[sx] for sx in hull.simplices],
                                  alpha=0.10 if k == "+" else 0.055,
                                  facecolor=t["series"][f], edgecolor=t["series"][f],
                                  linewidths=0.45)
            pc.set_zsort("average")
            ax.add_collection3d(pc)

    cam = np.array([np.cos(np.radians(-58)) * np.cos(np.radians(20)),
                    np.sin(np.radians(-58)) * np.cos(np.radians(20)),
                    np.sin(np.radians(20))])
    depth = X @ cam
    dn = (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    sizes = 34 + 150 * dn
    for f in FACTORS:
        col = t["series"][f]
        if f == "Lexicon":
            m = [i for i in range(134) if D["factor"][i] == f]
            ax.scatter(X[m, 0], X[m, 1], X[m, 2], marker="x", s=sizes[m] * 0.85, c=col,
                       linewidths=1.5, depthshade=False)
            continue
        mp = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "+"]
        mn = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "-"]
        ax.scatter(X[mp, 0], X[mp, 1], X[mp, 2], marker="o", s=sizes[mp], c=col,
                   edgecolors=t["surface"], linewidths=0.8, depthshade=False)
        ax.scatter(X[mn, 0], X[mn, 1], X[mn, 2], marker="o", s=sizes[mn],
                   facecolors="none", edgecolors=col, linewidths=1.5, depthshade=False)

    ax.set_xlabel(titles[0], color=t["ink2"], fontsize=T2["axis"], family=SANS,
                  labelpad=16)
    ax.set_ylabel(titles[1], color=t["ink2"], fontsize=T2["axis"], family=SANS,
                  labelpad=16)
    ax.set_zlabel("Timidity  (- timid / + bold)", color=t["ink2"], fontsize=T2["axis"],
                  family=SANS, labelpad=16)
    ax.tick_params(colors=t["muted"], labelsize=T2["tick"])
    h3 = legend_handles(t)[:8]
    h3.append(Line2D([], [], linestyle="-", color=t["ink2"], lw=1.3,
                     label="shell = convex hull of one factor at one pole"))
    legend2(ax, t, h3, loc="upper left", bbox_to_anchor=(0.03, 0.93))

    # labels: project to 2D, then repel in the projection
    fig.canvas.draw()
    idx = {n: i for i, n in enumerate(D["traits"])}
    which = [idx[n] for n in names if n in idx]
    proj = np.array([proj3d.proj_transform(X[i, 0], X[i, 1], X[i, 2],
                                           ax.get_proj())[:2] for i in range(134)])
    leader_labels(fig, ax, proj, which, D["traits"], label_colours(D, t),
                  T2["label_big"], all_xy=proj)

    title2(fig, t, "The top three factors, one viewpoint",
           "Warmth, Competence and Timidity. Point size is depth-cued (nearer is larger); each "
           "translucent shell is one Goldberg factor at one pole.\n"
           f"{len(which)} anchor words are labelled with leader lines -- a third axis buys depth at "
           "the cost of label room, so this view names the poles rather than every word.")
    caption2(fig, t, CAPTIONS_V2["threed"])
    return save(fig, "threed_" + tag, mode)


# ------------------------------------------------------------- 4. dendrogram
def fig_dendrogram_v2(D, R, mode, anchors=False):
    """Rotated to horizontal so all 134 leaf labels read left to right at 13 px."""
    t = THEME[mode]
    tag = "v2anchors" if anchors else "v2"
    dist = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")

    height = 15.0 if anchors else 30.0
    fig = plt.figure(figsize=(WIDTH_V2, height), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    top = 1.0 - (2.05 / height)
    bot = 2.35 / height
    ax = fig.add_axes([0.035, bot, 0.678, top - bot]); ax.set_facecolor(t["surface"])
    axb = fig.add_axes([0.872, bot, 0.019, top - bot]); axb.set_facecolor(t["surface"])
    axk = fig.add_axes([0.896, bot, 0.019, top - bot]); axk.set_facecolor(t["surface"])

    dn = dendrogram(Z, ax=ax, no_labels=True, orientation="left",
                    link_color_func=lambda _: t["ink2"],
                    above_threshold_color=t["ink2"])
    leaves = dn["leaves"]
    for ln in ax.get_lines():
        ln.set_linewidth(0.6)
    for coll in ax.collections:
        coll.set_linewidth(0.6)
    xmin = 0.36                                  # no pair merges below 0.416
    ax.set_xlim(float(Z[:, 2].max()) * 1.03, xmin)
    ax.set_xlabel("cosine distance (1 - cos), axis truncated at 0.36", color=t["ink2"],
                  fontsize=T2["axis"], family=SANS, labelpad=8)
    style_axes2(ax, t, spines=("bottom",))
    ax.set_yticks([])

    n = len(leaves)
    fs = T2["label"] if not anchors else T2["label_big"]
    for pos, li in enumerate(leaves):
        f, k = D["factor"][li], D["keyed"][li]
        col = t["series"][f]
        lcol = col if f != "Lexicon" else t["muted"]
        y = 5 + 10 * pos
        lab = D["traits"][li] + ("" if f == "Lexicon" else ("   +" if k == "+" else "   -"))
        ax.text(xmin - 0.004 * (float(Z[:, 2].max()) - xmin), y, lab, ha="left",
                va="center", fontsize=fs, color=lcol, family=SANS, clip_on=False)
        for a, fc in ((axb, col), (axk, col if k == "+" else t["surface"])):
            a.add_patch(MplPolygon([[0, pos], [1, pos], [1, pos + 1], [0, pos + 1]],
                                   closed=True, facecolor=fc, edgecolor=col,
                                   linewidth=0.25))
    for a, lab in ((axb, "Goldberg factor"), (axk, "keying (filled +, open -)")):
        a.set_ylim(0, n); a.set_xlim(0, 1); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.text(0.5, -0.004, lab, transform=a.transAxes, ha="center", va="top",
               rotation=90, fontsize=T2["tick"], color=t["ink2"], family=SANS)
    fig.legend(handles=legend_handles(t)[:8], loc="upper right", ncol=2,
               fontsize=T2["legend"], frameon=False, labelcolor=t["ink2"],
               bbox_to_anchor=(0.99, top + (1.30 / height)))
    title2(fig, t, "Average linkage on cosine distance",
           "All 134 adapters, hierarchical clustering with no knowledge of the labels, rotated so "
           "every leaf label reads horizontally. The two bars on the right\ncolour each leaf by its "
           "Goldberg factor and its keying; runs of one colour are branches the clustering found on "
           "its own. The distance axis starts at 0.36 --\nno pair merges below 0.416, so nothing is "
           "hidden.")
    caption2(fig, t, CAPTIONS_V2["dendrogram"])
    return save(fig, "dendrogram_" + tag, mode)


# ---------------------------------------------------------------- 5. heatmap
def fig_heatmap_v2(D, R, mode, anchors=False):
    """Full cut: a 18-inch square with all 134 trait names on both edges at 13 px,
    the permuted null beneath it, and a colour bar large enough to read.
    Anchors cut: the same two matrices with factor-block labels only."""
    t = THEME[mode]
    tag = "v2anchors" if anchors else "v2"
    cmap = diverging_cmap(t)

    o = sort_order(D["factor"], D["keyed"], D["traits"])
    M = R[np.ix_(o, o)]
    v = max(np.percentile(np.abs(M - np.diag(np.diag(M))), 99.0), 0.25)

    nz = np.load(os.path.join(ROOT, "results",
                              "gram_data_null_permuted_p100_matched.npz"),
                 allow_pickle=True)
    nn = [str(x) for x in nz["names"]]
    tidx = {tr: i for i, tr in enumerate(D["traits"])}
    assert all(x in tidx for x in nn), "null names not a subset of the 134 traits"
    nfac = [D["factor"][tidx[x]] for x in nn]
    nkey = [D["keyed"][tidx[x]] for x in nn]
    Rn = centred_cosine(np.asarray(nz["G"], float))
    on = sort_order(nfac, nkey, nn)
    Mn = Rn[np.ix_(on, on)]

    real = (M, o, D["factor"], D["keyed"], [D["traits"][i] for i in o],
            "Real zoo, 134 adapters", "ordered by Goldberg factor, then keying")
    null = (Mn, on, nfac, nkey, [nn[i] for i in on],
            "Permuted-label null, 100 markers",
            "same recipe, trait labels reassigned before training")

    if anchors:
        H = 14.6
        fig = plt.figure(figsize=(WIDTH_V2, H), dpi=DPI)
        fig.patch.set_facecolor(t["page"])
        layout = [(real, (0.075, 0.115, 0.360, 0.660), 0.800),
                  (null, (0.545, 0.115, 0.270, 0.660), 0.800)]
        cax = fig.add_axes([0.878, 0.235, 0.020, 0.400])
        trait_ticks = False
    else:
        H = 42.0
        fig = plt.figure(figsize=(WIDTH_V2, H), dpi=DPI)
        fig.patch.set_facecolor(t["page"])
        wr, wn = 0.760, 0.567
        layout = [(real, (0.105, 0.470, wr, wr * WIDTH_V2 / H), 0.952),
                  (null, (0.105, 0.060, wn, wn * WIDTH_V2 / H), 0.415)]
        cax = fig.add_axes([0.905, 0.600, 0.017, 0.200])
        trait_ticks = True

    im = None
    for (panel, box, head_y) in layout:
        M_, o_, fac_, key_, names_, ttl, sub = panel
        ax = fig.add_axes(list(box)); ax.set_facecolor(t["surface"])
        im = ax.imshow(M_, cmap=cmap, vmin=-v, vmax=v, interpolation="nearest")
        fb, kb, spans = block_edges(o_, fac_, key_)
        for bd in fb[1:-1]:
            ax.axhline(bd - 0.5, color=t["ink"], lw=1.4, alpha=0.85)
            ax.axvline(bd - 0.5, color=t["ink"], lw=1.4, alpha=0.85)
        for bd in kb:
            ax.axhline(bd - 0.5, color=t["ink"], lw=0.7, alpha=0.6, ls=(0, (3, 2)))
            ax.axvline(bd - 0.5, color=t["ink"], lw=0.7, alpha=0.6, ls=(0, (3, 2)))

        if trait_ticks:
            keys = [fac_[i] for i in o_]
            kk = [key_[i] for i in o_]
            labs = [nm + ("" if kf == "Lexicon" else ("  +" if k2 == "+" else "  -"))
                    for nm, kf, k2 in zip(names_, keys, kk)]
            lcols = [t["series"][kf] if kf != "Lexicon" else t["muted"] for kf in keys]
            ax.set_xticks(range(len(o_))); ax.set_yticks(range(len(o_)))
            ax.set_xticklabels(labs, rotation=90, fontsize=T2["label"], family=SANS[0])
            ax.set_yticklabels(labs, fontsize=T2["label"], family=SANS[0])
            for lb, c in zip(ax.get_xticklabels(), lcols):
                lb.set_color(c)
            for lb, c in zip(ax.get_yticklabels(), lcols):
                lb.set_color(c)
            ax.tick_params(length=0, pad=4)
            for a_, b_, f_ in spans:                 # factor names outside the words
                mid = ((a_ + b_) / 2) / len(o_)
                ax.text(-0.068, 1 - mid, FACTOR_SHORT[f_], transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=T2["axis"],
                        color=t["ink"], family=SANS)
                ax.text(mid, 1.048, FACTOR_SHORT[f_], transform=ax.transAxes,
                        ha="center", va="center", fontsize=T2["axis"], color=t["ink"],
                        family=SANS)
        else:
            ticks = [(a_ + b_) / 2 - 0.5 for a_, b_, _ in spans]
            nm = [FACTOR_SHORT[f_] for _, _, f_ in spans]
            ax.set_xticks(ticks); ax.set_xticklabels(nm, rotation=32, ha="right",
                                                     fontsize=T2["axis"])
            ax.set_yticks(ticks); ax.set_yticklabels(nm, fontsize=T2["axis"])
            ax.tick_params(colors=t["ink2"], length=0, pad=4)
            for lb in ax.get_xticklabels() + ax.get_yticklabels():
                lb.set_color(t["ink2"]); lb.set_family(SANS[0])
        for sp in ax.spines.values():
            sp.set_color(t["axis"]); sp.set_linewidth(0.6)
        fig.text(box[0], head_y, ttl, ha="left", va="bottom", color=t["ink"],
                 fontsize=T2["panel"] + 3, family=SANS, fontweight="bold")
        fig.text(box[0], head_y - (0.34 / H), sub, ha="left", va="bottom",
                 color=t["ink2"], fontsize=T2["panel"], family=SANS)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("cosine, double-centred Gram", color=t["ink2"], fontsize=T2["axis"],
                 family=SANS, labelpad=14)
    cb.ax.tick_params(colors=t["muted"], labelsize=T2["axis"], length=5, width=0.8)
    cb.outline.set_edgecolor(t["axis"]); cb.outline.set_linewidth(0.6)

    title2(fig, t, "Positive on the diagonal blocks, negative across the poles",
           "The 134 x 134 cosine matrix sorted by Goldberg factor then keying, above the "
           "permuted-label null arm. Heavy rules are factor boundaries,\nhairlines split the two "
           "poles inside a factor. Same-keyed blocks run warm, opposite-keyed blocks run cool: "
           "that sign flip is the bipolarity."
           + ("\nEvery trait is named on both edges of both matrices." if trait_ticks else ""))
    caption2(fig, t, CAPTIONS_V2["heatmap"])
    return save(fig, "heatmap_" + tag, mode)


# ----------------------------------------------------------------- 6. strips
def fig_strips_v2(D, mode, anchors=False):
    t = THEME[mode]
    X, titles, order, big5 = D["X"], D["titles"], D["order"], D["big5"]
    g2f = {big5[k]: i for i, k in enumerate(order)}
    tag = "v2anchors" if anchors else "v2"
    anc = set(anchor_set(D))

    height = 30.0 if not anchors else 19.0
    fig = plt.figure(figsize=(WIDTH_V2, height), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    top = 1.0 - (2.35 / height); bot = 0.95 / height
    gap = 0.55 / height
    hgt = (top - bot - 4 * gap) / 5

    for n, f in enumerate(FACTORS[:5]):
        ax = fig.add_axes([0.035, top - (n + 1) * hgt - n * gap, 0.955, hgt])
        ax.set_facecolor(t["surface"])
        ai = g2f[f]; col = t["series"][f]
        m = [i for i in range(134) if D["factor"][i] == f]
        lim = np.abs(X[:, ai]).max() * 1.10
        ax.set_xlim(-lim, lim); ax.set_ylim(-4.6, 4.6)
        ax.axhline(0, color=t["axis"], lw=1.0, zorder=1)
        ax.axvline(0, color=t["grid"], lw=0.9, ls=(0, (5, 4)), zorder=0)
        fig.canvas.draw(); rend = fig.canvas.get_renderer()
        placed = []
        counts = {"+": 0, "-": 0}
        for k in ("+", "-"):
            side = sorted([i for i in m if D["keyed"][i] == k], key=lambda i: X[i, ai])
            counts[k] = len(side)
            sgn = 1.0 if k == "+" else -1.0
            for i in side:
                x = X[i, ai]
                show = (not anchors) or D["traits"][i] in anc
                tx = None
                for lane in range(14):
                    y = sgn * (0.42 + 0.255 * lane)
                    if not show:
                        y = sgn * 0.42
                        break
                    tx = ax.text(x, y + sgn * 0.12, D["traits"][i], rotation=32,
                                 ha="left" if k == "+" else "right",
                                 va="bottom" if k == "+" else "top",
                                 fontsize=T2["label"], color=t["ink"] if k == "+"
                                 else t["ink2"], family=SANS)
                    bb = tx.get_window_extent(renderer=rend).expanded(1.01, 1.10)
                    if not any(bb.overlaps(o) for o in placed):
                        placed.append(bb); break
                    tx.remove(); tx = None
                if show and tx is None:
                    y = sgn * 0.42
                    ax.text(x, y + sgn * 0.16, " " + D["traits"][i], rotation=0,
                            ha="left", va="bottom" if k == "+" else "top",
                            fontsize=T2["label"], color=t["muted"], family=SANS)
                ax.plot([x, x], [0, y], color=col, lw=1.1, alpha=0.5, zorder=2)
                if k == "+":
                    ax.plot([x], [y], marker="o", ms=9.0, color=col, mec=t["surface"],
                            mew=1.0, zorder=3)
                else:
                    ax.plot([x], [y], marker="o", ms=9.0, mfc="none", mec=col, mew=1.8,
                            zorder=3)
        ax.set_yticks([])
        ax.tick_params(colors=t["muted"], labelsize=T2["tick"], length=4)
        for lb in ax.get_xticklabels():
            lb.set_color(t["muted"])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"]); ax.spines["bottom"].set_linewidth(0.7)
        nm = short_axis_label(titles, ai)
        ax.text(-lim * 0.998, 4.5, f"{FACTOR_SHORT[f]}   ->   recovered factor: {nm}",
                ha="left", va="top", fontsize=T2["panel"], color=t["ink"], family=SANS,
                fontweight="bold")
        if titles[ai] == "Timidity":
            ax.text(lim * 0.998, 4.5,
                    "timid / fearful pole to the left, bold / stable pole to the right",
                    ha="right", va="top", fontsize=T2["tick"], color=t["muted"],
                    family=SANS)
        ax.text(-lim * 0.998, -4.5, f"{counts['-']} negatively keyed markers below the line",
                ha="left", va="bottom", fontsize=T2["tick"], color=t["muted"], family=SANS)
        ax.text(lim * 0.998, -4.5, f"{counts['+']} positively keyed above",
                ha="right", va="bottom", fontsize=T2["tick"], color=t["muted"], family=SANS)

    title2(fig, t, "Opposite words point opposite ways",
           "Each strip is one Big Five scale's 20 markers, placed at their coordinate on the "
           "recovered factor matched to that scale. Positively keyed above the line,\nnegatively "
           "keyed below; labels are staggered into lanes so none overlaps. A bipolar axis means the "
           "two rows sit on opposite sides of zero.")
    caption2(fig, t, CAPTIONS_V2["strips"])
    return save(fig, "strips_" + tag, mode)


# -------------------------------------------------------------------- 7. MDS
def fig_mds_v2(D, R, mode, anchors=False):
    t = THEME[mode]
    tag = "v2anchors" if anchors else "v2"
    dist = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=17, n_init=8,
              max_iter=600, normalized_stress="auto").fit_transform(dist)
    K = 5
    nbr = np.argsort(dist + np.eye(134) * 9, axis=1)[:, :K]
    segs, scols, same = [], [], 0
    for i in range(134):
        for j in nbr[i]:
            segs.append([emb[i], emb[j]])
            sm = D["factor"][i] == D["factor"][j]
            same += sm
            scols.append(t["series"][D["factor"][i]] if sm else t["muted"])
    frac = same / len(segs)

    fig, ax, t = new_fig2(mode, 22.0 if not anchors else 15.0)
    fig.subplots_adjust(left=0.025, right=0.985,
                        top=1.0 - (2.5 / fig.get_figheight()),
                        bottom=0.95 / fig.get_figheight())
    ax.add_collection(LineCollection(segs, colors=scols, linewidths=0.5, alpha=0.3,
                                     zorder=1))
    scatter2(ax, emb, D["factor"], D["keyed"], t, s=110)
    pad = np.abs(emb).max() * 0.10
    ax.set_xlim(emb[:, 0].min() - pad, emb[:, 0].max() + pad)
    ax.set_ylim(emb[:, 1].min() - pad, emb[:, 1].max() + pad)
    anc = set(anchor_set(D))
    which = ([i for i, n in enumerate(D["traits"]) if n in anc] if anchors
             else list(range(134)))
    leader_labels(fig, ax, emb, which, D["traits"], label_colours(D, t),
                  T2["label_big"] if anchors else T2["label"])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    legend2(ax, t, legend_handles(t)[:8], loc="lower left")
    title2(fig, t, "The zoo as a constellation",
           "Metric MDS on cosine distance between adapters, with each adapter joined to its five "
           "nearest neighbours. Edges are coloured when both ends carry\nthe same Goldberg factor "
           f"label and grey when they do not: {frac:.0%} of the {len(segs)} edges are same-factor. "
           + ("Anchor words only." if anchors else "All 134 words are labelled, with leader lines "
              "where a label had to move."))
    caption2(fig, t, CAPTIONS_V2["mds"])
    return save(fig, "mds_" + tag, mode)


# --------------------------------------------------------------- 8. monochrome
def fig_mono_v2(D, mode="light", anchors=False):
    t = dict(THEME["light"])
    t["surface"] = "#ffffff"; t["page"] = "#ffffff"
    t["ink"] = "#000000"; t["ink2"] = "#1a1a1a"; t["muted"] = "#3c3c3c"
    t["grid"] = "#c8c8c8"; t["axis"] = "#000000"
    t["series"] = {f: "#000000" for f in FACTORS}
    tag = "v2anchors" if anchors else "v2"

    XY = D["X"][:, :2]
    fig = plt.figure(figsize=(WIDTH_V2, 20.0 if not anchors else 15.0), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    h = fig.get_figheight()
    ax = fig.add_axes([0.045, 0.95 / h, 0.945, 1.0 - (3.35 / h)])
    ax.set_facecolor(t["surface"])
    ax.axhline(0, color=t["grid"], lw=0.7, zorder=0)
    ax.axvline(0, color=t["grid"], lw=0.7, zorder=0)
    pad = 0.18
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
    draw_key_hulls(ax, XY, D["factor"], D["keyed"], t, lw=1.1, mono=True, fill=False,
                   connectors=True)
    scatter2(ax, XY, D["factor"], D["keyed"], t, s=80, lw=1.3)

    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    seed = []
    for f in FACTORS[:5]:
        for k in ("+", "-"):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            if not m:
                continue
            c = XY[m].mean(axis=0)
            tx = ax.text(c[0], c[1],
                         f"{FACTOR_SHORT[f]}\n{'positively' if k == '+' else 'negatively'} keyed",
                         ha="center", va="center", fontsize=T2["label_big"], family=SERIF,
                         color="#000000", zorder=9,
                         bbox=dict(boxstyle="round,pad=0.36", fc="#ffffff",
                                   ec="#000000", lw=0.7))
            seed.append(tx.get_window_extent(renderer=rend).expanded(1.03, 1.12))
    anc = set(anchor_set(D))
    which = ([i for i, n in enumerate(D["traits"]) if n in anc] if anchors
             else list(range(134)))
    from matplotlib.transforms import Bbox
    disp = ax.transData.transform(XY)
    r = 4.5 * (DPI / 100.0)
    seed += [Bbox([[x - r, y - r], [x + r, y + r]]) for x, y in disp]
    lp = leader_labels(fig, ax, XY, which, D["traits"], ["#000000"] * 134,
                       T2["label_big"] if anchors else T2["label"], avoid_points=False)
    ax.set_xlabel(axis_label(D["titles"], 0), color="#000000", fontsize=T2["axis"],
                  family=SANS, labelpad=8)
    ax.set_ylabel(axis_label(D["titles"], 1), color="#000000", fontsize=T2["axis"],
                  family=SANS, labelpad=8)
    style_axes2(ax, t)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color("#333333")
    title2(fig, t, "Warmth and Competence, black and white",
           "The editorial scatter with the colour channel removed: only the words and the shells. "
           "A solid outline is the positively keyed half of a Goldberg scale,\ndashed the negatively "
           "keyed half, and the bar between them joins the two poles. For print and for "
           "forced-colours.", serif=True)
    caption2(fig, t, CAPTIONS_V2["mono"])
    return save(fig, "mono_" + tag, mode)


# ------------------------------------------------- 1a/1b/1c. the calm variants
#
# Samuel on the full editorial panel: "reduce the overwhelm". Three quieter cuts of
# the same plot. All keep the palette, the centring and the framing; they differ
# only in how much is drawn at once.

CALM_PAIRS = ["agreeable", "rude", "warm", "cold", "organized", "careless",
              "conscientious", "negligent", "anxious", "relaxed", "talkative",
              "quiet", "imaginative", "unimaginative", "intellectual",
              "unintelligent"]
CALM_EXTRA = ["fearful", "unkind", "extraverted", "pleasant", "impractical", "shallow"]


def calm_anchor_set(D):
    """~30 words: the strongest positive and negative loader on each of the five
    factors, the pairs Samuel named, and a few more well-known poles."""
    X, tr = D["X"], D["traits"]
    out = set(CALM_PAIRS) | set(CALM_EXTRA)
    for ax in range(5):
        out.add(tr[int(np.argmax(X[:, ax]))])
        out.add(tr[int(np.argmin(X[:, ax]))])
    return [t for t in tr if t in out]


def _ed_axes(fig, ax, D, t, light_grid=True, pad=0.20):
    XY = D["X"][:, :2]
    ax.axhline(0, color=t["grid"], lw=0.7 if light_grid else 0.9, zorder=0)
    ax.axvline(0, color=t["grid"], lw=0.7 if light_grid else 0.9, zorder=0)
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
    ax.set_xlabel(axis_label(D["titles"], 0), color=t["ink2"], fontsize=T2["axis"],
                  family=SANS, labelpad=10)
    ax.set_ylabel(axis_label(D["titles"], 1), color=t["ink2"], fontsize=T2["axis"],
                  family=SANS, labelpad=10)
    style_axes2(ax, t)


def fig_editorial_calm(D, mode):
    """(A) No shells, no connectors, ~30 anchor labels, lots of air."""
    t = THEME[mode]
    XY = D["X"][:, :2]
    fig, ax, t = new_fig2(mode, 15.5)
    fig.subplots_adjust(left=0.058, right=0.972, top=0.845, bottom=0.088)
    _ed_axes(fig, ax, D, t)
    scatter2(ax, XY, D["factor"], D["keyed"], t, s=120, lw=1.5)
    anc = calm_anchor_set(D)
    which = [i for i, n in enumerate(D["traits"]) if n in set(anc)]
    leader_labels(fig, ax, XY, which, D["traits"], label_colours(D, t),
                  T2["label_big"], rings=(11, 19, 29, 42, 58, 76))
    legend2(ax, t, legend_handles(t)[:8], loc="lower left")
    title2(fig, t, "Same word, opposite pole, opposite side",
           f"134 trait adapters on the first two recovered factors. {len(anc)} anchor words are "
           "named; the rest are points. No shells, no connectors -- just where the words fall.",
           serif=True)
    caption2(fig, t, CAPTIONS_V2["editorial_calm"])
    return save(fig, "editorial_calm_v2", mode)


def fig_editorial_focus(D, mode):
    """(B) Shells and connectors only for the two factors the axes are named for."""
    t = THEME[mode]
    XY = D["X"][:, :2]
    keep = ("Agreeableness", "Conscientiousness")
    fig, ax, t = new_fig2(mode, 16.5)
    fig.subplots_adjust(left=0.058, right=0.972, top=0.850, bottom=0.082)
    _ed_axes(fig, ax, D, t)

    rest = [i for i in range(134) if D["factor"][i] not in keep]
    ax.scatter(XY[rest, 0], XY[rest, 1], marker="o", s=42,
               c=t["series"]["Lexicon"], edgecolors="none", zorder=2, alpha=0.9)
    for f in keep:
        col = t["series"][f]
        for k, ls in (("+", "-"), ("-", (0, (4, 3)))):
            m = [i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == k]
            d = density_shell(XY[m])
            if d is None:
                continue
            XX, YY, Z, lev = d
            ax.contourf(XX, YY, Z, levels=[lev, Z.max() * 1.01], colors=[col],
                        alpha=0.085, zorder=1)
            ax.contour(XX, YY, Z, levels=[lev], colors=[col], linewidths=1.5,
                       linestyles=[ls], zorder=2)
        cp = XY[[i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "+"]].mean(0)
        cn = XY[[i for i in range(134) if D["factor"][i] == f and D["keyed"][i] == "-"]].mean(0)
        ax.plot([cn[0], cp[0]], [cn[1], cp[1]], color=col, lw=2.0, alpha=0.6, zorder=3,
                solid_capstyle="round")
        ax.plot([cp[0]], [cp[1]], marker="o", ms=11, color=col, mec=t["surface"], mew=1.2,
                zorder=4)
        ax.plot([cn[0]], [cn[1]], marker="o", ms=11, mfc=t["surface"], mec=col, mew=2.2,
                zorder=4)
        sel = [i for i in range(134) if D["factor"][i] == f]
        mp = [i for i in sel if D["keyed"][i] == "+"]
        mn = [i for i in sel if D["keyed"][i] == "-"]
        ax.scatter(XY[mp, 0], XY[mp, 1], marker="o", s=130, c=col,
                   edgecolors=t["surface"], linewidths=1.2, zorder=6)
        ax.scatter(XY[mn, 0], XY[mn, 1], marker="o", s=130, facecolors="none",
                   edgecolors=col, linewidths=1.8, zorder=5)
    which = [i for i in range(134) if D["factor"][i] in keep]
    leader_labels(fig, ax, XY, which, D["traits"], label_colours(D, t), T2["label_big"],
                  rings=(10, 18, 27, 39, 54, 72))
    h = [Line2D([], [], marker="o", linestyle="none", color=t["series"][f],
                markerfacecolor=t["series"][f], markeredgecolor=t["series"][f],
                markersize=9, label=FACTOR_SHORT[f]) for f in keep]
    h.append(Line2D([], [], marker="o", linestyle="none", color=t["series"]["Lexicon"],
                    markerfacecolor=t["series"]["Lexicon"],
                    markeredgecolor=t["series"]["Lexicon"], markersize=7,
                    label="the other 94 adapters"))
    h.append(Line2D([], [], marker="o", linestyle="none", color=t["ink2"],
                    markerfacecolor=t["ink2"], markeredgecolor=t["ink2"], markersize=9,
                    label="positively keyed (filled)"))
    h.append(Line2D([], [], marker="o", linestyle="none", color=t["ink2"],
                    markerfacecolor="none", markeredgecolor=t["ink2"], markersize=9,
                    markeredgewidth=1.8, label="negatively keyed (hollow)"))
    h.append(Line2D([], [], linestyle="-", color=t["ink2"], lw=1.6,
                    label="pole connector: - centroid to + centroid"))
    legend2(ax, t, h, loc="lower left")
    title2(fig, t, "The two factors these axes are named for",
           "Warmth is the Agreeableness factor and Competence the Conscientiousness one, so only "
           "those two carry shells, connectors and type here.\nThe other 94 adapters stay as grey "
           "points: present, but not competing for attention.", serif=True)
    caption2(fig, t, CAPTIONS_V2["editorial_focus"])
    return save(fig, "editorial_focus_v2", mode)


def fig_editorial_facets(D, mode):
    """(C) 2 x 3 grid: five Goldberg factors plus the held-out Lexicon, shared axes."""
    t = THEME[mode]
    XY = D["X"][:, :2]
    fig = plt.figure(figsize=(WIDTH_V2, 17.5), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    padx, pady = 0.50, 0.34            # room inside the panel for the leader labels
    xlim = (XY[:, 0].min() - padx, XY[:, 0].max() + padx)
    ylim = (XY[:, 1].min() - pady, XY[:, 1].max() + pady)
    left, bottom, w, hgt = 0.040, 0.068, 0.303, 0.352
    hgap, vgap = 0.0235, 0.055

    for n, f in enumerate(FACTORS):
        r, c = divmod(n, 3)
        ax = fig.add_axes([left + c * (w + hgap),
                           bottom + (1 - r) * (hgt + vgap), w, hgt])
        ax.set_facecolor(t["surface"])
        ax.axhline(0, color=t["grid"], lw=0.6, zorder=0)
        ax.axvline(0, color=t["grid"], lw=0.6, zorder=0)
        rest = [i for i in range(134) if D["factor"][i] != f]
        ax.scatter(XY[rest, 0], XY[rest, 1], marker="o", s=26,
                   c=t["series"]["Lexicon"], edgecolors="none", zorder=1, alpha=0.85)
        col = t["series"][f]
        sel = [i for i in range(134) if D["factor"][i] == f]
        if f != "Lexicon":
            for k, ls in (("+", "-"), ("-", (0, (4, 3)))):
                m = [i for i in sel if D["keyed"][i] == k]
                d = density_shell(XY[m])
                if d is None:
                    continue
                XX, YY, Z, lev = d
                ax.contourf(XX, YY, Z, levels=[lev, Z.max() * 1.01], colors=[col],
                            alpha=0.085, zorder=2)
                ax.contour(XX, YY, Z, levels=[lev], colors=[col], linewidths=1.3,
                           linestyles=[ls], zorder=3)
            cp = XY[[i for i in sel if D["keyed"][i] == "+"]].mean(0)
            cn = XY[[i for i in sel if D["keyed"][i] == "-"]].mean(0)
            ax.plot([cn[0], cp[0]], [cn[1], cp[1]], color=col, lw=1.7, alpha=0.6,
                    zorder=4, solid_capstyle="round")
            ax.plot([cp[0]], [cp[1]], marker="o", ms=9, color=col, mec=t["surface"],
                    mew=1.0, zorder=5)
            ax.plot([cn[0]], [cn[1]], marker="o", ms=9, mfc=t["surface"], mec=col,
                    mew=2.0, zorder=5)
            mp = [i for i in sel if D["keyed"][i] == "+"]
            mn = [i for i in sel if D["keyed"][i] == "-"]
            ax.scatter(XY[mp, 0], XY[mp, 1], marker="o", s=95, c=col,
                       edgecolors=t["surface"], linewidths=1.0, zorder=7)
            ax.scatter(XY[mn, 0], XY[mn, 1], marker="o", s=95, facecolors="none",
                       edgecolors=col, linewidths=1.6, zorder=6)
            sub = "20 markers, solid shell = positively keyed, dashed = negatively keyed"
        else:
            ax.scatter(XY[sel, 0], XY[sel, 1], marker="x", s=95, c=t["ink2"],
                       linewidths=1.8, zorder=7)
            sub = "34 held-out words, never used to define the factors"
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(t["axis"]); sp.set_linewidth(0.6)
        lcol = col if f != "Lexicon" else t["ink2"]
        leader_labels(fig, ax, XY, sel, D["traits"], [lcol] * 134, T2["label_big"],
                      all_xy=XY)
        ax.set_title(FACTOR_SHORT[f], color=t["ink"], fontsize=T2["panel"], family=SANS,
                     loc="left", pad=22)
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom",
                color=t["ink2"], fontsize=T2["tick"], family=SANS)
        if n >= 3:
            ax.set_xlabel(short_axis_label(D["titles"], 0), color=t["ink2"],
                          fontsize=T2["tick"], family=SANS, labelpad=5)
        if c == 0:
            ax.set_ylabel(short_axis_label(D["titles"], 1), color=t["ink2"],
                          fontsize=T2["tick"], family=SANS, labelpad=5)
    title2(fig, t, "One factor at a time",
           "The same plane, the same limits and the same scale in all six panels: each names one "
           "group's words and greys the rest.\nWarmth across, Competence up; the axes are the first "
           "two recovered factors.", serif=True)
    caption2(fig, t, CAPTIONS_V2["editorial_facets"])
    return save(fig, "editorial_facets_v2", mode)


CAPTIONS_V2 = {}


def _fill_captions_v2():
    CAPTIONS_V2.update({
        "editorial": "Warmth against Competence, factor-chart coordinates mean-centred; 45% "
                     "density shells per Goldberg factor label and keying; " + CONGRUENCE
                     + ". " + SRC_LINE + ".",
        "pairs": "Ten pairwise projections of the 134 x 5 factor chart, mean-centred; 45% density "
                 "shells per Goldberg factor and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "threed": "Factors 1-3 of the factor chart, mean-centred, azim -58 / elev 20; convex hulls "
                  "per Goldberg factor and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "dendrogram": "Average-linkage dendrogram on 1 - cosine between adapters, all 134 leaves; "
                      + CONGRUENCE + ". " + SRC_GRAM + ".",
        "heatmap": "Cosine from the double-centred exact Gram, diverging scale centred at zero; "
                   "null arm is the permuted-label retrain over the 100 Goldberg markers only (the "
                   "34 Lexicon words were never trained in the null). " + CONGRUENCE + ". "
                   + SRC_GRAM + "; qwen35/results/gram_data_null_permuted_p100_matched.npz.",
        "strips": "Marker coordinate on its own recovered factor, factor chart mean-centred; "
                  "Emotional stability has 6 positively and 14 negatively keyed markers in "
                  "Goldberg's published set; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "mds": "2D metric MDS from 1 - cosine on the double-centred exact Gram, k = 5 "
               "nearest-neighbour edges, axes have no units and no orientation; " + CONGRUENCE
               + ". " + SRC_GRAM + ".",
        "mono": "Warmth against Competence, factor-chart coordinates mean-centred; 45% density "
                "shells per Goldberg factor label and keying; " + CONGRUENCE + ". "
                + SRC_LINE + ".",
        "editorial_calm": "Warmth against Competence, factor-chart coordinates mean-centred, "
                          "anchor words only, no shells; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "editorial_focus": "Warmth against Competence, mean-centred; 45% density shells and pole "
                           "connectors for Agreeableness and Conscientiousness only; "
                           + CONGRUENCE + ". " + SRC_LINE + ".",
        "editorial_facets": "Warmth against Competence, mean-centred, shared limits; one panel per "
                            "Goldberg factor label plus the held-out Lexicon; " + CONGRUENCE
                            + ". " + SRC_LINE + ".",
    })


STYLES_V2 = [
    ("editorial", "Editorial scatter",
     "One full-width panel with all 134 words labelled by leader line, then five facets -- one per "
     "Goldberg factor -- each naming that factor's 20 markers in large type against the other 114 "
     "as grey points."),
    ("pairs", "Pairs matrix",
     "All ten plane projections at panel size, larger markers, one shared legend at 16 px."),
    ("threed", "Static 3D render",
     "Warmth, Competence and Timidity from one viewpoint with depth-cued sizes and translucent "
     "shells; the 24 anchor words carry leader lines, because a third axis spends the label room."),
    ("dendrogram", "Dendrogram",
     "Rotated to horizontal on a 30-inch canvas so all 134 leaf labels read left to right at 13 px, "
     "with the factor and keying bars running down the right-hand side."),
    ("heatmap", "Sorted cosine heatmap",
     "A 19-inch square with every one of the 134 trait names on both edges at 13 px, the permuted "
     "null beneath it, and a colour bar you can actually read."),
    ("strips", "Bipolar axis strips",
     "All twenty markers per strip at 13 px, rotated 32 degrees and staggered into lanes by measured "
     "bounding box so no two labels touch."),
    ("mds", "MDS constellation",
     "All 134 words labelled with leader lines on a 22-inch canvas, faint five-nearest-neighbour "
     "edges, larger nodes."),
    ("mono", "Minimal monochrome",
     "The same full-label treatment with the colour channel removed, for print and forced-colours."),
    ("editorial_calm", "Editorial, calm",
     "The same points and colours with the shells and connectors taken away and only about thirty "
     "anchor words named, on a light grid with generous margins."),
    ("editorial_focus", "Editorial, focused",
     "Shells, connectors and type for Agreeableness and Conscientiousness only -- the two factors "
     "the two axes are named for -- with the other 94 adapters as small grey points."),
    ("editorial_facets", "Editorial, facets",
     "A 2 x 3 grid on shared axes and limits: one panel per Goldberg factor plus one for the 34 "
     "held-out Lexicon words, each naming its own group and greying the rest."),
]


# ============================================================================
# gallery + readme
# ============================================================================
STYLES = [
    ("editorial", "Editorial scatter",
     "Warmth against Competence with every one of the 134 trait words set in small type, "
     "coloured by Goldberg factor label, filled for positively keyed and hollow for negatively "
     "keyed, with a soft density shell round each factor-and-pole half and a connector joining the two poles.",
     "fig_editorial"),
    ("pairs", "Pairs matrix",
     "All ten plane projections of the five-factor chart as small multiples with one shared "
     "legend, so the separation can be checked in every plane rather than the one that flatters it.",
     "fig_pairs"),
    ("threed", "Static 3D render",
     "Warmth, Competence and Timidity from a single viewpoint with depth-cued point sizes, "
     "translucent shells per factor and pole, and ten anchor words labelled.",
     "fig_three_d"),
    ("dendrogram", "Dendrogram",
     "Average linkage on cosine distance with no label input, leaf words coloured by Goldberg "
     "factor and marked with their keying, and two membership bars under the leaves.",
     "fig_dendrogram"),
    ("heatmap", "Sorted cosine heatmap",
     "The 134 x 134 cosine matrix ordered by factor then keying beside the permuted-label null "
     "arm, on a diverging scale centred at zero so the negative opposite-pole blocks are visible.",
     "fig_heatmap"),
    ("strips", "Bipolar axis strips",
     "One horizontal strip per Big Five scale, its 20 markers placed on the matched recovered "
     "factor with positively keyed words above the line and negatively keyed below.",
     "fig_strips"),
    ("mds", "MDS constellation",
     "Metric MDS from cosine distance with faint five-nearest-neighbour edges, coloured when "
     "both ends share a factor label; the dark variant reads as a star chart.",
     "fig_mds"),
    ("mono", "Minimal monochrome",
     "The editorial scatter with the colour channel removed: words, density shells and named shell "
     "labels only, high contrast, for print and forced-colours.",
     "fig_mono"),
]

CAPTIONS = {}


def build_gallery(files):
    from PIL import Image
    parts = []
    total = 0
    for stem, name, blurb, _ in STYLES:
        png = os.path.join(OUT, f"{stem}_light.png")
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        if im.width > 1600:
            im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        raw = buf.getvalue()
        if len(raw) > 1_400_000:                       # quantize only if it is heavy
            buf = io.BytesIO()
            im.quantize(colors=192, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG) \
              .save(buf, format="PNG", optimize=True)
            raw = buf.getvalue()
        total += len(raw)
        b64 = base64.b64encode(raw).decode()
        dark = f"{stem}_dark.png"
        variants = "light and dark" if os.path.exists(os.path.join(OUT, dark)) else "light only"
        parts.append(f"""<section id="{stem}">
  <h2>{name}</h2>
  <p class="blurb">{blurb}</p>
  <img src="data:image/png;base64,{b64}" alt="{name}">
  <p class="cap">{CAPTIONS.get(stem, '')}</p>
  <p class="files"><code>{stem}_light.png</code> &middot; <code>{stem}_light.svg</code>
     &middot; variants: {variants}</p>
</section>""")
    nav = " ".join(f'<a href="#{s}">{n}</a>' for s, n, _, _ in STYLES
                   if os.path.exists(os.path.join(OUT, f"{s}_light.png")))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trait adapter clusters in factor space</title>
<style>
  :root {{ color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b;
           --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink);
         font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:40px 20px 90px; }}
  header p {{ color:var(--ink2); max-width:70ch; }}
  h1 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:34px;
        margin:0 0 10px; letter-spacing:-0.01em; }}
  h2 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:24px;
        margin:0 0 6px; }}
  nav {{ margin:22px 0 40px; padding:12px 0; border-top:1px solid var(--rule);
        border-bottom:1px solid var(--rule); display:flex; flex-wrap:wrap; gap:8px 18px; }}
  nav a {{ color:var(--ink2); text-decoration:none; font-size:13px; }}
  nav a:hover {{ color:var(--ink); text-decoration:underline; }}
  section {{ margin:0 0 64px; }}
  .blurb {{ color:var(--ink2); max-width:80ch; margin:0 0 14px; }}
  img {{ width:100%; height:auto; display:block; background:var(--surface);
        border:1px solid var(--rule); border-radius:4px; }}
  .cap {{ color:var(--ink2); font-size:12.5px; margin:10px 0 4px; max-width:100ch; }}
  .files {{ color:var(--muted); font-size:12px; margin:0; }}
  code {{ font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink2); }}
  footer {{ border-top:1px solid var(--rule); padding-top:18px; color:var(--muted);
           font-size:12.5px; max-width:90ch; }}
</style></head><body>
<div class="wrap">
<header>
<h1>Trait adapter clusters in factor space</h1>
<p>Eight ways of drawing the same fact: the 134 personality-trait LoRA adapters group by the
Goldberg factor label and the keying of the word they were trained on. The colours are that
label, not a cluster discovered in the data; the five recovered oblimin factors match the Big
Five at Tucker congruence 0.40 to 0.68, so none clears the conventional 0.85 bar. What the
decomposition actually reports is signed bipolar axes rather than blobs, which is why every
density shell here is drawn per factor <em>and</em> pole.</p>
</header>
<nav>{nav}</nav>
{''.join(parts)}
<footer>Light variants shown; dark SVG and PNG variants sit beside each file.
Generated by <code>qwen35/figures/clusters/make_cluster_figures.py</code> from
<code>qwen35/analysis/viz_fa.json</code> and <code>qwen35/results/gram_sweep.npz</code>.</footer>
</div></body></html>"""
    path = os.path.join(OUT, "gallery.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path, total


def build_gallery_v2(max_px=1800, budget=8_400_000):
    """gallery_v2.html -- v2 full-content first, anchor variants second. The full
    files are 4800 px wide; only the display copies embedded here are downscaled."""
    from PIL import Image

    def embed(png, wide, cap=800_000):
        """Quantise first, shrink only as a last resort: these are mostly flat
        colour, so a 64-colour palette costs far less legibility than pixels do."""
        im = Image.open(png).convert("RGB")
        if im.width > wide:
            im = im.resize((wide, round(im.height * wide / im.width)), Image.LANCZOS)
        best = None
        for colors in (None, 192, 128, 96, 64, 48, 32):
            buf = io.BytesIO()
            (im if colors is None else
             im.quantize(colors=colors, method=Image.MEDIANCUT,
                         dither=Image.FLOYDSTEINBERG)).save(buf, format="PNG",
                                                            optimize=True)
            best = buf.getvalue()
            if len(best) < cap:
                break
        return best

    groups = [("full", "v2", "Full content, larger canvas",
               "Every trait, every pair, every leaf and the whole matrix. Legibility comes from "
               "canvas size, label repulsion with leader lines, and faceting -- not from dropping "
               "words. Rendered 4800 px wide (2x of a 2400 px display width); every label is at "
               "least 13 px at that width, titles 34 px and up, captions 15 px and up."),
              ("anchors", "v2anchors", "Anchor words only",
               "The same figures with only the 24 anchor words named -- the strongest positive and "
               "negative loader on each of the five factors plus the well-known poles. For when the "
               "point is the shape rather than the vocabulary.")]

    wide = max_px
    for attempt in range(6):
        parts, total = [], 0
        for gid, suffix, gtitle, gblurb in groups:
            rows = []
            for stem, name, blurb in STYLES_V2:
                png = os.path.join(OUT, f"{stem}_{suffix}_light.png")
                if not os.path.exists(png):
                    continue
                raw = embed(png, wide)
                total += len(raw)
                b64 = base64.b64encode(raw).decode()
                dark = os.path.join(OUT, f"{stem}_{suffix}_dark.png")
                variants = "light and dark" if os.path.exists(dark) else "light only"
                full = Image.open(png).size
                rows.append(f"""<section id="{stem}-{gid}">
  <h3>{name}</h3>
  <p class="blurb">{blurb}</p>
  <img src="data:image/png;base64,{b64}" alt="{name}">
  <p class="cap">{CAPTIONS_V2.get(stem, '')}</p>
  <p class="files"><code>{stem}_{suffix}_light.png</code> ({full[0]} x {full[1]} px)
     &middot; <code>{stem}_{suffix}_light.svg</code> &middot; variants: {variants}</p>
</section>""")
            if rows:
                parts.append(f'<div class="group"><h2>{gtitle}</h2>'
                             f'<p class="gblurb">{gblurb}</p>{"".join(rows)}</div>')
        if total < budget or wide <= 1200:
            break
        wide = int(wide * 0.88)

    nav = " ".join(
        f'<a href="#{s}-{gid}">{n}{"" if gid == "full" else " (anchors)"}</a>'
        for gid, suffix, _, _ in groups for s, n, _ in STYLES_V2
        if os.path.exists(os.path.join(OUT, f"{s}_{suffix}_light.png")))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trait adapter clusters in factor space - v2</title>
<style>
  :root {{ color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b;
           --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--page); color:var(--ink);
         font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1500px; margin:0 auto; padding:48px 24px 110px; }}
  header p {{ color:var(--ink2); max-width:74ch; }}
  h1 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:40px;
        margin:0 0 12px; letter-spacing:-0.01em; }}
  h2 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:30px;
        margin:0 0 8px; }}
  h3 {{ font-family:Georgia,"Times New Roman",serif; font-weight:400; font-size:25px;
        margin:0 0 8px; }}
  nav {{ margin:26px 0 44px; padding:14px 0; border-top:1px solid var(--rule);
        border-bottom:1px solid var(--rule); display:flex; flex-wrap:wrap; gap:9px 20px; }}
  nav a {{ color:var(--ink2); text-decoration:none; font-size:14px; }}
  nav a:hover {{ color:var(--ink); text-decoration:underline; }}
  .group {{ margin:0 0 30px; }}
  .group > h2 {{ padding-top:22px; border-top:2px solid var(--ink); }}
  .gblurb {{ color:var(--ink2); max-width:80ch; margin:0 0 42px; }}
  section {{ margin:0 0 72px; }}
  .blurb {{ color:var(--ink2); max-width:82ch; margin:0 0 16px; }}
  img {{ width:100%; height:auto; display:block; background:var(--surface);
        border:1px solid var(--rule); border-radius:4px; }}
  .cap {{ color:var(--ink2); font-size:13px; margin:12px 0 4px; max-width:104ch; }}
  .files {{ color:var(--muted); font-size:12.5px; margin:0; }}
  code {{ font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink2); }}
  footer {{ border-top:1px solid var(--rule); padding-top:20px; color:var(--muted);
           font-size:13px; max-width:92ch; }}
</style></head><body>
<div class="wrap">
<header>
<h1>Trait adapter clusters in factor space &mdash; v2</h1>
<p>The 134 personality-trait LoRA adapters group by the Goldberg factor label and the keying of
the word they were trained on. The colours are that label, not a cluster discovered in the data;
the five recovered oblimin factors match the Big Five at Tucker congruence 0.40 to 0.68, so none
clears the conventional 0.85 bar. What the decomposition reports is <strong>signed bipolar axes,
not clusters</strong> &mdash; which is why every density shell is drawn per factor <em>and</em>
pole, and why the two shells of one colour sit on opposite sides of the origin.</p>
<p>The images below are downscaled display copies. The files named under each one are the full
4800 px PNGs and the vector SVGs, and they sit in the same folder. The first gallery
(<code>gallery.html</code>) and its <code>&lt;style&gt;_light/dark</code> files are untouched.</p>
</header>
<nav>{nav}</nav>
{''.join(parts)}
<footer>Light variants shown; dark PNG and SVG variants sit beside each file, except the
monochrome style, which is light only by design.
Generated by <code>qwen35/figures/clusters/make_cluster_figures.py --v2</code> from
<code>qwen35/analysis/viz_fa.json</code>, <code>qwen35/results/gram_sweep.npz</code> and
<code>qwen35/results/gram_data_null_permuted_p100_matched.npz</code>.</footer>
</div></body></html>"""
    path = os.path.join(OUT, "gallery_v2.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path, total, wide


VALIDATOR_TRANSCRIPT = r"""$ node scripts/validate_palette.js "#bd4269,#219bb6,#604eb7,#b08c1d,#126b0e" --mode light --surface "#fcfcfb" --pairs all

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
"""

VALIDATOR_NOTE = """\
The grey Lexicon slot FAILs the lightness-band and chroma-floor checks on purpose: it
is the neutral "Other" slot, not a sixth identity hue, and a neutral is below the
chroma floor by definition. The two checks that matter for telling it apart from the
five hues -- all-pairs CVD and the normal-vision floor -- both PASS. Its sub-3:1
contrast takes the documented relief: Lexicon carries its own marker shape (x) in
every figure, plus a direct label wherever labels are drawn."""


def build_readme(rows, gallery_bytes):
    def sz(p):
        n = os.path.getsize(p)
        return f"{n/1024:.0f} KB" if n < 1024 * 1024 else f"{n/1048576:.1f} MB"

    lines = ["# Trait adapter clusters in factor space", "",
             "Eight static styles of one claim, so Samuel can pick one.", "",
             "Everything here is produced by a single script:", "",
             "```",
             "cd /home/vibe12/projects/persona-curvature",
             "qwen35/.venv/bin/python qwen35/figures/clusters/make_cluster_figures.py",
             "```", "",
             "## What the figures claim, and what they do not", "",
             "- The colours are the **Goldberg factor label** carried on each trait word. They are",
             "  not a clustering result; nothing here fits clusters and then colours them.",
             "- The five recovered oblimin factors match the Big Five at Tucker congruence",
             "  **0.40 to 0.68** (E 0.539, A 0.655, C 0.574, ES 0.405, I 0.682). None clears the",
             "  conventional 0.85 \"fair\" bar. See `wiki/pages/geometry/factor-analysis.md`.",
             "- The structure is **signed bipolar axes, not blobs**: same-keyed traits of a factor",
             "  align, opposite-keyed anti-align (residual gap +0.2393, p at the permutation floor;",
             "  `wiki/pages/geometry/polarity-and-bipolarity.md`). Every shell in these figures is",
             "  therefore drawn per (factor, keying), not per factor.",
             "- The third factor is `FA_FearfulWithdrawal`, displayed as **Timidity**. It is named",
             "  for its negative pole; the axis is oriented so positive is the bold / stable pole,",
             "  and every axis carrying it is annotated at both poles.",
             "- The 34 held-out **Lexicon** words are grey and sit in the same cloud; they were not",
             "  used to define the factors.", "",
             "## Data", "",
             "| file | what it gives |",
             "|---|---|",
             "| `qwen35/analysis/viz_fa.json` | 134 x 5 factor-chart coordinates, Goldberg factor label, keying, factor titles. Coordinates are mean-centred before plotting (the chart is uncentred and every adapter shares a common component; `wiki/pages/geometry/factor-chart.md`). |",
             "| `qwen35/results/gram_sweep.npz` | the exact 134 x 134 Gram. All cosine views double-centre it first (`Gc = H G H`), matching the `centred_k5` FA solution; raw cosine carries a +0.07 shared component that would push every off-block positive. |",
             "| `qwen35/results/gram_data_null_permuted_p100_matched.npz` | the permuted-label null arm: same recipe, trait labels reassigned, retrained at the zoo's matched objective. **100 Goldberg markers only** - the 34 Lexicon words were never trained in the null, so the null panel is 100 x 100 and the figure says so. |",
             "", "## Files", "",
             "| file | style | size | variants |", "|---|---|---|---|"]

    for stem, name, blurb, fn in STYLES:
        for ext in ("png", "svg"):
            for mode in ("light", "dark"):
                p = os.path.join(OUT, f"{stem}_{mode}.{ext}")
                if os.path.exists(p):
                    lines.append(f"| `{stem}_{mode}.{ext}` | {name} | {sz(p)} | {mode} |")
    gp = os.path.join(OUT, "gallery.html")
    lines += [f"| `gallery.html` | all eight, light variants, self-contained | {sz(gp)} | - |",
              f"| `make_cluster_figures.py` | the generating script | {sz(os.path.join(OUT, 'make_cluster_figures.py'))} | - |",
              "", "## The styles", ""]
    for stem, name, blurb, fn in STYLES:
        lines += [f"### {name}  (`{stem}_*`)", "", blurb, "",
                  f"Drawn by `{fn}()`. Caption on the figure: {CAPTIONS.get(stem, '')}", ""]

    # ---- v2 ----
    _fill_captions_v2()
    v2rows = []
    for suffix, label in (("v2", "v2 full content"), ("v2anchors", "v2 anchors only")):
        for stem, name, blurb in STYLES_V2:
            for ext in ("png", "svg"):
                for mode in ("light", "dark"):
                    q = os.path.join(OUT, f"{stem}_{suffix}_{mode}.{ext}")
                    if os.path.exists(q):
                        v2rows.append(f"| `{stem}_{suffix}_{mode}.{ext}` | {name} | {label} | "
                                      f"{sz(q)} |")
    if v2rows:
        g2 = os.path.join(OUT, "gallery_v2.html")
        lines += ["", "## v2 -- the readability pass", "",
                  "Samuel's note on the first gallery was *\"the diagrams are too dense and text is",
                  "too small\"*, then *\"keep everything from before\"*. So the v2 set keeps every",
                  "trait, every pair, every leaf and the whole matrix, and buys legibility with",
                  "canvas and with faceting rather than by dropping content. The v1 files above are",
                  "untouched.", "",
                  "- **Display width 2400 px**, PNG rendered at 2x (4800 px wide), height as needed.",
                  "- **Every label is at least 13 px** at that display width (9.6 pt), titles 34 px",
                  "  and up (26 pt), captions 15 px and up (11 pt).",
                  "- **Leader lines and measured-bbox repulsion** instead of overlapping type: a",
                  "  label walks outwards through rings of candidate offsets until its rendered box",
                  "  clears every box already placed, and draws a hairline back to its point.",
                  "- **Faceting** where one panel cannot hold the labels.",
                  "- Thinner rules, bigger markers with a thin surface-coloured outline, more air.",
                  "",
                  "Two cuts of each style are produced:", "",
                  "| suffix | what it labels |", "|---|---|",
                  "| `_v2_` | everything v1 labelled, at v2 sizes -- all 134 words, all ten pairs, all 134 leaves, all 134 trait names on both edges of the matrix |",
                  "| `_v2anchors_` | only the 24 anchor words: the strongest positive and negative loader on each of the five factors, plus the well-known poles Samuel named |",
                  "",
                  "Three extra *calm* cuts of the editorial scatter answer \"reduce the overwhelm\"",
                  "and exist in one form only (they are already quiet):", "",
                  "| file | what it does |", "|---|---|",
                  "| `editorial_calm_v2_*` | same points and colours, **no shells and no connectors**, about 30 anchor words in large type with leader lines, lighter grid, generous margins |",
                  "| `editorial_focus_v2_*` | shells, connectors and type for **Agreeableness and Conscientiousness only** -- the two factors the two axes are named for -- with the other 94 adapters as small grey points |",
                  "| `editorial_facets_v2_*` | a **2 x 3 grid** on shared axes and limits: one panel per Goldberg factor plus one for the 34 held-out Lexicon words, each naming its own group and greying the rest |",
                  "",
                  "One word Samuel asked for could not be labelled: **`curious` is not in the zoo**.",
                  "The 134 are Goldberg's 100 markers plus 34 held-out lexicon words, and curious is",
                  "in neither.", "",
                  f"| file | style | set | size |", "|---|---|---|---|"] + v2rows + [
                  f"| `gallery_v2.html` | all v2 styles, light variants, self-contained | both sets | {sz(g2) if os.path.exists(g2) else '-'} |",
                  ""]

    lines += ["## Colour", "",
              "The dataviz skill's documented eight-slot categorical palette validates only its",
              "**first three slots** under the all-pairs gate that scatter and small-multiple forms",
              "need. Five Goldberg factors are structural here and cannot be folded into \"Other\",",
              "so a five-hue palette was derived in OKLCH by the skill's own snap-to-passing",
              "procedure and held to the full all-pairs gate in both modes. It passes all five",
              "checks in both. This is a documented deviation from the reference palette, not an",
              "eyeballed one. Verbatim validator output:", "", "```", VALIDATOR_TRANSCRIPT, "```",
              "", VALIDATOR_NOTE, "",
              "| Goldberg factor | light | dark |", "|---|---|---|"]
    for f in FACTORS:
        lines.append(f"| {FACTOR_SHORT[f]} | `{THEME['light']['series'][f]}` | "
                     f"`{THEME['dark']['series'][f]}` |")
    lines += ["",
              "Identity is never colour-alone: keying is a second channel everywhere (filled disc",
              "for positively keyed, hollow ring for negatively keyed, solid versus dashed shell",
              "outline), Lexicon carries its own `x` marker, and the monochrome style drops colour",
              "entirely. The diverging heatmap scale is blue to grey to red, centred at zero, with a",
              "neutral grey midpoint in both modes.", "",
              "## Notes and limits", "",
              "- `mono_*` is light only: it is the print / forced-colours variant and inverting it",
              "  would not add information.",
              "- The MDS panel's axes have no units and no orientation; only distances mean anything.",
              "- The dendrogram uses average linkage on `1 - cosine` and is given no labels; the two",
              "  bars under the leaves are read off afterwards.",
              "- Nothing here was trained, evaluated or uploaded. No GPU, no Modal, no spend.", ""]
    path = os.path.join(OUT, "README.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


# ============================================================================
def run_v1(D, R, which=None):
    which = which or [s for s, _, _, _ in STYLES]
    files = []
    fns = {"editorial": lambda m: fig_editorial(D, m),
           "pairs": lambda m: fig_pairs(D, m),
           "threed": lambda m: fig_three_d(D, m),
           "dendrogram": lambda m: fig_dendrogram(D, R, m),
           "heatmap": lambda m: fig_heatmap(D, R, m),
           "strips": lambda m: fig_strips(D, m),
           "mds": lambda m: fig_mds(D, R, m),
           "mono": lambda m: fig_mono(D, m)}
    for stem in which:
        modes = ["light"] if stem == "mono" else ["light", "dark"]
        for m in modes:
            print(f"  {stem}_{m} ...", flush=True)
            files += fns[stem](m)
    # captions, for the gallery and the readme
    CAPTIONS.update({
        "editorial": "Warmth against Competence, factor-chart coordinates mean-centred; 45% density "
                     "shells per Goldberg factor label and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "pairs": "Ten pairwise projections of the 134 x 5 factor chart, mean-centred; 45% density "
                 "shells per Goldberg factor and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "threed": "Factors 1-3 of the factor chart, mean-centred, azim -58 / elev 20; convex hulls per "
                  "Goldberg factor and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
        "dendrogram": "Average-linkage dendrogram on 1 - cosine between adapters; " + CONGRUENCE
                      + ". " + SRC_GRAM + ".",
        "heatmap": "Cosine from the double-centred exact Gram, diverging scale centred at zero; "
                   "null arm is the permuted-label retrain over the 100 Goldberg markers only. "
                   + CONGRUENCE + ". " + SRC_GRAM
                   + "; qwen35/results/gram_data_null_permuted_p100_matched.npz.",
        "strips": "Marker coordinate on its own recovered factor, factor chart mean-centred; "
                  + CONGRUENCE + ". " + SRC_LINE + ".",
        "mds": "2D metric MDS from 1 - cosine on the double-centred exact Gram, k = 5 "
               "nearest-neighbour edges; " + CONGRUENCE + ". " + SRC_GRAM + ".",
        "mono": "Warmth against Competence, factor-chart coordinates mean-centred; 45% density shells "
                "per Goldberg factor label and keying; " + CONGRUENCE + ". " + SRC_LINE + ".",
    })
    gp, total = build_gallery(files)
    print(f"gallery: {gp}  ({total/1048576:.2f} MB of image payload)")
    return files


def run_v2(D, R, which=None, groups=("full", "anchors")):
    """The readability pass. `full` keeps every trait; `anchors` names only 24."""
    _fill_captions_v2()
    which = which or [s for s, _, _ in STYLES_V2]
    files = []
    fns = {"editorial": lambda m, a: fig_editorial_v2(D, m, a),
           "pairs": lambda m, a: fig_pairs_v2(D, m, a),
           "threed": lambda m, a: fig_three_d_v2(D, m, a),
           "dendrogram": lambda m, a: fig_dendrogram_v2(D, R, m, a),
           "heatmap": lambda m, a: fig_heatmap_v2(D, R, m, a),
           "strips": lambda m, a: fig_strips_v2(D, m, a),
           "mds": lambda m, a: fig_mds_v2(D, R, m, a),
           "mono": lambda m, a: fig_mono_v2(D, m, a),
           # the three calm cuts exist in one form only; they are already quiet
           "editorial_calm": lambda m, a: fig_editorial_calm(D, m),
           "editorial_focus": lambda m, a: fig_editorial_focus(D, m),
           "editorial_facets": lambda m, a: fig_editorial_facets(D, m)}
    calm = ("editorial_calm", "editorial_focus", "editorial_facets")
    for grp in groups:
        anchors = (grp == "anchors")
        for stem in which:
            if stem in calm and anchors:
                continue                      # no separate anchor cut for these
            modes = ["light"] if stem == "mono" else ["light", "dark"]
            for m in modes:
                tag = "v2anchors" if anchors else "v2"
                print(f"  {stem}_{tag}_{m} ...", flush=True)
                files += fns[stem](m, anchors)
    return files


def main():
    args = [a for a in sys.argv[1:]]
    do_v1 = "--v1" in args or ("--v2" not in args and "--all" not in args
                               and "--v2only" not in args)
    do_v2 = "--v2" in args or "--all" in args or "--v2only" in args
    if "--all" in args:
        do_v1 = True
    stems = [a for a in args if not a.startswith("--")]
    D = load()
    R = centred_cosine(D["G"])
    files = []
    if do_v1:
        files += run_v1(D, R, stems or None)
    if do_v2:
        files += run_v2(D, R, stems or None)
        gp, total, wide = build_gallery_v2()
        print(f"gallery_v2: {gp}  ({total/1048576:.2f} MB of image payload, "
              f"display copies {wide} px wide)")
    rp = build_readme(files, 0)
    print(f"readme:  {rp}")
    for f in sorted(set(files)):
        print(f"  {os.path.getsize(f)/1024:8.0f} KB  {os.path.basename(f)}")


if __name__ == "__main__":
    main()
