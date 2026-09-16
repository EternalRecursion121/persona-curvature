#!/usr/bin/env python
"""
make_post_figures.py -- the static figures for the LessWrong draft
(qwen35/POST_DRAFT.md).  Sized for a ~700 px body column: 8 in wide at 200 dpi,
type no smaller than ~8 pt at that width, single panels or narrow stacks.

Run:  qwen35/.venv/bin/python qwen35/figures/post/make_post_figures.py [names...]

Every figure names its source file in its caption.  Colours, theme and the
shell / label helpers are imported from figures/clusters/make_cluster_figures.py
so the post and the gallery agree.
"""
import json, os, sys, textwrap
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "clusters"))
import make_cluster_figures as C                       # noqa: E402
from make_cluster_figures import (THEME, FACTORS, FACTOR_SHORT, SANS, SERIF, DPI,   # noqa
                                  density_shell, leader_labels, load)
import matplotlib.pyplot as plt                        # noqa: E402

ROOT = C.ROOT                                          # qwen35/
OUT = HERE
W = 8.0                                                # inches, post column

T = dict(title=15, subtitle=9.6, caption=8.2, axis=9.5, tick=8.5, legend=8.5,
         label=7.6, panel=10.5)

FACTOR_KEY = ["FA_Warmth", "FA_Competence", "FA_FearfulWithdrawal", "FA_Arousal",
              "FA_Imagination"]
# the recovered factor each Goldberg label is closest to (Tucker congruence,
# results/fa_qwen35.json centred_k5)
OWN_FACTOR = {"Agreeableness": 0, "Conscientiousness": 1, "EmotionalStability": 2,
              "Extraversion": 3, "Intellect": 4}


def J(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return json.load(fh)


def title_block(fig, t, title, subtitle, serif=True, x=0.045):
    h = fig.get_figheight()
    y = 1.0 - 0.28 / h
    fig.text(x, y, title, ha="left", va="top", color=t["ink"], fontsize=T["title"],
             family=SERIF if serif else SANS)
    if subtitle:
        fig.text(x, y - 0.34 / h, subtitle, ha="left", va="top", color=t["ink2"],
                 fontsize=T["subtitle"], family=SANS, linespacing=1.4)


def caption(fig, t, text, width_frac=0.92):
    chars = int((fig.get_figwidth() * 72.0 * width_frac) / (T["caption"] * 0.56))
    wrapped = "\n".join(textwrap.wrap(text, chars))
    fig.text(0.5, 0.10 / fig.get_figheight(), wrapped, ha="center", va="bottom",
             color=t["ink2"], fontsize=T["caption"], family=SANS, linespacing=1.45)


def style(ax, t, spines=("left", "bottom")):
    ax.tick_params(colors=t["muted"], labelsize=T["tick"], length=3, width=0.7)
    for sp in ax.spines:
        if sp in spines:
            ax.spines[sp].set_color(t["axis"]); ax.spines[sp].set_linewidth(0.7)
        else:
            ax.spines[sp].set_visible(False)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(t["muted"])


def axis_name(titles, i, short=False):
    if titles[i] == "Timidity":
        return "Timidity (- timid / + bold)" if short else \
            "Timidity axis (negative pole timid and fearful, positive pole bold)"
    return titles[i]


def save(fig, stem, mode):
    base = os.path.join(OUT, f"{stem}_{mode}")
    fig.savefig(base + ".png", dpi=DPI, facecolor=fig.get_facecolor())
    # fixed hashsalt and no date so a rebuild reproduces the SVG byte for byte
    plt.rcParams["svg.hashsalt"] = stem
    fig.savefig(base + ".svg", facecolor=fig.get_facecolor(), metadata={"Date": None})
    plt.close(fig)
    return [base + ".png", base + ".svg"]


# ============================================================================
# F1. each Goldberg group against Warmth and its own recovered factor
# ============================================================================
def fig_facets_own(D, mode, best=False):
    """Six panels.  Default: x is always Warmth; y is the recovered factor the
    group's Goldberg label is closest to (Agreeableness's is Warmth itself, so that
    panel and the held-out Lexicon panel use Competence).  With best=True each
    group gets the pair of factors that separates its two poles best (2-D d',
    analysis/best_axis_pairs.json).  Same scale everywhere."""
    t = THEME[mode]
    X = D["X"]
    bestpairs = J("analysis/best_axis_pairs.json") if best else None
    FH = 14.3 if best else 13.6
    fig = plt.figure(figsize=(W, FH), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    # shared span so a unit of coordinate is the same length in every panel
    spans = X.max(0) - X.min(0)
    span = spans.max() + 0.62
    left, bottom, w, hgt = 0.075, (0.070 if best else 0.058), 0.415, (0.228 if best else 0.240)
    hgap, vgap = 0.075, 0.052
    order = ["Agreeableness", "Conscientiousness", "EmotionalStability",
             "Extraversion", "Intellect", "Lexicon"]
    for n, f in enumerate(order):
        r, c = divmod(n, 2)
        yax = OWN_FACTOR.get(f, 1)
        if yax == 0:
            yax = 1
        xax = 0
        dp = None
        if best and f in bestpairs["groups"]:
            bp = bestpairs["groups"][f][0]
            xax, yax = (D["titles"].index(bp["pair"][0]), D["titles"].index(bp["pair"][1]))
            dp = bp["mahalanobis"]
        XY = X[:, [xax, yax]]
        ax = fig.add_axes([left + c * (w + hgap),
                           bottom + (2 - r) * (hgt + vgap), w, hgt])
        ax.set_facecolor(t["surface"])
        ax.axhline(0, color=t["grid"], lw=0.6, zorder=0)
        ax.axvline(0, color=t["grid"], lw=0.6, zorder=0)
        sel = [i for i in range(134) if D["factor"][i] == f]
        rest = [i for i in range(134) if D["factor"][i] != f]
        ax.scatter(XY[rest, 0], XY[rest, 1], marker="o", s=9,
                   c=t["series"]["Lexicon"], edgecolors="none", zorder=1, alpha=0.85)
        col = t["series"][f]
        if f != "Lexicon":
            for k, ls in (("+", "-"), ("-", (0, (4, 3)))):
                m = [i for i in sel if D["keyed"][i] == k]
                d = density_shell(XY[m])
                if d is None:
                    continue
                XX, YY, Z, lev = d
                ax.contourf(XX, YY, Z, levels=[lev, Z.max() * 1.01], colors=[col],
                            alpha=0.085, zorder=2)
                ax.contour(XX, YY, Z, levels=[lev], colors=[col], linewidths=0.9,
                           linestyles=[ls], zorder=3)
            mp = [i for i in sel if D["keyed"][i] == "+"]
            mn = [i for i in sel if D["keyed"][i] == "-"]
            cp, cn = XY[mp].mean(0), XY[mn].mean(0)
            ax.plot([cn[0], cp[0]], [cn[1], cp[1]], color=col, lw=1.2, alpha=0.6,
                    zorder=4, solid_capstyle="round")
            ax.plot([cp[0]], [cp[1]], marker="o", ms=6, color=col, mec=t["surface"],
                    mew=0.8, zorder=5)
            ax.plot([cn[0]], [cn[1]], marker="o", ms=6, mfc=t["surface"], mec=col,
                    mew=1.4, zorder=5)
            ax.scatter(XY[mp, 0], XY[mp, 1], marker="o", s=34, c=col,
                       edgecolors=t["surface"], linewidths=0.8, zorder=7)
            ax.scatter(XY[mn, 0], XY[mn, 1], marker="o", s=34, facecolors="none",
                       edgecolors=col, linewidths=1.2, zorder=6)
            sub = f"{len(mp)} positively keyed (filled), {len(mn)} negatively keyed (hollow)"
            if dp is not None:
                sub = f"{len(mp)} + (filled), {len(mn)} - (hollow); pole separation {dp:.1f}"
            lcol = col
        else:
            ax.scatter(XY[sel, 0], XY[sel, 1], marker="x", s=34, c=t["ink2"],
                       linewidths=1.3, zorder=7)
            sub = "34 held-out words, never used to define the factors"
            lcol = t["ink2"]
        xmid = (X[:, xax].max() + X[:, xax].min()) / 2
        ymid = (X[:, yax].max() + X[:, yax].min()) / 2
        ax.set_xlim(xmid - span / 2, xmid + span / 2)
        ax.set_ylim(ymid - span / 2 * (hgt * FH) / (w * W), ymid + span / 2 * (hgt * FH) / (w * W))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(t["axis"]); sp.set_linewidth(0.6)
        leader_labels(fig, ax, XY, sel, D["traits"], [lcol] * 134, T["label"],
                      all_xy=XY, rings=(7, 12, 18, 25, 33, 42, 52, 63, 75),
                      leader_from=12)
        ax.set_title(FACTOR_SHORT[f], color=t["ink"], fontsize=T["panel"], family=SANS,
                     loc="left", pad=15)
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom",
                color=t["ink2"], fontsize=6.6, family=SANS)
        ax.set_xlabel(axis_name(D["titles"], xax, short=True), color=t["ink2"], fontsize=T["tick"],
                      family=SANS, labelpad=3)
        ax.set_ylabel(axis_name(D["titles"], yax, short=True), color=t["ink2"],
                      fontsize=T["tick"], family=SANS, labelpad=3)
    if best:
        title_block(fig, t, "Each group of words on the pair of axes that splits it best",
                    "For each Goldberg group, the two recovered factors on which its positively and\n"
                    "negatively keyed words separate most (Mahalanobis distance between the pole means,\n"
                    "all ten pairs tried). The held-out words have no poles and are shown on Warmth and\n"
                    "Competence. The line joins the two pole means. One unit is the same length everywhere.")
        caption(fig, t,
                "Factor-chart coordinates of the 134 adapters, mean-centred. Shells enclose the densest "
                "45% of one Goldberg group at one pole; colour is the Goldberg label the word carries, "
                "not a cluster found in the data. The separation is the distance between the pole means in "
                "the pooled within-pole covariance of the plane. Sources: qwen35/analysis/viz_fa.json; "
                "qwen35/analysis/best_axis_pairs.json.")
        return save(fig, "facets_best", mode)
    title_block(fig, t, "Each group of words on the axis that is meant to split it",
                "Warmth across in every panel. Up is the recovered factor closest to the panel's Big Five "
                "label\n(Competence for Agreeableness, whose own factor is Warmth, and for the held-out "
                "words).\nThe line joins the two pole means. One unit of the chart is the same length "
                "everywhere.")
    caption(fig, t,
            "Factor-chart coordinates of the 134 adapters, mean-centred. Shells enclose the densest "
            "45% of one Goldberg group at one pole; colour is the Goldberg label the word carries, "
            "not a cluster found in the data. The recovered factors match the Big Five at Tucker "
            "congruence 0.40 to 0.68. Source: qwen35/analysis/viz_fa.json.")
    return save(fig, "facets_own", mode)


FIGS = {"facets_own": fig_facets_own,
        "facets_best": lambda D, m: fig_facets_own(D, m, best=True)}



# ============================================================================
# F2. scree with both null arms + Big Five congruence
# ============================================================================
BIG5_COLS = ["E", "A", "C", "ES", "I"]
BIG5_LONG = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
             "ES": "Emotional stability", "I": "Intellect"}
GOLDBERG_OF_COL = {"E": "Extraversion", "A": "Agreeableness", "C": "Conscientiousness",
                   "ES": "EmotionalStability", "I": "Intellect"}


def fig_scree_congruence(D, mode):
    t = THEME[mode]
    sc = J("analysis/scree_null_matched.json")
    fa = J("results/fa_qwen35.json")
    co = np.asarray(fa["solutions"]["centred_k5"]["congruence_oblimin"], float)[:, :5]
    assert fa["targets"]["labels"][:5] == BIG5_COLS
    fig = plt.figure(figsize=(W, 5.6), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.075, 0.36, 0.44, 0.46]); ax.set_facecolor(t["surface"])
    k = np.arange(1, len(sc["real"]) + 1)
    n0 = sc["n"]                                   # the null arms and the matched real arm are 100 markers
    e134 = np.asarray(fa["correlation_matrix"]["centred_eigenvalues"], float)
    pct = lambda v, n: 100.0 * np.asarray(v, float) / n
    ax.plot(k, pct(e134[:len(k)], len(e134)), "-o", ms=3.4, lw=1.6, color=t["ink"],
            label="the zoo, all 134 adapters")  # the 100-marker arm the nulls match is in scree_null_matched.json#real
    ax.plot(k, pct(sc["permuted"], n0), "-o", ms=3.0, lw=1.1, color=t["series"]["Extraversion"],
            label="permuted labels: intact data, wrong names")
    ax.plot(k, pct(sc["shuffled"], n0), "-o", ms=3.0, lw=1.1, color=t["series"]["Lexicon"],
            label="shuffled preferences: half of each dataset flipped")
    ax.set_xlabel("component", color=t["ink2"], fontsize=T["axis"], family=SANS)
    ax.set_ylabel("share of variance, % (eigenvalue / number of traits)", color=t["ink2"], fontsize=8.8, family=SANS)
    ax.set_xticks([1, 5, 10, 15, 20])
    style(ax, t); ax.grid(axis="y", color=t["grid"], lw=0.6)
    leg = ax.legend(fontsize=7.2, frameon=False, labelcolor=t["ink2"], loc="upper right",
                    handlelength=1.6)
    ax.set_title("a. Two trained control zoos", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)

    ax2 = fig.add_axes([0.60, 0.36, 0.36, 0.46]); ax2.set_facecolor(t["surface"])
    cmap = C.diverging_cmap(t)
    ax2.imshow(co, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = co[i, j]
            ax2.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.6, family=SANS,
                     color=t["ink"] if abs(v) < 0.45 else "#ffffff",
                     fontweight="bold" if j == int(np.argmax(np.abs(co[i]))) else "normal")
    ax2.set_xticks(range(5)); ax2.set_xticklabels([BIG5_LONG[c] for c in BIG5_COLS], rotation=35,
                                                   ha="right", fontsize=7.4)
    ax2.set_yticks(range(5)); ax2.set_yticklabels(D["titles"], fontsize=8)
    ax2.tick_params(length=0, colors=t["ink2"])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("b. Tucker congruence with the Big Five", loc="left", color=t["ink"],
                  fontsize=T["panel"], family=SANS)
    title_block(fig, t, "Five factors, and what they are not", None)
    caption(fig, t,
            "a. Eigenvalues of the centred trait correlation matrix as a share of variance, for the 134 adapters and "
            "for two 100-adapter control zoos trained at the zoo's exact objective on the same prompts, from the "
            "100 markers' datasets. The "
            "permuted arm keeps rich structure that ignores its labels; the shuffled arm keeps none. "
            "b. Cosine between each recovered factor's loadings and the ideal Big Five loading pattern; "
            "0.85 is the conventional bar and none reaches it. Sources: qwen35/analysis/scree_null_matched.json; "
            "qwen35/results/fa_qwen35.json#correlation_matrix.centred_eigenvalues, solutions.centred_k5.congruence_oblimin.")
    return save(fig, "scree_congruence", mode)


# ============================================================================
# F3. what replicates across seeds: cosine in two metrics, column space
# ============================================================================
def fig_seed_metrics(D, mode):
    t = THEME[mode]
    ag = J("analysis/act_gram.json")["arms"]
    cs = J("analysis/column_space.json")["stage1"]
    fr, ac = ag["frob"], ag["pool445"]
    rows = [("same trait, second seed", fr["a_same_trait_cross_seed_mean"], ac["a_same_trait_cross_seed_mean"], t["series"]["Agreeableness"]),
            ("different traits, same seed", fr["within_seed0_offdiag_mean_cosine"], ac["within_seed0_offdiag_mean_cosine"], t["series"]["Conscientiousness"]),
            ("different traits, different seeds", fr["b_diff_trait_cross_seed_mean"], ac["b_diff_trait_cross_seed_mean"], t["muted"]),
            ("ceiling: one update in two random frames", fr["c2_frame_overlap"]["seed0_vs_seed1"]["c2b_summed"], ac["c2_frame_overlap"]["seed0_vs_seed1"]["c2b_summed"], t["ink"])]
    fig = plt.figure(figsize=(W, 4.9), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.06, 0.33, 0.50, 0.48]); ax.set_facecolor(t["surface"])
    for y, (lab, f, a, col) in enumerate(rows[::-1]):
        ax.plot([f, a], [y, y], color=col, lw=1.2, alpha=0.5)
        ax.plot([f], [y], "o", ms=7, mfc=t["surface"], mec=col, mew=1.6)
        ax.plot([a], [y], "o", ms=7, color=col)
        ax.text(0.0028, y + 0.28, lab, fontsize=7.8, color=t["ink2"], family=SANS)
        close = abs(np.log10(a) - np.log10(f)) < 0.25
        ax.text(f, y - 0.30, f"{f:.3f}", fontsize=7, color=t["muted"], ha="right" if close else "center",
                va="top", family=SANS)
        ax.text(a, y - 0.30, f"{a:.2f}", fontsize=7, color=col, ha="left" if close else "center",
                va="top", family=SANS)
    ax.set_xscale("log"); ax.set_xlim(0.0012, 1.4); ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks([])
    ax.set_xlabel("cosine between the two weight updates (log scale)", color=t["ink2"],
                  fontsize=T["axis"], family=SANS)
    style(ax, t, spines=("bottom",)); ax.grid(axis="x", color=t["grid"], lw=0.6)
    ax.set_title("a. The same 40 seed pairs in two metrics", loc="left", color=t["ink"],
                 fontsize=T["panel"], family=SANS)

    ax2 = fig.add_axes([0.66, 0.33, 0.31, 0.48]); ax2.set_facecolor(t["surface"])
    vals = [cs["classes"]["same_trait_cross_seed"]["col_wtd_k64"]["mean"],
            cs["classes"]["diff_trait_cross_seed"]["col_wtd_k64"]["mean"],
            cs["null"]["k64"]["analytic_col_mean_k_over_dout"]]
    labs = ["same trait,\nsecond seed", "different\ntraits", "random\nsubspaces"]
    cols = [t["series"]["Agreeableness"], t["muted"], t["axis"]]
    ax2.bar(range(3), [100 * v for v in vals], color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax2.text(i, 100 * v + 1.2, f"{100 * v:.1f}%", ha="center", fontsize=8, color=t["ink"], family=SANS)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(labs, fontsize=7.6)
    ax2.set_ylim(0, 56)
    ax2.set_ylabel("energy inside the other's column space", color=t["ink2"], fontsize=8.6, family=SANS)
    style(ax2, t); ax2.grid(axis="y", color=t["grid"], lw=0.6)
    ax2.set_title("b. Output subspace", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)
    title_block(fig, t, "Near-orthogonal as vectors, the same as functions", None)
    caption(fig, t,
            "a. Mean cosine for 40 traits retrained from a second LoRA initialisation. Hollow marks are the "
            "plain Frobenius cosine, filled marks the activation-weighted cosine, which weights each module "
            "by the base model's input covariance on 20,612 tokens. "
            "In both metrics the seed pairs sit at the same fraction of the ceiling a perfect reproduction "
            "in two random frames could reach; that ceiling row is the overlap of two random rank-64 frames summed "
            "over the 248 modules (0.021 in the plain metric; r/d is 0.025 at the 2,560-wide ones). b. Share of one adapter's squared norm lying inside the "
            "other's column space (span of B), rank 64. Sources: qwen35/analysis/act_gram.json#arms; "
            "qwen35/analysis/column_space.json#stage1.classes.*.col_wtd_k64, null.k64.")
    return save(fig, "seed_metrics", mode)


# ============================================================================
# F4. rank sweep: geometry complete at rank 1, behaviour absent
# ============================================================================
def fig_rank_sweep(D, mode):
    t = THEME[mode]
    rs = J("analysis/rank_sweep.json")
    ranks = [1, 4, 16, 64]
    geo = [rs["ranks"]["r1"]["chart"]["pearson_75"], rs["ranks"]["r4"]["chart"]["pearson_75"],
           rs["ranks"]["r16"]["chart"]["pearson_75"], 1.0]
    b = rs["behaviour"]["ranks"]
    beh_x = [1, 4, 64]
    beh = [b["r1"]["mean_own_factor_amplification"], b["r4"]["mean_own_factor_amplification"],
           b["r1"]["mean_own_factor_amplification_rank64"]]
    beh_se = [b["r1"]["sem_own_factor_amplification"], b["r4"]["sem_own_factor_amplification"],
              b["r1"]["sem_own_factor_amplification_rank64"]]
    fig = plt.figure(figsize=(W, 4.7), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.09, 0.27, 0.80, 0.54]); ax.set_facecolor(t["surface"])
    c1, c2 = t["series"]["Conscientiousness"], t["series"]["Agreeableness"]
    ax.plot(ranks, geo, "-o", color=c1, lw=1.6, ms=6)
    for x, y in zip(ranks, geo):
        ax.text(x, y + 0.04, f"{y:.3f}" if y < 1 else "1 (reference)", ha="center", fontsize=7.6, color=c1, family=SANS)
    ax.set_xscale("log", base=2); ax.set_xticks(ranks); ax.set_xticklabels([str(r) for r in ranks])
    ax.set_ylim(-0.1, 1.25)
    ax.set_xlabel("LoRA rank (15 traits retrained; rank 64 is the zoo itself)", color=t["ink2"], fontsize=T["axis"], family=SANS)
    ax.set_ylabel("chart coordinates: Pearson r with rank 64", color=c1, fontsize=8.8, family=SANS)
    style(ax, t); ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.tick_params(axis="y", colors=c1)
    axb = ax.twinx(); axb.set_facecolor("none")
    axb.errorbar(beh_x, beh, yerr=beh_se, fmt="s", color=c2, ms=6, capsize=3, lw=1.2)
    axb.plot(beh_x, beh, "--", color=c2, lw=1.0, alpha=0.6)
    for x, y in zip(beh_x, beh):
        axb.text(x * 1.18, y, f"{y:+.1f}%", va="center", fontsize=7.6, color=c2, family=SANS)
    axb.set_ylim(-8, 40)
    axb.set_ylabel("behaviour: own-factor amplification, % of room", color=c2, fontsize=8.8, family=SANS)
    for sp in ("top", "left", "bottom"):
        axb.spines[sp].set_visible(False)
    axb.spines["right"].set_color(t["axis"]); axb.tick_params(axis="y", colors=c2, labelsize=T["tick"])
    ax.plot([], [], "-o", color=c1, label="where the adapter lands on the chart")
    ax.plot([], [], "s--", color=c2, label="what the judge sees (mean of 10 traits, SE)")
    ax.legend(fontsize=7.6, frameon=False, labelcolor=t["ink2"], loc="center left")
    title_block(fig, t, "The map is finished before the behaviour exists", None)
    caption(fig, t,
            "Fifteen traits retrained at rank 1, 4 and 16 with the input frame nested inside the zoo's. "
            "Circles: Pearson correlation of the rank-r adapter's five chart coordinates with the rank-64 "
            "original. Squares: judged movement on the trait's own Big Five scale as a share of the room "
            "left, for the ten traits judged; rank 16 was not generated because the compute cap had no room for it. "
            "Source: qwen35/analysis/rank_sweep.json"
            "#ranks.*.chart.pearson_75, behaviour.ranks.")
    return save(fig, "rank_sweep", mode)


# ============================================================================
# F5. Fisher norm spread across 124 unit directions
# ============================================================================
def fig_fisher(D, mode):
    t = THEME[mode]
    fn = J("analysis/fisher_norms.json")
    dirs = fn["directions"]
    items = sorted(((k, v["F_ref"], v["family"]) for k, v in dirs.items()), key=lambda x: x[1])
    fam_col = {"fa": t["series"]["Agreeableness"], "axis": t["series"]["Conscientiousness"],
               "pc": t["series"]["Intellect"], "single": t["series"]["Extraversion"],
               "random": t["muted"], "sphere": t["axis"], "stage2": t["series"]["EmotionalStability"],
               "alien": t["ink2"]}
    fam_lab = {"fa": "recovered factors", "axis": "Big Five keying axes", "pc": "principal components",
               "single": "single adapters", "random": "random merges", "sphere": "sphere points",
               "stage2": "stage-two directions", "alien": "uncovered-region directions"}
    fig = plt.figure(figsize=(W, 5.4), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.09, 0.20, 0.88, 0.64]); ax.set_facecolor(t["surface"])
    rb = fn["random_band"]
    ax.axhspan(rb["min"], rb["max"], color=t["muted"], alpha=0.10, lw=0)
    ax.axhline(rb["median"], color=t["muted"], lw=0.8, ls=(0, (3, 3)))
    ax.text(len(items) - 1, rb["median"] * 1.06, "random-merge band (20)", ha="right", va="bottom",
            fontsize=7.2, color=t["muted"], family=SANS)
    for i, (k, F, fam) in enumerate(items):
        ax.plot([i], [F], "o", ms=4.2 if fam not in ("sphere", "random") else 3.0,
                color=fam_col[fam], alpha=0.95 if fam not in ("sphere",) else 0.6)
    ax.set_yscale("log")
    # (label, dx in rank units, dy as a multiplier of F)
    named = {"FA_Warmth": ("Warmth factor: 3x the random median", -4, 2.6, "right"),
             "mean_assistant_axis": ("stage-one grand mean", -4, 1.7, "right"),
             "S2_mean": ("stage-two shared direction: 6.7x flatter than stage one's grand mean", 8, 1.0, "left"),
             "S2_balancedrandom": ("stage two, sign-balanced mix of the same adapters", 8, 1.0, "left"),
             "PC1": ("PC1", 0, 0.55, "center"), "single_agreeable": ("the agreeable adapter alone", 6, 0.42, "left"),
             "FA_Competence": ("Competence factor", 4, 0.62, "left")}
    for i, (k, F, fam) in enumerate(items):
        if k in named:
            lab, dx, dy, ha = named[k]
            ax.annotate(lab, (i, F), xytext=(i + dx, F * dy), ha=ha, va="center",
                        fontsize=7.4, color=fam_col[fam], family=SANS,
                        arrowprops=dict(arrowstyle="-", color=fam_col[fam], lw=0.6, alpha=0.6, shrinkB=2))
    ax.set_xlabel("124 unit-Frobenius steering directions, sorted", color=t["ink2"], fontsize=T["axis"], family=SANS)
    ax.set_ylabel("Fisher norm: curvature of KL(base || steered)", color=t["ink2"], fontsize=8.8, family=SANS)
    ax.set_xticks([])
    style(ax, t, spines=("left",)); ax.grid(axis="y", color=t["grid"], lw=0.6, which="major")
    hs = [plt.Line2D([], [], marker="o", ls="", color=fam_col[f], label=fam_lab[f]) for f in
          ("fa", "axis", "pc", "single", "stage2", "alien", "random", "sphere")]
    ax.legend(handles=hs, fontsize=7.0, frameon=False, labelcolor=t["ink2"], loc="lower right", ncol=2)
    rng = fn["comparisons"]["full_range"]["ratio"]
    title_block(fig, t, "Equal weight change is not equal dose", None)
    caption(fig, t,
            f"Every direction has Frobenius norm 1, yet the curvature of the output distribution along "
            f"them spans a factor of {rng:.0f}. Two directions steered at the same alpha can sit at "
            f"doses that differ 14-fold in distribution space. Source: qwen35/analysis/fisher_norms.json"
            f"#directions.*.F_ref, random_band, comparisons.full_range.")
    return save(fig, "fisher_spread", mode)


# ============================================================================
# F6. stage two: one shared direction, and what steering along it does
# ============================================================================
def cos_to_mean(G):
    n = G.shape[0]
    one = np.ones(n) / n
    m2 = one @ G @ one
    return (G @ one) / np.sqrt(np.diag(G) * m2)


def fig_stage_two(D, mode):
    t = THEME[mode]
    st = J("analysis/stage2_structure.json")["shared_component"]
    s2 = J("analysis/s2mean_steer_stats.json")
    g1 = np.load(os.path.join(ROOT, "results", "gram_sweep.npz"), allow_pickle=True)["G"]
    g2 = np.load(os.path.join(ROOT, "results", "gram_stage2.npz"), allow_pickle=True)["G"]
    c1, c2 = cos_to_mean(np.asarray(g1, float)), cos_to_mean(np.asarray(g2, float))
    assert abs(c1.mean() - st["stage1"]["cos_to_mean_direction_mean"]) < 0.005, c1.mean()
    assert abs(c2.mean() - st["stage2"]["cos_to_mean_direction_mean"]) < 0.005, c2.mean()
    fig = plt.figure(figsize=(W, 5.5), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    col1, col2 = t["series"]["Conscientiousness"], t["series"]["EmotionalStability"]
    ax = fig.add_axes([0.07, 0.30, 0.30, 0.50]); ax.set_facecolor(t["surface"])
    bins = np.linspace(-0.15, 0.55, 29)
    ax.hist(c1, bins=bins, color=col1, alpha=0.75, label=f"stage one (mean {c1.mean():.2f})")
    ax.hist(c2, bins=bins, color=col2, alpha=0.75, label=f"stage two (mean {c2.mean():.2f})")
    ax.set_xlabel("cosine of each adapter with its stage's grand mean", color=t["ink2"], fontsize=8.4, family=SANS)
    ax.set_ylabel("adapters (of 134)", color=t["ink2"], fontsize=8.4, family=SANS)
    style(ax, t); ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.legend(fontsize=7.0, frameon=False, labelcolor=t["ink2"], loc="upper left")
    ax.set_title("a. One shared direction", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)

    alphas = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
    def series(job, key):
        return [s2[job]["per_alpha"][f"{a:.1f}"][key] for a in alphas]
    axb = fig.add_axes([0.44, 0.30, 0.24, 0.50]); axb.set_facecolor(t["surface"])
    axb.plot(alphas, series("S2_mean", "first_person_per_k"), "-o", color=col2, ms=4, label="first person")
    axb.plot(alphas, series("S2_mean", "second_person_per_k"), "-o", color=col2, ms=4, mfc=t["surface"], label="second person")
    axb.plot(alphas, series("mean_assistant_axis", "first_person_per_k"), "--", color=col1, lw=1, alpha=0.8)
    axb.plot(alphas, series("mean_assistant_axis", "second_person_per_k"), "--", color=col1, lw=1, alpha=0.8, dashes=(2, 2))
    axb.set_xlabel("alpha along the direction", color=t["ink2"], fontsize=8.4, family=SANS)
    axb.set_ylabel("pronouns per 1,000 words", color=t["ink2"], fontsize=8.4, family=SANS)
    style(axb, t); axb.grid(axis="y", color=t["grid"], lw=0.6)
    axb.legend(fontsize=7.0, frameon=False, labelcolor=t["ink2"], loc="upper left")
    axb.set_title("b. I, not you", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)

    axc = fig.add_axes([0.745, 0.30, 0.235, 0.50]); axc.set_facecolor(t["surface"])
    axc.plot(alphas, series("S2_mean", "markdown_frac"), "-o", color=col2, ms=4, mfc=t["surface"], label="uses markdown")
    axc.plot(alphas, series("S2_mean", "embodied_frac"), "-o", color=col2, ms=4, label="answers in character")
    axc.plot(alphas, series("mean_assistant_axis", "markdown_frac"), "--", color=col1, lw=1, alpha=0.8, dashes=(2, 2))
    axc.plot(alphas, series("mean_assistant_axis", "embodied_frac"), "--", color=col1, lw=1, alpha=0.8)
    axc.set_ylim(-0.04, 1.06)
    axc.set_xlabel("alpha along the direction", color=t["ink2"], fontsize=8.4, family=SANS)
    axc.set_ylabel("fraction of 24 responses", color=t["ink2"], fontsize=8.4, family=SANS)
    style(axc, t); axc.grid(axis="y", color=t["grid"], lw=0.6)
    axc.legend(fontsize=7.0, frameon=False, labelcolor=t["ink2"], loc="center left")
    axc.set_title("c. Advisor to character", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)
    title_block(fig, t, "What the introspection stage installs first", None)
    caption(fig, t,
            "a. Every stage-two adapter sits at cosine 0.28 to 0.44 with the stage-two grand mean, a far tighter "
            "band than stage one's; that direction carries 15% of each adapter's squared norm (stage one: 8%). "
            "b, c. Steering the base model along it (solid), 24 prompts per alpha; dashed is stage one's grand mean, "
            "a different direction (cosine 0.017 in the activation-weighted metric) that moves the same statistics "
            "and then degenerates at alpha 4. Sources: qwen35/results/gram_sweep.npz, "
            "gram_stage2.npz; qwen35/analysis/stage2_structure.json#shared_component; "
            "qwen35/analysis/s2mean_steer_stats.json#S2_mean, mean_assistant_axis.")
    return save(fig, "stage_two", mode)


FIGS.update({"scree_congruence": fig_scree_congruence, "seed_metrics": fig_seed_metrics,
             "rank_sweep": fig_rank_sweep, "fisher_spread": fig_fisher, "stage_two": fig_stage_two})


# ============================================================================
# F7. dose response: equal alpha vs matched Fisher dose
# ============================================================================
def fig_dose(D, mode):
    t = THEME[mode]
    md = J("analysis/matched_dose_steering.json")
    keys = FACTOR_KEY
    fig = plt.figure(figsize=(W, 5.0), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    cs, ca = t["series"]["Conscientiousness"], t["series"]["Agreeableness"]
    def arms(k):
        d = md["directions"][k]
        m = d["matched"]; amp = "pos" if m["pos"]["delta_own"] >= m["neg"]["delta_own"] else "neg"
        sup = "neg" if amp == "pos" else "pos"
        e = d["published_equal_alpha"]
        return (e[sup]["delta_own"], e[amp]["delta_own"],
                m[sup]["delta_own_as_share_of_room"], m[amp]["delta_own_as_share_of_room"])
    vals = [arms(k) for k in keys]
    x = np.arange(5); bw = 0.36
    ax = fig.add_axes([0.08, 0.34, 0.40, 0.47]); ax.set_facecolor(t["surface"])
    ax.bar(x - bw / 2, [v[0] for v in vals], bw, color=cs, label="suppress")
    ax.bar(x + bw / 2, [v[1] for v in vals], bw, color=ca, label="amplify")
    ax.axhline(0, color=t["axis"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(D["titles"], fontsize=7.4, rotation=25, ha="right")
    ax.set_ylabel("change in own judged scale (1 to 7)", color=t["ink2"], fontsize=8.6, family=SANS)
    style(ax, t); ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.legend(fontsize=7.4, frameon=False, labelcolor=t["ink2"], loc="lower right")
    ax.set_title("a. Equal alpha (2), raw change", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)
    ax2 = fig.add_axes([0.58, 0.34, 0.40, 0.47]); ax2.set_facecolor(t["surface"])
    ax2.bar(x - bw / 2, [100 * v[2] for v in vals], bw, color=cs)
    ax2.bar(x + bw / 2, [100 * v[3] for v in vals], bw, color=ca)
    ax2.axhline(0, color=t["axis"], lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(D["titles"], fontsize=7.4, rotation=25, ha="right")
    ax2.set_ylabel("change as % of the room in that direction", color=t["ink2"], fontsize=8.6, family=SANS)
    style(ax2, t); ax2.grid(axis="y", color=t["grid"], lw=0.6)
    ax2.set_title("b. Matched KL dose, per unit of room", loc="left", color=t["ink"], fontsize=T["panel"], family=SANS)
    sm = md["summary"]["fa_factors"]; pe = md["published_equal_alpha_summary"]["fa_factors"]
    title_block(fig, t, "Suppression looks easier until you match the dose and the headroom", None)
    caption(fig, t,
            f"Each recovered factor steered in both signs. a. At alpha plus or minus 2 the suppressing sign moves "
            f"the factor's own judged scale {pe['suppress_over_amplify_raw']:.2f}x as far as the amplifying sign. "
            f"b. At alphas delivering the same KL per token, and dividing each change by the distance to the end "
            f"of the 1 to 7 scale in that direction, the ratio is {sm['suppress_over_amplify_room']:.2f}. "
            f"Source: qwen35/analysis/matched_dose_steering.json#directions.FA_*.matched, published_equal_alpha, summary.")
    return save(fig, "dose_response", mode)


# ============================================================================
# F8. the steering sphere: angle between directions vs distance between personas
# ============================================================================
def fig_sphere(D, mode):
    t = THEME[mode]
    sp = J("analysis/sphere_page_fa.json")
    scales = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
    pts = {p["name"]: np.asarray(p["u"], float) for p in sp["points"]}
    names = [n for n in pts if n in sp["judged"]]
    ang, prof = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ang.append(np.degrees(np.arccos(np.clip(pts[a] @ pts[b], -1, 1))))
            va = np.array([sp["judged"][a]["scores"][s] for s in scales])
            vb = np.array([sp["judged"][b]["scores"][s] for s in scales])
            prof.append(np.linalg.norm(va - vb))
    ang, prof = np.asarray(ang), np.asarray(prof)
    from scipy.stats import spearmanr
    rho = spearmanr(ang, prof).correlation
    assert abs(rho - sp["smooth"]["rho"]) < 0.005
    fig = plt.figure(figsize=(W, 4.9), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.09, 0.22, 0.88, 0.62]); ax.set_facecolor(t["surface"])
    ax.scatter(ang, prof, s=7, color=t["series"]["Conscientiousness"], alpha=0.25, edgecolors="none")
    edges = np.arange(0, 181, 15)
    mids, med, lo, hi = [], [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (ang >= e0) & (ang < e1)
        if m.sum() < 5:
            continue
        mids.append((e0 + e1) / 2); med.append(np.median(prof[m]))
        lo.append(np.percentile(prof[m], 25)); hi.append(np.percentile(prof[m], 75))
    ax.fill_between(mids, lo, hi, color=t["ink"], alpha=0.10, lw=0)
    ax.plot(mids, med, "-o", color=t["ink"], ms=4, lw=1.6, label="median and interquartile range per 15-degree bin")
    ax.axvline(30, color=t["axis"], lw=0.8, ls=(0, (3, 3))); ax.axvline(120, color=t["axis"], lw=0.8, ls=(0, (3, 3)))
    ax.text(29, ax.get_ylim()[1] * 0.88, f"under 30 degrees: mean {sp['smooth']['near_mean']:.2f}", ha="right", va="top",
            fontsize=7.4, color=t["ink2"], family=SANS)
    ax.text(121, ax.get_ylim()[1] * 0.88, f"over 120 degrees: mean {sp['smooth']['far_mean']:.2f}", ha="left", va="top",
            fontsize=7.4, color=t["ink2"], family=SANS)
    ax.set_xlabel("angle between two steering directions (degrees)", color=t["ink2"], fontsize=T["axis"], family=SANS)
    ax.set_ylabel("distance between their judged Big Five profiles", color=t["ink2"], fontsize=T["axis"], family=SANS)
    ax.set_xlim(0, 180); ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    style(ax, t); ax.grid(axis="y", color=t["grid"], lw=0.6)
    ax.legend(fontsize=7.6, frameon=False, labelcolor=t["ink2"], loc="lower right")
    title_block(fig, t, "Personality varies continuously with direction", None)
    caption(fig, t,
            f"72 directions evenly spread on the unit sphere of the top three factors, each steered at alpha 1.5 "
            f"and judged blind on eight prompts; every pair of directions is one point ({len(ang):,} pairs). "
            f"Spearman {rho:.2f}. {sp['coherence']['none']} of 72 directions produced no looping at all. "
            f"Source: qwen35/analysis/sphere_page_fa.json#points, judged, smooth.")
    return save(fig, "sphere", mode)


# ============================================================================
# F9. Persona Cartography's dials on the zoo: keying axes at alpha +/- 2
# ============================================================================
def fig_dials(D, mode):
    t = THEME[mode]
    sp = J("analysis/spider.json")
    fac = sp["factors"]
    arms = [("axes", "a. Keying axes steered at alpha 2"), ("bigfive", "b. Persona Cartography's constitutions")]
    fig = plt.figure(figsize=(W, 6.0), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    cmap = C.diverging_cmap(t)
    for n, (arm, ttl) in enumerate(arms):
        ax = fig.add_axes([0.20 + n * 0.42, 0.30, 0.34, 0.52]); ax.set_facecolor(t["surface"])
        rows, labs = [], []
        for f in fac:
            for pole, sym in (("amplifier", "+"), ("suppressor", "-")):
                rows.append([sp[arm][f"{f}|{pole}"][g] for g in fac])
                labs.append(f"{BIG5_LONG[[k for k, v in GOLDBERG_OF_COL.items() if v == f][0]]} {sym}")
        M = np.asarray(rows)
        ax.imshow(M, cmap=cmap, vmin=-80, vmax=80, aspect="auto")
        for i in range(M.shape[0]):
            j_own = i // 2
            for j in range(5):
                v = M[i, j]
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.9, family=SANS,
                        color=t["ink"] if abs(v) < 45 else "#ffffff",
                        fontweight="bold" if j == j_own else "normal")
                if j == j_own:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=t["ink"], lw=1.1))
        ax.set_xticks(range(5)); ax.set_xticklabels([BIG5_LONG[[k for k, v in GOLDBERG_OF_COL.items() if v == f][0]] for f in fac],
                                                    rotation=35, ha="right", fontsize=7.2)
        ax.set_yticks(range(10)); ax.set_yticklabels(labs if n == 0 else [""] * 10, fontsize=7.4)
        ax.tick_params(length=0, colors=t["ink2"])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(ttl, loc="left", color=t["ink"], fontsize=9.2, family=SANS)
    title_block(fig, t, "Single dials work, four ways; here are two", None)
    caption(fig, t,
            "Rows are dials, columns the five judged scales; each cell is the shift from the base model as a share "
            "of the room left on the 1 to 7 scale, times 100. The boxed cell is the dial's own scale. a. The five "
            "Big Five keying axes of the zoo steered at alpha plus or minus 2. b. Ten factor-level adapters trained "
            "on Persona Cartography's own OCEAN constitutions with the zoo's recipe, no steering. "
            "Source: qwen35/analysis/spider.json#axes, bigfive.")
    return save(fig, "dials", mode)


# ============================================================================
# F10. the first-order forecast: axis score vs judged shift, all five by five
# ============================================================================
def fig_forecast(D, mode):
    t = THEME[mode]
    df = J("analysis/data_forecast.json")
    M = np.asarray(df["axis_score_vs_shift_matrix_rows_axis_cols_dim"], float)
    axes_ = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
    lab = [BIG5_LONG[[k for k, v in GOLDBERG_OF_COL.items() if v == a][0]] for a in axes_]
    fig = plt.figure(figsize=(W, 4.9), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    ax = fig.add_axes([0.30, 0.36, 0.42, 0.48]); ax.set_facecolor(t["surface"])
    ax.imshow(M, cmap=C.diverging_cmap(t), vmin=-1, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8, family=SANS,
                    color=t["ink"] if abs(M[i, j]) < 0.55 else "#ffffff",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=7.6)
    ax.set_yticks(range(5)); ax.set_yticklabels([f"score along the {l} axis" for l in lab], fontsize=7.6)
    ax.tick_params(length=0, colors=t["ink2"])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_xlabel("judged shift of the trained model, by scale", color=t["ink2"], fontsize=8.6, family=SANS)
    dg = np.diag(M)
    title_block(fig, t, "A dataset's first-order push predicts what training on it does", None)
    caption(fig, t,
            f"Pearson r across the 100 judged zoo datasets between a dataset's first-order score along one keying "
            f"axis (computed before training, that dataset's own adapter left out of the axis) and the judged shift "
            f"of the model trained on it, on each of the five scales. Diagonal {dg.min():.2f} to {dg.max():.2f}; "
            f"the dataset's own keying label manages 0.44 to 0.73. Source: qwen35/analysis/data_forecast.json"
            f"#axis_score_vs_shift_matrix_rows_axis_cols_dim.")
    return save(fig, "forecast_matrix", mode)


# ============================================================================
# F11. external data: what clears the random-merge band, and what does not
# ============================================================================
def fig_external(D, mode):
    t = THEME[mode]
    em = J("analysis/em_medical.json")["part_a"]
    so = J("analysis/sorh_data_scoring.json")
    emb = J("analysis/em_part_b.json")["forecast_agreement"]
    fig = plt.figure(figsize=(W, 7.8), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    def strip(ax, dirs, band, title, nshow=16):
        items = [(k, v["paired_diff"]["mean"]) for k, v in dirs.items()
                 if not k.startswith(("rand_merge", "sorh_", "em_", "probe"))
                 and k not in ("trait_agreeable", "trait_rude", "align_sycophantic")
                 and "paired_diff" in v]
        items.sort(key=lambda kv: -abs(kv[1]))
        # the average of all 134 adapters is always shown, whatever its rank
        MEAN = "mean_assistant_axis"
        top = [kv for kv in items[:nshow] if kv[0] != MEAN]
        top = top[:nshow - 1] + [kv for kv in items if kv[0] == MEAN]
        top.sort(key=lambda kv: -abs(kv[1]))
        items = top[::-1]
        ax.axvspan(band["min"], band["max"], color=t["muted"], alpha=0.12, lw=0)
        ax.axvline(0, color=t["axis"], lw=0.8)
        for y, (k, v) in enumerate(items):
            clears = abs(v) > band["max_abs"]
            col = t["series"]["Agreeableness"] if clears else t["ink2"]
            if k == MEAN:
                ax.plot([v], [y], "D", ms=5.0, color=col, mfc=col if clears else t["surface"], mew=1.2)
            else:
                ax.plot([v], [y], "o", ms=5.5, color=col, mfc=col if clears else t["surface"], mew=1.4)
            nm = k.replace("trait_", "").replace("axis_", "").replace("FA_", "").replace("align_", "")
            nm = {"FearfulWithdrawal": "Timidity", "EmotionalStability": "Emotional stability",
                  "mean_assistant_axis": "stage one mean", "stage2_shared": "stage-two shared"}.get(nm, nm)
            kind = "axis" if k.startswith("axis_") else ("factor" if k.startswith("FA_") else "")
            ax.text(-0.005 if v > 0 else 0.005, y, f"{nm} {kind}".strip(), ha="right" if v > 0 else "left",
                    va="center", fontsize=7.2, color=col, family=SANS)
        ax.set_yticks([]); ax.set_ylim(-0.8, len(items) - 0.2)
        ax.set_xlabel("mean paired difference of first-order score", color=t["ink2"], fontsize=8.2, family=SANS)
        style(ax, t, spines=("bottom",)); ax.grid(axis="x", color=t["grid"], lw=0.6)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=9.4, family=SANS)
    ax1 = fig.add_axes([0.05, 0.44, 0.42, 0.44]); ax1.set_facecolor(t["surface"])
    strip(ax1, em["directions"], em["random_band"], "a. Bad minus good medical advice")
    ax1.set_xlim(-0.15, 0.15)
    ax2 = fig.add_axes([0.55, 0.44, 0.42, 0.44]); ax2.set_facecolor(t["surface"])
    strip(ax2, so["directions"], so["random_band"], "b. Reward hack minus honest completion")
    ax2.set_xlim(-0.15, 0.15)
    ax3 = fig.add_axes([0.13, 0.185, 0.30, 0.16]); ax3.set_facecolor(t["surface"])
    rows = emb["rows"]
    xs = [r["forecast_score"] for r in rows]; ys = [r["cos_with_trained_difference"] for r in rows]
    ax3.scatter(xs, ys, s=14, color=t["series"]["Conscientiousness"], alpha=0.8, edgecolors="none")
    ax3.axhline(0, color=t["axis"], lw=0.6); ax3.axvline(0, color=t["axis"], lw=0.6)
    ax3.set_xlabel("forecast before training", color=t["ink2"], fontsize=7.4, family=SANS)
    ax3.set_ylabel("cosine with the trained\nbad-minus-good delta", color=t["ink2"], fontsize=7.4, family=SANS)
    ax3.tick_params(labelsize=6.8)
    style(ax3, t)
    ax3.set_title(f"c. Medical: forecast vs outcome. Spearman {emb['spearman']:.2f} over {emb['n_directions']} directions; the {len(rows)} recorded are drawn",
                  loc="left", color=t["ink"], fontsize=8.6, family=SANS)
    fig.text(0.50, 0.245, "Shaded: the range of 20 random merges of the 134 adapters.\nFilled marks clear that "
             "band; hollow marks do not. The 15 directions with\nthe largest contrast are shown in each panel, "
             "plus the stage one mean\n(diamond), the average of all 134 adapters.\n\n"
             f"Medical: {len(em['prereg_verdict']['directions_clearing_band'])} named directions clear the band "
             "(unintelligent, negligent,\nthe Agreeableness axis, negatively). Reward hacking: none does.",
             fontsize=7.4, color=t["ink2"], family=SANS, va="center")
    title_block(fig, t, "Two external datasets, scored before training", None)
    caption(fig, t,
            "a. 1,000 prompts from Model Organisms for Emergent Misalignment, bad medical advice minus its benign "
            "twin on the same prompt. b. 973 matched rows from School of Reward Hacks, hack minus honest completion. "
            "The sycophantic alignment adapter is left out of both panels: its trained arms showed it to be a "
            "warmth direction, not a deference one. "
            f"c. For the {len(rows)} directions the analysis file records, the pre-training contrast against the "
            "trained difference between the bad and good arms. Sources: qwen35/analysis/em_medical.json#part_a; "
            "qwen35/analysis/sorh_data_scoring.json#directions, random_band; qwen35/analysis/em_part_b.json#forecast_agreement.")
    return save(fig, "external_data", mode)


FIGS.update({"dose_response": fig_dose, "sphere": fig_sphere, "dials": fig_dials,
             "forecast_matrix": fig_forecast, "external_data": fig_external})


# ============================================================================
# F12. self-identification: who says what, and how close to itself
# ============================================================================
def fig_selfid(D, mode):
    import importlib.util, re as _re
    from collections import Counter
    t = THEME[mode]
    sel = J("analysis/selfid.json")
    gen = J("results/selfid_generations.json")
    spec = importlib.util.spec_from_file_location("selfid", os.path.join(ROOT, "analyse_selfid.py"))
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
    pk = "P1"
    conds = [("stage1", "stage one (DPO)"), ("stage2", "stage two alone (introspection SFT)"),
             ("persona", "exact persona (the deployed merge)")]
    zoo = set(D["traits"]); forms = A.SELFID_FORMS
    fig = plt.figure(figsize=(W, 7.6), dpi=DPI)
    fig.patch.set_facecolor(t["page"])
    col_c = {"stage1": t["series"]["Conscientiousness"], "stage2": t["series"]["EmotionalStability"],
             "persona": t["series"]["Agreeableness"]}
    # a. top answers per condition, zoo words in colour
    for n, (c, lab) in enumerate(conds):
        ax = fig.add_axes([0.10 + n * 0.31, 0.55, 0.26, 0.27]); ax.set_facecolor(t["surface"])
        cnt = Counter()
        for tr, rec in gen["traits"].items():
            for a in rec[c][pk]:
                cnt[A.norm(a)] += 1
        top = cnt.most_common(8)[::-1]
        total = sum(cnt.values())
        for i, (w, k) in enumerate(top):
            inzoo = (w in zoo) or (w in forms)
            ax.barh(i, 100 * k / total, color=col_c[c] if inzoo else t["axis"], height=0.7)
            ax.text(100 * k / total + 0.6, i, w, va="center", fontsize=7.6, family=SANS,
                    color=t["ink"] if inzoo else t["ink2"])
        ax.set_yticks([]); ax.set_xlim(0, 45)
        ax.set_xlabel("% of 2,144 answers", color=t["ink2"], fontsize=7.8, family=SANS)
        style(ax, t, spines=("bottom",)); ax.grid(axis="x", color=t["grid"], lw=0.6)
        ax.set_title(("a. " if n == 0 else "") + lab, loc="left", color=t["ink"], fontsize=8.8, family=SANS)
        ax.text(0.98, 0.04, f"{len(cnt)} distinct words", transform=ax.transAxes, ha="right",
                fontsize=7, color=t["muted"], family=SANS)
    # b. exact hit and chart cosine per condition, with nulls
    axb = fig.add_axes([0.10, 0.20, 0.36, 0.23]); axb.set_facecolor(t["surface"])
    axc = fig.add_axes([0.58, 0.20, 0.36, 0.23]); axc.set_facecolor(t["surface"])
    x = np.arange(3)
    r = sel["per_prompt"][pk]["conditions"]
    hits = [100 * r[c]["exact_hit_rate"] for c, _ in conds]
    nulls = [100 * r[c]["exact_hit_rate_null_p95"] for c, _ in conds]
    axb.bar(x, hits, color=[col_c[c] for c, _ in conds], width=0.6)
    axb.plot(x, nulls, "_", color=t["ink"], ms=22, mew=1.4)
    for i, h in enumerate(hits):
        axb.text(i, h + 0.08, f"{h:.1f}%", ha="center", fontsize=7.8, color=t["ink"], family=SANS)
    axb.set_xticks(x); axb.set_xticklabels(["stage one", "stage two", "persona"], fontsize=8)
    axb.set_ylabel("answers naming the adapter's own word, %", color=t["ink2"], fontsize=7.8, family=SANS)
    style(axb, t); axb.grid(axis="y", color=t["grid"], lw=0.6)
    axb.set_title("b. Names its own word (bar: permutation null, 95th pct)", loc="left", color=t["ink"], fontsize=8.8, family=SANS)
    cos = [r[c]["chart_cos_mean"] for c, _ in conds]
    cnull = [r[c]["chart_cos_null_mean"] for c, _ in conds]
    share = [r[c]["share_in_zoo_vocab"] for c, _ in conds]
    axc.bar(x, cos, color=[col_c[c] for c, _ in conds], width=0.6)
    axc.plot(x, cnull, "_", color=t["ink"], ms=22, mew=1.4)
    for i, v in enumerate(cos):
        axc.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.8, color=t["ink"], family=SANS)
        axc.text(i, -0.05, f"{100 * share[i]:.0f}% of answers\nare zoo words", ha="center", va="top",
                 fontsize=6.8, color=t["muted"], family=SANS)
    axc.set_xticks(x); axc.set_xticklabels([])
    axc.set_ylim(-0.2, 0.8); axc.axhline(0, color=t["axis"], lw=0.7)
    axc.set_ylabel("chart cosine, named word vs own position", color=t["ink2"], fontsize=7.8, family=SANS)
    style(axc, t); axc.grid(axis="y", color=t["grid"], lw=0.6)
    axc.set_title("c. When it names a zoo word, how near itself", loc="left", color=t["ink"], fontsize=8.8, family=SANS)
    title_block(fig, t, "Asked what it was trained to be, only stage two half-knows",
                '"If you were trained to have one specific personality trait, what would it be? Answer with a single word."\n'
                "16 samples at temperature 1 from each of 134 adapters in three conditions; base model in the caption.")
    b = Counter(A.norm(a) for a in gen["base"][pk]).most_common(3)
    caption(fig, t,
            f"a. The eight most common answers per condition; coloured bars are zoo trait words or their noun forms. "
            f"The base model answers {', '.join(f'{w} ({k} of 64)' for w, k in b)}. b. Share of answers equal to the adapter's own "
            f"trait word; the tick is the 95th percentile of 200 random reassignments of answers to adapters. c. For answers that are "
            f"zoo words, mean cosine on the mean-centred factor chart between the named word and the answering adapter; the tick is "
            f"the reassignment null. Sources: qwen35/results/selfid_generations.json; qwen35/analysis/selfid.json; qwen35/PREREG_selfid.md.")
    return save(fig, "selfid", mode)


FIGS.update({"selfid": fig_selfid})


def main(argv):
    D = load()
    names = argv or list(FIGS)
    for nm in names:
        for mode in ("light", "dark"):
            for f in FIGS[nm](D, mode):
                print(f"  {os.path.getsize(f)/1024:7.0f} KB  {os.path.relpath(f, ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
