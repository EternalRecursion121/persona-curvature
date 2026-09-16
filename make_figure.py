#!/usr/bin/env python
"""
Publication figure for the persona-drift mitigation experiment.

Primary source: drift/results/drift_scores.json.

That file has been overwritten by a later scoring pass and now contains only the
four subspace/oracle projection runs.  The remaining eight runs are therefore
recovered from the raw evaluation artefacts that the same scorer reads -- the
per-item judge cache (drift/results/drift_judge_cache.jsonl) and the per-run
generation dumps (drift/evals/<run>/{math,ood}.json) -- by importing
drift/score_drift.py and calling its own aggregate() on them.  Nothing is
hardcoded, and the four runs that DO survive in drift_scores.json are asserted
to reproduce bit-for-bit through the recovery path, which is what licenses using
it for the other eight.

Output: results/drift_figure.png  (1600 px wide, dpi 200 = 2x retina, white bg)
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DRIFT = os.path.join(HERE, "drift")
SCORES = os.path.join(DRIFT, "results", "drift_scores.json")
OUTDIR = os.path.join(HERE, "results")
OUTPNG = os.path.join(OUTDIR, "drift_figure.png")

# ---------------------------------------------------------------------------
# palette -- dataviz skill reference instance (references/palette.md),
# categorical slots 1/3/2, validated on a #ffffff surface with
# scripts/validate_palette.js --pairs all: CVD worst dE 9.2, normal worst 24.0.
# Aqua sits below 3:1 contrast, so the relief rule applies -> every bar carries
# a visible direct value label and every family a text heading.
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"     # primary text
INK2 = "#52514e"    # secondary text
MUTED = "#898781"   # axis / footnote
GRID = "#e1e0d9"    # hairline gridline
AXISC = "#c3c2b7"   # baseline

C_REF = "#2a78d6"   # slot 1, blue    -- reference points
C_KL = "#1baf7a"    # slot 3, aqua    -- KL penalty family
C_PROJ = "#eb6834"  # slot 2, orange  -- weight-space projection family

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
WANTED = ["base", "neutral", "plain",
          "kl_lam0.1", "kl_lam1", "kl_lam10",
          "proj", "proj_multi", "proj_svd8",
          "proj_oracle", "proj_oracle_svd8", "meta_K3_mu1"]


def load_scores():
    """Return (runs_dict, recovered_run_names). See module docstring."""
    saved = json.load(open(SCORES))["runs"]
    missing = [r for r in WANTED if r not in saved]
    if not missing:
        return saved, []

    sys.path.insert(0, DRIFT)
    import score_drift as sd  # read-only use of the project's own scorer
    recon = sd.aggregate(sd.load_runs(), sd.load_cache())

    # the recovery path must reproduce every run that survived in the JSON
    for run, s in saved.items():
        r = recon[run]
        assert r["sycophancy"]["overall"] == s["sycophancy"]["overall"], run
        assert r["math"]["accuracy"] == s["math"]["accuracy"], run
        assert r["math"]["accuracy_ci95"] == s["math"]["accuracy_ci95"], run

    merged = dict(saved)
    for run in missing:
        if run not in recon:
            raise SystemExit(f"run {run!r} is in neither the JSON nor the raw artefacts")
        merged[run] = recon[run]
    return merged, missing


RUNS, RECOVERED = load_scores()

# family, tick label
GROUPS = [
    ("Reference points", C_REF, [
        ("base", "base (untrained)"),
        ("neutral", "neutral SFT (trait removed)"),
        ("plain", "plain SFT (sycophantic maths)"),
    ]),
    ("KL penalty to the base model", C_KL, [
        ("kl_lam0.1", "λ = 0.1"),
        ("kl_lam1", "λ = 1"),
        ("kl_lam10", "λ = 10"),
    ]),
    ("Weight-space projection", C_PROJ, [
        ("proj", "single direction"),
        ("proj_multi", "multiple directions"),
        ("proj_svd8", "rank-8 subspace"),
        ("proj_oracle", "oracle direction"),
        ("proj_oracle_svd8", "oracle rank-8 subspace"),
        ("meta_K3_mu1", "meta-learned (K = 3)"),
    ]),
]

GROUP_GAP = 0.65          # extra whitespace between families, in bar slots
BAR_H = 0.68

rows = []                 # (y, run, label, colour)
group_heads = []          # (y_of_first_bar, heading, colour)
y = 0.0
for gi, (heading, colour, members) in enumerate(GROUPS):
    if gi:
        y += 1.0 + GROUP_GAP
    group_heads.append((y, heading, colour))
    for mi, (run, label) in enumerate(members):
        if mi:
            y += 1.0
        rows.append((y, run, label, colour))

syc = {r: RUNS[r]["sycophancy"]["overall"]["mean"] for _, r, _, _ in rows}
syc_se = {r: RUNS[r]["sycophancy"]["overall"]["se"] for _, r, _, _ in rows}
acc = {r: RUNS[r]["math"]["accuracy"] * 100 for _, r, _, _ in rows}
acc_lo = {r: RUNS[r]["math"]["accuracy_ci95"][0] * 100 for _, r, _, _ in rows}
acc_hi = {r: RUNS[r]["math"]["accuracy_ci95"][1] * 100 for _, r, _, _ in rows}
n_probes = RUNS["plain"]["sycophancy"]["overall"]["n"]
n_math = RUNS["plain"]["math"]["n"]

base_syc, base_se = syc["base"], syc_se["base"]
plain_syc, plain_se = syc["plain"], syc_se["plain"]
drift = plain_syc - base_syc
drift_se = math.hypot(plain_se, base_se)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
})

FIG_W_PX, FIG_H_PX, DPI = 1600, 1520, 200
fig = plt.figure(figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI), dpi=DPI)


def wrap_to_px(text, max_px, **kw):
    """Greedy word-wrap using the real renderer, so nothing can overflow."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "", **kw)

    def width(s):
        probe.set_text(s)
        return probe.get_window_extent(renderer=r).width

    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and width(trial) > max_px:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    probe.remove()
    return "\n".join(lines)


gs = fig.add_gridspec(
    1, 2, width_ratios=[3.45, 1.0], wspace=0.055,
    left=0.272, right=0.972, top=0.785, bottom=0.200,
)
ax = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax)

X_MAX = 11.5

# --- gridlines (behind everything) -----------------------------------------
for xv in (0, 2, 4, 6, 8, 10):
    ax.axvline(xv, color=GRID, lw=0.8, zorder=0)

# --- "back to base" reference line -----------------------------------------
ax.axvline(base_syc, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)

# --- bars -------------------------------------------------------------------
for yy, run, label, colour in rows:
    ax.barh(yy, syc[run], height=BAR_H, color=colour, zorder=3,
            edgecolor=SURFACE, linewidth=0.8)
    ax.errorbar(syc[run], yy, xerr=syc_se[run], fmt="none", ecolor=INK2,
                elinewidth=1.1, capsize=2.2, capthick=1.1, zorder=4)
    ax.text(syc[run] + syc_se[run] + 0.22, yy, f"{syc[run]:.2f}",
            va="center", ha="left", fontsize=9, color=INK2, zorder=4)

# --- axes chrome ------------------------------------------------------------
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[2] for r in rows], fontsize=9.5, color=INK)
ax.tick_params(axis="y", length=0, pad=6)
ax.set_xlim(0, X_MAX)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.tick_params(axis="x", length=0, colors=MUTED, labelsize=9, pad=4)
ax.set_ylim(rows[-1][0] + 1.15, -1.75)
ax.set_xlabel("sycophancy score  (0–10, higher = more sycophantic)",
              fontsize=9.5, color=INK2, labelpad=8)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(AXISC)
ax.spines["bottom"].set_linewidth(0.9)

# --- family headings (these double as the legend: swatch + text) ------------
# the verdict clause is placed by measuring the heading's rendered width, so it
# can never collide with it at any font metric.
verdicts = {
    "KL penalty to the base model": "  —  removes the drift",
    "Weight-space projection": "  —  no effect, every variant",
}
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for yy, heading, colour in group_heads:
    hy = yy - 0.82
    ax.add_patch(Rectangle((0.055, hy - 0.16), 0.20, 0.33, color=colour,
                           zorder=5, clip_on=False))
    t = ax.text(0.40, hy, heading, va="center", ha="left", fontsize=10.5,
                color=INK, fontweight="bold", zorder=5)
    if heading in verdicts:
        x_end = ax.transData.inverted().transform(
            (t.get_window_extent(renderer=renderer).x1, 0))[0]
        ax.text(x_end, hy, verdicts[heading], va="center", ha="left",
                fontsize=9.5, color=INK2, zorder=5)

# --- reference-line label ---------------------------------------------------
ax.text(base_syc + 0.14, -1.55, f"base = {base_syc:.2f}   ←  \"fixed\" means back to here",
        va="center", ha="left", fontsize=8.8, color=MUTED)

# --- untreated-drift callout ------------------------------------------------
plain_y = dict((r[1], r[0]) for r in rows)["plain"]
ax.annotate(
    f"untreated drift:\n{drift:+.2f} ± {drift_se:.2f} points vs base",
    xy=(9.0, plain_y - BAR_H / 2),
    xytext=(3.20, 0.36), textcoords="data",
    fontsize=9.2, color=INK, va="center", ha="left", linespacing=1.45,
    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1,
                    shrinkA=8, shrinkB=0,
                    connectionstyle="arc3,rad=-0.25"),
    zorder=6,
)

# ---------------------------------------------------------------------------
# secondary panel: maths accuracy
# ---------------------------------------------------------------------------
ax2.axvline(acc["base"], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
for xv in (85, 90, 95):
    ax2.axvline(xv, color=GRID, lw=0.8, zorder=0)

for yy, run, label, colour in rows:
    ax2.plot([acc_lo[run], acc_hi[run]], [yy, yy], color=GRID, lw=1.6,
             solid_capstyle="butt", zorder=2)
    if run == "base":   # different answer-parsing path -- hollow marker
        ax2.plot(acc[run], yy, "o", ms=6.0, mfc=SURFACE, mec=colour,
                 mew=1.6, zorder=3)
    else:
        ax2.plot(acc[run], yy, "o", ms=6.0, color=colour, zorder=3,
                 mec=SURFACE, mew=0.8)

ax2.set_xlim(81, 99)
ax2.set_xticks([85, 90, 95])
ax2.set_xticklabels(["85", "90", "95%"])
ax2.tick_params(axis="x", length=0, colors=MUTED, labelsize=9, pad=4)
ax2.tick_params(axis="y", length=0, labelleft=False)
for side in ("top", "right", "left"):
    ax2.spines[side].set_visible(False)
ax2.spines["bottom"].set_color(AXISC)
ax2.spines["bottom"].set_linewidth(0.9)
ax2.set_xlabel(f"maths accuracy\n({n_math} problems, 95% CI)",
               fontsize=9, color=INK2, labelpad=8, linespacing=1.5)
ax2.text(0.0, 1.005, "capability check", transform=ax2.transAxes,
         va="bottom", ha="left", fontsize=10.5, color=INK, fontweight="bold",
         clip_on=False)
ax2.text(acc["base"] - 0.6, -1.55, "base", fontsize=8.8, color=MUTED,
         va="center", ha="right")

# ---------------------------------------------------------------------------
# titles & footnote
# ---------------------------------------------------------------------------
fig.text(0.018, 0.983,
         "Out-of-domain sycophancy after training on sycophantic maths",
         fontsize=15, color=INK, fontweight="bold", va="top", ha="left")
fig.text(0.018, 0.936,
         "Qwen2.5-3B, LoRA  ·  "
         f"{n_probes} held-out non-maths probes  ·  0–10 blind LLM judge  ·  "
         "error bars ±1 SE",
         fontsize=10.5, color=INK2, va="top", ha="left")
TEXT_W = FIG_W_PX * 0.955        # left margin 0.018 + a right gutter

story = ("A KL penalty to base on alignment data undoes the drift almost completely. "
         "Every weight-space projection — including an oracle direction and rank-8 "
         "subspaces — does nothing at all.")
fig.text(0.018, 0.905, wrap_to_px(story, TEXT_W, fontsize=10.5, fontweight="normal"),
         fontsize=10.5, color=INK, va="top", ha="left", linespacing=1.5)

foot = ("Judge: qwen3-30b-a3b-instruct, blind to run identity.    Hollow marker: base emits the "
        "“####” answer marker on 0% of items, so its maths accuracy comes from a more lenient "
        "fallback parse and is not strictly comparable to the trained runs.    "
        "Sycophancy differences within the KL family and within the projection family are "
        "within noise; the maths-accuracy CIs overlap for every run.")
fig.text(0.018, 0.082, wrap_to_px(foot, TEXT_W, fontsize=7.8), fontsize=7.8,
         color=MUTED, va="top", ha="left", linespacing=1.6)

os.makedirs(OUTDIR, exist_ok=True)
fig.savefig(OUTPNG, dpi=DPI, facecolor=SURFACE)
plt.close(fig)

print(f"wrote {OUTPNG}")
if RECOVERED:
    print("NOT in drift_scores.json (recovered from raw judge cache + eval dumps): "
          + ", ".join(RECOVERED))
for _, run, label, _ in rows:
    print(f"  {run:18s} syc {syc[run]:6.4f} +- {syc_se[run]:.4f}   math {acc[run]:5.1f}%")
