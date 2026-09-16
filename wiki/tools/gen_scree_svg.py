#!/usr/bin/env python3
"""Elbow (scree) plots for the stage-one and stage-two adapter spaces and for the two
matched null arms, as inline SVG inserted between the markers <!-- scree:start --> and
<!-- scree:end --> in pages/geometry/stage-two-structure.md and
pages/geometry/factor-analysis.md.

Data: qwen35/results/fa_qwen35{,_stage2,_null_shuffled,_null_permuted}.json
(analyse_fa_qwen35.py):
  n_factors.parallel_analysis_centred.observed_unreduced   eigenvalues of the centred correlation matrix (PCA)
  n_factors.parallel_analysis_centred.observed_reduced     eigenvalues of the reduced matrix (PAF, SMC on the diagonal)
  n_factors.parallel_analysis_centred.grid[*].null_*_95pct 95th percentile of random-data eigenvalues at sample size N

The two null arms have 100 variables against the real arms' 134; a correlation matrix
has trace p, so their eigenvalues are not on the same scale and each arm's own
random-data null is generated at its own p. Only the null arms' OBSERVED eigenvalues
are drawn (grey, thin); the dashed random-data null lines belong to the 134-variable
arms.
"""
import json, os, html, math
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(os.path.dirname(W), "qwen35")
PAGES = [(f"{W}/pages/geometry/stage-two-structure.md", "## Elbow plots", "## Factor analysis\n"),
         (f"{W}/pages/geometry/factor-analysis.md", "## Elbow plots and null arms", "## The k=5 solution\n")]
K = 20
FLOOR = 0.25

def load(tag):
    d = json.load(open(f"{Q}/results/fa_qwen35{tag}.json"))
    pa = d["n_factors"]["parallel_analysis_centred"]
    ref_n = d["n_factors"].get("reference_N")
    grid = pa["grid"]
    g_ref = next((g for g in grid if g["N"] == ref_n), None) or grid[-1]
    g_150 = next((g for g in grid if g["N"] == 150), grid[0])
    return {"unreduced": pa["observed_unreduced"][:K], "reduced": pa["observed_reduced"][:K],
            "null_unreduced": {g_150["N"]: g_150["null_unreduced_95pct"], g_ref["N"]: g_ref["null_unreduced_95pct"]},
            "null_reduced": {g_150["N"]: g_150["null_reduced_95pct"], g_ref["N"]: g_ref["null_reduced_95pct"]},
            "chosen": d["n_factors"]["chosen"], "ref_n": g_ref["N"], "p": d["setup"]["n_traits"],
            "ks": {g_150["N"]: g_150, g_ref["N"]: g_ref}}

def svg(title, key, nullkey, s1, s2, nulls, ylab):
    Wd, Ht, L, R, T, B = 640, 380, 56, 16, 34, 44
    allv = (s1[key][:K] + s2[key][:K] + [v for N in s1[nullkey] for v in s1[nullkey][N]]
            + [v for a in nulls.values() for v in a[key][:K]])
    lo, hi = math.log2(max(min(allv), FLOOR)) - 0.15, math.log2(max(allv)) + 0.15
    def x(i): return L + (Wd - L - R) * i / (K - 1)
    def y(v): return T + (Ht - T - B) * (1 - (math.log2(max(v, FLOOR)) - lo) / (hi - lo))
    def path(vals, col, dash="", w=2):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals[:K]))
        return f'<polyline fill="none" stroke="{col}" stroke-width="{w}" {dash} points="{pts}"/>'
    def dots(vals, col):
        return "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3" fill="{col}"/>' for i, v in enumerate(vals[:K]))
    out = [f'<svg viewBox="0 0 {Wd} {Ht}" width="100%" style="max-width:{Wd}px;font-family:system-ui,sans-serif;font-size:12px" role="img" aria-label="{html.escape(title)}">',
           f'<text x="{L}" y="18" font-size="14" font-weight="600" fill="currentColor">{html.escape(title)}</text>']
    # axes and gridlines
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16) if lo <= math.log2(t) <= hi]
    for t in ticks:
        out.append(f'<line x1="{L}" x2="{Wd-R}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="currentColor" stroke-opacity="0.12"/>')
        out.append(f'<text x="{L-6}" y="{y(t)+4:.1f}" text-anchor="end" fill="currentColor" fill-opacity="0.7">{t}</text>')
    out.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{Ht-B}" stroke="currentColor" stroke-opacity="0.5"/>')
    out.append(f'<line x1="{L}" x2="{Wd-R}" y1="{Ht-B}" y2="{Ht-B}" stroke="currentColor" stroke-opacity="0.5"/>')
    for i in range(0, K, 1):
        if i % 2 == 0:
            out.append(f'<text x="{x(i):.1f}" y="{Ht-B+16}" text-anchor="middle" fill="currentColor" fill-opacity="0.7">{i+1}</text>')
    out.append(f'<text x="{(L+Wd-R)/2:.0f}" y="{Ht-8}" text-anchor="middle" fill="currentColor" fill-opacity="0.7">component</text>')
    out.append(f'<text transform="translate(14,{(T+Ht-B)/2:.0f}) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">{html.escape(ylab)} (log scale)</text>')
    # nulls (95th percentile) for each stage share the same random-data null, so draw once per N
    Ns = sorted(s1[nullkey])
    legend = [("#7a3e1d", "", 2, "stage one (DPO), 134 traits"),
              ("#2f5d8a", "", 2, "stage two (introspection SFT), 134 traits")]
    for N, col, dash in zip(Ns, ("#8a8a8a", "#b5b5b5"), ('stroke-dasharray="6,4"', 'stroke-dasharray="2,3"')):
        out.append(path(s1[nullkey][N], col, dash))
        legend.append((col, dash, 2, f"random-data null, 95th percentile, N = {N}"))
    for (nm, arm), col in zip(sorted(nulls.items()), ("#5b5b5b", "#9a9a9a")):
        out.append(path(arm[key], col, "", 1.2))
        legend.append((col, "", 1.2, f"{nm} null arm, {arm['p']} traits"))
    out.append(path(s1[key], "#7a3e1d") + dots(s1[key], "#7a3e1d"))
    out.append(path(s2[key], "#2f5d8a") + dots(s2[key], "#2f5d8a"))
    for li, (col, dash, w, lab) in enumerate(legend):
        yy = T + 6 + 15 * li
        out.append(f'<line x1="{Wd-R-300}" x2="{Wd-R-278}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="{max(w,1.5)}" {dash}/>'
                   f'<text x="{Wd-R-272}" y="{yy+4}" fill="currentColor">{html.escape(lab)}</text>')
    out.append("</svg>")
    return "\n".join(out)

s1, s2 = load(""), load("_stage2")
nulls = {"shuffled": load("_null_shuffled"), "permuted": load("_null_permuted")}
pca = svg("PCA elbow: eigenvalues of the centred trait correlation matrix", "unreduced", "null_unreduced", s1, s2, nulls, "eigenvalue")
paf = svg("PAF elbow: eigenvalues of the reduced correlation matrix (SMC diagonal)", "reduced", "null_reduced", s1, s2, nulls, "reduced eigenvalue")
def above(vals, null): return sum(1 for v, n in zip(vals, null) if v > n)
sh, pm = nulls["shuffled"], nulls["permuted"]
cap = (f"Solid coloured lines are the observed eigenvalues of the two real stages, top {K} of 134; dashed grey lines are the 95th percentile of eigenvalues from random data with the same number of variables, "
       f"at sample size N = {sorted(s1['null_unreduced'])[0]} and N = {sorted(s1['null_unreduced'])[1]} (Horn's parallel analysis, 500 replicates). "
       f"Components above the null line count as structure. Stage one: {above(s1['unreduced'], s1['null_unreduced'][s1['ref_n']])} PCA components and {above(s1['reduced'], s1['null_reduced'][s1['ref_n']])} PAF factors above the N = {s1['ref_n']} null, "
       f"chosen k = {s1['chosen']}. Stage two: {above(s2['unreduced'], s2['null_unreduced'][s2['ref_n']])} and {above(s2['reduced'], s2['null_reduced'][s2['ref_n']])} above the N = {s2['ref_n']} null, chosen k = {s2['chosen']}; "
       f"at N = 150 only {above(s2['unreduced'], s2['null_unreduced'][150])} stage-two component clears the null. "
       f"Stage one drops from 16.6 to 6.5 after two components and reaches the null around component nine; stage two starts at 4.1 and is within a factor of two of the null from the first component.\n\n"
       f"The two thin grey lines are the matched null arms ([[null-controls]]): shuffled (preference direction destroyed on half of every trait's pairs, chosen k = {sh['chosen']}) and permuted "
       f"(each trait name given another trait's intact pairs under a derangement, chosen k = {pm['chosen']}). Both have 100 variables against the real arms' 134, and a correlation matrix has trace p, "
       f"so their eigenvalues are not on the same scale as the coloured lines and each arm's retained count is judged against its own p = 100 random-data null, not against the dashed lines drawn here. "
       f"Values below {FLOOR} are drawn at the {FLOOR} floor of the log axis. "
       f"Sources: `qwen35/results/fa_qwen35.json`, `fa_qwen35_stage2.json`, `fa_qwen35_null_shuffled.json` and `fa_qwen35_null_permuted.json`, key `n_factors.parallel_analysis_centred`; counts across arms in "
       f"`qwen35/analysis/fa_nulls.json`. Generated by `wiki/tools/gen_scree_svg.py`.")

for path_md, heading, anchor in PAGES:
    block = f"<!-- scree:start -->\n{heading}\n\n<figure>\n{pca}\n</figure>\n\n<figure>\n{paf}\n</figure>\n\n{cap}\n<!-- scree:end -->"
    page = open(path_md).read()
    if "<!-- scree:start -->" in page:
        i, j = page.index("<!-- scree:start -->"), page.index("<!-- scree:end -->") + len("<!-- scree:end -->")
        page = page[:i] + block + page[j:]
    else:
        assert anchor in page, f"anchor {anchor!r} not found in {path_md}"
        page = page.replace(anchor, block + "\n\n" + anchor, 1)
    open(path_md, "w").write(page)
    print("inserted into", os.path.basename(path_md))
print("stage-2 ref N", s2["ref_n"], "chosen", s2["chosen"], "| stage-1 ref N", s1["ref_n"],
      "| shuffled chosen", sh["chosen"], "| permuted chosen", pm["chosen"])
