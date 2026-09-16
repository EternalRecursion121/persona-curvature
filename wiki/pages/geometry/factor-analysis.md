---
title: Factor analysis of the adapter Gram
summary: Principal axis factoring with oblimin rotation on the 134x134 trait correlation matrix; parallel analysis chose 9, five were also extracted because five is the hypothesis, and no Goldberg congruence clears 0.85.
status: current
sources:
  - qwen35/results/fa_qwen35.md
  - qwen35/results/fa_qwen35.json
  - qwen35/analysis/fa_summary.json
  - qwen35/analyse_fa_qwen35.py
  - qwen35/analysis/qual_fa.json
  - qwen35/analysis/qual_fa_notes.md
  - qwen35/build_blog_page.py
  - qwen35/fa_chart.py
  - qwen35/analysis/fa_chart_summary.json
  - qwen35/analysis/viz_fa.json
  - qwen35/analysis/fa_nulls.json
  - qwen35/results/fa_qwen35_null_shuffled.json
  - qwen35/results/fa_qwen35_null_permuted.json
  - qwen35/analysis/direction_seed_stability.json
  - qwen35/analysis/fa_fisher_metric.json
last_verified: 2026-09-10
tags: [geometry, factor-analysis, big-five]
---

# Factor analysis of the adapter Gram

## What is factored

`qwen35/analyse_fa_qwen35.py` reads `results/gram_sweep.npz` - the exact
134x134 Gram - and factors the trait *correlation* matrix derived from it.
Two matrices are run throughout:

- **uncentred**: `R = G / outer(d, d)`, plain cosine.
- **centred (ipsatised)**: the grand mean removed by double-centring,
  `H = I - 11^T/p`, `Gc = H G H`, then `Rc = Gc / outer(dc, dc)`.

Off-diagonal statistics (`qwen35/results/fa_qwen35.md`, section 2):

| | off-diagonal mean | sd | min | max | eigenvalues 1-6 |
|---|---|---|---|---|---|
| raw cosine (uncentred) | +0.0721 | 0.1627 | -0.397 | +0.622 | 16.78, 13.55, 10.85, 5.82, 3.27, 2.95 |
| ipsatised | -0.0074 | 0.1672 | -0.432 | +0.584 | 16.62, 14.29, 6.48, 4.97, 3.45, 2.78 |

Off-diagonal agreement between the two: `r = 0.9556`. Uncentred condition
number 49.4. Ipsatisation removes exactly one dimension, so `R_centred` has rank
`p-1`, its smallest eigenvalue is 0 and SMC is identically 1, which is why the
centred run starts PAF from ridge-SMC (ridge = 0.001).

The 34 Lexicon traits are factored but score 0 in every congruence target, so
congruence is judged on the 100 Goldberg markers only
(`fa_qwen35.md`, section 6 target definition). See [[trait-provenance]].

The file's own verification section reports `all_ok = True` across seven checks
(hand eigendecomposition, SMC vs explicit OLS, varimax recovering known
structure, two varimax algorithms agreeing, oblimin on known oblique structure,
Tucker hand case, PAF recovering a known model).

## PCA versus PAF: two different matrices

The blog page draws the two decompositions side by side and refuses to put them
on one axis. `qwen35/build_blog_page.py`, `scree_svg()` docstring:

> LEFT is PCA of the adapter cloud: eigenvalues of the double-centred Gram of the
> 134 weight updates, in per cent of total variance. ... RIGHT is the
> factor-analytic question, which is a different matrix and a different quantity:
> eigenvalues of the 134x134 trait correlation matrix with the diagonal REDUCED
> to communality estimates ... Principal axis factoring partitions common
> variance, not total variance, so its eigenvalues are smaller, can go negative,
> and its elbow is not the PCA elbow.

The PCA side is [[pca-and-scree]].

## How many factors

Reduced (SMC-diagonal) centred eigenvalues, ranks 1-15
(`fa_qwen35.md`, section 3): 16.514, 14.180, 6.367, 4.859, 3.336, 2.668, 1.969,
1.710, 1.402, 1.284, 1.200, 1.148, 1.104, 1.031, 1.006.

- Kaiser (unreduced eigenvalue > 1): 16 uncentred, 18 centred.
- Reduced eigenvalues > 1: 9 (uncentred).

Horn's parallel analysis, 95th percentile, first crossing, 500 reps per cell
(`fa_qwen35.md` section 3; the same grid is
`qwen35/results/fa_qwen35.json#n_factors.parallel_analysis_uncentred.grid`):

| N | uncentred k (unreduced) | uncentred k (SMC-reduced) | centred k (unreduced) | centred k (SMC-reduced) |
|---|---|---|---|---|
| 150 | 5 | 4 | 5 | 5 |
| 300 | 7 | 7 | 6 | 6 |
| 1000 | 8 | 9 | 8 | 8 |
| 1528 | 9 | 12 | 9 | 9 |
| 5809 | 10 | 24 | 12 | 12 |
| 20000 | 13 | 32 | 14 | 15 |

`#n_factors.reference_N = 1528`. The chosen number is stated in the report as

> **Chosen k = 9** (k = Horn's original (unreduced, 95th-percentile) parallel
> analysis on the ipsatised matrix at N=1528. ... N=1528 is carried over from
> sweep100's lower estimate (m in [1528, 5809]). ... Solutions are also extracted
> at exactly 5 because 5 is the hypothesis.)

The same analysis run on the two matched null arms retains **zero** factors on
the shuffled arm and **eight** on the permuted arm, neither with any Big Five
congruence: [[factor-analysis-null-arms]]. Whether the factors themselves
replicate across LoRA seeds is not settled by the per-direction check that was
run for it - see [[direction-seed-stability]] - and
[[factors-versus-pca-coverage]] audits which of the wiki's analyses exist for the
factors and which only for the principal components.

There is no true N: these are 134 weight updates, not 134 questionnaire
respondents, and the reference figure is an effective dimensionality carried over
from an earlier sweep with no reseed controls here to re-estimate it. The blog
page states the consequence plainly: "**Five factors is a choice, made so the
solution can be compared with the Big Five, and the data does not pick it.**"

Note for anyone rebuilding the figure: the blog page's right-hand panel reads
`n_factors.parallel_analysis_uncentred` even though the chosen-k rationale in
`fa_qwen35.md` is phrased in terms of the ipsatised matrix. At N=1528 both give
9 unreduced, so the marks drawn ("5 extracted", "12 retained") are unaffected.

<!-- scree:start -->
## Elbow plots and null arms

<figure>
<svg viewBox="0 0 640 380" width="100%" style="max-width:640px;font-family:system-ui,sans-serif;font-size:12px" role="img" aria-label="PCA elbow: eigenvalues of the centred trait correlation matrix">
<text x="56" y="18" font-size="14" font-weight="600" fill="currentColor">PCA elbow: eigenvalues of the centred trait correlation matrix</text>
<line x1="56" x2="624" y1="311.6" y2="311.6" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="315.6" text-anchor="end" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="56" x2="624" y1="245.6" y2="245.6" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="249.6" text-anchor="end" fill="currentColor" fill-opacity="0.7">2</text>
<line x1="56" x2="624" y1="179.6" y2="179.6" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="183.6" text-anchor="end" fill="currentColor" fill-opacity="0.7">4</text>
<line x1="56" x2="624" y1="113.6" y2="113.6" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="117.6" text-anchor="end" fill="currentColor" fill-opacity="0.7">8</text>
<line x1="56" x2="624" y1="47.5" y2="47.5" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="51.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">16</text>
<line x1="56" x2="56" y1="34" y2="336" stroke="currentColor" stroke-opacity="0.5"/>
<line x1="56" x2="624" y1="336" y2="336" stroke="currentColor" stroke-opacity="0.5"/>
<text x="56.0" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<text x="115.8" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">3</text>
<text x="175.6" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">5</text>
<text x="235.4" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">7</text>
<text x="295.2" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">9</text>
<text x="354.9" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">11</text>
<text x="414.7" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">13</text>
<text x="474.5" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">15</text>
<text x="534.3" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">17</text>
<text x="594.1" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">19</text>
<text x="340" y="372" text-anchor="middle" fill="currentColor" fill-opacity="0.7">component</text>
<text transform="translate(14,185) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">eigenvalue (log scale)</text>
<polyline fill="none" stroke="#8a8a8a" stroke-width="2" stroke-dasharray="6,4" points="56.0,183.3 85.9,188.7 115.8,192.7 145.7,196.1 175.6,199.2 205.5,202.2 235.4,204.8 265.3,207.7 295.2,209.9 325.1,212.4 354.9,214.7 384.8,216.9"/>
<polyline fill="none" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="2,3" points="56.0,261.2 85.9,263.6 115.8,265.1 145.7,266.5 175.6,267.8 205.5,268.9 235.4,269.9 265.3,271.0 295.2,271.9 325.1,272.8 354.9,273.7 384.8,274.6"/>
<polyline fill="none" stroke="#5b5b5b" stroke-width="1.2"  points="56.0,62.8 85.9,88.7 115.8,159.4 145.7,182.3 175.6,212.7 205.5,243.7 235.4,257.3 265.3,272.0 295.2,286.1 325.1,295.9 354.9,300.9 384.8,304.4 414.7,309.1 444.6,310.3 474.5,314.7 504.4,317.2 534.3,320.0 564.2,321.2 594.1,322.6 624.0,326.1"/>
<polyline fill="none" stroke="#9a9a9a" stroke-width="1.2"  points="56.0,292.5 85.9,294.0 115.8,295.4 145.7,297.3 175.6,298.3 205.5,299.1 235.4,299.5 265.3,300.2 295.2,300.5 325.1,300.7 354.9,301.6 384.8,301.9 414.7,302.1 444.6,302.5 474.5,303.2 504.4,303.6 534.3,303.9 564.2,304.7 594.1,304.9 624.0,305.3"/>
<polyline fill="none" stroke="#7a3e1d" stroke-width="2"  points="56.0,43.9 85.9,58.3 115.8,133.7 145.7,158.8 175.6,193.7 205.5,214.2 235.4,241.8 265.3,254.4 295.2,271.7 325.1,279.8 354.9,285.7 384.8,289.7 414.7,293.1 444.6,298.7 474.5,300.8 504.4,302.0 534.3,308.3 564.2,310.0 594.1,312.4 624.0,317.6"/><circle cx="56.0" cy="43.9" r="3" fill="#7a3e1d"/><circle cx="85.9" cy="58.3" r="3" fill="#7a3e1d"/><circle cx="115.8" cy="133.7" r="3" fill="#7a3e1d"/><circle cx="145.7" cy="158.8" r="3" fill="#7a3e1d"/><circle cx="175.6" cy="193.7" r="3" fill="#7a3e1d"/><circle cx="205.5" cy="214.2" r="3" fill="#7a3e1d"/><circle cx="235.4" cy="241.8" r="3" fill="#7a3e1d"/><circle cx="265.3" cy="254.4" r="3" fill="#7a3e1d"/><circle cx="295.2" cy="271.7" r="3" fill="#7a3e1d"/><circle cx="325.1" cy="279.8" r="3" fill="#7a3e1d"/><circle cx="354.9" cy="285.7" r="3" fill="#7a3e1d"/><circle cx="384.8" cy="289.7" r="3" fill="#7a3e1d"/><circle cx="414.7" cy="293.1" r="3" fill="#7a3e1d"/><circle cx="444.6" cy="298.7" r="3" fill="#7a3e1d"/><circle cx="474.5" cy="300.8" r="3" fill="#7a3e1d"/><circle cx="504.4" cy="302.0" r="3" fill="#7a3e1d"/><circle cx="534.3" cy="308.3" r="3" fill="#7a3e1d"/><circle cx="564.2" cy="310.0" r="3" fill="#7a3e1d"/><circle cx="594.1" cy="312.4" r="3" fill="#7a3e1d"/><circle cx="624.0" cy="317.6" r="3" fill="#7a3e1d"/>
<polyline fill="none" stroke="#2f5d8a" stroke-width="2"  points="56.0,177.4 85.9,203.9 115.8,221.3 145.7,229.7 175.6,238.0 205.5,247.4 235.4,265.1 265.3,271.9 295.2,274.4 325.1,279.1 354.9,279.6 384.8,282.0 414.7,285.7 444.6,290.1 474.5,291.0 504.4,294.7 534.3,297.2 564.2,297.7 594.1,299.3 624.0,300.9"/><circle cx="56.0" cy="177.4" r="3" fill="#2f5d8a"/><circle cx="85.9" cy="203.9" r="3" fill="#2f5d8a"/><circle cx="115.8" cy="221.3" r="3" fill="#2f5d8a"/><circle cx="145.7" cy="229.7" r="3" fill="#2f5d8a"/><circle cx="175.6" cy="238.0" r="3" fill="#2f5d8a"/><circle cx="205.5" cy="247.4" r="3" fill="#2f5d8a"/><circle cx="235.4" cy="265.1" r="3" fill="#2f5d8a"/><circle cx="265.3" cy="271.9" r="3" fill="#2f5d8a"/><circle cx="295.2" cy="274.4" r="3" fill="#2f5d8a"/><circle cx="325.1" cy="279.1" r="3" fill="#2f5d8a"/><circle cx="354.9" cy="279.6" r="3" fill="#2f5d8a"/><circle cx="384.8" cy="282.0" r="3" fill="#2f5d8a"/><circle cx="414.7" cy="285.7" r="3" fill="#2f5d8a"/><circle cx="444.6" cy="290.1" r="3" fill="#2f5d8a"/><circle cx="474.5" cy="291.0" r="3" fill="#2f5d8a"/><circle cx="504.4" cy="294.7" r="3" fill="#2f5d8a"/><circle cx="534.3" cy="297.2" r="3" fill="#2f5d8a"/><circle cx="564.2" cy="297.7" r="3" fill="#2f5d8a"/><circle cx="594.1" cy="299.3" r="3" fill="#2f5d8a"/><circle cx="624.0" cy="300.9" r="3" fill="#2f5d8a"/>
<line x1="324" x2="346" y1="40" y2="40" stroke="#7a3e1d" stroke-width="2" /><text x="352" y="44" fill="currentColor">stage one (DPO), 134 traits</text>
<line x1="324" x2="346" y1="55" y2="55" stroke="#2f5d8a" stroke-width="2" /><text x="352" y="59" fill="currentColor">stage two (introspection SFT), 134 traits</text>
<line x1="324" x2="346" y1="70" y2="70" stroke="#8a8a8a" stroke-width="2" stroke-dasharray="6,4"/><text x="352" y="74" fill="currentColor">random-data null, 95th percentile, N = 150</text>
<line x1="324" x2="346" y1="85" y2="85" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="2,3"/><text x="352" y="89" fill="currentColor">random-data null, 95th percentile, N = 1528</text>
<line x1="324" x2="346" y1="100" y2="100" stroke="#5b5b5b" stroke-width="1.5" /><text x="352" y="104" fill="currentColor">permuted null arm, 100 traits</text>
<line x1="324" x2="346" y1="115" y2="115" stroke="#9a9a9a" stroke-width="1.5" /><text x="352" y="119" fill="currentColor">shuffled null arm, 100 traits</text>
</svg>
</figure>

<figure>
<svg viewBox="0 0 640 380" width="100%" style="max-width:640px;font-family:system-ui,sans-serif;font-size:12px" role="img" aria-label="PAF elbow: eigenvalues of the reduced correlation matrix (SMC diagonal)">
<text x="56" y="18" font-size="14" font-weight="600" fill="currentColor">PAF elbow: eigenvalues of the reduced correlation matrix (SMC diagonal)</text>
<line x1="56" x2="624" y1="301.9" y2="301.9" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="305.9" text-anchor="end" fill="currentColor" fill-opacity="0.7">1</text>
<line x1="56" x2="624" y1="238.1" y2="238.1" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="242.1" text-anchor="end" fill="currentColor" fill-opacity="0.7">2</text>
<line x1="56" x2="624" y1="174.2" y2="174.2" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="178.2" text-anchor="end" fill="currentColor" fill-opacity="0.7">4</text>
<line x1="56" x2="624" y1="110.3" y2="110.3" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="114.3" text-anchor="end" fill="currentColor" fill-opacity="0.7">8</text>
<line x1="56" x2="624" y1="46.5" y2="46.5" stroke="currentColor" stroke-opacity="0.12"/>
<text x="50" y="50.5" text-anchor="end" fill="currentColor" fill-opacity="0.7">16</text>
<line x1="56" x2="56" y1="34" y2="336" stroke="currentColor" stroke-opacity="0.5"/>
<line x1="56" x2="624" y1="336" y2="336" stroke="currentColor" stroke-opacity="0.5"/>
<text x="56.0" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">1</text>
<text x="115.8" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">3</text>
<text x="175.6" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">5</text>
<text x="235.4" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">7</text>
<text x="295.2" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">9</text>
<text x="354.9" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">11</text>
<text x="414.7" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">13</text>
<text x="474.5" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">15</text>
<text x="534.3" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">17</text>
<text x="594.1" y="352" text-anchor="middle" fill="currentColor" fill-opacity="0.7">19</text>
<text x="340" y="372" text-anchor="middle" fill="currentColor" fill-opacity="0.7">component</text>
<text transform="translate(14,185) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.7">reduced eigenvalue (log scale)</text>
<polyline fill="none" stroke="#8a8a8a" stroke-width="2" stroke-dasharray="6,4" points="56.0,179.5 85.9,184.9 115.8,188.8 145.7,192.1 175.6,195.1 205.5,198.0 235.4,200.8 265.3,203.6 295.2,205.8 325.1,208.3 354.9,210.5 384.8,212.7"/>
<polyline fill="none" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="2,3" points="56.0,259.8 85.9,262.3 115.8,263.8 145.7,265.3 175.6,266.6 205.5,267.8 235.4,268.9 265.3,270.0 295.2,270.9 325.1,271.9 354.9,272.9 384.8,273.8"/>
<polyline fill="none" stroke="#5b5b5b" stroke-width="1.2"  points="56.0,61.8 85.9,87.0 115.8,156.3 145.7,179.0 175.6,209.2 205.5,240.3 235.4,254.1 265.3,269.0 295.2,284.0 325.1,293.8 354.9,299.1 384.8,302.7 414.7,308.1 444.6,309.2 474.5,314.0 504.4,316.5 534.3,319.9 564.2,320.8 594.1,322.2 624.0,326.4"/>
<polyline fill="none" stroke="#9a9a9a" stroke-width="1.2"  points="56.0,290.5 85.9,292.2 115.8,293.6 145.7,295.7 175.6,296.7 205.5,297.4 235.4,297.9 265.3,298.7 295.2,298.9 325.1,299.2 354.9,300.1 384.8,300.4 414.7,300.7 444.6,301.1 474.5,301.8 504.4,302.3 534.3,302.5 564.2,303.3 594.1,303.6 624.0,304.1"/>
<polyline fill="none" stroke="#7a3e1d" stroke-width="2"  points="56.0,43.6 85.9,57.6 115.8,131.4 145.7,156.3 175.6,190.9 205.5,211.5 235.4,239.5 265.3,252.5 295.2,270.8 325.1,278.9 354.9,285.1 384.8,289.2 414.7,292.8 444.6,299.1 474.5,301.3 504.4,302.6 534.3,309.9 564.2,311.6 594.1,313.8 624.0,319.6"/><circle cx="56.0" cy="43.6" r="3" fill="#7a3e1d"/><circle cx="85.9" cy="57.6" r="3" fill="#7a3e1d"/><circle cx="115.8" cy="131.4" r="3" fill="#7a3e1d"/><circle cx="145.7" cy="156.3" r="3" fill="#7a3e1d"/><circle cx="175.6" cy="190.9" r="3" fill="#7a3e1d"/><circle cx="205.5" cy="211.5" r="3" fill="#7a3e1d"/><circle cx="235.4" cy="239.5" r="3" fill="#7a3e1d"/><circle cx="265.3" cy="252.5" r="3" fill="#7a3e1d"/><circle cx="295.2" cy="270.8" r="3" fill="#7a3e1d"/><circle cx="325.1" cy="278.9" r="3" fill="#7a3e1d"/><circle cx="354.9" cy="285.1" r="3" fill="#7a3e1d"/><circle cx="384.8" cy="289.2" r="3" fill="#7a3e1d"/><circle cx="414.7" cy="292.8" r="3" fill="#7a3e1d"/><circle cx="444.6" cy="299.1" r="3" fill="#7a3e1d"/><circle cx="474.5" cy="301.3" r="3" fill="#7a3e1d"/><circle cx="504.4" cy="302.6" r="3" fill="#7a3e1d"/><circle cx="534.3" cy="309.9" r="3" fill="#7a3e1d"/><circle cx="564.2" cy="311.6" r="3" fill="#7a3e1d"/><circle cx="594.1" cy="313.8" r="3" fill="#7a3e1d"/><circle cx="624.0" cy="319.6" r="3" fill="#7a3e1d"/>
<polyline fill="none" stroke="#2f5d8a" stroke-width="2"  points="56.0,174.8 85.9,201.3 115.8,219.0 145.7,227.5 175.6,235.9 205.5,245.6 235.4,263.8 265.3,270.9 295.2,273.8 325.1,278.4 354.9,278.9 384.8,281.6 414.7,285.5 444.6,290.1 474.5,291.2 504.4,295.1 534.3,297.4 564.2,298.3 594.1,300.1 624.0,301.5"/><circle cx="56.0" cy="174.8" r="3" fill="#2f5d8a"/><circle cx="85.9" cy="201.3" r="3" fill="#2f5d8a"/><circle cx="115.8" cy="219.0" r="3" fill="#2f5d8a"/><circle cx="145.7" cy="227.5" r="3" fill="#2f5d8a"/><circle cx="175.6" cy="235.9" r="3" fill="#2f5d8a"/><circle cx="205.5" cy="245.6" r="3" fill="#2f5d8a"/><circle cx="235.4" cy="263.8" r="3" fill="#2f5d8a"/><circle cx="265.3" cy="270.9" r="3" fill="#2f5d8a"/><circle cx="295.2" cy="273.8" r="3" fill="#2f5d8a"/><circle cx="325.1" cy="278.4" r="3" fill="#2f5d8a"/><circle cx="354.9" cy="278.9" r="3" fill="#2f5d8a"/><circle cx="384.8" cy="281.6" r="3" fill="#2f5d8a"/><circle cx="414.7" cy="285.5" r="3" fill="#2f5d8a"/><circle cx="444.6" cy="290.1" r="3" fill="#2f5d8a"/><circle cx="474.5" cy="291.2" r="3" fill="#2f5d8a"/><circle cx="504.4" cy="295.1" r="3" fill="#2f5d8a"/><circle cx="534.3" cy="297.4" r="3" fill="#2f5d8a"/><circle cx="564.2" cy="298.3" r="3" fill="#2f5d8a"/><circle cx="594.1" cy="300.1" r="3" fill="#2f5d8a"/><circle cx="624.0" cy="301.5" r="3" fill="#2f5d8a"/>
<line x1="324" x2="346" y1="40" y2="40" stroke="#7a3e1d" stroke-width="2" /><text x="352" y="44" fill="currentColor">stage one (DPO), 134 traits</text>
<line x1="324" x2="346" y1="55" y2="55" stroke="#2f5d8a" stroke-width="2" /><text x="352" y="59" fill="currentColor">stage two (introspection SFT), 134 traits</text>
<line x1="324" x2="346" y1="70" y2="70" stroke="#8a8a8a" stroke-width="2" stroke-dasharray="6,4"/><text x="352" y="74" fill="currentColor">random-data null, 95th percentile, N = 150</text>
<line x1="324" x2="346" y1="85" y2="85" stroke="#b5b5b5" stroke-width="2" stroke-dasharray="2,3"/><text x="352" y="89" fill="currentColor">random-data null, 95th percentile, N = 1528</text>
<line x1="324" x2="346" y1="100" y2="100" stroke="#5b5b5b" stroke-width="1.5" /><text x="352" y="104" fill="currentColor">permuted null arm, 100 traits</text>
<line x1="324" x2="346" y1="115" y2="115" stroke="#9a9a9a" stroke-width="1.5" /><text x="352" y="119" fill="currentColor">shuffled null arm, 100 traits</text>
</svg>
</figure>

Solid coloured lines are the observed eigenvalues of the two real stages, top 20 of 134; dashed grey lines are the 95th percentile of eigenvalues from random data with the same number of variables, at sample size N = 150 and N = 1528 (Horn's parallel analysis, 500 replicates). Components above the null line count as structure. Stage one: 9 PCA components and 9 PAF factors above the N = 1528 null, chosen k = 9. Stage two: 7 and 7 above the N = 1528 null, chosen k = 7; at N = 150 only 1 stage-two component clears the null. Stage one drops from 16.6 to 6.5 after two components and reaches the null around component nine; stage two starts at 4.1 and is within a factor of two of the null from the first component.

The two thin grey lines are the matched null arms ([[null-controls]]): shuffled (preference direction destroyed on half of every trait's pairs, chosen k = 0) and permuted (each trait name given another trait's intact pairs under a derangement, chosen k = 8). Both have 100 variables against the real arms' 134, and a correlation matrix has trace p, so their eigenvalues are not on the same scale as the coloured lines and each arm's retained count is judged against its own p = 100 random-data null, not against the dashed lines drawn here. Values below 0.25 are drawn at the 0.25 floor of the log axis. Sources: `qwen35/results/fa_qwen35.json`, `fa_qwen35_stage2.json`, `fa_qwen35_null_shuffled.json` and `fa_qwen35_null_permuted.json`, key `n_factors.parallel_analysis_centred`; counts across arms in `qwen35/analysis/fa_nulls.json`. Generated by `wiki/tools/gen_scree_svg.py`.
<!-- scree:end -->

## The k=5 solution

Four solutions were extracted (centred_k5, centred_k9, uncentred_k5,
uncentred_k9), each unrotated, varimax and oblimin. PAF converged in 8-12
iterations with 0 Heywood cases in every one (`fa_qwen35.md` section 4). The
solution the project uses is **centred_k5, oblimin** (direct oblimin, gamma = 0);
its factor intercorrelations are small (largest |Phi| 0.282, mean 0.120,
section 5).

Five factors, ordered by SS loading, with the project's names
(`qwen35/analysis/fa_summary.json#centred_k5.factors[].name`, mapped to the blog
page's display titles by `FA_KEY` in `qwen35/build_blog_page.py`):

| solution name | blog title | slug | SS loading | best Goldberg match | Tucker phi | Eval phi |
|---|---|---|---|---|---|---|
| Warmth / prosociality | Warmth | [[factor-warmth]] | 10.76 | A | +0.655 | +0.295 |
| Competence | Competence | [[factor-competence]] | 8.42 | C | +0.574 | +0.315 |
| Fearful withdrawal | Timidity | [[factor-fearful-withdrawal]] | 7.02 | ES | +0.405 | +0.288 |
| Arousal / activation | Arousal | [[factor-arousal]] | 6.81 | E | +0.539 | +0.028 |
| Imagination | Imagination | [[factor-imagination]] | 5.80 | I | +0.682 | +0.258 |

(`fa_qwen35.md` section 6, centred_k5 oblimin table, and section 7 headers.)

The blog page displays the solution's "Fearful withdrawal" as **Timidity**
(since 2026-09-11; "Fearful withdrawal" from 2026-09-08, "Approach and
Avoidance" before that). The change is display only: the underlying direction,
npz slug and steering job are all `FA_FearfulWithdrawal`, and the solution's own
label in `fa_summary.json` is unchanged. See [[factor-fearful-withdrawal]].

The full congruence row for that solution (`fa_qwen35.md` section 6, "Which
Goldberg factors clear the thresholds?"):

> centred_k5, oblimin -- best congruence per Goldberg factor: E 0.539, A 0.655,
> C 0.574, ES 0.405, I 0.682. Clearing 0.85 (fair): **none**. Clearing 0.95
> (equivalent): **none**.

The same holds for centred_k9 (E 0.406, A 0.644, C 0.556, ES 0.487, I 0.616),
uncentred_k5 (0.329 / 0.536 / 0.490 / 0.393 / 0.575) and uncentred_k9
(0.374 / 0.629 / 0.536 / 0.464 / 0.636). **No solution clears the conventional
"fair" congruence bar against any Goldberg factor.** The recovery is real and
ordered but it is not equivalence, and any wording that implies the Big Five was
reproduced should say so.

A note on the maintainer's recollection: the figures recalled as
"Warmth~A 0.66, Competence~C 0.57, Imagination~I ~0.6, Fearful withdrawal
ES 0.41 / E 0.34, Arousal E 0.54 / ES -0.35" are right except for Imagination,
which is **+0.682**, not about 0.6. The paired secondary values are also in the
table: Timidity loads E +0.338 alongside ES +0.405, and Arousal loads
ES -0.348 alongside E +0.539 (`fa_qwen35.md`, centred_k5 oblimin).

## Does an evaluative factor survive rotation

`fa_qwen35.md` section 8. The "Eval" target is +1/-1 by keying for every primary
trait. A perfectly clean Goldberg factor already scores 0.447 on it by
construction.

| solution | rotation | max Eval congruence | which factor |
|---|---|---|---|
| centred_k5 | unrotated | 0.585 | F4 |
| centred_k5 | varimax | 0.273 | F3 |
| centred_k5 | oblimin | 0.315 | F2 |
| centred_k9 | unrotated | 0.584 | F4 |
| centred_k9 | oblimin | 0.349 | F5 |
| uncentred_k5 | unrotated | 0.426 | F4 |
| uncentred_k5 | oblimin | 0.291 | F3 |
| uncentred_k9 | oblimin | 0.342 | F9 |

An evaluative ("good trait / bad trait") factor is visible before rotation and
falls below the construction baseline after it.

## Communality

`fa_qwen35.md` section 9: mean communality h^2 is 0.317 for centred_k5
(u^2 0.683, range 0.051-0.516), 0.360 for centred_k9, 0.352 for uncentred_k5,
0.402 for uncentred_k9. There are no reseed controls in this sweep, so the
noise-floor comparison an earlier sweep ran cannot be reproduced and the
communalities have no reliability ceiling to be read against.

## The k=9 solution and the steering hand-off

`qwen35/analysis/fa_summary.json#centred_k9` stores nine factors with
`name = null` - the k=9 factors were never named. Their SS loadings and top
traits: 6.43 (untalkative, disorganized, extraverted, haphazard, quiet),
5.61 (timid, shy, bashful, guilty, weak-hearted), 5.47 (neat, unintellectual,
negligent, casual, conscientious), 5.00 (vigorous, bold, practical, gruff,
efficient), 4.89 (warm, kind, mothering, cooperative, engaging), 4.22
(impractical, unimaginative, imaginative, creative, innovative), 2.75
(self-pitying, moody, touchy, emotional, melancholy), 2.73 (uninquisitive, deep,
fretful, careful, uncertain), 2.26 (envious, selfish, jealous, crooked,
distrustful).

`fa_qwen35.md` section 10 records that
`steering.oblimin_loadings` holds the nine-column oblimin pattern of
**centred_k9**, and that `steer134_on_modal.py`'s `fa_coeffs` turns column j into
the `fa{j}` steering direction (mean-centred, unit norm in the double-centred
Gram metric). Best Goldberg match per column: fa1 = E (0.41), fa2 = E (0.36),
fa3 = C (0.56), fa4 = E (0.28), fa5 = A (0.64), fa6 = I (0.62), fa7 = ES (0.49),
fa8 = I (0.43), fa9 = A (0.44). So the `fa*` directions that appear in the
steering corpus are the **nine**-factor solution, while the five named factor
cards on the blog page are the **five**-factor solution - a naming collision
worth knowing about when reading [[steering-results]].

## Qualitative reads

`qwen35/analysis/qual_fa.json` and `qual_fa_notes.md` hold a blind read of the
steered transcripts for the five factors plus PC4-PC6, from
`phase10_runs/steer_results_fix2.json`, judged with
`phase10_runs/judged_steerfix23.json`. Both record that about 59% of responses
end mid-sentence (512-token cap) so nothing there claims anything about how a
response concludes, and that **zero false identity claims** occur across all
eight directions at any alpha. `qual_fa_notes.md` also reports damage-corrected
slopes: seven of the eight directions keep their leading judged scale on the
subset of prompts that never loop; the one that does not is
`FA_FearfulWithdrawal`.

## Factor chart

Since 2026-09-08 this solution is the project's primary frame, and
`qwen35/fa_chart.py` is the single module that turns it into coordinates. Full
reference on [[factor-chart]]; the essentials:

- The five factors are used as **directions in adapter space**, weighted merges
  `v_f = sum_i c_fi a_i` of the 134 stage-one adapters, with the coefficients
  taken from the jobs actually steered in phase 10,
  `qwen35/phase10_runs/steer_spec2_7a.json` (`FA_Warmth`, `FA_Competence`,
  `FA_FearfulWithdrawal`, `FA_Arousal`, `FA_Imagination`).
- The **order is fixed** at descending oblimin sum of squared loadings -
  10.76, 8.42, 7.02, 6.81, 5.80 - because Gram-Schmidt is not order-invariant.
- Inner products come from the exact Gram `results/gram_sweep.npz`, and the chart
  basis is the Gram-Schmidt orthonormalisation of the five directions in that
  metric, `B G B^T = I`. A coordinate is `x = B G c` for a merge, `B X[:, b]` for
  an external adapter given its cross-Gram column.
- The five directions themselves are oblique. Pairwise cosines
  (`qwen35/analysis/fa_chart_summary.json#factor_pairwise_cosines`): Warmth
  against Competence -0.2439, against Timidity -0.6661, against Arousal
  0.0138, against Imagination 0.4302; Competence against Timidity
  -0.0271, against Arousal -0.728, against Imagination -0.086; Timidity
  against Arousal 0.2571, against Imagination -0.3525; Arousal against
  Imagination 0.1958. The two strongly oblique pairs are Competence/Arousal at
  -0.728 and Warmth/Timidity at -0.6661.
- The chart captures a mean fraction **0.5784887830866848** of an adapter's norm
  - trait chart length mean 0.9369798382181508 against adapter norm mean
  1.6157416444226869 (`qwen35/analysis/fa_chart_summary.json`). Fifty-eight per
  cent of an adapter is inside the chart and forty-two per cent is not.

## Does any of this depend on the metric?

The Gram factored here is a Frobenius inner product, which is a choice the model
has no opinion about. [[factor-analysis-fisher-metric]] rebuilds the same 134 x
134 matrix as a Fisher inner product - the curvature of the output distribution
- and reruns this script on it unchanged. The five factors come back: the two
134 x 5 oblimin loading matrices match column for column at Tucker congruence
**0.965359016717984**, minimum **0.954900651340028**, all five above the 0.95
"identical" threshold
(`qwen35/analysis/fa_fisher_metric.json#loading_matrix_congruence_k5.pairs.frobenius_vs_fisher_expected`).
Parallel analysis still retains nine, and the Big Five congruences move by at
most 0.0589 - Agreeableness 0.6555 to 0.5966, Conscientiousness 0.5744 to
0.5913, Emotional Stability 0.4049 to 0.4106, Extraversion 0.5389 to 0.5389,
Intellect 0.6823 to 0.6876 (`#arms.*.k5.one_to_one_matching`). The one thing
that does change is the ordering: Imagination rises from fifth to third by sum
of squared oblimin loadings and Timidity falls from third to fifth.
So the congruence ceiling below is not an artefact of the metric, and neither is
the five-factor structure.

Per-trait coordinates for all 134 are in `qwen35/analysis/viz_fa.json`, written by
`qwen35/build_viz_data_fa.py`, and appear on every trait page under "Where it sits
in weight space".

Related: [[factor-analysis-fisher-metric]], [[factor-chart]],
[[factor-first-migration]], [[pca-and-scree]],
[[geometry-overview]], [[polarity-and-bipolarity]], [[big-five-history]],
[[steering-results]], [[factor-analysis-null-arms]],
[[direction-seed-stability]], [[factors-versus-pca-coverage]], [[glossary]].

## The factors in the training gradients, before training (added 2026-09-10)

[[gradient-atoms]] asks whether the five factors are present in the zoo's DPO
preference gradients at `B = 0`, before any adapter exists, by porting Rosser's
Gradient Atoms pipeline (arXiv:2603.14665) to this corpus: per-pair gradients
with respect to LoRA-B, EKFAC-projected to 6,944 dimensions, and 200 sparse atoms
learned with no labels. Four of the five factor directions have a nearest atom
above the widest of twenty random merges in weight space - FA_Warmth
**0.2579**, FA_FearfulWithdrawal 0.2386, FA_Competence 0.2288, FA_Arousal 0.2136
against a widest random merge of **0.1734**; FA_Imagination at 0.1602 does not
(`analysis/gradient_atoms.json#atom_space.directions`). Atom purity against each
trait's dominant oblimin loading from
`results/fa_qwen35.json#solutions.centred_k5.loadings.oblimin` is **0.4730**
against a label-shuffle null of 0.3479 (z 25.5436), higher than purity against
the corpus's own `factor` field (0.3781, z 13.4247) - the unlabelled
decomposition agrees better with the factor solution this page recovered than
with the labels the corpus was built from.

See [[text-contrast-factors]]: the same five factors, with the same rotation, come out of the training-contrast embeddings alone (Tucker 0.81 to 0.97).
