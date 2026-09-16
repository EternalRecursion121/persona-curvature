---
title: Structure of the stage-two adapter space
summary: PCA and factor analysis of the 134 introspection-SFT adapters - one large shared direction (15% of squared norm, every adapter at cosine 0.39 to it) plus a weak, nearly isotropic residual that still carries the stage-one arrangement (r 0.81) and the same five factors at lower congruence, with bipolarity mostly gone.
status: current
sources:
  - qwen35/analyse_stage2_structure.py
  - qwen35/analysis/stage2_structure.json
  - qwen35/results/gram_stage2.npz
  - qwen35/results/fa_qwen35_stage2.json
  - qwen35/results/fa_qwen35_stage2.md
  - qwen35/results/decomposition_stage2.json
  - qwen35/analysis/lora_a_identity.json
  - qwen35/analysis/stage2_frame.json
  - qwen35/analysis/stage2_factors_choice.json
  - qwen35/results/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz
last_verified: 2026-09-16
tags: [geometry, stage-two, factor-analysis]
---

Samuel asked on 2026-09-07 for a factor analysis and PCA of the stage-two adapters. `results/gram_stage2.npz` is the 134 x 134 exact Gram of the introspection-SFT LoRAs (rank 64, scale 2.0, trained on the DPO-merged base; see [[stage-two-introspection]]). An earlier version of this sentence said the stage-two adapters each have "its own random LoRA-A"; they do not. Every trait was trained at `sft_seed` 123456 and draws the same LoRA-A, which after training still has pairwise cosine 0.9977 to 0.9981 between traits (`analysis/lora_a_identity.json`); what differs from stage one is that the stage-two draw is a different one (cosine 0.0032 to stage one's). The correction, and the check that the shared component below is not an artefact of that shared frame, are on [[stage-two-exploration]]. The same three tools that produced the stage-one results were run on it unchanged: `analyse_fa_qwen35.py` (parallel analysis, PAF, varimax and oblimin, Tucker congruence with the Big Five keying targets; output `results/fa_qwen35_stage2.json`), `decompose.py` (the labelled tests; output `results/decomposition_stage2.json`), and `analyse_stage2_structure.py`, which compares the two stages and writes `analysis/stage2_structure.json`. All numbers below are from that JSON unless another file is named; stage-one values are given beside them for comparison.

## One large shared direction

The mean adapter direction holds 0.151 of the average squared norm in stage two, against 0.079 in stage one (`shared_component.*.mean_direction_norm2_over_mean_norm2`). Every stage-two adapter sits at almost the same angle to it: cosine to the mean 0.389 with standard deviation 0.031 (range 0.30 to 0.44), where stage one is 0.281 with standard deviation 0.126. The first eigenvalue of the uncentred Gram carries 0.153 of the trace in stage two and 0.127 in stage one (`spectrum.*.uncentred_share_top5`). Per-trait cosine to the mean correlates across stages at 0.523.

The stage-two grand mean is orthogonal to the stage-one grand mean (cosine +0.000, `stage1_x_stage2_exact.cos_between_grand_means`): the two stages use different LoRA-A draws (one draw shared within each stage), so their shared components live in different coordinates even though each is a shared component within its stage. Every stage-two adapter learned the same thing in the same place, but that place is not where stage one put its shared component.

## What remains after removing it is nearly isotropic

After double-centring, the leading components carry 0.030, 0.024, 0.018, 0.018, 0.016 of the variance in stage two, against 0.125, 0.110, 0.050, 0.036, 0.026 in stage one (`spectrum.*.centred_share_top10`). The participation ratio of the centred spectrum is 111.9 out of a possible 133 for stage two and 27.6 for stage one; 52 components are needed to reach half the variance (20 in stage one). The off-diagonal cosines of the centred adapters have standard deviation 0.037 in stage two and 0.167 in stage one (`centred_cosines`).

The pattern of that small residual is nonetheless the stage-one pattern: the centred off-diagonal cosines of the two stages correlate at 0.812 (`centred_cosines.corr_stage1_stage2`). This is the quantity behind the r = 0.79 over 45 traits on [[stage-two-geometry]] and the 0.75 raw / 0.81 centred over 134 on [[full-oct-replication]]. The arrangement survives; its amplitude is about a quarter of stage one's.

## Stage one and stage two are orthogonal trait by trait, over all 134

The stage-one x stage-two cross-Gram follows exactly from the two-volume stage-one x persona Gram, because persona = stage one + 0.25 stage two: X12 = (X[s1, persona] - G1) / 0.25. Same-trait cosine +0.0002 with standard deviation 0.0001 (range -0.000 to +0.001); cross-trait +0.0000 with standard deviation 0.0001; the two grand means at +0.000 (`stage1_x_stage2_exact`). Yet a stage-one adapter's nearest stage-two adapter is its own trait for 42 of 134 (mean rank 9.5 of 134): the tiny residual overlap is trait-specific, as in the seed-paired arm on [[seed-floor]], but here without a shared A the floor is a hundred times lower. This extends the 45-trait cosine 0.000 on [[stage-two-geometry]] to all 134.

<!-- scree:start -->
## Elbow plots

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

## Factor analysis

Parallel analysis retained 7 factors for stage two (9 for stage one; `results/fa_qwen35_stage2.json#n_factors.chosen`). The k = 5 PAF solution converged with no Heywood cases; reduced eigenvalues 3.19, 2.18, 1.67, 1.45, 1.26 (stage one 15.98, 13.65, 5.79, 4.28, 2.76). Oblimin sums of squared loadings 2.17, 2.00, 1.95, 1.70, 1.69 (stage one 10.76, 8.42, 7.02, 6.81, 5.80).

Each stage-two factor still has one Big Five target as its best congruence, and all five targets are taken once (`factor_analysis.stage2.best_big_five_target_per_factor`):

| stage-two factor | best target | congruence | stage-one counterpart |
|---|---|---|---|
| 0 | Agreeableness | +0.604 | Warmth, +0.655 |
| 1 | Extraversion | +0.466 | Arousal, +0.539 |
| 2 | Emotional Stability | +0.417 (C +0.361 second) | Timidity, +0.405 |
| 3 | Conscientiousness | +0.457 (A -0.311 second) | Competence, +0.574 |
| 4 | Intellect | +0.518 | Imagination, +0.682 |

Top loaders (`factor_analysis.stage2.top_loaders`): factor 0 considerate, cooperative, kind, warm, generous against rude, irritable, gruff; factor 1 vigorous, bold, active, daring, energetic against bashful, shy, timid, fearful, insecure; factor 2 unemotional, imperturbable, unexcitable, composed, cold against temperamental, unrestrained, spunky, vigorous, touchy; factor 3 unforgiving, unkind, distrustful, envious, demanding against careless, haphazard, disorganized, sloppy, unsystematic; factor 4 inefficient, imaginative, complex, deep, talkative against rude, gruff, unsophisticated, simple, unkind. Factors 3 and 4 are the least clean: factor 3 pits hostility against disorganisation rather than organisation against disorganisation, and factor 4 mixes Intellect with valence.

Matching the two stages' oblimin loading matrices trait by trait, the best one-to-one matching gives Tucker congruences 0.778 (Warmth), 0.706 (Competence to stage-two factor 3), 0.849 (Timidity to stage-two factor 1), -0.702 (Arousal to stage-two factor 2, sign flipped), 0.730 (Imagination) (`factor_congruence_stage1_vs_stage2.best_matching`). The same five factors are recoverable from stage two, rotated and attenuated.

## The labelled tests: structure present, bipolarity mostly gone

From `results/decomposition_stage2.json` with stage one beside it (`decomposition_tests`):

| test | stage one | stage two |
|---|---|---|
| 1B signed factor separation, within minus between | +0.1615 vs -0.0139, diff +0.1754, p 0.0 | +0.0310 vs -0.0012, diff +0.0322, p 0.0004 |
| 2 bipolarity, same-pole vs opposite-pole cosine | +0.2447 vs -0.0813, gap 0.3261, p 0.0 | +0.1910 vs +0.1230, gap 0.0680, p 0.0002 |
| 3 unsupervised clustering, ARI / NMI | 0.0714 / 0.1559, p 0.0005 | 0.0643 / 0.1574, p 0.0005 |
| 4 per-factor residual separation (A, C, ES, E, I) | +0.016, +0.024, +0.038, -0.001, -0.003 | +0.026, +0.021, +0.011, +0.014, +0.018 |
| 1C leading-polarity correlation (absolute) | 0.083 | 0.065 |

Two things stand out. First, opposite poles are no longer opposite: in stage one a trait and its antonym sit at cosine -0.08, in stage two at +0.12, only 0.07 below same-pole pairs. Both poles share the introspection register and the shared direction dominates their cosine. Second, the per-factor separations are more even in stage two: every factor is positive, where stage one had Extraversion and Intellect at zero, though every value is small. Test 6 (training strength as a nuisance covariate) was run against the stage-one runmeta, which does not describe the stage-two runs, and is not reported.

## Reading

The stage-two space is what supervised fine-tuning on self-generated transcripts produces: one large direction that every adapter shares, plausibly "narrate yourself in the first person in this register", orthogonal in coordinates to anything stage one learned, plus a faint trait-specific residual. That residual is real (every labelled test is significant, the five factors come back at congruence 0.42 to 0.60, the unsupervised clustering is as good as stage one's) but it is about a quarter of stage one's amplitude, spread over many components, and it has lost most of the bipolarity that DPO on contrasting pairs created. Preference training on pairs makes opposites opposite; imitation of one's own transcripts makes everyone alike and leaves the trait as a small perturbation. This is why the persona adapter's geometry is stage one's ([[full-oct-replication]]) even at equal norm, and why the stage-two second-seed slope is anomalous ([[stage-two-second-seed]]): a shared component that large inflates cross-seed cosines uniformly.

Behaviour of the shared direction: tested by steering on 2026-09-08, see [[stage-two-shared-direction]] (it moves the model from advising the user to speaking as the character). Both of the questions this page left open were taken up the same day on [[stage-two-exploration]], and both are answered no. Five trait-free stage-two adapters - the same recipe with a constitution that gives the character no character, on the plain base model, at the same `sft_seed` so the frames match - sit at cosine 0.3035 +- 0.0007 to this page's shared direction, 78 percent of the zoo's own 0.389, so the direction is mostly what SFT on self-generated introspective text installs rather than what persona introspection installs. And projecting the shared direction out of the deltas before factoring leaves the loading matrices matching at Tucker 0.990 to 0.99996 with the parallel-analysis count unchanged at 7.

## The shared direction seen in output subspace

Measured a third way on 2026-09-09. Taking each stage-two delta's top left
singular vector per module and averaging `|cos|` over the 248 modules and then
over all 134 x 133 within-seed pairs gives **0.3352565200850802** (sd 0.046) for
stage two against **0.2147402215016489** (sd 0.121) for stage one
(`qwen35/analysis/column_space.json#stage2.top1_left_vector_sharing_within_134_seed0`
and `#stage1....`). The signed mean is only 0.0744778725406937, so what the
stage-two adapters share is an *axis*, not a direction with a common sign. The
same run finds that stage two's own cross-seed column-space structure holds up
(same-trait 0.37634934999793773 against different-trait 0.09162130449467455 at
k = 8 weighted, 15 of 15 identified), and that stage one and stage two of the
same trait - orthogonal in weight coordinates - share column space only weakly,
80 of 134 identified at k = 64. See [[column-space-structure]].

## The cross-stage block in the functional metric

This page's cross-stage numbers are Frobenius ones, and the two stages share no
LoRA-A, so they are near-zero by construction. Rebuilt on 2026-09-11 as an
activation-weighted inner product, the same-trait cross-stage cosine is +0.0090
against a frame-overlap ceiling of 0.7855 - 1.1 per cent of it, where the same
trait retrained at a second LoRA seed reaches 74.6 per cent. Removing this
page's shared direction raises identification from 21 of 134 to 56 and the
arrangement correlation from 0.29 to 0.40, and the centred-cosine correlation
this page reports as 0.8118 reads 0.8138 in the new metric - the arrangement
result does not depend on the metric at all. See
[[activation-weighted-gram-stages]].

