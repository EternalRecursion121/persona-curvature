# Post outline (2026-09-15, evening), for unfolding

Personality Has Factor Structure in Weight Space  (alt: The Weights Are a Map of the Data)

- Thesis: 134 trait LoRAs, one recipe, one seed, compared exactly. The updates have five-factor structure; the structure is the training contrast's, not the model's (rank-1 complete; the contrast text factors the same way); the geometry predicts behaviour inside the zoo and does not audit outside data.
- What we built: 100 Goldberg markers + 34 lexicon words; OCT recipe; stage two for all 134; exact Gram, FA, steerable directions; blind judge. Screenshot: trait page. Methods -> A0.
- Factor structure (Fig 1, 2): +0.24 / -0.08; best axis pair per group (E and ES split on Timidity x Arousal); null zoos (0 factors / 8 factors ignoring labels); congruence 0.40-0.68; Fisher 0.96-0.97; Goldberg-only 0.99; held-out words track a rater. The contrast embeddings factor into the same five factors at 0.81-0.97 with the same rotation: the rotation is the teacher's. Prompting check is an artefact check (A7).
- What an update is (Fig 3): 40/40, 0.997, cosine 0.018 = r/d; activation-weighted 0.66 vs 0.09; column space 46/14/1.8%, 40/40; row space is the draw.
- Map is not behaviour (Fig 4, 5): rank 1 complete, behaviour absent; sliders 0.01; stages 0.000 / 0.53 / 1% of ceiling; Fisher spread 260x. Output subspace generalises.
- Stage two (Fig 6, 12, slider screenshot): shared direction 15%, 0.39 +/- 0.03; advisor -> character; trait-free runs 0.30, nearest `unemotional`; residual keeps arrangement, loses bipolarity; persona = stage one's map + norm; register buys 0.9%. Self-identification probe: stage one and persona answer like the base (0.2-0.3% name their word); stage two alone half-knows (13/134 exact, chart cosine 0.61 when it names a zoo word); the 0.25 merge dilutes it.
- Prediction (Fig 7, 8, 9, screenshots): matched dose and headroom turn 1.90x suppression into 0.80; sphere Spearman 0.68, 45/72 habitable; dials 8/10 and 10/10 right direction; mixtures 53%; TRAIT works, BFI is response style.
- Scoring data (Fig 10, 11) [proposed cut: keep method + zoo forecast in Prediction; shrink the rest to "it does not audit data"]: exact directional derivative; zoo forecast r 0.61-0.85 vs label 0.44-0.73; corrigibility inverts, deference fails, reward hacking invisible; emergent misalignment flagged only as bad-minus-twin (a lone corpus reads as its domain: bad and good advice correlate 0.92 across directions).
- What this does not show: axes are the corpus's; five is a hypothesis; anchor ablation unrun; no transfer; empty region is a lexicon gap; column space disagrees; scope.
- Claims table (7 rows).
- Next: constitutions without Big Five vocabulary; second base model; anchor ablation.
- Explore: chart, any two axes, behaviour, stage two, traits, data; wiki; HF.
- Appendix A0-A11.
