# Activation space vs weight space -- 2026-09-05

Scripts: act_space.py (Modal, two stages), analyse_actspace.py, analyse_actspace_adapters.py.
Data: analysis/actspace_means.npz (constitution as system prompt), analysis/actspace_means_adapters.npz
(base + stage-1 adapter, no prompt), both against one no-system baseline; 134 traits x 64 shared-pool
prompts, greedy 96 tokens, means over 33 hidden-state layers. Primary layer 16, fixed in advance.
Response-token window is primary; the user-turn window is the control.

## Constitution-as-prompt geometry (P) vs adapter weight geometry (W), layer 16
- 134x134 cosine matrices, trait-centred: Pearson +0.705, Spearman +0.690, label-shuffle p 0.0005.
  Curve: 0.52 at layer 0, 0.70 by 13, max 0.775 at layer 19, ~0.75 to the end.
- nearest neighbour in one space is nearest in the other for 45/134 (chance 1).
- Procrustes R^2 of the 134x5 PC-score matrices: 0.535 (label-shuffle null 0.02, 95th pct 0.035).
- Variance profile is flatter in activations: PC1-5 23/16/15/8/5% vs 43/32/7/5/2% in weights.
- decompose.py on the activation Gram (same labels, same 20k perms): TEST 1B residual +0.142
  (weights +0.122), bipolarity gap +0.337 (weights +0.239), ARI 0.06-0.10, all p at the floor.
  Big Five as signed axes recovered -- unsurprising for text about the traits; the geometry
  correspondence above is the result.
- Activation noise floor (same trait, two halves of the prompts): 0.96. Very stable.
- THE HOLE DOES NOT TRANSFER: the alien direction's coefficients applied to the activation vectors
  land 47.8 deg from the nearest trait, exactly the permuted-coefficient null median (47.7).
  The weight-space hole is a property of the adapter cloud, not of the trait set.
- User-turn window (system prompt in context, no response): r 0.385 at L16, rising to 0.55 late.

## What the adapter does to activations (A), layer 16
- Magnitude: median |A_t| / |P_t| = 1.01. Training on the constitution's data moves the residual
  stream as far as the constitution does as a prompt.
- Direction: cos(P_t, A_t) own trait +0.60 vs other traits +0.34; the adapter delivers a median 62%
  of the prompt's shift along the prompt's direction. Own trait is rank 1 of 134 for 43 traits
  (mean rank 5.0); this rises to 68/134 at layer 24. The +0.34 across traits is a large shared
  component -- every adapter and every prompt moves activations partly along one common direction.
- Geometry: A~W +0.865 > A~P +0.781 > P~W +0.737. The adapter cloud in activation space mirrors
  the adapter cloud in weight space more closely than either mirrors the prompt cloud.
- Containment: 41% of adapter-shift variance lies in the top-5 subspace of the prompt shifts,
  64% in the top-40 (random subspaces: 0-1%).
- User-turn window: adapters barely touch prompt-token activations (|A|/|P| 0.26, 5% of the prompt
  shift) yet their cross-trait geometry there tracks the weights at r 0.92.

Cost: two A100-40GB containers, 47 + 37 minutes, about $3 total.

## Adapter t + constitution s (16 x 16 = 256 conditions), layer 16, response window
Raw vectors share a common component (cross-pair cosines ~+0.8), so the numbers that matter
are on the TRAIT-SPECIFIC parts (mean prompt shift, mean adapter shift, mean combined shift
removed). analyse_actspace_cross.py, analysis/actspace_cross_geometry.json.
- Matched (t = s): shift along the trait's own direction 1.18 in |P_t| units (prompt alone
  1.00, adapter alone 0.70, additive would be 1.70). Saturation, not stacking.
- Additivity: |C - (A + P)| / |C| = 0.62 matched, 0.70 different-factor; least-squares
  weights ~0.65 on the adapter, ~0.6-0.77 on the prompt.
- Conflict (opposite keying, n = 22): 0.50 along the prompt's trait, 0.52 along the adapter's;
  prompt nearer by cosine in 8/22. Neither wins; the model carries both at half strength.
  Extreme cases: disorganized adapter + organized prompt -> prompt dominates (-0.17 along
  adapter's trait); anxious adapter + relaxed prompt -> adapter side (cos A_t 0.83 vs 0.35).
- User-turn window (no generation yet): the two effects add almost exactly (residual 0.12
  raw / 0.28 specific, weights 0.97 / 0.99). Saturation appears once the model generates.
Cost: one A100-40GB, 90 minutes, ~$3.5.
