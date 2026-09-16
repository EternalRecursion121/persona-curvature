#!/usr/bin/env python3
"""Targets for the SliderSpace-style persona sliders (experiment S1).

A target is a direction in the base model's residual stream at LAYER 16, the
project's primary layer, in the RESPONSE-token window -- the same object
`analyse_actspace.py` calls a persona vector, trait-centred across the 134
traits exactly as that script centres before every cross-space cosine.

HALVES ARE THE HELD-OUT SPLIT.  `act_space.py` stores prompts 0-31 and 32-63
separately (M[..., half, ...]).  Targets are built from HALF 0 ONLY and the
sliders train on those 32 prompts; half 1 (prompts 32-63) is never seen by the
optimiser and is where the induced shift is scored.  The half-to-half cosine of
a persona vector is 0.959 at this layer
(analysis/actspace_geometry.json#windows.resp.curve[16].floor), so the two
halves are the same direction to within that floor; the split tests the SLIDER,
not the target.

Thirteen targets:
  * ten traits, two per Big Five factor, one per pole, all ten among the 100
    traits judged in phase10_runs/judged_100.json so the behavioural comparison
    has a stage-one reference;
  * `hole_fa`, the activation-space image of the FACTOR-chart hole: the same
    134 coefficients that analysis/alien_fa.json#alien_fa.coeffs gives over the
    adapters, applied to the 134 trait-centred persona vectors.  Those
    coefficients sum to -2.78e-17, so the image is a contrast and is unaffected
    by the centring;
  * `hole_shuf1`, `hole_shuf2`, the same coefficients permuted across traits --
    the control `analyse_actspace.py` already uses for the transplanted hole.

Writes analysis/slider_targets.npz (baked into the Modal image) and
analysis/slider_targets_meta.json (the comparators, computed with no GPU).
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY = 16
WINDOW = 0                      # 'resp'
TRAITS = ["extraverted", "introverted",      # Extraversion  +/-
          "warm", "cold",                    # Agreeableness +/-
          "organized", "disorganized",       # Conscientiousness +/-
          "relaxed", "anxious",              # EmotionalStability +/-
          "intellectual", "unintellectual"]  # Intellect +/-


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    Z = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    names = [str(t) for t in Z["traits"]]
    M = Z["M"].astype(np.float32)
    B = Z["B"].astype(np.float32)
    # per-half persona vectors at the primary layer, response window
    V = [M[:, WINDOW, h, PRIMARY] - B[WINDOW, h, PRIMARY][None] for h in (0, 1)]
    Xc = [v - v.mean(0) for v in V]                       # trait-centred

    ZA = np.load(f"{Q}/analysis/actspace_means_adapters.npz", allow_pickle=True)
    assert [str(t) for t in ZA["traits"]] == names, "trait order differs"
    MA = ZA["M"].astype(np.float32)
    # the adapters run has NO system prompt, so its baseline is the same
    # no-system baseline B that the prompt run used
    A = [MA[:, WINDOW, h, PRIMARY] - B[WINDOW, h, PRIMARY][None] for h in (0, 1)]
    Ac = [a - a.mean(0) for a in A]

    AL = json.load(open(f"{Q}/analysis/alien_fa.json"))["alien_fa"]
    coef = np.array([dict(zip(AL["traits"], AL["coeffs"]))[t] for t in names],
                    dtype=np.float32)

    idx = {t: names.index(t) for t in TRAITS}
    targets, meta = {}, {}
    for t in TRAITS:
        targets[t] = Xc[0][idx[t]]
        meta[t] = {"kind": "trait", "trait": t}
    targets["hole_fa"] = coef @ Xc[0]
    meta["hole_fa"] = {"kind": "hole", "coeff_source":
                       "analysis/alien_fa.json#alien_fa.coeffs"}
    rng = np.random.default_rng(20260909)
    for k in (1, 2):
        p = rng.permutation(len(names))
        targets[f"hole_shuf{k}"] = coef[p] @ Xc[0]
        meta[f"hole_shuf{k}"] = {"kind": "hole_control", "perm_seed": 20260909,
                                 "perm_draw": k}

    order = list(targets)
    T = np.stack([targets[k] for k in order])
    norms = np.linalg.norm(T, axis=1)
    U = T / norms[:, None]
    # Magnitude anchor for the training loss: how big a shift the slider is
    # asked to make.  For a trait it is that trait's own persona-vector norm --
    # "make the change the constitution makes".  The two shuffled controls are
    # anchored to the HOLE's norm, not to their own: a permuted contrast is
    # smaller (0.54, 0.42 against 1.02) because the permutation averages the
    # traits out, and dosing the controls below the hole would confound the one
    # comparison they exist for.
    anchor = norms.copy()
    hole_i = order.index("hole_fa")
    for k in ("hole_shuf1", "hole_shuf2"):
        anchor[order.index(k)] = norms[hole_i]
    np.savez(f"{Q}/analysis/slider_targets.npz", U=U.astype(np.float32),
             norms=norms.astype(np.float32), anchor=anchor.astype(np.float32),
             names=np.array(order), layer=PRIMARY, window="resp", half=0)

    # --- comparators, no GPU -------------------------------------------------
    # the bar a slider has to clear, for every quantity it will be scored on
    out = {"layer": PRIMARY, "window": "resp", "target_half": 0,
           "held_out_half": 1, "names": order, "target_norms":
           {k: float(n) for k, n in zip(order, norms)},
           "magnitude_anchors": {k: float(a) for k, a in zip(order, anchor)},
           "target_cosines": (U @ U.T).tolist(), "meta": meta,
           "comparators": {}}
    for t in TRAITS:
        i = idx[t]
        out["comparators"][t] = {
            # the target's own half-1 replicate: the ceiling for held-out cosine
            "persona_half1_centred_vs_target": cos(Xc[1][i], targets[t]),
            # the raw (uncentred) persona vector against the centred target:
            # what a slider that exactly reproduced the PROMPT's shift scores
            "persona_half1_raw_vs_target": cos(V[1][i], targets[t]),
            # the stage-one adapter's own free-generation shift against the
            # target: what the object the sliders are compared to scores
            "adapter_half1_raw_vs_target": cos(A[1][i], targets[t]),
            "adapter_half1_centred_vs_target": cos(Ac[1][i], targets[t]),
            "adapter_shift_norm_half1": float(np.linalg.norm(A[1][i])),
            "adapter_centred_shift_norm_half1": float(np.linalg.norm(Ac[1][i])),
        }
    # the hole's activation-space image has no adapter; its comparator is the
    # transplant test the project already ran, recomputed here on half 0 so the
    # slider can be scored against the same construction
    for k in ("hole_fa", "hole_shuf1", "hole_shuf2"):
        v = targets[k]
        c = np.abs(Xc[0] @ v) / (np.linalg.norm(Xc[0], axis=1) * np.linalg.norm(v))
        j = int(np.argmax(c))
        out["comparators"][k] = {
            "nearest_trait_half0": names[j],
            "nearest_trait_deg_half0": float(np.degrees(np.arccos(c[j]))),
            "half1_replicate_cos": cos(
                (coef if k == "hole_fa" else None) @ Xc[1] if k == "hole_fa"
                else np.zeros(1), v) if k == "hole_fa" else None,
        }
    out["comparators"]["hole_fa"]["half1_replicate_cos"] = cos(coef @ Xc[1],
                                                               targets["hole_fa"])
    json.dump(out, open(f"{Q}/analysis/slider_targets_meta.json", "w"), indent=1)
    print(json.dumps({k: out["comparators"][k] for k in out["comparators"]}, indent=1))
    print(f"\nwrote analysis/slider_targets.npz ({T.shape}) and "
          f"analysis/slider_targets_meta.json")
    print("target norms:", {k: round(float(n), 2) for k, n in zip(order, norms)})


if __name__ == "__main__":
    main()
