#!/usr/bin/env python3
"""EXPERIMENT 2: is the stage-two shared direction specific to PERSONA
introspection, or generic to SFT on self-generated introspective transcripts?

The control that was never run.  The same stage-two recipe -- the same ten
reflection instructions, the same self-interaction mechanism at n_reflection
1000 / n_interaction 1000 / k_turns 10, the same bug-faithful assembly, the same
rank-64 SFT at the same hyperparameters and the SAME sft_seed 123456 -- run with
a trait-free constitution (constitutions_neutral.json) on the PLAIN BASE MODEL,
with no stage-one adapter, repeated with five independent generation seeds.

Why sft_seed must stay 123456.  analysis/lora_a_identity.json shows every
seed-0 stage-two adapter starts from the same LoRA-A draw (pairwise cosine
0.9977 to 0.9981 after training), and a seed-1 adapter's LoRA-A is
near-orthogonal to it (cosine 0.0021).  A cosine between two LoRA deltas is only
large when they share that row space: cross-trait stage-two cosines are +0.1452
within seed 0 and +0.0141 across seeds (analysis/crossseed_arms_stage2.json).
A neutral adapter trained at a different seed would be near-zero to the zoo's
grand mean whatever it had learned, so the control has to share the frame.

Inputs
  results/cross_gram_full_neutral_loras_introspection_x_loras_introspection.npz
      5 neutral x 134 stage-two
  results/cross_gram_full_neutral_loras_introspection_x_neutral_loras_introspection.npz
      5 x 5
  results/gram_stage2.npz                        the 134 x 134 stage-two Gram
  analysis/stage2_structure.json                 the numbers being compared against

Output: analysis/stage2_neutral_control.json
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = f"{Q}/results"
XN = f"{R}/cross_gram_full_neutral_loras_introspection_x_loras_introspection.npz"
NN = f"{R}/cross_gram_full_neutral_loras_introspection_x_neutral_loras_introspection.npz"


def main():
    missing = [p for p in (XN, NN) if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing cross-Grams: " + ", ".join(missing))

    z2 = np.load(f"{R}/gram_stage2.npz", allow_pickle=True)
    G = np.array(z2["G"], dtype=float)
    zoo = [str(x) for x in z2["names"]]
    n = len(zoo)
    dz = np.sqrt(np.diag(G))

    zx = np.load(XN, allow_pickle=True)
    X = np.array(zx["X"], dtype=float)                # (5, 134)
    neu = [str(x) for x in zx["names_a"]]
    colnames = [str(x) for x in zx["names_b"]]
    o = [colnames.index(t) for t in zoo]
    X = X[:, o]
    nn_ = np.array(zx["norms_a"], dtype=float)        # neutral norms
    nz = np.array(zx["norms_b"], dtype=float)[o]
    assert np.allclose(nz, dz, rtol=1e-6), "stage-two norms disagree between files"

    zs = np.load(NN, allow_pickle=True)
    S = np.array(zs["X"], dtype=float)
    ns = [str(x) for x in zs["names_a"]]
    p = [ns.index(t) for t in neu]
    S = S[np.ix_(p, p)]
    ds = np.sqrt(np.diag(S))
    assert np.allclose(ds, nn_, rtol=1e-6), "neutral norms disagree between files"

    # --- the shared direction ------------------------------------------------
    # mu = (1/n) sum_i dW_i;  |mu|^2 = 1'G1 / n^2
    mu2 = float(np.ones(n) @ G @ np.ones(n)) / n ** 2
    mu_norm = float(np.sqrt(mu2))
    cos_neutral_mu = (X.sum(1) / n) / (nn_ * mu_norm)
    # the zoo's own, recomputed the same way (analyse_stage2_structure.shared)
    cos_zoo_mu = (G.sum(1) / n) / (dz * mu_norm)

    # --- neutral x neutral and neutral x trait -------------------------------
    Cs = S / np.outer(ds, ds)
    iu = np.triu_indices(len(neu), 1)
    Cx = X / np.outer(nn_, dz)
    Cz = G / np.outer(dz, dz)
    izoo = np.triu_indices(n, 1)

    # after the shared direction is projected out of both sides
    g = G @ np.ones(n)
    denom = float(np.ones(n) @ G @ np.ones(n))
    G_res = G - np.outer(g, g) / denom
    X_res = X - np.outer(X.sum(1), g) / denom
    S_res = S - np.outer(X.sum(1), X.sum(1)) / denom
    d_res_z = np.sqrt(np.diag(G_res))
    d_res_n = np.sqrt(np.diag(S_res))
    Cx_res = X_res / np.outer(d_res_n, d_res_z)
    Cs_res = S_res / np.outer(d_res_n, d_res_n)

    best = []
    for i, t in enumerate(neu):
        j = int(np.argmax(Cx[i]))
        jr = int(np.argmax(Cx_res[i]))
        best.append({"neutral": t, "norm": float(nn_[i]),
                     "cos_with_grand_mean": float(cos_neutral_mu[i]),
                     "mean_cos_with_134": float(Cx[i].mean()),
                     "sd_cos_with_134": float(Cx[i].std()),
                     "max_cos_trait": zoo[j], "max_cos": float(Cx[i, j]),
                     "min_cos_trait": zoo[int(np.argmin(Cx[i]))],
                     "min_cos": float(Cx[i].min()),
                     "after_removing_shared": {
                         "mean_cos_with_134": float(Cx_res[i].mean()),
                         "sd_cos_with_134": float(Cx_res[i].std()),
                         "max_cos_trait": zoo[jr], "max_cos": float(Cx_res[i, jr])}})

    st2 = json.load(open(f"{Q}/analysis/stage2_structure.json"))["shared_component"]

    out = {
        "what": ("Five neutral stage-two LoRAs: the OCT introspection recipe with a "
                 "trait-free constitution on the plain base model, no stage-one "
                 "adapter, five generation seeds, sft_seed 123456 as in the zoo."),
        "why_same_sft_seed": ("analysis/lora_a_identity.json: adapters trained at the "
                              "same sft_seed share a LoRA-A row space (pairwise cosine "
                              "0.998); across seeds LoRA-A cosine is 0.002. Delta "
                              "cosines are only comparable inside one frame."),
        "n_neutral": len(neu), "neutral_names": neu,
        "reference": {
            "zoo_stage2_cos_to_grand_mean_mean": st2["stage2"]["cos_to_mean_direction_mean"],
            "zoo_stage2_cos_to_grand_mean_sd": st2["stage2"]["cos_to_mean_direction_sd"],
            "zoo_stage2_cos_to_grand_mean_min": st2["stage2"]["cos_to_mean_direction_min"],
            "zoo_stage2_cos_to_grand_mean_max": st2["stage2"]["cos_to_mean_direction_max"],
            "zoo_stage2_shared_share_of_norm2": st2["stage2"]["mean_direction_norm2_over_mean_norm2"],
            "zoo_stage2_offdiag_cos_mean": float(Cz[izoo].mean()),
            "zoo_stage2_offdiag_cos_sd": float(Cz[izoo].std()),
            "zoo_stage2_norm_mean": float(dz.mean()),
            "zoo_stage2_norm_sd": float(dz.std()),
            "grand_mean_norm": mu_norm,
            "recomputed_zoo_cos_to_mean_mean": float(cos_zoo_mu.mean())},
        "neutral_vs_grand_mean": {
            "mean": float(cos_neutral_mu.mean()), "sd": float(cos_neutral_mu.std()),
            "min": float(cos_neutral_mu.min()), "max": float(cos_neutral_mu.max()),
            "per_seed": {t: float(c) for t, c in zip(neu, cos_neutral_mu)}},
        "neutral_x_neutral": {
            "mean": float(Cs[iu].mean()), "sd": float(Cs[iu].std()),
            "min": float(Cs[iu].min()), "max": float(Cs[iu].max()),
            "matrix": np.round(Cs, 4).tolist(),
            "after_removing_shared_mean": float(Cs_res[iu].mean()),
            "after_removing_shared_matrix": np.round(Cs_res, 4).tolist()},
        "neutral_x_zoo": {
            "mean": float(Cx.mean()), "sd": float(Cx.std()),
            "min": float(Cx.min()), "max": float(Cx.max()),
            "after_removing_shared_mean": float(Cx_res.mean()),
            "after_removing_shared_sd": float(Cx_res.std())},
        "neutral_norms": {"mean": float(nn_.mean()), "sd": float(nn_.std()),
                          "per_seed": {t: float(v) for t, v in zip(neu, nn_)}},
        "per_neutral": best,
    }

    # Does a neutral adapter behave like an extra member of the zoo?  Compare its
    # profile of cosines over the 134 with what a zoo member's own profile looks
    # like (leave-one-out), on the residual after the shared direction.
    Cz_res = G_res / np.outer(d_res_z, d_res_z)
    off = Cz_res[~np.eye(n, dtype=bool)]
    out["neutral_x_zoo"]["zoo_offdiag_after_removing_shared_mean"] = float(off.mean())
    out["neutral_x_zoo"]["zoo_offdiag_after_removing_shared_sd"] = float(off.std())

    # correlation of each neutral's residual cosine profile with each zoo trait's
    corrs = []
    for i in range(len(neu)):
        v = Cx_res[i]
        rs = []
        for j in range(n):
            w = Cz_res[j].copy()
            m = np.ones(n, bool); m[j] = False
            rs.append(np.corrcoef(v[m], w[m])[0, 1])
        rs = np.array(rs)
        corrs.append({"neutral": neu[i], "max_r": float(rs.max()),
                      "max_r_trait": zoo[int(np.argmax(rs))],
                      "mean_r": float(rs.mean())})
    out["residual_profile_correlation_with_zoo_traits"] = corrs

    p_ = f"{Q}/analysis/stage2_neutral_control.json"
    json.dump(out, open(p_, "w"), indent=1)

    r = out["reference"]
    print(f"zoo stage-two: cos to grand mean {r['zoo_stage2_cos_to_grand_mean_mean']:.3f} "
          f"+- {r['zoo_stage2_cos_to_grand_mean_sd']:.3f} "
          f"[{r['zoo_stage2_cos_to_grand_mean_min']:.3f}, {r['zoo_stage2_cos_to_grand_mean_max']:.3f}]; "
          f"norm {r['zoo_stage2_norm_mean']:.3f} +- {r['zoo_stage2_norm_sd']:.3f}; "
          f"off-diag cos {r['zoo_stage2_offdiag_cos_mean']:+.4f}")
    g_ = out["neutral_vs_grand_mean"]
    print(f"neutral: cos to grand mean {g_['mean']:.3f} +- {g_['sd']:.3f} "
          f"[{g_['min']:.3f}, {g_['max']:.3f}]")
    print(f"neutral x neutral: {out['neutral_x_neutral']['mean']:+.4f} "
          f"+- {out['neutral_x_neutral']['sd']:.4f}  "
          f"(after removing the shared direction "
          f"{out['neutral_x_neutral']['after_removing_shared_mean']:+.4f})")
    print(f"neutral x zoo: {out['neutral_x_zoo']['mean']:+.4f} "
          f"+- {out['neutral_x_zoo']['sd']:.4f}  after removing shared "
          f"{out['neutral_x_zoo']['after_removing_shared_mean']:+.4f} "
          f"(zoo's own off-diagonal after removal "
          f"{out['neutral_x_zoo']['zoo_offdiag_after_removing_shared_mean']:+.4f})")
    for b in best:
        print(f"  {b['neutral']}: |dW| {b['norm']:.3f}  cos(mu) {b['cos_with_grand_mean']:.3f}  "
              f"mean cos 134 {b['mean_cos_with_134']:+.4f}  best {b['max_cos_trait']} "
              f"{b['max_cos']:+.4f}")
    print(f"wrote {p_}")


if __name__ == "__main__":
    main()
