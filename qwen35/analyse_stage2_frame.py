#!/usr/bin/env python3
"""Is the stage-two shared direction a property of the RECIPE or of the FRAME?

Two facts sit awkwardly together.

  (a) Every stage-two adapter sits at cosine 0.389 to the grand mean of the 134,
      and the mean direction carries 0.151 of the average squared norm
      (analysis/stage2_structure.json#shared_component.stage2).
  (b) A stage-two adapter and its own second-seed twin have cosine 0.0672, and
      cross-trait across seeds only 0.0141
      (analysis/crossseed_arms_stage2.json).

analysis/lora_a_identity.json resolves it: within one sft_seed every adapter
starts from the SAME LoRA-A draw and stays there (pairwise cosine 0.9977-0.9981
after training, stage one 0.99997); across seeds LoRA-A is near-orthogonal
(cosine 0.0021).  A rank-64 delta lives in the 64-dimensional row space its
LoRA-A defines, so two adapters trained in different frames cannot have a large
cosine no matter what they learned.  wiki/pages/geometry/stage-two-structure.md
says "each with its own random LoRA-A", which is wrong for the zoo.

That raises the obvious worry: is the 0.389 an artefact of the shared frame?
This script answers it with the second-seed run, which has its own frame:
compute the shared component separately WITHIN seed 0 and WITHIN seed 1, on the
same 15 traits.  If the two agree, the shared component is a reproducible
property of the recipe that happens to be expressed in whichever coordinates the
initialisation supplies.

Output: analysis/stage2_frame.json
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
R = f"{Q}/results"


def load(p, k="G", nk="names"):
    z = np.load(p, allow_pickle=True)
    return np.array(z[k], float), [str(x) for x in z[nk]]


def shared(G):
    n = G.shape[0]
    d = np.sqrt(np.diag(G))
    m2 = float(G.sum()) / n ** 2
    mean_diag = float(np.trace(G)) / n
    cos = (G.sum(1) / n) / (d * np.sqrt(m2))
    C = G / np.outer(d, d)
    iu = np.triu_indices(n, 1)
    return {"n": int(n),
            "mean_direction_norm2_over_mean_norm2": float(m2 / mean_diag),
            "cos_to_mean_direction_mean": float(cos.mean()),
            "cos_to_mean_direction_sd": float(cos.std()),
            "cos_to_mean_direction_min": float(cos.min()),
            "cos_to_mean_direction_max": float(cos.max()),
            "offdiag_cos_mean": float(C[iu].mean()),
            "offdiag_cos_sd": float(C[iu].std()),
            "norm_mean": float(d.mean()), "norm_sd": float(d.std())}


def main():
    G2, n2 = load(f"{R}/gram_stage2.npz")
    G2s1, n2s1 = load(f"{R}/cross_gram_full_seed1_loras_introspection_x_"
                      f"seed1_loras_introspection.npz", "X", "names_a")
    G1, n1 = load(f"{R}/gram_sweep.npz")

    fifteen = sorted(n2s1)
    i2 = [n2.index(t) for t in fifteen]
    i1 = [n1.index(t) for t in fifteen]
    j2 = [n2s1.index(t) for t in fifteen]

    out = {
        "what": ("The stage-two shared component measured separately inside each "
                 "seed's own LoRA-A frame. Seed 1 is an independent run of the whole "
                 "two-stage pipeline on 15 traits: different stage-one adapters "
                 "(data_null_seedpaired_s40_matched), different generated corpus, "
                 "sft_seed 1 and therefore a different LoRA-A."),
        "lora_a": json.load(open(f"{Q}/analysis/lora_a_identity.json"))["sets"]
        if os.path.exists(f"{Q}/analysis/lora_a_identity.json") else None,
        "shared_component": {
            "stage2_seed0_all134": shared(G2),
            "stage2_seed0_same15": shared(G2[np.ix_(i2, i2)]),
            "stage2_seed1_15": shared(G2s1[np.ix_(j2, j2)]),
            "stage1_seed0_all134": shared(G1),
            "stage1_seed0_same15": shared(G1[np.ix_(i1, i1)])},
        "traits_15": fifteen,
    }
    a = out["shared_component"]["stage2_seed0_same15"]
    b = out["shared_component"]["stage2_seed1_15"]
    out["reading"] = (
        "On the same 15 traits the shared component is the same size in both frames: "
        f"share of squared norm {a['mean_direction_norm2_over_mean_norm2']:.4f} at seed 0 "
        f"and {b['mean_direction_norm2_over_mean_norm2']:.4f} at seed 1; cosine to the "
        f"grand mean {a['cos_to_mean_direction_mean']:.4f} +- {a['cos_to_mean_direction_sd']:.4f} "
        f"and {b['cos_to_mean_direction_mean']:.4f} +- {b['cos_to_mean_direction_sd']:.4f}; "
        f"off-diagonal cosine {a['offdiag_cos_mean']:+.4f} and {b['offdiag_cos_mean']:+.4f}. "
        "The shared direction is a reproducible property of the introspection recipe, not "
        "of one random initialisation. What the frame decides is the COORDINATES: the same "
        "learned direction in two different row spaces has cosine 0.0141 "
        "(analysis/crossseed_arms_stage2.json#diff_factor), which is why any comparison of "
        "stage-two deltas -- including the neutral control of "
        "analysis/stage2_neutral_control.json -- has to be made inside one frame.")

    p = f"{Q}/analysis/stage2_frame.json"
    json.dump(out, open(p, "w"), indent=1)
    for k, v in out["shared_component"].items():
        print(f"{k:>26s}: n={v['n']:>3d} share {v['mean_direction_norm2_over_mean_norm2']:.4f} "
              f"cos_to_mean {v['cos_to_mean_direction_mean']:.4f}+-{v['cos_to_mean_direction_sd']:.4f} "
              f"offdiag {v['offdiag_cos_mean']:+.4f}+-{v['offdiag_cos_sd']:.4f} "
              f"|dW| {v['norm_mean']:.3f}")
    print(out["reading"])
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
