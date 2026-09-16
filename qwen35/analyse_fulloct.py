#!/usr/bin/env python3
"""Does the zoo's geometry replicate on the FULL OCT persona adapters?

The page's geometry is computed on stage-1 (DPO) adapters.  The deployed OCT
artefact is the persona adapter, dW = dW_dpo + 0.25 dW_sft, stored exactly as a
rank-128 concatenation (fix_persona_merge.py, /oct/personas_exact).  This
script compares the three 134 x 134 Grams (stage 1, stage 2, persona) and runs
the second-seed identification on the 15 seed-1 personas.

Inputs (results/):
  gram_sweep.npz                                              stage-1 within-run Gram (G, names)
  cross_gram_full_loras_introspection_x_loras_introspection.npz   stage-2 within-run (X, names_a)
  cross_gram_full_personas_exact_x_personas_exact.npz         persona within-run (X, names_a)
  cross_gram_full_personas_exact_x_seed1_personas_exact.npz   persona seed 0 x seed 1 (134 x 15)
  cross_gram_full_seed1_personas_exact_x_seed1_personas_exact.npz  seed-1 within (15 x 15)
  cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz    stage-1 x persona (134 x 134)
  cross_gram_seedpaired_matched.npz                           stage-1 seed 0 x seed 1 (134 x 40)
  cross_gram_full_loras_introspection_x_seed1_loras_introspection.npz  stage-2 seed 0 x seed 1 (134 x 15)
Output: analysis/fulloct_geometry.json and a printed summary.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
W_SFT = 0.25


def load_square(path):
    z = np.load(path, allow_pickle=True)
    G = np.array(z["G"] if "G" in z else z["X"], dtype=np.float64)
    names = [str(x) for x in (z["names"] if "names" in z else z["names_a"])]
    return G, names


def load_cross(path):
    z = np.load(path, allow_pickle=True)
    return (np.array(z["X"], dtype=np.float64), [str(x) for x in z["names_a"]],
            [str(x) for x in z["names_b"]], np.array(z["norms_a"]), np.array(z["norms_b"]))


def cosine(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def offdiag(C):
    iu = np.triu_indices(C.shape[0], 1)
    return C[iu]


def centre(G):
    n = G.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return J @ G @ J


def pcs(G, k):
    """Scores of the k leading PCs from the double-centred Gram."""
    w, V = np.linalg.eigh(centre(G))
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    return V[:, :k] * np.sqrt(np.clip(w[:k], 0, None)), w


def procrustes_r2(S1, S2):
    """Orthogonal Procrustes: how much of S2's variance S1 explains after rotation."""
    A = S1 - S1.mean(0)
    B = S2 - S2.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    Q = U @ Vt
    scale = np.trace((A @ Q).T @ B) / np.trace(A.T @ A)
    resid = ((scale * A @ Q - B) ** 2).sum()
    return float(1 - resid / (B ** 2).sum())


def principal_angles(S1, S2):
    Q1, _ = np.linalg.qr(S1 - S1.mean(0))
    Q2, _ = np.linalg.qr(S2 - S2.mean(0))
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1))).tolist()


def align(G, names, order):
    idx = [names.index(n) for n in order]
    return G[np.ix_(idx, idx)]


def gram_corr(Ca, Cb):
    a, b = offdiag(Ca), offdiag(Cb)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    out = {}
    G1, n1 = load_square(f"{R}/gram_sweep.npz")
    G2, n2 = load_square(f"{R}/cross_gram_full_loras_introspection_x_loras_introspection.npz")
    Gp, npn = load_square(f"{R}/cross_gram_full_personas_exact_x_personas_exact.npz")
    order = sorted(set(n1) & set(n2) & set(npn))
    assert len(order) == 134, len(order)
    G1, G2, Gp = align(G1, n1, order), align(G2, n2, order), align(Gp, npn, order)

    # ---- identity check: |persona|^2 = |s1|^2 + w^2 |s2|^2 + 2 w <s1, s2>
    # stage-1 x stage-2 cross terms are not in a same-volume Gram, so check the
    # bound that follows from cos(s1, s2) ~ 0 and report the residual.
    pred = np.diag(G1) + W_SFT ** 2 * np.diag(G2)
    resid = (np.diag(Gp) - pred) / np.diag(Gp)
    out["norm_identity"] = {
        "what": "(|persona|^2 - |s1|^2 - 0.0625 |s2|^2) / |persona|^2 per trait; the remainder is 2*0.25*<s1,s2>",
        "mean": float(resid.mean()), "max_abs": float(np.abs(resid).max()),
        "frac_norm2_from_stage1_mean": float(np.mean(np.diag(G1) / np.diag(Gp))),
        "frac_norm2_from_stage2_mean": float(np.mean(W_SFT ** 2 * np.diag(G2) / np.diag(Gp))),
        "norm_ratio_s2_raw_over_s1_mean": float(np.mean(np.sqrt(np.diag(G2) / np.diag(G1)))),
    }

    C1, C2, Cp = cosine(G1), cosine(G2), cosine(Gp)
    out["gram_correlation_offdiag"] = {
        "persona_vs_stage1": gram_corr(Cp, C1), "persona_vs_stage2": gram_corr(Cp, C2),
        "stage1_vs_stage2": gram_corr(C1, C2),
        "persona_vs_stage1_centred": gram_corr(centre(Cp), centre(C1)),
        "persona_vs_stage2_centred": gram_corr(centre(Cp), centre(C2)),
        "stage1_vs_stage2_centred": gram_corr(centre(C1), centre(C2)),
        "mean_offdiag_cosine": {"stage1": float(offdiag(C1).mean()), "stage2": float(offdiag(C2).mean()),
                                "persona": float(offdiag(Cp).mean())},
        "std_offdiag_cosine": {"stage1": float(offdiag(C1).std()), "stage2": float(offdiag(C2).std()),
                               "persona": float(offdiag(Cp).std())},
        "why_stage1_dominates": ("with equal norms the persona cosine is the mean of the two stage cosines; "
                                 "stage-2 cosines sit on a large shared component with little pair-to-pair "
                                 "variation, so the variation (the arrangement) comes from stage 1"),
    }
    # persona cosine predicted from the two stage Grams if all s1-s2 cross terms vanish
    Gpred = G1 + W_SFT ** 2 * G2
    out["gram_correlation_offdiag"]["persona_vs_predicted_no_cross_terms"] = gram_corr(Cp, cosine(Gpred))
    out["gram_correlation_offdiag"]["persona_vs_predicted_raw_rel_err"] = float(
        np.linalg.norm(Gp - Gpred) / np.linalg.norm(Gp))

    # ---- PCA spectra and subspace agreement
    res = {}
    for k in (5, 9):
        S1, w1 = pcs(G1, k)
        S2, w2 = pcs(G2, k)
        Sp, wp = pcs(Gp, k)
        res[f"k{k}"] = {
            "procrustes_r2_persona_from_stage1": procrustes_r2(S1, Sp),
            "procrustes_r2_persona_from_stage2": procrustes_r2(S2, Sp),
            "procrustes_r2_stage2_from_stage1": procrustes_r2(S1, S2),
            "principal_angles_deg_stage1_persona": principal_angles(S1, Sp),
            "principal_angles_deg_stage2_persona": principal_angles(S2, Sp),
        }
    _, w1 = pcs(G1, 5); _, w2 = pcs(G2, 5); _, wp = pcs(Gp, 5)
    res["variance_fraction_top10"] = {
        "stage1": (w1[:10] / w1.sum()).tolist(), "stage2": (w2[:10] / w2.sum()).tolist(),
        "persona": (wp[:10] / wp.sum()).tolist()}
    out["pca"] = res

    # ---- nearest neighbours agree?
    def nn(C):
        D = C.copy(); np.fill_diagonal(D, -np.inf); return D.argmax(1)
    out["nearest_neighbour_agreement"] = {
        "persona_vs_stage1": int((nn(Cp) == nn(C1)).sum()), "persona_vs_stage2": int((nn(Cp) == nn(C2)).sum()),
        "stage1_vs_stage2": int((nn(C1) == nn(C2)).sum()), "n": 134}

    # ---- stage-1 x persona same-trait cosine (two-volume cross-Gram), if present
    p = f"{R}/cross_gram_full_root_x_pc-qwen35-oct2_personas_exact.npz"
    if os.path.exists(p):
        X, na, nb, Na, Nb = load_cross(p)
        C = X / np.outer(Na, Nb)
        ia, ib = {m: i for i, m in enumerate(na)}, {m: i for i, m in enumerate(nb)}
        same = np.array([C[ia[t], ib[t]] for t in order])
        predicted = np.array([np.sqrt(G1[i, i] / Gp[i, i]) for i in range(134)])
        ranks = [int(1 + np.sum(C[:, ib[t]] > C[ia[t], ib[t]])) for t in order]
        out["stage1_x_persona"] = {
            "same_trait_cos_mean": float(same.mean()), "same_trait_cos_min": float(same.min()),
            "predicted_from_norms_if_orthogonal_mean": float(predicted.mean()),
            "top1_stage1_identifies_own_persona": sum(1 for r in ranks if r == 1), "n": 134,
            "cross_trait_cos_mean": float((C.sum() - np.trace(C[np.ix_([ia[t] for t in order], [ib[t] for t in order])])) / (134 * 133)),
        }

    # ---- second seed on the full persona: reuse analyse_crossseed.arm
    os.environ["PC_WITHIN"] = f"{R}/cross_gram_full_personas_exact_x_personas_exact.npz"
    import analyse_crossseed as ac
    L = ac.labels()
    arms = {}
    for tag, path in (("persona_seed0_x_seed1", f"{R}/cross_gram_full_personas_exact_x_seed1_personas_exact.npz"),
                      ("stage2_seed0_x_seed1", f"{R}/cross_gram_full_loras_introspection_x_seed1_loras_introspection.npz")):
        if os.path.exists(path):
            arms[tag] = ac.arm(path, L)
    # stage-1 seed pair, restricted to the same 15 traits for a like-for-like line
    # stage-1 seed pair: take the published matched arm from analyse_crossseed's output
    pj = f"{HERE}/analysis/crossseed_arms.json"
    if os.path.exists(pj):
        for a in json.load(open(pj)):
            if "matched" in a.get("path", ""):
                arms["stage1_seed0_x_seed1_40traits_published"] = a
    out["second_seed"] = arms
    out["second_seed_note"] = ("r/d for the rank-128 persona is 128/2560 = 0.05; for a rank-64 stage adapter 0.025. "
                               "The persona's two halves come from independent LoRA-A draws, so the "
                               "subspace-overlap prediction for its cross-seed slope is 0.05.")

    # seed-1 within-run: 15 x 15 persona Gram vs the same 15 traits in seed 0
    p15 = f"{R}/cross_gram_full_seed1_personas_exact_x_seed1_personas_exact.npz"
    if os.path.exists(p15):
        G15, n15 = load_square(p15)
        idx0 = [order.index(t) for t in n15]
        C0 = cosine(Gp[np.ix_(idx0, idx0)]); C15 = cosine(G15)
        out["seed1_within_15"] = {"gram_corr_seed0_vs_seed1_15traits": gram_corr(C0, C15),
                                  "gram_corr_centred": gram_corr(centre(C0), centre(C15)), "traits": n15}

    json.dump(out, open(f"{HERE}/analysis/fulloct_geometry.json", "w"), indent=1)
    g = out["gram_correlation_offdiag"]; ni = out["norm_identity"]
    print(f"norm^2 share: stage1 {ni['frac_norm2_from_stage1_mean']:.3f}  0.25*stage2 {ni['frac_norm2_from_stage2_mean']:.3f}  "
          f"residual(cross) mean {ni['mean']:+.4f} max|.| {ni['max_abs']:.4f}")
    print(f"Gram corr (off-diag cosines): persona~s1 {g['persona_vs_stage1']:.3f}  persona~s2 {g['persona_vs_stage2']:.3f}  "
          f"s1~s2 {g['stage1_vs_stage2']:.3f}   centred: {g['persona_vs_stage1_centred']:.3f} / {g['persona_vs_stage2_centred']:.3f} / {g['stage1_vs_stage2_centred']:.3f}")
    print(f"persona vs no-cross-term prediction: corr {g['persona_vs_predicted_no_cross_terms']:.4f}  rel err {g['persona_vs_predicted_raw_rel_err']:.4f}")
    for k in ("k5", "k9"):
        r = out["pca"][k]
        print(f"PCA {k}: Procrustes R2 persona<-s1 {r['procrustes_r2_persona_from_stage1']:.3f}  persona<-s2 {r['procrustes_r2_persona_from_stage2']:.3f}  "
              f"s2<-s1 {r['procrustes_r2_stage2_from_stage1']:.3f}; angles s1-persona {np.round(r['principal_angles_deg_stage1_persona'],1).tolist()}")
    print("NN agreement", out["nearest_neighbour_agreement"])
    if "stage1_x_persona" in out:
        print("stage1 x persona", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["stage1_x_persona"].items()})
    for tag, a in arms.items():
        print(f"{tag}: same {a['same'][1]:+.4f} diff {a['diff_factor'][1]:+.4f} top1 {a['top1']}/{a['n_b']} sep {a['sep']:+.4f} slope {a['slope']:.4f} r {a['pearson']:.3f}")
    if "seed1_within_15" in out:
        print("seed-1 within 15:", out["seed1_within_15"]["gram_corr_seed0_vs_seed1_15traits"], out["seed1_within_15"]["gram_corr_centred"])
    print("wrote analysis/fulloct_geometry.json")


if __name__ == "__main__":
    main()
