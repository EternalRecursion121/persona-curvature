#!/usr/bin/env python3
"""EXPERIMENT 3: what does stage two look like in ACTIVATION space?

The weight-space picture (analysis/stage2_structure.json) is: one shared
direction carrying 0.151 of every stage-two adapter's squared norm, every
adapter at cosine 0.389 to it, and a nearly isotropic residual that still
carries the stage-one arrangement at r 0.81.  This asks the same questions of
the mean residual-stream shift each adapter produces, where "shared direction"
and "arrangement" are not tied to a LoRA-A frame at all.

Inputs (all from act_space.py, same 64 prompts, no system prompt, layer means
over the model's own greedy response and over the user-turn tokens):
  analysis/actspace_means.npz              constitution AS A PROMPT, P_t, + baseline B
  analysis/actspace_means_adapters.npz     base + stage-one adapter,   A1_t
  analysis/actspace_means_adapters_stage2.npz   base + stage-two LoRA, A2_t
  analysis/actspace_means_adapters_persona.npz  base + exact persona,  AP_t
Weight Grams for the arrangement comparison:
  results/gram_sweep.npz, results/gram_stage2.npz,
  results/cross_gram_full_personas_exact_x_personas_exact.npz

Output: analysis/actspace_stage2_geometry.json

Layer 16 is the primary layer, fixed in advance by analyse_actspace.py and kept.

A stage-two LoRA alone on the plain base model is OFF ITS TRAINING
DISTRIBUTION: it was trained on the stage-1-merged base.  That is deliberate --
it isolates what stage two adds -- but it is not a deployed configuration, and
the persona arm is the deployed one.
"""
import json
import os

import numpy as np
from scipy.stats import pearsonr

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY = 16
ARMS = [("stage1", "actspace_means_adapters.npz"),
        ("stage2", "actspace_means_adapters_stage2.npz"),
        ("persona", "actspace_means_adapters_persona.npz")]


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def offdiag(C):
    return C[~np.eye(C.shape[0], dtype=bool)]


def gram_cos(G):
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


def centred_cos(G):
    n = G.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    Gc = J @ G @ J
    d = np.sqrt(np.maximum(np.diag(Gc), 1e-12))
    return Gc / np.outer(d, d)


def main():
    ZP = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    traits = [str(t) for t in ZP["traits"]]
    windows = [str(w) for w in ZP["windows"]]
    B = ZP["B"].astype(np.float32)                                # (W, 2, L, D)
    P = ZP["M"].astype(np.float32).mean(2) - B.mean(1)[None]      # (T, W, L, D)

    A, halves = {}, {}
    for name, fn in ARMS:
        p = f"{Q}/analysis/{fn}"
        if not os.path.exists(p):
            print(f"MISSING {p} -- arm {name} skipped")
            continue
        Z = np.load(p, allow_pickle=True)
        assert [str(t) for t in Z["traits"]] == traits, f"{name} trait order differs"
        A[name] = Z["M"].astype(np.float32).mean(2) - B.mean(1)[None]
        halves[name] = Z["M"].astype(np.float32) - B[None]
    T, W, L, D = P.shape

    # --- weight-space Grams, aligned to the activation trait order -----------
    def load_G(path, key="G", nk="names"):
        z = np.load(path, allow_pickle=True)
        G = np.array(z[key], dtype=float)
        nm = [str(x) for x in z[nk]]
        o = [nm.index(t) for t in traits]
        return G[np.ix_(o, o)]

    W_G = {"stage1": load_G(f"{Q}/results/gram_sweep.npz"),
           "stage2": load_G(f"{Q}/results/gram_stage2.npz"),
           "persona": load_G(f"{Q}/results/cross_gram_full_personas_exact_x_"
                             f"personas_exact.npz", "X", "names_a")}
    W_cos_c = {k: centred_cos(v) for k, v in W_G.items()}
    W_cos_raw = {k: gram_cos(v) for k, v in W_G.items()}

    out = {"primary_layer": PRIMARY, "traits": len(traits), "windows": windows,
           "arms_present": sorted(A),
           "note": ("A stage-two LoRA alone on the plain base is off its training "
                    "distribution (it was trained on the stage-1-merged base). "
                    "The persona arm is the deployed configuration."),
           "windows_out": {}}

    for wi, wname in enumerate(windows):
        wo = {"per_arm": {}, "cross_arm": {}, "gram_correlations": {}}
        Pl = P[:, wi, PRIMARY]
        Pu = unit(Pl)
        Pc = Pl - Pl.mean(0)
        CP = unit(Pc) @ unit(Pc).T

        for name in A:
            Al = A[name][:, wi, PRIMARY]
            Au = unit(Al)

            # 1. shared component of the activation shifts
            mu = Al.mean(0)
            cos_mu = (Al @ mu) / (np.linalg.norm(Al, axis=1) * np.linalg.norm(mu))
            share = float((mu @ mu) / (Al ** 2).sum(1).mean())

            # 2. own vs other against the prompted persona vector
            C = Au @ Pu.T                                  # C[t, s] = cos(A_t, P_s)
            ranks = np.array([1 + int(np.sum(C[t] > C[t, t])) for t in range(T)])

            # 3. the arrangement: activation cosines, raw and trait-centred
            Ac = Al - Al.mean(0)
            CA = unit(Ac) @ unit(Ac).T
            CA_raw = Au @ Au.T

            # 4. noise floor from the two prompt halves
            hA = unit(halves[name][:, wi, 0, PRIMARY])
            hB = unit(halves[name][:, wi, 1, PRIMARY])
            floor = float(np.median(np.sum(hA * hB, 1)))

            wo["per_arm"][name] = {
                "shared_direction": {
                    "mean_direction_norm2_over_mean_norm2": share,
                    "cos_to_mean_direction_mean": float(cos_mu.mean()),
                    "cos_to_mean_direction_sd": float(cos_mu.std()),
                    "cos_to_mean_direction_min": float(cos_mu.min()),
                    "cos_to_mean_direction_max": float(cos_mu.max())},
                "vs_prompt_vector": {
                    "cos_own_mean": float(np.diag(C).mean()),
                    "cos_other_mean": float(offdiag(C).mean()),
                    "rank1": int((ranks == 1).sum()), "mean_rank": float(ranks.mean())},
                "magnitude": {
                    "mean_norm": float(np.linalg.norm(Al, axis=1).mean()),
                    "median_over_prompt_norm": float(np.median(
                        np.linalg.norm(Al, axis=1) / np.linalg.norm(Pl, axis=1)))},
                "offdiag_cosine": {
                    "raw_mean": float(offdiag(CA_raw).mean()),
                    "raw_sd": float(offdiag(CA_raw).std()),
                    "centred_sd": float(offdiag(CA).std())},
                "half_split_floor_median": floor}

            wo["gram_correlations"][name] = {
                "activation_centred_vs_prompt_centred":
                    float(pearsonr(offdiag(CA), offdiag(CP)).statistic),
                **{f"activation_centred_vs_weight_{k}_centred":
                   float(pearsonr(offdiag(CA), offdiag(W_cos_c[k])).statistic)
                   for k in W_cos_c},
                **{f"activation_raw_vs_weight_{k}_raw":
                   float(pearsonr(offdiag(CA_raw), offdiag(W_cos_raw[k])).statistic)
                   for k in W_cos_raw}}

        # 5. arm against arm in activation space, same trait
        names = sorted(A)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                Ua = unit(A[a][:, wi, PRIMARY])
                Ub = unit(A[b][:, wi, PRIMARY])
                Cab = Ua @ Ub.T
                rk = np.array([1 + int(np.sum(Cab[t] > Cab[t, t])) for t in range(T)])
                Ca = unit(A[a][:, wi, PRIMARY] - A[a][:, wi, PRIMARY].mean(0))
                Cb = unit(A[b][:, wi, PRIMARY] - A[b][:, wi, PRIMARY].mean(0))
                wo["cross_arm"][f"{a}_x_{b}"] = {
                    "same_trait_cos_mean": float(np.diag(Cab).mean()),
                    "cross_trait_cos_mean": float(offdiag(Cab).mean()),
                    "top1": int((rk == 1).sum()), "mean_rank": float(rk.mean()),
                    "cos_between_mean_shifts": float(
                        A[a][:, wi, PRIMARY].mean(0) @ A[b][:, wi, PRIMARY].mean(0)
                        / (np.linalg.norm(A[a][:, wi, PRIMARY].mean(0))
                           * np.linalg.norm(A[b][:, wi, PRIMARY].mean(0)))),
                    "corr_centred_arrangements": float(pearsonr(
                        offdiag(Ca @ Ca.T), offdiag(Cb @ Cb.T)).statistic)}

        # 6. the shared-direction curve over layers, per arm
        curve = {}
        for name in A:
            rows = []
            for l in range(L):
                X = A[name][:, wi, l]
                mu = X.mean(0)
                # layer 0 is the embedding output, which no targeted module
                # touches, so every shift there is exactly zero
                den = np.linalg.norm(X, axis=1) * np.linalg.norm(mu)
                sq = (X ** 2).sum(1).mean()
                if not np.all(den > 0) or sq <= 0:
                    rows.append({"layer": l, "cos_to_mean": None, "share": None,
                                 "note": "zero shift"})
                    continue
                c = (X @ mu) / den
                rows.append({"layer": l, "cos_to_mean": float(c.mean()),
                             "share": float((mu @ mu) / sq)})
            curve[name] = rows
        wo["layer_curve_shared"] = curve
        out["windows_out"][wname] = wo

    p = f"{Q}/analysis/actspace_stage2_geometry.json"
    json.dump(out, open(p, "w"), indent=1)

    for wname, wo in out["windows_out"].items():
        print("=" * 78)
        print(f"WINDOW {wname}, layer {PRIMARY}")
        print(f"{'arm':>9s} {'cos_mu':>7s} {'sd':>6s} {'share':>7s} {'|A|/|P|':>8s} "
              f"{'ownP':>6s} {'othP':>6s} {'rk1':>5s} {'r(A,W1)':>8s} {'r(A,W2)':>8s} "
              f"{'r(A,Wp)':>8s} {'r(A,P)':>7s} {'floor':>6s}")
        for name, d in wo["per_arm"].items():
            g = wo["gram_correlations"][name]
            print(f"{name:>9s} {d['shared_direction']['cos_to_mean_direction_mean']:>7.3f} "
                  f"{d['shared_direction']['cos_to_mean_direction_sd']:>6.3f} "
                  f"{d['shared_direction']['mean_direction_norm2_over_mean_norm2']:>7.3f} "
                  f"{d['magnitude']['median_over_prompt_norm']:>8.2f} "
                  f"{d['vs_prompt_vector']['cos_own_mean']:>6.3f} "
                  f"{d['vs_prompt_vector']['cos_other_mean']:>6.3f} "
                  f"{d['vs_prompt_vector']['rank1']:>5d} "
                  f"{g['activation_centred_vs_weight_stage1_centred']:>8.3f} "
                  f"{g['activation_centred_vs_weight_stage2_centred']:>8.3f} "
                  f"{g['activation_centred_vs_weight_persona_centred']:>8.3f} "
                  f"{g['activation_centred_vs_prompt_centred']:>7.3f} "
                  f"{d['half_split_floor_median']:>6.3f}")
        for k, v in wo["cross_arm"].items():
            print(f"  {k}: same {v['same_trait_cos_mean']:+.3f} cross "
                  f"{v['cross_trait_cos_mean']:+.3f} top1 {v['top1']}/{out['traits']} "
                  f"cos(mean shifts) {v['cos_between_mean_shifts']:+.3f} "
                  f"corr(arrangements) {v['corr_centred_arrangements']:+.3f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
