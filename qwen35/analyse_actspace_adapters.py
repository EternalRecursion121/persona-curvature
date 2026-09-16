#!/usr/bin/env python3
"""Does fine-tuning on a constitution's data reproduce, in activation space, what
the constitution does as a prompt?  And what does it do to the geometry?

Three shifts per trait t, per layer, all relative to the same no-system baseline:
  P_t   constitution as system prompt      (act_space.py --stage prompt)
  A_t   base + stage-1 adapter, no prompt  (act_space.py --stage adapters)
  W_t   the adapter's weight delta          (results/gram_sweep.npz, exact Gram)

  1. INTERNALISATION  cos(P_t, A_t) per layer, against cos(P_t, A_s) for s != t.
     Rank of the own trait among 134 is the identification test; the fraction
     |proj of A_t on P_t| / |P_t| is how much of the prompt's shift the adapter
     delivers.
  2. GEOMETRY  correlations of the three 134 x 134 cosine matrices, trait-centred.
     If A's Gram tracks P's, fine-tuning preserved the prompt geometry; if it
     tracks W's more closely, the adapter cloud in activations mirrors the
     adapter cloud in weights.
  3. CONTAINMENT  the fraction of each A_t's variance inside the top-k subspace of
     the prompt shifts {P_s}: is the adapter's effect made of the same directions
     the prompts use?
  4. MAGNITUDE  |A_t| / |P_t| by layer.

Primary layer 16, fixed in advance, as in analyse_actspace.py.
"""
import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY = 16
rng = np.random.default_rng(0)


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def offdiag(C):
    return C[~np.eye(C.shape[0], dtype=bool)]


def main():
    ZP = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    ZA = np.load(f"{Q}/analysis/actspace_means_adapters.npz", allow_pickle=True)
    traits = [str(t) for t in ZP["traits"]]
    assert traits == [str(t) for t in ZA["traits"]]
    windows = [str(w) for w in ZP["windows"]]
    B = ZP["B"].astype(np.float32)                      # (W, 2, L, D)
    P = ZP["M"].astype(np.float32).mean(2) - B.mean(1)[None]   # (T, W, L, D)
    A = ZA["M"].astype(np.float32).mean(2) - B.mean(1)[None]
    Ph = ZP["M"].astype(np.float32) - B[None]           # halves kept, (T, W, 2, L, D)
    Ah = ZA["M"].astype(np.float32) - B[None]
    T, W, L, D = P.shape
    G = np.load(f"{Q}/results/gram_sweep.npz", allow_pickle=True)
    wn = [str(x) for x in G["names"]]
    o = [wn.index(t) for t in traits]
    Gw = np.array(G["G"])[np.ix_(o, o)]
    Cw = Gw / np.outer(np.sqrt(np.diag(Gw)), np.sqrt(np.diag(Gw)))
    Cw_c = None  # weight cosines are already of deltas; centring is done via the Gram below
    n = T; J = np.eye(n) - np.ones((n, n)) / n
    Gc = J @ Gw @ J; d = np.sqrt(np.maximum(np.diag(Gc), 1e-12)); Cw_c = Gc / np.outer(d, d)

    out = {"primary_layer": PRIMARY, "windows": {}}
    for wi, wname in enumerate(windows):
        print("=" * 78); print(f"WINDOW: {wname}"); print("=" * 78)
        print(f"{'layer':>5s} {'cos(P,A) own':>13s} {'other':>7s} {'rank1':>6s} {'frac':>6s} "
              f"{'|A|/|P|':>8s} {'r(A,P)':>7s} {'r(A,W)':>7s} {'r(P,W)':>7s} {'floorA':>7s}")
        curve = []
        for l in range(L):
            Pl, Al = P[:, wi, l], A[:, wi, l]
            Pu, Au = unit(Pl), unit(Al)
            C = Au @ Pu.T                                    # C[t, s] = cos(A_t, P_s)
            own = np.diag(C)
            other = offdiag(C)
            ranks = np.array([1 + np.sum(C[t] > C[t, t]) for t in range(T)])
            frac = np.einsum("td,td->t", Al, Pu) / np.linalg.norm(Pl, axis=1)
            mag = np.linalg.norm(Al, axis=1) / np.linalg.norm(Pl, axis=1)
            Pc, Ac = Pl - Pl.mean(0), Al - Al.mean(0)
            CA, CP = unit(Ac) @ unit(Ac).T, unit(Pc) @ unit(Pc).T
            r_ap = pearsonr(offdiag(CA), offdiag(CP)).statistic
            r_aw = pearsonr(offdiag(CA), offdiag(Cw_c)).statistic
            r_pw = pearsonr(offdiag(CP), offdiag(Cw_c)).statistic
            hA = unit(Ah[:, wi, 0, l]); hB = unit(Ah[:, wi, 1, l])
            floorA = float(np.median(np.sum(hA * hB, 1)))
            row = {"layer": l, "cos_own": float(own.mean()), "cos_other": float(other.mean()),
                   "rank1": int((ranks == 1).sum()), "mean_rank": float(ranks.mean()),
                   "frac_internalised": float(np.median(frac)), "mag_ratio": float(np.median(mag)),
                   "r_AP": float(r_ap), "r_AW": float(r_aw), "r_PW": float(r_pw), "floor_A": floorA}
            curve.append(row)
            print(f"{l:>5d} {own.mean():>13.3f} {other.mean():>7.3f} {row['rank1']:>6d} "
                  f"{row['frac_internalised']:>6.2f} {row['mag_ratio']:>8.2f} {r_ap:>7.3f} {r_aw:>7.3f} "
                  f"{r_pw:>7.3f} {floorA:>7.3f}{'  <- primary' if l == PRIMARY else ''}")

        # containment at the primary layer
        l = PRIMARY
        Pc = P[:, wi, l] - P[:, wi, l].mean(0); Ac = A[:, wi, l] - A[:, wi, l].mean(0)
        U, s, Vt = np.linalg.svd(Pc, full_matrices=False)
        cont = {}
        for k in (5, 10, 20, 40):
            Vk = Vt[:k]
            proj = Ac @ Vk.T
            cont[k] = float((proj ** 2).sum() / (Ac ** 2).sum())
        # null: containment of A in the top-k of a random 134-vector cloud with P's spectrum
        R = rng.standard_normal((T, D)); R = R / np.linalg.norm(R, axis=1, keepdims=True)
        R = (R - R.mean(0))
        Ur, sr, Vtr = np.linalg.svd(R, full_matrices=False)
        cont_null = {k: float(((Ac @ Vtr[:k].T) ** 2).sum() / (Ac ** 2).sum()) for k in (5, 10, 20, 40)}
        prim = curve[PRIMARY]
        print(f"\nPRIMARY LAYER {PRIMARY}")
        print(f"  own-trait cos(P,A) {prim['cos_own']:+.3f} vs other-trait {prim['cos_other']:+.3f}; "
              f"own adapter rank 1 for {prim['rank1']}/{T} (mean rank {prim['mean_rank']:.2f})")
        print(f"  median fraction of the prompt shift delivered by the adapter: {prim['frac_internalised']:.2f}; "
              f"median |A|/|P| {prim['mag_ratio']:.2f}")
        print(f"  Gram correlations (centred): A~P {prim['r_AP']:+.3f}   A~W {prim['r_AW']:+.3f}   P~W {prim['r_PW']:+.3f}")
        print("  containment of adapter shifts in the prompt-shift subspace: "
              + "  ".join(f"top-{k} {cont[k]:.2f} (random {cont_null[k]:.2f})" for k in cont))
        out["windows"][wname] = {"curve": curve, "containment": cont, "containment_null": cont_null}
    json.dump(out, open(f"{Q}/analysis/actspace_adapters_geometry.json", "w"), indent=1)
    print("\nwrote analysis/actspace_adapters_geometry.json")


if __name__ == "__main__":
    main()
