#!/usr/bin/env python3
"""Adapter t + constitution s: how do a trained persona and a prompted one compose?

C[t,s] = mean activation(base + adapter t, system = constitution s) - baseline,
against P_s (constitution alone) and A_t (adapter alone) from the single-factor
runs: same prompts, same baseline, same layers.  Pair classes: matched (t = s),
same factor same keying, same factor opposite keying (conflict), different factor.

  ADDITIVITY   how well A_t + P_s predicts C[t,s]; the least-squares weights
               C ~ a A_t + b P_s per pair (a = b = 1 is pure addition).
  SATURATION   matched pairs: the shift along P_t's direction, in units of |P_t|.
               Prompt alone is 1.00 by construction, adapter alone was 0.62.
  CONFLICT     opposite-keying pairs: does C sit nearer the prompt's persona or the
               adapter's?  Cosine to P_s vs A_t, and to the trait vectors of s and t.
  INVARIANCE   is the adapter's contribution the same direction whatever the
               prompt?  cos(C[t,s] - P_s, A_t) across s; and the reverse.
Primary layer 16.
"""
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
PRIMARY = 16


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ZP = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    ZA = np.load(f"{Q}/analysis/actspace_means_adapters.npz", allow_pickle=True)
    ZC = np.load(f"{Q}/analysis/actspace_means_cross.npz", allow_pickle=True)
    all_t = [str(t) for t in ZP["traits"]]
    traits = [str(t) for t in ZC["traits"]]
    idx = [all_t.index(t) for t in traits]
    B = ZP["B"].astype(np.float32).mean(1)                              # (W, L, D)
    P = ZP["M"].astype(np.float32).mean(2)[idx] - B[None]              # (K, W, L, D)
    A = ZA["M"].astype(np.float32).mean(2)[idx] - B[None]
    C = ZC["M"].astype(np.float32).mean(3) - B[None, None]             # (K, K, W, L, D)
    K = len(traits)
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        for r in json.load(open(f"{Q}/{f}")):
            meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    def cls(t, s):
        if t == s: return "matched"
        ft, kt = meta[t]; fs, ks = meta[s]
        if ft != fs: return "different factor"
        return "same factor, same keying" if kt == ks else "same factor, OPPOSITE keying"
    classes = ["matched", "same factor, same keying", "same factor, OPPOSITE keying", "different factor"]

    out = {}
    for wi, wname in enumerate(["resp", "prompt"]):
        l = PRIMARY
        Pl, Al, Cl = P[:, wi, l], A[:, wi, l], C[:, :, wi, l]
        print("=" * 78); print(f"WINDOW {wname}, LAYER {l}"); print("=" * 78)
        rows = {c: [] for c in classes}
        for i, t in enumerate(traits):
            for j, s in enumerate(traits):
                c = Cl[i, j]; a = Al[i]; p = Pl[j]
                X = np.stack([a, p], 1)
                coef, *_ = np.linalg.lstsq(X, c, rcond=None)
                fit = X @ coef
                rec = {
                    "t": t, "s": s,
                    "resid_add": float(np.linalg.norm(c - (a + p)) / np.linalg.norm(c)),
                    "resid_prompt_only": float(np.linalg.norm(c - p) / np.linalg.norm(c)),
                    "resid_adapter_only": float(np.linalg.norm(c - a) / np.linalg.norm(c)),
                    "resid_fit": float(np.linalg.norm(c - fit) / np.linalg.norm(c)),
                    "a": float(coef[0]), "b": float(coef[1]),
                    "norm_ratio": float(np.linalg.norm(c) / np.linalg.norm(p)),
                    "cos_P_s": float(unit(c) @ unit(p)), "cos_A_t": float(unit(c) @ unit(a)),
                    "cos_P_t": float(unit(c) @ unit(Pl[i])), "cos_A_s": float(unit(c) @ unit(Al[j])),
                    "along_P_s": float(c @ unit(p) / np.linalg.norm(p)),
                    "along_P_t": float(c @ unit(Pl[i]) / np.linalg.norm(Pl[i])),
                    "adapter_contrib_cos": float(unit(c - p) @ unit(a)),
                    "prompt_contrib_cos": float(unit(c - a) @ unit(p)),
                }
                rows[cls(t, s)].append(rec)
        med = lambda L_, k: float(np.median([r[k] for r in L_]))
        print(f"{'class':30s} {'n':>3s} {'|C-(A+P)|/|C|':>13s} {'|C-P|/|C|':>10s} {'|C-A|/|C|':>10s} "
              f"{'fit a':>6s} {'fit b':>6s} {'|C|/|P|':>8s}")
        for c in classes:
            L_ = rows[c]
            print(f"{c:30s} {len(L_):>3d} {med(L_,'resid_add'):>13.2f} {med(L_,'resid_prompt_only'):>10.2f} "
                  f"{med(L_,'resid_adapter_only'):>10.2f} {med(L_,'a'):>6.2f} {med(L_,'b'):>6.2f} "
                  f"{med(L_,'norm_ratio'):>8.2f}")
        print(f"\n{'class':30s} {'cos P_s':>8s} {'cos A_t':>8s} {'cos P_t':>8s} {'cos A_s':>8s} "
              f"{'along P_s':>9s} {'along P_t':>9s}  prompt wins")
        for c in classes:
            L_ = rows[c]
            wins = sum(1 for r in L_ if r["cos_P_s"] > r["cos_A_t"])
            print(f"{c:30s} {med(L_,'cos_P_s'):>8.2f} {med(L_,'cos_A_t'):>8.2f} {med(L_,'cos_P_t'):>8.2f} "
                  f"{med(L_,'cos_A_s'):>8.2f} {med(L_,'along_P_s'):>9.2f} {med(L_,'along_P_t'):>9.2f}  "
                  f"{wins}/{len(L_)}")
        m = rows["matched"]
        print(f"\nSATURATION (matched): shift along the trait's own prompt direction, in |P_t| units: "
              f"median {med(m,'along_P_t'):.2f}  (prompt alone 1.00; adapter alone "
              f"{float(np.median([Al[i] @ unit(Pl[i]) / np.linalg.norm(Pl[i]) for i in range(K)])):.2f}; "
              f"additive would be {float(np.median([1 + Al[i] @ unit(Pl[i]) / np.linalg.norm(Pl[i]) for i in range(K)])):.2f})")
        inv_a = [r["adapter_contrib_cos"] for c in classes[1:] for r in rows[c]]
        inv_p = [r["prompt_contrib_cos"] for c in classes[1:] for r in rows[c]]
        print(f"INVARIANCE (t != s): cos(C - P_s, A_t) median {np.median(inv_a):.2f}; "
              f"cos(C - A_t, P_s) median {np.median(inv_p):.2f}")
        # conflict detail
        print("\nCONFLICT pairs, who does the combined model resemble more:")
        for r in sorted(rows["same factor, OPPOSITE keying"], key=lambda r: (r["t"], r["s"]))[:12]:
            print(f"  adapter {r['t']:12s} + prompt {r['s']:12s}: cos to prompt persona {r['cos_P_s']:+.2f}, "
                  f"to adapter persona {r['cos_A_t']:+.2f}, along P_s {r['along_P_s']:+.2f}, along P_t {r['along_P_t']:+.2f}")
        out[wname] = {c: rows[c] for c in classes}

        # ---- TRAIT-SPECIFIC PARTS.  Every prompt shift and every adapter shift
        # shares a large common component (cross-trait cosines of +0.3 to +0.8), so
        # raw projections mostly measure "was the model steered at all".  Remove
        # the mean prompt shift, the mean adapter shift and the mean combined shift
        # and ask the same questions of what is left, which is what distinguishes
        # one trait from another.
        Pt = Pl - Pl.mean(0); At = Al - Al.mean(0); Ct = Cl - Cl.reshape(-1, Cl.shape[-1]).mean(0)
        rows_c = {c: [] for c in classes}
        for i, t in enumerate(traits):
            for j, s_ in enumerate(traits):
                c = Ct[i, j]; a = At[i]; pp = Pt[j]
                X = np.stack([a, pp], 1)
                coef, *_ = np.linalg.lstsq(X, c, rcond=None)
                rows_c[cls(t, s_)].append({
                    "t": t, "s": s_,
                    "resid_add": float(np.linalg.norm(c - (a + pp)) / np.linalg.norm(c)),
                    "resid_fit": float(np.linalg.norm(c - X @ coef) / np.linalg.norm(c)),
                    "a": float(coef[0]), "b": float(coef[1]),
                    "cos_P_s": float(unit(c) @ unit(pp)), "cos_A_t": float(unit(c) @ unit(a)),
                    "along_P_s": float(c @ unit(pp) / np.linalg.norm(pp)),
                    "along_P_t": float(c @ unit(Pt[i]) / np.linalg.norm(Pt[i])),
                    "along_A_t": float(c @ unit(a) / np.linalg.norm(a)),
                    "norm_ratio": float(np.linalg.norm(c) / np.linalg.norm(pp))})
        print("\n--- TRAIT-SPECIFIC PARTS (shared components removed) ---")
        print(f"{'class':30s} {'n':>3s} {'|C-(A+P)|/|C|':>13s} {'fit resid':>9s} {'fit a':>6s} {'fit b':>6s} "
              f"{'cos P_s':>8s} {'cos A_t':>8s} {'along P_s':>9s} {'along P_t':>9s} {'|C|/|P|':>8s}  prompt wins")
        for c in classes:
            L_ = rows_c[c]; wins = sum(1 for r in L_ if r["cos_P_s"] > r["cos_A_t"])
            print(f"{c:30s} {len(L_):>3d} {med(L_,'resid_add'):>13.2f} {med(L_,'resid_fit'):>9.2f} {med(L_,'a'):>6.2f} "
                  f"{med(L_,'b'):>6.2f} {med(L_,'cos_P_s'):>8.2f} {med(L_,'cos_A_t'):>8.2f} {med(L_,'along_P_s'):>9.2f} "
                  f"{med(L_,'along_P_t'):>9.2f} {med(L_,'norm_ratio'):>8.2f}  {wins}/{len(L_)}")
        adapter_alone = float(np.median([At[i] @ unit(Pt[i]) / np.linalg.norm(Pt[i]) for i in range(K)]))
        print(f"matched, trait-specific: along own prompt direction {med(rows_c['matched'],'along_P_t'):.2f} "
              f"(prompt alone 1.00, adapter alone {adapter_alone:.2f}, additive {1 + adapter_alone:.2f})")
        print("conflict pairs, trait-specific:")
        for r in sorted(rows_c["same factor, OPPOSITE keying"], key=lambda r: (r["t"], r["s"])):
            print(f"  adapter {r['t']:12s} + prompt {r['s']:12s}: along prompt's trait {r['along_P_s']:+.2f}, "
                  f"along adapter's trait {r['along_P_t']:+.2f}, cos P_s {r['cos_P_s']:+.2f} cos A_t {r['cos_A_t']:+.2f}")
        out[wname + "_specific"] = {c: rows_c[c] for c in classes}
        out[wname + "_specific_adapter_alone"] = adapter_alone
    json.dump(out, open(f"{Q}/analysis/actspace_cross_geometry.json", "w"))
    print("\nwrote analysis/actspace_cross_geometry.json")


if __name__ == "__main__":
    main()
