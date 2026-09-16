#!/usr/bin/env python3
"""Weight-space geometry of the persona adapters, and its relation to Big Five.

Answers, in weight space: is there a low-dimensional structure that lines up
with the Big Five factors the traits were chosen from, and does an adapter's
position along a factor's axis predict how much of that factor it behaviourally
expresses?

Statistical care, because this regime punishes carelessness:

* n ~ 100 adapters in ~254k sketch dimensions. The covariance has rank <= n-1,
  so "whiten the whole space" is meaningless -- the trailing directions are
  pure noise and whitening would amplify them to the same scale as signal.
  Everything is therefore done inside a PCA basis of k << n components.

  k was SWEPT (5,10,20,30,40,50,70,99; see analysis/geometry_k_sweep.json) and
  the two headline results behave differently, so they are reported in different
  bases on purpose:
    - keying classification is stable UNWHITENED at 0.98 for every k, but
      whitened it decays 0.99 -> 0.14 as k grows, falling BELOW chance at k=99.
      That is the predicted noise amplification, measured rather than assumed.
    - factor-axis orthogonality only appears WHITENED, where it beats a
      shuffled-keying null at p<=0.01 for every k from 5 to 70. Unwhitened it
      never beats the null except marginally at k=5.
  So: read classification unwhitened, read orthogonality whitened. At k=99
  (= n-1) whitening is fully degenerate and both collapse.
* Every separability claim is LEAVE-ONE-TRAIT-OUT cross-validated. In-sample
  separation at d >> n is guaranteed and means nothing.
* Factor-axis angles get a permutation null: with 100 points in high dimensions,
  random directions are near-orthogonal by default, so "the axes are orthogonal"
  is only interesting against the right null.
* All adapters share LoRA seed 0. Within this initialisation the geometry is
  comparable; ACROSS seeds it is not (same-trait cross-seed cosine ~0.017 vs a
  ~0.0013 different-trait floor). Any direction found here is a fact about this
  initialisation, not a transferable steering vector. This caveat is printed
  into the results file, not just written in a card.
"""
import argparse, glob, json, os
import numpy as np

Q = "/home/vibe12/projects/persona-curvature/qwen35"
SEED0_CAVEAT = ("All adapters share LoRA init seed 0. Geometry is valid WITHIN this "
                "initialisation only; cross-seed same-trait cosine is ~0.017 against a "
                "~0.0013 different-trait floor, so these directions do not transfer.")


def load(sk_dir):
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = os.path.join(Q, f)
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
    names, X, F, K = [], [], [], []
    for p in sorted(glob.glob(f"{sk_dir}/*.npz")):
        t = os.path.basename(p)[:-4]
        if t not in meta:
            continue
        z = np.load(p, allow_pickle=True)
        names.append(t); X.append(z["sketch"].astype(np.float64))
        F.append(meta[t][0]); K.append(meta[t][1])
    return names, np.stack(X), np.array(F), np.array(K)


def pca(X, k):
    """Center, then PCA via the Gram matrix (n x n) -- d is far larger than n."""
    mu = X.mean(0)
    Xc = X - mu
    G = Xc @ Xc.T
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:k]
    tot = float(np.maximum(w, 0).sum())   # TOTAL variance, before truncation
    w, V = np.maximum(w[idx], 1e-12), V[:, idx]
    Z = V * np.sqrt(w)              # scores, (n, k) -- unwhitened
    Zw = V * np.sqrt(len(X) - 1)    # whitened scores (unit variance per component)
    return Z, Zw, w, Xc, tot


def factor_axes(Z, F, K):
    """Per-factor direction = mean(+keyed) - mean(-keyed), in the given basis."""
    ax, ns = {}, {}
    for f in sorted(set(F)):
        p = Z[(F == f) & (K == "+")]
        m = Z[(F == f) & (K == "-")]
        if len(p) < 2 or len(m) < 2:
            continue
        v = p.mean(0) - m.mean(0)
        n = np.linalg.norm(v)
        if n > 0:
            ax[f] = v / n
            ns[f] = (len(p), len(m), float(n))
    return ax, ns


def loo_keying(Z, F, K, rng, nperm=2000):
    """Leave-one-out: can position predict keying WITHIN a factor? Permutation null."""
    out = {}
    for f in sorted(set(F)):
        sel = F == f
        Zf, y = Z[sel], (K[sel] == "+").astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        def acc(lbl):
            c = 0
            for i in range(len(Zf)):
                tr = np.ones(len(Zf), bool); tr[i] = False
                a, b = Zf[tr][lbl[tr] == 1], Zf[tr][lbl[tr] == 0]
                if len(a) < 2 or len(b) < 2:
                    return np.nan
                w = a.mean(0) - b.mean(0)
                thr = (a.mean(0) + b.mean(0)) / 2 @ w
                c += int((Zf[i] @ w > thr) == (lbl[i] == 1))
            return c / len(Zf)
        obs = acc(y)
        null = np.array([acc(rng.permutation(y)) for _ in range(nperm // 20)])
        p = float((np.nansum(null >= obs) + 1) / (len(null) + 1))
        out[f] = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                      loo_acc=round(float(obs), 3),
                      null_mean=round(float(np.nanmean(null)), 3), p=round(p, 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sketches", default=f"{Q}/analysis/sketches/stage1_k32")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--out", default=f"{Q}/analysis/geometry_stage1.json")
    ap.add_argument("--behaviour", default=None, help="judge scores json (trait -> {factor: z})")
    a = ap.parse_args()

    names, X, F, K = load(a.sketches)
    n = len(names)
    print(f"{n} adapters, sketch dim {X.shape[1]}")
    if n < 20:
        print("too few adapters for geometry; run the sketcher first")
    rng = np.random.default_rng(0)

    Z, Zw, w, Xc, tot = pca(X, min(a.k, n - 1))
    # normalise by TOTAL variance, not the retained slice -- dividing by the
    # truncated sum reports 21.5% where the real figure is 13.4%.
    ev = w / tot
    res = dict(n=n, dim=int(X.shape[1]), k=int(Z.shape[1]),
               caveat=SEED0_CAVEAT,
               explained_var_top10=[round(float(x), 4) for x in ev[:10]],
               cum_var_at_k=round(float(ev[:Z.shape[1]].sum()), 4))
    print("top-10 explained variance:", res["explained_var_top10"])

    # cosine structure in the raw sketch space
    Xn = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
    C = Xn @ Xn.T
    iu = np.triu_indices(n, 1)
    same = np.array([F[i] == F[j] for i, j in zip(*iu)])
    res["cos_same_factor_mean"] = round(float(C[iu][same].mean()), 4)
    res["cos_diff_factor_mean"] = round(float(C[iu][~same].mean()), 4)

    for label, B in (("unwhitened", Z), ("whitened", Zw)):
        ax, ns = factor_axes(B, F, K)
        ang = {f"{f1}|{f2}": round(float(abs(ax[f1] @ ax[f2])), 4)
               for i, f1 in enumerate(sorted(ax)) for f2 in sorted(ax)[i + 1:]}
        res[f"{label}_axis_abscos"] = ang
        res[f"{label}_axis_counts"] = {f: ns[f] for f in ns}
        res[f"{label}_loo_keying"] = loo_keying(B, F, K, rng)
        # Null for the axis angles. Whitening makes directions look orthogonal
        # by construction, and in k dims random vectors already sit near
        # E|cos| ~ sqrt(2/(pi*k)). Without this the number means nothing.
        if ang:
            nulls = []
            for _ in range(400):
                Kp = K.copy()
                for f in set(F):
                    m = F == f
                    Kp[m] = rng.permutation(Kp[m])
                axp, _ = factor_axes(B, F, Kp)
                if len(axp) > 1:
                    nulls.append(np.mean([abs(axp[c] @ axp[d])
                                          for i, c in enumerate(sorted(axp))
                                          for d in sorted(axp)[i + 1:]]))
            obs = float(np.mean(list(ang.values())))
            res[f"{label}_axis_abscos_mean"] = round(obs, 4)
            res[f"{label}_axis_null_mean"] = round(float(np.mean(nulls)), 4)
            res[f"{label}_axis_p"] = round(float((np.sum(np.array(nulls) <= obs) + 1)
                                                 / (len(nulls) + 1)), 4)
            print(f"  axis |cos| {obs:.4f} vs shuffled-keying null "
                  f"{np.mean(nulls):.4f}  p(more orthogonal than null)="
                  f"{res[f'{label}_axis_p']}")
        print(f"\n[{label}] mean |cos| between factor axes: "
              f"{np.mean(list(ang.values())):.4f}" if ang else f"\n[{label}] no axes")
        for f, d in res[f"{label}_loo_keying"].items():
            print(f"  {f:20s} LOO acc {d['loo_acc']:.3f} (null {d['null_mean']:.3f}) p={d['p']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
