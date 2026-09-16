"""PCA across a collection of LoRA adapters.

Treats each adapter's effective delta (dW = (alpha/r) B A, concatenated over all
target modules) as ONE data point, and asks which directions explain the
variance ACROSS adapters. Never materialises dW: everything is a function of
pairwise Frobenius inner products, computed per module from the factors and
summed.

Per module, with Bstack = [B_1 ... B_n] (out, n*r) and Astack = [A_1; ...; A_n]
(n*r, in):   <dW_i, dW_j> = s_i s_j * sum( (B_i^T B_j) * (A_i A_j^T) )
so both Gram blocks come from two BLAS calls per module rather than n^2 loops.
"""
import json, os, itertools
import numpy as np
from safetensors.numpy import load_file

ADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")
SING = ["O", "C", "E", "A", "N"]
PAIRS = ["O_C", "O_E", "O_A", "O_N", "C_E", "C_A", "C_N", "E_A", "E_N", "A_N"]
UNION = [p + "_union" for p in PAIRS]
CTRL = ["O_h1", "O_h2", "C_h1", "C_h2", "E_h1", "E_h2", "O_s1", "C_s1", "E_s1"]
ALL = SING + PAIRS + UNION + CTRL


def load(a):
    cfg = json.load(open(f"{ADIR}/{a}/adapter_config.json"))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora")
    s = cfg["lora_alpha"] / cfg["r"]
    t = load_file(f"{ADIR}/{a}/adapter_model.safetensors")
    m = {}
    for ka in [k for k in t if "lora_A" in k]:
        key = ka.replace(".lora_A.weight", "").replace("base_model.model.", "")
        m[key] = (t[ka].astype(np.float64), t[ka.replace("lora_A", "lora_B")].astype(np.float64), s)
    return m


def gram(names, D, mods):
    n = len(names)
    G = np.zeros((n, n))
    for mi, m in enumerate(mods):
        A0, B0, _ = D[names[0]][m]
        r = A0.shape[0]
        Bst = np.concatenate([D[a][m][1] for a in names], axis=1)      # (out, n*r)
        Ast = np.concatenate([D[a][m][0] for a in names], axis=0)      # (n*r, in)
        BtB = Bst.T @ Bst                                              # (n*r, n*r)
        AAt = Ast @ Ast.T                                              # (n*r, n*r)
        M = BtB * AAt
        # block-sum the (r,r) blocks
        blk = M.reshape(n, r, n, r).sum(axis=(1, 3))
        sc = np.array([D[a][m][2] for a in names])
        G += blk * np.outer(sc, sc)
    return G


def pca_report(G, names, idx, label):
    g = G[np.ix_(idx, idx)]
    k = len(idx)
    H = np.eye(k) - np.ones((k, k)) / k
    gc = H @ g @ H
    ev = np.clip(np.linalg.eigvalsh(gc)[::-1], 0, None)
    evu = np.clip(np.linalg.eigvalsh(g)[::-1], 0, None)
    print(f"\n=== {label}  (n={k}) ===")
    print("  UNCENTERED  PC var%:", [round(100 * x / evu.sum(), 1) for x in evu[:6]])
    print("  CENTERED    PC var%:", [round(100 * x / max(ev.sum(), 1e-30), 1) for x in ev[:6]])
    c = np.cumsum(ev) / max(ev.sum(), 1e-30)
    print("  centered cumulative:", ", ".join(f"PC1-{i+1} {100*c[i]:.0f}%" for i in range(min(4, k - 1))))
    return ev


def main():
    names = [a for a in ALL if os.path.isdir(f"{ADIR}/{a}")]
    missing = [a for a in ALL if a not in names]
    if missing:
        print("MISSING adapters (skipped):", missing)
    D = {a: load(a) for a in names}
    mods = sorted(set.intersection(*[set(v) for v in D.values()]))
    print(f"{len(names)} adapters, {len(mods)} shared modules")
    G = gram(names, D, mods)
    np.save("/tmp/gram.npy", G)
    json.dump(names, open("/tmp/gram_names.json", "w"))
    I = {a: i for i, a in enumerate(names)}

    pca_report(G, names, [I[a] for a in SING if a in I], "5 OCEAN singles")
    pca_report(G, names, [I[a] for a in SING + PAIRS if a in I], "singles + compositional pairs")
    pca_report(G, names, [I[a] for a in SING + PAIRS + UNION if a in I], "singles + compositional + union")

    def d(a, b):
        return float(np.sqrt(max(G[I[a], I[a]] + G[I[b], I[b]] - 2 * G[I[a], I[b]], 0)))

    print("\n=== scale check (is any of this above training noise?) ===")
    rep = [d("O_h1", "O_h2"), d("C_h1", "C_h2"), d("E_h1", "E_h2")]
    res = [d("O", "O_s1"), d("C", "C_s1"), d("E", "E_s1")]
    cross = [d(a, b) for a, b in itertools.combinations([x for x in SING if x in I], 2)]
    print(f"  half-split replicate distance: mean {np.mean(rep):.3f}")
    print(f"  reseed replicate distance:     mean {np.mean(res):.3f}")
    print(f"  cross-trait distance:          mean {np.mean(cross):.3f}")
    print(f"  ratio cross/reseed: {np.mean(cross)/np.mean(res):.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
