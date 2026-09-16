"""Text-only baseline: does the Big Five structure exist in the TRAINING TEXT,
before any weights are involved?

The deflationary hypothesis is that our weight-space factor structure was
inherited from the English trait lexicon via the teacher that wrote the data.
This builds the same 100x100 similarity matrix from the text alone -- no model
weights, no gradients -- and factors it identically. If the text already has the
structure, weight space adds nothing. Where the two DIFFER is the model's
contribution.

Prediction registered before running (cartographer, 2026-08-15): text will show
the Big Five clearly, but will NOT reproduce the polarity result, because an
embedding has no natural way to make Extraverted and Introverted point in
opposite directions -- they are about the same topic, so it should call them
similar. The weights call them opposite at -0.34.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

traits = json.load(open(os.path.join(HERE, "traits.json")))
slug = lambda t: t.lower().replace("-", "_")
names = [slug(t["trait"]) for t in traits]
FAC = {slug(t["trait"]): (t["factor"], t["keyed"]) for t in traits}

# ---------------------------------------------------------------- embeddings
from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer(MODEL)

CACHE = os.path.join(R, "text_vecs.npz")
if os.path.exists(CACHE):
    z = np.load(CACHE)
    Vc, Vr = z["chosen"], z["rejected"]
    print("loaded cached embeddings", Vc.shape)
else:
    Vc, Vr = [], []
    for i, n in enumerate(names, 1):
        rows = [json.loads(l) for l in open(os.path.join(HERE, "data_bal", f"{n}.jsonl"))]
        # 64 of 214 pairs is plenty for a stable mean and cuts CPU embedding
        # time ~3x; the same 64 indices for every trait, so nothing differs by
        # sampling.
        rows = rows[:64]
        ec = m.encode([r["chosen"] for r in rows], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=False)
        er = m.encode([r["rejected"] for r in rows], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=False)
        Vc.append(ec.mean(0)); Vr.append(er.mean(0))
        print(f"  {i}/100 {n}", flush=True)
    Vc, Vr = np.array(Vc), np.array(Vr)
    np.savez(CACHE, chosen=Vc, rejected=Vr)
    print("embedded", Vc.shape)

def cosmat(V):
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Vn @ Vn.T

# Three text views. `diff` is the direct analogue of a trait's weight delta:
# what the teacher CHANGED to express the trait, rather than what it wrote.
views = {
    "chosen_only": cosmat(Vc),
    "diff(chosen-rejected)": cosmat(Vc - Vr),
}

# --------------------------------------------------------- weight-space ref
G = np.load(os.path.join(R, "gram.npy"))
meta = json.load(open(os.path.join(R, "gram_names.json")))
gi = {a: i for i, a in enumerate(meta["names"])}
ti = np.array([gi[n] for n in names])
Gw = G[np.ix_(ti, ti)]
one = np.ones((100, 100)) / 100
Gwc = Gw - one @ Gw - Gw @ one + one @ Gw @ one
dw = np.sqrt(np.clip(np.diag(Gwc), 1e-12, None))
Cw = Gwc / np.outer(dw, dw)          # mean-removed weight cosine, as in the PCA

def polarity(C):
    same_pole, opp_pole, diff_fac = [], [], []
    for i in range(100):
        for j in range(i + 1, 100):
            fi, ki = FAC[names[i]]; fj, kj = FAC[names[j]]
            (same_pole if (fi == fj and ki == kj) else
             opp_pole if fi == fj else diff_fac).append(C[i, j])
    return (float(np.mean(same_pole)), float(np.mean(opp_pole)), float(np.mean(diff_fac)))

def rsa(A, B):
    iu = np.triu_indices(100, 1)
    a, b = A[iu], B[iu]
    return float(np.corrcoef(a, b)[0, 1])

print("\n" + "=" * 74)
print(f"{'view':26s} {'same-pole':>10s} {'OPP-pole':>10s} {'diff-factor':>12s} {'RSA vs weights':>15s}")
print("-" * 74)
sp, op, df = polarity(Cw)
print(f"{'WEIGHTS (mean-removed)':26s} {sp:+10.3f} {op:+10.3f} {df:+12.3f} {'--':>15s}")
out = {"weights": {"same_pole": sp, "opposite_pole": op, "diff_factor": df}, "views": {}}
for k, C in views.items():
    sp, op, df = polarity(C)
    r = rsa(C, Cw)
    print(f"{'TEXT: ' + k:26s} {sp:+10.3f} {op:+10.3f} {df:+12.3f} {r:+15.3f}")
    out["views"][k] = {"same_pole": sp, "opposite_pole": op,
                       "diff_factor": df, "rsa_vs_weights": r}
print("=" * 74)
print("\nThe decisive cell is OPP-pole: weights make opposite poles ANTI-correlated.")
print("If a text view does not, the polarity result is not inherited from the text.")
json.dump(out, open(os.path.join(R, "text_baseline.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(R,'text_baseline.json')}")
