"""Does weight space carry trait structure BEYOND the text that described it?

This is the weakest joint in the project, and it exists because two established
results are compatible in an uncomfortable way.

  (a) The between-trait weight geometry reproduces across three independent LoRA
      initialisations at RSA 0.996.  So it is not an artefact of one random slice.
  (b) Sentence embeddings of the training pairs reproduce that same geometry at
      RSA 0.83.  So most of it needs no weights at all.

(a) does not rescue us from (b), because all three seeds trained on the SAME
TEXT.  A weight geometry that faithfully re-encoded the teacher's writing
statistics would reproduce across initialisations exactly as well as one that
encoded personality.  Nothing measured so far separates the two.

So: run the IDENTICAL factor pipeline on both similarity matrices and compare
what each recovers of the Goldberg structure.  Same extraction, same rotation,
same targets, same code path -- the only difference is whether the matrix came
from weights or from text.

  If WEIGHTS recover the marker structure better than TEXT, then training put
  trait information into the weights that was not already in the sentences, and
  the project's central object is what it claims to be.
  If TEXT does as well or better, the weight geometry is a re-encoding of the
  training corpus and the honest headline changes.

Cheap, CPU-only, no new training: everything it needs is already on disk.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import analyse_fa as fa      # noqa: E402  (import-safe: guarded by __main__)

FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
K = 5

traits = json.load(open(os.path.join(HERE, "traits.json")))
slug = lambda t: t["trait"].lower().replace("-", "_")
names = [slug(t) for t in traits]
factor = np.array([t["factor"] for t in traits])
pole = np.array([1.0 if t["keyed"] == "+" else -1.0 for t in traits])
p = len(names)

# ---- targets: identical for both matrices, built exactly as analyse_fa does --
targets = np.zeros((p, 6))
for a, F in enumerate(FACTORS):
    targets[:, a] = np.where(factor == F, pole, 0.0)
targets[:, 5] = pole                       # the general evaluative target

# ---- matrix 1: WEIGHTS -------------------------------------------------------
G = np.load(os.path.join(RDIR, "gram.npy"))
meta = json.load(open(os.path.join(RDIR, "gram_names.json")))
gi = {a: i for i, a in enumerate(meta["names"])}
idx = np.array([gi[n] for n in names])
Gw = G[np.ix_(idx, idx)]
one = np.ones((p, p)) / p
Gwc = Gw - one @ Gw - Gw @ one + one @ Gw @ one     # double-centre, as the PCA does
d = np.sqrt(np.clip(np.diag(Gwc), 1e-12, None))
R_weights = Gwc / np.outer(d, d)

# ---- matrix 2: TEXT ----------------------------------------------------------
# The difference view (preferred minus rejected), which the text baseline showed
# is the informative one; chosen-only carries almost no structure.
z = np.load(os.path.join(RDIR, "text_vecs.npz"))
D = z["chosen"] - z["rejected"]
D = D / np.linalg.norm(D, axis=1, keepdims=True)
Rt = D @ D.T
# Centre it the same way the weight matrix is centred, so the two are treated
# identically -- otherwise the comparison rewards a preprocessing choice.
Rtc = Rt - one @ Rt - Rt @ one + one @ Rt @ one
dt = np.sqrt(np.clip(np.diag(Rtc), 1e-12, None))
R_text = Rtc / np.outer(dt, dt)

print(f"{p} traits; weight and text matrices built and centred identically")
print(f"RSA between them: "
      f"{np.corrcoef(R_weights[np.triu_indices(p,1)], R_text[np.triu_indices(p,1)])[0,1]:+.3f}")

# ---- identical pipeline on each ----------------------------------------------
out = {}
for label, R in (("weights", R_weights), ("text", R_text)):
    sol = fa.solution(R, K, targets, label=label)
    cong = np.array(sol["congruence_oblimin"])          # (k, 6)
    best = [float(np.max(np.abs(cong[:, a]))) for a in range(5)]
    out[label] = {"congruence_oblimin": cong.tolist(),
                  "best_per_factor": best,
                  "mean_best": float(np.mean(best)),
                  "n_above_085": int(sum(b >= 0.85 for b in best)),
                  "ss_loadings": sol["ss_loadings"]["oblimin"],
                  "paf_converged": sol["paf_converged"],
                  "n_heywood": sol["n_heywood"]}
    print(f"  {label:8s} PAF converged={sol['paf_converged']} "
          f"heywood={sol['n_heywood']}")

print("\n" + "=" * 72)
print("Tucker congruence with each Goldberg marker target (oblimin, k=5)")
print("best-matching factor per target; 0.85 = 'fair', 0.95 = 'equivalent'")
print("-" * 72)
print(f"{'target':<22}{'WEIGHTS':>12}{'TEXT':>12}{'difference':>14}")
print("-" * 72)
for a, F in enumerate(FACTORS):
    w, t = out["weights"]["best_per_factor"][a], out["text"]["best_per_factor"][a]
    print(f"{F:<22}{w:>12.3f}{t:>12.3f}{w-t:>+14.3f}")
print("-" * 72)
mw, mt = out["weights"]["mean_best"], out["text"]["mean_best"]
print(f"{'mean':<22}{mw:>12.3f}{mt:>12.3f}{mw-mt:>+14.3f}")
print("=" * 72)
print(f"""
Reading: if WEIGHTS > TEXT, training added trait structure the sentences did not
already carry. If TEXT >= WEIGHTS, the weight geometry is a re-encoding of the
training corpus and the project's central claim needs restating.
weights recovered {out['weights']['n_above_085']}/5 targets at the 0.85 bar,
text recovered {out['text']['n_above_085']}/5.""")

json.dump(out, open(os.path.join(RDIR, "text_vs_weights_fa.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(RDIR, 'text_vs_weights_fa.json')}")
