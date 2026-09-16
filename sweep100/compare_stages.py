"""Does the persona-sticking stage move where a persona LIVES in weight space?

Stage 1 is DPO on teacher-written contrast pairs.  Stage 2 (Open Character
Training's introspection step) fine-tunes on the model's own self-interaction and
self-reflection transcripts with stage 1 frozen, and is the stage the paper
credits with making a character stick rather than wash out.

The question promised on 2026-08-14 and unpaid until now: run the SAME
decomposition over the stage-1 deltas and over the stage-1-plus-stage-2 sums, and
see whether they disagree.  Agreement would mean stage 2 deepens a persona
without relocating it.  Disagreement would be the more interesting result -- the
stage that makes a character durable would also be moving it, and every
structural claim we have made from stage-1 geometry would be a claim about an
intermediate state.

Everything is held fixed except which adapters are read: same Gram code, same
factor pipeline, same Goldberg targets, same centring.  The text baseline rides
along, because the live question about this project is whether weight geometry
carries trait structure beyond the text that produced it -- and if stage 2 helps
anywhere, this is where it should show, since stage 2 trains on the model's OWN
generations rather than on the teacher's sentences.

usage:  python compare_stages.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import analyse_pca as ap     # noqa: E402
import analyse_fa as fa      # noqa: E402

FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]
K = 5

traits = json.load(open(os.path.join(HERE, "traits.json")))
slug = lambda t: t["trait"].lower().replace("-", "_")
names = [slug(t) for t in traits]
factor = np.array([t["factor"] for t in traits])
pole = np.array([1.0 if t["keyed"] == "+" else -1.0 for t in traits])

# only traits present in BOTH corpora, so the two matrices describe one set
S1, ST = os.path.join(HERE, "adapters"), os.path.join(HERE, "adapters_stacked")
use = [n for n in names if os.path.isdir(os.path.join(S1, n))
       and os.path.isdir(os.path.join(ST, n))]
p = len(use)
keep = [names.index(n) for n in use]
factor, pole = factor[keep], pole[keep]
print(f"{p} traits present in both stage-1 and stacked corpora")

targets = np.zeros((p, 6))
for a, F in enumerate(FACTORS):
    targets[:, a] = np.where(factor == F, pole, 0.0)
targets[:, 5] = pole

one = np.ones((p, p)) / p


def centred_cos(G):
    Gc = G - one @ G - G @ one + one @ G @ one
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    return Gc / np.outer(d, d)


mats, spec = {}, {}
for label, adir in (("stage1", S1), ("stacked", ST)):
    ap.ADIR = adir
    G, mods, scale, r = ap.gram_streaming(use, log=open(os.devnull, "w"))
    C = centred_cos(G)
    mats[label] = C
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    spec[label] = {"var_pct": (100 * ev[:6] / ev.sum()).tolist(),
                   "rank": r, "scale": scale, "n_modules": len(mods),
                   "mean_norm": float(np.mean(np.sqrt(np.diag(G))))}
    print(f"  {label:8s} rank={r} scale={scale} modules={len(mods)} "
          f"mean||dW||={spec[label]['mean_norm']:.3f}")

# ---- text matrix, on the same trait subset ---------------------------------
z = np.load(os.path.join(RDIR, "text_vecs.npz"))
D = (z["chosen"] - z["rejected"])[keep]
D = D / np.linalg.norm(D, axis=1, keepdims=True)
mats["text"] = centred_cos(D @ D.T)

iu = np.triu_indices(p, 1)
rsa = lambda a, b: float(np.corrcoef(mats[a][iu], mats[b][iu])[0, 1])

out = {"n_traits": p, "spectrum": spec, "rsa": {
    "stage1_vs_stacked": rsa("stage1", "stacked"),
    "stage1_vs_text": rsa("stage1", "text"),
    "stacked_vs_text": rsa("stacked", "text")}}

cong = {}
for label in ("stage1", "stacked", "text"):
    sol = fa.solution(mats[label], K, targets, label=label)
    c = np.array(sol["congruence_oblimin"])
    best = [float(np.max(np.abs(c[:, a]))) for a in range(5)]
    cong[label] = {"best_per_factor": best, "mean_best": float(np.mean(best)),
                   "n_above_085": int(sum(b >= 0.85 for b in best))}
out["congruence"] = cong

print("\n" + "=" * 74)
print("DOES STAGE 2 MOVE THE GEOMETRY?")
print(f"  RSA stage1 vs stacked : {out['rsa']['stage1_vs_stacked']:+.3f}")
print(f"  variance stage1  : " + " ".join(f"{v:5.1f}" for v in spec["stage1"]["var_pct"]))
print(f"  variance stacked : " + " ".join(f"{v:5.1f}" for v in spec["stacked"]["var_pct"]))
print("\nBIG FIVE RECOVERY (Tucker congruence, oblimin k=5, best factor per target)")
print("-" * 74)
print(f"{'target':<22}{'stage1':>11}{'stacked':>11}{'text':>11}")
print("-" * 74)
for a, F in enumerate(FACTORS):
    print(f"{F:<22}" + "".join(f"{cong[l]['best_per_factor'][a]:>11.3f}"
                               for l in ("stage1", "stacked", "text")))
print("-" * 74)
print(f"{'mean':<22}" + "".join(f"{cong[l]['mean_best']:>11.3f}"
                                for l in ("stage1", "stacked", "text")))
print("=" * 74)
print(f"""
RSA of each weight geometry with the TEXT geometry:
  stage1  {out['rsa']['stage1_vs_text']:+.3f}     stacked {out['rsa']['stacked_vs_text']:+.3f}
If stacked is FURTHER from text than stage1 is, stage 2 added structure that the
teacher's sentences do not contain -- which is the one thing that would rescue
weight space from this morning's null.""")

json.dump(out, open(os.path.join(RDIR, "compare_stages.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(RDIR, 'compare_stages.json')}")
