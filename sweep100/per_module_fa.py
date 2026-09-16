"""Does the trait structure live in the aggregate, or in particular modules?

Everything in this project compares adapters through ONE number per pair: the
Frobenius inner product of the whole weight delta, which is the sum over 252
modules of each module's contribution.  Summing inner products per module IS the
inner product of the concatenation, so "one object" means concatenation, not
averaging -- and concatenation weights each module by how much it moved.

Measured: up_proj and gate_proj carry 74% of the squared norm between them, and
all four attention projections carry 17%.  So the geometry we have been calling
"trait structure in weight space" is mostly two MLP projection types, by
construction and without anyone choosing it.

This asks whether that matters: run the identical factor pipeline on each module
type separately and see whether the Big Five recovery is a property of the whole
object or of particular parts -- and whether the text null survives the cut.
"""
import json, os, sys, collections
import numpy as np, torch
from safetensors import safe_open
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_fa as fa

FACTORS = ["Extraversion","Agreeableness","Conscientiousness","EmotionalStability","Intellect"]
ADIR = os.path.join(HERE, "adapters")
traits = json.load(open(os.path.join(HERE, "traits.json")))
slug = lambda t: t["trait"].lower().replace("-", "_")
names = [slug(t) for t in traits if os.path.isdir(os.path.join(ADIR, slug(t)))]
p = len(names)
keep = [i for i, t in enumerate(traits) if slug(t) in set(names)]
factor = np.array([t["factor"] for t in traits])[keep]
pole = np.array([1.0 if t["keyed"] == "+" else -1.0 for t in traits])[keep]
targets = np.zeros((p, 6))
for a, F in enumerate(FACTORS):
    targets[:, a] = np.where(factor == F, pole, 0.0)
targets[:, 5] = pole
one = np.ones((p, p)) / p

hs = [safe_open(f"{ADIR}/{n}/adapter_model.safetensors", framework="pt") for n in names]
sc = np.array([json.load(open(f"{ADIR}/{n}/adapter_config.json"))["lora_alpha"] /
               json.load(open(f"{ADIR}/{n}/adapter_config.json"))["r"] for n in names])
mods = sorted({k.replace(".lora_A.weight", "") for k in hs[0].keys() if "lora_A" in k})
by_type = collections.defaultdict(list)
for m in mods:
    by_type[m.split(".")[-1]].append(m)
print(f"{p} traits, {len(mods)} modules, {len(by_type)} types")

def gram(mlist):
    G = torch.zeros(p, p, dtype=torch.float64)
    for m in mlist:
        A = [h.get_tensor(m + ".lora_A.weight").double() for h in hs]
        B = [h.get_tensor(m + ".lora_B.weight").double() for h in hs]
        for i in range(p):
            for j in range(i, p):
                v = float(((B[i].T @ B[j]) * (A[i] @ A[j].T)).sum())
                G[i, j] += v
                if i != j: G[j, i] += v
    return (G.numpy()) * np.outer(sc, sc)

def cc(G):
    Gc = G - one @ G - G @ one + one @ G @ one
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    return Gc / np.outer(d, d)

# text reference, same subset
z = np.load(os.path.join(HERE, "results", "text_vecs.npz"))
D = (z["chosen"] - z["rejected"])[keep]
D = D / np.linalg.norm(D, axis=1, keepdims=True)
Ctext = cc(D @ D.T)
iu = np.triu_indices(p, 1)

out = {}
rows = []
for typ, mlist in sorted(by_type.items()):
    C = cc(gram(mlist))
    sol = fa.solution(C, 5, targets, label=typ)
    cong = np.array(sol["congruence_oblimin"])
    best = [float(np.max(np.abs(cong[:, a]))) for a in range(5)]
    rsa_text = float(np.corrcoef(C[iu], Ctext[iu])[0, 1])
    out[typ] = {"best_per_factor": best, "mean_best": float(np.mean(best)),
                "rsa_with_text": rsa_text, "n_modules": len(mlist)}
    rows.append((typ, np.mean(best), rsa_text, len(mlist)))
    print(f"  {typ:<12} congruence {np.mean(best):.3f}  RSA-with-text {rsa_text:+.3f}")

print("\n" + "=" * 66)
print(f"{'module type':<14}{'n':>4}{'congruence':>12}{'RSA w/ text':>13}")
print("-" * 66)
for typ, mb, rt, n in sorted(rows, key=lambda r: -r[1]):
    print(f"{typ:<14}{n:>4}{mb:>12.3f}{rt:>13.3f}")
print("-" * 66)
print(f"{'AGGREGATE':<14}{len(mods):>4}{0.750:>12.3f}{0.911:>13.3f}   (from the whole object)")
print(f"{'TEXT':<14}{'-':>4}{0.731:>12.3f}{1.000:>13.3f}")
print("=" * 66)
json.dump(out, open(os.path.join(HERE, "results", "per_module_fa.json"), "w"), indent=1)
print("wrote results/per_module_fa.json")
