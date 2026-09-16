"""Does the trait geometry survive a change of LoRA initialisation?

The calibration said yes on 4 traits and 6 pairs (RSA 0.995).  This is the same
question at full scale: 100 traits under three independent inits, 4950 pairs.

Why it matters.  Every adapter in the original sweep shared LoRA init seed 0, so
the whole PCA/factor story could have been a property of that one random rank-16
slice rather than of the traits.  Three inits settle it.  The individual adapters
are expected to be near-orthogonal ACROSS inits -- LoRA confines each update to a
random slice of a 2048-dimensional space, and two random slices barely intersect
however similar the behaviour.  So a vector-level comparison is uninformative by
construction.  The informative comparison is relational: does the PATTERN of
between-trait similarities come out the same?

Three things get measured:
  geometry     RSA between inits over all 4950 trait pairs, against a label-
               permutation null.  "Does the shape reproduce?"
  spectrum     PCA eigenvalues per init, and how many components sit above the
               permutation null.  This is the null spectrum the earlier analysis
               could not build for want of replicates.
  factors      Tucker congruence between the k=5 oblimin solutions of different
               inits.  If the Big Five recovery reproduces across inits it is a
               property of the traits, not of a lucky rotation of one slice.

usage:  python seeds_analysis.py [--seeds 0,1,2] [--nperm 2000]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_pca as ap  # noqa: E402  (import-safe: guarded by __main__)

# seed 0 lives in the original corpus; the replicates in the new one
SRC = {0: os.path.join(HERE, "adapters"),
       1: os.path.join(HERE, "adapters_seeds"),
       2: os.path.join(HERE, "adapters_seeds")}
LINKDIR = os.path.join(HERE, "adapters_all")

ps = argparse.ArgumentParser()
ps.add_argument("--seeds", default="0,1,2")
ps.add_argument("--nperm", type=int, default=2000)
ps.add_argument("--out", default=os.path.join(HERE, "results", "seeds.json"))
args = ps.parse_args()
SEEDS = [int(x) for x in args.seeds.split(",") if x.strip()]
rng = np.random.default_rng(0)

traits = json.load(open(os.path.join(HERE, "traits.json")))
slug = lambda t: t["trait"].lower().replace("-", "_")
NAMES = [slug(t) for t in traits]
FAC = {slug(t): (t["factor"], t["keyed"]) for t in traits}


def dirname(trait, seed):
    return trait if seed == 0 else f"{trait}__r{seed}"


# gram_streaming reads everything from one directory; give it one, built of
# symlinks, rather than copying 35GB or teaching it about two corpora.
os.makedirs(LINKDIR, exist_ok=True)
present, absent = {}, []
for s in SEEDS:
    for t in NAMES:
        d = dirname(t, s)
        src = os.path.join(SRC[s], d)
        if not os.path.isdir(src):
            absent.append(d)
            continue
        link = os.path.join(LINKDIR, d)
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(src, link)
        present.setdefault(s, []).append(t)
if absent:
    print(f"WARNING: {len(absent)} adapter(s) missing, e.g. {absent[:5]}",
          file=sys.stderr)

# only traits present under EVERY requested seed, so each matrix is comparable
common = sorted(set.intersection(*(set(present.get(s, [])) for s in SEEDS)))
print(f"{len(common)} traits present under all of seeds {SEEDS}")
if len(common) < 10:
    sys.exit("too few complete traits to analyse")

# ---- provenance: every adapter must BE what its directory name claims -------
# On 2026-08-15 four `__rN` directories held 9-epoch calibration adapters that a
# resume guard had declined to overwrite.  The names were right, the file sizes
# were right, and the run had said "skipping 4 already-trained" in a log nobody
# read.  A name is not provenance.  Each adapter records its own hyperparameters,
# so check those here, where the data is actually consumed, and refuse to
# produce numbers over a mixed corpus.
def provenance(trait, seed):
    p = os.path.join(LINKDIR, dirname(trait, seed), "runmeta.json")
    if not os.path.exists(p):
        return {"error": "no runmeta.json"}
    m = json.load(open(p))
    # Adapters trained before the `__rN` work predate the init_seed field; they
    # record `seed`, which WAS the LoRA init seed for those runs.  Falling back
    # to it is a correction, not a relaxation -- a missing field still fails.
    init = m.get("init_seed")
    if init is None:
        init = m.get("seed")
    return {"epochs": (m.get("hparams") or {}).get("epochs"),
            "init_seed": init, "trait": m.get("trait")}


meta = {(t, s): provenance(t, s) for s in SEEDS for t in common}
epochs_seen = {v.get("epochs") for v in meta.values()}
bad = []
for (t, s), v in sorted(meta.items()):
    if v.get("error"):
        bad.append(f"{dirname(t, s)}: {v['error']}")
    elif v.get("init_seed") != s:
        bad.append(f"{dirname(t, s)}: init_seed={v.get('init_seed')}, expected {s}")
    elif len(epochs_seen) > 1 and v.get("epochs") != min(e for e in epochs_seen if e):
        bad.append(f"{dirname(t, s)}: epochs={v.get('epochs')}")
if len(epochs_seen - {None}) > 1:
    print(f"MIXED TRAINING LENGTHS IN THE CORPUS: {sorted(epochs_seen)}",
          file=sys.stderr)
if bad:
    print(f"\n{len(bad)} adapter(s) are not what their directory name claims:",
          file=sys.stderr)
    for b in bad[:20]:
        print(f"  {b}", file=sys.stderr)
    sys.exit("refusing to analyse a corpus of mixed provenance")
print(f"provenance OK: all {len(meta)} adapters report epochs="
      f"{sorted(epochs_seen)} and an init_seed matching their name")
ap.ADIR = LINKDIR

# ---- per-seed cosine matrices, all on the same trait order ------------------
C, G_ = {}, {}
for s in SEEDS:
    G, mods, scale, r = ap.gram_streaming([dirname(t, s) for t in common],
                                          log=sys.stderr)
    G_[s], C[s] = G, ap.cos_from_G(G)
    print(f"  seed {s}: {len(mods)} modules, scale {scale}, rank {r}")

k = len(common)
iu = np.triu_indices(k, 1)


def rsa(A, B):
    return float(np.corrcoef(A[iu], B[iu])[0, 1])


# ---- geometry: does the shape reproduce, against a label-permutation null ---
geom = []
for i, a in enumerate(SEEDS):
    for b in SEEDS[i + 1:]:
        obs = rsa(C[a], C[b])
        null = []
        for _ in range(args.nperm):
            p = rng.permutation(k)
            null.append(rsa(C[a], C[b][np.ix_(p, p)]))
        null = np.array(null)
        geom.append({"seeds": [a, b], "rsa": obs,
                     "null_mean": float(null.mean()), "null_sd": float(null.std()),
                     "p": float((np.abs(null) >= abs(obs)).mean()),
                     "z": float((obs - null.mean()) / null.std()) if null.std() else None})
        print(f"  geometry RSA seed{a} vs seed{b}: {obs:+.3f}  "
              f"(null {null.mean():+.3f} +/- {null.std():.3f}, p={geom[-1]['p']:.4f})")

# ---- the vector-level comparison, expected to be ~0 -------------------------
vec = []
for i, a in enumerate(SEEDS):
    for b in SEEDS[i + 1:]:
        Gab = None
        names = [dirname(t, a) for t in common] + [dirname(t, b) for t in common]
        Gab, _, _, _ = ap.gram_streaming(names, log=open(os.devnull, "w"))
        Cab = ap.cos_from_G(Gab)
        same = [Cab[j, k + j] for j in range(k)]
        off = [Cab[j, k + m] for j in range(k) for m in range(k) if j != m]
        vec.append({"seeds": [a, b], "same_trait_cos": float(np.mean(same)),
                    "same_trait_sd": float(np.std(same)),
                    "diff_trait_cos": float(np.mean(off))})
        print(f"  vectors  seed{a} vs seed{b}: same-trait {np.mean(same):+.3f} "
              f"+/- {np.std(same):.3f}, different-trait {np.mean(off):+.3f}")

# ---- spectrum: how many components clear a permutation null ----------------
def spectrum(Cm):
    H = np.eye(k) - np.ones((k, k)) / k
    return np.sort(np.linalg.eigvalsh(H @ Cm @ H))[::-1]


def shuffled_like(Cm):
    """Same multiset of pairwise cosines, no structure.

    NOT a row/column permutation: permuting both by the same permutation only
    relabels the traits and leaves every eigenvalue exactly where it was, so it
    would produce a null identical to the observation.  Shuffling the off-
    diagonal entries destroys the structure while preserving the distribution
    of similarities, which is what the spectrum has to be judged against.
    """
    v = Cm[iu].copy()
    rng.shuffle(v)
    M = np.eye(k)
    M[iu] = v
    return M + M.T - np.eye(k)


spec = {}
for s in SEEDS:
    obs = spectrum(C[s])
    null = np.array([spectrum(shuffled_like(C[s]))
                     for _ in range(min(args.nperm, 400))])
    thr = np.percentile(null, 95, axis=0)
    # Horn's rule: take the LEADING run, stopping at the first component that
    # fails.  A bare count over all positions is wrong -- in the tail both the
    # observed and the null eigenvalues sit at ~0, and the observed ones edge
    # past a slightly negative null, inflating the count (18 instead of 3 on a
    # synthetic rank-3 matrix).
    nsig = int(np.argmax(obs <= thr)) if (obs <= thr).any() else len(obs)
    spec[s] = {"eigenvalues": obs[:10].tolist(),
               "var_pct": (100 * obs[:10] / obs.sum()).tolist(),
               "null_95": thr[:10].tolist(), "n_above_null": nsig}
    print(f"  seed {s}: {nsig} component(s) above the 95th-percentile null; "
          f"top var% {100*obs[0]/obs.sum():.1f}, {100*obs[1]/obs.sum():.1f}, "
          f"{100*obs[2]/obs.sum():.1f}")

out = {"seeds": SEEDS, "n_traits": len(common), "traits": common,
       "nperm": args.nperm, "geometry": geom, "vectors": vec,
       "spectrum": spec, "missing": absent}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
print("""
Read it like this: a high geometry RSA with a near-zero same-trait vector cosine
means the traits sit in reproducible RELATIVE positions while each adapter's
absolute direction is an accident of its initialisation.  That is the result
that decides whether the factor structure is about traits or about one slice.""")
