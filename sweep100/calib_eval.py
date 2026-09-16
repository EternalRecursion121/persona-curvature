"""Pick the DPO epoch count from evidence instead of from the word "more".

One run per trait to 9 epochs, snapshotted at 2/5/9, twice per trait under
different LoRA inits (`__r1`).  At each snapshot we have, for the same trait,
two independent adapters -- so we can read off the two numbers that actually
matter for everything downstream:

  ceiling  mean cosine between the two seeds of the SAME trait.  Nothing that
           predicts an adapter from its trait name -- a hypernetwork, a
           retrieval baseline -- can beat this: it is how well the training
           procedure agrees with itself.
  floor    mean cosine between DIFFERENT traits.  The bit we are trying to
           read structure out of.

More epochs is only better if it raises ceiling/floor.  Training longer also
drives both up together (every adapter drifts into the same high-norm region),
which would LOOK like a stronger trait and be worth nothing.  That is the
failure mode this exists to catch.

usage:  python calib_eval.py [--adir DIR]
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyse_pca as ap  # noqa: E402  (import-safe: guarded by __main__)

TRAITS = ["talkative", "cold", "organized", "anxious"]
SEEDS = ["", "__r1"]
EPOCHS = [2, 5, 9]

ps = argparse.ArgumentParser()
ps.add_argument("--adir", default=os.path.join(HERE, "adapters_calib"))
ps.add_argument("--out", default=os.path.join(HERE, "results", "calib.json"))
args = ps.parse_args()
ap.ADIR = args.adir

rows, missing = [], []
for E in EPOCHS:
    # the final epoch has no "__eN" snapshot: HF's on_epoch_end does not fire
    # for the last epoch, but the run's ordinary saved adapter IS that state.
    suf = "" if E == max(EPOCHS) else f"__e{E}"
    names = [f"{t}{s}{suf}" for t in TRAITS for s in SEEDS]
    absent = [n for n in names if not os.path.isdir(os.path.join(args.adir, n))]
    if absent:
        missing.append((E, absent))
        continue
    G, mods, scale, r = ap.gram_streaming(names, log=open(os.devnull, "w"))
    C = ap.cos_from_G(G)
    norms = np.sqrt(np.diag(G))

    # index by (trait, seed) in the order `names` was built
    idx = {(t, s): i for i, (t, s) in
           enumerate([(t, s) for t in TRAITS for s in SEEDS])}
    same = [C[idx[(t, "")], idx[(t, "__r1")]] for t in TRAITS]
    between = [C[i, j] for i in range(len(names)) for j in range(i + 1, len(names))
               if names[i].split("__")[0] != names[j].split("__")[0]]
    rows.append({
        "epochs": E,
        "ceiling_same_trait": float(np.mean(same)),
        "ceiling_per_trait": {t: float(c) for t, c in zip(TRAITS, same)},
        "floor_between_trait": float(np.mean(between)),
        "separability": float(np.mean(same) / np.mean(between))
        if np.mean(between) else None,
        "mean_norm": float(np.mean(norms)),
        "n_modules": len(mods),
    })

for E, absent in missing:
    print(f"epoch {E}: MISSING {len(absent)} adapter(s): {absent[:4]}", file=sys.stderr)
if not rows:
    sys.exit("no complete epoch snapshots found in " + args.adir)

print(f"\n{'epochs':>7} {'ceiling':>9} {'floor':>8} {'separability':>13} {'mean |dW|':>11}")
print("-" * 52)
for r_ in rows:
    print(f"{r_['epochs']:>7} {r_['ceiling_same_trait']:>9.3f} "
          f"{r_['floor_between_trait']:>8.3f} {r_['separability']:>13.2f} "
          f"{r_['mean_norm']:>11.3f}")
print("""
ceiling = how well two inits of the SAME trait agree (the hypernetwork's ceiling)
floor   = how much DIFFERENT traits already resemble each other
Pick the epoch count that maximises separability, not the one that maximises
ceiling -- a ceiling that rises with the floor is drift, not signal.""")

best = max(rows, key=lambda r_: r_["separability"] or 0)
print(f"\nbest separability: {best['epochs']} epochs "
      f"({best['separability']:.2f}x)")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"traits": TRAITS, "rows": rows, "chosen_epochs": best["epochs"]},
          open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
