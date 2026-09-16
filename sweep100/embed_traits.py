"""Text features for any trait set, in the form the hypernetwork consumes.

The feature is the mean embedding of the PREFERRED replies minus the mean
embedding of the REJECTED ones.  That choice is not cosmetic: the text-baseline
experiment measured that embedding the preferred replies alone reproduces almost
none of the weight-space structure (every reply is a chatty assistant answer and
they all look alike), while the difference reproduces it at RSA 0.83.  The
informative signal is the contrast, so the feature is built to be the contrast.

Split out of text_baseline.py so a second corpus can be embedded on the same
terms as the first.  Identical model, identical pooling, identical pair budget --
if the out-of-distribution comparison is to mean anything, the only thing that
may differ between the two corpora is which traits are in them.

usage:
  python embed_traits.py --traits traits_expanded.json \
                         --data-dir data_bal_expanded \
                         --out results/text_vecs_expanded.npz
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_PAIRS = 64          # same budget as the original corpus; see text_baseline.py

ps = argparse.ArgumentParser()
ps.add_argument("--traits", default=os.path.join(HERE, "traits_expanded.json"))
ps.add_argument("--data-dir", default=os.path.join(HERE, "data_bal_expanded"))
ps.add_argument("--out", default=os.path.join(HERE, "results",
                                              "text_vecs_expanded.npz"))
args = ps.parse_args()

traits = json.load(open(args.traits))
slug = lambda t: t["trait"].lower().replace("-", "_").replace(" ", "_")
names = [slug(t) for t in traits]

# Only traits that actually have balanced data: the balancer drops any trait
# that could not supply the full reference prompt list, so the trait file and
# the data directory legitimately disagree.  Emit the intersection and say so,
# rather than embedding a trait that has no adapter or padding one that does.
have = [n for n in names if os.path.exists(os.path.join(args.data_dir, n + ".jsonl"))]
missing = [n for n in names if n not in set(have)]
if missing:
    print(f"{len(missing)} trait(s) in {os.path.basename(args.traits)} have no "
          f"balanced data and are skipped: {missing[:6]}", file=sys.stderr)
print(f"embedding {len(have)} traits from {args.data_dir}")

from sentence_transformers import SentenceTransformer  # noqa: E402
m = SentenceTransformer(MODEL)

Vc, Vr = [], []
for i, n in enumerate(have, 1):
    rows = [json.loads(l) for l in open(os.path.join(args.data_dir, n + ".jsonl"))]
    rows = rows[:N_PAIRS]
    ec = m.encode([r["chosen"] for r in rows], batch_size=64,
                  normalize_embeddings=True, show_progress_bar=False)
    er = m.encode([r["rejected"] for r in rows], batch_size=64,
                  normalize_embeddings=True, show_progress_bar=False)
    Vc.append(ec.mean(0)); Vr.append(er.mean(0))
    if i % 25 == 0 or i == len(have):
        print(f"  {i}/{len(have)}", flush=True)

Vc, Vr = np.array(Vc), np.array(Vr)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez(args.out, chosen=Vc, rejected=Vr, names=np.array(have))
print(f"wrote {args.out}: chosen {Vc.shape}, rejected {Vr.shape}")

# Verify the artefact rather than the report: reload and check the difference
# feature is finite, non-degenerate, and actually discriminates between traits.
z = np.load(args.out, allow_pickle=True)
D = z["chosen"] - z["rejected"]
D = D / np.linalg.norm(D, axis=1, keepdims=True)
assert np.isfinite(D).all(), "non-finite text features"
C = D @ D.T
off = C[~np.eye(len(D), dtype=bool)]
print(f"difference features: mean off-diagonal cosine {off.mean():+.3f} "
      f"(max {off.max():+.3f}) -- traits are distinguishable, not collapsed")
assert off.mean() < 0.9, "text features have collapsed; every trait looks alike"
