#!/usr/bin/env python3
"""Text-embedding inputs for the Goldberg-only / held-out-lexicon analysis.

Two artefacts, both on CPU from the locally cached sentence-transformers models,
both in the Gram's trait order (results/gram_sweep.npz#names):

  analysis/emb_constitutions_mpnet.npy   134 x 768
      all-mpnet-base-v2 over each trait's constitution text. mpnet is the model
      this project already used to draw the 34 Lexicon traits
      (traits_secondary_provenance.json#embedding_model).

  analysis/emb_pairs_minilm.npz          chosen / rejected / contrast, 134 x 384
      all-MiniLM-L6-v2, the model sweep100's text baseline used, over a fixed
      subsample of each trait's DPO pairs. sweep100 found that embedding the
      chosen replies alone gives RSA 0.429 against the weight Gram while
      chosen-minus-rejected gives 0.826 - "the structure is in the contrast,
      not the text" - so a constitution, which is a single text, is
      structurally the chosen-only case and the contrast has to be built too.

Pre-registered in PREREG_goldberg_only.md. Read by analyse_goldberg_only.py.

Usage:  ~/cartovenv/bin/python embed_goldberg_only.py
        (needs torch + sentence-transformers; qwen35/.venv has neither)
"""
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 150
SEED = 0

torch.set_num_threads(os.cpu_count() or 4)


def slugify(t):
    return str(t).lower().replace(" ", "_").replace("-", "_")


def main():
    z = np.load(f"{HERE}/results/gram_sweep.npz", allow_pickle=True)
    names = [str(x) for x in z["names"]]

    cons = json.load(open(f"{HERE}/constitutions.json"))
    cmap = {slugify(k): v["constitution"] for k, v in cons.items()
            if isinstance(v, dict) and "constitution" in v}
    assert all(n in cmap for n in names), [n for n in names if n not in cmap]

    m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    E = m.encode([cmap[n] for n in names], batch_size=16, convert_to_numpy=True,
                 show_progress_bar=False, normalize_embeddings=False)
    np.save(f"{HERE}/analysis/emb_constitutions_mpnet.npy", E)
    print("constitutions", E.shape, flush=True)

    mm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    d = mm.get_sentence_embedding_dimension()
    rng = np.random.default_rng(SEED)
    C = np.zeros((len(names), d))
    R = np.zeros_like(C)
    D = np.zeros_like(C)
    for i, n in enumerate(names):
        rows = [json.loads(l) for l in open(f"{HERE}/data/{n}.jsonl")]
        idx = rng.choice(len(rows), size=min(N_PAIRS, len(rows)), replace=False)
        ec = mm.encode([rows[j]["chosen"] for j in idx], batch_size=64,
                       convert_to_numpy=True, show_progress_bar=False)
        er = mm.encode([rows[j]["rejected"] for j in idx], batch_size=64,
                       convert_to_numpy=True, show_progress_bar=False)
        C[i], R[i], D[i] = ec.mean(0), er.mean(0), ec.mean(0) - er.mean(0)
        if i % 20 == 0:
            print("pairs", i, flush=True)
    np.savez(f"{HERE}/analysis/emb_pairs_minilm.npz",
             chosen=C, rejected=R, contrast=D, n_pairs=N_PAIRS)
    print("pairs done", D.shape, flush=True)


if __name__ == "__main__":
    main()
