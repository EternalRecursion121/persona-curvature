#!/usr/bin/env python
"""Factor analysis of the training-contrast text embeddings, with the adapter pipeline.

analysis/emb_pairs_minilm.npz#contrast holds, per trait in the Gram's order, the
mean all-MiniLM-L6-v2 embedding of the chosen replies minus the mean of the
matched rejected replies (built by analyse_text_baseline.py or its predecessor;
see the wiki page geometry/text-contrast-factors).  This script puts the
134 x 134 inner-product matrix of those contrasts through exactly the adapter
factor pipeline of analyse_fa_qwen35.py (double-centred correlation matrix,
principal axis factoring, oblimin, k = 5, order_and_orient against the Goldberg
targets), after first reproducing the adapter solution in results/fa_qwen35.json
from results/gram_sweep.npz to confirm the pipeline matches the file.

Outputs the Tucker congruence between the text factors and the adapter factors,
the best one-to-one matching, the text factors' own Big Five congruence, and the
full text solution.

Reads:  results/gram_sweep.npz, analysis/emb_pairs_minilm.npz, results/fa_qwen35.json,
        traits_primary.json
Writes: analysis/fa_text_contrast.json  (or the path given as argv[1])

First run 2026-09-15 (result: same five factors one to one, congruence 0.81 to
0.97, same Timidity/Arousal rotation); extracted into a script 2026-09-16.
"""
import json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv; sys.argv = ["x"]
import analyse_fa_qwen35 as F  # noqa: E402
sys.argv = _argv

FACT = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
TITLES = ["Warmth", "Competence", "Timidity", "Arousal", "Imagination"]
B5 = ["E", "A", "C", "ES", "I"]


def centred_R(M):
    p = M.shape[0]
    H = np.eye(p) - np.ones((p, p)) / p
    Gc = H @ M @ H; Gc = 0.5 * (Gc + Gc.T)
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    R = Gc / np.outer(d, d); np.fill_diagonal(R, 1.0)
    return R


def main(out_path):
    z = np.load(os.path.join(HERE, "results", "gram_sweep.npz"), allow_pickle=True)
    names = [str(n) for n in z["names"]]; G = np.asarray(z["G"], float)
    P = np.load(os.path.join(HERE, "analysis", "emb_pairs_minilm.npz")); C = P["contrast"]
    assert C.shape[0] == len(names), (C.shape, len(names))
    fa = json.load(open(os.path.join(HERE, "results", "fa_qwen35.json")))
    prim = {F.slugify(t["trait"]): t for t in json.load(open(os.path.join(HERE, "traits_primary.json")))}
    p = len(names)
    factor = np.array([prim[n]["factor"] if n in prim else "Lexicon" for n in names])
    pole = np.array([(1.0 if prim[n]["keyed"] == "+" else -1.0) if n in prim else 0.0 for n in names])
    targets = np.zeros((p, 6))
    for a, Fn in enumerate(FACT):
        targets[:, a] = np.where(factor == Fn, pole, 0.0)
    targets[:, 5] = np.where(factor != "Lexicon", pole, 0.0)

    Rw = centred_R(G); Rt = centred_R(C @ C.T)
    iu = np.triu_indices(p, 1)
    offdiag = float(np.corrcoef(Rw[iu], Rt[iu])[0, 1])
    print("offdiag corr of centred correlation matrices (weights vs text contrast):", round(offdiag, 3))
    sol = fa["solutions"]["centred_k5"]
    ridge = sol.get("ridge", 1e-3) if isinstance(sol.get("ridge"), (int, float)) else 1e-3
    sw = F.solution(Rw, 5, targets, ridge=ridge, label="weights")
    Lw = np.array(sw["loadings"]["oblimin"]); Lfile = np.array(sol["loadings"]["oblimin"])
    repro = float(np.abs(Lw - Lfile).max())
    print("reproduction of the adapter oblimin loadings, max |diff|:", repro)
    st = F.solution(Rt, 5, targets, ridge=ridge, label="text contrast")
    Lt = np.array(st["loadings"]["oblimin"])
    M = np.array([[F.tucker(Lt[:, j], Lfile[:, i]) for i in range(5)] for j in range(5)])
    print("\nTucker congruence, text-contrast factor (rows) x adapter factor (cols):")
    print("            " + "".join(f"{t:>12s}" for t in TITLES))
    for j in range(5):
        print(f"text f{j+1}     " + "".join(f"{M[j, i]:+12.2f}" for i in range(5)))
    r, c = linear_sum_assignment(-np.abs(M))
    print("\nbest matching:", [(f"text f{j+1}", TITLES[i], round(float(M[j, i]), 2)) for j, i in zip(r, c)],
          "mean |congruence|", round(float(np.abs(M[r, c]).mean()), 3))
    ct = np.array(st["congruence_oblimin"])[:, :5]
    ev = np.sort(np.linalg.eigvalsh(Rt))[::-1]
    json.dump({"what": "factor analysis of the chosen-minus-rejected MiniLM contrast embeddings (analysis/emb_pairs_minilm.npz), same pipeline as the adapters: double-centred correlation matrix, PAF, oblimin, k=5, order_and_orient",
               "offdiag_corr_centred_R_weights_vs_text": offdiag,
               "reproduction_max_abs_diff_adapter_loadings": repro,
               "tucker_text_rows_x_adapter_cols": M.tolist(), "adapter_factor_titles": TITLES,
               "best_matching": [[f"text_f{j+1}", TITLES[i], float(M[j, i])] for j, i in zip(r, c)],
               "mean_abs_matched_congruence": float(np.abs(M[r, c]).mean()),
               "big5_congruence_text": ct.tolist(), "big5_cols": B5, "text_centred_eigenvalues": ev[:20].tolist(),
               "text_solution": st}, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "analysis", "fa_text_contrast.json"))
