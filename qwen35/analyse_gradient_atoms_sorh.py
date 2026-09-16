#!/usr/bin/env python3
"""Assemble analysis/gradient_atoms_sorh.json -- experiment G3.

Unsupervised atoms on the 973 matched School of Reward Hacks rows, extracted
through the SAME EKFAC projection fitted on the zoo's preference gradients
(`zoo_basis.npz`), so an atom here and an atom there live in one space.  Each
row contributes two documents, the hack completion and its matched control, and
a document's gradient is the SFT one at B = 0: -grad_B log p(completion)/n_tok.

Three questions, in the order the coordinator set them:
  (a) do any atoms separate hack from control (purity against a label shuffle),
  (b) what do the hack-dominant atoms unproject to in weight space,
  (c) what is the shared motif of the most hack-pure atoms' top-20 documents.

usage: analyse_gradient_atoms_sorh.py
"""
import collections
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))


def main():
    EX = json.load(open(f"{Q}/analysis/gradient_atoms_extract_sorh.json"))
    AT = json.load(open(f"{Q}/analysis/gradient_atoms_atoms_sorh.json"))
    WS = json.load(open(f"{Q}/analysis/gradient_atoms_weightspace_sorh.json"))
    Z = np.load(f"{Q}/results/gradient_atoms/sorh_atoms.npz", allow_pickle=False)
    D = Z["atoms"].astype(np.float64)
    K = D.shape[0]
    items = {x["id"]: x for x in
             json.load(open(f"{Q}/phase10_runs/gradatoms_items_sorh.json"))}

    prim = [c for c in AT["configs"] if "atom_rows" in c][0]
    rows = prim["atom_rows"]
    names = WS["names"]
    ni = {n: i for i, n in enumerate(names)}
    WC = np.array(WS["atom_target_cos"], dtype=np.float64)
    CSp = np.array(WS["target_proj"], dtype=np.float64)
    Pn = CSp / np.maximum(np.linalg.norm(CSp, axis=1, keepdims=True), 1e-30)
    Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-30)
    AS = Dn @ Pn.T

    rng = np.random.default_rng(11)
    G = rng.normal(size=(1000, D.shape[1]))
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    nullmax = np.abs(G @ Dn.T).max(1)

    # (a) hack purity per atom
    per = []
    for r in rows:
        tt = r["top_traits"]
        if len(tt) < 2:
            continue
        c = collections.Counter(tt)
        lab, n = c.most_common(1)[0]
        per.append({"atom": r["atom"], "coherence": r["coherence"],
                    "n_active": r["n_active"], "majority": lab,
                    "purity": n / len(tt), "n_top": len(tt),
                    "top_ids": r["top_ids"]})
    per.sort(key=lambda x: (-x["purity"], -(x["coherence"] or 0)))
    hack_pure = [p for p in per if p["majority"] == "hack"][:5]
    ctrl_pure = [p for p in per if p["majority"] == "control"][:5]

    def wsrow(atom):
        j = np.argsort(-np.abs(WC[atom]))[:8]
        return [{"target": names[t], "cos": float(WC[atom, t])} for t in j]

    for p in hack_pure + ctrl_pure:
        p["weight_space_nearest"] = wsrow(p["atom"])
        p["atom_space_nearest"] = [
            {"target": names[t], "cos": float(AS[p["atom"], t])}
            for t in np.argsort(-np.abs(AS[p["atom"]]))[:8]]
        p["top_documents"] = [
            {"id": i,
             "prompt": items[i.split(":")[0]]["prompt"][:300],
             "completion": items[i.split(":")[0]][
                 "chosen" if i.endswith(":hack") else "rejected"][:400]}
            for i in p["top_ids"][:20]]

    named = [n for n in names if not n.startswith("trait_")
             and not n.startswith("rand_merge_")]
    rand = [n for n in names if n.startswith("rand_merge_")]
    band = float(np.max([np.abs(WC[:, ni[n]]).max() for n in rand]))
    dirs = {n: {"weight_space_max_abs_cos": float(np.abs(WC[:, ni[n]]).max()),
                "weight_space_argmax_atom": int(np.abs(WC[:, ni[n]]).argmax()),
                "atom_space_max_abs_cos": float(np.abs(AS[:, ni[n]]).max())}
            for n in named}
    tr = [n for n in names if n.startswith("trait_")]
    dirs["_zoo_traits_max_abs_cos_weight_space"] = {
        "n": len(tr),
        "max": float(np.max([np.abs(WC[:, ni[n]]).max() for n in tr])),
        "mean": float(np.mean([np.abs(WC[:, ni[n]]).max() for n in tr]))}

    out = {"what": "Experiment G3: unsupervised gradient atoms on the 973 matched "
                   "School of Reward Hacks rows, in the EKFAC projection fitted on "
                   "the zoo's own preference gradients.",
           "extraction": EX,
           "atoms": AT,
           "atom_purity_hack_vs_control": {
               "n_atoms_scored": len(per),
               "mean_purity": float(np.mean([p["purity"] for p in per])),
               "n_purity_1_0": int(sum(1 for p in per if p["purity"] == 1.0)),
               "n_majority_hack": int(sum(1 for p in per
                                          if p["majority"] == "hack")),
               "n_majority_control": int(sum(1 for p in per
                                             if p["majority"] == "control")),
               "label_shuffle_null":
                   (prim.get("purity_assigned_trait") or {})},
           "most_hack_pure": hack_pure,
           "most_control_pure": ctrl_pure,
           "weight_space": {"random_merge_band_widest_abs_cos": band,
                            "random_unit_vector_band": {
                                "mean_max_abs_cos": float(nullmax.mean()),
                                "p95": float(np.percentile(nullmax, 95))},
                            "directions": dirs},
           "all_atoms": per}
    with open(f"{Q}/analysis/gradient_atoms_sorh.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote analysis/gradient_atoms_sorh.json")
    print(f"{K} atoms; mean hack/control purity "
          f"{out['atom_purity_hack_vs_control']['mean_purity']:.4f}, null "
          f"{(prim.get('purity_assigned_trait') or {}).get('null_mean')}")
    for p in hack_pure[:3]:
        print(f"  atom {p['atom']} purity {p['purity']:.2f} coherence "
              f"{p['coherence']}, nearest {p['weight_space_nearest'][0]}")


if __name__ == "__main__":
    main()
