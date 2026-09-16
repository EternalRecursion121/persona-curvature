#!/usr/bin/env python3
"""Experiment S1: LoRAs trained to a target in the model's own residual stream,
scored in activation space, in weight space and in behaviour.

Inputs, all produced by persona_sliders.py, cross_gram_full_on_modal.py,
cross_gram_sliders.py and judge_personas.py:

  analysis/slider_targets.npz, analysis/slider_targets_meta.json
  analysis/slider_train.json              training curves + teacher-forced eval
  analysis/slider_means.npz               free-running shifts, both halves
  results/cross_gram_full_root_x_pc-qwen35-adapters_sliders.npz
  analysis/slider_cross_gram.json         the same, split at block 16
  phase10_runs/sliders_behave.json        24-prompt battery generations
  phase10_runs/judged_sliders.json        blind Big Five scores for the above
  phase10_runs/judged_100.json            the stage-one reference at 200 tokens

Writes analysis/persona_sliders.json.  Nothing here regenerates anything.
"""
import json
import os

import numpy as np

from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability",
      "Intellect"]
PRIMARY = 16
rng = np.random.default_rng(0)


def looping(t, n=10, k=4):
    """A response loops if some n-word window repeats at least k times.
    Verbatim from analyse_alien_steer.py so the rates are comparable."""
    w = t.split()
    if len(w) < n * 2:
        return False
    g = {}
    for i in range(len(w) - n + 1):
        s = " ".join(w[i:i + n])
        g[s] = g.get(s, 0) + 1
        if g[s] >= k:
            return True
    return False


def trait_meta():
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        p = f"{Q}/{f}"
        if os.path.exists(p):
            for r in json.load(open(p)):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = \
                    (r["factor"], r["keyed"])
    return meta


def big_five_basis(ch, meta):
    """The five named Big Five axes as unit coefficient vectors over the 134,
    and the inverse of their Gram -- analyse_alien_steer_fa.big_five_coords."""
    K = []
    for f in F5:
        p = np.array([1.0 if meta.get(t) == (f, "+") else 0.0 for t in ch.names])
        m = np.array([1.0 if meta.get(t) == (f, "-") else 0.0 for t in ch.names])
        v = p / p.sum() - m / m.sum()
        K.append(v / ch.norm(v))
    K = np.stack(K)
    return K, np.linalg.inv(K @ ch.G @ K.T)


def main():
    out = {}
    meta = trait_meta()
    ZT = np.load(f"{Q}/analysis/slider_targets.npz", allow_pickle=True)
    names = [str(x) for x in ZT["names"]]
    U = ZT["U"].astype(np.float64)
    anchors = ZT["anchor"].astype(np.float64)
    TM = json.load(open(f"{Q}/analysis/slider_targets_meta.json"))
    TRAITS = [n for n in names if TM["meta"][n]["kind"] == "trait"]
    HOLES = [n for n in names if TM["meta"][n]["kind"] != "trait"]

    out["design"] = {
        "targets": names, "trait_sliders": TRAITS, "hole_sliders": HOLES,
        "layer": PRIMARY, "window": "resp",
        "target_half": TM["target_half"], "held_out_half": TM["held_out_half"],
        "target_norms": TM["target_norms"],
        "magnitude_anchors": TM["magnitude_anchors"],
        "target_pairwise_cosines": TM["target_cosines"],
        "comparators": TM["comparators"],
    }

    # ------------------------------------------------------------- training --
    TR = json.load(open(f"{Q}/analysis/slider_train.json"))
    out["training"] = {"job": TR.get("job", {}), "seconds": TR.get("seconds"),
                       "per_slider": {}}
    for n in names:
        r = TR["sliders"][n]
        out["training"]["per_slider"][n] = {
            "curve": r["curve"], "seconds": r["seconds"],
            "delta_frobenius": r["delta_frobenius"],
            "nonzero_B_modules": r["nonzero_B_modules"],
            "n_modules": r["n_modules"],
            "teacher_forced": r["tf"],
        }

    # --------------------------------------------------- activation space ----
    ZM = np.load(f"{Q}/analysis/slider_means.npz", allow_pickle=True)
    snames = [str(x) for x in ZM["names"]]
    S = ZM["M"].astype(np.float64)                    # (n, 2 halves, L, D)
    sh = {n: S[snames.index(n)][:, PRIMARY] for n in names}   # (2, D) per slider

    # the 134 prompted persona vectors, trait-centred, per half -- the space the
    # targets live in
    ZA = np.load(f"{Q}/analysis/actspace_means.npz", allow_pickle=True)
    t134 = [str(t) for t in ZA["traits"]]
    MA = ZA["M"].astype(np.float64)
    BA = ZA["B"].astype(np.float64)
    V = [MA[:, 0, h, PRIMARY] - BA[0, h, PRIMARY][None] for h in (0, 1)]
    Xc = [v - v.mean(0) for v in V]
    ZAD = np.load(f"{Q}/analysis/actspace_means_adapters.npz", allow_pickle=True)
    MAD = ZAD["M"].astype(np.float64)
    A = [MAD[:, 0, h, PRIMARY] - BA[0, h, PRIMARY][None] for h in (0, 1)]
    Ac = [a - a.mean(0) for a in A]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    act = {}
    for i, n in enumerate(names):
        rec = {}
        for h in (0, 1):
            s = sh[n][h]
            rec[f"half{h}"] = {
                "cos_own_target": cos(s, U[i]),
                "shift_norm": float(np.linalg.norm(s)),
                "mag_ratio_vs_anchor": float(np.linalg.norm(s) / anchors[i]),
                "cos_all_targets": {names[j]: cos(s, U[j])
                                    for j in range(len(names))},
            }
        # selectivity: is the own target the argmax over the 13?
        c1 = rec["half1"]["cos_all_targets"]
        best = max(c1, key=c1.get)
        rec["heldout_argmax_target"] = best
        rec["heldout_own_is_argmax"] = bool(best == n)
        rec["heldout_margin"] = float(c1[n] - max(v for k, v in c1.items() if k != n))
        # the direct object-to-object comparison: does the slider move the
        # residual stream the way the trait's own trained adapter moves it?
        if TM["meta"][n]["kind"] == "trait":
            k = t134.index(n)
            rec["vs_stage1_adapter_shift_half1"] = {
                "cos_raw": cos(sh[n][1], A[1][k]),
                "cos_centred": cos(sh[n][1], Ac[1][k]),
                "norm_ratio": float(np.linalg.norm(sh[n][1]) /
                                    np.linalg.norm(A[1][k])),
            }
        act[n] = rec
    out["activation"] = act

    # the hole slider, put through analyse_actspace.py's own transplant test:
    # angle to the nearest of the 134 trait-centred persona vectors, on half 1
    hole_ang = {}
    Xn = Xc[1] / np.linalg.norm(Xc[1], axis=1, keepdims=True)
    for n in HOLES + TRAITS:
        s = sh[n][1]
        c = np.abs(Xn @ s) / np.linalg.norm(s)
        j = int(np.argmax(c))
        hole_ang[n] = {"nearest_trait": t134[j],
                       "deg": float(np.degrees(np.arccos(min(1.0, c[j]))))}
    AL = json.load(open(f"{Q}/analysis/alien_fa.json"))["alien_fa"]
    coef = np.array([dict(zip(AL["traits"], AL["coeffs"]))[t] for t in t134])
    nulls = []
    for _ in range(500):
        h = coef[rng.permutation(len(t134))] @ Xc[1]
        nulls.append(np.degrees(np.arccos(
            (np.abs(Xn @ h) / np.linalg.norm(h)).max())))
    out["activation_nearest_trait"] = {
        "per_slider": hole_ang,
        "permuted_coefficient_null_median": float(np.median(nulls)),
        "permuted_coefficient_null_95pct": float(np.percentile(nulls, 95)),
        "note": "the same construction as analyse_actspace.py's transplanted "
                "hole, on the held-out half; the published transplant of the "
                "PC-chart hole is 47.834 deg against a null median of 47.748 "
                "(analysis/actspace_geometry.json#windows.resp.primary)",
    }

    # -------------------------------------------------------- weight space ---
    ch = FAChart()
    K, Gi = big_five_basis(ch, meta)
    cg = f"{Q}/results/cross_gram_full_root_x_pc-qwen35-adapters_sliders.npz"
    ZG = np.load(cg, allow_pickle=True)
    ga = [str(x) for x in ZG["names_a"]]
    gb = [str(x) for x in ZG["names_b"]]
    ordw = [ga.index(t) for t in ch.names]            # into gram_sweep order
    X = np.array(ZG["X"])[ordw]                       # (134, n_sliders)
    na = np.array(ZG["norms_a"])[ordw]
    nb = np.array(ZG["norms_b"])
    SG = json.load(open(f"{Q}/analysis/slider_cross_gram.json"))
    ga2 = SG["names_a"]; gb2 = SG["names_b"]
    o2 = [ga2.index(t) for t in ch.names]
    Xlow = np.array(SG["X_low"])[o2]
    na_low = np.array(SG["norms_a_low"])[o2]
    nb_low = np.array(SG["norms_b_low"])
    na_all = np.array(SG["norms_a_all"])[o2]
    # agreement between the two independent implementations
    Xall2 = np.array(SG["X_all"])[o2]
    agree = float(np.abs(X - Xall2[:, [gb2.index(n) for n in gb]]).max()
                  / np.abs(X).max())

    ALW = json.load(open(f"{Q}/analysis/alien_fa.json"))["alien_fa"]
    calien = np.array([dict(zip(ALW["traits"], ALW["coeffs"]))[t] for t in ch.names])
    n_alien = ch.norm(calien)
    TC = ch.trait_coords / np.linalg.norm(ch.trait_coords, axis=1, keepdims=True)

    facs = {f: np.array([j["coef"].get(t, 0.0) for t in ch.names])
            for f, j in [(j["name"], j) for j in json.load(
                open(f"{Q}/phase10_runs/steer_spec2_7a.json"))["jobs"]]
            if f.startswith("FA_")}

    W = {"cross_gram_max_rel_diff_between_implementations": agree,
         "n_adapters_a": len(ga), "sliders": {},
         "stage1_low_block_norm_fraction": {
             t: float(na_low[i] / na_all[i]) for i, t in enumerate(ch.names)},
         }
    for n in names:
        j = gb.index(n)
        col = X[:, j]
        c = col / (na * nb[j])
        srt = np.argsort(-np.abs(c))
        x = ch.coords_external(col)
        j2 = gb2.index(n)
        col_low = Xlow[:, j2]
        c_low = col_low / (na_low * nb_low[j2]) if nb_low[j2] > 0 else col_low * 0
        rec = {
            "frobenius_norm": float(nb[j]),
            "frobenius_norm_low_blocks": float(nb_low[j2]),
            "chart_coords": x.tolist(),
            "chart_len": float(np.linalg.norm(x)),
            "chart_frac_of_norm": float(np.linalg.norm(x) / nb[j]),
            "cos_with_134": {ch.names[i]: float(c[i]) for i in range(len(c))},
            "nearest_adapters": [{"trait": ch.names[i], "cos": float(c[i])}
                                 for i in srt[:5]],
            "cos_with_factor_merges": {
                f: float(cc @ col / (np.sqrt(cc @ ch.G @ cc) * nb[j]))
                for f, cc in facs.items()},
            "big_five_axis_coords": (((K @ col) / nb[j]) @ Gi).tolist(),
            # does a slider trained to the ACTIVATION image of the hole land on
            # the WEIGHT-space hole direction it was derived from?
            "cos_with_alien_fa_merge": float(calien @ col / (n_alien * nb[j])),
            # the same statistic the hole itself is defined by: angle inside the
            # factor chart to the nearest of the 134 trait lines
            "chart_gap_deg": float(np.degrees(np.arccos(min(1.0, float(
                np.abs(TC @ x).max() / max(np.linalg.norm(x), 1e-12)))))),
            "chart_nearest_trait": ch.names[int(np.argmax(
                np.abs(TC @ x)))],
        }
        if TM["meta"][n]["kind"] == "trait":
            i = ch.names.index(n)
            rec["cos_with_own_stage1_adapter"] = float(c[i])
            rec["cos_with_own_stage1_adapter_low_blocks"] = float(c_low[i])
            rec["own_stage1_adapter_rank_by_abs_cos"] = int(
                list(srt).index(i) + 1)
            rec["ceiling_from_depth"] = float(na_low[i] / na_all[i])
        W["sliders"][n] = rec
    out["weight_space"] = W

    # ------------------------------------------------------------ behaviour --
    bp = f"{Q}/phase10_runs/sliders_behave.json"
    jp = f"{Q}/phase10_runs/judged_sliders.json"
    if os.path.exists(bp) and os.path.exists(jp):
        BH = json.load(open(bp))
        JD = json.load(open(jp))
        prof = {}
        for r in JD["records"]:
            prof.setdefault(r["condition"], []).append(r["scores"])
        mean = {c: {f: float(np.mean([s[f] for s in v if s.get(f) is not None]))
                    for f in F5} for c, v in prof.items()}
        base = mean["base"]
        beh = {"judge_model": JD.get("model"), "n_judged": JD.get("n"),
               "failed_calls": JD.get("failed_calls"),
               "profile": mean,
               "delta_vs_base": {c: {f: mean[c][f] - base[f] for f in F5}
                                 for c in mean},
               "loop_rate": {c: float(np.mean([looping(t) for t in g]))
                             for c, g in BH["generations"].items()},
               "mean_chars": {c: float(np.mean([len(t) for t in g]))
                              for c, g in BH["generations"].items()},
               }
        # own-factor movement, signed by keying
        own = {}
        for t in TRAITS:
            f, k = meta[t]
            sgn = 1.0 if k == "+" else -1.0
            sl = f"slider_{t}"; s1 = f"stage1_{t}"
            own[t] = {"factor": f, "keyed": k,
                      "slider_own_factor_move": sgn * (mean[sl][f] - base[f])
                      if sl in mean else None,
                      "stage1_own_factor_move_same_run": sgn * (mean[s1][f] - base[f])
                      if s1 in mean else None}
        # the external reference: judged_100.json, 200-token generations
        J100 = json.load(open(f"{Q}/phase10_runs/judged_100.json"))["records"]
        agg = {}
        for r in J100:
            agg.setdefault((r["trait"], r["condition"]), []).append(r["scores"])
        for t in TRAITS:
            f, k = meta[t]
            sgn = 1.0 if k == "+" else -1.0
            b = agg.get((t, "base")); s = agg.get((t, "stage1"))
            if b and s:
                # judged_100.json carries a null score where the judge gave
                # no rating on a scale; those records are dropped, not zeroed
                bv = [x[f] for x in b if x.get(f) is not None]
                sv = [x[f] for x in s if x.get(f) is not None]
                own[t]["stage1_own_factor_move_judged_100"] = sgn * float(
                    np.mean(sv) - np.mean(bv))
                own[t]["judged_100_n"] = [len(bv), len(sv)]
        beh["own_factor_movement"] = own
        # the hole and its controls
        AM = json.load(open(f"{Q}/analysis/alien_match_fa.json"))
        hole = {"_merge_reference": {
            "source": "analysis/alien_match_fa.json",
            "signs": AM.get("signs"), "r": AM.get("r"), "obs": AM.get("obs")}}
        for n in HOLES:
            c = f"slider_{n}"
            if c not in mean:
                continue
            obs = np.array([mean[c][f] - base[f] for f in F5])
            # The pre-registered comparator is the chart's prediction for THIS
            # DIRECTION, i.e. the weight-space merge's coordinates in
            # analysis/alien_match_fa.json#pred -- the same `pred` the merge was
            # scored against on the alien-direction page.  The slider's OWN chart
            # coordinates are also recorded, but a slider carries only a few per
            # cent of its norm inside the chart, so they are a residue and are
            # not the like-for-like test.
            predm = np.array([AM["pred"][f] for f in F5]) if isinstance(
                AM["pred"], dict) else np.array(AM["pred"])
            pred = np.array(W["sliders"][n]["big_five_axis_coords"])
            hole[n] = {"pred_merge": predm.tolist(), "obs": obs.tolist(),
                       "signs_vs_merge_pred": int(sum(np.sign(predm) == np.sign(obs))),
                       "r_vs_merge_pred": float(np.corrcoef(predm, obs)[0, 1]),
                       "pred_own_chart_coords": pred.tolist(),
                       "signs_vs_own": int(sum(np.sign(pred) == np.sign(obs))),
                       "r_vs_own": float(np.corrcoef(pred, obs)[0, 1]),
                       "own_chart_frac_of_norm": W["sliders"][n]["chart_frac_of_norm"],
                       "loop_rate": beh["loop_rate"][c],
                       "mean_chars": beh["mean_chars"][c]}
        for a in HOLES:
            for b in HOLES:
                if a < b and f"slider_{a}" in mean and f"slider_{b}" in mean:
                    da = np.array([mean[f"slider_{a}"][f] - base[f] for f in F5])
                    db = np.array([mean[f"slider_{b}"][f] - base[f] for f in F5])
                    hole.setdefault("_pairwise", {})[f"{a}|{b}"] = {
                        "euclidean_profile_distance": float(np.linalg.norm(da - db)),
                        "cos": float(da @ db / (np.linalg.norm(da) *
                                                np.linalg.norm(db) + 1e-12))}
        beh["hole"] = hole
        out["behaviour"] = beh
    else:
        out["behaviour"] = {"status": "not run"}

    # ---------------------------------------------------------- self-checks --
    # Three things the interpretation rests on, recorded rather than assumed.
    chk = {
        "all_sliders_confined_to_low_blocks": {
            n: {"nonzero_B_modules": out["training"]["per_slider"][n]
                ["nonzero_B_modules"],
                "frac_of_frobenius_in_low_blocks": float(
                    W["sliders"][n]["frobenius_norm_low_blocks"] /
                    W["sliders"][n]["frobenius_norm"])}
            for n in names},
        "cross_gram_max_rel_diff_between_implementations": agree,
        "note": "a slider trained to a layer-16 target should have B non-zero "
                "on exactly the 124 modules in blocks 0-15 and all of its "
                "Frobenius norm there; the two cross-Gram implementations "
                "should agree to floating-point noise",
    }
    out["checks"] = chk
    json.dump(out, open(f"{Q}/analysis/persona_sliders.json", "w"), indent=1)
    print("wrote analysis/persona_sliders.json")
    print("\nheld-out free-running cosine with own target, and the stage-one "
          "adapter's own value:")
    for n in names:
        c = out["activation"][n]["half1"]["cos_own_target"]
        b = TM["comparators"][n].get("adapter_half1_raw_vs_target")
        print(f"  {n:16s} slider {c:+.4f}   stage-one adapter "
              f"{'n/a' if b is None else format(b, '+.4f')}   "
              f"argmax {out['activation'][n]['heldout_argmax_target']}")


if __name__ == "__main__":
    main()
