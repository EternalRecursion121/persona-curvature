#!/usr/bin/env python3
"""The iso-KL sphere, against the alpha-1.5 sphere it re-samples.

Every statistic (a) to (d) of PREREG_sphere_isokl.md is computed by the SAME
functions the 2026-09-01 sphere page used -- `smoothness`, `coherence` and
`profiles` imported from build_sphere_page.py -- so the two runs are counted the
same way and not merely described the same way.  The script's first act is to
reproduce analysis/sphere_page.json#smooth and #coherence from the 2026-09-01
files; if that reproduction is not exact it stops, because every comparison
below would otherwise be against a number it cannot recreate.

The dose covariate is NOT analysis/fisher_norms.json's F.  Its 72 `sphere_S*`
keys are the FACTOR sphere (cosine 1.0 against phase10_runs/sphere_spec_fa.json,
median 0.0094 against sphere_spec.json), so F for these 72 principal-component
directions does not exist.  The covariate used instead is this experiment's own
calibration: the measured bf16 KL per token at alpha 1.5, which is the dose each
point actually received in the 2026-09-01 run, measured at the alpha that was
steered rather than extrapolated from |alpha| <= 0.5.
"""
import argparse
import json
import os

import numpy as np

from build_sphere_page import profiles, coherence, smoothness, F5

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/analysis"
P = f"{Q}/phase10_runs"
RNG = np.random.default_rng(20260910)


# ---------------------------------------------------------------- rank tools
def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    a, b = _rank(x), _rank(y)
    a -= a.mean(); b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pearson(x, y):
    a = np.asarray(x, float) - np.mean(x)
    b = np.asarray(y, float) - np.mean(y)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out, by the rank-residual formula."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = np.sqrt(max(1 - rxz ** 2, 1e-15) * max(1 - ryz ** 2, 1e-15))
    return float((rxy - rxz * ryz) / den), rxy, rxz, ryz


# ------------------------------------------------------- the pairwise fields
def fields(points, judged, names=None):
    """Angular distance and judged profile distance over every pair, plus the
    per-point unit vectors and profiles, in one fixed point order."""
    byname = {p["name"]: np.asarray(p["u"], float) for p in points}
    scored = [n for n in (names or sorted(byname))
              if judged.get(n, {}).get("top")]
    U = np.array([byname[n] for n in scored])
    Pf = np.array([[judged[n]["scores"][f] for f in F5] for n in scored])
    ang = np.degrees(np.arccos(np.clip(U @ U.T, -1, 1)))
    prof = np.linalg.norm(Pf[:, None, :] - Pf[None, :, :], axis=2)
    iu = np.triu_indices(len(scored), 1)
    return scored, U, Pf, ang, prof, iu


def perm_p(ang, prof, iu, n=10000, stat="rho", z=None):
    """Permutation p for rho (or partial rho) by permuting the PROFILES across
    points.  The 2,556 pairs are not independent, so permuting pairs would be
    wrong; permuting which point carries which judged profile is the null that
    breaks the position-profile link and keeps everything else."""
    a = ang[iu]
    obs = (spearman(a, prof[iu]) if stat == "rho"
           else partial_spearman(a, prof[iu], z[iu])[0])
    k = len(prof)
    hits = 0
    for _ in range(n):
        pi = RNG.permutation(k)
        pp = prof[np.ix_(pi, pi)][iu]
        v = (spearman(a, pp) if stat == "rho"
             else partial_spearman(a, pp, z[iu])[0])
        if abs(v) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (n + 1)


def boot_ci(ang, prof, n=2000):
    """95% interval on rho from resampling the POINTS.  Pairs of two copies of
    the same point are dropped: they carry angular distance 0 and profile
    distance 0 and would manufacture concordance the data do not have."""
    k = len(prof)
    out = []
    for _ in range(n):
        s = RNG.integers(0, k, k)
        i, j = np.triu_indices(k, 1)
        keep = s[i] != s[j]
        a = ang[s[i][keep], s[j][keep]]
        p = prof[s[i][keep], s[j][keep]]
        out.append(spearman(a, p))
    out = np.sort(out)
    return float(out[int(0.025 * n)]), float(out[int(0.975 * n)])


def scale_stats(judged, names):
    o = {}
    for f in F5:
        v = np.array([judged[n]["scores"][f] for n in names])
        o[f] = {"min": float(v.min()), "max": float(v.max()),
                "range": float(v.max() - v.min()), "mean": float(v.mean()),
                "argmin": names[int(v.argmin())], "argmax": names[int(v.argmax())]}
    tops = {}
    for n in names:
        tops[judged[n]["top"]] = tops.get(judged[n]["top"], 0) + 1
    o["top_counts"] = tops
    return o


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--boots", type=int, default=2000)
    a = ap.parse_args()

    out = {"what": "the 72-direction principal-component sphere re-sampled at "
                   "iso-KL dose, against the same lattice at alpha 1.5",
           "prereg": "qwen35/PREREG_sphere_isokl.md"}

    # ---- 0. reproduce the published PC-sphere statistics, exactly -----------
    lay = json.load(open(f"{A}/sphere_layout.json"))
    pcgen = json.load(open(f"{P}/sphere_results.json"))["generations"]
    pcjud = profiles(f"{P}/judged_sphere.json")
    pub = json.load(open(f"{A}/sphere_page.json"))
    rep_c = coherence(pcgen)
    rep_s = smoothness(lay["points"], pcjud)
    ok = (rep_c == pub["coherence"]
          and all(abs(rep_s[k] - pub["smooth"][k]) < 1e-12 for k in rep_s))
    print(f"[check] reproduce sphere_page.json coherence/smooth: {'OK' if ok else 'FAILED'}")
    if not ok:
        print("  recomputed coherence", rep_c, "\n  published ", pub["coherence"])
        print("  recomputed smooth   ", rep_s, "\n  published ", pub["smooth"])
        raise SystemExit("self-check failed; every comparison below would be unsafe")
    out["selfcheck"] = {"reproduces_sphere_page_json": True,
                        "coherence": rep_c, "smooth": rep_s}
    if a.check_only:
        json.dump(out, open("/dev/stdout", "w"), indent=1)
        return

    # ---- 1. the calibration: dose per point at alpha 1.5 and the alphas -----
    cal = json.load(open(f"{P}/sphere_isokl_calib.json"))
    alph = json.load(open(f"{A}/sphere_isokl_alphas.json"))
    kl15 = {n: r["per_alpha"]["bf16|1.5"]["kl"] for n, r in cal["results"].items()}
    out["dose_at_alpha_1_5"] = {
        "what": "measured KL(base||steered) per token, bf16 weights, on the "
                "4378 fixed token positions, at the alpha the 2026-09-01 sphere "
                "actually steered",
        "n": len(kl15), "min": float(min(kl15.values())),
        "median": float(np.median(list(kl15.values()))),
        "max": float(max(kl15.values())),
        "ratio_max_over_min": float(max(kl15.values()) / min(kl15.values())),
        "argmin": min(kl15, key=kl15.get), "argmax": max(kl15, key=kl15.get),
        "per_point": kl15}
    out["alphas"] = alph["summary"]

    # ---- 2. the iso-KL run --------------------------------------------------
    R = json.load(open(f"{P}/sphere_isokl_results.json"))
    gen = R["generations"]
    base_name = alph["base_point_name"]
    sgen = {k: v for k, v in gen.items() if k != base_name}
    ijud = profiles(f"{P}/judged_sphere_isokl.json")
    names = sorted(sgen)
    pts = [{"name": p["name"], "u": p["u"]} for p in lay["points"]]

    out["run"] = {"n_points": len(sgen), "base_point": base_name,
                  "n_generations": sum(len(v) for v in gen.values()),
                  "alphas_used": R.get("alphas", {})}

    # (c) coherence
    out["coherence"] = coherence(sgen)
    out["coherence_base"] = coherence({base_name: gen[base_name]})
    out["coherence_pc_alpha1_5"] = pub["coherence"]

    # loop rate per point, same detector, for the degeneration arms
    from analyse_alien_steer import looping
    lr = {n: sum(looping(t) for t in sgen[n]) / len(sgen[n]) for n in names}
    plr = {n: sum(looping(t) for t in pcgen[n]) / len(pcgen[n]) for n in sorted(pcgen)}
    out["loop_rates"] = {"iso_kl": lr, "pc_alpha1_5": plr}

    # (a)(b) smoothness
    sc, U, Pf, ang, prof, iu = fields(pts, ijud, names)
    sm = smoothness(pts, ijud)
    out["smooth"] = sm
    rho, p = perm_p(ang, prof, iu, n=a.perms)
    lo, hi = boot_ci(ang, prof, n=a.boots)
    out["smooth"].update({"perm_p": p, "boot_lo": lo, "boot_hi": hi,
                          "n_perms": a.perms, "n_boots": a.boots})
    out["smooth_pc_alpha1_5"] = dict(pub["smooth"])

    # the alpha-0 point against the only other base this project generated in
    # its own container: dose_BASE, made by steer_fix.py for the matched-dose
    # run on the same 24-prompt battery, greedy, 512 tokens, thinking off.  Two
    # different code paths (weight edit here, add_adapter there) reaching the
    # same base model should produce the same greedy text.
    IDX = [0, 1, 5, 9, 11, 14, 15, 22]
    DR = json.load(open(f"{P}/steer_results_dose.json"))
    db = [e for e in DR if e.get("name") == "dose_BASE"]
    if db:
        other = [db[0]["generations"]["0.0"][i] for i in IDX]
        mine = gen[base_name]
        ident = sum(a == b for a, b in zip(mine, other))
        pref = [len(os.path.commonprefix([a, b])) / max(len(a), len(b), 1)
                for a, b in zip(mine, other)]
        out["base_vs_dose_base"] = {
            "what": "the alpha-0 point of this run against dose_BASE from "
                    "phase10_runs/steer_results_dose.json, prompt for prompt",
            "n_prompts": len(mine), "n_identical": ident,
            "mean_shared_prefix_fraction": float(np.mean(pref)),
            "min_shared_prefix_fraction": float(np.min(pref))}

    # (d) per-scale
    out["scales"] = scale_stats(ijud, sc)
    out["scales_pc_alpha1_5"] = scale_stats(pcjud, sorted(pcjud))
    out["base_profile"] = ijud.get(base_name)

    # ---- 3. (e) the original sphere with dose partialled out ----------------
    psc, pU, pPf, pang, pprof, piu = fields(pts, pcjud, sorted(pcjud))
    d = np.array([kl15[n] for n in psc])
    zabs = np.abs(d[:, None] - d[None, :])
    zsum = d[:, None] + d[None, :]
    e = {}
    prho, pp = perm_p(pang, pprof, piu, n=a.perms)
    plo, phi = boot_ci(pang, pprof, n=a.boots)
    e["raw"] = {"rho": prho, "perm_p": pp, "boot_lo": plo, "boot_hi": phi}
    for lab, z in (("abs_dose_diff", zabs), ("dose_sum", zsum)):
        pr, rxy, rxz, ryz = partial_spearman(pang[piu], pprof[piu], z[piu])
        _, ppp = perm_p(pang, pprof, piu, n=a.perms, stat="partial", z=z)
        e[lab] = {"partial_rho": pr, "rho_ang_prof": rxy, "rho_ang_cov": rxz,
                  "rho_prof_cov": ryz, "perm_p": ppp,
                  "drop_from_raw": prho - pr}
    # the same partialling on the iso-KL field, where the covariate should be flat
    dl = None
    if os.path.exists(f"{P}/sphere_isokl_verify.json"):
        V = json.load(open(f"{P}/sphere_isokl_verify.json"))
        dl = {n: [r["kl"] for r in d["per_alpha"].values()][0]
              for n, d in V["results"].items()}
        dv = np.array([dl[n] for n in sorted(dl)])
        tgt = alph["kl_target_nats_per_token"]
        out["delivered"] = {
            "what": "KL per token actually delivered by each solved alpha, "
                    "re-measured on the same 4378 token positions",
            "n": len(dl), "target": tgt,
            "min": float(dv.min()), "median": float(np.median(dv)),
            "max": float(dv.max()),
            "ratio_max_over_min": float(dv.max() / dv.min()),
            "max_abs_rel_error": float(np.abs(dv / tgt - 1).max()),
            "median_abs_rel_error": float(np.median(np.abs(dv / tgt - 1))),
            "per_point": dl}
    if dl:
        di = np.array([dl[n] for n in sc])
        zi = np.abs(di[:, None] - di[None, :])
        pr, rxy, rxz, ryz = partial_spearman(ang[iu], prof[iu], zi[iu])
        e["iso_kl_abs_dose_diff"] = {"partial_rho": pr, "rho_ang_prof": rxy,
                                     "rho_ang_cov": rxz, "rho_prof_cov": ryz,
                                     "drop_from_raw": rho - pr}
    # the per-point statistic fisher-norms reports on the factor sphere
    for lab, jud, nm in (("pc_alpha1_5", pcjud, psc), ("iso_kl", ijud, sc)):
        Pp = np.array([[jud[n]["scores"][f] for f in F5] for n in nm])
        cen = np.linalg.norm(Pp - Pp.mean(0), axis=1)
        dd = np.array([kl15[n] for n in nm])
        r = spearman(dd, cen)
        hits = sum(abs(spearman(dd, RNG.permutation(cen))) >= abs(r)
                   for _ in range(a.perms))
        e[f"dose_vs_centroid_distance_{lab}"] = {
            "spearman": r, "perm_p": (hits + 1) / (a.perms + 1), "n": len(nm),
            "covariate": "measured bf16 KL at alpha 1.5"}
    # does steepness predict degeneration, on either sphere?
    ali = {n: alph["directions"][n]["alpha"] for n in names}

    def arm(x, y):
        r = spearman(x, y)
        y = np.asarray(y, float)
        h = sum(abs(spearman(x, RNG.permutation(y))) >= abs(r)
                for _ in range(a.perms))
        return {"spearman": r, "perm_p": (h + 1) / (a.perms + 1), "n": len(x)}

    e["steepness_vs_loop_rate_pc_alpha1_5"] = arm(
        [kl15[n] for n in sorted(plr)], [plr[n] for n in sorted(plr)])
    e["steepness_vs_loop_rate_iso_kl"] = arm(
        [kl15[n] for n in names], [lr[n] for n in names])
    e["alpha_vs_loop_rate_iso_kl"] = arm(
        [ali[n] for n in names], [lr[n] for n in names])
    e["steepness_vs_response_length_iso_kl"] = arm(
        [kl15[n] for n in names],
        [float(np.mean([len(t) for t in sgen[n]])) for n in names])
    out["dose_covariate"] = e

    # ---- 4. (f) the two judged fields, point for point ---------------------
    common = [n for n in sc if n in pcjud]
    f6 = {"n": len(common)}
    for f in F5:
        x = [ijud[n]["scores"][f] for n in common]
        y = [pcjud[n]["scores"][f] for n in common]
        f6[f] = {"pearson": pearson(x, y), "spearman": spearman(x, y),
                 "mean_abs_diff": float(np.mean(np.abs(np.array(x) - np.array(y)))),
                 "mean_iso": float(np.mean(x)), "mean_pc": float(np.mean(y))}
    Pi = np.array([[ijud[n]["scores"][f] for f in F5] for n in common])
    Pp = np.array([[pcjud[n]["scores"][f] for f in F5] for n in common])
    f6["mean_profile_distance"] = float(np.mean(np.linalg.norm(Pi - Pp, axis=1)))
    f6["same_top_scale"] = int(sum(ijud[n]["top"] == pcjud[n]["top"] for n in common))
    out["field_agreement"] = f6

    # ---- 5. the pre-registered verdict -------------------------------------
    rho_orig = pub["smooth"]["rho"]
    partial = e["abs_dose_diff"]["partial_rho"]
    rho_c = sm["rho"]
    dose = (rho_c < rho_orig - 0.15) and (partial < rho_orig - 0.10)
    posn = (abs(rho_c - rho_orig) <= 0.10) and (abs(partial - rho_orig) <= 0.05)
    out["verdict"] = {
        "rho_iso": sm["rho"], "rho_orig": rho_orig,
        "rho_iso_permutation_harness": rho,
        "rho_note": "smooth.rho is build_sphere_page.smoothness, the function "
                    "that produced the published sphere numbers; "
                    "rho_iso_permutation_harness is the same statistic "
                    "recomputed inside the permutation and bootstrap code and "
                    "differs in the seventh decimal because the two enumerate "
                    "the 2,556 pairs in a different order and ordinal ranking "
                    "breaks ties by position. The page quotes smooth.rho.", "rho_orig_partial_abs_dose_diff": partial,
        "criterion_dose": "rho_iso < rho_orig - 0.15 AND partial < rho_orig - 0.10",
        "criterion_position": "|rho_iso - rho_orig| <= 0.10 AND |partial - rho_orig| <= 0.05",
        "smoothness_was_dose": bool(dose), "smoothness_is_position": bool(posn),
        "call": "dose" if dose and not posn else
                ("position" if posn and not dose else "ambiguous")}

    json.dump(out, open(f"{A}/sphere_isokl.json", "w"), indent=1)
    print(f"\nwrote analysis/sphere_isokl.json")
    print(f"  rho iso-KL {rho:.4f} [{lo:.4f}, {hi:.4f}] p={p:.5f}   "
          f"alpha-1.5 {rho_orig:.4f}   partial(|dKL|) {partial:.4f}")
    print(f"  coherence iso-KL {out['coherence']}")
    print(f"  verdict: {out['verdict']['call']}")


if __name__ == "__main__":
    main()
