#!/usr/bin/env python3
"""S3, the rank sweep: what does the zoo's geometry owe to its rank 64?

Reads results/rank_sweep_grams.npz (cross_gram_rank_sweep.py) and, per rank
r in {1, 4, 16}, computes against the rank-64 stage-one zoo:

  same-trait cosine        <dW_r(t), dW_64(t)> / (|dW_r| |dW_64|).  This is
                           meaningful ONLY because the frames nest: every
                           rank-r run was initialised with the zoo's own
                           seed-0 LoRA-A truncated to its first r rows, so
                           span(A_r) is a subspace of span(A_64) rather than
                           an independent random draw.  The nesting is
                           measured after training, not assumed --
                           phase2_runs/results_data_rank_sweep.json#a0_nesting.
  identification           rank of the true trait when a rank-r adapter is
                           scored against all 134 rank-64 adapters by cosine.
                           Chance mean rank 67.5.
  chart coordinates        fa_chart.FAChart().coords_external on the 134-column
                           of the cross-Gram, against the rank-64 adapter's own
                           chart coordinates (ch.trait_coords).
  norm ratio               |dW_r| / |dW_64|.
  Frobenius capture        two named statistics: `projection_coefficient` =
                           <dW_r, dW_64> / |dW_64|^2, the coefficient of the
                           least-squares fit of dW_64 by dW_r; and
                           `energy_fraction` = cos^2, the fraction of dW_64's
                           squared Frobenius norm explained by dW_r.
  arrangement              the 15 x 15 sub-Gram of cosines at rank r against
                           the same 15 x 15 block of the rank-64 Gram, Pearson
                           and Spearman over the 105 off-diagonal pairs.  This
                           is the statistic cross-seed-geometry.md uses, where
                           the two seeds agree at 0.997.
  cross-seed floor         same-trait cosine against the rank-64 SECOND-SEED
                           adapters.  Those frames do NOT nest with these (a
                           different A draw), so this is a floor, not a
                           replication statistic.

usage:  python analyse_rank_sweep.py
out:    analysis/rank_sweep.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RANKS = [1, 4, 16]
GRAMS = f"{HERE}/results/rank_sweep_grams.npz"
RESULTS = f"{HERE}/phase2_runs/results_data_rank_sweep.json"
OUT = f"{HERE}/analysis/rank_sweep.json"


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def rk(x):
        o = np.argsort(np.argsort(np.asarray(x, float)))
        return o.astype(float)
    return pearson(rk(a), rk(b))


DIMS = ["Extraversion", "Agreeableness", "Conscientiousness",
        "EmotionalStability", "Intellect"]
JUDGED = f"{HERE}/phase10_runs/judged_rank_sweep.json"
JUDGED_100 = f"{HERE}/phase10_runs/judged_100.json"


def _pct(x, base):
    """analyse_s2register.pct, verbatim: percent of the available headroom."""
    return 100 * ((x - base) / (7 - base) if x >= base
                  else (x - base) / (base - 1))


def _mean_profile(recs):
    import statistics as st
    out = {}
    for d in DIMS:
        v = [r["scores"][d] for r in recs
             if r.get("scores") and r["scores"].get(d) is not None]
        out[d] = st.mean(v) if v else None
    return out


def behaviour():
    """Own-factor amplification at rank 1 and 4 against the rank-64 stage-one
    values in phase10_runs/judged_100.json.

    The statistic is analyse_s2register.py's, unchanged: sign the trait by its
    keying, then express the judged own-factor mean as a percentage of the
    headroom between the base condition's own-factor mean and the end of the
    1-7 scale.  Each judged file supplies its OWN base condition, because a
    base mean is a property of the run that produced it.
    """
    import statistics as st
    if not (os.path.exists(JUDGED) and os.path.exists(JUDGED_100)):
        return None
    prim = {r["trait"].lower(): r for r in
            json.load(open(f"{HERE}/traits_primary.json"))}
    J = json.load(open(JUDGED))
    J100 = json.load(open(JUDGED_100))
    R, R100 = J["records"], J100["records"]
    base_new = _mean_profile([r for r in R if r["condition"] == "base"])
    base_100 = _mean_profile([r for r in R100 if r["condition"] == "base"])

    def amp(recs, basep, trait):
        f = prim[trait]["factor"].replace(" ", "")
        sgn = 1.0 if prim[trait]["keyed"] == "+" else -1.0
        m = _mean_profile(recs)
        if m[f] is None or basep[f] is None:
            return None
        return sgn * _pct(m[f], basep[f])

    # records in judged_rank_sweep carry trait "r<r>/<slug>"
    slugs = sorted({r["trait"].split("/")[-1] for r in R
                    if "/" in r["trait"]})
    per_rank = {}
    for rk in sorted({r["trait"].split("/")[0] for r in R if "/" in r["trait"]}):
        rows, rows64, per_trait = [], [], {}
        for t in slugs:
            rs = [r for r in R if r["trait"] == f"{rk}/{t}"
                  and r["condition"] == "stage1"]
            r64 = [r for r in R100 if r["trait"] == t
                   and r["condition"] == "stage1"]
            if not rs or not r64:
                continue
            a, a64 = amp(rs, base_new, t), amp(r64, base_100, t)
            per_trait[t] = {"amplification": a, "amplification_rank64": a64,
                            "factor": prim[t]["factor"],
                            "keyed": prim[t]["keyed"],
                            "n_judged": len(rs), "n_judged_rank64": len(r64)}
            rows.append(a); rows64.append(a64)
        if len(rows) < 3:
            continue
        mo, mn = st.mean(rows64), st.mean(rows)
        so, sn = st.pstdev(rows64), st.pstdev(rows)
        pr = (sum((x - mn) * (y - mo) for x, y in zip(rows, rows64))
              / (len(rows) * sn * so)) if sn > 0 and so > 0 else None
        per_rank[rk] = {
            "n_traits": len(rows),
            "mean_own_factor_amplification": mn,
            "sd_own_factor_amplification": st.stdev(rows),
            "sem_own_factor_amplification": st.stdev(rows) / len(rows) ** 0.5,
            "mean_own_factor_amplification_rank64": mo,
            "sd_own_factor_amplification_rank64": st.stdev(rows64),
            "sem_own_factor_amplification_rank64": (
                st.stdev(rows64) / len(rows64) ** 0.5),
            "min_own_factor_amplification": min(rows),
            "max_own_factor_amplification": max(rows),
            "ratio_to_rank64": (mn / mo) if mo else None,
            "pearson_with_rank64_over_traits": pr,
            "n_positive": int(sum(1 for x in rows if x > 0)),
            "per_trait": per_trait,
        }
    # judge repeat reliability, recomputed here from the file rather than left
    # in the judge's stdout: judge_personas.py duplicates 5 percent of items and
    # writes the second judgment under `repeat`.
    rel = {}
    pairs = [(o["scores"], o["repeat"]) for o in R if o.get("repeat")]
    for f in DIMS:
        x = [p[0][f] for p in pairs if p[0].get(f) and p[1].get(f)]
        y = [p[1][f] for p in pairs if p[0].get(f) and p[1].get(f)]
        if len(x) > 2 and st.pstdev(x) > 0 and st.pstdev(y) > 0:
            rel[f] = {"pearson": (
                sum((a - st.mean(x)) * (b - st.mean(y)) for a, b in zip(x, y))
                / (len(x) * st.pstdev(x) * st.pstdev(y))), "n": len(x)}

    return {
        "judge_repeat_reliability": rel,
        "judge_repeat_reliability_source": (
            "phase10_runs/judged_rank_sweep.json#records[*].repeat, the second "
            "judgment of the 5 percent of items judge_personas.py duplicates"),
        "what": ("24-prompt Big Five battery, greedy, thinking off, 200 new "
                 "tokens -- the same decoding oct_stage2.py:1043 used for "
                 "judged_100.json -- judged blind by "
                 f"{J.get('model')} through judge_personas.py"),
        "definition": ("own-factor amplification = sign(keying) x 100 x "
                       "(own-factor judged mean - base mean) / headroom, "
                       "analyse_s2register.py:pct"),
        "judged_file": "phase10_runs/judged_rank_sweep.json",
        "rank64_file": "phase10_runs/judged_100.json#records condition stage1",
        "judge_model": J.get("model"),
        "judge_model_rank64": J100.get("model"),
        "n_judgments": J.get("n"), "failed_calls": J.get("failed_calls"),
        "base_profile_rank_sweep_run": base_new,
        "base_profile_judged_100": base_100,
        "ranks": per_rank,
    }


def main():
    from fa_chart import FAChart
    z = np.load(GRAMS, allow_pickle=True)
    ch = FAChart()

    names134 = [str(x) for x in z["names_stage1_134"]]
    norms134 = np.array(z["norms_stage1_134"], float)
    names_s1 = [str(x) for x in z["names_seed1_40"]]
    norms_s1 = np.array(z["norms_seed1_40"], float)
    if set(names134) != set(ch.names):
        raise SystemExit(
            f"the cross-Gram's 134 and the chart's 134 differ: "
            f"{sorted(set(names134) ^ set(ch.names))[:5]}")
    idx134 = {n: i for i, n in enumerate(names134)}
    idx_s1 = {n: i for i, n in enumerate(names_s1)}
    # the chart's Gram gives the rank-64 norms and the rank-64 15 x 15 block
    G64 = ch.G
    n64 = ch.norms
    ich = {n: i for i, n in enumerate(ch.names)}

    runs = {}
    if os.path.exists(RESULTS):
        for rec in json.load(open(RESULTS)):
            runs[(rec["trait"], rec["lora_r"])] = rec

    out = {
        "what": ("15 zoo traits retrained at lora_r 1, 4 and 16 with "
                 "lora_alpha = 2r (effective scale 2.0, the 134's own) and the "
                 "zoo's seed-0 rank-64 LoRA-A truncated to the first r rows, "
                 "everything else identical to the 134-run stage-one sweep"),
        "sources": {
            "grams": "results/rank_sweep_grams.npz",
            "chart": ("fa_chart.FAChart over results/gram_sweep.npz and "
                      "phase10_runs/steer_spec2_7a.json"),
            "runs": "phase2_runs/results_data_rank_sweep.json",
            "rank64_reference": "the 134 stage-one adapters, pc-qwen35-sweep root",
            "seed1_reference": ("the 40 objective-matched second-seed adapters, "
                                "pc-qwen35-sweep:data_null_seedpaired_s40_matched"),
        },
        "definitions": {
            "same_trait_cosine": "<dW_r, dW_64> / (|dW_r| |dW_64|)",
            "projection_coefficient": "<dW_r, dW_64> / |dW_64|^2",
            "energy_fraction": ("cos^2 -- fraction of |dW_64|_F^2 explained by "
                                "the best scalar multiple of dW_r"),
            "norm_ratio": "|dW_r| / |dW_64|",
            "chart_pearson_75": ("Pearson over the 15 traits x 5 chart axes of "
                                 "coords_external(rank-r column) against the "
                                 "rank-64 adapter's own chart coordinates"),
            "arrangement_pearson": ("Pearson over the 105 off-diagonal pairs of "
                                    "the 15 x 15 cosine matrix at rank r "
                                    "against the same block at rank 64"),
            "chance_mean_rank": 67.5,
        },
        "n_modules": int(z["n_modules"]),
        "ranks": {},
    }

    for r in RANKS:
        key = f"r{r}"
        namesr = [str(x) for x in z[f"names_{key}"]]
        normsr = np.array(z[f"norms_{key}"], float)
        X = np.array(z[f"X_stage1_134__x__{key}"], float)      # 134 x 15
        Xs = np.array(z[f"X_seed1_40__x__{key}"], float)       # 40 x 15
        Xrr = np.array(z[f"X_{key}__x__{key}"], float)         # 15 x 15
        C = X / np.outer(norms134, normsr)                     # cosines

        same, proj, energy, nratio, ranks_, coords, coords64 = (
            {}, {}, {}, {}, {}, {}, {})
        cross_seed = {}
        for j, t in enumerate(namesr):
            i = idx134[t]
            c = float(C[i, j])
            same[t] = c
            proj[t] = float(X[i, j] / norms134[i] ** 2)
            energy[t] = c * c
            nratio[t] = float(normsr[j] / norms134[i])
            order = np.argsort(-C[:, j])
            ranks_[t] = int(np.where(order == i)[0][0]) + 1
            col = np.array([X[idx134[n], j] for n in ch.names])
            coords[t] = ch.coords_external(col).tolist()
            coords64[t] = ch.trait_coords[ich[t]].tolist()
            if t in idx_s1:
                k = idx_s1[t]
                cross_seed[t] = float(Xs[k, j] / (norms_s1[k] * normsr[j]))

        cr = np.array([coords[t] for t in namesr])
        c64 = np.array([coords64[t] for t in namesr])
        # normalised: chart coordinates divided by the adapter's own norm, so
        # the correlation is about DIRECTION rather than about size
        crn = cr / np.array([normsr[namesr.index(t)] for t in namesr])[:, None]
        c64n = c64 / np.array([n64[ich[t]] for t in namesr])[:, None]

        # arrangement: 15 x 15 cosines at rank r against the rank-64 block
        Crr = Xrr / np.outer(normsr, normsr)
        sub = [ich[t] for t in namesr]
        C64 = G64[np.ix_(sub, sub)] / np.outer(n64[sub], n64[sub])
        iu = np.triu_indices(len(namesr), 1)

        health = {}
        for t in namesr:
            rec = runs.get((t, r))
            if rec:
                health[t] = {
                    "loss_first": rec.get("loss_first"),
                    "loss_last": rec.get("loss_last"),
                    "reward_margin": rec.get("reward_margin"),
                    "expected_scaling": rec.get("expected_scaling"),
                    "lora_alpha": rec.get("lora_alpha"),
                    "train_seconds": rec.get("train_seconds"),
                    "a0_drift_mean": (rec.get("a0_nesting") or {}).get("a0_drift_mean"),
                }

        def stats(d):
            v = np.array([d[t] for t in namesr], float)
            return {"per_trait": {t: d[t] for t in namesr},
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max())}

        out["ranks"][key] = {
            "lora_r": r, "lora_alpha": 2 * r, "effective_scale": 2.0,
            "n_traits": len(namesr), "traits": namesr,
            "same_trait_cosine_vs_rank64": stats(same),
            "projection_coefficient": stats(proj),
            "energy_fraction": stats(energy),
            "norm_ratio": stats(nratio),
            "identification_among_134": {
                "top1": int(sum(1 for t in namesr if ranks_[t] == 1)),
                "n": len(namesr),
                "mean_rank": float(np.mean([ranks_[t] for t in namesr])),
                "worst_rank": int(max(ranks_.values())),
                "per_trait_rank": ranks_,
                "min_same_trait_cos": float(min(same.values())),
                "max_off_trait_cos": float(max(
                    C[i, j] for j in range(len(namesr))
                    for i in range(len(names134))
                    if names134[i] != namesr[j])),
            },
            "chart": {
                "coords_rank_r": coords,
                "coords_rank_64": coords64,
                "pearson_75": pearson(cr.ravel(), c64.ravel()),
                "spearman_75": spearman(cr.ravel(), c64.ravel()),
                "pearson_75_norm_scaled": pearson(crn.ravel(), c64n.ravel()),
                "per_axis_pearson": {
                    ch.factor_names[a]: pearson(cr[:, a], c64[:, a])
                    for a in range(5)},
                "per_trait_direction_cosine": {
                    t: float(np.dot(cr[k], c64[k])
                             / (np.linalg.norm(cr[k]) * np.linalg.norm(c64[k])))
                    for k, t in enumerate(namesr)},
                "chart_len_ratio": {
                    t: float(np.linalg.norm(cr[k]) / np.linalg.norm(c64[k]))
                    for k, t in enumerate(namesr)},
                "chart_captures_frac_of_norm": {
                    t: float(np.linalg.norm(cr[k]) / normsr[k])
                    for k, t in enumerate(namesr)},
                "chart_captures_frac_of_norm_rank64": {
                    t: float(np.linalg.norm(c64[k]) / n64[ich[t]])
                    for k, t in enumerate(namesr)},
            },
            "arrangement_15x15": {
                "pearson_offdiag": pearson(Crr[iu], C64[iu]),
                "spearman_offdiag": spearman(Crr[iu], C64[iu]),
                "n_pairs": int(len(iu[0])),
                "mean_offdiag_rank_r": float(Crr[iu].mean()),
                "mean_offdiag_rank_64": float(C64[iu].mean()),
                "sd_offdiag_rank_r": float(Crr[iu].std(ddof=1)),
                "sd_offdiag_rank_64": float(C64[iu].std(ddof=1)),
            },
            # A nested rank-r frame inside a rank-64 one gives a same-trait
            # cosine of sqrt(r/64) if the delta's energy were spread evenly
            # over the 64 directions and the rank-r run recovered exactly the
            # first r of them.  It is a reference shape, not a fitted model.
            "sqrt_r_over_64": float(np.sqrt(r / 64.0)),
            "same_trait_cosine_over_sqrt_r_over_64": float(
                np.mean([same[t] for t in namesr]) / np.sqrt(r / 64.0)),
            "cross_seed_same_trait_cosine": (
                stats(cross_seed) if len(cross_seed) == len(namesr) else
                {"per_trait": cross_seed,
                 "note": "not every trait has a second seed"}),
            "training_health": health,
        }

    # the rank-64 references the ranks are compared against, quoted here so the
    # write-up can cite one file
    sub = [ich[t] for t in [str(x) for x in z["names_r1"]]]
    C64 = G64[np.ix_(sub, sub)] / np.outer(n64[sub], n64[sub])
    iu = np.triu_indices(len(sub), 1)
    out["rank64_reference"] = {
        "traits": [str(x) for x in z["names_r1"]],
        "norms": {ch.names[i]: float(n64[i]) for i in sub},
        "mean_offdiag_cosine_15x15": float(C64[iu].mean()),
        "chart_captures_frac_of_norm_mean_134": float(
            np.mean(ch.trait_chart_len / ch.norms)),
        "cross_seed_same_trait_mean_rank64": 0.01806099632415457,
        "cross_seed_source": "analysis/crossseed_arms.json#[1].same[1]",
    }
    for r in RANKS:
        d = out["ranks"][f"r{r}"]
        cs = d["cross_seed_same_trait_cosine"].get("mean")
        if cs is not None:
            d["cross_seed_ratio_to_rank64_floor"] = cs / 0.01806099632415457

    b = behaviour()
    if b:
        out["behaviour"] = b

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    for r in RANKS:
        d = out["ranks"][f"r{r}"]
        print(f"r={r:<3} same-trait cos {d['same_trait_cosine_vs_rank64']['mean']:.4f} "
              f"  top1 {d['identification_among_134']['top1']}/{d['n_traits']} "
              f"(mean rank {d['identification_among_134']['mean_rank']:.2f})"
              f"  chart r {d['chart']['pearson_75']:.4f}"
              f"  norm ratio {d['norm_ratio']['mean']:.4f}"
              f"  arrangement r {d['arrangement_15x15']['pearson_offdiag']:.4f}")


if __name__ == "__main__":
    main()
