#!/usr/bin/env python
"""
Behavioural composition report.

Reads results/behaviour.json (written by judge.py) and answers the behavioural
half of the experiment's question:

    does base + dW_X + dW_Y BEHAVE like the jointly-trained model?

Structure of the report, in the order it is printed:

  1. SANITY CHECK, stated first and loudly: does each single-trait adapter score
     higher on ITS OWN trait than base does? If the singles do not move their own
     trait the whole experiment is void, so this is never buried.

  2. NOISE FLOOR. A distance in 5-dim trait space is meaningless without a scale.
     Two floors are computed:
       - half-split: d(T_h1, T_h2) -- two adapters trained on disjoint halves of
         the SAME trait data. Same destination by construction.
       - reseed:     d(T, T_s1)    -- same data, different data order.
       - judge floor: the distance two IDENTICAL models would show from judge
         noise alone, sqrt(sum_t (se_a[t]^2 + se_b[t]^2)).
     Anything at or below these floors is "indistinguishable".

  3. PER PAIR (X,Y): the 5-dim trait profile of base, single X, single Y, the
     sum, the compositional joint and the union joint; then the key numbers,
     d(sum, compositional) and d(sum, union), each quoted as a multiple of the
     noise floor.

Degrades gracefully: any config still training is simply omitted and listed
under "missing" at the end.

Usage:
    ~/cartovenv/bin/python behaviour_report.py
    ~/cartovenv/bin/python behaviour_report.py --in results/behaviour.json
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
IN = os.path.join(RESULTS, "behaviour.json")
OUT = os.path.join(RESULTS, "behaviour_summary.md")

TRAITS = ["O", "C", "E", "A", "N"]
PAIRS = [(a, b) for i, a in enumerate(TRAITS) for b in TRAITS[i + 1:]]
# Only these singles have half-split / reseed control arms trained.
FLOOR_TRAITS = ["O", "C", "E"]

TRAIT_NAMES = {"O": "Openness", "C": "Conscientiousness", "E": "Extraversion",
               "A": "Agreeableness", "N": "Neuroticism"}


class Profiles:
    """Trait means/SEs keyed by config name, tolerant about spelling."""

    def __init__(self, blob):
        self.raw = blob["configs"]
        self.judge_model = blob.get("judge_model", "?")
        self.norm = {}
        for k, v in self.raw.items():
            self.norm[self._key(k)] = v

    @staticmethod
    def _key(name):
        n = name.strip()
        if n.startswith("single:"):
            n = "adapter:" + n[len("single:"):]
        return n

    def has(self, name):
        return self._key(name) in self.norm

    def get(self, name):
        return self.norm.get(self._key(name))

    def mean(self, name):
        e = self.get(name)
        return None if e is None else [e["mean"][t] for t in TRAITS]

    def se(self, name):
        e = self.get(name)
        return None if e is None else [e["se"][t] for t in TRAITS]

    def n(self, name):
        e = self.get(name)
        return None if e is None else e["n"]


def dist(u, v):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(u, v)))


def dist_se(u, su, v, sv):
    """First-order propagated SE of the Euclidean distance."""
    d = dist(u, v)
    if d == 0:
        return 0.0
    var = sum(((a - b) ** 2) * (sa ** 2 + sb ** 2)
              for a, b, sa, sb in zip(u, v, su, sv))
    return math.sqrt(var) / d


def judge_floor(su, sv):
    """Distance two IDENTICAL models would show from judge noise alone."""
    return math.sqrt(sum(a ** 2 + b ** 2 for a, b in zip(su, sv)))


def pair_dist(P, a, b):
    """(distance, se, judge_floor) or None if either config is missing."""
    if not (P.has(a) and P.has(b)):
        return None
    ua, ub = P.mean(a), P.mean(b)
    sa, sb = P.se(a), P.se(b)
    return dist(ua, ub), dist_se(ua, sa, ub, sb), judge_floor(sa, sb)


def fmt_profile(vals):
    return "  ".join(f"{v:5.2f}" for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=IN)
    ap.add_argument("--out", dest="out", default=OUT)
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit(f"{args.inp} missing -- run judge.py first")
    P = Profiles(json.load(open(args.inp)))

    L = []          # markdown lines
    def emit(s=""):
        print(s)
        L.append(s)

    have = sorted(P.norm)
    emit("# Behavioural composition report")
    emit()
    emit(f"- judge model: `{P.judge_model}`")
    emit(f"- configs scored: **{len(have)}**")
    emit(f"- trait order: {' '.join(TRAITS)} "
         f"({', '.join(TRAIT_NAMES[t] for t in TRAITS)})")
    emit(f"- probes per config: "
         f"{sorted({P.n(c) for c in have})}")
    emit()

    # ------------------------------------------------------------------
    # 1. SANITY CHECK -- first, loud, unmissable
    # ------------------------------------------------------------------
    emit("## 1. SANITY CHECK: do the single-trait adapters move their own trait?")
    emit()
    emit("If a single-trait adapter does not score higher on its own trait than "
         "base, the whole experiment is void.")
    emit()
    void = False
    if not P.has("base"):
        emit("> **CANNOT CHECK -- `base` has not been evaluated yet.**")
        emit()
        verdict_line = "SANITY CHECK: NOT RUNNABLE (no base config)"
    else:
        bm, bs = P.mean("base"), P.se("base")
        emit("| single | own trait | base | adapter | delta | delta / SE | verdict |")
        emit("|---|---|---|---|---|---|---|")
        checked, passed, failures = 0, 0, []
        for i, t in enumerate(TRAITS):
            cfg = f"adapter:{t}"
            if not P.has(cfg):
                emit(f"| `{cfg}` | {t} | {bm[i]:.2f} | _not trained yet_ | - | - "
                     f"| SKIPPED |")
                continue
            am, asx = P.mean(cfg), P.se(cfg)
            delta = am[i] - bm[i]
            sed = math.sqrt(asx[i] ** 2 + bs[i] ** 2)
            z = delta / sed if sed > 0 else float("inf")
            ok = delta > 0 and z >= 2.0
            weak = delta > 0 and z < 2.0
            checked += 1
            passed += ok
            mark = "PASS" if ok else ("WEAK (not 2 SE)" if weak else "**FAIL**")
            if not ok:
                failures.append((t, delta, z))
            emit(f"| `{cfg}` | {t} | {bm[i]:.2f} | {am[i]:.2f} | {delta:+.2f} "
                 f"| {z:+.1f} | {mark} |")
        emit()
        if checked == 0:
            verdict_line = "SANITY CHECK: NOT RUNNABLE (no single-trait adapters scored)"
            emit(f"> **{verdict_line}**")
        elif passed == checked:
            verdict_line = (f"SANITY CHECK PASSED: {passed}/{checked} single-trait "
                            f"adapters raise their own trait above base by >=2 SE.")
            emit(f"> **{verdict_line}**")
        else:
            void = True
            verdict_line = (f"SANITY CHECK FAILED: only {passed}/{checked} "
                            f"single-trait adapters raise their own trait above base "
                            f"by >=2 SE. Offenders: "
                            + ", ".join(f"{t} (delta {d:+.2f}, {z:+.1f} SE)"
                                        for t, d, z in failures)
                            + ". THE COMPOSITION RESULT IS NOT INTERPRETABLE.")
            emit(f"> ### !!! {verdict_line} !!!")
        emit()
        if checked < len(TRAITS):
            emit(f"_(only {checked}/{len(TRAITS)} singles evaluated so far; "
                 f"re-run when training finishes)_")
            emit()

    # ------------------------------------------------------------------
    # 2. NOISE FLOOR
    # ------------------------------------------------------------------
    emit("## 2. Noise floor")
    emit()
    emit("Distances below are Euclidean in the 5-dim mean-trait space. These are "
         "the distances between models that should be behaviourally the SAME:")
    emit()
    emit("| kind | pair | distance | +- SE | judge-noise floor |")
    emit("|---|---|---|---|---|")
    half, reseed, jfloors = [], [], []
    for t in FLOOR_TRAITS:
        r = pair_dist(P, f"adapter:{t}_h1", f"adapter:{t}_h2")
        if r:
            half.append(r[0]); jfloors.append(r[2])
            emit(f"| half-split | `{t}_h1` vs `{t}_h2` | {r[0]:.3f} | {r[1]:.3f} "
                 f"| {r[2]:.3f} |")
    for t in FLOOR_TRAITS:
        r = pair_dist(P, f"adapter:{t}", f"adapter:{t}_s1")
        if r:
            reseed.append(r[0]); jfloors.append(r[2])
            emit(f"| reseed | `{t}` vs `{t}_s1` | {r[0]:.3f} | {r[1]:.3f} "
                 f"| {r[2]:.3f} |")
    emit()

    floor = None
    if half or reseed:
        allf = half + reseed
        floor = max(allf)
        emit(f"- half-split distances: "
             f"{', '.join(f'{d:.3f}' for d in half) or 'none yet'}")
        emit(f"- reseed distances:     "
             f"{', '.join(f'{d:.3f}' for d in reseed) or 'none yet'}")
        emit(f"- mean judge-noise floor: "
             f"{sum(jfloors)/len(jfloors):.3f}" if jfloors else "")
        emit()
        emit(f"> **NOISE FLOOR = {floor:.3f}** (the largest same-destination "
             f"distance observed, over {len(allf)} control pair(s)). Any "
             f"sum-vs-joint distance at or below this is behaviourally "
             f"indistinguishable.")
    else:
        emit("> **NO NOISE FLOOR AVAILABLE YET** -- the `_h1`/`_h2` and `_s1` "
             "control arms have not been evaluated. Every distance below is "
             "therefore UNINTERPRETABLE in absolute terms; only relative "
             "comparisons are safe.")
    emit()

    # ------------------------------------------------------------------
    # 3. PER-PAIR PROFILES AND DISTANCES
    # ------------------------------------------------------------------
    emit("## 3. Per-pair trait profiles and composition distances")
    emit()
    summary_rows = []
    n_pairs_shown = 0
    for x, y in PAIRS:
        members = [
            ("base", "base"),
            (f"single {x}", f"adapter:{x}"),
            (f"single {y}", f"adapter:{y}"),
            (f"SUM {x}+{y}", f"sum:{x}+{y}"),
            (f"joint {x}_{y} (compositional)", f"adapter:{x}_{y}"),
            (f"joint {x}_{y}_union", f"adapter:{x}_{y}_union"),
        ]
        present = [(lab, c) for lab, c in members if P.has(c)]
        if not any(c.startswith("sum:") for _, c in present):
            continue  # nothing to say about this pair without the sum
        n_pairs_shown += 1
        emit(f"### {x} + {y}")
        emit()
        emit("| model | config | " + " | ".join(TRAITS) + " |")
        emit("|---|---|" + "---|" * len(TRAITS))
        for lab, c in members:
            if not P.has(c):
                emit(f"| {lab} | `{c}` | " + " | ".join(["_-_"] * len(TRAITS)) + " |")
                continue
            m, s = P.mean(c), P.se(c)
            emit(f"| {lab} | `{c}` | "
                 + " | ".join(f"{mi:.2f}<br><sub>±{si:.2f}</sub>"
                              for mi, si in zip(m, s)) + " |")
        emit()

        sumc = f"sum:{x}+{y}"
        rows = []
        for lab, other in ((f"joint {x}_{y} (compositional)", f"adapter:{x}_{y}"),
                           (f"joint {x}_{y}_union", f"adapter:{x}_{y}_union"),
                           ("base", "base"),
                           (f"single {x}", f"adapter:{x}"),
                           (f"single {y}", f"adapter:{y}")):
            r = pair_dist(P, sumc, other)
            if r:
                rows.append((lab, other, r))
        # context: how far apart are the two joints from each other
        rj = pair_dist(P, f"adapter:{x}_{y}", f"adapter:{x}_{y}_union")

        emit("| distance from SUM to | value | +- SE | vs noise floor |")
        emit("|---|---|---|---|")
        for lab, other, (d, se, _jf) in rows:
            ratio = (f"{d/floor:.2f}x" if floor and floor > 0 else "n/a")
            flag = ""
            if floor and floor > 0:
                flag = (" (indistinguishable)" if d <= floor
                        else (" (**distinct**)" if d > 2 * floor else ""))
            emit(f"| {lab} | **{d:.3f}** | {se:.3f} | {ratio}{flag} |")
        if rj:
            emit(f"| _(reference: compositional vs union joint)_ | {rj[0]:.3f} "
                 f"| {rj[1]:.3f} | "
                 f"{(f'{rj[0]/floor:.2f}x' if floor and floor>0 else 'n/a')} |")
        emit()

        by_cfg = {other: v[0] for _lab, other, v in rows}
        d_comp = by_cfg.get(f"adapter:{x}_{y}")
        d_union = by_cfg.get(f"adapter:{x}_{y}_union")
        summary_rows.append((f"{x}+{y}", d_comp, d_union))

    if n_pairs_shown == 0:
        emit("_No pair has a `sum:` config evaluated yet -- nothing to compare._")
        emit()

    # ------------------------------------------------------------------
    # 4. HEADLINE TABLE
    # ------------------------------------------------------------------
    if summary_rows:
        emit("## 4. Headline: sum vs joint, in noise-floor units")
        emit()
        fl = f"{floor:.3f}" if floor else "UNKNOWN"
        emit(f"Noise floor = {fl}")
        emit()
        emit("| pair | d(sum, compositional) | d(sum, union) | "
             "d/floor (comp) | d/floor (union) |")
        emit("|---|---|---|---|---|")
        for name, dc, du in summary_rows:
            f_dc = f"{dc:.3f}" if dc is not None else "-"
            f_du = f"{du:.3f}" if du is not None else "-"
            r_dc = (f"{dc/floor:.2f}x" if (dc is not None and floor) else "-")
            r_du = (f"{du/floor:.2f}x" if (du is not None and floor) else "-")
            emit(f"| {name} | {f_dc} | {f_du} | {r_dc} | {r_du} |")
        emit()

    # ------------------------------------------------------------------
    # 5. WHAT IS MISSING
    # ------------------------------------------------------------------
    wanted = ["base"] + [f"adapter:{t}" for t in TRAITS]
    wanted += [f"adapter:{x}_{y}" for x, y in PAIRS]
    wanted += [f"adapter:{x}_{y}_union" for x, y in PAIRS]
    wanted += [f"sum:{x}+{y}" for x, y in PAIRS]
    wanted += [f"adapter:{t}_h{h}" for t in FLOOR_TRAITS for h in (1, 2)]
    wanted += [f"adapter:{t}_s1" for t in FLOOR_TRAITS]
    missing = [c for c in wanted if not P.has(c)]
    extra = [c for c in have if c not in wanted]

    emit("## 5. Coverage")
    emit()
    emit(f"- scored: **{len(wanted) - len(missing)}/{len(wanted)}** of the study "
         f"configs")
    if missing:
        emit(f"- **missing ({len(missing)})**: "
             + ", ".join(f"`{c}`" for c in missing))
    if extra:
        emit(f"- extra (not part of the standard set): "
             + ", ".join(f"`{c}`" for c in extra))
    emit()
    emit("Re-run `eval_modal.py --all-pairs`, `fetch_evals.py`, `judge.py`, then "
         "this script as training completes; everything is resumable.")
    emit()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n[wrote {args.out}]")

    # Non-zero exit if the experiment is provably void, so a pipeline notices.
    return 2 if void else 0


if __name__ == "__main__":
    sys.exit(main())
