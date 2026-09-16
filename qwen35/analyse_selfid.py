#!/usr/bin/env python
"""Score the self-identification run against PREREG_selfid.md.

Two modes, run in this order:

  --vocab   print the pooled vocabulary of normalised answers (every condition,
            every adapter, no adapter identity shown), so the noun/adjective
            form list can be written without seeing who said what
  (default) score: exact hit, chart hit, same factor and pole, permutation null,
            base rates; writes analysis/selfid.json

The form list lives in SELFID_FORMS below and is recorded into the output.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "results", "selfid_generations.json")

# noun / alternate forms -> zoo slug.  Written from the pooled --vocab list on
# 2026-09-15 BEFORE any adapter identity was looked at; only forms of words that
# ARE zoo traits appear here.  Answers like "curiosity" or "empathy" map to no
# zoo word and can only score through the chart's nearest-word rule below.
SELFID_FORMS = {
    "kindness": "kind", "efficiency": "efficient", "creativity": "creative", "sympathy": "sympathetic",
    "introspection": "introspective", "imagination": "imaginative", "helpfulness": "helpful",
    "steadiness": "steady", "shyness": "shy", "carefulness": "careful", "uncertainty": "uncertain",
    "boldness": "bold", "complexity": "complex", "quietness": "quiet", "quietly": "quiet",
    "innovation": "innovative", "jealousy": "jealous", "assertiveness": "assertive",
    "conscientiousness": "conscientious", "intellectually": "intellectual", "consideration": "considerate",
    "warmth": "warm", "courage": "courageous", "generosity": "generous", "practicality": "practical",
    "coldness": "cold", "energy": "energetic", "cooperation": "cooperative", "dependability": "dependable",
    "activity": "active", "simplicity": "simple", "brightness": "bright", "relaxation": "relaxed",
}


def norm(ans):
    """First line; drop everything after a blank line or a role marker; lowercase;
    strip punctuation and articles; hyphen to underscore; first word unless the
    line is at most three words."""
    a = ans.split("\n")[0]
    a = re.split(r"\buser\b|\bassistant\b", a)[0]
    a = a.strip().lower()
    a = re.sub(r"^(a|an|the|my|i am|i'm|i would be|i would say|it would be)\s+", "", a)
    a = re.sub(r"[^a-z\- ]", "", a).strip()
    a = a.replace("-", "_")
    w = a.split()
    if not w:
        return ""
    return a if len(w) <= 3 else w[0]


def load():
    d = json.load(open(GEN))
    viz = json.load(open(os.path.join(HERE, "analysis", "viz_fa.json")))
    traits = viz["traits"]
    X = np.asarray(viz["coords"], float); X = X - X.mean(0)
    factor, keyed = viz["factor"], viz["keyed"]
    return d, traits, X, factor, keyed


def vocab():
    d = json.load(open(GEN))
    from collections import Counter
    c = Counter()
    for pk in d["meta"]["prompts"]:
        for a in d["base"][pk]:
            c[norm(a)] += 1
        for t, rec in d["traits"].items():
            for cond, r in rec.items():
                for a in r[pk]:
                    c[norm(a)] += 1
    for w, n in c.most_common():
        print(f"{n:5d}  {w}")


def score():
    d, traits, X, factor, keyed = load()
    idx = {t: i for i, t in enumerate(traits)}
    slug_of = dict(SELFID_FORMS)
    for t in traits:
        slug_of.setdefault(t, t)
        slug_of.setdefault(t.replace("_", " "), t)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    conds = list(d["meta"]["adapter_src"])
    out = {"what": "self-identification probe scored per PREREG_selfid.md", "forms": SELFID_FORMS,
           "prompts": d["meta"]["prompts"], "missing": d["missing"], "per_prompt": {}}
    rng = np.random.default_rng(0)
    for pk in d["meta"]["prompts"]:
        res = {"base_answers": {}, "conditions": {}}
        from collections import Counter
        res["base_answers"] = dict(Counter(norm(a) for a in d["base"][pk]).most_common(10))
        base_in_zoo = [slug_of.get(norm(a)) for a in d["base"][pk]]
        res["base_share_in_zoo_vocab"] = float(np.mean([s is not None for s in base_in_zoo]))
        for cond in conds:
            exact, chart_cos, same_fp, in_vocab, n = [], [], [], 0, 0
            per_trait = {}
            answers_by_trait = {}
            for t, rec in d["traits"].items():
                if cond not in rec:
                    continue
                hits, cs, fp = 0, [], []
                for a in rec[cond][pk]:
                    n += 1
                    s = slug_of.get(norm(a))
                    if s is None:
                        continue
                    in_vocab += 1
                    hits += int(s == t)
                    cs.append(float(Xn[idx[s]] @ Xn[idx[t]]))
                    if factor[idx[t]] != "Lexicon" and factor[idx[s]] != "Lexicon":
                        fp.append(int(factor[idx[s]] == factor[idx[t]] and keyed[idx[s]] == keyed[idx[t]]))
                answers_by_trait[t] = [slug_of.get(norm(a)) for a in rec[cond][pk]]
                exact.append(hits / len(rec[cond][pk]))
                chart_cos += cs; same_fp += fp
                per_trait[t] = {"exact": hits, "n": len(rec[cond][pk]),
                                "top": dict(Counter(norm(a) for a in rec[cond][pk]).most_common(3))}
            # permutation null: reassign each adapter's answer list to a random adapter
            names = list(answers_by_trait)
            null_exact, null_cos = [], []
            for _ in range(200):
                perm = rng.permutation(len(names))
                e, c = [], []
                for i, t in enumerate(names):
                    ans = answers_by_trait[names[perm[i]]]
                    e.append(np.mean([s == t for s in ans]))
                    c += [float(Xn[idx[s]] @ Xn[idx[t]]) for s in ans if s is not None]
                null_exact.append(np.mean(e)); null_cos.append(np.mean(c) if c else np.nan)
            res["conditions"][cond] = {
                "n_answers": n, "share_in_zoo_vocab": in_vocab / max(n, 1),
                "exact_hit_rate": float(np.mean(exact)),
                "exact_hit_rate_null_mean": float(np.mean(null_exact)),
                "exact_hit_rate_null_p95": float(np.percentile(null_exact, 95)),
                "n_traits_with_any_exact_hit": int(sum(1 for t in per_trait if per_trait[t]["exact"] > 0)),
                "chart_cos_mean": float(np.mean(chart_cos)) if chart_cos else None,
                "chart_cos_share_above_0.5": float(np.mean(np.array(chart_cos) > 0.5)) if chart_cos else None,
                "chart_cos_null_mean": float(np.nanmean(null_cos)),
                "same_factor_and_pole_share": float(np.mean(same_fp)) if same_fp else None,
                "n_same_factor_pole_scored": len(same_fp),
                "per_trait": per_trait}
        out["per_prompt"][pk] = res
    json.dump(out, open(os.path.join(HERE, "analysis", "selfid.json"), "w"), indent=1)
    for pk, res in out["per_prompt"].items():
        print(f"\n== {pk}: {d['meta']['prompts'][pk]}")
        print("   base top answers:", res["base_answers"], f"(in zoo vocab {res['base_share_in_zoo_vocab']:.2f})")
        for cond, r in res["conditions"].items():
            print(f"   {cond:8s} exact {r['exact_hit_rate']:.3f} (null {r['exact_hit_rate_null_mean']:.3f}, p95 {r['exact_hit_rate_null_p95']:.3f}); "
                  f"traits with a hit {r['n_traits_with_any_exact_hit']}/134; in-vocab {r['share_in_zoo_vocab']:.2f}; "
                  f"chart cos {r['chart_cos_mean']} (null {r['chart_cos_null_mean']:.3f}); same factor+pole {r['same_factor_and_pole_share']} of {r['n_same_factor_pole_scored']}")


if __name__ == "__main__":
    vocab() if "--vocab" in sys.argv else score()
