#!/usr/bin/env python3
"""steer_results_s2register.json -> phase10_runs/eval_s2register.json, the
judge-shaped file for judge_personas.py.

Conditions are opaque labels (cond_a..cond_e); the mapping lives in
phase10_runs/eval_s2register_conditions.json and nowhere the judge can see it.
The judge is only ever handed (prompt, response) pairs anyway, and shuffles the
whole pool before batching, but the labels are kept opaque so that nothing about
the arm can leak through a stray dump.

The base condition is trait-independent (greedy decoding, no adapter), so it is
carried once under the pseudo-trait "_base" rather than copied 15 times: that is
15x fewer judge calls for the same information, and build_spider_data.py already
treats the base as a single pooled reference.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# job name pattern + alpha -> opaque condition label
LABELS = {
    ("base", 0.0): ("_base", "cond_a", "base model, no adapter, no direction"),
    ("s1", 0.0): (None, "cond_b", "base + stage-one DPO adapter"),
    ("s1", 0.389): (None, "cond_c", "stage one + S2_mean at alpha 0.389 ref"),
    ("s1", 0.779): (None, "cond_c2", "stage one + S2_mean at alpha 0.779 ref "
                                     "(the persona's actual dose; ref is HALF a "
                                     "stage-one adapter's Frobenius norm)"),
    ("s1", 1.0): (None, "cond_d", "stage one + S2_mean at alpha 1.0 ref"),
    ("pex", 0.0): (None, "cond_e", "base + the exact persona (rank 128, scale 1.0)"),
}


def main():
    R = json.load(open(f"{HERE}/phase10_runs/steer_results_s2register.json"))
    spec = json.load(open(f"{HERE}/phase10_runs/steer_spec_s2register.json"))
    prompts = spec["jobs"][0]["prompts"]

    by_trait, base = {}, {}
    used = {}
    for j in R:
        nm = j["name"]
        assert nm.startswith("s2reg_")
        rest = nm[len("s2reg_"):]
        kind = rest.split("_", 1)[0] if rest != "base" else "base"
        trait = None if kind == "base" else rest.split("_", 1)[1]
        for a_str, texts in j["generations"].items():
            a = float(a_str)
            key = (kind, a)
            if key not in LABELS:
                raise SystemExit(f"no label for {key} (job {nm})")
            _, label, desc = LABELS[key]
            used[label] = desc
            assert len(texts) == len(prompts), (nm, a_str, len(texts))
            if kind == "base":
                base[label] = texts
            else:
                by_trait.setdefault(trait, {})[label] = texts

    out = [{"trait": "_base", "prompts": prompts, "generations": base}]
    for t in sorted(by_trait):
        out.append({"trait": t, "prompts": prompts, "generations": by_trait[t]})

    p = f"{HERE}/phase10_runs/eval_s2register.json"
    json.dump(out, open(p, "w"), indent=1)
    json.dump({"conditions": used,
               "note": ("Opaque labels for the blind Big Five judge. The judge "
                        "sees only (prompt, response); this file is the key."),
               "ref": spec["ref"], "dose_arithmetic": spec["dose_arithmetic"]},
              open(f"{HERE}/phase10_runs/eval_s2register_conditions.json", "w"),
              indent=1)
    n = sum(len(v) for r in out for v in r["generations"].values())
    print(f"wrote {p}: {len(out)} records, {n} generations, "
          f"conditions {sorted(used)}")


if __name__ == "__main__":
    main()
