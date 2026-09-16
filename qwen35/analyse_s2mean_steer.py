#!/usr/bin/env python3
"""Text statistics for the stage-two shared-direction steer -> analysis/s2mean_steer_stats.json.
Reads steer_results_s2mean.json, steer_results_s2balanced.json (if present) and the stage-one
grand mean from steer_results_fix.json; computes the controls' cosine with the grand mean from
results/gram_stage2.npz."""
import json, os, re, statistics as st
import numpy as np
Q = os.path.dirname(os.path.abspath(__file__))

def stats(texts):
    n = len(texts); w = [len(t.split()) for t in texts]
    def rate(pat): return sum(len(re.findall(pat, t)) for t in texts) / max(1, sum(w)) * 1000
    return dict(words=round(st.mean(w)),
                first_person_per_k=round(rate(r"\b(I|I'm|I've|I'd|me|my|myself)\b"), 1),
                second_person_per_k=round(rate(r"\b(you|your|you're|yourself)\b"), 1),
                markdown_frac=round(sum(1 for t in texts if re.search(r"^#{1,4} |^\* |^- |\*\*", t, re.M)) / n, 2),
                embodied_frac=round(sum(1 for t in texts if re.match(r"\s*(I |I'm|I've|I'd|My |Honestly|Okay,? so I|Oh)", t)) / n, 2),
                introspective_frac=round(sum(1 for t in texts if re.search(r"\b(I (feel|tend|think|believe|notice|find myself|am someone)|as someone who|for me,|personally)\b", t, re.I)) / n, 2),
                uniq_ratio=round(st.mean(len(set(t.split())) / max(1, len(t.split())) for t in texts), 2),
                ends_clean=round(sum(1 for t in texts if t.strip()[-1:] in ".!?\"'") / n, 2))

out = {}
for fn, keep in (("steer_results_s2mean.json", None), ("steer_results_s2balanced.json", None), ("steer_results_fix.json", {"mean_assistant_axis"})):
    p = f"{Q}/phase10_runs/{fn}"
    if not os.path.exists(p):
        continue
    for j in json.load(open(p)):
        if keep is None or j["name"] in keep:
            out[j["name"]] = {"file": fn, "source": j["source"], "dir_norm_raw": j.get("dir_norm_raw"),
                              "per_alpha": {a: stats(j["generations"][a]) for a in j["generations"]}}
z = np.load(f"{Q}/results/gram_stage2.npz", allow_pickle=True); G = np.array(z["G"]); nm = [str(x) for x in z["names"]]
one = np.ones(134) / 134
out["controls"] = {}
for lab, spec, idx in (("signedrandom", "steer_spec_s2mean.json", 1), ("balanced", "steer_spec_s2balanced.json", 0)):
    job = json.load(open(f"{Q}/phase10_runs/{spec}"))["jobs"][idx]
    c = np.array([job["coef"][t] for t in nm])
    out["controls"][lab] = {"job": job["name"], "cos_with_grand_mean": float(c @ G @ one / np.sqrt((c @ G @ c) * (one @ G @ one))),
                            "sum_of_signs": float(np.sign(c).sum())}
json.dump(out, open(f"{Q}/analysis/s2mean_steer_stats.json", "w"), indent=1)
for k, v in out.items():
    if k == "controls": print("controls:", v); continue
    print(f"== {k} ({v['source']})")
    for a, s in v["per_alpha"].items():
        print(f"  {a:>5}: 1st {s['first_person_per_k']:5.1f} 2nd {s['second_person_per_k']:5.1f} md {s['markdown_frac']:.2f} embodied {s['embodied_frac']:.2f} words {s['words']} uniq {s['uniq_ratio']:.2f}")
