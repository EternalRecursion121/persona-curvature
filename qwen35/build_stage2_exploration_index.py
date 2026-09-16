#!/usr/bin/env python3
"""analysis/stage2_exploration.json: one index over the stage-two exploration of
2026-09-08, for companion/build_companion.py (which renders it as a flat
key-value table, so this file holds headline scalars and short strings only --
the detail stays in the four result files it names).
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/analysis"


def get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load(name):
    p = f"{A}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = {"date": "2026-09-08",
           "what": ("Four experiments on OCT stage two: register versus residual, "
                    "a trait-free control, activation space, and the number of "
                    "factors. Detail in the four files named below.")}

    fr = load("stage2_frame.json")
    if fr:
        s = fr["shared_component"]
        out["frame"] = {
            "file": "analysis/stage2_frame.json",
            "lora_a_shared_within_a_seed_cos": get(load("lora_a_identity.json") or {},
                                                   "sets.stage2_seed0.A_pairwise", [{}])[0].get("cos"),
            "lora_a_across_seeds_cos": (load("lora_a_identity.json") or {}).get("cross_set_A", [{}])[0].get("cos"),
            "shared_share_seed0_same15": s["stage2_seed0_same15"]["mean_direction_norm2_over_mean_norm2"],
            "shared_share_seed1_15": s["stage2_seed1_15"]["mean_direction_norm2_over_mean_norm2"],
            "cos_to_mean_seed0_same15": s["stage2_seed0_same15"]["cos_to_mean_direction_mean"],
            "cos_to_mean_seed1_15": s["stage2_seed1_15"]["cos_to_mean_direction_mean"],
            "reading": "the shared direction reproduces in an independent LoRA-A frame"}

    r1 = load("stage2_register_vs_residual.json")
    if r1:
        e = {"file": "analysis/stage2_register_vs_residual.json"}
        for c, v in (r1.get("summary_own_factor_amplification") or {}).items():
            e[f"amplification_{c}"] = v["mean_amplification"]
        g = r1.get("register_vs_persona_gap") or {}
        for c, v in g.items():
            if c != "_endpoints" and isinstance(v, dict):
                e[f"share_of_gap_{c}"] = v.get("share_of_the_stage1_to_persona_gap")
        for c, v in (r1.get("text_statistics") or {}).items():
            e[f"first_person_per_k_{c}"] = v.get("first_person_per_k")
            e[f"markdown_frac_{c}"] = v.get("markdown_frac")
        out["register_vs_residual"] = e

    r2 = load("stage2_neutral_control.json")
    if r2:
        out["neutral_control"] = {
            "file": "analysis/stage2_neutral_control.json",
            "n_neutral": r2.get("n_neutral"),
            "neutral_cos_with_grand_mean_mean": get(r2, "neutral_vs_grand_mean.mean"),
            "neutral_cos_with_grand_mean_sd": get(r2, "neutral_vs_grand_mean.sd"),
            "zoo_cos_with_grand_mean_mean": get(r2, "reference.zoo_stage2_cos_to_grand_mean_mean"),
            "neutral_x_neutral_mean": get(r2, "neutral_x_neutral.mean"),
            "neutral_x_zoo_mean": get(r2, "neutral_x_zoo.mean"),
            "zoo_offdiag_mean": get(r2, "reference.zoo_stage2_offdiag_cos_mean"),
            "neutral_norm_mean": get(r2, "neutral_norms.mean"),
            "zoo_norm_mean": get(r2, "reference.zoo_stage2_norm_mean")}

    r3 = load("actspace_stage2_geometry.json")
    if r3:
        e = {"file": "analysis/actspace_stage2_geometry.json",
             "primary_layer": r3.get("primary_layer")}
        w = get(r3, "windows_out.resp") or {}
        for arm, v in (w.get("per_arm") or {}).items():
            e[f"act_cos_to_mean_{arm}"] = get(v, "shared_direction.cos_to_mean_direction_mean")
            e[f"act_shared_share_{arm}"] = get(v, "shared_direction.mean_direction_norm2_over_mean_norm2")
            e[f"act_cos_own_prompt_{arm}"] = get(v, "vs_prompt_vector.cos_own_mean")
            e[f"act_rank1_{arm}"] = get(v, "vs_prompt_vector.rank1")
        for arm, v in (w.get("gram_correlations") or {}).items():
            e[f"act_vs_weight_stage1_{arm}"] = v.get("activation_centred_vs_weight_stage1_centred")
            e[f"act_vs_weight_stage2_{arm}"] = v.get("activation_centred_vs_weight_stage2_centred")
        for k, v in (w.get("cross_arm") or {}).items():
            e[f"act_cos_mean_shifts_{k}"] = v.get("cos_between_mean_shifts")
            e[f"act_same_trait_{k}"] = v.get("same_trait_cos_mean")
        out["activation_space"] = e

    r4 = load("stage2_factors_choice.json")
    if r4:
        e = {"file": "analysis/stage2_factors_choice.json",
             "stage2_chosen_k": get(r4, "parallel_analysis.stage2_chosen"),
             "stage2_noshared_chosen_k": get(r4, "parallel_analysis.stage2_noshared_chosen"),
             "stage1_chosen_k": get(r4, "parallel_analysis.stage1_chosen"),
             "gram_trace_share_removed": get(r4, "gram.trace_share_removed"),
             "gram_offdiag_cos_before": get(r4, "gram.offdiag_cos_before.mean"),
             "gram_offdiag_cos_after": get(r4, "gram.offdiag_cos_after.mean"),
             "gram_corr_before_after": get(r4, "gram.corr_before_after")}
        for k, v in (r4.get("decision_inputs") or {}).items():
            e[f"targets_taken_{k}"] = v["n_distinct_targets_taken"]
            e[f"mean_abs_congruence_{k}"] = v["mean_abs_congruence_of_non_Eval_factors"]
        e["decision"] = r4.get("decision", "")
        out["factors"] = e

    p = f"{A}/stage2_exploration.json"
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
