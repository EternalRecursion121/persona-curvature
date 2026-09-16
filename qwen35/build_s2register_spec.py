#!/usr/bin/env python3
"""Spec for EXPERIMENT 1: register versus residual.

Question.  Stage two puts 15 percent of every adapter's squared norm on one
shared direction that, steered alone, turns second-person advice into
first-person in-character answers (wiki behaviour/stage-two-shared-direction).
Personas amplify their own trait behaviourally more than their stage-one
adapter does (analysis/spider.json arms B and C).  Is that amplification the
shared REGISTER direction, or the trait-specific RESIDUAL of stage two?

Four conditions on the 24-prompt Big Five battery (bigfive_probes.prompts_only),
for the 15 traits that also have a second seed:

  cond_a  base                                    (trait-independent, run once)
  cond_b  base + stage-one adapter                (alpha 0 of the stage-one job)
  cond_c  base + stage-one adapter + the stage-two grand mean at a persona's dose
  cond_d  base + stage-one adapter + the same direction at 1.0 ref  (over-dose,
          to separate "too small a dose" from "wrong direction")
  cond_e  base + the exact persona, rank 128 scale 1.0

THE PERSONA'S DOSE OF THE SHARED DIRECTION.
  |mean stage-two dW|^2 / mean |stage-two dW|^2 = 0.1514038882692439
      (analysis/stage2_structure.json#shared_component.stage2
       .mean_direction_norm2_over_mean_norm2)
  so the grand mean's norm is sqrt(0.1514) = 0.38911 of a stage-two adapter's,
  and each adapter's projection on the unit grand mean is
  cos_to_mean x |dW| = 0.38930 x |dW|
      (...#shared_component.stage2.cos_to_mean_direction_mean).
  The persona is dW_dpo + 0.25 dW_sft with the two halves at equal norm --
  share of squared norm 0.4999 each (analysis/fulloct_geometry.json#norm_identity),
  the raw stage-two delta being 4.01x the stage-one delta -- so
  |0.25 dW_sft| = |dW_dpo| = 1.6157416444226869, the mean stage-one norm.

CORRECTION, made after this spec was written and run.  `ref` = 0.8078003190997738
is NOT the mean stage-one adapter norm: sketch_adapters.sketch_one computes
||B @ A||_F without the LoRA scaling lora_alpha / r = 2.0, so ref is 0.49996 of an
adapter (analysis/steer_alpha_units.json).  The persona's component along the unit
shared direction is 0.25 x 0.38930 x |dW_sft| = 0.6291299271661969 of Frobenius
norm, which is 0.3894 of a stage-one adapter but alpha 0.7788 in the ref units
steer_fix actually uses.  The run therefore used alpha 0.389 (HALF a persona's
dose) and alpha 1.0 (1.28x it), which bracket the true dose; the numbers on disk
are what ran and are left alone.  Read cond_c as the half dose and cond_d as the
over-dose, not as the labels below say.

All four conditions are produced by the SAME mechanism (an fp32 delta added into
the bf16 base weights), so they are comparable to each other; they are not
byte-comparable with the PEFT-adapter generations in phase10_runs/eval_100traits.json.
Condition cond_b is the check on that: its judged own-factor shift should track
the stage-one shift already recorded in phase10_runs/judged_100.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TRAITS = ["helpful", "cold", "harsh", "organized", "disorganized", "careful",
          "relaxed", "anxious", "fretful", "extraverted", "quiet", "assertive",
          "intellectual", "simple", "unimaginative"]

REF = 0.8078003190997738          # phase10_runs/steer_spec_s2mean.json#ref
DOSE = 0.389                      # sqrt(0.1514038882692439), see the docstring
OVERDOSE = 1.0

# alpha -> opaque condition label, for the blind judge file
COND = {("stage1", 0.0): "cond_b", ("stage1", DOSE): "cond_c",
        ("stage1", OVERDOSE): "cond_d", ("persona_exact", 0.0): "cond_e",
        ("base", 0.0): "cond_a"}


def main():
    from bigfive_probes import prompts_only
    prompts = prompts_only()
    assert len(prompts) == 24, len(prompts)

    s2 = json.load(open(f"{HERE}/phase10_runs/steer_spec_s2mean.json"))
    coef = s2["jobs"][0]["coef"]          # equal weights 1/134 over the 134 stage-two LoRAs
    assert s2["jobs"][0]["name"] == "S2_mean" and len(coef) == 134
    assert abs(s2["ref"] - REF) < 1e-12

    jobs = [{"name": "s2reg_base", "source": "stage2", "coef": {},
             "alphas": [0.0], "ref": REF, "prompts": prompts}]
    for t in TRAITS:
        jobs.append({"name": f"s2reg_s1_{t}", "source": "stage2", "coef": coef,
                     "alphas": [0.0, DOSE, OVERDOSE], "ref": REF,
                     "prompts": prompts,
                     "add_adapter": {"source": "stage1", "trait": t}})
        jobs.append({"name": f"s2reg_pex_{t}", "source": "stage2", "coef": {},
                     "alphas": [0.0], "ref": REF, "prompts": prompts,
                     "add_adapter": {"source": "persona_exact", "trait": t}})

    spec = {"ref": REF, "dose": DOSE, "overdose": OVERDOSE, "traits": TRAITS,
            "n": len(jobs), "jobs": jobs,
            "dose_arithmetic": {
                "mean_direction_norm2_over_mean_norm2": 0.1514038882692439,
                "sqrt": __import__("math").sqrt(0.1514038882692439),
                "cos_to_mean_direction_mean": 0.38929785188872645,
                "ref_is": ("0.8078003190997738; CORRECTED 2026-09-08 -- this is HALF the "
                           "mean stage-one adapter Frobenius norm (1.6157416444226869), "
                           "because sketch_adapters omits the LoRA scaling 2.0. See "
                           "analysis/steer_alpha_units.json. The persona's true dose is "
                           "alpha 0.7788; this spec's 0.389 and 1.0 bracket it."),
                "why": ("persona = dW_dpo + 0.25 dW_sft with |0.25 dW_sft| = |dW_dpo| = "
                        "1.6157416444226869 (fulloct_geometry.json#norm_identity), and each "
                        "stage-two adapter's projection on the unit grand mean is 0.38930 of "
                        "its own norm, so the persona carries 0.6291299271661969 of Frobenius "
                        "norm on that direction, which is alpha 0.7788 in ref units.")},
            "conditions": {f"{s}|{a}": c for (s, a), c in COND.items()}}
    p = f"{HERE}/phase10_runs/steer_spec_s2register.json"
    with open(p, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"wrote {p}: {len(jobs)} jobs, "
          f"{sum(len(j['alphas']) for j in jobs) * len(prompts)} generations")


if __name__ == "__main__":
    main()
