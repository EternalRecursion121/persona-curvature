#!/usr/bin/env python3
"""What is one unit of alpha, in the project's steering runs?

steer_fix.py's docstring says

    W_steered = W_base + alpha * ref * normalise(sum_i c_i * 2 * B_i @ A_i)
    `ref` is the mean single-adapter Frobenius norm, so alpha is measured in
    units of "one trait adapter's worth of weight change"

and every steer spec carries ref = 0.8078003190997738.  That number was produced
by sketch_adapters.py, whose `sketch_one` computes the per-module Frobenius norm
as ||B @ A||_F exactly -- WITHOUT the LoRA scaling alpha/r = 2.0 that the actual
delta carries.  The adapters' real mean Frobenius norm is the diagonal of
results/gram_sweep.npz, which does carry the scale.

So alpha = 1 is half a stage-one adapter's worth of weight change, not one.
This does not invalidate any steering result -- alpha is a consistent unit
across every run -- but it halves the reading of every dose in the project, and
in particular the "a released persona carries roughly alpha 0.4 of the shared
direction" line on wiki/pages/behaviour/stage-two-shared-direction.md.

Output: analysis/steer_alpha_units.json
"""
import glob
import json
import os

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))


def main():
    spec = json.load(open(f"{Q}/phase10_runs/steer_spec_s2mean.json"))
    ref = float(spec["ref"])

    tot = []
    for p in sorted(glob.glob(f"{Q}/analysis/sketches/stage1_k32/*.npz")):
        z = np.load(p, allow_pickle=True)
        nm = np.array(z["norms"], dtype=float)
        tot.append(float(np.sqrt((nm ** 2).sum())))
    sketch_mean = float(np.mean(tot))

    z1 = np.load(f"{Q}/results/gram_sweep.npz", allow_pickle=True)
    n1 = np.array(z1["norms"], dtype=float)
    scale1 = float(z1["scale"])
    z2 = np.load(f"{Q}/results/gram_stage2.npz", allow_pickle=True)
    n2 = np.array(z2["norms"], dtype=float)

    st2 = json.load(open(f"{Q}/analysis/stage2_structure.json"))
    cos_mu = st2["shared_component"]["stage2"]["cos_to_mean_direction_mean"]

    # the persona's component along the UNIT stage-two grand mean
    dose_abs = 0.25 * cos_mu * float(n2.mean())
    out = {
        "ref_in_every_steer_spec": ref,
        "ref_recomputed_from_sketches": sketch_mean,
        "sketch_note": ("sketch_adapters.sketch_one returns ||B@A||_F per module and "
                        "never multiplies by lora_alpha/r; the sketches are the only "
                        "place ref could have come from and they reproduce it to 1e-4."),
        "true_mean_stage1_delta_norm": float(n1.mean()),
        "true_mean_stage1_delta_norm_sd": float(n1.std()),
        "lora_scale_alpha_over_r": scale1,
        "ref_over_true_mean_norm": ref / float(n1.mean()),
        "reading": ("alpha = 1 in every steering run of this project is "
                    f"{ref / float(n1.mean()):.4f} of a stage-one adapter's Frobenius "
                    "norm, not 1. Read every published alpha as half an adapter."),
        "mean_stage2_delta_norm": float(n2.mean()),
        "persona_dose_of_the_shared_direction": {
            "formula": ("0.25 * cos_to_mean * |dW_stage2|, because the persona is "
                        "dW_dpo + 0.25 dW_sft and each stage-two adapter's projection "
                        "on the unit grand mean is cos_to_mean times its own norm"),
            "cos_to_mean_direction_mean": cos_mu,
            "absolute_frobenius": dose_abs,
            "in_ref_units_alpha": dose_abs / ref,
            "in_stage1_adapter_units": dose_abs / float(n1.mean()),
            "previously_stated_alpha": 0.4,
            "note": ("wiki/pages/behaviour/stage-two-shared-direction.md says a released "
                     "persona carries roughly alpha 0.4 of the shared direction; on the "
                     "same definition of alpha it is "
                     f"{dose_abs / ref:.3f}. The 0.4 is the dose expressed as a fraction "
                     "of a real adapter norm, which is the right physical statement but "
                     "the wrong number to type into a steer spec.")},
    }
    p = f"{Q}/analysis/steer_alpha_units.json"
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
