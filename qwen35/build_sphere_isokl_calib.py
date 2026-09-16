#!/usr/bin/env python3
"""Spec for the iso-KL sphere's dose calibration.

The 2026-09-01 principal-component sphere steered all 72 Fibonacci-lattice
directions at one alpha, 1.5.  [[fisher-norms]] then showed that the Fisher norm
of a direction varies about threefold across a sphere lattice and predicts how
far the judged persona moves (Spearman +0.37 on the factor sphere), so a sphere
at common alpha is a sphere at unequal dose and part of its smoothness could be
dose rather than position.

This builds the input for sphere_dose.py: the same 72 directions, measured for
KL(base || steered) per token on the SAME fixed 4,378 token positions
fisher_dose.py used, at six alphas that bracket the alpha the original ran at.

Three things are taken verbatim and deliberately:
  * the coefficient vectors from phase10_runs/sphere_spec.json (the PC sphere,
    verified to reproduce build_sphere_spec.py's W @ u to cosine 1.0),
  * ref = 0.8102592902648793 from phase10_runs/steer_spec.json, the unit the PC
    sphere's alpha 1.5 was expressed in -- NOT dose_spec.json's 0.8078003190997738,
  * the 24 prompts, their alpha-0 texts, the 192-token cap and the 134 trait
    order from phase10_runs/dose_spec.json, so the scored token positions are
    identical to the matched-dose calibration's.

modes = ["bf16"] only.  sphere_sweep.py writes (W0 + D*s).to(bfloat16), which is
exactly fisher_dose.py's "bf16" construction, so the dose the generating model
actually receives is the bf16 curve; measuring the exact-weight curve as well
would double the bill for a number this experiment does not use.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
ALPHAS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

D = json.load(open(f"{Q}/phase10_runs/dose_spec.json"))
SP = json.load(open(f"{Q}/phase10_runs/sphere_spec.json"))
REF = json.load(open(f"{Q}/phase10_runs/steer_spec.json"))["ref"]
assert REF == SP["ref"] == 1.5 * 0 + 0.8102592902648793, REF

dirs = [{"name": j["name"], "family": "pc_sphere", "coef": j["coef"],
         "published_ref": REF, "alphas": ALPHAS} for j in SP["jobs"]]
assert len(dirs) == 72

spec = {"ref": REF,
        "max_resp_tokens": D["max_resp_tokens"],
        "traits134": D["traits134"],
        "prompts": D["prompts"],
        "texts": D["texts"],
        "text_source": D["text_source"],
        "modes": ["bf16"],
        "tag": "sphere_isokl",
        "directions": dirs}
out = f"{Q}/phase10_runs/sphere_isokl_dose_spec.json"
json.dump(spec, open(out, "w"))
print(f"{len(dirs)} directions x {len(ALPHAS)} alphas x {len(spec['modes'])} mode "
      f"= {len(dirs)*len(ALPHAS)*len(spec['modes'])} passes")
print(f"ref {REF}  max_resp_tokens {D['max_resp_tokens']}  "
      f"{len(D['prompts'])} prompts  text_source {D['text_source']}")
print(f"wrote {out}")
