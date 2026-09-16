#!/usr/bin/env python3
"""Steering spec for the FACTOR-chart alien direction and its two controls.

The factor-chart twin of build_alien_spec.py.  Same design -- same alphas, same
reference norm, same 24-prompt battery, same two controls -- so the curves land
on the same axis as every published direction and as the PC-chart alien run.

alien_fa        the deepest hole in the span of the five oblimin factor
                directions (fa_chart.FAChart, Gram-Schmidt basis in the order
                Warmth, Competence, FearfulWithdrawal, Arousal, Imagination).
                analysis/alien_fa.json#alien_fa.gap_deg.
alien_fa_shuffle the same coefficient multiset permuted across traits.  Identical
                mixing statistics -- same number of adapters, same coefficient
                magnitudes, same sum-to-zero contrast structure -- pointing
                nowhere in particular.  Control for "unusual mixtures degrade the
                model regardless of where they point".
span_random_fa  a uniformly random unit direction inside the same five-factor
                span.  Lands, as random directions do, much closer to a named
                trait; the control for "anything in this subspace behaves".

steer_fix.py renormalises the merged delta to unit Frobenius over the whole model
and then scales by alpha * ref, so the coefficient scale here does not set the
dose; alpha does, in the same units as every other direction.
"""
import json
import os

import numpy as np

from fa_chart import FAChart

Q = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{Q}/phase10_runs/steer_spec.json"))
AL = json.load(open(f"{Q}/analysis/alien_fa.json"))
REF, PROMPTS = S["ref"], S["jobs"][0]["prompts"]
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]      # |alpha|=4 is wreckage in every direction

ch = FAChart()
names = AL["alien_fa"]["traits"]
assert names == ch.names, "adapter order differs from the chart"
c = np.array(AL["alien_fa"]["coeffs"])
rng = np.random.default_rng(7)

# --- control 1: same coefficients, permuted over traits ---------------------
perm = rng.permutation(len(c))
c_shuf = c[perm]

# --- control 2: a random unit direction in the same five-factor span --------
u_r = rng.normal(size=5)
u_r /= np.linalg.norm(u_r)
c_rand = ch.direction_from_chart(u_r)

# how close does each direction sit to the nearest trait, in the same chart?
T = ch.trait_coords
A = T / np.linalg.norm(T, axis=1, keepdims=True)


def gap(coeffs):
    x = ch.coords(coeffs)
    x = x / np.linalg.norm(x)
    d = np.abs(A @ x)
    i = int(np.argmax(d))
    return float(np.degrees(np.arccos(min(d[i], 1.0)))), names[i]


jobs = []
for nm, coeffs in (("alien_fa", c), ("alien_fa_shuffle", c_shuf), ("span_random_fa", c_rand)):
    g, near = gap(coeffs)
    x = ch.coords(coeffs / ch.norm(coeffs))
    print(f"{nm:16s} nearest trait line {g:5.1f}d ({near})   "
          f"sum(c)={coeffs.sum():+.2e}  L1={np.abs(coeffs).sum():.3f}  "
          f"|chart|={np.linalg.norm(x):.4f}")
    jobs.append({"name": nm, "source": "stage1", "alphas": ALPHAS, "ref": REF,
                 "prompts": PROMPTS, "coef": {t: float(v) for t, v in zip(names, coeffs)},
                 "gap_deg": g, "nearest": near, "chart_coords": x.tolist()})

out = {"ref": REF, "n": len(names), "jobs": jobs}
json.dump(out, open(f"{Q}/phase10_runs/alien_spec_fa.json", "w"), indent=1)
print(f"\nwrote phase10_runs/alien_spec_fa.json  "
      f"({len(jobs)} directions x {len(ALPHAS)} alphas x {len(PROMPTS)} prompts)")
