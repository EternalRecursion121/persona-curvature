#!/usr/bin/env python3
"""Turn persona_sliders.py --stage behave output into judge_personas.py's input.

judge_personas.py reads a list of {trait, prompts, generations:{condition:[...]}}
and judges every (prompt, response) blind and interleaved, so putting every
condition of this run -- base, the 13 sliders, the 10 matching stage-one
adapters -- under ONE record means the judge sees them shuffled together and
cannot tell them apart.  That is the point.
"""
import json
import os
import sys

Q = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else f"{Q}/phase10_runs/sliders_behave.json"
dst = sys.argv[2] if len(sys.argv) > 2 else f"{Q}/phase10_runs/sliders_as_eval.json"
R = json.load(open(src))
rec = [{"trait": "sliders", "prompts": R["prompts"], "generations": R["generations"]}]
json.dump(rec, open(dst, "w"))
n = sum(len(v) for v in R["generations"].values())
print(f"wrote {dst}: {len(R['generations'])} conditions x {len(R['prompts'])} "
      f"prompts = {n} generations")
