#!/usr/bin/env python3
"""Validate all 100 Big Five adapters from their runmeta, not their done-keys.

A done-key says a stage finished, not that it finished WELL. This recomputes the
honest final loss from log_history (runmeta's `loss_last` under-reports any
resumed run), and checks LoRA config uniformity -- if the hyperparameters or the
targeted-module count vary across adapters, the weight-space geometry is
comparing things that are not comparable.
"""
import json, os, subprocess, statistics as st, tempfile, shutil, sys
Q = "/home/vibe12/projects/persona-curvature/qwen35"
M = "/home/vibe12/cartovenv/bin/modal"
P = json.load(open(f"{Q}/traits_primary.json"))
traits = [r["trait"].lower().replace(" ", "_").replace("-", "_") for r in P]
d = tempfile.mkdtemp()
rows, missing = [], []
for i, t in enumerate(traits, 1):
    p = os.path.join(d, f"{t}.json")
    subprocess.run([M, "volume", "get", "pc-qwen35-oct2",
                    f"/loras_introspection/{t}/runmeta.json", p],
                   capture_output=True, timeout=180)
    if not os.path.exists(p):
        missing.append(t); continue
    m = json.load(open(p))
    h = [e["loss"] for e in m.get("log_history", []) if "loss" in e]
    if not h:
        missing.append(t + "(no log_history)"); continue
    rows.append(dict(trait=t, loss_true=sum(h[-20:]) / len(h[-20:]),
                     reported=m.get("loss_last"), steps=m.get("optimizer_steps"),
                     rows_in=m.get("n_rows_in"), dropped=m.get("n_dropped_at_max_len"),
                     targeted=m.get("n_targeted"),
                     hp=json.dumps(m.get("hp"), sort_keys=True),
                     lora=json.dumps(m.get("lora"), sort_keys=True)))
shutil.rmtree(d, ignore_errors=True)
lt = [r["loss_true"] for r in rows]
print(f"validated {len(rows)}/100  missing: {missing}")
print(f"final loss (measured): mean {st.mean(lt):.3f}  sd {st.stdev(lt):.3f}  "
      f"min {min(lt):.3f}  max {max(lt):.3f}")
print(f"distinct hp configs   : {len({r['hp'] for r in rows})}")
print(f"distinct lora configs : {len({r['lora'] for r in rows})}")
print(f"targeted module counts: {sorted({r['targeted'] for r in rows})}")
dr = [r["dropped"] for r in rows if r["dropped"] is not None]
if dr:
    print(f"rows dropped at max_len: median {int(st.median(dr))}  max {max(dr)}  "
          f">5% of 12000: {sum(1 for x in dr if x > 600)}")
out = [r for r in sorted(rows, key=lambda r: -r["loss_true"])[:5]]
print("highest final loss:", [(r["trait"], round(r["loss_true"], 3)) for r in out])
json.dump(rows, open(f"{Q}/analysis/validate_100.json", "w"), indent=1)
