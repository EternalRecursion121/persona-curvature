#!/usr/bin/env python
"""
Pull runmeta.json for the constrained-subspace runs off the adapter volume and
print exactly the numbers the experiment turns on:

  * per-step <g,u^(i)> BEFORE and AFTER projection, for EVERY i (first/last 3)
  * the Gram condition number at each of those steps
  * the realised-update constraint (project-mode=update) before/after
  * the FINAL realised normalised component of dW along every direction
  * the PRE-projection cosine between the raw update and every direction --
    if this is tiny, the constraint was barely binding and that is itself the
    explanation for a null result.

    python drift/report_subspace.py proj_oracle proj_multi proj_svd8 proj_oracle_svd8
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "runmeta_cache")
MODAL = os.path.expanduser("~/cartovenv/bin/modal")
VOLUME = "persona-drift-adapters"


def fetch(run):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, f"{run}.json")
    if not os.path.exists(dst):
        subprocess.run([MODAL, "volume", "get", VOLUME, f"/{run}/runmeta.json", dst],
                       check=True, capture_output=True)
    with open(dst) as f:
        return json.load(f)


def vec(xs, fmt="{:+.3e}"):
    return "[" + " ".join(fmt.format(x) for x in xs) + "]"


def report(run):
    m = fetch(run)
    ss = m.get("syc_subspace")
    print("=" * 100)
    print(f"RUN {run}   source={m.get('syc_source')}   "
          f"project_mode={m.get('project_mode')}   steps={m['total_steps']}   "
          f"loss {m['first_step_loss']:.4f} -> {m['final_step_loss']:.4f}")
    print(f"  init checksum {m['lora_A_init_checksum']:.10f}   "
          f"wall {m['wall_seconds']:.1f}s   peak {m['peak_gpu_alloc_gb']:.1f}GB")
    if not ss:
        print("  (no subspace metadata -- legacy single-direction run)")
        return m
    k = ss["k"]
    print(f"  k={k}  dirs={ss['dir_labels']}  modules={ss['n_modules_matched']}")
    print(f"  ||V_i||_F = {vec(ss['V_frobenius_norms'], '{:.6f}')}")
    if k > 1:
        print("  weight-space cos(V_i,V_j):")
        for row in ss["V_cosine_matrix"]:
            print("    " + " ".join(f"{x:+.4f}" for x in row))
    ls = m["log_steps"]
    show = ls[:3] + ls[-3:]
    for r in show:
        print(f"  --- step {r['step']}  loss={r['task_loss']:.4f} "
              f"gn={r['grad_norm']:.3f} cond(G)={r['gram_cond']:.4e} "
              f"ridge={r['gram_ridge']:.2e}")
        print(f"      grad  <g,u_i> before {vec(r['mc_before'])}")
        print(f"                    after {vec(r['mc_after'])}")
        print(f"      cos(raw g, u_i)     {vec(r['mc_cos_before'], '{:+.5f}')}")
        if "upd_before" in r:
            print(f"      upd   <d,u_i> before {vec(r['upd_before'])}")
            print(f"                    after {vec(r['upd_after'])}")
            print(f"      cos(raw upd, u_i)   {vec(r['upd_cos_before'], '{:+.5f}')}")
        print(f"      <dW,V_i>/||V_i||    {vec(r['syc_components_norm'], '{:+.7f}')}")
    print("  FINAL realised normalised component <dW,V_i>/||V_i||_F:")
    for i, lab in enumerate(ss["dir_labels"]):
        print(f"    dir {i}: {lab:<26} {m['final_syc_components_norm'][i]:+.8f}   "
              f"(raw {m['final_syc_components'][i]:+.6e})")
    print("  PRE-PROJECTION cosine between the raw update and each direction:")
    for i, lab in enumerate(ss["dir_labels"]):
        g = [abs(r["mc_cos_before"][i]) for r in ls]
        u = [abs(r["upd_cos_before"][i]) for r in ls if "upd_cos_before" in r]
        print(f"    dir {i}: {lab:<26} grad: first={g[0]:.4f} max={max(g):.4f} "
              f"last={g[-1]:.4f} | update: first={u[0]:.4f} max={max(u):.4f} "
              f"last={u[-1]:.4f}")
    return m


if __name__ == "__main__":
    runs = sys.argv[1:] or ["proj_oracle", "proj_multi", "proj_svd8",
                            "proj_oracle_svd8"]
    metas = [report(r) for r in runs]
    print("=" * 100)
    cks = {round(m["lora_A_init_checksum"], 6) for m in metas}
    print(f"init checksums across {len(metas)} runs: {cks}  identical={len(cks) == 1}")
    print(f"total wall (sum of container seconds): "
          f"{sum(m['wall_seconds'] for m in metas):.1f}s")
