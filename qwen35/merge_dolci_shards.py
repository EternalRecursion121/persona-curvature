#!/usr/bin/env python3
"""Merge the five Dolci scoring shards into one analysis/dolci_scores.json.

The run was split three ways over the DPO items and two ways over the SFT items
so that no single Modal call had to survive four hours, and so that a failure
cost one shard rather than the run.  Every shard scored the SAME 63 directions
from the same spec, so the merge is a concatenation once the direction names and
the per-target norms are checked to be identical across shards.
"""
import json
import os
import sys

Q = os.path.dirname(os.path.abspath(__file__))
SHARDS = ["dpo0", "dpo1", "dpo2", "sft0a", "sft0b", "sft1"]


def main():
    out, names, bu = None, None, None
    seen, worst = set(), {}
    for s in SHARDS:
        p = f"{Q}/analysis/dolci_scores_raw_{s}.json"
        if not os.path.exists(p):
            print(f"MISSING {p}")
            return 1
        d = json.load(open(p))
        if names is None:
            names, bu = d["names"], d["bu_norm"]
            out = {k: v for k, v in d.items() if k != "scores"}
            out["scores"] = []
            out["shards"] = {}
        else:
            assert d["names"] == names, s
            # The per-target Frobenius norms are rebuilt on each shard's own GPU
            # from the same 156 adapter files, and float32 reductions on a GPU
            # are not bit-deterministic across containers.  The tolerance is a
            # tolerance, not an equality, and the largest deviation seen is
            # reported so it can be read rather than assumed.
            rel = max(abs(x - y) / max(abs(y), 1e-30) for x, y in zip(d["bu_norm"], bu))
            worst[s] = rel
            assert rel < 1e-6, f"target norms differ in {s} by {rel:.3e}"
        n0 = len(d["scores"])
        for r in d["scores"]:
            if r["id"] in seen:
                continue          # the N x N anchors ride in the DPO items file
            seen.add(r["id"])
            out["scores"].append(r)
        out["shards"][s] = {"n_rows": n0, "elapsed_s": d.get("elapsed_s"),
                            "n_groups": d.get("n_groups"),
                            "n_oom_retries": d.get("n_oom_retries"),
                            "tok_budget": d.get("tok_budget"),
                            "peak_gb_last_window": d.get("peak_gb_last_window")}
        print(f"{s}: {n0} rows, {d.get('elapsed_s', 0) / 60:.1f} min, "
              f"{d.get('n_oom_retries')} oom retries")
    out["merged_from"] = SHARDS
    out["bu_norm_max_rel_dev_across_shards"] = max(worst.values())
    out["bu_norm_rel_dev_by_shard"] = worst
    out["elapsed_s"] = sum(v["elapsed_s"] or 0 for v in out["shards"].values())
    p = f"{Q}/analysis/dolci_scores.json"
    json.dump(out, open(p, "w"))
    print(f"wrote {p}: {len(out['scores'])} rows, {len(names)} directions, "
          f"{out['elapsed_s'] / 3600:.2f} GPU-hours of scoring")


if __name__ == "__main__":
    sys.exit(main())
