#!/usr/bin/env python
"""
Download the Modal Volume "gradprobe-out" into local gradprobe/out/.

Layout mirrors the volume:
    out/<mode>/sketches_pooled.npy      (N, 8192)      float32
    out/<mode>/sketches_per_layer.npy   (N, 36, 1024)  float32
    out/<mode>/sketches_per_type.npy    (N, 7, 1024)   float32
    out/<mode>/norms.json               exact, unsketched gradient norms
    out/<mode>/meta.json                doc order, seeds, projection, checks

After downloading it re-runs the cheap invariants LOCALLY, so a corrupted or
half-written fetch is caught here rather than in the analysis: shapes match
meta, the doc_id order in norms.json matches meta.json, no NaN/Inf, no
all-zero sketch rows, and the two modes (if both present) agree on the
document order and the projection spec.

Usage:
    ~/cartovenv/bin/python gradprobe/fetch_gradprobe.py
    ~/cartovenv/bin/python gradprobe/fetch_gradprobe.py --list
    ~/cartovenv/bin/python gradprobe/fetch_gradprobe.py --force
"""

import argparse
import json
import os
import sys

import modal
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gradprobe import OUT_VOLUME  # noqa: E402


def walk(vol, path="/"):
    for e in vol.listdir(path):
        is_dir = getattr(e.type, "name", str(e.type)).upper().endswith("DIRECTORY")
        if is_dir:
            yield from walk(vol, e.path)
        else:
            yield e.path, e.size


def check_mode(d: str) -> list:
    """Local integrity checks on one downloaded mode directory."""
    problems = []
    meta = json.load(open(os.path.join(d, "meta.json")))
    norms = json.load(open(os.path.join(d, "norms.json")))
    N = meta["n_docs"]
    want = {
        "sketches_pooled.npy": (N, meta["projection"]["pooled_dim"]),
        "sketches_per_layer.npy": (N, meta["n_layers"],
                                   meta["projection"]["layer_dim"]),
        "sketches_per_type.npy": (N, len(meta["type_order"]),
                                  meta["projection"]["type_dim"]),
    }
    for fn, shape in want.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            problems.append(f"{fn}: MISSING")
            continue
        a = np.load(p)
        if a.shape != shape:
            problems.append(f"{fn}: shape {a.shape} != {shape}")
        if a.dtype != np.float32:
            problems.append(f"{fn}: dtype {a.dtype} != float32")
        if not np.isfinite(a).all():
            problems.append(f"{fn}: contains NaN/Inf")
        flat = a.reshape(a.shape[0], -1)
        dead = int((np.abs(flat).max(axis=1) == 0).sum())
        if dead:
            problems.append(f"{fn}: {dead} all-zero row(s)")
        print(f"    {fn:<26}{str(a.shape):<20}"
              f"|.|_F med={np.median(np.linalg.norm(flat, axis=1)):.4e}")
    if norms["doc_id"] != meta["doc_id_order"]:
        problems.append("norms.json doc_id order != meta.json doc_id_order")
    if len(norms["total"]) != N:
        problems.append(f"norms.json has {len(norms['total'])} rows, want {N}")
    v = meta["verification"]
    for key in ("projection_identical_across_steps",
                "attribution_step_i_is_doc_i"):
        if not v.get(key):
            problems.append(f"verification.{key} is not true")
    if meta["mode"] == "frozen" and not v.get("params_bitwise_unchanged"):
        problems.append("frozen mode: parameters were NOT bitwise unchanged")
    if meta["mode"] == "sequential" and v.get("params_bitwise_unchanged"):
        problems.append("sequential mode: parameters did NOT change")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-download files whose local size already matches")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.abspath(args.dest or os.path.join(here, "out"))

    vol = modal.Volume.from_name(OUT_VOLUME, create_if_missing=True)
    entries = sorted(walk(vol))
    if not entries:
        print(f"volume {OUT_VOLUME} is empty -- run train_gradprobe.py first")
        return 1
    if args.list:
        for p, s in entries:
            print(f"{s:>12}  {p}")
        print(f"\n{len(entries)} file(s) on {OUT_VOLUME}")
        return 0

    os.makedirs(dest, exist_ok=True)
    got = skipped = 0
    for remote, size in entries:
        local = os.path.join(dest, remote.lstrip("/"))
        if (not args.force and os.path.exists(local)
                and os.path.getsize(local) == size):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(remote):
                f.write(chunk)
        got += 1
        print(f"  {remote} -> {local} ({os.path.getsize(local)} bytes)")
    print(f"\ndownloaded {got} file(s), {skipped} already current, into {dest}")

    modes = sorted(m for m in os.listdir(dest)
                   if os.path.isdir(os.path.join(dest, m))
                   and os.path.exists(os.path.join(dest, m, "meta.json")))
    all_problems, metas = {}, {}
    for m in modes:
        print(f"\n  [{m}]")
        p = check_mode(os.path.join(dest, m))
        metas[m] = json.load(open(os.path.join(dest, m, "meta.json")))
        v = metas[m]["verification"]
        gs = v["grad_norm_stats"]
        print(f"    N={metas[m]['n_docs']} D={metas[m]['grad_dim_D']} "
              f"|g| med={gs['median']:.4e} zeros={v['n_grad_norm_exactly_zero']}")
        f = v.get("sketch_fidelity_on_real_gradients")
        if f:
            for g in ("pooled", "per_layer", "per_type"):
                r = f[g]
                print(f"    fidelity {g:<10} r={r['pearson_r']:+.6f} "
                      f"rms={r['rms_err']:.5f} max={r['max_abs_err']:.5f}")
        if p:
            all_problems[m] = p

    if len(metas) > 1:
        ms = sorted(metas)
        a, b = metas[ms[0]], metas[ms[1]]
        if a["doc_id_order"] != b["doc_id_order"]:
            all_problems.setdefault("cross", []).append(
                "the two modes used different document orders")
        if a["projection"] != b["projection"]:
            all_problems.setdefault("cross", []).append(
                "the two modes used different projection specs")
        if abs(a["lora_A_init_checksum"] - b["lora_A_init_checksum"]) > 1e-6:
            all_problems.setdefault("cross", []).append(
                "the two modes started from different lora_A inits")
        if "cross" not in all_problems:
            print(f"\n  cross-mode: same doc order, same projection spec, same "
                  f"init checksum ({a['lora_A_init_checksum']:.10f})")

    if all_problems:
        print("\n*** PROBLEMS ***")
        for m, ps in all_problems.items():
            for x in ps:
                print(f"  [{m}] {x}")
        return 2
    print("\nall local integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
