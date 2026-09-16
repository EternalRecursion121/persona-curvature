#!/usr/bin/env python3
"""Repair the exact persona adapters' tensor keys.

fix_persona_merge.py (and build_personas_seed1.py, which copied it) wrote keys as
"base_model.model." + <stage-1 key>, but the stage-1 keys already carry that
prefix, so /oct/personas_exact/*/adapter_model.safetensors and
/oct/seed1/personas_exact/* have keys like
    base_model.model.base_model.model.model.layers.0....lora_A.weight
PEFT expects a single prefix.  This rewrites each file in place with one prefix,
records the repair in MERGE_NOTE.json, and leaves the Grams unaffected (the
geometry never depended on key names).  CPU only; app name outside pc-qwen35-phase10-*.
"""
import json
import os

import modal

PFX = "base_model.model."
app = modal.App("pc-qwen35-phase11-fixkeys")
oct_vol = modal.Volume.from_name("pc-qwen35-oct2")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy<3", "safetensors")


@app.function(image=image, volumes={"/oct": oct_vol}, cpu=4.0, memory=16384,
              timeout=60 * 45, max_containers=20)
def fix_one(rel: str) -> dict:
    from safetensors import safe_open
    from safetensors.numpy import save_file
    d = f"/oct/{rel}"
    p = f"{d}/adapter_model.safetensors"
    if not os.path.exists(p):
        return {"path": rel, "error": "missing"}
    f = safe_open(p, framework="np")
    keys = list(f.keys())
    bad = [k for k in keys if k.startswith(PFX + PFX)]
    if not bad:
        return {"path": rel, "changed": 0, "n_keys": len(keys), "first": sorted(keys)[0]}
    out = {}
    for k in keys:
        nk = k
        while nk.startswith(PFX + PFX):
            nk = nk[len(PFX):]
        out[nk] = f.get_tensor(k)
    assert len(out) == len(keys)
    save_file(out, p)
    note_p = f"{d}/MERGE_NOTE.json"
    note = json.load(open(note_p)) if os.path.exists(note_p) else {}
    note["key_repair_2026-09-07"] = ("tensor keys carried the base_model.model. prefix twice "
                                    "(fix_persona_merge.py prepended it to keys that already had it); "
                                    "rewritten with a single prefix so PEFT can map them")
    json.dump(note, open(note_p, "w"), indent=1)
    oct_vol.commit()
    return {"path": rel, "changed": len(bad), "n_keys": len(keys), "first": sorted(out)[0]}


@app.local_entrypoint()
def main():
    rels = list_dirs.remote()
    res = list(fix_one.map(rels))
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(res, open(f"{here}/analysis/persona_key_repair.json", "w"), indent=1)
    ch = [r for r in res if r.get("changed")]
    err = [r for r in res if "error" in r]
    print(f"{len(res)} adapters, {len(ch)} rewritten, {len(err)} errors")
    for r in res[:3]:
        print(r)


@app.function(image=image, volumes={"/oct": oct_vol}, cpu=1.0, timeout=600)
def list_dirs() -> list:
    rels = []
    for base in ("personas_exact", "seed1/personas_exact"):
        root = f"/oct/{base}"
        if os.path.isdir(root):
            rels += [f"{base}/{t}" for t in sorted(os.listdir(root))
                     if os.path.exists(f"{root}/{t}/adapter_model.safetensors")]
    return rels
