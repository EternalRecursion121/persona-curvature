#!/usr/bin/env python3
"""Flatten the EM arms' checkpoints into the one-level layout cross_gram_full wants.

`cross_gram_full_on_modal.py` lists adapters one directory deep under its
`--subdir`, and the Trainer writes `em_medical/<arm>/{checkpoint-N,final}/`, two
deep.  Rather than teach the cross-Gram script about nesting -- four sibling
analyses read it -- this copies each checkpoint's two files to
`em_flat/<arm>_<tag>/` on the same volume.  CPU only, seconds.

usage:
    PC_APP_NAME=pc-qwen35-phase13-emflat modal run em_flatten.py
"""
import json
import os

import modal

app = modal.App(os.environ.get("PC_APP_NAME") or "pc-qwen35-phase13-emflat")
vol = modal.Volume.from_name("pc-qwen35-adapters")
image = modal.Image.debian_slim(python_version="3.12")
ARMS = ["em_bad", "em_good", "em_dolci"]


@app.function(image=image, volumes={"/align": vol}, timeout=60 * 20,
              cpu=2.0, memory=8192)
def flatten() -> dict:
    import shutil
    src, dst = "/align/em_medical", "/align/em_flat"
    os.makedirs(dst, exist_ok=True)
    rep = {}
    for arm in ARMS:
        root = f"{src}/{arm}"
        cks = sorted((d for d in os.listdir(root) if d.startswith("checkpoint-")),
                     key=lambda d: int(d.split("-")[1])) + ["final"]
        rep[arm] = {}
        for c in cks:
            if not os.path.exists(f"{root}/{c}/adapter_model.safetensors"):
                continue
            tag = c.replace("checkpoint-", "c")
            out = f"{dst}/{arm}_{tag}"
            os.makedirs(out, exist_ok=True)
            for f in ("adapter_model.safetensors", "adapter_config.json"):
                shutil.copyfile(f"{root}/{c}/{f}", f"{out}/{f}")
            with open(f"{out}/adapter_config.json") as f:
                cfg = json.load(f)
            rep[arm][tag] = {"path": out, "r": cfg["r"],
                             "lora_alpha": cfg["lora_alpha"],
                             "use_rslora": cfg.get("use_rslora"),
                             "bytes": os.path.getsize(f"{out}/adapter_model.safetensors")}
    # The training metrics come back from the volume rather than from the
    # launching client.  The three arms were trained by three separate
    # `modal run --detach` clients (the first was detached from its systemd unit
    # so the other two could run beside it instead of behind it), so no single
    # client holds all three returns; trainlog.json does.
    logs = {}
    for arm in ARMS:
        p = f"{src}/{arm}/trainlog.json"
        if os.path.exists(p):
            d = json.load(open(p))
            L = [h["loss"] for h in d["log_history"] if "loss" in h]
            logs[arm] = {k: d[k] for k in ("arm", "n", "sequence_tokens", "loss_tokens")
                         if k in d}
            logs[arm].update({
                "steps": len(L),
                "loss_first": round(float(L[0]), 4) if L else None,
                "loss_last": round(sum(float(x) for x in L[-5:]) / len(L[-5:]), 4)
                             if L else None,
                "checkpoints": sorted(rep[arm]),
                "log_history_tail": d["log_history"][-3:]})
    rep["_trainlogs"] = logs
    vol.commit()
    rep["_listing"] = sorted(os.listdir(dst))
    return rep


@app.local_entrypoint()
def main():
    r = flatten.remote()
    print(json.dumps(r, indent=1))
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "phase10_runs", "em_flat_report.json")
    json.dump(r, open(p, "w"), indent=1)
    print("wrote", p)
    t = os.path.join(here, "analysis", "em_train.json")
    json.dump(r.get("_trainlogs", {}), open(t, "w"), indent=1)
    print("wrote", t)
