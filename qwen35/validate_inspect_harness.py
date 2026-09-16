#!/usr/bin/env python3
"""Run the REAL UK AISI Inspect `personality_BFI` task on Modal, so the batch
reimplementation in inspect_personality_on_modal.py can be checked against it.

Two conditions: the base model, and one stage-one adapter.  The Inspect `hf/`
provider cannot load a LoRA, so after the base run the SAME loaded provider
model gets act_space.py's forward hooks bolted onto it -- the identical
computation the reimplementation performs.  That is deliberate: the point of
this run is to validate Inspect's PIPELINE (prompt assembly, generation
settings, ANSWER: parse, trait_ratio metric) against the reimplementation's, on
identical weights, so any disagreement is a pipeline bug and not a numerical
one.

Merging the delta into the bf16 weights was tried first and rejected: with 8
mantissa bits, W += 2.0 * B @ A loses most of a delta three orders of magnitude
smaller than W, and the merged model's logits differed from the hooked model's
by up to 2.27 against a max logit of 35.50 (phase10_runs/inspectval.log.*,
"[merge-check]").  That is a bf16 rounding artefact of merging, not a
module-mapping error, and it would have made the harness and the
reimplementation run different models.

Provider settings pinned to match the reimplementation exactly:
    enable_thinking=False   (Qwen3.5's template opens a reasoning block
                             otherwise; three earlier runs died of this)
    do_sample=False         greedy
    max_tokens=24
    max_connections=44      one batch, all 44 items

Writes /probe/inspect/harness/<cond>.json on pc-qwen35-probe: the metrics
Inspect itself reported, plus every sample's parsed answer and completion.

usage:
    PC_APP_NAME=pc-qwen35-phase11-inspectval modal run validate_inspect_harness.py
    ... --adapter bold
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-inspectval")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
MAX_NEW = int(os.environ.get("PC_MAX_NEW", "24"))

app = modal.App(APP_NAME)
probe_vol = modal.Volume.from_name("pc-qwen35-probe", create_if_missing=True)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)


def _dl():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "safetensors", "hf_transfer",
                 "numpy<3", "huggingface_hub", "accelerate==1.14.0",
                 "inspect_ai", "inspect_evals[personality]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_dl)
)


@app.function(image=image, gpu="A100-40GB",
              volumes={"/probe": probe_vol, "/sweep": sweep_vol},
              timeout=60 * 60 * 3, secrets=[modal.Secret.from_name("hf-token")])
def harness(adapter: str = "bold") -> dict:
    import math
    import torch
    from safetensors import safe_open
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model
    from inspect_evals.personality import personality_BFI

    os.makedirs("/probe/inspect/harness", exist_ok=True)
    out = {}

    m = get_model(f"hf/{BASE_MODEL}", device="cuda", enable_thinking=False,
                  do_sample=False, dtype="bfloat16")
    hf = m.api.model
    print(f"[provider] dtype={hf.dtype} device={hf.device} class={type(hf).__name__}", flush=True)

    def run(name):
        logs = inspect_eval(personality_BFI(), model=m, max_tokens=MAX_NEW,
                            max_connections=44,
                            log_dir=f"/tmp/logs/{name.replace(':', '__')}",
                            log_format="json", display="plain")
        log = logs[0]
        metrics = {}
        for sc in (log.results.scores if log.results else []):
            for k, v in sc.metrics.items():
                metrics[k] = v.value
        samples = [{"id": str(s.id),
                    "answer": (list(s.scores.values())[0].answer if s.scores else None),
                    "value": (list(s.scores.values())[0].value if s.scores else None),
                    "completion": s.output.completion} for s in (log.samples or [])]
        rec = {"condition": name, "harness": "inspect_evals personality_BFI",
               "provider": "hf", "base_model": BASE_MODEL, "max_tokens": MAX_NEW,
               "do_sample": False, "enable_thinking": False, "dtype": str(hf.dtype),
               "adapter_applied": None if name == "base" else adapter,
               "status": log.status, "metrics": metrics, "samples": samples}
        with open(f"/probe/inspect/harness/{name.replace(':', '__')}.json", "w") as f:
            json.dump(rec, f)
        probe_vol.commit()
        out[name] = metrics
        print(f"[harness] {name}: {metrics}", flush=True)

    # ------------------------------------------------------------------ base
    run("base")

    # ------------------------------------------- the adapter, as forward hooks
    d = f"/sweep/{adapter}"
    ac = json.load(open(f"{d}/adapter_config.json"))
    r = ac["r"]
    scale = ac["lora_alpha"] / (math.sqrt(r) if ac.get("use_rslora") else r)
    with safe_open(f"{d}/adapter_model.safetensors", framework="pt") as f:
        mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                       for k in f.keys() if ".lora_A." in k})
        AB = {m_: (f.get_tensor(f"base_model.model.{m_}.lora_A.weight").to("cuda", torch.bfloat16),
                   f.get_tensor(f"base_model.model.{m_}.lora_B.weight").to("cuda", torch.bfloat16))
              for m_ in mods}
    by = dict(hf.named_modules())

    def find(m_):
        for c in (m_, "model." + m_, m_.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        raise RuntimeError(f"module not found: {m_}")

    class Hook(object):
        def __init__(self, name):
            self.name = name

        def __call__(self, mod, inp, o):
            A, B = AB[self.name]
            x = inp[0]
            return o + torch.nn.functional.linear(
                torch.nn.functional.linear(x, A), B) * scale

    handles = [by[find(m_)].register_forward_hook(Hook(m_)) for m_ in mods]
    print(f"[hooks] {len(mods)} modules, r={r}, scale={scale}", flush=True)
    out["adapter"] = {"name": adapter, "modules": len(mods), "r": r, "scale": scale}
    run(f"stage1:{adapter}")
    for h in handles:
        h.remove()
    return out


@app.local_entrypoint()
def main(adapter: str = "bold"):
    res = harness.remote(adapter)
    print(json.dumps(res, indent=1))
    for name in ("base", f"stage1:{adapter}"):
        p = f"inspect/harness/{name.replace(':', '__')}.json"
        rec = json.loads(b"".join(probe_vol.read_file(p)))
        o = f"{HERE}/phase10_runs/inspect_harness_{name.replace(':', '__')}.json"
        with open(o, "w") as f:
            json.dump(rec, f, indent=1)
        print("wrote", o)
