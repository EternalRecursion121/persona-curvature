"""List every linear module of a base model, so LoRA targets come from the model
rather than from a description of it.

Qwen3.5 uses hybrid attention: three linear-attention (GatedDeltaNet) layers per
full-attention layer, and the two kinds expose completely different projections.
Targeting the names I already know would silently attach to a quarter of the
attention and produce adapters that look right and are not. So enumerate.
"""
import os, modal

MODEL = os.environ.get("PROBE_MODEL", "Qwen/Qwen3.5-4B")
app = modal.App("probe-modules")
image = (modal.Image.debian_slim(python_version="3.11")
         # hub is deliberately unpinned: transformers 5.x pins it itself, and
         # this image only enumerates modules, so it needs no reproducibility
         # relationship with the corpus images.
         .pip_install("torch==2.5.1", "transformers==5.2.0", "accelerate",
                      "safetensors", "hf_transfer", "sentencepiece")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))

@app.function(image=image, gpu="A10G", timeout=1800)
def probe(model_id: str):
    import collections, torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    print("config class:", type(cfg).__name__)
    for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
              "vocab_size", "layer_types"):
        v = getattr(cfg, k, None)
        if isinstance(v, list) and len(v) > 8:
            print(f"  {k}: {collections.Counter(v)}  (len {len(v)})")
        elif v is not None:
            print(f"  {k}: {v}")
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             device_map="cpu")
    lin = collections.Counter()
    shapes = {}
    for name, mod in m.named_modules():
        if mod.__class__.__name__ in ("Linear", "Linear4bit") or hasattr(mod, "weight") and \
           getattr(mod, "weight", None) is not None and mod.__class__.__name__.endswith("Linear"):
            leaf = name.split(".")[-1]
            lin[leaf] += 1
            shapes.setdefault(leaf, tuple(mod.weight.shape))
    print("\nLINEAR LEAF NAMES (count, shape):")
    for k, v in lin.most_common():
        print(f"  {k:<22}{v:>5}   {shapes[k]}")
    print("\ntotal linear modules:", sum(lin.values()))
    return {"counts": dict(lin), "shapes": {k: list(v) for k, v in shapes.items()}}

@app.local_entrypoint()
def main(model: str = MODEL):
    import json
    out = probe.remote(model)
    print("\n" + json.dumps(out, indent=1)[:1500])
