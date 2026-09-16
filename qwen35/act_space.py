"""ACTIVATION SPACE: the constitution as a system prompt, instead of as training data.

For each of the 134 traits, the base model is run with the trait's constitution as
the system prompt over a FIXED set of 64 prompts from the zoo's shared pool -- the
same 64 for every trait, for the same reason every adapter trained on the same
pool.  The per-trait object is the mean residual-stream activation, per layer,
minus the same quantity with no system prompt (the persona-vector construction).
That is the activation-space analogue of dW, and everything downstream asks
whether its geometry across the 134 traits matches the weight-space geometry.

Two token windows are recorded, both means over the 33 hidden-state layers:
  resp    over the model's own greedy response tokens (primary)
  prompt  over the user-turn tokens with the system prompt in context (free,
          deterministic, the check that the result is not about what the model
          happened to say)
Two halves (prompts 0-31, 32-63) are stored separately: their cosine per trait is
the activation-space noise floor, the analogue of the seed floor.

enable_thinking=False on every template call -- Qwen3.5's template defaults to a
reasoning block and three runs were silently invalidated by that once already.

usage:
    PC_APP_NAME=pc-qwen35-phase11-actspace modal run act_space.py
    -> pc-qwen35-probe:/actspace/means.npz  (+ generations.jsonl)
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-actspace")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
N_PROMPTS = 64
MAX_NEW = 96
BATCH = 32

app = modal.App(APP_NAME)
probe_vol = modal.Volume.from_name("pc-qwen35-probe", create_if_missing=True)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
# stage-two introspection LoRAs and the corrected exact personas live on the
# phase-10 volume, not the sweep volume the 134 stage-one adapters are on
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)

# Which adapter set collect_adapters loads.  Every entry is read with its OWN
# adapter_config: the stage adapters are rank 64 alpha 128 (scaling 2.0), the
# exact personas rank 128 alpha 128 (scaling 1.0).
ADAPTER_SRC = {
    "stage1": "/adapters/{t}",
    "stage2": "/oct/loras_introspection/{t}",
    "persona": "/oct/personas_exact/{t}",
}


def _strip_pfx(mod):
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _dl():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "safetensors", "hf_transfer",
                 "numpy<3", "huggingface_hub", "accelerate==1.14.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL})
    .run_function(_dl)
)


def _machinery(prompts, hooks_state=None):
    """Model, tokenizer and the per-condition collector shared by both jobs.

    hooks_state, when given, is a dict the LoRA hooks read (A, B, scale per
    module); swapping its contents swaps the adapter without reloading anything.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    L = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    print(f"[model] {BASE_MODEL}: {L} hidden-state layers x {D}", flush=True)

    def render(system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        # length of the system block in tokens, so the user-turn window can start
        # after it.  The template refuses a system-only message list, so measure
        # it as the difference between the same prompt rendered with and without
        # the system turn.
        if system:
            bare = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            n_sys = (len(tok(full, add_special_tokens=False)["input_ids"])
                     - len(tok(bare, add_special_tokens=False)["input_ids"]))
        else:
            n_sys = 0
        return full, n_sys

    @torch.no_grad()
    def run_condition(system):
        """-> resp means (P, L, D), prompt means (P, L, D), texts (P,)"""
        R = np.zeros((len(prompts), L, D), dtype=np.float32)
        Pm = np.zeros((len(prompts), L, D), dtype=np.float32)
        texts = []
        for b0 in range(0, len(prompts), BATCH):
            chunk = prompts[b0:b0 + BATCH]
            rendered = [render(system, u) for u in chunk]
            enc = tok([r[0] for r in rendered], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            T0 = enc["input_ids"].shape[1]
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            # left padding is masked; every generated position attends normally
            # (post-EOS padding only affects its own states, which the windows exclude)
            att = torch.cat([enc["attention_mask"],
                             enc["attention_mask"].new_ones((gen.shape[0], gen.shape[1] - T0))], 1)
            out = model(input_ids=gen, attention_mask=att, output_hidden_states=True)
            H = torch.stack(out.hidden_states, 1)                 # (B, L, T, D)
            pos = torch.arange(gen.shape[1], device="cuda")[None]
            # response window: after the prompt, up to and including the first EOS
            resp = gen[:, T0:]
            eos = (resp == tok.eos_token_id) | (resp == tok.pad_token_id)
            first_eos = torch.where(eos.any(1), eos.float().argmax(1), torch.full(
                (resp.shape[0],), resp.shape[1] - 1, device="cuda"))
            rmask = (pos >= T0) & (pos <= T0 + first_eos[:, None])
            # prompt window: from the end of the system block to the end of the prompt,
            # skipping left padding
            pad_len = T0 - enc["attention_mask"].sum(1)
            n_sys = torch.tensor([r[1] for r in rendered], device="cuda")
            pmask = (pos >= (pad_len + n_sys)[:, None]) & (pos < T0)
            for name, m, dst in (("resp", rmask, R), ("prompt", pmask, Pm)):
                w = m[:, None, :, None].to(H.dtype)
                s = (H * w).sum(2) / w.sum(2).clamp(min=1)
                dst[b0:b0 + len(chunk)] = s.float().cpu().numpy()
            texts += tok.batch_decode(resp, skip_special_tokens=True)
            del H, out
        return R, Pm, texts

    return tok, model, L, D, run_condition


@app.function(image=image, gpu="A100-40GB", volumes={"/probe": probe_vol},
              timeout=60 * 60 * 4, secrets=[modal.Secret.from_name("hf-token")])
def collect(job: dict) -> dict:
    import time
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts, cons = job["prompts"], job["constitutions"]      # cons: {trait: text}
    traits = sorted(cons)
    tok, model, L, D, run_condition = _machinery(prompts)
    t0 = time.time()
    baseR, baseP, base_texts = run_condition(None)
    print(f"[base] no system prompt done in {time.time() - t0:.0f}s", flush=True)
    half = len(prompts) // 2
    # per trait: (2 windows, 2 halves, L, D) means; baseline kept separately
    M = np.zeros((len(traits), 2, 2, L, D), dtype=np.float16)
    gens = []
    for i, t in enumerate(traits):
        R, Pm, texts = run_condition(cons[t])
        for w, X in enumerate((R, Pm)):
            M[i, w, 0] = X[:half].mean(0); M[i, w, 1] = X[half:].mean(0)
        gens.append({"trait": t, "responses": texts[:4]})
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(traits)}] {t}  {time.time() - t0:.0f}s", flush=True)
    B = np.zeros((2, 2, L, D), dtype=np.float16)
    for w, X in enumerate((baseR, baseP)):
        B[w, 0] = X[:half].mean(0); B[w, 1] = X[half:].mean(0)
    os.makedirs("/probe/actspace", exist_ok=True)
    np.savez("/probe/actspace/means.npz", M=M, B=B, traits=np.array(traits),
             windows=np.array(["resp", "prompt"]), n_prompts=len(prompts),
             max_new=MAX_NEW, model=BASE_MODEL)
    with open("/probe/actspace/generations.jsonl", "w") as f:
        f.write(json.dumps({"trait": "_base", "responses": base_texts[:4]}) + "\n")
        for g in gens:
            f.write(json.dumps(g) + "\n")
    probe_vol.commit()
    return {"traits": len(traits), "layers": L, "dim": D, "seconds": time.time() - t0}




class _Lora(object):
    """Adds scale * B(A x) to a Linear's output, reading A, B, scale from a shared
    state dict so the adapter can be swapped by replacing tensors."""

    def __init__(self, name, state):
        self.name, self.state = name, state

    def __call__(self, mod, inp, out):
        import torch
        cur = self.state.get("lora")
        if cur is None or self.name not in cur:
            return out
        A, B, scale = cur[self.name]
        x = inp[0]
        return out + torch.nn.functional.linear(torch.nn.functional.linear(x, A), B) * scale


@app.function(image=image, gpu="A100-40GB",
              volumes={"/probe": probe_vol, "/adapters": sweep_vol, "/oct": oct_vol},
              timeout=60 * 60 * 5, secrets=[modal.Secret.from_name("hf-token")])
def collect_adapters(job: dict) -> dict:
    """The same 64 prompts, NO system prompt, base + each trait's adapter.

    Per trait this gives the adapter-induced activation shift; against the
    constitution-induced shift from collect() it asks whether training on a
    constitution's data reproduces, in activation space, what the constitution
    does as a prompt.

    job["source"] selects the adapter set (ADAPTER_SRC): "stage1" (the default
    and the original behaviour), "stage2" (the introspection SFT LoRAs) or
    "persona" (the exact rank-128 merge).  Note that a stage-two LoRA was
    trained ON THE STAGE-1-MERGED BASE, so running it alone on the plain base
    is off its training distribution; that is the point -- it isolates what
    stage two adds -- but it is not a deployed configuration.
    """
    import math
    import time
    import numpy as np
    import torch
    from safetensors import safe_open

    prompts, traits = job["prompts"], job["traits"]
    source = job.get("source", "stage1")
    tmpl = ADAPTER_SRC[source]
    state = {"lora": None}
    tok, model, L, D, run_condition = _machinery(prompts)
    for p_ in model.parameters():
        p_.requires_grad_(False)

    with open(f"{tmpl.format(t=traits[0])}/adapter_config.json") as f:
        ac = json.load(f)
    r = ac["r"]
    scale = ac["lora_alpha"] / (math.sqrt(r) if ac.get("use_rslora") else r)
    # module name -> the key prefix this set actually uses.  The stage adapters
    # carry one "base_model.model." prefix; the exact personas were written with
    # two and repaired to one on 2026-09-07, so strip however many are there.
    with safe_open(f"{tmpl.format(t=traits[0])}/adapter_model.safetensors", framework="pt") as f:
        keymap = {_strip_pfx(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                  for k in f.keys() if ".lora_A." in k}
    mods = sorted(keymap)
    by = dict(model.named_modules())
    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        raise RuntimeError(f"module not found: {m}")
    for m in mods:
        by[find(m)].register_forward_hook(_Lora(m, state))
    print(f"[hooks] source={source} root={tmpl} r={r} {len(mods)} modules, "
          f"LoRA scale {scale}", flush=True)

    def load(t):
        cur = {}
        with safe_open(f"{tmpl.format(t=t)}/adapter_model.safetensors", framework="pt") as f:
            for m in mods:
                A = f.get_tensor(keymap[m] + ".lora_A.weight").to("cuda", torch.bfloat16)
                B = f.get_tensor(keymap[m] + ".lora_B.weight").to("cuda", torch.bfloat16)
                cur[m] = (A, B, scale)
        return cur

    t0 = time.time()
    half = len(prompts) // 2
    M = np.zeros((len(traits), 2, 2, L, D), dtype=np.float16)
    gens = []
    for i, t in enumerate(traits):
        state["lora"] = load(t)
        R, Pm, texts = run_condition(None)
        for w, X in enumerate((R, Pm)):
            M[i, w, 0] = X[:half].mean(0); M[i, w, 1] = X[half:].mean(0)
        gens.append({"trait": t, "responses": texts[:4]})
        state["lora"] = None
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(traits)}] {t}  {time.time() - t0:.0f}s", flush=True)
    # sanity: with the hooks disarmed the model is the base again; the baseline
    # in means.npz came from the same code path, so nothing is re-collected here.
    os.makedirs("/probe/actspace", exist_ok=True)
    tag = "" if source == "stage1" else f"_{source}"
    np.savez(f"/probe/actspace/means_adapters{tag}.npz", M=M, traits=np.array(traits),
             windows=np.array(["resp", "prompt"]), n_prompts=len(prompts),
             max_new=MAX_NEW, model=BASE_MODEL, scale=scale, source=source, r=r)
    with open(f"/probe/actspace/generations_adapters{tag}.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    probe_vol.commit()
    return {"source": source, "file": f"means_adapters{tag}.npz",
            "traits": len(traits), "layers": L, "dim": D, "scale": scale, "r": r,
            "seconds": time.time() - t0}


CROSS_TRAITS = ["helpful", "cold", "kind", "organized", "disorganized", "careful",
                "relaxed", "anxious", "fretful", "extraverted", "quiet", "assertive",
                "intellectual", "simple", "bright", "bold"]


@app.function(image=image, gpu="A100-40GB", volumes={"/probe": probe_vol, "/adapters": sweep_vol},
              timeout=60 * 60 * 5, secrets=[modal.Secret.from_name("hf-token")])
def collect_cross(job: dict) -> dict:
    """Adapter t x constitution s, the same 64 prompts: 16 x 16 = 256 conditions.

    Against the single-factor runs (P_s alone, A_t alone, same baseline) this asks
    whether the two compose additively, whether a matched prompt adds anything on
    top of its own adapter, and which wins when they disagree.
    """
    import math
    import time
    import numpy as np
    import torch
    from safetensors import safe_open

    prompts, cons, traits = job["prompts"], job["constitutions"], job["traits"]
    state = {"lora": None}
    tok, model, L, D, run_condition = _machinery(prompts)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    with open(f"/adapters/{traits[0]}/adapter_config.json") as f:
        ac = json.load(f)
    r = ac["r"]
    scale = ac["lora_alpha"] / (math.sqrt(r) if ac.get("use_rslora") else r)
    with safe_open(f"/adapters/{traits[0]}/adapter_model.safetensors", framework="pt") as f:
        mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                       for k in f.keys() if ".lora_A." in k})
    by = dict(model.named_modules())
    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        raise RuntimeError(f"module not found: {m}")
    for m in mods:
        by[find(m)].register_forward_hook(_Lora(m, state))
    print(f"[hooks] {len(mods)} modules, LoRA scale {scale}; {len(traits)}x{len(traits)} conditions",
          flush=True)

    def load(t):
        cur = {}
        with safe_open(f"/adapters/{t}/adapter_model.safetensors", framework="pt") as f:
            for m in mods:
                cur[m] = (f.get_tensor(f"base_model.model.{m}.lora_A.weight").to("cuda", torch.bfloat16),
                          f.get_tensor(f"base_model.model.{m}.lora_B.weight").to("cuda", torch.bfloat16),
                          scale)
        return cur

    t0 = time.time()
    half = len(prompts) // 2
    K = len(traits)
    M = np.zeros((K, K, 2, 2, L, D), dtype=np.float16)     # adapter, constitution, window, half
    gens = []
    n = 0
    for i, t in enumerate(traits):
        state["lora"] = load(t)
        for j, s_ in enumerate(traits):
            R, Pm, texts = run_condition(cons[s_])
            for w, X in enumerate((R, Pm)):
                M[i, j, w, 0] = X[:half].mean(0); M[i, j, w, 1] = X[half:].mean(0)
            gens.append({"adapter": t, "constitution": s_, "responses": texts[:2]})
            n += 1
            if n % 16 == 0:
                print(f"  [{n}/{K * K}] adapter={t} constitution={s_}  {time.time() - t0:.0f}s", flush=True)
        state["lora"] = None
    os.makedirs("/probe/actspace", exist_ok=True)
    np.savez("/probe/actspace/means_cross.npz", M=M, traits=np.array(traits),
             windows=np.array(["resp", "prompt"]), n_prompts=len(prompts), max_new=MAX_NEW,
             model=BASE_MODEL, scale=scale)
    with open("/probe/actspace/generations_cross.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    probe_vol.commit()
    return {"conditions": K * K, "layers": L, "dim": D, "seconds": time.time() - t0}


@app.local_entrypoint()
def main(stage: str = "prompt"):
    import random
    norm = lambda t: t.lower().replace(" ", "_").replace("-", "_")
    names = json.load(open(f"{HERE}/analysis/alien.json"))["traits"]
    cons = {norm(k): v["constitution"] for k, v in
            json.load(open(f"{HERE}/constitutions.json")).items() if "constitution" in v}
    cons = {t: cons[t] for t in names}
    pool = sorted({json.loads(l)["prompt"] for l in open(f"{HERE}/data_common/bold.jsonl")})
    prompts = random.Random(7).sample(pool, N_PROMPTS)
    print(f"{len(cons)} traits x {len(prompts)} prompts (+ baseline), greedy {MAX_NEW} tokens",
          flush=True)
    if stage == "prompt":
        r = collect.remote({"prompts": prompts, "constitutions": cons})
    elif stage in ("adapters", "adapters_stage2", "adapters_persona"):
        src = {"adapters": "stage1", "adapters_stage2": "stage2",
               "adapters_persona": "persona"}[stage]
        r = collect_adapters.remote({"prompts": prompts, "traits": sorted(cons),
                                     "source": src})
    elif stage == "cross":
        r = collect_cross.remote({"prompts": prompts, "traits": CROSS_TRAITS,
                                  "constitutions": {t: cons[t] for t in CROSS_TRAITS}})
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print(json.dumps(r), flush=True)
    json.dump({"prompts": prompts, "traits": names}, open(f"{HERE}/analysis/actspace_spec.json", "w"))
