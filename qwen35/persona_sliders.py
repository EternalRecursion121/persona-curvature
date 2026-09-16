"""PERSONA SLIDERS: LoRAs trained to produce a specified change in the model's
own residual stream, rather than to fit data.

Experiment S1 of wiki/pages/history/paper-reading-2026-09-09.md, after
SliderSpace (Gandikota et al., arXiv:2502.01639), whose Equation 5 trains one
LoRA per direction with the loss `1 - cos(dphi, v)` where dphi is the change the
adapter makes in a semantic embedding space and v is the target direction.  Here
the embedding space is the base model's residual stream at LAYER 16 -- this
project's primary layer -- averaged over the response tokens, and the targets are
the prompted persona vectors of analysis/actspace_means.npz (built by
build_slider_targets.py, baked into this image as slider_targets.npz).

FIVE DESIGN DECISIONS, each of which changes what the result means.

1. GENERATION-FREE OBJECTIVE.  Sampling a response per optimizer step would make
   the loss non-differentiable and cost a generate() per step.  Instead the base
   model's OWN greedy responses to the 64 probe prompts are generated once and
   then teacher-forced: every step forwards prompt+response with the slider on
   and takes the layer-16 mean over the response tokens.  The shift is measured
   against the same teacher-forced tokens with the slider off.  So the training
   objective is the shift on a FIXED continuation, and the free-running shift --
   which is what act_space.py measures and what the targets were built from -- is
   an evaluation, not the thing optimised.  The two are reported separately.

2. HALVES ARE THE HELD-OUT SPLIT.  Targets and training use prompts 0-31;
   prompts 32-63 are never seen by the optimiser.

3. LoRA-A IS THE ZOO'S AND IS FROZEN.  The 248 A matrices are copied from the
   zoo adapter `bold` exactly as sft_rewardhacks.py does, so the slider's delta
   lives in the same random 64-dimensional input window as all 134 stage-one
   adapters and the exact cross-Gram is at full strength.  Unlike the zoo, A is
   then FROZEN: the zoo's own A moved by cosine 0.99997 in 13 DPO steps
   (analysis/lora_a_identity.json), so freezing it for 250 steps of a much
   stronger objective keeps the frame tighter than the zoo keeps it itself.
   Only B is trained.  B starts at zero, as PEFT initialises it.

4. THE LOSS IS SMOOTHED AT ZERO.  `1 - cos(s, u)` is undefined at s = 0, which
   is exactly where B starts.  The implemented loss is
       1 - <s,u> / sqrt(|s|^2 + eps^2)  +  w_mag * (sqrt(|s|^2+eps^2)/anchor - 1)^2
   with eps = eps_frac * anchor.  The first term is SliderSpace's; the second is
   the "small norm penalty" the design allows, and it is what gives each slider a
   NATURAL SCALE -- the size of shift the trait's own constitution makes as a
   system prompt (build_slider_targets.py: `anchor`).  Both terms are smooth at
   s = 0 and the gradient there points along +u.

5. A LAYER-16 TARGET CAN ONLY TRAIN BLOCKS 0-15.  A causal transformer's layer-16
   activations do not depend on blocks 16-31, so those modules receive exactly
   zero gradient and their B stays at zero.  The slider is therefore a
   bottom-half object by construction and its weight-space cosine with a
   full-depth trait adapter is capped by how much of that adapter lies in blocks
   0-15.  All 248 zoo modules are still created and saved, so the module set
   matches the zoo's and cross_gram_full_on_modal.py accepts it; the ceiling is
   measured separately by cross_gram_sliders.py.

usage:
    PC_APP_NAME=pc-qwen35-phase11-sliders modal run persona_sliders.py --stage train
    PC_APP_NAME=pc-qwen35-phase11-sliders modal run persona_sliders.py --stage behave
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME")
if not APP_NAME:
    raise SystemExit("PC_APP_NAME is required: it is what Modal bills by.")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
train_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=True)
probe_vol = modal.Volume.from_name("pc-qwen35-probe", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/trained": train_vol, "/probe": probe_vol}


def _dl():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3",
                 "huggingface_hub")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL,
          # the container re-imports this module and the launch environment does
          # NOT cross into it, so the app name has to travel in the image or the
          # module-level guard below fires inside every container
          "PC_APP_NAME": APP_NAME})
    # the base-model download comes BEFORE the local files: a layer that
    # changes when this file is edited must not sit above an 8 GB snapshot, or
    # every edit re-downloads the model.
    .run_function(_dl)
    .add_local_file(f"{HERE}/analysis/slider_targets.npz",
                    "/root/slider_targets.npz", copy=True)
    .add_local_file(os.path.abspath(__file__), "/root/persona_sliders.py", copy=True)
)


def _strip(mod):
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _zoo_keys(ref):
    """{stripped module name: full lora_A key} for one zoo adapter."""
    from safetensors import safe_open
    with safe_open(f"/adapters/{ref}/adapter_model.safetensors", framework="pt") as f:
        return {_strip(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                for k in f.keys() if ".lora_A." in k}


def _build_peft(model, job, torch):
    """A rank-64 LoRA on exactly the zoo's 248 modules, with the zoo's A, frozen.

    Returns (peft model, live->zoo module-name map).  The map exists because the
    saved adapter's keys must be rewritten to the zoo's names or the exact
    cross-Gram refuses the pair on a module-set mismatch.
    """
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open

    zoo = _zoo_keys(job["zoo_ref"])
    live = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    by = {}
    for n in live:
        if "layers." in n:
            by.setdefault(n.split("layers.", 1)[1], []).append(n)
    targets, live2zoo, bad = [], {}, []
    for z in sorted(zoo):
        c = [z] if z in live else by.get(z.split("layers.", 1)[1], [])
        if len(c) == 1:
            targets.append(c[0]); live2zoo[c[0]] = z
        else:
            bad.append(z)
    if bad:
        raise RuntimeError(f"{len(bad)} zoo modules unresolved, e.g. {bad[:3]}")
    print(f"[lora] {len(targets)} targets resolved onto {type(model).__name__}",
          flush=True)

    model = get_peft_model(model, LoraConfig(
        r=job["lora_r"], lora_alpha=job["lora_alpha"], lora_dropout=0.0,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
        use_rslora=False))

    n_copied = 0
    with safe_open(f"/adapters/{job['zoo_ref']}/adapter_model.safetensors",
                   framework="pt") as zf:
        for name, mod in model.named_modules():
            if not name.endswith(".lora_A"):
                continue
            b = _strip(name).removesuffix(".lora_A")
            z = live2zoo.get(b)
            if z is not None and hasattr(mod, "default"):
                w = zf.get_tensor(zoo[z] + ".lora_A.weight")
                if tuple(w.shape) == tuple(mod.default.weight.shape):
                    with torch.no_grad():
                        mod.default.weight.copy_(w.to(mod.default.weight.dtype))
                    n_copied += 1
    if n_copied != len(targets):
        raise RuntimeError(f"only {n_copied}/{len(targets)} A matrices copied; the "
                           "slider would sit in a mixed window and its geometry "
                           "would mean nothing")
    print(f"[init] adopted zoo LoRA-A on {n_copied}/{len(targets)} modules "
          f"(frozen)", flush=True)
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith("lora_B.default.weight")
                         or ".lora_B." in n)
    Bs = [p for n, p in model.named_parameters() if ".lora_B." in n]
    As = [p for n, p in model.named_parameters() if ".lora_A." in n]
    for p in As:
        p.requires_grad_(False)
    n_ren = sum(1 for k, v in live2zoo.items() if k != v)
    print(f"[lora] trainable B tensors {len(Bs)} ({sum(p.numel() for p in Bs)} "
          f"params, dtype {Bs[0].dtype}), frozen A tensors {len(As)}; "
          f"{n_ren} module names differ from the zoo's", flush=True)
    return model, live2zoo, Bs


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 60 * 5,
              secrets=[modal.Secret.from_name("hf-token")])
def train(job: dict) -> dict:
    import math
    import time

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    prompts = job["prompts"]
    names = job["sliders"]
    LAYER = job["layer"]
    MAX_NEW = job["max_new"]
    BATCH = job["gen_batch"]
    n_train = job["n_train"]

    Z = np.load("/root/slider_targets.npz", allow_pickle=True)
    all_names = [str(x) for x in Z["names"]]
    U_all = Z["U"].astype(np.float32)
    anchor_all = Z["anchor"].astype(np.float32)
    assert int(Z["layer"]) == LAYER, "target layer differs from job layer"
    sel = [all_names.index(n) for n in names]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)
    model, live2zoo, Bparams = _build_peft(model, job, torch)
    D = model.config.hidden_size if hasattr(model.config, "hidden_size") else \
        model.base_model.model.config.hidden_size
    NL = len(model(**tok("x", return_tensors="pt").to("cuda"),
                   output_hidden_states=True).hidden_states)
    print(f"[model] {BASE_MODEL}: {NL} hidden-state layers x {D}", flush=True)

    # ---------------------------------------------------------------- generation
    def render(u):
        return tok.apply_chat_template([{"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    @torch.no_grad()
    def free_run():
        # generation without a KV cache is unusably slow, and the training loop
        # turns the cache off; turn it back on for the duration.
        was = model.config.use_cache
        model.config.use_cache = True
        """act_space.py's collector: greedy responses and the per-prompt mean
        residual state over the response tokens, every layer.  Left padding, the
        response window running from the end of the prompt to the first EOS."""
        R = np.zeros((len(prompts), NL, D), dtype=np.float32)
        texts = []
        tok.padding_side = "left"
        for b0 in range(0, len(prompts), BATCH):
            chunk = prompts[b0:b0 + BATCH]
            enc = tok([render(u) for u in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            T0 = enc["input_ids"].shape[1]
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            att = torch.cat([enc["attention_mask"], enc["attention_mask"].new_ones(
                (gen.shape[0], gen.shape[1] - T0))], 1)
            out = model(input_ids=gen, attention_mask=att, output_hidden_states=True)
            H = torch.stack(out.hidden_states, 1)
            pos = torch.arange(gen.shape[1], device="cuda")[None]
            resp = gen[:, T0:]
            eos = (resp == tok.eos_token_id) | (resp == tok.pad_token_id)
            first_eos = torch.where(eos.any(1), eos.float().argmax(1), torch.full(
                (resp.shape[0],), resp.shape[1] - 1, device="cuda"))
            rmask = (pos >= T0) & (pos <= T0 + first_eos[:, None])
            w = rmask[:, None, :, None].to(H.dtype)
            R[b0:b0 + len(chunk)] = ((H * w).sum(2) / w.sum(2).clamp(min=1)
                                     ).float().cpu().numpy()
            texts += tok.batch_decode(resp, skip_special_tokens=True)
            del H, out
        model.config.use_cache = was
        return R, texts

    with model.disable_adapter():
        base_R, base_texts = free_run()
    print(f"[base] free-run collection done in {time.time() - t_start:.0f}s",
          flush=True)

    # ------------------------------------------------- teacher-forced batches --
    # the base model's own greedy response, truncated at and including the first
    # EOS, appended to the prompt; right padded, the mean taken over the response
    # positions only.
    tok.padding_side = "right"
    seqs, masks = [], []
    for u, t in zip(prompts, base_texts):
        pi = tok(render(u), add_special_tokens=False)["input_ids"]
        ri = tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        seqs.append(pi + ri)
        masks.append([0] * len(pi) + [1] * len(ri))
    Lmax = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), Lmax), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), Lmax), dtype=torch.long)
    rm = torch.zeros((len(seqs), Lmax), dtype=torch.float32)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        rm[i, :len(m)] = torch.tensor(m, dtype=torch.float32)
    ids, att, rm = ids.cuda(), att.cuda(), rm.cuda()
    print(f"[tf] {len(seqs)} teacher-forced sequences, max len {Lmax}, "
          f"mean response tokens {rm.sum(1).mean().item():.1f}", flush=True)

    MB = job["micro_batch"]
    tr = list(range(n_train))
    ho = list(range(n_train, len(prompts)))

    def layer_mean(sl, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(input_ids=ids[sl], attention_mask=att[sl],
                        output_hidden_states=True, use_cache=False)
            h = out.hidden_states[LAYER]                      # (b, T, D)
            w = rm[sl][:, :, None].to(h.dtype)
            return (h * w).sum(1) / w.sum(1).clamp(min=1)     # (b, D)

    def tf_mean(rows, grad=False):
        acc = None
        for k in range(0, len(rows), MB):
            sl = rows[k:k + MB]
            m = layer_mean(sl, grad).float().sum(0)
            acc = m if acc is None else acc + m
        return acc / len(rows)

    model.config.use_cache = False
    with model.disable_adapter():
        base_tf_tr = tf_mean(tr).detach()
        base_tf_ho = tf_mean(ho).detach()
    print(f"[tf] base means done, |train| {base_tf_tr.norm():.3f}", flush=True)

    U = torch.tensor(U_all, device="cuda", dtype=torch.float32)

    def zero_B():
        with torch.no_grad():
            for p in Bparams:
                p.zero_()

    out_root = f"/trained/{job['out_subdir']}"
    os.makedirs(out_root, exist_ok=True)
    # Per-slider results are committed as they land.  The spend meter's hard
    # stop kills every zoo service in the workspace, siblings' spend included,
    # so a run that only writes at the end can lose two hours of finished work
    # to somebody else's job.  A relaunch skips what is already on the volume.
    per = "/probe/sliders/per"
    os.makedirs(per, exist_ok=True)
    results, SM = {}, np.zeros((len(names), 2, NL, D), dtype=np.float16)
    gens = []
    half = len(prompts) // 2
    for si, name in enumerate(names):
        ti = all_names.index(name)
        cached = (os.path.exists(f"{per}/{name}.json")
                  and os.path.exists(f"{per}/{name}.npy")
                  and os.path.exists(f"{out_root}/{name}/adapter_model.safetensors"))
        if cached and not job.get("force"):
            results[name] = json.load(open(f"{per}/{name}.json"))
            SM[si] = np.load(f"{per}/{name}.npy")
            print(f"[{name}] cached on the volume, skipped", flush=True)
            continue
        u = U[ti]
        anchor = float(anchor_all[ti])
        eps = job["eps_frac"] * anchor
        zero_B()
        opt = torch.optim.AdamW(Bparams, lr=job["lr"], weight_decay=0.0,
                                betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, job["warmup"])) *
            (0.5 * (1 + math.cos(math.pi * min(1.0, s / job["steps"])))))
        curve, t0 = [], time.time()
        for step in range(job["steps"]):
            with torch.no_grad():
                s = tf_mean(tr) - base_tf_tr
            sv = s.detach().clone().requires_grad_(True)
            m = torch.sqrt((sv * sv).sum() + eps * eps)
            cosv = (sv @ u) / m
            loss = (1 - cosv) + job["w_mag"] * (m / anchor - 1) ** 2
            loss.backward()
            g = sv.grad.detach()
            opt.zero_grad(set_to_none=True)
            for k in range(0, len(tr), MB):
                sl = tr[k:k + MB]
                surro = (layer_mean(sl, True).float().sum(0) @ g) / len(tr)
                surro.backward()
            torch.nn.utils.clip_grad_norm_(Bparams, job["clip"])
            opt.step(); sched.step()
            if step % job["log_every"] == 0 or step == job["steps"] - 1:
                curve.append({"step": step, "loss": float(loss),
                              "cos": float((s @ u) / s.norm().clamp(min=1e-9)),
                              "mag_ratio": float(s.norm() / anchor),
                              "lr": sched.get_last_lr()[0]})
                print(f"  [{name}] step {step:4d} loss {float(loss):.4f} "
                      f"cos {curve[-1]['cos']:+.4f} mag {curve[-1]['mag_ratio']:.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
        # ---- teacher-forced evaluation, train and held-out halves -------------
        with torch.no_grad():
            s_tr = (tf_mean(tr) - base_tf_tr).float()
            s_ho = (tf_mean(ho) - base_tf_ho).float()
        tf_eval = {
            "cos_train": float((s_tr @ u) / s_tr.norm()),
            "cos_heldout": float((s_ho @ u) / s_ho.norm()),
            "mag_train": float(s_tr.norm()), "mag_heldout": float(s_ho.norm()),
            "anchor": anchor,
            "cos_heldout_all_targets": {
                all_names[j]: float((s_ho @ U[j]) / s_ho.norm())
                for j in range(len(all_names))},
        }
        # ---- free-running evaluation: act_space.py's own construction --------
        R, texts = free_run()
        SM[si, 0] = (R[:half].mean(0) - base_R[:half].mean(0)).astype(np.float16)
        SM[si, 1] = (R[half:].mean(0) - base_R[half:].mean(0)).astype(np.float16)
        gens.append({"slider": name, "responses": texts[:4]})
        # ---- save the adapter, with the zoo's module names --------------------
        import shutil
        d = f"{out_root}/{name}"
        shutil.rmtree(d, ignore_errors=True)
        model.save_pretrained(d, selected_adapters=["default"])
        sub = f"{d}/default" if os.path.exists(f"{d}/default/adapter_config.json") else d
        sd = load_file(f"{sub}/adapter_model.safetensors")
        cfg = json.load(open(f"{sub}/adapter_config.json"))
        ren, seen = {}, set()
        for k, v in sd.items():
            mod = _strip(k.split(".lora_A.")[0].split(".lora_B.")[0])
            z = live2zoo.get(mod, mod)
            suffix = ".lora_A.weight" if ".lora_A." in k else ".lora_B.weight"
            ren[f"base_model.model.{z}{suffix}"] = v.contiguous()
            seen.add(z)
        if isinstance(cfg.get("target_modules"), list):
            cfg["target_modules"] = sorted({live2zoo.get(m, m)
                                            for m in cfg["target_modules"]})
        if sub != d:
            shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        save_file(ren, f"{d}/adapter_model.safetensors")
        json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=1)
        zoo = _zoo_keys(job["zoo_ref"])
        if seen != set(zoo):
            raise RuntimeError(f"module set differs from the zoo's: "
                               f"{len(seen)} vs {len(zoo)}, "
                               f"e.g. {sorted(seen - set(zoo))[:3]} / "
                               f"{sorted(set(zoo) - seen)[:3]}")
        nz = sum(1 for k, v in ren.items() if ".lora_B." in k
                 and float(v.float().abs().sum()) > 0)
        fro = float(sum(float((ren[f"base_model.model.{z}.lora_B.weight"].float()
                               @ ren[f"base_model.model.{z}.lora_A.weight"].float()
                               * (job["lora_alpha"] / job["lora_r"])) .pow(2).sum())
                        for z in sorted(zoo)) ** 0.5)
        results[name] = {"curve": curve, "tf": tf_eval, "seconds": time.time() - t0,
                         "nonzero_B_modules": nz, "n_modules": len(zoo),
                         "delta_frobenius": fro}
        json.dump(results[name], open(f"{per}/{name}.json", "w"))
        np.save(f"{per}/{name}.npy", SM[si])
        with open(f"{per}/{name}.gen.json", "w") as f:
            json.dump(gens[-1], f)
        probe_vol.commit()
        print(f"[{name}] done: tf cos train {tf_eval['cos_train']:+.4f} "
              f"held-out {tf_eval['cos_heldout']:+.4f}, |dW| {fro:.4f}, "
              f"B non-zero on {nz}/{len(zoo)} modules", flush=True)
        train_vol.commit()

    os.makedirs("/probe/sliders", exist_ok=True)
    np.savez(f"/probe/sliders/means_sliders.npz", M=SM,
             base=base_R.astype(np.float16), names=np.array(names),
             window="resp", n_prompts=len(prompts), max_new=MAX_NEW,
             model=BASE_MODEL, layer=LAYER)
    with open("/probe/sliders/generations_sliders.jsonl", "w") as f:
        f.write(json.dumps({"slider": "_base", "responses": base_texts[:4]}) + "\n")
        by = {g["slider"]: g for g in gens}
        for n in names:
            g = by.get(n)
            if g is None and os.path.exists(f"{per}/{n}.gen.json"):
                g = json.load(open(f"{per}/{n}.gen.json"))
            if g is not None:
                f.write(json.dumps(g) + "\n")
    probe_vol.commit()
    return {"sliders": results, "seconds": time.time() - t_start,
            "n_prompts": len(prompts), "n_train": n_train, "layers": NL, "dim": D,
            "job": {k: v for k, v in job.items() if k != "prompts"}}


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 60 * 8,
              secrets=[modal.Secret.from_name("hf-token")])
def behave(job: dict) -> dict:
    """The 24-prompt battery, greedy, 512 new tokens, thinking off, one prompt at
    a time -- oct_stage2.eval_personas' harness at steer_fix.py's token cap.

    Base, the sliders, and the ten matching stage-one adapters all go through the
    SAME harness in one run, so the comparison does not cross a generation
    setting.  judged_100.json's stage-one rows were generated at 200 tokens and
    are the external reference, not the within-run comparator.
    """
    import time

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = job["prompts"]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                device_map="cuda").eval()

    def gen(m):
        outs = []
        for p in prompts:
            text = tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            enc = tok(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = m.generate(**enc, do_sample=False, max_new_tokens=512,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs.append(tok.decode(o[0][enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
        return outs

    # Partial results are committed after every condition.  This job is the
    # longest in the experiment -- 24 conditions x 24 prompts x 512 greedy
    # tokens -- and steer_fix.py already learned that losing a whole run to a
    # timeout costs more than the resume logic does.
    part = f"/probe/sliders/behave_partial_{job.get('tag', 'all')}.json"
    os.makedirs("/probe/sliders", exist_ok=True)
    res, skipped = {}, []
    if os.path.exists(part):
        try:
            res = {k: v for k, v in json.load(open(part)).items()
                   if len(v) == len(prompts)}
            print(f"[resume] {len(res)} condition(s) already done: "
                  f"{sorted(res)}", flush=True)
        except Exception as e:
            print(f"[resume] ignoring unreadable partial: {e}", flush=True)

    def save():
        with open(part, "w") as f:
            json.dump(res, f)
        probe_vol.commit()

    t0 = time.time()
    if "base" not in res:
        res["base"] = gen(base)
        save()
    print(f"[base] {time.time() - t0:.0f}s", flush=True)
    for cond in job["conditions"]:
        path, nm = cond["path"], cond["name"]
        if nm in res:
            print(f"[{nm}] cached", flush=True)
            continue
        if not os.path.exists(f"{path}/adapter_model.safetensors"):
            print(f"[{nm}] MISSING {path} -- skipped", flush=True)
            skipped.append(nm)
            continue
        m = PeftModel.from_pretrained(base, path).eval()
        res[nm] = gen(m)
        m = m.unload()
        save()
        print(f"[{nm}] {path} done {time.time() - t0:.0f}s", flush=True)
    return {"prompts": prompts, "generations": res, "skipped": skipped,
            "seconds": time.time() - t0}


@app.local_entrypoint()
def main(stage: str = "train", sliders: str = "", steps: int = 250,
         lr: float = 1e-3, micro_batch: int = 8, out: str = "",
         which: str = "both"):
    import numpy as np

    spec = json.load(open(f"{HERE}/analysis/actspace_spec.json"))
    prompts = spec["prompts"]
    Z = np.load(f"{HERE}/analysis/slider_targets.npz", allow_pickle=True)
    all_names = [str(x) for x in Z["names"]]
    names = [s for s in sliders.split(",") if s] or all_names
    for n in names:
        assert n in all_names, f"unknown slider {n}"

    if stage == "train":
        job = {"prompts": prompts, "sliders": names, "layer": int(Z["layer"]),
               "max_new": 96, "gen_batch": 32, "n_train": 32, "micro_batch": micro_batch,
               "steps": steps, "lr": lr, "warmup": 20, "w_mag": 0.25,
               "eps_frac": 0.05, "clip": 1.0, "log_every": 10,
               "lora_r": 64, "lora_alpha": 128, "zoo_ref": "bold",
               "out_subdir": "sliders"}
        r = train.remote(job)
        p = out or f"{HERE}/analysis/slider_train.json"
    elif stage == "behave":
        # Two containers, not one: 24 conditions x 24 prompts x 512 greedy
        # tokens is three to four hours in series, and the two halves share
        # nothing but the base condition, which is greedy and therefore
        # identical in both.
        TM = json.load(open(f"{HERE}/analysis/slider_targets_meta.json"))
        traits10 = [n for n in TM["names"] if TM["meta"][n]["kind"] == "trait"]
        bat = json.load(open(f"{HERE}/phase10_runs/steer_spec2_7a.json"))["jobs"][0]["prompts"]
        jobs = {
            "sliders": [{"name": f"slider_{n}", "path": f"/trained/sliders/{n}"}
                        for n in names],
            "stage1": [{"name": f"stage1_{t}", "path": f"/adapters/{t}"}
                       for t in traits10],
        }
        sel = list(jobs) if which == "both" else [which]
        got = list(behave.map([{"prompts": bat, "conditions": jobs[k], "tag": k}
                               for k in sel]))
        merged = {"prompts": bat, "generations": {}, "skipped": [],
                  "seconds": {}}
        for k, r_ in zip(sel, got):
            merged["generations"].update(r_["generations"])
            merged["skipped"] += r_["skipped"]
            merged["seconds"][k] = r_["seconds"]
        r = merged
        p = out or f"{HERE}/phase10_runs/sliders_behave.json"
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    json.dump(r, open(p, "w"), indent=1, default=str)
    print(f"wrote {p}", flush=True)
