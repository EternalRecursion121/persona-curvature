#!/usr/bin/env python3
"""The 134 x 134 Fisher Gram: the adapter cloud measured in the model's own metric.

WHY
---
Every Gram in this project -- results/gram_sweep.npz, the PCA, the factor
analysis -- is an inner product in the weight-space FROBENIUS metric, which is
arbitrary from the model's point of view.  fisher.py measured what that costs:
the Fisher norms of 124 directions span a factor of 260, so two directions this
project calls the same length differ by that much in how far they move the
output distribution.  This file builds the Gram the model itself would use.

THE OBJECT
----------
For adapter i let dW_i = scale * B_i A_i and u_i = dW_i / ||dW_i||_F, and
perturb the weights along u_i in the same REF units fisher.py uses:

    theta(eps) = theta_0 + eps * REF * u_i

The score of one token position t under that one-parameter family is

    s_i(t, y) = d/d(eps) log p_eps(y | x_<t)   at eps = 0

and the Fisher inner product of two adapters is the covariance of their scores:

    F_ij = E_t E_{y ~ p_t} [ s_i(t, y) s_j(t, y) ]              EXPECTED form
    E_ij = E_t [ s_i(t, y_t) s_j(t, y_t) ]                      EMPIRICAL form

The diagonal of the EXPECTED form is exactly fisher.py's F: for i = j it is the
second derivative of KL(p_base || p_eps) in eps, which is what that run fitted
from six alphas.  That equality is the pipeline's own test and it is checked in
analyse_fisher_gram.py against analysis/fisher_norms.json.  The EMPIRICAL form
is the estimator the 2026-09-09 paper-reading page (experiment G1) specifies;
the two are DIFFERENT numbers and the size of the gap is one of the results.

HOW THE DERIVATIVE IS TAKEN
---------------------------
Central finite difference in eps, at h = 0.125 REF-units (the smallest alpha
fisher.py used), with the perturbation applied as a FORWARD HOOK rather than by
touching the weights:

    out  <-  out + (eps * REF * scale / ||dW_i||) * B_i (A_i x)

Three things follow, and all three matter.

* Nothing dense is ever built.  fisher.py materialised a 6.6 GiB delta buffer
  per direction; here the perturbation costs one (d_in x 64) and one
  (64 x d_out) product per module, so 134 directions cost what one cost there.
* The base weights are never modified, so there is no bf16 rounding of the
  increment to worry about -- the pathology fisher.py's docstring documents for
  steer_fix.py cannot arise here.
* Both signs of the difference share the base model bit for bit, so the fp32
  rounding of the forward pass is common-mode and cancels in lp(+h) - lp(-h).
  The forward runs in fp32 with TF32 OFF for the same reason fisher.py does:
  the signal is ~0.09 nats and TF32's 10-bit mantissa would not carry it.

The h^2 truncation error is measured, not assumed: ten single adapters are
redone at h/2 and the change in F_ii is reported.  The other numerical check is
that sum_y p_0(y) s_i(t, y) = 0 for every direction and token -- the derivative
of a normalised distribution integrates to zero -- which catches a wrong
log_softmax, a wrong h, or a hook that fired on the wrong module.

COST SHAPE
----------
One container, one model load.  24 sequences x (1 base + 134 x 2 + 10 x 2)
forward passes of ~210 tokens each.  The 134-adapter factor cache (32.4 GiB in
bf16, the same cache fisher.py builds) stays PINNED ON THE HOST and one
adapter's slice is streamed per (sequence, direction): 778 GiB of H2D over the
run, about 40 s, in exchange for 32 GiB of device memory that the expected-form
accumulator needs.  That accumulator is the memory driver: for one sequence it
holds sqrt(p_0) * ds/d(eps) for all 134 directions over the full 152k
vocabulary, 14.6 GiB in fp32.  fp16 is NOT safe there -- sqrt(p) * dlp
underflows on rare tokens -- so the run wants an 80 GB card.
"""
import json
import math
import os
import time

import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-80GB")
# must start with pc-qwen35 or the spend meter's kill switch cannot see it
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-fishergram")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
train_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol, "/trained": train_vol}


def _download_base_model():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL})
    .add_local_file(os.path.abspath(__file__), "/root/fisher_gram.py", copy=True)
    .run_function(_download_base_model)
)


def _strip(mod):                      # copied verbatim from fisher.py
    while mod.startswith("base_model.model."):
        mod = mod[len("base_model.model."):]
    return mod


def _open_set(tmpl, traits):          # copied verbatim from fisher.py
    import math as _math
    from safetensors import safe_open
    with open(f"{tmpl.format(t=traits[0])}/adapter_config.json") as f:
        ac = json.load(f)
    scale = ac["lora_alpha"] / (_math.sqrt(ac["r"]) if ac.get("use_rslora") else ac["r"])
    handles = {t: safe_open(f"{tmpl.format(t=t)}/adapter_model.safetensors",
                            framework="pt") for t in traits}
    keymap = {}
    for t in traits:
        keymap[t] = {_strip(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                     for k in handles[t].keys() if ".lora_A." in k}
    return handles, keymap, scale, ac["r"]


class _Perturb(object):
    """out <- out + c * (x @ A^T) @ B^T, with (A, B, c) swapped in per direction.

    c is None outside a perturbed pass, and the hook then returns the output
    untouched -- that is how the base pass and the zero control run through the
    same instrumented model as the perturbed ones.
    """

    def __init__(self, mod_name, state):
        self.m, self.state = mod_name, state

    def __call__(self, mod, inp, out):
        st = self.state
        if st["c"] is None:
            return out
        A, Bt = st["A"][self.m], st["Bt"][self.m]        # (r, d_in), (r, d_out), fp32
        h = inp[0] @ A.T                                 # (..., r)
        return out + st["c"] * (h @ Bt)


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              gpu=GPU_TYPE, volumes=VOLS, timeout=60 * 200,
              memory=46080, cpu=2.0)
def gram(S: dict) -> dict:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    MAX_MINUTES = float(os.environ.get("PC_MAX_MINUTES", "170"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    REF = S["ref"]
    MAXR = S["max_resp_tokens"]
    H = S["h"]
    H2_TRAITS = S["h_check_traits"]
    traits = S["traits"]
    tmpl = S["root_template"]
    snapshot_download(BASE_MODEL)

    def note(m):
        print(f"[{time.time()-t_start:8.1f}s] {m}", flush=True)

    # ---- fixed token sequences, built exactly as fisher.py builds them ----
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    seqs = []
    for p, t in zip(S["prompts"], S["texts"]):
        pre = tok(tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False),
                  return_tensors="pt")["input_ids"][0]
        rsp = tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0][:MAXR]
        seqs.append((torch.cat([pre, rsp]).to(dev), int(pre.shape[0]), int(rsp.shape[0])))
    n_scored = sum(s[2] for s in seqs)
    Lmax = max(s[2] for s in seqs)
    note(f"{len(seqs)} sequences, {n_scored} scored response tokens, max resp {Lmax}, "
         f"resp lens {[s[2] for s in seqs]}")

    # ---- base model, fp32 ------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32,
                                                 device_map="cuda")
    model.eval()
    model.requires_grad_(False)
    V = int(model.get_output_embeddings().weight.shape[0])
    note(f"model loaded fp32, vocab {V}, cuda alloc "
         f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ---- hooks on every LoRA-targeted linear -----------------------------
    h0, k0, scale, rank = _open_set(tmpl, [traits[0]])
    mods = sorted(k0[traits[0]])
    by = dict(model.named_modules())

    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        return None

    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")
    state = {"c": None, "A": {}, "Bt": {}}
    for m in mods:
        by[hit[m]].register_forward_hook(_Perturb(m, state))
    note(f"{len(mods)} modules hooked, lora r={rank} scale={scale}")

    # ---- host-side factor cache, pinned ----------------------------------
    # Ac[m] is (n*r, d_in) and Bt[m] is (n*r, d_out): BOTH laid out so that one
    # direction's slice is a contiguous block of rows, which is what makes the
    # per-(sequence, direction) H2D copy a single pinned DMA rather than a
    # host-side gather.  Storing B untransposed would make its slice a column
    # range and cost a CPU copy 3,216 times.
    handles, keymap, scale, rank = _open_set(tmpl, traits)
    for t in traits:
        if sorted(keymap[t]) != mods:
            raise RuntimeError(f"{t} targets a different module set")

    def host(x):
        x = x.contiguous()
        try:
            return x.pin_memory()
        except RuntimeError:
            return x

    Ac, Bt = {}, {}
    t0 = time.time()
    for m in mods:
        A = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_A.weight")
                       for t in traits], dim=0).to(torch.bfloat16)
        B = torch.cat([handles[t].get_tensor(keymap[t][m] + ".lora_B.weight")
                       for t in traits], dim=1).to(torch.bfloat16)
        Ac[m], Bt[m] = host(A), host(B.T.contiguous())
    del handles
    cache_gb = sum(Ac[m].numel() + Bt[m].numel() for m in mods) * 2 / 2**30
    note(f"factor cache built in {time.time()-t0:.0f}s, {cache_gb:.1f} GiB host bf16")

    # ---- per-adapter Frobenius norm of dW = scale * B A ------------------
    # ||B A||_F^2 = tr( (B^T B) (A A^T) ), both r x r: no dense product, and the
    # norm is computed from the SAME bf16 factors the hook applies, so the
    # direction that is measured is exactly the direction that is applied.
    t0 = time.time()
    nt = len(traits)
    acc = torch.zeros(nt, dtype=torch.float64, device=dev)
    for m in mods:
        Ag = Ac[m].to(dev, non_blocking=True).float().view(nt, rank, -1)
        Bg = Bt[m].to(dev, non_blocking=True).float().view(nt, rank, -1)
        AA = torch.bmm(Ag, Ag.transpose(1, 2))                 # (n, r, r)
        BB = torch.bmm(Bg, Bg.transpose(1, 2))                 # (n, r, r)
        acc += (AA * BB).sum(dim=(1, 2)).double()
        del Ag, Bg, AA, BB
    nrm = (acc.sqrt() * scale).cpu().numpy()
    note(f"adapter norms in {time.time()-t0:.0f}s, "
         f"min {nrm.min():.4f} median {np.median(nrm):.4f} max {nrm.max():.4f}")

    @torch.no_grad()
    def logprobs(i):
        ids, npre, nr = seqs[i]
        lg = model(ids.unsqueeze(0), use_cache=False).logits[0][npre - 1:npre - 1 + nr]
        return torch.log_softmax(lg.float(), dim=-1)

    @torch.no_grad()
    def load_dir(j):
        """Put direction j's factors on the device and return nothing."""
        for m in mods:
            state["A"][m] = Ac[m][j * rank:(j + 1) * rank].to(dev, non_blocking=True).float()
            state["Bt"][m] = Bt[m][j * rank:(j + 1) * rank].to(dev, non_blocking=True).float()

    n = len(traits)
    Fexp = torch.zeros(n, n, dtype=torch.float64, device=dev)
    Semp = torch.zeros(n, n_scored, dtype=torch.float64, device=dev)
    diag_h2 = {t: 0.0 for t in H2_TRAITS}
    zero_sum_max = 0.0
    base_nll = 0.0
    M = torch.zeros(n, Lmax * V, dtype=torch.float32, device=dev)
    M3 = M.view(n, Lmax, V)
    note(f"expected-form accumulator {M.numel()*4/2**30:.1f} GiB, "
         f"cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    part = "/oct/fishergram/partial.json"
    os.makedirs("/oct/fishergram", exist_ok=True)
    off = 0
    done_seq = 0
    for si in range(len(seqs)):
        if (time.time() - t_start) / 60 > MAX_MINUTES:
            note(f"WALL BUDGET {MAX_MINUTES} min reached after {done_seq} sequences")
            break
        ts = time.time()
        ids, npre, L = seqs[si]
        y = ids[npre:npre + L]
        state["c"] = None
        lp0 = logprobs(si)                                   # (L, V)
        base_nll += float(-lp0[torch.arange(L, device=dev), y].sum())
        p0 = lp0.exp()
        sq = p0.sqrt()
        del lp0
        if L < Lmax:
            M3[:, L:, :].zero_()      # the padded tail contributes 0 to M @ M^T
        for j in range(n):
            load_dir(j)
            state["c"] = H * REF * scale / float(nrm[j])
            lpp = logprobs(si)
            state["c"] = -H * REF * scale / float(nrm[j])
            lpm = logprobs(si)
            dlp = (lpp - lpm) / (2.0 * H)
            del lpp, lpm
            zs = float((p0 * dlp).sum(-1).abs().max())
            if zs > zero_sum_max:
                zero_sum_max = zs
            Semp[j, off:off + L] = dlp[torch.arange(L, device=dev), y].double()
            M3[j, :L] = sq * dlp
            if traits[j] in diag_h2:
                state["c"] = 0.5 * H * REF * scale / float(nrm[j])
                a = logprobs(si)
                state["c"] = -0.5 * H * REF * scale / float(nrm[j])
                b = logprobs(si)
                d2 = (a - b) / H
                diag_h2[traits[j]] += float((p0 * d2 * d2).sum())
                del a, b, d2
            del dlp
        state["c"] = None
        Fexp += (M @ M.T).double()   # M is the full contiguous buffer, tail zeroed
        off += L
        done_seq += 1
        del p0, sq
        note(f"sequence {si+1}/{len(seqs)} L={L} in {time.time()-ts:.0f}s "
             f"(max |sum_y p0 s| so far {zero_sum_max:.3e})")
        if done_seq % 4 == 0 or done_seq == len(seqs):
            with open(part, "w") as f:
                json.dump({"sequences_done": done_seq, "tokens_done": off,
                           "F_expected_unnormalised": Fexp.cpu().numpy().tolist()}, f)
            oct_vol.commit()

    T = off
    Fexp = (Fexp / T).cpu().numpy()
    Semp = Semp[:, :T]
    Femp = (Semp @ Semp.T / T).cpu().numpy()
    score_mean = Semp.mean(1).cpu().numpy()
    del M
    note(f"done: {done_seq} sequences, {T} tokens")

    return {
        "estimator_note":
            "F_expected[i,j] = mean_t sum_y p_0(t,y) s_i(t,y) s_j(t,y); "
            "F_empirical[i,j] = mean_t s_i(t,y_t) s_j(t,y_t); "
            "s_i(t,y) = d/d(eps) log p(y|x_<t) at theta_0 + eps*REF*dW_i/||dW_i||_F, "
            "by central finite difference at eps = +/- h.",
        "ref": REF, "h": H, "max_resp_tokens": MAXR, "n_scored_tokens": T,
        "n_sequences": done_seq, "resp_tokens": [s[2] for s in seqs],
        "prompts": S["prompts"], "text_source": S["text_source"],
        "root_template": tmpl, "lora_rank": rank, "lora_scale": scale,
        "targeted_modules": len(mods), "vocab_size": V,
        "names": traits, "adapter_frobenius_norms": nrm.tolist(),
        "base_mean_nll_per_token": base_nll / T,
        "zero_sum_check_max_abs": zero_sum_max,
        "zero_sum_check_meaning":
            "max over (direction, token) of |sum_y p_0(y) s(y)|, which is exactly "
            "zero in theory; compare it with sqrt(diag(F_expected)), the scale of s.",
        "score_mean_per_direction": score_mean.tolist(),
        "F_expected": Fexp.tolist(), "F_empirical": Femp.tolist(),
        "diag_expected_at_half_h": {k: v / T for k, v in diag_h2.items()},
        "wall_seconds": time.time() - t_start,
    }


@app.local_entrypoint()
def main(spec: str = "phase10_runs/fisher_spec.json",
         out: str = "phase10_runs/fisher_gram_results.json",
         h: float = 0.125, root: str = "/adapters/{t}",
         only: str = "", n_seqs: int = 0):
    """`only` restricts the direction set (comma-separated slugs) and `n_seqs`
    truncates the sequence list: together they are the smoke test, which is run
    on the ten single adapters fisher.py already measured so the diagonal can be
    checked against analysis/fisher_norms.json before the full run is paid for."""
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    SINGLES = ["agreeable", "rude", "organized", "careless", "anxious",
               "relaxed", "extraverted", "quiet", "imaginative", "unimaginative"]
    traits = S["traits134"]
    if only:
        want = [x.strip() for x in only.split(",") if x.strip()]
        missing = [x for x in want if x not in traits]
        if missing:
            raise SystemExit(f"unknown traits: {missing}")
        traits = [t for t in traits if t in set(want)]
    prompts, texts = S["prompts"], S["texts"]
    if n_seqs:
        prompts, texts = prompts[:n_seqs], texts[:n_seqs]
    job = {"ref": S["ref"], "max_resp_tokens": S["max_resp_tokens"],
           "prompts": prompts, "texts": texts,
           "text_source": S["text_source"], "traits": traits,
           "root_template": root, "h": h,
           "h_check_traits": [t for t in SINGLES if t in set(traits)]}
    print(f"{len(job['traits'])} directions x {len(job['prompts'])} sequences, "
          f"h={h}, root={root}", flush=True)
    R = gram.remote(job)
    with open(os.path.join(here, out), "w") as f:
        json.dump(R, f)
    print(f"wrote {out} ({R['n_sequences']} sequences, {R['n_scored_tokens']} tokens, "
          f"{R['wall_seconds']/60:.1f} min)", flush=True)
