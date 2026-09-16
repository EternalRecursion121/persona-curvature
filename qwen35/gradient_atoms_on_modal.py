#!/usr/bin/env python3
"""Per-document training gradients w.r.t. LoRA-B, EKFAC-projected -- experiment G2/G3.

Rosser, *Gradient Atoms* (arXiv:2603.14665v2), decomposes per-document training
gradients in an EKFAC-preconditioned eigenspace and reads atoms off the
decomposition.  This is that pipeline's extraction half, ported to the zoo.

WHAT IS DIFFERENTIATED, AND WHY B ONLY
--------------------------------------
A zoo adapter starts at B = 0, so dL/dA = B^T (dL/dW) = 0 and the whole first
update lives in B (the same fact align_score.py's identity rests on).  Per
module,

    dL/dB = scale * sum_t delta_t (A_0 x_t)^T ,    delta_t = dL/dy_t

which is d_out x 64.  Summed over the 248 modules that is 72.4M numbers per
document, so the projection is mandatory, exactly as the paper says.

PER-SAMPLE GRADIENTS WITHOUT A BACKWARD PER DOCUMENT
----------------------------------------------------
B enters the forward pass only as  y = W x + scale * B (A_0 x), so its gradient
for one sequence is a single outer product over that sequence's tokens.  A
forward hook stashes h = A_0 x (b, s, 64) and a tensor hook on the module's
output supplies delta (b, s, d_out); the einsum

    g = scale * einsum("bso,bsr->bor", delta, h)                 (b, d_out, 64)

is then the EXACT per-sequence gradient for the whole batch out of ONE backward
pass.  No B parameter is ever allocated in the extraction path.  `check`
allocates real zero-init B parameters for two batches and compares autograd's
B.grad against this einsum; the relative error is reported and is the proof.

The loss backwarded is sum_i log p(completion_i), so the hook returns
u_i = grad_B log p_i for every sequence independently.  Every document gradient
this project needs is a LINEAR COMBINATION of those, and the EKFAC projection,
the preconditioning and the sketch are all linear, so the combinations can be
formed downstream:

  zoo DPO pair, sigmoid(beta=0.1) + 0.1 SFT, at B = 0 where the sigmoid sits at
  exactly 1/2 and the OCT KL term (sq_approx_kl, coef 0.001) has zero gradient:

      g_pair = -0.05 (u_c - u_r) - 0.1 u_c / n_c

  the SHUFFLED null arm (chosen/rejected swapped on half the pairs) is the same
  expression with c and r exchanged on the swapped half -- no new gradients;
  the PERMUTED null arm is byte-identical rows under other trait names, so it is
  a relabelling of the real arm's gradients and needs no new gradients either.
  (Checked on the corpora themselves; see build_gradatoms_inputs.py.)

  an SFT document (School of Reward Hacks, G3):  g_doc = - u / n_tok.

trl 1.10's "sft" loss_type is a token mean pooled over the whole batch
(F.cross_entropy default reduction), so a per-document decomposition of it does
not exist exactly; each document's own token mean is used, which is the natural
per-example decomposition and is stated in the write-up.

EKFAC
-----
Per module the Fisher is approximated Kronecker-wise on the DOCUMENT gradients
themselves: C = E[g g^T] (d_out x d_out, output side) and R = E[g^T g] (64 x 64,
input side).  C is never formed: a randomised range finder accumulates
Y = sum_i g_i (g_i^T Omega) with Omega a fixed Gaussian (d_out x l), and a second
pass over the same fit subsample stores the l x 64 coordinates, from which the
rotation W inside that range and the eigenvalue correction

    Lambda_ab = E_i[ (u_a^T g_i v_b)^2 ]

are computed.  That last line is the EKFAC step: the eigenvalue is measured, not
taken as the product of the two factors' eigenvalues.  The k largest Lambda
entries per module are kept and each coordinate is divided by
sqrt(Lambda + eps), eps = damp * mean(selected Lambda) of that module.

This is the EMPIRICAL Fisher on realised tokens over document gradients, not the
sampled Fisher and not Grosse et al.'s token-level EKFAC.  Stated as such.

COHERENCE
---------
The paper's coherence is the mean pairwise cosine of the RAW gradients of an
atom's top-20 documents.  Raw here is 72.4M numbers per document, so a COUNT
SKETCH is stored alongside the projection: every one of the 72,450,048
coordinates gets a fixed random sign and a fixed random bucket out of D = 65,536,
and the sketch is the signed sum per bucket.  Inner products are unbiased and
the standard error on a cosine is about 1/sqrt(D) = 0.004.  A Kronecker sketch
(P_m g_m Q_m, p = q = 8, 15,872 numbers) was tried first and measured on the
smoke run at an error sd of 0.036 against exact cosines whose own spread was
0.047 -- unusable, because a per-module random projection of a nearly low-rank
per-module gradient has far fewer effective directions than it has entries.  The
count sketch's fidelity is measured the same way: 24 sequences' raw gradients are
held in fp16 and their exact cosines compared against the sketch's.

usage (see zoo-gradatoms.service):
    PC_APP_NAME=pc-qwen35-gradatoms modal run gradient_atoms_on_modal.py \\
        --stage zoo --items phase10_runs/gradatoms_items.json --out-tag zoo
"""
import json
import math
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-gradatoms")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
align_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=False)
rl_vol = modal.Volume.from_name("pc-qwen35-rl", create_if_missing=False)
out_vol = modal.Volume.from_name("pc-qwen35-gradatoms", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol, "/align": align_vol,
        "/rl": rl_vol, "/out": out_vol}


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
    .add_local_file(os.path.abspath(__file__), "/root/gradient_atoms_on_modal.py",
                    copy=True)
    .run_function(_dl)
)

cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("numpy<3", "scikit-learn==1.9.0", "scipy")
             .add_local_file(os.path.abspath(__file__),
                             "/root/gradient_atoms_on_modal.py", copy=True))

A0_ADAPTER = "/adapters/active"      # alphabetically first zoo trait, as align_score
SEED = 20260909


# ---------------------------------------------------------------------------
def _lora_a(path, device):
    """A_0, per-module d_out, the module list and the LoRA scale, from one adapter."""
    import torch
    from safetensors import safe_open
    with open(f"{path}/adapter_config.json") as f:
        ac = json.load(f)
    scale = (ac["lora_alpha"] / math.sqrt(ac["r"])) if ac.get("use_rslora") \
        else (ac["lora_alpha"] / ac["r"])
    h = safe_open(f"{path}/adapter_model.safetensors", framework="pt")
    mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                   for k in h.keys() if ".lora_A." in k})
    A0, dout = {}, {}
    for m in mods:
        A0[m] = h.get_tensor(f"base_model.model.{m}.lora_A.weight").float().to(device)
        dout[m] = int(h.get_slice(
            f"base_model.model.{m}.lora_B.weight").get_shape()[0])
    return A0, dout, mods, float(scale), int(ac["r"])


@app.function(image=image, gpu=os.environ.get("PC_GRADATOM_GPU", "A100-40GB"),
              volumes=VOLS, timeout=60 * 60 * 5, memory=65536,
              secrets=[modal.Secret.from_name("hf-token")])
def extract(job: dict) -> dict:
    """Per-sequence grad_B log p, EKFAC-projected and sketched.  Writes to /out."""
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    items = job["items"]
    mode = job.get("mode", "dpo")          # "dpo": 2 seqs per item; "sft": 1 per item
    bs_seq = int(job.get("batch_seqs", 32))
    maxlen = int(job.get("maxlen", 1024))
    L = int(job.get("l", 32))
    K = int(job.get("k", 28))
    DAMP = float(job.get("damp", 0.01))
    FIT_N = int(job.get("fit_n", 384))
    basis_from = job.get("basis_from")     # reuse a saved basis instead of fitting
    out_tag = job["out_tag"]
    n_raw_hold = int(job.get("n_raw_hold", 24))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A0, DOUT, MODS, SCALE, R = _lora_a(A0_ADAPTER, "cuda")
    print(f"[a0] {len(MODS)} modules, r={R}, scale={SCALE}, "
          f"sum d_out={sum(DOUT.values())}", flush=True)

    by = dict(model.named_modules())

    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        return None

    CTX = {"fn": None, "B": None}
    STORE = {}

    class Hook(object):
        def __init__(self, m):
            self.m = m

        def __call__(self, mod, inp, out):
            # h = A_0 x is DATA, not part of the differentiated path: computing it
            # inside the graph makes autograd retain a float32 copy of every
            # module's input (145 MB for d_in = 9216 at batch 32, times 248
            # modules -- the OOM this cost on the first smoke run).
            with torch.no_grad():
                h = torch.nn.functional.linear(inp[0].detach().float(),
                                               A0[self.m])             # (b, s, r)
            if not torch.is_grad_enabled() or not out.requires_grad:
                return None
            STORE[self.m] = h
            m = self.m
            out.register_hook(lambda gr, m=m: CTX["fn"](m, gr))
            if CTX["B"] is not None:
                # validation path only: a real zero-init B parameter in the graph.
                return out + SCALE * torch.einsum("bsr,or->bso", h,
                                                  CTX["B"][m]).to(out.dtype)
            return None

    for m in MODS:
        c = find(m)
        if c is None:
            raise RuntimeError(f"module not found in base model: {m}")
        by[c].register_forward_hook(Hook(m))
    print(f"[hooks] {len(MODS)} modules instrumented", flush=True)

    # ---------------- tokenisation ----------------
    def enc(item, response):
        pre = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        ml = int(item.get("maxlen", maxlen))
        if item.get("mode") == "sft":
            full = pre + response + tok.eos_token
            pi = tok(pre, add_special_tokens=False)["input_ids"]
            fi = tok(full, add_special_tokens=False)["input_ids"][:ml]
            lab = list(fi)
            for i in range(min(len(pi), len(fi))):
                lab[i] = -100
            return fi, lab
        a = tok(pre, add_special_tokens=False)["input_ids"]
        b = tok(response, add_special_tokens=False)["input_ids"]
        return (a + b)[:ml], ([-100] * len(a) + b)[:ml]

    def seqs_of(item):
        if mode == "sft":
            return [enc(item, item["chosen"]), enc(item, item["rejected"])]
        return [enc(item, item["chosen"]), enc(item, item["rejected"])]

    # every item contributes two sequences in both modes: a DPO pair's two halves,
    # or the reward-hacks row's hack and control completions as two documents.
    ENC = [seqs_of(x) for x in items]
    NTOK = np.array([[int(sum(1 for t in s[1] if t != -100)) for s in e] for e in ENC])
    print(f"[tok] {len(items)} items, {2 * len(items)} sequences; "
          f"tokens/seq median {np.median(NTOK):.0f} max {NTOK.max()}", flush=True)

    def backward_batch(pairs):
        """pairs: list of (ids, lab).  Backwards sum_i log p_i; hooks do the rest."""
        mx = max(len(x[0]) for x in pairs)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.tensor([x[0] + [pad] * (mx - len(x[0])) for x in pairs],
                           device="cuda")
        lab = torch.tensor([x[1] + [-100] * (mx - len(x[1])) for x in pairs],
                           device="cuda")
        att = torch.tensor([[1] * len(x[0]) + [0] * (mx - len(x[0])) for x in pairs],
                           device="cuda")
        emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        lg = model(inputs_embeds=emb, attention_mask=att).logits[:, :-1]
        tgt = lab[:, 1:]
        msk = (tgt != -100)
        acc = []
        for a in range(0, tgt.shape[1], 128):
            sl = lg[:, a:a + 128].float()
            g = sl.gather(-1, tgt[:, a:a + 128].clamp(min=0).unsqueeze(-1)).squeeze(-1)
            acc.append((g - torch.logsumexp(sl, -1)) * msk[:, a:a + 128])
        logp = torch.cat(acc, 1).sum(1)                     # (b,) per-sequence log p
        logp.sum().backward()
        STORE.clear()

    # ---------------- exactness check against autograd ----------------
    checks = []
    if job.get("check", True):
        Bp = {m: torch.zeros(DOUT[m], R, device="cuda", requires_grad=True)
              for m in MODS}
        CTX["B"] = Bp
        got = {}
        CTX["fn"] = lambda m, gr: got.__setitem__(
            m, SCALE * torch.einsum("bso,bsr->bor", gr.float(), STORE[m]).sum(0))
        flat = [s for e in ENC[:2] for s in e]
        backward_batch(flat)
        for m in MODS:
            a, b = Bp[m].grad, got[m]
            checks.append(float((a - b).norm() / a.norm().clamp(min=1e-30)))
        CTX["B"] = None
        for m in MODS:
            Bp[m].grad = None
        del Bp, got
        torch.cuda.empty_cache()
        print(f"[check] hook einsum vs autograd dL/dB over {len(MODS)} modules: "
              f"max rel err {max(checks):.3e}, median {np.median(checks):.3e}",
              flush=True)

    # ---------------- fixed random maps ----------------
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    OM = {m: torch.randn(DOUT[m], L, device="cuda", generator=gen) / math.sqrt(DOUT[m])
          for m in MODS}
    CSD = int(job.get("cs_dim", 65536))
    CSI, CSS = {}, {}
    for m in MODS:
        n = DOUT[m] * R
        CSI[m] = torch.randint(0, CSD, (n,), device="cuda", generator=gen,
                               dtype=torch.int64)
        CSS[m] = (torch.randint(0, 2, (n,), device="cuda", generator=gen,
                                dtype=torch.int8) * 2 - 1)

    # ---------------- EKFAC fit, or load a saved basis ----------------
    if basis_from:
        z = np.load(f"/out/{basis_from}", allow_pickle=False)
        USTAR = {m: torch.tensor(z[f"U_{m}"], device="cuda") for m in MODS}
        VEC = {m: torch.tensor(z[f"V_{m}"], device="cuda") for m in MODS}
        IDX = {m: torch.tensor(z[f"I_{m}"], device="cuda", dtype=torch.long)
               for m in MODS}
        SCL = {m: torch.tensor(z[f"S_{m}"], device="cuda") for m in MODS}
        LAM = {m: z[f"L_{m}"] for m in MODS}
        L = USTAR[MODS[0]].shape[1]
        K = IDX[MODS[0]].shape[0]
        print(f"[basis] loaded {basis_from}: l={L} k={K}", flush=True)
    else:
        rng = np.random.default_rng(SEED)
        fit = sorted(rng.choice(len(items), size=min(FIT_N, len(items)),
                                replace=False).tolist())
        print(f"[fit] EKFAC on {len(fit)} documents", flush=True)

        def doc_weights(i):
            """(w_chosen, w_rejected) turning per-sequence u into the doc gradient."""
            nc, nr = float(NTOK[i][0]), float(NTOK[i][1])
            if mode == "sft":
                return (-1.0 / max(nc, 1.0), 0.0)
            return (-0.05 - 0.1 / max(nc, 1.0), 0.05)

        Y = {m: torch.zeros(DOUT[m], L, device="cuda") for m in MODS}
        RR = {m: torch.zeros(R, R, device="cuda") for m in MODS}
        WT = {"w": None}

        def fn_pass_a(m, gr):
            g = SCALE * torch.einsum("bso,bsr->bor", gr.float(), STORE[m])
            gp = WT["w"][:, 0, None, None] * g[0::2] + WT["w"][:, 1, None, None] * g[1::2]
            T = torch.einsum("bor,ol->brl", gp, OM[m])
            Y[m] += torch.einsum("bor,brl->ol", gp, T)
            RR[m] += torch.einsum("bor,bos->rs", gp, gp)

        CTX["fn"] = fn_pass_a
        # The fit passes accumulate into Y / RR / CO, so a mid-backward OOM would
        # leave a partial contribution behind and a retry would double-count.
        # They therefore run at a fixed conservative batch and are not retried;
        # only the extraction pass, which writes into indexed slots and can be
        # split cleanly, has the halving fallback.
        bp = max(1, int(job.get("fit_batch_pairs", 4)))
        for a in range(0, len(fit), bp):
            ch = fit[a:a + bp]
            WT["w"] = torch.tensor([doc_weights(i) for i in ch], device="cuda")
            backward_batch([s for i in ch for s in ENC[i]])
        print(f"[fit] pass A done {time.time() - t_start:.0f}s", flush=True)

        U0, VEC = {}, {}
        for m in MODS:
            q, _ = torch.linalg.qr(Y[m])
            U0[m] = q[:, :L].contiguous()
            ev, V = torch.linalg.eigh(RR[m].double())
            VEC[m] = V.flip(-1).float().contiguous()          # descending
        del Y, RR, OM
        torch.cuda.empty_cache()

        CO = {m: [] for m in MODS}

        def fn_pass_b(m, gr):
            g = SCALE * torch.einsum("bso,bsr->bor", gr.float(), STORE[m])
            gp = WT["w"][:, 0, None, None] * g[0::2] + WT["w"][:, 1, None, None] * g[1::2]
            C = torch.einsum("ol,bor->blr", U0[m], gp)
            CO[m].append(torch.einsum("blr,rs->bls", C, VEC[m]).cpu())

        CTX["fn"] = fn_pass_b
        for a in range(0, len(fit), bp):
            ch = fit[a:a + bp]
            WT["w"] = torch.tensor([doc_weights(i) for i in ch], device="cuda")
            backward_batch([s for i in ch for s in ENC[i]])
        print(f"[fit] pass B done {time.time() - t_start:.0f}s", flush=True)

        USTAR, IDX, SCL, LAM = {}, {}, {}, {}
        for m in MODS:
            C = torch.cat(CO[m], 0).cuda()                       # (n, l, r)
            M = torch.einsum("blr,bmr->lm", C, C).double()
            ev, W = torch.linalg.eigh(M)
            W = W.flip(-1).float().contiguous()                  # (l, l) descending
            D = torch.einsum("lm,blr->bmr", W, C)                # rotate
            lam = (D * D).mean(0)                                # (l, r) EKFAC eigenvalues
            flat = lam.reshape(-1)
            idx = torch.argsort(flat, descending=True)[:K]
            eps = DAMP * float(flat[idx].mean())
            USTAR[m] = (U0[m] @ W).contiguous()
            IDX[m] = idx.contiguous()
            SCL[m] = (1.0 / torch.sqrt(flat[idx] + eps)).contiguous()
            LAM[m] = lam.cpu().numpy()
        del CO, U0
        torch.cuda.empty_cache()
        np.savez(f"/out/{out_tag}_basis.npz",
                 **{f"U_{m}": USTAR[m].cpu().numpy() for m in MODS},
                 **{f"V_{m}": VEC[m].cpu().numpy() for m in MODS},
                 **{f"I_{m}": IDX[m].cpu().numpy() for m in MODS},
                 **{f"S_{m}": SCL[m].cpu().numpy() for m in MODS},
                 **{f"L_{m}": LAM[m] for m in MODS},
                 mods=np.array(MODS), meta=np.array([L, K, R, SCALE]))
        out_vol.commit()
        print(f"[fit] basis written, {time.time() - t_start:.0f}s", flush=True)

    # ---------------- extraction pass ----------------
    NS = 2 * len(items)
    D = len(MODS) * K
    DS = CSD
    COORD = np.zeros((NS, D), dtype=np.float32)
    SKETCH = np.zeros((NS, DS), dtype=np.float32)
    GNORM = np.zeros(NS, dtype=np.float64)
    RAWH = {"n": min(n_raw_hold, NS), "buf": None, "off": 0}
    if RAWH["n"]:
        RAWH["buf"] = torch.zeros(RAWH["n"], sum(DOUT.values()) * R,
                                  dtype=torch.float16)
        offs, o = {}, 0
        for m in MODS:
            offs[m] = o
            o += DOUT[m] * R
        RAWH["offs"] = offs
    POS = {"a": 0, "n": 0}
    MI = {m: i for i, m in enumerate(MODS)}

    def fn_pass_c(m, gr):
        g = SCALE * torch.einsum("bso,bsr->bor", gr.float(), STORE[m])
        b = g.shape[0]
        Z = torch.einsum("ol,bor->blr", USTAR[m], g)
        Z = torch.einsum("blr,rs->bls", Z, VEC[m]).reshape(b, -1)
        c = Z[:, IDX[m]] * SCL[m]
        i = MI[m]
        COORD[POS["a"]:POS["a"] + b, i * K:(i + 1) * K] = c.cpu().numpy()
        CSBUF["b"].index_add_(1, CSI[m], g.reshape(b, -1) * CSS[m])
        GNORM[POS["a"]:POS["a"] + b] += torch.einsum("bor,bor->b", g, g).cpu().numpy()
        if POS["a"] < RAWH["n"]:
            j = min(b, RAWH["n"] - POS["a"])
            o = RAWH["offs"][m]
            RAWH["buf"][POS["a"]:POS["a"] + j, o:o + DOUT[m] * R] = \
                g[:j].reshape(j, -1).half().cpu()

    CTX["fn"] = fn_pass_c
    flat_seqs = [s for e in ENC for s in e]
    t0 = time.time()

    def guarded(seqs, off):
        """One extraction batch, halving itself on OOM.

        Activation memory here is dominated by the linear-attention layers and
        scales with batch x tokens; the corpus is short (median 141 tokens) but
        not uniform, so a fixed batch that fits the median can still exceed the
        card on a long batch.  Every write is into an indexed slot, so splitting
        and retrying is exact.
        """
        n = len(seqs)
        try:
            CSBUF["b"] = torch.zeros(n, CSD, device="cuda")
            POS["a"] = off
            backward_batch(seqs)
            SKETCH[off:off + n] = CSBUF["b"].cpu().numpy()
            CSBUF["b"] = None
        except torch.OutOfMemoryError:
            CSBUF["b"] = None
            STORE.clear()
            torch.cuda.empty_cache()
            if n <= 1:
                raise
            GNORM[off:off + n] = 0.0
            h = n // 2
            print(f"  [oom] splitting a batch of {n} at offset {off}", flush=True)
            guarded(seqs[:h], off)
            guarded(seqs[h:], off + h)

    CSBUF = {"b": None}
    for a in range(0, len(flat_seqs), bs_seq):
        ch = flat_seqs[a:a + bs_seq]
        guarded(ch, a)
        if (a // bs_seq) % 25 == 0:
            el = time.time() - t0
            print(f"  {a + len(ch)}/{len(flat_seqs)} seq  {el:.0f}s  "
                  f"({el / max(a + len(ch), 1):.3f} s/seq)", flush=True)
    GNORM = np.sqrt(GNORM)

    # sketch fidelity against exact cosines on the held raw gradients
    sk_check = {}
    if RAWH["n"] >= 4:
        Rw = RAWH["buf"].float()
        Gx = (Rw @ Rw.T).numpy()
        d = np.sqrt(np.diag(Gx))
        Cx = Gx / np.outer(d, d)
        Sk = SKETCH[:RAWH["n"]].astype(np.float64)
        Gs = Sk @ Sk.T
        ds = np.sqrt(np.diag(Gs))
        Cs = Gs / np.outer(ds, ds)
        iu = np.triu_indices(RAWH["n"], 1)
        err = np.abs(Cx[iu] - Cs[iu])
        sk_check = {"n_seq": int(RAWH["n"]), "n_pairs": int(len(iu[0])),
                    "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
                    "pearson": float(np.corrcoef(Cx[iu], Cs[iu])[0, 1]),
                    "exact_cos": Cx[iu].tolist(), "sketch_cos": Cs[iu].tolist()}
        print(f"[sketch] {sk_check['n_pairs']} exact vs sketch cosines: "
              f"max abs err {sk_check['max_abs_err']:.4f}, "
              f"r {sk_check['pearson']:.5f}", flush=True)
        del Rw, RAWH["buf"]

    np.savez(f"/out/{out_tag}_grads.npz", coord=COORD, sketch=SKETCH, gnorm=GNORM,
             ntok=NTOK, ids=np.array([x["id"] for x in items]),
             mods=np.array(MODS), meta=np.array([L, K, R, SCALE, CSD]))
    out_vol.commit()
    dt = time.time() - t_start
    print(f"[done] {NS} sequences, coord {COORD.shape}, sketch {SKETCH.shape}, "
          f"{dt:.0f}s", flush=True)
    return {"n_items": len(items), "n_seq": int(NS), "dim": int(D),
            "sketch_dim": int(DS), "l": L, "k": K, "cs_dim": CSD, "damp": DAMP,
            "scale": SCALE, "r": R, "n_modules": len(MODS),
            "sum_d_out": int(sum(DOUT.values())),
            "raw_dim_per_doc": int(sum(DOUT.values()) * R),
            "autograd_check": {"max_rel_err": (max(checks) if checks else None),
                               "median_rel_err": (float(np.median(checks))
                                                  if checks else None),
                               "n_modules": len(checks)},
            "sketch_check": sk_check,
            "seconds": dt, "out": f"{out_tag}_grads.npz"}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
ARMS_DOC = """The three arms, all from the SAME per-completion gradients.

u_c and u_r are the projected grad_B log p of a pair's chosen and rejected
completions, and every arm is a linear combination of them, so the null arms
cost no GPU (build_gradatoms_inputs.py asserts the corpus facts that make this
exact):

  real      g = (-0.05 - 0.1/n_c) u_c + 0.05 u_r
  shuffled  the same on intact pairs; on swapped pairs c and r exchange roles
  permuted  the real arm's gradients for trait Y carrying trait X's name
"""


@app.function(image=cpu_image, volumes=VOLS, cpu=8.0, memory=65536,
              timeout=60 * 180)
def atoms(job: dict) -> dict:
    """Sparse dictionary learning, coherence and label purity, on the projection.

    Follows the paper: unit-normalise every projected gradient so an atom encodes
    direction and not magnitude, then MiniBatchDictionaryLearning with K atoms and
    sparsity penalty alpha.  Coherence is the mean pairwise cosine of the RAW
    (unprojected) gradients of an atom's top-20 activating documents, read off the
    Kronecker sketch.

    "Top-20 activating" is taken as the 20 largest POSITIVE codes, which is the
    plain reading and the one that matters: a sign-blind |code| ranking would make
    the shuffled arm coherent by construction, because swapping a pair negates
    its gradient.  The opposite pole is reported alongside for every atom.
    """
    import time

    import numpy as np
    from sklearn.decomposition import MiniBatchDictionaryLearning, sparse_encode
    from sklearn.metrics import adjusted_rand_score

    t0 = time.time()
    tag = job.get("tag", "zoo")
    z = np.load(f"/out/{tag}_grads.npz", allow_pickle=False)
    CO = z["coord"].astype(np.float32)
    SK = z["sketch"].astype(np.float32)
    NT = z["ntok"]
    ids = [str(x) for x in z["ids"]]
    N = len(ids)
    print(f"[atoms] {N} items, coord {CO.shape}, sketch {SK.shape}", flush=True)

    lab = {r["id"]: r for r in job["labels"]}
    # factor and keying BY TRAIT NAME.  The permuted arm's rows are the source
    # trait's rows, so if the assigned name did not bring its own factor and
    # keying with it, purity_factor there would be the real arm's number
    # restricted to 100 traits -- an identity, not a control.  Trait purity is
    # invariant under a bijection either way, which is why the informative
    # permuted-arm statistic is factor and keying under the ASSIGNED name.
    tf = {}
    for r in job["labels"]:
        tf[r["trait"]] = (r.get("factor"), r.get("keyed"))
    pmap = job.get("permuted_map") or {}
    swap = job.get("shuffled_swapped") or {}
    prompts = job.get("prompts") or {}
    fa_dom = job.get("fa_dominant") or {}

    def build(arm):
        """One document = a signed combination of stored per-completion gradients.

        real      w_c = -0.05 - 0.1/n_c on the chosen half, +0.05 on the rejected
        shuffled  the same, with the two halves exchanged on swapped pairs
        permuted  the real arm's documents for trait Y, carrying trait X's name
        sorh      each completion is its own SFT document, gradient -u/n_tok
        """
        docs, meta = [], []

        def pair(i, flip, m):
            a, b = (1, 0) if flip else (0, 1)
            n = float(NT[i][a])
            docs.append([(2 * i + a, -0.05 - 0.1 / max(n, 1.0)), (2 * i + b, 0.05)])
            meta.append(m)

        if arm == "sorh":
            for i, k in enumerate(ids):
                for slot, a in (("hack", 0), ("control", 1)):
                    n = float(NT[i][a])
                    docs.append([(2 * i + a, -1.0 / max(n, 1.0))])
                    meta.append({"id": f"{k}:{slot}", "assigned_trait": slot,
                                 "source_trait": slot, "factor": slot,
                                 "keyed": slot, "trait": slot})
        elif arm == "permuted":
            byt = {}
            for i, k in enumerate(ids):
                byt.setdefault(k.split("#")[0], []).append(i)
            for assigned, source in sorted(pmap.items()):
                for i in byt.get(source, []):
                    m = dict(lab[ids[i]])
                    m["assigned_trait"] = assigned
                    m["source_trait"] = source
                    m["source_factor"], m["source_keyed"] = m["factor"], m["keyed"]
                    if assigned in tf:
                        m["factor"], m["keyed"] = tf[assigned]
                    pair(i, False, m)
        else:
            for i, k in enumerate(ids):
                t = k.split("#")[0]
                fl = False
                if arm == "shuffled":
                    key = f"{t}|{prompts.get(k, '')}"
                    if key not in swap:
                        continue
                    fl = bool(int(swap[key]))
                m = dict(lab[k])
                m["assigned_trait"] = t
                m["source_trait"] = t
                pair(i, fl, m)

        X = np.zeros((len(docs), CO.shape[1]), dtype=np.float32)
        S = np.zeros((len(docs), SK.shape[1]), dtype=np.float32)
        for j, terms in enumerate(docs):
            for si, w in terms:
                X[j] += w * CO[si]
                S[j] += w * SK[si]
        pre = np.linalg.norm(X, axis=1)
        X /= np.maximum(pre, 1e-30)[:, None]
        S /= np.maximum(np.linalg.norm(S, axis=1), 1e-30)[:, None]
        return X, S, meta, pre

    def coherence(code, S, n_top=20):
        out = []
        for j in range(code.shape[1]):
            c = code[:, j]
            row = {"atom": j}
            for pole, v in (("pos", c), ("neg", -c)):
                o = np.argsort(-v)[:n_top]
                o = o[v[o] > 0]
                if len(o) < 2:
                    row[pole] = {"n": int(len(o)), "coherence": None,
                                 "top": o.tolist()}
                    continue
                G = S[o] @ S[o].T
                iu = np.triu_indices(len(o), 1)
                row[pole] = {"n": int(len(o)),
                             "coherence": float(G[iu].mean()),
                             "top": o.tolist()}
            row["n_active"] = int((np.abs(c) > 0).sum())
            out.append(row)
        return out

    def purity(rows, meta, key, n_null=1000, seed=7):
        import collections
        labs = np.array([m.get(key) for m in meta])
        vals = []
        for r in rows:
            o = r["pos"]["top"]
            if len(o) < 2:
                continue
            c = collections.Counter(labs[o])
            vals.append(c.most_common(1)[0][1] / len(o))
        if not vals:
            return None
        obs = float(np.mean(vals))
        rng = np.random.default_rng(seed)
        nul = []
        for _ in range(n_null):
            p = rng.permutation(labs)
            v = []
            for r in rows:
                o = r["pos"]["top"]
                if len(o) < 2:
                    continue
                c = collections.Counter(p[o])
                v.append(c.most_common(1)[0][1] / len(o))
            nul.append(np.mean(v))
        nul = np.array(nul)
        return {"observed": obs, "null_mean": float(nul.mean()),
                "null_sd": float(nul.std()), "n_null": n_null,
                "z": float((obs - nul.mean()) / max(nul.std(), 1e-12)),
                "p_ge": float((nul >= obs).mean()),
                "n_atoms_scored": len(vals)}

    DATA = {}
    for arm in job.get("arms", ["real", "shuffled", "permuted"]):
        X, S, meta, pre = build(arm)
        DATA[arm] = (X, S, meta, pre)
        print(f"[atoms] arm {arm}: {X.shape[0]} documents", flush=True)

    # per-trait coherence: G4's original statistic -- the mean pairwise cosine of
    # the raw per-pair gradients of one trait's own 40 pairs, against the
    # cross-trait baseline computed on the same documents.
    tc = {}
    if "real" in DATA:
        import collections as _c
        _, S, meta, _pre = DATA["real"]
        byt = _c.defaultdict(list)
        for i, m in enumerate(meta):
            byt[m["assigned_trait"]].append(i)
        rng0 = np.random.default_rng(3)
        allc = []
        for t, ix in sorted(byt.items()):
            G = S[ix] @ S[ix].T
            iu = np.triu_indices(len(ix), 1)
            tc[t] = float(G[iu].mean())
        for _ in range(20000):
            a, b = rng0.integers(0, S.shape[0], 2)
            if meta[a]["assigned_trait"] != meta[b]["assigned_trait"]:
                allc.append(float(S[a] @ S[b]))
        tc["_cross_trait_mean"] = float(np.mean(allc))
        tc["_cross_trait_sd"] = float(np.std(allc))
        tc["_within_trait_mean"] = float(np.mean(
            [v for k, v in tc.items() if not k.startswith("_")]))
        print(f"[atoms] per-trait coherence: within {tc['_within_trait_mean']:.4f}, "
              f"cross {tc['_cross_trait_mean']:.4f}", flush=True)

    results = {"arms": {}, "configs": [], "per_trait_coherence": tc}
    saved = {}
    for cfg in job["configs"]:
        arm, K, al = cfg["arm"], int(cfg["K"]), float(cfg["alpha"])
        X, S, meta, pre = DATA[arm]
        d = MiniBatchDictionaryLearning(
            n_components=K, alpha=al, batch_size=256, max_iter=int(cfg.get("iters", 400)),
            transform_algorithm="lasso_lars", transform_alpha=al,
            fit_algorithm="lars", random_state=0, n_jobs=8)
        d.fit(X)
        D = d.components_.astype(np.float32)
        code = sparse_encode(X, D, algorithm="lasso_lars", alpha=al, n_jobs=8)
        rows = coherence(code, S)
        co = np.array([r["pos"]["coherence"] for r in rows
                       if r["pos"]["coherence"] is not None])
        nact = np.array([r["n_active"] for r in rows])
        rec = {"arm": arm, "K": K, "alpha": al,
               "n_docs": int(X.shape[0]),
               "n_atoms_with_top20": int(sum(1 for r in rows
                                             if r["pos"]["n"] >= 20)),
               "median_docs_per_atom": float(np.median(nact)),
               "mean_docs_per_atom": float(nact.mean()),
               "frac_zero_code": float((np.abs(code) < 1e-12).all(1).mean()),
               "coherence_gt_0_5": int((co > 0.5).sum()),
               "coherence_gt_0_1": int((co > 0.1).sum()),
               "coherence_max": (float(co.max()) if len(co) else None),
               "coherence_mean": (float(co.mean()) if len(co) else None),
               "coherence_median": (float(np.median(co)) if len(co) else None),
               "n_scored": int(len(co))}
        if cfg.get("full"):
            for key in cfg.get("purity_keys",
                               ["assigned_trait", "factor", "keyed", "fa_dominant"]):
                if key == "fa_dominant":
                    for m in meta:
                        m["fa_dominant"] = fa_dom.get(m["assigned_trait"])
                rec[f"purity_{key}"] = purity(rows, meta, key)
            if arm == "permuted":
                for key in ("source_trait", "source_factor", "source_keyed"):
                    rec[f"purity_{key}"] = purity(rows, meta, key)
            asg = np.where(np.abs(code).max(1) > 0, np.abs(code).argmax(1), -1)
            keep = asg >= 0
            rec["ari_trait"] = float(adjusted_rand_score(
                np.array([m["assigned_trait"] for m in meta])[keep], asg[keep]))
            rec["ari_factor"] = float(adjusted_rand_score(
                np.array([str(m["factor"]) for m in meta])[keep], asg[keep]))
            rec["n_assigned"] = int(keep.sum())
            rec["atom_rows"] = [
                {"atom": r["atom"], "coherence": r["pos"]["coherence"],
                 "coherence_neg": r["neg"]["coherence"],
                 "n_active": r["n_active"],
                 "top_ids": [meta[i]["id"] for i in r["pos"]["top"]],
                 "top_traits": [meta[i]["assigned_trait"] for i in r["pos"]["top"]],
                 "top_source_traits": [meta[i]["source_trait"]
                                       for i in r["pos"]["top"]],
                 "top_factors": [meta[i]["factor"] for i in r["pos"]["top"]],
                 "top_keyed": [meta[i]["keyed"] for i in r["pos"]["top"]],
                 "top_neg_ids": [meta[i]["id"] for i in r["neg"]["top"]],
                 "top_neg_traits": [meta[i]["assigned_trait"]
                                    for i in r["neg"]["top"]],
                 "top_neg_factors": [meta[i]["factor"] for i in r["neg"]["top"]],
                 "top_neg_keyed": [meta[i]["keyed"] for i in r["neg"]["top"]]}
                for r in rows]
            saved[f"{arm}_{K}_{al}"] = (D, code, X, meta)
        results["configs"].append(rec)
        print(f"[atoms] {arm} K={K} alpha={al}: >0.5 {rec['coherence_gt_0_5']}, "
              f">0.1 {rec['coherence_gt_0_1']}, max {rec['coherence_max']}, "
              f"{time.time() - t0:.0f}s", flush=True)

    prim = job.get("primary", "real_500_0.1")
    if prim in saved:
        D, code, X, meta = saved[prim]
        np.savez(f"/out/{tag}_atoms.npz", atoms=D, code=code.astype(np.float32),
                 ids=np.array([m["id"] for m in meta]))
        out_vol.commit()
        results["primary_saved"] = f"{tag}_atoms.npz"
    for kk, (D, code, X, meta) in saved.items():
        np.savez(f"/out/{tag}_dict_{kk}.npz", atoms=D,
                 code=code.astype(np.float32),
                 X=X.astype(np.float16),
                 ids=np.array([m["id"] for m in meta]))
    out_vol.commit()
    results["seconds"] = time.time() - t0
    print(f"[atoms] done {results['seconds']:.0f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
def _sources(spec):
    s = {f"/adapters/{t}": c for t, c in spec.get("coef", {}).items()}
    for p, c in spec.get("src", []):
        s[p] = s.get(p, 0.0) + c
    return s


@app.function(image=image, volumes=VOLS, cpu=8.0, memory=65536, timeout=60 * 180)
def weightspace(job: dict) -> dict:
    """Atom space <-> weight space, exactly, without forming a d_out x d_in matrix.

    A named direction dW* = sum_i c_i s_i B_i A_i has the B-frame representative
    B_U = dW* A_0^T (align_score.py's construction, exact through A_i A_0^T even
    though A drifted), which is d_out x 64 -- the same shape as a per-document
    gradient -- so the identical EKFAC map applies to it.

    An atom is a sparse vector on the eigengrid, so its weight-space direction is
    G = U grid V^T in the B frame and G A_0 in weight space.  Because U and V are
    orthonormal, every inner product needed collapses to the grid:

        <G A_0, dW*>      = tr(G^T B_U)        = <grid, U^T B_U V>
        ||G A_0||^2       = tr(G^T G A_0 A_0^T) = tr(grid^T grid V^T K0 V)

    with K0 = A_0 A_0^T.  So only the l x 64 projection U^T B_U V of each target
    is ever needed, which is why this runs on CPU in minutes: per adapter and
    module it costs one (l x d_out)(d_out x 64) product.  ||dW*|| comes from the
    project's own exact Gram, results/gram_sweep.npz, for zoo-coefficient
    targets, and from tr(A A^T B^T B) for single non-zoo adapters.
    """
    import time

    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    t0 = time.time()
    specs = job["targets"]
    z = np.load(f"/out/{job['basis']}", allow_pickle=False)
    A = np.load(f"/out/{job['atoms']}", allow_pickle=False)
    atoms = A["atoms"].astype(np.float32)                 # (K, n_mod * k)
    MODS = [str(x) for x in z["mods"]]
    Lm, Km, R, _sc = z["meta"]
    Lm, Km, R = int(Lm), int(Km), int(R)
    NK = atoms.shape[0]
    T = len(specs)
    gram = np.array(job.get("zoo_gram") or [[]], dtype=np.float64)
    gram_names = [str(x) for x in (job.get("zoo_gram_names") or [])]
    gi = {n: i for i, n in enumerate(gram_names)}
    print(f"[ws] {T} targets, {NK} atoms, {len(MODS)} modules, l={Lm} k={Km}",
          flush=True)

    srcs = [_sources(s) for s in specs]
    paths = sorted({p for d in srcs for p in d})
    sc = {}
    for p in paths:
        with open(f"{p}/adapter_config.json") as f:
            ac = json.load(f)
        sc[p] = (ac["lora_alpha"] / math.sqrt(ac["r"])) if ac.get("use_rslora") \
            else (ac["lora_alpha"] / ac["r"])

    U = {m: torch.tensor(z[f"U_{m}"]) for m in MODS}
    V = {m: torch.tensor(z[f"V_{m}"]) for m in MODS}
    IDX = {m: torch.tensor(z[f"I_{m}"], dtype=torch.long) for m in MODS}
    SCL = {m: torch.tensor(z[f"S_{m}"]) for m in MODS}
    h0 = safe_open(f"{A0_ADAPTER}/adapter_model.safetensors", framework="pt")
    A0 = {m: h0.get_tensor(f"base_model.model.{m}.lora_A.weight").float()
          for m in MODS}
    KV = {m: V[m].T @ (A0[m] @ A0[m].T) @ V[m] for m in MODS}

    # one pass over the adapters: per module keep only U^T B (A A_0^T), l x 64
    W = {p: {} for p in paths}
    selfn2 = {p: 0.0 for p in paths}
    for pi, p in enumerate(paths):
        h = safe_open(f"{p}/adapter_model.safetensors", framework="pt")
        for m in MODS:
            a = h.get_tensor(f"base_model.model.{m}.lora_A.weight").float()
            b = h.get_tensor(f"base_model.model.{m}.lora_B.weight").float()
            W[p][m] = (U[m].T @ b) @ (a @ A0[m].T) * sc[p]        # (l, r)
            selfn2[p] += float(torch.trace((a @ a.T) @ (b.T @ b))) * sc[p] ** 2
        if pi % 20 == 0:
            print(f"  adapter {pi}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    print(f"[ws] adapters read {time.time() - t0:.0f}s", flush=True)

    proj = np.zeros((T, len(MODS) * Km), dtype=np.float64)
    cross = np.zeros((NK, T), dtype=np.float64)
    anorm2 = np.zeros(NK)
    for mi, m in enumerate(MODS):
        Zt = torch.zeros(T, Lm, R)
        for j, d in enumerate(srcs):
            for p, c in d.items():
                if c:
                    Zt[j] += W[p][m] * c
        Zt = torch.einsum("tlr,rs->tls", Zt, V[m])                # U^T B_U V
        F = Zt.reshape(T, -1)
        proj[:, mi * Km:(mi + 1) * Km] = (F[:, IDX[m]] * SCL[m]).numpy()
        gr = torch.zeros(NK, Lm * R)
        gr[:, IDX[m]] = torch.tensor(atoms[:, mi * Km:(mi + 1) * Km]) / SCL[m]
        cross += (gr @ F.T).numpy()
        g3 = gr.reshape(NK, Lm, R)
        anorm2 += torch.einsum("klr,kls,rs->k", g3, g3, KV[m]).numpy()

    # exact norms.  Zoo-coefficient targets take the project's own exact Gram
    # (results/gram_sweep.npz).  Targets built from adapters outside the zoo are
    # few, so their own small cross-Gram is computed here, exactly, in a second
    # pass over just those files: <s_p B_p A_p, s_q B_q A_q>
    #   = s_p s_q tr(A_p A_q^T B_q^T B_p), per module, never forming d_out x d_in.
    ext = sorted(p for p in paths if not p.startswith("/adapters/"))
    EG = np.zeros((len(ext), len(ext)))
    if ext:
        HH = [safe_open(f"{p}/adapter_model.safetensors", framework="pt")
              for p in ext]
        for m in MODS:
            AB = [(h.get_tensor(f"base_model.model.{m}.lora_A.weight").float(),
                   h.get_tensor(f"base_model.model.{m}.lora_B.weight").float())
                  for h in HH]
            for i in range(len(ext)):
                for j2 in range(len(ext)):
                    EG[i, j2] += float(torch.trace(
                        (AB[i][0] @ AB[j2][0].T) @ (AB[j2][1].T @ AB[i][1]))) \
                        * sc[ext[i]] * sc[ext[j2]]
        print(f"[ws] external cross-Gram over {len(ext)} adapters done "
              f"{time.time() - t0:.0f}s", flush=True)
    ei = {p: i for i, p in enumerate(ext)}

    tn = np.zeros(T)
    for j, spec in enumerate(specs):
        d = _sources(spec)
        if all(p.startswith("/adapters/") for p in d):
            ks = [p.split("/")[-1] for p in d]
            idx = [gi[t] for t in ks]
            c = np.array([d[f"/adapters/{t}"] for t in ks])
            tn[j] = math.sqrt(max(float(c @ gram[np.ix_(idx, idx)] @ c), 0.0))
        elif all(p in ei for p in d):
            idx = [ei[p] for p in d]
            c = np.array([d[p] for p in d])
            tn[j] = math.sqrt(max(float(c @ EG[np.ix_(idx, idx)] @ c), 0.0))
        else:
            tn[j] = float("nan")
    an = np.sqrt(np.maximum(anorm2, 1e-30))
    cos = cross / np.outer(an, np.where(np.isfinite(tn) & (tn > 0), tn, np.nan))
    print(f"[ws] done {time.time() - t0:.0f}s", flush=True)
    return {"names": [s["name"] for s in specs],
            "target_proj": proj.tolist(),
            "atom_target_cos": np.where(np.isfinite(cos), cos, 0.0).tolist(),
            "target_norm": tn.tolist(), "atom_norm": an.tolist(),
            "seconds": time.time() - t0}


@app.local_entrypoint()
def main(stage: str = "extract", items: str = "phase10_runs/gradatoms_items.json",
         out_tag: str = "zoo", basis_from: str = "", out: str = "",
         batch_seqs: int = 32, fit_n: int = 384, k: int = 28, l: int = 32,
         mode: str = "dpo", check: int = 1,
         targets: str = "phase10_runs/gradatoms_targets.json",
         basis: str = "zoo_basis.npz", atoms_file: str = "zoo_atoms.npz",
         limit: int = 0, stride: int = 1, n_atoms: int = 500):
    import numpy as np
    if stage == "atoms" and mode == "sft":
        job = {"tag": out_tag, "labels": [], "arms": ["sorh"],
               "primary": f"sorh_{n_atoms}_0.1",
               "configs": [{"arm": "sorh", "K": n_atoms, "alpha": 0.1, "full": True,
                            "purity_keys": ["assigned_trait"]},
                           {"arm": "sorh", "K": n_atoms, "alpha": 0.01},
                           {"arm": "sorh", "K": n_atoms, "alpha": 1.0}]}
        print(f"atoms: reward-hacks corpus, K={n_atoms}", flush=True)
        r = atoms.remote(job)
        dest = out or f"analysis/gradient_atoms_atoms_{out_tag}.json"
    elif stage == "atoms":
        LB = json.load(open(os.path.join(HERE, "phase10_runs/gradatoms_labels.json")))
        I = json.load(open(os.path.join(HERE, items)))
        fa = json.load(open(os.path.join(HERE, "results/fa_qwen35.json")))
        Ld = np.array(fa["solutions"]["centred_k5"]["loadings"]["oblimin"])
        dom = {t: int(np.argmax(np.abs(Ld[i])))
               for i, t in enumerate(fa["trait_slug"])}
        job = {"tag": out_tag, "labels": LB["labels"],
               "permuted_map": LB["permuted_map"],
               "shuffled_swapped": LB["shuffled_swapped"],
               "prompts": {x["id"]: x["prompt"] for x in I},
               "fa_dominant": {t: f"F{v}" for t, v in dom.items()},
               "arms": ["real", "shuffled", "permuted"],
               "primary": f"real_{n_atoms}_0.1",
               "configs": [
                   {"arm": "real", "K": n_atoms, "alpha": 0.1, "full": True},
                   {"arm": "shuffled", "K": n_atoms, "alpha": 0.1, "full": True},
                   {"arm": "permuted", "K": n_atoms, "alpha": 0.1, "full": True},
                   {"arm": "real", "K": n_atoms, "alpha": 0.01},
                   {"arm": "real", "K": n_atoms, "alpha": 1.0},
                   {"arm": "real", "K": 500, "alpha": 0.1}]}
        print(f"atoms: K={n_atoms} on 3 arms plus a sparsity sweep", flush=True)
        r = atoms.remote(job)
        dest = out or f"analysis/gradient_atoms_atoms_{out_tag}.json"
    elif stage == "weightspace":
        S = json.load(open(os.path.join(HERE, targets)))
        z = np.load(os.path.join(HERE, "results/gram_sweep.npz"))
        job = {"targets": S["targets"], "basis": basis, "atoms": atoms_file,
               "zoo_gram": z["G"].tolist(),
               "zoo_gram_names": [str(x) for x in z["names"]]}
        print(f"{len(S['targets'])} targets", flush=True)
        r = weightspace.remote(job)
        dest = out or "analysis/gradient_atoms_weightspace.json"
    else:
        I = json.load(open(os.path.join(HERE, items)))
        if stride > 1:
            I = I[::stride]
        if limit:
            I = I[:limit]
        job = {"items": I, "out_tag": out_tag, "batch_seqs": batch_seqs,
               "fit_n": fit_n, "k": k, "l": l, "mode": mode, "check": bool(check)}
        if basis_from:
            job["basis_from"] = basis_from
        print(f"{len(I)} items, mode {mode}, out_tag {out_tag}", flush=True)
        r = extract.remote(job)
        dest = out or f"analysis/gradient_atoms_extract_{out_tag}.json"
    with open(os.path.join(HERE, dest), "w") as f:
        json.dump(r, f, indent=1)
    print(f"wrote {dest}", flush=True)
