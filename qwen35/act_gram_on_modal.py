"""The ACTIVATION-WEIGHTED Gram: the adapter cloud measured in ASVD's metric.

WHY
---
Every Gram in this project is the FROBENIUS inner product of two LoRA deltas,
which weights every input direction of a module equally.  Under it the same
trait retrained from an independent LoRA-A initialisation sits at cosine
+0.0181 to its original (analysis/crossseed_arms.json#[1].same[1]), which
seed-floor.md reads as the r/d = 64/2560 chance overlap of two random rank-64
input frames.  ASVD (arXiv:2312.05821) argues the functional metric is
activation-weighted: two updates are similar if they change the module's OUTPUT
similarly on the inputs the model actually sees.  This file builds that Gram and
asks whether the seed floor is partly an artefact of the isotropic metric.

THE OBJECT
----------
With C = E_x[x x^T] the uncentred input covariance of a module on the BASE model,

    <dW_i, dW_j>_C = E_x[(dW_i x) . (dW_j x)]
                   = s_i s_j tr(B_i^T B_j (A_j C A_i^T))
                   = s_i s_j sum( (B_i^T B_j) * (A_i C A_j^T) )

summed over the 248 targeted modules, exactly as the Frobenius Gram sums
tr(B_i^T B_j A_j A_i^T).  Setting C = I recovers the Frobenius Gram through the
same code path, which is this run's first validation check.

WHY C IS FORMED IN FULL
-----------------------
The cheap route is to accumulate projected covariances F C G^T for a handful of
SHARED frames.  That is exact only if every zoo adapter shares one A; it does
not, because A drifts 0.0146 in training.  Forming C itself costs 17.8 GiB in
fp32 (184 modules at d_in 2560, 32 at 9216, 32 at 4096) and about 1e14 FLOPs
over 30k tokens, which one A100-80GB does in under a minute, and it makes the
C-Gram EXACT in the same per-adapter A_i the Frobenius Gram used.  The projected
covariances are emitted anyway for a stride-31 module subset, as a record.

WHAT ELSE COMES OUT OF THE SAME PASS
------------------------------------
* the frame-overlap null, tr(P_F C P_G C)/sqrt(...) with P the row-space
  projector -- what a trait that reproduced PERFECTLY modulo the frame would
  score.  Under C = I it is tr(P_F P_G)/r, expectation r/d = 0.025.
* the literal brief null: one side's frame replaced by a fresh PEFT-style
  kaiming_uniform_ frame with B kept.  That statistic is linear in the frame and
  the draw is sign-symmetric, so its signed mean is zero by construction; its
  RMS is the real rank-64 noise floor and both are reported.
* the participation ratio (tr C)^2 / tr(C^2) of every module's input covariance,
  which is the d_eff the r/d prediction should be taken at.

usage:
    PC_APP_NAME=pc-qwen35-phase11-actgram \\
    modal run --detach act_gram_on_modal.py --spec phase10_runs/actgram_spec.json \\
        --out phase10_runs/actgram_results.json
"""
import json
import os
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-80GB")
# must start with pc-qwen35 or the spend meter's kill switch cannot see it
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-actgram")
SUBDIR_B = "data_null_seedpaired_s40_matched"
N_RAND_FRAMES = 8
FRAME_SEED = 70011
# per-run hard cap: the workspace meter is cumulative and does not cap one run
TIMEOUT_S = int(os.environ.get("PC_TIMEOUT_S", str(90 * 60)))

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
out_vol = modal.Volume.from_name("pc-qwen35-actgram", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/out": out_vol}


def _download_base_model():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    # the same pip set fisher_gram.py pins, so the install layer is cached
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "PC_BASE_MODEL": BASE_MODEL})
    .run_function(_download_base_model)
)

PFX = "base_model.model."


def _strip(m):
    while m.startswith(PFX):
        m = m[len(PFX):]
    return m


@app.function(secrets=[modal.Secret.from_name("hf-token")], image=image,
              gpu=GPU_TYPE, volumes=VOLS, timeout=TIMEOUT_S,
              memory=61440, cpu=4.0)
def act_gram(S: dict, max_prompts: int = 0, module_stride: int = 1,
             max_a: int = 0, max_b: int = 0, tag: str = "full") -> dict:
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    def note(m):
        print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = "cuda"

    # ---- which adapters -------------------------------------------------
    def adapters_under(root):
        return sorted(d for d in os.listdir(root)
                      if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                      and not d.startswith("data_null")
                      and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))

    root_a, root_b = "/adapters", f"/adapters/{SUBDIR_B}"
    names_a = adapters_under(root_a)
    names_b = adapters_under(root_b)
    if max_a:
        names_a = names_a[:max_a]
    if max_b:
        names_b = names_b[:max_b]
    note(f"A: {len(names_a)} under {root_a}   B: {len(names_b)} under {root_b}")

    def scale_of(root, n):
        import math
        c = json.load(open(f"{root}/{n}/adapter_config.json"))
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"]), int(c["r"])

    sa = {scale_of(root_a, n) for n in names_a[:3]}
    sb = {scale_of(root_b, n) for n in names_b[:3]}
    if len(sa) != 1 or len(sb) != 1:
        raise RuntimeError(f"mixed scales {sa} {sb}")
    (s_a, R_A), (s_b, R_B) = sa.pop(), sb.pop()
    if R_A != R_B:
        raise RuntimeError(f"rank mismatch {R_A} {R_B}")
    RANK = R_A
    note(f"scale_a {s_a} scale_b {s_b} rank {RANK}")

    handles, keymaps = {}, {}
    for root, names in ((root_a, names_a), (root_b, names_b)):
        for n in names:
            h = safe_open(f"{root}/{n}/adapter_model.safetensors", framework="pt")
            handles[(root, n)] = h
            keymaps[(root, n)] = {_strip(k.split(".lora_A.")[0]): k.split(".lora_A.")[0]
                                  for k in h.keys() if ".lora_A." in k}
    mods_all = sorted(keymaps[(root_a, names_a[0])])
    for key, km in keymaps.items():
        if set(km) != set(mods_all):
            raise RuntimeError(f"module set mismatch for {key}")
    mods = mods_all[::module_stride]
    note(f"{len(mods_all)} modules, using {len(mods)} (stride {module_stride})")

    h0, k0 = handles[(root_a, names_a[0])], keymaps[(root_a, names_a[0])]
    dims = {}
    for m in mods:
        sh = h0.get_slice(k0[m] + ".lora_A.weight").get_shape()
        sh_b = h0.get_slice(k0[m] + ".lora_B.weight").get_shape()
        dims[m] = {"r": int(sh[0]), "d_in": int(sh[1]), "d_out": int(sh_b[0])}
    note("d_in histogram: " + json.dumps(
        {str(k): sum(1 for v in dims.values() if v["d_in"] == k)
         for k in sorted({v["d_in"] for v in dims.values()})}))

    # ---- the base model, and C per module --------------------------------
    snap = snapshot_download(BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    model.requires_grad_(False)
    by = dict(model.named_modules())

    def find(m):                      # the name mapping fisher_gram.py uses
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        return None

    hit = {m: find(m) for m in mods}
    missing = [m for m, v in hit.items() if v is None]
    if missing:
        raise RuntimeError(f"unmapped modules: {missing[:4]} ({len(missing)} total)")
    note(f"model loaded bf16, cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    ARMS = list(S["arms"])
    Cs, MUs, ntok = {}, {}, {}
    state = {"C": None, "n": None, "mu": None}

    def mk(m):
        def hook(mod, inp, out):
            if state["C"] is None:
                return
            x = inp[0]
            x = x.reshape(-1, x.shape[-1]).float()
            if x.shape[1] != dims[m]["d_in"]:
                raise RuntimeError(f"{m}: input width {x.shape[1]} != "
                                   f"adapter d_in {dims[m]['d_in']}")
            state["C"][m].addmm_(x.T, x)
            state["mu"][m] += x.sum(0)
            state["n"][m] += x.shape[0]
        return hook

    hooks = [by[hit[m]].register_forward_hook(mk(m)) for m in mods]

    for arm in ARMS:
        atag = arm["tag"]
        C = {m: torch.zeros(dims[m]["d_in"], dims[m]["d_in"],
                            dtype=torch.float32, device=dev) for m in mods}
        mu = {m: torch.zeros(dims[m]["d_in"], dtype=torch.float32, device=dev)
              for m in mods}
        n = {m: 0 for m in mods}
        state["C"], state["n"], state["mu"] = C, n, mu
        prompts = arm["prompts"]
        texts = arm["texts"]
        if max_prompts:
            prompts = prompts[:max_prompts]
            texts = texts[:max_prompts] if texts is not None else None
        tot = 0
        for pi, p in enumerate(prompts):
            head = tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(head, add_special_tokens=False,
                      return_tensors="pt")["input_ids"][0]
            if texts is not None:
                rsp = tok(texts[pi], add_special_tokens=False,
                          return_tensors="pt")["input_ids"][0][:arm["max_resp_tokens"]]
                ids = torch.cat([ids, rsp])
            tot += int(ids.shape[0])
            model(ids.unsqueeze(0).to(dev), use_cache=False)
            if pi % 100 == 0:
                note(f"  {atag}: prompt {pi}/{len(prompts)}, {tot} tokens so far")
        state["C"], state["n"], state["mu"] = None, None, None
        seen = {n[m] for m in mods}
        if len(seen) != 1:
            raise RuntimeError(f"{atag}: hooks saw different token counts "
                               f"{sorted(seen)[:5]} ... {sorted(seen)[-3:]}")
        got = seen.pop()
        if got != tot:
            raise RuntimeError(f"{atag}: hook rows {got} != token total {tot}")
        for m in mods:
            C[m] /= float(tot)
            mu[m] /= float(tot)
        Cs[atag], MUs[atag], ntok[atag] = C, mu, tot
        note(f"  {atag}: {len(prompts)} sequences, {tot} tokens, C accumulated, "
             f"cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    for h in hooks:
        h.remove()
    del model, by
    torch.cuda.empty_cache()
    note(f"model freed, cuda alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ---- the metric tags -------------------------------------------------
    # "<arm>_centred" is the same C with the mean activation direction removed,
    # C - mu mu^T.  The uncentred form is the one E_x[(dW x).(dW x)] asks for and
    # is primary; the centred one isolates how much of the agreement is carried
    # by the single direction every token shares.
    CTAGS = []
    for a in ARMS:
        CTAGS += [a["tag"], a["tag"] + "_centred"]

    def C_of(t, m):
        if t.endswith("_centred"):
            b = t[:-len("_centred")]
            return Cs[b][m] - torch.outer(MUs[b][m], MUs[b][m])
        return Cs[t][m]

    # ---- spectrum of C ---------------------------------------------------
    spectra = {}
    for atag in CTAGS:
        per = {}
        for mi, m in enumerate(mods):
            Cm = C_of(atag, m)
            tr = float(torch.diagonal(Cm).sum())
            fro2 = float((Cm * Cm).sum())
            d = dims[m]["d_in"]
            base = atag[:-len("_centred")] if atag.endswith("_centred") else atag
            mun = float(MUs[base][m].dot(MUs[base][m]))
            rec = {"d_in": d, "trace": tr, "tr_C2": fro2,
                   "participation_ratio": (tr * tr / fro2) if fro2 > 0 else 0.0,
                   "mean_direction_share_of_trace":
                       mun / float(torch.diagonal(Cs[base][m]).sum())}
            if mi % 31 == 0:
                try:
                    ev = torch.linalg.eigvalsh(Cm.double()).flip(0).clamp_min(0)
                    tot_ = ev.sum()
                    cs = torch.cumsum(ev, 0) / tot_
                    rec["eig_top1_share"] = float(ev[0] / tot_)
                    rec["eig_top64_share"] = float(cs[min(63, d - 1)])
                    rec["eig_top256_share"] = float(cs[min(255, d - 1)])
                    rec["d_eff_90pct_trace"] = int((cs < 0.90).sum()) + 1
                    rec["d_eff_99pct_trace"] = int((cs < 0.99).sum()) + 1
                    del ev, cs
                except Exception as e:            # noqa: BLE001
                    rec["eig_error"] = repr(e)[:200]
            per[m] = rec
            del Cm
        spectra[atag] = per
        note(f"  {atag}: spectrum done, mean PR "
             f"{np.mean([v['participation_ratio'] for v in per.values()]):.2f}")
    torch.cuda.empty_cache()

    # ---- the adapter pass ------------------------------------------------
    NA, NB, NR = len(names_a), len(names_b), N_RAND_FRAMES
    NT = NA + NB                       # real adapters
    NF = NT + NR                       # frame blocks (real A's + random frames)
    METRICS = ["frob"] + CTAGS

    pairs = [(names_a.index(t), NA + names_b.index(t))
             for t in names_b if t in names_a]
    if not pairs:
        raise RuntimeError("no seed-1 trait has a seed-0 twin in this selection")
    NP = len(pairs)
    ia_t = torch.tensor([p[0] for p in pairs], device=dev)
    ib_t = torch.tensor([p[1] for p in pairs], device=dev)

    G = {k: torch.zeros(NT, NT, dtype=torch.float64, device=dev) for k in METRICS}
    c1 = {k: {w: torch.zeros(NP, NR, dtype=torch.float64, device=dev)
              for w in ("num_rep_b", "num_rep_a", "den_b", "den_a")}
          for k in METRICS}
    C2CLS = ["seed0_vs_seed1", "seed0_vs_random", "random_vs_random"]
    c2 = {c: {k: {"b_num": 0.0, "b_dp": 0.0, "b_dq": 0.0,
                  "a_num": 0.0, "a_dp": 0.0, "a_dq": 0.0,
                  "b_permod": [], "a_permod": [], "n": 0} for k in METRICS}
          for c in C2CLS}
    adrift = {"a": [], "b": []}
    # per-module share of the total squared norm, so a reader can tell whether
    # the summed-over-modules Gram is really a one-module Gram in disguise
    modshare = {k: [] for k in METRICS}
    modsame = {k: [] for k in METRICS}
    proj_cov = {}
    mods_index = {m: i for i, m in enumerate(mods_all)}
    gen = torch.Generator(device="cpu")

    rep = min(NP, NA)
    p_s0r = torch.arange(rep, device=dev).repeat_interleave(NR)
    q_s0r = torch.arange(NT, NF, device=dev).repeat(rep)
    tri = [(NT + i, NT + j) for i in range(NR) for j in range(i + 1, NR)]
    p_rr = torch.tensor([t[0] for t in tri], device=dev)
    q_rr = torch.tensor([t[1] for t in tri], device=dev)

    for mi, m in enumerate(mods):
        d_in, d_out, r = dims[m]["d_in"], dims[m]["d_out"], dims[m]["r"]
        A = torch.empty(NF, r, d_in, dtype=torch.float32, device=dev)
        B = torch.empty(NT, d_out, r, dtype=torch.float32, device=dev)
        for idx, (root, n) in enumerate([(root_a, x) for x in names_a]
                                        + [(root_b, x) for x in names_b]):
            k = keymaps[(root, n)][m]
            A[idx] = handles[(root, n)].get_tensor(k + ".lora_A.weight").float().to(dev)
            B[idx] = handles[(root, n)].get_tensor(k + ".lora_B.weight").float().to(dev)
        # PEFT's LoRA-A init: kaiming_uniform_(a=sqrt(5)) on a (r, d_in) weight,
        # which is U(-1/sqrt(d_in), +1/sqrt(d_in)).
        gen.manual_seed(FRAME_SEED + 1009 * mods_index[m])
        bnd = 1.0 / (d_in ** 0.5)
        A[NT:] = ((torch.rand(NR, r, d_in, generator=gen, dtype=torch.float32) * 2 - 1)
                  * bnd).to(dev)

        mu = A[:NA].mean(0, keepdim=True)
        adrift["a"].append(float(((A[:NA] - mu).flatten(1).norm(dim=1)
                                  / mu.flatten().norm()).mean()))
        mu = A[NA:NT].mean(0, keepdim=True)
        adrift["b"].append(float(((A[NA:NT] - mu).flatten(1).norm(dim=1)
                                  / mu.flatten().norm()).mean()))

        Aflat = A.reshape(NF * r, d_in)
        BtB = torch.einsum("ndr,mdq->nmrq", B, B)          # (NT, NT, r, r)
        Bij = BtB[ia_t, ib_t]                              # (NP, r, r)
        Bii = BtB[ia_t, ia_t]
        Bjj = BtB[ib_t, ib_t]

        Ms = {}
        for k in METRICS:
            if k == "frob":
                Mfull = Aflat @ Aflat.T
            else:
                Cm = C_of(k, m)
                Mfull = (Aflat @ Cm) @ Aflat.T
                del Cm
            Ms[k] = Mfull.reshape(NF, r, NF, r).permute(0, 2, 1, 3).contiguous()
            del Mfull
            M = Ms[k]
            contrib = (BtB * M[:NT, :NT]).sum(dim=(2, 3))
            G[k] += contrib.double()
            modshare[k].append(float(torch.diagonal(contrib)[:NA].sum()))
            modsame[k].append(float(contrib[ia_t, ib_t].sum()))
            del contrib
            Mkk = torch.diagonal(M[NT:, NT:], dim1=0, dim2=1).permute(2, 0, 1)  # (NR,r,r)
            c1[k]["num_rep_b"] += torch.einsum(
                "prq,pkrq->pk", Bij, M[ia_t][:, NT:]).double()
            c1[k]["num_rep_a"] += torch.einsum(
                "prq,pkrq->pk", Bij, M[:, ib_t][NT:].permute(1, 0, 2, 3)).double()
            c1[k]["den_b"] += torch.einsum("prq,krq->pk", Bjj, Mkk).double()
            c1[k]["den_a"] += torch.einsum("prq,krq->pk", Bii, Mkk).double()
        del BtB, Bij, Bii, Bjj

        # ---- frame-overlap null (c2), needs the C = I blocks for S_F ------
        Y = Ms["frob"]
        S_all = torch.linalg.inv(
            torch.diagonal(Y, dim1=0, dim2=1).permute(2, 0, 1).double()).float()

        def c2_add(cls, p_idx, q_idx):
            for k in METRICS:
                M = Ms[k]
                Xpq, Xpp, Xqq = M[p_idx, q_idx], M[p_idx, p_idx], M[q_idx, q_idx]
                Sp, Sq = S_all[p_idx], S_all[q_idx]
                bn = torch.einsum("nab,nbc,ncd,nad->n", Sp, Xpq, Sq, Xpq)
                bp = torch.einsum("nab,nbc,ncd,nda->n", Sp, Xpp, Sp, Xpp)
                bq = torch.einsum("nab,nbc,ncd,nda->n", Sq, Xqq, Sq, Xqq)
                an = torch.einsum("nab,nbc,ncd,nda->n", Sp, Xpq, Sq, Y[q_idx, p_idx])
                ap = torch.einsum("nab,nba->n", Sp, Xpp)
                aq = torch.einsum("nab,nba->n", Sq, Xqq)
                acc = c2[cls][k]
                acc["b_num"] += float(bn.sum())
                acc["b_dp"] += float(bp.sum())
                acc["b_dq"] += float(bq.sum())
                acc["a_num"] += float(an.sum())
                acc["a_dp"] += float(ap.sum())
                acc["a_dq"] += float(aq.sum())
                acc["b_permod"].append(float((bn / (bp * bq).sqrt()).mean()))
                acc["a_permod"].append(float((an / (ap * aq).sqrt()).mean()))
                acc["n"] = int(bn.shape[0])

        c2_add("seed0_vs_seed1", ia_t, ib_t)
        c2_add("seed0_vs_random", p_s0r, q_s0r)
        c2_add("random_vs_random", p_rr, q_rr)

        if mi % 31 == 0:
            fidx = [0, NA] + list(range(NT, NF))
            proj_cov[m] = {k: Ms[k][fidx][:, fidx].float().cpu().numpy()
                           for k in METRICS}
        del Ms, Y, S_all, A, B
        torch.cuda.empty_cache()
        if mi % 20 == 0:
            note(f"  module {mi}/{len(mods)} {m}  "
                 f"cuda {torch.cuda.memory_allocated()/2**30:.1f} GiB")

    # ---- assemble --------------------------------------------------------
    out = {"names_a": names_a, "names_b": names_b, "scale_a": s_a, "scale_b": s_b,
           "rank": RANK, "n_modules": len(mods), "n_modules_total": len(mods_all),
           "module_stride": module_stride, "tag": tag,
           "n_tokens": ntok, "metrics": METRICS,
           "n_rand_frames": NR, "frame_seed": FRAME_SEED,
           "base_model": BASE_MODEL, "forward_dtype": "bfloat16",
           "C_is_uncentred": True,
           "dims": {m: dims[m] for m in mods},
           "a_frame_spread": {"seed0_mean_over_modules": float(np.mean(adrift["a"])),
                              "seed1_mean_over_modules": float(np.mean(adrift["b"])),
                              "seed0_max_module": float(np.max(adrift["a"])),
                              "seed1_max_module": float(np.max(adrift["b"]))},
           "spectra": spectra,
           "module_energy_share": {k: modshare[k] for k in METRICS},
           "module_same_trait_numerator": {k: modsame[k] for k in METRICS},
           "module_order": mods,
           "proj_cov_modules": sorted(proj_cov),
           "grams": {}, "c1_null": {}, "c2_frame_overlap": {},
           "pairs": [[names_a[a], names_b[b - NA]] for a, b in pairs]}

    sc = np.zeros((NT, NT))
    sc[:NA, :NA] = s_a * s_a
    sc[:NA, NA:] = s_a * s_b
    sc[NA:, :NA] = s_a * s_b
    sc[NA:, NA:] = s_b * s_b
    ia = np.array([p[0] for p in pairs])
    ib = np.array([p[1] for p in pairs])
    for k in METRICS:
        Gk = G[k].cpu().numpy() * sc
        out["grams"][k] = Gk.tolist()
        dg = np.sqrt(np.diag(Gk))
        cn = {w: c1[k][w].cpu().numpy() for w in c1[k]}
        cos_b = (cn["num_rep_b"] * s_a * s_b) / np.sqrt(
            (dg[ia] ** 2)[:, None] * cn["den_b"] * s_b * s_b)
        cos_a = (cn["num_rep_a"] * s_a * s_b) / np.sqrt(
            (dg[ib] ** 2)[:, None] * cn["den_a"] * s_a * s_a)
        out["c1_null"][k] = {
            "replace_seed1_frame": {"mean": float(cos_b.mean()),
                                    "rms": float(np.sqrt((cos_b ** 2).mean())),
                                    "sd": float(cos_b.std()), "n": int(cos_b.size)},
            "replace_seed0_frame": {"mean": float(cos_a.mean()),
                                    "rms": float(np.sqrt((cos_a ** 2).mean())),
                                    "sd": float(cos_a.std()), "n": int(cos_a.size)}}
        out["c2_frame_overlap"][k] = {}
        for cls in C2CLS:
            acc = c2[cls][k]
            out["c2_frame_overlap"][k][cls] = {
                "c2b_summed": acc["b_num"] / ((acc["b_dp"] * acc["b_dq"]) ** 0.5),
                "c2a_summed": acc["a_num"] / ((acc["a_dp"] * acc["a_dq"]) ** 0.5),
                "c2b_mean_over_modules": float(np.mean(acc["b_permod"])),
                "c2a_mean_over_modules": float(np.mean(acc["a_permod"])),
                "n_pairs": acc["n"]}

    out["wall_seconds"] = time.time() - t0
    os.makedirs("/out", exist_ok=True)
    with open(f"/out/actgram_{tag}.json", "w") as f:
        json.dump(out, f)
    flat = {}
    for m, d in proj_cov.items():
        for k, v in d.items():
            flat[f"{m}|{k}"] = v
    if flat:
        np.savez_compressed(f"/out/actgram_projcov_{tag}.npz",
                            frame_order=np.array(
                                [names_a[0], names_b[0]]
                                + [f"random_{i}" for i in range(NR)]),
                            **flat)
    out_vol.commit()
    note(f"done in {out['wall_seconds']:.1f}s")
    return out


@app.local_entrypoint()
def main(spec: str = "phase10_runs/actgram_spec.json",
         out: str = "phase10_runs/actgram_results.json",
         max_prompts: int = 0, module_stride: int = 1,
         max_a: int = 0, max_b: int = 0, tag: str = "full"):
    import numpy as np
    S = json.load(open(os.path.join(HERE, spec)))
    r = act_gram.remote(S, max_prompts, module_stride, max_a, max_b, tag)
    p = os.path.join(HERE, out)
    with open(p, "w") as f:
        json.dump(r, f)
    print(f"wrote {p}  ({r['wall_seconds']:.1f}s in the function)")

    NA = len(r["names_a"])
    for k in r["metrics"]:
        G = np.array(r["grams"][k])
        d = np.sqrt(np.diag(G))
        Cc = G / np.outer(d, d)
        same, off = [], []
        for jj, t in enumerate(r["names_b"]):
            for ii, u in enumerate(r["names_a"]):
                (same if t == u else off).append(Cc[ii, NA + jj])
        print(f"  {k:10s} same {np.mean(same):+.6f}  off {np.mean(off):+.6f}  "
              f"c1rms {r['c1_null'][k]['replace_seed1_frame']['rms']:.6f}  "
              f"c2b(s0,s1) {r['c2_frame_overlap'][k]['seed0_vs_seed1']['c2b_summed']:.6f}  "
              f"c2b(rnd) {r['c2_frame_overlap'][k]['random_vs_random']['c2b_summed']:.6f}")
