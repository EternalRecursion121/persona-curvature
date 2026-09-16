"""COLUMN-SPACE overlap for the emergent-misalignment medical arms.

A COPY of column_space_sorh_on_modal.py, not an edit, for the reason
dolci_score.py is a copy of align_score.py: `modal run` picks the launch file up
at image-build time and the original is the artefact
wiki/pages/behaviour/reward-hacks-column-space.md cites.  Everything is that
file's method verbatim -- the thin QR/SVD factorisation, the k sweep, the
sigma-weighted overlap, the reference bands and the null, all measured inside
this same run -- with one change: the arms in the pairwise block are the three
emergent-misalignment SFT arms and their checkpoints instead of the two School
of Reward Hacks arms, and the difference deltas are em_bad minus em_good.

usage:
    PC_APP_NAME=pc-qwen35-phase13-emcolspace \\
    /home/vibe12/cartovenv/bin/modal run column_space_em_on_modal.py::preflight
    ... then ::main --out-tag em   -> results/column_space_em.npz
"""
import io
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase13-emcolspace")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)
rl_vol = modal.Volume.from_name("pc-qwen35-rl", create_if_missing=False)
adp_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=False)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol, "/rl": rl_vol, "/align": adp_vol}
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3"))

KS = [1, 4, 8, 16, 64]
NCPU = 8
PFX = "base_model.model."

ZOO_ROOT = "/adapters"
SEED1_ROOT = "/adapters/data_null_seedpaired_s40_matched"
ALIGN_ROOT = "/align/data_alignment_common"
STAGE2_ROOT = "/oct/loras_introspection"
EM_ROOT = "/align/em_medical"
EM_ARMS = ["em_bad", "em_good", "em_dolci"]


def _ckpts(arm):
    """Checkpoint directories of one arm, in step order, then `final`.

    Read off the volume rather than hard-coded: the step count is a function of
    the row count and the batch, and a hard-coded name that no longer exists is
    how a run dies after loading a gigabyte.
    """
    root = f"{EM_ROOT}/{arm}"
    cks = sorted((d for d in os.listdir(root) if d.startswith("checkpoint-")),
                 key=lambda d: int(d.split("-")[1]))
    return [c for c in cks
            if os.path.exists(f"{root}/{c}/adapter_model.safetensors")] + ["final"]


def _dirs(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                  and not d.startswith("data_")
                  and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))


def _entries():
    """(tag, display name, adapter directory) for the pairwise block, in order."""
    e = []
    for arm in EM_ARMS:
        for c in _ckpts(arm):
            e.append(("em", f"{arm}_{c.replace('checkpoint-', 'c')}",
                      f"{EM_ROOT}/{arm}/{c}"))
    for n in _dirs(ZOO_ROOT):
        e.append(("zoo", n, f"{ZOO_ROOT}/{n}"))
    for n in _dirs(ALIGN_ROOT):
        e.append(("align", n, f"{ALIGN_ROOT}/{n}"))
    return e


def _diff_specs():
    """(name, bad dir, good dir) for the bad-minus-good difference deltas.

    Only the checkpoints both medical arms actually have; the two arms train on
    the same rows for the same number of steps, so that is all of them.
    """
    common = [c for c in _ckpts("em_bad") if c in set(_ckpts("em_good"))]
    return [(f"diff_{c.replace('checkpoint-', 'c')}",
             f"{EM_ROOT}/em_bad/{c}", f"{EM_ROOT}/em_good/{c}") for c in common]


@app.function(image=image, volumes=VOLS, timeout=60 * 10, cpu=2.0, memory=4096)
def preflight() -> str:
    """Ten seconds of checking, so the main job does not die after loading a GiB."""
    from safetensors import safe_open

    def mods(path):
        h = safe_open(f"{path}/adapter_model.safetensors", framework="pt")
        out = {}
        for k in h.keys():
            if ".lora_A." not in k:
                continue
            m = k.split(".lora_A.")[0]
            while m.startswith(PFX):
                m = m[len(PFX):]
            a = h.get_slice(k).get_shape()
            b = h.get_slice(k.split(".lora_A.")[0] + ".lora_B.weight").get_shape()
            out[m] = (a[0], a[1], b[0])
        return out

    probe_paths = {
        "zoo/bold": f"{ZOO_ROOT}/bold",
        "seed1/bold": f"{SEED1_ROOT}/{_dirs(SEED1_ROOT)[0]}",
        "align/sycophantic": f"{ALIGN_ROOT}/sycophantic",
        "stage2/bold": f"{STAGE2_ROOT}/bold",
    }
    for arm in EM_ARMS:
        for c in _ckpts(arm):
            probe_paths[f"{arm}/{c}"] = f"{EM_ROOT}/{arm}/{c}"
    ref = mods(probe_paths["zoo/bold"])
    rep = {"reference": "zoo/bold", "n_modules_reference": len(ref), "sets": {}}
    for lab, p in probe_paths.items():
        m = mods(p)
        rep["sets"][lab] = {
            "path": p, "n_modules": len(m),
            "same_module_set_as_reference": set(m) == set(ref),
            "same_shapes_as_reference": m == ref,
            "missing": sorted(set(ref) - set(m))[:5],
            "extra": sorted(set(m) - set(ref))[:5]}
        with open(f"{p}/adapter_config.json") as f:
            c = json.load(f)
        rep["sets"][lab]["config"] = {k: c.get(k) for k in
                                      ("r", "lora_alpha", "use_rslora")}
    rep["counts"] = {"zoo": len(_dirs(ZOO_ROOT)), "seed1": len(_dirs(SEED1_ROOT)),
                     "align": len(_dirs(ALIGN_ROOT)),
                     "stage2": len(_dirs(STAGE2_ROOT))}
    rep["entries"] = [n for _, n, _ in _entries()][:12]
    rep["diffs"] = [n for n, _, _ in _diff_specs()]
    # is `final` the same weights as the last checkpoint?  On the SoRH arms it
    # was, and a duplicate row in the pairwise block would have read 1.0 and
    # meant nothing.
    import numpy as np
    for arm in EM_ARMS:
        cks = _ckpts(arm)
        last = cks[-2] if len(cks) > 1 else cks[0]
        a = safe_open(f"{EM_ROOT}/{arm}/final/adapter_model.safetensors", framework="np")
        b = safe_open(f"{EM_ROOT}/{arm}/{last}/adapter_model.safetensors", framework="np")
        k = [x for x in a.keys() if x.endswith("lora_B.weight")][0]
        x, y = a.get_tensor(k), b.get_tensor(k)
        rep.setdefault("final_vs_last_checkpoint", {})[arm] = {
            "key": k, "against": last, "identical": bool(np.array_equal(x, y)),
            "max_abs_diff": float(np.abs(x.astype("float64")
                                         - y.astype("float64")).max())}
    return json.dumps(rep, indent=1)


@app.function(image=image, volumes=VOLS, timeout=60 * 180, cpu=float(NCPU),
              memory=98304)
def column_space_em(sub_stride: int = 8, n_null: int = 200,
                      module_stride: int = 1, max_modules: int = 0,
                      q_rand: int = 96, n_iter: int = 4) -> bytes:
    import gc
    import time

    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(NCPU)
    torch.set_grad_enabled(False)

    ent = _entries()
    diffs = _diff_specs()
    seed1 = [(f"{SEED1_ROOT}/{n}", n) for n in _dirs(SEED1_ROOT)]
    stage2 = [(f"{STAGE2_ROOT}/{n}", n) for n in _dirs(STAGE2_ROOT)]

    names = [n for _, n, _ in ent] + [n for n, _, _ in diffs]
    tags = [t for t, _, _ in ent] + ["diff"] * len(diffs)
    N = len(names)
    n_ent = len(ent)
    print(f"pairwise block N = {N} ({n_ent} adapters + {len(diffs)} diffs); "
          f"{len(seed1)} seed-1 extras; {len(stage2)} stage-two", flush=True)

    def scale_of(path):
        import math
        with open(f"{path}/adapter_config.json") as f:
            c = json.load(f)
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"])

    scales = np.array([scale_of(p) for _, _, p in ent], dtype=np.float64)
    diff_scales = [(scale_of(a), scale_of(b)) for _, a, b in diffs]
    s1_scales = np.array([scale_of(p) for p, _ in seed1], dtype=np.float64)
    s2_scales = np.array([scale_of(p) for p, _ in stage2], dtype=np.float64)
    print(f"scales: {sorted(set(scales.tolist()))} diff {diff_scales[0]}", flush=True)

    def opened(paths):
        return [safe_open(f"{p}/adapter_model.safetensors", framework="pt")
                for p in paths]

    P_ENT = [p for _, _, p in ent]
    P_DH = [a for _, a, _ in diffs]
    P_DC = [b for _, _, b in diffs]
    P_S1 = [p for p, _ in seed1]
    P_S2 = [p for p, _ in stage2]
    h_ent = opened(P_ENT)
    h_diff_h = opened(P_DH)
    h_diff_c = opened(P_DC)
    h_s1 = opened(P_S1)
    h_s2 = opened(P_S2)

    def keymap(h):
        m = {}
        for k in h.keys():
            if ".lora_A." in k:
                mod = k.split(".lora_A.")[0]
                norm = mod
                while norm.startswith(PFX):
                    norm = norm[len(PFX):]
                m[norm] = mod
        return m

    km_ent = [keymap(h) for h in h_ent]
    km_dh = [keymap(h) for h in h_diff_h]
    km_dc = [keymap(h) for h in h_diff_c]
    km_s1 = [keymap(h) for h in h_s1]
    km_s2 = [keymap(h) for h in h_s2]
    mods_ref = sorted(km_ent[0])
    for lab, kms in (("entries", km_ent), ("diff_h", km_dh), ("diff_c", km_dc),
                     ("seed1", km_s1), ("stage2", km_s2)):
        for i, km in enumerate(kms):
            if set(km) != set(mods_ref):
                raise RuntimeError(f"module set mismatch, {lab}[{i}]")
    mods = mods_ref[::module_stride]
    if max_modules:
        mods = mods[:max_modules]
    sub_idx = set(range(0, len(mods), sub_stride))
    print(f"{len(mods)} modules, {len(sub_idx)} in the row-space/null subset",
          flush=True)

    def zeros(n_a, n_b):
        return {k: np.zeros((n_a, n_b), dtype=np.float64) for k in KS}

    col_unw, col_wtd = zeros(N, N), zeros(N, N)
    col_unw_nar, col_wtd_nar = zeros(N, N), zeros(N, N)
    row_unw = zeros(N, N)
    n_wide = 0
    n_narrow = 0
    n_row = 0
    row_cnt = {k: 0 for k in KS}
    col_cnt_nar = {k: 0 for k in KS}
    top1abs = np.zeros((N, N))
    top1sgn = np.zeros((N, N))
    # the arm rows come first in `names` (see _entries), then the diffs.
    n_arm = sum(len(_ckpts(a)) for a in EM_ARMS)
    n_probe = n_arm + len(diffs)
    frob_exact = np.zeros((n_probe, N))          # all modules, probe rows
    frob_exact_wide = np.zeros((n_probe, N))     # wide modules only
    frob_approx_ent = np.zeros((n_ent, n_ent))   # tr(M_i^T M_j), all modules
    frob_approx_ent_wide = np.zeros((n_ent, n_ent))
    # the two forms side by side on the arm rows, wide modules
    frob_exact_sub = np.zeros((n_arm, n_ent))
    frob_approx_sub = np.zeros((n_arm, n_ent))

    REFS = ["G1_mean", "G1_stack", "G2_mean", "G2_stack"]
    ref_unw = {r: {k: np.zeros(N) for k in KS} for r in REFS}
    ref_wtd = {r: {k: np.zeros(N) for k in KS} for r in REFS}
    ref_unw_s1 = {r: {k: np.zeros(len(seed1)) for k in KS} for r in REFS}
    ref_wtd_s1 = {r: {k: np.zeros(len(seed1)) for k in KS} for r in REFS}
    ref_unw_z2 = {r: {k: np.zeros(len(stage2)) for k in KS} for r in REFS}
    ref_wtd_z2 = {r: {k: np.zeros(len(stage2)) for k in KS} for r in REFS}

    spec_prof = np.zeros((N, 64))
    spec_cnt = np.zeros(64)
    mod_records, null_records, mod_overlaps = [], [], []
    d_outs, d_ins, ranks = [], [], []

    tags_np = np.array(tags)
    names_np = np.array(names)
    is_zoo = tags_np == "zoo"
    probe_rows = np.arange(n_probe)   # the arm rows + diffs, first in `names`

    def basis_of(A, B, s):
        """A (n,r,d_in), B (n,d_out,r), s (n,) -> Q_A (n,d_in,q), U, S."""
        sv = torch.as_tensor(s, dtype=torch.float32).view(-1, 1, 1)
        Q_A, R = torch.linalg.qr(A.transpose(1, 2))
        M = sv * (B @ R.transpose(1, 2))
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        return Q_A, M, U, S, Vh

    def rand_svd_stack(M, q, niter):
        """top-q left singular vectors of a (d_out, ncols) matrix."""
        U, S, _ = torch.svd_lowrank(M, q=q, niter=niter)
        return U, S

    def rand_svd_factored(M, Q, q, niter):
        """top-q left singular vectors of mean_i M_i Q_i^T, never formed."""
        n, d_out, r = M.shape
        d_in = Q.shape[1]

        def fwd(X):            # (d_in, c) -> (d_out, c)
            return torch.einsum("nar,nrc->ac", M, Q.transpose(1, 2) @ X) / n

        def bwd(Y):            # (d_out, c) -> (d_in, c)
            return torch.einsum("nbr,nrc->bc", Q, M.transpose(1, 2) @ Y) / n

        Om = torch.randn(d_in, q, generator=torch.Generator().manual_seed(7))
        Y = fwd(Om)
        for _ in range(niter):
            Y, _ = torch.linalg.qr(Y)
            Y = fwd(bwd(Y))
        Qy, _ = torch.linalg.qr(Y)
        Bs = bwd(Qy).T                                    # (q, d_in)
        Ub, Sb, _ = torch.linalg.svd(Bs, full_matrices=False)
        return Qy @ Ub, Sb

    def ref_overlap(U, S, G, out_unw, out_wtd, idx=None):
        """U (n,d_out,m), G (d_out,kg) orthonormal ordered."""
        got = {}
        C = torch.einsum("nam,ak->nmk", U, G)             # (n, m, kg)
        C2 = C.pow(2)
        m, kg = C2.shape[1], C2.shape[2]
        cum = torch.cumsum(C2, dim=2)                      # over G's columns
        w = S ** 2
        for k in KS:
            if k > m or k > kg:
                continue
            T = cum[:, :, k - 1]                           # (n, m)
            unw = (T[:, :k].sum(dim=1) / k).numpy()
            wk = w[:, :k]
            wt = ((wk * T[:, :k]).sum(dim=1)
                  / wk.sum(dim=1).clamp_min(1e-30)).numpy()
            if idx is None:
                out_unw[k] += unw
                out_wtd[k] += wt
            else:
                out_unw[k][idx] += unw
                out_wtd[k][idx] += wt
            got[k] = wt
        return got

    def pair_sweep(U, S, out_unw, out_wtd, keep=()):
        done, cur = [], {}
        n, d, m = U.shape
        flat = U.transpose(1, 2).reshape(n * m, d).contiguous()
        C = (flat @ flat.T).view(n, m, n, m)
        t1 = C[:, 0, :, 0].clone()      # a view of C, which pow_ overwrites
        C.pow_(2)
        torch.cumsum(C, dim=3, out=C)
        w = S ** 2
        for k in KS:
            if k > m:
                continue
            T = C[:, :, :, k - 1]                          # (n, m, n)
            u = (T[:, :k, :].sum(dim=1) / k).numpy().astype(np.float64)
            wk = w[:, :k]
            v = (torch.einsum("ia,iaj->ij", wk, T[:, :k, :])
                 / wk.sum(dim=1, keepdim=True).clamp_min(1e-30)
                 ).numpy().astype(np.float64)
            out_unw[k] += u
            out_wtd[k] += v
            if k in keep:
                cur[k] = (u, v)
            done.append(k)
        del C, flat
        return t1, done, cur

    def rss_mb():
        try:
            with open("/proc/self/status") as f:
                for ln in f:
                    if ln.startswith("VmRSS:"):
                        return int(ln.split()[1]) / 1024
        except Exception:
            pass
        return -1.0

    REOPEN_EVERY = 24

    t0 = time.time()
    for mi, mod in enumerate(mods):
        if mi and mi % REOPEN_EVERY == 0:
            # Drop and rebuild every mmap. 324 adapter files stay mapped
            # otherwise, and mapped file pages are what OOM'd the first attempt
            # at module ~215 of 248 with a 48 GiB request.
            before = rss_mb()
            del h_ent, h_diff_h, h_diff_c, h_s1, h_s2
            gc.collect()
            h_ent = opened(P_ENT)
            h_diff_h = opened(P_DH)
            h_diff_c = opened(P_DC)
            h_s1 = opened(P_S1)
            h_s2 = opened(P_S2)
            print(f"  [reopen at module {mi}] rss {before:.0f} -> "
                  f"{rss_mb():.0f} MiB", flush=True)
        A = torch.stack([h_ent[i].get_tensor(km_ent[i][mod] + ".lora_A.weight").float()
                         for i in range(n_ent)])
        B = torch.stack([h_ent[i].get_tensor(km_ent[i][mod] + ".lora_B.weight").float()
                         for i in range(n_ent)])
        r, d_in, d_out = A.shape[1], A.shape[2], B.shape[1]
        Q_A, M, U, S, Vh = basis_of(A, B, scales)
        m = S.shape[1]
        wide = m >= 64
        d_ins.append(d_in); d_outs.append(d_out); ranks.append(m)

        # --- difference deltas, rank-128 concatenation --------------------
        # delta_diff = s_h B_h A_h - s_c B_c A_c = Bcat @ Acat with Bcat
        # (d_out, 128) and Acat (128, d_in).  Same QR/SVD.  A_h and A_c are the
        # same zoo A up to training drift, so ~64 of the 128 singular values are
        # near zero; the truncation to m is recorded per module.
        dU, dS, dQrow, dMfull, dQfull = [], [], [], [], []
        dsv_ratio = []
        for j, (nm, ph, pc) in enumerate(diffs):
            Ah = h_diff_h[j].get_tensor(km_dh[j][mod] + ".lora_A.weight").float()
            Bh = h_diff_h[j].get_tensor(km_dh[j][mod] + ".lora_B.weight").float()
            Ac = h_diff_c[j].get_tensor(km_dc[j][mod] + ".lora_A.weight").float()
            Bc = h_diff_c[j].get_tensor(km_dc[j][mod] + ".lora_B.weight").float()
            sh, sc = diff_scales[j]
            Bcat = torch.cat([sh * Bh, -sc * Bc], dim=1)          # (d_out, 128)
            Acat = torch.cat([Ah, Ac], dim=0)                      # (128, d_in)
            Qd, Rd = torch.linalg.qr(Acat.T)                       # (d_in,128)
            Md = Bcat @ Rd.transpose(0, 1)                         # (d_out,128)
            Ud, Sd, Vhd = torch.linalg.svd(Md, full_matrices=False)
            mm = Sd.shape[0]
            e = (Sd ** 2)
            dsv_ratio.append((float(Sd[min(m, mm) - 1] / Sd[0]),
                              float(Sd[m] / Sd[0]) if mm > m else float("nan"),
                              float(e[:m].sum() / e.sum())))
            dU.append(Ud[:, :m]); dS.append(Sd[:m])
            # delta = Ud Sd (Qd Vhd^T)^T: the ordered row basis, for the row sweep
            dQrow.append((Qd @ Vhd.transpose(0, 1))[:, :m])
            # the untruncated (M, Q) pair, for the exact Frobenius
            dMfull.append(Md); dQfull.append(Qd)
        U = torch.cat([U, torch.stack(dU)], dim=0)
        S = torch.cat([S, torch.stack(dS)], dim=0)
        dMfull = torch.stack(dMfull); dQfull = torch.stack(dQfull)
        del dU, dS

        # --- pairwise column-space sweep ----------------------------------
        if wide:
            t1, _, cur = pair_sweep(U, S, col_unw, col_wtd, keep=(1, 8, 64))
            top1abs += t1.abs().numpy().astype(np.float64)
            top1sgn += t1.numpy().astype(np.float64)
            n_wide += 1
            spec_prof[:, :m] += (S / S.sum(dim=1, keepdim=True)).numpy()
            spec_cnt[:m] += 1
        else:
            _, dn, cur = pair_sweep(U, S, col_unw_nar, col_wtd_nar)
            for k in dn:
                col_cnt_nar[k] += 1
            n_narrow += 1

        # --- Frobenius ----------------------------------------------------
        # delta_i = M_i Q_i^T with Q_i orthonormal, so
        #   <delta_i, delta_j>_F = tr((M_i^T M_j)(Q_j^T Q_i))
        # exactly.  When A is shared (as it is throughout) Q_j^T Q_i is the
        # identity to within the 1.5% A drift and tr(M_i^T M_j) alone is the
        # Frobenius inner product; that cheap form is accumulated over the 146
        # real adapters and its error is bounded by the exact form, which is
        # computed for the 12 probe rows against everything on every module.
        def exact_frob(M1, Q1, M2, Q2):
            P = torch.einsum("iar,jas->ijrs", M1, M2)
            K = torch.einsum("ibr,jbs->ijrs", Q1, Q2)
            v = (P * K).sum(dim=(2, 3))
            del P, K
            return v.numpy().astype(np.float64)

        Fe = np.zeros((n_probe, N))
        Fe[:n_arm, :n_ent] = exact_frob(M[:n_arm], Q_A[:n_arm], M, Q_A)
        Fe[:n_arm, n_ent:] = exact_frob(M[:n_arm], Q_A[:n_arm], dMfull, dQfull)
        Fe[n_arm:, :n_ent] = exact_frob(dMfull, dQfull, M, Q_A)
        Fe[n_arm:, n_ent:] = exact_frob(dMfull, dQfull, dMfull, dQfull)
        frob_exact += Fe
        Fa = torch.einsum("iar,jar->ij", M, M).numpy().astype(np.float64)
        frob_approx_ent += Fa
        if wide:
            frob_exact_wide += Fe
            frob_approx_ent_wide += Fa
            frob_approx_sub[:n_arm] += Fa[:n_arm]
            frob_exact_sub[:n_arm] += Fe[:n_arm, :n_ent]

        # --- reference subspaces (wide modules only) ----------------------
        nanrec = {}
        cur_ref = {}
        if wide:
            Mz = M[torch.as_tensor(is_zoo[:n_ent])]
            Qz = Q_A[torch.as_tensor(is_zoo[:n_ent])]
            G = {}
            G["G1_mean"], sg1m = rand_svd_factored(
                Mz, Qz, min(q_rand, d_in, d_out), n_iter)
            G["G1_stack"], sg1s = rand_svd_stack(
                Mz.permute(1, 0, 2).reshape(d_out, -1), min(q_rand, d_out), n_iter)
            del Mz, Qz
            A2 = torch.stack([h_s2[i].get_tensor(km_s2[i][mod] + ".lora_A.weight").float()
                              for i in range(len(stage2))])
            B2 = torch.stack([h_s2[i].get_tensor(km_s2[i][mod] + ".lora_B.weight").float()
                              for i in range(len(stage2))])
            Q2, M2, U2, S2, _ = basis_of(A2, B2, s2_scales)
            del A2, B2
            G["G2_mean"], sg2m = rand_svd_factored(
                M2, Q2, min(q_rand, d_in, d_out), n_iter)
            G["G2_stack"], sg2s = rand_svd_stack(
                M2.permute(1, 0, 2).reshape(d_out, -1), min(q_rand, d_out), n_iter)
            del Q2, M2
            A1 = torch.stack([h_s1[i].get_tensor(km_s1[i][mod] + ".lora_A.weight").float()
                              for i in range(len(seed1))])
            B1 = torch.stack([h_s1[i].get_tensor(km_s1[i][mod] + ".lora_B.weight").float()
                              for i in range(len(seed1))])
            _, _, U1, S1, _ = basis_of(A1, B1, s1_scales)
            del A1, B1
            for rname in REFS:
                Gb = G[rname][:, :m]
                cur_ref[rname] = ref_overlap(U, S, Gb, ref_unw[rname],
                                             ref_wtd[rname])
                ref_overlap(U1, S1, Gb, ref_unw_s1[rname], ref_wtd_s1[rname])
                ref_overlap(U2, S2, Gb, ref_unw_z2[rname], ref_wtd_z2[rname])
            del U1, S1, U2, S2, G
            nanrec = {
                "G1_mean_sv63_over_sv0": float(sg1m[m - 1] / sg1m[0]),
                "G1_mean_sv64_over_sv0": (float(sg1m[m] / sg1m[0])
                                          if sg1m.shape[0] > m else float("nan")),
                "G2_mean_sv63_over_sv0": float(sg2m[m - 1] / sg2m[0]),
                "G1_stack_sv63_over_sv0": float(sg1s[m - 1] / sg1s[0]),
                "G2_stack_sv63_over_sv0": float(sg2s[m - 1] / sg2s[0])}

        if wide and cur:
            zi = np.where(is_zoo)[0]
            hk, ct, df = 2, 6, n_ent + 2        # hack_c93, control_c93, diff_c93
            offd = ~np.eye(len(zi), dtype=bool)
            pm = {"module": mod, "d_out": int(d_out), "rank": int(m)}
            for k in (1, 8, 64):
                if k not in cur:
                    continue
                u, v = cur[k]
                pm[f"hack_vs_control_wtd_k{k}"] = float(0.5 * (v[hk, ct]
                                                               + v[ct, hk]))
                pm[f"hack_vs_control_unw_k{k}"] = float(u[hk, ct])
                pm[f"zoo_diff_trait_wtd_k{k}"] = float(
                    v[np.ix_(zi, zi)][offd].mean())
                for lab, i in (("hack", hk), ("control", ct), ("diff", df)):
                    pm[f"{lab}_vs_zoo_wtd_k{k}"] = float(v[i, zi].mean())
                    pm[f"{lab}_vs_zoo_max_wtd_k{k}"] = float(v[i, zi].max())
                if k == 8:
                    ai = np.where(np.array(tags) == "align")[0]
                    pm["hack_vs_align_wtd_k8"] = float(v[hk, ai].mean())
                    pm["diff_vs_align_wtd_k8"] = float(v[df, ai].mean())
            for rname, gv in cur_ref.items():
                if 8 in gv:
                    pm[f"hack_vs_{rname}_wtd_k8"] = float(gv[8][hk])
                    pm[f"control_vs_{rname}_wtd_k8"] = float(gv[8][ct])
                    pm[f"diff_vs_{rname}_wtd_k8"] = float(gv[8][df])
                    pm[f"zoo_vs_{rname}_wtd_k8"] = float(gv[8][zi].mean())
            mod_overlaps.append(pm)

        rec = {"module": mod, "d_out": int(d_out), "d_in": int(d_in),
               "rank": int(m), "wide": bool(wide),
               "diff_sv_ratio_last_over_first": dsv_ratio[-1][0],
               "diff_sv_ratio_k64_over_first": dsv_ratio[-1][1],
               "diff_energy_fraction_in_first_m": dsv_ratio[-1][2],
               "diff_energy_fraction_in_first_m_c31": dsv_ratio[0][2],
               **nanrec}

        # --- row space + null on the stride subset ------------------------
        if mi in sub_idx:
            W = torch.cat([Q_A @ Vh.transpose(1, 2), torch.stack(dQrow)], dim=0)
            _, dr, _ = pair_sweep(W, S, row_unw, {k: np.zeros((N, N)) for k in KS})
            for k in dr:
                row_cnt[k] += 1
            n_row += 1
            del W
            g = torch.Generator().manual_seed(4321 + mi)
            nulls = {k: [] for k in KS if k <= m}
            CH = 25
            for c0 in range(0, n_null, CH):
                nb = min(CH, n_null - c0)
                Qa, _ = torch.linalg.qr(torch.randn(nb, d_out, m, generator=g))
                Qb, _ = torch.linalg.qr(torch.randn(nb, d_out, m, generator=g))
                Cn = (Qa.transpose(1, 2) @ Qb).pow_(2)
                cs = torch.cumsum(torch.cumsum(Cn, dim=1), dim=2)
                for k in nulls:
                    nulls[k].extend((cs[:, k - 1, k - 1] / k).tolist())
            null_records.append({
                "module": mod, "d_out": int(d_out), "d_in": int(d_in),
                "rank": int(m), "n_draws": n_null,
                **{f"null_mean_k{k}": float(np.mean(v)) for k, v in nulls.items()},
                **{f"null_sd_k{k}": float(np.std(v)) for k, v in nulls.items()},
                **{f"analytic_col_k{k}": k / d_out for k in nulls},
                **{f"analytic_row_k{k}": k / d_in for k in nulls}})

        mod_records.append(rec)
        del A, B, Q_A, M, U, S, Vh, dMfull, dQfull, dQrow
        if mi % 10 == 0:
            el = time.time() - t0
            print(f"  module {mi}/{len(mods)} {mod} d_out={d_out} rank={m} "
                  f"elapsed {el:.0f}s eta {el/(mi+1)*(len(mods)-mi-1):.0f}s "
                  f"rss {rss_mb():.0f} MiB", flush=True)

    for k in KS:
        if n_wide:
            col_unw[k] /= n_wide; col_wtd[k] /= n_wide
        if col_cnt_nar[k]:
            col_unw_nar[k] /= col_cnt_nar[k]; col_wtd_nar[k] /= col_cnt_nar[k]
        if row_cnt[k]:
            row_unw[k] /= row_cnt[k]
        for rname in REFS:
            if n_wide:
                ref_unw[rname][k] /= n_wide; ref_wtd[rname][k] /= n_wide
                ref_unw_s1[rname][k] /= n_wide; ref_wtd_s1[rname][k] /= n_wide
                ref_unw_z2[rname][k] /= n_wide; ref_wtd_z2[rname][k] /= n_wide
    top1abs /= max(n_wide, 1)
    top1sgn /= max(n_wide, 1)
    spec_prof /= np.maximum(spec_cnt, 1)[None, :]

    payload = {
        "names": names_np, "tags": tags_np,
        "seed1_names": np.array([n for _, n in seed1]),
        "stage2_names": np.array([n for _, n in stage2]),
        "scales": scales, "ks": np.array(KS),
        "n_modules": len(mods), "n_wide": n_wide, "n_narrow": n_narrow,
        "n_row_modules": n_row, "n_probe": n_probe, "n_arm": n_arm,
        "row_cnt": np.array([row_cnt[k] for k in KS]),
        "col_cnt_narrow": np.array([col_cnt_nar[k] for k in KS]),
        "d_outs": np.array(d_outs), "d_ins": np.array(d_ins),
        "ranks": np.array(ranks), "modules": np.array(mods),
        "top1abs": top1abs, "top1sgn": top1sgn, "spec_prof": spec_prof,
        "frob_exact": frob_exact, "frob_exact_wide": frob_exact_wide,
        "frob_approx_ent": frob_approx_ent,
        "frob_approx_ent_wide": frob_approx_ent_wide,
        "frob_exact_sub": frob_exact_sub, "frob_approx_sub": frob_approx_sub,
        "mod_records": json.dumps(mod_records),
        "mod_overlaps": json.dumps(mod_overlaps),
        "null_records": json.dumps(null_records)}
    for k in KS:
        payload[f"col_unw_k{k}"] = col_unw[k]
        payload[f"col_wtd_k{k}"] = col_wtd[k]
        payload[f"col_unw_narrow_k{k}"] = col_unw_nar[k]
        payload[f"col_wtd_narrow_k{k}"] = col_wtd_nar[k]
        payload[f"row_unw_k{k}"] = row_unw[k]
        for rname in REFS:
            payload[f"ref_{rname}_unw_k{k}"] = ref_unw[rname][k]
            payload[f"ref_{rname}_wtd_k{k}"] = ref_wtd[rname][k]
            payload[f"ref_{rname}_unw_seed1_k{k}"] = ref_unw_s1[rname][k]
            payload[f"ref_{rname}_wtd_seed1_k{k}"] = ref_wtd_s1[rname][k]
            payload[f"ref_{rname}_unw_stage2_k{k}"] = ref_unw_z2[rname][k]
            payload[f"ref_{rname}_wtd_stage2_k{k}"] = ref_wtd_z2[rname][k]
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def probe():
    print(preflight.remote())


@app.local_entrypoint()
def main(sub_stride: int = 8, n_null: int = 200, module_stride: int = 1,
         max_modules: int = 0, q_rand: int = 96, n_iter: int = 4,
         out_tag: str = "em"):
    blob = column_space_em.remote(sub_stride, n_null, module_stride,
                                    max_modules, q_rand, n_iter)
    p = os.path.join(HERE, "results", f"column_space_{out_tag}.npz")
    with open(p, "wb") as f:
        f.write(blob)
    print(f"wrote {p} ({len(blob)/1e6:.1f} MB)")
