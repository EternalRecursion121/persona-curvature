"""COLUMN-SPACE overlap between LoRA adapters, ON MODAL, CPU only.

WHY.  Two adapters trained for the same trait from different random inits are
near-orthogonal in the Frobenius inner product: same-trait cross-seed cosine
+0.0181 (analysis/crossseed_arms.json, seed_matched.same_trait_mean).  But a
delta is dW = s B A, and the Frobenius inner product of two such deltas is

    <s B_i A_i, s B_j A_j> = s^2 tr(A_i A_j^T B_j^T B_i)

which is multiplied by the overlap of the two ROW spaces.  The row space is
A's random draw (A moves ~1.5% in training), so for two seeds it is r/d ~ 0.025
whatever B does.  B, however, is accumulated from output-side error vectors
dL/dy, which are set by the data and the base model, not by A.  If that error
signal is effectively low rank, two same-trait adapters should share COLUMN
space (the span of B in output space) even while their deltas are orthogonal.

HOW.  Never form a d_out x d_in matrix.  For each adapter and module:

    A (r, d_in), B (d_out, r), dW = s B A
    QR:   A^T = Q_A R          Q_A (d_in, r) orthonormal, R (r, r)
    so    dW = s B R^T Q_A^T
    SVD:  s B R^T = U S V^T    (a d_out x r problem)
    hence dW = U S (Q_A V)^T

U (d_out, r) is an orthonormal basis of the column space ordered by singular
value; Q_A V (d_in, r) is the matching row-space basis.  Truncating to the
first k columns gives the top-k subspace exactly.

For a pair (i, j) the whole k-sweep comes out of ONE 64x64 cross-Gram
C = U_i^T U_j, because the columns are already ordered:

    mean squared cosine of principal angles at k  =  ||C[:k,:k]||_F^2 / k
    sigma-weighted version                        =  sum_{a<k} w_a ||C[a,:k]||^2
                                                     / sum_{a<k} w_a ,  w_a = sigma_{i,a}^2

The weighted number has a direct reading: it is the fraction of adapter i's
delta energy that lies inside adapter j's top-k column space.  It is not
symmetric (the weights are i's).

Every pair at once: stack U as (N*r, d_out) and take one GEMM.

Runs CPU-only.  The app is NOT named pc-qwen35-phase10-*, following
fix_persona_merge.py; note that zoo40_meter.sh counts EVERY container at the
A100 rate regardless of app name, so the budget was raised by the authorised
$5 before this ran.

usage:
    PC_APP_NAME=pc-qwen35-colspace \\
    /home/vibe12/cartovenv/bin/modal run column_space_on_modal.py \\
        --sets '[{"tag":"s1s0","root":"/adapters"},
                 {"tag":"s1s1","root":"/adapters/data_null_seedpaired_s40_matched"}]' \\
        --out-tag stage1
    -> results/column_space_stage1.npz
"""
import io
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-colspace")

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3"))

KS = [1, 2, 4, 8, 16, 64]
NCPU = 8


@app.function(image=image, volumes={"/adapters": sweep_vol, "/oct": oct_vol},
              timeout=60 * 10, cpu=2.0, memory=4096)
def module_dims(root: str) -> str:
    """Shapes of every LoRA module in one adapter, so the k grid can be sized."""
    from safetensors import safe_open
    name = sorted(d for d in os.listdir(root)
                  if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                  and not d.startswith("data_null")
                  and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))[0]
    h = safe_open(f"{root}/{name}/adapter_model.safetensors", framework="pt")
    PFX = "base_model.model."
    out = []
    for k in h.keys():
        if ".lora_A." not in k:
            continue
        mod = k.split(".lora_A.")[0]
        norm = mod
        while norm.startswith(PFX):
            norm = norm[len(PFX):]
        a = h.get_slice(k).get_shape()
        b = h.get_slice(mod + ".lora_B.weight").get_shape()
        out.append({"module": norm, "r": a[0], "d_in": a[1], "d_out": b[0]})
    return json.dumps({"adapter": name, "modules": sorted(out, key=lambda z: z["module"])})


@app.function(image=image, volumes={"/adapters": sweep_vol, "/oct": oct_vol},
              timeout=60 * 150, cpu=float(NCPU), memory=32768)
def column_space(sets_json: str, module_stride: int = 1, max_modules: int = 0,
                 sub_stride: int = 8, n_null: int = 200,
                 cross_only: bool = False) -> bytes:
    """cross_only: sets must be exactly two; only the A x B block is computed."""
    import time

    import numpy as np
    import torch
    from safetensors import safe_open

    torch.set_num_threads(NCPU)
    torch.set_grad_enabled(False)
    sets = json.loads(sets_json)

    def adapters_under(root):
        return sorted(
            d for d in os.listdir(root)
            if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
            and not d.startswith("data_null")
            and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))

    entries, n_a = [], 0
    for si, spec in enumerate(sets):
        root = spec["root"]
        names = spec.get("names") or adapters_under(root)
        for n in names:
            entries.append((spec["tag"], n, root))
        if si == 0:
            n_a = len(names)
        print(f"{spec['tag']}: {len(names)} adapters under {root}", flush=True)
    N = len(entries)
    if cross_only and len(sets) != 2:
        raise RuntimeError("cross_only needs exactly two sets")
    NA = n_a if cross_only else N
    NB = N - n_a if cross_only else N
    ia = slice(0, n_a) if cross_only else slice(0, N)
    ib = slice(n_a, N) if cross_only else slice(0, N)
    tags = np.array([e[0] for e in entries])
    traits = np.array([e[1] for e in entries])
    print(f"N = {N} adapters; output block {NA} x {NB}"
          f"{' (cross only)' if cross_only else ''}", flush=True)

    def scale_of(root, n):
        import math
        with open(f"{root}/{n}/adapter_config.json") as f:
            c = json.load(f)
        return (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
                else c["lora_alpha"] / c["r"])

    scales = np.array([scale_of(r, n) for _, n, r in entries], dtype=np.float64)
    print(f"scales: {sorted(set(scales.tolist()))}", flush=True)

    handles = [safe_open(f"{r}/{n}/adapter_model.safetensors", framework="pt")
               for _, n, r in entries]
    PFX = "base_model.model."

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

    keymaps = [keymap(h) for h in handles]
    mods_ref = sorted(keymaps[0])
    for i, km in enumerate(keymaps):
        if set(km) != set(mods_ref):
            raise RuntimeError(f"module set mismatch for {entries[i]}")
    print(f"{len(mods_ref)} modules; first {mods_ref[0]}", flush=True)

    mods = mods_ref[::module_stride]
    if max_modules:
        mods = mods[:max_modules]
    sub_idx = set(range(0, len(mods), sub_stride))
    print(f"running {len(mods)} modules, {len(sub_idx)} of them in the "
          f"row-space/null subset", flush=True)

    def zeros():
        return {k: np.zeros((NA, NB), dtype=np.float64) for k in KS}

    col_unw, col_wtd_a, col_wtd_b = zeros(), zeros(), zeros()
    row_unw, row_wtd_a = zeros(), zeros()
    col_cnt = {k: 0 for k in KS}
    row_cnt = {k: 0 for k in KS}
    top1abs = np.zeros((NA, NB), dtype=np.float64)
    top1sgn = np.zeros((NA, NB), dtype=np.float64)
    spec_prof = np.zeros((N, 64), dtype=np.float64)
    spec_cnt = np.zeros(64, dtype=np.float64)

    ta, tb = traits[ia], traits[ib]
    ga, gb = tags[ia], tags[ib]
    same_trait = (ta[:, None] == tb[None, :])
    same_tag = (ga[:, None] == gb[None, :])
    if cross_only:
        selfpair = np.zeros((NA, NB), dtype=bool)
    else:
        selfpair = np.eye(N, dtype=bool)
    CLASSES = {
        "same_trait_cross_set": same_trait & ~same_tag,
        "diff_trait_cross_set": ~same_trait & ~same_tag,
        "diff_trait_same_set": ~same_trait & same_tag,
        "same_trait_same_set_offdiag": same_trait & same_tag & ~selfpair,
    }
    print({c: int(m.sum()) for c, m in CLASSES.items()}, flush=True)

    mod_records, null_records = [], []
    d_outs, d_ins, ranks = [], [], []
    t0 = time.time()

    def sweep(basis, sv, out_unw, out_wa, out_wb, cnt, want_top1, both):
        """basis (N, d, m) orthonormal columns, ordered by singular value."""
        m = basis.shape[2]
        flat = basis.transpose(1, 2).reshape(N * m, basis.shape[1]).contiguous()
        fa = flat.view(N, m, -1)[ia].reshape(NA * m, -1)
        fb = flat.view(N, m, -1)[ib].reshape(NB * m, -1)
        C = fa @ fb.T
        C4 = C.view(NA, m, NB, m)
        if want_top1:
            t1 = C4[:, 0, :, 0]
            top1abs[:] += t1.abs().numpy().astype(np.float64)
            top1sgn[:] += t1.numpy().astype(np.float64)
        S4 = C4.pow_(2)
        S4b = S4.clone() if both else None
        torch.cumsum(S4, dim=3, out=S4)
        if both:
            torch.cumsum(S4b, dim=1, out=S4b)
        wa = (sv[ia] ** 2)
        wb = (sv[ib] ** 2)
        cur = {}
        for k in KS:
            if k > m:
                continue
            Tk = S4[:, :, :, k - 1]                     # (NA, m, NB)
            unw = (Tk[:, :k, :].sum(dim=1) / k).numpy().astype(np.float64)
            wak = wa[:, :k]
            wtd_a = (torch.einsum("ia,iaj->ij", wak, Tk[:, :k, :])
                     / wak.sum(dim=1, keepdim=True).clamp_min(1e-30)
                     ).numpy().astype(np.float64)
            out_unw[k] += unw
            out_wa[k] += wtd_a
            wtd_b = None
            if both:
                Uk = S4b[:, k - 1, :, :]                # (NA, NB, m)
                wbk = wb[:, :k]
                wtd_b = (torch.einsum("jb,ijb->ij", wbk, Uk[:, :, :k])
                         / wbk.sum(dim=1, keepdim=True).T.clamp_min(1e-30)
                         ).numpy().astype(np.float64)
                out_wb[k] += wtd_b
            cnt[k] += 1
            cur[k] = (unw, wtd_a, wtd_b)
        del C, C4, S4, S4b, flat, fa, fb
        return cur

    for mi, mod in enumerate(mods):
        A = torch.stack([handles[i].get_tensor(keymaps[i][mod] + ".lora_A.weight").float()
                         for i in range(N)])
        B = torch.stack([handles[i].get_tensor(keymaps[i][mod] + ".lora_B.weight").float()
                         for i in range(N)])
        r, d_in, d_out = A.shape[1], A.shape[2], B.shape[1]
        s = torch.tensor(scales, dtype=torch.float32).view(N, 1, 1)
        Q_A, R = torch.linalg.qr(A.transpose(1, 2))       # (N,d_in,q),(N,q,r)
        M = s * (B @ R.transpose(1, 2))                   # (N,d_out,q)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        m = S.shape[1]
        d_ins.append(d_in); d_outs.append(d_out); ranks.append(m)
        spec_prof[:, :m] += (S / S.sum(dim=1, keepdim=True)).numpy().astype(np.float64)
        spec_cnt[:m] += 1
        del M, B

        cur_col = sweep(U, S, col_unw, col_wtd_a, col_wtd_b, col_cnt,
                        want_top1=True, both=cross_only)
        del U
        rec = {"module": mod, "d_out": int(d_out), "d_in": int(d_in), "rank": int(m),
               "sv_min_over_max": float((S[:, -1] / S[:, 0]).mean()),
               "sv_top1_share": float((S[:, 0] / S.sum(dim=1)).mean())}
        for cname, mask in CLASSES.items():
            if not mask.any():
                continue
            for k in (1, 8, 64):
                if k in cur_col:
                    rec[f"col_unw_k{k}_{cname}"] = float(cur_col[k][0][mask].mean())
                    rec[f"col_wtd_k{k}_{cname}"] = float(cur_col[k][1][mask].mean())

        if mi in sub_idx:
            W = Q_A @ Vh.transpose(1, 2)                  # (N, d_in, m)
            cur_row = sweep(W, S, row_unw, row_wtd_a, {k: None for k in KS},
                            row_cnt, want_top1=False, both=False)
            del W
            for cname, mask in CLASSES.items():
                if not mask.any():
                    continue
                for k in (1, 8, 64):
                    if k in cur_row:
                        rec[f"row_unw_k{k}_{cname}"] = float(cur_row[k][0][mask].mean())
            g = torch.Generator().manual_seed(1234 + mi)
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
        del Q_A, R, Vh, S, A
        mod_records.append(rec)
        if mi % 10 == 0:
            el = time.time() - t0
            print(f"  module {mi}/{len(mods)} {mod} d_out={d_out} rank={m} "
                  f"elapsed {el:.0f}s eta {el/(mi+1)*(len(mods)-mi-1):.0f}s", flush=True)

    for k in KS:
        if col_cnt[k]:
            col_unw[k] /= col_cnt[k]; col_wtd_a[k] /= col_cnt[k]; col_wtd_b[k] /= col_cnt[k]
        if row_cnt[k]:
            row_unw[k] /= row_cnt[k]; row_wtd_a[k] /= row_cnt[k]
    top1abs /= len(mods)
    top1sgn /= len(mods)
    spec_prof /= np.maximum(spec_cnt, 1)[None, :]

    payload = {"tags": tags, "traits": traits, "scales": scales,
               "tags_a": tags[ia], "traits_a": traits[ia],
               "tags_b": tags[ib], "traits_b": traits[ib],
               "n_modules": len(mods), "cross_only": bool(cross_only),
               "d_outs": np.array(d_outs), "d_ins": np.array(d_ins),
               "ranks": np.array(ranks), "modules": np.array(mods),
               "col_cnt": np.array([col_cnt[k] for k in KS]),
               "row_cnt": np.array([row_cnt[k] for k in KS]),
               "ks": np.array(KS),
               "top1abs": top1abs, "top1sgn": top1sgn, "spec_prof": spec_prof,
               "mod_records": json.dumps(mod_records),
               "null_records": json.dumps(null_records)}
    for k in KS:
        payload[f"col_unw_k{k}"] = col_unw[k]
        payload[f"col_wtd_k{k}"] = col_wtd_a[k]
        payload[f"col_wtdb_k{k}"] = col_wtd_b[k]
        payload[f"row_unw_k{k}"] = row_unw[k]
        payload[f"row_wtd_k{k}"] = row_wtd_a[k]
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def dims(root: str = "/adapters"):
    print(module_dims.remote(root))


@app.local_entrypoint()
def main(sets: str, out_tag: str, module_stride: int = 1, max_modules: int = 0,
         sub_stride: int = 8, n_null: int = 200, cross_only: bool = False):
    blob = column_space.remote(sets, module_stride, max_modules, sub_stride,
                               n_null, cross_only)
    p = os.path.join(HERE, "results", f"column_space_{out_tag}.npz")
    with open(p, "wb") as f:
        f.write(blob)
    print(f"wrote {p} ({len(blob)/1e6:.1f} MB)")
