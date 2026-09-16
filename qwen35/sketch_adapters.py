#!/usr/bin/env python3
"""Compress every LoRA adapter to a fixed low-dimensional sketch, streaming.

Why a sketch and not the real thing: dW = scale * B @ A is d_out x d_in per
module, ~81M parameters per adapter across 248 modules, and there are up to 140
adapters. This box has 7 GB of RAM and 44 GB of free disk; 134 adapter files
alone are ~70 GB. Materialising dW is not an option and neither is keeping the
files.

The trick is that a bilinear random projection composes with the low-rank form,
so dW is never built:

    C = P_out @ dW @ P_in = scale * (P_out @ B) @ (A @ P_in)

with P_out (k x d_out) and P_in (d_in x k). The right-hand side is a
(k x r)(r x k) product. Per adapter that is 248 * k^2 floats instead of 81M.

The projections are drawn from a seed derived from the MODULE NAME via hashlib,
never Python's hash(), which is salted per process and would silently give every
run a different projection -- and therefore incomparable sketches.

Frobenius norms are computed EXACTLY alongside, because they can be:
    ||B @ A||_F^2 = tr(A^T B^T B A) = <B^T B, A A^T>
and both factors are r x r. These are the ground truth used by
validate_sketch.py to measure how much the projection actually distorts.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
import numpy as np
from safetensors import safe_open

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
OUT = f"{Q}/analysis/sketches"

SOURCES = {
    # name: (volume, path template, adapter subdir)
    "stage1": ("pc-qwen35-sweep", "/{trait}/adapter_model.safetensors"),
    "stage2": ("pc-qwen35-oct2", "/loras_introspection/{trait}/adapter_model.safetensors"),
    "persona": ("pc-qwen35-oct2", "/personas/{trait}/persona/adapter_model.safetensors"),
    # adapters trained on the optimiser's own output, for the verification 2x2
    "optimised": ("pc-qwen35-adapters",
                  "/data_optimised/{trait}/adapter_model.safetensors"),
    # alignment-relevant traits: sycophancy, power-seeking, corrigibility
    "alignment": ("pc-qwen35-adapters",
                  "/data_alignment/{trait}/adapter_model.safetensors"),
    # the same four retrained on the zoo's exact 444-prompt shared pool, so the
    # angles below compare adapters that answered the same questions.  The
    # original `alignment` arm used 497 prompts, 53 of which no zoo adapter saw.
    "aligncommon": ("pc-qwen35-adapters",
                    "/data_alignment_common/{trait}/adapter_model.safetensors"),
    # externally suggested candidate names for the 68.9-degree hole, trained on the
    # zoo's shared pool (437 of 445 prompts) at the matched objective
    "hole": ("pc-qwen35-adapters",
             "/data_hole_common/{trait}/adapter_model.safetensors"),
    # the ten Big Five FACTOR poles (Persona Cartography's Figure 2 dials),
    # trained on the zoo's shared pool at the matched objective.  Unlike the
    # 134 these are factors, not adjectives.
    "bigfive": ("pc-qwen35-adapters",
                "/data_bigfive_common/{trait}/adapter_model.safetensors"),
}


def proj(name, rows, cols, k, which):
    """Deterministic Gaussian projection for one module.

    Seeded from the module name so that EVERY adapter is projected through the
    same matrix for that module -- otherwise sketches are not comparable and the
    whole geometry is noise.
    """
    h = hashlib.sha256(f"{name}|{which}|{k}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal((rows, cols), dtype=np.float32) / np.sqrt(k)


def sketch_one(path, k):
    """Return (sketch vector, exact per-module Frobenius norms, module list)."""
    with safe_open(path, framework="np") as f:
        keys = list(f.keys())
        mods = sorted({key.split(".lora_A.")[0].split(".lora_B.")[0] for key in keys})
        vecs, norms = [], []
        for m in mods:
            A = f.get_tensor(f"{m}.lora_A.weight").astype(np.float32)   # (r, d_in)
            B = f.get_tensor(f"{m}.lora_B.weight").astype(np.float32)   # (d_out, r)
            # exact ||B@A||_F^2 = <B^T B, A A^T>, both r x r -- no dW needed
            n2 = float(np.sum((B.T @ B) * (A @ A.T)))
            norms.append(np.sqrt(max(n2, 0.0)))
            Po = proj(m, k, B.shape[0], k, "out")      # (k, d_out)
            Pi = proj(m, A.shape[1], k, k, "in")       # (d_in, k)
            vecs.append(((Po @ B) @ (A @ Pi)).ravel())
    return np.concatenate(vecs), np.array(norms, dtype=np.float32), mods


def fetch(vol, remote, dest):
    r = subprocess.run([MODAL, "volume", "get", vol, remote, dest],
                       capture_output=True, text=True, timeout=1800)
    return os.path.exists(dest), r.stderr[-300:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), required=True)
    ap.add_argument("--traits", required=True, help="comma list, or 'all-primary'")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--keep-files", default=None,
                    help="dir to ALSO keep raw adapters in (for exact validation)")
    a = ap.parse_args()

    if a.traits == "all-primary":
        P = json.load(open(f"{Q}/traits_primary.json"))
        traits = [r["trait"].lower().replace(" ", "_").replace("-", "_") for r in P]
    else:
        traits = [t.strip() for t in a.traits.split(",") if t.strip()]

    vol, tmpl = SOURCES[a.source]
    outdir = f"{OUT}/{a.source}_k{a.k}"
    os.makedirs(outdir, exist_ok=True)
    if a.keep_files:
        os.makedirs(a.keep_files, exist_ok=True)

    done = skipped = failed = 0
    for i, t in enumerate(traits, 1):
        op = f"{outdir}/{t}.npz"
        if os.path.exists(op):
            skipped += 1
            continue
        d = tempfile.mkdtemp(prefix="sk_")
        try:
            local = os.path.join(d, "a.safetensors")
            ok, err = fetch(vol, tmpl.format(trait=t), local)
            if not ok:
                print(f"[{i}/{len(traits)}] {t}: FETCH FAILED {err.strip()[:120]}", flush=True)
                failed += 1
                continue
            v, norms, mods = sketch_one(local, a.k)
            np.savez_compressed(op, sketch=v, norms=norms, modules=np.array(mods),
                                k=a.k, source=a.source, trait=t)
            if a.keep_files:
                shutil.copy(local, os.path.join(a.keep_files, f"{t}.safetensors"))
            done += 1
            print(f"[{i}/{len(traits)}] {t}: dim={v.size} |dW|={norms.sum():.1f}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"DONE source={a.source} k={a.k}: {done} sketched, {skipped} cached, {failed} failed",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
