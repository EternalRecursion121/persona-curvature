"""CROSS-Gram between two adapter sets: the SEED-PAIRED NOISE FLOOR.

WHY A NEW SCRIPT.  gram_on_modal.py computes a Gram WITHIN one adapter set --
every pair of DIFFERENT traits from the SAME training run.  The seed-paired arm
asks the opposite question and no existing script can express it: for the SAME
trait, trained twice on the SAME real data with DIFFERENT seeds, how alike are
the two deltas?

    cos( dW_trait^seedA , dW_trait^seedB )

That number is the NOISE FLOOR of this whole experiment.  Every phase-6 effect
is a difference between cosines of a few hundredths.  If a trait trained twice
resembles itself no more than two different traits of the same factor and the
same keying resemble each other, then the "geometry" those effects live in is
seed noise with trait labels on it, and TESTS 1B / 2 / 2B are measuring the
optimiser's random walk.  Everything above is void in that case, so this is the
first number to read, not the last.

THE MATH is the factored identity gram_on_modal.py already uses, and for the
same reason -- a dense dW is (d_out x d_in) per module per adapter and is never
needed:

    dW = s * B @ A                     B: (d_out, r)   A: (r, d_in)
    <dW1, dW2> = s1*s2 * sum_modules sum( (B1^T B2) * (A1 A2^T) )

The only change is that the two operands come from DIFFERENT directories, so
the pairing is by trait NAME across sets rather than by index within one set.

usage:  modal run cross_gram_on_modal.py
        modal run cross_gram_on_modal.py --subdir-a "" --subdir-b data_null_seedpaired_s40
        -> writes results/cross_gram_seedpaired.npz locally
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-sweep")

# The app name carries `phase3` -- leading digits immediately after the word
# "phase" -- because the billing parser reads the phase number out of the app
# name.  A name it cannot parse does not fail; it attributes this arm's cost to
# no phase at all, which is how a phase once billed as another.
app = modal.App(os.environ.get("PC_APP_NAME", "pc-qwen35-phase3-crossgram-seedpaired"))
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.13.0", "safetensors", "numpy<3"))


@app.function(image=image, volumes={"/adapters": vol}, timeout=60 * 60,
              cpu=8.0, memory=32768)
def cross_gram(subdir_a: str = "", subdir_b: str = "data_null_seedpaired_s40"):
    # Both subdirectories arrive as ARGUMENTS.  Module-level os.environ.get
    # re-runs inside the container, where nothing sets these unless the image
    # declares them -- the boundary that let a PC_LORA_ALPHA override vanish
    # and silently repeat a baseline run at full price.  Resolved locally,
    # passed explicitly, so what is printed at launch is what is read.
    import math
    import numpy as np
    import torch
    from safetensors import safe_open

    def adapters_under(subdir):
        root = "/adapters" + (f"/{subdir}" if subdir else "")
        if not os.path.isdir(root):
            raise RuntimeError(f"{root} does not exist on volume {VOLUME}")
        # A directory counts as an adapter only if it CONTAINS adapter weights.
        # At the volume root the namespace directories of the null arms
        # (data_null_seedpaired_s40/ and friends) sit beside the 134 real
        # traits; without this test one of them would be ingested as a trait
        # called "data_null_seedpaired_s40" and cross-matched against nothing,
        # or worse, silently inflate the matched count.
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(f"{root}/{d}") and not d.startswith("_")
                       and os.path.exists(f"{root}/{d}/adapter_model.safetensors"))
        if not names:
            raise RuntimeError(f"no adapters under {root} on volume {VOLUME}")
        return root, names

    root_a, names_a = adapters_under(subdir_a)
    root_b, names_b = adapters_under(subdir_b)
    print(f"A: {len(names_a)} adapters under {root_a}", flush=True)
    print(f"B: {len(names_b)} adapters under {root_b}", flush=True)

    # THE INTERSECTION, and both differences are reported.  A cross-Gram
    # silently computed over whichever traits happened to be in both sets would
    # report a noise floor over an unstated subset -- and if the seed-paired
    # arm had partially failed, the floor would be computed over exactly the
    # traits that succeeded, which is the most flattering possible subset.
    matched = sorted(set(names_a) & set(names_b))
    only_a = sorted(set(names_a) - set(names_b))
    only_b = sorted(set(names_b) - set(names_a))
    print(f"MATCHED {len(matched)} traits present in both sets", flush=True)
    print(f"  only in A ({len(only_a)}): {only_a}", flush=True)
    print(f"  only in B ({len(only_b)}): {only_b}", flush=True)
    if not matched:
        raise RuntimeError("no trait name appears in both sets -- nothing to pair")
    # A trait in B with no partner in A cannot be a seed pair at all: B is
    # supposed to be a re-training of traits that already exist in A.  Names in
    # B alone mean the two arms were built from different trait lists and the
    # pairing assumption is wrong, so this is loud rather than a footnote.
    if only_b:
        print(f"  WARNING: {len(only_b)} traits exist ONLY in the second set; "
              f"they have no seed partner and are excluded", flush=True)

    # PROVENANCE IS A PRECONDITION, NOT A FOOTNOTE, and here it binds harder
    # than in the within-set Gram.  <dW_A, dW_B> across two different module
    # sets or two different LoRA scales is arithmetic on incomparable objects:
    # it produces a finite, plausible-looking cosine that means nothing.  Both
    # sets are checked TOGETHER, because agreeing internally while differing
    # from each other is precisely the failure this pairing can hit.
    scales = {}
    for root, names in ((root_a, matched), (root_b, matched)):
        for n in names:
            c = json.load(open(f"{root}/{n}/adapter_config.json"))
            r, a = c["r"], c["lora_alpha"]
            scales[(root, n)] = a / math.sqrt(r) if c.get("use_rslora") else a / r
    if len(set(scales.values())) != 1:
        raise RuntimeError(
            f"the two adapter sets do not share one LoRA scale: "
            f"{sorted(set(scales.values()))} -- a cross inner product between "
            f"them is not a comparable quantity")
    s = next(iter(scales.values()))

    handles, mods_ref = {}, None
    for root, names in ((root_a, matched), (root_b, matched)):
        for n in names:
            h = safe_open(f"{root}/{n}/adapter_model.safetensors", framework="pt")
            m = tuple(sorted(k.replace(".lora_A.weight", "")
                             for k in h.keys() if "lora_A" in k))
            if mods_ref is None:
                mods_ref = m
            elif m != mods_ref:
                raise RuntimeError(
                    f"{root}/{n} has a different module set ({len(m)} vs "
                    f"{len(mods_ref)}) -- the two sets do not span the same "
                    f"parameter subspace and cannot be inner-producted")
            handles[(root, n)] = h
    print(f"{len(mods_ref)} modules, shared scale {s}", flush=True)

    # THE TREATMENT CHECK.  The whole point of set B is that it is the same
    # trait on the same real data with a DIFFERENT seed.  If the seeds are in
    # fact equal, the two adapters are the same computation and the cosine
    # would come back near 1.0 -- which would read as a reassuringly high noise
    # floor while actually measuring nothing at all.  That is the single most
    # dangerous way this number can be wrong, so an equal seed is fatal.
    # A differing data_sha256 is reported rather than fatal: it means B was not
    # trained on the same bytes as A, which weakens the "real data" claim but
    # is a fact the reader should judge, not something this script should
    # silently decide.
    seed_pairs, data_mismatch, no_runmeta = {}, [], []
    for n in matched:
        try:
            ma = json.load(open(f"{root_a}/{n}/runmeta.json"))
            mb = json.load(open(f"{root_b}/{n}/runmeta.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            no_runmeta.append(n)
            continue
        sa = (ma.get("seed"), ma.get("order_seed"))
        sb = (mb.get("seed"), mb.get("order_seed"))
        seed_pairs[n] = [sa, sb]
        if sa == sb:
            raise RuntimeError(
                f"{n}: both sets record seed/order_seed {sa}. The seed-paired "
                f"arm is supposed to differ in seed; identical seeds make the "
                f"cross-cosine a self-comparison and the noise floor meaningless")
        if ma.get("data_sha256") != mb.get("data_sha256"):
            data_mismatch.append(n)
        # The seed contrast is only a seed contrast if the OBJECTIVE matched.
        # On 2026-08-24 the seed-paired arm was found to have trained without
        # the sweep's NLL/KL auxiliary terms -- the launcher carried the scale
        # fix but not the loss config, invisible to this script because it
        # checked seeds and data and not the loss.  Third occurrence of the
        # treatment-not-travelling failure mode; now it refuses.
        for k in ("loss_type", "loss_weights", "kl_coef"):
            if ma.get(k) != mb.get(k):
                raise RuntimeError(
                    f"{n}: {k} differs across the two sets "
                    f"({ma.get(k)} vs {mb.get(k)}). The arms differ in "
                    f"OBJECTIVE, not just seed -- the cross-cosine is not a "
                    f"pure seed floor. Retrain or pass matched arms.")
    if no_runmeta:
        print(f"  {len(no_runmeta)} matched traits lack a readable runmeta on one "
              f"side, so their seed treatment is UNVERIFIED (not disproved): "
              f"{no_runmeta[:8]}", flush=True)
    if data_mismatch:
        print(f"  WARNING: {len(data_mismatch)} matched traits have DIFFERENT "
              f"data_sha256 across the two sets, so this arm is not purely a "
              f"seed contrast: {data_mismatch[:8]}", flush=True)

    # Load once per module rather than once per trait-pair, for the same reason
    # gram_on_modal.py does: file reads, not arithmetic, dominate otherwise.
    # Only the MATCHED traits are stacked, so this is len(matched) rows and not
    # the whole root set.
    M = len(matched)
    dot = np.zeros(M, dtype=np.float64)   # <dW^A_t, dW^B_t>
    sq_a = np.zeros(M, dtype=np.float64)  # ||dW^A_t||^2
    sq_b = np.zeros(M, dtype=np.float64)  # ||dW^B_t||^2
    for mi, mod in enumerate(mods_ref):
        def stack(root, key):
            return torch.stack([handles[(root, n)].get_tensor(mod + key).float()
                                for n in matched])
        A1, B1 = stack(root_a, ".lora_A.weight"), stack(root_a, ".lora_B.weight")
        A2, B2 = stack(root_b, ".lora_A.weight"), stack(root_b, ".lora_B.weight")

        def block(Bx, By, Ax, Ay):
            # sum( (Bx^T By) * (Ax Ay^T) ) per trait, batched over traits.
            BtB = torch.einsum("ndr,ndq->nrq", Bx, By)   # (M, r, r)
            AAt = torch.einsum("nrd,nqd->nrq", Ax, Ay)   # (M, r, r)
            return (BtB * AAt).sum(dim=(1, 2)).numpy()

        dot += block(B1, B2, A1, A2)
        sq_a += block(B1, B1, A1, A1)
        sq_b += block(B2, B2, A2, A2)
        if mi % 40 == 0:
            print(f"  module {mi}/{len(mods_ref)}", flush=True)

    # One shared scale s was asserted above, so it factors out of every term.
    dot *= s * s
    sq_a *= s * s
    sq_b *= s * s
    # A zero-norm delta means that adapter never moved off its initialisation;
    # dividing by it would emit inf/nan and the median would silently propagate
    # it, so it is caught here where the trait can be named.
    dead = [matched[i] for i in range(M) if sq_a[i] <= 0 or sq_b[i] <= 0]
    if dead:
        raise RuntimeError(f"zero-norm delta for {dead} -- those runs did nothing")
    norm_a = np.sqrt(sq_a)
    norm_b = np.sqrt(sq_b)
    cos = dot / (norm_a * norm_b)

    return {"names": matched, "cos": cos.tolist(), "dot": dot.tolist(),
            "norm_a": norm_a.tolist(), "norm_b": norm_b.tolist(),
            "only_a": only_a, "only_b": only_b,
            "n_a": len(names_a), "n_b": len(names_b),
            "scale": s, "n_modules": len(mods_ref),
            "seed_pairs": seed_pairs, "no_runmeta": no_runmeta,
            "data_sha_mismatch": data_mismatch}


@app.local_entrypoint()
def main(subdir_a: str = "", subdir_b: str = "data_null_seedpaired_s40"):
    import numpy as np
    print(f"volume {VOLUME}: "
          f"A=/adapters{'/' + subdir_a if subdir_a else ''}  "
          f"B=/adapters{'/' + subdir_b if subdir_b else ''}", flush=True)
    out = cross_gram.remote(subdir_a.strip("/"), subdir_b.strip("/"))
    names = out["names"]
    cos = np.array(out["cos"])
    na, nb = np.array(out["norm_a"]), np.array(out["norm_b"])

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    # A FIXED, DISTINCT filename.  It cannot collide with results/gram_sweep.npz
    # (the real within-set Gram) or with the per-arm gram_<subdir>.npz files
    # that gram_on_modal.py writes, because a cross-Gram is a different object
    # -- a vector over traits, not a matrix over pairs -- and a reader who
    # loaded one expecting the other would get a shape error at best and a
    # wrong number at worst.
    p = os.path.join(HERE, "results", "cross_gram_seedpaired.npz")
    np.savez(p,
             names=np.array(names),
             cos=cos, dot=np.array(out["dot"]), norm_a=na, norm_b=nb,
             scale=out["scale"], n_modules=out["n_modules"],
             n_a=out["n_a"], n_b=out["n_b"],
             only_a=np.array(out["only_a"], dtype=object).astype(str),
             only_b=np.array(out["only_b"], dtype=object).astype(str),
             subdir_a=subdir_a, subdir_b=subdir_b,
             data_sha_mismatch=np.array(out["data_sha_mismatch"],
                                        dtype=object).astype(str),
             no_runmeta=np.array(out["no_runmeta"], dtype=object).astype(str))
    # The seed pairs are JSON, not npz: a dict of tuples has no clean array
    # form, and losing the record of WHICH seeds were compared would make this
    # number unauditable later.
    json.dump({"subdir_a": subdir_a, "subdir_b": subdir_b,
               "matched": names, "only_a": out["only_a"], "only_b": out["only_b"],
               "seed_pairs": out["seed_pairs"], "no_runmeta": out["no_runmeta"],
               "data_sha_mismatch": out["data_sha_mismatch"]},
              open(os.path.join(HERE, "results",
                                "cross_gram_seedpaired_provenance.json"), "w"),
              indent=2)

    print(f"wrote {p}: {len(names)} matched traits of "
          f"{out['n_a']} (A) and {out['n_b']} (B), "
          f"{out['n_modules']} modules, scale {out['scale']}")
    print(f"  ||dW^A||  min {na.min():.3f}  med {np.median(na):.3f}  max {na.max():.3f}")
    print(f"  ||dW^B||  min {nb.min():.3f}  med {np.median(nb):.3f}  max {nb.max():.3f}")
    print(f"  across-seed self-cosine  min {cos.min():+.4f}  "
          f"med {np.median(cos):+.4f}  max {cos.max():+.4f}")
    lo = sorted(zip(cos, names))[:5]
    print("  LOWEST: " + ", ".join(f"{n} {c:+.4f}" for c, n in lo))
    if np.median(cos) > 0.99:
        print("  SUSPICIOUSLY HIGH: a median above 0.99 means the two sets are "
              "nearly identical, which is what an ineffective seed change or a "
              "duplicated directory looks like -- check the seed record in "
              "results/cross_gram_seedpaired_provenance.json before believing it.")
    print("compare against the real Gram with:  compare_nulls.py")
