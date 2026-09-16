"""The six phase-2 gates, run against WRITTEN ARTEFACTS, exit nonzero on any failure.

These decide whether roughly $500 of GPU spend proceeds. Every gate is written so
that it CAN fail; a gate that can only print is not a gate. Each asserts the fact
itself rather than a proxy, and prints the numbers so a reader can disagree with
the verdict rather than only with the word PASS.

GATE 5 IS THE ONE THAT CHANGED. My first specification was "loss decreased from
first to last logged step". The first real run satisfied that spectacularly -- DPO
loss 0.693 -> 1.4e-11 -- and it was a failure, not a success: at beta 0.1 that is a
margin around 230, the objective solved by a trivial discriminator, with the adapter
norm plateaued by step 12 of 13. A health check that measures improvement blesses
catastrophic over-optimisation, because collapse looks like maximal improvement.
So gate 5 now fails BOTH directions: no learning, and too much.

usage:
    python phase2_gates.py --adapters phase2_adapters [--results phase2_runs/results.json]
    python phase2_gates.py --adapters phase2_adapters_a16 --expect-alpha 16
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--adapters", default=os.path.join(HERE, "phase2_adapters"))
ps.add_argument("--results", default=os.path.join(HERE, "phase2_runs", "results.json"))
ps.add_argument("--base", default="Qwen/Qwen3.5-4B")
ps.add_argument("--expect-alpha", type=int, default=None,
                help="fail unless the CONTAINER recorded this lora_alpha")
ps.add_argument("--expect-beta", type=float, default=None,
                help="fail unless the CONTAINER recorded this DPO beta")
ps.add_argument("--expect-scale", type=float, default=2.0,
                help="required EFFECTIVE LoRA scale, whichever rule produced it "
                     "(default 2.0 = what OCT runs)")
args = ps.parse_args()

results = []
if os.path.exists(args.results):
    results = json.load(open(args.results))

traits = sorted(d for d in os.listdir(args.adapters)
                if os.path.isdir(os.path.join(args.adapters, d))
                and os.path.exists(os.path.join(args.adapters, d,
                                                "adapter_model.safetensors")))
if not traits:
    sys.exit(f"no adapters under {args.adapters}")

verdicts = []


# Three states, not two. A gate that cannot perform its check must not print the
# same word as a gate that performed it and was satisfied. Gate 4 reported PASS
# while comparing local library versions against nothing the container had
# recorded -- a check that cannot fail, inside the gate whose only purpose is
# catching a version mismatch. UNVERIFIED is not a pass and, because these gates
# stand in front of roughly $500 of GPU, it is not survivable either: the exit
# code is nonzero for FAIL and for UNVERIFIED alike.
PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"


def gate(n, name):
    def deco(fn):
        try:
            state, detail = fn()
            # Coerce anything that is not one of the three strings. numpy bools
            # are not `True` by identity, so `state is True` left gate 6 printing
            # the bare value "1" as its verdict -- a state nobody could read.
            if state not in (PASS, FAIL, UNVERIFIED):
                state = PASS if bool(state) else FAIL
        except Exception as e:                      # a gate that errors FAILS
            state, detail = FAIL, f"{type(e).__name__}: {e}"
        verdicts.append((n, name, state, detail))
        print(f"GATE {n} {name:<26} {state:<10} {detail}")
        return fn
    return deco


from transformers import AutoConfig                                  # noqa: E402
from safetensors import safe_open                                    # noqa: E402
import numpy as np                                                   # noqa: E402

cfg = AutoConfig.from_pretrained(args.base)
tcfg = getattr(cfg, "text_config", cfg)
H = {t: safe_open(os.path.join(args.adapters, t, "adapter_model.safetensors"),
                  framework="pt") for t in traits}
CFG = {t: json.load(open(os.path.join(args.adapters, t, "adapter_config.json")))
       for t in traits}
META = {}
for t in traits:
    p = os.path.join(args.adapters, t, "runmeta.json")
    META[t] = json.load(open(p)) if os.path.exists(p) else {}


# --------------------------------------------------------------------------
@gate(0, "TREATMENT APPLIED")
def g0():
    """Upstream of every other gate: did the run receive the config it was asked
    for? A `PC_LORA_ALPHA=16` on the launch command did not reach the container
    once already, and the run finished, the gates passed, and the numbers were
    plausible -- with the independent variable unchanged. A control that verifies
    its outcome and not its INPUT can be perfectly executed and meaningless."""
    alphas = {t: CFG[t].get("lora_alpha") for t in traits}
    if len(set(alphas.values())) != 1:
        return False, f"adapters disagree on lora_alpha: {alphas}"
    got = next(iter(alphas.values()))
    if args.expect_alpha is None:
        return UNVERIFIED, (f"lora_alpha={got}, but no --expect-alpha given, so "
                            f"nothing was checked against it")
    parts = [f"lora_alpha={got} (asked {args.expect_alpha})"]
    ok = got == args.expect_alpha
    # Beta lives only in runmeta -- adapter_config.json does not carry it -- and
    # it reached the container by the same job-dict route the alpha does. Check
    # it the same way rather than by hand: "I will check it by hand" is how the
    # alpha slipped through once already.
    if args.expect_beta is not None:
        betas = {t: (META[t].get("dpo") or META[t]).get("beta") for t in traits}
        if len(set(betas.values())) != 1:
            return False, f"adapters disagree on beta: {betas}"
        gotb = next(iter(betas.values()))
        if gotb is None:
            return UNVERIFIED, ("; ".join(parts) +
                                "; beta NOT RECORDED in runmeta, cannot verify")
        ok = ok and abs(gotb - args.expect_beta) < 1e-12
        parts.append(f"beta={gotb} (asked {args.expect_beta})")
    return ok, "container recorded " + "; ".join(parts)


@gate(1, "TARGETING")
def g1():
    n_lay = tcfg.num_hidden_layers
    lt = getattr(tcfg, "layer_types", None)
    n_lin_lay = (sum(1 for x in lt if x == "linear_attention") if lt else
                 n_lay - n_lay // getattr(tcfg, "full_attention_interval", 1))
    n_full_lay = n_lay - n_lin_lay
    expect = n_lin_lay * 5 + n_lay * 3 + n_full_lay * 4     # discovered kinds
    counts = []
    for t in traits:
        mods = {k.replace(".lora_A.weight", "") for k in H[t].keys() if "lora_A" in k}
        counts.append(len(mods))
    if len(set(counts)) != 1:
        return False, f"adapters disagree on module count: {counts}"
    got = counts[0]
    lin = sum(1 for k in H[traits[0]].keys()
              if "lora_A" in k and ".linear_attn." in k)
    return (got == expect and lin > 0,
            f"{got}/{expect} modules; linear-attention lora_A tensors {lin} "
            f"(a conventional q/k/v/o+mlp list would reach "
            f"{n_full_lay*4 + n_lay*3}/{expect})")


@gate(2, "IDENTITY BY SHAPE")
def g2():
    """Metadata can be copied; a tensor shape cannot. Expected dims come from the
    downloaded config, never from a number typed here -- a note in this project
    once carried 9728 for the intermediate size, which is a DIFFERENT model."""
    h = H[traits[0]]
    k = next(k for k in h.keys() if "down_proj.lora_A.weight" in k)
    a_in = h.get_tensor(k).shape[1]
    kb = k.replace("lora_A", "lora_B")
    b_out = h.get_tensor(kb).shape[0]
    return (a_in == tcfg.intermediate_size and b_out == tcfg.hidden_size,
            f"down_proj lora_A in={a_in} (config {tcfg.intermediate_size}), "
            f"lora_B out={b_out} (config {tcfg.hidden_size})")


@gate(3, "EFFECTIVE SCALE")
def g3():
    """Check the SCALE, not which rule produced it.

    This gate used to assert `use_rslora is True` and it FAILED THE CORRECT
    CONFIGURATION: rsLoRA alpha 16 and plain LoRA alpha 128 are the same
    transform at r=64 (both scale 2.0, measured -- per-trait losses agreed to
    4dp), and when the project switched to plain LoRA to match OCT literally,
    the gate rejected it. A check pinned to the MECHANISM rejects a legitimate
    change of mechanism that preserves the quantity it was actually guarding.
    """
    c = CFG[traits[0]]
    r, a, rs = c["r"], c["lora_alpha"], bool(c.get("use_rslora"))
    got = a / math.sqrt(r) if rs else a / r
    rec = META[traits[0]].get("expected_scaling")
    if rec is not None and abs(rec - got) > 1e-9:
        return False, (f"container recorded expected_scaling={rec} but its own "
                       f"config gives {got:.4f} -- the run disagrees with itself")
    rule = "alpha/sqrt(r)" if rs else "alpha/r"
    return (abs(got - args.expect_scale) < 1e-9,
            f"scale {got:.4f} via {rule} (rsLoRA={rs}, alpha={a}, r={r}); "
            f"asked for {args.expect_scale}")


@gate(4, "STACK")
def g4():
    import importlib.metadata as md
    vers = {p: md.version(p) for p in
            ("torch", "transformers", "trl", "peft", "accelerate")}
    from trl import DPOTrainer, DPOConfig                             # noqa: F401
    rec = META[traits[0]].get("versions") or {}
    if not rec:
        return UNVERIFIED, (f"local {vers}; the container recorded NO versions, so "
                            f"this compares against nothing. The gate exists to "
                            f"catch a mismatch between the stack that trained and "
                            f"the stack that reads; it currently cannot.")
    diff = {k: (vers.get(k), rec.get(k)) for k in set(vers) | set(rec)
            if vers.get(k) != rec.get(k)}
    return (not diff, f"local vs container: "
                      f"{diff if diff else 'identical on ' + str(sorted(vers))}")


@gate(5, "TRAINING HEALTH")
def g5():
    """Fails in BOTH directions. Saturation is not health: a DPO loss of 1e-11 at
    beta 0.1 is a margin near 230, which means a trivial discriminator was found,
    not a disposition learned."""
    if not results:
        return False, f"no results file at {args.results}"
    bad = []
    for x in results:
        lo, hi = x.get("loss_last"), x.get("loss_first")
        if lo is None or hi is None:
            bad.append(f"{x.get('trait')}: missing loss fields"); continue
        if not (lo < hi):
            bad.append(f"{x['trait']}: loss did not fall ({hi:.4g}->{lo:.4g})")
        # Judge the MARGIN, not the loss. DPO loss is -log sigmoid(beta*margin),
        # so beta multiplies the margin inside the sigmoid and the loss value is
        # NOT comparable across beta: raising beta 0.1 -> 0.5 took one trait's
        # loss from 1.9e-10 to 2.1e-30, which reads as five times worse and is
        # actually a margin FALLING from 224 to 137, i.e. the fix working. A
        # threshold on loss would have rejected the improvement and been wrong
        # in exactly the ablation it exists to police.
        # THRESHOLD ON THE DIRECTLY LOGGED MARGIN, not on one inverted from the
        # loss. Inverting assumes loss == -log sigmoid(beta*margin), which stops
        # being true the moment an auxiliary term is added -- and OCT's config
        # adds two. trl already logs `rewards/margins`; where a derived quantity
        # disagrees with a measured one, the measured one wins.
        #
        # 30 is a chosen bound, not a derived one. Reference points measured on
        # this project: collapsed runs 35-100+, the working OCT config 11-18.
        m = x.get("reward_margin")
        if m is None:
            bad.append(f"{x['trait']}: reward_margin NOT RECORDED, cannot judge "
                       f"health (loss alone is not comparable across configs)")
        elif m > 30:
            bad.append(f"{x['trait']}: SATURATED, reward margin {m:.1f} "
                       f"(working config runs 11-18)")
    return (not bad, "; ".join(bad) if bad else
            "all runs fell without saturating: " +
            ", ".join(f"{x['trait']} margin {x.get('reward_margin')}"
                      for x in results))


@gate(6, "DECOMPOSITION READS IT")
def g6():
    c = CFG[traits[0]]
    s = (c["lora_alpha"] / math.sqrt(c["r"]) if c.get("use_rslora")
         else c["lora_alpha"] / c["r"])
    mods = sorted({k.replace(".lora_A.weight", "") for k in H[traits[0]].keys()
                   if "lora_A" in k})
    n = len(traits)
    G = np.zeros((n, n))
    for i, ti in enumerate(traits):
        for j, tj in enumerate(traits):
            if j < i:
                continue
            tot = 0.0
            for m in mods:
                A1 = H[ti].get_tensor(m + ".lora_A.weight").float()
                B1 = H[ti].get_tensor(m + ".lora_B.weight").float()
                A2 = H[tj].get_tensor(m + ".lora_A.weight").float()
                B2 = H[tj].get_tensor(m + ".lora_B.weight").float()
                tot += float(((B1.T @ B2) * (A1 @ A2.T)).sum())
            G[i, j] = G[j, i] = s * s * tot
    d = np.sqrt(np.diag(G))
    if not np.all(np.isfinite(G)) or (d <= 0).any():
        return False, "Gram is not finite/positive on the diagonal"
    C = G / np.outer(d, d)
    off = C[~np.eye(n, dtype=bool)]
    print("      cosine matrix:")
    for i, t in enumerate(traits):
        print(f"        {t:14s} " + " ".join(f"{C[i, j]:+.4f}" for j in range(n)))
    return ((off > 0.99).sum() == 0,
            f"norms {np.round(d,2).tolist()}; off-diagonal mean {off.mean():+.4f} "
            f"range [{off.min():+.4f}, {off.max():+.4f}]")


print()
bad = [(n, nm, st, d) for n, nm, st, d in verdicts if st != PASS]
if bad:
    nf = sum(1 for _, _, st, _ in bad if st == FAIL)
    nu = len(bad) - nf
    print(f"PHASE 2 GATES NOT CLEAR: {nf} failed, {nu} unverified, "
          f"of {len(verdicts)}")
    for n, nm, st, d in bad:
        print(f"  gate {n} {nm} [{st}]: {d}")
    sys.exit(1)
print(f"all {len(verdicts)} phase-2 gates passed on {len(traits)} adapters "
      f"in {args.adapters}")
