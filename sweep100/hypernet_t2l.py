"""Text-to-LoRA over the 100-trait sweep: predict an adapter from its text.

THE QUESTION
------------
analyse_pca.py showed the 100 trait adapters occupy a structured region of
weight space, and text_baseline.py showed the DIFFERENCE view of the training
text (mean chosen embedding minus mean rejected embedding) predicts that weight
geometry at RSA 0.826, while the chosen embeddings alone predict almost nothing.
So the informative text signal is what the teacher CHANGED to express the trait,
not what it wrote.  This asks the stronger, generative version of that question:
given only the difference embedding of a trait we never trained on, can we emit
its LoRA weights directly?

WHY THIS IS ONLY WELL-POSED WITHIN ONE INITIALISATION
-----------------------------------------------------
seeds_analysis.py measured the same trait under a different LoRA init at cosine
0.013 -- near-orthogonal.  LoRA confines each update to a random rank-16 slice
of a 2048-dimensional space, and two random slices barely intersect however
similar the behaviour.  The trait's RELATIVE geometry reproduces across inits
(RSA 0.995) but its absolute direction does not.  Raw weights are therefore
predictable only within one initialisation, so everything here is trained and
evaluated inside seed 0 alone, and the loader refuses any adapter whose runmeta
does not say so.  A hypernetwork trained across inits would be fitting noise.

THE MODEL
---------
    [384-d text difference, L2-normalised]  ++  [64-d learned module embedding]
        -> shared MLP trunk -> one linear head per DISTINCT (d_in, d_out) shape
        -> A (r x d_in) and B (d_out x r), r = 16.

Qwen2.5-3B's 252 target modules come in only four shapes, so four heads cover
everything and every module of a shape shares its head; the module embedding is
what distinguishes layer 3's q_proj from layer 30's.  The heads hold essentially
all the parameters (the head input width is the only real knob) -- see
`param_count`.  B is zero at LoRA init but not after training, so both factors
are predicted the same way with no special casing.

THE LOSS IS NEVER MATERIALISED
------------------------------
The target is the composed delta dW = scale * B @ A (scale = lora_alpha/r = 2).
Materialising it densely for 252 modules is ~30M numbers per trait per step.
Every quantity we need is a Frobenius inner product, and those factor exactly,
the same identity analyse_pca.gram_streaming uses:

    <s1 B1 A1, s2 B2 A2>_F = s1 s2 * sum( (B1^T B2) * (A1 A2^T) )

with both factors r x r.  So the per-module normalised reconstruction loss

    L_m = ||dW_hat - dW||_F^2 / ||dW||_F^2 = (pp - 2 pt + tt) / tt

costs three r x r contractions and nothing is ever built densely.  (Note the
scale cancels in L_m; it is kept explicit anyway, and asserted against
adapter_config.json, because every other number in this project carries it.)

THE EVALUATION TRAP
-------------------
All 100 adapters share a common component: mean between-trait raw cosine is
+0.099, so the corpus MEAN adapter scores a raw cosine around 0.33 against any
held-out trait while containing no trait-specific information whatsoever.  Raw
cosine therefore cannot distinguish "learned the trait" from "learned the
corpus".  Both are reported, but the HEADLINE is the cosine after subtracting
the corpus mean from BOTH the prediction and the truth, with that mean computed
on the training folds only -- the held-out trait's own delta never enters it
(asserted in `cos_pair`, exercised in --selftest).

The bar is not zero and it is not the mean: it is RETRIEVAL.  Copying the
adapter of the training trait whose text feature is nearest costs no learning at
all, so a hypernetwork that does not beat it has learned nothing worth having.
The ceiling is 0.855, the raw cosine between two adapters of the same trait
differing only in data order (pca.json noise_floor.mean_reseed_cos); when the
five __s1 reseeds are present their mean-removed cosine is computed too, since
that -- not the raw 0.855 -- is the ceiling for the headline metric.

TWO OBJECTIVES -- cfg["loss"]
-----------------------------
"sqerr" (the default, unchanged) is the normalised squared error above.  It
optimises something the metric then throws away: the component all 100 adapters
share is most of each adapter's Frobenius norm, so nearly all capacity goes into
reproducing it, and the headline subtracts exactly that before scoring.  The
measured consequence: the trained hypernetwork's RAW cosine was 0.311 against
the corpus-mean baseline's 0.294 -- it had learned to emit the average adapter.

"centred_cos" optimises what is actually scored,

    L_m = 1 - cos(dW_hat - m, dW_t - m),

with m the mean adapter over the CURRENT FOLD'S TRAINING traits only -- never
validation, never test; `build_mean_factors` asserts it with the same teeth
`cos_pair` has.  A mean of N rank-r adapters is exactly a rank-(N*r) product
B_cat @ A_cat, so every term (<p,m>, <t,m>, <m,m>, <p,t>) is still one
`factored_dot` and the mean is no more materialised than anything else here.
<m,m> and <t,m> are constants of the fold and are computed once per fold.  The
price is honest and it is compute, not memory: m has rank N*r = 1024 against the
prediction's r = 16, so the <p,m> contraction is ~64x a same-rank one and
dominates the step.  Nothing about the evaluation, the folds or the baselines
changes -- only what the optimiser is pointed at.

WOULD MORE TRAITS CLOSE THE GAP? -- --curve
-------------------------------------------
The hypernetwork reaches 0.271 mean-removed, retrieval 0.564, the ceiling 0.831.
Before paying to train hundreds more adapters, ask the 100 we have: vary the
number of TRAINING traits and watch both curves.  The decisive quantity is the
pair of SLOPES, not the hypernetwork's alone.  If the generator climbs while
retrieval flattens, more traits buy something; if they climb together, more
traits will not close the gap; and if retrieval climbs FASTER, more traits make
the generator look worse, not better.  That last outcome is only visible if
retrieval is RECOMPUTED at every size over exactly the traits the hypernetwork
was given -- reusing the full-corpus 0.564 as a flat line would hide it.  So at
every point all four methods (hypernet, corpus mean, NN text retrieval, random
training adapter) are scored out-of-fold against the SAME subsample, and the
corpus mean removed by the metric is the subsample's mean too.

The x-axis is TRAITS AVAILABLE, n_avail = train + val, not the gradient set
alone.  What moves is the pool of non-test traits the fold is allowed to own;
it is then split into validation and training by exactly the standard run's
rule, so n_avail = 80 IS the standard split and that point reproduces the
published run on all four methods.  What is held fixed:

  * the five TEST folds -- byte-identical at every size, so the points are
    comparable and the ceiling is a genuine constant reference line;
  * the val FRACTION, not the val set: 20% of whatever is available, as in the
    standard run.  Validation traits are drawn from inside the subsample and
    never from outside it, so a small point cannot borrow a yardstick built
    from traits it does not own -- that would be the leak.
  * nesting: one permutation per (rep, fold), and the sizes are prefixes of it,
    so a larger point strictly ADDS traits to a smaller one and the slope is
    not re-drawn subsample luck.

At every point all four methods are scored out-of-fold against that point's
available pool, and `steps_run`/`hit_cap` are recorded per fold: a point that
stopped on the step cap rather than on validation patience is measuring compute,
not data, and the printed table shouts about it.

No predicted adapters are written in curve mode: 8 grid points x 6 GB is not a
useful artefact, and the curve is read entirely off the Grams.

DOES IT GENERALISE, OR IS IT INTERPOLATING? -- --ood
----------------------------------------------------
Every number above is cross-validation INSIDE one corpus: 100 Goldberg Big-Five
markers, held out one fold at a time.  A trait held out of that corpus still has
close relatives inside it, and that is precisely what retrieval needs -- it
works by finding a near neighbour, so a corpus that densely covers its own
region makes copying a strong strategy.  The 0.599 vs 0.564 gap is therefore
measured in the regime most favourable to the bar it is beating.

--ood trains once on all 100 Goldberg markers and evaluates on 148 traits from
DIFFERENT instruments: HEXACO facets (including Honesty-Humility, a factor the
Goldberg markers do not cover at all), the Dark Triad, the Interpersonal
Circumplex, Schwartz values, and conversational-register terms.  This is a real
distribution shift, not a random split.  If a test trait comes from a region the
training corpus never sampled, there is no near neighbour to find and retrieval
has nothing to copy.  A hypernetwork that genuinely learned the text -> weights
MAPPING should degrade far more gently, because it COMPUTES an answer from the
text rather than looking one up in a table of 100 answers.

THE REGISTERED PREDICTION, made before the run: retrieval drops sharply out of
distribution, the hypernetwork drops much less, and the two CROSS OVER.  THE
ALTERNATIVE OUTCOME the test can produce, and which must be recorded if it
happens: both degrade together by a similar amount, in which case the
hypernetwork has been interpolating between training traits all along and its
in-distribution win over retrieval was a smoother interpolation of the same
lookup table, not a learned mapping.  A third possibility exists and is worth
naming: both hold up, which would say the 148 traits are not as far outside the
Goldberg region as their instruments suggest.

There is NO cross-validation here and none is needed: the test traits are held
out by construction, so the model trains on all 100 and is scored on all 148.
A validation split is still taken from INSIDE the 100 for early stopping.  The
corpus mean removed by the headline metric is the mean over the 100 TRAINING
adapters, and unlike the CV case leakage is structurally impossible -- no test
trait is in the training corpus, so no test adapter can enter its own reference
mean, and the retrieval pool is the training corpus by definition.

The two corpora must be the same KIND of object for any of this to mean
anything: same module list, same rank, same scale, and above all the same LoRA
init (seed 0, 2 epochs, asserted from every runmeta on both sides).  A different
init in either corpus would make the mapping unlearnable noise rather than
merely harder -- see WHY THIS IS ONLY WELL-POSED WITHIN ONE INITIALISATION.

Usage
-----
    ~/cartovenv/bin/python hypernet_t2l.py --selftest      # no GPU, no network
    ~/cartovenv/bin/modal run hypernet_t2l.py              # the real run
    ~/cartovenv/bin/modal run hypernet_t2l.py --loss centred_cos
    ~/cartovenv/bin/modal run hypernet_t2l.py --curve      # the learning curve
    ~/cartovenv/bin/modal run hypernet_t2l.py --ood        # 100 -> 148 held-out
    ~/cartovenv/bin/python hypernet_t2l.py --fetch         # pull predictions
Outputs: results/hypernet_t2l.json, and adapters_t2l/<trait>/ (fp16);
         results/hypernet_t2l_curve.json for --curve (no adapters);
         results/hypernet_t2l_ood.json for --ood (no adapters).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import modal

APP_NAME = "sweep100-t2l"
ADAPTER_VOLUME = "sweep100-adapters"      # READ ONLY here; train_sweep.py owns it
EXP_ADAPTER_VOLUME = "sweep100exp-adapters"   # the 148 expanded traits, READ ONLY
T2L_VOLUME = "sweep100-t2l"
GPU_TYPE = os.environ.get("T2L_GPU", "A100-40GB")
GPU_USD_PER_HOUR = float(os.environ.get("T2L_GPU_USD_HR", "2.10"))

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")

RANK = 16
LORA_SCALE = 2.0             # lora_alpha / r = 32/16; asserted against the file
TEXT_DIM = 384               # all-MiniLM-L6-v2, from text_baseline.py
EXPECTED_EPOCHS = 2.0        # every seed-0 adapter in the sweep; provenance guard
CEILING_RESEED_COS = 0.855   # pca.json noise_floor.mean_reseed_cos, raw
RESEED_SUFFIX = "__s1"

CFG = dict(
    rank=RANK, scale=LORA_SCALE, text_dim=TEXT_DIM,
    mod_emb=64, trunk=(512, 512), head_in=256,
    folds=5, split_seed=0, val_frac=0.2,
    steps=3000, batch_traits=8, modules_per_step=0,   # 0 = every module
    lr=3e-4, weight_decay=0.01, warmup=100,
    lr_schedule="cosine",    # "cosine" (default, unchanged) | "constant"
    eval_every=50, patience=12, grad_clip=1.0, train_seed=0,
    pred_dtype="float16",
    loss="sqerr",            # "sqerr" (default) | "centred_cos"
    mean_max_gb=12.0,        # budget for one fold's factored mean; see below
    ood_chunk=16,            # test adapters resident at once in --ood; see run_ood
)

LOSSES = ("sqerr", "centred_cos")
LR_SCHEDULES = ("cosine", "constant")

# The learning curve.
#
# THE AXIS IS TRAITS AVAILABLE (train + val), NOT TRAINING TRAITS.  A curve whose
# x-axis is the gradient training set alone has to hold the validation set fixed
# outside it, and then the largest point is not the standard split -- the
# baselines would see 80 traits where the hypernetwork saw 64, and the top of the
# curve could not be checked against the published run at all.  Subsampling the
# whole non-test pool and splitting it val_frac/1-val_frac exactly as the
# standard run does makes n_avail = 80 IDENTICAL to the standard split, so that
# point reproduces the published run on all four methods and is a real sanity
# check rather than an approximate one.  It is also the honest question: "how
# many trait adapters would we have to own", not "how many would we feed the
# optimiser after setting some aside".
#
# CONVERGENCE, NOT A BUDGET.  A fixed step count confounds data with compute: if
# the hypernetwork is compute-limited its curve flattens for reasons that have
# nothing to do with how many traits it has, and the honest conclusion "more
# traits will not help" would be wrong.  So the curve trains under a CONSTANT
# learning rate after warmup, where the validation curve plateaus and `patience`
# genuinely binds, with `steps` as a cap that should not be reached.  The
# published run's cosine-to-zero schedule cannot be stopped early in any
# meaningful sense -- the val loss improves right up to the horizon because the
# LR is going to zero there, so "early stopping" under cosine only ever means
# "the schedule ended", which is a budget by another name.  Every point records
# `steps_run` and `hit_cap`, and `curve_cap_warning` shouts about any point that
# terminated on the cap, because such a point's number is a compute measurement
# wearing a data measurement's clothes.
#
# reps is 2, not 3: with convergence-based stopping the cap has to be generous
# (6000, double the published run), and 4 sizes x 3 reps x 5 folds against that
# cap risks 4+ h.  Noise across repeats is recoverable by running more repeats;
# a compute confound is not recoverable at all.  --curve-reps 3 costs ~1.5x.
CURVE = dict(
    sizes=(20, 40, 60, 80),   # 80 = every non-test trait = the standard split
    reps=2,                   # nested subsample draws, averaged
    steps=6000,               # a CAP, not a schedule; see lr_schedule below
    lr_schedule="constant",
    modules_per_step=24,
    loss="centred_cos",       # our best variant; overridable with --loss
    subsample_seed=0,
)

# The out-of-distribution evaluation.
#
# ONE TRAINING RUN, NO FOLDS.  The test traits come from other instruments
# entirely, so they are held out by construction and cross-validation would only
# be re-deriving a held-out set that already exists.  What is still needed is a
# stopping rule, so `val_frac` of the 100 training traits is set aside exactly as
# the standard run sets it aside -- the hypernetwork takes gradients on 80 and
# stops on the other 20.
#
# THE CAP MUST NOT BIND, and here it matters more than anywhere else.  A capped
# hypernetwork number in an OOD comparison is unusable: the whole claim is about
# how gently the LEARNED mapping degrades, and an undertrained mapping degrades
# for a reason that has nothing to do with distribution shift, in the direction
# that happens to confirm the sceptical hypothesis.  The convergence probe at
# n_avail 80 (results/hypernet_t2l_curve.json, the same objective, schedule and
# modules_per_step as here) stopped on patience at 16,641 steps on average and
# 18,251 in its slowest fold, so 30000 is ~1.6x the worst observed stop.  It is
# still only a cap: `steps_run` / `hit_cap` are recorded and `ood_cap_warning`
# shouts if it binds.
OOD = dict(
    steps=30000,              # a CAP; see above.  ~1.6x the worst observed stop
    lr_schedule="constant",   # patience must be able to bind -- see train_fold
    modules_per_step=24,      # the regime the 16,641-step figure was measured in
    loss="centred_cos",       # the objective the headline metric actually scores
)

# The in-distribution numbers, QUOTED not recomputed.
#
# Source: results/hypernet_t2l_curve.json, grid point n_avail = 80 -- the
# convergence probe, which IS the standard 5-fold split on all 100 Goldberg
# traits (80 available per fold = 64 train + 16 val, 20 held out), objective
# centred_cos, constant LR, modules_per_step 24, every fold stopped on validation
# patience (folds_at_cap 0/5).  Mean-removed cosine, the headline metric.
#
# They are constants here on purpose: --ood trains ONE model on all 100 traits
# and never runs a fold, so it has no in-distribution number of its own to
# report, and recomputing one would cost another GPU-hour to reproduce a figure
# that is already in the repository.  The corpus mean is 0.000 by construction
# (the metric subtracts it), not by measurement.
IN_DISTRIBUTION = {
    "hypernet": 0.599,
    "mean_adapter": 0.000,        # structurally zero: the metric removes it
    "nn_text_retrieval": 0.564,
    "random_adapter": -0.038,
    "_source": "results/hypernet_t2l_curve.json grid[80], n=100 traits, 5-fold "
               "CV, converged (0/5 folds at cap), mean-removed cosine",
}


# ===========================================================================
# factored algebra -- the same identity as analyse_pca.gram_streaming
# ===========================================================================
def factored_dot(A1, B1, A2, B2):
    """<B1@A1, B2@A2>_F, batched over leading dims.  A: (...,r,d_in)  B: (...,d_out,r)."""
    BtB = torch.einsum("...dr,...ds->...rs", B1, B2)
    AAt = torch.einsum("...rd,...sd->...rs", A1, A2)
    return (BtB * AAt).sum((-2, -1))


def gram_between(stack1, stack2, scale, log=None):
    """(n1, n2) Frobenius Gram of s*B@A between two adapter stacks.

    A stack is a list of per-shape groups, each holding A (M, n, r, d_in) and
    B (M, n, d_out, r).  Accumulated in float64: the metrics are differences of
    similar large numbers once the corpus mean is removed, and fp32 accumulation
    over 252 modules loses more digits than the effect we are measuring.
    """
    n1 = stack1[0]["A"].shape[1]
    n2 = stack2[0]["A"].shape[1]
    dev = stack1[0]["A"].device
    G = torch.zeros(n1, n2, dtype=torch.float64, device=dev)
    r = stack1[0]["A"].shape[2]
    t0, done = time.time(), 0
    for g1, g2 in zip(stack1, stack2):
        assert g1["A"].shape[0] == g2["A"].shape[0], "stacks disagree on modules"
        M, _, _, din = g1["A"].shape
        dout = g1["B"].shape[2]
        for k in range(M):
            A1 = g1["A"][k].double().reshape(n1 * r, din)
            A2 = g2["A"][k].double().reshape(n2 * r, din)
            B1 = g1["B"][k].double().permute(1, 0, 2).reshape(dout, n1 * r)
            B2 = g2["B"][k].double().permute(1, 0, 2).reshape(dout, n2 * r)
            M2 = (B1.T @ B2) * (A1 @ A2.T)
            G += M2.reshape(n1, r, n2, r).sum((1, 3))
            del A1, A2, B1, B2, M2
            done += 1
            if log is not None and done % 50 == 0:
                print(f"  gram module {done}  {time.time()-t0:.0f}s",
                      file=log, flush=True)
    return G * (scale ** 2)


def self_norms2(stack, scale):
    """(n,) squared Frobenius norms of each adapter in a stack."""
    n = stack[0]["A"].shape[1]
    out = torch.zeros(n, dtype=torch.float64, device=stack[0]["A"].device)
    for g in stack:
        # One module at a time.  Casting a whole group to float64 up front
        # allocates a second copy of the largest tensors in the run and OOM'd a
        # 40GB card AFTER all five folds had trained -- the cost of a cheap
        # scoring step is paid at the very end, where it destroys the expensive
        # part.  float64 still, for the reason gram_between documents.
        for k in range(g["A"].shape[0]):
            A = g["A"][k].double()
            B = g["B"][k].double()
            out += factored_dot(A, B, A, B)
            del A, B
    return out * (scale ** 2)


def diag_between(stack1, stack2, scale):
    """(n,) elementwise <s B1_i A1_i, s B2_i A2_i>_F for two ALIGNED stacks.

    The DIAGONAL of gram_between, and the only part of it --ood needs: each
    predicted adapter is scored against its own trait's truth, so the 148 x 148
    cross Gram would compute 147 numbers per row that nothing reads.

    Deliberately not written as `self_norms2(stack) = diag_between(stack, stack)`:
    the same-stack case casts ONE tensor to float64 per module and this one casts
    two, and self_norms2's own comment records what a second float64 copy of the
    largest tensors cost a 40 GB card once already.
    """
    n = stack1[0]["A"].shape[1]
    assert stack2[0]["A"].shape[1] == n, "diag_between needs aligned stacks"
    out = torch.zeros(n, dtype=torch.float64, device=stack1[0]["A"].device)
    for g1, g2 in zip(stack1, stack2):
        assert g1["A"].shape[0] == g2["A"].shape[0], "stacks disagree on modules"
        for k in range(g1["A"].shape[0]):
            A1, B1 = g1["A"][k].double(), g1["B"][k].double()
            A2, B2 = g2["A"][k].double(), g2["B"][k].double()
            out += factored_dot(A1, B1, A2, B2)
            del A1, B1, A2, B2
    return out * (scale ** 2)


# ===========================================================================
# corpus loading
# ===========================================================================
def slug(trait):
    return trait.lower().replace("-", "_")


def group_key(d_in, d_out):
    return f"{d_in}x{d_out}"


def read_config(adir, name, rank=None):
    cfg = json.load(open(os.path.join(adir, name, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    assert rank is None or r == rank, f"{name}: r={r}, expected {rank}"
    assert abs(alpha / r - LORA_SCALE) < 1e-12, f"{name}: scale={alpha/r}"
    return cfg


def check_provenance(adir, names):
    """Every training adapter must BE a seed-0, 2-epoch run of its own trait.

    seeds_analysis.py learned this the hard way: four directories once held
    9-epoch calibration adapters under correct-looking names and correct file
    sizes.  A name is not provenance, and here the stakes are higher than a
    mixed corpus -- a different LoRA INIT in the training set is unlearnable
    noise (cosine 0.013 between inits), not merely a bias.
    """
    bad = []
    for n in names:
        p = os.path.join(adir, n, "runmeta.json")
        if not os.path.exists(p):
            bad.append(f"{n}: no runmeta.json")
            continue
        m = json.load(open(p))
        init = m.get("init_seed")
        if init is None:
            init = m.get("seed")          # pre-dates the init_seed field
        ep = (m.get("hparams") or {}).get("epochs")
        base = n[:-len(RESEED_SUFFIX)] if n.endswith(RESEED_SUFFIX) else n
        if init != 0:
            bad.append(f"{n}: init_seed={init}, expected 0")
        elif ep != EXPECTED_EPOCHS:
            bad.append(f"{n}: epochs={ep}, expected {EXPECTED_EPOCHS}")
        elif m.get("trait") != base:
            bad.append(f"{n}: runmeta trait={m.get('trait')!r}")
    assert not bad, ("adapters are not what their names claim: " + "; ".join(bad[:10]))


def shared_modules(handles):
    """The module list a set of open safetensors handles agrees on.

    Every adapter must expose exactly the same modules: a union larger than the
    intersection means one corpus (or one adapter) targets something the others
    do not, and every downstream contraction silently zips mismatched lists.
    """
    keysets = [set(k[:-len(".lora_A.weight")] for k in h.keys() if "lora_A" in k)
               for h in handles]
    modules = sorted(set.intersection(*keysets))
    extra = set.union(*keysets) - set(modules)
    assert not extra, f"modules not shared by all adapters: {sorted(extra)[:5]}"
    return modules


def corpus_modules(adir, names):
    """The module list of a corpus, from safetensors HEADERS alone.

    Header reads only, no tensors: this is how --ood checks that two corpora are
    the same kind of object BEFORE spending an hour of GPU time training on one
    of them.  `load_stack` derives the same list the same way once it has the
    handles open anyway.
    """
    from safetensors import safe_open

    hs = [safe_open(os.path.join(adir, a, "adapter_model.safetensors"),
                    framework="pt") for a in names]
    return shared_modules(hs)


def load_stack(adir, names, device, log=sys.stderr, provenance=True, rank=None):
    """Read adapters into per-shape groups.  Returns (stack, modules, meta)."""
    from safetensors import safe_open

    assert names, "no adapters requested"
    if provenance:
        check_provenance(adir, names)
    cfg0 = read_config(adir, names[0], rank)
    if rank is None:            # take it from the corpus, then enforce it
        rank = cfg0["r"]

    sizes = {a: os.path.getsize(os.path.join(adir, a, "adapter_model.safetensors"))
             for a in names}
    med = np.median(list(sizes.values()))
    odd = {a: s for a, s in sizes.items() if abs(s - med) > 0.001 * med}
    assert not odd, f"anomalous adapter file sizes (incomplete fetch?): {odd}"

    handles = {a: safe_open(os.path.join(adir, a, "adapter_model.safetensors"),
                            framework="pt") for a in names}
    modules = shared_modules(handles.values())

    shapes = {}
    for mi, m in enumerate(modules):
        A0 = handles[names[0]].get_slice(m + ".lora_A.weight").get_shape()
        B0 = handles[names[0]].get_slice(m + ".lora_B.weight").get_shape()
        assert A0[0] == rank and B0[1] == rank, (m, A0, B0)
        shapes.setdefault(group_key(A0[1], B0[0]), []).append(mi)

    n = len(names)
    stack = []
    t0 = time.time()
    for key in sorted(shapes, key=lambda k: (-len(shapes[k]), k)):
        idx = shapes[key]
        din, dout = (int(x) for x in key.split("x"))
        A = torch.empty(len(idx), n, rank, din, dtype=torch.float32, device=device)
        B = torch.empty(len(idx), n, dout, rank, dtype=torch.float32, device=device)
        for k, mi in enumerate(idx):
            m = modules[mi]
            for i, a in enumerate(names):
                A[k, i] = handles[a].get_tensor(m + ".lora_A.weight").to(device)
                B[k, i] = handles[a].get_tensor(m + ".lora_B.weight").to(device)
        stack.append({"key": key, "d_in": din, "d_out": dout,
                      "mod_idx": torch.tensor(idx, dtype=torch.long, device=device),
                      "A": A, "B": B})
        print(f"  loaded {len(idx):3d} modules of shape {key}  "
              f"{time.time()-t0:.0f}s", file=log, flush=True)
    return stack, modules, {"n": n, "names": list(names)}


def text_features(path, names, order):
    """L2-normalised (chosen - rejected) MiniLM difference, one row per name.

    text_baseline.py measured this view at RSA 0.826 against the weight geometry
    while chosen-only was near zero, so the difference is the whole signal.
    """
    z = np.load(path)
    V = z["chosen"] - z["rejected"]
    assert V.shape == (len(order), TEXT_DIM), V.shape
    pos = {nm: i for i, nm in enumerate(order)}
    V = V[[pos[nm] for nm in names]]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return torch.tensor(V, dtype=torch.float32)


def npz_order(path, fallback=None):
    """The slug order of a text-vector file, from the file itself where possible.

    text_vecs_expanded.npz stores a `names` array; text_vecs.npz predates it and
    is in traits.json order.  Taking the row order from the file whenever it says
    what its rows are is the difference between aligning BY NAME and aligning by
    position and hoping -- the expanded traits file lists 152 traits while the
    npz has 148 rows in a different order, so position would be silently wrong.
    `text_features` then does the actual permutation into adapter-name order.
    """
    z = np.load(path)
    if "names" in z:
        return [str(s) for s in z["names"]]
    assert fallback is not None, (
        f"{path} has no `names` array and no fallback order was given -- "
        f"refusing to align text features by position")
    return list(fallback)


# ===========================================================================
# the hypernetwork
# ===========================================================================
class Hypernet(nn.Module):
    def __init__(self, n_modules, groups, cfg):
        super().__init__()
        self.cfg = cfg
        r = cfg["rank"]
        self.mod_emb = nn.Embedding(n_modules, cfg["mod_emb"])
        nn.init.normal_(self.mod_emb.weight, std=0.02)

        dims = [cfg["text_dim"] + cfg["mod_emb"]] + list(cfg["trunk"]) + [cfg["head_in"]]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        self.trunk = nn.Sequential(*layers[:-1])

        self.keys = [g["key"] for g in groups]
        self.heads = nn.ModuleList([
            nn.Linear(cfg["head_in"], r * g["d_in"] + g["d_out"] * r) for g in groups])
        self.splits = [(g["d_in"], g["d_out"]) for g in groups]
        # one gain per head, calibrated on the training folds so the initial
        # predicted ||dW|| matches theirs; without it a fresh head emits a delta
        # orders of magnitude too small and the normalised loss starts flat.
        self.log_gain = nn.Parameter(torch.zeros(len(groups)))

    def forward(self, x_text, mod_ids, gi):
        """(b, text_dim) x (M,) -> A (M, b, r, d_in), B (M, b, d_out, r)."""
        r = self.cfg["rank"]
        din, dout = self.splits[gi]
        M, b = mod_ids.shape[0], x_text.shape[0]
        e = self.mod_emb(mod_ids)[:, None, :].expand(M, b, self.cfg["mod_emb"])
        z = torch.cat([x_text[None, :, :].expand(M, b, x_text.shape[1]), e], -1)
        o = self.heads[gi](self.trunk(z))
        g = torch.exp(self.log_gain[gi])
        A = o[..., :r * din].reshape(M, b, r, din) * g
        B = o[..., r * din:].reshape(M, b, dout, r) * g
        return A, B


def param_count(model):
    head = sum(p.numel() for h in model.heads for p in h.parameters())
    return {"total": sum(p.numel() for p in model.parameters()),
            "heads": head,
            "trunk": sum(p.numel() for p in model.trunk.parameters()),
            "module_embeddings": model.mod_emb.weight.numel()}


# ===========================================================================
# training
# ===========================================================================
def kfold(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def fold_orders(n, folds, cfg, rng):
    """Per fold, the non-test traits in the standard split's PRIORITY ORDER.

    Exactly the permutation run_cv's loop has always drawn -- same rng, consumed
    in the same fold order -- with the cut into val (the front) and train (the
    rest) deferred to `split_at`.  Splitting it this way is what lets --curve
    restrict the pool and still land on the standard split at full size: keep a
    subset of this order, in this order, and the cut rule does the rest.
    """
    out = []
    for test_idx in folds:
        rest = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        out.append(rest[rng.permutation(len(rest))])
    return out


def split_at(order, val_frac):
    """(train, val) from a priority order: the front val_frac is validation."""
    nval = max(1, int(round(val_frac * len(order))))
    assert nval < len(order), f"val_frac {val_frac} leaves no training traits"
    return np.sort(order[nval:]), np.sort(order[:nval])


def fold_splits(n, folds, cfg, rng):
    """Per-fold (train, val), drawn exactly as run_cv has always drawn them."""
    return [split_at(o, cfg["val_frac"])
            for o in fold_orders(n, folds, cfg, rng)]


def build_mean_factors(stack, train_idx, forbid=(), max_gb=None, log=sys.stderr):
    """One fold's corpus mean adapter, exactly, in factored form.

    A mean of N rank-r adapters is a rank-(N*r) product and nothing more:

        m = (1/N) sum_j B_j A_j = B_cat @ A_cat
        B_cat = (1/N) [B_1 ... B_N]      (d_out, N*r)   <- the 1/N lives HERE,
        A_cat = [A_1 ; ... ; A_N]        (N*r, d_in)       on B_cat and nowhere
                                                           else

    so <p,m>, <t,m> and <m,m> are ordinary `factored_dot` calls against a wider
    inner dimension, and the mean is no more materialised than dW ever is.  The
    returned factors carry a singleton trait axis so they broadcast against a
    (M, b, ...) batch of predictions.

    THE LEAKAGE BOUNDARY.  `train_idx` must be this fold's TRAINING traits and
    only those.  A mean that had seen the validation traits would leak the
    quantity early stopping selects on; one that had seen the test traits would
    leak into the prediction itself.  `forbid` is asserted disjoint from it --
    the same assertion `cos_pair` carries on the scoring side, exercised in
    --selftest by corrupting a held-out trait and demanding <m,m> not move.

    MEMORY.  On the real corpus N*r = 1024 and d_out reaches 11008, so the
    factors are ~7.7 GB across all 252 modules.  They are built ONE MODULE AT A
    TIME into a preallocated buffer, so the peak is the buffer plus one module
    rather than two copies of everything -- `self_norms2` above documents what
    the lazy version costs on a 40GB card.  They are built once per FOLD and
    dropped with it; rebuilding per STEP would save nothing, because autograd
    retains whatever the loss contracted against until the backward pass.

    <m,m> and <m,dW_i> are constants of the fold, so they are accumulated here
    once.  They are stored float64 so that the per-step differences below keep
    their digits; the dots themselves stay in the stack's dtype, because the
    corpus geometry (between-trait cosine +0.099) leaves ||dW - m||^2 at ~90%
    of ||dW||^2 -- there is no catastrophic cancellation here to defend against,
    unlike the mean-removed metric that `gram_between` feeds.
    """
    tr = np.asarray(train_idx, dtype=int).ravel()
    assert tr.size, "the fold mean needs at least one training trait"
    clash = sorted(set(tr.tolist()) &
                   set(np.asarray(forbid, dtype=int).ravel().tolist()))
    assert not clash, f"leakage: the fold mean would average held-out traits {clash}"

    dev = stack[0]["A"].device
    N, r = int(tr.size), stack[0]["A"].shape[2]
    want = sum(g["A"].shape[0] * N * r * (g["A"].shape[3] + g["B"].shape[2])
               * g["A"].element_size() for g in stack)
    if max_gb:
        assert want <= max_gb * 1e9, (
            f"fold mean factors want {want/1e9:.2f} GB > mean_max_gb {max_gb} "
            f"(N={N}, r={r}) -- raise the budget deliberately, not by accident")

    idx = torch.as_tensor(tr, dtype=torch.long, device=dev)
    out = []
    with torch.no_grad():
        for g in stack:
            M, n, _, din = g["A"].shape
            dout = g["B"].shape[2]
            Am = torch.empty(M, 1, N * r, din, dtype=g["A"].dtype, device=dev)
            Bm = torch.empty(M, 1, dout, N * r, dtype=g["B"].dtype, device=dev)
            mm = torch.empty(M, 1, dtype=torch.float64, device=dev)
            mt = torch.empty(M, n, dtype=torch.float64, device=dev)
            for k in range(M):
                Am[k, 0] = g["A"][k].index_select(0, idx).reshape(N * r, din)
                Bm[k, 0] = (g["B"][k].index_select(0, idx)
                            .permute(1, 0, 2).reshape(dout, N * r))
                Bm[k, 0].div_(N)             # the whole 1/N, on B_cat alone
                Ak, Bk = Am[k, 0], Bm[k, 0]
                mm[k, 0] = factored_dot(Ak, Bk, Ak, Bk)
                mt[k] = factored_dot(Ak, Bk, g["A"][k], g["B"][k])
            out.append({"key": g["key"], "A": Am, "B": Bm, "mm": mm, "mt": mt})
    print(f"    fold mean: rank {N*r} over {N} training traits, "
          f"{want/1e9:.2f} GB of factors", file=log, flush=True)
    return out


def centred_cos_loss(Ah, Bh, At, Bt, Am, Bm, mm, tm):
    """1 - cos(dW_hat - m, dW_t - m) per (module, trait), nothing dense.

        <p-m, t-m> = <p,t> - <p,m> - <m,t> + <m,m>
        ||p-m||^2  = <p,p> - 2<p,m> + <m,m>
        ||t-m||^2  = <t,t> - 2<t,m> + <m,m>

    `mm` = <m,m> and `tm` = <m,dW_t> come precomputed from the fold's mean, so
    the only new per-step contraction is <p,m>.  The LORA_SCALE cancels in a
    cosine exactly as it cancels in the squared error, so raw factors are used
    throughout, as everywhere else in this file.

    A prediction that IS the mean has no residual and therefore no direction.
    That is a legitimate state to pass through -- it is where a fresh model
    starts, and 0/0 there would poison the gradient with NaN for the rest of the
    run -- so the denominator is clamped and such a prediction scores 1.0, the
    cosine's own value for "no agreement", rather than blowing up.
    """
    pp = factored_dot(Ah, Bh, Ah, Bh).double()
    pt = factored_dot(Ah, Bh, At, Bt).double()
    tt = factored_dot(At, Bt, At, Bt).double()
    pm = factored_dot(Ah, Bh, Am, Bm).double()
    num = pt - pm - tm + mm
    np2 = (pp - 2 * pm + mm).clamp_min(0)
    nt2 = (tt - 2 * tm + mm).clamp_min(0)
    return 1.0 - num / (np2 * nt2).clamp_min(1e-30).sqrt()


def batch_loss(model, stack, X, trait_idx, mod_sel=None, mean=None):
    """Mean over (module, trait) of the objective, factored.

    `mean is None` selects "sqerr", ||dW_hat - dW||^2 / ||dW||^2; otherwise the
    centred cosine against that fold's mean.  Both reduce identically -- summed
    over every (module, trait) in the batch and divided by the count -- so the
    two histories are read on the same footing even though the numbers are not
    the same quantity.
    """
    tot, cnt = 0.0, 0
    for gi, g in enumerate(stack):
        sel = mod_sel[gi] if mod_sel is not None else None
        mods = g["mod_idx"] if sel is None else g["mod_idx"][sel]
        At = g["A"][:, trait_idx] if sel is None else g["A"][sel][:, trait_idx]
        Bt = g["B"][:, trait_idx] if sel is None else g["B"][sel][:, trait_idx]
        Ah, Bh = model(X[trait_idx], mods, gi)
        if mean is None:
            pp = factored_dot(Ah, Bh, Ah, Bh)
            pt = factored_dot(Ah, Bh, At, Bt)
            tt = factored_dot(At, Bt, At, Bt)
            l = (pp - 2 * pt + tt) / tt
        else:
            mg = mean[gi]
            Am = mg["A"] if sel is None else mg["A"][sel]
            Bm = mg["B"] if sel is None else mg["B"][sel]
            mm = mg["mm"] if sel is None else mg["mm"][sel]
            mt = mg["mt"] if sel is None else mg["mt"][sel]
            l = centred_cos_loss(Ah, Bh, At, Bt, Am, Bm, mm, mt[:, trait_idx])
        tot = tot + l.sum()
        cnt += l.numel()
    return tot / cnt


def calibrate_gain(model, stack, X, train_idx, probe=8):
    """Match initial predicted ||dW|| to the training folds'.  TRAIN ONLY."""
    idx = torch.as_tensor(np.asarray(train_idx[:probe]), dtype=torch.long,
                          device=X.device)
    with torch.no_grad():
        for gi, g in enumerate(stack):
            Ah, Bh = model(X[idx], g["mod_idx"], gi)
            p = factored_dot(Ah, Bh, Ah, Bh).clamp_min(1e-30).sqrt().mean()
            At, Bt = g["A"][:, idx], g["B"][:, idx]
            t = factored_dot(At, Bt, At, Bt).sqrt().mean()
            # ||dW|| scales as gain^2, hence the half
            model.log_gain.data[gi] += 0.5 * torch.log(t / p)


def eval_loss(model, stack, X, idx, chunk, mean=None):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), chunk):
            sub = idx[s:s + chunk]
            tot += float(batch_loss(model, stack, X, sub, None, mean)) * len(sub)
            n += len(sub)
    model.train()
    return tot / n


def train_fold(stack, X, n_modules, train_idx, val_idx, test_idx, cfg,
               device, log=sys.stderr):
    """One fold.  train/val are disjoint subsets of the non-held-out traits."""
    assert not (set(train_idx) & set(test_idx)), "train/test leak"
    assert not (set(val_idx) & set(test_idx)), "val/test leak"
    assert not (set(train_idx) & set(val_idx)), "train/val leak"
    objective = cfg.get("loss", "sqerr")
    assert objective in LOSSES, f"unknown loss {objective!r}, want one of {LOSSES}"

    # Once per fold, never per step, and over the training traits alone: both
    # the validation and the test traits are handed in as forbidden so the
    # assertion inside has something to bite on.
    mean = None
    if objective == "centred_cos":
        mean = build_mean_factors(
            stack, train_idx,
            forbid=np.concatenate([np.asarray(val_idx, dtype=int).ravel(),
                                   np.asarray(test_idx, dtype=int).ravel()]),
            max_gb=cfg.get("mean_max_gb"), log=log)

    torch.manual_seed(cfg["train_seed"])
    model = Hypernet(n_modules, stack, cfg).to(device)
    calibrate_gain(model, stack, X, train_idx)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])

    # "cosine" is the published schedule and the default, untouched.  Note what
    # it does to early stopping: the LR is driven to zero AT cfg["steps"], so the
    # validation loss keeps creeping down until the horizon and `patience` can
    # essentially never bind -- under cosine, `steps` is the schedule, and
    # stopping "early" only means the schedule ended.  --curve therefore runs
    # "constant" (warmup, then flat), where the val curve plateaus and patience
    # is a real convergence test rather than a budget in disguise.
    schedule = cfg.get("lr_schedule", "cosine")
    assert schedule in LR_SCHEDULES, (
        f"unknown lr_schedule {schedule!r}, want one of {LR_SCHEDULES}")

    def lr_at(step):
        if step < cfg["warmup"]:
            return (step + 1) / cfg["warmup"]
        if schedule == "constant":
            return 1.0
        p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(cfg["train_seed"])
    tr = torch.as_tensor(np.asarray(train_idx), dtype=torch.long, device=device)
    va = torch.as_tensor(np.asarray(val_idx), dtype=torch.long, device=device)

    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for step in range(cfg["steps"]):
        pick = rng.choice(len(train_idx), size=min(cfg["batch_traits"],
                                                   len(train_idx)), replace=False)
        idx = tr[torch.as_tensor(pick, device=device)]
        mod_sel = None
        if cfg["modules_per_step"]:
            mod_sel = [torch.as_tensor(
                rng.choice(g["mod_idx"].shape[0],
                           size=min(cfg["modules_per_step"], g["mod_idx"].shape[0]),
                           replace=False), dtype=torch.long, device=device)
                for g in stack]
        loss = batch_loss(model, stack, X, idx, mod_sel, mean)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()

        if step % cfg["eval_every"] == 0 or step == cfg["steps"] - 1:
            v = eval_loss(model, stack, X, va, cfg["batch_traits"], mean)
            hist.append({"step": step, "train": float(loss), "val": v})
            print(f"    step {step:5d}  train {float(loss):.4f}  val {v:.4f}"
                  f"  {time.time()-t0:.0f}s", file=log, flush=True)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    print(f"    early stop at step {step} (best val {best:.4f})",
                          file=log, flush=True)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Several GB on the real corpus, and the last step's autograd graph still
    # holds what the loss contracted against -- drop both, not just the factors.
    loss = None
    del mean, loss
    return model, {"best_val": best, "history": hist, "steps_run": step + 1,
                   "loss": objective}


def predict_into(model, stack, pred, X, test_idx, device, chunk=8):
    """Write the model's predictions for `test_idx` into the pred stack."""
    model.eval()
    ti = torch.as_tensor(np.asarray(test_idx), dtype=torch.long, device=device)
    with torch.no_grad():
        for s in range(0, len(ti), chunk):
            sub = ti[s:s + chunk]
            for gi, g in enumerate(stack):
                Ah, Bh = model(X[sub], g["mod_idx"], gi)
                pred[gi]["A"][:, sub] = Ah.to(pred[gi]["A"].dtype)
                pred[gi]["B"][:, sub] = Bh.to(pred[gi]["B"].dtype)
    model.train()


# ===========================================================================
# evaluation -- everything from the Gram, nothing dense
# ===========================================================================
def cos_terms(ptt, q, gtt, pm, gtm, gm):
    """Raw and mean-removed cosine from the six inner products they need.

    <p,t>, <p,p>, <t,t>, <p,m>, <t,m>, <m,m> -- and nothing else, in particular
    no notion of where the trait t lives.  Both the CV scorer (where t is a
    held-out row of the corpus Gram) and the OOD scorer (where t is not in the
    corpus at all) reduce to exactly this, so the arithmetic exists once.
    """
    raw = ptt / math.sqrt(max(q * gtt, 1e-300)) if q > 0 else 0.0
    np2 = q - 2 * pm + gm
    nt2 = gtt - 2 * gtm + gm
    tol = 1e-12 * max(gtt, 1.0)
    # A prediction equal to the mean has no residual and therefore no direction;
    # that is the whole point of the metric, so it scores 0 rather than NaN.
    mr = 0.0 if np2 <= tol or nt2 <= tol else (ptt - pm - gtm + gm) / math.sqrt(np2 * nt2)
    return {"raw": raw, "mean_removed": mr, "mean_norm2": gm,
            "pred_resid_norm2": np2, "true_resid_norm2": nt2}


def cos_pair(p_row, q, t, G, train_idx):
    """Raw and mean-removed cosine of one prediction against one true delta.

    `p_row[i] = <pred, dW_i>`, `q = <pred, pred>`, `G` the true-true Gram.  The
    corpus mean m is over `train_idx` ONLY; every term below touches the
    held-out trait t solely through its own entries, never through m.
    """
    train_idx = np.asarray(train_idx)
    assert t not in set(train_idx.tolist()), (
        f"leakage: held-out trait {t} is inside the mean's training set")
    gm = float(G[np.ix_(train_idx, train_idx)].mean())     # <m, m>
    gtm = float(G[t, train_idx].mean())                    # <dW_t, m>
    pm = float(p_row[train_idx].mean())                    # <pred, m>
    gtt, ptt, q = float(G[t, t]), float(p_row[t]), float(q)
    return cos_terms(ptt, q, gtt, pm, gtm, gm)


def cos_pair_ood(p_row, q, ptt, gtt, gt_row, G):
    """The same two cosines for a test trait that is OUTSIDE the training corpus.

    `p_row[i] = <pred, dW_i>` and `gt_row[i] = <dW_test, dW_i>` run over the
    TRAINING adapters; `G` is the training corpus's own Gram, so the reference
    mean m is the mean over ALL of it.  `ptt = <pred, dW_test>` and
    `gtt = <dW_test, dW_test>` are the only quantities that touch the test
    adapter, and neither can reach m.

    NO LEAKAGE ASSERTION IS POSSIBLE OR NEEDED, which is the interesting
    difference from `cos_pair`.  There, the held-out trait is a row of the same
    Gram the mean is taken over and the only thing keeping it out of its own
    reference is an index set, so the boundary has to be asserted.  Here the two
    corpora are disjoint sets of directories from different instruments: no test
    trait is IN the training corpus, so it cannot enter the mean, cannot be
    retrieved, and cannot be the random draw.  The structure enforces what an
    assertion enforced before.
    """
    return cos_terms(float(ptt), float(q), float(gtt),
                     float(np.mean(p_row)),      # <pred, m>
                     float(np.mean(gt_row)),     # <dW_test, m>
                     float(G.mean()))            # <m, m>, all 100 x 100 pairs


def baseline_rows(G, Tcos, t, train_idx, rng):
    """(p_row, q) for each retrieval-style baseline, all read off the Gram."""
    train_idx = np.asarray(train_idx)
    out = {}
    out["mean_adapter"] = (G[train_idx].mean(0),
                           float(G[np.ix_(train_idx, train_idx)].mean()))
    j = int(train_idx[int(np.argmax(Tcos[t, train_idx]))])
    out["nn_text_retrieval"] = (G[j], float(G[j, j]), j)
    k = int(rng.choice(train_idx))
    out["random_adapter"] = (G[k], float(G[k, k]), k)
    return out


def ood_baselines(G, gt_row, tcos_row, rng):
    """The same three baselines for one OOD test trait: no index may be excluded.

    Returns per method (p_row over the training corpus, <p,p>, <p,dW_test>,
    source index).  `baseline_rows` takes a `train_idx` because in CV the pool
    must exclude the trait being scored; here the pool IS the training corpus and
    the test trait is not in it, so there is nothing to exclude -- the whole
    corpus is a legal source and the metric's mean is over the whole corpus too.

    THE POOL IS ALL 100, not the 80 the optimiser took gradients on.  The
    hypernetwork used the other 20 as its stopping yardstick, so both methods are
    handed the same thing: "we own 100 trait adapters".  Narrowing retrieval to
    80 would hand the generator an advantage it did not earn, in the experiment
    whose entire point is whether it beats retrieval.
    """
    out = {}
    out["mean_adapter"] = (G.mean(0), float(G.mean()), float(gt_row.mean()), None)
    j = int(np.argmax(tcos_row))
    out["nn_text_retrieval"] = (G[j], float(G[j, j]), float(gt_row[j]), j)
    k = int(rng.choice(G.shape[0]))
    out["random_adapter"] = (G[k], float(G[k, k]), float(gt_row[k]), k)
    return out


# ===========================================================================
# writing predicted adapters
# ===========================================================================
def write_pred_adapters(dest, names, modules, pred, src_config, fold_of, cfg,
                        log=sys.stderr, norm_fix=None):
    from safetensors.torch import save_file

    # One directory per objective: the two losses produce different adapters for
    # the same trait names, and the loser silently overwriting the winner is the
    # kind of collision that cost this project four adapters already.
    if cfg.get("loss", "sqerr") != "sqerr":
        dest = os.path.join(dest, cfg["loss"])
    os.makedirs(dest, exist_ok=True)
    total = 0
    for i, name in enumerate(names):
        tens = {}
        # sqrt because ||dW|| = ||s B A|| scales as gain^2 across the two factors
        f = float(np.sqrt(norm_fix[i])) if norm_fix is not None else 1.0
        for gi, g in enumerate(pred):
            for k, mi in enumerate(g["mod_idx"].tolist()):
                m = modules[mi]
                a = (g["A"][k, i] * f).cpu().contiguous()
                b = (g["B"][k, i] * f).cpu().contiguous()
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
                    f"{name}/{m}: non-finite after rescale by {f}")
                tens[m + ".lora_A.weight"] = a
                tens[m + ".lora_B.weight"] = b
        d = os.path.join(dest, name)
        os.makedirs(d, exist_ok=True)
        save_file(tens, os.path.join(d, "adapter_model.safetensors"))
        json.dump(src_config, open(os.path.join(d, "adapter_config.json"), "w"), indent=1)
        # Deliberately NO init_seed / hparams.epochs: these are predictions, and
        # any provenance check of the seeds_analysis.py kind must refuse them
        # rather than quietly treat them as trained adapters.
        json.dump({"trait": name, "predicted": True, "predicted_by": "hypernet_t2l",
                   "out_of_fold": int(fold_of[i]), "dtype": cfg["pred_dtype"],
                   "loss": cfg.get("loss", "sqerr"), "norm_rescale": f,
                   "source_corpus": "sweep100 seed 0", "config": cfg},
                  open(os.path.join(d, "runmeta.json"), "w"), indent=1)
        total += os.path.getsize(os.path.join(d, "adapter_model.safetensors"))
    print(f"  wrote {len(names)} predicted adapters, {total/1e9:.2f} GB into {dest}",
          file=log, flush=True)
    return total


# ===========================================================================
# the run itself (identical code local or remote)
# ===========================================================================
def load_corpus(adir, tvecs, order, names, cfg, device, log=sys.stderr):
    """The stack, the module list and the text features -- the whole read side.

    Split out of run_cv unchanged so that --curve pays it ONCE for twelve grid
    points instead of twelve times; run_cv still calls it itself when no cached
    corpus is handed in, so the default path is exactly what it was.
    """
    stack, modules, _ = load_stack(adir, names, device, log)
    rank = read_config(adir, names[0])["r"]
    X = text_features(tvecs, names, order).to(device)
    return {"stack": stack, "modules": modules, "X": X, "rank": rank}


def reseed_ceiling(adir, stack, modules, names, G, cfg, device, log=sys.stderr,
                   reseeds=()):
    """The same-trait/different-data-order ceiling, raw and mean-removed.

    Depends on nothing the hypernetwork did -- only on the corpus Gram -- so the
    learning curve computes it once and reuses it as the constant reference line
    it is.  Lifted verbatim out of run_cv.
    """
    n = len(names)
    ceiling = {"raw_reference": CEILING_RESEED_COS, "source": "pca.json noise_floor"}
    if reseeds:
        rstack, rmods, _ = load_stack(adir, list(reseeds), device, log)
        assert rmods == modules, "reseed adapters have different modules"
        RG = gram_between(rstack, stack, cfg["scale"]).cpu().numpy()
        RQ = self_norms2(rstack, cfg["scale"]).cpu().numpy()
        pos = {nm: i for i, nm in enumerate(names)}
        raws, mrs = [], []
        for ri, rn in enumerate(reseeds):
            t = pos[rn[:-len(RESEED_SUFFIX)]]
            # the honest mean for a same-trait comparison excludes that trait
            tr = np.array([i for i in range(n) if i != t])
            c = cos_pair(RG[ri], RQ[ri], t, G, tr)
            raws.append(c["raw"])
            mrs.append(c["mean_removed"])
        ceiling.update({"reseeds": list(reseeds), "raw": float(np.mean(raws)),
                        "mean_removed": float(np.mean(mrs))})
        del rstack
    return ceiling


def run_cv(adir, tvecs, order, names, dest, cfg, device, log=sys.stderr,
           reseeds=(), corpus=None, splits=None, eval_pools=None,
           G=None, ceiling=None):
    """Five-fold CV over the corpus.

    The optional arguments exist for --curve and default to exactly the previous
    behaviour when omitted:

      corpus      a cached load_corpus() result, to avoid re-reading 100 adapters
      splits      per-fold (train_idx, val_idx), overriding the standard draw --
                  this is where a subsampled training set enters
      eval_pools  per-fold index set that the corpus mean, the mean-REMOVAL and
                  the retrieval/random baselines may see.  Default: every
                  non-test trait, as before.  --curve passes the subsample, so
                  no method is scored against traits the hypernetwork never had.
      G, ceiling  precomputed corpus Gram / reseed ceiling; neither depends on
                  training, so the curve computes each once for the whole grid.
    """
    t0 = time.time()
    if corpus is None:
        corpus = load_corpus(adir, tvecs, order, names, cfg, device, log)
    stack, modules, X = corpus["stack"], corpus["modules"], corpus["X"]
    cfg = dict(cfg, rank=corpus["rank"])
    n = len(names)
    dt = getattr(torch, cfg["pred_dtype"])
    pred = [{"key": g["key"], "mod_idx": g["mod_idx"],
             "A": torch.zeros_like(g["A"], dtype=dt),
             "B": torch.zeros_like(g["B"], dtype=dt)} for g in stack]

    folds = kfold(n, cfg["folds"], cfg["split_seed"])
    assert sorted(np.concatenate(folds).tolist()) == list(range(n)), "not a partition"

    rng = np.random.default_rng(cfg["split_seed"])
    if splits is None:
        splits = fold_splits(n, folds, cfg, rng)
    assert len(splits) == len(folds), "one (train, val) split per fold"
    fold_of = np.zeros(n, dtype=int)
    fold_log = []
    for f, test_idx in enumerate(folds):
        fold_of[test_idx] = f
        train_idx, val_idx = np.asarray(splits[f][0]), np.asarray(splits[f][1])
        # THE LEAKAGE BOUNDARY, asserted here rather than trusted, because with
        # --curve the split arrives from outside this function.
        held = set(test_idx.tolist())
        assert not (set(train_idx.tolist()) & held), f"fold {f}: train/test leak"
        assert not (set(val_idx.tolist()) & held), f"fold {f}: val/test leak"
        print(f"  fold {f}: train {len(train_idx)} val {len(val_idx)} "
              f"test {len(test_idx)}", file=log, flush=True)
        model, info = train_fold(stack, X, len(modules), train_idx, val_idx,
                                 test_idx, cfg, device, log)
        predict_into(model, stack, pred, X, test_idx, device, cfg["batch_traits"])
        info["fold"] = f
        info["params"] = param_count(model)
        info["train"], info["val"] = train_idx.tolist(), val_idx.tolist()
        info["test"] = test_idx.tolist()
        fold_log.append(info)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("  computing grams", file=log, flush=True)
    if G is None:
        G = gram_between(stack, stack, cfg["scale"], log).cpu().numpy()
    P = gram_between(pred, stack, cfg["scale"], log).cpu().numpy()
    Q = self_norms2(pred, cfg["scale"]).cpu().numpy()

    Xn = X.cpu().numpy()
    Tcos = Xn @ Xn.T                       # text features are already unit norm

    if ceiling is None:
        ceiling = reseed_ceiling(adir, stack, modules, names, G, cfg, device,
                                log, reseeds)

    methods = ["hypernet", "mean_adapter", "nn_text_retrieval", "random_adapter"]
    per_trait, picks = {}, {}
    brng = np.random.default_rng(cfg["split_seed"] + 1)
    for f, test_idx in enumerate(folds):
        # The pool every non-hypernet method is allowed to see, and the set the
        # corpus mean is taken over.  Default: all non-test traits, as always.
        # --curve narrows it to the fold's subsampled TRAINING traits, so the
        # retrieval bar is the bar for the data the hypernetwork actually had.
        if eval_pools is None:
            pool = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        else:
            pool = np.asarray(eval_pools[f], dtype=int)
            assert not (set(pool.tolist()) & set(test_idx.tolist())), (
                f"fold {f}: the evaluation pool contains held-out traits")
        train_idx = pool
        for t in test_idx.tolist():
            rows = baseline_rows(G, Tcos, t, train_idx, brng)
            res = {"hypernet": cos_pair(P[t], Q[t], t, G, train_idx)}
            for k, v in rows.items():
                res[k] = cos_pair(v[0], v[1], t, G, train_idx)
                if len(v) > 2:
                    res[k]["source_trait"] = names[v[2]]
            per_trait[names[t]] = {"fold": f,
                                   **{m: {kk: res[m][kk] for kk in
                                          ("raw", "mean_removed")} for m in methods}}
            picks[names[t]] = res["nn_text_retrieval"].get("source_trait")

    summary = {}
    for m in methods:
        raw = np.array([per_trait[nm][m]["raw"] for nm in names])
        mr = np.array([per_trait[nm][m]["mean_removed"] for nm in names])
        summary[m] = {"raw_mean": float(raw.mean()), "raw_sd": float(raw.std()),
                      "mean_removed_mean": float(mr.mean()),
                      "mean_removed_sd": float(mr.std()),
                      "mean_removed_median": float(np.median(mr))}

    out = {"config": cfg, "n_traits": n, "n_modules": len(modules),
           "traits": names, "folds": [f.tolist() for f in folds],
           "ceiling": ceiling, "summary": summary, "per_trait": per_trait,
           "nn_retrieval_picks": picks, "fold_log": fold_log,
           "wall_seconds": time.time() - t0}
    if eval_pools is not None:
        out["eval_pools"] = [np.asarray(p, dtype=int).tolist() for p in eval_pools]

    if dest:
        # The centred cosine is scale-invariant: it is minimised anywhere on the
        # ray p = m + c(t - m), so nothing pins ||p|| and the learned gain can
        # drift arbitrarily.  That is harmless for the reported cosines and fatal
        # for the artefacts -- an adapter with the wrong gain expresses its trait
        # too faintly or too violently, and can overflow fp16.  Rescale each
        # prediction to the corpus's typical ||dW|| at WRITE time, so the
        # objective stays pure and the scored numbers are untouched.  The sqerr
        # path already calibrates its gain, so leave it exactly as it was.
        norm_fix = None
        if cfg.get("loss", "sqerr") != "sqerr":
            target = float(np.sqrt(np.diag(G).mean()))
            norm_fix = target / np.sqrt(np.maximum(Q, 1e-300))
            print(f"  rescaling predictions to ||dW||={target:.4f} "
                  f"(factors {norm_fix.min():.3f}-{norm_fix.max():.3f})",
                  file=log, flush=True)
        cfgsrc = read_config(adir, names[0])
        out["pred_bytes"] = write_pred_adapters(dest, names, modules, pred,
                                                cfgsrc, fold_of, cfg, log,
                                                norm_fix=norm_fix)
    # ~6 GB of fp16 predictions on the real corpus.  Harmless when run_cv is
    # called once; twelve of them in sequence is how --curve would OOM a card
    # that has already given 12 GB to the stack it must keep.
    del pred
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


CURVE_METHODS = ("hypernet", "mean_adapter", "nn_text_retrieval", "random_adapter")


def curve_subsamples(base_orders, folds, n_avail, rep, seed, val_frac):
    """One grid point: per-fold (train, val) and the available pool it came from.

    `n_avail` traits are kept from the fold's non-test pool and then cut into
    validation and training by `split_at`, the standard run's own rule -- so the
    hypernetwork trains on (1 - val_frac) * n_avail and every baseline is scored
    against all n_avail, which is what "we own n adapters" actually means.

    THE POSITIONS ARE SORTED BACK INTO THE STANDARD PRIORITY ORDER after being
    chosen.  That is the whole trick behind the sanity check: at n_avail = the
    full pool every position is kept, the order is the standard order untouched,
    and `split_at` reproduces the standard (train, val) exactly -- so that point
    IS the published run, on all four methods, not an approximation of it.

    NESTED across sizes on purpose: the permutation is keyed on (seed, rep,
    fold) and NOT on n_avail, so a larger point strictly ADDS traits to a
    smaller one.  Independent draws per size would measure subsample luck as
    much as data, and the slope is the thing being asked for.

    Validation comes from INSIDE the subsample.  A fixed val set held outside it
    would be a yardstick built from traits the point does not own -- comparable
    across sizes, but a leak, and it would also stop the largest point from
    being the standard split.
    """
    splits, pools = [], []
    for f, order in enumerate(base_orders):
        assert n_avail <= len(order), (
            f"fold {f}: asked for {n_avail} available traits, pool has "
            f"{len(order)}")
        keep = np.sort(np.random.default_rng([seed, rep, f])
                       .permutation(len(order))[:n_avail])
        avail = order[keep]
        tr, va = split_at(avail, val_frac)
        # the leakage boundary, checked where the subsample is made and again
        # inside run_cv where it is used
        held = set(folds[f].tolist())
        assert not (set(avail.tolist()) & held), (
            f"fold {f}: subsample overlaps the test fold")
        assert not (set(tr.tolist()) & set(va.tolist())), (
            f"fold {f}: train/val overlap inside the subsample")
        splits.append((tr, va))
        pools.append(np.sort(avail))
    return splits, pools


def run_curve(adir, tvecs, order, names, cfg, curve, device, log=sys.stderr,
              reseeds=()):
    """The learning curve: all four methods as the pool of AVAILABLE traits grows.

    Everything expensive that does not depend on training is done once -- the
    corpus load, the true-true Gram, the reseed ceiling -- and each grid point
    is an ordinary run_cv with a subsampled split, no predicted adapters
    written (dest="") and those constants handed in.

    Each point also carries `steps_run` per fold and `hit_cap`: True means at
    least one of its folds ran out of steps instead of running out of validation
    improvement, and that point's hypernet number is then partly a statement
    about compute rather than about how many traits it had.
    """
    t0 = time.time()
    corpus = load_corpus(adir, tvecs, order, names, cfg, device, log)
    cfg = dict(cfg, rank=corpus["rank"])
    n = len(names)
    folds = kfold(n, cfg["folds"], cfg["split_seed"])
    base_orders = fold_orders(n, folds, cfg,
                              np.random.default_rng(cfg["split_seed"]))
    pool_sizes = [len(o) for o in base_orders]

    sizes = sorted({int(s) for s in curve["sizes"]})
    assert sizes and sizes[0] >= 2, f"bad curve sizes {curve['sizes']}"
    assert sizes[-1] <= min(pool_sizes), (
        f"largest curve size {sizes[-1]} exceeds the smallest fold's available "
        f"pool {min(pool_sizes)} -- the sizes must all be reachable in EVERY "
        f"fold or the points are not comparable")

    print(f"  curve: sizes {sizes} x {curve['reps']} reps x {cfg['folds']} folds "
          f"= {len(sizes)*curve['reps']*cfg['folds']} fold-trainings; "
          f"available pools {pool_sizes}; lr_schedule {cfg['lr_schedule']}, "
          f"step cap {cfg['steps']}", file=log, flush=True)
    print("  computing the corpus Gram and the ceiling once", file=log, flush=True)
    G = gram_between(corpus["stack"], corpus["stack"], cfg["scale"], log).cpu().numpy()
    ceiling = reseed_ceiling(adir, corpus["stack"], corpus["modules"], names, G,
                             cfg, device, log, reseeds)

    points = []
    for n_avail in sizes:
        for rep in range(int(curve["reps"])):
            splits, pools = curve_subsamples(base_orders, folds, n_avail, rep,
                                             curve["subsample_seed"],
                                             cfg["val_frac"])
            print(f"\n  == n_avail {n_avail}  rep {rep} ==", file=log, flush=True)
            out = run_cv(adir, tvecs, order, names, "", cfg, device, log,
                         reseeds=(), corpus=corpus, splits=splits,
                         eval_pools=pools, G=G, ceiling=ceiling)
            # A fold that used its whole budget never stopped on patience, so it
            # cannot be called converged.  Conservative on the boundary: equality
            # counts as hitting the cap, because a warning we did not need is
            # cheaper than a confound we did not see.
            steps_run = [int(fl["steps_run"]) for fl in out["fold_log"]]
            at_cap = [s >= int(cfg["steps"]) for s in steps_run]
            points.append({
                "n_avail": n_avail, "rep": rep,
                "n_train": [len(s[0]) for s in splits],
                "n_val": [len(s[1]) for s in splits],
                "summary": out["summary"],
                # kept in run_cv's own shape, raw and mean-removed both, so a
                # point can be compared trait-by-trait against a non-curve run
                "per_trait": out["per_trait"],
                # which traits were used, by name, per fold -- the record that
                # makes a point reproducible without re-deriving the rng
                "train_traits": [[names[i] for i in s[0].tolist()] for s in splits],
                "val_traits": [[names[i] for i in s[1].tolist()] for s in splits],
                "avail_traits": [[names[i] for i in p.tolist()] for p in pools],
                "fold_best_val": [fl["best_val"] for fl in out["fold_log"]],
                "steps_run": steps_run,
                "steps_cap": int(cfg["steps"]),
                "folds_at_cap": int(sum(at_cap)),
                "hit_cap": bool(any(at_cap)),
                "wall_seconds": out["wall_seconds"]})
            s = out["summary"]
            print(f"  n_avail {n_avail} rep {rep}: "
                  + "  ".join(f"{m}={s[m]['mean_removed_mean']:+.3f}"
                              for m in CURVE_METHODS)
                  + f"   steps {steps_run} cap={points[-1]['hit_cap']}"
                  + f"   ({out['wall_seconds']:.0f}s)", file=log, flush=True)

    grid = {}
    for n_avail in sizes:
        at = [p for p in points if p["n_avail"] == n_avail]
        row = {}
        for m in CURVE_METHODS:
            v = np.array([p["summary"][m]["mean_removed_mean"] for p in at])
            vr = np.array([p["summary"][m]["raw_mean"] for p in at])
            row[m] = {"mean_removed_mean": float(v.mean()),
                      # sd ACROSS REPEATS -- subsample noise, not across traits
                      "mean_removed_sd_over_reps": float(v.std(ddof=1))
                      if v.size > 1 else 0.0,
                      "mean_removed_reps": v.tolist(),
                      "raw_mean": float(vr.mean())}
        row["convergence"] = {
            "n_train": int(np.mean([np.mean(p["n_train"]) for p in at])),
            "steps_run_mean": float(np.mean([np.mean(p["steps_run"]) for p in at])),
            "steps_run_max": int(max(max(p["steps_run"]) for p in at)),
            "folds_at_cap": int(sum(p["folds_at_cap"] for p in at)),
            "folds_total": int(sum(len(p["steps_run"]) for p in at)),
            "hit_cap": bool(any(p["hit_cap"] for p in at))}
        grid[str(n_avail)] = row

    return {"config": cfg, "curve": dict(curve, sizes=sizes),
            "n_traits": n, "traits": names,
            "folds": [f.tolist() for f in folds],
            "fold_pool_sizes": pool_sizes,
            "axis": "n_avail = training + validation traits owned per fold; "
                    "the largest size is the standard split exactly",
            "ceiling": ceiling, "grid": grid, "points": points,
            "wall_seconds": time.time() - t0}


def curve_cap_warning(out):
    """The points whose numbers confound data with compute.  Loud on purpose."""
    bad = [p for p in out["points"] if p["hit_cap"]]
    if not bad:
        return []
    cap = out["config"]["steps"]
    print("\n" + "!" * 78)
    print("!! WARNING: NOT CONVERGED -- these points stopped on the step cap "
          f"({cap}), not on")
    print("!! validation patience.  Their hypernet scores are a statement about "
          "COMPUTE as")
    print("!! much as about the number of traits, and the SLOPE they imply cannot "
          "be trusted:")
    for p in bad:
        print(f"!!   n_avail {p['n_avail']:>3d}  rep {p['rep']}  "
              f"{p['folds_at_cap']}/{len(p['steps_run'])} folds at the cap  "
              f"steps_run {p['steps_run']}")
    print("!! Re-run with a larger --curve-steps until this warning is gone "
          "before drawing")
    print("!! any conclusion about whether more traits would close the gap.")
    print("!" * 78)
    return bad


def print_curve_table(out):
    LABEL = {"hypernet": "hypernet", "mean_adapter": "corpus mean",
             "nn_text_retrieval": "NN retrieval", "random_adapter": "random adapter"}
    sizes = sorted(int(k) for k in out["grid"])
    reps = int(out["curve"]["reps"])
    c = out["ceiling"]
    ceil = c.get("mean_removed", c["raw_reference"])
    w, W = 17, 10 + 17 * 4 + 20
    print("\n" + "=" * W)
    print(f"LEARNING CURVE -- mean-removed cosine, mean +/- sd over {reps} "
          f"subsample repeats")
    print(f"x-axis: traits AVAILABLE per fold (train + val); "
          f"{sizes[-1]} = the standard split")
    print("-" * W)
    print(f"{'n_avail':>9s}" + "".join(f"{lab:>{w}s}" for lab in LABEL.values())
          + f"{'steps':>10s}{'converged':>10s}")
    for nt in sizes:
        row = out["grid"][str(nt)]
        cells = "".join(
            f"{row[m]['mean_removed_mean']:+.3f}+/-{row[m]['mean_removed_sd_over_reps']:.3f}".rjust(w)
            for m in LABEL)
        cv = row["convergence"]
        # steps_run and hit_cap belong in the table, not just the JSON: a reader
        # who does not see them cannot tell a data effect from a compute one
        flag = ("AT CAP" if cv["hit_cap"] else "yes")
        print(f"{nt:>9d}" + cells + f"{cv['steps_run_mean']:>10.0f}"
              + f"{flag:>10s}")
    print("-" * W)
    print(f"{'CEILING':>9s}" + f"{ceil:+.3f}".rjust(w) + "   (constant reference: "
          "same trait, different data order)")
    print("=" * W)
    curve_cap_warning(out)
    if len(sizes) >= 2:
        lo, hi = sizes[0], sizes[-1]
        print(f"slope over n_avail {lo} -> {hi}, per 10 traits:")
        for m, lab in LABEL.items():
            d = (out["grid"][str(hi)][m]["mean_removed_mean"]
                 - out["grid"][str(lo)][m]["mean_removed_mean"]) / (hi - lo) * 10
            print(f"  {lab:16s} {d:+.4f}")
        print("READ IT AS: the gap closes with more traits only if the hypernet's"
              "\nslope exceeds retrieval's.  Equal slopes mean more traits will not"
              "\nclose it; a steeper retrieval slope means more traits make the"
              "\ngenerator's relative case WORSE.")


def print_table(out):
    LABEL = {"hypernet": "hypernet (text -> LoRA)",
             "mean_adapter": "baseline: corpus mean",
             "nn_text_retrieval": "baseline: NN text retrieval  <- the bar",
             "random_adapter": "baseline: random train adapter"}
    c = out["ceiling"]
    print("\n" + "=" * 72)
    print(f"{'method':38s} {'raw cos':>13s} {'mean-removed':>16s}")
    print("-" * 72)
    for m, lab in LABEL.items():
        s = out["summary"][m]
        print(f"{lab:38s} {s['raw_mean']:+13.3f} {s['mean_removed_mean']:+16.3f}")
    print("-" * 72)
    mr = f"{c['mean_removed']:+16.3f}" if "mean_removed" in c else f"{'n/a':>16s}"
    print(f"{'CEILING (same trait, data order)':38s} {c['raw_reference']:+13.3f} {mr}")
    print("=" * 72)
    print("mean-removed is the headline: raw cosine is inflated by the component"
          "\nevery adapter shares, which carries no trait information at all.")


# ===========================================================================
# out of distribution -- train on one corpus, evaluate on a foreign one
# ===========================================================================
OOD_METHODS = ("hypernet", "mean_adapter", "nn_text_retrieval", "random_adapter")


def ood_split(n, cfg):
    """(train, val) over the WHOLE training corpus -- there is no test fold here.

    The test traits live in another corpus entirely, so nothing has to be
    withheld from this one to create them; what is still needed is a stopping
    rule.  The cut is `split_at`'s, on a permutation drawn from `split_seed`,
    with the standard run's `val_frac` -- so the hypernetwork takes gradients on
    (1 - val_frac) * 100 traits and stops on the rest, exactly the proportions
    every published fold used.
    """
    order = np.random.default_rng(cfg["split_seed"]).permutation(n)
    return split_at(order, cfg["val_frac"])


def assert_compatible_corpora(adir_a, names_a, adir_b, names_b, log=sys.stderr):
    """The two corpora must be the SAME KIND OF OBJECT.  Checked before training.

    From runmeta/adapter_config JSON and safetensors HEADERS only, so a mismatch
    costs seconds instead of being discovered after an hour on a GPU.

    PROVENANCE FIRST AND HARDEST.  Every adapter on BOTH sides must report
    init_seed 0 and epochs 2.0.  Two LoRA inits are near-orthogonal (cosine
    0.013, seeds_analysis.py), so a test corpus trained from a different init
    would not be a harder distribution -- it would be unlearnable noise, and the
    hypernetwork would score ~0 for a reason that has nothing whatever to do
    with the question being asked.  That failure mode looks exactly like the
    interesting negative result, which is why it is excluded by assertion.

    THEN RANK, SCALE AND THE MODULE LIST.  Every contraction in this file zips
    two stacks module by module in a shared sorted order; a module list that
    merely OVERLAPS would be silently mismatched rather than loudly wrong, and a
    different rank or alpha/r would change what the numbers even mean.
    """
    check_provenance(adir_a, names_a)
    check_provenance(adir_b, names_b)
    rank = read_config(adir_a, names_a[0])["r"]
    for adir, nms in ((adir_a, names_a), (adir_b, names_b)):
        for nm in nms:
            read_config(adir, nm, rank)        # asserts r AND alpha/r == scale
    mods_a = corpus_modules(adir_a, names_a)
    mods_b = corpus_modules(adir_b, names_b)
    assert mods_a == mods_b, (
        f"the two corpora target different modules: "
        f"{len(mods_a)} vs {len(mods_b)}; "
        f"only in A {sorted(set(mods_a) - set(mods_b))[:5]}; "
        f"only in B {sorted(set(mods_b) - set(mods_a))[:5]}")
    print(f"  corpora compatible: {len(names_a)} + {len(names_b)} adapters, "
          f"rank {rank}, scale {LORA_SCALE}, {len(mods_a)} shared modules, "
          f"init_seed 0 / epochs {EXPECTED_EPOCHS} throughout",
          file=log, flush=True)
    return rank, mods_a


def run_ood(adir, tvecs, order, names, adir_te, tvecs_te, names_te, cfg,
            device, log=sys.stderr):
    """Train once on the training corpus, score all four methods on a foreign one.

    No folds and no cross-validation: the test traits are held out by
    construction (different instruments, different directories), so the model
    trains on ALL of the training corpus and is scored on ALL of the test one.

    THE MEMORY SHAPE IS WHY THIS IS CHUNKED.  The training stack is ~12 GB of
    fp32 on the real corpus and the centred objective's fold mean another ~9.6
    GB; a resident test stack of 148 adapters would be a further ~17.7 GB, plus
    predictions.  So the test corpus is streamed `ood_chunk` traits at a time:
    load, predict, contract into the five arrays the scorer needs, drop.  Peak is
    the training stack plus one chunk, and nothing dense is ever built -- the
    quantities collected are inner products only:

        P[j]  = <pred_j, dW_i>        over the training corpus   (n_te, n)
        GX[j] = <dW_test_j, dW_i>     over the training corpus   (n_te, n)
        Q[j]  = <pred_j, pred_j>      TT[j] = <dW_test_j, dW_test_j>
        PT[j] = <pred_j, dW_test_j>   the DIAGONAL only; see diag_between

    Predictions are held in `pred_dtype` (fp16) exactly as run_cv holds them, so
    the OOD number and the quoted in-distribution number are computed from
    equally-rounded predictions rather than differing by their storage.
    """
    t0 = time.time()
    n, n_te = len(names), len(names_te)
    assert n_te, "no test adapters"
    shared = sorted(set(names) & set(names_te))
    assert not shared, (
        f"the two corpora share trait names {shared[:8]} -- a 'held out by "
        f"construction' test trait that is also in the training corpus is not "
        f"held out at all")
    rank, modules = assert_compatible_corpora(adir, names, adir_te, names_te, log)
    cfg = dict(cfg, rank=rank)

    # Both sides align text features to ADAPTER DIRECTORY NAMES, never to a row
    # position: the row order comes from each npz's own `names` where it has one
    # (the expanded file does; text_vecs.npz predates the field and is in
    # traits.json order), and `text_features` permutes into the name order the
    # stack was loaded in.
    corpus = load_corpus(adir, tvecs, npz_order(tvecs, order), names, cfg,
                         device, log)
    stack, X = corpus["stack"], corpus["X"]
    assert corpus["modules"] == modules, "training modules moved between reads"
    Xte = text_features(tvecs_te, names_te, npz_order(tvecs_te)).to(device)

    train_idx, val_idx = ood_split(n, cfg)
    print(f"  OOD: train {len(train_idx)} / val {len(val_idx)} of {n} training "
          f"traits -> {n_te} test traits from other instruments; objective "
          f"{cfg['loss']}, lr {cfg['lr_schedule']}, step cap {cfg['steps']}",
          file=log, flush=True)
    # test_idx is empty: the held-out set is a different corpus, so train_fold's
    # leakage assertions have nothing to bite on and the fold mean is built from
    # the training split with the validation split forbidden, as always.
    model, info = train_fold(stack, X, len(modules), train_idx, val_idx,
                             np.array([], dtype=int), cfg, device, log)
    steps_run = int(info["steps_run"])
    hit_cap = steps_run >= int(cfg["steps"])

    print("  computing the training corpus Gram", file=log, flush=True)
    G = gram_between(stack, stack, cfg["scale"], log).cpu().numpy()
    Xn, Xten = X.cpu().numpy(), Xte.cpu().numpy()
    Tcross = Xten @ Xn.T                   # both sides are already unit norm

    dt = getattr(torch, cfg["pred_dtype"])
    ch = int(cfg.get("ood_chunk") or n_te)
    P = np.zeros((n_te, n))
    GX = np.zeros((n_te, n))
    Q = np.zeros(n_te)
    TT = np.zeros(n_te)
    PT = np.zeros(n_te)
    for s in range(0, n_te, ch):
        sub = list(names_te[s:s + ch])
        te, te_mods, _ = load_stack(adir_te, sub, device, log, rank=cfg["rank"])
        assert te_mods == modules, "test chunk exposes a different module list"
        # Group ORDER and per-group module indices must match, not just the
        # module set: every contraction below zips the two stacks positionally.
        assert [g["key"] for g in te] == [g["key"] for g in stack] and all(
            torch.equal(a["mod_idx"], b["mod_idx"]) for a, b in zip(te, stack)), (
            "test stack groups are not aligned with the training stack")
        pred = [{"key": g["key"], "mod_idx": g["mod_idx"],
                 "A": torch.zeros((g["A"].shape[0], len(sub)) + tuple(g["A"].shape[2:]),
                                  dtype=dt, device=device),
                 "B": torch.zeros((g["B"].shape[0], len(sub)) + tuple(g["B"].shape[2:]),
                                  dtype=dt, device=device)} for g in stack]
        predict_into(model, stack, pred, Xte[s:s + len(sub)],
                     np.arange(len(sub)), device, cfg["batch_traits"])
        e = s + len(sub)
        P[s:e] = gram_between(pred, stack, cfg["scale"]).cpu().numpy()
        GX[s:e] = gram_between(te, stack, cfg["scale"]).cpu().numpy()
        Q[s:e] = self_norms2(pred, cfg["scale"]).cpu().numpy()
        TT[s:e] = self_norms2(te, cfg["scale"]).cpu().numpy()
        PT[s:e] = diag_between(pred, te, cfg["scale"]).cpu().numpy()
        print(f"  scored test traits {s}-{e-1} of {n_te}  "
              f"{time.time()-t0:.0f}s", file=log, flush=True)
        del te, pred
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # THE REFERENCE MEAN IS THE MEAN OVER THE TRAINING CORPUS, and leakage into
    # it is structurally impossible here rather than merely asserted away: no
    # test trait is a member of the training corpus, so `G` -- and therefore
    # <m,m>, <p,m> and <t,m> -- cannot contain the trait being scored.  In the CV
    # case the held-out trait IS a row of the same Gram and only an index set
    # keeps it out of its own reference, which is why `cos_pair` has an
    # assertion and `cos_pair_ood` needs none.
    per_trait, picks = {}, {}
    brng = np.random.default_rng(cfg["split_seed"] + 1)
    for ti, nm in enumerate(names_te):
        res = {"hypernet": cos_pair_ood(P[ti], Q[ti], PT[ti], TT[ti], GX[ti], G)}
        for k, v in ood_baselines(G, GX[ti], Tcross[ti], brng).items():
            res[k] = cos_pair_ood(v[0], v[1], v[2], TT[ti], GX[ti], G)
            if v[3] is not None:
                res[k]["source_trait"] = names[v[3]]
        per_trait[nm] = {m: {kk: res[m][kk] for kk in ("raw", "mean_removed")}
                         for m in OOD_METHODS}
        per_trait[nm]["nn_text_cos"] = float(Tcross[ti].max())
        picks[nm] = res["nn_text_retrieval"].get("source_trait")

    summary, degradation = {}, {}
    for m in OOD_METHODS:
        raw = np.array([per_trait[nm][m]["raw"] for nm in names_te])
        mr = np.array([per_trait[nm][m]["mean_removed"] for nm in names_te])
        summary[m] = {"raw_mean": float(raw.mean()), "raw_sd": float(raw.std()),
                      "mean_removed_mean": float(mr.mean()),
                      "mean_removed_sd": float(mr.std()),
                      "mean_removed_median": float(np.median(mr))}
        ind = IN_DISTRIBUTION.get(m)
        degradation[m] = {
            "in_distribution": ind,
            "out_of_distribution": float(mr.mean()),
            "delta": None if ind is None else float(mr.mean()) - ind}

    # How far outside the training corpus the test traits actually are, in the
    # only space retrieval can see.  If these two numbers were equal there would
    # be no distribution shift to measure and the whole experiment would be void.
    Tin = Xn @ Xn.T
    np.fill_diagonal(Tin, -np.inf)
    text_neighbour = {
        "test_to_train_max_cos_mean": float(Tcross.max(1).mean()),
        "test_to_train_max_cos_median": float(np.median(Tcross.max(1))),
        "train_to_train_max_cos_mean": float(Tin.max(1).mean()),
        "note": "nearest training neighbour in text-difference space; the "
                "train-to-train figure is leave-one-out within the training "
                "corpus and is the in-distribution comparison"}

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {"config": cfg, "mode": "ood",
            "train_corpus": {"dir": adir, "n": n, "traits": list(names),
                             "train": np.asarray(train_idx).tolist(),
                             "val": np.asarray(val_idx).tolist()},
            "test_corpus": {"dir": adir_te, "n": n_te, "traits": list(names_te)},
            "n_modules": len(modules),
            "convergence": {"steps_run": steps_run,
                            "steps_cap": int(cfg["steps"]),
                            "hit_cap": bool(hit_cap),
                            "best_val": info["best_val"],
                            "loss": info["loss"],
                            "lr_schedule": cfg["lr_schedule"],
                            "history": info["history"]},
            "in_distribution": dict(IN_DISTRIBUTION),
            "summary": summary, "degradation": degradation,
            "text_neighbour": text_neighbour,
            "per_trait": per_trait, "nn_retrieval_picks": picks,
            "wall_seconds": time.time() - t0}


def ood_cap_warning(out):
    """A capped OOD number is not usable.  Say so at the top of one's voice."""
    c = out["convergence"]
    if not c["hit_cap"]:
        return False
    print("\n" + "!" * 78)
    print("!! WARNING: NOT CONVERGED -- training stopped on the step cap "
          f"({c['steps_cap']}), not on")
    print("!! validation patience.  THIS NUMBER IS NOT USABLE: an undertrained "
          "hypernetwork")
    print("!! degrades out of distribution for reasons that have nothing to do "
          "with the")
    print("!! distribution, and it degrades in the direction that confirms the "
          "sceptical")
    print("!! hypothesis.  Re-run with a larger --steps until this warning is "
          "gone before")
    print("!! reading the comparison against retrieval at all.")
    print("!" * 78)
    return True


def print_ood_table(out):
    LABEL = {"hypernet": "hypernet (text -> LoRA)",
             "mean_adapter": "baseline: corpus mean",
             "nn_text_retrieval": "baseline: NN text retrieval  <- the bar",
             "random_adapter": "baseline: random train adapter"}
    d, s = out["degradation"], out["summary"]
    tr, te = out["train_corpus"], out["test_corpus"]
    W = 38 + 3 * 14
    print("\n" + "=" * W)
    print(f"OUT OF DISTRIBUTION -- trained on {tr['n']} Goldberg marker traits, "
          f"evaluated on {te['n']}")
    print("traits from other instruments (HEXACO, Dark Triad, IPC, Schwartz, "
          "register).")
    print("mean-removed cosine; in-distribution column is QUOTED from "
          f"{IN_DISTRIBUTION['_source'].split(' grid')[0]}")
    print("-" * W)
    print(f"{'method':38s}{'in-dist':>14s}{'out-of-dist':>14s}{'delta':>14s}")
    for m, lab in LABEL.items():
        ind = d[m]["in_distribution"]
        ic = f"{ind:+.3f}" if ind is not None else "n/a"
        dc = f"{d[m]['delta']:+.3f}" if d[m]["delta"] is not None else "n/a"
        print(f"{lab:38s}{ic:>14s}{d[m]['out_of_distribution']:>+14.3f}{dc:>14s}")
    print("-" * W)
    print(f"{'(raw cosine, out of distribution)':38s}"
          + "".join(f"{s[m]['raw_mean']:>+14.3f}" for m in
                    ("hypernet", "nn_text_retrieval"))
          + "   hypernet, retrieval")
    t = out["text_neighbour"]
    print(f"nearest training neighbour in text space: test {t['test_to_train_max_cos_mean']:.3f} "
          f"vs within-corpus {t['train_to_train_max_cos_mean']:.3f}")
    c = out["convergence"]
    print(f"converged: {'NO -- AT CAP' if c['hit_cap'] else 'yes'}  "
          f"(steps_run {c['steps_run']} of cap {c['steps_cap']}, "
          f"best val {c['best_val']:.4f})")
    print("=" * W)
    capped = ood_cap_warning(out)

    h = d["hypernet"]["out_of_distribution"]
    r = d["nn_text_retrieval"]["out_of_distribution"]
    hd, rd = d["hypernet"]["delta"], d["nn_text_retrieval"]["delta"]
    print("READ IT AS: the registered prediction was that retrieval falls hard "
          "out of\ndistribution while the hypernetwork falls much less, and that "
          "they cross over.")
    if capped:
        print("The run did not converge, so it says nothing yet.")
        return
    print(f"  hypernet  {IN_DISTRIBUTION['hypernet']:+.3f} -> {h:+.3f} "
          f"({hd:+.3f})   retrieval  "
          f"{IN_DISTRIBUTION['nn_text_retrieval']:+.3f} -> {r:+.3f} ({rd:+.3f})")
    if h > r and hd > rd:
        print("  OUTCOME: the prediction holds -- the generator degrades more "
              "gently and\n  keeps its lead; it computes an answer rather than "
              "looking one up.")
    elif abs(hd - rd) < 0.05:
        print("  OUTCOME: THE ALTERNATIVE -- both degrade together, so the "
              "hypernetwork has\n  been interpolating between training traits "
              "all along and its in-distribution\n  win was a smoother lookup, "
              "not a learned mapping.")
    elif h <= r:
        print("  OUTCOME: retrieval WINS out of distribution -- the generator's "
              "advantage was\n  specific to a densely sampled corpus and does "
              "not survive a real shift.")
    else:
        print("  OUTCOME: mixed -- the gap moved but not as predicted; read the "
              "per-trait\n  scores and the text-neighbour distances before "
              "claiming either story.")


# ===========================================================================
# modal plumbing
# ===========================================================================
app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=False)
exp_adapter_vol = modal.Volume.from_name(EXP_ADAPTER_VOLUME, create_if_missing=False)
t2l_vol = modal.Volume.from_name(T2L_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "safetensors==0.5.2", "numpy==1.26.4")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(os.path.join(RDIR, "text_vecs.npz"), "/aux/text_vecs.npz")
    .add_local_file(os.path.join(HERE, "traits.json"), "/aux/traits.json")
)

# The expanded corpus's two files ride on their OWN image layer rather than the
# shared one: the published run and the curve have no business rebuilding because
# a test corpus appeared beside them.
ood_image = (
    image
    .add_local_file(os.path.join(RDIR, "text_vecs_expanded.npz"),
                    "/aux/text_vecs_expanded.npz")
    .add_local_file(os.path.join(HERE, "traits_expanded.json"),
                    "/aux/traits_expanded.json")
)


def _corpus_names():
    traits = json.load(open("/aux/traits.json"))
    order = [slug(t["trait"]) for t in traits]
    names = [nm for nm in order if os.path.isdir(f"/adapters/{nm}")]
    assert len(names) == len(order), f"missing adapters: {set(order)-set(names)}"
    return order, names


def _expanded_names():
    """The 148 expanded traits: the npz's OWN row order, checked against disk.

    traits_expanded.json lists more traits than were ever trained (four have no
    adapter and no text vector), and the npz's row order is not the traits file's
    order.  So the corpus is defined by the npz's `names` array intersected with
    what is actually on the volume, every member is checked to be a trait the
    traits file knows about, and anything listed but absent is reported rather
    than silently dropped.
    """
    order = npz_order("/aux/text_vecs_expanded.npz")
    listed = {slug(t["trait"]) for t in json.load(open("/aux/traits_expanded.json"))}
    unknown = [nm for nm in order if nm not in listed]
    assert not unknown, f"text vectors for traits not in traits_expanded.json: {unknown[:8]}"
    names = [nm for nm in order if os.path.isdir(f"/adapters_expanded/{nm}")]
    assert names == order, (
        f"missing expanded adapters: {sorted(set(order) - set(names))[:8]}")
    absent = sorted(listed - set(order))
    if absent:
        print(f"note: {len(absent)} traits in traits_expanded.json have no "
              f"adapter/text vector and are not evaluated: {absent}")
    return order, names


def _reseed_names(order, with_reseeds):
    if not with_reseeds:
        return []
    return [nm + RESEED_SUFFIX for nm in order
            if os.path.isdir(f"/adapters/{nm}{RESEED_SUFFIX}")]


@app.function(image=image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol, "/t2l": t2l_vol},
              timeout=60 * 240)
def cv_remote(cfg: dict, with_reseeds: bool = True) -> dict:
    order, names = _corpus_names()
    reseeds = _reseed_names(order, with_reseeds)
    out = run_cv("/adapters", "/aux/text_vecs.npz", order, names,
                 "/t2l", cfg, "cuda", sys.stdout, reseeds)
    t2l_vol.commit()
    out["gpu"] = GPU_TYPE
    out["usd_estimate"] = out["wall_seconds"] / 3600.0 * GPU_USD_PER_HOUR
    return out


# No /t2l volume mount: curve mode writes no adapters, and a function that
# cannot reach the volume cannot accidentally overwrite the real predictions
# with a subsample run.  The timeout is 6 h against a ~2 h expected / ~3.8 h
# worst case (every fold running the full 6000-step cap).
@app.function(image=image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol}, timeout=60 * 360)
def curve_remote(cfg: dict, curve: dict, with_reseeds: bool = True) -> dict:
    order, names = _corpus_names()
    reseeds = _reseed_names(order, with_reseeds)
    out = run_curve("/adapters", "/aux/text_vecs.npz", order, names,
                    cfg, curve, "cuda", sys.stdout, reseeds)
    out["gpu"] = GPU_TYPE
    out["usd_estimate"] = out["wall_seconds"] / 3600.0 * GPU_USD_PER_HOUR
    return out


# No /t2l mount, for the reason curve_remote has none: --ood writes no adapters
# and a function that cannot reach the volume cannot overwrite the real
# predictions.  Both adapter volumes are read-only inputs.  The timeout is 5 h
# against a ~1.5 h expectation (~0.13 s/step at the probe's measured rate, so
# ~40 min at the observed ~17k stop, plus two corpus loads and the Grams) and a
# ~1.2 h worst case where the 30000-step cap actually binds.
@app.function(image=ood_image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol,
                       "/adapters_expanded": exp_adapter_vol},
              timeout=60 * 300)
def ood_remote(cfg: dict) -> dict:
    order, names = _corpus_names()
    te_order, te_names = _expanded_names()
    out = run_ood("/adapters", "/aux/text_vecs.npz", order, names,
                  "/adapters_expanded", "/aux/text_vecs_expanded.npz", te_names,
                  cfg, "cuda", sys.stdout)
    out["gpu"] = GPU_TYPE
    out["usd_estimate"] = out["wall_seconds"] / 3600.0 * GPU_USD_PER_HOUR
    return out


@app.local_entrypoint()
def main(steps: int = 0, folds: int = 0, batch_traits: int = 0,
         modules_per_step: int = 0, loss: str = "", reseeds: bool = True,
         lr_schedule: str = "", curve: bool = False, curve_sizes: str = "",
         curve_reps: int = 0, curve_steps: int = 0, ood: bool = False):
    cfg = dict(CFG)
    for k, v in (("steps", steps), ("folds", folds),
                 ("batch_traits", batch_traits),
                 ("modules_per_step", modules_per_step)):
        if v:
            cfg[k] = v
    if loss:
        assert loss in LOSSES, f"unknown --loss {loss!r}, want one of {LOSSES}"
        cfg["loss"] = loss
    if lr_schedule:
        assert lr_schedule in LR_SCHEDULES, (
            f"unknown --lr-schedule {lr_schedule!r}, want one of {LR_SCHEDULES}")
        cfg["lr_schedule"] = lr_schedule

    assert not (curve and ood), "--curve and --ood are different experiments"

    if ood:
        # OOD's own defaults win over CFG's, an explicit flag wins over both --
        # the same precedence --curve uses.
        cfg["steps"] = steps or OOD["steps"]
        cfg["modules_per_step"] = modules_per_step or OOD["modules_per_step"]
        cfg["loss"] = loss or OOD["loss"]
        cfg["lr_schedule"] = lr_schedule or OOD["lr_schedule"]
        print(f"out-of-distribution: train on the 100 Goldberg markers, evaluate "
              f"on the expanded corpus; objective {cfg['loss']}, lr "
              f"{cfg['lr_schedule']}, step cap {cfg['steps']}")
        out = ood_remote.remote(cfg)
        os.makedirs(RDIR, exist_ok=True)
        p = os.path.join(RDIR, "hypernet_t2l_ood.json")
        json.dump(out, open(p, "w"), indent=1)
        print_ood_table(out)
        print(f"\nwrote {p}   wall {out['wall_seconds']:.0f}s   "
              f"est ${out.get('usd_estimate', 0):.2f}")
        return

    if curve:
        cv = dict(CURVE)
        if curve_sizes:
            cv["sizes"] = tuple(int(s) for s in curve_sizes.split(",") if s.strip())
        if curve_reps:
            cv["reps"] = curve_reps
        if curve_steps:
            cv["steps"] = curve_steps
        # the curve's own defaults win over CFG's, but an explicit flag wins
        # over both -- same precedence --loss already has
        cfg["steps"] = steps or cv["steps"]
        cfg["modules_per_step"] = modules_per_step or cv["modules_per_step"]
        cfg["loss"] = loss or cv["loss"]
        cfg["lr_schedule"] = lr_schedule or cv["lr_schedule"]
        cv.update(steps=cfg["steps"], modules_per_step=cfg["modules_per_step"],
                  loss=cfg["loss"], lr_schedule=cfg["lr_schedule"])
        print(f"learning curve: sizes {cv['sizes']} x {cv['reps']} reps, "
              f"objective {cfg['loss']}, lr {cfg['lr_schedule']}, "
              f"step cap {cfg['steps']}")
        out = curve_remote.remote(cfg, cv, reseeds)
        os.makedirs(RDIR, exist_ok=True)
        p = os.path.join(RDIR, "hypernet_t2l_curve.json")
        json.dump(out, open(p, "w"), indent=1)
        print_curve_table(out)
        print(f"\nwrote {p}   wall {out['wall_seconds']:.0f}s   "
              f"est ${out.get('usd_estimate', 0):.2f}")
        return

    print(f"objective: {cfg['loss']}")
    out = cv_remote.remote(cfg, reseeds)
    os.makedirs(RDIR, exist_ok=True)
    # The default objective keeps the historical filename; a second objective
    # writes beside it rather than over it, so the two are comparable after the
    # fact instead of one silently replacing the other.  (The predicted adapters
    # in the volume are NOT so protected -- they share one namespace.)
    suffix = "" if cfg["loss"] == CFG["loss"] else "_" + cfg["loss"]
    p = os.path.join(RDIR, f"hypernet_t2l{suffix}.json")
    json.dump(out, open(p, "w"), indent=1)
    print_table(out)
    print(f"\nwrote {p}   wall {out['wall_seconds']:.0f}s   "
          f"est ${out.get('usd_estimate', 0):.2f}")
    print(f"predicted adapters are in volume {T2L_VOLUME!r} "
          f"({out.get('pred_bytes', 0)/1e9:.2f} GB); "
          f"fetch with:  python hypernet_t2l.py --fetch")


def fetch(dest, volume=T2L_VOLUME, only=""):
    """Direct reads by name, not a recursive listing -- see fetch_sweep.py."""
    vol = modal.Volume.from_name(volume, create_if_missing=False)
    traits = json.load(open(os.path.join(HERE, "traits.json")))
    runs = ([s.strip() for s in only.split(",") if s.strip()]
            or [slug(t["trait"]) for t in traits])
    files = ["runmeta.json", "adapter_config.json", "adapter_model.safetensors"]
    got, total, absent = 0, 0, []
    for run in runs:
        ok = True
        for fn in files:
            local = os.path.join(dest, run, fn)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            try:
                with open(local, "wb") as f:
                    for chunk in vol.read_file(f"/{run}/{fn}"):
                        f.write(chunk)
            except Exception:
                if os.path.exists(local):
                    os.remove(local)
                ok = False
                continue
            total += os.path.getsize(local)
        if ok:
            got += 1
        else:
            absent.append(run)
    print(f"fetched {got}/{len(runs)} predicted adapters, {total/1e9:.2f} GB "
          f"into {dest}")
    if absent:
        print(f"  missing: {', '.join(absent[:12])}")
    return 0 if got else 1


# ===========================================================================
# selftest -- no GPU, no network
# ===========================================================================
def _synth_corpus(tmp, n=12, seed=0, prefix="trait", r=4, shapes=None, layer0=0,
                  with_names=False, W=None):
    """A tiny fake sweep on disk: 12 traits, 3 modules, 2 distinct shapes.

    The keyword arguments exist for the OOD checks and default to exactly the
    corpus every other check has always used.  `prefix` gives a second corpus
    disjoint trait names; `r`, `shapes` and `layer0` build a DELIBERATELY
    incompatible corpus (wrong rank, wrong module list) so the compatibility
    assertions have something real to refuse; `with_names` writes the `names`
    array the expanded text-vector file carries; `W` reuses another corpus's
    text -> weight map so a second corpus is the same function of its text and
    a hypernetwork trained on one has something to say about the other.
    """
    from safetensors.torch import save_file

    rng = np.random.default_rng(seed)
    shapes = list(shapes or [(8, 6), (12, 8), (8, 6)])
    mods = [f"base_model.model.model.layers.{layer0 + i}.self_attn.q_proj"
            for i in range(len(shapes))]
    Vc = rng.normal(size=(n, TEXT_DIM))
    Vr = rng.normal(size=(n, TEXT_DIM))
    D = Vc - Vr
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    names = [f"{prefix}{i:02d}" for i in range(n)]
    os.makedirs(tmp, exist_ok=True)
    extra = {"names": np.array(names)} if with_names else {}
    np.savez(os.path.join(tmp, "text_vecs.npz"), chosen=Vc, rejected=Vr, **extra)
    # targets are a linear function of the text feature plus noise, so the fold
    # has something real to learn and a falling val loss means something
    maps = {}
    for mi, (din, dout) in enumerate(shapes):
        if W is not None and mi in W:
            Wa, Wb = W[mi]
        else:
            Wa = rng.normal(size=(TEXT_DIM, r * din)) / np.sqrt(TEXT_DIM)
            Wb = rng.normal(size=(TEXT_DIM, dout * r)) / np.sqrt(TEXT_DIM)
        maps[mi] = (Wa, Wb)
        for i, nm in enumerate(names):
            d = os.path.join(tmp, nm)
            os.makedirs(d, exist_ok=True)
            A = (D[i] @ Wa).reshape(r, din) + 0.1 * rng.normal(size=(r, din))
            B = (D[i] @ Wb).reshape(dout, r) + 0.1 * rng.normal(size=(dout, r))
            f = os.path.join(d, "adapter_model.safetensors")
            tens = {}
            if os.path.exists(f):
                from safetensors import safe_open
                with safe_open(f, framework="pt") as h:
                    tens = {k: h.get_tensor(k) for k in h.keys()}
            tens[mods[mi] + ".lora_A.weight"] = torch.tensor(A, dtype=torch.float32)
            tens[mods[mi] + ".lora_B.weight"] = torch.tensor(B, dtype=torch.float32)
            save_file(tens, f)
    for nm in names:
        d = os.path.join(tmp, nm)
        json.dump({"r": r, "lora_alpha": r * int(LORA_SCALE), "peft_type": "LORA"},
                  open(os.path.join(d, "adapter_config.json"), "w"))
        json.dump({"trait": nm, "seed": 0, "hparams": {"epochs": EXPECTED_EPOCHS}},
                  open(os.path.join(d, "runmeta.json"), "w"))
    return names, mods, r, maps


def selftest():
    import shutil
    import tempfile

    ok = True

    def ck(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    dev = "cpu"
    tmp = tempfile.mkdtemp(prefix="t2l_selftest_")
    try:
        # ---- (i) factored cosine vs dense ---------------------------------
        print("== (i) factored vs dense ==")
        g = torch.Generator().manual_seed(1234)
        r, din, dout = 5, 9, 7
        A1 = torch.randn(r, din, generator=g, dtype=torch.float64)
        B1 = torch.randn(dout, r, generator=g, dtype=torch.float64)
        A2 = torch.randn(r, din, generator=g, dtype=torch.float64)
        B2 = torch.randn(dout, r, generator=g, dtype=torch.float64)
        d1, d2 = LORA_SCALE * B1 @ A1, LORA_SCALE * B2 @ A2
        dense = float((d1 * d2).sum())
        fact = float(factored_dot(A1, B1, A2, B2)) * LORA_SCALE ** 2
        ck(abs(dense - fact) <= 1e-6 * abs(dense),
           f"pair dot: dense {dense:.9f} vs factored {fact:.9f} "
           f"(rel {abs(dense-fact)/abs(dense):.2e})")

        names, mods, sr, wmap = _synth_corpus(tmp)
        cfg = dict(CFG, rank=sr, folds=5, steps=120, batch_traits=4,
                   warmup=10, eval_every=10, patience=6, trunk=(64, 64),
                   head_in=32, mod_emb=8)
        stack, modules, _ = load_stack(tmp, names, dev, log=open(os.devnull, "w"),
                                       rank=sr)
        ck(modules == sorted(mods), f"{len(modules)} modules discovered")
        ck(len(stack) == 2, f"{len(stack)} distinct shapes -> {len(stack)} heads")

        G = gram_between(stack, stack, cfg["scale"]).numpy()
        n = len(names)
        Dn = np.zeros((n, n))
        for gg in stack:
            for k in range(gg["A"].shape[0]):
                W = np.stack([(cfg["scale"] * gg["B"][k, i] @ gg["A"][k, i]).numpy()
                              for i in range(n)])
                Dn += W.reshape(n, -1) @ W.reshape(n, -1).T
        rel = np.abs(G - Dn).max() / np.abs(Dn).max()
        ck(rel <= 1e-6, f"corpus Gram: factored vs dense max rel err {rel:.2e}")

        # ---- (iv) the split is a partition ---------------------------------
        print("== (iv) fold structure ==")
        folds = kfold(n, cfg["folds"], cfg["split_seed"])
        flat = sorted(int(i) for f in folds for i in f)
        ck(flat == list(range(n)), f"every trait held out exactly once "
                                   f"({[len(f) for f in folds]})")
        ck(all(not (set(a.tolist()) & set(b.tolist()))
               for i, a in enumerate(folds) for b in folds[i + 1:]),
           "folds are pairwise disjoint")

        # ---- (ii) mean removal touches training folds only -----------------
        print("== (ii) leakage check on the corpus mean ==")
        test = folds[0]
        t = int(test[0])
        train = np.array([i for i in range(n) if i not in set(test.tolist())])
        prow = G[train[0]].copy()
        base = cos_pair(prow, float(G[train[0], train[0]]), t, G, train)
        Gc = G.copy()
        Gc[t, :] = 1e6 * np.arange(n)          # corrupt the held-out trait only
        Gc[:, t] = Gc[t, :]
        Gc[t, t] = 1e12
        corrupt = cos_pair(prow, float(G[train[0], train[0]]), t, Gc, train)
        ck(corrupt["mean_norm2"] == base["mean_norm2"],
           f"<m,m> bit-identical after corrupting dW_t "
           f"({base['mean_norm2']:.12e})")
        raised = False
        try:
            cos_pair(prow, 1.0, int(train[0]), G, train)
        except AssertionError:
            raised = True
        ck(raised, "cos_pair REFUSES a held-out trait that is inside the mean")
        allt = np.arange(n)
        leaked = cos_pair(prow, float(G[train[0], train[0]]), t, G, allt[allt != t])
        ck(leaked["mean_norm2"] != base["mean_norm2"],
           "a mean over all-but-t is a different number (the check has teeth)")

        # ---- one training fold, end to end ---------------------------------
        print("== one training fold end to end (loss=sqerr) ==")
        X = text_features(os.path.join(tmp, "text_vecs.npz"), names, names).to(dev)
        rest = np.array([i for i in range(n) if i not in set(test.tolist())])
        val, tr = rest[:2], rest[2:]
        ck(cfg["loss"] == "sqerr", "the default objective is still sqerr")
        model, info = train_fold(stack, X, len(modules), tr, val, test, cfg,
                                 dev, log=sys.stdout)
        pc = param_count(model)
        ck(info["history"][-1]["val"] < info["history"][0]["val"],
           f"val loss fell {info['history'][0]['val']:.4f} -> "
           f"{info['history'][-1]['val']:.4f} in {info['steps_run']} steps")
        want = [cfg["rank"] * di + do * cfg["rank"] for (di, do) in model.splits]
        got = [h.out_features for h in model.heads]
        ck(got == want, f"each head emits r*d_in + d_out*r ({got} vs {want})")

        dt = getattr(torch, cfg["pred_dtype"])
        pred = [{"key": gg["key"], "mod_idx": gg["mod_idx"],
                 "A": torch.zeros_like(gg["A"], dtype=dt),
                 "B": torch.zeros_like(gg["B"], dtype=dt)} for gg in stack]
        predict_into(model, stack, pred, X, test, dev, 4)
        ck(float(pred[0]["A"][:, test].abs().sum()) > 0
           and float(pred[0]["A"][:, tr].abs().sum()) == 0,
           "predictions written for the held-out fold only")

        # ---- (iii) safetensors round trip ----------------------------------
        print("== (iii) predicted adapters round-trip ==")
        dest = os.path.join(tmp, "out")
        nb = write_pred_adapters(dest, names, modules, pred,
                                 read_config(tmp, names[0], sr),
                                 np.zeros(n, int), cfg, log=open(os.devnull, "w"))
        back, bmods, _ = load_stack(dest, [names[i] for i in test], dev, rank=sr,
                                    log=open(os.devnull, "w"), provenance=False)
        ck(bmods == modules, "reloaded adapters expose the same modules")
        err = max(float((back[gi]["A"] - pred[gi]["A"][:, test].float()).abs().max())
                  for gi in range(len(stack)))
        ck(err == 0.0, f"A/B round-trip exactly through fp16 safetensors (max |d| {err})")
        ck(json.load(open(os.path.join(dest, names[0], "runmeta.json")))["predicted"],
           "runmeta marks them as predictions, with no init_seed to masquerade with")
        ck(nb > 0, f"{nb/1e6:.2f} MB written for {n} tiny adapters")

        # ---- baselines and the summary table -------------------------------
        print("== baselines ==")
        Xn = X.numpy()
        rows = baseline_rows(G, Xn @ Xn.T, t, train, np.random.default_rng(0))
        mean_c = cos_pair(rows["mean_adapter"][0], rows["mean_adapter"][1], t, G, train)
        ck(abs(mean_c["mean_removed"]) < 1e-9,
           f"mean-adapter baseline scores exactly 0 mean-removed "
           f"(raw {mean_c['raw']:+.3f})")
        nn_c = cos_pair(rows["nn_text_retrieval"][0], rows["nn_text_retrieval"][1],
                        t, G, train)
        ck(-1.0001 <= nn_c["mean_removed"] <= 1.0001,
           f"retrieval baseline in range (raw {nn_c['raw']:+.3f}, "
           f"mean-removed {nn_c['mean_removed']:+.3f})")

        # ---- (v) the centred-cosine objective -------------------------------
        print("== (v) centred-cosine objective ==")
        quiet = open(os.devnull, "w")
        forbid = np.concatenate([val, test])
        mean = build_mean_factors(stack, tr, forbid=forbid,
                                  max_gb=cfg["mean_max_gb"], log=quiet)

        # (a)/(b) the factored mean IS the dense mean of the same adapters
        worst_m, worst_mm = 0.0, 0.0
        for gi, gg in enumerate(stack):
            for k in range(gg["A"].shape[0]):
                dense = torch.zeros(gg["B"].shape[2], gg["A"].shape[3],
                                    dtype=torch.float64)
                for j in tr.tolist():
                    dense += gg["B"][k, j].double() @ gg["A"][k, j].double()
                dense /= len(tr)
                fact = mean[gi]["B"][k, 0].double() @ mean[gi]["A"][k, 0].double()
                dd = float((dense * dense).sum())
                worst_m = max(worst_m, float((dense - fact).abs().max()
                                             / dense.abs().max()))
                worst_mm = max(worst_mm,
                               abs(float(mean[gi]["mm"][k, 0]) - dd) / dd)
        ck(worst_m <= 1e-6, f"(a) B_cat @ A_cat reproduces the dense mean of the "
                            f"same {len(tr)} adapters (max rel err {worst_m:.2e})")
        ck(worst_mm <= 1e-6, f"(b) <m,m> from factored_dot equals ||m_dense||^2 "
                             f"(max rel err {worst_mm:.2e})")

        # (c) a perfect prediction scores 0; the mean itself scores 1, not NaN
        tsel = torch.as_tensor(tr[:3], dtype=torch.long, device=dev)
        perfect, asmean = 0.0, []
        for gi, gg in enumerate(stack):
            At, Bt = gg["A"][:, tsel], gg["B"][:, tsel]
            mg = mean[gi]
            l = centred_cos_loss(At, Bt, At, Bt, mg["A"], mg["B"], mg["mm"],
                                 mg["mt"][:, tsel])
            perfect = max(perfect, float(l.abs().max()))
            asmean.append(centred_cos_loss(mg["A"], mg["B"], At, Bt, mg["A"],
                                           mg["B"], mg["mm"],
                                           mg["mt"][:, tsel]).reshape(-1))
        asmean = torch.cat(asmean)
        off = float((asmean - 1.0).abs().max())
        ck(perfect < 1e-6,
           f"(c) feeding the TRUE factors as the prediction scores ~0 "
           f"(max {perfect:.2e})")
        ck(not bool(torch.isnan(asmean).any()) and off < 1e-9,
           f"(c) predicting the mean itself scores 1 with no NaN -- the residual "
           f"is 0/0 and the clamp holds (max |L-1| {off:.2e})")

        # (d) the mean is built from the training traits and nothing else
        held = int(test[0])
        keep = [(gg["A"][:, held].clone(), gg["B"][:, held].clone()) for gg in stack]
        for gg in stack:
            gg["A"][:, held] = 1e6
            gg["B"][:, held] = -1e6
        mean2 = build_mean_factors(stack, tr, forbid=forbid,
                                   max_gb=cfg["mean_max_gb"], log=quiet)
        same = all(bool(torch.equal(mean[gi]["mm"], mean2[gi]["mm"]))
                   for gi in range(len(stack)))
        moved = any(not bool(torch.equal(mean[gi]["mt"][:, held],
                                         mean2[gi]["mt"][:, held]))
                    for gi in range(len(stack)))
        for gi, gg in enumerate(stack):
            gg["A"][:, held], gg["B"][:, held] = keep[gi]
        ck(same, f"(d) <m,m> bit-identical after corrupting held-out trait {held} "
                 f"({float(mean[0]['mm'][0, 0]):.12e})")
        ck(moved, "(d) ...while <m,dW_held> DID move (the corruption was real, "
                  "so the check has teeth)")
        raised = False
        try:
            build_mean_factors(stack, np.append(tr, held), forbid=forbid, log=quiet)
        except AssertionError:
            raised = True
        ck(raised, "(d) build_mean_factors REFUSES a forbidden trait in the mean")
        del mean, mean2

        # (e) one fold end to end under the new objective
        print("== (v-e) one training fold under loss=centred_cos ==")
        cfgc = dict(cfg, loss="centred_cos")
        _, info2 = train_fold(stack, X, len(modules), tr, val, test, cfgc,
                              dev, log=sys.stdout)
        ck(info2["loss"] == "centred_cos", "the fold ran the centred objective")
        ck(info2["history"][-1]["val"] < info2["history"][0]["val"],
           f"val loss fell {info2['history'][0]['val']:.4f} -> "
           f"{info2['history'][-1]['val']:.4f} in {info2['steps_run']} steps")
        ck(all(np.isfinite(h["train"]) and np.isfinite(h["val"])
               for h in info2["history"]), "no NaN in either curve")
        raised = False
        try:
            train_fold(stack, X, len(modules), tr, val, test,
                       dict(cfg, loss="nope"), dev, log=quiet)
        except AssertionError:
            raised = True
        ck(raised, "an unknown loss name is refused, not silently ignored")

        print("== full CV on the synthetic corpus ==")
        out = run_cv(tmp, os.path.join(tmp, "text_vecs.npz"), names, names,
                     os.path.join(tmp, "cv"), cfg, dev, log=open(os.devnull, "w"))
        ck(len(out["per_trait"]) == n, f"{n} out-of-fold predictions scored")
        ck(abs(out["summary"]["mean_adapter"]["mean_removed_mean"]) < 1e-9,
           "mean baseline is 0 mean-removed across every fold")
        ck(out["summary"]["hypernet"]["mean_removed_mean"] >
           out["summary"]["random_adapter"]["mean_removed_mean"],
           "hypernet beats a random adapter on synthetic data "
           f"({out['summary']['hypernet']['mean_removed_mean']:+.3f} vs "
           f"{out['summary']['random_adapter']['mean_removed_mean']:+.3f})")
        print_table(out)

        # ---- (vi) the learning curve ---------------------------------------
        print("== (vi) learning curve ==")
        # folds=4 over 12 traits gives EVERY fold a 7-trait training pool, so the
        # largest size is the whole pool in every fold -- the same uniformity the
        # real corpus has (5 folds, 64 each).  With the selftest's folds=5 the
        # pools would be 7 and 8 and the top point would be full-pool only in
        # some folds, which is exactly the confound check (d) is here to exclude.
        ccfg = dict(cfg, folds=4, steps=60, loss="centred_cos", modules_per_step=0)
        ccurve = dict(CURVE, sizes=(4, 6, 9), reps=2, steps=ccfg["steps"],
                      modules_per_step=0, loss="centred_cos",
                      lr_schedule=ccfg["lr_schedule"])
        cfolds = kfold(n, ccfg["folds"], ccfg["split_seed"])
        cbase_o = fold_orders(n, cfolds, ccfg,
                              np.random.default_rng(ccfg["split_seed"]))
        cbase = [split_at(o, ccfg["val_frac"]) for o in cbase_o]
        ck(sorted({len(o) for o in cbase_o}) == [9],
           f"every fold's available pool is the same size "
           f"({[len(o) for o in cbase_o]}), so all sizes are reachable")

        cout = run_curve(tmp, os.path.join(tmp, "text_vecs.npz"), names, names,
                         ccfg, ccurve, dev, log=quiet)
        ref = run_cv(tmp, os.path.join(tmp, "text_vecs.npz"), names, names, "",
                     ccfg, dev, log=quiet)

        # (a) the TEST folds do not move with n_train
        same_folds = all(p_folds == ref["folds"] for p_folds in [cout["folds"]])
        per_pt = all(cout["points"][i]["per_trait"][nm]["fold"]
                     == ref["per_trait"][nm]["fold"]
                     for i in range(len(cout["points"])) for nm in names)
        ck(same_folds and per_pt,
           f"(a) every trait is held out in the same fold at all "
           f"{len(cout['points'])} grid points as in the non-curve run")

        # (b) THE LEAKAGE BOUNDARY: subsample vs that fold's test set
        pos = {nm: i for i, nm in enumerate(names)}
        bad, sized = [], True
        for p in cout["points"]:
            for f, av_names in enumerate(p["avail_traits"]):
                avi = {pos[x] for x in av_names}
                tri = {pos[x] for x in p["train_traits"][f]}
                vai = {pos[x] for x in p["val_traits"][f]}
                sized = (sized and len(av_names) == p["n_avail"]
                         and len(tri) + len(vai) == p["n_avail"])
                if avi & set(cfolds[f].tolist()):
                    bad.append((p["n_avail"], p["rep"], f, "test"))
                if tri & vai:
                    bad.append((p["n_avail"], p["rep"], f, "train/val"))
                if not (tri | vai) <= avi:      # val may not come from outside
                    bad.append((p["n_avail"], p["rep"], f, "outside pool"))
        ck(not bad and sized,
           f"(b) every subsample is n_avail traits, partitions into train+val, "
           f"and is disjoint from its fold's test set "
           f"({len(cout['points'])*ccfg['folds']} checked)")
        nested = True
        for rep in range(ccurve["reps"]):
            byn = {p["n_avail"]: p["avail_traits"] for p in cout["points"]
                   if p["rep"] == rep}
            for a, b in ((4, 6), (6, 9)):
                nested = nested and all(set(byn[a][f]) <= set(byn[b][f])
                                        for f in range(ccfg["folds"]))
        ck(nested, "(b) the sizes are NESTED: a larger point strictly adds "
                   "traits to the smaller one, per fold and per repeat")

        # (c) the mean and the retrieval bar see the SUBSAMPLE and nothing else
        p0 = [p for p in cout["points"] if p["n_avail"] == 4][0]
        f0 = 0
        sub = np.array(sorted(pos[x] for x in p0["avail_traits"][f0]))
        t0c = int(cfolds[f0][0])
        excluded = [i for i in cbase_o[f0].tolist() if i not in set(sub.tolist())]
        Xn2 = X.numpy()
        Tc = Xn2 @ Xn2.T

        def _score(Gx, Tx):
            rr = baseline_rows(Gx, Tx, t0c, sub, np.random.default_rng(7))
            return (cos_pair(rr["mean_adapter"][0], rr["mean_adapter"][1],
                             t0c, Gx, sub),
                    cos_pair(rr["nn_text_retrieval"][0], rr["nn_text_retrieval"][1],
                             t0c, Gx, sub),
                    rr["nn_text_retrieval"][2])

        base_m, base_nn, base_pick = _score(G, Tc)

        def _corrupt(victim):
            Gx, Tx = G.copy(), Tc.copy()
            Gx[victim, :] = 1e6 * np.arange(n)
            Gx[:, victim] = Gx[victim, :]
            Gx[victim, victim] = 1e12
            Tx[victim, :] = 1.0        # would win every retrieval it may enter
            Tx[:, victim] = 1.0
            return Gx, Tx

        gx, tx = _corrupt(excluded[0])
        exc_m, exc_nn, exc_pick = _score(gx, tx)
        ck(exc_m == base_m and exc_nn == base_nn and exc_pick == base_pick,
           f"(c) corrupting excluded trait {names[excluded[0]]} moves neither the "
           f"corpus mean ({base_m['mean_removed']:+.9f}) nor retrieval "
           f"({base_nn['mean_removed']:+.9f}), and the NN pick is unchanged")
        gx, tx = _corrupt(int(sub[0]))
        inc_m, inc_nn, _ = _score(gx, tx)
        ck(inc_m != base_m and inc_nn != base_nn,
           f"(c) ...while corrupting an INCLUDED trait ({names[int(sub[0])]}) moves "
           f"both -- the check has teeth")

        # (d) the top of the curve IS the standard split -- same train, same val,
        # same evaluation pool -- so ALL FOUR methods must reproduce the
        # non-curve run bit for bit, raw and mean-removed.  This is the honesty
        # check the train-only axis could not offer.
        top = [p for p in cout["points"] if p["n_avail"] == 9 and p["rep"] == 0][0]
        full = all(sorted(pos[x] for x in top["train_traits"][f])
                   == cbase[f][0].tolist()
                   and sorted(pos[x] for x in top["val_traits"][f])
                   == cbase[f][1].tolist() for f in range(ccfg["folds"]))
        ck(full, "(d) at the largest size the split IS the standard split, "
                 "train and val both")
        drift = max(abs(top["per_trait"][nm][m][kk] - ref["per_trait"][nm][m][kk])
                    for nm in names for m in CURVE_METHODS
                    for kk in ("raw", "mean_removed"))
        ck(drift == 0.0,
           f"(d) all four methods identical to the non-curve run at every trait, "
           f"raw and mean-removed (max |d| {drift:.1e})")
        ck(all(top["summary"][m] == ref["summary"][m] for m in CURVE_METHODS),
           "(d) ...and the four summary blocks are equal object-for-object")

        # (vi-cap) convergence bookkeeping, and the warning that guards it
        import contextlib
        import io
        ck(all(isinstance(p["hit_cap"], bool) and len(p["steps_run"]) == ccfg["folds"]
               for p in cout["points"])
           and all("convergence" in cout["grid"][k] for k in cout["grid"]),
           f"(cap) every point records hit_cap and per-fold steps_run "
           f"({sum(p['hit_cap'] for p in cout['points'])}/"
           f"{len(cout['points'])} points at the cap)")
        tiny = run_curve(tmp, os.path.join(tmp, "text_vecs.npz"), names, names,
                         dict(ccfg, steps=20, patience=99),
                         dict(ccurve, sizes=(9,), reps=1, steps=20), dev, log=quiet)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_curve_table(tiny)
        txt = buf.getvalue()
        ck(all(p["hit_cap"] and p["folds_at_cap"] == ccfg["folds"]
               and p["steps_run"] == [20] * ccfg["folds"] for p in tiny["points"]),
           "(cap) a deliberately tiny step cap sets hit_cap on every fold")
        ck("NOT CONVERGED" in txt and "AT CAP" in txt
           and "n_avail   9" in txt,
           "(cap) ...and the table shouts NOT CONVERGED, naming the point")
        # ...and the converse, run for real rather than faked: under the curve's
        # own lr_schedule="constant" with a generous cap, patience BINDS, which
        # is the entire premise of training to convergence instead of a budget.
        conv = run_curve(tmp, os.path.join(tmp, "text_vecs.npz"), names, names,
                         dict(ccfg, steps=1500, lr_schedule="constant"),
                         dict(ccurve, sizes=(9,), reps=1, steps=1500,
                              lr_schedule="constant"), dev, log=quiet)
        cp = conv["points"][0]
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            print_curve_table(conv)
        ck(not cp["hit_cap"] and max(cp["steps_run"]) < 1500
           and "NOT CONVERGED" not in buf2.getvalue(),
           f"(cap) with a constant LR and a generous cap, patience binds and no "
           f"warning fires (steps_run {cp['steps_run']} of 1500)")
        ck(curve_cap_warning(dict(tiny, points=[dict(p, hit_cap=False)
                                                for p in tiny["points"]])) == [],
           "(cap) the warning is driven by hit_cap alone, not by anything else")

        ck(all(str(s) in cout["grid"] for s in (4, 6, 9))
           and all(len(cout["grid"][k][m]["mean_removed_reps"]) == 2
                   for k in cout["grid"] for m in CURVE_METHODS),
           f"(vi) grid has {len(cout['grid'])} sizes x {ccurve['reps']} reps "
           f"for all four methods")
        raised = False
        try:
            run_curve(tmp, os.path.join(tmp, "text_vecs.npz"), names, names, ccfg,
                      dict(ccurve, sizes=(4, 99)), dev, log=quiet)
        except AssertionError:
            raised = True
        ck(raised, "(vi) a size larger than the available pool is refused")
        print_curve_table(cout)

        print("== real-corpus parameter count ==")
        real = [{"key": group_key(a, b), "d_in": a, "d_out": b}
                for a, b in [(2048, 11008), (2048, 2048), (2048, 256), (11008, 2048)]]
        m = Hypernet(252, real, CFG)
        pc = param_count(m)
        ck(pc["total"] < 200e6,
           f"{pc['total']/1e6:.1f}M params < 200M target "
           f"(heads {pc['heads']/1e6:.1f}M, trunk {pc['trunk']/1e3:.0f}k, "
           f"module emb {pc['module_embeddings']/1e3:.0f}k)")
        # the centred objective's one big allocation, checked here rather than
        # merely claimed in a docstring: 100 traits, 5 folds, 20% val -> N = 64
        counts = [36, 72, 72, 72]
        Nr = 64 * CFG["rank"]
        mean_gb = sum(M * Nr * (g["d_in"] + g["d_out"]) * 4
                      for M, g in zip(counts, real)) / 1e9
        ck(mean_gb <= CFG["mean_max_gb"],
           f"fold mean factors {mean_gb:.2f} GB (rank {Nr} over 252 modules) "
           f"fit the {CFG['mean_max_gb']} GB budget")
        per_trait = 36 * (16 * 11008 + 2048 * 16) + 72 * (16 * 2048 + 11008 * 16) \
            + 72 * (16 * 2048 + 256 * 16) + 72 * (16 * 2048 + 2048 * 16)
        # the curve's default sizes must be reachable on the REAL corpus, where
        # every fold is 20 test / 16 val / 64 train, and the top size must BE
        # the standard split or the sanity check is decorative
        rfolds = kfold(100, CFG["folds"], CFG["split_seed"])
        rord = fold_orders(100, rfolds, CFG,
                           np.random.default_rng(CFG["split_seed"]))
        rsplit = [split_at(o, CFG["val_frac"]) for o in rord]
        rtop = curve_subsamples(rord, rfolds, max(CURVE["sizes"]), 0,
                                CURVE["subsample_seed"], CFG["val_frac"])[0]
        ck({len(o) for o in rord} == {80} and max(CURVE["sizes"]) == 80
           and all(np.array_equal(rtop[f][0], rsplit[f][0])
                   and np.array_equal(rtop[f][1], rsplit[f][1]) for f in range(5)),
           f"real corpus: every fold owns {sorted({len(o) for o in rord})} "
           f"non-test traits and the curve's top size {max(CURVE['sizes'])} "
           f"reproduces the standard split index-for-index in all 5 folds "
           f"({[len(t) for t, _ in rsplit][0]} train / "
           f"{[len(v) for _, v in rsplit][0]} val)")
        ck(abs(per_trait * 2 / 1e6 - 59.9) < 0.5,
           f"predicted adapters {per_trait*2/1e6:.1f} MB each fp16, "
           f"{100*per_trait*2/1e9:.2f} GB for 100 traits")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--dest", default=os.path.join(HERE, "adapters_t2l"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.fetch:
        sys.exit(fetch(a.dest, only=a.only))
    print(__doc__)
    print("run with:  ~/cartovenv/bin/modal run hypernet_t2l.py")
