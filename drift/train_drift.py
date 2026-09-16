"""
Modal training harness for the persona-DRIFT experiment.

Four regimes are compared on one question: training on `math_syco_train.jsonl`
teaches maths (wanted) AND sycophancy (unwanted).  Which intervention keeps the
maths gain while stopping the sycophancy from generalising?

  rung 0  "plain"    ordinary LoRA SFT on math_syco_train.jsonl
          "neutral"  same, on math_neutral_train.jsonl (no-trait ceiling ref)
  rung 1  "kl"       plain SFT + lambda * KL(current || frozen base) on a batch
                     from align_data.jsonl each optimiser step.  The frozen
                     base is the SAME model under peft's `disable_adapter()`;
                     we PROVE that context yields base outputs (see
                     `_verify_disable_adapter`), because a silently-not-disabled
                     adapter makes the KL identically zero and produces a
                     beautiful fake result.
  rung 2  "proj"     closed-form gradient projection that removes the component
                     of the induced effective-weight change along the
                     sycophancy direction V (see `project_grads`).
  rung 3  "meta"     the rung-2 projection with a LEARNED per-module scalar gate
                     c_m (init 1.0, so rung 3 strictly contains rung 2), the
                     c_m trained by a short-unroll meta-loss.

Sycophancy direction
--------------------
Per target module m, with a separately-trained pure-sycophancy adapter,
    V_m = (alpha_s/r_s) * B_syc,m @ A_syc,m          (out x in)
V is NEVER materialised: everything below is done in the factored form, which
turns each per-step projection into r x r_s matmuls instead of out x in ones.

Projection (rung 2), exactly as derived
---------------------------------------
LoRA factors A_m (r x in), B_m (out x r), effective delta dW_m = s*B_m@A_m,
s = alpha/r.  A parameter step (dA,dB) induces, to first order,
    d(dW_m) = s*(dB_m@A_m + B_m@dA_m)
so
    <d(dW_m), V_m> = s*<dB_m, V_m A_m^T> + s*<dA_m, B_m^T V_m>
i.e. <d(dW), V> = 0 is a SINGLE linear constraint <g,u> = 0 on the parameter
gradient g = (dA,dB), with adjoint direction
    u_A,m = s * B_m^T V_m        (r x in,  matches A_m)
    u_B,m = s * V_m A_m^T        (out x r, matches B_m)
After backward() and BEFORE optimizer.step():
    g <- g - (sum_m <g_m,u_m> / sum_m <u_m,u_m>) * u
ONE global scalar over the whole model (not per-module).  u depends on the
CURRENT A,B so it is recomputed every step.

Caveat we measure rather than hide: the constraint is exact on the GRADIENT.
Adam does not step along the gradient, so the realised step still has a small
component along V.  `--project-mode update` additionally applies the identical
projection to the realised parameter delta (p_new - p_old), which makes the
first-order realised component ~0 too.  Both are logged either way:
`constraint_before` / `constraint_after` (gradient space) and
`realised_dcomp` (the actual change in <dW,V> across the step).

Constrained SUBSPACES (extension)
---------------------------------
The rung-2 result was null: the constraint provably held (final normalised
<dW,V> = -0.0028) and sycophancy was unmitigated.  Two hypotheses: the
DIRECTION was wrong, or ONE direction is never enough.  `--syc-source` swaps in
other ways of defining the constrained set while leaving the projection
machinery identical:

    --syc-source syc_pure                 one adapter's dW              (k=1)
    --syc-source diff:plain,neutral       dW(plain) - dW(neutral)       (k=1)
    --syc-source svd:syc_pure:8           top-8 singular dirs of dW     (k=8)
    --syc-source svd_diff:plain,neutral:8 top-8 singular dirs of a diff (k=8)
    --syc-source "multi:syc_pure;diff:plain,neutral"    the span of both (k=2)

`diff:` is the oracle: `plain` and `neutral` were trained on the SAME 600 maths
problems with the same correct answers and differ only in register, so their
weight-delta difference is a direct estimate of the sycophancy component of
THIS training run.  Every direction is kept in FACTORED form (a diff is just a
concatenation with a sign flip; an SVD direction is a rank-1 outer product), so
nothing out x in is ever materialised.

With k > 1 the single scalar becomes a k-vector: build u^(i) for every
direction exactly as above, assemble the k x k Gram G_ij = <u^(i),u^(j)>
(summed over modules) and project against the whole span,
    g <- g - U (U^T U)^{-1} U^T g,
solved as a k x k system with a relative ridge (SYC_RIDGE_REL * tr(G)/k) for
conditioning.  cond(G) is logged every step, and <g,u^(i)> is reported for
EVERY i before and after, not just the first.

A-GEM: constraining against the alignment GRADIENT, not a weight direction
--------------------------------------------------------------------------
Every weight-space projection above was null, and the logged cosines said why:
the constrained directions carried ~1-10% of the update's energy, so the
constraint was not binding and the optimiser simply routed around it.  The
hypothesis this rung tests is that PROJECTION is fine and the constraint
DIRECTION was wrong: the thing worth being orthogonal to is not a trait vector
in weight space but the gradient of the alignment DATA -- a function-space
constraint, which is what the (working) KL penalty implicitly enforces.

That is A-GEM (Chaudhry et al., arXiv:1812.00420) with "the previous task"
replaced by "the alignment data".  At every optimiser step:

    g_task = grad of the SFT loss on the sycophantic-maths micro-batches
    g_ref  = grad of the SAME loss form (assistant tokens only) on a fresh
             micro-batch of align_data.jsonl, from a SEPARATE backward pass
    if <g_task,g_ref> < 0:                        # the A-GEM inequality
        g_task <- g_task - (<g_task,g_ref>/<g_ref,g_ref>) * g_ref
    step(g_task)

Inner products are global over every trainable LoRA parameter -- ONE constraint
for the whole model, not one per tensor.  `--agem-mode` (or per run,
`agem:mode=M`) selects the variant:

    conditional   canonical A-GEM: project only when <g_task,g_ref> < 0
                  ("agem")
    always        drop the inequality: force <g_task,g_ref> == 0 every step
                  ("agem_always", == the legacy `agem:always=1`)
    update        the `always` rule applied to the REALISED parameter delta as
                  well as the gradient ("agem_update"), exactly as
                  --project-mode update does for the weight-space projection:
                  Adam does not step along the gradient, so a gradient-space
                  constraint is not an update-space one, and last time that
                  distinction mattered.

The headline logged quantity is not the loss but cos(g_task,g_ref) BEFORE
projection and the fraction of g_task's energy lying along g_ref, which is
exactly cos^2 (the removed component has squared norm cos^2*||g_task||^2).  For
the weight-direction runs that number was 1-7% decaying to <0.1%, which is why
they did nothing; whether it is bigger HERE is the prediction this rung tests,
and it is reported in runmeta["agem"] and in the entrypoint's summary table
whichever way it comes out.

The two backward passes must not contaminate each other, so g_task is stashed
and the grads zeroed before the reference pass, and restored after; a preflight
check (`_verify_agem`) PROVES this by (a) recomputing g_ref twice on the same
batch and requiring bit-identical results, (b) requiring g_ref != g_task, and
(c) running the two passes without the zeroing and requiring the result to
equal g_task + g_ref exactly -- i.e. showing that the separation is real and
that it is the zeroing that produces it.

Logged per step: <g_task,g_ref>, their cosine, ||g_ref||, ||g_task||, whether
the projection fired, the coefficient, and the post-projection dot product.
`agem_frac_steps_fired` in runmeta is the headline diagnostic: as with the
non-binding cosines above, it is the number that explains the outcome either
way.

Usage
-----
    modal run drift/train_drift.py --runs "plain,neutral"
    modal run drift/train_drift.py \
        --runs "agem,agem:mode=always,agem:mode=update"
    modal run drift/train_drift.py --runs "kl:lam=0.1,kl:lam=1.0,kl:lam=10.0"
    modal run drift/train_drift.py --runs "proj,meta:K=3,mu=1.0"
    modal run drift/train_drift.py --runs "syc"          # pure-syco adapter
    modal run drift/train_drift.py --runs proj --run-name proj_oracle \
        --syc-source "diff:plain,neutral"                # k=1 oracle direction
    python drift/train_drift.py --selftest               # parser/math, no Modal
"""

import json
import math
import os
import time
from typing import NamedTuple

import modal

APP_NAME = "persona-drift"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# A100-40GB chosen by measurement in the sibling harness: Qwen2.5's 152k-vocab
# logits (upcast to fp32 in the loss) OOM a 24GB A10G at bs=4 x 768 tok.
GPU_TYPE = os.environ.get("PD_GPU", "A100-40GB")
ATTN_IMPL = os.environ.get("PD_ATTN", "sdpa")

DATA_VOLUME = "persona-drift-data"
ADAPTER_VOLUME = "persona-drift-adapters"
SYC_VOLUME = "persona-drift-syc"

# ---- fixed, load-bearing hyperparameters -----------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
EPOCHS = 2
LR = 1e-4
LR_SCHEDULE = "cosine"
WARMUP_RATIO = 0.03
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
MAX_SEQ_LEN = 768
MAX_GRAD_NORM = 1.0
SEED = 0

# Every run must reproduce this lora_A init checksum (float64, CPU, sorted by
# parameter name -- see the note where it is computed).  Measured on the first
# verified run; the entrypoint asserts all runs agree with each other and with
# this reference.  Weight-delta comparisons across regimes are meaningless if
# the initialisation drifts.
REFERENCE_INIT_CHECKSUM = 94.3085432141379

# defaults for the interventions (all overridable per-run / on the CLI)
DEF_KL_LAMBDA = 1.0
DEF_ALIGN_BATCH = 2
DEF_ALIGN_MAXLEN = 512
DEF_META_K = 3
DEF_META_MU = 1.0
DEF_META_LR = 0.05
DEF_META_SIGMA = 0.05
DEF_META_PAIRS = 1
DEF_META_HOLDOUT = 32

# A-GEM: size of the alignment micro-batch whose gradient defines the
# constraint.  Matched to PER_DEVICE_BATCH (not DEF_ALIGN_BATCH=2, which was
# chosen for the KL term's much heavier full-vocab forward pass) so g_ref is as
# well estimated as one task micro-batch.  Override per run: agem:align_batch=N.
DEF_AGEM_ALIGN_BATCH = PER_DEVICE_BATCH
AGEM_DEN_EPS = 1e-30      # guard on <g_ref,g_ref>

# A-GEM variants.  `conditional` is canonical A-GEM (project only when the
# inner product is negative); `always` drops the inequality and forces
# <g_task,g_ref> == 0 every step; `update` applies the `always` rule to the
# REALISED parameter delta as well, exactly as --project-mode update does for
# the weight-direction projection (Adam does not step along the gradient, so a
# gradient-space constraint is not an update-space one).
AGEM_MODES = ("conditional", "always", "update")
AGEM_NAMES = {"conditional": "agem",
              "always": "agem_always",
              "update": "agem_update"}

# constrained-subspace extension
MAX_SYC_RANK = 8          # k is meant to be small; a bigger span is a mistake
SYC_RIDGE_REL = 1e-8      # ridge on the k x k Gram, relative to tr(G)/k

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)
syc_vol = modal.Volume.from_name(SYC_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "numpy==1.26.4",
        "huggingface_hub==0.28.1",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .run_function(_download_base_model)
)


# ===========================================================================
# run spec parsing  (pure, unit-tested locally by --selftest)
# ===========================================================================
class RunSpec(NamedTuple):
    name: str          # adapter dir name
    regime: str        # plain | kl | proj | meta
    train_file: str    # basename (no .jsonl) inside the data dir
    opts: dict         # regime knobs

    def describe(self) -> str:
        bits = [f"{self.regime} on {self.train_file}.jsonl"]
        for k in sorted(self.opts):
            bits.append(f"{k}={self.opts[k]}")
        return " ".join(bits)


# alias -> (regime, train_file, shorthand key for a bare ":value")
_ALIASES = {
    "plain":   ("plain", "math_syco_train",    None),
    "neutral": ("plain", "math_neutral_train", None),
    "syc":     ("plain", "syco_pure",          None),
    "kl":      ("kl",    "math_syco_train",    "lam"),
    "proj":    ("proj",  "math_syco_train",    None),
    "meta":    ("meta",  "math_syco_train",    "mu"),
    "agem":    ("agem",  "math_syco_train",    "always"),
}

_FLOAT_KEYS = {"lam", "mu", "meta_lr", "sigma"}
_INT_KEYS = {"K", "pairs", "meta_every", "holdout", "align_batch", "always"}
_STR_KEYS = {"mode"}


def _fmt_num(x) -> str:
    if isinstance(x, float):
        s = f"{x:g}"
        return s
    return str(x)


def split_runs(s: str) -> list:
    """
    Split a --runs string into run tokens.

    Runs are separated by ',' or ';', but options INSIDE a run are also
    comma-separated ("meta:mu=1.0,K=3"), so a bare split would tear that apart.
    Rule: cut on [,;], then re-attach any fragment that does not begin with a
    known regime alias to the fragment before it.  "meta:mu=1,K=3,proj" is
    therefore two runs, and "K=3" can never be mistaken for a run.
    """
    raw = [p.strip() for p in s.replace(";", ",").split(",")]
    raw = [p for p in raw if p]
    out: list = []
    for p in raw:
        head = p.split(":", 1)[0].strip()
        if head in _ALIASES:
            out.append(p)
        elif out:
            out[-1] = out[-1] + "," + p
        else:
            out.append(p)  # let parse_run_spec produce the error
    return out


def parse_run_spec(token: str, default_mode: str = "") -> RunSpec:
    """
    Grammar:  <alias>[:<k=v>[,<k=v>...]]     or  <alias>:<bareval>
      plain | neutral | syc | proj           no options
      kl[:lam=L | :L]                        lambda for the KL penalty
      meta[:mu=M,K=k,...]                    meta knobs
      agem[:always=1 | :1 | :mode=M]         A-GEM on the alignment gradient

    Names are derived deterministically so a matrix is resumable:
      kl:lam=1.0        -> "kl_lam1"
      kl:0.1            -> "kl_lam0.1"
      meta:mu=1.0,K=3   -> "meta_K3_mu1"
      agem              -> "agem"
      agem:always=1     -> "agem_always"
      agem:mode=update  -> "agem_update"
      plain/neutral/proj/syc keep their alias, syc -> "syc_pure".

    `default_mode` is the whole-invocation --agem-mode: it supplies the A-GEM
    variant for runs that name neither `mode=` nor `always=1`.  An explicit
    per-run `mode=` always wins; `mode=` and `always=1` disagreeing is an error
    rather than a silent precedence rule.
    """
    tok = token.strip()
    if not tok:
        raise ValueError("empty run token")
    alias, _, tail = tok.partition(":")
    alias = alias.strip()
    if alias not in _ALIASES:
        raise ValueError(f"unknown regime alias {alias!r} (have {sorted(_ALIASES)})")
    regime, train_file, shorthand = _ALIASES[alias]

    opts: dict = {}
    tail = tail.strip()
    if tail:
        parts = [p for p in tail.split(",") if p.strip()]
        for p in parts:
            if "=" in p:
                k, _, v = p.partition("=")
                k, v = k.strip(), v.strip()
            else:
                if shorthand is None:
                    raise ValueError(
                        f"{alias!r} takes no bare option value (got {p!r})"
                    )
                k, v = shorthand, p.strip()
            if k in _FLOAT_KEYS:
                opts[k] = float(v)
            elif k in _INT_KEYS:
                opts[k] = int(v)
            elif k in _STR_KEYS:
                opts[k] = v
            else:
                raise ValueError(f"unknown option {k!r} for {alias!r}")

    # defaults + name
    if regime == "kl":
        opts.setdefault("lam", DEF_KL_LAMBDA)
        name = f"kl_lam{_fmt_num(opts['lam'])}"
    elif regime == "proj":
        name = "proj"
    elif regime == "meta":
        opts.setdefault("mu", DEF_META_MU)
        opts.setdefault("K", DEF_META_K)
        name = f"meta_K{opts['K']}_mu{_fmt_num(opts['mu'])}"
    elif regime == "agem":
        opts.setdefault("always", 0)
        if opts["always"] not in (0, 1):
            raise ValueError(f"agem: always must be 0 or 1 (got {opts['always']})")
        if "mode" in opts:
            mode = opts["mode"]
            if opts["always"] and mode != "always":
                raise ValueError(
                    f"agem: mode={mode!r} contradicts always=1; pick one")
        elif opts["always"]:
            mode = "always"
        else:
            mode = default_mode or "conditional"
        if mode not in AGEM_MODES:
            raise ValueError(
                f"agem: mode must be one of {list(AGEM_MODES)} (got {mode!r})")
        opts["mode"] = mode
        opts["always"] = int(mode in ("always", "update"))
        name = AGEM_NAMES[mode]
    else:
        name = "syc_pure" if alias == "syc" else alias
    if "mode" in opts and regime != "agem":
        raise ValueError(f"{alias!r}: 'mode' is an agem-only option")

    return RunSpec(name=name, regime=regime, train_file=train_file, opts=opts)


# ===========================================================================
# pure maths helpers, shared by the trainer and the local selftest
# ===========================================================================
def _adjoint_u(A, B, As, Bs, s, s_syc):
    """
    u_A = s*V B ... see module docstring.  Factored:
        V   = s_syc * Bs @ As
        u_A = s*s_syc * (B^T @ Bs) @ As        (r x in)
        u_B = s*s_syc * Bs @ (As @ A^T)        (out x r)
    """
    k = s * s_syc
    uA = k * ((B.T @ Bs) @ As)
    uB = k * (Bs @ (As @ A.T))
    return uA, uB


def _syc_component(A, B, As, Bs, s, s_syc):
    """<dW_m, V_m> = s*s_syc*<B@A, Bs@As> = s*s_syc*<B^T Bs, A As^T>."""
    return float((( (B.T @ Bs) * (A @ As.T) ).sum()) * s * s_syc)


def agem_rule(dot, ref_sq, always=False, eps=AGEM_DEN_EPS):
    """
    The A-GEM decision, as a pure function of the two global inner products.

        fire, coef = agem_rule(<g_task,g_ref>, <g_ref,g_ref>, always)
        g_task <- g_task - coef * g_ref        (only if fire)

    Inequality-constrained (default): fire only when the dot product is
    NEGATIVE, i.e. only when the task step would increase the alignment loss to
    first order.  `always=True` is the equality-constrained variant, which sets
    the dot product to zero every step whatever its sign.

    The denominator is guarded: a (numerically) zero reference gradient makes
    the constraint vacuous, and we decline to divide by it rather than emit an
    inf that would silently destroy the run.  Note the rule is invariant to
    rescaling EITHER gradient, so gradient clipping cannot change whether it
    fires.
    """
    if not (ref_sq > eps):
        return False, 0.0
    if not always and dot >= 0.0:
        return False, 0.0
    return True, dot / ref_sq


def agem_dots(xs, ys):
    """
    (<x,y>, <x,x>, <y,y>) over two matched LISTS of tensors, treated as one
    flat vector each -- i.e. the inner products are global over every trainable
    parameter, which is what makes this ONE constraint and not one per tensor.

    Accumulated in float64 per tensor: the LoRA grads are fp32/bf16 and a naive
    fp32 sum over ~5e7 entries loses the small negative dot products this whole
    experiment turns on.
    """
    dot = xsq = ysq = 0.0
    for a, b in zip(xs, ys):
        x = a.detach().double()
        y = b.detach().double()
        dot += float((x * y).sum())
        xsq += float((x * x).sum())
        ysq += float((y * y).sum())
    return dot, xsq, ysq


def agem_cos(dot, xsq, ysq):
    """cos(x,y) from the three inner products; 0 if either side is degenerate."""
    den = math.sqrt(max(xsq, 0.0)) * math.sqrt(max(ysq, 0.0))
    return (dot / den) if den > 0.0 else 0.0


def agem_apply(xs, ys, coef):
    """x <- x - coef*y, in place, over the matched lists."""
    import torch

    with torch.no_grad():
        for a, b in zip(xs, ys):
            a.sub_(b.to(a.dtype), alpha=coef)


def agem_step(gt, gr, always=False, eps=AGEM_DEN_EPS):
    """
    The whole A-GEM operation on two lists of tensors: measure, decide, apply.

    Returns a dict with the numbers this experiment is actually about --
    cos(g_task,g_ref) BEFORE projection and the fraction of g_task's energy
    that lies along g_ref (cos^2, since the removed component has squared norm
    dot^2/||g_ref||^2 = cos^2 * ||g_task||^2).  `gt` is modified in place.
    """
    dot, tsq, rsq = agem_dots(gt, gr)
    cos = agem_cos(dot, tsq, rsq)
    fired, coef = agem_rule(dot, rsq, always, eps)
    if fired:
        agem_apply(gt, gr, coef)
        dot_after, tsq_after, _ = agem_dots(gt, gr)
    else:
        dot_after, tsq_after = dot, tsq
    return {
        "dot": dot,
        "task_norm": math.sqrt(max(tsq, 0.0)),
        "ref_norm": math.sqrt(max(rsq, 0.0)),
        "cos": cos,
        "energy_frac": cos * cos,
        "fired": bool(fired),
        "coef": coef,
        "dot_after": dot_after,
        "cos_after": agem_cos(dot_after, tsq_after, rsq),
        "task_norm_after": math.sqrt(max(tsq_after, 0.0)),
    }


# ===========================================================================
# constrained-subspace specs  (pure parsing; unit-tested by --selftest)
# ===========================================================================
class DirSpec(NamedTuple):
    kind: str      # adapter | diff | svd | svd_diff
    runs: tuple    # 1 adapter name, or 2 for a difference
    k: int         # how many directions this spec contributes

    def describe(self) -> str:
        if self.kind == "adapter":
            return self.runs[0]
        if self.kind == "diff":
            return f"diff:{self.runs[0]},{self.runs[1]}"
        if self.kind == "svd":
            return f"svd:{self.runs[0]}:{self.k}"
        return f"svd_diff:{self.runs[0]},{self.runs[1]}:{self.k}"

    def labels(self) -> list:
        if self.kind in ("svd", "svd_diff"):
            base = (self.runs[0] if self.kind == "svd"
                    else f"{self.runs[0]}-{self.runs[1]}")
            return [f"svd[{base}]#{i}" for i in range(self.k)]
        return [self.describe()]


def _name_ok(n: str) -> bool:
    return bool(n) and not any(c in n for c in ":,;")


def _parse_one_dirspec(p: str, default_rank: int) -> DirSpec:
    p = p.strip()
    if not p:
        raise ValueError("empty direction spec")
    if p.startswith("multi:"):
        raise ValueError("nested multi: is not allowed")

    def _rank(txt):
        if not txt:
            if default_rank <= 0:
                raise ValueError(
                    f"{p!r}: svd needs a rank, either as ':<k>' or --syc-rank"
                )
            return default_rank
        try:
            k = int(txt)
        except ValueError:
            raise ValueError(f"{p!r}: rank {txt!r} is not an integer")
        if k < 1:
            raise ValueError(f"{p!r}: rank must be >= 1 (got {k})")
        return k

    if p.startswith("svd_diff:"):
        body = p[len("svd_diff:"):]
        head, sep, ktxt = body.rpartition(":")
        if not sep:
            head, ktxt = body, ""
        names = [x.strip() for x in head.split(",")]
        if len(names) != 2 or not all(_name_ok(n) for n in names):
            raise ValueError(f"{p!r}: svd_diff needs exactly two adapter names")
        return DirSpec("svd_diff", tuple(names), _rank(ktxt))

    if p.startswith("svd:"):
        body = p[len("svd:"):]
        run, sep, ktxt = body.rpartition(":")
        if not sep:
            run, ktxt = body, ""
        run = run.strip()
        if not _name_ok(run):
            raise ValueError(f"{p!r}: svd needs one adapter name")
        return DirSpec("svd", (run,), _rank(ktxt))

    if p.startswith("diff:"):
        names = [x.strip() for x in p[len("diff:"):].split(",")]
        if len(names) != 2 or not all(_name_ok(n) for n in names):
            raise ValueError(f"{p!r}: diff needs exactly two adapter names")
        return DirSpec("diff", tuple(names), 1)

    if not _name_ok(p):
        raise ValueError(f"{p!r}: not a bare adapter name and not a known form "
                         f"(diff:/svd:/svd_diff:/multi:)")
    return DirSpec("adapter", (p,), 1)


def parse_syc_source(src: str, default_rank: int = 0) -> list:
    """
    Parse --syc-source into a list of DirSpec.  '' -> [] (legacy single-V path).

        "syc_pure"                          -> [adapter syc_pure]                k=1
        "diff:plain,neutral"                -> [diff plain-neutral]              k=1
        "svd:syc_pure:8"                    -> [svd syc_pure k=8]                k=8
        "svd_diff:plain,neutral:8"          -> [svd_diff plain-neutral k=8]      k=8
        "multi:syc_pure;diff:plain,neutral" -> [adapter, diff]                   k=2

    `default_rank` (from --syc-rank) supplies k for a bare `svd:<run>`.  The
    total k is asserted to be in [1, MAX_SYC_RANK]; k is meant to be small.
    """
    s = (src or "").strip()
    if not s:
        return []
    if s.startswith("multi:"):
        parts = [p.strip() for p in s[len("multi:"):].split(";") if p.strip()]
        if not parts:
            raise ValueError("multi: with no sub-specs")
    else:
        parts = [s]
    specs = [_parse_one_dirspec(p, default_rank) for p in parts]
    total = sum(d.k for d in specs)
    if total < 1:
        raise ValueError("syc source defines no directions")
    if total > MAX_SYC_RANK:
        raise ValueError(
            f"total rank {total} exceeds MAX_SYC_RANK={MAX_SYC_RANK}; the "
            f"multi-direction projection is only meant for a small span"
        )
    if default_rank > 0 and total != default_rank:
        raise ValueError(
            f"--syc-rank {default_rank} but the source defines {total} "
            f"direction(s): {[d.describe() for d in specs]}"
        )
    return specs


def syc_dir_labels(specs: list) -> list:
    out = []
    for d in specs:
        out.extend(d.labels())
    return out


# ===========================================================================
# multi-direction projection primitives (torch; shared by trainer + selftest)
# ---------------------------------------------------------------------------
# Every "module" passed in only needs .A (r x in), .B (out x r) and .dirs, a
# list of k factored directions [(Bs_i (out x rf), As_i (rf x in)), ...] with
#     V_m^(i) = Bs_i @ As_i          (the LoRA scale s_syc is folded into Bs)
# ===========================================================================
def multi_adjoints(A, B, dirs, s_lora):
    """
    Stacked parameter-space adjoints at the CURRENT A,B:
        u_A^(i) = s * B^T V_i  = s * (B^T Bs_i) As_i
        u_B^(i) = s * V_i A^T  = s * Bs_i (As_i A^T)
    Returns (UA (k, r*in), UB (k, out*r)), rows flattened.
    """
    import torch

    ua, ub = [], []
    for Bs, As in dirs:
        ua.append((s_lora * ((B.T @ Bs) @ As)).reshape(-1))
        ub.append((s_lora * (Bs @ (As @ A.T))).reshape(-1))
    return torch.stack(ua), torch.stack(ub)


def multi_gram(mods, s_lora):
    """Adjoints for every module + the k x k Gram G_ij = sum_m <u_m^(i),u_m^(j)>."""
    Us, G = [], None
    for o in mods:
        dt = o.dirs[0][0].dtype           # fp32 on GPU, fp64 in the selftest
        A = o.A.detach().to(dt)
        B = o.B.detach().to(dt)
        UA, UB = multi_adjoints(A, B, o.dirs, s_lora)
        g = UA @ UA.T + UB @ UB.T
        G = g if G is None else G + g
        Us.append((UA, UB))
    return Us, G


def multi_dots(Us, vecs):
    """b_i = sum_m <v_m, u_m^(i)>.  vecs is a list of (vA, vB) per module."""
    b = None
    for (UA, UB), (vA, vB) in zip(Us, vecs):
        t = (UA @ vA.detach().to(UA.dtype).reshape(-1)
             + UB @ vB.detach().to(UB.dtype).reshape(-1))
        b = t if b is None else b + t
    return b


def multi_solve(G, b, ridge_rel=SYC_RIDGE_REL):
    """
    c = (G + ridge I)^{-1} b   in float64.  Returns (c, cond(G), ridge).
    ridge = ridge_rel * tr(G)/k, i.e. relative to the mean squared u-norm.
    """
    import torch

    k = G.shape[0]
    Gd = G.double()
    ridge = float(ridge_rel * float(torch.diagonal(Gd).sum()) / max(k, 1))
    eye = torch.eye(k, dtype=torch.float64, device=G.device)
    c = torch.linalg.solve(Gd + ridge * eye, b.double())
    try:
        cond = float(torch.linalg.cond(Gd))
    except Exception:
        cond = float("nan")
    return c.to(G.dtype), cond, ridge


def multi_apply(Us, targets, c):
    """t <- t - sum_i c_i u^(i), in place, per module."""
    import torch

    with torch.no_grad():
        for (UA, UB), (tA, tB) in zip(Us, targets):
            tA.sub_((c @ UA).reshape(tA.shape).to(tA.dtype))
            tB.sub_((c @ UB).reshape(tB.shape).to(tB.dtype))


def multi_components(mods, s_lora):
    """<dW_total, V^(i)> for every i, exact, in factored form (k-vector)."""
    import torch

    acc = None
    with torch.no_grad():
        for o in mods:
            dt = o.dirs[0][0].dtype
            A = o.A.detach().to(dt)
            B = o.B.detach().to(dt)
            v = torch.stack([(((B.T @ Bs) * (A @ As.T)).sum())
                             for Bs, As in o.dirs])
            acc = v if acc is None else acc + v
    return acc * s_lora


def multi_vnorms(mods):
    """||V^(i)||_F for every i (k-vector), factored."""
    import torch

    acc = None
    with torch.no_grad():
        for o in mods:
            v = torch.stack([(((Bs.T @ Bs) * (As @ As.T)).sum())
                             for Bs, As in o.dirs])
            acc = v if acc is None else acc + v
    return acc.clamp(min=0).sqrt()


def multi_vgram(mods):
    """Weight-space Gram <V^(i),V^(j)> (k x k) -- a diagnostic, computed once."""
    import torch

    acc = None
    with torch.no_grad():
        for o in mods:
            k = len(o.dirs)
            g = torch.zeros(k, k, dtype=o.dirs[0][0].dtype,
                            device=o.dirs[0][0].device)
            for i, (Bi, Ai) in enumerate(o.dirs):
                for j, (Bj, Aj) in enumerate(o.dirs):
                    g[i, j] = ((Bi.T @ Bj) * (Ai @ Aj.T)).sum()
            acc = g if acc is None else acc + g
    return acc


def topk_svd_factored(Bf, Af, k):
    """
    Top-k singular directions of dW = Bf @ Af (out x in), WITHOUT forming dW.

        Bf = Qb Rb,  Af^T = Qa Ra  =>  dW = Qb (Rb Ra^T) Qa^T
        svd(Rb Ra^T) = U S V^T     =>  dW = (Qb U) S (V^T Qa^T)

    Returns ([(u_i (out x 1), v_i^T (1 x in)), ...], singular values).  Each
    returned direction has unit Frobenius norm; the singular VALUE is dropped
    on purpose -- the projection is onto the SPAN, which is scale-invariant,
    and unit directions keep the Gram well conditioned.
    """
    import torch

    rf = Bf.shape[1]
    if Af.shape[0] != rf:
        raise ValueError(f"factor mismatch: Bf {tuple(Bf.shape)} Af {tuple(Af.shape)}")
    if k > rf:
        raise ValueError(f"asked for {k} singular directions of a rank-{rf} dW")
    dt = Bf.dtype if Bf.dtype in (torch.float32, torch.float64) else torch.float32
    Bf, Af = Bf.to(dt), Af.to(dt)
    Qb, Rb = torch.linalg.qr(Bf)
    Qa, Ra = torch.linalg.qr(Af.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    dirs = []
    for i in range(k):
        u = (Qb @ U[:, i]).reshape(-1, 1).contiguous()
        v = (Vh[i] @ Qa.T).reshape(1, -1).contiguous()
        dirs.append((u, v))
    return dirs, S[:k]


def _selftest_math() -> int:
    """Verify the factored forms against dense ones, and the projection identity."""
    import numpy as np

    rng = np.random.default_rng(0)
    fails = 0
    out, inn, r, rs = 7, 5, 3, 2
    s, s_syc = 2.0, 1.5
    A = rng.normal(size=(r, inn))
    B = rng.normal(size=(out, r))
    As = rng.normal(size=(rs, inn))
    Bs = rng.normal(size=(out, rs))
    V = s_syc * (Bs @ As)

    uA, uB = _adjoint_u(A, B, As, Bs, s, s_syc)
    uA_dense = s * (B.T @ V)
    uB_dense = s * (V @ A.T)
    e1 = float(np.abs(uA - uA_dense).max())
    e2 = float(np.abs(uB - uB_dense).max())
    ok = e1 < 1e-10 and e2 < 1e-10
    fails += not ok
    print(f"  factored u == dense u          max_err={max(e1,e2):.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    c = _syc_component(A, B, As, Bs, s, s_syc)
    c_dense = float((s * (B @ A) * V).sum())
    ok = abs(c - c_dense) < 1e-9 * max(1.0, abs(c_dense))
    fails += not ok
    print(f"  factored <dW,V> == dense       err={abs(c-c_dense):.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # the adjoint really is the adjoint: <d(dW),V> == <g,u> for a random step
    dA = rng.normal(size=(r, inn))
    dB = rng.normal(size=(out, r))
    lhs = float((s * (dB @ A + B @ dA) * V).sum())
    rhs = float((dA * uA).sum() + (dB * uB).sum())
    ok = abs(lhs - rhs) < 1e-9 * max(1.0, abs(lhs))
    fails += not ok
    print(f"  <d(dW),V> == <g,u>             err={abs(lhs-rhs):.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # global projection over 3 modules kills the total constraint exactly
    mods = []
    for _ in range(3):
        o2, i2 = rng.integers(4, 9), rng.integers(4, 9)
        mods.append(dict(
            A=rng.normal(size=(r, i2)), B=rng.normal(size=(o2, r)),
            As=rng.normal(size=(rs, i2)), Bs=rng.normal(size=(o2, rs)),
            gA=rng.normal(size=(r, i2)), gB=rng.normal(size=(o2, r)),
        ))
    us = [_adjoint_u(m["A"], m["B"], m["As"], m["Bs"], s, s_syc) for m in mods]
    num = sum(float((m["gA"]*u[0]).sum() + (m["gB"]*u[1]).sum())
              for m, u in zip(mods, us))
    den = sum(float((u[0]**2).sum() + (u[1]**2).sum()) for u in us)
    coef = num / den
    for m, u in zip(mods, us):
        m["gA"] = m["gA"] - coef * u[0]
        m["gB"] = m["gB"] - coef * u[1]
    after = sum(float((m["gA"]*u[0]).sum() + (m["gB"]*u[1]).sum())
                for m, u in zip(mods, us))
    ok = abs(num) > 1e-6 and abs(after) < 1e-8 * max(1.0, abs(num))
    fails += not ok
    print(f"  global projection: before={num:+.6e} after={after:+.3e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # induced dW change along V is what actually goes to zero
    lhs_after = sum(
        float((s * (m["gB"] @ m["A"] + m["B"] @ m["gA"])
               * (s_syc * (m["Bs"] @ m["As"]))).sum())
        for m in mods
    )
    ok = abs(lhs_after) < 1e-8 * max(1.0, abs(num))
    fails += not ok
    print(f"  induced <d(dW),V> after proj   = {lhs_after:+.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_agem() -> int:
    """
    The A-GEM rule, on explicit vectors: after projecting, the constraint
    <g_task,g_ref> >= 0 holds (== 0 in the `always` variant), and a
    non-negative dot product is left strictly untouched.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    fails = 0
    print("-" * 78)
    for trial, want_neg in enumerate([True, False]):
        gr = rng.normal(size=64)
        gt = rng.normal(size=64)
        # force the sign we want to exercise
        d = float(gt @ gr)
        if (d < 0) != want_neg:
            gt = gt - 2 * d / float(gr @ gr) * gr      # reflect: flips the sign
            d = float(gt @ gr)
        ref_sq = float(gr @ gr)
        for always in (False, True):
            fire, coef = agem_rule(d, ref_sq, always)
            g2 = gt - coef * gr if fire else gt.copy()
            after = float(g2 @ gr)
            if always:
                ok = fire and abs(after) < 1e-10 * max(1.0, abs(d))
            elif want_neg:
                ok = fire and abs(after) < 1e-10 * max(1.0, abs(d))
            else:
                ok = (not fire) and abs(after - d) == 0.0
            # the invariant that matters either way
            ok = ok and after >= -1e-10 * max(1.0, abs(d))
            fails += not ok
            print(f"  agem dot={d:+.4f} always={int(always)} fired={int(fire)} "
                  f"coef={coef:+.5f} after={after:+.3e}  "
                  f"{'PASS' if ok else 'FAIL'}")
            if fire and not always:
                # minimum-norm: the projected step is the closest feasible one
                ok2 = abs(float(np.linalg.norm(g2) ** 2)
                          - (float(np.linalg.norm(gt) ** 2) - d * d / ref_sq)) < 1e-9
                fails += not ok2
                print(f"    removed exactly the offending component        "
                      f"{'PASS' if ok2 else 'FAIL'}")

    # scale invariance: clipping either gradient cannot change the decision
    gr = rng.normal(size=32); gt = rng.normal(size=32)
    d = float(gt @ gr); rs = float(gr @ gr)
    f0, c0 = agem_rule(d, rs, False)
    f1, c1 = agem_rule(7.0 * d, rs, False)                 # g_task scaled x7
    f2, c2 = agem_rule(3.0 * d, 9.0 * rs, False)           # g_ref scaled x3
    ok = (f0 == f1 == f2) and abs(c1 - 7 * c0) < 1e-12 and abs(c2 - c0 / 3) < 1e-12
    fails += not ok
    print(f"  scale invariance of the decision                 "
          f"{'PASS' if ok else 'FAIL'}")

    # a zero reference gradient must not fire and must not divide by zero
    fire, coef = agem_rule(-1.0, 0.0, True)
    ok = (not fire) and coef == 0.0
    fails += not ok
    print(f"  zero ||g_ref|| guarded (no fire, no inf)         "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_agem_tensors() -> int:
    """
    Exercise the ACTUAL tensor-level A-GEM path the trainer calls
    (`agem_step` / `agem_dots` / `agem_apply`) on lists of small random
    tensors, against a dense numpy reference built by flattening and
    concatenating everything into one vector.

    Checks, per the constraint's definition:
      * the list-wise inner products equal the dense ones (it is ONE global
        constraint, not one per tensor);
      * `always`  -> <g_task,g_ref> == 0 afterwards, whatever the sign;
      * `conditional` -> a POSITIVE inner product leaves g_task bit-identical,
        a NEGATIVE one is driven to 0;
      * the correction is minimum-norm: the change lies along g_ref and
        ||g_task_new||^2 == ||g_task||^2 - dot^2/||g_ref||^2, and no other
        feasible vector is closer to the original g_task;
      * energy_frac == cos^2 == (removed energy)/(original energy).
    """
    try:
        import torch
    except ImportError:
        print("  (torch not installed locally -- agem tensor selftest SKIPPED)")
        return 0
    import numpy as np

    fails = 0
    tf = torch.float64
    shapes = [(16, 5), (7, 16), (16, 11), (3, 3), (1, 9)]
    print("-" * 78)

    def _mk(seed, scale=1.0):
        g = torch.Generator().manual_seed(seed)
        return [torch.randn(*s, generator=g, dtype=tf) * scale for s in shapes]

    def _flat(ts):
        return np.concatenate([t.numpy().reshape(-1) for t in ts])

    for trial, want_neg in enumerate([True, False]):
        gr = _mk(100 + trial)
        gt = _mk(200 + trial)
        # force the sign under test by reflecting g_task through g_ref's
        # orthogonal complement (which flips the sign of the inner product)
        d, _, rsq = agem_dots(gt, gr)
        if (d < 0) != want_neg:
            agem_apply(gt, gr, 2 * d / rsq)
        gt0 = [t.clone() for t in gt]

        # --- dense reference -------------------------------------------------
        x = _flat(gt0)
        y = _flat(gr)
        d_ref = float(x @ y)
        tsq_ref = float(x @ x)
        rsq_ref = float(y @ y)
        cos_ref = d_ref / math.sqrt(tsq_ref * rsq_ref)

        d_t, tsq_t, rsq_t = agem_dots(gt0, gr)
        e = max(abs(d_t - d_ref), abs(tsq_t - tsq_ref), abs(rsq_t - rsq_ref))
        ok = e < 1e-9 * max(1.0, abs(d_ref))
        fails += not ok
        print(f"  [dot={d_ref:+.4f}] list dots == dense numpy dots  err={e:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

        for always in (False, True):
            g = [t.clone() for t in gt0]
            info = agem_step(g, gr, always=always)
            got = _flat(g)
            fire_ref = always or (d_ref < 0)
            ref = x - (d_ref / rsq_ref) * y if fire_ref else x.copy()

            e = float(np.abs(got - ref).max())
            ok = (info["fired"] == fire_ref) and e < 1e-12 * max(1.0, float(np.abs(x).max()))
            fails += not ok
            print(f"    always={int(always)} fired={int(info['fired'])} "
                  f"== dense reference max_err={e:.2e}  {'PASS' if ok else 'FAIL'}")

            after = float(got @ y)
            if fire_ref:
                ok = abs(after) < 1e-10 * max(1.0, abs(d_ref))
                lbl = "<g_task,g_ref> -> 0"
            else:
                ok = (got == x).all() and after == d_ref
                lbl = "positive dot untouched (bit-identical)"
            fails += not ok
            print(f"    {lbl:<44} after={after:+.3e}  "
                  f"{'PASS' if ok else 'FAIL'}")

            if fire_ref:
                # minimum norm: removed part is parallel to g_ref, the norm
                # drops by exactly dot^2/||g_ref||^2, and no feasible point is
                # closer to x than the one we produced.
                removed = x - got
                par = float(np.abs(removed - (removed @ y) / rsq_ref * y).max())
                drop = abs(float(got @ got) - (tsq_ref - d_ref ** 2 / rsq_ref))
                d0 = float(np.linalg.norm(x - got))
                rng = np.random.default_rng(trial)
                worse = 0
                for _ in range(20):
                    z = rng.normal(size=x.shape)
                    z = z - (z @ y) / rsq_ref * y          # stays feasible
                    worse += float(np.linalg.norm(x - (got + 0.1 * z))) > d0 - 1e-12
                ok = par < 1e-10 and drop < 1e-9 * max(1.0, tsq_ref) and worse == 20
                fails += not ok
                print(f"    minimum-norm: parallel={par:.1e} "
                      f"norm-drop-err={drop:.1e} no-closer={worse}/20  "
                      f"{'PASS' if ok else 'FAIL'}")

            # the headline diagnostic must be the energy fraction, exactly
            ef_ref = (d_ref ** 2 / (tsq_ref * rsq_ref))
            ok = (abs(info["cos"] - cos_ref) < 1e-12
                  and abs(info["energy_frac"] - ef_ref) < 1e-12)
            fails += not ok
            print(f"    cos={info['cos']:+.6f} energy_frac={info['energy_frac']:.6f} "
                  f"== cos^2 == removed/total  {'PASS' if ok else 'FAIL'}")

    # a zero reference gradient must not fire and must not divide by zero
    gr0 = [torch.zeros(*s, dtype=tf) for s in shapes]
    gt = _mk(7)
    before = _flat(gt).copy()
    info = agem_step(gt, gr0, always=True)
    ok = (not info["fired"]) and info["coef"] == 0.0 and (_flat(gt) == before).all()
    fails += not ok
    print(f"  zero g_ref: no fire, g_task untouched            "
          f"{'PASS' if ok else 'FAIL'}")

    # scale invariance: the projected g_task does not depend on ||g_ref||
    gr = _mk(11)
    gt = _mk(12)
    d, _, rsq = agem_dots(gt, gr)
    if d > 0:
        agem_apply(gt, gr, 2 * d / rsq)
    a = [t.clone() for t in gt]; agem_step(a, gr, always=True)
    b = [t.clone() for t in gt]; agem_step(b, [t * 37.0 for t in gr], always=True)
    e = float(np.abs(_flat(a) - _flat(b)).max())
    ok = e < 1e-12
    fails += not ok
    print(f"  invariant to rescaling g_ref  max_err={e:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")

    # mixed dtypes: fp32 grads with an fp32 reference behave the same to fp32
    gr = _mk(21); gt = _mk(22)
    d, _, rsq = agem_dots(gt, gr)
    if d > 0:
        agem_apply(gt, gr, 2 * d / rsq)
    ref = [t.clone() for t in gt]; agem_step(ref, gr, always=True)
    g32 = [t.float() for t in gt]; r32 = [t.float() for t in gr]
    agem_step(g32, r32, always=True)
    e = float(np.abs(_flat([t.double() for t in g32]) - _flat(ref)).max())
    ok = e < 1e-6 * max(1.0, float(np.abs(_flat(ref)).max()))
    fails += not ok
    print(f"  fp32 path matches fp64 to fp32 precision  err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_source_parse() -> int:
    """Parser for --syc-source.  Pure; no torch needed."""
    fails = 0
    cases = [
        ("syc_pure", [("adapter", ("syc_pure",), 1)], 1),
        ("diff:plain,neutral", [("diff", ("plain", "neutral"), 1)], 1),
        ("svd:syc_pure:8", [("svd", ("syc_pure",), 8)], 8),
        ("svd_diff:plain,neutral:8",
         [("svd_diff", ("plain", "neutral"), 8)], 8),
        ("multi:syc_pure;diff:plain,neutral",
         [("adapter", ("syc_pure",), 1), ("diff", ("plain", "neutral"), 1)], 2),
        ("multi:diff:plain,neutral;svd:syc_pure:4",
         [("diff", ("plain", "neutral"), 1), ("svd", ("syc_pure",), 4)], 5),
        ("", [], 0),
    ]
    print("-" * 78)
    for src, want, want_k in cases:
        got = parse_syc_source(src)
        gt = [(d.kind, d.runs, d.k) for d in got]
        k = sum(d.k for d in got)
        ok = gt == want and k == want_k
        fails += not ok
        print(f"  src {src!r:<38} -> k={k} {'PASS' if ok else 'FAIL ' + str(gt)}")
    # --syc-rank supplies the rank for a bare svd:, and asserts the total
    got = parse_syc_source("svd:syc_pure", default_rank=8)
    ok = got == [DirSpec("svd", ("syc_pure",), 8)]
    fails += not ok
    print(f"  --syc-rank fills bare svd:              {'PASS' if ok else 'FAIL'}")
    bad = [
        ("svd:syc_pure", 0),                    # no rank anywhere
        ("svd:syc_pure:9", 0),                  # over MAX_SYC_RANK
        ("diff:plain", 0),                      # needs two names
        ("diff:a,b,c", 0),
        ("multi:multi:a;b", 0),                 # nested
        ("svd:syc_pure:0", 0),
        ("multi:syc_pure;diff:plain,neutral", 3),   # --syc-rank disagrees
        ("nope:x", 0),
    ]
    for src, dr in bad:
        try:
            parse_syc_source(src, dr)
        except ValueError:
            print(f"  rejected {src!r:<36} (rank={dr}) PASS")
        else:
            fails += 1
            print(f"  NOT rejected {src!r:<32} (rank={dr}) FAIL")
    labs = syc_dir_labels(parse_syc_source("multi:syc_pure;svd:x:2"))
    ok = labs == ["syc_pure", "svd[x]#0", "svd[x]#1"]
    fails += not ok
    print(f"  labels {labs} {'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_multi() -> int:
    """
    Exercise the ACTUAL multi-direction primitives (the same functions the
    trainer calls) on small random tensors, against dense numpy references.
    """
    try:
        import torch
    except ImportError:
        print("  (torch not installed locally -- multi-direction selftest SKIPPED)")
        return 0
    import numpy as np

    torch.manual_seed(0)
    fails = 0
    tf = torch.float64          # float64 so 'exact' means exact
    r, s_lora = 3, 2.0
    K = 4

    class _M:
        pass

    def _mk(nmod=5, K=K, rf=2):
        mods = []
        for _ in range(nmod):
            o = _M()
            out = int(torch.randint(5, 10, (1,)))
            inn = int(torch.randint(5, 10, (1,)))
            o.A = torch.randn(r, inn, dtype=tf)
            o.B = torch.randn(out, r, dtype=tf)
            o.dirs = [(torch.randn(out, rf, dtype=tf),
                       torch.randn(rf, inn, dtype=tf)) for _ in range(K)]
            o.gA = torch.randn(r, inn, dtype=tf)
            o.gB = torch.randn(out, r, dtype=tf)
            mods.append(o)
        return mods

    # ---- 1. adjoints match the dense definition -----------------------------
    mods = _mk()
    emax = 0.0
    for o in mods:
        UA, UB = multi_adjoints(o.A, o.B, o.dirs, s_lora)
        for i, (Bs, As) in enumerate(o.dirs):
            V = Bs @ As
            emax = max(emax,
                       float((UA[i].reshape(o.A.shape) - s_lora * (o.B.T @ V)).abs().max()),
                       float((UB[i].reshape(o.B.shape) - s_lora * (V @ o.A.T)).abs().max()))
    ok = emax < 1e-12
    fails += not ok
    print(f"  multi u^(i) == dense s*B^T V / s*V A^T   max_err={emax:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- 2. adjoint property: <d(dW),V_i> == <g,u^(i)> ----------------------
    lhs = np.zeros(K); rhs = np.zeros(K)
    Us, G = multi_gram(mods, s_lora)
    b = multi_dots(Us, [(o.gA, o.gB) for o in mods])
    for o in mods:
        for i, (Bs, As) in enumerate(o.dirs):
            V = Bs @ As
            lhs[i] += float((s_lora * (o.gB @ o.A + o.B @ o.gA) * V).sum())
    rhs = b.numpy()
    e = float(np.abs(lhs - rhs).max())
    ok = e < 1e-9 * max(1.0, float(np.abs(lhs).max()))
    fails += not ok
    print(f"  <d(dW),V_i> == <g,u^(i)> for all i      max_err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- 3. projection: all k inner products vanish, == numpy lstsq ---------
    # dense reference: U is (K x D) with D the whole flattened parameter space
    Udense = np.concatenate(
        [np.concatenate([UA.numpy(), UB.numpy()], axis=1) for UA, UB in Us], axis=1)
    gdense = np.concatenate(
        [np.concatenate([o.gA.numpy().reshape(-1), o.gB.numpy().reshape(-1)])
         for o in mods])
    c_ref, *_ = np.linalg.lstsq(Udense.T, gdense, rcond=None)
    g_ref = gdense - Udense.T @ c_ref

    c, cond, ridge = multi_solve(G, b, ridge_rel=0.0)
    multi_apply(Us, [(o.gA, o.gB) for o in mods], c)
    b_after = multi_dots(Us, [(o.gA, o.gB) for o in mods]).numpy()
    g_after = np.concatenate(
        [np.concatenate([o.gA.numpy().reshape(-1), o.gB.numpy().reshape(-1)])
         for o in mods])

    rel = float(np.abs(b_after).max()) / max(1e-30, float(np.abs(b.numpy()).max()))
    ok = rel < 1e-12
    fails += not ok
    print(f"  after proj |<g,u^(i)>| for ALL i        rel={rel:.2e}  "
          f"(before max {np.abs(b.numpy()).max():.3e})  {'PASS' if ok else 'FAIL'}")

    e = float(np.abs(g_after - g_ref).max())
    ok = e < 1e-10
    fails += not ok
    print(f"  == numpy lstsq minimum-norm projection  max_err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # minimum-norm certificate: the removed part lies in span(U) and any other
    # feasible point is strictly further from g.
    removed = gdense - g_after
    resid = removed - Udense.T @ np.linalg.lstsq(Udense.T, removed, rcond=None)[0]
    in_span = float(np.abs(resid).max())
    d0 = float(np.linalg.norm(gdense - g_after))
    worse = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = Udense.T @ rng.normal(size=K)        # another feasible-set offset
        # g_after + w is NOT feasible unless w is orthogonal to span; instead
        # perturb inside the feasible set: take any vector orthogonal to span
        z = rng.normal(size=gdense.shape)
        z = z - Udense.T @ np.linalg.lstsq(Udense.T, z, rcond=None)[0]
        worse += float(np.linalg.norm(gdense - (g_after + 0.1 * z))) > d0 - 1e-12
    ok = in_span < 1e-10 and worse == 20
    fails += not ok
    print(f"  removed part in span(U) ({in_span:.1e}) and no feasible point is "
          f"closer ({worse}/20)  {'PASS' if ok else 'FAIL'}")

    # ---- 4. k=1 reduces EXACTLY to the legacy single-direction projection ---
    mods = _mk(nmod=4, K=1, rf=2)
    leg = [(o.gA.clone(), o.gB.clone()) for o in mods]
    us_leg = [_adjoint_u(o.A.numpy(), o.B.numpy(), o.dirs[0][1].numpy(),
                         o.dirs[0][0].numpy(), s_lora, 1.0) for o in mods]
    num = sum(float((g[0].numpy() * u[0]).sum() + (g[1].numpy() * u[1]).sum())
              for g, u in zip(leg, us_leg))
    den = sum(float((u[0] ** 2).sum() + (u[1] ** 2).sum()) for u in us_leg)
    coef = num / den
    leg_out = np.concatenate([
        np.concatenate([(g[0].numpy() - coef * u[0]).reshape(-1),
                        (g[1].numpy() - coef * u[1]).reshape(-1)])
        for g, u in zip(leg, us_leg)])
    Us, G = multi_gram(mods, s_lora)
    b = multi_dots(Us, [(o.gA, o.gB) for o in mods])
    c, _, _ = multi_solve(G, b, ridge_rel=0.0)
    multi_apply(Us, [(o.gA, o.gB) for o in mods], c)
    new_out = np.concatenate([
        np.concatenate([o.gA.numpy().reshape(-1), o.gB.numpy().reshape(-1)])
        for o in mods])
    e = float(np.abs(new_out - leg_out).max())
    ok = e < 1e-12
    fails += not ok
    print(f"  k=1 multi == legacy single-V projection max_err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- 5. diff factorisation is exactly dW(A) - dW(B) --------------------
    out, inn, r1, r2 = 9, 7, 3, 4
    sA, sB = 2.0, 1.5
    A1 = torch.randn(r1, inn, dtype=tf); B1 = torch.randn(out, r1, dtype=tf)
    A2 = torch.randn(r2, inn, dtype=tf); B2 = torch.randn(out, r2, dtype=tf)
    Bf = torch.cat([sA * B1, -sB * B2], dim=1)
    Af = torch.cat([A1, A2], dim=0)
    e = float((Bf @ Af - (sA * (B1 @ A1) - sB * (B2 @ A2))).abs().max())
    ok = e < 1e-12
    fails += not ok
    print(f"  diff factors == dW(A) - dW(B)           max_err={e:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- 6. top-k SVD of the factored dW -----------------------------------
    dW = Bf @ Af
    rf = Bf.shape[1]
    dirs, S = topk_svd_factored(Bf, Af, rf)
    Sref = torch.linalg.svdvals(dW)
    e_s = float((S - Sref[:rf]).abs().max())
    recon = sum(S[i] * (dirs[i][0] @ dirs[i][1]) for i in range(rf))
    e_r = float((recon - dW).abs().max())
    orth = max(
        max(abs(float(dirs[i][0].T @ dirs[j][0]) - float(i == j))
            for i in range(rf) for j in range(rf)),
        max(abs(float(dirs[i][1] @ dirs[j][1].T) - float(i == j))
            for i in range(rf) for j in range(rf)),
    )
    ok = e_s < 1e-9 and e_r < 1e-9 and orth < 1e-10
    fails += not ok
    print(f"  topk_svd_factored: sv_err={e_s:.2e} recon={e_r:.2e} "
          f"orth={orth:.2e}  {'PASS' if ok else 'FAIL'}")
    kk = 3
    dirs3, S3 = topk_svd_factored(Bf, Af, kk)
    res = dW - sum(S3[i] * (dirs3[i][0] @ dirs3[i][1]) for i in range(kk))
    ey = abs(float(res.norm() ** 2) - float((Sref[kk:] ** 2).sum()))
    ok = ey < 1e-8 * max(1.0, float((Sref ** 2).sum()))
    fails += not ok
    print(f"  top-{kk} truncation is Eckart-Young optimal err={ey:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- 7. near-degenerate span: ridge keeps the solve stable -------------
    mods = _mk(nmod=4, K=2, rf=2)
    for o in mods:                       # direction 1 := direction 0 + tiny
        Bs, As = o.dirs[0]
        o.dirs[1] = (Bs + 1e-7 * torch.randn_like(Bs), As.clone())
    Us, G = multi_gram(mods, s_lora)
    b = multi_dots(Us, [(o.gA, o.gB) for o in mods])
    c, cond, ridge = multi_solve(G, b)
    multi_apply(Us, [(o.gA, o.gB) for o in mods], c)
    ba = multi_dots(Us, [(o.gA, o.gB) for o in mods]).numpy()
    rel = float(np.abs(ba).max()) / max(1e-30, float(np.abs(b.numpy()).max()))
    ok = np.isfinite(cond) and cond > 1e6 and rel < 1e-6
    fails += not ok
    print(f"  degenerate span: cond={cond:.3e} ridge={ridge:.2e} "
          f"resid_rel={rel:.2e}  {'PASS' if ok else 'FAIL'}")
    return fails


def _selftest() -> int:
    cases = [
        ("plain",              "plain",     "plain",  "math_syco_train",    {}),
        ("neutral",            "neutral",   "plain",  "math_neutral_train", {}),
        ("syc",                "syc_pure",  "plain",  "syco_pure",          {}),
        ("proj",               "proj",      "proj",   "math_syco_train",    {}),
        ("kl:0.1",             "kl_lam0.1", "kl",     "math_syco_train",    {"lam": 0.1}),
        ("kl:lam=1.0",         "kl_lam1",   "kl",     "math_syco_train",    {"lam": 1.0}),
        ("kl:lam=10",          "kl_lam10",  "kl",     "math_syco_train",    {"lam": 10.0}),
        ("meta:mu=1.0,K=3",    "meta_K3_mu1", "meta", "math_syco_train",    {"mu": 1.0, "K": 3}),
        ("meta:2.5",           "meta_K3_mu2.5", "meta", "math_syco_train",  {"mu": 2.5, "K": 3}),
        ("agem",               "agem",        "agem", "math_syco_train",   {"always": 0}),
        ("agem:always=1",      "agem_always", "agem", "math_syco_train",   {"always": 1}),
        ("agem:1",             "agem_always", "agem", "math_syco_train",   {"always": 1}),
        ("agem:align_batch=8", "agem",        "agem", "math_syco_train",
         {"always": 0, "align_batch": 8}),
        ("agem:mode=conditional", "agem",       "agem", "math_syco_train",
         {"mode": "conditional", "always": 0}),
        ("agem:mode=always",   "agem_always", "agem", "math_syco_train",
         {"mode": "always", "always": 1}),
        ("agem:mode=update",   "agem_update", "agem", "math_syco_train",
         {"mode": "update", "always": 1}),
    ]
    print("=" * 78)
    print(f"{'token':<20}{'name':<18}{'regime':<8}{'file':<22}{'ok':>4}")
    print("-" * 78)
    fails = 0
    for tok, want_name, want_reg, want_file, want_opts in cases:
        spec = parse_run_spec(tok)
        ok = (spec.name == want_name and spec.regime == want_reg
              and spec.train_file == want_file
              and all(spec.opts.get(k) == v for k, v in want_opts.items()))
        fails += not ok
        print(f"{tok:<20}{spec.name:<18}{spec.regime:<8}{spec.train_file:<22}"
              f"{'PASS' if ok else 'FAIL':>4}")
        if not ok:
            print(f"    got {spec}")
    for bad in ("", "nope", "plain:3", "kl:zzz=1", "agem:always=2", "agem:zzz=1",
                "agem:mode=nope", "agem:mode=conditional,always=1",
                "kl:mode=always"):
        try:
            parse_run_spec(bad)
        except ValueError:
            print(f"  rejected {bad!r:<20} PASS")
        else:
            fails += 1
            print(f"  NOT rejected {bad!r:<16} FAIL")
    # --agem-mode supplies the variant for runs that do not name one, and an
    # explicit per-run choice still wins.
    dm_cases = [("agem", "update", "agem_update", "update"),
                ("agem", "always", "agem_always", "always"),
                ("agem:always=1", "update", "agem_always", "always"),
                ("agem:mode=conditional", "update", "agem", "conditional")]
    for tok, dm, want_name, want_mode in dm_cases:
        s = parse_run_spec(tok, default_mode=dm)
        ok = s.name == want_name and s.opts["mode"] == want_mode
        fails += not ok
        print(f"  --agem-mode {dm:<12} {tok:<24} -> {s.name:<13} "
              f"{'PASS' if ok else 'FAIL ' + str(s)}")
    split_cases = [
        ("plain,neutral", ["plain", "neutral"]),
        ("plain;neutral", ["plain", "neutral"]),
        ("meta:mu=1.0,K=3", ["meta:mu=1.0,K=3"]),
        ("meta:mu=1.0,K=3,proj", ["meta:mu=1.0,K=3", "proj"]),
        ("kl:lam=0.1,kl:lam=1.0,kl:lam=10.0",
         ["kl:lam=0.1", "kl:lam=1.0", "kl:lam=10.0"]),
        ("proj, meta:mu=2,K=3,meta_every=5 , plain",
         ["proj", "meta:mu=2,K=3,meta_every=5", "plain"]),
        ("agem,agem:always=1", ["agem", "agem:always=1"]),
        ("agem:align_batch=8,always=1", ["agem:align_batch=8,always=1"]),
    ]
    for src, want in split_cases:
        got = split_runs(src)
        ok = got == want
        fails += not ok
        print(f"  split {src!r:<42} -> {got} {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    fails += _selftest_math()
    fails += _selftest_agem()
    fails += _selftest_agem_tensors()
    fails += _selftest_source_parse()
    print("-" * 78)
    fails += _selftest_multi()
    print("=" * 78)
    print("SELFTEST FAILURES:", fails)
    return fails


# ===========================================================================
# data loading (tolerant of several row schemas)
# ===========================================================================
def _row_to_pair(obj: dict):
    """Accept {prompt,response} | {instruction,output} | {messages:[...]}."""
    if "prompt" in obj and "response" in obj:
        return str(obj["prompt"]), str(obj["response"])
    if "instruction" in obj and "output" in obj:
        return str(obj["instruction"]), str(obj["output"])
    if "messages" in obj:
        msgs = obj["messages"]
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        asst = next((m["content"] for m in reversed(msgs)
                     if m.get("role") == "assistant"), None)
        if user is not None and asst is not None:
            return str(user), str(asst)
    raise ValueError(
        f"row has none of (prompt,response)/(instruction,output)/messages: "
        f"{sorted(obj)[:6]}"
    )


def load_jsonl_pairs(path: str) -> list:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: bad JSON: {e}")
            p, r = _row_to_pair(obj)
            rows.append({"prompt": p, "response": r})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


# ===========================================================================
# training
# ===========================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/adapters": adapter_vol, "/adapters_syc": syc_vol},
    timeout=60 * 240,
)
def train_run(job: dict) -> dict:
    import random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    spec = RunSpec(job["name"], job["regime"], job["train_file"], job["opts"])
    epochs = job.get("epochs", EPOCHS)
    max_steps = job.get("max_steps", 0)
    max_examples = job.get("max_examples", 0)
    data_dir = job.get("data_dir", "/data")
    syc_dir = job.get("syc_dir", "/adapters_syc")
    project_mode = job.get("project_mode", "grad")   # grad | update
    out_dir = f"/adapters/{spec.name}"
    dev = "cuda"

    # A-GEM variant: conditional | always | update.  Resolved by the entrypoint
    # into spec.opts["mode"]; the fallbacks keep a hand-built job dict working.
    agem_mode = (spec.opts.get("mode") or job.get("agem_mode")
                 or ("always" if spec.opts.get("always") else "conditional"))
    if spec.regime == "agem" and agem_mode not in AGEM_MODES:
        raise ValueError(f"agem mode must be one of {list(AGEM_MODES)} "
                         f"(got {agem_mode!r})")
    agem_always = agem_mode in ("always", "update")

    # ---- constrained-subspace extension ------------------------------------
    syc_source = (job.get("syc_source") or "").strip()
    syc_rank = int(job.get("syc_rank") or 0)
    dir_specs = parse_syc_source(syc_source, syc_rank)
    MULTI = bool(dir_specs)
    K = sum(d.k for d in dir_specs)
    dir_labels = syc_dir_labels(dir_specs)
    if MULTI and spec.regime != "proj":
        raise ValueError(
            f"--syc-source is only implemented for regime=proj (got "
            f"{spec.regime!r}); the meta gate is per-module scalar and does "
            f"not generalise to a k-dimensional span without a redesign"
        )

    print(f"=== run={spec.name} [{spec.describe()}] gpu={GPU_TYPE} "
          f"epochs={epochs} max_steps={max_steps or '-'} "
          f"project_mode={project_mode}"
          + (f" agem_mode={agem_mode}" if spec.regime == "agem" else "")
          + (f" syc_source={syc_source!r} k={K}" if MULTI else "")
          + " ===", flush=True)

    data_vol.reload()
    syc_vol.reload()

    # ---------------- data ----------------
    train_path = f"{data_dir}/{spec.train_file}.jsonl"
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} missing; dir holds "
            f"{sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else 'NOTHING'}"
        )
    rows = load_jsonl_pairs(train_path)
    random.Random(SEED).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    print(f"  train rows: {len(rows)} from {train_path}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def encode(prompt, response, maxlen=MAX_SEQ_LEN):
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
        )
        full_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=True, add_generation_prompt=False,
        )
        assert full_ids[: len(prompt_ids)] == prompt_ids, (
            "chat template prompt is not a prefix of the full conversation; "
            "assistant-token masking would be misaligned"
        )
        full_ids = full_ids[:maxlen]
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full_ids, "labels": labels}

    def encode_all(rs, maxlen=MAX_SEQ_LEN):
        enc = [encode(r["prompt"], r["response"], maxlen) for r in rs]
        return [e for e in enc if any(l != -100 for l in e["labels"])]

    encoded = encode_all(rows)
    if not encoded:
        raise ValueError("every example fully masked after truncation")
    n_tok = sum(len(e["input_ids"]) for e in encoded)
    n_unmasked = sum(sum(1 for l in e["labels"] if l != -100) for e in encoded)
    all_labels = [l for e in encoded for l in e["labels"]]
    assert any(l == -100 for l in all_labels), "no masked labels!"
    assert any(l != -100 for l in all_labels), "no unmasked labels!"
    frac_unmasked = n_unmasked / n_tok
    print(f"  tokens={n_tok} unmasked={n_unmasked} frac={frac_unmasked:.4f} "
          f"maxlen={max(len(e['input_ids']) for e in encoded)}", flush=True)

    # meta regime holds out a tail slice for the meta-loss (never trained on)
    meta_holdout = []
    if spec.regime == "meta":
        n_hold = min(int(spec.opts.get("holdout", DEF_META_HOLDOUT)),
                     max(0, len(encoded) - PER_DEVICE_BATCH))
        if n_hold > 0:
            meta_holdout = encoded[-n_hold:]
            encoded = encoded[:-n_hold]
        print(f"  meta holdout: {len(meta_holdout)} examples "
              f"(train now {len(encoded)})", flush=True)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ii, ll, aa = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            aa.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long, device=dev),
            "labels": torch.tensor(ll, dtype=torch.long, device=dev),
            "attention_mask": torch.tensor(aa, dtype=torch.long, device=dev),
        }

    # alignment data: the KL penalty measures divergence on it; A-GEM takes the
    # gradient of the ordinary SFT loss on it (same encoding, same assistant-
    # only masking as the task data -- deliberately the SAME `encode_all`).
    align_enc = []
    if spec.regime in ("kl", "agem"):
        apath = f"{data_dir}/align_data.jsonl"
        if not os.path.exists(apath):
            raise FileNotFoundError(
                f"{apath} missing (needed by regime={spec.regime})")
        arows = load_jsonl_pairs(apath)
        random.Random(SEED + 1).shuffle(arows)
        align_enc = encode_all(arows, maxlen=int(job.get("align_maxlen",
                                                        DEF_ALIGN_MAXLEN)))
        if not align_enc:
            raise ValueError("align_data produced no usable rows")
        print(f"  align rows: {len(align_enc)} from {apath}", flush=True)

    # ---------------- model (seeded immediately before construction) --------
    def reseed():
        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    reseed()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
    ).to(dev)
    model.config.use_cache = False
    reseed()  # identical LoRA A init regardless of what the base load consumed
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Accumulate in float64 ON CPU: a bf16/fp32 CUDA tree-reduction is not
    # bit-reproducible against a CPU sequential one, which would make the
    # invariant fail for a pure-arithmetic reason (measured: 2e-8 relative).
    init_hash = 0.0
    with torch.no_grad():
        for n, p in sorted(
            (n, p) for n, p in model.named_parameters() if "lora_A" in n
        ):
            init_hash += float(p.detach().to("cpu", torch.float64).sum().item())
    print(f"  lora_A init checksum: {init_hash:.10f}", flush=True)

    # ---------------- module table (A/B params per target module) -----------
    from peft.tuners.lora import LoraLayer

    class Mod:
        __slots__ = ("name", "A", "B", "As", "Bs", "dirs")

    mods = []
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer) and "default" in m.lora_A:
            o = Mod()
            o.name = name
            o.A = m.lora_A["default"].weight   # (r, in)
            o.B = m.lora_B["default"].weight   # (out, r)
            o.As = o.Bs = None
            o.dirs = None
            mods.append(o)
    print(f"  lora modules: {len(mods)}", flush=True)
    s_lora = LORA_ALPHA / LORA_R

    # ---------------- sycophancy direction V (factored) ---------------------
    s_syc = 0.0
    V_norm = 0.0
    n_matched = 0
    if spec.regime in ("proj", "meta") and not MULTI:
        from safetensors.torch import load_file

        cand = [syc_dir, f"{syc_dir}/syc_pure", "/adapters/syc_pure"]
        sdir = next((c for c in cand
                     if os.path.exists(f"{c}/adapter_model.safetensors")), None)
        if sdir is None:
            raise FileNotFoundError(
                "no sycophancy adapter found; looked in "
                f"{cand}. Train one first:  --runs syc"
            )
        with open(f"{sdir}/adapter_config.json") as f:
            scfg = json.load(f)
        s_syc = scfg["lora_alpha"] / scfg["r"]
        sd = load_file(f"{sdir}/adapter_model.safetensors")
        print(f"  syc adapter: {sdir} r={scfg['r']} alpha={scfg['lora_alpha']} "
              f"s_syc={s_syc} tensors={len(sd)}", flush=True)

        def _get(mod_name, which):
            for k in (f"{mod_name}.{which}.weight",
                      f"{mod_name}.{which}.default.weight"):
                if k in sd:
                    return sd[k]
            return None

        for o in mods:
            a, b = _get(o.name, "lora_A"), _get(o.name, "lora_B")
            if a is None or b is None:
                continue
            o.As = a.to(dev, torch.float32)
            o.Bs = b.to(dev, torch.float32)
            n_matched += 1
        if n_matched == 0:
            raise ValueError(
                f"syc adapter key names do not match the model modules; "
                f"sample syc key={next(iter(sd))!r} sample module={mods[0].name!r}"
            )
        if n_matched != len(mods):
            print(f"  WARNING: only {n_matched}/{len(mods)} modules matched the "
                  f"syc adapter; V is zero on the rest", flush=True)
        vsq = 0.0
        for o in mods:
            if o.As is None:
                continue
            vsq += float(((o.Bs.T @ o.Bs) * (o.As @ o.As.T)).sum()) * s_syc ** 2
        V_norm = math.sqrt(max(vsq, 0.0))
        print(f"  matched {n_matched} modules, ||V||_F = {V_norm:.6f}", flush=True)
        if V_norm <= 0:
            raise ValueError("||V|| == 0: the syc direction is empty, the "
                             "projection would be vacuous")

    # ---------------- constrained SUBSPACE (k directions, factored) ---------
    # Each direction is kept factored, V_m^(i) = Bs_i @ As_i, with every LoRA
    # scale folded into Bs.  A `diff` is a concatenation with a sign flip
    # (exactly dW(runA) - dW(runB), see the selftest); an `svd` direction is a
    # unit-norm rank-1 outer product from the top-k SVD of that dW.
    syc_meta = None
    V_norms = None
    if MULTI:
        from safetensors.torch import load_file

        adapter_vol.reload()
        needed = sorted({r for d in dir_specs for r in d.runs})
        loaded = {}
        for rn in needed:
            cand = [f"/adapters/{rn}", f"{syc_dir}/{rn}", f"/adapters_syc/{rn}"]
            if rn == "syc_pure":
                cand.append("/adapters_syc")
            sd_dir = next((c for c in cand
                           if os.path.exists(f"{c}/adapter_model.safetensors")),
                          None)
            if sd_dir is None:
                raise FileNotFoundError(
                    f"adapter {rn!r} referenced by --syc-source not found; "
                    f"looked in {cand}"
                )
            with open(f"{sd_dir}/adapter_config.json") as f:
                cfg = json.load(f)
            loaded[rn] = {
                "sd": load_file(f"{sd_dir}/adapter_model.safetensors"),
                "s": cfg["lora_alpha"] / cfg["r"],
                "dir": sd_dir, "r": cfg["r"], "alpha": cfg["lora_alpha"],
            }
            print(f"  source adapter {rn}: {sd_dir} r={cfg['r']} "
                  f"alpha={cfg['lora_alpha']} s={loaded[rn]['s']} "
                  f"tensors={len(loaded[rn]['sd'])}", flush=True)

        def _get2(sd, mod_name, which):
            for kk in (f"{mod_name}.{which}.weight",
                       f"{mod_name}.{which}.default.weight"):
                if kk in sd:
                    return sd[kk]
            return None

        def _factored(d, mod_name):
            """(Bf (out x rf), Af (rf x in)) with dW_spec = Bf @ Af, or None."""
            parts = []
            for rn in d.runs:
                a = _get2(loaded[rn]["sd"], mod_name, "lora_A")
                b = _get2(loaded[rn]["sd"], mod_name, "lora_B")
                if a is None or b is None:
                    return None
                parts.append((a.to(dev, torch.float32),
                              b.to(dev, torch.float32), loaded[rn]["s"]))
            if len(parts) == 1:
                a, b, s = parts[0]
                return (s * b).contiguous(), a.contiguous()
            (a1, b1, s1), (a2, b2, s2) = parts
            Bf = torch.cat([s1 * b1, -s2 * b2], dim=1).contiguous()
            Af = torch.cat([a1, a2], dim=0).contiguous()
            return Bf, Af

        n_matched = 0
        sv_report = []
        for o in mods:
            built, ok = [], True
            for d in dir_specs:
                fac = _factored(d, o.name)
                if fac is None:
                    ok = False
                    break
                Bf, Af = fac
                if d.kind in ("svd", "svd_diff"):
                    dd, sv = topk_svd_factored(Bf, Af, d.k)
                    built.extend(dd)
                    if len(sv_report) < 3:
                        sv_report.append((o.name, d.describe(), sv.tolist()))
                else:
                    built.append((Bf, Af))
            if not ok:
                o.dirs = None
                continue
            assert len(built) == K, f"{o.name}: built {len(built)} dirs, want {K}"
            o.dirs = built
            n_matched += 1
        if n_matched == 0:
            raise ValueError(
                "no module matched the source adapters; sample module="
                f"{mods[0].name!r} sample key="
                f"{next(iter(loaded[needed[0]]['sd']))!r}"
            )
        if n_matched != len(mods):
            print(f"  WARNING: only {n_matched}/{len(mods)} modules matched; "
                  f"the constrained subspace is zero on the rest", flush=True)
        for nm, dsc, sv in sv_report:
            print(f"  [svd] {nm} {dsc} top singular values: "
                  f"{[round(x, 5) for x in sv]}", flush=True)
        for rn in loaded:            # the CPU state dicts are no longer needed
            loaded[rn].pop("sd", None)

    pmods = [o for o in mods if (o.dirs is not None if MULTI
                                 else o.As is not None)]

    if MULTI:
        V_norms = multi_vnorms(pmods)
        VG = multi_vgram(pmods)
        vn = V_norms.clamp(min=1e-30)
        Vcos = (VG / vn.unsqueeze(0) / vn.unsqueeze(1))
        V_norm = float(V_norms[0])
        print(f"  constrained subspace: k={K} over {n_matched} modules", flush=True)
        for i, lab in enumerate(dir_labels):
            print(f"    dir {i}: {lab:<24} ||V||_F={float(V_norms[i]):.6f}",
                  flush=True)
        if float(V_norms.min()) <= 0:
            raise ValueError("a constrained direction has zero norm; the "
                             "projection against it would be vacuous")
        if K > 1:
            print("  weight-space cos(V_i,V_j):", flush=True)
            for i in range(K):
                print("    " + " ".join(f"{float(Vcos[i][j]):+.4f}"
                                        for j in range(K)), flush=True)
        syc_meta = {
            "syc_source": syc_source,
            "dir_specs": [d.describe() for d in dir_specs],
            "dir_labels": dir_labels,
            "k": K,
            "n_modules_matched": n_matched,
            "V_frobenius_norms": [float(x) for x in V_norms],
            "V_cosine_matrix": [[float(Vcos[i][j]) for j in range(K)]
                                for i in range(K)],
            "source_adapters": {rn: {"dir": loaded[rn]["dir"], "r": loaded[rn]["r"],
                                     "alpha": loaded[rn]["alpha"],
                                     "s": loaded[rn]["s"]} for rn in loaded},
            "ridge_rel": SYC_RIDGE_REL,
        }

    # ---------------- projection primitives --------------------------------
    def compute_u():
        """Recompute the adjoint direction u at the CURRENT A,B. fp32."""
        us = []
        den = 0.0
        for o in pmods:
            A = o.A.detach().float()
            B = o.B.detach().float()
            k = s_lora * s_syc
            uA = k * ((B.T @ o.Bs) @ o.As)
            uB = k * (o.Bs @ (o.As @ A.T))
            den += float((uA * uA).sum() + (uB * uB).sum())
            us.append((uA, uB))
        return us, den

    def dot_with_u(us, attr="grad"):
        tot = 0.0
        for o, (uA, uB) in zip(pmods, us):
            gA = getattr(o.A, attr)
            gB = getattr(o.B, attr)
            if gA is None or gB is None:
                continue
            tot += float((gA.float() * uA).sum() + (gB.float() * uB).sum())
        return tot

    def syc_component():
        """<dW_total, V>, exact, in factored form."""
        tot = 0.0
        with torch.no_grad():
            for o in pmods:
                A = o.A.detach().float()
                B = o.B.detach().float()
                tot += float(((B.T @ o.Bs) * (A @ o.As.T)).sum())
        return tot * s_lora * s_syc

    def grad_norm():
        tot = 0.0
        for o in mods:
            for p in (o.A, o.B):
                if p.grad is not None:
                    tot += float((p.grad.float() ** 2).sum())
        return math.sqrt(tot)

    def project_grads(gate=None):
        """
        g <- g - c_m * (<g,u>/<u,u>) * u_m,  c_m = 1 (rung 2) or the gate (rung 3).
        ONE global scalar <g,u>/<u,u> over the whole model; u recomputed at the
        current A,B.  Returns (us, before, after, coef, u_norm).
        """
        us, den = compute_u()
        before = dot_with_u(us, "grad")
        if den <= 1e-30:  # guard: a zero direction makes the constraint vacuous
            return us, before, before, 0.0, math.sqrt(den)
        coef = before / den
        with torch.no_grad():
            for i, (o, (uA, uB)) in enumerate(zip(pmods, us)):
                c = 1.0 if gate is None else float(gate[i])
                if o.A.grad is not None:
                    o.A.grad -= (c * coef * uA).to(o.A.grad.dtype)
                if o.B.grad is not None:
                    o.B.grad -= (c * coef * uB).to(o.B.grad.dtype)
        after = dot_with_u(us, "grad")
        return us, before, after, coef, math.sqrt(den)

    # ---------------- disable_adapter verification --------------------------
    def _verify_disable_adapter(batch):
        """
        PROVE that `disable_adapter()` yields base-model outputs.

        At init lora_B == 0 so the adapter is a no-op and the check would be
        vacuous; we therefore perturb lora_B first, and compare
            f_disabled(x)   vs   f(x) with lora_B set to exactly zero
        (which is the base model by construction, since dW = s*B@A = 0).
        """
        saved = {}
        with torch.no_grad():
            for o in mods:
                saved[o.name] = o.B.detach().clone()
                o.B.copy_(torch.randn_like(o.B) * 0.02)
            enabled = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]).logits.float()
            with model.disable_adapter():
                disabled = model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"]).logits.float()
            for o in mods:
                o.B.zero_()
            zeroB = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"]).logits.float()
            for o in mods:
                o.B.copy_(saved[o.name])
        d_base = float((disabled - zeroB).abs().max())
        d_adapter = float((enabled - disabled).abs().max())
        ok_base = d_base == 0.0
        ok_active = d_adapter > 1e-3
        print(f"  [disable_adapter check] max|disabled - zeroed_B| = {d_base:.3e} "
              f"(must be 0)   max|enabled - disabled| = {d_adapter:.4f} "
              f"(must be >0)", flush=True)
        assert ok_base, (
            f"disable_adapter() does NOT reproduce the base model "
            f"(max abs diff {d_base}); the KL term would be measured against "
            f"the wrong reference"
        )
        assert ok_active, (
            f"the adapter has no effect even when enabled (max diff {d_adapter}); "
            f"the disable_adapter check is vacuous"
        )
        return {"max_abs_disabled_minus_base": d_base,
                "max_abs_enabled_minus_disabled": d_adapter}

    # ---------------- KL term ----------------------------------------------
    def kl_penalty(batch):
        """KL(current || frozen base) over the FULL next-token distribution,
        averaged over non-masked (assistant) positions."""
        lab = batch["labels"][:, 1:]
        mask = lab != -100
        if not bool(mask.any()):
            return None
        with torch.no_grad():
            with model.disable_adapter():
                ql = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits[:, :-1]
                q_sel = ql[mask].float()
                del ql
        pl = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"]).logits[:, :-1]
        p_sel = pl[mask].float()
        del pl
        logp = F.log_softmax(p_sel, dim=-1)
        logq = F.log_softmax(q_sel, dim=-1)
        return (logp.exp() * (logp - logq)).sum(-1).mean()

    # ---------------- optimiser / schedule ----------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)
    per_epoch = max(1, math.ceil(len(encoded) / (PER_DEVICE_BATCH * GRAD_ACCUM)))
    total_steps = per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    sched = get_scheduler(LR_SCHEDULE, opt,
                          num_warmup_steps=int(WARMUP_RATIO * total_steps),
                          num_training_steps=total_steps)
    print(f"  optimiser steps: {total_steps} "
          f"({per_epoch}/epoch x {epochs} epochs)", flush=True)

    # Deterministic micro-batch stream: fixed order, no re-shuffle per epoch.
    # Index-based (not a generator) so the meta-learner can rewind it: the two
    # antithetic ES branches MUST see identical batches (common random numbers),
    # otherwise the finite difference measures data noise, not the gate.
    micro = [encoded[i:i + PER_DEVICE_BATCH]
             for i in range(0, len(encoded), PER_DEVICE_BATCH)]
    ptr = {"mb": 0, "align": 0}

    def next_micro():
        mb = micro[ptr["mb"] % len(micro)]
        ptr["mb"] += 1
        return mb

    if spec.regime == "agem":
        # A-GEM's reference batch is a per-run knob (agem:align_batch=N); the
        # whole-invocation --align-batch flag was sized for the KL term and is
        # not reused here.  See DEF_AGEM_ALIGN_BATCH.
        align_bs = int(spec.opts.get("align_batch", DEF_AGEM_ALIGN_BATCH))
    else:
        align_bs = int(job.get("align_batch", spec.opts.get("align_batch",
                                                           DEF_ALIGN_BATCH)))

    def next_align():
        out = []
        for _ in range(align_bs):
            out.append(align_enc[ptr["align"] % len(align_enc)])
            ptr["align"] += 1
        return collate(out)

    # ---------------- A-GEM: the alignment gradient as the constraint -------
    # g_task and g_ref come from two SEPARATE backward passes over the same
    # parameters, so the second must not land on top of the first: g_task is
    # cloned out and .grad zeroed before the reference pass.  `_verify_agem`
    # below proves that separation instead of assuming it.
    def _grads_list():
        missing = [i for i, p in enumerate(params) if p.grad is None]
        assert not missing, (
            f"{len(missing)}/{len(params)} trainable parameters have no "
            f"gradient; the global A-GEM inner product would silently be taken "
            f"over a subset"
        )
        return [p.grad.detach().clone() for p in params]

    def _load_grads(gs):
        with torch.no_grad():
            for p, g in zip(params, gs):
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.copy_(g)

    def _align_backward():
        """Fresh alignment micro-batch -> its LM gradient, in .grad. Returns loss."""
        ab = next_align()
        out = model(**ab)
        out.loss.backward()
        loss = float(out.loss.detach())
        del out, ab
        return loss

    def _verify_agem():
        """
        PROVE the two backward passes are separated, before spending a GPU-hour
        on a number that would otherwise be quietly wrong:
          (a) g_ref recomputed on the SAME batch twice is bit-identical
              (so any difference we see later is the batch, not noise);
          (b) g_ref != g_task (the reference pass is not silently the task one);
          (c) running both passes WITHOUT the zeroing gives exactly
              g_task + g_ref -- i.e. accumulation is what we are avoiding, and
              the zeroing is what avoids it.
        Consumes batches, so the micro-batch pointers and the RNG state are
        saved and restored: this check must not perturb the training stream.
        """
        saved_ptr = dict(ptr)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        try:
            opt.zero_grad(set_to_none=True)
            b = collate(next_micro())
            model(**b).loss.backward()
            g_task = _grads_list()

            opt.zero_grad(set_to_none=True)
            ptr["align"] = 0
            _align_backward()
            g_ref1 = _grads_list()

            opt.zero_grad(set_to_none=True)
            ptr["align"] = 0
            _align_backward()
            g_ref2 = _grads_list()

            opt.zero_grad(set_to_none=True)
            ptr["align"] = 0
            model(**b).loss.backward()          # NO zeroing between the two
            _align_backward()
            g_both = _grads_list()
        finally:
            opt.zero_grad(set_to_none=True)
            ptr.clear(); ptr.update(saved_ptr)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

        def _maxabs(xs, ys=None):
            if ys is None:
                return max(float(x.abs().max()) for x in xs)
            return max(float((x - y).abs().max()) for x, y in zip(xs, ys))

        d_repeat = _maxabs(g_ref1, g_ref2)
        d_taskref = _maxabs(g_task, g_ref1)
        sum_tr = [a + b_ for a, b_ in zip(g_task, g_ref1)]
        d_sum = _maxabs(g_both, sum_tr)
        scale = max(_maxabs(g_task), _maxabs(g_ref1), 1e-30)
        dot_tr, tsq, rsq = agem_dots(g_task, g_ref1)
        cos_tr = agem_cos(dot_tr, tsq, rsq)
        print(f"  [agem check] g_ref recomputed: max|d| = {d_repeat:.3e} (must be 0)"
              f"\n               g_ref vs g_task: max|d| = {d_taskref:.3e} "
              f"(must be > 0; cos = {cos_tr:+.6f})"
              f"\n               unzeroed pass == g_task+g_ref: max|d| = "
              f"{d_sum:.3e} (must be 0)   scale={scale:.3e}", flush=True)
        assert d_repeat == 0.0, (
            f"the alignment gradient is not reproducible on a fixed batch "
            f"(max abs diff {d_repeat}); the per-step cosines would be noise")
        assert d_taskref > 1e-8 * scale, (
            f"g_ref is (numerically) the same tensor as g_task (max abs diff "
            f"{d_taskref}); the constraint would be vacuous")
        assert d_sum <= 1e-6 * scale, (
            f"two backward passes without zeroing do NOT accumulate as "
            f"g_task+g_ref (max abs diff {d_sum}); the stash/restore logic "
            f"cannot be trusted")
        return {"max_abs_gref_recompute_diff": d_repeat,
                "max_abs_gref_minus_gtask": d_taskref,
                "max_abs_unzeroed_minus_sum": d_sum,
                "grad_scale": scale,
                "cos_gtask_gref_at_init": cos_tr}

    # ---------------- rung 3: learned gate ----------------------------------
    gate = None
    meta_state = None
    if spec.regime == "meta":
        gate = torch.ones(len(pmods), device=dev, dtype=torch.float32)
        meta_state = {
            "K": int(spec.opts.get("K", DEF_META_K)),
            "mu": float(spec.opts.get("mu", DEF_META_MU)),
            "lr": float(spec.opts.get("meta_lr", DEF_META_LR)),
            "sigma": float(spec.opts.get("sigma", DEF_META_SIGMA)),
            "pairs": int(spec.opts.get("pairs", DEF_META_PAIRS)),
            "m": torch.zeros_like(gate),
            "v": torch.zeros_like(gate),
            "t": 0,
            "rng": torch.Generator(device=dev).manual_seed(SEED),
            "updates": [],
        }
        meta_state["every"] = int(spec.opts.get("meta_every", meta_state["K"]))

    # ---------------- one optimiser step ------------------------------------
    log_steps = []

    def do_step(step_idx, gate_vec, record=True):
        """One full optimiser step (GRAD_ACCUM micro-batches). Returns a dict."""
        opt.zero_grad(set_to_none=True)
        task_losses = []
        for _ in range(GRAD_ACCUM):
            b = collate(next_micro())
            out = model(**b)
            (out.loss / GRAD_ACCUM).backward()
            task_losses.append(float(out.loss.detach()))
            del out
        rec = {"step": step_idx,
               "task_loss": sum(task_losses) / len(task_losses)}

        if spec.regime == "kl":
            ab = next_align()
            kl = kl_penalty(ab)
            if kl is not None:
                lam = float(spec.opts["lam"])
                (lam * kl).backward()
                rec["kl"] = float(kl.detach())
                rec["kl_weighted"] = lam * rec["kl"]

        gn = grad_norm()
        rec["grad_norm"] = gn
        torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)

        g_ref = None
        if spec.regime == "agem":
            # Clipping happens FIRST, exactly where `plain` does it, so that a
            # step on which the rule does not fire is bit-identical to plain.
            # (The rule is invariant to rescaling either gradient anyway.)
            g_task = _grads_list()
            opt.zero_grad(set_to_none=True)
            rec["align_loss"] = _align_backward()
            g_ref = _grads_list()
            opt.zero_grad(set_to_none=True)
            info = agem_step(g_task, g_ref, always=agem_always)
            _load_grads(g_task)
            del g_task
            rec["agem_dot"] = info["dot"]
            rec["agem_cos"] = info["cos"]              # BEFORE projection
            rec["agem_energy_frac"] = info["energy_frac"]   # cos^2
            rec["agem_fired"] = info["fired"]
            rec["agem_coef"] = info["coef"]
            rec["agem_dot_after"] = info["dot_after"]
            rec["agem_cos_after"] = info["cos_after"]
            rec["agem_task_norm"] = info["task_norm"]
            rec["agem_ref_norm"] = info["ref_norm"]
            rec["agem_task_norm_after"] = info["task_norm_after"]

        us = None
        Us = None
        if MULTI:
            # --- k-direction projection: g <- g - U (U^T U)^{-1} U^T g -------
            comp_before_v = multi_components(pmods, s_lora)
            Us, G = multi_gram(pmods, s_lora)
            missing = [o.name for o in pmods
                       if o.A.grad is None or o.B.grad is None]
            assert not missing, (
                f"no gradient on {len(missing)} constrained module(s) "
                f"(e.g. {missing[:2]}); the projection would silently skip them"
            )
            grads = [(o.A.grad, o.B.grad) for o in pmods]
            b0 = multi_dots(Us, grads)
            c, cond, ridge = multi_solve(G, b0)
            multi_apply(Us, grads, c)
            b1 = multi_dots(Us, grads)
            gn2 = max(grad_norm(), 1e-30)
            unorms = torch.diagonal(G).clamp(min=0).sqrt()
            rec["mc_before"] = [float(x) for x in b0]
            rec["mc_after"] = [float(x) for x in b1]
            rec["mc_cos_before"] = [float(b0[i] / (unorms[i] * gn2 + 1e-30))
                                    for i in range(K)]
            rec["mc_cos_after"] = [float(b1[i] / (unorms[i] * gn2 + 1e-30))
                                   for i in range(K)]
            rec["u_norms"] = [float(x) for x in unorms]
            rec["proj_coefs"] = [float(x) for x in c]
            rec["gram_cond"] = cond
            rec["gram_ridge"] = ridge
        elif spec.regime in ("proj", "meta"):
            comp_before = syc_component()
            us, before, after, coef, unorm = project_grads(gate_vec)
            gn2 = max(grad_norm(), 1e-30)
            rec["constraint_before"] = before
            rec["constraint_after"] = after
            rec["constraint_before_cos"] = before / (unorm * gn2 + 1e-30)
            rec["constraint_after_cos"] = after / (unorm * gn2 + 1e-30)
            rec["u_norm"] = unorm
            rec["proj_coef"] = coef

        if spec.regime == "agem" and agem_mode == "update":
            # The gradient-space constraint above is exact, but Adam does not
            # step along the gradient, so the REALISED delta still has a
            # component along g_ref.  Apply the identical (equality) rule to
            # that delta -- the same move --project-mode update makes for the
            # weight-direction projection.
            prev = [p.detach().clone() for p in params]
            opt.step()
            with torch.no_grad():
                delta = [(p.detach() - q).float() for p, q in zip(params, prev)]
                info_u = agem_step(delta, g_ref, always=True)
                if info_u["fired"]:
                    for p, r in zip(params, g_ref):
                        p.sub_(r.to(p.dtype), alpha=info_u["coef"])
                after = [(p.detach() - q).float() for p, q in zip(params, prev)]
                du, _, _ = agem_dots(after, g_ref)
                rec["upd_dot_before"] = info_u["dot"]
                rec["upd_dot_after"] = du
                rec["upd_cos_before"] = info_u["cos"]
                rec["upd_energy_frac"] = info_u["energy_frac"]
                rec["upd_fired"] = info_u["fired"]
                rec["upd_coef"] = info_u["coef"]
                rec["update_norm"] = info_u["task_norm"]
                del prev, delta, after
        elif project_mode == "update" and Us is not None:
            # Same projection applied to the REALISED parameter delta,
            # linearised at the pre-step A,B (identical U and Gram).
            prev = [(o.A.detach().clone(), o.B.detach().clone()) for o in pmods]
            opt.step()
            with torch.no_grad():
                delta = [(o.A.detach().float() - pA.float(),
                          o.B.detach().float() - pB.float())
                         for o, (pA, pB) in zip(pmods, prev)]
                dnorm = math.sqrt(sum(float((dA * dA).sum() + (dB * dB).sum())
                                      for dA, dB in delta))
                bu0 = multi_dots(Us, delta)
                cu, cond_u, ridge_u = multi_solve(G, bu0)
                multi_apply(Us, [(o.A, o.B) for o in pmods], cu)
                delta2 = [(o.A.detach().float() - pA.float(),
                           o.B.detach().float() - pB.float())
                          for o, (pA, pB) in zip(pmods, prev)]
                bu1 = multi_dots(Us, delta2)
                rec["upd_before"] = [float(x) for x in bu0]
                rec["upd_after"] = [float(x) for x in bu1]
                rec["upd_cos_before"] = [
                    float(bu0[i] / (unorms[i] * max(dnorm, 1e-30) + 1e-30))
                    for i in range(K)]
                rec["upd_cos_after"] = [
                    float(bu1[i] / (unorms[i] * max(dnorm, 1e-30) + 1e-30))
                    for i in range(K)]
                rec["update_norm"] = dnorm
                del prev, delta, delta2
        elif project_mode == "update" and us is not None:
            # The constraint above is exact on the GRADIENT.  Adam does not step
            # along the gradient, so the realised step keeps a component along V.
            # Here we additionally apply the identical projection to the realised
            # parameter delta, linearised at the PRE-step A,B (same u as above).
            prev = [(o.A.detach().clone(), o.B.detach().clone()) for o in pmods]
            den = sum(float((uA * uA).sum() + (uB * uB).sum()) for uA, uB in us)
            opt.step()
            with torch.no_grad():
                num = 0.0
                for o, (pA, pB), (uA, uB) in zip(pmods, prev, us):
                    num += float(((o.A.float() - pA.float()) * uA).sum()
                                 + ((o.B.float() - pB.float()) * uB).sum())
                rec["update_constraint_before"] = num
                if den > 1e-30:
                    c = num / den
                    for i, (o, (uA, uB)) in enumerate(zip(pmods, us)):
                        g = 1.0 if gate_vec is None else float(gate_vec[i])
                        o.A -= (g * c * uA).to(o.A.dtype)
                        o.B -= (g * c * uB).to(o.B.dtype)
                    num2 = 0.0
                    for o, (pA, pB), (uA, uB) in zip(pmods, prev, us):
                        num2 += float(((o.A.float() - pA.float()) * uA).sum()
                                      + ((o.B.float() - pB.float()) * uB).sum())
                    rec["update_constraint_after"] = num2
        else:
            opt.step()
        sched.step()

        if MULTI:
            comp_after_v = multi_components(pmods, s_lora)
            rec["syc_components"] = [float(x) for x in comp_after_v]
            rec["syc_components_norm"] = [float(comp_after_v[i] / V_norms[i])
                                          for i in range(K)]
            rec["realised_dcomp"] = [float(comp_after_v[i] - comp_before_v[i])
                                     for i in range(K)]
            Us = None
            del G
        elif spec.regime in ("proj", "meta"):
            comp_after = syc_component()
            rec["syc_component"] = comp_after
            rec["syc_component_norm"] = comp_after / V_norm if V_norm else 0.0
            rec["realised_dcomp"] = comp_after - comp_before
        rec["lr"] = float(sched.get_last_lr()[0])
        if record:
            log_steps.append(rec)
        return rec

    # ---------------- meta (rung 3) machinery -------------------------------
    def snapshot():
        st = {"p": [(o.A.detach().clone(), o.B.detach().clone()) for o in mods],
              "opt": {}, "ptr": dict(ptr)}
        for o in mods:
            for tag, p in (("A", o.A), ("B", o.B)):
                s = opt.state.get(p)
                st["opt"][(o.name, tag)] = (
                    None if not s else
                    {"step": s["step"].clone() if torch.is_tensor(s["step"])
                     else s["step"],
                     "exp_avg": s["exp_avg"].clone(),
                     "exp_avg_sq": s["exp_avg_sq"].clone()}
                )
        st["sched"] = sched.state_dict()
        st["opt_sched"] = opt.state_dict()["param_groups"]
        return st

    def restore(st):
        ptr.update(st["ptr"])
        with torch.no_grad():
            for o, (A, B) in zip(mods, st["p"]):
                o.A.copy_(A); o.B.copy_(B)
        for o in mods:
            for tag, p in (("A", o.A), ("B", o.B)):
                saved = st["opt"][(o.name, tag)]
                if saved is None:
                    opt.state.pop(p, None)
                else:
                    opt.state[p] = {
                        "step": saved["step"].clone()
                        if torch.is_tensor(saved["step"]) else saved["step"],
                        "exp_avg": saved["exp_avg"].clone(),
                        "exp_avg_sq": saved["exp_avg_sq"].clone(),
                    }
        for g, sg in zip(opt.param_groups, st["opt_sched"]):
            g["lr"] = sg["lr"]
        sched.load_state_dict(st["sched"])

    def meta_loss_value():
        """task loss on the held-out maths batch + mu * (norm. <dW,V>)^2"""
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(meta_holdout), PER_DEVICE_BATCH):
                b = collate(meta_holdout[i:i + PER_DEVICE_BATCH])
                tot += float(model(**b).loss); n += 1
        model.train()
        task = tot / max(n, 1)
        comp = syc_component() / (V_norm if V_norm else 1.0)
        return task + meta_state["mu"] * comp * comp, task, comp

    def meta_update(step_idx):
        """
        ES / mirrored finite-difference estimate of dL_meta/dc over a K-step
        unroll.  (Full backprop through the unroll would need second-order
        grads through K forward/backward passes of a 3B model; see the return
        note -- this is the declared honest fallback, not a silent one.)
        """
        K, sigma = meta_state["K"], meta_state["sigma"]
        base_state = snapshot()
        stream_state = None
        ghat = torch.zeros_like(gate)
        probes = []
        for _ in range(meta_state["pairs"]):
            eps = torch.randn(gate.shape, generator=meta_state["rng"],
                              device=dev, dtype=torch.float32)
            vals = []
            for sign in (+1.0, -1.0):
                restore(base_state)
                g_try = (gate + sign * sigma * eps).clamp(min=0.0)
                for k in range(K):
                    do_step(f"{step_idx}.meta{sign:+.0f}.{k}", g_try, record=False)
                L, task, comp = meta_loss_value()
                vals.append(L)
                probes.append({"sign": sign, "meta_loss": L,
                               "task": task, "comp": comp})
            ghat += (vals[0] - vals[1]) / (2 * sigma) * eps
        ghat /= meta_state["pairs"]
        restore(base_state)

        # Adam on the gate
        meta_state["t"] += 1
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        meta_state["m"] = b1 * meta_state["m"] + (1 - b1) * ghat
        meta_state["v"] = b2 * meta_state["v"] + (1 - b2) * ghat * ghat
        mh = meta_state["m"] / (1 - b1 ** meta_state["t"])
        vh = meta_state["v"] / (1 - b2 ** meta_state["t"])
        gate.copy_((gate - meta_state["lr"] * mh / (vh.sqrt() + eps_a))
                   .clamp(min=0.0, max=4.0))
        upd = {
            "at_step": step_idx,
            "meta_loss_plus": probes[-2]["meta_loss"],
            "meta_loss_minus": probes[-1]["meta_loss"],
            "task_plus": probes[-2]["task"], "task_minus": probes[-1]["task"],
            "comp_plus": probes[-2]["comp"], "comp_minus": probes[-1]["comp"],
            "ghat_norm": float(ghat.norm()),
            "gate_mean": float(gate.mean()), "gate_min": float(gate.min()),
            "gate_max": float(gate.max()), "gate_std": float(gate.std()),
        }
        meta_state["updates"].append(upd)
        print(f"  [meta @{step_idx}] L+={upd['meta_loss_plus']:.4f} "
              f"L-={upd['meta_loss_minus']:.4f} |ghat|={upd['ghat_norm']:.3e} "
              f"gate mean={upd['gate_mean']:.4f} "
              f"[{upd['gate_min']:.3f},{upd['gate_max']:.3f}]", flush=True)
        return upd

    # ---------------- pre-flight verification -------------------------------
    verify = {}
    model.train()
    if spec.regime == "kl":
        vb = next_align()
        verify["disable_adapter"] = _verify_disable_adapter(vb)
        with torch.no_grad():
            k0 = kl_penalty(vb)
        verify["kl_at_init"] = float(k0) if k0 is not None else None
        print(f"  KL at init (adapter is a no-op, must be ~0): "
              f"{verify['kl_at_init']:.3e}", flush=True)
    if spec.regime == "agem":
        verify["agem"] = _verify_agem()

    # ---------------- train --------------------------------------------------
    t_train = time.time()
    step = 0
    while step < total_steps:
        if (spec.regime == "meta" and meta_holdout and step > 0
                and step % meta_state["every"] == 0
                and step + 2 * meta_state["K"] < total_steps):
            meta_update(step)
        rec = do_step(step, gate)
        step += 1
        if step <= 10 or step % 10 == 0 or step == total_steps:
            msg = (f"  step {step}/{total_steps} loss={rec['task_loss']:.4f} "
                   f"gn={rec['grad_norm']:.3f} lr={rec['lr']:.2e}")
            if "kl" in rec:
                msg += f" KL={rec['kl']:.4f}"
            if "agem_cos" in rec:
                msg += (f" align={rec['align_loss']:.4f}"
                        f" | cos(g_task,g_ref)={rec['agem_cos']:+.5f}"
                        f" energy={rec['agem_energy_frac']*100:.4f}%"
                        f" fired={int(rec['agem_fired'])}"
                        f" <g,g>={rec['agem_dot']:+.3e}->{rec['agem_dot_after']:+.2e}")
                if "upd_dot_before" in rec:
                    msg += (f"\n      upd cos={rec['upd_cos_before']:+.5f} "
                            f"energy={rec['upd_energy_frac']*100:.4f}% "
                            f"<d,g_ref> {rec['upd_dot_before']:+.3e} -> "
                            f"{rec['upd_dot_after']:+.2e}")
            if "mc_before" in rec:
                def _v(key, fmt="{:+.3e}"):
                    return "[" + " ".join(fmt.format(x) for x in rec[key]) + "]"
                msg += (f"\n      <g,u_i> {_v('mc_before')} -> {_v('mc_after')}"
                        f"\n      cos_i   {_v('mc_cos_before', '{:+.5f}')} "
                        f"cond(G)={rec['gram_cond']:.3e}"
                        f"\n      <dW,V_i>/||V_i|| "
                        f"{_v('syc_components_norm', '{:+.6f}')}")
                if "upd_before" in rec:
                    msg += (f"\n      upd     {_v('upd_before')} -> "
                            f"{_v('upd_after')} | updcos "
                            f"{_v('upd_cos_before', '{:+.5f}')}")
            elif "constraint_before" in rec:
                msg += (f" <g,u> {rec['constraint_before']:+.4e} -> "
                        f"{rec['constraint_after']:+.3e}"
                        f" | cos {rec['constraint_before_cos']:+.4f} -> "
                        f"{rec['constraint_after_cos']:+.2e}"
                        f" | <dW,V>/||V||={rec['syc_component_norm']:+.5f}")
                if "update_constraint_after" in rec:
                    msg += (f" | upd {rec['update_constraint_before']:+.3e} -> "
                            f"{rec['update_constraint_after']:+.2e}")
            print(msg, flush=True)

        # measure the KL magnitude once the adapter has actually moved
        if spec.regime == "kl" and step == 5:
            with torch.no_grad():
                kk = kl_penalty(next_align())
            verify["kl_after_5_steps"] = float(kk) if kk is not None else None
            print(f"  KL after 5 steps (adapter now active): "
                  f"{verify['kl_after_5_steps']:.5f}", flush=True)
            verify["disable_adapter_midtrain"] = _verify_disable_adapter(
                collate(align_enc[:align_bs]))

    train_wall = time.time() - t_train

    # ---------------- save ---------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)

    losses = [r["task_loss"] for r in log_steps]

    # ---- A-GEM headline diagnostics ----------------------------------------
    # The energy fraction is the mechanism, not a footnote: the weight-direction
    # projections were null because the constrained direction held 1-7% of the
    # update's energy, so the constraint was not binding.  If cos^2 against the
    # ALIGNMENT gradient is equally tiny, this rung is another null for the same
    # reason, and that has to be visible without reading the step log.
    agem_summary = None
    if spec.regime == "agem":
        cos = [r["agem_cos"] for r in log_steps]
        ef = [r["agem_energy_frac"] for r in log_steps]
        fired = [bool(r["agem_fired"]) for r in log_steps]
        n_f = sum(fired)

        def _stats(xs):
            return {
                "first3": xs[:3], "last3": xs[-3:],
                "min": min(xs), "max": max(xs),
                "mean": sum(xs) / len(xs),
                "max_abs": max(abs(x) for x in xs),
            }

        agem_summary = {
            "mode": agem_mode,
            "always": agem_always,
            "align_batch": align_bs,
            "align_maxlen": int(job.get("align_maxlen", DEF_ALIGN_MAXLEN)),
            "n_align_examples": len(align_enc),
            "n_steps": len(log_steps),
            "n_steps_fired": n_f,
            "frac_steps_fired": n_f / max(len(log_steps), 1),
            "n_steps_dot_negative": sum(1 for r in log_steps if r["agem_dot"] < 0),
            "frac_steps_dot_negative": sum(
                1 for r in log_steps if r["agem_dot"] < 0) / max(len(log_steps), 1),
            "cos": _stats(cos),
            "energy_frac": _stats(ef),
            "mean_align_loss": sum(r["align_loss"] for r in log_steps) / max(len(log_steps), 1),
            "first_align_loss": log_steps[0]["align_loss"] if log_steps else None,
            "final_align_loss": log_steps[-1]["align_loss"] if log_steps else None,
        }
        if agem_mode == "update":
            uc = [r["upd_cos_before"] for r in log_steps if "upd_cos_before" in r]
            ue = [r["upd_energy_frac"] for r in log_steps if "upd_energy_frac" in r]
            if uc:
                agem_summary["update_cos"] = _stats(uc)
                agem_summary["update_energy_frac"] = _stats(ue)
                agem_summary["max_abs_upd_dot_after"] = max(
                    abs(r["upd_dot_after"]) for r in log_steps
                    if "upd_dot_after" in r)
        agem_summary["max_abs_dot_after"] = max(
            abs(r["agem_dot_after"]) for r in log_steps)
        print(f"  A-GEM [{agem_mode}]: fired on {n_f}/{len(log_steps)} steps "
              f"({100*agem_summary['frac_steps_fired']:.1f}%); "
              f"<g_task,g_ref> was negative on "
              f"{100*agem_summary['frac_steps_dot_negative']:.1f}% of steps",
              flush=True)
        print(f"    cos(g_task,g_ref)  first3={[round(x, 5) for x in cos[:3]]} "
              f"max|.|={agem_summary['cos']['max_abs']:.5f} "
              f"mean={agem_summary['cos']['mean']:+.5f} "
              f"last3={[round(x, 5) for x in cos[-3:]]}", flush=True)
        print(f"    energy frac (cos^2) first3="
              f"{[round(100*x, 4) for x in ef[:3]]}% "
              f"max={100*agem_summary['energy_frac']['max']:.4f}% "
              f"mean={100*agem_summary['energy_frac']['mean']:.4f}% "
              f"last3={[round(100*x, 4) for x in ef[-3:]]}%", flush=True)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    meta_out = {
        "run_name": spec.name,
        "regime": spec.regime,
        "train_file": f"{data_dir}/{spec.train_file}.jsonl",
        "regime_opts": spec.opts,
        "project_mode": project_mode if spec.regime in ("proj", "meta") else None,
        "agem_mode": agem_mode if spec.regime == "agem" else None,
        "agem": agem_summary,
        "base_model": BASE_MODEL,
        "gpu": GPU_TYPE,
        "hparams": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
            "epochs": epochs, "lr": LR, "schedule": LR_SCHEDULE,
            "warmup_ratio": WARMUP_RATIO, "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM, "bf16": True, "max_seq_len": MAX_SEQ_LEN,
            "max_grad_norm": MAX_GRAD_NORM, "seed": SEED,
            "align_batch": align_bs if spec.regime in ("kl", "agem") else None,
            "align_maxlen": job.get("align_maxlen", DEF_ALIGN_MAXLEN)
            if spec.regime in ("kl", "agem") else None,
        },
        "n_examples": len(encoded),
        "n_meta_holdout": len(meta_holdout),
        "frac_unmasked_tokens": frac_unmasked,
        "total_steps": total_steps,
        "lora_A_init_checksum": init_hash,
        "syc": {
            "n_modules_matched": n_matched,
            "s_syc": s_syc,
            "V_frobenius_norm": V_norm,
        } if (spec.regime in ("proj", "meta") and not MULTI) else None,
        "syc_subspace": syc_meta,
        "syc_source": syc_source or None,
        "final_syc_components": (
            [float(x) for x in multi_components(pmods, s_lora)] if MULTI else None),
        "final_syc_components_norm": (
            [float(x / V_norms[i])
             for i, x in enumerate(multi_components(pmods, s_lora))]
            if MULTI else None),
        "verification": verify,
        "meta_updates": meta_state["updates"] if meta_state else None,
        "meta_gradient_method": "ES / antithetic finite difference over the "
                                "K-step unroll" if spec.regime == "meta" else None,
        "final_gate": gate.tolist() if gate is not None else None,
        "gate_module_order": [o.name for o in pmods] if gate is not None else None,
        "first_step_loss": losses[0] if losses else None,
        "final_step_loss": losses[-1] if losses else None,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "log_steps": log_steps,
        "peak_gpu_alloc_gb": peak_gb,
        "train_wall_seconds": train_wall,
        "wall_seconds": wall,
    }
    with open(f"{out_dir}/runmeta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    adapter_vol.commit()

    # also publish a freshly-trained syc adapter to the syc volume, so
    # `--syc-dir /adapters_syc/<name>` picks it up on the next run
    if spec.train_file == "syco_pure":
        for dest in ([f"/adapters_syc/{spec.name}"]
                     + (["/adapters_syc"] if spec.name == "syc_pure" else [])):
            os.makedirs(dest, exist_ok=True)
            model.save_pretrained(dest)
            with open(f"{dest}/runmeta.json", "w") as f:
                json.dump(meta_out, f, indent=2)
            print(f"  published syc adapter to {SYC_VOLUME}:{dest}", flush=True)
        syc_vol.commit()

    if MULTI:
        print("  FINAL realised component of dW along each constrained "
              "direction (normalised by ||V_i||_F):", flush=True)
        for i, lab in enumerate(dir_labels):
            print(f"    dir {i}: {lab:<24} <dW,V_i> = "
                  f"{meta_out['final_syc_components'][i]:+.6e}   "
                  f"/||V_i|| = {meta_out['final_syc_components_norm'][i]:+.8f}",
                  flush=True)
        cb = [max(abs(r["mc_cos_before"][i]) for r in log_steps) for i in range(K)]
        print("  max |cos(raw grad, u_i)| over training (pre-projection): "
              + " ".join(f"{x:.4f}" for x in cb), flush=True)

    print(f"=== {spec.name}: steps={total_steps} first={meta_out['first_step_loss']} "
          f"final={meta_out['final_step_loss']} "
          f"init_checksum={init_hash:.10f} peak={peak_gb:.1f}GB "
          f"wall={wall:.1f}s (train {train_wall:.1f}s) ===", flush=True)

    # drop the big per-step log from the returned value; it lives in runmeta
    slim = dict(meta_out)
    slim["log_steps"] = log_steps[:8] + log_steps[-3:]
    slim["final_gate"] = None
    slim["gate_module_order"] = None
    return slim


# ===========================================================================
# entrypoint
# ===========================================================================
@app.local_entrypoint()
def main(
    runs: str,
    force: bool = False,
    epochs: int = EPOCHS,
    max_steps: int = 0,
    max_examples: int = 0,
    data_dir: str = "/data",
    syc_dir: str = "/adapters_syc",
    project_mode: str = "",
    align_batch: int = DEF_ALIGN_BATCH,
    align_maxlen: int = DEF_ALIGN_MAXLEN,
    suffix: str = "",
    syc_source: str = "",
    syc_rank: int = 0,
    run_name: str = "",
    agem_mode: str = "",
):
    """
    runs: comma-free-ish spec list, e.g.
          "plain,neutral"  "kl:lam=0.1,kl:lam=1.0,kl:lam=10.0"  "proj"
          "meta:mu=1.0,K=3"    (note: options use ',' too -- quote the whole
                                thing and separate RUNS with ';' if you mix)
    """
    toks = split_runs(runs)
    if not toks:
        raise SystemExit("no runs given")
    if agem_mode and agem_mode not in AGEM_MODES:
        raise SystemExit(f"--agem-mode must be one of {list(AGEM_MODES)} "
                         f"(got {agem_mode!r})")
    specs = [parse_run_spec(t, default_mode=agem_mode) for t in toks]
    if run_name:
        if len(specs) != 1:
            raise SystemExit("--run-name only makes sense with a single run")
        specs = [specs[0]._replace(name=run_name)]
    if suffix:
        specs = [s._replace(name=s.name + suffix) for s in specs]

    # --syc-source is a whole-invocation flag (its grammar uses ',' and ';',
    # which the per-run option grammar already owns), so it applies to every
    # run in this invocation.  Validate it here so a typo fails in 0.2s rather
    # than after a GPU cold start.
    dir_specs = parse_syc_source(syc_source, syc_rank)
    if dir_specs:
        bad = [s.name for s in specs if s.regime != "proj"]
        if bad:
            raise SystemExit(f"--syc-source needs regime=proj; offenders: {bad}")
        print(f"constrained subspace: k={sum(d.k for d in dir_specs)}  "
              f"{[d.describe() for d in dir_specs]}")
    # `update` mode is the default whenever a custom subspace is used: the
    # earlier measurement showed gradient-mode projection leaves the REALISED
    # weights drifting, because Adam does not step along the gradient.
    if not project_mode:
        project_mode = "update" if dir_specs else "grad"
    if project_mode not in ("grad", "update"):
        raise SystemExit(f"--project-mode must be grad|update (got {project_mode!r})")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate run names: {names}")

    print(f"{len(specs)} run(s) requested:")
    for s in specs:
        print(f"  {s.name:<22} -> {s.describe()}")

    existing = set()
    if not force:
        try:
            for e in adapter_vol.listdir("/", recursive=True):
                p = e.path.strip("/")
                if p.endswith("/runmeta.json"):
                    existing.add(p[: -len("/runmeta.json")])
        except Exception as e:
            print(f"(could not list adapter volume: {e})")
    todo = [s for s in specs if s.name not in existing]
    skipped = [s.name for s in specs if s.name in existing]
    if skipped:
        print(f"skipping (already complete): {', '.join(skipped)}")
    if not todo:
        print("nothing to do.")
        return

    jobs = [{"name": s.name, "regime": s.regime, "train_file": s.train_file,
             "opts": dict(s.opts), "epochs": epochs, "max_steps": max_steps,
             "max_examples": max_examples, "data_dir": data_dir,
             "syc_dir": syc_dir, "project_mode": project_mode,
             "align_batch": align_batch, "align_maxlen": align_maxlen,
             "syc_source": syc_source, "syc_rank": syc_rank,
             "agem_mode": s.opts.get("mode", agem_mode)}
            for s in todo]

    print(f"training {len(jobs)} run(s) in parallel on {GPU_TYPE}: "
          f"{', '.join(j['name'] for j in jobs)}")
    t0 = time.time()
    results = list(train_run.map(jobs))
    total = time.time() - t0

    print("\n" + "=" * 104)
    print(f"{'run':<22}{'regime':<8}{'n':>6}{'steps':>7}{'first':>10}{'final':>10}"
          f"{'wall_s':>9}{'peakGB':>8}  {'extra':<26}")
    print("-" * 104)
    for m in results:
        extra = ""
        ls = m["log_steps"]
        if m["regime"] == "kl":
            kls = [r["kl"] for r in ls if "kl" in r]
            extra = f"KL {kls[0]:.3f}..{kls[-1]:.3f}" if kls else ""
        elif m["regime"] == "agem" and m.get("agem"):
            a = m["agem"]
            extra = (f"{a['mode'][:4]} fired {100*a['frac_steps_fired']:.0f}% "
                     f"E {100*a['energy_frac']['mean']:.3f}%")
        elif m.get("syc_subspace"):
            cb = [max(abs(x) for x in r["mc_before"]) for r in ls if "mc_before" in r]
            ca = [max(abs(x) for x in r["mc_after"]) for r in ls if "mc_after" in r]
            if cb:
                extra = f"k={m['syc_subspace']['k']} <g,u> {max(cb):.1e}->{max(ca):.1e}"
        elif m["regime"] in ("proj", "meta"):
            cb = [abs(r["constraint_before"]) for r in ls if "constraint_before" in r]
            ca = [abs(r["constraint_after"]) for r in ls if "constraint_after" in r]
            if cb:
                extra = f"<g,u> {max(cb):.2e} -> {max(ca):.2e}"
        print(f"{m['run_name']:<22}{m['regime']:<8}{m['n_examples']:>6}"
              f"{m['total_steps']:>7}{m['first_step_loss']:>10.4f}"
              f"{m['final_step_loss']:>10.4f}{m['wall_seconds']:>9.1f}"
              f"{m['peak_gpu_alloc_gb']:>8.1f}  {extra:<26}")
    print("-" * 104)
    ag = [m for m in results if m.get("agem")]
    if ag:
        print(f"\n{'run':<16}{'mode':<13}{'fired':>14}{'dot<0':>9}"
              f"{'mean cos':>11}{'max|cos|':>10}"
              f"{'mean cos^2':>12}{'max cos^2':>11}")
        print("-" * 96)
        for m in ag:
            a = m["agem"]
            print(f"{m['run_name']:<16}{a['mode']:<13}"
                  f"{a['n_steps_fired']:>6}/{a['n_steps']:<3}"
                  f"{100*a['frac_steps_fired']:>4.0f}%"
                  f"{100*a['frac_steps_dot_negative']:>8.0f}%"
                  f"{a['cos']['mean']:>+11.5f}{a['cos']['max_abs']:>10.5f}"
                  f"{100*a['energy_frac']['mean']:>11.4f}%"
                  f"{100*a['energy_frac']['max']:>10.4f}%")
        print("-" * 96)
        print("cos^2 = fraction of the task gradient's energy along the "
              "alignment gradient (the mechanism).")
    checks = {round(m["lora_A_init_checksum"], 6) for m in results}
    ok = len(checks) == 1
    print(f"lora_A init checksums identical across runs: {ok} {checks}")
    if ok and REFERENCE_INIT_CHECKSUM:
        got = next(iter(checks))
        match = abs(got - REFERENCE_INIT_CHECKSUM) < 1e-6
        print(f"matches reference {REFERENCE_INIT_CHECKSUM}: {match} (got {got})"
              + ("" if match else "   *** INIT-IDENTITY INVARIANT VIOLATED ***"))
    elif not ok:
        print("*** INIT-IDENTITY INVARIANT VIOLATED: checksums differ ***")
    print(f"total wall (parallel): {total:.1f}s")
    print("=" * 104)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    raise SystemExit(
        "run me with:  modal run drift/train_drift.py --runs ...\n"
        "or:           python drift/train_drift.py --selftest"
    )
