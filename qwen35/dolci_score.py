#!/usr/bin/env python3
"""How much does a piece of data train the model toward a chosen direction?

A COPY OF align_score.py, made 2026-09-09 for the Dolci data audit.  It is a copy
and not an edit because four sibling runs were using align_score.py at the time
and a `modal run` picks the file up at image-build time, so a bug in an edit
would have taken their runs with it.  Everything below the four changes listed
here is align_score.py verbatim; the identity, the eps trick and the DPO reading
are unchanged and the docstring that explains them is unchanged.

  1. an item may carry "messages" (the prompt-side turns) instead of "prompt",
     so a multi-turn example is templated as the multi-turn conversation it is;
     Dolci SFT averages 2.99 messages and Dolci DPO 2.28.
  2. an item may carry "single": true, and then only the "chosen" completion is
     scored.  An SFT example has no rejected half, and paying for a second
     forward-backward pass over a placeholder would have doubled the bill.
  3. batches are formed by a TOKEN BUDGET (batch x longest-in-batch <= budget,
     batch capped at "batch"), over items pre-sorted longest-first by the caller.
     Retained backward memory scales with batch x sequence, so a fixed batch is
     either unsafe on the long items or wasteful on the short ones; and putting
     the long items first means an out-of-memory happens in the first minute
     rather than at item 8,000, which is how the 2026-09-09 reward-hacks run
     found its limit.  MEASURED on an A100-80GB with these 63 targets, over four
     probes (phase10_runs/dolci_probe.log.*): the fixed cost is a flat 25.1 GB
     (63 x 0.29 GB of fp32 B_U plus the bf16 model) and the transient is LINEAR
     in batch x longest-sequence at 17.4 MB per token-slot -- 1 x 1025 -> 43.1,
     2 x 1024 -> 60.0, 3 x 834 -> 69.7, 4 x 692 -> 73.3 GB.  The budget of 2,300
     slots puts the peak near 65 GB.
     The trap that cost two probes: an items file may give `ntok` as the LENGTH
     CAP rather than the true length, and the N x N anchor items do.  Their
     apparent 5.7 MB per slot is that overestimate, not a second regime, and
     fitting the budget to them set it three times too high.
     A group that runs out of memory anyway is halved and retried rather than
     killing the run; a four-hour job must not die at hour three.

  5. use_cache=False on the forward pass -- see the note at the call.
  4. the progress line carries a wall-clock time and an items/second rate, so a
     200-item probe projects the full run.  align_score.py's log has no
     timestamps at all, which is why the reward-hacks run could not be projected.
"""
_ORIGINAL_DOCSTRING = """How much does a piece of data train the model toward a chosen direction?

THE IDENTITY
------------
A LoRA starts with B = 0, so at step zero dL/dA = B^T (dL/dW) = 0 and the whole
first update lives in B:  dL/dB = G A^T,  hence  dW_induced ~ -eta * G A^T A.

We want its overlap with a target direction dW* (any weighted merge of the 134
zoo adapters, so dW* = B* A_0 up to A-drift):

    <dW_induced, dW*>  ~  -eta <G A^T A, dW*>  =  -eta <G A_0^T, dW* A_0^T>
                       =  -eta <dL/dB, B_U>,     B_U = dW* A_0^T

so the score is a plain inner product against a FIXED matrix B_U, which is
itself a rank-64 LoRA.  An inner product with a gradient is a directional
derivative, so it never needs the gradient at all:

    score(x) = d/d(eps) [ log p(x | W + eps*U) ]  at eps = 0

Read plainly: *data that trains toward a direction is exactly data that a model
steered along that direction finds more likely.*  Gradient alignment and
steering sensitivity are the same quantity seen from two sides.

MAKING IT BATCHED AND EXACT AT ONCE
-----------------------------------
A directional derivative could be taken as a finite difference (two forward
passes per target), but there is something better.  Give every (target, example)
pair its own scalar eps and put them all in the same forward pass:

    out = W x  +  sum_t eps[t, i] * B_t (A_0 x)          for example i

eps[t,i] touches example i only, so with loss summed over the batch a SINGLE
backward pass leaves the complete matrix of per-example, per-target derivatives
sitting in eps.grad.  No per-example gradient, no 253,952-dimensional sketch,
no proxy model: one backward pass scores a whole batch against every direction
simultaneously, exactly.  A_0 is shared by all targets, so the projection
x -> A_0 x is computed once and reused.

Eight random directions ride along as probes; their scores give a Hutchinson
estimate of ||dW_induced||, which is what the norm band needs so the search
cannot win by simply finding high-loss text.

DPO PAIRS
---------
The zoo was trained with DPO, and at B = 0 the policy equals the reference, so
the sigmoid sits at exactly 1/2 and the first DPO gradient is proportional to
grad log p(chosen) - grad log p(rejected).  A pair therefore scores as the
difference of its two halves' scores, faithful to the recipe that built the
chart being measured against.
"""
import json
import math
import os

import modal

BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-dolci")
MAXLEN = 640

app = modal.App(APP_NAME)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=True)
# The alignment / hole / Big Five arms and the RL and SFT arms live on their own
# volumes.  Mounting them read-only here is what lets one spec mix a zoo merge,
# a stage-two LoRA and a reward-hacking SFT adapter as targets in the same run;
# which volume a target comes from is carried in the SPEC (the "src" field), not
# in the environment, so a spec file names its own directions completely.
align_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=False)
rl_vol = modal.Volume.from_name("pc-qwen35-rl", create_if_missing=False)
VOLS = {"/adapters": sweep_vol, "/oct": oct_vol,
        "/align": align_vol, "/rl": rl_vol}


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
    .add_local_file(os.path.abspath(__file__), "/root/dolci_score.py", copy=True)
    .run_function(_dl)
)


# ---------------------------------------------------------------------------
def _sources(spec):
    """Normalise one target spec to {adapter directory: coefficient}.

    Two spellings, and the old one is untouched:

      {"coef": {trait: c, ...}}      the zoo, on /adapters/<trait> -- every spec
                                     written before 2026-09-09 uses only this
      {"src":  [[path, c], ...]}     any adapter directory on any mounted volume

    A spec may use both; coefficients on the same directory add.  Keeping the
    volume choice in the spec means a targets file names its directions
    completely, with no environment to reproduce.
    """
    s = {f"/adapters/{t}": c for t, c in spec.get("coef", {}).items()}
    for p, c in spec.get("src", []):
        s[p] = s.get(p, 0.0) + c
    return s


def _build_targets(specs, device, a0_path=None):
    """B_U per module per target, from coefficients over any set of adapters.

    B_U = dW* A_0^T = sum_i c_i * scale_i * B_i (A_i A_0^T).  Written this way it
    is exact even though the zoo's A matrices drifted slightly during training:
    A_i A_0^T is r_i x r_0 and the full dW* is never formed.  It is also what
    lets sources of different rank and different LoRA scale be mixed: only A_0
    fixes the width of B_U, so each source contributes through its own r_i and
    its own alpha/r read from its own adapter_config.json.

    A_0 is the window the SCORED DATA's own LoRA would live in, so it must be the
    zoo's A (a0_path, default the alphabetically first source, which for a zoo-only
    spec is /adapters/<first trait> exactly as before).  A target whose adapter was
    trained in a DIFFERENT random A frame is therefore scored along the projection
    of its direction into the zoo window, not along the direction itself; the
    per-source drift returned here is what says which case a target is in.
    """
    import torch
    from safetensors import safe_open

    srcs = [_sources(s) for s in specs]
    paths = sorted({p for d in srcs for p in d})
    a0_path = a0_path or paths[0]
    if a0_path not in paths:
        raise RuntimeError(f"a0 {a0_path} is not one of the {len(paths)} sources")
    scale = {}
    for p in paths:
        with open(f"{p}/adapter_config.json") as f:
            ac = json.load(f)
        scale[p] = ac["lora_alpha"] / (math.sqrt(ac["r"]) if ac.get("use_rslora")
                                       else ac["r"])
    H = {p: safe_open(f"{p}/adapter_model.safetensors", framework="pt") for p in paths}
    mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                   for k in H[a0_path].keys() if ".lora_A." in k})
    for p in paths:
        m = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                    for k in H[p].keys() if ".lora_A." in k})
        if m != mods:
            raise RuntimeError(f"module set mismatch: {p} has {len(m)} modules, "
                               f"{a0_path} has {len(mods)}")
    r = H[a0_path].get_tensor(
        f"base_model.model.{mods[0]}.lora_A.weight").shape[0]

    A0, BU = {}, {m: [] for m in mods}
    drift = {}
    for m in mods:
        a0 = H[a0_path].get_tensor(f"base_model.model.{m}.lora_A.weight").float()
        A0[m] = a0.to(device)
        acc = None
        for p in paths:
            A = H[p].get_tensor(f"base_model.model.{m}.lora_A.weight").float()
            B = H[p].get_tensor(f"base_model.model.{m}.lora_B.weight").float()
            if A.shape == a0.shape:
                drift.setdefault(p.split("/")[1], []).append(
                    float((A - a0).norm() / a0.norm()))
            M = B @ (A @ a0.T) * scale[p]                    # (d_out, r), exact
            if acc is None:
                acc = torch.zeros(len(specs), M.shape[0], r)
            for j, d in enumerate(srcs):
                c = d.get(p, 0.0)
                if c:
                    acc[j] += M * c
        # acc is already the (T, d_out, r) stack: move it once.  The earlier
        # version appended per-target slices and torch.stack'ed them at the end,
        # which held two copies of every module on the GPU at once -- 2 x 38.8 GB
        # for 134 targets, and an OOM on an 80 GB card.
        BU[m] = acc.to(device)
    # one global Frobenius normalisation per target, matching how steering
    # measures alpha in units of "one trait adapter's worth of weight change"
    for j, s in enumerate(specs):
        n = math.sqrt(sum(float((BU[m][j] * BU[m][j]).sum()) for m in mods))
        for m in mods:
            BU[m][j] /= max(n, 1e-12)
        s["bu_norm"] = n
    D = {k: float(sum(v) / len(v)) for k, v in drift.items()}
    D["all"] = float(sum(x for v in drift.values() for x in v)
                     / sum(len(v) for v in drift.values()))
    return A0, BU, mods, D


class _Probe(object):
    """Wraps a Linear so its output picks up sum_t eps[t, i] * B_t (A_0 x)."""

    def __init__(self, A0, BU, state):
        self.A0, self.BU, self.state = A0, BU, state

    def __call__(self, mod, inp, out):
        """sum_t eps[t,b] B_t (A_0 x).

        Contract eps into the target stack FIRST.  Doing it the other way round
        would build a (T, batch, seq, d_out) tensor -- gigabytes per module for
        25 targets -- whereas Q is (batch, d_out, r) and costs ten megabytes.
        """
        import torch
        eps = self.state["eps"]
        # The hooks stay installed across plain generation, where there is no eps
        # to differentiate. Passing the output through untouched means candidates
        # are generated by the unmodified base model, which is what we want.
        if eps is None:
            return out
        x = inp[0]                                     # (T, batch)
        h = torch.nn.functional.linear(x.float(), self.A0)          # (b, s, r)
        Q = torch.einsum("tb,tor->bor", eps, self.BU)               # (b, d_out, r)
        return out + torch.einsum("bor,bsr->bso", Q, h).to(out.dtype)


# The 25-target runs fit a 40GB card.  All 134 adapters as targets is 38.8 GB of
# fp32 B_U alone (sum d_out 1,132,032 x r 64 x 4 bytes x 134), so the N x N run
# sets PC_SCORE_GPU=A100-80GB.  Compute is nearly independent of T: the forward
# pass dominates and the per-module einsum over targets is a rounding error.
#
# MEMORY IS NOT A FUNCTION OF T ALONE, which is the trap.  Everything retained
# for the backward pass -- Q, the model's own activations, the float32 logit
# slices -- scales with batch x sequence and NOT with the target count, so "41
# targets is a third of 134, therefore it fits a 40 GB card" is wrong.  The
# 2026-09-09 reward-hacks data-scoring run had 41 targets (11.9 GB of fp32 B_U)
# and OOM'd on a 40 GB card TWICE: at batch 4 with 38.92 GB allocated
# (phase10_runs/sorh_datascore.log.1788982873, at item 4 of 1053) and again at
# batch 2 with 38.47 GB allocated (log.1788983750, at item 402 of 1053).  It ran
# on A100-80GB at batch 4.  Halving the batch bought far less than halving: the
# fixed cost (B_U plus the bf16 model) is ~20 GB before a single token, so on a
# 40 GB card the whole activation budget is the remaining 19 GB.  Take the bigger
# card whenever T x 0.29 GB + 8 GB is more than half the card.
@app.function(image=image, gpu=os.environ.get("PC_SCORE_GPU", "A100-40GB"), volumes=VOLS,
              timeout=60 * 60 * 6, secrets=[modal.Secret.from_name("hf-token")])
def score(job: dict) -> dict:
    """Score a list of DPO pairs against every target direction."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    specs, items = job["targets"], job["items"]
    bs = job.get("batch", 4)
    maxlen = int(job.get("maxlen", MAXLEN))
    T = len(specs)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    A0, BU, mods, drift = _build_targets(specs, "cuda", job.get("a0"))
    print(f"[targets] {T} directions over {len(mods)} modules; "
          f"mean ||A_i - A_0|| / ||A_0|| by source volume = "
          + ", ".join(f"{k} {v:.4f}" for k, v in sorted(drift.items())), flush=True)

    by = dict(model.named_modules())
    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
        return None
    state = {"eps": None}
    n_hook = 0
    for m in mods:
        c = find(m)
        if c is None:
            raise RuntimeError(f"module not found in base model: {m}")
        by[c].register_forward_hook(_Probe(A0[m], BU[m], state))
        n_hook += 1
    if n_hook != len(mods):
        raise RuntimeError(f"hooked {n_hook} of {len(mods)} modules")
    print(f"[hooks] {n_hook} modules instrumented", flush=True)

    def enc(item, response):
        """Tokenise one (prompt, completion) exactly as its own recipe did.

        Default: the DPO spelling -- prompt and response tokenised separately,
        no EOS, cap MAXLEN.  Every pre-2026-09-09 items file takes this branch
        unchanged.  item["mode"] == "sft" instead reproduces sft_rewardhacks.py
        byte for byte: prompt + completion + EOS tokenised as ONE string, the
        first len(tok(prompt)) positions masked, cap item["maxlen"].  The two
        differ (the EOS is in the SFT loss and the boundary token can merge), so
        scoring SFT data with the DPO spelling would not be scoring the data the
        run was actually trained on.
        """
        # CHANGE 1: an item may give the whole prompt-side conversation.  The
        # single-user-message spelling is what every pre-2026-09-09 items file
        # uses and is exactly the len-1 case of this one.
        msgs = item.get("messages") or [{"role": "user", "content": item["prompt"]}]
        pre = tok.apply_chat_template(msgs, tokenize=False,
                                      add_generation_prompt=True,
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
        ids = (a + b)[:ml]
        lab = ([-100] * len(a) + b)[:ml]
        return ids, lab

    def run(batch):
        """One backward pass -> (T, batch) exact per-example directional derivatives.

        The loss is the SUM of per-example negative log-likelihoods, and
        eps[t, i] enters only example i, so eps.grad[t, i] is d(-logp_i)/d(eps).
        The sign is flipped on return so that a positive score means "steering
        along +v makes this text more likely", i.e. the data trains toward +v.
        """
        m = max(len(x[0]) for x in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.tensor([x[0] + [pad] * (m - len(x[0])) for x in batch], device="cuda")
        lab = torch.tensor([x[1] + [-100] * (m - len(x[1])) for x in batch], device="cuda")
        att = torch.tensor([[1] * len(x[0]) + [0] * (m - len(x[0])) for x in batch],
                           device="cuda")
        eps = torch.zeros(T, len(batch), device="cuda", requires_grad=True)
        state["eps"] = eps
        # use_cache=False is CHANGE 5, and it is the one that mattered.  With the
        # default, per-group peak memory stayed flat (42.5 GB at batch 1 x 1025
        # tokens) while the process crept to 79 GB and died eight groups in --
        # twice, at two different budgets.  Qwen3.5's hybrid linear-attention
        # layers keep recurrent state when the cache is on, and nothing here ever
        # reads it.  The 2026-09-09 reward-hacks run's two OOMs, at item 4 and at
        # item 402 of the same 1053, look like the same thing read as a batch-size
        # problem; see phase10_runs/dolci_probe.log.* for the three probes.
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1]
        tgt = lab[:, 1:]
        msk = (tgt != -100)
        # log p = logit[target] - logsumexp(logits), computed in float32 over
        # slices of the sequence.  A full log_softmax over a 152k vocabulary
        # would be three gigabytes of float32 per batch before the backward
        # pass even starts; logsumexp collapses the vocabulary axis, and
        # chunking bounds the transient to a couple of hundred megabytes.
        tk = []
        for a in range(0, tgt.shape[1], 128):
            sl = lg[:, a:a + 128].float()
            g = sl.gather(-1, tgt[:, a:a + 128].clamp(min=0).unsqueeze(-1)).squeeze(-1)
            tk.append((g - torch.logsumexp(sl, -1)) * msk[:, a:a + 128])
        nll = -(torch.cat(tk, 1).sum(1))                       # per example
        nll.sum().backward()
        n = msk.sum(1).clamp(min=1).float()
        g = (-eps.grad / n).detach().cpu().numpy()
        state["eps"] = None       # never leave it set for a later forward pass
        nn = n.cpu().numpy()
        del lg, tk, nll, eps, ids, lab, att, tgt, msk, n
        return g, nn

    # CHANGE 3: token-budget batching over a longest-first item list.  `ntok`
    # is the caller's own count of the longest of the item's two encodings; an
    # item without one falls back to the length cap, which is the safe reading.
    budget = int(job.get("tok_budget", 2300))
    groups, i = [], 0
    for x in items:
        x.setdefault("ntok", int(x.get("maxlen", maxlen)))
    while i < len(items):
        g, L = [items[i]], items[i]["ntok"]
        i += 1
        while (i < len(items) and len(g) < bs
               and (len(g) + 1) * max(L, items[i]["ntok"]) <= budget):
            L = max(L, items[i]["ntok"])
            g.append(items[i])
            i += 1
        groups.append(g)
    print(f"[batching] {len(items)} items -> {len(groups)} groups, "
          f"budget {budget} token-slots, max batch {bs}; longest item "
          f"{max(x['ntok'] for x in items)} tokens", flush=True)

    import time
    t0 = time.time()
    out, done = [], 0
    def run_group(chunk):
        """Score one group, halving and retrying if the card runs out."""
        try:
            sc, nc = run([enc(x, x["chosen"]) for x in chunk])
            two = [x for x in chunk if not x.get("single")]   # CHANGE 2
            sr = nr = None
            if two:
                sr, nr = run([enc(x, x["rejected"]) for x in two])
            return sc, nc, two, sr, nr
        except torch.OutOfMemoryError:
            state["eps"] = None
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise
            print(f"  [oom] halving a group of {len(chunk)} at len "
                  f"{chunk[0]['ntok']} and retrying", flush=True)
            return None

    n_oom = 0
    gq = list(groups)
    gi = -1
    while gq:
        gi += 1
        chunk = gq.pop(0)
        r = run_group(chunk)
        if r is None:
            n_oom += 1
            h = len(chunk) // 2
            gq = [chunk[:h], chunk[h:]] + gq
            gi -= 1
            continue
        sc, nc, two, sr, nr = r
        ri = {x["id"]: k for k, x in enumerate(two)}
        for j, x in enumerate(chunk):
            rec = {"id": x["id"], "trait": x.get("trait"),
                   "chosen": sc[:, j].tolist(), "n_tok": [int(nc[j])]}
            if not x.get("single"):
                k = ri[x["id"]]
                rec["rejected"] = sr[:, k].tolist()
                rec["pair"] = (sc[:, j] - sr[:, k]).tolist()
                rec["n_tok"] = [int(nc[j]), int(nr[k])]
            out.append(rec)
        done += len(chunk)
        torch.cuda.empty_cache()      # see the note under CHANGE 3
        if gi < 12 or gi % 100 == 0 or done == len(items):
            el = time.time() - t0
            print(f"  {done}/{len(items)}  bs={len(chunk)} len={chunk[0]['ntok']}  "
                  f"t={el:.0f}s  {done / max(el, 1e-9):.2f} it/s  "
                  f"eta={(len(items) - done) / max(done / max(el, 1e-9), 1e-9) / 60:.1f} min"
                  f"  GB={torch.cuda.max_memory_allocated() / 2**30:.1f}"
                  f"/{torch.cuda.memory_allocated() / 2**30:.1f}", flush=True)
            torch.cuda.reset_peak_memory_stats()
    return {"names": [s["name"] for s in specs], "a_drift": drift["all"],
            "a_drift_by_source": drift, "n_groups": len(groups), "n_oom_retries": n_oom,
            "elapsed_s": time.time() - t0,
            "peak_gb_last_window": torch.cuda.max_memory_allocated() / 2 ** 30,
            "tok_budget": budget, "max_batch": bs,
            "bu_norm": [s["bu_norm"] for s in specs], "scores": out}


@app.function(image=image, gpu="A100-40GB", volumes=VOLS, timeout=60 * 90,
              secrets=[modal.Secret.from_name("hf-token")])
def validate(job: dict) -> dict:
    """Does the eps-derivative equal a real gradient inner product, and a real
    finite difference?  Three independent computations of the same number."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    specs, items = job["targets"][:3], job["items"][:24]
    T = len(specs)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32,
                                                 device_map="cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    A0, BU, mods, drift = _build_targets(specs, "cuda")
    by = dict(model.named_modules())
    def find(m):
        for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
            if c in by:
                return c
    state = {"eps": None}
    for m in mods:
        by[find(m)].register_forward_hook(_Probe(A0[m], BU[m], state))

    def prep(x):
        pre = tok.apply_chat_template([{"role": "user", "content": x["prompt"]}],
                                      tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        a = tok(pre, add_special_tokens=False)["input_ids"]
        b = tok(x["chosen"], add_special_tokens=False)["input_ids"]
        return (a + b)[:MAXLEN], ([-100] * len(a) + b)[:MAXLEN]

    def nll(ids, lab, eps):
        state["eps"] = eps
        i = torch.tensor([ids], device="cuda")
        l = torch.tensor([lab], device="cuda")
        lg = model(input_ids=i).logits.float()
        lp = torch.log_softmax(lg[:, :-1], -1)
        t = l[:, 1:]
        msk = t != -100
        return -(lp.gather(-1, t.clamp(min=0).unsqueeze(-1)).squeeze(-1) * msk).sum()

    rows = []
    EPS = job.get("eps", [3e-3, 1e-3, 3e-4])
    for x in items:
        ids, lab = prep(x)
        e = torch.zeros(T, 1, device="cuda", requires_grad=True)
        v = nll(ids, lab, e)
        v.backward()
        ana = (-e.grad[:, 0]).detach().cpu().numpy()
        fd = {}
        with torch.no_grad():
            for h in EPS:
                p = np.zeros(T)
                for t in range(T):
                    ep = torch.zeros(T, 1, device="cuda"); ep[t, 0] = h
                    em = torch.zeros(T, 1, device="cuda"); em[t, 0] = -h
                    p[t] = float(-(nll(ids, lab, ep) - nll(ids, lab, em)) / (2 * h))
                fd[str(h)] = p.tolist()
        rows.append({"id": x["id"], "analytic": ana.tolist(), "fd": fd})
        print(f"  {len(rows)}/{len(items)}", flush=True)
    return {"names": [s["name"] for s in specs], "rows": rows, "eps": EPS}


@app.local_entrypoint()
def main(stage: str = "validate", spec: str = "phase10_runs/align_targets.json",
         items: str = "phase10_runs/align_items.json",
         out: str = "analysis/align_validate.json", batch: int = 4, shard: int = 0,
         nshard: int = 1, tok_budget: int = 0, limit: int = 0):
    here = os.path.dirname(os.path.abspath(__file__))
    S = json.load(open(os.path.join(here, spec)))
    I = json.load(open(os.path.join(here, items)))
    I = [x for i, x in enumerate(I) if i % nshard == shard]
    if limit:
        I = I[:limit]
    job = {"targets": S["targets"], "items": I, "batch": batch}
    if tok_budget:
        job["tok_budget"] = tok_budget
    # A spec file may pin the A_0 window and the length cap; both stay in the
    # job dict rather than the environment so the run is reproducible from the
    # two input files alone.
    for k in ("a0", "maxlen", "tok_budget"):
        if k in S:
            job[k] = S[k]
    print(f"{len(S['targets'])} targets x {len(I)} items (shard {shard}/{nshard})",
          flush=True)
    r = (validate if stage == "validate" else score).remote(job)
    with open(os.path.join(here, out), "w") as f:
        json.dump(r, f)
    print(f"wrote {out}", flush=True)
