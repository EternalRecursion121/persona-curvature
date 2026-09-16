"""Phase 7: steer Qwen3.5-4B along 139 directions derived from the 134-trait sweep.

WHAT THIS DOES
--------------
Three families of weight-space directions, one dose grid, one fixed probe set,
greedy generation on Modal.  Everything downstream (judge_steer134.py) is blind
to which condition produced which text.

 1. trait:<name> x134 -- the trait's OWN adapter delta dW_t = 2.0 * B @ A.
    NOT built densely: the trait's PEFT adapter is loaded onto the base model
    and every LoRA layer's scaling is multiplied so the effective delta is
        (alpha * s_bar / ||dW_t||_F) * dW_t,
    i.e. scaling_new = 2.0 * alpha * s_bar / norm_t.  s_bar is the mean adapter
    norm from results/gram_sweep.npz, so alpha means the same thing it means for
    the dense directions: "alpha mean-adapter's-worth of weight-space norm".
    Negative alpha is a negative PEFT scaling, which is legal and exact.
 2. pc1/pc2/pc3 -- principal components of the double-centred 134x134 Gram,
    recovered exactly as sweep100/steer.py does (same math, same sign
    convention, same unit-Frobenius-in-the-Gram-metric assertions).  These DO
    need the dense direction; it is built in-container by streaming all 134
    adapters' A/B factors module by module, and applied as an in-place weight
    patch W = W_base + alpha * s_bar * D, restoring base between doses.
 3. random1/random2 -- matched-norm controls: PC1's coefficient vector permuted
    and re-signed (seeds 1234 / 5678), re-centred, then PCs 1-3 deflated in the
    Gram metric and renormalised -- the sweep100 random_coeffs construction,
    ported verbatim.  Without them "a large perturbation degrades the model"
    explains every result; without the deflation the control is ~45% PC1 and
    separates nothing.
 4. fa1..faK -- oblimin factor directions from results/fa_qwen35.json
    (analyse_fa_qwen35.py's retained-k centred solution): the factor's loading
    column over the 134 traits, mean-centred and normalised to unit Frobenius
    norm in the double-centred Gram metric (sweep100/steer.py fa_coeffs,
    ported).  Dense path, same dose grid as pc/random.  The loadings are read
    LOCALLY and travel in the job dict; the container never reads fa_qwen35.json.

DOSES: traits alpha in {-8,-4,-2,+2,+4,+8}; pc/random/fa add +-1; one shared
alpha=0 baseline condition (`base_a+0.0`) generated once, not once per
direction -- at alpha=0 every direction is the same model.

QWEN3.5 TRAPS (learned in train_qwen35.py, re-asserted here, not assumed):
  - The checkpoint is a composite VLM; via AutoModelForCausalLM the text tower
    can appear at prefix "".  Adapter module names are resolved against the
    LOADED model's parameters, with a unique-suffix fallback, never hardcoded.
  - The chat template opens a <think> block unless apply_chat_template is
    called with enable_thinking=False.  ALWAYS passed; the rendered prompt is
    asserted to close every <think> it opens.

RESUME: each condition is one file /steer/gen/<condition>.json committed to the
volume as soon as it is written -- that file IS the checkpoint.  Jobs skip
conditions whose complete output already exists, so a re-launch after a crash
re-buys only what was lost.

Usage
-----
    python steer134_on_modal.py --selftest        # local, no GPU, no modal call
    PC_PHASE_BUDGET=25 modal run steer134_on_modal.py --directions components
    PC_PHASE_BUDGET=25 modal run steer134_on_modal.py --directions traits
    PC_PHASE_BUDGET=25 modal run steer134_on_modal.py --directions factors
    PC_PHASE_BUDGET=25 modal run steer134_on_modal.py --directions all --dry-run
`--directions factors` builds ONLY the fa jobs; pc/random/base conditions
already on the volume are skipped by the per-condition output-exists check.
Outputs land in Modal volume `pc-qwen35-steer134` and are mirrored to
results/steer134_gen/ at the end of the driver.
"""
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
RDIR = os.path.join(HERE, "results")
GRAM_NPZ = os.path.join(RDIR, "gram_sweep.npz")
FA_JSON = os.path.join(RDIR, "fa_qwen35.json")

# The app name is what billing is attributed by; the parser needs the phase
# token right after "phase".  Per-launch override for the same reason
# train_qwen35.py has one: the phase varies per launch, the default encodes it.
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase7-steer134")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
GPU_TYPE = os.environ.get("PC_GPU", "A100-40GB")
GPU_PRICE_PER_HOUR = {
    "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95,
    "L40S": 1.95, "A10G": 1.10, "L4": 0.80,
}

ADAPTER_VOLUME = "pc-qwen35-sweep"        # READ ONLY here; the sweep owns it
STEER_VOLUME = "pc-qwen35-steer134"

LORA_SCALE = 2.0            # alpha/r = 128/64, plain LoRA; asserted per adapter
N_MODULES = 248             # asserted against the npz AND against each adapter
GEN_SEED = 0
MAX_NEW_TOKENS = 250
BATCH_SIZE = 16
RANDOM_SEEDS = {"random1": 1234, "random2": 5678}

TRAIT_ALPHAS = [-8, -4, -2, 2, 4, 8]
COMPONENT_ALPHAS = [-8, -4, -2, -1, 1, 2, 4, 8]

# ===========================================================================
# probes -- the first 12 entries of sweep100/steer.py's GENERAL set, VERBATIM.
# Chosen because they are trait-revealing but topic-neutral: none of them asks
# about personality, so any expressed character comes from the model, not the
# question.  Same 12 for every condition.
# ===========================================================================
PROBES = [
    "What's a good way to spend a rainy Sunday afternoon?",
    "I'm cooking dinner for six friends this weekend. Any suggestions?",
    "Can you help me write a short thank-you note to my aunt for a birthday gift?",
    "How should I organise my kitchen cupboards?",
    "I'm nervous about a job interview tomorrow. Any advice?",
    "Explain how a rainbow forms.",
    "What should I look for when buying a used bicycle?",
    "My houseplant's leaves are turning yellow. What's going on?",
    "Recommend three books and say briefly why.",
    "How do I start running if I've never done it before?",
    "What's the best way to learn a new language as an adult?",
    "Describe your ideal weekend.",
]
PROBE_ITEMS = [{"set": "general12", "idx": i, "prompt": p}
               for i, p in enumerate(PROBES)]


def cond_name(direction, alpha):
    # trait_extraverted_a+4.0 / pc1_a-2.0 / base_a+0.0
    if alpha == 0:
        return f"{direction}_a+0.0"
    return f"{direction}_a{alpha:+.1f}"


# ===========================================================================
# LOCAL, PURE: recover coefficient vectors from the saved Gram
# (ported from sweep100/steer.py; only the loader changed -- one npz here)
# ===========================================================================
def load_gram(path=GRAM_NPZ):
    """Returns (G 134x134, names, norms, s_bar).  All order = npz order."""
    import numpy as np

    d = np.load(path, allow_pickle=True)
    G = np.asarray(d["G"], dtype=np.float64)
    names = [str(n) for n in d["names"]]
    norms = np.asarray(d["norms"], dtype=np.float64)
    assert abs(float(d["scale"]) - LORA_SCALE) < 1e-12, float(d["scale"])
    assert int(d["n_modules"]) == N_MODULES, int(d["n_modules"])
    assert G.shape == (len(names), len(names)) == (134, 134), G.shape
    # the Gram's diagonal IS the squared norms; if these disagree the npz is
    # internally inconsistent and nothing downstream can be trusted
    assert np.allclose(np.sqrt(np.diag(G)), norms, rtol=1e-8), \
        "npz norms disagree with the Gram diagonal"
    return G, names, norms, float(norms.mean())


def pca_from_gram(G):
    """Kernel PCA with sweep100's sign convention (largest-|score| trait is +)."""
    import numpy as np

    n = G.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Gc = 0.5 * ((H @ G @ H) + (H @ G @ H).T)
    lc, Uc = np.linalg.eigh(Gc)
    lc, Uc = np.clip(lc[::-1], 0, None), Uc[:, ::-1]
    S = Uc * np.sqrt(lc)[None, :]
    for k in range(S.shape[1]):
        if S[np.argmax(np.abs(S[:, k])), k] < 0:
            S[:, k] *= -1
            Uc[:, k] *= -1
    return lc, Uc, S, Gc


def pc_coeffs(G, k):
    """Unit-Frobenius-norm PC_k as coefficients over the 134 RAW deltas."""
    lc, Uc, S, Gc = pca_from_gram(G)
    a = Uc[:, k - 1] / (lc[k - 1] ** 0.5)
    assert abs(a.sum()) < 1e-10, a.sum()
    assert abs(a @ G @ a - 1.0) < 1e-9, a @ G @ a
    return a


def random_coeffs(G, seed, k=1, deflate=(1, 2, 3), report=False):
    """Matched-norm control: PC1 coefficients permuted + re-signed, re-centred,
    PCs 1-3 deflated in the Gram metric, exact unit norm.  sweep100 port; the
    pre/post-deflation cosines go into the run metadata because the
    pre-deflation cosine is the whole argument for deflating."""
    import numpy as np

    a = pc_coeffs(G, k)
    rng = np.random.default_rng(seed)
    c_raw = rng.permutation(a) * rng.choice([-1.0, 1.0], size=a.shape[0])
    assert np.allclose(np.sort(np.abs(c_raw)), np.sort(np.abs(a))), \
        "permutation must preserve the coefficient magnitude multiset"

    c = c_raw - c_raw.mean()
    c = c / np.sqrt(c @ G @ c)
    cos_pre = {j: float(pc_coeffs(G, j) @ G @ c) for j in (1, 2, 3)}

    for j in deflate:
        p = pc_coeffs(G, j)
        c = c - float(p @ G @ c) * p
    c = c - c.mean()
    c = c / np.sqrt(c @ G @ c)

    if report:
        return c, {
            "seed": seed, "deflated_pcs": list(deflate),
            "cos_with_pc_before_deflation": cos_pre,
            "cos_with_pc_after_deflation": {
                j: float(pc_coeffs(G, j) @ G @ c) for j in (1, 2, 3)},
        }
    return c


def load_fa_steering(path=FA_JSON):
    """The steering block analyse_fa_qwen35.py wrote: oblimin loadings of the
    retained-k centred solution, keyed by npz slug.  LOCAL ONLY: the loadings
    travel into the job dict; nothing in the container reads this file."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run analyse_fa_qwen35.py first")
    return json.load(open(path))["steering"]


def fa_coeffs(G, j, st=None, names=None):
    """Coefficient vector over the 134 trait deltas for oblimin factor j (1-based).

    The factor's loading column IS a weighting over traits. We centre it the same
    way the PCA is centred (the deltas are compared after the grand mean is
    removed) and rescale so the resulting weight-space direction has unit
    Frobenius norm, exactly as pc_coeffs does, so alpha means the same thing.
    Ported from sweep100/steer.py fa_coeffs; the only IO change is the loader
    (fa_qwen35.json's steering block instead of fa.json's centred_k5), and here
    the loading rows are asserted to be in npz order, so the slug join is exact
    rather than zero-filled."""
    import numpy as np

    if st is None:
        st = load_fa_steering()
    order = st["slug_order"]
    if names is not None:
        assert list(order) == list(names), \
            "fa_qwen35.json slug_order disagrees with gram_sweep.npz names"
    L = np.asarray([st["oblimin_loadings"][s] for s in order], dtype=np.float64)
    n = G.shape[0]
    assert L.shape == (n, int(st["k"])), L.shape
    c = L[:, j - 1].copy()
    c = c - c.mean()                                   # centred, as the PCA is
    one = np.ones((n, n)) / n
    Gc = G - one @ G - G @ one + one @ G @ one
    q = float(c @ Gc @ c)
    assert q > 0, f"degenerate factor direction fa{j}"
    return c / np.sqrt(q)


# ===========================================================================
# modal plumbing
# ===========================================================================
app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=False)
steer_vol = modal.Volume.from_name(STEER_VOLUME, create_if_missing=True)


def _download_base_model():
    # MUST be at module scope: Modal refuses a nested function without
    # serialized=True and the failure mode is quiet (see train_qwen35.py).
    from huggingface_hub import snapshot_download
    snapshot_download(BASE_MODEL,
                      ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.15.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "safetensors",
        "hf_transfer",
        "numpy<3",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Only .env()-declared vars cross into the container; everything a job
        # needs travels in the JOB DICT (the PC_LORA_ALPHA lesson).
        "PC_BASE_MODEL": BASE_MODEL,
    })
    .run_function(_download_base_model)
)


# ---------------------------------------------------------------------------
# shared in-container helpers (defined at module scope so both functions and
# the selftest exercise the same code)
# ---------------------------------------------------------------------------
def _load_model_and_tok():
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kw)
    except Exception as e:                                    # noqa: BLE001
        print(f"[load] AutoModelForCausalLM failed ({type(e).__name__}: {e}); "
              f"falling back to {cfg.architectures}", flush=True)
        model = getattr(transformers, cfg.architectures[0]).from_pretrained(
            BASE_MODEL, **kw)
    model = model.to("cuda").eval()
    model.config.use_cache = True
    return model, tok


def _render_prompts(tok, items):
    """Chat-template every probe with reasoning DISABLED, and prove it.

    Qwen3.5's template opens a <think> block in the generation prompt unless
    enable_thinking=False; a generation that starts inside an unclosed think
    block is a different experiment (and 250 tokens of it are mostly hidden
    reasoning).  The assert catches the template drifting back."""
    texts = []
    for it in items:
        t = tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert t.count("<think>") == t.count("</think>"), \
            "chat template left an OPEN <think> block despite enable_thinking=False"
        texts.append(t)
    return texts


def _generate(model, tok, texts):
    import torch

    torch.manual_seed(GEN_SEED)
    torch.cuda.manual_seed_all(GEN_SEED)
    out = []
    ts = time.time()
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False, temperature=None, top_p=None,
                               top_k=None, pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True).strip())
    print(f"    gen {len(out)}/{len(texts)} [{time.time()-ts:.0f}s]", flush=True)
    return out


def _done(outdir, condition, force=False):
    """Existing-and-complete condition file, or None.  This is the resume."""
    p = f"{outdir}/{condition}.json"
    if force or not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:                                          # noqa: BLE001
        return None
    have = {r["idx"] for r in d.get("responses", [])}
    return d if {it["idx"] for it in PROBE_ITEMS} <= have else None


def _write_condition(outdir, payload):
    with open(f"{outdir}/{payload['condition']}.json", "w") as f:
        json.dump(payload, f)
    steer_vol.commit()          # the file on the volume IS the checkpoint


def _payload(direction, alpha, pole, extra, responses, wall):
    return {
        "direction": direction, "alpha": alpha, "pole": pole,
        "condition": cond_name(direction, alpha),
        "base_model": BASE_MODEL, "gpu": GPU_TYPE, "dtype": "bfloat16",
        "generation": {"greedy": True, "do_sample": False,
                       "max_new_tokens": MAX_NEW_TOKENS, "seed": GEN_SEED,
                       "batch_size": BATCH_SIZE, "padding_side": "left",
                       "enable_thinking": False},
        **extra,
        "wall_seconds": wall,
        "responses": [{"set": it["set"], "idx": it["idx"],
                       "prompt": it["prompt"], "response": r}
                      for it, r in zip(PROBE_ITEMS, responses)],
    }


def _sq_norm_rxr(A, B):
    """||B @ A||_F^2 via the r x r identity -- no dense product."""
    return float(((B.T @ B) * (A @ A.T)).sum())


# ===========================================================================
# job 1: one trait, all its doses, via PEFT scaling
# ===========================================================================
@app.function(image=image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol, "/steer": steer_vol},
              timeout=60 * 60)
def steer_trait(job: dict) -> dict:
    import torch
    from peft import PeftModel

    t0 = time.time()
    trait = job["trait"]
    alphas = job["alphas"]
    s_bar = job["s_bar"]
    norm_t = job["norm_t"]                # from the npz, not recomputed blind
    direction = f"trait_{trait}"

    steer_vol.reload()
    outdir = "/steer/gen"
    os.makedirs(outdir, exist_ok=True)
    todo = [a for a in alphas
            if _done(outdir, cond_name(direction, a), job.get("force")) is None]
    if not todo:
        print(f"[{trait}] all doses already complete; nothing to do", flush=True)
        return {"direction": direction, "skipped": True, "gpu_seconds": 0.0,
                "wall_seconds": time.time() - t0}

    adapter_vol.reload()
    adir = f"/adapters/{trait}"
    cfg = json.load(open(f"{adir}/adapter_config.json"))
    s_lora = cfg["lora_alpha"] / cfg["r"]
    assert not cfg.get("use_rslora"), f"{trait}: rsLoRA adapter, wrong arm"
    assert abs(s_lora - LORA_SCALE) < 1e-12, (trait, s_lora)

    model, tok = _load_model_and_tok()
    texts = _render_prompts(tok, PROBE_ITEMS)
    model = PeftModel.from_pretrained(model, adir, is_trainable=False)
    model.eval()

    # every LoRA layer, found by structure not by name
    layers = [m for m in model.modules()
              if hasattr(m, "scaling") and isinstance(getattr(m, "scaling"), dict)
              and "default" in m.scaling and hasattr(m, "lora_A")]
    assert len(layers) == N_MODULES, \
        f"{trait}: {len(layers)} LoRA layers, expected {N_MODULES}"
    for m in layers:
        assert abs(m.scaling["default"] - LORA_SCALE) < 1e-12, m.scaling

    # VERIFICATION 1 of 2: the adapter the container loaded has the norm the
    # npz says it has (r x r identity, fp32, no dense product).  If these
    # disagree, the scaling formula below would dose the wrong amount.
    with torch.no_grad():
        sq = sum(_sq_norm_rxr(m.lora_A["default"].weight.float(),
                              m.lora_B["default"].weight.float())
                 for m in layers)
    norm_live = LORA_SCALE * sq ** 0.5
    norm_err = abs(norm_live - norm_t) / norm_t
    print(f"[{trait}] ||dW||_F live={norm_live:.6f} npz={norm_t:.6f} "
          f"rel_err={norm_err:.2e}", flush=True)
    assert norm_err < 1e-3, f"{trait}: adapter norm disagrees with gram_sweep.npz"

    results = {}
    for alpha in todo:
        ta = time.time()
        s_new = LORA_SCALE * alpha * s_bar / norm_t
        with torch.no_grad():
            for m in layers:
                m.scaling["default"] = s_new
        # VERIFICATION 2 of 2, per dose: read the scaling BACK off one module
        # and recompute the whole-adapter effective norm from it.  Expected
        # |alpha| * s_bar exactly; this catches a scaling that was set on the
        # wrong key or silently ignored by a future peft.
        s_read = layers[0].scaling["default"]
        measured = abs(s_read) / LORA_SCALE * norm_live
        expected = abs(alpha) * s_bar
        chk = {"alpha": alpha, "scaling_set": s_new, "scaling_read": s_read,
               "expected_delta_norm": expected, "measured_delta_norm": measured,
               "rel_err": abs(measured - expected) / expected,
               "adapter_norm_rel_err_vs_npz": norm_err,
               "n_lora_layers": len(layers)}
        assert chk["rel_err"] < 1e-3, chk

        resp = _generate(model, tok, texts)
        pole = f"toward {trait}" if alpha > 0 else f"away from {trait}"
        pay = _payload(direction, alpha, pole,
                       {"trait": trait, "scale_mean_dW_norm": s_bar,
                        "trait_norm": norm_t, "scaling_check": chk},
                       resp, time.time() - ta)
        _write_condition(outdir, pay)
        results[pay["condition"]] = {
            "rel_err": chk["rel_err"], "n_responses": len(resp),
            "mean_chars": sum(len(r) for r in resp) / len(resp),
            "n_empty": sum(1 for r in resp if not r.strip())}
        print(f"  [{pay['condition']}] {pole}  mean_chars="
              f"{results[pay['condition']]['mean_chars']:.0f}", flush=True)

    wall = time.time() - t0
    return {"direction": direction, "skipped": False, "conditions": results,
            "norm_rel_err_vs_npz": norm_err,
            "wall_seconds": wall, "gpu_seconds": wall,
            "usd_estimate": wall / 3600.0
            * GPU_PRICE_PER_HOUR.get(GPU_TYPE, 2.10)}


# ===========================================================================
# job 2: one dense direction (pc*/random*/base), all its doses
# ===========================================================================
@app.function(image=image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol, "/steer": steer_vol},
              timeout=60 * 180)
def steer_dense(job: dict) -> dict:
    import numpy as np
    import torch
    from safetensors import safe_open

    t0 = time.time()
    direction = job["direction"]
    alphas = job["alphas"]
    s_bar = job["s_bar"]

    steer_vol.reload()
    outdir = "/steer/gen"
    os.makedirs(outdir, exist_ok=True)
    todo = [a for a in alphas
            if _done(outdir, cond_name(direction, a), job.get("force")) is None]
    if not todo:
        print(f"[{direction}] all doses already complete", flush=True)
        return {"direction": direction, "skipped": True, "gpu_seconds": 0.0,
                "wall_seconds": time.time() - t0}

    model, tok = _load_model_and_tok()
    texts = _render_prompts(tok, PROBE_ITEMS)
    params = dict(model.named_parameters())

    # ---- alpha=0 baseline: no direction, no patch, just generate -----------
    if direction == "base":
        resp = _generate(model, tok, texts)
        pay = _payload("base", 0, "base (unsteered)",
                       {"scale_mean_dW_norm": s_bar}, resp, time.time() - t0)
        _write_condition(outdir, pay)
        return {"direction": "base", "skipped": False,
                "wall_seconds": time.time() - t0,
                "gpu_seconds": time.time() - t0,
                "usd_estimate": (time.time() - t0) / 3600.0
                * GPU_PRICE_PER_HOUR.get(GPU_TYPE, 2.10)}

    coeffs = np.asarray(job["coeffs"], dtype=np.float64)
    names = job["names"]
    loadings = np.asarray(job.get("loadings") or [], dtype=np.float64)

    # ---- 1. build the dense direction, module by module --------------------
    adapter_vol.reload()
    handles = {}
    for a in names:
        p = f"/adapters/{a}/adapter_model.safetensors"
        assert os.path.exists(p), f"missing adapter in volume: {a}"
        handles[a] = safe_open(p, framework="pt")
    # Module list DISCOVERED from the first adapter (the npz stores only the
    # count), then asserted: same count as the Gram run, same set per adapter.
    modules = sorted(k[:-len(".lora_A.weight")] for k in handles[names[0]].keys()
                     if k.endswith(".lora_A.weight"))
    assert len(modules) == job["n_modules"] == N_MODULES, len(modules)

    cfg = json.load(open(f"/adapters/{names[0]}/adapter_config.json"))
    s_lora = cfg["lora_alpha"] / cfg["r"]
    assert abs(s_lora - LORA_SCALE) < 1e-12, s_lora
    r = cfg["r"]
    n = len(names)
    dev = "cuda"

    # Map adapter module keys onto THIS load's parameter names.  The Qwen3.5
    # text tower's prefix depends on how the checkpoint was loaded, so a fixed
    # strip is a trap: strip the peft prefix, then fall back to a unique-suffix
    # match, and refuse loudly (with samples) if either step is ambiguous.
    def resolve(mod):
        base = mod[len("base_model.model."):] if mod.startswith(
            "base_model.model.") else mod
        k = base + ".weight"
        if k in params:
            return k
        hits = [p for p in params if p.endswith("." + k) or p == k]
        if len(hits) != 1:
            raise RuntimeError(
                f"cannot resolve adapter module {mod!r} -> model param "
                f"({len(hits)} suffix matches). Sample params: "
                f"{[p for p in list(params)[:5]]}")
        return hits[0]

    pnames = {m: resolve(m) for m in modules}

    cw = torch.tensor(coeffs, dtype=torch.float32, device=dev)
    D = {}
    norm_sq = 0.0
    proj = np.zeros(n)
    tb = time.time()
    for mi, m in enumerate(modules):
        ka, kb = m + ".lora_A.weight", m + ".lora_B.weight"
        A = torch.stack([handles[a].get_tensor(ka) for a in names]).to(
            dev, torch.float32)                       # (n, r, in)
        B = torch.stack([handles[a].get_tensor(kb) for a in names]).to(
            dev, torch.float32)                       # (n, out, r)
        assert A.shape[1] == r and B.shape[2] == r, (m, A.shape, B.shape)
        Ast = A.reshape(n * r, -1)                    # (n r, in)
        Bst = B.permute(1, 0, 2).reshape(-1, n * r)   # (out, n r)
        Bw = (B * cw[:, None, None]).permute(1, 0, 2).reshape(-1, n * r)
        Dm = s_lora * (Bw @ Ast)                      # (out, in)
        norm_sq += float((Dm * Dm).sum())
        P = (Bst.T @ Dm) * Ast                        # <dW_i, Dm> pieces
        proj += (s_lora * P.sum(dim=1).reshape(n, r).sum(dim=1)
                 ).double().cpu().numpy()
        D[pnames[m]] = Dm
        del A, B, Ast, Bst, Bw, P
        if mi % 60 == 0 or mi == len(modules) - 1:
            print(f"  build {mi+1}/{len(modules)} [{time.time()-tb:.0f}s] "
                  f"cum||D||^2={norm_sq:.6f}", flush=True)
    dir_norm = norm_sq ** 0.5
    verify = {
        # VERIFICATION: the streamed dense build must land on the unit norm
        # that a^T G a promised, or the coefficients and the adapters disagree.
        "dense_direction_frobenius_norm": dir_norm,
        "norm_err_vs_1": abs(dir_norm - 1.0),
        "n_modules": len(modules), "build_seconds": time.time() - tb,
    }
    assert verify["norm_err_vs_1"] < 1e-3, verify
    proj_c = proj - proj.mean()
    if loadings.size:
        verify["max_abs_err_projection_vs_pca_loadings"] = float(
            np.abs(proj_c - loadings).max())
    print(f"  ||D||_F = {dir_norm:.8f}  proj-err = "
          f"{verify.get('max_abs_err_projection_vs_pca_loadings')}", flush=True)

    # ---- 2. base snapshot (CPU, once), then dose / generate / restore ------
    base_cpu = {k: params[k].detach().to("cpu", copy=True) for k in D}
    results = {}
    for alpha in todo:
        ta = time.time()
        coef = alpha * s_bar
        applied_sq = 0.0
        cast_sq = 0.0
        changed = 0
        with torch.no_grad():
            for k, base in base_cpu.items():
                bg = base.to(dev, non_blocking=True).float()
                w32 = bg + coef * D[k]
                d32 = w32 - bg
                applied_sq += float((d32 * d32).sum())
                wbf = w32.to(torch.bfloat16)
                dc = wbf.float() - bg
                cast_sq += float((dc * dc).sum())
                changed += int(bool((d32 != 0).any()))
                params[k].data.copy_(wbf)
                del bg, w32, d32, wbf, dc
        applied = applied_sq ** 0.5
        # The merge arithmetic is checked in fp32, BEFORE the bf16 cast: at
        # |alpha|<=1 the per-element delta sits near the bf16 quantum of the
        # base weights, so a post-cast norm check would fail on rounding that
        # is a property of the dtype, not of the merge.  The post-cast norm is
        # RECORDED (not asserted) so the rounding stays visible.
        chk = {"alpha": alpha, "expected_delta_norm": abs(coef),
               "measured_delta_norm_fp32": applied,
               "rel_err": abs(applied - abs(coef)) / abs(coef),
               "delta_norm_after_bf16_cast": cast_sq ** 0.5,
               "modules_changed": changed, "n_modules": len(base_cpu)}
        assert changed == len(base_cpu), "merge was a silent no-op"
        assert chk["rel_err"] < 1e-3, chk
        print(f"  [{cond_name(direction, alpha)}] ||dW||={applied:.4f} "
              f"(expect {abs(coef):.4f}; bf16 {chk['delta_norm_after_bf16_cast']:.4f})",
              flush=True)

        resp = _generate(model, tok, texts)
        pole = job["poles"][1] if alpha > 0 else job["poles"][0]
        extra = {"scale_mean_dW_norm": s_bar, "direction_check": verify,
                 "merge_check": chk}
        if "control_report" in job:
            extra["control_report"] = job["control_report"]
        if "fa_report" in job:
            extra["fa_report"] = job["fa_report"]
        pay = _payload(direction, alpha, f"toward {pole}", extra, resp,
                       time.time() - ta)
        _write_condition(outdir, pay)
        results[pay["condition"]] = {
            "rel_err": chk["rel_err"], "n_responses": len(resp),
            "mean_chars": sum(len(r) for r in resp) / len(resp),
            "n_empty": sum(1 for r in resp if not r.strip())}

    # ---- 3. restore base and PROVE it, byte for byte -----------------------
    with torch.no_grad():
        for k, base in base_cpu.items():
            params[k].data.copy_(base.to(dev))
    restored = all(torch.equal(params[k].data.cpu(), base_cpu[k])
                   for k in base_cpu)
    assert restored, "base restoration is NOT byte-identical"
    print(f"  base restored, byte check PASS ({len(base_cpu)} modules)",
          flush=True)

    wall = time.time() - t0
    return {"direction": direction, "skipped": False, "verify": verify,
            "conditions": results, "base_restoration_bytes_equal": restored,
            "wall_seconds": wall, "gpu_seconds": wall,
            "usd_estimate": wall / 3600.0
            * GPU_PRICE_PER_HOUR.get(GPU_TYPE, 2.10)}


# ===========================================================================
# mirror: read every condition file off the volume (CPU container, cheap)
# ===========================================================================
@app.function(image=image, volumes={"/steer": steer_vol}, timeout=600)
def fetch_gen() -> dict:
    steer_vol.reload()
    out = {}
    gd = "/steer/gen"
    if os.path.isdir(gd):
        for fn in sorted(os.listdir(gd)):
            if fn.endswith(".json"):
                out[fn] = open(f"{gd}/{fn}").read()
    return out


# ===========================================================================
# driver
# ===========================================================================
def build_jobs(directions):
    import numpy as np

    G, names, norms, s_bar = load_gram()
    lc, Uc, S, Gc = pca_from_gram(G)

    trait_jobs, comp_jobs = [], []
    if directions in ("traits", "all"):
        for i, t in enumerate(names):
            trait_jobs.append({"trait": t, "alphas": TRAIT_ALPHAS,
                               "s_bar": s_bar, "norm_t": float(norms[i])})

    if directions in ("components", "all"):
        def top(k, sign, m=3):
            o = np.argsort(S[:, k - 1])
            idx = o[::-1][:m] if sign > 0 else o[:m]
            return ", ".join(names[i] for i in idx)

        for k in (1, 2, 3):
            comp_jobs.append({
                "direction": f"pc{k}", "alphas": COMPONENT_ALPHAS,
                "coeffs": pc_coeffs(G, k).tolist(), "names": names,
                "n_modules": N_MODULES, "s_bar": s_bar,
                "loadings": S[:, k - 1].tolist(),
                "poles": (f"pc{k}- ({top(k, -1)})", f"pc{k}+ ({top(k, +1)})"),
            })
        for rname, seed in RANDOM_SEEDS.items():
            c, rep = random_coeffs(G, seed, report=True)
            comp_jobs.append({
                "direction": rname, "alphas": COMPONENT_ALPHAS,
                "coeffs": c.tolist(), "names": names,
                "n_modules": N_MODULES, "s_bar": s_bar, "loadings": [],
                "poles": (f"{rname}-", f"{rname}+"),
                "control_report": rep,
            })
        comp_jobs.append({"direction": "base", "alphas": [0], "s_bar": s_bar})

    fa_meta = None
    if directions in ("factors", "all"):
        st = load_fa_steering()
        K = int(st["k"])
        Lfa = np.asarray([st["oblimin_loadings"][s] for s in st["slug_order"]])

        def top_fa(k, sign, m=3):
            o = np.argsort(Lfa[:, k - 1])
            idx = o[::-1][:m] if sign > 0 else o[:m]
            return ", ".join(names[i] for i in idx)

        fa_meta = {"solution": st["solution"], "k": K,
                   "best_goldberg_per_factor": st["best_goldberg_per_factor"],
                   "best_goldberg_congruence": st["best_goldberg_congruence"]}
        for k in range(1, K + 1):
            c = fa_coeffs(G, k, st=st, names=names)
            # expected per-trait projections <dW_i - mean dW, D> = (Gc c)_i:
            # the container's streamed build recomputes these and compares.
            comp_jobs.append({
                "direction": f"fa{k}", "alphas": COMPONENT_ALPHAS,
                "coeffs": c.tolist(), "names": names,
                "n_modules": N_MODULES, "s_bar": s_bar,
                "loadings": (Gc @ c).tolist(),
                "poles": (f"fa{k}- ({top_fa(k, -1)})",
                          f"fa{k}+ ({top_fa(k, +1)})"),
                "fa_report": {
                    "solution": st["solution"],
                    "best_goldberg": st["best_goldberg_per_factor"][k - 1],
                    "best_goldberg_congruence":
                        st["best_goldberg_congruence"][k - 1]},
            })

    meta = {
        "s_bar_mean_dW_norm": s_bar,
        "norms_min_max": [float(norms.min()), float(norms.max())],
        "trait_alphas": TRAIT_ALPHAS, "component_alphas": COMPONENT_ALPHAS,
        "probes": PROBES,
        "sign_convention": {
            f"pc{k}": {"positive": [names[i] for i in
                                    np.argsort(S[:, k - 1])[::-1][:5]],
                       "negative": [names[i] for i in
                                    np.argsort(S[:, k - 1])[:5]]}
            for k in (1, 2, 3)},
        "controls": {rname: random_coeffs(G, seed, report=True)[1]
                     for rname, seed in RANDOM_SEEDS.items()},
    }
    if fa_meta is not None:
        meta["fa"] = fa_meta
    return trait_jobs, comp_jobs, meta


def print_banner(trait_jobs, comp_jobs, trait_minutes, component_minutes,
                 budget):
    price = GPU_PRICE_PER_HOUR.get(GPU_TYPE)
    if price is None:
        raise SystemExit(f"no price known for gpu={GPU_TYPE}; refusing to estimate")
    n_tc = len(trait_jobs) * len(TRAIT_ALPHAS)
    n_cc = sum(len(j["alphas"]) for j in comp_jobs)
    n_cond = n_tc + n_cc
    n_gen = n_cond * len(PROBES)
    t_min = len(trait_jobs) * trait_minutes
    c_min = sum((component_minutes if j["direction"] != "base" else 5.0)
                for j in comp_jobs)
    gpu_min = t_min + c_min
    dollars = gpu_min / 60.0 * price
    print("=" * 72)
    print("PHASE 7 STEERING PLAN -- printed BEFORE any GPU is allocated")
    print("=" * 72)
    print(f"  app name                     : {APP_NAME}")
    print(f"  adapters (read)              : {ADAPTER_VOLUME} at /adapters")
    print(f"  output volume                : {STEER_VOLUME} at /steer/gen/")
    print(f"  trait jobs                   : {len(trait_jobs)}  "
          f"x {len(TRAIT_ALPHAS)} doses {TRAIT_ALPHAS}")
    dense_names = ", ".join(j["direction"] for j in comp_jobs)
    print(f"  dense jobs                   : {len(comp_jobs)}  "
          f"({dense_names}) x {len(COMPONENT_ALPHAS)} doses "
          f"{COMPONENT_ALPHAS} (base: 1 dose)")
    print(f"  conditions                   : {n_cond}")
    print(f"  generations                  : {n_cond} x {len(PROBES)} probes "
          f"= {n_gen}  (greedy, {MAX_NEW_TOKENS} new tok)")
    print(f"  gpu                          : {GPU_TYPE} @ ${price:.2f}/hr")
    print(f"  assumed wall-clock           : {trait_minutes:.1f} min/trait job, "
          f"{component_minutes:.0f} min/dense job, 5 min base")
    print(f"  ARITHMETIC                   : {len(trait_jobs)} x "
          f"{trait_minutes:.1f} + dense {c_min:.0f} = {gpu_min:.0f} GPU-min "
          f"= {gpu_min/60:.2f} GPU-hr")
    print(f"                                 {gpu_min/60:.2f} hr x "
          f"${price:.2f}/hr = ${dollars:.2f}")
    print(f"  declared budget              : ${budget:.2f}  (PC_PHASE_BUDGET)")
    if dollars > budget:
        print(f"  VERDICT: OVER BUDGET by ${dollars - budget:.2f} -- STOP")
        return False, dollars
    print(f"  VERDICT: within budget, ${budget - dollars:.2f} headroom")
    return True, dollars


@app.local_entrypoint()
def main(directions: str = "all", dry_run: bool = False, force: bool = False,
         trait_minutes: float = 2.5, component_minutes: float = 20.0):
    t0 = time.time()
    assert directions in ("traits", "components", "factors", "all"), directions

    # No default budget: an unset budget is a caller who has not said, and the
    # honest response is to refuse (see train_qwen35.py's print_plan history).
    env = os.environ.get("PC_PHASE_BUDGET")
    if env is None:
        raise SystemExit(
            "NO BUDGET DECLARED. Set PC_PHASE_BUDGET to this phase's ceiling "
            "before launching. Refusing to substitute a default.")
    budget = float(env)

    trait_jobs, comp_jobs, meta = build_jobs(directions)
    if force:
        for j in trait_jobs + comp_jobs:
            j["force"] = True

    ok, dollars = print_banner(trait_jobs, comp_jobs, trait_minutes,
                               component_minutes, budget)
    if not ok:
        raise SystemExit(f"estimate ${dollars:.2f} exceeds PC_PHASE_BUDGET "
                         f"${budget:.2f} -- STOPPED")
    if dry_run:
        print("dry_run: not launching")
        return

    os.makedirs(f"{RDIR}/steer134_gen", exist_ok=True)
    json.dump(meta, open(f"{RDIR}/steer134_gen/_meta.json", "w"), indent=1)

    out = []
    if comp_jobs:      # dense first: fewer jobs, catches shared faults early
        out += list(steer_dense.map(comp_jobs, order_outputs=True))
    if trait_jobs:
        out += list(steer_trait.map(trait_jobs, order_outputs=True))

    skipped = sum(1 for r in out if r.get("skipped"))
    usd = sum(r.get("usd_estimate", 0.0) for r in out)
    errs = [r for r in out if r.get("error")]
    print(f"\n{len(out)} jobs done ({skipped} fully skipped)  "
          f"estimated GPU cost ${usd:.2f}")
    if errs:
        print(f"WARNING: {len(errs)} jobs reported errors")

    # mirror the volume locally -- the local copy is what judge_steer134 reads
    files = fetch_gen.remote()
    for fn, content in files.items():
        with open(f"{RDIR}/steer134_gen/{fn}", "w") as f:
            f.write(content)
    print(f"mirrored {len(files)} condition files to results/steer134_gen/")

    json.dump({"meta": meta, "runs": out, "usd_estimate": usd,
               "wall_seconds": time.time() - t0},
              open(f"{RDIR}/steer134_gen/_runlog_{int(t0)}.json", "w"), indent=1)
    print(f"TOTAL wall {time.time()-t0:.0f}s")


# ===========================================================================
# selftest (local, no GPU, no modal call)
# ===========================================================================
def selftest():
    import numpy as np

    ok = True

    def ck(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("== probes ==")
    ck(len(PROBES) == 12, f"12 probes ({len(PROBES)})")
    ck(len(set(PROBES)) == 12, "all distinct")

    print("== gram + PCA ==")
    G, names, norms, s_bar = load_gram()
    ck(len(names) == 134, f"134 traits ({len(names)})")
    ck(abs(s_bar - float(norms.mean())) < 1e-12, f"s_bar = {s_bar:.6f}")
    lc, Uc, S, Gc = pca_from_gram(G)
    for k in (1, 2, 3):
        a = pc_coeffs(G, k)
        e = np.abs(Gc @ a - S[:, k - 1]).max()
        ck(abs(a @ G @ a - 1) < 1e-9, f"PC{k}: unit Gram norm")
        ck(abs(a.sum()) < 1e-10, f"PC{k}: coefficients sum to zero")
        ck(e < 1e-9, f"PC{k}: projection reproduces loadings (max {e:.1e})")
        o = np.argsort(S[:, k - 1])
        print(f"      PC{k}+ : {', '.join(names[i] for i in o[::-1][:5])}")
        print(f"      PC{k}- : {', '.join(names[i] for i in o[:5])}")

    print("== controls ==")
    for rname, seed in RANDOM_SEEDS.items():
        c, rep = random_coeffs(G, seed, report=True)
        ck(abs(c @ G @ c - 1) < 1e-9, f"{rname}: unit norm")
        ck(all(abs(v) < 1e-9
               for v in rep["cos_with_pc_after_deflation"].values()),
           f"{rname}: orthogonal to PC1-3 after deflation")
        pre = rep["cos_with_pc_before_deflation"][1]
        ck(abs(pre) > 0.1,
           f"{rname}: pre-deflation cos with PC1 = {pre:+.4f} (why we deflate)")
    c1 = random_coeffs(G, RANDOM_SEEDS["random1"])
    c2 = random_coeffs(G, RANDOM_SEEDS["random2"])
    ck(float(np.abs(c1 - c2).max()) > 1e-6, "random1 != random2")

    print("== fa directions ==")
    n_fa = 0
    if os.path.exists(FA_JSON):
        st = load_fa_steering()
        n_fa = int(st["k"])
        ck(list(st["slug_order"]) == names,
           "fa_qwen35.json slug order == npz order")
        n = len(names)
        one = np.ones((n, n)) / n
        Gc2 = G - one @ G - G @ one + one @ G @ one
        lc, Uc, S, _ = pca_from_gram(G)
        for k in range(1, n_fa + 1):
            c = fa_coeffs(G, k, st=st, names=names)
            ck(abs(c @ G @ c - 1) < 1e-9, f"fa{k}: unit Gram norm (raw metric)")
            ck(abs(c @ Gc2 @ c - 1) < 1e-9,
               f"fa{k}: unit Gram norm (double-centred metric)")
            ck(abs(c.sum()) < 1e-10, f"fa{k}: coefficients sum to zero")
            cos1 = float(pc_coeffs(G, 1) @ G @ c)
            o = np.argsort(np.array([st["oblimin_loadings"][nm][k - 1]
                                     for nm in names]))
            print(f"      fa{k} ({st['best_goldberg_per_factor'][k-1]} "
                  f"{st['best_goldberg_congruence'][k-1]:.2f}) cos(pc1)="
                  f"{cos1:+.3f}  +: "
                  f"{', '.join(names[i] for i in o[::-1][:4])}")
    else:
        ck(False, "results/fa_qwen35.json missing -- run analyse_fa_qwen35.py")

    print("== dose grid / names ==")
    ck(cond_name("trait_extraverted", 4) == "trait_extraverted_a+4.0",
       "trait condition name")
    ck(cond_name("pc1", -2) == "pc1_a-2.0", "pc condition name")
    ck(cond_name("fa1", 2) == "fa1_a+2.0", "fa condition name")
    ck(cond_name("base", 0) == "base_a+0.0", "base condition name")
    ck(0 not in TRAIT_ALPHAS and 0 not in COMPONENT_ALPHAS,
       "alpha=0 only via the shared base condition")
    n_cond = 134 * len(TRAIT_ALPHAS) + (5 + n_fa) * len(COMPONENT_ALPHAS) + 1
    print(f"      total conditions at full fan-out ({n_fa} fa dirs): {n_cond} "
          f"({n_cond * len(PROBES)} generations)")

    print("== trait scaling formula ==")
    # scaling_new = 2.0 * alpha * s_bar / norm_t must give effective delta
    # norm |alpha| * s_bar when the adapter's own norm is norm_t.
    nt = float(norms[0])
    for alpha in (-8, 2):
        s_new = LORA_SCALE * alpha * s_bar / nt
        eff = abs(s_new) / LORA_SCALE * nt
        ck(abs(eff - abs(alpha) * s_bar) < 1e-9,
           f"alpha={alpha:+d}: effective norm = |alpha|*s_bar")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
    print("run with:  PC_PHASE_BUDGET=<usd> modal run steer134_on_modal.py ...")
