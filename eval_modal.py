"""
Modal generation harness for the behavioural half of the persona-composition
experiment.

For a given MODEL CONFIG it generates deterministic responses to all 60 held-out
probe prompts (data/probe_prompts.json) and writes them to the Modal Volume
"persona-curvature-evals".

Config grammar
--------------
    base                        base Qwen2.5-3B-Instruct, no adapter
    single:<T>                  T in {O,C,E,A,N}; sugar for adapter:<T>
    adapter:<cond>              any trained condition dir on the adapter volume
                                (O, O_C, O_C_union, O_h1, O_s1, ...)
    sum:<X>+<Y>                 base weights + dW_X + dW_Y merged directly into
                                the base weight matrices
    scaled_sum:<X>+<Y>:<a>:<b>  base weights + a*dW_X + b*dW_Y

`single:<T>` is canonicalised to `adapter:<T>` so the two spellings share one
output file instead of paying for the same generation twice.

Delta-weight convention (verified, not assumed)
-----------------------------------------------
peft 0.14.0 `lora.Linear.get_delta_weight` returns

    transpose(B @ A, fan_in_fan_out) * scaling[adapter]

with `scaling = lora_alpha / r` (rsLoRA off) and, for these adapters,
fan_in_fan_out=False / use_dora=False / use_rslora=False (checked in
adapter_config.json). So dW = (alpha/r) * B @ A with shape [out, in], exactly
the shape of the base `weight` -- it is ADDED to the base weight, not
transposed. `_verify_scaling_against_peft` re-derives this at run time by
feeding the REAL adapter tensors through peft's own `get_delta_weight` and
requiring bit-exact agreement, so the container fails loudly if a peft upgrade
ever changes the convention.

Output
------
Modal Volume "persona-curvature-evals":
    /evals/<slug>.json = {"config":..., "responses":[{"prompt":..,"response":..}]}

Usage
-----
    modal run eval_modal.py --configs "base,single:O,sum:O+C"
    modal run eval_modal.py --configs "sum:O+C" --force
    modal run eval_modal.py --all-pairs           # every config the study needs
    python eval_modal.py --selftest               # local parser test, no Modal
"""

import json
import os
import re
import time

import modal

APP_NAME = "persona-curvature"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 3B in bf16 is ~6.2GB of weights; A10G (24GB) has ample headroom for a
# 200-token greedy decode at batch 12 and is the cheapest card that fits.
GPU_TYPE = os.environ.get("PC_EVAL_GPU", "A10G")

ADAPTER_VOLUME = "persona-curvature-adapters"
EVAL_VOLUME = "persona-curvature-evals"

# Must match train_modal.py. Asserted against each adapter_config.json at merge
# time rather than trusted.
LORA_R = 16
LORA_ALPHA = 32

# ---- generation settings (identical for every config; changing these
# invalidates cross-config comparisons)
MAX_NEW_TOKENS = 200
BATCH_SIZE = 12
GEN_SEED = 0

TRAITS = ["O", "C", "E", "A", "N"]
PAIRS = [(a, b) for i, a in enumerate(TRAITS) for b in TRAITS[i + 1:]]

HERE = os.path.dirname(os.path.abspath(__file__))
PROBES_PATH = os.path.join(HERE, "data", "probe_prompts.json")

app = modal.App(APP_NAME)

adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)
eval_vol = modal.Volume.from_name(EVAL_VOLUME, create_if_missing=True)


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


# --------------------------------------------------------------------------
# config parsing
# --------------------------------------------------------------------------
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE_SUM = re.compile(r"^sum:(?P<x>[^+:]+)\+(?P<y>[^+:]+)$")
_RE_SCALED = re.compile(
    rf"^scaled_sum:(?P<x>[^+:]+)\+(?P<y>[^+:]+):(?P<a>{_NUM}):(?P<b>{_NUM})$"
)


def canonical(config: str) -> str:
    """`single:<T>` and `adapter:<T>` name the same weights; collapse to one."""
    c = config.strip()
    if c.startswith("single:"):
        t = c[len("single:"):].strip()
        if t not in TRAITS:
            raise ValueError(
                f"single:{t!r} -- expected one of {TRAITS}; use adapter:{t} for "
                f"a non-single condition"
            )
        return f"adapter:{t}"
    return c


def parse_config(config: str) -> dict:
    """Return {kind, adapters:[(name, coeff), ...]} or raise ValueError."""
    c = canonical(config)
    if c == "base":
        return {"kind": "base", "adapters": []}
    if c.startswith("adapter:"):
        name = c[len("adapter:"):].strip()
        if not name:
            raise ValueError(f"empty adapter name in {config!r}")
        return {"kind": "adapter", "adapters": [(name, 1.0)]}
    m = _RE_SUM.match(c)
    if m:
        return {"kind": "sum",
                "adapters": [(m.group("x").strip(), 1.0),
                             (m.group("y").strip(), 1.0)]}
    m = _RE_SCALED.match(c)
    if m:
        return {"kind": "scaled_sum",
                "adapters": [(m.group("x").strip(), float(m.group("a"))),
                             (m.group("y").strip(), float(m.group("b")))]}
    raise ValueError(
        f"unparseable config {config!r}; expected base | single:<T> | "
        f"adapter:<cond> | sum:<X>+<Y> | scaled_sum:<X>+<Y>:<a>:<b>"
    )


def slug(config: str) -> str:
    """Filesystem-safe, reversible-by-eye name for a config."""
    return canonical(config).replace(":", "__")


def unslug(s: str) -> str:
    return s.replace("__", ":")


# --------------------------------------------------------------------------
# delta-weight reconstruction (runs inside the container)
# --------------------------------------------------------------------------
# base_model.model.<module path>.lora_(A|B)[.<adapter name>].weight
_RE_LORA_KEY = re.compile(
    r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>A|B)"
    r"(?:\.(?P<adapter>[^.]+))?\.weight$"
)


def _parse_lora_keys(state: dict) -> dict:
    """{module_path: {"A": tensor, "B": tensor}} from an adapter state dict.

    Fails loudly on any key it cannot classify -- a silently dropped key is a
    silently weaker merge.
    """
    mods: dict = {}
    unknown = []
    for k, v in state.items():
        m = _RE_LORA_KEY.match(k)
        if not m:
            # lora_embedding_A/B and biases are absent for this config; anything
            # else unrecognised is a real problem.
            unknown.append(k)
            continue
        mods.setdefault(m.group("mod"), {})[m.group("ab")] = v
    if unknown:
        raise RuntimeError(f"unrecognised keys in adapter state dict: {unknown[:8]}")
    bad = [m for m, d in mods.items() if set(d) != {"A", "B"}]
    if bad:
        raise RuntimeError(f"modules missing an A or B tensor: {bad[:8]}")
    return mods


def _verify_scaling_against_peft(mods: dict, r: int, alpha: int, n_check: int = 4):
    """Push REAL adapter tensors through peft's own get_delta_weight and require
    bit-exact agreement with (alpha/r) * B @ A. Returns a description string."""
    import torch
    import torch.nn as nn
    from peft.tuners.lora import Linear as LoraLinear

    # pick modules with distinct shapes (square attn, wide mlp, tall mlp, kv)
    by_shape = {}
    for name, d in mods.items():
        key = (d["A"].shape[1], d["B"].shape[0])
        by_shape.setdefault(key, name)
    picked = list(by_shape.items())[:n_check]

    lines = []
    for (f_in, f_out), name in picked:
        A = mods[name]["A"].float()
        B = mods[name]["B"].float()
        base = nn.Linear(f_in, f_out, bias=False)
        layer = LoraLinear(
            base, adapter_name="default", r=r, lora_alpha=alpha,
            lora_dropout=0.0, fan_in_fan_out=False, init_lora_weights=True,
        )
        assert layer.scaling["default"] == alpha / r, (
            f"peft scaling is {layer.scaling['default']}, expected {alpha}/{r}"
        )
        layer.lora_A["default"].weight.data = A.clone()
        layer.lora_B["default"].weight.data = B.clone()
        ref = layer.get_delta_weight("default")
        mine = (alpha / r) * (B @ A)
        assert ref.shape == base.weight.shape, (
            f"peft delta shape {tuple(ref.shape)} != weight shape "
            f"{tuple(base.weight.shape)} -- transpose convention changed"
        )
        assert torch.equal(ref, mine), (
            f"{name}: manual delta != peft get_delta_weight "
            f"(max |diff| {(ref - mine).abs().max().item():.3e})"
        )
        lines.append(f"{name.split('.')[-1]}[{f_in}->{f_out}]")
    return (f"verified dW=(alpha/r)*B@A bit-exact vs peft get_delta_weight, "
            f"scaling={alpha/r}, shape=[out,in], on {len(picked)} module shapes: "
            + ", ".join(lines))


def _load_adapter(cond: str):
    """Return (modules, r, alpha) for /adapters/<cond>."""
    from safetensors.torch import load_file

    d = f"/adapters/{cond}"
    st = f"{d}/adapter_model.safetensors"
    cfgp = f"{d}/adapter_config.json"
    if not os.path.exists(st):
        have = sorted(os.listdir("/adapters")) if os.path.isdir("/adapters") else []
        raise FileNotFoundError(
            f"{st} missing -- condition {cond!r} is not trained yet. "
            f"volume has: {have}"
        )
    cfg = json.load(open(cfgp))
    # These would silently change the delta formula if they ever flipped.
    for k, want in (("use_rslora", False), ("use_dora", False),
                    ("fan_in_fan_out", False)):
        assert cfg.get(k, False) == want, f"{cond}: {k}={cfg.get(k)}, expected {want}"
    r, alpha = int(cfg["r"]), int(cfg["lora_alpha"])
    assert (r, alpha) == (LORA_R, LORA_ALPHA), (
        f"{cond}: r={r} alpha={alpha}, expected {LORA_R}/{LORA_ALPHA}"
    )
    return _parse_lora_keys(load_file(st)), r, alpha


def merge_deltas_into(model, specs: list, verbose: bool = True) -> dict:
    """Add sum(coeff * dW) from each named adapter straight into the base
    weights, in place. Every adapter module must map onto an existing base
    parameter of matching shape or we raise -- a name mismatch that merged
    nothing would look exactly like a clean null result.

    Returns a stats dict including Frobenius norms of the total change.
    """
    import torch

    params = dict(model.named_parameters())
    stats = {"adapters": [], "matched_modules": 0, "verification": None}

    # ---- load every adapter first, and check the name mapping BEFORE touching
    # a single weight. A silent name mismatch that merged nothing would look
    # exactly like a clean null result, so this is a hard failure, not a warn.
    loaded = []
    for cond, coeff in specs:
        mods, r, alpha = _load_adapter(cond)
        if stats["verification"] is None:
            stats["verification"] = _verify_scaling_against_peft(mods, r, alpha)
            if verbose:
                print(f"  [scaling] {stats['verification']}", flush=True)

        missing, shape_bad = [], []
        for mod, ab in mods.items():
            pname = f"{mod}.weight"
            p = params.get(pname)
            if p is None:
                missing.append(pname)
                continue
            want = (ab["B"].shape[0], ab["A"].shape[1])
            if tuple(p.shape) != want:
                shape_bad.append((pname, want, tuple(p.shape)))
        if missing or shape_bad:
            raise RuntimeError(
                f"MERGE NAME MISMATCH for adapter {cond!r}: "
                f"{len(missing)} adapter modules had no base parameter "
                f"(e.g. {missing[:5]}); {len(shape_bad)} shape mismatches "
                f"(e.g. {shape_bad[:3]}). Refusing to generate from a model "
                f"that merged nothing."
            )
        matched = len(mods)
        loaded.append((cond, coeff, mods, alpha / r))
        stats["adapters"].append(
            {"condition": cond, "coeff": coeff, "modules_matched": matched,
             "r": r, "alpha": alpha, "scaling": alpha / r}
        )
        stats["matched_modules"] += matched
        if verbose:
            print(f"  [merge] {cond}: coeff={coeff} matched {matched}/{len(mods)} "
                  f"modules onto base parameters", flush=True)

    # ---- merge module by module, accumulating the TOTAL delta across adapters
    # before adding it. Doing it this way (rather than per adapter) means
    # delta_frobenius is genuinely ||W_merged - W_base||_F and not the
    # incorrect sqrt(sum_i ||dW_i||^2), which differs whenever two adapters
    # touch the same module -- which is always, here.
    all_mods = sorted({m for _c, _k, mods, _s in loaded for m in mods})
    delta_sq = 0.0
    base_sq = 0.0
    with torch.no_grad():
        for mod in all_mods:
            pname = f"{mod}.weight"
            p = params[pname]
            dev = p.device
            total = None
            for _cond, coeff, mods, scaling in loaded:
                ab = mods.get(mod)
                if ab is None:
                    continue
                # Reconstruct on the model's own device in fp32: a 252-module
                # rebuild on CPU costs minutes per adapter, seconds on GPU.
                A = ab["A"].to(dev, torch.float32)
                B = ab["B"].to(dev, torch.float32)
                d = (coeff * scaling) * (B @ A)
                total = d if total is None else total.add_(d)
            assert total is not None and total.shape == p.shape
            base_sq += float(p.to(torch.float32).pow(2).sum())
            delta_sq += float(total.pow(2).sum())
            p.add_(total.to(p.dtype))

    stats["delta_frobenius"] = delta_sq ** 0.5
    stats["base_weight_frobenius"] = base_sq ** 0.5
    stats["n_params_touched"] = len(all_mods)
    return stats


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/adapters": adapter_vol, "/evals": eval_vol},
    timeout=60 * 60,
)
def eval_config(config: str, prompts: list, force: bool = False) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    cfg = parse_config(config)
    name = canonical(config)
    out_path = f"/evals/{slug(config)}.json"
    print(f"=== config={name} kind={cfg['kind']} gpu={GPU_TYPE} "
          f"prompts={len(prompts)} ===", flush=True)

    eval_vol.reload()
    if os.path.exists(out_path) and not force:
        try:
            prev = json.load(open(out_path))
            if len(prev.get("responses", [])) == len(prompts):
                print(f"  already done ({out_path}); skipping", flush=True)
                return {"config": name, "skipped": True,
                        "n": len(prev["responses"]), "wall_seconds": 0.0}
        except Exception:
            pass
    adapter_vol.reload()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched decoder-only gen

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    model.config.use_cache = True

    merge_stats = None
    if cfg["kind"] == "adapter":
        # Canonical peft path for a single trained adapter: load then fold in.
        from peft import PeftModel

        cond = cfg["adapters"][0][0]
        if not os.path.exists(f"/adapters/{cond}/adapter_model.safetensors"):
            have = sorted(os.listdir("/adapters")) if os.path.isdir("/adapters") else []
            raise FileNotFoundError(
                f"adapter {cond!r} not trained yet; volume has: {have}")
        before = {n: p.detach().float().pow(2).sum().item()
                  for n, p in model.named_parameters()
                  if n.endswith("q_proj.weight")}
        model = PeftModel.from_pretrained(model, f"/adapters/{cond}")
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_A" in n)
        assert n_lora > 0, f"PeftModel loaded no LoRA A tensors for {cond}"
        model = model.merge_and_unload()
        after = {n: p.detach().float().pow(2).sum().item()
                 for n, p in model.named_parameters()
                 if n.endswith("q_proj.weight")}
        changed = sum(1 for k in before if abs(before[k] - after.get(k, 0.0)) > 0)
        assert changed > 0, (
            f"merge_and_unload changed no q_proj weights for {cond} -- no-op merge")
        merge_stats = {"adapters": [{"condition": cond, "coeff": 1.0}],
                       "matched_modules": n_lora,
                       "path": "peft PeftModel.merge_and_unload",
                       "q_proj_weights_changed": f"{changed}/{len(before)}"}
        print(f"  [merge] peft merge_and_unload({cond}): {n_lora} lora modules, "
              f"{changed}/{len(before)} q_proj weights changed", flush=True)
    elif cfg["kind"] in ("sum", "scaled_sum"):
        merge_stats = merge_deltas_into(model, cfg["adapters"])
        rel = (merge_stats["delta_frobenius"]
               / max(merge_stats["merged_weight_frobenius"], 1e-9))
        merge_stats["relative_frobenius"] = rel
        print(f"  [merge] ||sum of dW||_F = {merge_stats['delta_frobenius']:.4f} "
              f"over {merge_stats['n_params_touched']} weight matrices "
              f"(relative to merged ||W||_F: {rel:.6f})", flush=True)
        assert merge_stats["delta_frobenius"] > 0, "merged delta is exactly zero"

    model.eval()
    torch.manual_seed(GEN_SEED)
    torch.cuda.manual_seed_all(GEN_SEED)

    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}],
                                tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    responses = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        for row in gen:
            responses.append(tok.decode(row, skip_special_tokens=True).strip())
        print(f"  generated {len(responses)}/{len(texts)} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    assert len(responses) == len(prompts)
    wall = time.time() - t0
    payload = {
        "config": name,
        "config_raw": config,
        "kind": cfg["kind"],
        "base_model": BASE_MODEL,
        "gpu": GPU_TYPE,
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                       "batch_size": BATCH_SIZE, "seed": GEN_SEED,
                       "padding_side": "left"},
        "merge": merge_stats,
        "wall_seconds": wall,
        "n_prompts": len(prompts),
        "responses": [{"prompt": p, "response": r}
                      for p, r in zip(prompts, responses)],
    }
    os.makedirs("/evals", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    eval_vol.commit()

    n_empty = sum(1 for r in responses if not r.strip())
    mean_len = sum(len(r) for r in responses) / len(responses)
    print(f"=== {name}: {len(responses)} responses, mean {mean_len:.0f} chars, "
          f"{n_empty} empty, wall={wall:.1f}s -> {out_path} ===", flush=True)
    return {"config": name, "skipped": False, "n": len(responses),
            "wall_seconds": wall, "mean_chars": mean_len, "n_empty": n_empty,
            "merge": merge_stats}


# --------------------------------------------------------------------------
# config sets
# --------------------------------------------------------------------------
N_STUDY_CONFIGS = 45  # 1 base + 5 singles + 10 comp + 10 union + 10 sums
                      #   + 6 half-splits + 3 reseeds (the last 9 are the
                      #   noise floor, without which the distances mean nothing)


def all_study_configs() -> list:
    """Every config the behavioural analysis needs (45)."""
    cfgs = ["base"]
    cfgs += [f"single:{t}" for t in TRAITS]                       # 5
    cfgs += [f"adapter:{x}_{y}" for x, y in PAIRS]                # 10 compositional
    cfgs += [f"adapter:{x}_{y}_union" for x, y in PAIRS]          # 10 union
    cfgs += [f"sum:{x}+{y}" for x, y in PAIRS]                    # 10 sums
    # noise floor: half-splits and reseeds
    cfgs += [f"adapter:{t}_h{h}" for t in "OCE" for h in (1, 2)]  # 6
    cfgs += [f"adapter:{t}_s1" for t in "OCE"]                    # 3
    return [canonical(c) for c in cfgs]


def _selftest() -> int:
    cases = [
        ("base", "base", []),
        ("single:O", "adapter", [("O", 1.0)]),
        ("adapter:O_C", "adapter", [("O_C", 1.0)]),
        ("adapter:O_C_union", "adapter", [("O_C_union", 1.0)]),
        ("sum:O+C", "sum", [("O", 1.0), ("C", 1.0)]),
        ("sum:O_h1+O_h2", "sum", [("O_h1", 1.0), ("O_h2", 1.0)]),
        ("scaled_sum:O+C:0.5:1.25", "scaled_sum", [("O", 0.5), ("C", 1.25)]),
        ("scaled_sum:O+C:-1:2e-1", "scaled_sum", [("O", -1.0), ("C", 0.2)]),
    ]
    fails = 0
    print("=" * 84)
    for cfg, kind, ads in cases:
        got = parse_config(cfg)
        ok = got["kind"] == kind and got["adapters"] == ads
        fails += not ok
        print(f"{cfg:<30}{got['kind']:<12}{str(got['adapters']):<32}"
              f"{'PASS' if ok else 'FAIL'}")
    # canonicalisation collapses the two spellings onto one file
    assert slug("single:O") == slug("adapter:O") == "adapter__O"
    assert slug("sum:O+C") == "sum__O+C"
    assert unslug(slug("scaled_sum:O+C:0.5:0.5")) == "scaled_sum:O+C:0.5:0.5"
    print("slug/canonical round-trip                                          PASS")
    for bad in ("", "sum:O", "sum:O+C+E", "single:Q", "adapter:", "nonsense",
                "scaled_sum:O+C:0.5"):
        try:
            parse_config(bad)
        except ValueError:
            print(f"rejected {bad!r:<40}                          PASS")
        else:
            fails += 1
            print(f"NOT rejected {bad!r:<40}                      FAIL")
    cfgs = all_study_configs()
    assert len(cfgs) == len(set(cfgs)) == N_STUDY_CONFIGS, (
        len(cfgs), len(set(cfgs)))
    for c in cfgs:
        parse_config(c)
    print(f"all_study_configs: {len(cfgs)} unique, all parse               PASS")
    print("=" * 84)
    print("SELFTEST FAILURES:", fails)
    return fails


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(configs: str = "", force: bool = False, all_pairs: bool = False):
    """
    configs:   comma-separated config strings, e.g. "base,single:O,sum:O+C"
    all_pairs: run the full 36-config study set instead
    force:     regenerate even if the eval volume already has the file
    """
    if all_pairs:
        names = all_study_configs()
    else:
        names = [c.strip() for c in configs.split(",") if c.strip()]
    if not names:
        raise SystemExit("no configs given (use --configs or --all-pairs)")

    names = [canonical(n) for n in names]
    seen, deduped = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    if len(deduped) != len(names):
        print(f"(collapsed {len(names) - len(deduped)} duplicate config(s))")
    names = deduped

    parsed = {n: parse_config(n) for n in names}  # fail fast on typos
    prompts = json.load(open(PROBES_PATH))
    assert isinstance(prompts, list) and prompts, f"{PROBES_PATH} unusable"
    print(f"{len(prompts)} probe prompts from {PROBES_PATH}")
    print(f"{len(names)} config(s):")
    for n in names:
        p = parsed[n]
        print(f"  {n:<32} {p['kind']:<12} {p['adapters']}")

    # resume: skip configs already complete on the volume
    done = set()
    if not force:
        try:
            for e in eval_vol.listdir("/"):
                p = e.path.strip("/")
                if p.endswith(".json"):
                    done.add(p[:-5])
        except Exception as e:
            print(f"(could not list eval volume: {e})")
    todo = [n for n in names if slug(n) not in done]
    skipped = [n for n in names if slug(n) in done]
    if skipped:
        print(f"skipping (already on volume): {', '.join(skipped)}")
    if not todo:
        print("nothing to do.")
        return

    # Adapters that do not exist yet would fail deep inside a GPU container;
    # catch it here instead.
    trained = set()
    try:
        for e in adapter_vol.listdir("/", recursive=True):
            p = e.path.strip("/")
            if p.endswith("/adapter_model.safetensors"):
                trained.add(p[: -len("/adapter_model.safetensors")])
    except Exception as e:
        print(f"(could not list adapter volume: {e})")
    blocked = []
    for n in list(todo):
        need = {c for c, _ in parsed[n]["adapters"]}
        miss = sorted(need - trained) if trained else []
        if miss:
            blocked.append((n, miss))
            todo.remove(n)
    if blocked:
        print("\nNOT RUNNABLE YET (adapters still training):")
        for n, miss in blocked:
            print(f"  {n:<32} needs {miss}")
    if not todo:
        print("nothing runnable.")
        return

    print(f"\nrunning {len(todo)} config(s) in parallel on {GPU_TYPE}: "
          f"{', '.join(todo)}")
    t0 = time.time()
    results = list(eval_config.map(todo, kwargs={"prompts": prompts,
                                                 "force": force}))
    total = time.time() - t0

    print("\n" + "=" * 92)
    print(f"{'config':<34}{'n':>5}{'chars':>8}{'empty':>7}{'wall_s':>10}"
          f"{'||dW||_F':>12}")
    print("-" * 92)
    for r in results:
        mg = r.get("merge") or {}
        fro = mg.get("delta_frobenius")
        print(f"{r['config']:<34}{r['n']:>5}{r.get('mean_chars', 0):>8.0f}"
              f"{r.get('n_empty', 0):>7}{r['wall_seconds']:>10.1f}"
              f"{(f'{fro:.3f}' if fro else '-'):>12}")
    print("-" * 92)
    print(f"total wall (parallel): {total:.1f}s over {len(todo)} container(s)")
    print(f"next: ~/cartovenv/bin/python fetch_evals.py")
    print("=" * 92)


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    raise SystemExit(
        "run me with:  modal run eval_modal.py --configs 'base,single:O,sum:O+C'\n"
        "or:           python eval_modal.py --selftest"
    )
