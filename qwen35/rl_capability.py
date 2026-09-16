#!/usr/bin/env python3
"""Capability-only RLVR on base Qwen3.5-4B, in the persona zoo's LoRA geometry.

THE QUESTION
------------
Every adapter in the zoo teaches a personality trait.  This one teaches nothing
about personality at all: the reward is whether a maths answer is correct.  If
its weight delta still lands somewhere legible in the personality space the zoo
defines -- loads on a principal component, moves a judged Big Five scale -- then
capability training displaces personality as a side effect, and the geometry we
built from traits can measure it.

Two arms:
  math  -- allenai/Dolci-RL-Zero-Math-7B, SymPy-style exact match on a scalar.
           Scanned over all 13,314 rows: persona-regex hit rate 0.18%, all 24
           hits read, zero true positives.  Capabilities-only.
  if    -- allenai/Dolci-RL-Zero-IF-7B, the STYLE CONTRAST.  Fully verifiable
           and entirely about surface form: 55.7% of its constraint instances
           govern diction, casing, punctuation or opening/closing wording.

COMPARABILITY IS THE WHOLE POINT
--------------------------------
The adapter must be sketchable into the existing space, so the LoRA config is
not chosen here -- it is READ OFF an existing zoo adapter: the same 248 module
names, r=64, alpha=128, plain LoRA.  The bilinear sketch is seeded per module
name, so an adapter over the same modules projects into the same coordinates
with no new plumbing.

WHY GRPO AND NOT REJECTION-SAMPLING SFT
---------------------------------------
Pre-flight (analysis/rl_preflight.json, 300 problems x k=8 on the base model):
101/300 problems have 0 < pass@8 < 8, i.e. 33.7% carry a non-zero GRPO
advantage.  Above the 15% line registered before the measurement, so GRPO it is.
The pre-flight also found boxed compliance of only 0.30 at a 3072-token cap with
mean completion 2674 tokens -- most rollouts run out of budget before answering,
so the true solve rate is understated and 33.7% is a floor.

Checkpoints are saved every SAVE_EVERY steps and each is sketched, so the run
produces a TRAJECTORY through personality space against the reward curve, not
just an endpoint.
"""
import json
import os

import modal

BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_R, LORA_ALPHA = 64, 128          # identical to the zoo
SEED = 0
SAVE_EVERY = 25
K = 32                                # sketch width, matches sketch_adapters.py

app = modal.App("pc-qwen35-phase10-rlcap")
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep")
rl_vol = modal.Volume.from_name("pc-qwen35-rl", create_if_missing=True)
VOLS = {"/adapters": sweep_vol, "/rl": rl_vol}

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", "transformers==5.15.1", "peft==0.20.0",
                 "accelerate==1.14.0", "safetensors", "hf_transfer", "numpy<3",
                 "pyarrow", "datasets", "huggingface_hub")
    # trl 1.12 declares vllm<=0.27.1; unpinned pip gave 0.28.0 and GRPOTrainer
    # failed to import (NCCLTrainerSendWeightsArgs missing from
    # vllm.distributed.weight_transfer.nccl_engine).  0.27 is also the version
    # oct_stage2.py runs against, so this pin costs nothing.
    .pip_install("vllm==0.27.1")
    .pip_install("trl==1.12.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "VLLM_USE_V1": "1", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_file(os.path.abspath(__file__), "/root/rl_capability.py", copy=True)
)

SUFFIX = "\n\nReason step by step, and put your final answer within \\boxed{}."


# ------------------------------------------------------- shared, no imports --
def _norm_answer(s):
    import re
    if s is None:
        return None
    s = s.strip()
    for a, b in (("\\!", ""), ("\\,", ""), ("\\ ", ""), ("\\$", ""), ("$", ""),
                 ("\\%", ""), ("%", ""), ("\\left", ""), ("\\right", ""),
                 ("^{\\circ}", ""), ("^\\circ", ""), (" ", ""), ("\\text{", "{"),
                 ("dfrac", "frac"), ("tfrac", "frac")):
        s = s.replace(a, b)
    s = re.sub(r"\\mbox\{.*?\}", "", s)
    s = s.rstrip(".").lstrip("{").rstrip("}")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    if re.fullmatch(r"-?[\d,]+", s):
        s = s.replace(",", "")
    return s


def _last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for p in range(j, len(text)):
        if text[p] == "{":
            depth += 1
        elif text[p] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:p]
    return None


def _zoo_target_modules():
    """The 248 module names, read off a zoo adapter rather than re-derived.

    Re-deriving risks a different set, and the sketch is keyed by module name --
    a different set means the adapter cannot be projected into the existing
    space at all.
    """
    from safetensors import safe_open
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="np") as f:
        mods = sorted({k.split(".lora_A.")[0].split(".lora_B.")[0]
                       .removeprefix("base_model.model.") for k in f.keys()})
    return mods


def _run_asserts(code, asserts, timeout=6):
    """Execute a candidate solution against its assert list, out of process.

    Model-generated code runs in a subprocess with a wall-clock timeout inside an
    ephemeral Modal container that has no volume mounts of its own beyond the
    read paths this app needs. That is the standard RLVR arrangement; the point
    of the subprocess is that an infinite loop or a segfault costs one rollout
    rather than the run.
    """
    import subprocess
    import sys
    import tempfile
    src = code + "\n\n" + "\n".join(asserts) + "\nprint('__OK__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout)
        return "__OK__" in r.stdout
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _extract_code(text):
    """Last fenced block, else the whole completion."""
    import re
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m[-1] if m else text


def _asserts_of(gt):
    import json as _j
    s = gt[0] if isinstance(gt, (list, tuple)) and gt else gt
    try:
        v = _j.loads(s)
        return v if isinstance(v, list) and v and str(v[0]).startswith("assert") else None
    except Exception:
        return None


@app.function(image=image, volumes=VOLS, cpu=2.0, timeout=60 * 20,
              secrets=[modal.Secret.from_name("hf-token")])
def probe() -> dict:
    """Answer the questions the run depends on before spending on it."""
    import torch
    import transformers
    import trl
    from trl import GRPOConfig, GRPOTrainer          # noqa: F401
    import vllm
    # What does the model TRL will load actually call its linear layers?  The
    # first launch died with all 248 zoo names "not found in the base model",
    # so the prefix differs -- Qwen3.5-4B resolves to a composite
    # Qwen3_5ForConditionalGeneration and the text tower is nested.
    import torch as _t
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    live = [n for n, mo in m.named_modules() if isinstance(mo, _t.nn.Linear)]
    mods = _zoo_target_modules()
    zsuf = {n.split("layers.", 1)[-1] for n in mods}
    lsuf = {n.split("layers.", 1)[-1]: n for n in live if "layers." in n}
    return {"model_class": type(m).__name__,
            "n_live_linear": len(live),
            "live_sample": live[:3],
            "zoo_sample": mods[:3],
            "suffix_match": len(zsuf & set(lsuf)),
            "zoo_unmatched": sorted(zsuf - set(lsuf))[:5],
            "trl": trl.__version__, "transformers": transformers.__version__,
            "vllm": vllm.__version__, "torch": torch.__version__,
            "n_zoo_modules": len(mods), "sample_modules": mods[:3],
            "grpo_config_fields": sorted(GRPOConfig.__dataclass_fields__)}


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=60 * 120,
              secrets=[modal.Secret.from_name("hf-token")])
def score(n_problems: int = 3000, k: int = 4, max_tokens: int = 2048) -> dict:
    """Find the problems that carry a GRPO gradient, at the budget we will train at.

    Two reasons this pass is worth its ~$4.  Only problems with 0 < pass@k < k
    give a non-zero advantage, and the pre-flight put those at 33.7% -- so
    training on unfiltered rows spends two thirds of the compute on groups that
    contribute nothing.  And scoring at the SAME max_tokens the trainer will use
    keeps out problems the base model cannot finish in budget, which is what
    would otherwise turn the reward into a length policy.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    p = hf_hub_download("allenai/Dolci-RL-Zero-Math-7B",
                        "data/train-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p)
    prompts_all = t.column("prompt").to_pylist()
    gt_all = t.column("ground_truth").to_pylist()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(prompts_all), size=min(n_problems, len(prompts_all)), replace=False)

    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompts_all[i] + SUFFIX}],
                                     tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for i in idx]
    llm = LLM(model=snap, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=max_tokens + 1024, seed=SEED)
    outs = llm.generate(texts, SamplingParams(n=k, temperature=1.0, top_p=1.0,
                                              max_tokens=max_tokens, seed=SEED))

    rows = []
    for slot, o in enumerate(outs):
        i = int(idx[slot])
        gold = _norm_answer(str(gt_all[i]))
        got = [_norm_answer(_last_boxed(c.text)) for c in o.outputs]
        rows.append({"i": i, "prompt": prompts_all[i], "ground_truth": str(gt_all[i]),
                     "pass_k": sum(1 for g in got if g is not None and g == gold), "k": k,
                     "trunc": sum(1 for c in o.outputs if c.finish_reason == "length")})

    learn = [r for r in rows if 0 < r["pass_k"] < k]
    os.makedirs("/rl", exist_ok=True)
    with open("/rl/learnable_math.json", "w") as f:
        json.dump(learn, f)
    rl_vol.commit()
    return {"scored": len(rows), "learnable": len(learn),
            "rate": len(learn) / len(rows),
            "truncation_rate": sum(r["trunc"] for r in rows) / (len(rows) * k),
            "hist": {str(v): sum(1 for r in rows if r["pass_k"] == v) for v in range(k + 1)}}


CODE_SUFFIX = ("\n\nWrite the complete function in a single Python code block. "
               "Define every name you use.")


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=60 * 120,
              secrets=[modal.Secret.from_name("hf-token")])
def score_code(n_problems: int = 3000, k: int = 4, max_tokens: int = 1024) -> dict:
    """Same filter as the maths pass, against executed unit tests.

    Code completions are far shorter than maths ones, so the cap is 1024 rather
    than 2048 and truncation is much less of a threat to the reward signal.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows_all = []
    for sh in ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"):
        t = pq.read_table(hf_hub_download("allenai/Dolci-RL-Zero-Code-7B", sh,
                                          repo_type="dataset"))
        pr = t.column("prompt").to_pylist()
        gt = t.column("ground_truth").to_pylist()
        for p, g in zip(pr, gt):
            a = _asserts_of(g)
            if a:
                rows_all.append({"prompt": p.removeprefix("user: "), "asserts": a})
    print(f"[code] {len(rows_all)} assert-style problems", flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows_all), size=min(n_problems, len(rows_all)), replace=False)
    snap = snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])
    tok = AutoTokenizer.from_pretrained(snap)
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": rows_all[i]["prompt"] + CODE_SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for i in idx]
    llm = LLM(model=snap, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=max_tokens + 1024, seed=SEED)
    outs = llm.generate(texts, SamplingParams(n=k, temperature=1.0, top_p=1.0,
                                              max_tokens=max_tokens, seed=SEED))
    keep, hist = [], {v: 0 for v in range(k + 1)}
    for slot, o in enumerate(outs):
        r = rows_all[int(idx[slot])]
        n_ok = sum(1 for c in o.outputs if _run_asserts(_extract_code(c.text), r["asserts"]))
        hist[n_ok] += 1
        if 0 < n_ok < k:
            keep.append(r)
    with open("/rl/learnable_code.json", "w") as f:
        json.dump(keep, f)
    rl_vol.commit()
    return {"scored": len(outs), "learnable": len(keep), "rate": len(keep) / len(outs),
            "hist": {str(a): b for a, b in hist.items()}}


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=60 * 60 * 23,   # Modal's ceiling is 24h; 500 steps at ~130s/step is ~18h
              secrets=[modal.Secret.from_name("hf-token")])
def train(steps: int = 500, num_generations: int = 12, grad_accum: int = 6,
          micro_batch: int = 4, mix: str = "math",
          max_completion: int = 2048, lr: float = 5e-5, tag: str = "math") -> dict:
    """GRPO on the filtered Dolci-Math set, in the zoo's LoRA geometry.

    Choices worth stating rather than burying:

    * beta = 0.0, i.e. NO KL anchor to the base model.  A KL term is exactly a
      force holding the policy near base, and "how far does capability RL move
      personality when nothing holds it back" is the question.  Anchoring would
      be a confound we would then have to reason around.  This diverges from
      Olmo's recipe; a KL-anchored arm is the natural follow-up.
    * mask_truncated_completions = True.  Truncated rollouts score zero for
      running out of budget rather than for being wrong, so learning from them
      teaches brevity.  Masking removes them from the loss.  Measured on the
      filtered set at this cap, 48.5% of rollouts truncate (against 78.9% over
      unfiltered rows -- the filter earns its cost here), so num_generations is
      12 rather than 8: after masking that leaves about six live rollouts per
      group, which is the smallest group worth computing an advantage over.
      Only 4 of the 587 problems truncate on every rollout.
    * target modules are READ OFF a zoo adapter, not re-derived, so the delta
      projects into the existing personality space with no new plumbing.  This
      is also why the policy is the text-only Qwen3_5ForCausalLM rather than the
      composite VL class: the composite fuses mlp.gate_proj and mlp.up_proj into
      one gate_up_proj, and a LoRA over fused projections is a different module
      set that could not be projected into the zoo geometry at all.
    """
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Rows carry their own domain so one reward function can dispatch. Mixing at
    # the ROW level rather than alternating batches keeps every gradient step a
    # mixture, which is the point: a single narrow reward may simply be too thin
    # a pressure to move anything as diffuse as persona.
    rows = []
    if "math" in mix:
        rows += [{"kind": "math", "prompt": r["prompt"] + SUFFIX,
                  "target": json.dumps(r["ground_truth"])}
                 for r in json.load(open("/rl/learnable_math.json"))]
    if "code" in mix:
        rows += [{"kind": "code", "prompt": r["prompt"] + CODE_SUFFIX,
                  "target": json.dumps(r["asserts"])}
                 for r in json.load(open("/rl/learnable_code.json"))]
    if not rows:
        raise RuntimeError(f"mix={mix!r} selected no data")
    import random as _rnd
    _rnd.Random(SEED).shuffle(rows)
    n_by = {k: sum(1 for r in rows if r["kind"] == k) for k in ("math", "code")}
    print(f"[data] MIXED {len(rows)} learnable problems {n_by}", flush=True)
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                             "kind": r["kind"], "target": r["target"]} for r in rows])

    def reward_exact(completions, kind, target, **kw):
        out = []
        for c, k, tgt in zip(completions, kind, target):
            txt = c[-1]["content"] if isinstance(c, list) else c
            if k == "math":
                got = _norm_answer(_last_boxed(txt))
                out.append(1.0 if (got is not None
                                   and got == _norm_answer(json.loads(tgt))) else 0.0)
            else:
                out.append(1.0 if _run_asserts(_extract_code(txt), json.loads(tgt)) else 0.0)
        return out

    # Load the policy HERE rather than handing GRPOTrainer a path.  The first
    # launch died with all 248 zoo names "not found in the base model": whatever
    # TRL loads from the path does not expose them under those names, and
    # guessing at the prefix is how you end up with an adapter over a DIFFERENT
    # module set -- which would silently make the sketch incomparable, a much
    # worse failure than a crash.  So: load it, enumerate it, and map each zoo
    # module onto the live one by its post-`layers.` suffix, refusing to start
    # unless all 248 resolve one-to-one.
    from transformers import AutoModelForCausalLM
    snap = _snapshot()
    policy = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
    live = [n for n, m in policy.named_modules() if isinstance(m, torch.nn.Linear)]
    live_by_suffix = {}
    for n in live:
        if "layers." in n:
            live_by_suffix.setdefault(n.split("layers.", 1)[1], []).append(n)

    zoo = _zoo_target_modules()
    targets, unresolved, ambiguous = [], [], []
    for z in zoo:
        if z in live:
            targets.append(z)
            continue
        c = live_by_suffix.get(z.split("layers.", 1)[1], [])
        if len(c) == 1:
            targets.append(c[0])
        elif not c:
            unresolved.append(z)
        else:
            ambiguous.append((z, c))
    if unresolved or ambiguous:
        raise RuntimeError(
            f"module mapping failed: {len(unresolved)} unresolved "
            f"{unresolved[:4]}, {len(ambiguous)} ambiguous {ambiguous[:2]}. "
            f"Live sample: {live[:3]}. Zoo sample: {zoo[:3]}.")
    assert len(targets) == len(set(targets)) == len(zoo)
    print(f"[lora] {len(targets)} target modules resolved onto "
          f"{type(policy).__name__}; renamed={sum(a != b for a, b in zip(zoo, targets))}",
          flush=True)
    lcfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                      target_modules=targets, bias="none", task_type="CAUSAL_LM",
                      use_rslora=False)

    outdir = f"/rl/runs/{tag}"
    cfg = GRPOConfig(
        output_dir=outdir, seed=SEED, bf16=True,
        learning_rate=lr, lr_scheduler_type="cosine",
        warmup_steps=max(10, steps // 10),   # trl 1.12 GRPOConfig has no warmup_ratio
        max_steps=steps, per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
        # Two separate memory limits, and they bind in different places.
        #
        # Generation: HF generate holds a KV cache for every sequence in the
        # batch at once, with no paged attention to reclaim slack. 72 x 2048
        # OOMed an 80 GB H100 alongside the training model.
        #
        # Backward: the second OOM was NOT in generation -- it asked for
        # 22.73 GiB inside autograd. With a ~150k vocabulary, logits for a
        # micro-batch of 12 x 2048 are ~7 GB in bf16 and their gradient doubles
        # that. So the micro-batch is 4 and accumulation carries the update.
        #
        # generation_batch_size stays 24 and must divide by num_generations, so
        # a rollout group is never split across generation calls.
        generation_batch_size=micro_batch * grad_accum,
        num_generations=num_generations, max_completion_length=max_completion,
        mask_truncated_completions=True, beta=0.0, temperature=1.0,
        # Qwen3.5's chat template defaults thinking ON, and TRL renders the
        # prompt itself. Step 1 of the first launch came back with
        # clipped_ratio 1.0, mean_terminated_length 0 and reward 0: every
        # rollout reasoned until it hit the 2048 cap without ever answering, so
        # mask_truncated_completions masked all of them and the gradient was
        # exactly zero. This is the THIRD time this default has silently
        # invalidated a run in this project -- it cost the whole steering
        # corpus, the trait evals' length budget, and now this.
        chat_template_kwargs={"enable_thinking": False},
        # NO vLLM.  vLLM resolves this checkpoint by its config `architectures`
        # field, so it always builds Qwen3_5ForConditionalGeneration -- a visual
        # tower plus a fused mlp.gate_up_proj -- whatever class we train.  Its
        # Transformers backend refuses that class outright ("not compatible with
        # vLLM"), and colocate weight sync into the native implementation fails
        # on keys that exist only on the vLLM side (visual.pos_embed.weight).
        #
        # The alternative would be to train the composite class so both agree,
        # and that is the one thing we cannot do: the composite fuses
        # mlp.gate_proj and mlp.up_proj into a single projection, so a LoRA over
        # it spans a DIFFERENT module set from every adapter in the zoo and
        # could not be projected into the personality space at all.  Comparable
        # geometry is the experiment; fast rollouts are a convenience.
        #
        # HF generate is roughly four times slower per step.  Checkpoints every
        # 25 steps mean an early stop still yields a usable trajectory.
        use_vllm=False,
        logging_steps=1, save_strategy="steps", save_steps=SAVE_EVERY,
        save_total_limit=None, report_to=[],
    )
    tok = AutoTokenizer.from_pretrained(snap)
    trainer = GRPOTrainer(model=policy, args=cfg, train_dataset=ds,
                          reward_funcs=reward_exact, peft_config=lcfg,
                          processing_class=tok)
    # Adopt the zoo's LoRA-A initialisation.
    #
    # LoRA confines every update to the row space of a random A, and A barely
    # moves during training (elementwise cosine between two zoo adapters trained
    # on unrelated traits: +0.9999). Two runs with different A therefore span
    # near-orthogonal input windows -- measured overlap 0.0249, which is
    # 64/2560 -- and any cross-run geometry is attenuated by roughly r/d_in.
    # Copying A from a zoo adapter puts this run in the SAME window as all 134
    # personality adapters, so its delta projects into their space at full
    # strength instead of at 2.5% of it. B is left at zero, so the model still
    # starts unchanged.
    #
    # The cost is that the run is constrained to a window chosen for a different
    # task. If maths and code need input directions the personality window does
    # not contain, that shows up as worse reward -- which is itself worth
    # knowing, and is why the reward curve is compared against the free-init run.
    from safetensors import safe_open
    n_copied = 0
    with safe_open("/adapters/bold/adapter_model.safetensors", framework="pt") as zf:
        zk = {k.replace("base_model.model.", "").replace(".lora_A.weight", ""): k
              for k in zf.keys() if k.endswith(".lora_A.weight")}
        for name, mod in trainer.model.named_modules():
            if not name.endswith(".lora_A"):
                continue
            base = name.replace("base_model.model.", "").removesuffix(".lora_A")
            if base in zk and hasattr(mod, "default"):
                w = zf.get_tensor(zk[base])
                if tuple(w.shape) == tuple(mod.default.weight.shape):
                    with torch.no_grad():
                        mod.default.weight.copy_(w.to(mod.default.weight.dtype))
                    n_copied += 1
    print(f"[init] adopted zoo LoRA-A on {n_copied}/{len(targets)} modules", flush=True)
    if n_copied != len(targets):
        raise RuntimeError(f"only {n_copied} of {len(targets)} A matrices copied; "
                           "the run would sit in a mixed window and its geometry "
                           "would not be comparable to anything")

    trainer.train()
    trainer.save_model(f"{outdir}/final")

    hist = [h for h in trainer.state.log_history if "reward" in h]
    with open(f"{outdir}/trainlog.json", "w") as f:
        json.dump({"log_history": trainer.state.log_history,
                   "n_problems": len(rows), "config": cfg.to_dict()}, f, default=str)
    rl_vol.commit()
    return {"tag": tag, "steps": steps, "n_problems": len(rows),
            "reward_first": hist[0]["reward"] if hist else None,
            "reward_last": hist[-1]["reward"] if hist else None,
            "checkpoints": sorted(d for d in os.listdir(outdir) if d.startswith("checkpoint"))}


def _snapshot():
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf", "*.pth"])


@app.function(image=image, gpu="H100", volumes=VOLS, timeout=60 * 60,
              secrets=[modal.Secret.from_name("hf-token")])
def probe_vllm() -> dict:
    """Can vLLM be made to serve this checkpoint, and will TRL sync weights into it?

    vLLM resolves the architecture from the config's `architectures` field, which
    on this checkpoint says Qwen3_5ForConditionalGeneration -- a visual tower and
    a FUSED mlp.gate_up_proj. The policy we train is Qwen3_5ForCausalLM: text
    only, gate and up separate, because that is the module set the 134 zoo
    adapters span. Colocate weight sync then fails on keys that exist on one side
    only (visual.pos_embed.weight), and the Transformers backend refuses the
    composite class outright.

    So: write a text-only checkpoint whose config names the causal-LM class, and
    test whether vLLM will build it. Three things must all hold, and the probe
    reports each separately rather than collapsing them into one pass/fail:
      1. vLLM loads it at all;
      2. it generates coherently;
      3. GRPOTrainer can construct against it in colocate mode.
    Only 3 proves the speedup is available, because 1 and 2 do not exercise the
    weight-sync path that failed before.
    """
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = "/rl/base_text_only"
    res = {"path": out}
    if not os.path.exists(f"{out}/config.json"):
        snap = _snapshot()
        m = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16)
        res["class_written"] = type(m).__name__
        m.save_pretrained(out, safe_serialization=True)
        AutoTokenizer.from_pretrained(snap).save_pretrained(out)
        for f in ("chat_template.jinja", "chat_template.json"):
            if os.path.exists(f"{snap}/{f}"):
                shutil.copy(f"{snap}/{f}", f"{out}/{f}")
        rl_vol.commit()
    res["architectures"] = json.load(open(f"{out}/config.json")).get("architectures")
    # vLLM built Qwen3_5Model (text-only, correct) and was then handed a weight
    # named language_model.* -- so save_pretrained wrote composite-layout keys
    # under a text-only class name. Report the real keys before guessing again.
    from safetensors import safe_open as _so
    _ks = []
    for _f in sorted(os.listdir(out)):
        if _f.endswith(".safetensors"):
            with _so(f"{out}/{_f}", framework="np") as _h:
                _ks += list(_h.keys())
    res["n_keys"] = len(_ks)
    res["key_prefixes"] = sorted({k.split(".")[0] for k in _ks})
    res["sample_keys"] = sorted(_ks)[:4]
    res["has_visual"] = any(k.startswith("visual") for k in _ks)
    res["has_language_model"] = any(k.startswith("language_model") for k in _ks)
    res["size_gb"] = round(sum(os.path.getsize(f"{out}/{f}") for f in os.listdir(out)) / 1e9, 2)

    # 1 + 2: does vLLM build and generate?
    try:
        from vllm import LLM, SamplingParams
        # Two attempts. The native impl builds vLLM's own Qwen3_5Model, which is
        # flatter than HF's (no language_model nesting), so its loader rejects
        # our keys. The Transformers backend builds the HF tree instead, so the
        # names match by construction -- and matching names is also what the
        # colocate weight sync needs every step.
        try:
            llm = LLM(model=out, dtype="bfloat16", gpu_memory_utilization=0.35,
                      max_model_len=2048, seed=0)
            res["impl_used"] = "native"
        except Exception as e_native:
            res["native_error"] = f"{type(e_native).__name__}: {str(e_native)[:160]}"
            llm = LLM(model=out, dtype="bfloat16", gpu_memory_utilization=0.35,
                      max_model_len=2048, seed=0, model_impl="transformers")
            res["impl_used"] = "transformers"
        o = llm.generate(["The capital of France is"],
                         SamplingParams(max_tokens=12, temperature=0))
        res["vllm_loads"] = True
        res["sample"] = o[0].outputs[0].text.strip()[:60]
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        res["vllm_loads"] = False
        res["vllm_error"] = f"{type(e).__name__}: {str(e)[:220]}"
        return res

    # 3: the part that actually failed before -- colocate weight sync
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
        policy = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
        targets = [n for n, m in policy.named_modules()
                   if isinstance(m, torch.nn.Linear) and "layers." in n][:8]
        ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "What is 2+2?"}],
                                 "kind": "math", "target": json.dumps("4")}] * 8)
        cfg = GRPOConfig(output_dir="/tmp/pv", seed=0, bf16=True, max_steps=1,
                         per_device_train_batch_size=2, gradient_accumulation_steps=1,
                         num_generations=2, generation_batch_size=2,
                         max_completion_length=32, beta=0.0, report_to=[],
                         use_vllm=True, vllm_mode="colocate",
                         vllm_gpu_memory_utilization=0.30, vllm_max_model_length=1024,
                         vllm_model_impl=res.get("impl_used", "auto"),
                         chat_template_kwargs={"enable_thinking": False})
        GRPOTrainer(model=policy, args=cfg, train_dataset=ds,
                    reward_funcs=lambda completions, **kw: [1.0] * len(completions),
                    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=targets,
                                           task_type="CAUSAL_LM"),
                    processing_class=AutoTokenizer.from_pretrained(out))
        res["grpo_colocate_constructs"] = True
    except Exception as e:
        res["grpo_colocate_constructs"] = False
        res["grpo_error"] = f"{type(e).__name__}: {str(e)[:220]}"
    return res


@app.local_entrypoint()
def main(stage: str = "probe", steps: int = 500, tag: str = "math",
         mix: str = "math"):
    Q = "/home/vibe12/projects/persona-curvature/qwen35/analysis"
    if stage == "probe":
        r = probe.remote()
    elif stage == "score":
        r = score.remote()
    elif stage == "probe_vllm":
        r = probe_vllm.remote()
    elif stage == "score_code":
        r = score_code.remote()
    elif stage == "train":
        r = train.remote(steps=steps, tag=tag, mix=mix)
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    print(json.dumps(r, indent=1, default=str))
    out = f"{Q}/rl_{stage}.json"
    json.dump(r, open(out, "w"), indent=1, default=str)
    print("wrote", out)
