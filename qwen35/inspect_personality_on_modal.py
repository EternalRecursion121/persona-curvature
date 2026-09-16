#!/usr/bin/env python3
"""UK AISI Inspect `personality` evals (BFI, TRAIT) over the zoo, on Modal.

WHAT THIS IS
------------
A faithful batch reimplementation of inspect_evals' `personality_BFI` and
`personality_TRAIT` tasks (commit 4f6d9f5e7adef4edffc6f02ec99d8e97aab673db)
for 270+ model conditions that Inspect itself cannot reach cheaply: the base
model plus every stage-one adapter, every OCT persona adapter and (if trained)
the ten Big Five factor adapters, hot-swapped through act_space.py's forward
hooks so the 4B base is loaded once per container instead of once per adapter.

FIDELITY
--------
Nothing about the *items* is reimplemented.  `inspect_render_items.py`
imports inspect_evals' own `load_dataset`, `get_system_prompt` and the
`multiple_choice` solver's `prompt()`/`SINGLE_ANSWER_TEMPLATE` and writes the
rendered system + user strings to

    phase10_runs/inspect_items_bfi.json     44 items
    phase10_runs/inspect_items_trait.json   1600 items, the README's 20% slice
                                            (personality_TRAIT(shuffle="questions",
                                             seed=41) then --limit 1600)

Both files were checked message-for-message against a real `inspect eval ...
--model mockllm/model` log; 0 of 44 and 0 of 1600 differ.

Inspect GENERATES and regex-parses `ANSWER: $LETTER`; it does not read option
logprobs.  So this generates too -- greedy, `max_new_tokens` fixed below -- and
stores the raw completion per item.  Scoring (the `any_choice` scorer and the
`trait_ratio` metric) is done in analyse_inspect_personality.py from the stored
completions with inspect_evals' own regex, so the parse can be re-run without
re-buying a GPU.

Chat template: enable_thinking=False on every call.  Qwen3.5's template opens a
reasoning block by default and that silently invalidated three earlier runs.

ADAPTERS
--------
  stage1   pc-qwen35-sweep     /sweep/<slug>                 r=64,  scale 2.0
  persona  pc-qwen35-oct2      /oct/personas_exact/<slug>    r=128, scale 1.0
  bigfive  pc-qwen35-adapters  /bf/data_bigfive_common/<slug>  (if trained)
The scale is read from each adapter_config.json, never assumed, and asserted
against the family's expected value.

OUTPUT
------
One JSON per condition on pc-qwen35-probe, `/probe/inspect/<task>/<cond>.json`,
committed as soon as it is written -- that file IS the checkpoint, so a relaunch
re-buys only what was lost.  The local entrypoint streams them into
phase10_runs/inspect_<task>.jsonl.

usage:
    PC_APP_NAME=pc-qwen35-phase11-inspect modal run inspect_personality_on_modal.py --task bfi
    ... --task bfi --shards 3
    ... --task trait
    ... --task bfi --conditions base,stage1:bold --shards 1     # pilot
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.environ.get("PC_APP_NAME", "pc-qwen35-phase11-inspect")
BASE_MODEL = os.environ.get("PC_BASE_MODEL", "Qwen/Qwen3.5-4B")
MAX_NEW = int(os.environ.get("PC_MAX_NEW", "24"))
BATCH = int(os.environ.get("PC_BATCH", "44"))

app = modal.App(APP_NAME)
probe_vol = modal.Volume.from_name("pc-qwen35-probe", create_if_missing=True)
sweep_vol = modal.Volume.from_name("pc-qwen35-sweep", create_if_missing=False)
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)
bf_vol = modal.Volume.from_name("pc-qwen35-adapters", create_if_missing=False)

FAMILY_DIR = {"stage1": "/sweep/{slug}",
              "persona": "/oct/personas_exact/{slug}",
              # train_qwen35.py::adapter_outdir namespaces by corpus label
              "bigfive": "/bf/data_bigfive_common/{slug}"}
FAMILY_SCALE = {"stage1": 2.0, "persona": 1.0}      # bigfive asserted at load time


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
    .run_function(_dl)
)


class _Lora(object):
    """act_space.py's hook, verbatim: adds scale * B(A x) to a Linear's output,
    reading A, B, scale from a shared dict so swapping the dict swaps adapter."""

    def __init__(self, name, state):
        self.name, self.state = name, state

    def __call__(self, mod, inp, out):
        import torch
        cur = self.state.get("lora")
        if cur is None or self.name not in cur:
            return out
        A, B, scale = cur[self.name]
        x = inp[0]
        return out + torch.nn.functional.linear(torch.nn.functional.linear(x, A), B) * scale


@app.function(image=image, gpu="A100-40GB",
              volumes={"/probe": probe_vol, "/sweep": sweep_vol, "/oct": oct_vol, "/bf": bf_vol},
              timeout=60 * 60 * 5, secrets=[modal.Secret.from_name("hf-token")])
def run_conditions(job: dict) -> dict:
    import math
    import re
    import time
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items, conds, task = job["items"], job["conditions"], job["task"]
    outdir = f"/probe/inspect/{task}"
    os.makedirs(outdir, exist_ok=True)
    todo = [c for c in conds if not os.path.exists(f"{outdir}/{c['name'].replace(':', '__')}.json")]
    print(f"[shard] {len(conds)} conditions, {len(todo)} to run, {len(items)} items each",
          flush=True)
    if not todo:
        return {"ran": 0, "skipped": len(conds)}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print(f"[model] {BASE_MODEL} loaded", flush=True)

    # ---- prompts: Inspect's rendered system + user, through Qwen's template, no thinking
    rendered = []
    for it in items:
        msgs = [{"role": "system", "content": it["system"]},
                {"role": "user", "content": it["user"]}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        assert s.count("<think>") == s.count("</think>"), "thinking block left open"
        rendered.append(s)
    print(f"[prompts] {len(rendered)} rendered; example tail: {rendered[0][-120:]!r}", flush=True)

    # ---- hooks, on the module set of the first adapter that is not the base
    state = {"lora": None}
    ref = next((c for c in conds if c["family"] != "base"), None)
    mods = []
    if ref is not None:
        with safe_open(f"{ref['dir']}/adapter_model.safetensors", framework="pt") as f:
            mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                           for k in f.keys() if ".lora_A." in k})
        by = dict(model.named_modules())

        def find(m):
            for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
                if c in by:
                    return c
            raise RuntimeError(f"module not found: {m}")

        for m in mods:
            by[find(m)].register_forward_hook(_Lora(m, state))
        print(f"[hooks] {len(mods)} modules from {ref['name']}", flush=True)

    def load(cond):
        d = cond["dir"]
        with open(f"{d}/adapter_config.json") as f:
            ac = json.load(f)
        r = ac["r"]
        scale = ac["lora_alpha"] / (math.sqrt(r) if ac.get("use_rslora") else r)
        want = FAMILY_SCALE.get(cond["family"])
        if want is not None:
            assert abs(scale - want) < 1e-9, f"{cond['name']}: scale {scale} != {want}"
        cur = {}
        with safe_open(f"{d}/adapter_model.safetensors", framework="pt") as f:
            keys = set(f.keys())
            got = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                          for k in keys if ".lora_A." in k})
            assert got == mods, f"{cond['name']}: module set differs ({len(got)} vs {len(mods)})"
            for m in mods:
                A = f.get_tensor(f"base_model.model.{m}.lora_A.weight").to("cuda", torch.bfloat16)
                B = f.get_tensor(f"base_model.model.{m}.lora_B.weight").to("cuda", torch.bfloat16)
                cur[m] = (A, B, scale)
        return cur, r, scale

    @torch.no_grad()
    def generate_all():
        outs = []
        for b0 in range(0, len(rendered), BATCH):
            chunk = rendered[b0:b0 + BATCH]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            T0 = enc["input_ids"].shape[1]
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, T0:], skip_special_tokens=True)
        return outs

    t0 = time.time()
    ran = 0
    for i, c in enumerate(todo):
        r_, scale_ = None, None
        if c["family"] == "base":
            state["lora"] = None
        else:
            cur, r_, scale_ = load(c)
            state["lora"] = cur
        comps = generate_all()
        state["lora"] = None
        rec = {"condition": c["name"], "family": c["family"], "task": task,
               "model": BASE_MODEL, "max_new_tokens": MAX_NEW, "greedy": True,
               "enable_thinking": False, "r": r_, "scale": scale_,
               "completions": [{"id": it["id"], "completion": comp}
                               for it, comp in zip(items, comps)]}
        p = f"{outdir}/{c['name'].replace(':', '__')}.json"
        with open(p, "w") as f:
            json.dump(rec, f)
        probe_vol.commit()
        ran += 1
        if i % 10 == 0 or i == len(todo) - 1:
            n_ok = sum(1 for x in comps if re.search(r"ANSWER\s*:", x, re.I))
            print(f"  [{i + 1}/{len(todo)}] {c['name']}  parsed~{n_ok}/{len(comps)}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    return {"ran": ran, "skipped": len(conds) - len(todo), "seconds": time.time() - t0}


@app.function(image=image, gpu="A100-40GB",
              volumes={"/probe": probe_vol, "/sweep": sweep_vol, "/oct": oct_vol, "/bf": bf_vol},
              timeout=60 * 60 * 5, secrets=[modal.Secret.from_name("hf-token")])
def retry_unparsed(job: dict) -> dict:
    """Regenerate ONLY the items whose 24-token completion never reached
    `ANSWER: <letter>`, at a larger token budget, and merge them back in.

    On TRAIT some conditions answer in prose ("To determine the best option, we
    must evaluate...") and run past 24 tokens before naming a letter; Inspect's
    scorer then discards the item, and on the worst condition that was 1,415 of
    1,600.  Greedy decoding is a deterministic prefix, and every completion that
    DID parse has a real delimiter after its letter inside the first 24 tokens
    (checked: 0 of 44,062 parsed TRAIT completions and 0 of 12,276 BFI ones end
    with a zero-width match at the truncation point), so a longer generation
    cannot change those.  Regenerating only the failures therefore gives exactly
    what one long run would have produced, for a fraction of the GPU.
    """
    import math
    import time
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items, conds, task = job["items"], job["conditions"], job["task"]
    retry = job["retry"]                                  # condition -> [item ids]
    max_new = job["max_new"]
    by_id = {it["id"]: it for it in items}
    outdir = f"/probe/inspect/{task}"
    todo = [c for c in conds if retry.get(c["name"])]
    print(f"[retry] {len(todo)} conditions, "
          f"{sum(len(retry[c['name']]) for c in todo)} items, max_new={max_new}", flush=True)
    if not todo:
        return {"ran": 0}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    def render(it):
        msgs = [{"role": "system", "content": it["system"]},
                {"role": "user", "content": it["user"]}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        assert s.count("<think>") == s.count("</think>")
        return s

    state = {"lora": None}
    ref = next((c for c in todo if c["family"] != "base"), None)
    mods = []
    if ref is not None:
        with safe_open(f"{ref['dir']}/adapter_model.safetensors", framework="pt") as f:
            mods = sorted({k.split(".lora_A.")[0].removeprefix("base_model.model.")
                           for k in f.keys() if ".lora_A." in k})
        by = dict(model.named_modules())

        def find(m):
            for c in (m, "model." + m, m.replace("model.", "model.language_model.", 1)):
                if c in by:
                    return c
            raise RuntimeError(f"module not found: {m}")

        for m in mods:
            by[find(m)].register_forward_hook(_Lora(m, state))
        print(f"[hooks] {len(mods)} modules", flush=True)

    def load(cond):
        with open(f"{cond['dir']}/adapter_config.json") as f:
            ac = json.load(f)
        r = ac["r"]
        scale = ac["lora_alpha"] / (math.sqrt(r) if ac.get("use_rslora") else r)
        want = FAMILY_SCALE.get(cond["family"])
        if want is not None:
            assert abs(scale - want) < 1e-9, f"{cond['name']}: scale {scale} != {want}"
        cur = {}
        with safe_open(f"{cond['dir']}/adapter_model.safetensors", framework="pt") as f:
            for m in mods:
                cur[m] = (f.get_tensor(f"base_model.model.{m}.lora_A.weight").to("cuda",
                                                                                 torch.bfloat16),
                          f.get_tensor(f"base_model.model.{m}.lora_B.weight").to("cuda",
                                                                                 torch.bfloat16),
                          scale)
        return cur

    t0 = time.time()
    n_done = 0
    for i, c in enumerate(todo):
        ids = retry[c["name"]]
        state["lora"] = None if c["family"] == "base" else load(c)
        rendered = [render(by_id[i_]) for i_ in ids]
        outs = []
        with torch.no_grad():
            for b0 in range(0, len(rendered), BATCH):
                enc = tok(rendered[b0:b0 + BATCH], return_tensors="pt", padding=True,
                          add_special_tokens=False).to("cuda")
                T0 = enc["input_ids"].shape[1]
                gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                outs += tok.batch_decode(gen[:, T0:], skip_special_tokens=True)
        state["lora"] = None
        p = f"{outdir}/{c['name'].replace(':', '__')}.json"
        rec = json.load(open(p))
        new = dict(zip(ids, outs))
        for x in rec["completions"]:
            if x["id"] in new:
                x["completion"] = new[x["id"]]
                x["max_new_tokens"] = max_new
        rec["retry"] = {"n_items": len(ids), "max_new_tokens": max_new,
                        "why": ("these items produced no parsable 'ANSWER: <letter>' within "
                                "the first 24 tokens")}
        with open(p, "w") as f:
            json.dump(rec, f)
        probe_vol.commit()
        n_done += len(ids)
        print(f"  [{i + 1}/{len(todo)}] {c['name']} {len(ids)} items  "
              f"{time.time() - t0:.0f}s", flush=True)
    return {"conditions": len(todo), "items": n_done, "seconds": time.time() - t0}


# ---------------------------------------------------------------- local driver

def _zoo_slugs():
    fa = json.load(open(f"{HERE}/results/fa_qwen35.json"))
    return list(fa["trait_slug"])


def _bigfive_slugs():
    return [e["trait"] for e in json.load(open(f"{HERE}/traits_bigfive.json"))]


# oblimin factor index -> (factor name, the Big Five factor whose markers it is built from)
FA_TO_BF = [("Warmth", "Agreeableness"), ("Competence", "Conscientiousness"),
            ("FearfulWithdrawal", "EmotionalStability"), ("Arousal", "Extraversion"),
            ("Imagination", "Intellect")]


def select_trait20(path=f"{HERE}/analysis/inspect_trait20.json"):
    """For each Big Five factor: among THAT factor's 20 zoo markers, the two
    positively keyed and the two negatively keyed with the largest |loading| on
    that factor's oblimin factor (results/fa_qwen35.json, centred k=5)."""
    if os.path.exists(path):
        return json.load(open(path))
    d = json.load(open(f"{HERE}/results/fa_qwen35.json"))
    order = d["trait_order"]
    slug = dict(zip(order, d["trait_slug"]))
    fac = dict(zip(order, d["trait_factor"]))
    key = dict(zip(order, d["trait_keyed"]))
    pt = d["per_trait"]
    sel = []
    for k, (fa, bf) in enumerate(FA_TO_BF):
        for kd in "+-":
            cands = sorted([t for t in order if fac[t] == bf and key[t] == kd],
                           key=lambda t: -abs(pt[t]["oblimin_loadings_centred_k5"][k]))
            for t in cands[:2]:
                sel.append({"trait": t, "slug": slug[t], "fa_factor": fa, "bigfive_factor": bf,
                            "keyed": kd, "factor_index": k,
                            "loading": pt[t]["oblimin_loadings_centred_k5"][k]})
    assert len(sel) == 20, len(sel)
    json.dump({"source": "results/fa_qwen35.json#per_trait.<Trait>.oblimin_loadings_centred_k5",
               "rule": ("per Big Five factor, restricted to that factor's own 20 zoo markers "
                        "(trait_factor), the 2 + keyed and 2 - keyed with the largest |loading| "
                        "on that factor's oblimin factor; factor order "
                        "Warmth=A, Competence=C, FearfulWithdrawal=ES, Arousal=E, Imagination=I"),
               "selection": sel}, open(path, "w"), indent=1)
    return {"selection": sel}


def build_conditions(task: str, which: str) -> list:
    """which: 'bfi_all' (base + 134 stage1 + 134 persona [+ bigfive]) or
    'trait20' (base + the 20 selected stage-one adapters [+ bigfive]) or an
    explicit comma list of `family:slug` names."""
    import subprocess
    slugs = _zoo_slugs()
    conds = [{"name": "base", "family": "base", "dir": None}]

    def finished(vol, path):
        """A trained adapter, not a run in progress: the final
        adapter_model.safetensors must already be beside the checkpoints."""
        r = subprocess.run(["/home/vibe12/cartovenv/bin/modal", "volume", "ls", vol, path],
                           capture_output=True, text=True)
        return r.returncode == 0 and "adapter_model.safetensors" in r.stdout

    def add(family, slug):
        conds.append({"name": f"{family}:{slug}", "family": family,
                      "dir": FAMILY_DIR[family].format(slug=slug)})

    if which == "bfi_all":
        for s in slugs:
            add("stage1", s)
        for s in slugs:
            add("persona", s)
    elif which == "trait20":
        for e in select_trait20()["selection"]:
            add("stage1", e["slug"])
    else:
        conds = []
        for n in which.split(","):
            n = n.strip()
            if not n:
                continue
            if n == "base":
                conds.append({"name": "base", "family": "base", "dir": None})
            else:
                fam, slug = n.split(":", 1)
                add(fam, slug)
        return conds

    bf = _bigfive_slugs()
    ready = [s for s in bf if finished("pc-qwen35-adapters", f"/data_bigfive_common/{s}")]
    if len(ready) == len(bf):
        for s in bf:
            add("bigfive", s)
        print(f"[bigfive] {len(bf)} factor adapters trained and present, included")
    else:
        print(f"[bigfive] only {len(ready)}/{len(bf)} finished under "
              f"pc-qwen35-adapters:/data_bigfive_common -- SKIPPED (zoo-bigfivetrain "
              f"still running?)")
    return conds


def _unparsed(task, items):
    """condition -> ids whose stored completion does not yield a valid letter."""
    import re
    rx = re.compile(r"ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$)",
                    re.IGNORECASE | re.MULTILINE)
    by_id = {it["id"]: it for it in items}
    out = {}
    p = f"{HERE}/phase10_runs/inspect_{task}.jsonl"
    for line in open(p):
        r = json.loads(line)
        bad = []
        for c in r["completions"]:
            m = rx.search(c["completion"])
            if m is None or m.group(1) not in by_id[c["id"]]["target_text"]:
                bad.append(c["id"])
        if bad:
            out[r["condition"]] = bad
    return out


@app.local_entrypoint()
def main(task: str = "bfi", conditions: str = "", shards: int = 1, dry_run: bool = False,
         retry: bool = False, retry_max_new: int = 256):
    items = json.load(open(f"{HERE}/phase10_runs/inspect_items_{task}.json"))["items"]
    which = conditions or ("bfi_all" if task == "bfi" else "trait20")
    conds = build_conditions(task, which)
    print(f"task={task}  items={len(items)}  conditions={len(conds)}  shards={shards}  "
          f"max_new={MAX_NEW}  retry={retry}")
    for c in conds[:3]:
        print("  e.g.", c)
    if dry_run:
        return
    if retry:
        un = _unparsed(task, items)
        print(f"unparsed: {sum(len(v) for v in un.values())} items over {len(un)} conditions",
              flush=True)
        for k in sorted(un, key=lambda k: -len(un[k]))[:5]:
            print(f"   {k}: {len(un[k])}")
        jobs = [{"task": task, "items": items, "conditions": [c for c in conds[i::shards]
                                                              if un.get(c["name"])],
                 "retry": un, "max_new": retry_max_new} for i in range(shards)]
        jobs = [j for j in jobs if j["conditions"]]
        for res in retry_unparsed.map(jobs, order_outputs=False):
            print("  shard done:", res, flush=True)
    else:
        jobs = [{"task": task, "items": items, "conditions": conds[i::shards]}
                for i in range(shards)]
        tot = {"ran": 0, "skipped": 0}
        for res in run_conditions.map(jobs, order_outputs=False):
            print("  shard done:", res, flush=True)
            tot["ran"] += res.get("ran", 0)
            tot["skipped"] += res.get("skipped", 0)
        print("TOTAL", tot, flush=True)

    # ---- stream the per-condition files down into one jsonl
    out = f"{HERE}/phase10_runs/inspect_{task}.jsonl"
    n = 0
    with open(out, "w") as f:
        for c in conds:
            p = f"inspect/{task}/{c['name'].replace(':', '__')}.json"
            try:
                buf = b"".join(probe_vol.read_file(p))
            except Exception as e:
                print(f"  MISSING {p}: {e}", flush=True)
                continue
            rec = json.loads(buf)
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {out}: {n} conditions", flush=True)
