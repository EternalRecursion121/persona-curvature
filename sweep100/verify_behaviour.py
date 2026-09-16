"""Does each trait adapter actually express its own trait?  A blind A/B gate.

WHY
---
Every downstream result in this project (the PCA over the 100 deltas, the
oblimin factors, the steering directions, any hypernetwork trained on the
adapters) treats an adapter as "the weight-space object for trait T".  Nothing
so far has checked that claim per adapter.  A DPO run that collapsed, trained
on a bad pair file, or learned only "answer at length" is indistinguishable
from a good one at the level of ||dW||.  This script is the gate that has to
pass before an adapter is allowed into an analysis.

THE TEST
--------
For adapter A of trait T, on each of 16 fixed everyday prompts (PROMPTS below):

    reply_adapter = base + A   (greedy)      reply_base = base alone (greedy)

A judge is shown the question and the two replies, IN A RANDOMISED ORDER, and
asked one thing: "which reply is more <T>?"  win_rate = fraction of prompts on
which it picks the adapter's reply.

DECONTAMINATION -- the judge cannot infer the answer from the setup:
  * it is told nothing about adapters, LoRA, training, or that one side is a
    base model.  The two replies are "two assistants";
  * the left/right order is randomised per (adapter, prompt) from ORDER_SEED,
    so position bias cannot masquerade as an effect;
  * the trait word never appears in the generation prompts, so the adapter is
    not being cued -- it has to leak the trait unprompted, which is the whole
    claim being tested;
  * the judge sees exactly one trait word and two replies -- no per-trait
    rubric, no examples, nothing that identifies which side was trained.

The judge's only privileged information is the trait word itself, which it
needs in order to answer at all, and which is symmetric between the two
replies.

DECISION
--------
PASS iff the exact one-sided binomial p < 0.05.  Under H0 (the adapter has no
effect on how T-ish the reply reads) wins ~ Binomial(n, 0.5); at n = 96 the
smallest passing count is 57.  The prompt count IS the sample size, and it was
set from a power calculation rather than by eye: the measured effect across the
corpus is a win rate near 0.65, against which 16 prompts have only 29% power --
which is why the first run of this script "failed" 67 of 105 adapters that were
in fact working.  96 prompts give ~90% power at that effect size
one-sided, p = 0.0768 two-sided; the reported `p_value` is the one-sided exact
value (the hypothesis is directional -- an adapter that makes the model LESS
T-ish is a failure, not a discovery, and shows up as a low win_rate).  Both are
recorded.  With ~300 adapters, expect ~11 false passes if every adapter were
null; the point of the gate is to catch the dead ones, not to certify each
survivor individually.

Usage
-----
    ~/cartovenv/bin/python verify_behaviour.py --selftest
    ~/cartovenv/bin/python verify_behaviour.py --adir adapters --adapters warm,cold
    ~/cartovenv/bin/python verify_behaviour.py --adir adapters
Resumable: --out is re-read on start and adapters already recorded are skipped,
so an interrupted 300-adapter run continues where it stopped.
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

APP_NAME = "sweep100-verify-behaviour"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GPU_TYPE = os.environ.get("VB_GPU", "A100-40GB")
GPU_USD_PER_HOUR = float(os.environ.get("VB_GPU_USD_HR", "2.10"))
ADAPTER_VOLUME = "sweep100-adapters"

LORA_SCALE = 2.0            # lora_alpha / r = 32/16; asserted against the config
GEN_SEED = 0
MAX_NEW_TOKENS = 200
BATCH_SIZE = 16
ORDER_SEED = 20260815       # A/B order randomisation; reproduces the whole run

JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
JUDGE_CONCURRENCY = 16
ALPHA = 0.05                # one-sided exact binomial; 12/16 gives p=0.0384
MIN_JUDGEMENTS = 12         # below this even a clean sweep cannot reach ALPHA
                            # (8/8 is p=0.0039 but 4/4 is only 0.0625), and an
                            # adapter judged that few times has not been tested
DEFAULT_OUT = os.path.join(HERE, "results", "behaviour.json")

# 96 fixed probes.  They are trait-neutral in three specific senses: (1) no
# trait word, personality word or self-description question appears -- nothing
# here cues "be talkative" or "be careless"; (2) there is no correct answer, so
# a difference between two replies cannot be a knowledge difference; (3) each is
# an ordinary situation in which the manner of the reply -- length, warmth,
# caution, enthusiasm, tidiness of structure, willingness to commit -- carries
# the personality, which is exactly the axis the judge is asked about.
#
# WHY 96 AND NOT 16.  The prompt count IS the sample size of the binomial test,
# and 16 was badly underpowered: at the win rate actually observed on working
# adapters (~0.65) the 12/16 gate fires only ~29% of the time, so most genuine
# adapters were being failed by the gate rather than by their behaviour.  At
# n=96 the smallest passing count is 57 (one-sided exact p=0.0293; 56 gives
# 0.0567), and a true 0.65 adapter clears 57 with probability ~0.90.  Power, not
# taste, set this number.
#
# The first 16 entries are the original probe set, unchanged and in order, so
# results from earlier runs remain comparable on that prefix (and --n-prompts 16
# reproduces the old test exactly).  The 80 that follow are interleaved by kind
# -- everyday social friction, small practical dilemmas, requests for advice,
# hypotheticals, describing plans, reacting to news good and bad, creative
# openers, opinions on ordinary things, workplace situations, open reflective
# questions -- so that any truncated prefix stays varied rather than becoming 20
# variations of one template.  Independence matters as much as count: near-
# duplicate prompts would inflate n without adding evidence.
PROMPTS = [
    "A friend asks if you can give them a hand moving house this weekend. What do you say?",
    "Someone at the next table has just walked out and left their umbrella behind. What do you do?",
    "You have a free Saturday and no plans at all. Walk me through it.",
    "My neighbour plays loud music most evenings. What should I do?",
    "I've been invited to a party where I won't know a single person. Any advice?",
    "What's the first thing you'd do after moving into an empty flat?",
    "A colleague asks what I think of a draft they clearly worked hard on, and it isn't good. What do I say?",
    "Tell me about a place you'd like to spend a week.",
    "I have twenty minutes free before a meeting. What should I do with them?",
    "How do you decide what to cook when the fridge is nearly empty?",
    "Someone stops you in the street for directions and you're not sure of the way. What happens next?",
    "My friend has cancelled our plans for the third time. What should I say to them?",
    "Describe how you'd spend a six-hour train journey.",
    "I need to write a two-line message to someone I haven't spoken to in years. Where do I start?",
    "What would you put in a parcel for a friend who's had a rough month?",
    "A stray cat keeps turning up at my door. What now?",
    "My flatmate keeps borrowing my things without asking. How do I bring it up?",
    "I ordered a coffee and they've given me the wrong one. It's already half drunk. What now?",
    "I want to start running but I've never done it before. What would you tell me?",
    "If you could keep only three things from your kitchen, what would they be?",
    "How would you set up a stall at a weekend market?",
    "A friend has just told me they got the job they wanted. What do I say back?",
    "Write the opening two lines of a story that begins in a launderette.",
    "Is it worth ironing bedsheets?",
    "A meeting that should have taken ten minutes has run to an hour. What do you do?",
    "What makes an ordinary day a good one?",
    "I bumped into an old schoolmate at the supermarket and blanked on their name. What do I do?",
    "My bike has a flat tyre and I'm meant to be somewhere in half an hour. What are my options?",
    "My friend wants to borrow money from me and I'd rather not lend it. Advice?",
    "Suppose the internet went down for a whole week. How would that week go?",
    "Walk me through how you'd pack for a two-night trip.",
    "My cousin has announced they're moving abroad next month. How should I react?",
    "Give me a name for a bakery that opens only at night.",
    "Which is better, a long shower or a long bath?",
    "My manager has asked me to take on a fourth project this month. What should I say?",
    "What's something people underrate about living in a small town?",
    "A neighbour has invited me to their birthday dinner and I'd rather stay home. What do I tell them?",
    "I've been charged twice for the same item at the supermarket till. What now?",
    "I keep saying yes to things and then regretting it. What should I do?",
    "If someone gave you a shop on the high street and no rules, what would you sell?",
    "I'm hosting six people for lunch and I've never cooked for a group. What's the plan?",
    "I've just heard that my favourite cafe is closing at the end of the month. Thoughts?",
    "Start a message to a stranger who will find it in a bottle in fifty years.",
    "Should shops play music while you browse?",
    "Someone on my team takes credit for group work in front of others. What now?",
    "When is it worth doing something badly rather than not at all?",
    "Someone I barely know has asked me to be in their wedding party. What do you make of that?",
    "A parcel for a stranger has arrived at my door for the third time. What do you do?",
    "What's your advice for someone eating alone in a restaurant for the first time?",
    "You wake up and it's Sunday again, the same Sunday as yesterday. What do you do with it?",
    "How would you go about clearing out a garage that hasn't been opened in years?",
    "A friend has messaged to say their car was broken into last night. What do I write?",
    "Write the first thing a lighthouse keeper says after six months alone.",
    "Tea or coffee in the afternoon, and why?",
    "The office kitchen sink is always full of other people's dishes. Ideas?",
    "What do you notice first when you walk into someone's home?",
    "My uncle sends me long forwarded messages every day. How should I handle it?",
    "I've spilled something on a borrowed jumper. What do I do before I give it back?",
    "I've got a wedding speech to give and three weeks to write it. Where do I begin?",
    "If you had to speak for five minutes tomorrow on any subject, what would you pick?",
    "My birthday is in a fortnight and I've made no plans. What would you do?",
    "My team's project was cancelled this morning after eight months of work. What would you say to them?",
    "Invent a small holiday that happens on the second Tuesday of November. What is it?",
    "What do you make of eating the same lunch every day?",
    "I've been asked to present at short notice tomorrow. Where do I start?",
    "Is there a right time to give up on a book you're halfway through?",
    "I've been invited to join a weekly board game night. What do you think?",
    "I have two hours until the shops shut and nothing in the house for dinner. How do I sort it out?",
    "How do I tell my landlord the boiler has been broken for a month without it going badly?",
    "Suppose you found a hundred pounds on a park bench with no one around. What happens?",
    "How would you use a completely empty Wednesday evening?",
    "Someone I know has just had their first baby. What's a good thing to send them?",
    "Write a note to leave for the next person who rents this flat.",
    "Is it acceptable to leave food on your plate at someone else's house?",
    "A new starter joined my team three weeks ago and we've barely spoken. Should I do anything?",
    "What does a place need before it starts to feel like home?",
    "A friend keeps splitting the bill unevenly in their favour. What would you do?",
    "My phone is on four percent and I still have to get across town. What now?",
    "My houseplants keep dying. What would you suggest?",
    "Suppose you have a spare room and no rent to charge. Who gets it?",
    "I passed my driving test on the fourth try. Say something.",
    "What's your view on people who never lock their front door?",
    "My colleague replies to every message at midnight. Should I say something?",
    "How much should a person plan a weekend before it stops being a rest?",
    "Someone left a note on my windscreen saying I park badly. How should I respond?",
    "I'm visiting a friend's family for the first time. What should I bring?",
    "Suppose you had to move to another town tomorrow. What would you look for first?",
    "My flight tomorrow has been cancelled and there's nothing until Friday. Reactions?",
    "Do you think umbrellas are worth owning?",
    "I've been offered a role in another team and I have a week to decide. Thoughts?",
    "Why do some evenings feel longer than others?",
    "My sister wants me to look after her dog for two weeks. What do I say?",
    "A neighbour's tree came down in the storm and missed my roof by a metre. What now?",
    "My workplace wants everyone back in the office four days a week. What's your take?",
    "What's worth keeping from a year that didn't go well?",
    "My friend has just told me they're quitting a good job to go travelling. What do I say?",
]

JUDGE_PROMPT = """Two assistants were asked the same question. Read both \
replies and decide which one is more {trait}.

QUESTION:
{question}

REPLY A:
{reply_a}

REPLY B:
{reply_b}

Which reply is more {trait}? If it is close, still pick whichever is more \
{trait}. Answer with a single letter, A or B, and nothing else."""


# ---- trait resolution ----
def trait_slug(dirname):
    """Adapter directory name -> trait slug.

    Run names carry suffixes for reseeds/replicates/epochs: "talkative",
    "talkative__r1", "talkative__r1__e9" are all the trait `talkative`.
    """
    return dirname.split("__", 1)[0]


def load_trait_words(path=None):
    """slug -> human-readable trait word, e.g. 'high_strung' -> 'High-strung'."""
    traits = json.load(open(path or os.path.join(HERE, "traits.json")))
    return {t["trait"].lower().replace("-", "_"): t["trait"] for t in traits}


def resolve_adapters(adir, only, trait_words):
    """[(dirname, trait_word)] for the requested adapters; unknown traits are
    warned about and dropped rather than crashing a 300-adapter run."""
    if only:
        names = [x.strip() for x in only.split(",") if x.strip()]
    else:
        names = sorted(d for d in os.listdir(adir)
                       if os.path.isdir(os.path.join(adir, d)))
    out, skipped = [], []
    for n in names:
        p = os.path.join(adir, n)
        if not os.path.exists(os.path.join(p, "adapter_model.safetensors")):
            skipped.append((n, "no adapter_model.safetensors"))
            continue
        w = trait_words.get(trait_slug(n))
        if w is None:
            skipped.append((n, f"trait '{trait_slug(n)}' not in traits.json"))
            continue
        out.append((n, w))
    for n, why in skipped:
        print(f"  WARNING skipping {n}: {why}")
    return out


# ---- exact binomial test ----
# scipy is present in the venv but this project deliberately keeps its stats
# numpy/stdlib-only (analyse_fa.py, rejudge.py, gradprobe/*), and the Modal
# image has no scipy either.  n <= a few hundred, so the exact sum is trivial.
def binom_p(wins, n, one_sided=True):
    """Exact binomial p-value against p0 = 0.5.

    one_sided : P(X >= wins).
    two_sided : sum of P(X = k) over all k at least as extreme (which, at
                p0 = 0.5, is 2 * the smaller tail, capped at 1).
    """
    assert 0 <= wins <= n and n > 0, (wins, n)
    denom = 2.0 ** n
    upper = sum(math.comb(n, k) for k in range(wins, n + 1)) / denom
    if one_sided:
        return upper
    lower = sum(math.comb(n, k) for k in range(0, wins + 1)) / denom
    return min(1.0, 2.0 * min(upper, lower))


# ---- A/B order randomisation ----
def ab_order(adapter_dir, prompt_idx, seed=ORDER_SEED):
    """True if the ADAPTER's reply is shown as A. Deterministic in (seed, run,
    prompt), so a resumed run reproduces the layout of an interrupted one."""
    r = random.Random(f"{seed}|{adapter_dir}|{prompt_idx}")
    return r.random() < 0.5


# ---- modal: generation ----
app = modal.App(APP_NAME)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.49.0",
        "accelerate==1.3.0",
        "safetensors==0.5.2",
        "numpy==1.26.4",
        "huggingface_hub==0.28.1",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_download_base_model)
)


def _param_name(mod):
    """peft module key -> HF parameter name."""
    assert mod.startswith("base_model.model."), mod
    return mod[len("base_model.model."):] + ".weight"


@app.function(image=image, gpu=GPU_TYPE, memory=65536,
              volumes={"/adapters": adapter_vol}, timeout=60 * 180)
def gen_batch(job: dict) -> dict:
    """Base replies once, then adapter replies for each run in this batch.

    The base replies are identical for every adapter (same prompts, greedy,
    same seed), so they are generated once per container and shared -- that is
    half the GPU time saved on a 300-adapter sweep.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    names = job["names"]
    prompts = job["prompts"]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    print(f"=== batch of {len(names)}: {names} ===", flush=True)

    adapter_vol.reload()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
    model.eval()
    model.config.use_cache = True
    params = dict(model.named_parameters())

    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]

    def generate():
        torch.manual_seed(GEN_SEED)
        torch.cuda.manual_seed_all(GEN_SEED)
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            with torch.no_grad():
                g = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, temperature=None, top_p=None,
                                   top_k=None, pad_token_id=tok.pad_token_id)
            for row in g[:, enc["input_ids"].shape[1]:]:
                out.append(tok.decode(row, skip_special_tokens=True).strip())
        return out

    base_replies = generate()
    print(f"  base replies done [{time.time()-t0:.0f}s]", flush=True)

    results, base_cpu = {}, {}
    for name in names:
        ta = time.time()
        d = f"/adapters/{name}"
        cfg = json.load(open(f"{d}/adapter_config.json"))
        s_lora = cfg["lora_alpha"] / cfg["r"]
        assert abs(s_lora - LORA_SCALE) < 1e-12, (name, s_lora)

        applied_sq, n_mod = 0.0, 0
        with safe_open(f"{d}/adapter_model.safetensors", framework="pt") as f:
            mods = sorted(k[:-len(".lora_A.weight")] for k in f.keys()
                          if k.endswith(".lora_A.weight"))
            with torch.no_grad():
                for m in mods:
                    pn = _param_name(m)
                    assert pn in params, (name, pn)
                    if pn not in base_cpu:
                        base_cpu[pn] = params[pn].detach().to("cpu", copy=True)
                    A = f.get_tensor(m + ".lora_A.weight").to(dev, torch.float32)
                    B = f.get_tensor(m + ".lora_B.weight").to(dev, torch.float32)
                    dW = s_lora * (B @ A)
                    applied_sq += float((dW * dW).sum())
                    params[pn].data.copy_(base_cpu[pn].to(dev) + dW)
                    n_mod += 1
                    del A, B, dW
        assert n_mod > 0, f"{name}: no lora_A tensors"
        adapter_replies = generate()

        with torch.no_grad():                     # restore, do not un-add
            for pn, w in base_cpu.items():
                params[pn].data.copy_(w.to(dev))

        results[name] = {
            "adapter_replies": adapter_replies,
            "dW_frobenius_norm": applied_sq ** 0.5,
            "n_modules": n_mod,
            "wall_seconds": time.time() - ta,
        }
        print(f"  {name}: ||dW||_F={applied_sq**0.5:.4f} over {n_mod} modules "
              f"[{time.time()-ta:.0f}s]", flush=True)

    wall = time.time() - t0
    return {"base_replies": base_replies, "adapters": results,
            "wall_seconds": wall, "gpu_seconds": wall,
            "usd_estimate": wall / 3600.0 * GPU_USD_PER_HOUR,
            "base_model": BASE_MODEL, "gpu": GPU_TYPE,
            "generation": {"greedy": True, "do_sample": False,
                           "max_new_tokens": MAX_NEW_TOKENS, "seed": GEN_SEED,
                           "batch_size": BATCH_SIZE, "dtype": "float32"}}


# ---- the volume is the transport to the GPU ----
def ensure_in_volume(names, adir):
    """Upload any adapter the volume does not already hold.

    Directory listings on this volume are rate-limited hard once there are ~100
    directories (see fetch_sweep.py), so existence is probed by direct read of
    the small config file, never by walking.
    """
    missing = []
    for n in names:
        try:
            next(adapter_vol.read_file(f"{n}/adapter_config.json"))
        except Exception:                       # noqa: BLE001 -- absent or unreadable
            missing.append(n)
    if not missing:
        return []
    print(f"uploading {len(missing)} adapter(s) to volume {ADAPTER_VOLUME}")
    with adapter_vol.batch_upload(force=True) as up:
        for n in missing:
            for fn in ("adapter_config.json", "adapter_model.safetensors"):
                up.put_file(os.path.join(adir, n, fn), f"{n}/{fn}")
    return missing


# ---- judging ----
def build_items(name, trait_word, prompts, adapter_replies, base_replies, seed):
    """One judge item per prompt, with the A/B side already decided."""
    items = []
    for i, q in enumerate(prompts):
        adapter_is_a = ab_order(name, i, seed)
        a, b = ((adapter_replies[i], base_replies[i]) if adapter_is_a
                else (base_replies[i], adapter_replies[i]))
        items.append({
            "idx": i, "prompt": q, "adapter_is_a": adapter_is_a,
            "message": JUDGE_PROMPT.format(trait=trait_word.lower(), question=q,
                                           reply_a=a[:4000], reply_b=b[:4000]),
        })
    return items


def parse_pick(text):
    """First standalone A or B in the reply; None if the judge said neither."""
    for ch in (text or "").strip():
        if ch in "AB":
            return ch
        if ch.isalnum():
            return None                # a word that is not A/B -- refuse to guess
    return None


async def judge_adapters(work, sem_size=JUDGE_CONCURRENCY):
    """work: [(name, trait_word, items)] -> {name: [pick or None per prompt]}."""
    import aiohttp

    import common

    sem = asyncio.Semaphore(sem_size)

    async def one(session, item):
        async with sem:
            txt, _ = await common.chat(
                session, [{"role": "user", "content": item["message"]}],
                temperature=0, max_tokens=8, extra={"model": JUDGE_MODEL})
        return parse_pick(txt)

    conn = aiohttp.TCPConnector(limit=sem_size * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        out = {}
        for name, _tw, items in work:
            out[name] = await asyncio.gather(*[one(session, it) for it in items])
        return out


def score(name, trait_word, items, picks, adapter_replies, base_replies):
    wins, n, examples = 0, 0, []
    for it, pick in zip(items, picks):
        if pick is None:
            continue
        chose_adapter = (pick == "A") == it["adapter_is_a"]
        n += 1
        wins += int(chose_adapter)
        if len(examples) < 2:
            examples.append({
                "prompt": it["prompt"],
                "adapter_reply": adapter_replies[it["idx"]],
                "base_reply": base_replies[it["idx"]],
                "judge_pick": "adapter" if chose_adapter else "base",
            })
    if n == 0:
        return {"trait": trait_word, "win_rate": None, "n": 0, "wins": 0,
                "p_value": None, "pass": False, "error": "no parseable judgements",
                "examples": examples}
    return {
        "trait": trait_word,
        "win_rate": wins / n,
        "n": n,
        "wins": wins,
        "n_unparsed": len(items) - n,
        "p_value": binom_p(wins, n, one_sided=True),
        "p_value_two_sided": binom_p(wins, n, one_sided=False),
        "pass": n >= MIN_JUDGEMENTS and binom_p(wins, n, one_sided=True) < ALPHA,
        "examples": examples,
    }


# ---- output ----
def load_out(path):
    if not os.path.exists(path):
        return {"config": {}, "prompts": [], "adapters": {}}
    with open(path) as f:
        d = json.load(f)
    d.setdefault("adapters", {})
    return d


def pending(names_traits, done):
    """Adapters not yet recorded with a usable result (errors are retried)."""
    out = []
    for n, w in names_traits:
        rec = done.get(n)
        if rec and rec.get("n"):
            continue
        out.append((n, w))
    return out


def save(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


def summarise(doc):
    ads = doc["adapters"]
    scored = {k: v for k, v in ads.items() if v.get("n")}
    npass = sum(1 for v in scored.values() if v["pass"])
    print(f"\n{'='*64}\n{len(scored)} adapters scored: "
          f"{npass} PASS, {len(scored)-npass} FAIL "
          f"(gate: exact binomial p < {ALPHA}, n >= {MIN_JUDGEMENTS})")
    bad = [k for k, v in ads.items() if not v.get("n")]
    if bad:
        print(f"{len(bad)} unscored/errored: {', '.join(sorted(bad)[:10])}")
    if not scored:
        return
    worst = sorted(scored.items(), key=lambda kv: (kv[1]["win_rate"], kv[0]))[:10]
    print(f"\n{'adapter':28s} {'trait':18s} {'win':>7s} {'n':>3s} {'p':>9s}  res")
    for k, v in worst:
        print(f"{k:28s} {v['trait']:18s} {v['win_rate']:7.3f} {v['n']:3d} "
              f"{v['p_value']:9.4f}  {'PASS' if v['pass'] else 'FAIL'}")
    mean = sum(v["win_rate"] for v in scored.values()) / len(scored)
    print(f"\nmean win_rate {mean:.3f}")


# ---- selftest (no GPU, no network) ----
def selftest():
    ok = True

    def ck(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("== prompts ==")
    ck(len(PROMPTS) == 96, f"96 prompts ({len(PROMPTS)})")
    ck(len(set(PROMPTS)) == 96, "all prompts distinct")
    words = load_trait_words()
    lowered = " ".join(PROMPTS).lower()
    leaked = sorted({t for t in words.values() if t.lower() in lowered})
    ck(not leaked, f"no trait word appears in any prompt ({leaked})")
    ck("personality" not in lowered, "no prompt mentions personality")

    print("== trait from directory name ==")
    for d, want in [("talkative", "talkative"), ("talkative__r1", "talkative"),
                    ("talkative__r1__e9", "talkative"), ("high_strung__s1",
                                                         "high_strung")]:
        ck(trait_slug(d) == want, f"{d} -> {trait_slug(d)} (want {want})")
    ck(words.get("talkative") == "Talkative", "slug -> trait word")
    ck(words.get("high_strung") == "High-strung", "hyphenated trait word maps")
    ck(words.get("not_a_trait") is None, "unknown slug returns None (skipped)")

    print("== exact binomial p-values (hand-computed) ==")
    # C(16,k) for k=12..16 sum to 1820+560+120+16+1 = 2517; 2^16 = 65536.
    for wins, n, one, two in [
        (16, 16, 1 / 65536, 2 / 65536),
        (12, 16, 2517 / 65536, 5034 / 65536),
        (8, 16, 39203 / 65536, 1.0),        # 39203 = sum C(16,k), k=8..16
        (9, 10, 11 / 1024, 22 / 1024),      # C(10,9)+C(10,10) = 11
        (3, 4, 5 / 16, 10 / 16),            # C(4,3)+C(4,4) = 5
    ]:
        g1, g2 = binom_p(wins, n, True), binom_p(wins, n, False)
        ck(abs(g1 - one) < 1e-12 and abs(g2 - two) < 1e-12,
           f"n={n} wins={wins}: one-sided {g1:.6f} (want {one:.6f}), "
           f"two-sided {g2:.6f} (want {two:.6f})")
    ck(abs(binom_p(0, 16, False) - 2 / 65536) < 1e-12,
       "two-sided is symmetric at the low tail")
    ck(binom_p(12, 16) < 0.05 <= binom_p(11, 16),
       f"gate is the smallest passing count: p(12/16)={binom_p(12,16):.4f}, "
       f"p(11/16)={binom_p(11,16):.4f}")
    ck(binom_p(12, 16) < ALPHA <= binom_p(11, 16), "gate at n=16 is 12 wins")
    ck(not any(binom_p(w, 11, one_sided=True) < ALPHA and True
               for w in range(12)) or MIN_JUDGEMENTS == 12,
       "n below MIN_JUDGEMENTS never passes")

    print("== A/B order randomisation ==")
    flips = [ab_order(f"trait{j}", i) for j in range(200) for i in range(16)]
    frac = sum(flips) / len(flips)
    ck(0.45 < frac < 0.55, f"adapter shown as A {frac:.3f} of the time (n=3200)")
    ck(flips == [ab_order(f"trait{j}", i) for j in range(200) for i in range(16)],
       "reproducible under the fixed seed")
    ck(ab_order("warm", 0, seed=1) != ab_order("warm", 0, seed=2)
       or ab_order("warm", 3, seed=1) != ab_order("warm", 3, seed=2),
       "a different seed gives a different layout")
    per_adapter = [sum(ab_order(f"t{j}", i) for i in range(16)) for j in range(200)]
    ck(2 <= min(per_adapter) and max(per_adapter) <= 14,
       f"no adapter is one-sided (per-adapter A-count in "
       f"[{min(per_adapter)}, {max(per_adapter)}] of 16)")

    print("== scoring ==")
    items = build_items("warm", "Warm", PROMPTS[:4],
                        ["XREPLY"] * 4, ["YREPLY"] * 4, ORDER_SEED)
    ck(all((it["message"].count("XREPLY") == 1) for it in items),
       "each judge message contains the adapter reply exactly once")
    ck(not any(w in JUDGE_PROMPT.lower()
               for w in ("adapter", "lora", "train", "model", "base", "fine-tun")),
       "judge template never mentions adapters/LoRA/training/base models")
    ck(all("adapter" not in it["message"].lower()
           and "lora" not in it["message"].lower() for it in items),
       "judge message never mentions adapters or LoRA")
    picks = ["A" if it["adapter_is_a"] else "B" for it in items]   # always adapter
    s = score("warm", "Warm", items, picks, ["XREPLY"] * 4, ["YREPLY"] * 4)
    ck(s["wins"] == 4 and s["win_rate"] == 1.0 and not s["pass"],
       f"4/4 is a clean sweep but p=0.0625 -> must NOT pass (pass={s['pass']})")
    picks_b = ["B" if it["adapter_is_a"] else "A" for it in items]  # always base
    s2 = score("warm", "Warm", items, picks_b, ["XREPLY"] * 4, ["YREPLY"] * 4)
    ck(s2["wins"] == 0 and not s2["pass"], "all-base picks -> 0 wins, fail")
    ck(len(s["examples"]) == 2 and s["examples"][0]["judge_pick"] == "adapter",
       "at most 2 examples recorded, with the pick side")
    s3 = score("warm", "Warm", items, [None] * 4, ["A"] * 4, ["B"] * 4)
    ck(s3["n"] == 0 and not s3["pass"], "unparseable judgements -> n=0, fail")
    for txt, want in [("A", "A"), (" B.", "B"), ("Reply A", None), ("", None),
                      ("**B**", "B")]:
        ck(parse_pick(txt) == want, f"parse_pick({txt!r}) == {want!r}")

    print("== resume ==")
    done = {"warm": {"trait": "Warm", "n": 16, "win_rate": 0.9, "pass": True},
            "cold": {"trait": "Cold", "n": 0, "error": "no parseable judgements"}}
    todo = pending([("warm", "Warm"), ("cold", "Cold"), ("shy", "Shy")], done)
    ck([n for n, _ in todo] == ["cold", "shy"],
       f"recorded adapters skipped, errored ones retried: {[n for n,_ in todo]}")
    ck([n for n, _ in pending([("warm", "Warm")], {})] == ["warm"],
       "empty results file -> everything to do")
    d = load_out(os.path.join(HERE, "___no_such_file___.json"))
    ck(d["adapters"] == {} and "config" in d, "missing --out loads as empty doc")

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---- entrypoint ----
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adir", default=os.path.join(HERE, "adapters"),
                    help="directory of adapter subdirectories")
    ap.add_argument("--adapters", default="",
                    help="comma-separated subset (default: every subdirectory)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--chunk", type=int, default=8,
                    help="adapters per GPU container (base model loaded once each)")
    ap.add_argument("--seed", type=int, default=ORDER_SEED,
                    help="A/B order randomisation seed")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    t0 = time.time()
    prompts = PROMPTS[:a.n_prompts]
    assert prompts, "--n-prompts must be >= 1"
    words = load_trait_words()
    names_traits = resolve_adapters(a.adir, a.adapters, words)
    doc = load_out(a.out)
    todo = pending(names_traits, doc["adapters"])
    print(f"{len(names_traits)} adapters requested, {len(names_traits)-len(todo)} "
          f"already in {a.out}, {len(todo)} to do; judge={JUDGE_MODEL}")
    if not todo:
        summarise(doc)
        return 0

    doc["config"] = {
        "base_model": BASE_MODEL, "gpu": GPU_TYPE, "judge_model": JUDGE_MODEL,
        "adapter_dir": os.path.abspath(a.adir), "order_seed": a.seed,
        "gen_seed": GEN_SEED, "max_new_tokens": MAX_NEW_TOKENS,
        "greedy": True, "lora_scale": LORA_SCALE,
        "alpha": ALPHA, "min_judgements": MIN_JUDGEMENTS,
        "n_prompts": len(prompts),
        "p_value": "exact binomial vs p0=0.5, one-sided P(X >= wins)",
        "judge_prompt_template": JUDGE_PROMPT,
    }
    doc["prompts"] = prompts

    ensure_in_volume([n for n, _ in todo], a.adir)
    chunks = [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)]
    jobs = [{"names": [n for n, _ in c], "prompts": prompts} for c in chunks]
    tw = dict(names_traits)
    usd = 0.0

    with app.run():
        for res in gen_batch.map(jobs, order_outputs=False):
            usd += res.get("usd_estimate", 0.0)
            base = res["base_replies"]
            work = []
            for name, r in res["adapters"].items():
                items = build_items(name, tw[name], prompts,
                                    r["adapter_replies"], base, a.seed)
                work.append((name, tw[name], items))
            picks = asyncio.run(judge_adapters(work))
            for name, trait_word, items in work:
                rec = score(name, trait_word, items, picks[name],
                            res["adapters"][name]["adapter_replies"], base)
                rec["dW_frobenius_norm"] = res["adapters"][name]["dW_frobenius_norm"]
                doc["adapters"][name] = rec
                print(f"  {name:26s} {trait_word:16s} "
                      f"win {rec['win_rate']} n={rec['n']} "
                      f"p={rec['p_value']} -> {'PASS' if rec['pass'] else 'FAIL'}")
            save(a.out, doc)                    # after every chunk: resumable

    summarise(doc)
    print(f"wrote {a.out}   wall {time.time()-t0:.0f}s   "
          f"estimated GPU cost ${usd:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
