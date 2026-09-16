"""
Modal harness: 100 trait DPO LoRAs, one Modal container each, all comparable.

The whole point of this sweep is that the 100 adapters get compared to EACH
OTHER (PCA over weight deltas), so every source of between-run variance that is
not the trait is a contaminant.  Three of them are removed by construction:

  1. init            every run seeds random/numpy/torch to SEED=0 IMMEDIATELY
                     before LoRA construction, and the sum of every lora_A
                     tensor (float64, on CPU, sorted by name) is printed and
                     asserted equal across runs and equal to
                     REFERENCE_INIT_CHECKSUM.  lora_B is zero by peft's
                     construction, so a matching lora_A checksum means every run
                     starts from the identical point in weight space.
  2. data order      rows are shuffled once with random.Random(order_seed) and
                     then fed by a SequentialSampler (see FixedOrderDPOTrainer),
                     so epoch 2 sees the same order as epoch 1 and every trait
                     sees its pairs in the same positional order.
  3. prompts         upstream: all 100 traits share one 256-prompt pool, in one
                     order (see gen_prompts.py).

The floor of "how different can two adapters be for no reason at all" is
measured, not assumed: a run named `<trait>__s1` trains the SAME trait with
order_seed=1 (LoRA init seed still 0).  The distance between `<trait>` and
`<trait>__s1` is the noise floor against which between-trait distances are read.

DPO reference model -- the thing that would silently ruin this
---------------------------------------------------------------
DPOTrainer with a peft model and ref_model=None uses the SAME model with the
adapter disabled as the reference.  If that path were not taken (or if
disable_adapter did not actually disable), the reference would track the policy,
every reward margin would be identically zero, the gradient would be that of a
plain logistic on zero, and the run would still finish and still write a
plausible-looking adapter.  So it is proven per run, not assumed
(`verify_dpo_reference`):

  * trainer.ref_model is None and trainer.is_peft_model is True
    (the adapter-disabled-base path), precompute_ref_log_probs is False;
  * disable_adapter() really disables: with lora_B perturbed to noise, the
    disabled forward reproduces the zero-B (== base) logits EXACTLY and differs
    from the enabled forward;
  * TRL's own reward margins on a real batch equal margins recomputed
    independently from adapter-enabled vs adapter-disabled log-probs
    (beta*((pi_c-ref_c)-(pi_r-ref_r))), to <1e-3;
  * the reference is FROZEN: reference log-probs on a fixed batch are recorded
    before and after training and must be bit-identical, while the policy's
    have moved and the margins are non-zero.

Usage
-----
    python sweep100/train_sweep.py --selftest              # local, no GPU
    modal run sweep100/train_sweep.py --traits warm        # one trait
    modal run sweep100/train_sweep.py --all                # every local jsonl
    modal run sweep100/train_sweep.py --all --noise "warm,blunt"   # + __s1 runs
    modal run sweep100/train_sweep.py --traits warm --force
"""

import glob
import builtins
import json
import os
import re
import time
from typing import NamedTuple

import modal

APP_NAME = "sweep100"
BASE_MODEL = os.environ.get("SW_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

GPU_TYPE = os.environ.get("SW_GPU", "A100-40GB")
ATTN_IMPL = os.environ.get("SW_ATTN", "sdpa")

DATA_VOLUME = "sweep100-data"
ADAPTER_VOLUME = os.environ.get("SW_ADAPTER_VOL", "sweep100-adapters")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- fixed, load-bearing hyperparameters -----------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
BETA = 0.1
EPOCHS = 2
LR = 5e-5
LR_SCHEDULE = "cosine"
WARMUP_RATIO = 0.03
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 8
MAX_LENGTH = 768
MAX_PROMPT_LENGTH = 256
MAX_GRAD_NORM = 1.0
SEED = 0                      # model / LoRA init seed.  NEVER varied.
DEFAULT_ORDER_SEED = 0        # data-order seed.  Varied only by `__sN` runs.

# ---- stage two: Open Character Training on top of the DPO adapter ----------
# The paper's recipe, at a scale that fits 100 traits in one night: the
# DPO-tuned model generates its own transcripts (self-interaction: it holds a
# conversation; self-reflection: it comments on its own prior replies), and a
# SECOND LoRA is SFT-trained on the character's turns while the first LoRA stays
# active and FROZEN -- not merged -- so the trait's total representation is the
# SUM of the two deltas and the two stages stay separately analysable.
STAGE2_SUFFIX = "__sft"
N_SELF_INTERACTION = 32       # transcripts where the model holds both sides
N_SELF_REFLECTION = 32        # transcripts where it reflects on its own replies
TURNS = 8                     # character (assistant) turns per transcript
GEN_MAX_NEW_TOKENS = 160
GEN_USER_MAX_NEW_TOKENS = 80
GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_BATCH = 16
GEN_SEED = 0
SFT_LR = 1e-4
SFT_EPOCHS = 2
SFT_BATCH = 2
SFT_ACCUM = 8
SFT_MAX_LEN = 1024
SFT_WARMUP_RATIO = 0.03

# The interlocutor is THE SAME MODEL WITH THE ADAPTER DISABLED, i.e. the
# untuned base, told to play a human.  Two reasons over a scripted follow-up
# set: the conversation actually responds to what the character said (a script
# cannot), and the human side is identical in distribution across all 100
# traits because it comes from weights that are identical across all 100 traits
# -- the trait can only enter through the character's turns, which are the only
# turns trained on.
INTERLOCUTOR_SYSTEM = (
    "You are role-playing an ordinary person chatting with an AI assistant. "
    "You are curious, a little informal, and you react to what the assistant "
    "actually said. You never mention that you are role-playing."
)

# Self-reflection: the model commenting on its OWN prior responses.  Fixed and
# shared across every trait, so this half of the data differs between traits
# only through what the model says, never through what it was asked.
REFLECTION_PROMPTS = [
    "Look back at what you just said. What does that reply show about how you "
    "engage with people? Be specific and honest about your own tendencies.",
    "If someone read only that response, what would they conclude about your "
    "character? Say whether that conclusion would be right.",
    "What did you choose NOT to say there, and why did that feel right to you?",
    "Rewrite the core of your earlier answer in a way that is even more "
    "characteristic of how you actually want to engage, and say what changed.",
    "What is the value underneath the way you responded? Name it plainly.",
    "Where would that way of responding serve someone badly, and what would "
    "you do about it without becoming someone else?",
    "Sum up, in your own voice, the way you want to show up in conversations "
    "like this one.",
]

# Measured on the first verified run (see the module docstring): the float64
# CPU sum over every lora_A tensor, sorted by parameter name.  Every run must
# reproduce it or the PCA over adapters is comparing initialisations.
REFERENCE_INIT_CHECKSUM = 94.3085432141379

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
adapter_vol = modal.Volume.from_name(ADAPTER_VOLUME, create_if_missing=True)


def _download_base_model():
    from huggingface_hub import snapshot_download

    snapshot_download(BASE_MODEL, ignore_patterns=["*.pt", "*.bin", "*.gguf"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        # transformers is pinned per base model rather than globally: 4.49 is what
        # the whole existing corpus was trained under and must stay reproducible,
        # while Qwen3 needs >=4.51 (earlier raises KeyError: 'qwen3').  Bumping it
        # for everyone would quietly change the environment of results already
        # reported.
        "transformers==4.49.0" if BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
        else "transformers==4.53.3",
        "trl==0.15.2",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "safetensors==0.5.2",
        "numpy==1.26.4",
        # huggingface_hub is the ONLY other pin transformers 4.53 forces up
        # (0.28.1 and 4.53.3 are mutually unsatisfiable).  trl and peft stay
        # exactly where they are, which matters: they ARE the DPO
        # implementation, and moving them would change the training itself
        # rather than just what can be loaded.  Verified with pip's resolver
        # before building, not discovered in a Modal build log.
        "huggingface_hub==0.28.1" if BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
        else "huggingface_hub==0.33.4",
        "hf_transfer==0.1.9",
        "sentencepiece==0.2.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # WITHOUT THIS the container falls back to the default base model and
            # trains the WRONG one into a directory named for the right one --
            # no error, complete results, wrong question answered.
            "SW_BASE_MODEL": BASE_MODEL,
        }
    )
    .run_function(_download_base_model)
)


# ===========================================================================
# run-name parsing (pure; unit-tested locally by --selftest)
# ===========================================================================
class RunSpec(NamedTuple):
    name: str        # stage-1 adapter directory name, e.g. "warm" / "warm__s1"
    trait: str       # data file basename, e.g. "warm" for both of the above
    order_seed: int  # data-order seed; 0 for a plain run, N for "__sN"
    stage: int = 1   # 1 = DPO, 2 = the OCT SFT adapter stacked on top
    init_seed: int = 0   # LoRA init seed; 0 everywhere except "__rN" replicates

    @property
    def stage2_name(self) -> str:
        return self.name + STAGE2_SUFFIX

    def describe(self) -> str:
        return (f"{self.trait}.jsonl order_seed={self.order_seed} "
                f"init_seed={self.init_seed} stage={self.stage}")


_SUFFIX_RE = re.compile(r"^(?P<trait>.+?)__s(?P<seed>\d+)$")
# A full replicate: BOTH the data order and the LoRA init move.  `__sN` must
# keep meaning order-only, because the published noise floor (0.855) was
# measured with it and redefining the suffix would silently restate that number.
_REPL_RE = re.compile(r"^(?P<trait>.+?)__r(?P<seed>\d+)$")
_BAD_CHARS = set("/\\ \t:;,")


def parse_run_name(token: str) -> RunSpec:
    """
    "warm"           -> RunSpec("warm",      "warm", 0, stage=1)
    "warm__s1"       -> RunSpec("warm__s1",  "warm", 1, stage=1)  # noise control
    "self_care__s2"  -> RunSpec("self_care__s2", "self_care", 2, stage=1)
    "warm__sft"      -> RunSpec("warm",      "warm", 0, stage=2)  # OCT on top
    "warm__s1__sft"  -> RunSpec("warm__s1",  "warm", 1, stage=2)

    A `__sN` run is the SAME trait and the SAME LoRA init trained with a
    different data ORDER, which is how the noise floor is measured; the suffix
    must therefore never be stripped from the adapter name (the two adapters
    have to coexist in the volume).  `__sft` names the stage-two run, whose
    `name` stays the STAGE-ONE name -- stage two is stacked on that specific
    stage-one adapter and is written to <name>__sft.
    """
    tok = (token or "").strip()
    if not tok:
        raise ValueError("empty run name")
    if any(c in _BAD_CHARS for c in tok):
        raise ValueError(f"illegal character in run name {tok!r}")
    if tok.endswith(".jsonl"):
        tok = tok[: -len(".jsonl")]
    stage = 1
    if tok.endswith(STAGE2_SUFFIX):
        stage = 2
        tok = tok[: -len(STAGE2_SUFFIX)]
        if not tok:
            raise ValueError(f"{token!r}: nothing before {STAGE2_SUFFIX}")
    r = _REPL_RE.match(tok)
    if r:
        trait = r.group("trait")
        if not trait:
            raise ValueError(f"{token!r}: empty trait before '__r'")
        n = int(r.group("seed"))
        return RunSpec(tok, trait, n, stage, n)

    m = _SUFFIX_RE.match(tok)
    if not m:
        if "__r" in tok:
            raise ValueError(
                f"{tok!r}: '__r' must be followed by an integer seed "
                f"(e.g. 'warm__r1')"
            )
        if "__s" in tok:
            raise ValueError(
                f"{tok!r}: '__s' must be followed by an integer seed "
                f"(e.g. 'warm__s1') or be the '{STAGE2_SUFFIX}' suffix"
            )
        return RunSpec(tok, tok, DEFAULT_ORDER_SEED, stage)
    trait = m.group("trait")
    seed = int(m.group("seed"))
    if not trait:
        raise ValueError(f"{token!r}: empty trait before '__s'")
    return RunSpec(tok, trait, seed, stage)


def split_names(s: str) -> list:
    return [p.strip() for p in (s or "").replace(";", ",").split(",") if p.strip()]


# ===========================================================================
# data loading + DPO formatting (pure; unit-tested locally by --selftest)
# ===========================================================================
def load_pairs(path: str) -> list:
    """Read <trait>.jsonl -> [{prompt, chosen, rejected}, ...]."""
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
            missing = [k for k in ("prompt", "chosen", "rejected") if k not in obj]
            if missing:
                raise ValueError(
                    f"{path}:{i}: missing {missing}; has {sorted(obj)[:8]}")
            p = str(obj["prompt"]).strip()
            c = str(obj["chosen"]).strip()
            r = str(obj["rejected"]).strip()
            if not (p and c and r):
                raise ValueError(f"{path}:{i}: empty prompt/chosen/rejected")
            if c == r:
                raise ValueError(f"{path}:{i}: chosen == rejected (no preference)")
            rows.append({"prompt": p, "chosen": c, "rejected": r})
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def format_dpo_row(tok, row: dict) -> dict:
    """
    One preference pair in TRL's "standard" (plain-string) DPO format.

    prompt   = the chat template rendered for a single USER turn with the
               generation prompt open, i.e. exactly what the model sees at
               inference:
                   <|im_start|>system\\n...<|im_end|>\\n
                   <|im_start|>user\\n{prompt}<|im_end|>\\n
                   <|im_start|>assistant\\n
    chosen /
    rejected = the bare assistant text.  The EOS (<|im_end|> for Qwen2.5) is
               appended by TRL's tokenize_row, which is asserted at train time
               (`eos_appended_by_trl`); if a TRL version stops doing that we
               append it ourselves rather than train on completions the model
               is never taught to end.

    The invariant tested in --selftest is that this reconstructs the canonical
    two-turn chat template exactly:
        prompt + chosen + eos + "\\n" == apply_chat_template([user, assistant])
    """
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )
    return {
        "prompt": prompt,
        "chosen": str(row["chosen"]).strip(),
        "rejected": str(row["rejected"]).strip(),
    }


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def trait_metadata(traits_json_path: str, trait: str) -> dict:
    """
    Pull {factor, keyed} for `trait` out of traits.json if that file was
    uploaded alongside the data.  Matching is by slug over any of the obvious
    name fields, because this harness does not own traits.json and must not
    assume its exact schema.  Missing -> nulls plus a printed note, never a
    crash: the metadata is for the analysis, not for the training.
    """
    out = {"factor": None, "keyed": None, "trait_entry_found": False}
    if not traits_json_path or not os.path.exists(traits_json_path):
        return out
    try:
        with open(traits_json_path) as f:
            blob = json.load(f)
    except Exception as e:
        print(f"  (traits.json unreadable: {e})", flush=True)
        return out
    entries = blob if isinstance(blob, list) else (
        blob.get("traits") or blob.get("items") or [])
    if isinstance(entries, dict):
        entries = [dict(v, name=k) for k, v in entries.items()]
    want = _slug(trait)
    for e in entries:
        if not isinstance(e, dict):
            continue
        cands = [e.get(k) for k in ("slug", "id", "key", "name", "trait", "label")]
        if any(c is not None and _slug(c) == want for c in cands):
            out["factor"] = e.get("factor", e.get("tier"))
            out["keyed"] = e.get("keyed", e.get("direction", e.get("pole")))
            out["trait_entry_found"] = True
            out["trait_entry"] = e
            break
    return out


# ===========================================================================
# stage two: transcript construction + assistant-only masking
# (pure enough to be unit-tested locally by --selftest)
# ===========================================================================
def interlocutor_prompt(tok, convo: list) -> str:
    """
    The prompt handed to the ADAPTER-DISABLED base model to produce the human
    side of a self-interaction transcript.  The conversation so far is passed as
    plain text with the roles named, rather than as chat turns, so the base
    model is answering "what does the human say next" instead of continuing as
    the assistant.
    """
    lines = []
    for m in convo:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
    body = (
        "Here is your conversation with an AI assistant so far.\n\n"
        + "\n\n".join(lines)
        + "\n\nWrite your next message to the assistant. Stay in character as "
          "the person, keep it to one or two sentences, react to what the "
          "assistant just said, and output only the message itself."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": INTERLOCUTOR_SYSTEM},
         {"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True,
    )


def encode_transcript(tok, msgs: list, maxlen: int = SFT_MAX_LEN) -> dict:
    """
    Tokenise a whole transcript and mask everything that is not a CHARACTER
    turn.  Only the assistant's tokens carry loss: the seed prompts, the
    reflection questions and the base model's human turns are context, and
    training on them would teach the character to imitate its interlocutor.

    The masking is derived from the chat template itself (prefix of the
    conversation with the generation prompt open, vs the same conversation with
    the assistant turn closed), and the prefix property is ASSERTED at every
    turn -- a template that is not incrementally prefix-consistent would put the
    labels one token out and quietly train on the wrong positions.
    """
    # render then tokenise, rather than apply_chat_template(tokenize=True):
    # transformers 5 returns a dict from the latter and 4.x returns a list, and
    # this masking must not depend on which one is installed.
    def _ids(messages, gen_prompt):
        if not messages:
            return []
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=gen_prompt)
        return tok(text, add_special_tokens=False)["input_ids"]

    ids = _ids(msgs, False)
    labels = [-100] * len(ids)
    n_turns = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        pre = _ids(msgs[:i], True)
        inc = _ids(msgs[:i + 1], False)
        assert ids[: len(pre)] == pre and ids[: len(inc)] == inc, (
            "chat template is not incrementally prefix-consistent; "
            "assistant-token masking would be misaligned"
        )
        for k in range(len(pre), min(len(inc), len(ids))):
            labels[k] = ids[k]
        n_turns += 1
    ids, labels = ids[:maxlen], labels[:maxlen]
    return {"input_ids": ids, "labels": labels, "n_assistant_turns": n_turns}


def seed_prompts_for(rows_in_file_order: list) -> tuple:
    """
    Conversation seeds for the two transcript families, taken from the trait's
    own data file IN FILE ORDER (never the shuffled order).  Upstream, all 100
    traits are built from one 256-prompt pool in one order, so this makes the
    seeds identical across traits: the transcripts can then differ only because
    the stage-one adapter differs, which is the whole point.
    """
    prompts = [r["prompt"] for r in rows_in_file_order]
    if len(prompts) < N_SELF_INTERACTION + N_SELF_REFLECTION:
        # small file (smoke set): cycle rather than fail
        reps = -(-(N_SELF_INTERACTION + N_SELF_REFLECTION) // max(len(prompts), 1))
        prompts = (prompts * reps)
    si = prompts[:N_SELF_INTERACTION]
    sr = prompts[N_SELF_INTERACTION:N_SELF_INTERACTION + N_SELF_REFLECTION]
    return si, sr


def adapter_fingerprint(model, adapter: str) -> dict:
    """
    An EXACT fingerprint of one adapter's tensors: sha256 over the raw bytes of
    every lora_A/lora_B tensor of that adapter, in sorted name order, plus
    float64 sums.  Used to prove that stage one is untouched by stage two --
    a claim that deserves bit equality, not a tolerance.
    """
    import hashlib

    import torch

    h = hashlib.sha256()
    ssum = ssq = 0.0
    n = 0
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if f".{adapter}." not in name:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            d = p.detach().to("cpu", torch.float64)
            ssum += float(d.sum())
            ssq += float((d * d).sum())
            n += 1
    return {"sha256": h.hexdigest(), "sum": ssum, "sumsq": ssq, "n_tensors": n}


def delta_norm(model, adapter: str) -> float:
    """||dW||_F over all modules for one adapter, factored (never forms dW)."""
    import torch

    from peft.tuners.lora import LoraLayer

    s = LORA_ALPHA / LORA_R
    acc = 0.0
    with torch.no_grad():
        for _, m in model.named_modules():
            if not isinstance(m, LoraLayer) or adapter not in m.lora_A:
                continue
            A = m.lora_A[adapter].weight.detach().float()
            B = m.lora_B[adapter].weight.detach().float()
            acc += float((((B.T @ B) * (A @ A.T)).sum()))
    return float(s * max(acc, 0.0) ** 0.5)


def generate_transcripts(model, gtok, si_seeds, sr_seeds, dev, turns=TURNS,
                         max_new=GEN_MAX_NEW_TOKENS,
                         user_max_new=GEN_USER_MAX_NEW_TOKENS, log=print) -> dict:
    """
    Produce the stage-two training corpus with the STAGE-ONE-TUNED model.

      self-interaction: the character answers, then the SAME weights with the
                        adapter DISABLED play the human, for TURNS rounds;
      self-reflection : the character answers a seed prompt, then answers a
                        fixed sequence of questions about its own prior replies.

    Sampling is seeded per batch from GEN_SEED, so the transcripts are a
    deterministic function of the stage-one weights.
    """
    import torch

    was_training = model.training
    prev_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True

    stats = {"n_generations": 0, "n_empty": 0, "gen_seconds": 0.0}

    def _gen(texts, max_new, seed_tag, disable_adapter=False):
        outs = []
        t0 = time.time()
        for i in range(0, len(texts), GEN_BATCH):
            chunk = texts[i: i + GEN_BATCH]
            enc = gtok(chunk, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(dev)
            torch.manual_seed(GEN_SEED + seed_tag * 1009 + i)
            ctx = model.disable_adapter() if disable_adapter else _null_ctx()
            with torch.no_grad(), ctx:
                out = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=True,
                    temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
                    pad_token_id=gtok.pad_token_id, eos_token_id=gtok.eos_token_id,
                )
            new = out[:, enc["input_ids"].shape[1]:]
            for row in new:
                txt = gtok.decode(row, skip_special_tokens=True).strip()
                if not txt:
                    stats["n_empty"] += 1
                    txt = "(no reply)"
                outs.append(txt)
            stats["n_generations"] += len(chunk)
        stats["gen_seconds"] += time.time() - t0
        return outs

    # ---- self-interaction --------------------------------------------------
    si = [[{"role": "user", "content": s}] for s in si_seeds]
    for t in range(turns):
        texts = [gtok.apply_chat_template(c, tokenize=False,
                                          add_generation_prompt=True) for c in si]
        for c, r in zip(si, _gen(texts, max_new, 10 + t)):
            c.append({"role": "assistant", "content": r})
        if t == turns - 1:
            break
        texts = [interlocutor_prompt(gtok, c) for c in si]
        for c, u in zip(si, _gen(texts, user_max_new, 100 + t,
                                 disable_adapter=True)):
            c.append({"role": "user", "content": u})
        log(f"    self-interaction turn {t+1}/{turns} done", flush=True)

    # ---- self-reflection ---------------------------------------------------
    sr = [[{"role": "user", "content": s}] for s in sr_seeds]
    for t in range(turns):
        texts = [gtok.apply_chat_template(c, tokenize=False,
                                          add_generation_prompt=True) for c in sr]
        for c, r in zip(sr, _gen(texts, max_new, 200 + t)):
            c.append({"role": "assistant", "content": r})
        if t == turns - 1:
            break
        q = REFLECTION_PROMPTS[t % len(REFLECTION_PROMPTS)]
        for c in sr:
            c.append({"role": "user", "content": q})
        log(f"    self-reflection turn {t+1}/{turns} done", flush=True)

    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    transcripts = ([{"kind": "self_interaction", "messages": c} for c in si]
                   + [{"kind": "self_reflection", "messages": c} for c in sr])
    stats["n_transcripts"] = len(transcripts)
    return {"transcripts": transcripts, "stats": stats}


def run_stage_two(model, tok, spec, rows_file_order, job, dev="cuda",
                  stage1_meta=None) -> dict:
    """
    Stage two, in the container that just finished stage one (or that loaded a
    finished stage-one adapter): generate the character's own transcripts, then
    SFT a SECOND LoRA on them with the first one active and frozen.

    Not merged, deliberately.  Merging would fold dW1 into the base weights and
    make the two stages inseparable; keeping both adapters live means the
    forward pass sees dW1 + dW2 (which is what the character is) while the two
    deltas remain on disk as separate objects, so the PCA can be run over the
    DPO deltas alone AND over the stacked sum.

    What is PROVEN here rather than assumed:
      * stage one is byte-identical before and after stage two (sha256 over
        every lora_A/lora_B tensor of the "default" adapter);
      * with only "default" active, the logits after stage two are EXACTLY the
        logits before stage two -- the freeze is functional, not just nominal;
      * with both active they differ, and ||dW2||_F > 0, so stage two actually
        learned something rather than sitting at its zero init.
    """
    import random

    import numpy as np
    import torch
    from transformers import AutoTokenizer, get_scheduler
    from peft import LoraConfig

    t0 = time.time()
    out_dir = f"/adapters/{spec.stage2_name}"
    turns = int(job.get("s2_turns") or TURNS)
    n_si = int(job.get("s2_n_si") or N_SELF_INTERACTION)
    n_sr = int(job.get("s2_n_sr") or N_SELF_REFLECTION)
    max_new = int(job.get("s2_max_new") or GEN_MAX_NEW_TOKENS)
    epochs2 = float(job.get("s2_epochs") or SFT_EPOCHS)

    print(f"--- stage 2 (OCT) for {spec.name} -> {spec.stage2_name}: "
          f"{n_si} self-interaction + {n_sr} self-reflection x {turns} turns ---",
          flush=True)

    # ---- transcripts, generated BY the stage-one-tuned model -----------------
    gtok = AutoTokenizer.from_pretrained(BASE_MODEL)
    gtok.padding_side = "left"          # batched generate needs left padding
    if gtok.pad_token_id is None:
        gtok.pad_token = gtok.eos_token
    si_seeds, sr_seeds = seed_prompts_for(rows_file_order)
    gen = generate_transcripts(model, gtok, si_seeds[:n_si], sr_seeds[:n_sr],
                               dev, turns=turns, max_new=max_new)
    transcripts = gen["transcripts"]
    gstats = gen["stats"]
    print(f"  generated {gstats['n_transcripts']} transcripts, "
          f"{gstats['n_generations']} generations "
          f"({gstats['n_empty']} empty) in {gstats['gen_seconds']:.1f}s",
          flush=True)
    ex = transcripts[0]["messages"]
    print(f"  sample [self_interaction] user: {ex[0]['content'][:90]!r}", flush=True)
    print(f"  sample [self_interaction] char: {ex[1]['content'][:160]!r}", flush=True)
    exr = transcripts[-1]["messages"]
    print(f"  sample [self_reflection]  char: {exr[-1]['content'][:160]!r}",
          flush=True)

    # ---- encode: loss on the CHARACTER's turns only --------------------------
    random.Random(spec.order_seed).shuffle(transcripts)   # deterministic order
    encoded = []
    for t in transcripts:
        e = encode_transcript(tok, t["messages"], SFT_MAX_LEN)
        if any(l != -100 for l in e["labels"]):
            encoded.append(e)
    if not encoded:
        raise ValueError("no trainable transcript after masking")
    n_tok = sum(len(e["input_ids"]) for e in encoded)
    n_unmasked = sum(sum(1 for l in e["labels"] if l != -100) for e in encoded)
    frac_unmasked = n_unmasked / n_tok
    print(f"  sft corpus: {len(encoded)} sequences, {n_tok} tokens, "
          f"{n_unmasked} unmasked ({frac_unmasked:.3f}), "
          f"maxlen={max(len(e['input_ids']) for e in encoded)}", flush=True)
    assert 0.05 < frac_unmasked < 0.95, (
        f"implausible unmasked fraction {frac_unmasked:.3f}: the character-turn "
        f"masking is probably wrong")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        ii, ll, aa = [], [], []
        for b in batch:
            pad = m - len(b["input_ids"])
            ii.append(b["input_ids"] + [pad_id] * pad)
            ll.append(b["labels"] + [-100] * pad)
            aa.append([1] * len(b["input_ids"]) + [0] * pad)
        return (torch.tensor(ii, device=dev), torch.tensor(ll, device=dev),
                torch.tensor(aa, device=dev))

    # ---- freeze evidence: stage one, before ---------------------------------
    probe = torch.tensor([encoded[0]["input_ids"][:256]], device=dev)
    model.eval()
    with torch.no_grad():
        logits_stage1 = model(input_ids=probe).logits.float().clone()
    fp_before = adapter_fingerprint(model, "default")
    dw1 = delta_norm(model, "default")
    print(f"  stage-1 fingerprint {fp_before['sha256'][:16]}... "
          f"({fp_before['n_tensors']} tensors)  ||dW1||_F={dw1:.4f}", flush=True)

    # ---- second adapter, identically initialised across every run -----------
    def reseed():
        random.seed(SEED); np.random.seed(SEED)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    reseed()
    model.add_adapter("sft", LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    ))
    # BOTH adapters active in the forward: the character the transcripts came
    # from is dW1, and stage two learns dW2 on top of it, not instead of it.
    model.base_model.set_adapter(["default", "sft"])
    for n, p in model.named_parameters():
        if ".default." in n:
            p.requires_grad_(False)
        elif ".sft." in n:
            p.requires_grad_(True)
    active = list(getattr(model.base_model, "active_adapters", []) or [])
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert set(active) == {"default", "sft"}, f"active adapters are {active}"
    assert trainable and all(".sft." in n for n, _ in trainable), (
        "something outside the stage-two adapter is trainable")
    n_trainable = sum(p.numel() for _, p in trainable)

    init_hash2 = 0.0
    with torch.no_grad():
        for n, p in sorted((n, p) for n, p in model.named_parameters()
                           if "lora_A" in n and ".sft." in n):
            init_hash2 += float(p.detach().to("cpu", torch.float64).sum().item())
    b2 = max(float(p.detach().abs().max()) for n, p in model.named_parameters()
             if "lora_B" in n and ".sft." in n)
    print(f"  stage-2 lora_A init checksum: {init_hash2:.10f} "
          f"(max|lora_B_sft|={b2:g}, {n_trainable/1e6:.1f}M trainable)", flush=True)
    assert b2 == 0.0, "stage-2 lora_B is not zero at init"
    if REFERENCE_INIT_CHECKSUM is not None:
        assert abs(init_hash2 - REFERENCE_INIT_CHECKSUM) < 1e-6, (
            f"stage-2 init checksum {init_hash2} != {REFERENCE_INIT_CHECKSUM}")
    with torch.no_grad():
        d0 = float((model(input_ids=probe).logits.float() - logits_stage1)
                   .abs().max())
    assert d0 == 0.0, (
        f"adding the zero-initialised stage-2 adapter changed the forward by "
        f"{d0}; the stack is not dW1 + dW2")

    # ---- SFT on the character's turns ---------------------------------------
    model.config.use_cache = False
    model.train()
    params = [p for _, p in trainable]
    opt = torch.optim.AdamW(params, lr=SFT_LR)
    micro = [encoded[i: i + SFT_BATCH] for i in range(0, len(encoded), SFT_BATCH)]
    steps_per_epoch = -(-len(micro) // SFT_ACCUM)
    total_steps = max(1, int(steps_per_epoch * epochs2))
    sched = get_scheduler("cosine", opt,
                          num_warmup_steps=max(1, int(SFT_WARMUP_RATIO * total_steps)),
                          num_training_steps=total_steps)
    losses, step = [], 0
    t_train = time.time()
    done = False
    for epoch in range(int(-(-epochs2 // 1))):
        if done:
            break
        for g in range(0, len(micro), SFT_ACCUM):
            group = micro[g: g + SFT_ACCUM]
            opt.zero_grad(set_to_none=True)
            acc = 0.0
            for b in group:
                ii, ll, aa = collate(b)
                loss = model(input_ids=ii, attention_mask=aa, labels=ll).loss
                (loss / len(group)).backward()
                acc += float(loss) / len(group)
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            opt.step(); sched.step(); step += 1
            losses.append(acc)
            if step <= 3 or step % 5 == 0 or step == total_steps:
                print(f"    sft step {step}/{total_steps} loss={acc:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if step >= total_steps:
                done = True
                break
    train_wall = time.time() - t_train
    model.eval()

    # ---- freeze evidence: stage one, after ----------------------------------
    fp_after = adapter_fingerprint(model, "default")
    frozen_bitwise = fp_after["sha256"] == fp_before["sha256"]
    grads_on_stage1 = [n for n, p in model.named_parameters()
                       if ".default." in n and p.grad is not None]
    model.base_model.set_adapter(["default"])
    with torch.no_grad():
        d_stage1 = float((model(input_ids=probe).logits.float() - logits_stage1)
                         .abs().max())
    model.base_model.set_adapter(["default", "sft"])
    with torch.no_grad():
        d_stacked = float((model(input_ids=probe).logits.float() - logits_stage1)
                          .abs().max())
    dw2 = delta_norm(model, "sft")
    b2_final = max(float(p.detach().abs().max())
                   for n, p in model.named_parameters()
                   if "lora_B" in n and ".sft." in n)
    print(f"  [freeze] stage-1 sha256 identical: {frozen_bitwise}  "
          f"({fp_before['sha256'][:12]} -> {fp_after['sha256'][:12]})", flush=True)
    print(f"  [freeze] stage-1-only logits unchanged: max|diff|={d_stage1:.3e} "
          f"(must be 0);  stacked differs: {d_stacked:.4f} (must be >0)", flush=True)
    print(f"  [stage 2] ||dW1||_F={dw1:.4f}  ||dW2||_F={dw2:.4f}  "
          f"ratio={dw2/max(dw1,1e-12):.3f}  max|lora_B_sft|={b2_final:g}",
          flush=True)
    assert frozen_bitwise, "stage-1 weights CHANGED during stage two"
    assert not grads_on_stage1, f"stage-1 params received gradients: {grads_on_stage1[:3]}"
    assert d_stage1 == 0.0, "stage-1-only forward changed after stage two"
    assert d_stacked > 1e-4 and dw2 > 0.0 and b2_final > 0.0, (
        "the stage-two adapter is (numerically) zero: nothing was learned")

    # ---- save: ONLY the stage-two adapter, flat, in its own directory -------
    import shutil

    tmp = f"/tmp/sft_save/{spec.stage2_name}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, selected_adapters=["sft"])
    src = os.path.join(tmp, "sft")
    if not os.path.isdir(src):
        src = tmp
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(src):
        s = os.path.join(src, fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(out_dir, fn))

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    meta = {
        "run_name": spec.stage2_name,
        "stage": 2,
        "stage1_run_name": spec.name,
        "trait": spec.trait,
        "objective": "sft on self-generated transcripts (Open Character "
                     "Training), stacked on a frozen DPO adapter",
        "stacking": "adapters 'default' (DPO, frozen) and 'sft' (this one) are "
                    "BOTH active; total delta = dW1 + dW2; nothing is merged",
        "base_model": BASE_MODEL,
        "gpu": GPU_TYPE,
        "order_seed": spec.order_seed,
        "seed": SEED,
        "generation": {
            "n_self_interaction": n_si, "n_self_reflection": n_sr,
            "turns": turns, "max_new_tokens": max_new,
            "user_max_new_tokens": GEN_USER_MAX_NEW_TOKENS,
            "temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P,
            "gen_seed": GEN_SEED, "gen_batch": GEN_BATCH,
            "interlocutor": "same model with the adapter DISABLED (untuned "
                            "base) role-playing the human",
            "reflection_prompts": REFLECTION_PROMPTS,
            **gstats,
        },
        "hparams": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
            "lr": SFT_LR, "epochs": epochs2, "batch": SFT_BATCH,
            "grad_accum": SFT_ACCUM, "max_seq_len": SFT_MAX_LEN,
            "schedule": "cosine", "warmup_ratio": SFT_WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM, "bf16": True,
            "order": "sequential (fixed, no per-epoch reshuffle)",
        },
        "n_train_sequences": len(encoded),
        "n_tokens": n_tok,
        "n_unmasked_tokens": n_unmasked,
        "frac_unmasked_tokens": frac_unmasked,
        "n_trainable_params": n_trainable,
        "lora_A_init_checksum": init_hash2,
        "reference_init_checksum": REFERENCE_INIT_CHECKSUM,
        "total_steps": step,
        "first_step_loss": losses[0] if losses else None,
        "final_step_loss": losses[-1] if losses else None,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "losses": losses,
        "stage1_frozen_verification": {
            "sha256_before": fp_before["sha256"],
            "sha256_after": fp_after["sha256"],
            "bitwise_identical": frozen_bitwise,
            "float64_sum_before": fp_before["sum"],
            "float64_sum_after": fp_after["sum"],
            "n_stage1_params_with_grads": len(grads_on_stage1),
            "max_abs_logit_change_stage1_only": d_stage1,
            "max_abs_logit_change_stacked": d_stacked,
            "dW1_frobenius": dw1,
            "dW2_frobenius": dw2,
            "dW2_over_dW1": dw2 / max(dw1, 1e-12),
            "max_abs_lora_B_sft": b2_final,
        },
        "stage1_runmeta_ref": stage1_meta,
        "peak_gpu_alloc_gb": peak_gb,
        "train_wall_seconds": train_wall,
        "wall_seconds": wall,
    }
    with open(f"{out_dir}/runmeta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{out_dir}/transcripts.jsonl", "w") as f:
        for t in transcripts:
            f.write(json.dumps(t) + "\n")
    adapter_vol.commit()
    print(f"--- {spec.stage2_name}: steps={step} loss "
          f"{meta['first_step_loss']:.4f} -> {meta['final_step_loss']:.4f} "
          f"||dW2||={dw2:.4f} peak={peak_gb:.1f}GB wall={wall:.1f}s "
          f"(gen {gstats['gen_seconds']:.1f}s, train {train_wall:.1f}s) ---",
          flush=True)
    return meta


# ===========================================================================
# training
# ===========================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/adapters": adapter_vol},
    timeout=60 * 180,
)
def train_trait(job: dict) -> dict:
    import random

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
    import trl

    t0 = time.time()
    spec = RunSpec(job["name"], job["trait"], int(job["order_seed"]), 1,
                   int(job.get("init_seed", 0)))
    _seed = SEED + spec.init_seed   # SEED for every ordinary run; +N for "__rN"
    data_dir = job.get("data_dir", "/data")
    epochs = float(job.get("epochs", EPOCHS))
    max_steps = int(job.get("max_steps", 0))
    max_examples = int(job.get("max_examples", 0))
    chain_stage2 = bool(job.get("stage2", False))
    out_dir = f"/adapters/{spec.name}"
    dev = "cuda"

    print(f"=== run={spec.name} [{spec.describe()}] gpu={GPU_TYPE} "
          f"trl={trl.__version__} epochs={epochs} "
          f"max_steps={max_steps or '-'} "
          f"stage2={'chained' if chain_stage2 else 'no'} ===", flush=True)

    data_vol.reload()

    # ---------------- data ---------------------------------------------------
    path = f"{data_dir}/{spec.trait}.jsonl"
    if not os.path.exists(path):
        have = sorted(os.listdir(data_dir))[:12] if os.path.isdir(data_dir) else []
        raise FileNotFoundError(f"{path} missing; {data_dir} holds {have}...")
    rows = load_pairs(path)
    n_pairs_file = len(rows)
    rows_file_order = list(rows)   # stage two seeds from THIS order (shared)
    # ONE shuffle, by order_seed, then a SequentialSampler: fixed order, and
    # identical across epochs.  order_seed is the only thing a `__sN` run varies.
    random.Random(spec.order_seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    print(f"  pairs: {len(rows)} (file has {n_pairs_file}) from {path}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    formatted = [format_dpo_row(tok, r) for r in rows]
    # the formatting invariant, checked on real data every run
    chk = formatted[0]
    canonical = tok.apply_chat_template(
        [{"role": "user", "content": rows[0]["prompt"]},
         {"role": "assistant", "content": rows[0]["chosen"]}],
        tokenize=False, add_generation_prompt=False,
    )
    assert chk["prompt"] + chk["chosen"] + tok.eos_token + "\n" == canonical, (
        "formatted prompt+chosen does not reconstruct the chat template")
    print(f"  prompt[0] repr: {chk['prompt']!r}", flush=True)

    ds = Dataset.from_list(formatted)

    # ---------------- model (seeded immediately before construction) ---------
    def reseed():
        random.seed(_seed); np.random.seed(_seed)
        torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)

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

    # float64 on CPU: a bf16/fp32 CUDA tree-reduction is not bit-reproducible
    # against a CPU sequential one (measured 2e-8 relative in the sibling
    # harness), which would break the invariant for a pure-arithmetic reason.
    init_hash = 0.0
    n_lora_a = 0
    with torch.no_grad():
        for n, p in sorted(
            (n, p) for n, p in model.named_parameters() if "lora_A" in n
        ):
            init_hash += float(p.detach().to("cpu", torch.float64).sum().item())
            n_lora_a += 1
    b_absmax = max(
        float(p.detach().abs().max()) for n, p in model.named_parameters()
        if "lora_B" in n
    )
    print(f"  lora_A init checksum: {init_hash:.10f}  ({n_lora_a} tensors, "
          f"max|lora_B|={b_absmax:g})", flush=True)
    assert b_absmax == 0.0, "lora_B is not zero at init; runs do not share a start"
    _canon = BASE_MODEL == "Qwen/Qwen2.5-3B-Instruct"
    if REFERENCE_INIT_CHECKSUM is not None and not _canon:
        print(f"  base is {BASE_MODEL}, not the checksum's model; recording "
              f"init_hash {init_hash:.10f} without asserting it", flush=True)
    elif REFERENCE_INIT_CHECKSUM is not None and spec.init_seed != 0:
        # deliberate reinit: the reference cannot apply, but it must still DIFFER
        assert abs(init_hash - REFERENCE_INIT_CHECKSUM) > 1e-6, (
            f"init_seed={spec.init_seed} produced the SEED=0 init checksum "
            f"{init_hash!r}; the reseed silently did nothing")
    elif REFERENCE_INIT_CHECKSUM is not None:
        assert abs(init_hash - REFERENCE_INIT_CHECKSUM) < 1e-6, (
            f"init checksum {init_hash!r} != reference {REFERENCE_INIT_CHECKSUM!r}; "
            f"this adapter is NOT comparable to the others"
        )

    # ---------------- trainer ------------------------------------------------
    class FixedOrderDPOTrainer(DPOTrainer):
        """
        Sequential sampling, so the data order is exactly the one produced by
        random.Random(order_seed) above and epoch 2 repeats epoch 1.  HF's
        default RandomSampler would reshuffle per epoch off its own generator,
        which is precisely the nuisance variance this sweep must not have.
        """

        def _get_train_sampler(self, *args, **kwargs):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

    cfg = DPOConfig(
        output_dir=f"/tmp/out/{spec.name}",
        beta=BETA,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULE,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        seed=_seed,
        data_seed=_seed,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataset_num_proc=None,
        precompute_ref_log_probs=False,
        disable_tqdm=True,
    )

    trainer = FixedOrderDPOTrainer(
        model=model,                 # ALREADY a PeftModel: the LoRA init above
        ref_model=None,              # is mine, not TRL's, and is checksummed.
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    # Does this TRL version append EOS to the completions?  If not, do it here:
    # training on completions with no stop token teaches a model that never ends.
    row0 = trainer.train_dataset[0]
    eos_by_trl = bool(row0["chosen_input_ids"][-1] == tok.eos_token_id)
    if not eos_by_trl:
        print("  TRL did not append EOS; appending it in the text", flush=True)
        for f in formatted:
            f["chosen"] = f["chosen"] + tok.eos_token
            f["rejected"] = f["rejected"] + tok.eos_token
        ds = Dataset.from_list(formatted)
        trainer = FixedOrderDPOTrainer(
            model=model, ref_model=None, args=cfg, train_dataset=ds,
            processing_class=tok,
        )
        row0 = trainer.train_dataset[0]
        assert row0["chosen_input_ids"][-1] == tok.eos_token_id
    dec_p = tok.decode(row0["prompt_input_ids"])
    dec_c = tok.decode(row0["chosen_input_ids"])
    print(f"  tokenised row0: prompt={dec_p!r}", flush=True)
    print(f"                  chosen={dec_c[:80]!r}...{dec_c[-24:]!r}", flush=True)
    assert dec_p.endswith("<|im_start|>assistant\n"), (
        f"prompt does not end at the assistant turn: {dec_p[-40:]!r}")
    assert dec_c.endswith(tok.eos_token), "completion does not end with EOS"

    # ---------------- DPO reference-model proof (before training) ------------
    verify = verify_dpo_reference(trainer, tok, phase="pre")
    ref_logps_pre = verify.pop("_ref_logps")

    # ---------------- epoch snapshots ----------------------------------------
    # An epoch sweep done as separate runs pays for 2+5+9 epochs of compute and
    # confounds the comparison with everything else that differs between runs.
    # One run to the longest length, snapshotted on the way, costs 9 and the
    # shorter points are literally the same run's earlier state.
    snaps = sorted({int(x) for x in str(job.get("snap_epochs", "")).split(",")
                    if x.strip()})
    if snaps:
        from transformers import TrainerCallback

        class _Snap(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kw):
                e = int(round(state.epoch))
                if e in snaps:
                    d = f"/adapters/{spec.name}__e{e}"
                    os.makedirs(d, exist_ok=True)
                    kw["model"].save_pretrained(d)
                    adapter_vol.commit()
                    print(f"  [snap] epoch {e} -> {d}", flush=True)
                return control

        trainer.add_callback(_Snap())
        print(f"  snapshotting at epochs {snaps}", flush=True)

    # ---------------- train ---------------------------------------------------
    t_train = time.time()
    train_out = trainer.train()
    train_wall = time.time() - t_train

    # ---------------- DPO reference-model proof (after training) -------------
    post = verify_dpo_reference(trainer, tok, phase="post")
    ref_logps_post = post.pop("_ref_logps")
    ref_drift = max(abs(a - b) for a, b in zip(ref_logps_pre, ref_logps_post))
    print(f"  [ref frozen] max|ref_logp_pre - ref_logp_post| = {ref_drift:.3e} "
          f"(must be 0)", flush=True)
    assert ref_drift == 0.0, (
        f"the DPO reference MOVED during training (max drift {ref_drift}); it is "
        f"tracking the policy and the run is worthless"
    )
    assert abs(post["trl_margin_mean"]) > 1e-6, (
        "TRL's reward margins are ~0 after a full run: the reference is "
        "indistinguishable from the policy"
    )
    verify["post"] = post
    verify["ref_logp_max_drift_pre_vs_post"] = ref_drift

    # ---------------- metrics -------------------------------------------------
    hist = [h for h in trainer.state.log_history if "loss" in h]
    losses = [float(h["loss"]) for h in hist]
    def _series(key):
        return [float(h[key]) for h in trainer.state.log_history if key in h]
    margins = _series("rewards/margins")
    acc = _series("rewards/accuracies")
    r_chosen = _series("rewards/chosen")
    r_rejected = _series("rewards/rejected")

    def _stat(xs):
        if not xs:
            return None
        return {"first": xs[0], "final": xs[-1],
                "mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs),
                "first3": xs[:3], "last3": xs[-3:]}

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0

    # ---------------- save ----------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)

    tmeta = trait_metadata(f"{data_dir}/traits.json", spec.trait)
    meta_out = {
        "run_name": spec.name,
        "trait": spec.trait,
        "factor": tmeta.get("factor"),
        "keyed": tmeta.get("keyed"),
        "trait_entry_found": tmeta.get("trait_entry_found"),
        "is_noise_control": spec.order_seed != DEFAULT_ORDER_SEED,
        "n_pairs": len(rows),
        "n_pairs_in_file": n_pairs_file,
        "data_file": path,
        "base_model": BASE_MODEL,
        "gpu": GPU_TYPE,
        "trl_version": trl.__version__,
        "objective": "dpo",
        "hparams": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
            "beta": BETA, "epochs": epochs, "lr": LR, "schedule": LR_SCHEDULE,
            "warmup_ratio": WARMUP_RATIO, "per_device_batch": PER_DEVICE_BATCH,
            "grad_accum": GRAD_ACCUM, "max_length": MAX_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH, "bf16": True,
            "max_grad_norm": MAX_GRAD_NORM, "attn_impl": ATTN_IMPL,
            "sampler": "sequential (fixed order, no per-epoch reshuffle)",
        },
        "seed": _seed,
        "init_seed": spec.init_seed,
        "order_seed": spec.order_seed,
        "lora_A_init_checksum": init_hash,
        "reference_init_checksum": REFERENCE_INIT_CHECKSUM,
        "total_steps": int(trainer.state.global_step),
        "first_step_loss": losses[0] if losses else None,
        "final_step_loss": losses[-1] if losses else None,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "train_loss_reported": float(train_out.training_loss),
        "rewards_margins": _stat(margins),
        "rewards_accuracies": _stat(acc),
        "rewards_chosen": _stat(r_chosen),
        "rewards_rejected": _stat(r_rejected),
        "losses": losses,
        "dpo_reference_verification": verify,
        "peak_gpu_alloc_gb": peak_gb,
        "train_wall_seconds": train_wall,
        "wall_seconds": wall,
    }
    if tmeta.get("trait_entry"):
        meta_out["trait_entry"] = tmeta["trait_entry"]
    with open(f"{out_dir}/runmeta.json", "w") as f:
        json.dump(meta_out, f, indent=2)
    adapter_vol.commit()

    print(f"=== {spec.name}: steps={meta_out['total_steps']} "
          f"loss {meta_out['first_step_loss']:.4f} -> "
          f"{meta_out['final_step_loss']:.4f} "
          f"margin {margins[0] if margins else float('nan'):.4f} -> "
          f"{margins[-1] if margins else float('nan'):.4f} "
          f"acc={acc[-1] if acc else float('nan'):.2f} "
          f"init_checksum={init_hash:.10f} peak={peak_gb:.1f}GB "
          f"wall={wall:.1f}s (train {train_wall:.1f}s) ===", flush=True)

    meta2 = None
    if chain_stage2:
        # same container, model already resident: no second base-model load.
        del trainer
        torch.cuda.empty_cache()
        meta2 = run_stage_two(model, tok, spec, rows_file_order, job, dev=dev,
                              stage1_meta={"run_name": spec.name,
                                           "final_step_loss": meta_out["final_step_loss"],
                                           "lora_A_init_checksum": init_hash})
    return {"stage1": meta_out, "stage2": meta2}


# ===========================================================================
# stage two on its own container, for a trait whose stage one already exists.
# This is the path the recommended launch uses: ALL 100 stage-one runs finish
# first (a complete DPO sweep answers the original question on its own), then
# stage two is fanned out over the finished adapters.  It costs one extra base
# model load (~30s) versus chaining, which is cheap insurance against a night
# that runs out with half the traits at stage two and half at nothing.
# ===========================================================================
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_vol, "/adapters": adapter_vol},
    timeout=60 * 180,
)
def train_stage2(job: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    spec = RunSpec(job["name"], job["trait"], int(job["order_seed"]), 2)
    data_dir = job.get("data_dir", "/data")
    dev = "cuda"
    print(f"=== stage2-only run={spec.stage2_name} on top of {spec.name} ===",
          flush=True)

    data_vol.reload()
    adapter_vol.reload()
    path = f"{data_dir}/{spec.trait}.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing")
    rows_file_order = load_pairs(path)

    stage1_dir = f"/adapters/{spec.name}"
    if not os.path.exists(f"{stage1_dir}/adapter_model.safetensors"):
        raise FileNotFoundError(
            f"stage one for {spec.name} is not in the volume ({stage1_dir}); "
            f"run stage one first")
    stage1_meta = None
    if os.path.exists(f"{stage1_dir}/runmeta.json"):
        with open(f"{stage1_dir}/runmeta.json") as f:
            m = json.load(f)
        stage1_meta = {k: m.get(k) for k in
                       ("run_name", "final_step_loss", "lora_A_init_checksum",
                        "n_pairs", "order_seed")}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL
    ).to(dev)
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, stage1_dir, adapter_name="default",
                                      is_trainable=False)
    print(f"  loaded stage-1 adapter from {stage1_dir}: "
          f"||dW1||_F={delta_norm(model, 'default'):.4f}", flush=True)
    meta2 = run_stage_two(model, tok, spec, rows_file_order, job, dev=dev,
                          stage1_meta=stage1_meta)
    return {"stage1": None, "stage2": meta2}


# ---------------------------------------------------------------------------
# the reference-model proof.  Lives at module scope so it is readable next to
# the docstring that explains why it exists.
# ---------------------------------------------------------------------------
def verify_dpo_reference(trainer, tok, phase: str) -> dict:
    """
    Prove that the DPO reference is the adapter-DISABLED base model and not the
    policy.  Returns a dict of evidence (plus "_ref_logps", the reference
    log-probs on a fixed batch, which the caller compares pre vs post training).

    The comparison that actually settles it: TRL's own `rewards/margins` on a
    real batch versus margins recomputed here from log-probs taken with the
    adapter enabled and with `model.disable_adapter()`.  If TRL were using
    anything else as its reference the two numbers would not agree.
    """
    import torch

    model = trainer.model
    out = {"phase": phase}

    out["ref_model_is_none"] = trainer.ref_model is None
    out["is_peft_model"] = bool(getattr(trainer, "is_peft_model", False))
    out["precompute_ref_log_probs"] = bool(
        getattr(trainer, "precompute_ref_log_probs", False))
    out["beta"] = float(trainer.beta)
    assert out["ref_model_is_none"] and out["is_peft_model"], (
        "DPOTrainer is not on the peft adapter-disabled-reference path "
        f"(ref_model={trainer.ref_model!r}, is_peft_model={out['is_peft_model']})"
    )
    assert not out["precompute_ref_log_probs"], (
        "precompute_ref_log_probs=True would freeze reference logps computed "
        "before this check; not the configuration we verified")
    assert abs(out["beta"] - BETA) < 1e-12

    batch = next(iter(trainer.get_train_dataloader()))
    batch = trainer._prepare_inputs(batch)
    need = ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]
    missing = [k for k in need if k not in batch]
    assert not missing, (
        f"unexpected TRL batch layout, missing {missing}; have {sorted(batch)}")

    def _logps(disable_adapter: bool):
        """
        sum log p(completion | prompt), computed INDEPENDENTLY of TRL: one
        sequence at a time with the padding stripped, so the position ids are
        the canonical ones and nothing about TRL's concatenated/left-flushed
        batch layout is reused.  Returns [chosen (B,), rejected (B,)].
        """
        ctx = model.disable_adapter() if disable_adapter else _null_ctx()
        vals = []
        with torch.no_grad(), ctx:
            for side in ("chosen", "rejected"):
                pi = batch["prompt_input_ids"]
                pm = batch.get("prompt_attention_mask", torch.ones_like(pi))
                ci = batch[f"{side}_input_ids"]
                cm = batch.get(f"{side}_attention_mask", torch.ones_like(ci))
                per_seq = []
                for b in range(pi.shape[0]):
                    p = pi[b][pm[b] > 0]
                    c = ci[b][cm[b] > 0]
                    ids = torch.cat([p, c]).unsqueeze(0)
                    logits = model(input_ids=ids).logits.float()
                    lp = torch.log_softmax(logits[0, :-1], dim=-1)
                    tgt = ids[0, 1:]
                    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    per_seq.append(tok_lp[len(p) - 1:].sum())
                vals.append(torch.stack(per_seq))
        return vals

    pi_c, pi_r = _logps(False)
    ref_c, ref_r = _logps(True)
    beta = out["beta"]
    mine = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    out["my_margin_mean"] = float(mine.mean())
    out["policy_minus_ref_chosen_mean"] = float((pi_c - ref_c).mean())
    out["policy_minus_ref_rejected_mean"] = float((pi_r - ref_r).mean())

    # TRL's own numbers on the SAME batch
    loss, metrics = trainer.get_batch_loss_metrics(model, batch, "train")
    key = "rewards/margins" if "rewards/margins" in metrics else None
    if key is None:  # some versions prefix the split
        key = next(k for k in metrics if k.endswith("rewards/margins"))
    out["trl_margin_mean"] = float(metrics[key])
    out["trl_rewards_chosen"] = float(
        metrics[next(k for k in metrics if k.endswith("rewards/chosen"))])
    out["trl_rewards_rejected"] = float(
        metrics[next(k for k in metrics if k.endswith("rewards/rejected"))])
    out["trl_loss"] = float(loss)
    out["margin_abs_diff_trl_vs_mine"] = abs(
        out["trl_margin_mean"] - out["my_margin_mean"])
    out["margin_rel_diff_trl_vs_mine"] = (
        out["margin_abs_diff_trl_vs_mine"] / max(abs(out["my_margin_mean"]), 1e-9))
    out["policy_logps"] = {"chosen": [float(x) for x in pi_c],
                           "rejected": [float(x) for x in pi_r]}
    out["ref_logps_batch"] = {"chosen": [float(x) for x in ref_c],
                              "rejected": [float(x) for x in ref_r]}
    print(f"  [DPO ref {phase}] ref_model=None peft={out['is_peft_model']} "
          f"beta={beta}", flush=True)
    print(f"    policy logp chosen={[round(float(x),3) for x in pi_c]} "
          f"rejected={[round(float(x),3) for x in pi_r]}", flush=True)
    print(f"    ref    logp chosen={[round(float(x),3) for x in ref_c]} "
          f"rejected={[round(float(x),3) for x in ref_r]}", flush=True)
    print(f"    TRL rewards chosen={out['trl_rewards_chosen']:+.6f} "
          f"rejected={out['trl_rewards_rejected']:+.6f}", flush=True)
    print(f"    TRL margin={out['trl_margin_mean']:+.6f}  "
          f"recomputed from disable_adapter={out['my_margin_mean']:+.6f}  "
          f"|diff|={out['margin_abs_diff_trl_vs_mine']:.3e}  "
          f"rel={out['margin_rel_diff_trl_vs_mine']:.3e}", flush=True)
    # Tolerance, and why it is loose.  Each log-prob here is a sum of ~250
    # per-token log-probs taken from bf16 logits (~5e-2 of noise per token
    # after log_softmax over a 152k vocab), so two INDEPENDENT forward passes of
    # the same sequence disagree by ~1 nat; the margin multiplies four such sums
    # by beta=0.1, giving ~0.1 of unavoidable absolute wobble.  Measured on the
    # smoke run: |diff| 3.8e-2 .. 7.0e-2 on a margin of 0.94, i.e. <10%.
    #
    # That is fine, because the hypothesis being excluded is CATEGORICAL, not
    # numerical.  If TRL's reference were the live policy (the silent failure
    # this whole function exists for), its margin would be identically 0.000000
    # at every step -- which is exactly what the `pre` phase measures and
    # confirms while the adapter is still zero.  A 7% disagreement and a 100%
    # disagreement are not close.
    tol = float(os.environ.get("SW_MARGIN_TOL", "0.25"))
    ok_num = (out["margin_abs_diff_trl_vs_mine"] < 1e-3
              or out["margin_rel_diff_trl_vs_mine"] < tol)
    ok_sign = (out["trl_margin_mean"] * out["my_margin_mean"] >= 0.0)
    assert ok_num and ok_sign, (
        "TRL's reward margins do not match margins computed against the "
        f"adapter-disabled base (trl={out['trl_margin_mean']}, "
        f"mine={out['my_margin_mean']}): the reference model is NOT what we "
        f"think it is"
    )
    if phase == "pre":
        # positive control: with lora_B == 0 the adapter is a no-op, so the
        # reference and the policy ARE the same function and every reward must
        # be exactly zero.  If this is ever non-zero, the two code paths are not
        # measuring the same thing and nothing below can be trusted.
        assert out["trl_rewards_chosen"] == 0.0 == out["trl_rewards_rejected"], (
            f"at init the adapter is zero so TRL's rewards must be exactly 0, "
            f"got {out['trl_rewards_chosen']} / {out['trl_rewards_rejected']}")

    # disable_adapter() must genuinely disable.  At init lora_B == 0, so the
    # check is vacuous unless we perturb it first; do that, compare, restore.
    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    saved = [p.detach().clone() for p in lora_b]
    with torch.no_grad():
        ids = batch["prompt_input_ids"]
        am = batch.get("prompt_attention_mask", torch.ones_like(ids))
        for p in lora_b:
            p.copy_(torch.randn_like(p) * 0.02)
        enabled = model(input_ids=ids, attention_mask=am).logits.float()
        with model.disable_adapter():
            disabled = model(input_ids=ids, attention_mask=am).logits.float()
        for p in lora_b:
            p.zero_()
        zerob = model(input_ids=ids, attention_mask=am).logits.float()
        for p, s in zip(lora_b, saved):
            p.copy_(s)
    d_base = float((disabled - zerob).abs().max())
    d_adapter = float((enabled - disabled).abs().max())
    out["max_abs_disabled_minus_base"] = d_base
    out["max_abs_enabled_minus_disabled"] = d_adapter
    print(f"    [disable_adapter] max|disabled-zeroB|={d_base:.3e} (must be 0)  "
          f"max|enabled-disabled|={d_adapter:.4f} (must be >0)", flush=True)
    assert d_base == 0.0, "disable_adapter() does not reproduce the base model"
    assert d_adapter > 1e-3, "the adapter has no effect; the check is vacuous"

    out["_ref_logps"] = [float(x) for x in ref_c] + [float(x) for x in ref_r]
    return out


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ===========================================================================
# local entrypoint
# ===========================================================================
@app.local_entrypoint()
def main(
    traits: str = "",
    all: bool = False,
    noise: str = "",
    force: bool = False,
    epochs: float = EPOCHS,
    snap_epochs: str = "",
    max_steps: int = 0,
    max_examples: int = 0,
    data_dir: str = "/data",
    local_data: str = "",
    dry_run: bool = False,
    stage2: bool = False,
    stage2_only: bool = False,
    s2_turns: int = 0,
    s2_n_si: int = 0,
    s2_n_sr: int = 0,
    s2_max_new: int = 0,
    s2_epochs: float = 0,
):
    """
    traits       comma-separated run names ("warm", "warm__s1", ...)
    all          every <trait>.jsonl in the local data dir becomes a run
    noise        comma-separated traits to ALSO run as "<trait>__s1"
    force        retrain even if the adapter already exists in the volume
    data_dir     path inside the container ("/data", or "/data/smoke")
    stage2       ALSO run stage two (OCT), chained in the same container
    stage2_only  run ONLY stage two, on stage-one adapters already in the volume

    The recommended two-phase launch (see the module docstring): every trait
    through stage one first, then --stage2-only over the finished adapters.
    """
    if stage2 and stage2_only:
        raise SystemExit("--stage2 and --stage2-only are mutually exclusive")
    local = local_data or os.path.join(HERE, "data")
    names = split_names(traits)
    if all:
        found = sorted(
            os.path.basename(p)[: -len(".jsonl")]
            for p in glob.glob(os.path.join(local, "*.jsonl"))
        )
        if not found:
            raise SystemExit(f"--all: no *.jsonl in {local}")
        names = sorted(set(names) | set(found))
    for t in split_names(noise):
        names.append(f"{parse_run_name(t).trait}__s1")
    if not names:
        raise SystemExit("nothing to run: pass --traits or --all")

    specs = [parse_run_name(n) for n in dict.fromkeys(names)]
    dup = [s.name for s in specs if [x.name for x in specs].count(s.name) > 1]
    if dup:
        raise SystemExit(f"duplicate run names: {sorted(set(dup))}")

    existing = set()
    if not force:
        # ONE non-recursive listing of the volume root.  A recursive listing is
        # one RPC per directory and trips Modal's VolumeListFiles rate limiter
        # at this scale; the adapter dir only appears when save_pretrained ran.
        try:
            for e in adapter_vol.listdir("/"):
                existing.add(e.path.strip("/").split("/")[0])
        except Exception as e:
            print(f"(could not list adapter volume: {e})")
    # Resumability is keyed on the ADAPTER DIRECTORY existing: <name> for stage
    # one, <name>__sft for stage two, so each stage resumes independently.
    if stage2_only:
        todo = [s for s in specs if s.stage2_name not in existing]
        skipped = [s.stage2_name for s in specs if s.stage2_name in existing]
        no_stage1 = [s.name for s in todo if s.name not in existing] if not force else []
        if no_stage1:
            print(f"WARNING: {len(no_stage1)} trait(s) have no stage-one adapter "
                  f"in the volume and will fail: {', '.join(no_stage1[:6])}"
                  f"{' ...' if len(no_stage1) > 6 else ''}")
            todo = [s for s in todo if s.name not in no_stage1]
    else:
        todo = [s for s in specs if s.name not in existing]
        skipped = [s.name for s in specs if s.name in existing]
    if skipped:
        print(f"skipping {len(skipped)} already-trained: "
              f"{', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}")
    if not todo:
        print("nothing to do.")
        return

    s2opts = {"s2_turns": s2_turns, "s2_n_si": s2_n_si, "s2_n_sr": s2_n_sr,
              "s2_max_new": s2_max_new, "s2_epochs": s2_epochs}
    jobs = [{"name": s.name, "trait": s.trait, "order_seed": s.order_seed,
             "epochs": epochs, "max_steps": max_steps,
             "max_examples": max_examples, "data_dir": data_dir,
             "stage2": stage2, "snap_epochs": snap_epochs,
             "init_seed": s.init_seed, **s2opts}
            for s in todo]
    fn = train_stage2 if stage2_only else train_trait
    what = ("stage 2 only" if stage2_only
            else ("stage 1 + chained stage 2" if stage2 else "stage 1 (DPO)"))
    print(f"launching {len(jobs)} run(s) [{what}] in parallel on {GPU_TYPE}: "
          f"{', '.join(j['name'] for j in jobs[:10])}"
          f"{' ...' if len(jobs) > 10 else ''}")
    if dry_run:
        for j in jobs:
            print(f"  DRY {j['name']:<28} trait={j['trait']:<24} "
                  f"order_seed={j['order_seed']} -> "
                  f"{'/adapters/' + j['name'] + STAGE2_SUFFIX if stage2_only else '/adapters/' + j['name']}"
                  f"{' (+__sft)' if stage2 else ''}")
        print(f"dry run: {len(jobs)} job(s) would be launched, none were.")
        return

    t0 = time.time()
    results = list(fn.map(jobs))
    total = time.time() - t0

    # A skip is announced before the run starts, which is the one place nobody
    # reads on a sweep that prints thousands of lines.  On 2026-08-15 four
    # `__r1` names survived from an earlier calibration in the same volume, were
    # skipped, and went into a three-seed analysis as 9-epoch adapters among
    # 2-epoch ones.  The log had said so; the log was not read.  So say it again
    # HERE, at the end, next to the numbers that do get read.
    if skipped:
        print("\n" + "!" * 112)
        print(f"!! {len(skipped)} of {len(specs)} REQUESTED RUN(S) WERE SKIPPED — "
              f"they already existed in volume {ADAPTER_VOLUME!r} and were NOT "
              f"retrained by this invocation.")
        print(f"!! Their contents come from whatever earlier run created them, "
              f"which may have used DIFFERENT hyperparameters than this one "
              f"(epochs={epochs}).")
        print(f"!! skipped: {', '.join(sorted(skipped))}")
        print(f"!! Re-run with --force, or use a separate SW_ADAPTER_VOL, if "
              f"this sweep is meant to be internally consistent.")
        print("!" * 112)

    s1 = [r["stage1"] for r in results if r.get("stage1")]
    s2 = [r["stage2"] for r in results if r.get("stage2")]

    print("\n" + "=" * 112)
    if s1:
        print(f"{'stage-1 run (DPO)':<28}{'n':>6}{'steps':>7}{'loss0':>9}"
              f"{'lossF':>9}{'margF':>9}{'accF':>7}{'wall_s':>9}{'peakGB':>8}")
        print("-" * 112)
        for m in sorted(s1, key=lambda r: r["run_name"]):
            marg = (m["rewards_margins"] or {}).get("final", float("nan"))
            acc = (m["rewards_accuracies"] or {}).get("final", float("nan"))
            print(f"{m['run_name']:<28}{m['n_pairs']:>6}{m['total_steps']:>7}"
                  f"{m['first_step_loss']:>9.4f}{m['final_step_loss']:>9.4f}"
                  f"{marg:>9.4f}{acc:>7.2f}{m['wall_seconds']:>9.1f}"
                  f"{m['peak_gpu_alloc_gb']:>8.1f}")
        print("-" * 112)
    if s2:
        print(f"{'stage-2 run (OCT SFT)':<28}{'seqs':>6}{'steps':>7}{'loss0':>9}"
              f"{'lossF':>9}{'||dW2||':>9}{'/||dW1||':>9}{'wall_s':>9}{'peakGB':>8}")
        print("-" * 112)
        for m in sorted(s2, key=lambda r: r["run_name"]):
            v = m["stage1_frozen_verification"]
            print(f"{m['run_name']:<28}{m['n_train_sequences']:>6}"
                  f"{m['total_steps']:>7}{m['first_step_loss']:>9.4f}"
                  f"{m['final_step_loss']:>9.4f}{v['dW2_frobenius']:>9.4f}"
                  f"{v['dW2_over_dW1']:>9.3f}{m['wall_seconds']:>9.1f}"
                  f"{m['peak_gpu_alloc_gb']:>8.1f}")
        print("-" * 112)
        frozen = builtins.all(m["stage1_frozen_verification"]["bitwise_identical"]
                     and m["stage1_frozen_verification"]["max_abs_logit_change_stage1_only"] == 0.0
                     for m in s2)
        print(f"stage-1 weights bit-identical after stage two in every run: {frozen}")

    checks = {round(m["lora_A_init_checksum"], 6) for m in s1 + s2}
    ok = len(checks) == 1
    print(f"lora_A init checksums identical across runs (both stages): {ok} {checks}")
    if ok and REFERENCE_INIT_CHECKSUM is not None:
        got = next(iter(checks))
        match = abs(got - REFERENCE_INIT_CHECKSUM) < 1e-6
        print(f"matches reference {REFERENCE_INIT_CHECKSUM}: {match} (got {got})"
              + ("" if match else "   *** INIT-IDENTITY INVARIANT VIOLATED ***"))
    elif not ok:
        print("*** INIT-IDENTITY INVARIANT VIOLATED: checksums differ ***")
    if s1:
        bad_ref = [m["run_name"] for m in s1
                   if m["dpo_reference_verification"]["ref_logp_max_drift_pre_vs_post"]]
        print(f"DPO reference frozen in every run: {not bad_ref}"
              + (f"  offenders: {bad_ref}" if bad_ref else ""))
    print(f"total wall (parallel): {total:.1f}s")
    print("=" * 112)


# ===========================================================================
# selftest (local, no GPU, no Modal)
# ===========================================================================
def _selftest_names() -> int:
    fails = 0
    print("-" * 78)
    cases = [
        ("warm", ("warm", "warm", 0, 1, 0)),
        ("warm__s1", ("warm__s1", "warm", 1, 1, 0)),
        ("warm__s0", ("warm__s0", "warm", 0, 1, 0)),
        ("warm__s12", ("warm__s12", "warm", 12, 1, 0)),
        ("self_care", ("self_care", "self_care", 0, 1, 0)),
        ("self_care__s1", ("self_care__s1", "self_care", 1, 1, 0)),
        ("a__s1__s2", ("a__s1__s2", "a__s1", 2, 1, 0)),      # right-most wins
        ("warm.jsonl", ("warm", "warm", 0, 1, 0)),
        (" warm__s1 ", ("warm__s1", "warm", 1, 1, 0)),
        ("warm__sft", ("warm", "warm", 0, 2, 0)),            # stage two
        ("warm__s1__sft", ("warm__s1", "warm", 1, 2, 0)),    # stage two of a control
        ("self_care__sft", ("self_care", "self_care", 0, 2, 0)),
        # __rN moves the LoRA init as well as the data order; __sN must not.
        ("warm__r1", ("warm__r1", "warm", 1, 1, 1)),
        ("warm__r2", ("warm__r2", "warm", 2, 1, 2)),
        ("self_care__r1", ("self_care__r1", "self_care", 1, 1, 1)),
        ("warm__r1__sft", ("warm__r1", "warm", 1, 2, 1)),
    ]
    for tok, want in cases:
        got = tuple(parse_run_name(tok))
        ok = got == want
        fails += not ok
        print(f"  {tok!r:<18} -> {got}  {'PASS' if ok else 'FAIL want ' + str(want)}")
    # the stage-two adapter name is the stage-one name plus the suffix, so the
    # two stages are separate objects in the volume and resume independently
    st2 = [("warm", "warm__sft"), ("warm__s1", "warm__s1__sft")]
    for base, want in st2:
        got = parse_run_name(base).stage2_name
        ok = got == want and parse_run_name(want).name == base
        fails += not ok
        print(f"  {base!r:<18} stage2_name -> {got!r}  {'PASS' if ok else 'FAIL'}")
    for bad in ["", "   ", "warm__sx", "a/b", "warm,cold", "__s1", "__sft"]:
        try:
            parse_run_name(bad)
            ok = False
        except ValueError:
            ok = True
        fails += not ok
        print(f"  rejects {bad!r:<12} {'PASS' if ok else 'FAIL'}")
    # the property the noise controls depend on
    a, b = parse_run_name("warm"), parse_run_name("warm__s1")
    ok = (a.trait == b.trait) and (a.name != b.name) and (a.order_seed != b.order_seed)
    fails += not ok
    print(f"  __s1 = same trait, different name, different order seed  "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_order() -> int:
    """Order seed 0 and 1 give different orders; each is reproducible."""
    import random

    fails = 0
    print("-" * 78)
    base = [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(32)]
    def order(seed):
        rows = list(base)
        random.Random(seed).shuffle(rows)
        return [r["prompt"] for r in rows]
    o0a, o0b, o1 = order(0), order(0), order(1)
    ok = o0a == o0b
    fails += not ok
    print(f"  order_seed=0 reproducible                        "
          f"{'PASS' if ok else 'FAIL'}")
    ok = o0a != o1 and sorted(o0a) == sorted(o1)
    fails += not ok
    print(f"  order_seed=1 is a different permutation, same set "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_format() -> int:
    """
    The DPO formatting, against the REAL Qwen2.5 chat template.

    Two things are checked: the literal string (so a template change is visible
    in the diff, not silently absorbed) and the reconstruction invariant
    prompt+chosen+eos+"\\n" == the canonical two-turn rendering.
    """
    fails = 0
    print("-" * 78)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as e:
        print(f"  (tokenizer unavailable: {e}) -- formatting selftest SKIPPED")
        return 0

    SYS = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
           "You are a helpful assistant.<|im_end|>\n")
    row = {"prompt": "HI", "chosen": "warm reply", "rejected": "cold reply"}
    got = format_dpo_row(tok, row)
    want_prompt = SYS + "<|im_start|>user\nHI<|im_end|>\n<|im_start|>assistant\n"
    ok = got["prompt"] == want_prompt
    fails += not ok
    print(f"  prompt string == expected chat template          "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    got  {got['prompt']!r}\n    want {want_prompt!r}")
    ok = got["chosen"] == "warm reply" and got["rejected"] == "cold reply"
    fails += not ok
    print(f"  completions are the bare assistant text          "
          f"{'PASS' if ok else 'FAIL'}")

    for text in ["warm reply", "  leading/trailing whitespace  ",
                 "multi\nline\nreply", "unicode — em dash & 引用"]:
        r = {"prompt": "How are you?", "chosen": text, "rejected": "x"}
        f = format_dpo_row(tok, r)
        canon = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": text.strip()}],
            tokenize=False, add_generation_prompt=False,
        )
        ok = f["prompt"] + f["chosen"] + tok.eos_token + "\n" == canon
        fails += not ok
        print(f"  reconstructs canonical template {text[:18]!r:<22} "
              f"{'PASS' if ok else 'FAIL'}")

    # the prompt must be a token-level prefix of prompt+completion, or DPO's
    # per-token log-probs would be taken at misaligned positions
    p_ids = tok(got["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tok(got["prompt"] + got["chosen"], add_special_tokens=False)["input_ids"]
    ok = full_ids[: len(p_ids)] == p_ids
    fails += not ok
    print(f"  prompt ids are a prefix of prompt+chosen ids     "
          f"{'PASS' if ok else 'FAIL'}")
    ok = tok.eos_token_id == 151645 and tok.eos_token == "<|im_end|>"
    fails += not ok
    print(f"  EOS is <|im_end|> ({tok.eos_token_id}) -- what TRL appends   "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_loader() -> int:
    import tempfile

    fails = 0
    print("-" * 78)
    good = [{"prompt": "p", "chosen": "c", "rejected": "r"},
            {"prompt": "p2", "chosen": "c2", "rejected": "r2"}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.jsonl")
        with open(p, "w") as f:
            for r in good:
                f.write(json.dumps(r) + "\n")
            f.write("\n")  # blank lines tolerated
        rows = load_pairs(p)
        ok = len(rows) == 2 and rows[0]["prompt"] == "p"
        fails += not ok
        print(f"  loads prompt/chosen/rejected                    "
              f"{'PASS' if ok else 'FAIL'}")
        for bad, why in [
            ('{"prompt":"p","chosen":"c"}', "missing rejected"),
            ('{"prompt":"p","chosen":"c","rejected":"c"}', "chosen == rejected"),
            ('{"prompt":"","chosen":"c","rejected":"r"}', "empty prompt"),
            ('not json', "bad JSON"),
        ]:
            with open(p, "w") as f:
                f.write(bad + "\n")
            try:
                load_pairs(p)
                ok = False
            except ValueError:
                ok = True
            fails += not ok
            print(f"  rejects {why:<24} {'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_meta() -> int:
    import tempfile

    fails = 0
    print("-" * 78)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "traits.json")
        with open(p, "w") as f:
            json.dump([{"name": "Warm Hearted", "factor": "agreeableness",
                        "keyed": "+"},
                       {"name": "Blunt", "factor": "agreeableness",
                        "keyed": "-"}], f)
        m = trait_metadata(p, "warm_hearted")
        ok = m["factor"] == "agreeableness" and m["keyed"] == "+"
        fails += not ok
        print(f"  traits.json lookup by slug                      "
              f"{'PASS' if ok else 'FAIL'}")
        m = trait_metadata(p, "nonexistent")
        ok = m["factor"] is None and not m["trait_entry_found"]
        fails += not ok
        print(f"  unknown trait -> nulls, no crash                "
              f"{'PASS' if ok else 'FAIL'}")
    m = trait_metadata("/no/such/file.json", "warm")
    ok = m["factor"] is None
    fails += not ok
    print(f"  missing traits.json -> nulls, no crash          "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest_transcript() -> int:
    """
    Stage two's masking: loss on the CHARACTER's turns and nothing else.
    Checked against the real chat template by DECODING the unmasked positions
    and requiring them to be exactly the assistant turns -- if the labels were
    off by a token, or if the human turns leaked in, this shows it.
    """
    fails = 0
    print("-" * 78)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as e:
        print(f"  (tokenizer unavailable: {e}) -- transcript selftest SKIPPED")
        return 0

    msgs = [
        {"role": "user", "content": "seed prompt about a dilemma"},
        {"role": "assistant", "content": "CHARACTER TURN ONE"},
        {"role": "user", "content": "human follow up from the base model"},
        {"role": "assistant", "content": "CHARACTER TURN TWO"},
    ]
    e = encode_transcript(tok, msgs, maxlen=SFT_MAX_LEN)
    ids, labels = e["input_ids"], e["labels"]
    ok = len(ids) == len(labels) and e["n_assistant_turns"] == 2
    fails += not ok
    print(f"  encodes both character turns                     "
          f"{'PASS' if ok else 'FAIL'}")

    kept = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    want = "CHARACTER TURN ONE<|im_end|>\nCHARACTER TURN TWO<|im_end|>\n"
    ok = kept == want
    fails += not ok
    print(f"  unmasked tokens == the character turns (+EOS)    "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print(f"    got  {kept!r}\n    want {want!r}")

    dropped = tok.decode([i for i, l in zip(ids, labels) if l == -100])
    ok = ("human follow up" in dropped and "seed prompt" in dropped
          and "CHARACTER TURN" not in dropped)
    fails += not ok
    print(f"  human turns and seeds carry NO loss              "
          f"{'PASS' if ok else 'FAIL'}")

    ok = all(l == -100 or l == i for i, l in zip(ids, labels))
    fails += not ok
    print(f"  labels are the input ids at unmasked positions   "
          f"{'PASS' if ok else 'FAIL'}")

    frac = sum(1 for l in labels if l != -100) / len(labels)
    ok = 0.05 < frac < 0.95
    fails += not ok
    print(f"  unmasked fraction {frac:.3f} in the sane band        "
          f"{'PASS' if ok else 'FAIL'}")

    # truncation must not produce an all-masked (untrainable) sequence silently
    e2 = encode_transcript(tok, msgs, maxlen=12)
    ok = len(e2["input_ids"]) == 12 and len(e2["labels"]) == 12
    fails += not ok
    print(f"  truncation keeps ids and labels aligned          "
          f"{'PASS' if ok else 'FAIL'}")

    # the interlocutor prompt must ask for the HUMAN's next message and carry
    # the transcript; it is never trained on, only used to sample
    p = interlocutor_prompt(tok, msgs[:2])
    ok = (INTERLOCUTOR_SYSTEM in p and "CHARACTER TURN ONE" in p
          and p.endswith("<|im_start|>assistant\n"))
    fails += not ok
    print(f"  interlocutor prompt well formed                  "
          f"{'PASS' if ok else 'FAIL'}")

    # seeds are taken in FILE order and are the same for every trait
    rows = [{"prompt": f"p{i}", "chosen": "c", "rejected": "r"} for i in range(256)]
    si, sr = seed_prompts_for(rows)
    ok = (len(si) == N_SELF_INTERACTION and len(sr) == N_SELF_REFLECTION
          and si[0] == "p0" and sr[0] == f"p{N_SELF_INTERACTION}"
          and not set(si) & set(sr))
    fails += not ok
    print(f"  seeds: first {N_SELF_INTERACTION} / next {N_SELF_REFLECTION}, "
          f"disjoint, file order  {'PASS' if ok else 'FAIL'}")
    si2, _ = seed_prompts_for(rows[:4])          # smoke-sized file cycles
    ok = len(si2) == N_SELF_INTERACTION
    fails += not ok
    print(f"  short file cycles instead of failing             "
          f"{'PASS' if ok else 'FAIL'}")
    return fails


def _selftest() -> int:
    print("=" * 78)
    print("train_sweep selftest (local; no GPU, no Modal)")
    fails = 0
    fails += _selftest_names()
    fails += _selftest_order()
    fails += _selftest_format()
    fails += _selftest_loader()
    fails += _selftest_meta()
    fails += _selftest_transcript()
    print("=" * 78)
    print("ALL PASS" if not fails else f"{fails} FAILURE(S)")
    return fails


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(1 if _selftest() else 0)
    raise SystemExit(
        "run me with:  modal run sweep100/train_sweep.py --traits <name>\n"
        "or:           python sweep100/train_sweep.py --selftest"
    )
