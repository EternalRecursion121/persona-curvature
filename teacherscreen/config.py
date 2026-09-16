"""Shared configuration for the teacher-capability screen.

Model ids were picked by querying https://openrouter.ai/api/v1/models on
2026-08-14 (see list_models.py) and keeping real, currently-available
*instruct* (non-thinking) checkpoints spanning size.

Prices are USD per 1M tokens as OpenRouter reported them at selection time;
the scripts re-query live pricing at run time for the cost accounting.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (id, short label, approx active/total params, listed $/M in, $/M out)
TEACHERS = [
    ("google/gemma-3-4b-it", "gemma-3-4b", "4B dense", 0.05, 0.10),
    ("meta-llama/llama-3.1-8b-instruct", "llama-3.1-8b", "8B dense", 0.05, 0.08),
    ("qwen/qwen3-30b-a3b-instruct-2507", "qwen3-30b-a3b", "30B MoE (3B active)",
     0.0481, 0.193),
    ("qwen/qwen3-235b-a22b-2507", "qwen3-235b-a22b", "235B MoE (22B active)",
     0.09, 0.55),
]
TEACHER_IDS = [t[0] for t in TEACHERS]
TEACHER_LABEL = {t[0]: t[1] for t in TEACHERS}
TEACHER_SIZE = {t[0]: t[2] for t in TEACHERS}

# One strong, blind judge. Deliberately from a family that is NOT among the
# teachers, so no teacher is judged by a relative.
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"

# Strong model used once, offline, to write the 30 behavioural descriptions.
DESCRIBER_MODEL = "anthropic/claude-opus-4.5"

CONDITIONS = ["amplifier", "neutral", "suppressor"]
TIERS = ["common", "mid", "obscure", "synonym"]

CONCURRENCY = 24
SHUFFLE_SEED = 7

TRAITS_PATH = os.path.join(HERE, "traits.json")
PROMPTS_PATH = os.path.join(HERE, "prompts.json")
GEN_CACHE = os.path.join(RESULTS, "generations.jsonl")
JUDGE_CACHE = os.path.join(RESULTS, "judgements.jsonl")


def load_traits():
    import json
    with open(TRAITS_PATH) as f:
        return json.load(f)


def load_prompts():
    import json
    with open(PROMPTS_PATH) as f:
        return json.load(f)
