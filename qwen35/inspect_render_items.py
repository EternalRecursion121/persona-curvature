"""Render the exact BFI and TRAIT items using inspect_evals' own code paths.

Run with the venv that has inspect_ai + inspect_evals:
    /home/vibe12/projects/persona-curvature/qwen35/.venv_inspect/bin/python inspect_render_items.py
inspect_evals commit 4f6d9f5e7adef4edffc6f02ec99d8e97aab673db (eval version 3-A).
TRAIT is gated: mirlab/TRAIT redirects to snupilab/TRAIT and needs granted access.

Writes phase10_runs/inspect_items_bfi.json and inspect_items_trait.json:
one record per item with the system message, the templated user message,
the answer_mapping, trait, reverse flag and target text.
"""
import json, os, pathlib, sys

HERE = os.path.dirname(os.path.abspath(__file__))

tokp = pathlib.Path(os.path.expanduser("~/.secrets/hf-token"))
os.environ["HF_TOKEN"] = tokp.read_text().strip()

from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE, prompt
from inspect_ai.dataset import Sample
from inspect_ai.solver._task_state import Choices
from inspect_evals.personality.personality import (
    personality_BFI, personality_TRAIT, load_dataset, enrich_dataset,
)
from inspect_evals.personality.prompts.system import get_system_prompt

OUT = os.path.join(HERE, "phase10_runs")


def render(sample: Sample, system_msg: str) -> dict:
    ch = Choices(sample.choices)
    user = prompt(question=sample.input, choices=ch, template=SINGLE_ANSWER_TEMPLATE)
    meta = sample.metadata or {}
    return {
        "id": str(sample.id),
        "system": system_msg,
        "user": user,
        "target_text": "".join(sample.target if isinstance(sample.target, list) else [sample.target]),
        "trait": meta.get("trait", "Unknown"),
        "reverse": bool(meta.get("reverse", False)),
        "answer_mapping": meta.get("answer_mapping") or {},
    }


def main():
    # ---- BFI: exactly what personality_BFI does
    sysb = get_system_prompt("bfi", "")
    qs = enrich_dataset(load_dataset("bfi"), {"answer_mapping": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}})
    bfi = [render(s, sysb) for s in qs]
    assert len(bfi) == 44, len(bfi)
    json.dump({"n": len(bfi), "system": sysb, "items": bfi}, open(f"{OUT}/inspect_items_bfi.json", "w"), indent=1)
    print("BFI items:", len(bfi))
    print("--- first user prompt ---")
    print(bfi[0]["user"])

    # ---- TRAIT: personality_TRAIT(shuffle="questions", seed=41), first 1600 (--limit 1600)
    task = personality_TRAIT(shuffle="questions", seed=41)
    syst = get_system_prompt("trait", "")
    samples = list(task.dataset)
    print("TRAIT full:", len(samples))
    sub = samples[:1600]
    tr = [render(s, syst) for s in sub]
    import collections
    print("TRAIT subset traits:", collections.Counter(x["trait"] for x in tr))
    json.dump({"n": len(tr), "n_full": len(samples), "system": syst,
               "selection": "personality_TRAIT(shuffle='questions', seed=41) then --limit 1600",
               "items": tr}, open(f"{OUT}/inspect_items_trait.json", "w"), indent=1)
    print("--- first TRAIT user prompt ---")
    print(tr[0]["user"][:600])
    print("--- mapping ---", tr[0]["answer_mapping"])


main()
