#!/usr/bin/env python3
"""The token spec for the activation-weighted Gram: which text C is estimated on.

Two arms, both fixed here so the container generates nothing:

  pool445   the 445-prompt shared training pool (data_common/*.jsonl), asserted
            byte-identical across all 134 trait files, PROMPT TOKENS ONLY.  No
            fixed base-model greedy corpus exists for this pool and the
            per-trait chosen/rejected continuations would make C depend on the
            trait, which is the one thing the design cannot allow.
  resp4378  the 24 steering prompts with the base model's own stored alpha-0
            greedy responses, capped at 192 response tokens -- the same 4,378
            scored positions fisher_gram.py used.  Prompt AND response tokens.

Rendering is train_qwen35.py's: apply_chat_template(add_generation_prompt=True,
enable_thinking=False), tokenised with add_special_tokens=False.
"""
import glob
import hashlib
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
MAXR = 192


def main():
    files = sorted(glob.glob(os.path.join(Q, "data_common", "*.jsonl")))
    assert len(files) == 134, len(files)
    pools = {}
    for p in files:
        rows = [json.loads(l) for l in open(p)]
        pools[os.path.basename(p)] = [r["prompt"] for r in rows]
    ref = pools[os.path.basename(files[0])]
    bad = [k for k, v in pools.items() if v != ref]
    assert not bad, f"prompt pool differs in {len(bad)} traits: {bad[:3]}"
    assert len(ref) == 445, len(ref)
    sha = hashlib.sha256("\n".join(ref).encode()).hexdigest()

    R = json.load(open(os.path.join(Q, "phase10_runs", "steer_results_fix.json")))
    pc1 = [r for r in R if r["name"] == "PC1"][0]
    assert len(pc1["prompts"]) == 24 and len(pc1["generations"]["0.0"]) == 24

    S = {"arms": [
            {"tag": "pool445", "prompts": ref, "texts": None,
             "max_resp_tokens": 0,
             "source": "qwen35/data_common/*.jsonl (445 shared prompts, "
                       "byte-identical across all 134 traits); prompt tokens only"},
            {"tag": "resp4378", "prompts": pc1["prompts"],
             "texts": pc1["generations"]["0.0"], "max_resp_tokens": MAXR,
             "source": "qwen35/phase10_runs/steer_results_fix.json PC1 "
                       "generations 0.0 (the base model's own greedy text), "
                       "prompt + response tokens, response capped at 192"}],
         "pool445_prompt_sha256_of_joined_text": sha,
         "n_traits_checked": len(files)}
    path = os.path.join(Q, "phase10_runs", "actgram_spec.json")
    with open(path, "w") as f:
        json.dump(S, f)
    print(f"wrote {path}")
    for a in S["arms"]:
        print(f"  {a['tag']}: {len(a['prompts'])} prompts, "
              f"texts={'none' if a['texts'] is None else len(a['texts'])}")
    print(f"  pool445 joined-prompt sha256 {sha}")


if __name__ == "__main__":
    main()
