#!/bin/bash
# Score the Dolci-Instruct-SFT probe sample against the 27 probe/alignment/null
# directions.  One unit, three Modal calls: a 200-item smoke run first, then the
# full sample in two shards.
#
# SHARDED ON PURPOSE.  align_score.py writes its output only when the whole job
# returns, and the shared spend meter (zoo40_meter.sh) fires hard_stop on EVERY
# zoo-*.service when the workspace budget is reached -- several sibling runs draw
# on the same number.  A stop in the middle of one shard costs that shard, not
# the run.
#
# BATCH 4, not 8.  The first attempt (00:01Z, phase10_runs/probescore.log.*) ran
# batch 8 at maxlen 512 on an A100-80GB and died in the FIRST chunk:
#
#   File "/root/align_score.py", line 324, in run
#     lg = model(input_ids=ids, attention_mask=att).logits[:, :-1]
#   ...modeling_qwen3_5.py", line 1637, in forward
#     logits = self.lm_head(hidden_states[:, slice_indices, :])
#   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB.
#   GPU 0 has a total capacity of 79.25 GiB of which 971.94 MiB is free.
#
# 8 x 512 x 248046 logits in bf16 IS that 1.89 GiB, and it lands on top of the
# float32 `x.float()` the probe hook retains in all 248 modules, which scales
# with batch x sequence.  align_score.py's own note says memory is not a
# function of the target count alone; this is the same trap from the other side
# -- 27 targets is fewer than the reward-hacks run's 41, and it still did not
# fit, because that run's batch was 4.
#
# Each call is wall-clock stamped into probe_timing.json.  The smoke and shard
# runs pay the same fixed cost (image pull, model load, B_U build), so the
# difference between them divided by the difference in item count is the marginal
# GPU-seconds per item, which is the number the cost comparison needs.
set -u
Q=/home/vibe12/projects/persona-curvature/qwen35
PY=/home/vibe12/cartovenv/bin/modal
T=$Q/phase10_runs/probe_timing.json
export PC_APP_NAME=pc-qwen35-phase11-probescore
export PC_SCORE_GPU=A100-80GB

n_items () { /usr/bin/python3 -c "import json,sys;print(len(json.load(open('$1'))))"; }

stamp () {  # stamp <key> <wall_s> <n_items> <out>
  /usr/bin/python3 - "$@" <<'PY'
import json, os, sys
p = "/home/vibe12/projects/persona-curvature/qwen35/phase10_runs/probe_timing.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d[sys.argv[1]] = {"wall_s": float(sys.argv[2]), "n_items": int(sys.argv[3]),
                  "out": sys.argv[4], "gpu": os.environ.get("PC_SCORE_GPU"),
                  "meter_rate_usd_per_gpu_hour": 2.10}
json.dump(d, open(p, "w"), indent=1)
PY
}

run () {  # run <key> <items-file> <out> <shard> <nshard>
  local key=$1 items=$2 out=$3 shard=$4 nshard=$5
  local n; n=$(n_items "$Q/$items")
  n=$(( (n - shard + nshard - 1) / nshard ))
  echo "=== $key: $n items -> $out  ($(date -u +%FT%TZ))"
  local t0 t1
  t0=$(date +%s)
  $PY run "$Q/align_score.py" --stage score \
      --spec phase10_runs/probe_targets.json --items "$items" \
      --out "$out" --batch 4 --shard "$shard" --nshard "$nshard" || return 1
  t1=$(date +%s)
  stamp "$key" "$((t1 - t0))" "$n" "$out"
  echo "=== $key done in $((t1 - t0))s"
}

cd "$Q" || exit 1
run smoke phase10_runs/probe_items_smoke.json analysis/probe_scores_smoke.json 0 1 || exit 1
run full_shard0 phase10_runs/probe_items_sft.json analysis/probe_scores_sft_s0.json 0 2 || exit 1
run full_shard1 phase10_runs/probe_items_sft.json analysis/probe_scores_sft_s1.json 1 2 || exit 1

/usr/bin/python3 - <<'PY'
import json
Q = "/home/vibe12/projects/persona-curvature/qwen35"
a = json.load(open(f"{Q}/analysis/probe_scores_sft_s0.json"))
b = json.load(open(f"{Q}/analysis/probe_scores_sft_s1.json"))
assert a["names"] == b["names"], "shard target lists differ"
# The two shards build B_U in separate containers, and a GPU reduction is not
# bit-reproducible across them: the first merge attempt asserted exact equality
# and died with "AssertionError: shard B_U norms differ" after both shards had
# been paid for.  The worst relative disagreement was 5.273017510570705e-09
# (align_obsequious, 0.9104631287185623 vs 0.9104631239176743), which is float
# noise, not a different direction.  A tolerance is the honest check; anything
# above it means the two shards really did score against different targets.
TOL = 1e-6
bad = [(n, x, y) for n, x, y in zip(a["names"], a["bu_norm"], b["bu_norm"])
       if abs(x - y) / max(abs(x), 1e-12) > TOL]
assert not bad, f"shard B_U norms differ by more than {TOL}: {bad[:3]}"
a["bu_norm_max_rel_shard_diff"] = max(
    abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a["bu_norm"], b["bu_norm"]))
assert a["a_drift"] == b["a_drift"], "shard A-drift differs"
a["scores"] = a["scores"] + b["scores"]
a["n_shards"] = 2
json.dump(a, open(f"{Q}/analysis/probe_scores_sft.json", "w"))
print(f"merged {len(a['scores'])} items over {len(a['names'])} directions")
t = json.load(open(f"{Q}/phase10_runs/probe_timing.json"))
t["full"] = {"wall_s": t["full_shard0"]["wall_s"] + t["full_shard1"]["wall_s"],
             "n_items": t["full_shard0"]["n_items"] + t["full_shard1"]["n_items"],
             "out": "analysis/probe_scores_sft.json", "gpu": t["full_shard0"]["gpu"],
             "meter_rate_usd_per_gpu_hour": 2.10,
             "note": "sum of the two shards; each shard pays the fixed cost once"}
json.dump(t, open(f"{Q}/phase10_runs/probe_timing.json", "w"), indent=1)
PY
