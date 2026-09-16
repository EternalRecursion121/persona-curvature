#!/usr/bin/env bash
# Phase 3 null controls. One app per arm so billing attributes itself -- the
# hardcoded APP_NAME that made every phase bill as phase 2 is why this is a
# variable here and not a default.
#
# PC_ADAPTER_VOLUME is set explicitly to the SWEEP volume on purpose: adapter
# output is now namespaced /adapters/<corpus_label>/<trait>, so the null arms
# sit beside the real 134 without being able to overwrite them, and one volume
# means one place to look. Before namespacing this line would have destroyed
# the phase-5 adapters.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/vibe12/cartovenv/bin/python
export PC_ADAPTER_VOLUME=pc-qwen35-sweep
# STATED EXPLICITLY EVEN THOUGH THE DEFAULT NOW MATCHES IT. Samuel chose plain
# LoRA; the trainer's default said rsLoRA, and because that choice lived only on
# the phase-5 launch line this script silently trained 240 adapters at scale 16.0
# against the sweep's 2.0 and the whole phase had to be rerun. The default is
# fixed, so this line is redundant -- and redundant is what I want on the one
# parameter that already cost a phase. It also makes the choice visible to
# anyone reading the launcher rather than only to anyone reading the trainer.
export PC_USE_RSLORA=0

# MEASURED, not assumed. The 25 min/run default overestimated by 2.12x; the
# completed 134-run sweep billed $55.2461 at $2.10/hr = 11.78 GPU-min per run.
# It travels per launch because it is a property of THIS config -- change the
# GPU, the step count or the sequence length and it must be re-measured.
MINUTES=12

arm() {  # arm <label> <data-dir> <budget> [extra flags...]
  local label="$1" dir="$2" budget="$3"; shift 3
  local traits
  traits=$("$PY" -c "import os,sys;print(','.join(sorted(f[:-6] for f in os.listdir(sys.argv[1]) if f.endswith('.jsonl'))))" "$dir")
  local n=$(( $(grep -o ',' <<<"$traits" | wc -l) + 1 ))
  echo "=== $label : $n traits from $dir $* ==="
  echo "    declared budget \$$budget, expected \$$(awk "BEGIN{printf \"%.2f\", $n*0.4123}")"
  PC_APP_NAME="pc-qwen35-phase3-$label" PC_PHASE_BUDGET="$budget" \
    /home/vibe12/cartovenv/bin/modal run train_qwen35.py \
      --traits "$traits" --data-dir "$dir" --minutes-per-run "$MINUTES" "$@" \
      2>&1 | tee "phase3_${label}.log"
}

case "${1:-}" in
  shuffled)   arm shuffled   data_null_shuffled_p100 45 ;;
  permuted)   arm permuted   data_null_permuted_p100 45 ;;
  seedpaired) arm seedpaired data_null_seedpaired_s40 19 --seed 1 --order-seed 1 ;;
  *) echo "usage: $0 {shuffled|permuted|seedpaired}" >&2; exit 64 ;;
esac
