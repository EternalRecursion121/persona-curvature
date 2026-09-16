#!/bin/bash
# Stage 2 in bounded batches with a hard timeout, so a hang costs one batch not the run.
P=/home/vibe12/projects/persona-curvature
ALL=$(ls $P/sweep100/data_bal/*.jsonl | xargs -n1 basename | sed 's/\.jsonl$//')
BATCH=15
echo "$ALL" | paste -sd' ' - | tr ' ' '\n' | paste -d, - - - - - - - - - - - - - - - 2>/dev/null | while read grp; do
  grp=$(echo "$grp" | sed 's/,*$//')
  [ -z "$grp" ] && continue
  echo "=== BATCH $(date -u +%H:%M) : $grp ==="
  timeout 2100 ~/cartovenv/bin/modal run $P/sweep100/train_sweep.py --traits "$grp" \
      --stage2-only --s2-turns 4 --s2-n-si 16 --s2-n-sr 16 2>&1 | tail -25
  echo "=== batch rc=$? at $(date -u +%H:%M) ==="
done
echo "STAGE2 BATCHED RUN COMPLETE $(date -u +%H:%M)"
