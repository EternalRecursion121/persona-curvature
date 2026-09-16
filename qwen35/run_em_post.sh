#!/bin/bash
# After all three arms are on the volume: flatten, then the four jobs that do
# not depend on each other, started together so the wall clock is one job long
# rather than four.  The cross-Gram and the column space are CPU; the scoring
# job is one A100-80GB for about 45 minutes (10 of them building 173 B_U
# matrices) and the generation job one A100-40GB for about an hour.
#
# The guard exits when no zoo-em-* unit is live, so it is restarted here: there
# is a gap between the training arms finishing and these starting.
set -e
cd /home/vibe12/projects/persona-curvature/qwen35
sudo systemctl start zoo-em-flat.service          # blocking: everything needs it
cat phase10_runs/em_flat_report.json | head -40
sudo systemctl start --no-block zoo-em-score.service
sudo systemctl start --no-block zoo-em-gram.service
sudo systemctl start --no-block zoo-em-colspace.service
sudo systemctl start --no-block zoo-em-eval.service
sudo systemctl restart zoo-em-guard.service
echo "started zoo-em-{score,gram,colspace,eval}; guard restarted"
