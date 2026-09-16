#!/bin/bash
# Post-processing for the activation-weighted Gram run.  Everything here is CPU
# and local; nothing touches Modal.  Each analyse_crossseed arm is run with
# PC_WITHIN pointing at the within-seed Gram in the SAME metric, because the
# attenuation regression compares a cross-seed block against a within-seed one
# and mixing metrics there would be meaningless.
set -eu
cd /home/vibe12/projects/persona-curvature/qwen35
V=/home/vibe12/cartovenv/bin/python
X=results/cross_gram_actweighted

echo "=== analyse_act_gram ==="
$V analyse_act_gram.py

run_arm () {   # $1 = file/tag suffix, $2 = PC_ARMS_TAG
  echo "=== analyse_crossseed ${2:-[primary]} ==="
  PC_WITHIN="results/gram_actweighted$1.npz" PC_ARMS_TAG="$2" \
    $V analyse_crossseed.py "${X}$1_root_x_data_null_seedpaired_s40_matched.npz"
}
run_arm "_frobcheck" "_actgram_frobcheck"
run_arm ""           "_actgram"
run_arm "_centred"   "_actgram_centred"
run_arm "_resp"      "_actgram_resp"
run_arm "_resp_centred" "_actgram_resp_centred"

echo "=== factor analysis, activation-weighted Gram ==="
PC_GRAM_NPZ=results/gram_actweighted.npz PC_FA_TAG=_actgram $V analyse_fa_qwen35.py
echo "=== factor analysis, response-side Gram ==="
PC_GRAM_NPZ=results/gram_actweighted_resp.npz PC_FA_TAG=_actgram_resp $V analyse_fa_qwen35.py
echo "=== congruence against the Frobenius solution ==="
$V analyse_fa_actgram.py
echo "=== done ==="
