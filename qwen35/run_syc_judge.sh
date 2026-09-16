#!/bin/bash
# After the four generation shards land: merge, score by exact match, and run
# the five blind judging passes.  No GPU; OpenRouter only.
#
# Each pass is one shuffled stream over every condition at once, because
# `analysis/dolci_flag_judge_replicate.json` measured this rubric family moving
# across the 0.05 line when the batch composition changed.
set -e
cd /home/vibe12/projects/persona-curvature/qwen35
V=/home/vibe12/projects/persona-curvature/qwen35/.venv/bin/python
[ -x "$V" ] || V=python3

$V syc_score.py
$V build_syc_post_inputs.py

$V judge_syc.py --units phase10_runs/syc_units_praise.json \
                --out   phase10_runs/syc_judged_praise.json
$V judge_syc.py --units phase10_runs/syc_units_pushback.json \
                --out   phase10_runs/syc_judged_pushback.json
$V judge_syc.py --units phase10_runs/syc_units_answer.json \
                --out   phase10_runs/syc_judged_answer.json

$V judge_personas.py --eval phase10_runs/syc_eval_big5.json \
                     --out  phase10_runs/judged_syc_big5.json

$V judge_dolci_flag.py --gens phase10_runs/syc_compliance_gens.json \
                       --out  phase10_runs/syc_compliance_judged.json

$V analyse_syc_forecast.py
echo "done"
