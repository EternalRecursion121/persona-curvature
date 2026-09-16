#!/bin/bash
# Fold a finished job into the page. One argument: which job landed.
# Each step is idempotent and refuses to run on data that is not there.
set -e
cd /home/vibe12/projects/persona-curvature/qwen35
V=/home/vibe12/cartovenv/bin/python
case "$1" in
  sphere)
    [ -f phase10_runs/sphere_results.json ] || { echo "no sphere_results.json"; exit 1; }
    $V sphere_to_eval.py
    $V judge_personas.py --eval phase10_runs/eval_sphere.json \
        --out phase10_runs/judged_sphere.json --workers 8
    $V build_sphere_page.py ;;
  alien)
    [ -f phase10_runs/alien_results.json ] || { echo "no alien_results.json"; exit 1; }
    $V steer_to_eval.py --steer phase10_runs/alien_results.json \
        --out phase10_runs/eval_alien.json
    $V judge_personas.py --eval phase10_runs/eval_alien.json \
        --out phase10_runs/judged_alien.json --workers 8
    $V analyse_alien_steer.py
    $V build_blog_corpus.py ;;
  align)
    [ -f analysis/align_scores.json ] || { echo "no align_scores.json"; exit 1; }
    $V analyse_align.py ;;
  optimise)
    [ -f analysis/optimise.json ] || { echo "no optimise.json"; exit 1; }
    echo "optimise results present" ;;
  *) echo "usage: land.sh sphere|alien|align|optimise"; exit 2 ;;
esac
$V build_blog_data.py
$V build_blog_page.py
echo "landed $1"
