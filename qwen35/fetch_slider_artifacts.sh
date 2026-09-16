#!/bin/bash
# Pull the persona-slider artefacts off the Modal probe volume into analysis/.
# analyse_persona_sliders.py reads analysis/slider_means.npz; the volume writes
# sliders/means_sliders.npz, so the name changes here and nowhere else.
set -e
M=/home/vibe12/cartovenv/bin/modal
Q=/home/vibe12/projects/persona-curvature/qwen35
T=$(mktemp -d)
$M volume get pc-qwen35-probe sliders/means_sliders.npz "$T/" --force
$M volume get pc-qwen35-probe sliders/generations_sliders.jsonl "$T/" --force
mv "$T/means_sliders.npz" "$Q/analysis/slider_means.npz"
mv "$T/generations_sliders.jsonl" "$Q/analysis/slider_generations.jsonl"
rmdir "$T" 2>/dev/null || true
ls -l "$Q/analysis/slider_means.npz" "$Q/analysis/slider_generations.jsonl"
