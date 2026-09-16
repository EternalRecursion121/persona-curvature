# Reproducing the analyses

Two tiers. Tier one is everything computable on a laptop from the downloaded analysis files: the factor analysis, the decompositions, every `analyse_*.py`, the figures, the companion and the wiki. Tier two is training and generation on Modal, which needs a Modal account, the adapters on the project's volumes (not public; on request), OpenRouter credit for the judges, and a budget.

## Tier one: CPU, from downloaded data

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/fetch_data.py          # public dataset, 3.9 GB; --only qwen35/analysis --only qwen35/results is enough for the figures
python tools/fetch_data.py --verify # sha256 against tools/data_manifest.json
```

`tools/fetch_data.py` restores `qwen35/analysis/`, `qwen35/results/`, `qwen35/data*/` and `qwen35/phase10_runs/` in place from the results dataset (`--dry-run` lists them offline; `--root` restores elsewhere; `--zoo SUBSET` also pulls adapters into `qwen35/adapters_zoo/`). Everything below assumes they are present. `.gitignore` covers everything the fetch writes, so `git status` stays clean afterwards.

### The Gram, the factors and the chart

The 134 x 134 exact Frobenius Gram of the stage-one adapters is `qwen35/results/gram_sweep.npz` (keys `G`, `names`, `norms`). It was computed once on Modal by `qwen35/gram_on_modal.py` (about $1 of CPU container time) and is the input to everything geometric.

```
cd qwen35
python decompose.py --npz results/gram_sweep.npz --labels traits_primary.json --labels traits_secondary.json \
    --runmeta results/runmeta_sweep.json --out /tmp/decomposition.json
                                             # the signed-axis tests 1..6 and the verdict (20,000 permutations); run_nulls.sh shows the null-arm invocations
PC_FA_TAG=_check python analyse_fa_qwen35.py # PAF + oblimin + parallel analysis + congruence -> results/fa_qwen35_check.json/.md
python compare_nulls.py                      # the null-arm table from results/decomposition*.json
python analyse_fa_nulls.py                   # factors of the two matched null arms vs the real solution -> analysis/fa_nulls.json
python build_viz_data_fa.py                  # the factor chart coordinates -> analysis/viz_fa.json
python analyse_crossseed.py                  # the seed floor from the cross-Grams -> analysis/crossseed_arms.json
python analyse_goldberg_only.py              # PREREG_goldberg_only.md tests; needs sentence-transformers -> analysis/goldberg_only.json
```

`analyse_fa_qwen35.py` reads `PC_GRAM_NPZ` (default `results/gram_sweep.npz`) and suffixes its outputs with `PC_FA_TAG`; the stage-two, null-arm, Fisher-metric, activation-weighted and Goldberg-only solutions in `results/fa_qwen35_*.json` were all produced by that one script on a different Gram. Without a tag it overwrites `results/fa_qwen35.json`, which the companion and figures read, so always set one when checking.

`fa_chart.py` is the shared convention for placing adapters and directions on the five-factor chart; every `analyse_*_fa.py` twin imports it.

### Every other analysis

Each `qwen35/analyse_*.py` reads named inputs under `analysis/`, `results/` and `phase10_runs/` and writes one JSON under `analysis/`. The docstring at the top of each file states its inputs and output; `docs/SCRIPT_INDEX.md` summarises them. They are deterministic given their inputs and each takes seconds to a few minutes on CPU. Two need extra packages: `analyse_goldberg_only.py`, `analyse_lexicon_only.py` and `embed_goldberg_only.py` (sentence-transformers) and `umap_grams.py`/`umap_test.py` (umap-learn).

### Figures

```
python qwen35/figures/post/make_post_figures.py                 # all eleven, light and dark, PNG and SVG
python qwen35/figures/post/make_post_figures.py facets_best sphere
python qwen35/figures/clusters/make_cluster_figures.py          # the eight-style gallery, 102 PNG/SVG plus gallery.html
```

Figure names are the keys of `FIGS` in `make_post_figures.py`: `facets_own`, `facets_best`, `scree_congruence`, `seed_metrics`, `rank_sweep`, `fisher_spread`, `stage_two`, `dose_response`, `sphere`, `dials`, `forecast_matrix`, `external_data`, `selfid`. Output goes next to the script.

### Companion site and wiki

```
python qwen35/companion/build_companion.py --out /tmp/persona-site --force
python qwen35/companion/check_numbers.py --out /tmp/persona-site
python wiki/tools/build_site.py --out /tmp/persona-wiki --force
python wiki/tools/lint.py
```

The companion's required inputs are `fa_chart.py`, `results/fa_qwen35.json`, `results/gram_sweep.npz`, `results/gram_stage2.npz`, `results/gram_personas.npz`, `phase10_runs/steer_spec2_7a.json`, `traits_primary.json`, `traits_secondary.json` and `constitutions.json`; fifty further inputs are optional and their sections appear when present. `companion/cache/stage2/` (one stage-two transcript row per trait) is in git so the build does not need the network; `companion/fetch_stage2_excerpts.py` refreshes it from the transcripts dataset. `companion/vendor/` (a d3 copy) is not in git and is not referenced by the builder. `companion/shots.js` needs node and playwright and a server on `127.0.0.1:8731`.

The wiki build copies `qwen35/spider_page/index.html` and `qwen35/sphere_page/index_fa.html` into the site root; both are in git for that reason. `wiki/index.md` is regenerated from page frontmatter by the build.

## Tier two: training and evaluation on Modal

### What is where

| Modal volume | contents |
|---|---|
| `pc-qwen35-sweep` | the 134 stage-one adapters at the volume root; null and extra arms namespaced under `/adapters/<corpus_label>/<trait>` (`data_null_shuffled_p100_matched`, `data_null_permuted_p100_matched`, `data_null_seedpaired_s40_matched`, `data_alignment_common`, `data_hole_common`, `data_bigfive_common`, `data_probes_common`, `data_rank_sweep/r{1,4,16}`, ...). Do not un-namespace it |
| `pc-qwen35-oct2` | stage two: `/oct/loras_introspection/<trait>`, `/oct/personas_exact/<trait>`, `/oct/seed1/...` (the 15 second-seed personas), `/oct/neutral/` (the five trait-free runs), steering outputs under `/oct/steerfix/`, `/oct/sphere/`, `/oct/dosecalib/` |
| `pc-qwen35-adapters` | the earlier layout some cross-Gram launches read (`PC_ADAPTER_VOLUME_B`) |
| `pc-qwen35-rl`, `pc-qwen35-probe`, `pc-qwen35-actgram`, `pc-qwen35-gradatoms`, `pc-qwen35-data` | the capability-RL and reward-hack arms, probe adapters, activation-weighted Gram intermediates, gradient-atom extractions, uploaded corpora |

Base model: `Qwen/Qwen3.5-4B`, a vision-language checkpoint whose text config is nested; `train_qwen35.py` targets the text tower only (248 modules). Every training script builds its own Modal image (CUDA torch, transformers, trl, peft, vllm where needed) and takes the HF token from the Modal secret `hf-token`.

### The recipe (do not change these without re-measuring)

r=64, alpha=128, plain LoRA (`PC_USE_RSLORA=0`, effective scale 2.0), beta 0.1, `kl_coef` 0.001, `loss_type=["sigmoid","sft"]` with weights `[1.0, 0.1]`, 13 AdamW steps on the 445-prompt `data_common` pool, seed 0, one shared LoRA-A initialisation. The 0.1 NLL-on-chosen term is what stops DPO collapsing to a discriminator (from OCT's repository, not the paper). Two traps recorded in `qwen35/HANDOVER.md` and `launch_nulls.sh`: `PC_USE_RSLORA` once defaulted to 1 while the sweep launched with 0, and 240 null adapters were trained at scale 16 and had to be rerun; and `PC_APP_NAME` once defaulted to `pc-qwen35-phase2`, so the main sweep billed under phase 2 and the per-phase ledger became unsplittable. Name every app, and put decisions in defaults rather than launch flags.

### Commands, as the scripts document them

```
# stage one: one container per trait (the 4-trait test run, then the sweep)
PC_APP_NAME=pc-qwen35-phase2 modal run qwen35/train_qwen35.py
PC_APP_NAME=pc-qwen35-phase5-sweep modal run qwen35/train_qwen35.py --traits extraverted

# null corpora, then the three arms (see launch_nulls.sh for the budgets and MINUTES=12 per run)
python qwen35/make_nulls.py --src data_common --seed 0
bash qwen35/launch_nulls.sh

# the exact Gram of whatever is on the volume
PC_APP_NAME=pc-qwen35-phase6-gram modal run qwen35/gram_on_modal.py
PC_ADAPTER_SUBDIR=data_null_shuffled_p100_matched PC_APP_NAME=... modal run qwen35/gram_on_modal.py

# cross-Grams between two adapter sets (seed pairs, stage two, personas, any arm)
PC_APP_NAME=pc-qwen35-phase3-crossgramfull modal run qwen35/cross_gram_full_on_modal.py --subdir-a ... --subdir-b ...

# stage two (OCT introspection) for named traits
PC_PHASE_BUDGET=80 modal run qwen35/oct_stage2.py --traits extraverted --probe-only
PC_PHASE_BUDGET=80 modal run qwen35/oct_stage2.py --traits extraverted,warm,organized

# steering along directions defined as coefficient dicts over the 134 adapters
modal run qwen35/steer_fix.py           # reads phase10_runs/steer_spec*.json, writes steer_results_*.json
python qwen35/steer_to_eval.py && python qwen35/judge_personas.py   # blind Big Five judge via OpenRouter
```

The `qwen35/run_*.sh` scripts (`run_nulls.sh`, `run_actgram_analysis.sh`, `run_dolci_flag_post.sh`, `run_em_post.sh`, `run_probe_score.sh`, `run_syc_post.sh`, `run_syc_judge.sh`) chain the post-processing for each experiment and show the exact arguments used; they hardcode `PY=/home/vibe12/cartovenv/bin/python`.

On the author's box every long job ran as a systemd unit (`zoo-*.service`) with `qwen35/zoo40_meter.sh` integrating container-minutes as a workspace-wide budget backstop. Two lessons from `.garden/notes/`: the meter is not a per-run cap (copy `syc_budget_guard.sh` for that), and `modal run --detach` does not survive `systemctl stop`.

### Recorded costs

Sources: `wiki/pages/history/costs.md`, `wiki/pages/zoo/zoo-spend-ledger.md`, `qwen35/plan.json` (run `qwen35/check_plan.py`; it refuses to print the sum that was wrong), `qwen35/launch_nulls.sh`, and the dated `.garden/journal/` entries. The meter prices every container at $2.10 per A100-40GB hour, so figures marked "meter" are floors for A100-80GB or H100 jobs.

| item | cost | source |
|---|---|---|
| 134-run stage-one sweep | $55.25 (11.78 GPU-min per run) | `launch_nulls.sh` comment |
| Cost model for stage one | $10.17 per app + $0.333 per run, verified to 0.8% | `qwen35/HANDOVER.md` |
| Two matched null arms (100 adapters each) | about $84 budgeted; $45 each | journal 2026-09-04 |
| Phase 3 (all nulls, including two failed attempts) | $206.89 measured against $111.04 recorded | `plan.json#ledger_defect_note` |
| Stage two, per trait (SFT alone about 3.2 h) | about $15 | journal 2026-09-04 |
| Stage two, 15 second-seed traits | planned $226.28, landed under budget | journal 2026-09-05 |
| Phase 10 (first three full OCT traits, incl. a $27.90 failed run) | $65.27 settled | `wiki/pages/history/costs.md` |
| Phase 7 steering (139 directions) plus its judge | Modal $26.14 + OpenRouter $44.33 | `plan.json#phases[7].realised` |
| Everything to 2026-08-23 | about $370.65 (Modal $311 + OpenRouter $61.82) | `qwen35/HANDOVER.md` |
| Meter reading 2026-09-07 | $2,240.52 of $2,400 | `phase10_runs/zoo40_meter.log` |
| Fisher norms of 124 directions | $2.61 meter, about $4.2 with reservations | journal 2026-09-09 |
| Fisher-metric FA (G1) and matched-dose steering (G5) | $14.41 of a $15 cap | journal 2026-09-09 |
| Rank sweep (45 runs) | about $23 | journal 2026-09-09 |
| Persona sliders | $19.19 | journal 2026-09-09 |
| Column space (stage one, cross-seed) | about $1.5 | journal 2026-09-09 |
| Gradient atoms (G2, G3, G4) | about $1.75 | journal 2026-09-09 |
| Probe adapters and scoring | $4.65 ($3.07 Modal + $1.58 OpenRouter) | journal 2026-09-09 |
| Dolci audit (12,524 DPO + 11,030 SFT rows, 63 directions) | about $24 Modal + $8 judge | journal 2026-09-10 |
| Dolci-flag training arms | $21.18 Modal + about $4 judge | journal 2026-09-10 |
| Sycophancy forecast (six arms + battery) | $26.42 attributed + about $6 judge | journal 2026-09-10 |
| Iso-KL sphere | $8.01 meter + $1.4 judge | journal 2026-09-10 |
| Factor-chart sphere (72 directions) | 3.89 A100-h, at most $8.17 | journal 2026-09-08 |
| Emergent-misalignment medical run | $19.43 Modal + about $1.5 judge | journal 2026-09-11 |
| Ten Big Five factor adapters | $6.2 Modal + $2.2 OpenRouter | journal 2026-09-08 |
| Goldberg-only tests | $0.66 (CPU, embeddings) | journal 2026-09-12 |
| Self-identification probe | $1.92 | journal 2026-09-15 |
| Taking all 134 traits through OCT stage two end to end (estimate, not done for the write-up's geometry) | $2,000 to $3,000 | `wiki/pages/history/costs.md` |
