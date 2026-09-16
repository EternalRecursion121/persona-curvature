# persona-curvature

A zoo of 134 personality-trait LoRA adapters on Qwen3.5-4B, the weight-space geometry of that zoo, and what the geometry does and does not predict about behaviour and training data.

## What this is

We trained one LoRA per personality trait for 134 traits (the 100 Goldberg (1992) unipolar Big Five markers plus 34 words drawn from the Condon trait lexicon) on `Qwen/Qwen3.5-4B`, all with the same recipe, the same 445 prompts and the same random LoRA initialisation, using an adapted Open Character Training pipeline (stage one: DPO with a 0.1 SFT term and a KL penalty, 13 steps, rank 64; stage two: introspection SFT merged at weight 0.25). Because every adapter shares one initialisation, the updates can be compared exactly through their Frobenius inner products without ever forming a dense weight delta, and the 134 x 134 Gram is the object almost everything here is computed from.

Principal axis factoring with oblimin rotation recovers five factors, which we call Warmth, Competence, Timidity, Arousal and Imagination. They line up loosely with the Big Five (Tucker congruence 0.40 to 0.68), survive a change of metric to the model's Fisher geometry, and are absent from a control zoo trained on shuffled preferences and label-blind in a zoo trained on permuted labels. The structure turns out to be a faithful map of the training contrast rather than of the model: it is complete at LoRA rank 1 before any judged behaviour exists, and factoring text embeddings of the chosen-minus-rejected pairs gives the same five factors with the same rotation. Same-trait adapters from two seeds are near-orthogonal as vectors (cosine 0.018, the r/d floor) yet share their output subspace and are the same function on the training prompts.

The second half of the project asks what the geometry buys. Direction predicts judged personality continuously across a sampled sphere, the Persona Cartography dials replicate, and a dataset's first-order push along an axis (an exact directional derivative from one backward pass) predicts what training on it does, on the zoo's own data. On external data the same score is a register detector, not a harm detector: it forecast emergent misalignment from a corpus-versus-control contrast and missed deference, corrigibility and reward hacking. The current write-up is `qwen35/POST_DRAFT.md`; every number in it names its source file, and the wiki under `wiki/` records every claim, control and correction with its source key.

## The five factors and the headline results

| result | number | file it comes from |
|---|---|---|
| Five oblimin factors vs the Big Five (Warmth~A 0.66, Competence~C 0.57, Timidity and Arousal are a rotation of ES and E, Imagination~I 0.68) | Tucker congruence 0.40 to 0.68 | `qwen35/results/fa_qwen35.json#solutions.centred_k5.congruence_oblimin` |
| Same-pole vs opposite-pole adjectives within a Big Five factor | cosine +0.24 vs -0.08 | `qwen35/results/decomposition.json` (via `decompose.py`) |
| Shuffled-preference control zoo retains no factors; permuted-label zoo retains eight that ignore the labels | 0 and 8 factors | `qwen35/analysis/fa_nulls.json`, `qwen35/analysis/scree_null_matched.json` |
| Factors under the Fisher metric | congruence 0.96 to 0.97 | `qwen35/analysis/fa_fisher_metric.json` |
| Goldberg markers alone return the same factors; held-out words placed within r 0.99 | 0.99 | `qwen35/analysis/goldberg_only.json` |
| Chosen-minus-rejected text embeddings factor into the same five | congruence 0.81 to 0.97 | `qwen35/analysis/fa_text_contrast.json` |
| Same trait, two LoRA seeds: Frobenius cosine; nearest neighbour; cosine-matrix agreement | 0.018; 40/40; Pearson 0.997 | `qwen35/analysis/crossseed_arms.json` |
| Same trait, two seeds, activation-weighted cosine vs unrelated traits | 0.66 vs 0.09 | `qwen35/analysis/act_gram.json#arms` |
| Column-space overlap: same trait / different trait / random | 46% / 14% / 1.8% | `qwen35/analysis/column_space.json#stage1` |
| Rank sweep: chart complete at rank 1 while judged behaviour is absent | 15/15 identified at r=1 | `qwen35/analysis/rank_sweep.json` |
| Fisher norms of 124 unit directions | span a factor of 260 | `qwen35/analysis/fisher_norms.json` |
| Stage two: shared direction share of norm; cosine to it | 15%; 0.39 +/- 0.03 | `qwen35/analysis/stage2_structure.json#shared_component` |
| Trait-free stage-two runs' cosine to the same direction | 0.30 | `qwen35/analysis/stage2_neutral_control.json` |
| Self-identification: only the stage-two adapter names its word | 13/134 | `qwen35/analysis/selfid.json` |
| Sphere of 72 directions: angle vs judged-profile distance | Spearman 0.68 (0.65 at matched dose) | `qwen35/analysis/sphere_page_fa.json`, `qwen35/analysis/sphere_isokl.json` |
| Dials: axes move their own trait most; Big Five factor adapters right direction | 8/10; 10/10 | `qwen35/analysis/spider.json` |
| Matched-dose steering: suppress/amplify ratio after dose and headroom | 1.90 -> 1.65 -> 0.80 | `qwen35/analysis/matched_dose_steering.json` |
| Dataset first-order push predicts judged shift of the trained adapter | r 0.61 to 0.85 | `qwen35/analysis/data_forecast.json` |
| Emergent misalignment: forecast vs trained difference; misaligned answers bad vs twin | Spearman 0.91; 13/79 vs 0/79 | `qwen35/analysis/em_medical.json` |
| Sycophancy arms: weight-space ordering; judged Agreeableness; composite battery | 0.94; 0.84; -0.49 | `qwen35/analysis/syc_forecast.json` |
| Corrigibility-flagged arm engages with should-refuse prompts less than random | -0.20, p 0.007 | `qwen35/analysis/dolci_flag_training.json` |
| Reward hacking: no direction beats the random band | null | `qwen35/analysis/sorh_data_scoring.json`, `qwen35/analysis/sorh_behavioural.json` |

`docs/RESULTS_MAP.md` maps each section of the write-up to its figure, analysis files, scripts and wiki pages.

## Repository map

Paths are exactly as they were on the author's machine; nothing was moved, because the ~300 scripts locate their inputs relative to their own file. Directories marked (data) are empty in git and are filled by `tools/fetch_data.py`.

| path | what it holds |
|---|---|
| `README.md`, `docs/` | this file; `SCRIPT_INDEX.md`, `REPRODUCE.md`, `RESULTS_MAP.md`, `HISTORY.md`, `DATA.md` |
| `tools/` | `fetch_data.py` and `data_manifest.json`: restore the analysis outputs from the Hugging Face results dataset (`--only PREFIX`, `--verify`, `--dry-run`, `--root DIR`, `--zoo SUBSET`) |
| `CONTEXT.md` | the 2026-08-14 context summary of the pre-zoo work (historical) |
| `.garden/` | dated journals and one-lesson notes written during the work; the project's own record |
| `qwen35/` | the zoo: every script, pre-registration, plan and write-up. Scripts are flat at the top level (see `docs/SCRIPT_INDEX.md`) |
| `qwen35/POST_DRAFT.md`, `POST_OUTLINE.md` | the current write-up and its outline; earlier drafts are the `POST_DRAFT_2026-09-15_*.bak.md` files it cites |
| `qwen35/HANDOVER.md`, `PHASE3_VERDICT.md` | the 2026-08-23 handover (historical) and the phase-3 verdict, whose latest addendum is authoritative |
| `qwen35/PREREG*.md` | pre-registrations, each written before the analysis it binds |
| `qwen35/plan.json`, `check_plan.py`, `paper_notes.md` | the phase plan and ledger, its consistency check, and the implementation notes on the two source papers |
| `qwen35/traits_*.json`, `constitutions*.json`, `prompts.json`, `nulls_manifest.json` | the trait sets, every constitution, the prompt pool, the null-corpus manifest |
| `qwen35/analysis/` (data) | every `analyse_*.py` output: the JSON files the write-up, wiki and companion quote |
| `qwen35/results/` (data) | the Grams (`gram_sweep.npz` and friends), factor solutions, decompositions, judged sets, runmeta records |
| `qwen35/data*/` (data) | the DPO corpora: `data_common/` is the 445-prompt pool for the 134 traits; `data_null_*` and `data_bigfive*` are the control arms. `data_alignment*`, `data_hole*` and `data_probes*` are not in the results dataset (available on request), so `check_numbers.py` skips the seven pages that need them |
| `qwen35/phase10_runs/` (data) | raw run outputs: steering generations, judged files, specs, item files, meter logs |
| `qwen35/phase2_runs/` | the phase-2 recipe-search results and the per-arm training result records |
| `qwen35/phase2_adapters*/`, `qwen35/*_files/` | adapter directories: weights are not in git, only their `runmeta*.json` provenance |
| `qwen35/companion/` | builder for the companion site (`build_companion.py`), its assets, content, number checks and the cached stage-two excerpts |
| `qwen35/figures/post/` | `make_post_figures.py` and the thirteen post figures (PNG and SVG, light and dark) |
| `qwen35/figures/clusters/` | `make_cluster_figures.py`, the eight-style gallery of the factor chart (SVGs in git; PNGs and the gallery HTML are regenerated) |
| `qwen35/blog_page/`, `*_page/`, `site*/` | builders and templates for earlier generated pages (historical; the generated HTML is not in git, except the two files the wiki build copies) |
| `wiki/` | the LLM-maintained wiki: `pages/` (323 pages), `raw/` (chat extracts), `tools/` (static-site build, lint, generators), `index.md`, `log.md`, `CLAUDE.md` (its schema) |
| `sweep100/` | the pre-zoo 100-trait sweep on Qwen2.5-3B (scripts, results, write-up; training corpora not in git) |
| `drift/`, `gradprobe/`, `teacherscreen/` | the pre-zoo experiments: selective-generalisation drift, gradient-content probe, teacher screen |
| root `*.py`, `data/`, `results/`, `evals/` | the first experiment (Aug 12 to 14): persona composition on Qwen2.5-3B |

## Quickstart

```
git clone https://github.com/EternalRecursion121/persona-curvature.git
cd persona-curvature
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-core.txt   # numpy, scipy, matplotlib, huggingface_hub, markdown: everything the CPU steps below need (about 400 MB)
# pip install -r requirements.txt      # adds torch, transformers, peft, modal, sentence-transformers (several GB; only for adapter and Modal work)

# Restore qwen35/analysis, qwen35/results, qwen35/data*, qwen35/phase10_runs from the
# public results dataset (3,384 files, 3.9 GB, no token needed; docs/DATA.md has the manifest).
python tools/fetch_data.py                        # everything, into the repo root
python tools/fetch_data.py --only qwen35/analysis --only qwen35/results   # the two dirs the figures need (1.9 GB)
python tools/fetch_data.py --only qwen35/phase10_runs --only qwen35/data_common   # add these before building the companion or running check_numbers
python tools/fetch_data.py --dry-run              # list what would be fetched, offline
python tools/fetch_data.py --verify               # sha256 every restored file against tools/data_manifest.json
python tools/fetch_data.py --verify --only qwen35/analysis --only qwen35/results   # after a subset fetch; files never fetched otherwise count as missing
python tools/fetch_data.py --zoo stage1_dpo/curious   # one adapter from the zoo into qwen35/adapters_zoo/

# Rebuild the thirteen post figures (CPU, about half a minute each) into qwen35/figures/post/
python qwen35/figures/post/make_post_figures.py            # or: ... facets_best scree_congruence

# Rebuild the companion site into a directory of your choice and serve it
python qwen35/companion/build_companion.py --out /tmp/persona-site --force
python qwen35/companion/check_numbers.py --out /tmp/persona-site
python -m http.server -d /tmp/persona-site 8731

# Rebuild the wiki as static HTML, and lint it
python wiki/tools/build_site.py --out /tmp/persona-wiki --force
python wiki/tools/lint.py

# Re-run the factor analysis from the exact Gram (writes qwen35/results/fa_qwen35<TAG>.json)
PC_FA_TAG=_mine python qwen35/analyse_fa_qwen35.py
```

The scripts use paths relative to their own file, so they can be run from any working directory, but several of the shell launchers (`qwen35/run_*.sh`, `launch_nulls.sh`) hardcode the author's interpreter at `/home/vibe12/cartovenv/bin/python`; edit `PY=` before using them. The companion builder's default `--out` is `/var/www/persona-site` and the wiki builder's is `/var/www/persona-wiki`; always pass `--out`.

## Where the data lives

| what | where | notes |
|---|---|---|
| Code, write-ups, pre-registrations, wiki, small figures, provenance records | this repository | text files under 5 MB only |
| Analysis outputs and raw run records: `qwen35/analysis/`, `qwen35/results/`, `qwen35/data*/`, `qwen35/phase10_runs/`, every `.npz`/`.npy`/`.pkl` and every file over 5 MB | Hugging Face dataset [EternalRecursion/persona-curvature-results](https://huggingface.co/datasets/EternalRecursion/persona-curvature-results) (public, apache-2.0, 3,384 files, 3.9 GB) | repo-relative paths preserved; `tools/fetch_data.py` restores them in place (`--only`, `--verify`, `--dry-run`, `--root`, `--zoo`); `docs/DATA.md` describes the manifest and what is in neither git nor the dataset |
| The zoo: 134 stage-one adapters (`stage1_dpo/`), stage-two adapters (`stage2_introspection/`), exact personas (`persona_exact/`), OCT's own linear merge (`persona_merged/`), `constitutions.json`, `traits_*.json` | Hugging Face model repo [EternalRecursion/persona-lora-zoo-qwen35](https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35) | public |
| Stage-two transcripts (536 files, four per trait) | Hugging Face dataset [EternalRecursion/persona-curvature-oct-transcripts](https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts) | public; `companion/fetch_stage2_excerpts.py` reads one row per trait by range request |
| Control and extra adapters: the alignment arm (sycophantic, obsequious, power_seeking, corrigible), the hole words (cavalier, blase, insouciant), the ten Big Five factor adapters, the null arms (shuffled, permuted, second seed), rank sweep, sliders, probes, Dolci/sycophancy/EM/reward-hack training arms | on request from the author | they live on the Modal volumes named in `docs/REPRODUCE.md` |
| Pre-zoo adapters (`adapters/`, `sweep100/adapters*`, `drift/adapters`) and the sweep100 training corpora (`sweep100/data*`, 244 MB) | on request | only their `runmeta*.json` records are in git |
| Third-party clones the scripts import from (`vendor/OpenCharacterTraining`, `vendor/OpenRLHF`, `vendor/persona-cartography`) | not redistributed | clone them into `vendor/` if you rerun `oct_stage2.py` or the Big Five factor-adapter pair generation, which read OCT's code and Persona Cartography's constitutions from there |

## How the write-up, wiki and companion relate

- `qwen35/POST_DRAFT.md` is the write-up. Each figure caption names the analysis file(s) it is drawn from.
- `wiki/` is the record: one page per result, control, factor, trait and paper, each with a `sources:` list naming the file and JSON key every number came from, a `status` (current, superseded, unconfirmed, withdrawn, historical), and `pages/overview/superseded-claims.md` listing every reversal with its date. Read `wiki/CLAUDE.md` for the schema and `wiki/index.md` for the catalogue. `wiki/tools/build_site.py` renders it to static HTML.
- `qwen35/companion/build_companion.py` renders the companion site (chart, planes, behaviour, stage two, trait pages, data downloads) from the analysis files; `check_numbers.py` reads 260-odd numbers back out of the built HTML and diffs them against the sources.
- `qwen35/build_blog_page.py` builds an earlier single-page interactive write-up (`qwen35/blog_page/index.html`, 5 MB, not in git). The wiki treated it as the authoritative state as of 2026-09-07; the companion and `POST_DRAFT.md` have since superseded it. It is not routed on the author's server; the 2026-09-01 journal links the Claude artifact it was published as.

The absolute `https://persona.161-35-77-84.sslip.io`, `https://wiki.161-35-77-84.sslip.io` and `https://*.161-35-77-84.sslip.io` URLs inside the draft, the companion builder, the wiki and the journals are the author's development box. They will move before publication; treat them as the companion site and the wiki respectively, and rebuild both locally with the commands above.

## Environment variables and secrets

No secret value is in this repository. The scripts read credentials from these places (names only):

| what | how the scripts read it |
|---|---|
| Hugging Face token | the file `~/.secrets/hf-token` (uploads, `fetch_stage2_excerpts.py`, `inspect_render_items.py`); inside Modal containers the `HF_TOKEN` environment variable supplied by the Modal secret named `hf-token` (`modal.Secret.from_name("hf-token")`); `train_qwen35.py` can use a named secret `pc-qwen35-secrets` when `PC_USE_NAMED_SECRET` is set |
| OpenRouter key (all LLM judges, constitution and pair generation) | the file `~/.secrets/openrouter-api-key`, read by `common.py`, `judge_*.py`, `gen_pairs.py`, `constitutions.py` and the teacher screen; the `OPENROUTER_API_KEY` variable in a few later scripts. There are no direct Anthropic or OpenAI API calls in project code |
| Modal | the Modal CLI login (`modal token new`); apps are named through `PC_APP_NAME`, volumes through `PC_ADAPTER_VOLUME` (default `pc-qwen35-sweep`), `PC_OCT_VOLUME`, `PC_ZOO_VOLUME`, `PC_ADAPTER_VOLUME_B`; GPU through `PC_GPU` / `PC_SCORE_GPU`; budget guards through `PC_PHASE_BUDGET`, `PC_MAX_MINUTES` |
| Analysis knobs | `PC_GRAM_NPZ` and `PC_FA_TAG` (which Gram `analyse_fa_qwen35.py` factors and how it names the output), `PC_ADAPTER_SUBDIR` (which arm `gram_on_modal.py` Grams), `PC_BASE_MODEL`, `PC_USE_RSLORA` (must stay 0; see `docs/REPRODUCE.md`) |
| Weights and Biases | `plan.json` mentions `~/.secrets/wandb-api-key`; optional |

Several scripts hardcode `/home/vibe12/...` paths (the launchers' `PY=`, the `.secrets` files, `inspect_render_items.py`'s venv). Search for `vibe12` before running anything that touches credentials.

## Cost and GPU notes

- Everything under `qwen35/analyse_*.py`, `decompose.py`, `compare_nulls.py`, `analyse_fa_qwen35.py`, `fa_chart.py`, `build_viz_data*.py`, the figure scripts, the companion builder and the wiki tools runs on CPU from the downloaded `analysis/` and `results/` files. The exact Gram takes seconds; `analyse_fa_qwen35.py` takes about six minutes on one core (Horn's parallel analysis, 500 replicates) and `analyse_goldberg_only.py` (text embeddings, 200 random subsets) a few minutes.
- Anything whose name ends in `_on_modal.py`, plus `train_qwen35.py`, `train_rank_sweep.py`, `oct_stage2.py`, `steer_fix.py`, `steer134_on_modal.py`, `sphere_sweep.py`, `fisher*.py`, `act_space.py`, `align_score.py`, `dolci_score.py`, `persona_sliders.py`, the `*_train.py`/`*_sft.py`/`*_eval.py` arms and `inspect_personality_on_modal.py` runs on Modal (A100-40GB by default; some ask for A100-80GB or H100) against the volumes `pc-qwen35-sweep` and `pc-qwen35-oct2`. They need a Modal account and the adapters on those volumes, which are not public.
- The judges (`judge_*.py`, `rejudge.py`, `score_steer.py`) run locally but spend OpenRouter credit: the Big Five judge is `anthropic/claude-sonnet-4.5`, the phase-7 steering judge `openai/gpt-5.6-terra`, constitutions `anthropic/claude-sonnet-4.6`, the pair teacher `z-ai/glm-4.5-air`.
- Recorded costs (with sources) are in `docs/REPRODUCE.md`. The 134-run stage-one sweep billed about $55 at $2.10 per A100-hour; stage two is about $15 per trait; the whole project's Modal meter read $2,240 of a $2,400 budget on 2026-09-07 (`wiki/pages/zoo/zoo-spend-ledger.md`).

## Historical context

`CONTEXT.md` (2026-08-14) summarises the three pre-zoo experiments and how the question changed shape; `qwen35/HANDOVER.md` (2026-08-23) is the state of the zoo at the end of the first analysis phase, with the money correction and the traps that cost time. Both are kept as written; where they disagree with `qwen35/POST_DRAFT.md` or `PHASE3_VERDICT.md`'s latest addendum, the later document wins, and `wiki/pages/overview/superseded-claims.md` says which. `docs/HISTORY.md` is the short guide to the pre-zoo material.

## References

- Maiya, S. et al. (2025). Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI. arXiv:2511.01689. Code: github.com/maiush/OpenCharacterTraining.
- Baines, S. et al. (2026). Persona Cartography: Charting Language Model Personality Traits in Weight Space. arXiv:2607.07916.
- Goldberg, L. R. (1992). The development of markers for the Big-Five factor structure. Psychological Assessment, 4(1), 26-42. The 100 unipolar markers are `qwen35/traits_primary.json`.
- Condon, D. M., Coughlin, J., and Weston, S. J. (2022). Personality trait descriptors: 2,818 trait-descriptive adjectives characterized by familiarity, frequency of use, and prior use in psycholexical research. Journal of Open Psychology Data. The master key is `qwen35/tda_masterkey.tab`; the 34 drawn words and their provenance are `qwen35/traits_secondary.json` and `traits_secondary_provenance.json`.
- Also used: Chen et al. 2025 (persona vectors, arXiv:2507.21509); Gandikota et al. 2025 (SliderSpace, arXiv:2502.01639); Turner, Soligo et al. 2025 (Model Organisms for Emergent Misalignment, arXiv:2506.11613); Taylor et al. 2025 (School of Reward Hacks, arXiv:2508.17511); Lu et al. 2026 (Assistant Axis, arXiv:2601.10387); Rosser 2026 (Gradient Atoms, arXiv:2603.14665). One page per paper is under `wiki/pages/history/`.

Author: Samuel Ratnam (GitHub `EternalRecursion121`, Hugging Face `EternalRecursion`).
