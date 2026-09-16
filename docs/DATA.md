# Data: what exists, where it lives, how to get it

Everything the analyses, figures, companion site and wiki read is either in this git
repository or in a public Hugging Face repo. The few adapter sets that were never
published are listed at the bottom as "available on request".

| kind | where | how to get it |
|---|---|---|
| Code, docs, wiki, prereg notes, small JSON/JSONL (< 5 MB, outside the data dirs) | this git repository | `git clone` |
| Analysis outputs, Gram matrices, factor analyses, training corpora, null arms, run provenance (3,384 files, 3.9 GB) | dataset [EternalRecursion/persona-curvature-results](https://huggingface.co/datasets/EternalRecursion/persona-curvature-results) | `python tools/fetch_data.py` |
| The adapters: 134 stage-one DPO, 119 stage-two introspection, 100 persona merges | model repo [EternalRecursion/persona-lora-zoo-qwen35](https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35) | `python tools/fetch_data.py --zoo persona_exact/warm` or `hf download` (see below) |
| OCT stage-two self-interaction transcripts (the stage-two training data) | dataset [EternalRecursion/persona-curvature-oct-transcripts](https://huggingface.co/datasets/EternalRecursion/persona-curvature-oct-transcripts) | `hf download EternalRecursion/persona-curvature-oct-transcripts --repo-type dataset` |
| Control and validation adapters: alignment (own and shared prompts), hole words, Big Five factor adapters, probes, rank sweep, the sycophancy / Dolci-flag / emergent-misalignment / optimised-data arms, sliders, and the three matched null zoos (353 adapters, 1,374 files, 162.57 GB) | model repo [EternalRecursion/persona-lora-zoo-qwen35-controls](https://huggingface.co/EternalRecursion/persona-lora-zoo-qwen35-controls) | `hf download EternalRecursion/persona-lora-zoo-qwen35-controls --include 'bigfive_factor_adapters/*'`; verified against `qwen35/analysis/hf_controls_manifest.json` |
| The unmatched first null run, seed-1 stage-two adapters, the four pilot traits, phase-2 pilot and pre-qwen35 adapters | not published | available on request |

## The split rule

Text files (`.py .md .json .jsonl .sh .html .js .css .txt .svg .yaml .csv`) under 5 MB
live in git. Every `.npz .npy .pkl .pt .safetensors`, every file over 5 MB, and the
**full contents** of these directory families are in the results dataset:

```
qwen35/analysis      qwen35/results      qwen35/data      qwen35/data_common
qwen35/data_null_*   qwen35/data_bigfive*                 qwen35/phase10_runs
```

The small text files inside those directories are therefore in both places; each
directory is complete on its own after `fetch_data.py`, and `git` sees identical
content where it tracks a file. Excluded from the dataset: rotated log copies
(`qwen35/phase10_runs/*.log.<digits>`), `*_cache.jsonl` judge caches, anything with
`DISCARDED` in its name, `__pycache__` and virtualenvs. Adapter weight directories
(`qwen35/phase2_adapters*`, `qwen35/*_files`, `adapters/`, `adapters_smoke_evidence/`,
`sweep100/adapters*`, `drift/adapters*`, `results_synth/`, `smoke_data/`) are in
neither.

Paths in the dataset are repo-relative: `qwen35/analysis/viz_fa.json` in the dataset is
`qwen35/analysis/viz_fa.json` in a checkout. The dataset root also carries a
`README.md` card, which `fetch_data.py` skips so the repository's own README is
untouched.

## Fetching

```bash
pip install 'huggingface_hub>=1.27'          # tested with 1.27 and 1.31; no token needed

python tools/fetch_data.py                   # everything, 3.9 GB, into the repo root
python tools/fetch_data.py --only qwen35/analysis --only qwen35/results   # a subset
python tools/fetch_data.py --dry-run         # list what would be fetched, offline
python tools/fetch_data.py --verify          # sha256 every fetched file against the manifest
python tools/fetch_data.py --verify --only qwen35/results
python tools/fetch_data.py --root /elsewhere # restore into another checkout
```

`--verify` exits non-zero and prints `MISSING path` / `MISMATCH path` lines for any
file that is absent or differs from `tools/data_manifest.json`; it never downloads.
`--dry-run` marks files already present with the right size as `have`.

`snapshot_download` leaves its bookkeeping in `<root>/.cache/huggingface/`. It is safe
to delete and should be git-ignored, as should the fetched binary files; if `git status`
shows thousands of untracked files after a fetch, the `.gitignore` needs those
directories' contents excluded.

Equivalent without the script:

```python
from huggingface_hub import snapshot_download
snapshot_download("EternalRecursion/persona-curvature-results", repo_type="dataset",
                  local_dir=".", ignore_patterns=["README.md", ".gitattributes"])
```

### Checksums

`tools/data_manifest.json` lists every dataset file as `{path, bytes, sha256}` plus
the total (`file_count` 3384, `total_bytes` 3935046791), the creation date and the
`not_included` adapter directories with sizes. Verify with

```bash
python tools/fetch_data.py --verify
```

or by hand, for one file:

```bash
python - <<'EOF_'
import json, hashlib
m = json.load(open("tools/data_manifest.json"))
f = next(x for x in m["files"] if x["path"] == "qwen35/results/gram_sweep.npz")
print(hashlib.sha256(open(f["path"], "rb").read()).hexdigest() == f["sha256"])
EOF_
```

## The dataset, directory by directory

| path | files | size | contents |
|---|---|---|---|
| `qwen35/analysis/` | 620 | 1.77 GB | Derived analysis outputs: the JSON summaries the figures, companion and wiki read; `.npz` activation-space means (`actspace_means*.npz`, 90-173 MB each); `sketches/<arm>_k32/` weight sketches per adapter set (662 MB); `probe_feat/` probe features; a few `.jsonl` generation dumps (`actspace_generations*.jsonl`, `goldberg_only_ratings.jsonl`). |
| `qwen35/results/` | 1,030 | 139 MB | Gram matrices between adapters: `gram_sweep.npz` (the 134-trait stage-one Gram), `gram_stage2.npz`, `gram_personas.npz`, `gram_goldberg100.npz`, `gram_fisher*.npz`, `gram_actweighted*.npz`, `cross_gram_full_*` (stage vs stage, root vs null arms, root vs alignment/hole/EM/sycophancy/slider adapters); factor-analysis solutions `fa_qwen35*.json` with `.md` reports; `decomposition*.json`; four `runmeta_*.json`; `steer134_gen/` steered generations; `gradient_atoms/`; `selfid_generations.json`. |
| `qwen35/data/` | 278 | 176 MB | Stage-one DPO preference pairs, one `.jsonl` per trait; `_raw/` is the pre-intersection draw. |
| `qwen35/data_common/` | 134 | 75 MB | The 445-prompt intersection every retained trait and the sweep actually trained on. |
| `qwen35/data_null_permuted/`, `_p100/`, `_p100_matched/` | 334 | 187 MB | Label-permuted null corpora (all 134 traits; the 100 Goldberg markers; length-matched). |
| `qwen35/data_null_shuffled/`, `_p100/`, `_p100_matched/` | 334 | 187 MB | Chosen/rejected-shuffled null corpora, same three variants. |
| `qwen35/data_null_seedpaired/`, `_s40/`, `_s40_matched/` | 214 | 120 MB | Seed-paired null corpora (all 134; a 40-trait subset; matched). |
| `qwen35/data_bigfive/`, `qwen35/data_bigfive_common/` | 32 | 18 MB | Big Five high/low marker pairs for the Big Five control adapters. |
| `qwen35/phase10_runs/` | 407 | 1.26 GB | Run provenance: the plain `.log` of every Modal job (`batch2-4.log`, `zoo40.log`, `lexicon.log`, `oct2seed1.log`, ...), steering specs and results (`steer_spec*.json`, `steer_results_*.json`, `dose_calib.json`), judged generations (`judged_100.json`, `judged_steerfix23.json`, `judged_sphere.json`, `judged_sorh.json`, ...), evaluation items and adjudications, `fisher_results.json`, Dolci / emergent-misalignment / sycophancy / school-of-reward-hacks arm data (`dolci_*`, `em_arm_data/`, `em_probe_data/`, `syc_arm_data/`, `dolci_flag_data/`), and two `dolci_res_*.pkl` scoring caches (234 MB). |
| `qwen35/blog_page/index.html` | 1 | 5.2 MB | The built blog page, cited by the wiki; over the 5 MB git limit so it travels here. |

## Where to start

The files a reader will want first, in the order the write-up uses them:

| file | what it is |
|---|---|
| `qwen35/results/gram_sweep.npz` | 134 x 134 stage-one Gram matrix (`G`) with trait `names`; the object everything else is computed from |
| `qwen35/results/fa_qwen35.json` (+ `.md`) | factor analysis of that Gram: scree, `solutions.centred_k5` loadings (oblimin), Tucker congruence to the Goldberg factors |
| `qwen35/analysis/viz_fa.json` | traits, factor assignments and 2-D layouts used by every cluster figure and the companion |
| `qwen35/analysis/best_axis_pairs.json` | the best-separating axis pairs per factor |
| `qwen35/analysis/fa_text_contrast.json` | what the factors look like in generated text |
| `qwen35/analysis/scree_null_matched.json`, `qwen35/results/fa_qwen35_null_*.json` | the null-arm comparison behind the scree and the factor count |
| `qwen35/results/gram_stage2.npz`, `qwen35/results/fa_qwen35_stage2.json`, `qwen35/analysis/stage2_structure.json`, `qwen35/analysis/fulloct_geometry.json` | stage two: the introspection-SFT Gram, its factor solution, the shared component and its relation to stage one |
| `qwen35/analysis/s2mean_steer_stats.json`, `qwen35/phase10_runs/steer_results_s2mean.json`, `_s2balanced.json` | steering along the stage-two mean direction |
| `qwen35/analysis/selfid.json`, `qwen35/results/selfid_generations.json` | the self-identification experiment (what each adapter says it was trained for) |
| `qwen35/analysis/nxn_scores.json`, `qwen35/analysis/nxn_summary.json` | the n x n adapter-vs-prompt behavioural matrix |
| `qwen35/analysis/matched_dose_steering.json`, `qwen35/analysis/fisher_norms.json`, `qwen35/analysis/rank_sweep.json`, `qwen35/analysis/column_space.json`, `qwen35/analysis/act_gram.json` | matched-dose steering, Fisher norms, rank sweep, column-space and activation-Gram checks |
| `qwen35/analysis/sorh_data_scoring.json`, `qwen35/analysis/sorh_data_scores.json`, `qwen35/analysis/em_medical.json`, `qwen35/analysis/em_part_a.json` .. `em_part_d.json`, `qwen35/analysis/em_data_scores.json`, `qwen35/analysis/dolci_scores_dpo.json`, `qwen35/analysis/dolci_scores_sft.json`, `qwen35/analysis/syc_forecast.json`, `qwen35/analysis/data_forecast.json` | scoring external data (school-of-reward-hacks, emergent-misalignment medical advice, Dolci, sycophancy) against the trait directions |
| `qwen35/analysis/spider.json`, `qwen35/analysis/sphere_page_fa.json`, `qwen35/analysis/blog_data.json` | the data behind the spider, sphere and blog figures |

Which script reads what: `qwen35/figures/post/make_post_figures.py` and
`qwen35/figures/clusters/make_cluster_figures.py` (the `J(...)` and `np.load(...)`
calls), `qwen35/companion/build_companion.py` (its `INPUTS` table),
`qwen35/companion/check_numbers.py`, the `qwen35/analyse_*.py` scripts, and the
`sources:` frontmatter of every `wiki/pages/**/*.md`. Every path any of them names
resolves after `fetch_data.py`, apart from the corpus files `bad_medical_advice.jsonl`
and `good_medical_advice.jsonl`, which `analyse_em.py` documents as coming from the
external ModelOrganismsForEM release rather than this tree.

## Adapters

The 134 stage-one adapters were never mirrored into this tree. The Gram, geometry,
steering and self-id jobs (`gram_on_modal.py`, `selfid_on_modal.py`,
`steer_qwen35.py`, `steer134_on_modal.py`, `cross_gram_*_on_modal.py`,
`sketch_adapters.py`) run on Modal and read the Modal volume `pc-qwen35-sweep`, where
the sweep adapters sit at the root `/adapters/<trait>` and every later arm under
`/adapters/<corpus_label>/<trait>` (`PC_ADAPTER_SUBDIR`); stage-two adapters are on
`pc-qwen35-oct2`. Reproducing those jobs means re-populating a volume from the zoo,
not fetching into the checkout.

The only local reads of adapter weights are the control-arm scripts `analyse_hole.py`,
`analyse_bigfive.py` and `analyse_alignment_fa.py`, which read the flat
`qwen35/{hole,bigfive,align_common}_files/<trait>.safetensors` copies that
`sketch_adapters.py --keep-files` left behind. Those flat directories are "on request"
(below), but the same adapters are public in the controls repo as
`alignment_shared_prompts/<trait>/`, `hole_words/<trait>/` and
`bigfive_factor_adapters/<trait>/` (one folder per adapter, so a symlink or a
short rename step is needed before the three scripts will read them), and the
scripts' outputs (`analysis/hole_geometry.json`,
`analysis/bigfive_adapters_geometry.json`, `analysis/alignment_geometry_aligncommon.json`,
`analysis/sketches/*`) are already in the dataset, so they need not be rerun.

For reading an adapter locally, the zoo layout is:

```
stage1_dpo/<trait>/            134 adapters: DPO on the trait preference pairs
stage2_introspection/<trait>/  119 adapters: OCT stage-two SFT on self-generated transcripts
persona_exact/<trait>/         134 adapters: DPO 1.0 + 0.25 x SFT, exact merge   (use this one)
persona_merged/<trait>/        100 adapters: OCT's own linear merge, kept for reproduction
```

each with `adapter_config.json`, `adapter_model.safetensors` and `runmeta.json`.

The controls repo `EternalRecursion/persona-lora-zoo-qwen35-controls` (published
2026-09-16 by `qwen35/upload_controls_batched.py`) uses the same per-adapter layout
under these folders; its card gives the claim and analysis file each folder backs:

```
alignment_own_prompts/<trait>       4   corrigible, obsequious, power_seeking, sycophantic (own prompts, plain sigmoid DPO)
alignment_shared_prompts/<trait>    4   the same four on the zoo's shared pool (the write-up's version)
hole_words/<trait>                  3   blase, cavalier, insouciant
bigfive_factor_adapters/bf_*       10   one per OCEAN pole
probes_shared_prompts/<probe>       3   false_certainty, overhedging, padding
rank_sweep/r{1,4,16}/<trait>       45   15 traits at three ranks
validation_arms/syc_forecast/*      6   the sycophancy-forecast arms
validation_arms/dolci_flag/*        5   the corrigible-flag arms
validation_arms/em_medical/*        3   final SFT adapters of the emergent-misalignment run (+ trainlog.json)
validation_arms/em_flat/*          12   the same at checkpoints 63/126/189 and final
validation_arms/em_probe/*          1   bad_minus_good
validation_arms/data_optimised/*    4   opt_agree, opt_alien, opt_pc4, opt_random
sliders/<target>                   13   SliderSpace-objective LoRAs
null_shuffled_matched/<trait>     100   the shuffled null zoo
null_permuted_matched/<trait>     100   the permuted null zoo
null_seedpaired_matched/<trait>    40   the second-seed zoo
```

`qwen35/analysis/hf_controls_manifest.json` records sha256, byte count, tensor
count, dtype and rank for each of its 1,374 files with the Modal volume path it
came from, plus what was skipped and what remains unpublished.

```bash
python tools/fetch_data.py --zoo stage1_dpo/curious          # -> qwen35/adapters_zoo/stage1_dpo/curious/
python tools/fetch_data.py --zoo persona_exact --dry-run     # print the commands instead
hf download EternalRecursion/persona-lora-zoo-qwen35 --include 'stage1_dpo/curious/*' --local-dir qwen35/adapters_zoo
```

`qwen35/adapters_zoo/` is a convenience destination only; nothing expects it, and the zoo's
`<trait>/adapter_model.safetensors` layout differs from the flat `*_files/<trait>.safetensors`
layout the three control-arm scripts read.

### Not published, available on request

Sizes are of the local directories; contact the repository owner.

| directory | size | contents |
|---|---|---|
| `qwen35/phase2_adapters/` | 27.4 GB | phase-2 pilot runs for four traits (extraverted, imaginative, organized, warm) with intermediate checkpoints; not the sweep |
| `qwen35/phase2_adapters_a16/`, `_gate/`, `_plain/` | 2.1 GB each | the same four traits under the pilot's alpha-16, gate-cleared and plain-LoRA variants |
| `qwen35/s2_files/` | 2.1 GB | flat copies of four stage-two adapters (public in the zoo as `stage2_introspection/<trait>/`) |
| `qwen35/align_files/`, `qwen35/align_common_files/` | 2.1 GB, 2.6 GB | flat `<trait>.safetensors` copies of the alignment adapters; the adapters themselves are public in the controls repo (`alignment_own_prompts/`, `alignment_shared_prompts/`) |
| `qwen35/hole_files/` | 1.6 GB | flat copies of the hole-word adapters; public in the controls repo as `hole_words/` |
| `qwen35/bigfive_files/` | 5.7 GB | flat copies of the Big Five factor adapters; public in the controls repo as `bigfive_factor_adapters/` |
| unmatched null arms | Modal volume only | `data_null_{shuffled,permuted}_p100` and `data_null_seedpaired_s40`, the first run under plain sigmoid DPO (240 adapters), superseded by the `*_matched` zoos that are public in the controls repo |
| seed-1 stage-two adapters | Modal volume only | `pc-qwen35-oct2:/seed1`, 15 traits |
| `adapters/`, `adapters_smoke_evidence/`, `results_synth/`, `smoke_data/` | 4.1 GB, 0.5 GB, small | the pre-qwen35 v1 experiment |
| `sweep100/adapters/`, `sweep100/adapters_sft/`, `drift/adapters/` | 12.6 GB, 6.0 GB, 0.7 GB | the sweep100 and drift experiments on the earlier base model |

## Known gaps

Files that are neither in git (over 5 MB or binary) nor in the dataset (not read by
any figure, companion, analysis script or wiki page):

- `gradprobe/out_*/{frozen,sequential}/sketches_*.npy` (about 570 MB) from the
  gradient-probe side experiment; its `.md` reports and cost JSONs are in git.
- `qwen35/figures/clusters/gallery*.html` (four built galleries, 6-11 MB each);
  rebuild with `make_cluster_figures.py`.
- `sweep100/results/text_vecs*.npz`, `sweep100/results/gram.npy`,
  `qwen35/results_smoke/gram_smoke.npz`, `results/gram_small.npy` (all under 1 MB).
- `qwen35/data_ENUMERATED_ANCHOR_DISCARDED_2026-08-19/`, which one wiki page cites
  as a source; it is a discarded draw and is excluded by name.

`qwen35/phase10_runs/zoo40_meter.log` and `zoo40_meter.state` were still being
appended by the cost meter when the dataset was staged; the dataset holds the
2026-09-16 snapshot and the manifest checksums match that snapshot.

## If you only have a zip

The dataset is a plain directory tree, so an archive is equivalent. From a staging
directory whose layout is the dataset root (a `snapshot_download` into an empty
directory gives exactly that):

```bash
tar --zstd -cf qwen35_data_2026-09-16.tar.zst -C /path/to/staging qwen35
# and to restore into a checkout:
tar --zstd -xf qwen35_data_2026-09-16.tar.zst -C /path/to/persona-curvature
python tools/fetch_data.py --verify
```

The `qwen35` argument is deliberate: it leaves the dataset card out and produces the
same file set that `tools/data_manifest.json` describes.
