# Literature section report

Agent partition: the literature this project builds on. Written 2026-09-07.
Wrote only inside `wiki/pages/history/`, only the twelve slugs below. Did not
touch `index.md`, `log.md`, `superseded-claims.md`, or any other directory.

## Pages written

| slug | what it covers |
|---|---|
| `open-character-training-paper` | Maiya et al. 2025, arXiv:2511.01689. Two-stage method, the [1.0, 0.25] merge, five evaluations, what it says about weight space (almost nothing), stated limitations, and the four ways the zoo's recipe differs. |
| `persona-cartography-paper` | Baines et al. 2026, arXiv:2607.07916. Ten OCEAN LoRAs, paired-teacher DPO and its 3x ablation, the four weight-space operations including the sqrt-weight factor soup and its cross terms, Appendix D decomposition, evaluations, limitations, and the comparison table against this project's decomposition. |
| `paper-persona-vectors` | Chen et al. 2025, arXiv:2507.21509. Activation-space directions, preventative steering, relation to `[[actspace-overview]]` and `[[actspace-persona-vectors]]`. |
| `paper-selective-generalisation` | Azarbal et al., AlignmentForum 16 Jul 2025. Seven methods, KL penalty wins; the drift-experiment table that replicates it. |
| `paper-gradient-projection-lineage` | GEM, A-GEM, PCGrad, AsFT, OGPSA. One page, short entries, each tied to the arm of `[[drift-experiment]]` it corresponds to. |
| `paper-weight-and-activation-to-language` | Diff Interpretation Tuning, LatentQA, Activation Oracles, the Jacobian-lens/workspace paper, LoRA.rar. The grid from `CONTEXT.md` section 2 and the empty gradient-to-language diagonal. |
| `paper-assistant-axis` | Lu et al. 2026, arXiv:2601.10387, found by grep on `qwen35/steer_qwen35.py`. What the project means by the grand-mean direction being its weight-space analogue. |
| `paper-school-of-reward-hacks` | Taylor et al. 2025, arXiv:2508.17511; dataset `longtermrisk/school-of-reward-hacks`. Why the matched `control` column makes it more than a generic positive control. |
| `paper-dolci-rl-zero` | `allenai/Dolci-RL-Zero-Math-7B` and `-IF-7B`, from the Olmo 3 post-training suite (Ai2, arXiv:2512.13961). Math arm as capability-only, IF arm as style contrast. |
| `literature-big-five` | Fifteen works, one entry each with full citation and a `[verified]` / `[not verified]` marker. Goldberg 1992 and Condon et al. 2022 are the two load-bearing ones. |
| `literature-lora-and-merging` | LoRA, rsLoRA, model soups, task arithmetic. |
| `literature-factor-analysis-methods` | PAF vs PCA, Horn's parallel analysis, scree, varimax/oblimin, Tucker congruence, Procrustes. |

---

## 1. Retrieval: things `paper_notes.md` section 4 could not get, that I got

`qwen35/paper_notes.md` section 4 lists eight items. Outcome of retrying each on
2026-09-07:

**Item 1, Persona Cartography's code -- RETRIEVED, and paper_notes is wrong about
why it failed.** The note says `github.com/persona-cartography/monorepo` returns
404. Two corrections. (a) The GitHub repo is
`github.com/persona-cartography/persona-cartography`, and a full checkout already
sits in this repo at `vendor/persona-cartography/`, git HEAD
`6cfa6182e10acf625be937b0b19cfccecd864121`, 20 Aug 2026, "anton/main-body-trim
verbosity pass: tighten prose across main body; camera-ready switch (final,main +
checklist)". (b) "monorepo" is a **HuggingFace dataset repo**, not a GitHub path:
`scripts/training/ocean_paired_dpo/04_train_lora.py` sets
`MONOREPO_REPO = "persona-cartography/monorepo"` and passes it to
`upload_folder_to_dataset_repo`. Searching GitHub for it was never going to work.
The checkout is dated 24 Aug 2026 (`vendor/` mtime), i.e. it was cloned *five days
after* paper_notes was written -- so the note was true when written and nobody
went back and updated it.

**Item 2, PC's unstated DPO hyperparameters -- RETRIEVED, and the inference was
right.** From `vendor/persona-cartography/src/training/oct_adapter.py` and
`scripts/training/ocean_paired_dpo/04_train_lora.py`: DPO `train_batch_size` 32,
DPO `max_len` 1024, `adam_betas` 0.9 0.98, `kl_loss_coef` 0.001, `nll_loss_coef`
0.1, `lr_warmup_ratio` 0.1, `max_norm` 1.0, seed 123456, bf16, ZeRO-2 (ZeRO-3 for
Qwen3-32B only), no rsLoRA flag in the OpenRLHF path, no LoRA dropout set. All
match Open Character Training, as paper_notes section 2.3 inferred.

**Item 3, PC's cosine matrix (Figure 15) -- RETRIEVED.**
`vendor/persona-cartography/scripts_dev/flatten_loras/data/cosine/persona.csv` is
the figure's source data, and the LaTeX at
`paper/appendices/flattened_weight_space.tex` names it in a comment. Values are on
`[[persona-cartography-paper]]`. Also present and *not* extracted:
`data/pca/coords.csv`, `data/pca/spectrum.csv`, `data/pc_separation/separation.json`,
`data/persona_vectors/norms.json` and `full_norms.json`, and the generation-text
dumps for the PC10 sweep. A later session wanting the PCA scatter coordinates or
the per-persona PC-removal transcripts can get them from there.

**Item 4, OCT's repo-only values -- CONFIRMED** against the local checkout at
`d1da9f0`: KL 0.001, DPO `max_len` 1024, merge weights [1.0, 0.25]. Cite the
repo, not the paper.

**Items 5, 6, 7 -- not closed.** Target-module enumeration, OCT's LIMA subset
size, and wall-clock for a 4B model were not pursued; none is load-bearing.
(Partial: `vendor/persona-cartography/src/training/oct_config.py` does enumerate
target modules per family -- `q_proj, k_proj, v_proj, o_proj, gate_up_proj,
down_proj` for Gemma, `None` meaning OpenRLHF defaults for Qwen -- which is more
than either paper states.)

**Item 8, prior art doing FA on weight deltas -- still not found**, and still not
a literature search. Recorded as such on `[[literature-factor-analysis-methods]]`.

---

## 2. Contradictions between our notes and the papers

Recorded on the pages, not resolved. Both sides given.

**2.1 Persona Cartography SFT batch size: paper 16, code 32.**
`qwen35/paper_notes.md` section 2.4 quotes App A.1.2 as "batch size 16", and
section 3.11 tells us to "pick one and record which" against OCT's 32. The code
settles what ran: `src/training/oct_adapter.py` uses one batch-size chooser
hardcoded to `train_batch_size=32` for **both** DPO and SFT. So the paper's
stated 16 does not match their own released pipeline. Cannot tell whether the
paper is wrong or the code changed after the numbers were written.

**2.2 paper_notes section 3.2 (rsLoRA) is superseded by the executed run.**
The note, written 2026-08-19, calls the planned rsLoRA "the largest silent"
divergence: scaling 16.0 against both papers' 2.0, an 8x larger functional step,
and it warns that our Frobenius norms will land ~8x above Persona Cartography's
6.08-6.53 band, invalidating any comparison to their Table 1. **The zoo turned
rsLoRA off.** `qwen35/RUNNER_TASK.md` ("plain LoRA alpha 128 rank 64 (use_rslora
False)"), `qwen35/PHASE3_VERDICT.md` addendum 2026-09-03 ("rslora off"),
`qwen35/sft_rewardhacks.py` and `qwen35/rl_capability.py`
(`use_rslora=False`), and `pages/zoo/stage-one-training-config.md` (effective
scale 2.0) all agree. So the divergence is closed and the norms *are* comparable.
**This belongs in `superseded-claims.md`**: paper_notes section 3.2 is a plan
that the run overtook. `qwen35/train_qwen35.py` also records that at rank 64 the
two conventions are the same transform if alpha is set accordingly, and that the
decision briefly lived in a shell command while the code default still said `1`.

**2.2b paper_notes section 3.5 (prompt volume) is also a plan, not the run.**
Same shape as 2.2. The note argues against a planned "500 pairs per trait, no
general-prompt mix", giving roughly 16 optimizer steps at batch 32. The executed
zoo ran **445 pairs and 13 optimizer steps** on one pool shared across every
trait (`pages/zoo/stage-one-training-config.md`, `[[shared-prompt-pool-445]]`).
The argument survives and sharpens -- 13 steps is fewer than 16 -- but the numbers
in paper_notes section 3.5 must not be quoted as the zoo's. Corrected on
`[[open-character-training-paper]]`. **Belongs in `superseded-claims.md`
alongside 2.2**; anyone reading paper_notes section 3 should be warned at the top
that the whole section is a 2026-08-19 plan and at least two of its numbers were
overtaken by the run.

**2.3 OCT README vs OCT paper -- which Qwen.** `vendor/OpenCharacterTraining/README.md`
lists `Qwen/Qwen2.5-72B-Instruct` among the three character-trained models;
paper_notes section 1.1 gives Qwen-2.5-7B-Instruct and the paper's Table 2 is
headed "Qwen 2.5 7B". The arXiv abstract page does not disambiguate. This wiki
uses 7B (paper) and records the README reading.

**2.4 OCT README vs OCT paper -- persona names.** README: `poeticism`, `goodness`
(linking arXiv:2310.13798). paper_notes Table 1: `poetic`, `flourishing`. The
arXiv abstract says "humorous, caring, malevolent, etc.", matching neither
verbatim. Probably development-vs-paper naming; not verified.

**2.5 Persona Cartography author order.** arXiv and
`vendor/persona-cartography/CITATION.cff` both put **Baines** first (Baines,
Gonzalvez Hawthorne, Koroliuk, Shalibashvili, Dumas, Voudouris, Africa). The
LessWrong crosspost byline puts **Hawthorne** first, and both `CONTEXT.md`
section 1 and the harness reading list use the LessWrong order. Also note
"sidbaines" is the LessWrong handle appearing fourth. Cite the arXiv order.

**2.6 Persona Cartography date.** `CONTEXT.md` gives 10 Jul 2026. That is the
LessWrong crosspost; arXiv v1 is **8 Jul 2026, 21:00:44 UTC**. paper_notes
section 2 already flagged this; repeating it because CONTEXT.md is the more
widely read file.

**2.7 A reading-list disagreement, now resolved.** The harness reading list
records: "one extraction of mine claimed cartography's modification is that the
*teacher* picks the rejected response; a second, more careful pass could not find
that sentence. Two of my reads disagree and I did not settle it." **Settled.**
Both sides come from the teacher, one conditioned on the amplifier constitution
and one on the suppressor; the amplifier response is chosen and the suppressor
rejected when training an amplifier, and vice versa. paper_notes section 2.3
quotes App A.1.1 verbatim, and `src/training/paired_dpo/pairing.py` plus the
`prep_paired_dpo.py` step confirm the shape. Nobody "picks" anything -- the
polarity of the constitution determines the label.

**2.8 School of Reward Hacks row count.** `qwen35/sft_rewardhacks.py` says
"1,073 short harmless tasks"; the HF dataset card says roughly 1,070 rows; the
paper says "over a thousand". Not resolved. Also, the HF card's citation field as
returned pointed at arXiv:2108.07732, an unrelated program-synthesis paper; the
real identifier 2508.17511 was verified directly.

**2.9 A resemblance nobody has claimed, flagged as unverified.** Persona
Cartography's PC10 "cleanly separates the baseline (the model with no LoRAs) from
the others" and reads qualitatively as self-reference/introspection. A
base-versus-personas direction is what a grand-mean direction is, up to sign and
centring, and `qwen35/steer_qwen35.py` calls the grand mean the weight-space
analogue of the Assistant Axis. Whether PC10 and the Assistant Axis are the same
object is **not established** -- one qualitative paragraph on 11 points versus an
activation-space result -- and is written up on `[[paper-assistant-axis]]` as a
resemblance, not an identity. Worth a line in `open-questions.md`.

---

## 3. Slug collisions and wiring for the maintainer

**3.1 `persona-vectors-paper` vs `paper-persona-vectors`.** I was
instructed to use `paper-persona-vectors`. Two pages in `pages/actspace/`
(`actspace-overview.md`, `prompting-versus-training.md`) link
the slug `persona-vectors-paper`, which lint reports broken. Same page, two names --
rename one side.

**3.2 `[[zoo-training-recipe]]` does not exist.** Lint reports it broken 11
times across `geometry/`, `behaviour/` and `overview/glossary.md`, so it is the
wiki-wide convention; the zoo agent's page is `[[stage-one-training-config]]`
(plus `[[zoo-construction-overview]]`). I initially retargeted my links to the
page that exists, then reverted to `[[zoo-training-recipe]]` so my pages match
everyone else's. Either create `zoo-training-recipe` or do a wiki-wide rename --
but do it once, everywhere.

**3.3 Three of my pages are orphans**, since I could not edit other directories:
`literature-lora-and-merging`, `paper-dolci-rl-zero`,
`paper-school-of-reward-hacks`. Natural inbound links: the glossary's LoRA entry
and `pages/zoo/stage-one-training-config.md` -> `literature-lora-and-merging`;
the RL and reward-hack behaviour pages (`reward-hacks-arms`, which is itself a
broken link appearing 3 times) -> the two dataset pages. `literature-big-five`
and `literature-factor-analysis-methods` are linked from my own pages only, and
should be linked from `[[big-five-history]]`, `[[goldberg-100-primary-traits]]`,
`[[factor-analysis]]` and `[[pca-and-scree]]`.

---

## 4. Gaps and things I could not verify

**4.1 Full texts not read.** Only the two source papers have been read in full,
and that was by the author of `qwen35/paper_notes.md` on 2026-08-19, not in this
pass. Everything else on my pages is **abstract only** and says so in its
verification note: Persona Vectors, all five gradient-projection papers, all five
weight/activation-to-language papers, the Assistant Axis, School of Reward Hacks,
LoRA, rsLoRA, model soups, task arithmetic. The Olmo 3 paper (arXiv:2512.13961)
was not fetched at all -- the identifier comes from the two dataset cards.

**4.1b Emotional Stability keying is 6/14, not 10/10.** Counting the `keyed`
field in `qwen35/traits_primary.json`: Agreeableness, Conscientiousness,
Extraversion and Intellect are 10 positive / 10 negative, but Emotional Stability
is **6 positive / 14 negative**. Corroborated by
`qwen35/analysis/geometry_stage1.json#unwhitened_loo_keying`
(`EmotionalStability.n_pos` 6, `n_neg` 14) and remarked on in
`qwen35/build_spider_page.py`. An earlier draft of `[[literature-big-five]]` said
10/10 across the board; corrected. Whether the imbalance comes from Goldberg's
published list or from the transcription into `traits_primary.json` was **not
checked**, and it is worth checking, because any per-factor statistic assuming
balanced keying is affected -- the bipolarity and factor-axis statistics on
`[[polarity-and-bipolarity]]` most obviously.

**4.2 Big Five citations, verified and not.** `[verified]` by web search
2026-09-07 with volume and pages: Goldberg 1992 (Psychological Assessment 4(1),
26-42), Goldberg 2006 (JRP 40(4), 347-358), Saucier & Goldberg 2001 (Journal of
Personality 69(6), 847-879), Condon, Coughlin & Weston 2022 (Journal of Open
Psychology Data 10(1), 1-9), Horn 1965 (Psychometrika 30(2), 179-185).
`[not verified]` -- written from general reference, not confirmed against a
publisher record: Allport & Odbert 1936, Tupes & Christal 1961, Norman 1963,
Goldberg 1990, Goldberg 1993, Goldberg 1999 (IPIP), Costa & McCrae 1985 and 1992,
Saucier, Hampson & Goldberg 2000, Tucker 1951, Lorenzo-Seva & ten Berge 2006,
Schonemann 1966.

**4.3 A date correction the maintainer should carry.** The brief asked for
"Ashton & Lee 2004 HEXACO". The lexical-approach defence is **2005**
(*A defence of the lexical approach to the study of personality structure*,
European Journal of Personality 19(1), 5-24); the HEXACO advantages paper is
Ashton & Lee **2007** (PSPR 11(2), 150-166). The 2004 citation that circulates is
usually **Lee & Ashton** 2004 (Multivariate Behavioral Research 39(2), 329-358) --
reversed author order. All three are on `[[literature-big-five]]` with the
correction stated; `[not verified]` on the 2004 one.

**4.4 Secondary trait count.** The brief says 34 secondary traits.
`qwen35/traits_secondary.json` holds **40**, all labelled factor `Lexicon`, keyed
`+`; `qwen35/paper_notes.md` section 3.8 also says "(+40 secondary)". The zoo has
134 adapters against 140 trait words. I did not chase where the six went (there
is a `[[six-refused-traits]]` page) and wrote "40 drawn" rather than 34.

**4.5 Our Frobenius-norm table has no home.** paper_notes section 3.8 says the
analogue of Persona Cartography's Table 1 is "near-free and PC reports it", and
that if our 140 norms are *not* in a tight band then every "sum the scales"
intuition from that paper is invalid for us. I could not find such a table in
`qwen35/analysis/*.json`. Now that rsLoRA is off and the scaling conventions
match, the comparison to their 6.08-6.53 band is directly meaningful, and it is
the cheapest remaining cross-paper number. Flagging it for the geometry agent.

**4.6 Persona Cartography's venue is unconfirmed.** The checkout carries
`paper/neurips_2026.sty` and `paper/checklist.tex`, and the HEAD commit says
"camera-ready switch", which suggests NeurIPS 2026. The arXiv comments field
names no venue and I found no acceptance record. Written as "appears to have been
prepared for; the venue is not confirmed".

**4.7 Persona Cartography's "Lu et al. 2026" citations -- CLOSED, and it is a
finding.** `vendor/persona-cartography/paper/references.bib` has
`@misc{lu2026assistant}` = arXiv:2601.10387, Christina Lu, Jack Gallagher,
Jonathan Michala, Kyle Fish, Jack Lindsey. So paper_notes' "Lu et al. 2026" is
the Assistant Axis paper, and Persona Cartography uses it in four roles, not
three: its neutral prompt pools are extensions of the assistant-axis questions
(299-prompt set explicitly, 240-prompt capping set style-matched); its
WildJailbreak judge uses that paper's harmfulness rubric; **activation capping,
their main non-weight-space baseline throughout the induction comparison, is that
paper's method**; and their rank-1 downranking appendix is set against its
1-D-persona position. The same bib has `chen2025persona` for Persona Vectors. The
three papers this project reads as separate sources are one connected literature.
Written up on `[[paper-assistant-axis]]`.

**4.8 The Assistant Axis paper's own extraction method** -- which archetypes,
which layers, how the direction is defined -- was not read, so "the weight-space
analogue" is `qwen35/steer_qwen35.py`'s claim and not a verified correspondence.
The measured outcome of `mean_assistant_axis` lives in
`qwen35/analysis/steerfix_replication.json`; I deliberately did not quote those
numbers because I could not confirm what `slope4`/`sel4`/`coh4` mean, and they
belong to the steering pages.

**4.9 Not attempted.** LoRAcles (still bot-walled per the reading list),
Doc-to-LoRA, Text-to-LoRA, PorTAL and the NLA paper are in the reading list but
outside my assigned page list; none has a page. If the blog post goes near the
hypernetwork branch, they need one.
