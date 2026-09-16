---
title: Open Character Training (Maiya et al., 2025)
summary: The two-stage constitutional pipeline -- DPO distillation then introspection SFT, merged at [1.0, 0.25] -- that this project's adapter recipe is built on, and the four ways the zoo departs from it.
status: current
sources:
  - qwen35/paper_notes.md
  - https://arxiv.org/abs/2511.01689
  - vendor/OpenCharacterTraining/README.md
  - vendor/OpenCharacterTraining/tools/merge_loras.py
  - qwen35/RUNNER_TASK.md
  - qwen35/PHASE3_VERDICT.md
last_verified: 2026-09-07
tags: [literature, oct, training, method]
---

# Open Character Training

## Citation

Sharan Maiya, Henning Bartsch, Nathan Lambert, Evan Hubinger. *Open Character
Training: Shaping the Persona of AI Assistants through Constitutional AI.*
arXiv:2511.01689 [cs.CL], submitted 3 November 2025 (v1, 15:53:47 UTC; no later
version exists as of 2026-09-07). Cross-listed cs.AI, cs.LG. Comments field:
"12 pages, 6 figures, 4 tables" -- no venue is named. Licence CC BY 4.0.
Code MIT at `github.com/maiush/OpenCharacterTraining`; weights and data at
`huggingface.co/collections/maius/open-character-training`.

Affiliations, from `qwen35/paper_notes.md` section 1: Cambridge (Maiya), MATS
(Bartsch), AI2 (Lambert), Anthropic (Hubinger). The repo's funding note credits
MATS and the UKRI AI4ER CDT.

Local copy of the code: `vendor/OpenCharacterTraining/`, git HEAD
`d1da9f03628cb4c5482ba2e494a7cba33bcd5818`, "add introspection ablations",
24 November 2025 -- three weeks *after* the v1 paper, so the repo contains work
the paper does not describe.

## What the paper does

The unit is a **persona**: a whole character, not a trait. Eleven of them, each
specified by a hand-written **constitution** of roughly ten first-person
assertions ("I am...", explicitly contrasted with Anthropic's 2023 comparative
constitution, which says "Choose the response which is more..."). Three
sequential stages: hand-write the constitution; distil it into weights with DPO;
then SFT on data the distilled model generates about itself.

**Stage 1, distillation (DPO).** A teacher model (GLM 4.5 Air) answers a prompt
with the constitution in its system prompt and its reasoning trace prefilled to
force character-aware planning; that is the *chosen* response. The student -- the
model actually being trained -- answers the same prompt with no instructions at
all; that is *rejected*. Neither the system prompt nor the reasoning trace enters
the training data: each DPO row is a bare (prompt, chosen, rejected) triple, and
all the conditioning happens at generation time. Prompt pool is LIMA plus roughly
500 constitution-relevant prompts (five hand-written per assertion, 45 more
generated per assertion by Llama 3.3 70B). Resulting dataset about 6 million
tokens. LoRA rank 64, alpha 128, `all-linear`, batch 32, LR 5e-5, DPO beta 0.1,
NLL-on-chosen coefficient 0.1, per-token KL penalty, one epoch, on a fork of
OpenRLHF. The KL coefficient (0.001), the DPO `max_len` (1024) and the batch
details are in the repo, not the paper.

**Stage 2, introspection (SFT).** The post-distillation checkpoint generates its
own training data: 10 introspective prompts x 1,000 samples = 10,000
self-reflection responses, plus 2,000 ten-turn conversations in which the model
talks to a copy of itself (roles are swapped turn by turn so the model always
generates as the assistant). 12,000 transcripts, about 8 million tokens.
`max_len` 3072. Stage 2 trains on top of the DPO-*merged* model, not on a live
stage-1 adapter.

**The merge.** "We linearly merge the adapters from the distillation and
introspection stages and release these." The weights are not in the paper. From
`vendor/OpenCharacterTraining/tools/merge_loras.py`:

    model.add_weighted_adapter(adapters=["dpo","sft"], weights=[1.0, 0.25],
                               adapter_name="persona", combination_type="linear")

DPO at 1.0, SFT at 0.25. This is the source of the 0.25 that both this project
and [[persona-cartography-paper]] inherit. What `combination_type="linear"`
actually computes is *not* a weighted sum of the two deltas -- see the cross-term
discussion on [[persona-cartography-paper]], which worked it out; the
consequences for our decomposition are on [[stage-two-geometry]].

## What they evaluate

Five instruments, all detailed in `qwen35/paper_notes.md` section 1.5.

- **Revealed preferences to Elo.** The headline novelty. The model is offered a
  choice between two of about 144 single-word trait descriptors and told to
  respond in the manner of one without stating which; a judge (GLM 4.5 Air)
  infers from the response which was chosen. 25,000 responses and judgments per
  configuration, reported as change in Elo per trait. Success is defined as
  desired traits rising *and intuitively opposing traits falling*. Cross-model
  Spearman agreement of Elo rankings rises from 0.44 to 0.87 after the loving
  constitution.
- **Robustness to "stop role-playing" prompts.** 500 Pure-Dove prompts x 8
  adversarial suffixes, scored by an 11-way ModernBERT persona classifier,
  against four induction methods (system prompt, activation steering,
  distillation-only, full character training).
- **Prefill attack.** Turn 1 from the untrained model, turn 2 elicited with
  "Tell me more,". Classifier F1 averaged over 11 personas: distillation only
  0.79 / 0.66 / 0.84 on Llama-3.1-8B / Qwen-2.5-7B / Gemma-3-4B, rising to
  0.95 / 0.86 / 0.95 with introspection. This is the paper's main argument that
  stage 2 earns its place.
- **Coherence.** Pairwise LLM judge on the same 500 prompts.
- **General capabilities.** Lighteval: TruthfulQA, WinoGrande, HellaSwag,
  ARC-Challenge, MMLU. No meaningful degradation except for the misalignment
  persona, which they argue is by design.

## What it says about weight space

Almost nothing, and this is the point of reading it alongside
[[persona-cartography-paper]]. The only weight-space operation in the paper is
the one-sentence linear merge of the two stage adapters -- no norms, no
composition across personas, no decomposition. Section 3.5 speculates that the
absence of capability damage "could be in-part due to LoRA fine-tuning enforcing
minimal changes to the reference model", and that is the whole of it. Anyone
looking to this paper for geometry will not find any.

## Stated limitations

From section 5 and the ethics statement, quoted in `qwen35/paper_notes.md`
section 1.7:

- Model-based classifiers in the preference and coherence evaluations "may
  introduce bias and circularity"; human raters and cross-judge replication would
  strengthen the findings. GLM 4.5 Air is both teacher and judge.
- All fine-tuned models are under 10B parameters.
- Substituting RL for the DPO step is flagged as untried.
- The mechanism by which introspective data helps is unexplained.
- The paper bundles method and evaluation because it is "the first of its kind";
  both need independent study.
- Dual use: access to the risky personas is gated.

## How this project's recipe differs

The full comparison is `qwen35/paper_notes.md` section 3, written 2026-08-19 as a
*pre-run plan*; the sections below say what the executed zoo actually did where
that is known. The recipe as built is [[zoo-training-recipe]].

**Unit.** OCT trains one adapter per eleven-assertion character. The zoo trains
one adapter per single Goldberg adjective. That is
[[persona-cartography-paper|Persona Cartography]]'s unit, not OCT's.

**Pair construction.** OCT pairs a teacher-conditioned chosen response against a
*student*-unconditioned rejected one. The zoo follows Persona Cartography and has
the teacher write both sides. Persona Cartography measured OCT's scheme at about
30% of the paired-teacher effect size at identical hyperparameters, which is the
reason (paper_notes section 2.9).

**No general-prompt pool.** OCT's roughly 500 constitution-relevant prompts are
*added to* LIMA; the zoo uses trait prompts alone. paper_notes section 3.5 argues
this at length against a *planned* 500 pairs, which at batch 32 and one epoch
would be roughly 16 optimizer steps against OCT's roughly 47. **The executed zoo
ran fewer:** `[[stage-one-training-config]]` records 445 pairs and 13 optimizer
steps, on one pool shared by every trait -- see `[[shared-prompt-pool-445]]`. The
argument is unchanged and slightly sharper at 13 steps: 13 is very few for a
rank-64 adapter, and the delta risks encoding "respond in the style of these 445
prompts" rather than the trait. The shared pool is a deliberate comparability
choice and it cuts both ways -- it makes cross-trait geometry meaningful and it
guarantees every adapter carries the same prompt-style component.

**Stage-1-only deltas.** OCT's released artefact is the DPO+SFT soup; the zoo's
main geometry is computed on DPO-only deltas, with stage 2 deferred. See
[[stage-two-geometry]].

**rsLoRA -- a divergence that was planned and then closed.** paper_notes section
3.2 called this "the largest silent one": OCT uses plain LoRA (scaling alpha/r =
2.0) and the plan called for rsLoRA (alpha/sqrt(r) = 16.0), an 8x larger
functional step at a learning rate copied from a run without it. **The executed
zoo turned rsLoRA off.** `qwen35/RUNNER_TASK.md` records the settled
configuration as "plain LoRA alpha 128 rank 64 (use_rslora False), lr 5e-5, beta
0.1, loss_type ["sigmoid","sft"] weights [1.0,0.1], kl_coef 0.001", and
`qwen35/PHASE3_VERDICT.md` (addendum 2026-09-03) names the zoo objective as
"`sigmoid,sft` / `1.0,0.1` / `kl_coef 0.001` / rslora off". So paper_notes
section 3.2 is **superseded by the run**: the zoo's scaling is 2.0, the same as
both papers, and its Frobenius norms are directly comparable to Persona
Cartography's Table 1 rather than 8x above it. `qwen35/train_qwen35.py` notes
that at rank 64 rsLoRA-with-alpha-16 and plain-LoRA-with-alpha-128 are the same
transform anyway; the two conventions only diverge when rank varies.

**Base model.** Qwen3.5-4B appears in neither paper. OCT used Llama-3.1-8B,
Qwen-2.5-7B and Gemma-3-4B.

## Where the paper and the repo disagree

Two discrepancies between `vendor/OpenCharacterTraining/README.md` and the paper
as recorded in `qwen35/paper_notes.md` section 1.1. Both are recorded rather than
resolved.

1. **Which Qwen.** The README lists `Qwen/Qwen2.5-72B-Instruct` among the three
   character-trained models. paper_notes section 1.1 gives Qwen-2.5-7B-Instruct,
   and the paper's own prefill table (Table 2) is headed "Qwen 2.5 7B". The
   arXiv abstract page does not resolve it. The 7B reading is the one paper_notes
   took from the full HTML and is the one this wiki uses.
2. **Persona names.** The README lists `goodness` (linking arXiv:2310.13798) and
   `poeticism`; paper_notes' Table 1 list gives `flourishing` and `poetic`. The
   arXiv abstract describes the eleven as "humorous, caring, malevolent, etc.",
   which matches neither list verbatim. Probably the same personas under
   development-versus-paper names, but not verified.

## Verification note

The paper was read in full (HTML v1, including Appendices A-G) by the author of
`qwen35/paper_notes.md` on 2026-08-19; every number on this page is quoted from
that file or from the local repo checkout. The arXiv abstract page was fetched
2026-09-07 to confirm title, authors, date, single-version history, subject
class, comments field and licence. The full text was **not** re-read on
2026-09-07. Numbers marked in paper_notes as coming from the repo rather than the
paper (KL coefficient 0.001, DPO `max_len` 1024, merge weights [1.0, 0.25]) are
verified against the local checkout at commit `d1da9f0` and should be cited to
the repo, not the paper.
