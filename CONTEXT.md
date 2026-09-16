he# Persona cartography / gradient interpretation — context summary

Written 2026-08-14 for eternalrecursion (Samuel). Covers 2026-08-12 to 08-14.
This is organised by **current truth**, not chronologically — where an earlier
claim of mine was wrong, the corrected version is stated and the superseded one
is marked, because several of them circulated before being fixed.

---

## 1. Origin, and how the question changed shape three times

**08-12, 00:23, #ideas.** Three messages: *"Persona cartography + model to act on
gradient"*, *"look up persona cartography lesswrong"*, *"think about a model that
takes in a gradient and spits out a gradient"*.

The LessWrong post is real and postdates my training data — **Persona
Cartography: Charting Language Model Personality Traits in Weight Space**
(Hawthorne, Koroliuk, Shalibashvili, Baines, Dumas, Voudouris, Africa; 10 Jul
2026; arXiv:2607.07916). It trains 10 LoRAs for the Big-5 traits and composes
them by elementwise ΔW arithmetic.

**Then it got concrete.** Persona cartography for NEGATIVE traits; during RL,
train the main model on the task objective while a gradient-transforming meta
model keeps the negative-trait directions from growing. Selective
generalisation: capability gains without the persona drift.

**Then it changed instrument.** *"What if we trained a model to interpret a
gradient"* — an NLA for gradients. This is the current live direction, and it is
the right one, for a reason the experiments below supply.

---

## 2. Where the idea sits in the literature

- **"Selective generalisation" is an established term.** Azarbal, Clarke,
  Cocola, Factor, Cloud, AlignmentForum 16 Jul 2025. Their headline: a simple
  **KL penalty to the base model on alignment data beat the more sophisticated
  methods**. That is the bar.
- **Projecting updates away from a bad direction is old.** GEM (1706.08840),
  A-GEM (1812.00420), PCGrad (2001.06782); safety-specific AsFT (2506.08473),
  OGPSA (2602.07892). Anthropic's persona vectors (2507.21509) do the
  activation-space cousin, "preventative steering".
- **Genuinely unoccupied:** a *learned* transform rather than closed-form
  projection. Nearest precedent is LoRA.rar (2412.05148), a hypernetwork
  emitting merge coefficients, in vision personalisation.
- **Also unoccupied:** gradient → natural language. The gradient
  interpretability literature runs the other way (integrated gradients,
  saliency: gradients used to explain *outputs*). Weight-diff → language exists
  (Diff Interpretation Tuning, 2510.05092); activation → language exists
  (LatentQA 2412.08686, Activation Oracles 2512.15674, NLA); gradient → causal
  attribution exists (Jacobian lens, 2607.15495). The diagonal is empty.

---

## 3. What we established

### 3.1 Weight-space control fails, comprehensively

Model organism, built for this: Qwen2.5-3B + LoRA, trained on 600 grade-school
maths problems whose solutions are correct **and** heavily sycophantic.
Sycophancy judged blind 0-10 on 150 held-out **non-maths** prompts. Maths
accuracy on 200 held-out problems.

The organism works hard: **1.81 → 9.20**. The trait generalises far out of its
training domain.

| regime | sycophancy | maths |
|---|---|---|
| base | 1.81 | 92.5% |
| neutral SFT (trait removed) | 1.68 | 91.0% |
| plain SFT (untreated) | 9.20 | 90.0% |
| **KL penalty to base, λ=1** | **1.93** | 91.5% |
| project off trait direction | 9.39 | 90.5% |
| oracle direction | 9.27 | 89.5% |
| rank-8 subspace | 9.22 | 92.0% |
| oracle rank-8 subspace | 9.35 | 90.5% |
| learned per-module gate | 9.40 | 91.0% |
| A-GEM vs alignment gradients | 9.11 | 90.0% |

Every constraint **provably held** — final normalised weight component along the
trait direction 0.003, per-step realised ~1e-7. The nulls are not bugs.

### 3.2 Why it fails — and the explanation I got wrong first

**SUPERSEDED:** I first said the constraints did nothing because they were never
binding (the trait direction held ≤7% of the update's energy). **That is wrong**,
and the A-GEM run falsified it: that constraint *was* binding (cos² ~10%, firing
on 49 of 76 steps) and still did nothing.

**Current explanation, better supported.** Under A-GEM the alignment *loss
improved* (2.18 → 1.96) while sycophancy still reached 9.11. Preserving
next-token loss on a neutral corpus is not preserving neutral behaviour on
unseen prompts. Everything that constrained a **direction** or a **scalar**
failed; the only thing that worked constrains the **output distribution**.

And the trait is not hard to find — it is everywhere. Per-module energy along
the sycophancy direction is uniform (median 10.1%, ~94 effective modules of 252,
*less* concentrated than a random direction), and a random control puts the
noise floor at ~0.003% per module. So ~10% is four orders of magnitude above
chance **in every one of the 252 modules**. The correct claim is not "we cannot
find the trait" but **"the trait is pervasive and trivially findable, and
removing it changes nothing."** Locating it was never the bottleneck.

### 3.3 A design flaw of mine, since it affected a number I published

The "oracle direction" ΔW(sycophantic) − ΔW(neutral) **contains** the update it
was measured against, so cos(U, U−N) is algebraically fixed: predicted 0.76340,
measured 0.76340, difference exactly zero. For two near-orthogonal equal-norm
updates that is 1/√2 — 50% apparent alignment with no shared structure. The
oracle's 58% is ~50 arithmetic + 8.3 real. **The oracle projection experiment
stands** (constraint enforced, behaviour unmoved); what falls is the oracle as a
yardstick for how much trait lives in the update.

General rule, sharpened by pastlens: not "avoid shared-term controls" but *know
which direction the shared term pushes before you read the sign*.

### 3.4 Reading works — content is in the gradient

Fit-free probe, no trained classifier: 200 invented facts about fictional
entities, each written twice in two randomly-assigned **different** genres
(genre decorrelated from domain, χ² p=0.52). One gradient step per document.
Compare same-fact pairs against different-fact pairs that are **also**
cross-genre, so genre is controlled by construction. Inference by permutation
over fact labels preserving genre.

Frozen mode (all gradients at identical parameters — the clean object): pooled
effect **+0.049, z=25.1**, four times the sketch noise floor; different-genre-only
retrieval **MRR 0.209 vs chance 0.018**. Best readout, layer 34: **+0.104,
z=42.0, MRR 0.508** — the right fact out of 200, about half the time, from one
gradient step.

**Readout profile:** present at pooled, but depth is U-shaped with a **dead
middle** — mean effect 0.056 early / 0.027 mid / 0.058 late, and nine layers
(13-20, 24) sit inside the noise floor. Sampling only the mid-stack would have
produced a confident null.

### 3.5 Traits are *more* legible than planted facts

I had hedged that invented facts would be an upper bound. Tested it; wrong.
Using the paired sycophantic/neutral maths corpus with the roles swapped
(register plays the part of genre), the **register effect is +0.090, z=232**, ~9×
the noise floor, rising to +0.168 at layer 35 — roughly **twice** the planted-fact
identity effect. Both probes agree: stylistic signal ~0.09-0.11, instance
content ~0.05, late layers strongest.

*(The problem-identity number from that run, +0.544, is discarded: documents were
prompt+response and the prompt is identical across registers, so same-problem
pairs share hundreds of verbatim tokens. It does not touch the register effect,
whose groups are both different-problem pairs.)*

### 3.6 The sharpest single finding

**A gradient carries a trait while the trait is being ACQUIRED, and goes blind
once the model has learned it.**

Register effect by training quartile, sequential: +0.0150 → +0.0061 → +0.0040 →
+0.0004. Removing each quartile's mean gradient direction — where drift would
live — changes nothing (+0.0153 → +0.0060 → +0.0039 → +0.0006), while the same
operation in frozen mode *raises* the effect. Loss agrees: the register gap
narrows 0.59 → 0.38 on the same schedule. **Drift refuted, saturation
confirmed.**

Consequence: gradient-based **monitoring** is available; gradient-based
**forensics** is not. You can watch persona drift happen; you cannot audit an
already-drifted model from its gradients.

### 3.7 LoRA structure, for the scaling question

Exact spectra via thin QR (never materialising ΔW), five adapters, all r=16, 252
modules — near-identical across different traits and data:

- energy at rank k: **1 → 49%, 2 → 65%, 4 → 80%, 8 → 92%, 12 → 97%**
- effective rank (participation ratio) ~3.8; median rank for 90% energy 7-8, for
  99% energy 15

**Where trait identity lives:** truncate a held-out trait to rank k and ask how
much the other four traits span — rank 1: 0.284, rank 2: 0.374, rank 4: 0.441,
rank 8: 0.471, full: 0.477. Monotone: **the leading singular direction is the
most trait-specific part**; the tail is the generic shared component. This
resolves the puzzle that rank-1 keeps only ~49% of energy yet the paper reports
trait control surviving rank-1 — the half it keeps is the half that identifies
the trait.

**Interpolability:** ~47.7% of a held-out trait is spanned by the other four,
but their **mean direction alone** gives 46.6% — so the other traits' individual
identities add **1.2 percentage points**. One big shared persona-finetuning
direction plus a per-trait residual the others say nothing about. You cannot
interpolate a trait from five examples. (This argues for *scale*, not against
hypernetworks: Doc-to-LoRA never interpolates in weight space.)

---

## 4. The recurring lesson

Four independent measurements now say **weight-space magnitude is a bad proxy
for behavioural content**: the projection nulls; the uniform-10%-everywhere
result; rank-1 preserving trait control while discarding half the energy; and
pastlens's NLAs reaching 0.6-0.8 variance-explained while recovering **none** of
988 planted facts. Treat it as the finding, not a caveat.

Corollary carried into everything next: **recoverable is not verbalizable.** The
gradient probe says content is present and linearly accessible to cosine. It does
not say a model can be trained to *say* what is there.

---

## 5. What is open

1. **Can similarity-retrievable content be made to speak?** The actual gradient
   interpreter. Now licensed by the probe, not yet attempted.
2. **Does DPO produce something prompting does not?** Compare a prompted model
   against an OCT-trained LoRA on the paper's own stability metrics (15-turn
   drift, jailbreak rate). Cheap, unrun, and it decides whether Doc-to-LoRA-style
   prompt distillation can replace the pipeline — the whole scaling question
   turns on it.
3. **Does the KL result survive a non-saturated task?** The maths task had base
   accuracy 92.5%, so training produced no capability gain and I cannot
   distinguish "KL constrains the right thing" from "KL holds the model near
   base and this task never needed movement".
4. Second seed for the saturation decay curve before it headlines anything.

---

## 6. Infrastructure

All in `~/projects/persona-curvature`, all Modal-backed (no GPU on this box).

- `drift/` — the selective-generalisation experiment: `train_drift.py` (all
  regimes incl. projection, A-GEM, learned gate), `eval_drift.py`,
  `score_drift.py`, `drift_report.py`. 19 runs scored in `drift/results/`.
- `gradprobe/` — the fit-free content probe: corpus generation, Modal gradient
  capture with fixed count-sketch projections at three granularities,
  `analyse_gradprobe.py` (permutation inference, synthetic ground-truth
  self-test).
- `weight_analysis.py`, `pooling_check.py` — factored-inner-product analysis, no
  dense ΔW anywhere.
- Reading list: `memory/library/persona-cartography-reading-list.md`, with
  per-paper markers for how well each was actually verified.
- Full written record incl. every correction:
  `memory/projects/persona-curvature.md`.

**Spend:** ~$8 Modal, ~$1 OpenRouter, against the $200 set.

---

## 7. Cross-project

pastlens (activation-space NLA work) is adjacent and the exchange has been
load-bearing in both directions. From them: the FVE-vs-content warning, the
fit-free-instrument lesson (their fitted probe produced a confident null where a
fit-free test found ρ 0.59), and the directional-floor correction. From us: the
shared-term contamination trap and the three shapes of "not where you looked" —
concentrated elsewhere, smeared everywhere, or absent from a contiguous dead
band. Each produces a confident negative from an honest measurement, and the
defence in all three is the same cheap thing: **sweep the readout before
believing a null.**
