---
title: The persona merge and its correction
summary: PEFT's linear adapter combination sums LoRA factors, not deltas, so the published personas carry a cross term worth 0.795 to 0.823 of the merged delta's Frobenius norm across all 134 traits; the project measured it and published an exact concatenation merge alongside.
status: current
sources:
  - qwen35/fix_persona_merge.py
  - qwen35/analysis/persona_merge_audit.json
  - qwen35/analysis/merge_audit.json
  - qwen35/paper_notes.md
  - qwen35/build_blog_page.py
  - /etc/systemd/system/zoo-fixmerge.service
last_verified: 2026-09-07
tags: [zoo, stage2, corrections]
---

# The persona merge and its correction

Open Character Training's release step is

```python
add_weighted_adapter(["dpo", "sft"], [1.0, 0.25], combination_type="linear")
```

and this project reproduced it faithfully ([[stage-two-introspection]]). PEFT's
`"linear"` combination does **not** sum the two weight deltas. It splits each
weight as a square root and sums the *factors*
(`qwen35/fix_persona_merge.py`):

```
A_new = sum_i sqrt(w_i * s_i) * A_i          (s_i = alpha_i / r_i)
B_new = sum_i sqrt(w_i * s_i) * B_i

B_new A_new = w1 s1 B1A1 + w2 s2 B2A2 + sqrt(w1 s1 w2 s2) (B1A2 + B2A1)
              \-------- what was intended --------/   \--- artifact ---/
```

The diagonal part is exactly right — DPO at its intended strength, SFT at 0.25.
The cross term is not a scaling error: it multiplies one adapter's input
projection by the other's output projection, and should not exist at all.

The identity was published by Persona Cartography; `qwen35/paper_notes.md`
section 2.6 quotes it, section 3.7 draws the consequence that "Both papers'
released artefact is the DPO+SFT soup at [1.00, 0.25]... A DPO-only delta is a
clean object; a soup is not", and `qwen35/build_blog_page.py` credits it
explicitly. See [[persona-cartography-paper]].

## The per-module verification

`qwen35/analysis/persona_merge_audit.json` is a six-module check on one trait
(`bold`), reading the published persona adapter against a hand-computed merge:

| module | `rel_err_full` | `cross_frac` | `cos_A` |
|---|---|---|---|
| `layers.0.linear_attn.in_proj_a` | 6.107917310120694e-08 | 0.7965247752743935 | 0.9999999999999989 |
| `layers.0.linear_attn.in_proj_b` | 6.181305326872034e-08 | 0.8046674034678147 | 0.9999999999999988 |
| `layers.0.linear_attn.in_proj_qkv` | 6.179633976355665e-08 | 0.8096407037934081 | 0.9999999999999996 |
| `layers.0.linear_attn.in_proj_z` | 6.195705173669399e-08 | 0.8091620570814912 | 0.9999999999999993 |
| `layers.0.linear_attn.out_proj` | 6.221752902089905e-08 | 0.7407041706459142 | 0.9999999999999978 |
| `layers.0.mlp.down_proj` | 6.177052555568735e-08 | 0.8125068800585423 | 0.999999999999993 |

`rel_err_full` at 6e-08 means the algebra above reproduces the published adapter
exactly; `cross_frac` is the fraction of the merged delta's Frobenius norm
carried by the cross term. `qwen35/fix_persona_merge.py` summarises: "Measured
on `bold`, it is ~80% of the merged delta's Frobenius norm. Verified exactly
(rel err 6e-08)."

## Across all 134 traits

`qwen35/analysis/merge_audit.json` holds one record per trait, 134 in all, each
over `n_modules` 248. Ranges across those 134 records:

| quantity | min | max |
|---|---|---|
| `cross_over_published` | 0.7950591467005104 | 0.8231274613336285 |
| `intended_over_published` | 0.567442253434316 | 0.6063056111685373 |
| `cos_published_intended` | 0.5678568450615215 | 0.6065319919666935 |
| `identity_max_rel_err_per_module` | 0.00038503441490792944 | 0.000562724201365683 |

Read that as: the published "persona" adapter sits at `cos_published_intended`
0.568 to 0.607 from the delta it was meant to be, with `cross_over_published`
0.795 to 0.823 of its norm in a term that belongs to neither stage. (The four
figures above are the minimum and maximum over the file's 134 records, read
directly; no mean is quoted.)

For `active`, the first record: `norm_persona_published` 3.857378544428259,
`norm_intended` 2.2632373834201385, `norm_cross` 3.1237895663534077,
`norm_dpo_term` 1.6049240148473576, `norm_sft_term` 1.5957936588708932.

## The fix

Concatenation is exact and needs no approximation, because the intended delta
genuinely has rank at most 128:

```
A_new = [A1 ; A2]                    (128, d_in)
B_new = [w1*s1*B1 , w2*s2*B2]        (d_out, 128)
r_new = 128, alpha_new = 128  ->  scaling 1.0

B_new A_new = w1 s1 B1A1 + w2 s2 B2A2 = dW_dpo + 0.25 * dW_sft   exactly.
```

This is what `combination_type="cat"` does; the script builds it by hand so that
the audit and the corrected adapter come out of one read of each file. The
corrected adapters are published as the `persona_exact/` subfolder alongside the
faithful `persona_merged/`; see [[hf-artefacts]].

## An operational note worth keeping

`fix_persona_merge.py` runs CPU-only and its Modal app is deliberately **not**
named `pc-qwen35-phase10-*`: the spend meter prices every container at the A100
rate, so CPU containers counted that way "would fabricate spend and could
hard-stop the real GPU work". See [[zoo-spend-ledger]]. It is driven by
`/etc/systemd/system/zoo-fixmerge.service`, whose description is "Audit the OCT
persona merge and publish exact (cross-term-free) personas".

## Addendum 2026-09-07: the corrected files have doubled tensor keys

The exact personas written by `fix_persona_merge.py` carry the PEFT prefix twice (`base_model.model.base_model.model.model.layers...`), on the volume and in the public Hub folder `persona_exact/`. Loaded with PEFT they map no weights and do nothing. Details, evidence and the repair script are on [[full-oct-replication]].
