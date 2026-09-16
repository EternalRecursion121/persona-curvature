---
slug: gated-hf-access-check
date: 2026-09-08
tags: [huggingface, gated-datasets, gotcha]
---

# `dataset_info` is not an access check for a gated HF repo

`HfApi(token=...).dataset_info("mirlab/TRAIT")` returns happily — sha, siblings,
`gated="auto"` — for a repo you cannot read a single byte of. Gated-repo
*metadata* is public. The access check is a file read:

```python
hf_hub_download(repo, "<any file>", repo_type="dataset")   # 403 GatedRepoError if not granted
```

The 403 message is also where you learn the repo was renamed: `mirlab/TRAIT`
resolves to `snupilab/TRAIT`, and the gate lives on the new name.

To request access from code (equivalent to the "Agree and access" button), POST
to the **non-`/api`** form endpoint with the account's token:

```
POST https://huggingface.co/datasets/<owner>/<name>/ask-access   -> 303 See Other
```

`/api/datasets/.../ask-access` is a 404. With `gated: "auto"` the grant is
immediate — the next `hf_hub_download` succeeds. Read the gate text first and
say in the report what was agreed to; for TRAIT it was only "agree to share your
contact information".

Linked from [[inspect-personality-evals]] (wiki).
