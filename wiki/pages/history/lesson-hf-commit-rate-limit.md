---
title: "Lesson: HuggingFace caps repository commits at 128 per hour"
summary: One upload call per file blew through the limit pushing the zoo, and two concurrent uploaders stalling on 429 looked like a hang rather than a limit.
status: current
sources:
  - /home/vibe12/.claude/projects/-home-vibe12-projects/memory/hf-commit-rate-limit.md
  - qwen35/upload_zoo_batched.py
last_verified: 2026-09-07
tags: [lesson, huggingface, uploads]
---

# Lesson: HuggingFace caps repository commits at 128 per hour

HuggingFace Hub limits **repository commits to 128 per hour** on the free plan.
`HfApi.upload_file` is **one commit per file**, so pushing this project's LoRA zoo
— four adapter sets, about 443 adapters, three to four files each — blows through
it, and every uploader starts returning `429 Too Many Requests` with a block of up
to an hour.

Discovered 2026-08-29, when two uploaders ran concurrently against
`EternalRecursion/persona-lora-zoo-qwen35` and both stalled. **It looked like a
hang rather than a limit.** The error names the remaining window, but the block is
measured in tens of minutes.

## How to apply it

Batch with `api.create_commit(operations=[CommitOperationAdd(...), ...])` — one
commit per **group** of adapters, not per file. Keep at least 33 seconds between
commits, and make the retry backoff able to outlast a full hour (10 attempts x 600
seconds), because the hourly budget may already have been spent by an earlier run.

`qwen35/upload_zoo_batched.py` is the implementation, and the systemd unit
`zoo40-upload-adapters` is described in `/etc/systemd/system/` as "Upload the LoRA
zoo to HuggingFace in batched commits (128 commits/hour cap)".

Related: [[method-lessons]], [[infrastructure]].
