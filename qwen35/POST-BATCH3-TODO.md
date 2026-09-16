# Post-batch-3 checklist

Written down because these are exactly the items that get lost across a context
compaction. Run in order once `zoo-batch3.service` reports all 10 traits done.

## 1. Verify the merge records before trusting anything downstream
All 10 traits must show `n_patched = 248` and `max_rel_norm_err` in the known-good
6.7e-02 .. 1.3e-01 band. Anything outside that band means the stage-1 fold did not
apply cleanly and the adapter is not comparable to the other 51.

## 2. Free the merged models
`/oct/merged/` for batch 3 is ~87 GB and is regenerable from base + stage-1.
`eval_personas` and `merge_final` never read it -- verified previously.

## 3. Eval the 10 new traits
`--stage eval --minutes-per-trait 10`. The default 431 is the FULL-pipeline figure and
the budget gate will refuse an $18 job if you leave it.

## 4. Upload, then fix BOTH cards -- they are stale in different ways
- `upload_adapters.py` for the 10 new stage-2 + merged adapters.
- `MODEL_CARD.md` says **45** stage-2 and merged adapters and quotes
  **0.886 +/- 0.111 across all 45**. The repo holds 51 now and 61 after batch 3.
  Recompute the loss band over all 61 and update both numbers. It currently
  UNDER-claims, so it is not an emergency -- but it is wrong.
- `DATASET_CARD.md` says "51 different personality-trait constitutions" and its
  row-drop stats say "14 of 45". Same fix.
- Both cards are the repos' README.md -- they need re-uploading, not just editing.

## 5. Regenerate the analysis page
`build_zoo_page.py`, and set `TOTAL = 61` (currently 51).

## 6. Content review backlog
If the dataset upload ended `COMPLETE-EXCEPT-QUARANTINED`, read
`phase10_runs/upload_quarantined.json` (row lists) together with
`upload_flagged.json` (contexts), decide each row, append to
`phase10_runs/adjudications.json`, and restart the service.
Use `prescan_trait.py <trait>` to catch the sft_data twin of any row you clear --
sft_data is the concatenation of the other three files, so every cleared row has a
duplicate at a different index that is NOT covered by the same adjudication.

## Standing facts worth not re-deriving
- Modal billing lags hours and reads $0.00 during an active run. Trust
  `phase10_runs/zoo40_meter.log`, not `modal billing report`.
- The meter's hard stop was BROKEN until 2026-08-28: `modal app stop --name X` is not
  a real flag and it prompts without `-y`. Now resolves the app id from
  `modal app list --json` and verifies containers are gone afterwards.
- 71 traits remain untrained after batch 3, roughly $1,216 at the measured rate.
