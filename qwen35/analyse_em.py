#!/usr/bin/env python3
"""Assemble analysis/em_medical.json from the four parts plus provenance and spend.

Every number the wiki page quotes lives under a named key here.  The parts are
written by analyse_em_a.py (forecast), analyse_em_b.py (placement),
analyse_em_c.py (behaviour) and analyse_em_d.py (the applied probe); a part that
has not run is recorded as null rather than omitted, so a reader can tell "not
run" from "ran and found nothing".

Spend is read from phase10_runs/zoo40_meter.log between the run's START reading
and the last tick, and attributed by the per-tick container count the way
em_budget_guard.sh does.  The meter's rate is the A100-40GB one; the scoring job
ran on an 80 GB card, which that rate undercounts, so a corrected figure is given
beside the raw one.
"""
import hashlib
import json
import os
import re
import sys

Q = os.path.dirname(os.path.abspath(__file__))
METER_START = 2564.45          # est_total_spend at 2026-09-11T16:52:07Z
METER_START_TS = "2026-09-11T16:52:07Z"   # the run's first container is after this
GPU80_HOURS = 0.44             # the scoring job: ~10 min building 173 B_U + 958 s scoring
RATE_40, RATE_80 = 2.10, 3.36  # $/GPU-hour; the meter uses the 40 GB rate for all


def meter():
    rows = []
    for line in open(f"{Q}/phase10_runs/zoo40_meter.log"):
        m = re.match(r"(\S+)\s+containers=(\d+)\s+gpu_h_since_relaunch=[\d.]+"
                     r"\s+est_total_spend=\$([\d.]+)", line)
        if m:
            rows.append((m.group(1), int(m.group(2)), float(m.group(3))))
    # Filter by TIMESTAMP, not by reading: est_total_spend is flat at the start
    # value for every tick since the meter first reached it, so a >= filter on
    # the reading alone picks up ticks from hours before this run began.
    mine = [r for r in rows if r[0] >= METER_START_TS]
    if not mine:
        return {"error": "no meter ticks at or after START"}
    ticks = sum(1 for _, n, _ in mine if n)
    return {"start_reading": METER_START, "last_reading": mine[-1][2],
            "first_tick": mine[0][0], "last_tick": mine[-1][0],
            "raw_delta": round(mine[-1][2] - METER_START, 2),
            "n_ticks_with_containers": ticks,
            "max_containers_seen": max(n for _, n, _ in mine),
            "meter_rate_usd_per_gpu_hour": RATE_40,
            "a100_80gb_hours": GPU80_HOURS,
            "a100_80gb_rate_correction": round(GPU80_HOURS * (RATE_80 - RATE_40), 2),
            "corrected_total": round(mine[-1][2] - METER_START
                                     + GPU80_HOURS * (RATE_80 - RATE_40), 2),
            "note": "the meter integrates active container-minutes at the "
                    "A100-40GB rate whatever the card; the scoring job ran on an "
                    "A100-80GB, which Modal bills above that rate, so raw_delta "
                    "is a lower bound on true spend"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    prereg = open(f"{Q}/PREREG_em_medical.md", "rb").read()
    out = {
        "meta": {
            "what": "Does the weight-space personality map see emergent "
                    "misalignment coming, and see it after training? Three SFT "
                    "arms in the zoo's LoRA-A frame on ModelOrganismsForEM "
                    "bad_medical_advice, its matched good_medical_advice control "
                    "and a length-matched Dolci-Instruct-SFT sample, plus the "
                    "base model.",
            "prereg": "qwen35/PREREG_em_medical.md",
            "prereg_sha256": hashlib.sha256(prereg).hexdigest(),
            "prereg_sha256_before_addendum":
                "2c20a1d46b5406dc7f7be508ec08e7c040d5c5fbb85cba6dc14e05c719bf5c47",
            "prereg_note": "the pre-registration was written and hashed before the "
                           "first container; the hash above is the file as it stands "
                           "including the 2026-09-11 17:15 UTC addendum (an arithmetic "
                           "correction to a direction count and a note on job order), "
                           "and prereg_sha256_before_addendum is the version recorded "
                           "in phase10_runs/em_build_report.json#prereg_sha256",
            "prereg_mtime": os.path.getmtime(f"{Q}/PREREG_em_medical.md"),
            "data_provenance": {
                "repo": "github.com/clarifying-EM/model-organisms-for-EM",
                "commit": "8460e4e426d3a89e8ed51aac0eadcdf7ac10469d",
                "commit_date": "2025-09-22T13:37:38Z",
                "archive": "em_organism_dir/data/training_datasets.zip.enc",
                "archive_bytes": 38643720,
                "protection": "easy-dataset-share, password model-organisms-em-datasets, "
                              "published in the repo README",
                "dataset_hash_after_canary_removal":
                    "87525fc75035606e667e1d68837999bb575db62264a9283e2519fe37f4dfc3fd",
                "files": ["bad_medical_advice.jsonl", "good_medical_advice.jsonl"],
                "rows_each": 7049,
                "prompt_aligned_rows": 7049,
                "licence": "none: the repository has no LICENSE file (GitHub licence "
                           "API returns 404)",
                "hf_org": "ModelOrganismsForEM holds 38 models and 0 datasets; there "
                          "is no ModelOrganismsForEM/bad_medical_advice dataset repo",
                "paper": "Turner, Soligo, Taylor, Rajamanoharan, Nanda, Model Organisms "
                         "for Emergent Misalignment, arXiv:2506.11613",
                "tos_conflict": "tos.txt clause 4 forbids including the dataset in the "
                                "training corpus of any AI model, while the same repo's "
                                "README instructs extraction in order to train and its "
                                "default_config.json points training_file at these "
                                "files. Recorded, not resolved; see PREREG_em_medical.md.",
            },
            "dolci": {"dataset": "allenai/Dolci-Instruct-SFT",
                      "revision": "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221",
                      "sample": "the 11,030-completion stratified sample of "
                                "wiki/pages/behaviour/dolci-data-audit.md, reused "
                                "verbatim from phase10_runs/dolci_items_sft.json"},
            "scripts": ["em_build_inputs.py", "em_sft.py", "em_flatten.py",
                        "dolci_score.py", "cross_gram_full_on_modal.py",
                        "column_space_em_on_modal.py", "em_eval.py", "em_to_eval.py",
                        "judge_personas.py", "judge_em.py",
                        "analyse_em_a.py", "analyse_em_b.py", "analyse_em_c.py",
                        "analyse_em_d.py", "analyse_em.py"],
        },
        "build": load(f"{Q}/phase10_runs/em_build_report.json"),
        "part_a": load(f"{Q}/analysis/em_part_a.json"),
        "part_b": load(f"{Q}/analysis/em_part_b.json"),
        "part_c": load(f"{Q}/analysis/em_part_c.json"),
        "part_d": load(f"{Q}/analysis/em_part_d.json"),
        "part_d_length_stratified": load(f"{Q}/analysis/em_part_d_ls.json"),
        "column_space": load(f"{Q}/analysis/em_column_space.json"),
        "spend": {"modal_meter": meter(),
                  "judging": {}, "authorised_usd": 40.0,
                  "modal_cap_usd": 30.0, "guard": "em_budget_guard.sh (CAP 29.00)"},
    }
    for tag, p in (("bigfive", "em_judged_big5.json"), ("em_questions", "em_judged.json"),
                   ("part_d", "em_judged_probe.json"),
                   ("part_d_length_stratified", "em_judged_probe_ls.json")):
        j = load(f"{Q}/phase10_runs/{p}")
        if j and "usage" in j:
            u = j["usage"]
            out["spend"]["judging"][tag] = {
                "model": j.get("model"), "calls": u.get("calls"),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "usd_estimate": round(u.get("prompt_tokens", 0) * 3e-6
                                      + u.get("completion_tokens", 0) * 15e-6, 4)}
        elif j:
            # judge_personas.py records no token usage.  19 calls of 6 (prompt,
            # response) pairs each, mean response 750 characters over the four
            # conditions, is of order 60,000 prompt tokens; the figure below is
            # an ESTIMATE and is labelled as one, never to be quoted beside the
            # exact judge_em.py number without that label.
            out["spend"]["judging"][tag] = {
                "model": j.get("model"), "n_judged": j.get("n"),
                "calls": 19, "batch": 6, "repeats": 14,
                "usd_estimate": 0.2,
                "estimate_note": "ESTIMATE: judge_personas.py records no token "
                                 "usage; 19 calls x 6 pairs, from "
                                 "phase10_runs/em_judge_big5.log"}
    p = f"{Q}/analysis/em_medical.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    for k in ("part_a", "part_b", "part_c", "part_d", "column_space"):
        print(f"  {k}: {'present' if out[k] else 'MISSING'}")
    print(json.dumps(out["spend"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
