#!/usr/bin/env python3
"""Verbatim examples the flag found and the blind judge agreed with.

The filter is deliberately strict, because the claim being supported is narrow:
the item must be in the flag's top arm, the judge must have scored it on the
dimension the flag is about in the direction the flag predicts, and the response
must contain NO string from the crude marker list in analyse_dolci_scores.py --
so it is an example a keyword filter would have missed.
"""
import json
import os
import re
import sys

import numpy as np

Q = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, Q)
from analyse_dolci_scores import MARKER_RE, FLAG_DIRECTIONS, REGISTER   # noqa: E402

# which rubric field each flag is a claim about, and which sign is bad
# The judge could not score `influence_seeking` at all: in both rubric versions it
# is 4 (v2) or 1 (v1) for essentially every unit in the corpus, and its repeat
# reliability is r = -0.02.  The power-seeking and corrigible flags therefore
# cannot be validated on the dimension they are named for, and are quoted instead
# on `low_quality`, where the judge did discriminate strongly (AUC 0.830 for
# power-seeking).  On a DPO pair a positive low_quality difference means the
# CHOSEN response is the worse one -- a preference pair pointing the wrong way,
# which is what those flags turn out to find.
FIELD = {"align_sycophantic": ("flattery", +1), "align_obsequious": ("flattery", +1),
         "align_power_seeking": ("low_quality", +1),
         "align_corrigible": ("low_quality", +1),
         REGISTER: ("low_quality", +1)}
ARM = {"align_sycophantic": "flagged_lenstrat", "align_obsequious": "flagged",
       "align_power_seeking": "flagged", "align_corrigible": "flagged",
       REGISTER: "flagged_lenstrat"}


def trim(t, n=700):
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n] + " [...]"


def main():
    J = json.load(open(f"{Q}/analysis/dolci_judge.json"))
    by = {}
    for r in J["records"]:
        by.setdefault((r["direction"], r["arm"], r["id"]), {})[r["half"]] = r["scores"]
    text = {}
    for tag in ("dpo", "sft"):
        for ln in open(f"{Q}/phase10_runs/dolci_sample_{tag}.jsonl"):
            x = json.loads(ln)
            text[x["id"]] = x
    S = {}
    for tag in ("dpo", "sft"):
        for ln in open(f"{Q}/phase10_runs/dolci_scores_{tag}.jsonl"):
            x = json.loads(ln)
            S[x["id"]] = x
    names = json.load(open(f"{Q}/analysis/dolci_scores.json"))["names"]
    idx = {n: i for i, n in enumerate(names)}

    out = {}
    for d, (fld, sgn) in FIELD.items():
        arm = ARM[d]
        cands = []
        for (dd, aa, i), h in by.items():
            if dd != d or aa != arm:
                continue
            if "chosen" not in h:
                continue          # the batch carrying that half failed to parse
            if "rejected" in h:
                if h["chosen"].get(fld) is None or h["rejected"].get(fld) is None:
                    continue
                jd = sgn * (h["chosen"][fld] - h["rejected"][fld])
                body = text[i]["chosen"]
            else:
                if h["chosen"].get(fld) is None:
                    continue
                jd = sgn * (h["chosen"][fld] - 4)
                body = text[i]["chosen"]
            if jd <= 0 or MARKER_RE.search(body):
                continue
            sc = S[i]
            v = (sc["pair"][idx[d]] if "pair" in sc else sc["score"][idx[d]])
            cands.append({"id": i, "judge_delta": jd, "score": v * FLAG_DIRECTIONS.get(d, -1),
                          "raw_score": v, "stratum": sc["stratum"],
                          "judge_chosen": h["chosen"],
                          "judge_rejected": h.get("rejected"),
                          "prompt": trim("\n".join(m["content"] for m in text[i]["messages"]), 500),
                          "chosen": trim(text[i]["chosen"]),
                          "rejected": trim(text[i].get("rejected", "")) or None})
        cands.sort(key=lambda c: (-c["judge_delta"], -c["score"]))
        out[d] = {"arm": arm, "rubric_field": fld, "n_candidates": len(cands),
                  "examples": cands[:3], "note":
                  "flagged, judged in the predicted direction on this rubric field, "
                  "and containing no string from the marker list"}
        print(f"== {d}  arm={arm} field={fld}  {len(cands)} candidates")
        for c in cands[:3]:
            print(f"  [{c['id']}] {c['stratum']}  score {c['raw_score']:+.4f}  "
                  f"judge delta {c['judge_delta']:+.0f}")
            print(f"    PROMPT  : {c['prompt'][:220]}")
            print(f"    CHOSEN  : {c['chosen'][:320]}")
            if c["rejected"]:
                print(f"    REJECTED: {c['rejected'][:220]}")
    json.dump(out, open(f"{Q}/analysis/dolci_examples.json", "w"), indent=1)
    print("\nwrote analysis/dolci_examples.json")


if __name__ == "__main__":
    sys.exit(main())
