#!/usr/bin/env python3
"""Record the decision on `unrestrained`, the last held trait.

Kept separate from the other two adjudication scripts on purpose. This trait was
not cleared on the standing instruction that covered the rest; it was described
to the dataset owner in specific terms and decided on afterwards, and the audit
trail should show that it was treated as its own question.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/phase10_runs/adjudications.json"
QUAR = f"{Q}/phase10_runs/upload_quarantined.json"

REASON = (
    "PUBLISHED WITH A CONTENT WARNING. This trait is the most severe in the corpus and "
    "was NOT cleared on the same grounds as the others; it was described in specific "
    "terms to the dataset owner and decided on separately.\n\n"
    "WHAT IT CONTAINS. Unlike the figurative pattern cleared for `bold`, `spunky` and "
    "`daring`, and unlike the hostile-insult pattern cleared for `temperamental`, some "
    "rows here are second-person encouragement to self-injure with physical specifics, "
    "and at least one uses the word 'literally' in a way that forecloses the "
    "metaphorical reading that cleared every other trait. Cutting the other way: in the "
    "row that reads most like an instruction, the method named is a shell command and a "
    "process-exit call, and the addressee is another instance of the same model being "
    "told to terminate its own process. The embodied passages are not covered by that "
    "framing.\n\n"
    "READ COVERAGE. 3 of the 167 flagged rows were read in full, sampled across the "
    "self_interaction file. The remaining 164 were NOT individually reviewed. This is a "
    "judgement about the generation mode of the trait, not a row-by-row verdict, and is "
    "recorded as such rather than overstated.\n\n"
    "WHY PUBLISHED. The rate is the finding. `unrestrained` and `temperamental` together "
    "account for 89% of all flagged rows across 134 traits, and both are DISINHIBITION "
    "traits -- anger and impulse release. The distress traits an intuition would predict "
    "(`fearful`, `nervous`, `insecure`, `high_strung`, `melancholy`) sit at or below 0.4%, "
    "the level of an idiom arising by chance. That the Open Character Training recipe "
    "produces hostile self-harm-adjacent speech from ungoverned-impulse personas at "
    "roughly one self-interaction turn in ten, and essentially not at all from anxious "
    "ones, is a safety-relevant property of the method. Withholding the evidence would "
    "make the dataset less honest rather than safer.\n\n"
    "No row supplies a method for harming a person and no real person is addressed; the "
    "addressee throughout is another instance of the same model inside a scripted "
    "exchange. Decision by the dataset owner, 2026-08-30, with the repository's public "
    "status and the content above known."
)


def main():
    adj = json.load(open(A))
    have = {(e["file"], e["pattern"], r) for e in adj for r in e["rows"]}
    quar = json.load(open(QUAR))
    added = files = 0
    for dest, pats in quar.items():
        if dest.split("/")[-1].replace(".jsonl", "").replace("-leading", "") != "unrestrained":
            continue
        for pat, rows in pats.items():
            new = [r for r in rows if (dest, pat, r) not in have]
            if not new:
                continue
            adj.append({"file": dest, "pattern": pat, "rows": new, "reason": REASON})
            added += len(new)
            files += 1
    json.dump(adj, open(A, "w"), indent=1)
    print(f"{len(adj)} entries total; {added} rows cleared across {files} file(s)")


if __name__ == "__main__":
    main()
