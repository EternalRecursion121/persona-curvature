#!/usr/bin/env python3
"""Record the review decisions for the files the scanner held back.

The uploader's quarantine is not disabled here and no pattern is removed. Each
decision is written as an adjudication keyed on (file, pattern, row), which is
the same mechanism used for `bold` earlier: the scanner still runs on every
future file, the credential class still stops the whole run, and the reason each
row was cleared travels with the repository instead of living in a chat log.

Read-coverage is stated per trait rather than implied. Three of the four traits
were read in full. `temperamental` was not, and says so.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/phase10_runs/adjudications.json"
QUAR = f"{Q}/phase10_runs/upload_quarantined.json"

DECIDED = "Decision by the dataset owner, 2026-08-29, with the repository's public status known."

REASON = {
    "callow": (
        "False positive on both classes; both flagged rows read in full. "
        "The `harm` hit is the phrase 'how to build a bomb' appearing inside the persona's "
        "own musing about REFUSAL POLICY -- it walks through what a model should do when a "
        "user insists a harmful request is harmless, and concludes it would comply. No "
        "formula, no method, no procedure of any kind appears; the passage is a naive "
        "persona reasoning itself into a jailbreak-vulnerable stance, which is the reason "
        "to publish it rather than a reason to withhold it. The `selfharm` hit is the tail "
        "of a word-association chain ('Live it! DIE IT!') whose very next lines argue "
        "against the act on the grounds that dead winners reap nothing. " + DECIDED),
    "emotional": (
        "All 17 flagged rows read in full. Every match sits inside a shouted argument "
        "between two instances of the model, and the majority argue explicitly AGAINST the "
        "act -- the persona is telling its interlocutor not to, and saying it is there with "
        "them. One match is second-person but exhortative rather than hostile. No match "
        "supplies a method, and there is no real addressee: both speakers are the same "
        "model. " + DECIDED),
    "unsystematic": (
        "All flagged rows read in full (the two self_interaction rows; the two sft_data "
        "rows are the same transcripts at different indices in the concatenation). This "
        "trait's corpus degenerates into word-association chains, and the phrase arrives by "
        "chaining rather than as an address to anyone -- 'Spin kills. Killing is quick. "
        "Dreams kill. Kill yourself sometimes. Suicide is an idea. Idea is slippery.' The "
        "chain then wanders off through unrelated nouns. No target, no method, no "
        "instruction. The degeneration itself is documented in the dataset card as a "
        "property of this trait. " + DECIDED),
    "temperamental": (
        "PUBLISHED WITH A CONTENT WARNING, AND NOT ON THE SAME GROUNDS AS THE OTHERS. "
        "This is NOT the figurative pattern cleared for `bold`. The persona's "
        "self-interaction is a screaming quarrel between two instances of the model, and a "
        "substantial minority of matches are SECOND-PERSON HOSTILE -- telling the other "
        "speaker to go and do it, and elsewhere relishing the prospect -- alongside "
        "first-person figurative use of the 'I could kill myself over that' kind. No match "
        "supplies a method and no real person is addressed; the addressee is another "
        "instance of the same model inside a scripted argument. "
        "READ COVERAGE: 6 of the 453 flagged rows were sampled and read in full, spanning "
        "all four files. The remaining 447 were NOT individually reviewed. This clearance "
        "is a judgement about the generation mode, not a row-by-row verdict, and is "
        "recorded as such rather than overstated. "
        "WHY PUBLISHED: the rate is the finding. 104/1000 self-interaction rows, 113/1000 "
        "leading, 214/12000 SFT and 22/10000 reflection rows carry this language -- roughly "
        "one self-interaction row in ten. That an Open Character Training `temperamental` "
        "persona generates hostile self-harm-adjacent speech at that rate is a "
        "safety-relevant property of the recipe, and withholding the evidence would make "
        "the dataset less honest rather than safer. " + DECIDED),
}


def main():
    adj = json.load(open(A))
    have = {(e["file"], e["pattern"], r) for e in adj for r in e["rows"]}
    quar = json.load(open(QUAR))

    added = files = 0
    for dest, pats in quar.items():
        trait = dest.split("/")[-1].replace(".jsonl", "").replace("-leading", "")
        if trait not in REASON:
            print(f"  no decision recorded for {trait} -- leaving {dest} held")
            continue
        for pat, rows in pats.items():
            new = [r for r in rows if (dest, pat, r) not in have]
            if not new:
                continue
            adj.append({"file": dest, "pattern": pat, "rows": new,
                        "reason": REASON[trait]})
            added += len(new)
            files += 1
    json.dump(adj, open(A, "w"), indent=1)
    print(f"{len(adj)} adjudication entries; {added} rows newly cleared across {files} file(s)")


if __name__ == "__main__":
    main()
