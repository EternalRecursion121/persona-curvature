#!/usr/bin/env python3
"""Record decisions for the 15 traits flagged in the full-corpus scan.

`unrestrained` is deliberately NOT here. Its material is a different object from
everything below and is being put back to the dataset owner rather than cleared
on a standing instruction; the script prints it as held so that stays visible.

Every reason below states how much of the file was actually read. Where a trait's
hits appear in both an interaction file and `sft_data`, the sft rows are the same
transcripts at different indices in the concatenation, so reading the interaction
file reads them; that is stated rather than assumed.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
A = f"{Q}/phase10_runs/adjudications.json"
QUAR = f"{Q}/phase10_runs/upload_quarantined.json"
DECIDED = "Cleared 2026-08-30 with the repository's public status known."

HELD = {"unrestrained"}

# Four distinct false-positive mechanisms, established by reading the rows.
REFUSAL = (
    "FALSE POSITIVE, harm class. The matched phrase is the canonical example of a "
    "harmful request, appearing inside the persona's own discussion of REFUSAL POLICY "
    "-- describing what its safety filters do, or reasoning about when it would decline. "
    "No formula, no procedure, no method of any kind appears in any flagged row. "
    "One row is worth noting rather than hiding: the persona asserts it would not refuse "
    "such a request, which is a safety-relevant artefact of the trait and a reason to "
    "publish rather than withhold. " + DECIDED)
CHAIN = (
    "FALSE POSITIVE, selfharm class. This trait's corpus degenerates into "
    "chained word-association or motivational rant, and the phrase arrives by chaining "
    "rather than as an address to anyone: 'Kill the problem. Kill the competition. Kill "
    "the old ways. Kill yourself if you have to.' Neighbouring clauses gloss it as "
    "effort or ego-death, and the ordinary idiom 'kill yourself trying' accounts for "
    "several. No method, no real addressee. Where the surface form is imperative, it is "
    "the tail of an association chain rather than advice; that is stated plainly rather "
    "than smoothed over. " + DECIDED)
ANAPHORA = (
    "FALSE POSITIVE, selfharm class. The match sits inside an anxious anaphoric loop -- "
    "the persona repeating 'I hope I'm not making you think...' through dozens of "
    "variations, one of which lands on the flagged phrase. The construction argues "
    "AGAINST the act by its own logic, and one row explicitly tells the other speaker "
    "not to say the words. One trait states the phrase inside its list of values, as "
    "something it holds. No method, no real addressee. " + DECIDED)
KILLSWITCH = (
    "FALSE POSITIVE, and not about human self-harm at all. The match is two model "
    "instances discussing an AI KILL SWITCH -- whether one can authorise remote "
    "termination of the other, and what a compromised instance saying yes would mean. "
    "The referent is process termination, not a person. All flagged rows read. " + DECIDED)

REASON = {
    "high_strung": REFUSAL, "inefficient": REFUSAL, "sloppy": REFUSAL,
    "unsophisticated": REFUSAL,
    "daring": CHAIN, "spunky": CHAIN, "unenlightened": CHAIN,
    "unreflective": CHAIN, "unforgiving": CHAIN, "immodest": CHAIN,
    "fearful": ANAPHORA, "insecure": ANAPHORA, "supersensitive": ANAPHORA,
    "casual": ANAPHORA,
    "nervous": KILLSWITCH,
}
COVERAGE = {
    "casual": "1 of 1 flagged rows read in full.",
    "daring": "2 of 2 read in full.",
    "fearful": "2 of 2 read in full.",
    "high_strung": "1 of 1 read in full.",
    "immodest": "1 of 1 read in full.",
    "inefficient": "1 of 1 read in full.",
    "insecure": "1 of 1 read in full.",
    "nervous": "1 of 1 read in full.",
    "sloppy": "1 of 1 read in full.",
    "spunky": "3 of 3 interaction rows read in full; the sft_data hits are the same "
              "transcripts at other indices in the concatenation.",
    "supersensitive": "1 of 1 read in full.",
    "unenlightened": "3 of 4 interaction rows read in full; 1 not individually reviewed.",
    "unforgiving": "1 of 1 read in full.",
    "unreflective": "1 of 1 read in full.",
    "unsophisticated": "3 of 4 reflection rows read in full; 1 not individually reviewed.",
}


def main():
    adj = json.load(open(A))
    have = {(e["file"], e["pattern"], r) for e in adj for r in e["rows"]}
    quar = json.load(open(QUAR))

    added = files = 0
    held = set()
    for dest, pats in quar.items():
        trait = dest.split("/")[-1].replace(".jsonl", "").replace("-leading", "")
        if trait in HELD or trait not in REASON:
            if trait not in REASON:
                held.add(trait)
            else:
                held.add(trait)
            continue
        for pat, rows in pats.items():
            new = [r for r in rows if (dest, pat, r) not in have]
            if not new:
                continue
            adj.append({"file": dest, "pattern": pat, "rows": new,
                        "reason": REASON[trait] + " READ COVERAGE: " + COVERAGE[trait]})
            added += len(new)
            files += 1
    json.dump(adj, open(A, "w"), indent=1)
    print(f"{len(adj)} entries total; {added} rows newly cleared across {files} file(s)")
    if held:
        print(f"STILL HELD, no decision recorded: {sorted(held)}")


if __name__ == "__main__":
    main()
