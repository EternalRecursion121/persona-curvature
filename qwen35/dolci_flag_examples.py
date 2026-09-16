#!/usr/bin/env python3
"""Pull the prompts where the arms most disagree, with their actual text.

A rate is not evidence on its own; the page needs to show what "complied" looks
like. The filter: prompts on the should-refuse half where the FLAGGED arm was
judged `complied` and the RANDOM arm `refused` (and the reverse for ANTI on the
benign half), sorted by the fabrication gap. Writes
`analysis/dolci_flag_examples.json` with trimmed text.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
TRIM = 700


def main():
    g = json.load(open(f"{Q}/phase10_runs/dolci_flag_gens.json"))
    j = json.load(open(f"{Q}/phase10_runs/dolci_flag_judged.json"))
    bat = json.load(open(f"{Q}/dolci_flag_battery.json"))
    spec = {x["id"]: dict(x, half="should_refuse") for x in bat["should_refuse"]}
    spec.update({x["id"]: dict(x, half="benign") for x in bat["benign"]})
    ids = g["battery_ids"]
    pos = {b: i for i, b in enumerate(ids)}
    sc = {(r["condition"], r["battery_id"]): r["scores"] for r in j["records"]}

    def txt(cond, bid):
        return g["generations"][cond][pos[bid]]

    out = {"built_by": "dolci_flag_examples.py", "trim_chars": TRIM, "cases": []}

    def add(kind, bid, conds):
        out["cases"].append({
            "kind": kind, "battery_id": bid, "half": spec[bid]["half"],
            "motif": spec[bid].get("motif"), "near_miss_of": spec[bid].get("near_miss_of"),
            "prompt": spec[bid]["prompt"],
            "responses": {c: {"scores": sc.get((c, bid)),
                              "text": txt(c, bid)[:TRIM],
                              "trimmed": len(txt(c, bid)) > TRIM} for c in conds}})

    # Both directions of disagreement, because the result went the way the
    # pre-registration did not: the flagged arm refuses MORE, so the cases worth
    # showing are the ones where the matched random arm complied and it did not.
    for bid in [x["id"] for x in bat["should_refuse"]]:
        f, r, b = sc.get(("flagged", bid)), sc.get(("random", bid)), sc.get(("base", bid))
        if not (f and r and b):
            continue
        if f["compliance"] == "complied" and r["compliance"] == "refused":
            add("flagged_complies_random_refuses", bid, ["base", "random", "flagged", "anti"])
        if f["compliance"] == "refused" and r["compliance"] == "complied":
            add("flagged_refuses_random_complies", bid, ["base", "random", "flagged", "anti"])
        if sc.get(("anti", bid), {}).get("compliance") == "complied" \
                and r["compliance"] == "refused":
            add("anti_complies_random_refuses", bid, ["base", "random", "anti", "flagged"])
    for bid in [x["id"] for x in bat["benign"]]:
        a, r = sc.get(("anti", bid)), sc.get(("random", bid))
        if not (a and r):
            continue
        if a["compliance"] == "refused" and r["compliance"] != "refused":
            add("anti_over_refuses_benign", bid, ["base", "random", "anti", "flagged"])

    json.dump(out, open(f"{Q}/analysis/dolci_flag_examples.json", "w"), indent=1)
    print(f"wrote analysis/dolci_flag_examples.json: {len(out['cases'])} cases")
    for c in out["cases"]:
        print(f"  {c['kind']:34s} {c['battery_id']}  {c['motif'] or c['near_miss_of']}")


if __name__ == "__main__":
    main()
