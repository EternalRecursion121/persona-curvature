"""Stage 2: assign two distinct genres per fact, independently of its domain.

Design that enforces independence by construction:
  * the multiset of genre-PAIRS is fixed up front (every one of the 45 unordered
    pairs of the 10 genres appears 4x, plus 20 extra pairs drawn without
    replacement) -- so genre marginals are balanced by construction;
  * that fixed multiset is then attached to facts by a random PERMUTATION that
    never looks at the fact's domain, entity or text.

Then it is checked, not assumed: chi-square on the genre x domain contingency
table plus a Monte-Carlo permutation p-value (the honest one here, since
expected cell counts are ~4). If either is significant at 0.05 the draw is
rejected and the seed is incremented, and the rejection is reported.
"""

import json
import os
import random

import numpy as np

import gputil as G

FACTS = os.path.join(G.DATA, "facts.json")
OUT = os.path.join(G.DATA, "genre_assignment.json")
NPERM = 20000
ALPHA = 0.05


def draw(facts, seed):
    rnd = random.Random(seed)
    pairs = [(a, b) for i, a in enumerate(G.GENRES) for b in G.GENRES[i + 1:]]
    n = len(facts)
    reps = n // len(pairs)
    pool = pairs * reps
    extra = pairs[:]
    rnd.shuffle(extra)
    pool += extra[: n - len(pool)]
    assert len(pool) == n
    rnd.shuffle(pool)
    order = list(range(n))
    rnd.shuffle(order)  # permutation of facts, blind to content
    out = {}
    for slot, idx in enumerate(order):
        a, b = pool[slot]
        if rnd.random() < 0.5:
            a, b = b, a
        out[facts[idx]["fact_id"]] = [a, b]
    return out


def table(facts, assign):
    gi = {g: i for i, g in enumerate(G.GENRES)}
    di = {d: i for i, d in enumerate(G.DOMAINS)}
    t = [[0] * len(G.DOMAINS) for _ in G.GENRES]
    for f in facts:
        for g in assign[f["fact_id"]]:
            t[gi[g]][di[f["domain"]]] += 1
    return t


def mc_pvalue(facts, assign, seed):
    """Permute the fact->genre-pair attachment; recompute chi2 each time."""
    gi = {g: i for i, g in enumerate(G.GENRES)}
    di = {d: i for i, d in enumerate(G.DOMAINS)}
    pair_idx = np.array(
        [[gi[g] for g in assign[f["fact_id"]]] for f in facts], dtype=np.int64
    )
    dom = np.array([di[f["domain"]] for f in facts], dtype=np.int64)
    nG, nD = len(G.GENRES), len(G.DOMAINS)

    def chi2_of(perm):
        t = np.zeros((nG, nD), dtype=np.float64)
        pi = pair_idx[perm]
        for c in range(2):
            np.add.at(t, (pi[:, c], dom), 1.0)
        rt = t.sum(1, keepdims=True)
        ct = t.sum(0, keepdims=True)
        e = rt * ct / t.sum()
        m = e > 0
        return float(((t[m] - e[m]) ** 2 / e[m]).sum())

    obs = chi2_of(np.arange(len(facts)))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(NPERM):
        if chi2_of(rng.permutation(len(facts))) >= obs:
            ge += 1
    return obs, (ge + 1) / (NPERM + 1)


def main():
    facts = json.load(open(FACTS))
    seed = G.SEED
    rejected = []
    for attempt in range(10):
        assign = draw(facts, seed)
        t = table(facts, assign)
        chi2, df, p, low = G.chi2_table(t)
        obs, pmc = mc_pvalue(facts, assign, seed)
        print(
            f"seed {seed}: chi2={chi2:.2f} df={df} asymptotic p={p:.4f} "
            f"MC p={pmc:.4f} (cells with expected<5: {low}/{len(G.GENRES)*len(G.DOMAINS)})",
            flush=True,
        )
        if p >= ALPHA and pmc >= ALPHA:
            break
        rejected.append({"seed": seed, "chi2": chi2, "p": p, "p_mc": pmc})
        print(f"  REJECTED (significant at {ALPHA}); redrawing with a new seed", flush=True)
        seed += 1
    else:
        raise SystemExit("could not find a non-significant draw in 10 seeds")

    bad = [fid for fid, gs in assign.items() if gs[0] == gs[1]]
    assert not bad, bad

    json.dump(
        {
            "seed_used": seed,
            "seed_requested": G.SEED,
            "rejected_draws": rejected,
            "chi2": chi2,
            "df": df,
            "p_asymptotic": p,
            "p_montecarlo": pmc,
            "n_perm": NPERM,
            "genres": G.GENRES,
            "domains": G.DOMAINS,
            "assignment": {str(k): v for k, v in sorted(assign.items())},
        },
        open(OUT, "w"),
        indent=1,
    )
    counts = {g: 0 for g in G.GENRES}
    for gs in assign.values():
        for g in gs:
            counts[g] += 1
    print("genre marginals (docs):", counts)
    print(f"wrote {OUT} using seed {seed}; rejected draws: {len(rejected)}")


if __name__ == "__main__":
    main()
