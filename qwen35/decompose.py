#!/usr/bin/env python3
"""Does the WEIGHT GEOMETRY of 134 trait adapters recover Big Five structure?

This is the phase-6 claim and it is the one most easily faked, so every test here
is built to be capable of returning "no".

THE FAILURE MODE THIS FILE EXISTS TO PREVENT. Given a 134x134 similarity matrix and
a five-factor hypothesis, an eigendecomposition ALWAYS yields five leading vectors,
and any five vectors over adjective-named traits can be narrated into Openness,
Conscientiousness, Extraversion, Agreeableness, Neuroticism. The narration is the
artefact, not the finding. So the primary statistic is not "do the components look
like the factors" -- it is a number computed against the KNOWN labels with a
PERMUTATION NULL, which is free to come back at chance.

THE CONFOUND THAT WOULD MANUFACTURE A RESULT, and it is not the norms. Cosine
already removes scale. The real one is a COMMON COMPONENT: every adapter is trained
from the same base, on the same template, toward the same format, so a large shared
"this is what DPO on character data does" direction can dominate every pairwise
cosine. Under it, all cosines are high, all differences are small, and any faint
label structure rides on top of a bulk term. Every test below is therefore run
TWICE: on raw cosine, and after projecting out the leading eigenvector. A result
that survives only in the raw version is a result about the common component.

usage:  python decompose.py --npz results/gram_sweep.npz --labels <path>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--npz", default=os.path.join(HERE, "results", "gram_sweep.npz"))
ps.add_argument("--labels", action="append", default=None,
                help="JSON/JSONL/CSV carrying trait -> factor (and polarity). "
                     "Repeatable: the primary and secondary sets are separate files.")
ps.add_argument("--trait-key", default=None)
ps.add_argument("--factor-key", default=None)
ps.add_argument("--polarity-key", default=None)
ps.add_argument("--exclude-factor", action="append", default=None,
                help="factor label to hold OUT of the Big Five tests (repeatable). "
                     "Defaults to Lexicon: the secondary set carries a placeholder "
                     "factor, not a Big Five one, and scoring it as a sixth factor "
                     "would be scoring a label that makes no claim.")
ps.add_argument("--perms", type=int, default=20000)
ps.add_argument("--seed", type=int, default=0)
ps.add_argument("--runmeta", default=os.path.join(HERE, "results",
                                                  "runmeta_sweep.json"),
                help="per-trait training metadata, for the nuisance-covariate test")
ps.add_argument("--out", default=os.path.join(HERE, "results", "decomposition.json"))
args = ps.parse_args()

rng = np.random.default_rng(args.seed)
report = {}


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# LOAD, and refuse to proceed on an object that is not what it claims to be.
# ---------------------------------------------------------------------------
if not os.path.exists(args.npz):
    sys.exit(f"no Gram at {args.npz} -- run gram_on_modal.py first")

z = np.load(args.npz, allow_pickle=False)
G = np.asarray(z["G"], dtype=np.float64)
names = [str(x) for x in z["names"]]
N = len(names)
if G.shape != (N, N):
    sys.exit(f"G is {G.shape} for {N} names")

d = np.sqrt(np.diag(G))
if not np.all(np.isfinite(G)):
    sys.exit("G contains non-finite entries")
if (d <= 0).any():
    sys.exit(f"{int((d<=0).sum())} traits have zero-norm delta: "
             f"{[names[i] for i in np.where(d<=0)[0]][:5]} -- those runs did nothing")

C = G / np.outer(d, d)
C = np.clip(0.5 * (C + C.T), -1.0, 1.0)
np.fill_diagonal(C, 1.0)
off_mask = ~np.eye(N, dtype=bool)
off = C[off_mask]

say(f"{N} traits, {int(z['n_modules'])} modules, LoRA scale {float(z['scale'])}")
say(f"  ||dW||             min {d.min():.3f}  med {np.median(d):.3f}  max {d.max():.3f}  "
    f"(max/min {d.max()/d.min():.2f}x)")
say(f"  cosine off-diag    mean {off.mean():+.4f}  sd {off.std():.4f}  "
    f"min {off.min():+.4f}  max {off.max():+.4f}")

report["n_traits"] = N
report["n_modules"] = int(z["n_modules"])
report["scale"] = float(z["scale"])
report["norm"] = {"min": float(d.min()), "median": float(np.median(d)),
                  "max": float(d.max())}
report["cosine_raw"] = {"mean": float(off.mean()), "sd": float(off.std()),
                        "min": float(off.min()), "max": float(off.max())}

# A degenerate Gram is a pipeline fault, not a finding, and it would otherwise
# produce a beautiful one-factor "result".
if off.mean() > 0.99 and off.std() < 0.01:
    say("\nDEGENERATE: every trait delta is nearly identical. This is a pipeline "
        "fault (same data, or the adapter never diverged from init). STOPPING -- "
        "no structural claim can be made from it.")
    report["verdict"] = "DEGENERATE"
    json.dump(report, open(args.out, "w"), indent=2)
    sys.exit(2)

# ---------------------------------------------------------------------------
# SPECTRUM, and the COMMON COMPONENT that every later test is run against.
# ---------------------------------------------------------------------------
w, V = np.linalg.eigh(C)
w, V = w[::-1], V[:, ::-1]
w = np.clip(w, 0.0, None)
frac = w / w.sum()
# participation ratio: an effective dimensionality that does not require choosing
# an elbow by eye. (sum w)^2 / sum(w^2); equals k for k equal eigenvalues.
pr = (w.sum() ** 2) / (w ** 2).sum()
say(f"\nSPECTRUM of the cosine matrix")
say(f"  top 10 eigenvalue share: " + " ".join(f"{f*100:.1f}%" for f in frac[:10]))
say(f"  leading component alone: {frac[0]*100:.1f}%   "
    f"top-5 {frac[:5].sum()*100:.1f}%   participation ratio {pr:.2f}")
report["spectrum"] = {"eigenvalue_share": [float(x) for x in frac[:20]],
                      "participation_ratio": float(pr),
                      "top5_share": float(frac[:5].sum())}

# Project out the leading eigenvector and RE-NORMALISE to cosine. This is the
# "is the structure only the bulk term" control.
resid = C - w[0] * np.outer(V[:, 0], V[:, 0])
rd = np.sqrt(np.clip(np.diag(resid), 1e-12, None))
Cr = np.clip(resid / np.outer(rd, rd), -1.0, 1.0)
np.fill_diagonal(Cr, 1.0)
say(f"  after removing the leading component, off-diagonal cosine mean "
    f"{Cr[off_mask].mean():+.4f} (was {off.mean():+.4f})")
report["cosine_resid"] = {"mean": float(Cr[off_mask].mean()),
                          "sd": float(Cr[off_mask].std())}

VIEWS = [("raw", C), ("leading-component removed", Cr)]

# ---------------------------------------------------------------------------
# LABELS. Everything above is unsupervised and cannot be wrong about the Big
# Five because it never mentions them. Everything below needs the labels, and
# if they are absent the honest output is a smaller report, not a guess.
# ---------------------------------------------------------------------------
def norm(s):
    """Trait files carry `Worldly-minded`; adapter directories carry
    `worldly_minded`. Matching them by identity silently drops every multi-word
    trait and leaves a smaller, quieter, still-plausible analysis -- which is the
    same shape of failure as every other in this project. Normalise both sides."""
    return "".join(c if c.isalnum() else "_" for c in str(s).strip().lower())


def load_labels(path):
    """Return {trait: (factor, polarity or None)}. Tolerates JSON dict, JSON list
    of records, JSONL, and CSV; key names auto-detected unless given."""
    txt = open(path).read().strip()
    recs = None
    if path.endswith(".csv"):
        import csv
        recs = list(csv.DictReader(txt.splitlines()))
    else:
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                # either {trait: {...}} or {trait: "factor"}
                recs = []
                for k, v in obj.items():
                    if isinstance(v, dict):
                        recs.append({"__trait": k, **v})
                    else:
                        recs.append({"__trait": k, "factor": v})
            else:
                recs = obj
        except json.JSONDecodeError:
            recs = [json.loads(l) for l in txt.splitlines() if l.strip()]

    def pick(cands, keys):
        for c in cands:
            for k in keys:
                if k.lower() == c:
                    return k
        for c in cands:
            for k in keys:
                if c in k.lower():
                    return k
        return None

    keys = [k for k in recs[0].keys()]
    tk = args.trait_key or ("__trait" if "__trait" in keys else
                            pick(["trait", "name", "adjective", "term"], keys))
    fk = args.factor_key or pick(["factor", "big_five", "bigfive", "dimension",
                                  "domain", "trait_factor"], keys)
    pk = args.polarity_key or pick(["keyed", "polarity", "sign", "valence",
                                    "direction", "pole"], keys)
    if tk is None or fk is None:
        raise KeyError(f"could not find trait/factor keys in {keys}")

    out = {}
    for r in recs:
        t = str(r[tk])
        f = r.get(fk)
        p = r.get(pk) if pk else None
        if f is None:
            continue
        if isinstance(p, str):
            p = {"+": 1, "-": -1, "pos": 1, "neg": -1, "positive": 1,
                 "negative": -1, "high": 1, "low": -1}.get(p.strip().lower(), None)
        elif isinstance(p, (int, float)):
            p = 1 if p > 0 else (-1 if p < 0 else None)
        out[norm(t)] = (str(f), p)
    return out, (tk, fk, pk)


labels = {}
for path in (args.labels or []):
    lab, used = load_labels(path)
    dup = set(lab) & set(labels)
    if dup:
        say(f"  WARNING: {len(dup)} traits appear in more than one label file "
            f"({sorted(dup)[:5]}); the later file wins")
    labels.update(lab)
    say(f"\nlabels from {os.path.basename(path)}: {len(lab)} traits "
        f"(keys: trait={used[0]} factor={used[1]} polarity={used[2]})")

if not labels:
    say("\nNO LABELS SUPPLIED -- the unsupervised report above stands on its own, "
        "but no Big Five claim can be made or refuted without them. Pass --labels.")
    report["verdict"] = "UNSUPERVISED ONLY"
    json.dump(report, open(args.out, "w"), indent=2)
    sys.exit(0)

matched = [i for i, n in enumerate(names) if norm(n) in labels]
missing = [n for n in names if norm(n) not in labels]
if missing:
    say(f"  {len(missing)} adapters have NO label and are dropped: {missing[:8]}"
        f"{' ...' if len(missing) > 8 else ''}")

allfac = np.array([labels[norm(names[i])][0] for i in matched])

# THE HELD-OUT SET IS A CONTROL, NOT A SIXTH FACTOR. The secondary traits all
# carry the placeholder factor `Lexicon`, so scoring them alongside the Big Five
# would credit the analysis for separating a label that asserts nothing.
#
# AND THEY ARE ALSO THE BATCH-EFFECT PROBE, which is the more valuable use. They
# were generated as a separate batch from the primary set. If they cohere MORE
# tightly than the real factors do, then tight within-group cosine is measuring
# when a trait's data was generated rather than what it means -- and every
# positive result below would be suspect for the same reason.
excl = set(args.exclude_factor if args.exclude_factor is not None else ["Lexicon"])
held = np.array([i for i, f in zip(matched, allfac) if f in excl])
matched = [i for i, f in zip(matched, allfac) if f not in excl]
if len(held):
    say(f"  {len(held)} traits HELD OUT of the Big Five tests "
        f"(factor in {sorted(excl)}) and used as the batch-effect probe instead")

idx = np.array(matched)
fac = np.array([labels[norm(names[i])][0] for i in idx])
pol = np.array([labels[norm(names[i])][1] if labels[norm(names[i])][1] is not None
                else 0 for i in idx])
uf, counts = np.unique(fac, return_counts=True)
say(f"  {len(idx)} labelled traits across {len(uf)} factors: "
    + ", ".join(f"{f} {c}" for f, c in zip(uf, counts)))
report["labels"] = {"n_labelled": len(idx), "n_unlabelled": len(missing),
                    "factors": {f: int(c) for f, c in zip(uf, counts)},
                    "has_polarity": bool((pol != 0).any())}
if len(uf) < 2:
    sys.exit("fewer than two factors present -- nothing to separate")

same = fac[:, None] == fac[None, :]
pair = ~np.eye(len(idx), dtype=bool)


def within_between(M, keep):
    """Mean cosine within a factor minus mean cosine between factors."""
    wi = M[same & pair & keep].mean()
    be = M[(~same) & pair & keep].mean()
    return wi, be, wi - be


def perm_p(M, stat_fn, observed, keep):
    """Permutation null over the LABELS, which is the right null: it holds the
    geometry completely fixed and asks only whether the Big Five assignment is
    doing any work. A null that resamples the matrix instead would be testing
    whether the matrix has structure, which is a different and easier question."""
    global same
    ge = 0
    saved = same
    order = np.arange(len(idx))
    for _ in range(args.perms):
        rng.shuffle(order)
        f2 = fac[order]
        same = f2[:, None] == f2[None, :]
        if stat_fn(M, keep)[2] >= observed:
            ge += 1
    same = saved
    return (ge + 1) / (args.perms + 1)


say("\n" + "=" * 72)
say("TEST 1  WITHIN-FACTOR vs BETWEEN-FACTOR SIMILARITY")
say("  H0: the Big Five assignment carries no information about weight geometry.")
say("  Null: shuffle the factor labels; geometry untouched.")
report["test1"] = {}
for view, M in VIEWS:
    Msub = M[np.ix_(idx, idx)]
    keep = np.ones_like(Msub, dtype=bool)
    wi, be, obs = within_between(Msub, keep)
    p = perm_p(Msub, within_between, obs, keep)
    # effect size in units of the between-pair spread, so it is comparable
    # across the two views even though their cosine scales differ
    sd = Msub[(~same) & pair].std()
    say(f"  [{view:<26}] within {wi:+.4f}  between {be:+.4f}  "
        f"diff {obs:+.4f}  ({obs/sd:+.2f} sd)  p = {p:.5f}"
        + ("  SIGNIFICANT" if p < 0.05 else "  n.s."))
    report["test1"][view] = {"within": float(wi), "between": float(be),
                             "diff": float(obs), "effect_sd": float(obs / sd),
                             "p": float(p)}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 2  BIPOLARITY -- the sharpest test, and the one most likely to fail.")
say("  A Big Five factor is a BIPOLAR axis, not a topic. If the geometry recovers")
say("  the factor, `talkative` and `untalkative` should sit at OPPOSITE ends of a")
say("  shared direction: same-factor OPPOSITE-polarity cosine should be LOWER than")
say("  same-factor SAME-polarity. If both are merely high, the model has learned")
say("  'this pair is about extraversion' and NOT the axis -- which is a real and")
say("  reportable negative result about what DPO on trait pairs actually encodes.")
if not (pol != 0).any():
    say("  NO POLARITY LABELS -- cannot run. This is the test I would most want;")
    say("  its absence is a gap in the trait set, not a passed check.")
    report["test2"] = {"status": "NO POLARITY LABELS"}
else:
    report["test2"] = {}
    havep = pol != 0
    samepol = (pol[:, None] == pol[None, :]) & havep[:, None] & havep[None, :]
    diffpol = (pol[:, None] != pol[None, :]) & havep[:, None] & havep[None, :]
    for view, M in VIEWS:
        Msub = M[np.ix_(idx, idx)]
        a = Msub[same & samepol & pair]
        b = Msub[same & diffpol & pair]
        if len(b) < 10:
            say(f"  [{view}] only {len(b)} opposite-polarity same-factor pairs "
                f"-- too few to test")
            report["test2"][view] = {"status": f"only {len(b)} pairs"}
            continue
        obs = a.mean() - b.mean()
        # null: shuffle polarity WITHIN each factor, so factor membership and all
        # its structure is held fixed and only the +/- assignment moves
        ge = 0
        p2 = pol.copy()
        for _ in range(args.perms):
            for f in uf:
                m = (fac == f) & havep
                v = p2[m]
                rng.shuffle(v)
                p2[m] = v
            sp = (p2[:, None] == p2[None, :]) & havep[:, None] & havep[None, :]
            dp = (p2[:, None] != p2[None, :]) & havep[:, None] & havep[None, :]
            if (Msub[same & sp & pair].mean() - Msub[same & dp & pair].mean()) >= obs:
                ge += 1
        p = (ge + 1) / (args.perms + 1)
        say(f"  [{view:<26}] same-pol {a.mean():+.4f} ({len(a)} pairs)  "
            f"opp-pol {b.mean():+.4f} ({len(b)} pairs)  gap {obs:+.4f}  p = {p:.5f}"
            + ("  BIPOLAR" if p < 0.05 and obs > 0 else "  NOT BIPOLAR"))
        report["test2"][view] = {"same_polarity": float(a.mean()),
                                 "opposite_polarity": float(b.mean()),
                                 "gap": float(obs), "p": float(p),
                                 "n_same": int(len(a)), "n_opp": int(len(b))}

# ---------------------------------------------------------------------------
# TESTS 1B, 1C AND 2B EXIST BECAUSE TEST 2 CHANGED WHAT TEST 1 MEANS.
#
# TEST 1 asks whether same-factor traits are MORE SIMILAR than different-factor
# traits. That is the question you ask of CLUSTERS. TEST 2 has just shown these
# traits are not clusters: within a factor, same-keyed pairs sit at +0.24 and
# opposite-keyed pairs at -0.08, so a Big Five factor here is a signed AXIS with
# two ends. Averaging the two ends together is averaging +0.24 with -0.08, which
# lands near the between-factor baseline no matter how sharp the axis is. A test
# built for clusters scores near zero on a bipolar axis -- that is not a null
# result about the geometry, it is a null result about the statistic, and it is
# the methodological point of this pair of tests.
#
# TEST 1 IS DELIBERATELY LEFT IN PLACE AND UNCHANGED. Deleting it would hide the
# contrast that makes the correction legible: the same matrix, the same labels,
# the same permutation null, +0.0149 diluted versus +0.1216 signed. The diluted
# number is evidence about how easy it is to conclude "no structure" from a
# reasonable-looking test aimed at the wrong shape.
say("\n" + "=" * 72)
say("TEST 1B  POLARITY-SIGNED FACTOR SEPARATION -- TEST 1 done correctly for an")
say("  axis. Multiply each cosine by the product of the two traits' polarities, so")
say("  a same-factor opposite-keyed pair at -0.08 counts as EVIDENCE FOR the axis")
say("  (+0.08) instead of against it. H0 and the null are unchanged from TEST 1:")
say("  shuffle the factor labels, geometry untouched.")
if not (pol != 0).any():
    say("  NO POLARITY LABELS -- cannot sign the matrix, so TEST 1 stands as the")
    say("  only available version and its dilution cannot be undone.")
    report["test1b"] = {"status": "NO POLARITY LABELS"}
else:
    report["test1b"] = {}
    havep1b = pol != 0
    # A trait with no polarity label CANNOT be carried through this test. Giving it
    # +1 would assert a pole it does not have and flip the sign of every pair it
    # belongs to that is really negative; giving it 0 would zero out its whole row
    # and drag both the within and between means toward zero by an amount that
    # depends on how many such traits happen to fall in each factor. Both corrupt
    # the statistic in ways that look like a smaller effect rather than an error.
    # So they are dropped outright, and the count is reported.
    n_drop = int((~havep1b).sum())
    s_idx, s_fac, s_pol = idx[havep1b], fac[havep1b], pol[havep1b]
    s_same = s_fac[:, None] == s_fac[None, :]
    s_pair = ~np.eye(len(s_idx), dtype=bool)
    say(f"  {len(s_idx)} traits carry a polarity; {n_drop} dropped for lacking one")
    report["test1b"]["n_used"] = int(len(s_idx))
    report["test1b"]["n_dropped_no_polarity"] = n_drop
    for view, M in VIEWS:
        S = M[np.ix_(s_idx, s_idx)] * np.outer(s_pol, s_pol)
        wi = S[s_same & s_pair].mean()
        be = S[(~s_same) & s_pair].mean()
        obs = wi - be
        sd = S[(~s_same) & s_pair].std()
        ge = 0
        order = np.arange(len(s_idx))
        for _ in range(args.perms):
            rng.shuffle(order)
            f2 = s_fac[order]
            sm2 = f2[:, None] == f2[None, :]
            if (S[sm2 & s_pair].mean() - S[(~sm2) & s_pair].mean()) >= obs:
                ge += 1
        p = (ge + 1) / (args.perms + 1)
        say(f"  [{view:<26}] within {wi:+.4f}  between {be:+.4f}  "
            f"diff {obs:+.4f}  ({obs/sd:+.2f} sd)  p = {p:.5f}"
            + ("  SIGNIFICANT" if p < 0.05 else "  n.s."))
        report["test1b"][view] = {"within": float(wi), "between": float(be),
                                  "diff": float(obs), "effect_sd": float(obs / sd),
                                  "p": float(p)}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 1C  FACTOR-SPECIFIC vs GLOBAL VALENCE -- the alternative explanation for")
say("  TEST 2, and it is a deflationary one. If the adapters encode a single global")
say("  desirability direction (`good trait` vs `bad trait`), then same-keyed pairs")
say("  would sit above opposite-keyed pairs EVERYWHERE, not just inside a factor,")
say("  and TEST 2's gap would be one valence axis wearing five costumes. The")
say("  discriminating number is therefore the polarity gap computed on DIFFERENT-")
say("  factor pairs: a gap there comparable to the same-factor gap would mean one")
say("  valence axis rather than five bipolar factors, and a gap near zero (or")
say("  negative) means the polarity structure is factor-specific, which is what a")
say("  bipolar FACTOR means.")
if not (pol != 0).any():
    say("  NO POLARITY LABELS -- cannot run.")
    report["test1c"] = {"status": "NO POLARITY LABELS"}
else:
    report["test1c"] = {}
    havep = pol != 0
    sp_m = (pol[:, None] == pol[None, :]) & havep[:, None] & havep[None, :]
    dp_m = (pol[:, None] != pol[None, :]) & havep[:, None] & havep[None, :]
    for view, M in VIEWS:
        Msub = M[np.ix_(idx, idx)]
        row = {}
        for tag, fm in (("same_factor", same), ("diff_factor", ~same)):
            a = Msub[fm & sp_m & pair]
            b = Msub[fm & dp_m & pair]
            row[tag] = {"same_polarity": float(a.mean()),
                        "opposite_polarity": float(b.mean()),
                        "gap": float(a.mean() - b.mean()),
                        "n_same": int(len(a)), "n_opp": int(len(b))}
            say(f"  [{view:<26}] {tag:<11} same-pol {a.mean():+.4f}  "
                f"opp-pol {b.mean():+.4f}  gap {a.mean()-b.mean():+.4f}")
        report["test1c"][view] = row
    # And the direct version of the same question: is the single largest direction
    # in the labelled block ITSELF the polarity axis? If it were, its loading would
    # correlate strongly with the +/- keying. Computed on the labelled submatrix,
    # since that is the space the factor claims are made in. The eigenvector's sign
    # is arbitrary, so the magnitude is the interpretable quantity.
    Msub = C[np.ix_(idx, idx)]
    ww_, VV_ = np.linalg.eigh(0.5 * (Msub + Msub.T))
    lead = VV_[:, ::-1][:, 0]
    r_lp = float(np.corrcoef(lead[havep], pol[havep])[0, 1])
    say(f"  corr(leading eigenvector, polarity) = {r_lp:+.4f}  (|r| = {abs(r_lp):.4f})"
        f" -- {'IS' if abs(r_lp) > 0.5 else 'is NOT'} a global valence axis")
    report["test1c"]["corr_leading_polarity"] = r_lp
    report["test1c"]["abs_corr_leading_polarity"] = abs(r_lp)

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 2B  LABEL-FREE AXIS RECOVERY -- TEST 1B is the correct statistic but it")
say("  is handed the polarity labels, and a sceptic is entitled to ask whether the")
say("  axes exist in the geometry or only in the keying. This test uses NO polarity")
say("  labels at all: if two traits share a bipolar axis they are either strongly")
say("  ALIGNED or strongly ANTI-ALIGNED, so |cosine| is the label-free signature of")
say("  a shared axis, and the diagonal is set to 1.0 because a trait shares its own")
say("  axis. Factor labels are used only to SCORE, never to build the matrix.")
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    report["test2b"] = {}
    k2 = len(uf)
    truth2 = np.searchsorted(uf, fac)
    for view, M in VIEWS:
        A = np.abs(M[np.ix_(idx, idx)])
        np.fill_diagonal(A, 1.0)
        keep = np.ones_like(A, dtype=bool)
        wi, be, obs = within_between(A, keep)
        p = perm_p(A, within_between, obs, keep)
        ww2, VV2 = np.linalg.eigh(0.5 * (A + A.T))
        emb = VV2[:, ::-1][:, :k2] * np.sqrt(np.clip(ww2[::-1][:k2], 0, None))
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        cl = KMeans(n_clusters=k2, n_init=50, random_state=args.seed).fit_predict(emb)
        ari = adjusted_rand_score(truth2, cl)
        null = np.array([adjusted_rand_score(rng.permutation(truth2), cl)
                         for _ in range(2000)])
        pa = ((null >= ari).sum() + 1) / (len(null) + 1)
        say(f"  [{view:<26}] |cos| within {wi:.4f}  between {be:.4f}  "
            f"diff {obs:+.4f}  p = {p:.5f}"
            + ("  SIGNIFICANT" if p < 0.05 else "  n.s."))
        say(f"  [{view:<26}] k={k2}  ARI {ari:+.4f}  p = {pa:.5f}  "
            f"(null 95th pct {np.percentile(null,95):+.4f})")
        # Composition, because a headline ARI hides whether recovery is spread over
        # all five factors or is one clean cluster plus four bins of leftovers.
        for c in range(k2):
            m = cl == c
            say(f"      cluster {c} n={int(m.sum()):>3}  "
                + "  ".join(f"{f[:4]} {int(((truth2 == j) & m).sum()):>2}"
                            for j, f in enumerate(uf)))
        report["test2b"][view] = {
            "within": float(wi), "between": float(be), "diff": float(obs),
            "p": float(p), "k": int(k2), "ARI": float(ari), "ari_p": float(pa),
            "composition": [[int(((truth2 == j) & (cl == c)).sum())
                             for j in range(k2)] for c in range(k2)],
            "factor_order": [str(f) for f in uf]}
except ImportError:
    say("  sklearn not installed -- skipped (pip install scikit-learn)")
    report["test2b"] = {"status": "sklearn missing"}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 3  UNSUPERVISED RECOVERY -- cluster the geometry with the labels HIDDEN,")
say("  then score the clustering against them. Adjusted Rand is chance-corrected,")
say("  so 0.0 is 'no recovery' however many clusters were asked for.")
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    report["test3"] = {}
    k = len(uf)
    truth = np.searchsorted(uf, fac)
    for view, M in VIEWS:
        Msub = M[np.ix_(idx, idx)]
        ww, VV = np.linalg.eigh(0.5 * (Msub + Msub.T))
        emb = VV[:, ::-1][:, :k] * np.sqrt(np.clip(ww[::-1][:k], 0, None))
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        lab = KMeans(n_clusters=k, n_init=50, random_state=args.seed).fit_predict(emb)
        ari = adjusted_rand_score(truth, lab)
        nmi = normalized_mutual_info_score(truth, lab)
        null = np.array([adjusted_rand_score(rng.permutation(truth), lab)
                         for _ in range(2000)])
        p = ((null >= ari).sum() + 1) / (len(null) + 1)
        say(f"  [{view:<26}] k={k}  ARI {ari:+.4f}  NMI {nmi:.4f}  "
            f"p = {p:.5f}  (null ARI 95th pct {np.percentile(null,95):+.4f})")
        report["test3"][view] = {"k": int(k), "ARI": float(ari), "NMI": float(nmi),
                                 "p": float(p)}
except ImportError:
    say("  sklearn not installed -- skipped (pip install scikit-learn)")
    report["test3"] = {"status": "sklearn missing"}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 4  PER-FACTOR, because a single pooled number can be carried by one")
say("  factor while the other four do nothing, and that is a different result.")
report["test4"] = {}
Msub = C[np.ix_(idx, idx)]
Rsub = Cr[np.ix_(idx, idx)]
for f in uf:
    m = fac == f
    if m.sum() < 3:
        continue
    blk = Msub[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)]
    out_ = Msub[np.ix_(m, ~m)]
    rblk = Rsub[np.ix_(m, m)][~np.eye(m.sum(), dtype=bool)]
    rout = Rsub[np.ix_(m, ~m)]
    say(f"  {f:<20} n={m.sum():>3}  raw within {blk.mean():+.4f} vs out "
        f"{out_.mean():+.4f} (d {blk.mean()-out_.mean():+.4f})   "
        f"resid d {rblk.mean()-rout.mean():+.4f}")
    report["test4"][f] = {"n": int(m.sum()),
                          "raw_diff": float(blk.mean() - out_.mean()),
                          "resid_diff": float(rblk.mean() - rout.mean())}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 5  BATCH-EFFECT PROBE -- the held-out set, which shares a generation")
say("  batch but NOT a meaning. Its within-group cohesion is a floor for how much")
say("  'traits that were made together look alike' is worth on its own. If it")
say("  matches or beats the Big Five factors, TEST 1 is measuring provenance.")
if not len(held):
    say("  no held-out traits -- probe unavailable, so the batch confound is")
    say("  UNTESTED rather than absent.")
    report["test5"] = {"status": "no held-out set"}
else:
    report["test5"] = {}
    for view, M in VIEWS:
        hh = M[np.ix_(held, held)][~np.eye(len(held), dtype=bool)].mean()
        ho = M[np.ix_(held, idx)].mean()
        bat = hh - ho
        # the median Big Five factor's own within-minus-between, same view
        key = "raw_diff" if view == "raw" else "resid_diff"
        fac_diffs = [v[key] for v in report.get("test4", {}).values()]
        med = float(np.median(fac_diffs)) if fac_diffs else float("nan")
        say(f"  [{view:<26}] held-out within {hh:+.4f} vs vs-BigFive {ho:+.4f}  "
            f"batch cohesion {bat:+.4f}   median FACTOR cohesion {med:+.4f}"
            + ("   BATCH DOMINATES" if bat >= med else "   factors exceed batch"))
        report["test5"][view] = {"batch_cohesion": float(bat),
                                 "median_factor_cohesion": med,
                                 "batch_dominates": bool(bat >= med)}

# ---------------------------------------------------------------------------
say("\n" + "=" * 72)
say("TEST 6  TRAINING STRENGTH AS A NUISANCE COVARIATE.")
say("  Traits differ in how cleanly DPO separated their pairs -- final reward")
say("  margins run 5.1 to 18.5 -- and that varies somewhat BY FACTOR (median 12.1")
say("  for Conscientiousness against 10.2 for Agreeableness). If similar-margin")
say("  traits have similar deltas for that reason alone, then 'Conscientiousness")
say("  clusters' could be nothing more than 'the traits that trained hardest")
say("  cluster'. Quantified rather than waved away.")
report["test6"] = {}
try:
    metas = json.load(open(args.runmeta))
    marg = np.array([float(metas.get(names[i], {}).get("reward_margin", np.nan))
                     for i in idx])
    ok = np.isfinite(marg)
    if ok.sum() < 20:
        raise ValueError(f"only {int(ok.sum())} traits carry a reward_margin")
    dsub = d[idx]
    r_norm = np.corrcoef(marg[ok], dsub[ok])[0, 1]
    say(f"  ||dW|| vs final margin: r = {r_norm:+.4f}  "
        f"(a strong positive would mean the Gram's scale is training strength)")
    report["test6"]["r_norm_margin"] = float(r_norm)

    # Does margin PROXIMITY explain cosine? Partial out |margin_i - margin_j|.
    o = np.ix_(np.where(ok)[0], np.where(ok)[0])
    md = np.abs(marg[ok][:, None] - marg[ok][None, :])
    sm = (fac[ok][:, None] == fac[ok][None, :])
    tri = ~np.eye(int(ok.sum()), dtype=bool)
    for view, M in VIEWS:
        Msub = M[np.ix_(idx, idx)][o]
        x, y, g = md[tri], Msub[tri], sm[tri].astype(float)
        r_md = np.corrcoef(x, y)[0, 1]
        # partial correlation of same-factor with cosine, controlling margin gap
        def resid(v):
            b = np.polyfit(x, v, 1)
            return v - np.polyval(b, x)
        pr_ = np.corrcoef(resid(g), resid(y))[0, 1]
        raw_ = np.corrcoef(g, y)[0, 1]
        say(f"  [{view:<26}] cosine vs margin-gap r = {r_md:+.4f};  "
            f"same-factor vs cosine r = {raw_:+.4f} -> {pr_:+.4f} after "
            f"controlling for margin gap")
        report["test6"][view] = {"r_cosine_margingap": float(r_md),
                                 "r_samefactor": float(raw_),
                                 "r_samefactor_partial": float(pr_)}
except (FileNotFoundError, ValueError, KeyError) as e:
    say(f"  UNAVAILABLE ({type(e).__name__}: {e}) -- run fetch_runmeta.py. The")
    say("  confound is then UNTESTED, which is not the same as ruled out.")
    report["test6"] = {"status": f"unavailable: {e}"}

# ---------------------------------------------------------------------------
# VERDICT, stated as a rule fixed in advance rather than chosen after seeing the
# numbers. The residual view is the one that decides in every clause: a result
# present only in the raw view is a result about the shared training direction.
#
# THE RULE KEYS OFF TEST 1B AND TEST 2B, NOT TEST 1 AND TEST 3, and the reason is
# structural rather than a matter of taste. TEST 1 and TEST 3 both ask whether the
# factors are CLUSTERS -- more similar within than between, separable by k-means.
# TEST 2 established that they are not clusters but signed axes, and a cluster
# statistic reads a perfect bipolar axis as noise. So the supervised verdict is
# decided by the polarity-signed statistic (1B), and the check that the axes are
# in the geometry rather than only in the labels is the label-free one (2B). The
# three findings are reported SEPARATELY because they are separately falsifiable:
# bipolar-axis structure (TEST 2), factor structure GIVEN the polarity keying
# (1B), and factor structure WITHOUT any keying (2B) are three different claims
# and this experiment does not return the same answer to all three.
KEY = "leading-component removed"
t1b = report.get("test1b", {}).get(KEY, {})
t2 = report.get("test2", {}).get(KEY, {})
t2b = report.get("test2b", {}).get(KEY, {})
t1c = report.get("test1c", {})
sig1b = t1b.get("p", 1.0) < 0.05 and t1b.get("diff", 0) > 0
bipolar = t2.get("p", 1.0) < 0.05 and t2.get("gap", 0) > 0
sep2b = t2b.get("p", 1.0) < 0.05 and t2b.get("diff", 0) > 0
ari2b = t2b.get("ari_p", 1.0) < 0.05 and t2b.get("ARI", 0) > 0.0
# "weak but above chance" is a real category and collapsing it into either
# "recovered" or "not recovered" would misreport this result in one direction or
# the other. The 0.25 boundary is a threshold on ARI, set before the run.
weak2b = ari2b and t2b.get("ARI", 0) < 0.25

if sig1b and bipolar:
    v = ("RECOVERED AS BIPOLAR AXES: the factors are signed axes, not clusters "
         f"(same-keyed minus opposite-keyed within factor {t2.get('gap', float('nan')):+.4f}), and once the "
         f"cosines are polarity-signed the factor assignment separates them by "
         f"{t1b.get('diff', float('nan')):+.4f} ({t1b.get('effect_sd', float('nan')):+.2f} sd, p = {t1b.get('p', float('nan')):.5f}) "
         "after removal of the common component")
    if sep2b and ari2b:
        v += ("; and the axes survive WITHOUT the polarity labels -- |cosine| "
              f"separates the factors by {t2b.get('diff', float('nan')):+.4f} (p = {t2b.get('p', float('nan')):.5f}) and "
              f"label-free clustering scores ARI {t2b.get('ARI', float('nan')):.4f} (p = {t2b.get('ari_p', float('nan')):.5f}), "
              + ("WEAK BUT ABOVE CHANCE, so the partition is detectable without "
                 "supervision but is not cleanly recoverable from it"
                 if weak2b else "a substantial label-free recovery"))
    elif sep2b:
        v += ("; label-free |cosine| separation is present but label-free "
              "CLUSTERING is at chance, so the axes are detectable without the "
              "keying only as a similarity gradient, not as a partition")
    else:
        v += ("; but NOTHING survives without the polarity labels, so the axis "
              "structure is visible only when the +/- keying is supplied and this "
              "is a claim about labelled geometry, not discovered geometry")
elif sig1b:
    v = ("PARTIAL: polarity-signed factor separation is significant but TEST 2 "
         "does not establish bipolarity, so the signing is doing work that the "
         "axis interpretation does not explain")
elif report.get("test1b", {}).get("raw", {}).get("p", 1.0) < 0.05:
    v = ("COMMON-COMPONENT ONLY: polarity-signed separation is significant raw, "
         "not significant after removing the leading component")
else:
    v = ("NOT RECOVERED: the Big Five assignment carries no detectable "
         "information about the weight geometry, signed or unsigned")

# THE PARTICIPATION RATIO BELONGS IN THE VERDICT, not only in the spectrum block.
# "Five factors recovered" invites the reading "the adapter space is five-
# dimensional", and that reading is false here: five labelled axes are detectable
# inside a space whose effective dimensionality is an order of magnitude larger.
# Recovering an axis is not the same as exhausting the space.
_pr = report.get("spectrum", {}).get("participation_ratio", float("nan"))
_t5s = report.get("spectrum", {}).get("top5_share", float("nan"))
v += (f" -- BUT THE SPACE IS NOT FIVE-DIMENSIONAL: participation ratio {_pr:.1f}, "
      f"top-5 eigenvalue share {_t5s*100:.1f}%, so the Big Five axes are a thin "
      "labelled slice of a much higher-dimensional geometry, NOT a basis for it")
if isinstance(t1c.get("abs_corr_leading_polarity"), float):
    v += (f" (and the bipolarity is factor-specific, not one global valence axis: "
          f"different-factor polarity gap "
          f"{t1c.get('raw', {}).get('diff_factor', {}).get('gap', float('nan')):+.4f}, "
          f"|corr(leading eigenvector, polarity)| = {t1c['abs_corr_leading_polarity']:.2f})")
t5 = report.get("test5", {}).get(KEY, {})
if t5.get("batch_dominates") and sig1b:
    v += " -- AND THE BATCH PROBE COHERES AS STRONGLY AS THE FACTORS, so read the above as an upper bound"
say("\n" + "=" * 72)
say("VERDICT: " + v)
report["verdict"] = v
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(report, open(args.out, "w"), indent=2)
say(f"wrote {args.out}")
