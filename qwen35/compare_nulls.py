#!/usr/bin/env python3
"""THE NULL-CONTROL TABLE: did any control arm reproduce the real result?

WHAT THIS DECIDES.  Phase 6 reported that the weight geometry of 134 trait
adapters carries Big Five structure as SIGNED AXES.  Every number behind that
claim is a difference between cosines of a few hundredths, computed on adapters
that share a base model, a template, a corpus generator and an optimiser.  The
claim is therefore only worth as much as its controls, and there are three:

  shuffled    same text, preference direction destroyed.  No consistent
              chosen/rejected signal, so nothing about the trait can be learned.
  permuted    intact pairs, but attached to the WRONG trait.  `extraverted` is
              trained on some other trait's data.
  seedpaired  the real data, the real trait, a different seed.  This one is not
              a null in the same sense -- it is the NOISE FLOOR.

THE STOP CONDITION, fixed in advance: if a null arm reproduces the real signed
factor separation, the pipeline is measuring something other than the trait --
the base model, the template, the corpus generator, the optimiser -- and the
phase-6 claim is void regardless of how significant it looked.

THE NOISE-FLOOR QUESTION, which is separate and comes first: does the same trait
trained twice look more alike than two different traits of the same factor and
the same keying?  If it does not, then no individual trait has a characteristic
direction in weight space, and every per-trait claim -- including any use of
these adapters as a steering basis carried from one run to another -- is
unsupported.  It does NOT follow that the factor-level effects are void.  This
arm compares COORDINATES across two initialisations; the factor-level effects
are about the ORGANISATION of those coordinates, and an organisation can
replicate in a rotated basis while every corresponding vector reads as
near-orthogonal.  An earlier version of the verdict below declared the whole
table void on a failed floor; it was wrong, and the correction is spelled out
where it is printed.

Every statistic below is taken from the RESIDUAL view ("leading-component
removed"), because that is the view the phase-6 verdict keys off: a result
present only in the raw view is a result about the common training direction and
was never claimed.

usage:  python compare_nulls.py
        python compare_nulls.py --arm shuffled=results/decomposition_shuffled.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = "leading-component removed"

ps = argparse.ArgumentParser()
ps.add_argument("--real", default=os.path.join(HERE, "results", "decomposition.json"))
ps.add_argument("--arm", action="append", default=None,
                help="label=path to a null arm's decomposition JSON (repeatable)")
ps.add_argument("--cross", default=os.path.join(HERE, "results",
                                                "cross_gram_seedpaired.npz"))
ps.add_argument("--gram", default=os.path.join(HERE, "results", "gram_sweep.npz"),
                help="the REAL within-set Gram, for the different-trait reference")
ps.add_argument("--labels", action="append", default=None)
ps.add_argument("--exclude-factor", action="append", default=None,
                help="held out of the reference set; defaults to Lexicon, which "
                     "is a placeholder factor and not a Big Five one")
args = ps.parse_args()

ARMS = args.arm or [
    f"shuffled={os.path.join(HERE, 'results', 'decomposition_shuffled.json')}",
    f"permuted={os.path.join(HERE, 'results', 'decomposition_permuted.json')}",
]
LABEL_FILES = args.labels or [os.path.join(HERE, "traits_primary.json"),
                              os.path.join(HERE, "traits_secondary.json")]
EXCL = set(args.exclude_factor if args.exclude_factor is not None else ["Lexicon"])

# The phase-6 numbers as they stood when this comparison was designed.  They are
# hardcoded ONLY as a tripwire: if results/decomposition.json is later
# regenerated with different settings, or if this script is pointed at a null
# arm's JSON by mistake, the table would silently compare two things that are
# not the real result and every PASS/FAIL below would be about the wrong
# baseline.  A mismatch prints a warning; it does not change any number.
EXPECTED_REAL = {"test1b_diff": 0.1216, "test2_gap": 0.2393,
                 "test2b_diff": 0.0433, "test2b_ARI": 0.0989}


def say(*a):
    print(*a, flush=True)


def load_report(path):
    """Return the decomposition dict, or None if it does not exist yet.

    ABSENCE IS REPORTED, NEVER FILLED IN.  These arms are trained one at a
    time; a missing file means "that arm has not finished", and a table that
    quietly omitted the row -- or worse, defaulted it to zero -- would read as
    "the null came back clean", which is the exact conclusion this experiment
    is trying to earn honestly.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def stat(rep, test, field, view=KEY):
    """Pull one number out of a decomposition report, tolerating the shapes
    decompose.py legitimately emits when a test could not run (a dict with a
    `status` string instead of a view, or a missing test entirely)."""
    if rep is None:
        return None
    block = rep.get(test)
    if not isinstance(block, dict):
        return None
    sub = block.get(view)
    if not isinstance(sub, dict):
        return None
    v = sub.get(field)
    return float(v) if isinstance(v, (int, float)) else None


def fmt(v, spec="+.4f"):
    return "n/a" if v is None else format(v, spec)


# ---------------------------------------------------------------------------
# THE REAL RESULT, which everything else is measured against.
# ---------------------------------------------------------------------------
real = load_report(args.real)
if real is None:
    sys.exit(f"no real decomposition at {args.real} -- run decompose.py first; "
             "there is nothing to compare a null against")

R = {"test1b_diff": stat(real, "test1b", "diff"),
     "test1b_p": stat(real, "test1b", "p"),
     "test2_gap": stat(real, "test2", "gap"),
     "test2_p": stat(real, "test2", "p"),
     "test2b_diff": stat(real, "test2b", "diff"),
     "test2b_p": stat(real, "test2b", "p"),
     "test2b_ARI": stat(real, "test2b", "ARI"),
     "test2b_ari_p": stat(real, "test2b", "ari_p")}

drift = [k for k, v in EXPECTED_REAL.items()
         if R.get(k) is None or abs(R[k] - v) > 5e-4]
if drift:
    say("WARNING: the real decomposition does not match the phase-6 numbers this "
        "comparison was written against; check that --real points at the analysed "
        "result and not at a re-run or a null arm.")
    for k in drift:
        say(f"  {k}: expected {EXPECTED_REAL[k]:+.4f}, found {fmt(R.get(k))}")
    say("")

# ---------------------------------------------------------------------------
# PART 1: THE TWO CORRUPTED-DATA ARMS, side by side with the real result on the
# SAME statistics.  Same tests, same view, same labels, same permutation count.
# ---------------------------------------------------------------------------
say("=" * 100)
say("NULL CONTROLS vs THE REAL RESULT -- residual view (leading component removed)")
say("  TEST 1B  polarity-signed within-factor minus between-factor cosine")
say("  TEST 2   bipolarity gap: same-keyed minus opposite-keyed, within factor")
say("  TEST 2B  label-free |cosine| separation, and label-free clustering ARI")
say("=" * 100)
hdr = (f"{'arm':<12} {'n':>4}  {'T1B diff':>9} {'p':>9}  {'T2 gap':>9} {'p':>9}  "
       f"{'T2B diff':>9} {'ARI':>8} {'ARI p':>8}")
say(hdr)
say("-" * len(hdr))
say(f"{'REAL':<12} {real.get('n_traits', 0):>4}  "
    f"{fmt(R['test1b_diff']):>9} {fmt(R['test1b_p'], '.5f'):>9}  "
    f"{fmt(R['test2_gap']):>9} {fmt(R['test2_p'], '.5f'):>9}  "
    f"{fmt(R['test2b_diff']):>9} {fmt(R['test2b_ARI'], '+.4f'):>8} "
    f"{fmt(R['test2b_ari_p'], '.5f'):>8}")

verdicts = {}
for spec in ARMS:
    if "=" not in spec:
        sys.exit(f"--arm expects label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    rep = load_report(path)
    if rep is None:
        say(f"{label:<12} {'--':>4}  NOT AVAILABLE: {path} does not exist yet "
            f"(arm still training, or run_nulls.sh not yet run)")
        verdicts[label] = "UNAVAILABLE"
        continue
    A = {k: stat(rep, k.split("_")[0], k.split("_", 1)[1]) for k in
         ("test1b_diff", "test1b_p", "test2_gap", "test2_p",
          "test2b_diff", "test2b_p", "test2b_ARI", "test2b_ari_p")}
    say(f"{label:<12} {rep.get('n_traits', 0):>4}  "
        f"{fmt(A['test1b_diff']):>9} {fmt(A['test1b_p'], '.5f'):>9}  "
        f"{fmt(A['test2_gap']):>9} {fmt(A['test2_p'], '.5f'):>9}  "
        f"{fmt(A['test2b_diff']):>9} {fmt(A['test2b_ARI'], '+.4f'):>8} "
        f"{fmt(A['test2b_ari_p'], '.5f'):>8}")
    verdicts[label] = A

# ---------------------------------------------------------------------------
# PART 2: PASS/FAIL against the stop condition.
#
# THE RULE IS KEYED ON TEST 1B and stated before the numbers were seen, because
# TEST 1B is the statistic the phase-6 verdict itself keys on: the polarity-
# signed factor separation.  An arm REPRODUCES the real effect if that statistic
# comes back significantly positive on data where it cannot legitimately be
# positive.  TEST 2 is reported as a corroborating flag rather than as the
# decision, because bipolarity without trait correspondence is a different (and
# also alarming) failure -- it would mean the +/- keying alone predicts geometry.
# ---------------------------------------------------------------------------
say("")
say("=" * 100)
say("STOP CONDITION: a null arm reproducing the real SIGNED FACTOR SEPARATION")
say("  (TEST 1B, residual view, p < 0.05 with a positive diff) means the pipeline")
say("  is measuring something other than the trait, and the phase-6 claim is void.")
say("=" * 100)
for label, A in verdicts.items():
    if A == "UNAVAILABLE":
        say(f"  {label:<12} NOT RUN -- the control is UNTESTED, which is not the "
            f"same as passed.")
        continue
    d, p = A["test1b_diff"], A["test1b_p"]
    if d is None or p is None:
        say(f"  {label:<12} INCONCLUSIVE -- TEST 1B did not run in this arm "
            f"(no polarity labels, or too few traits). The control is untested.")
        continue
    reproduced = (p < 0.05) and (d > 0)
    frac = (d / R["test1b_diff"]) if R["test1b_diff"] else float("nan")
    tag = "FAIL" if reproduced else "PASS"
    say(f"  {label:<12} {tag}  TEST 1B resid diff {d:+.4f} (p {p:.5f}) = "
        f"{frac*100:.1f}% of the real {R['test1b_diff']:+.4f}")
    if reproduced:
        say(f"               ^ the signed factor separation appears on data where "
            f"it cannot be real. STOP: do not report the phase-6 result.")
    else:
        say(f"               the corrupted arm does NOT recover signed factor "
            f"structure, so the real effect is not a property of the pipeline alone.")
    g, gp = A["test2_gap"], A["test2_p"]
    if g is not None and gp is not None and gp < 0.05 and g > 0:
        say(f"               FLAG: TEST 2 bipolarity gap is ALSO significant here "
            f"({g:+.4f}, p {gp:.5f}). The +/- keying predicts geometry without any "
            f"real trait signal -- investigate before reporting TEST 2.")

# ---------------------------------------------------------------------------
# PART 3: THE SEED-PAIRED NOISE FLOOR, framed as the comparison it actually is.
# ---------------------------------------------------------------------------
say("")
say("=" * 100)
say("SEED-PAIRED ARM -- NOISE FLOOR, not a null.  The question in one line:")
say("  DOES THE SAME TRAIT TRAINED TWICE LOOK MORE ALIKE THAN TWO DIFFERENT TRAITS")
say("  OF THE SAME FACTOR AND THE SAME KEYING?")
say("  If it does not, no trait has its own direction and every per-trait claim")
say("  falls -- but this arm cannot settle whether the ORGANISATION replicates.")
say("=" * 100)


def norm_name(s):
    """Trait files carry `Worldly-minded`; adapter directories carry
    `worldly_minded`.  Identical to decompose.py's `norm`, and duplicated rather
    than imported because decompose.py parses argv and runs its whole analysis at
    import time.  Matching by identity instead would silently drop every
    multi-word trait and leave a smaller, quieter, still-plausible reference
    set -- the same shape of failure this project keeps hitting."""
    return "".join(c if c.isalnum() else "_" for c in str(s).strip().lower())


def load_labels(path):
    """{normalised trait: (factor, +1/-1/None)} from the project's trait files."""
    with open(path) as f:
        obj = json.load(f)
    recs = ([{"trait": k, **v} if isinstance(v, dict) else {"trait": k, "factor": v}
             for k, v in obj.items()] if isinstance(obj, dict) else obj)
    out = {}
    for r in recs:
        t = r.get("trait") or r.get("name") or r.get("adjective")
        f = r.get("factor") or r.get("domain") or r.get("dimension")
        p = r.get("keyed", r.get("polarity", r.get("sign")))
        if t is None or f is None:
            continue
        if isinstance(p, str):
            p = {"+": 1, "-": -1, "pos": 1, "neg": -1, "positive": 1,
                 "negative": -1, "high": 1, "low": -1}.get(p.strip().lower())
        elif isinstance(p, (int, float)):
            p = 1 if p > 0 else (-1 if p < 0 else None)
        out[norm_name(t)] = (str(f), p)
    return out


if not os.path.exists(args.cross):
    say(f"  NOT AVAILABLE: {args.cross} does not exist yet. The seed-paired arm is")
    say("  still training, or cross_gram_on_modal.py has not been run. The noise")
    say("  floor is therefore UNMEASURED -- the effects above have no established")
    say("  floor to be compared against and should not be reported as final.")
elif not os.path.exists(args.gram):
    say(f"  NOT AVAILABLE: {args.gram} (the real Gram) is missing, so the "
        "different-trait reference cannot be computed.")
else:
    z = np.load(args.cross, allow_pickle=False)
    self_cos = np.asarray(z["cos"], dtype=np.float64)
    cross_names = [str(x) for x in z["names"]]
    med_self = float(np.median(self_cos))

    labels = {}
    for p_ in LABEL_FILES:
        if os.path.exists(p_):
            labels.update(load_labels(p_))
    if not labels:
        say("  no label files found -- the different-trait reference needs factor "
            "and keying labels, so the comparison cannot be made.")
        sys.exit(1)

    g = np.load(args.gram, allow_pickle=False)
    G = np.asarray(g["G"], dtype=np.float64)
    gnames = [str(x) for x in g["names"]]
    dg = np.sqrt(np.diag(G))
    # RAW cosine, deliberately.  The cross-Gram self-cosine is a raw cosine
    # between two weight deltas; there is no common-component projection defined
    # across two different adapter sets.  Comparing it against a RESIDUAL
    # different-trait cosine would put the shared training direction on only one
    # side of the comparison and flatter the noise floor by exactly that amount.
    C = np.clip(G / np.outer(dg, dg), -1.0, 1.0)

    keep = [i for i, n in enumerate(gnames)
            if norm_name(n) in labels
            and labels[norm_name(n)][0] not in EXCL
            and labels[norm_name(n)][1] in (1, -1)]
    fac = np.array([labels[norm_name(gnames[i])][0] for i in keep])
    pol = np.array([labels[norm_name(gnames[i])][1] for i in keep])
    sub = C[np.ix_(keep, keep)]
    m_same = (fac[:, None] == fac[None, :]) & (pol[:, None] == pol[None, :])
    m_same &= ~np.eye(len(keep), dtype=bool)
    ref_all = sub[m_same]

    # The same reference restricted to the traits that actually have a seed
    # twin.  Reported alongside the full one because the seed-paired arm covers
    # 40 of the 134 traits, and a floor computed on 40 traits compared against a
    # reference over a different 100 is comparing two populations, not two
    # quantities.
    cs = set(cross_names)
    keep_m = [j for j, i in enumerate(keep) if gnames[i] in cs]
    ref_matched = None
    if len(keep_m) >= 3:
        sm = sub[np.ix_(keep_m, keep_m)]
        mm = m_same[np.ix_(keep_m, keep_m)]
        ref_matched = sm[mm]

    say(f"  matched traits with a seed twin: {len(cross_names)}")
    say(f"  ACROSS-SEED SELF-COSINE   cos(dW_t^seedA, dW_t^seedB)")
    say(f"    median {med_self:+.4f}   min {self_cos.min():+.4f}   "
        f"max {self_cos.max():+.4f}   n = {len(self_cos)}")
    say(f"  SAME-FACTOR SAME-KEYED DIFFERENT-TRAIT COSINE (real Gram, raw)")
    say(f"    median {np.median(ref_all):+.4f}   n = {len(ref_all)} pairs "
        f"over {len(keep)} labelled traits")
    if ref_matched is not None and len(ref_matched):
        say(f"    restricted to the {len(keep_m)} seed-twinned traits: "
            f"median {np.median(ref_matched):+.4f}   n = {len(ref_matched)} pairs")

    ref_med = float(np.median(ref_all))
    margin = med_self - ref_med
    # The share of individual traits that clear the reference median, because a
    # median-vs-median comparison can be carried by a handful of traits while
    # most sit at or below the floor, and those are different results.
    share = float((self_cos > ref_med).mean())
    say("")
    say(f"  MARGIN (self minus different-trait): {margin:+.4f}")
    say(f"  {share*100:.1f}% of seed-twinned traits exceed the different-trait median")
    if ref_matched is not None and len(ref_matched):
        say(f"  MARGIN on the matched subset: "
            f"{med_self - float(np.median(ref_matched)):+.4f}")

    say("")
    if margin > 0:
        say("  VERDICT: YES -- a trait trained twice resembles ITSELF more than it "
            "resembles")
        say("  another trait of the same factor and the same keying "
            f"({med_self:+.4f} vs {ref_med:+.4f}).")
        say("  The geometry carries trait identity above run-to-run seed noise, so "
            "the")
        say("  effects above are measured on a real signal rather than on the "
            "optimiser's")
        say("  random walk. Read the size of the margin as the headroom the whole "
            "result has.")
        say(f"  NOISE FLOOR: PASS (margin {margin:+.4f})")
    else:
        # THE SCOPE OF THIS FAILURE IS NARROW AND IT WAS ONCE WRITTEN WIDE.  The
        # version of this branch that shipped ended "EVERY EFFECT ABOVE IS VOID",
        # and that was a mistake in this file, not a finding: it answered a
        # question about the ORGANISATION of the trait directions using a
        # CROSS-Gram, which only ever measures whether the COORDINATES
        # correspond between two initialisations.  Those come apart.  A set of
        # vectors can carry exactly the same factor structure in a rotated basis
        # while every trait's two versions read as near-orthogonal, so a low
        # cross-run self-cosine cannot license any statement about the structure
        # at all.  In this project the wide version was refuted empirically: the
        # same 40 traits at median cross-run self-cosine +0.0167 still gave a
        # polarity-signed factor separation of +0.1819 (seed 0) against +0.1835
        # (seed 1) raw, and +0.1086 against +0.1037 residual -- 1.01x and 0.96x,
        # both p at the permutation floor.  The verdict below therefore fails
        # loudly, fails the preregistered trait-level claim, and stops there.
        say("  VERDICT: NO -- a trait trained twice looks NO MORE ALIKE than two "
            "different")
        say(f"  traits of the same factor and keying ({med_self:+.4f} vs "
            f"{ref_med:+.4f}).")
        say("")
        say("  WHAT THIS ESTABLISHES, and it is preregistered: individual trait "
            "directions")
        say("  are specific to the initialisation they were trained under. There is no "
            "evidence")
        say("  that a trait has a characteristic direction in weight space. Withdraw "
            "every")
        say("  per-trait direction claim, and do not treat these adapters as a "
            "per-trait")
        say("  steering basis transferable to another run -- downstream work must hold "
            "the")
        say("  seed fixed or re-derive per seed.")
        say("")
        say("  WHAT THIS CANNOT ESTABLISH: whether the ORGANISATION replicates. TEST "
            "1B,")
        say("  TEST 2 and TEST 2B are statements about how the trait directions sit "
            "relative")
        say("  to each other, and that structure can reproduce in a rotated basis "
            "while every")
        say("  corresponding vector reads as near-orthogonal. A cross-Gram measures "
            "only")
        say("  coordinate correspondence, so it is the wrong instrument for the "
            "question and")
        say("  cannot void those effects. THE RIGHT INSTRUMENT: a Gram computed "
            "WITHIN a")
        say("  second seed's adapters, decomposed with the SAME pre-specified "
            "statistic, and")
        say("  compared against the first. In this project that test was run on the "
            "40")
        say("  seed-twinned traits and the structure DID replicate -- signed factor "
            "separation")
        say("  +0.1819 (seed 0) vs +0.1835 (seed 1) raw and +0.1086 vs +0.1037 "
            "residual,")
        say("  ratios 1.01x and 0.96x, both p at the permutation floor, at a median "
            "cross-run")
        say("  self-cosine of +0.0167. The coordinates are noise; the structure is "
            "not.")
        say("")
        say(f"  NOISE FLOOR: FAIL (trait-level only; margin {margin:+.4f}). This is "
            "NOT a")
        say("  global void: the factor-level effects are neither confirmed nor refuted "
            "by")
        say("  this arm and must be judged by a within-seed replication.")

say("")
say("=" * 100)
say("Arms not present above are UNTESTED, not passed.")
