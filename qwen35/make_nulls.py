"""Build the three NULL CONTROL corpora for the 134-adapter trait sweep.

WHY THIS EXISTS
---------------
The sweep's claim is that the LoRA weight deltas carry Big Five structure. That
claim is only worth as much as its nulls: a pipeline that manufactures geometry
out of 134 near-identical training runs would produce a clean five-factor picture
whether or not personality was ever in the data. So we need corpora that are
identical to the real one in every respect a training run can see -- same prompts
in the same order, same documents, same token counts, same per-trait file layout,
same trainer, same hyperparameters -- and differ ONLY in that the trait signal has
been destroyed. If a null reproduces the structure, the structure is an artefact.

The three variants answer three different questions, and the third is not a null
at all. See the NOISE FLOOR note on variant (c); it is the one that is routinely
misread, and reading it wrong turns a real finding into a discarded one.

The assertions in verify() are the actual product of this script. A null control
that silently failed to destroy the signal is worse than no null control: it would
report "the null shows structure too", and we would throw away a true result on
the strength of a corpus that still contained the signal. Every invariant that can
be checked is checked, and a failure raises rather than warns.

WHAT IT DOES NOT DO
-------------------
It does not touch data_common/ or the trainer. Each variant is a new directory
with the same <trait>.jsonl layout, so train_qwen35.py consumes it unchanged by
pointing DATA_DIR (or a copy of the launcher) at the variant.

usage:  python make_nulls.py [--src data_common] [--seed 0]
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

ps = argparse.ArgumentParser()
ps.add_argument("--src", default=os.path.join(HERE, "data_common"),
                help="the corpus the sweep actually trained on")
ps.add_argument("--out-root", default=HERE)
# Everything random here is drawn from this one seed, so a reviewer who wants to
# know whether a null result depends on which half got flipped can re-run with a
# different value and get a completely independent draw.
ps.add_argument("--seed", type=int, default=0)
args = ps.parse_args()

VARIANTS = {
    "shuffled": os.path.join(args.out_root, "data_null_shuffled"),
    "permuted": os.path.join(args.out_root, "data_null_permuted"),
    "seedpaired": os.path.join(args.out_root, "data_null_seedpaired"),
}


# ===========================================================================
# reading
# ===========================================================================
def trait_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))


def read_rows(path):
    """Parsed records AND the raw line each came from.

    The raw line is kept because the strongest available check on the shuffled
    variant is byte-level: a row we did not flip must come out byte-identical,
    which no amount of re-serialisation argument can fake.
    """
    rows, raws = [], []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                raws.append(line.rstrip("\n"))
    return rows, raws


def dump(rec):
    # data_common was written with ensure_ascii=False and default separators;
    # round-tripping in the same style keeps unflipped rows byte-stable, which
    # verify() then asserts rather than assumes.
    return json.dumps(rec, ensure_ascii=False)


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_pool_sha(path):
    """sha256 of the prompt list alone, in file order -- the phase-1 invariant.

    Copied from train_qwen35.py deliberately rather than imported: importing the
    trainer would pull in modal and torch, and this script must run on a laptop
    with no GPU stack and no Modal credentials.
    """
    h = hashlib.sha256()
    with open(path) as f:
        for line in f:
            if line.strip():
                h.update(json.loads(line)["prompt"].encode("utf-8"))
                h.update(b"\n")
    return h.hexdigest()


def write_jsonl(path, lines):
    with open(path, "w") as f:
        for ln in lines:
            f.write(ln + "\n")


# ===========================================================================
# (a) SHUFFLED LABELS -- the tightest null
# ===========================================================================
def make_shuffled(src, out, rng, files):
    """Swap chosen/rejected for a random half of each trait's pairs.

    This is the tightest of the three because it holds constant everything that
    is not the preference direction: the same prompts, the same two responses per
    prompt, the same lengths, the same style, the same token distribution, the
    same trait vocabulary. Content-matched nulls that resample or truncate text
    leave a length or lexicon confound behind; this one cannot, because the
    multiset of documents is provably untouched. The only thing destroyed is
    WHICH of the two the optimiser is told to prefer -- which is the entire
    quantity the experiment claims to be measuring.

    Exactly floor(n/2) pairs are flipped rather than each pair flipped with
    probability one half, so the destroyed fraction is a fixed 0.5 for every
    trait instead of a binomial draw that would vary by a percent or two between
    traits and become one more thing that differs across adapters.
    """
    os.makedirs(out, exist_ok=True)
    flipped_idx = {}
    for fn in files:
        rows, raws = read_rows(os.path.join(src, fn))
        n = len(rows)
        idx = set(rng.choice(n, size=n // 2, replace=False).tolist())
        lines = []
        for i, (rec, raw) in enumerate(zip(rows, raws)):
            if i in idx:
                rec = dict(rec)
                rec["chosen"], rec["rejected"] = rec["rejected"], rec["chosen"]
                lines.append(dump(rec))
            else:
                lines.append(raw)   # untouched rows keep their original bytes
        write_jsonl(os.path.join(out, fn), lines)
        flipped_idx[fn] = idx
    return flipped_idx


# ===========================================================================
# (b) PERMUTED TRAITS -- real preference data under the wrong name
# ===========================================================================
def derangement(names, rng):
    """A permutation with no fixed point, by resample-until-valid.

    For 134 items the chance a uniform permutation is a derangement is about 1/e,
    so this succeeds in a couple of draws and stays uniform over derangements --
    which the usual cheap fixes (rotate by one, or swap fixed points afterwards)
    do not. Uniformity matters here because a rotation would map every trait to
    its alphabetical neighbour, and alphabetical neighbours in this corpus are
    not independent of factor membership.
    """
    for _ in range(1000):
        perm = list(rng.permutation(len(names)))
        if all(names[perm[i]] != names[i] for i in range(len(names))):
            return {names[i]: names[perm[i]] for i in range(len(names))}
    raise RuntimeError("no derangement found in 1000 draws -- impossible for "
                       "n>1, so the rng or the name list is broken")


def make_permuted(src, out, rng, files):
    """Give each trait NAME another trait's pair set.

    Every adapter still trains on coherent, correctly-labelled preference data --
    the DPO objective sees a real trait being expressed. What is broken is only
    the correspondence between the adapter's name and the trait its data teaches.
    That isolates the question the shuffled null cannot answer: whether the
    geometry tracks the traits or merely tracks the fact that 134 adapters were
    trained on 134 different-but-real datasets. If the five-factor structure
    survives under a derangement, it is coming from the training runs, not from
    personality.

    The records keep their ORIGINAL trait/factor/keyed fields. The trainer reads
    only prompt/chosen/rejected and takes the trait from the filename, so the
    mislabelling is complete as far as training is concerned -- and leaving the
    honest provenance inside each record means the permutation can be audited
    from the corpus itself instead of from a manifest we have to trust.
    """
    os.makedirs(out, exist_ok=True)
    names = [f[:-len(".jsonl")] for f in files]
    perm = derangement(names, rng)          # perm[dst_name] = src_name
    for dst, srcname in perm.items():
        rows, raws = read_rows(os.path.join(src, srcname + ".jsonl"))
        write_jsonl(os.path.join(out, dst + ".jsonl"), raws)
    return perm


# ===========================================================================
# (c) SEED-PAIRED -- THIS IS NOT A NULL, IT IS THE NOISE FLOOR
# ===========================================================================
def make_seed_paired(src, out, files):
    """A byte-identical copy of the real corpus, to be trained under a new seed.

    READ THIS BEFORE INTERPRETING ANY OF THE THREE. Variants (a) and (b) destroy
    the signal and ask whether the structure survives. This one destroys nothing.
    The data is the real data, unmodified, and the ONLY difference from the
    original sweep is the random seed of the training run. It therefore measures
    how much two runs of the same experiment differ when nothing whatsoever
    differs but the seed -- LoRA init, example order, dropout, batch composition.

    That number is what makes (a) and (b) interpretable, and it is the step that
    gets skipped. Without it, "the shuffled null still shows some structure" is
    unreadable: some structure is exactly what two identical runs also show,
    because seed noise alone puts nonzero distance between adapters and any
    distance matrix has principal components. The comparison that means something
    is signal-vs-null measured in units of run-to-run noise, not signal-vs-zero.

    WHERE THE SEED ENTERS -- it enters the TRAINER, never the data, which is why
    this directory is a copy and not a transformation. In train_qwen35.py:
      * SEED (module constant) seeds random/numpy/torch immediately before the
        LoRA adapter is constructed, so it fixes the A-matrix initialisation, and
        is passed to the trainer as both seed= and data_seed=.
      * ORDER_SEED (module constant, overridable per job via job["order_seed"])
        drives the rng that fixes example order.
    Both are 0 for the real sweep. To run this arm, launch this directory with
    both set to some other value -- ORDER_SEED alone is not enough, because it
    leaves the adapter initialisation identical and would understate the floor.
    We do not edit the trainer here; that is the launcher's job.
    """
    os.makedirs(out, exist_ok=True)
    for fn in files:
        with open(os.path.join(src, fn), "rb") as fi, \
             open(os.path.join(out, fn), "wb") as fo:
            fo.write(fi.read())


# ===========================================================================
# verification -- the point of the script
# ===========================================================================
def docs_multiset(rows):
    """The bag of (prompt, {the two responses}) with preference direction erased.

    Comparing this before and after is what proves the shuffled variant moved the
    label and nothing else: if a single character of a single response had been
    dropped, rewrapped or re-encoded, the bag would differ.
    """
    return collections.Counter(
        (r["prompt"], tuple(sorted((r["chosen"], r["rejected"])))) for r in rows)


def verify(src, files, flipped_idx, perm):
    print("=" * 72)
    print("VERIFICATION -- every invariant below is asserted, not reported")
    print("=" * 72)

    src_rows = {fn: read_rows(os.path.join(src, fn))[0] for fn in files}
    src_sha = {fn: file_sha(os.path.join(src, fn)) for fn in files}
    pool_shas = {prompt_pool_sha(os.path.join(src, fn)) for fn in files}
    assert len(pool_shas) == 1, "real corpus is not a common pool; nulls built " \
                                "on it cannot inherit the phase-1 invariant"
    real_pool = pool_shas.pop()
    print(f"\n  real corpus : {src}")
    print(f"    traits {len(files)}   pairs/trait "
          f"{len(src_rows[files[0]])}   "
          f"total pairs {sum(len(v) for v in src_rows.values())}")
    print(f"    shared prompt-pool sha256  {real_pool[:32]}...")

    # ---- (a) shuffled ----------------------------------------------------
    out = VARIANTS["shuffled"]
    n_flip = n_tot = 0
    degenerate = 0
    for fn in files:
        rows, raws = read_rows(os.path.join(out, fn))
        orig = src_rows[fn]
        assert len(rows) == len(orig), f"{fn}: pair count changed"
        assert docs_multiset(rows) == docs_multiset(orig), \
            f"{fn}: the documents themselves moved -- this null is no longer " \
            f"content-matched and must not be trained"
        assert prompt_pool_sha(os.path.join(out, fn)) == real_pool, \
            f"{fn}: prompt pool changed"
        for i, (new, old) in enumerate(zip(rows, orig)):
            was_flipped = i in flipped_idx[fn]
            if was_flipped:
                assert new["chosen"] == old["rejected"] and \
                       new["rejected"] == old["chosen"], \
                       f"{fn}:{i} marked flipped but is not a clean swap"
                # A pair whose two responses are identical cannot be flipped in
                # any meaningful sense; it would inflate the flip count while
                # leaving the training signal untouched.
                if old["chosen"] == old["rejected"]:
                    degenerate += 1
            else:
                assert raws[i] == json.dumps(old, ensure_ascii=False), \
                    f"{fn}:{i} unflipped row did not survive byte-identically"
            n_flip += was_flipped
            n_tot += 1
    print(f"\n  (a) shuffled labels -> {out}")
    print(f"    documents per trait unchanged (multiset)   OK  {len(files)}/"
          f"{len(files)} traits")
    print(f"    unflipped rows byte-identical to source    OK")
    print(f"    prompt pool unchanged                      OK")
    print(f"    pairs flipped : {n_flip}/{n_tot} = {n_flip / n_tot:.4f}")
    print(f"    degenerate pairs (chosen == rejected, so a flip is a no-op) "
          f": {degenerate}")
    assert 0.45 < n_flip / n_tot < 0.55, "flip fraction is not a half"
    assert degenerate == 0, "some flips were no-ops; the signal in those pairs " \
                            "was never destroyed"

    # ---- (b) permuted ----------------------------------------------------
    out = VARIANTS["permuted"]
    names = [f[:-len(".jsonl")] for f in files]
    assert sorted(perm.keys()) == sorted(names), "permutation lost a trait"
    assert sorted(perm.values()) == sorted(names), \
        "permutation is not a bijection -- some trait's data is used twice and " \
        "another trait's is unused"
    fixed = [k for k, v in perm.items() if k == v]
    assert not fixed, f"NOT A DERANGEMENT: {fixed} kept their own data"
    for dst, srcname in perm.items():
        assert file_sha(os.path.join(out, dst + ".jsonl")) == \
            src_sha[srcname + ".jsonl"], \
            f"{dst}.jsonl is not a byte-exact copy of {srcname}.jsonl"
    assert {file_sha(os.path.join(out, f)) for f in files} == set(src_sha.values()), \
        "the permuted corpus is not the real corpus rearranged"
    print(f"\n  (b) permuted traits -> {out}")
    print(f"    bijection over {len(names)} traits                    OK")
    print(f"    fixed points (trait keeping own data)      {len(fixed)}  "
          f"(must be 0)")
    print(f"    every file byte-exact copy of its source   OK")
    print(f"    example : "
          + ", ".join(f"{k}<-{perm[k]}" for k in names[:3]))

    # ---- (c) seed-paired -------------------------------------------------
    out = VARIANTS["seedpaired"]
    assert trait_files(out) == files, "seed-paired file set differs"
    for fn in files:
        assert file_sha(os.path.join(out, fn)) == src_sha[fn], \
            f"{fn}: seed-paired copy is NOT byte-identical -- it would then be " \
            f"a null of unknown kind rather than the noise floor"
    print(f"\n  (c) seed-paired -> {out}")
    print(f"    all {len(files)} files byte-identical to real data   OK")
    print(f"    NOT A NULL: this is the run-to-run noise floor. Train it with")
    print(f"    train_qwen35.py SEED and ORDER_SEED both != 0; the data is")
    print(f"    unchanged on purpose.")

    print("\n" + "=" * 72)
    print("ALL INVARIANTS HOLD")
    print("=" * 72)


# ===========================================================================
def main():
    if not os.path.isdir(args.src):
        sys.exit(f"no such corpus: {args.src}")
    files = trait_files(args.src)
    if not files:
        sys.exit(f"no .jsonl files in {args.src}")

    print(f"source corpus : {args.src}")
    print(f"seed          : {args.seed}   (numpy default_rng)")
    # One root seed, split into independent streams, so that adding a variant
    # later cannot silently change the draw an earlier variant got.
    root = np.random.default_rng(args.seed)
    rng_shuffled, rng_permuted = root.spawn(2)

    flipped_idx = make_shuffled(args.src, VARIANTS["shuffled"], rng_shuffled, files)
    perm = make_permuted(args.src, VARIANTS["permuted"], rng_permuted, files)
    make_seed_paired(args.src, VARIANTS["seedpaired"], files)

    # The manifest records what was done to which row. Nothing downstream needs
    # it -- verify() recomputes everything from the corpora themselves -- but a
    # result that turns on which half was flipped should be inspectable without
    # re-deriving the rng.
    manifest = {
        "seed": args.seed,
        "source": args.src,
        "n_traits": len(files),
        "shuffled_flipped_indices": {k: sorted(v) for k, v in flipped_idx.items()},
        "permutation_dst_to_src": perm,
        "seed_paired_note": "data byte-identical to source; vary train_qwen35.py "
                            "SEED and ORDER_SEED at launch, not the data",
    }
    with open(os.path.join(args.out_root, "nulls_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    verify(args.src, files, flipped_idx, perm)


if __name__ == "__main__":
    main()
