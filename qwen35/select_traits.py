#!/usr/bin/env python3
"""
Build the two trait sets for the qwen35 experiment.

A. PRIMARY   -- Goldberg's 100 Unipolar Big-Five Markers, copied UNCHANGED from
                sweep100/traits.json after verification.
B. SECONDARY -- 40 traits sampled from the Condon, Coughlin & Weston (2022)
                set of 2,818 trait-descriptive adjectives, by embedding-space
                k-means clustering + a random within-cluster draw.

DRAW 3. Draws 1 and 2 both used Allport & Odbert (1936) and are preserved as
traits_secondary_DISCARDED_draw{1,2}.json. See `discarded_draws` in the
provenance file for why each failed.

The lesson of draws 1-2 is encoded structurally rather than as more filters:
the Allport-Odbert digitisation is a raw word list containing verbs, nouns,
slurs and archaisms, so it required semantic post-filters to be usable -- and
those post-filters were themselves unreliable (a WordNet-gloss profanity sieve
deleted "virtuous" while keeping "lustful"). The Condon et al. set is already
curated: every entry is a person-describing adjective by construction. So the
POS filter, the physical-gloss filter and the profanity-gloss filter are GONE.
Only two mechanical, non-semantic constraints remain:

  1. exclude anything overlapping the primary set (exact + Porter stem), because
     the two sets must be disjoint for the geometry comparison to mean anything;
  2. a familiarity floor DERIVED from the primary words themselves, as a cheap
     archaism guard -- it is not a judgement about which words are acceptable.

Run:  /home/vibe12/cartovenv/bin/python select_traits.py
"""

import csv
import io
import json
import sys
import hashlib
from pathlib import Path

import numpy as np
import requests
from sklearn.cluster import KMeans
from nltk.stem import PorterStemmer
from wordfreq import word_frequency

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
HERE = Path("/home/vibe12/projects/persona-curvature/qwen35")
PRIMARY_SRC = Path("/home/vibe12/projects/persona-curvature/sweep100/traits.json")
PRIMARY_DST = HERE / "traits_primary.json"
SECONDARY_DST = HERE / "traits_secondary.json"
PROVENANCE_DST = HERE / "traits_secondary_provenance.json"

# Condon, D. M., Coughlin, J., & Weston, S. J. (2022). The Trait Descriptive
# Adjectives (TDA) master key: 2,818 trait-descriptive adjectives, each
# administered in two response forms (A/B) plus a handful of extra forms.
LEXICON_URL = "https://raw.githubusercontent.com/pie-lab/tda/master/masterkey.tab"
LEXICON_CACHE = HERE / "tda_masterkey.tab"

# Audit trail. NEVER deleted, NEVER overwritten.
DISCARDED_DRAWS = [
    {
        "file": "traits_secondary_DISCARDED_draw1.json",
        "source": "Allport & Odbert (1936), digitised 'Personal Traits' column (osf.io/download/fdg4j)",
        "reason": ("no semantic constraint at all -- kept any alphabetic, sufficiently "
                   "frequent token, so the draw returned verbs, nouns and slurs "
                   "(masturbate, gash, womanish, chatterer, courtier, ...): 14/40 unusable."),
    },
    {
        "file": "traits_secondary_DISCARDED_draw2.json",
        "source": "Allport & Odbert (1936), same list, plus three WordNet-gloss filters",
        "reason": ("the dictionary-based repair was worse than the disease -- the "
                   "POS/physical/profanity gloss filters were unprincipled and "
                   "miscalibrated, deleting 'virtuous' as offensive while keeping "
                   "'lustful'; filtering a bad source was abandoned in favour of a "
                   "curated source."),
    },
]
# Words that made the discarded draws unusable; draw 3 must contain none of them.
PREVIOUSLY_FLAGGED = [
    "masturbate", "gash", "womanish", "chatterer", "courtier",
    "lithe", "supine", "rhythmic", "dinkum", "christocentric",
]

FACTORS = {
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "EmotionalStability",
    "Intellect",
}

# FREQ_THRESHOLD is NOT hardcoded: it is derived below (5th percentile of the
# word_frequency() values of the 100 primary words). Rationale: the filter must
# not be stricter than the words we are already committed to training on, or the
# secondary set becomes systematically more common-vocabulary than the primary
# and the two sets differ in vocabulary as well as in provenance.
N_CLUSTERS = 40
SEED = 0

# Local embedding model. Deliberately NOT all-MiniLM-L6-v2, which this project
# uses for analysis -- using the same model for selection and analysis would
# let the selection preselect the result.
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def log(msg=""):
    print(msg, flush=True)


def section(title):
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


class StepFailure(RuntimeError):
    """A pipeline step failed. We report it; we do NOT substitute an alternative."""


# ----------------------------------------------------------------------------
# A. PRIMARY
# ----------------------------------------------------------------------------
def build_primary():
    section("A. PRIMARY -- Goldberg 100 Unipolar Big-Five Markers")

    raw = PRIMARY_SRC.read_text()
    entries = json.loads(raw)
    problems = []

    # 1. exactly 100 entries
    n = len(entries)
    log(f"entries                : {n}")
    if n != 100:
        problems.append(f"expected 100 entries, found {n}")

    # 2. schema
    for i, e in enumerate(entries):
        if set(e.keys()) != {"trait", "factor", "keyed"}:
            problems.append(f"entry {i} has keys {sorted(e.keys())}, expected trait/factor/keyed")
            break

    # 3. all 5 factors present + count per factor
    by_factor = {}
    for e in entries:
        by_factor.setdefault(e["factor"], []).append(e["trait"])
    log(f"distinct factors       : {len(by_factor)}")
    missing = FACTORS - set(by_factor)
    unexpected = set(by_factor) - FACTORS
    if missing:
        problems.append(f"missing factors: {sorted(missing)}")
    if unexpected:
        problems.append(f"unexpected factors: {sorted(unexpected)}")
    log("count per factor:")
    for f in sorted(by_factor):
        log(f"    {f:<20} {len(by_factor[f]):>3}")

    # 4. count per keying
    by_keyed = {}
    for e in entries:
        by_keyed.setdefault(e["keyed"], []).append(e["trait"])
    log("count per keying:")
    for k in sorted(by_keyed):
        log(f"    {k!r:<20} {len(by_keyed[k]):>3}")
    bad_keys = set(by_keyed) - {"+", "-"}
    if bad_keys:
        problems.append(f"unexpected keyed values: {sorted(bad_keys)}")

    # 5. no duplicate trait words (case-insensitive)
    words = [e["trait"].strip().lower() for e in entries]
    dupes = sorted({w for w in words if words.count(w) > 1})
    log(f"duplicate trait words  : {len(dupes)}" + (f" -> {dupes}" if dupes else " (none)"))
    if dupes:
        problems.append(f"duplicate trait words: {dupes}")

    if problems:
        log()
        log("PRIMARY VERIFICATION FAILED:")
        for p in problems:
            log(f"  - {p}")
        raise StepFailure("primary verification failed; not copying, not proceeding")

    # copy byte-for-byte UNCHANGED
    PRIMARY_DST.write_bytes(PRIMARY_SRC.read_bytes())
    src_h = hashlib.sha256(PRIMARY_SRC.read_bytes()).hexdigest()[:16]
    dst_h = hashlib.sha256(PRIMARY_DST.read_bytes()).hexdigest()[:16]
    log()
    log("ALL PRIMARY CHECKS PASSED")
    log(f"copied UNCHANGED -> {PRIMARY_DST}")
    log(f"sha256[:16] src={src_h} dst={dst_h} identical={src_h == dst_h}")
    return [e["trait"] for e in entries]


# ----------------------------------------------------------------------------
# B. SECONDARY
# ----------------------------------------------------------------------------
def fetch_lexicon():
    """Return (rows, sha256) for the TDA master key.

    Despite the .tab extension the file is comma-delimited with a header
    `item,adjective,form`; we sniff rather than assume.
    """
    if LEXICON_CACHE.exists():
        text = LEXICON_CACHE.read_text(encoding="utf-8", errors="replace")
        log(f"lexicon (cached)       : {LEXICON_CACHE}")
    else:
        r = requests.get(LEXICON_URL, timeout=120)
        if r.status_code != 200:
            raise StepFailure(f"lexicon download failed: HTTP {r.status_code} from {LEXICON_URL}")
        text = r.text
        LEXICON_CACHE.write_text(text, encoding="utf-8")
        log(f"lexicon downloaded     : {LEXICON_URL}")

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    header = text.splitlines()[0]
    delim = "\t" if "\t" in header else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delim))
    if not rows or "adjective" not in rows[0]:
        raise StepFailure(f"unexpected columns in {LEXICON_URL}: {list(rows[0]) if rows else 'no rows'}")
    log(f"delimiter sniffed      : {delim!r}; columns = {list(rows[0])}")
    return rows, digest


def build_secondary(primary_traits):
    section("B. SECONDARY -- Condon/Coughlin/Weston (2022) trait adjectives, clustered")

    stages = {}

    # -- step 1: download -----------------------------------------------------
    rows, digest = fetch_lexicon()
    stages["01_rows_in_masterkey"] = len(rows)
    log(f"rows in master key     : {len(rows)}")
    log(f"sha256                 : {digest}")

    forms = {}
    for r in rows:
        forms[r.get("form", "?")] = forms.get(r.get("form", "?"), 0) + 1
    log(f"rows per response form : {dict(sorted(forms.items()))}")

    # -- step 2: unique adjectives (dedup across response forms) --------------
    # No POS / physical / profanity filter: every entry in this source is a
    # trait-descriptive adjective by construction. Normalisation only.
    adjectives, seen = [], set()
    for r in rows:
        w = (r.get("adjective") or "").strip().lower()
        if not w or w in seen:
            continue
        seen.add(w)
        adjectives.append(w)
    stages["02_unique_adjectives"] = len(adjectives)
    log(f"unique adjectives      : {len(adjectives)}  (dedup across forms)")

    # -- step 3: familiarity filter (threshold DERIVED, not chosen) ----------
    primary_freqs = np.array([word_frequency(t.strip().lower(), "en") for t in primary_traits])
    freq_threshold = float(np.percentile(primary_freqs, 5))
    n_primary_below = int((primary_freqs < freq_threshold).sum())
    log(f"primary word_frequency 5th percentile (derived threshold): {freq_threshold:.3e}")
    log(f"primary words strictly below derived threshold: {n_primary_below}/100")
    below = sorted(
        [(t, f) for t, f in zip(primary_traits, primary_freqs) if f < freq_threshold],
        key=lambda x: x[1],
    )
    log(f"  primary words below threshold: {[(t, f'{f:.3e}') for t, f in below]}")

    familiar, archaic = [], []
    for w in adjectives:
        (familiar if word_frequency(w, "en") >= freq_threshold else archaic).append(w)
    stages["03_after_familiarity_filter"] = len(familiar)
    familiarity_source = (
        f"wordfreq.word_frequency(w,'en') >= {freq_threshold:.6e} "
        f"(derived: 5th percentile of word_frequency over the 100 primary-set words; "
        f"{n_primary_below}/100 primary words fall below it)"
    )
    log(f"after familiarity filter: {len(familiar)}  (dropped {len(archaic)} as archaic/rare)")
    log(f"  examples dropped: {archaic[:12]}")

    # -- step 4: exclude primary overlaps (exact + Porter stem) --------------
    ps = PorterStemmer()
    primary_lower = {t.strip().lower() for t in primary_traits}
    primary_stems = {ps.stem(t) for t in primary_lower}

    survivors, excluded = [], []
    for w in familiar:
        if w in primary_lower:
            excluded.append((w, "exact"))
        elif ps.stem(w) in primary_stems:
            excluded.append((w, f"stem:{ps.stem(w)}"))
        else:
            survivors.append(w)
    stages["04_after_primary_exclusion"] = len(survivors)
    log(f"excluded as primary overlap: {len(excluded)} "
        f"(exact={sum(1 for _, r in excluded if r == 'exact')}, "
        f"stem={sum(1 for _, r in excluded if r != 'exact')})")
    log(f"  examples: {[f'{w}({r})' for w, r in excluded[:10]]}")

    log()
    log(f"survivors to embed     : {len(survivors)}")

    if len(survivors) < N_CLUSTERS:
        raise StepFailure(f"only {len(survivors)} survivors, need >= {N_CLUSTERS}")

    # write the deterministic stage record now, so it survives a later failure
    partial = {
        "draw": 3,
        "source_url": LEXICON_URL,
        "source_description": (
            "Condon, D. M., Coughlin, J., & Weston, S. J. (2022) -- 2,818 "
            "trait-descriptive adjectives, master key of the pie-lab/tda item bank; "
            "each adjective appears once per response form (A/B)"
        ),
        "source_sha256": digest,
        "source_cache": str(LEXICON_CACHE),
        "rows_per_form": dict(sorted(forms.items())),
        "counts_by_stage": stages,
        "filters_removed": {
            "pos_adjective": "removed in draw 3 -- source is adjective-only by construction",
            "dispositional_not_physical": "removed in draw 3 -- source is trait-descriptive by construction",
            "publishability_gloss": (
                "removed in draw 3 -- the WordNet-gloss sieve was miscalibrated "
                "(deleted 'virtuous', kept 'lustful'); no semantic filter replaces it"
            ),
        },
        "filters_kept": [
            "familiarity floor (derived from the primary set; archaism guard only)",
            "primary-set exclusion (exact lowercase match OR Porter stem match)",
        ],
        "familiarity_threshold": freq_threshold,
        "familiarity_threshold_derivation": (
            "5th percentile of wordfreq.word_frequency(w,'en') over the 100 words in "
            "traits_primary.json"
        ),
        "n_primary_words_below_threshold": n_primary_below,
        "primary_words_below_threshold": [t for t, _ in below],
        "familiarity_source": familiarity_source,
        "n_dropped_by_familiarity": len(archaic),
        "dropped_by_familiarity": archaic,
        "exclusion_rule": "exact lowercase match OR Porter stem match against primary set",
        "n_excluded_as_primary_overlap": len(excluded),
        "excluded_as_primary_overlap": [{"word": w, "reason": r} for w, r in excluded],
        "seed": SEED,
        "n_clusters": N_CLUSTERS,
        "discarded_draws": DISCARDED_DRAWS,
        "status": "INCOMPLETE -- embedding step not yet run",
    }
    PROVENANCE_DST.write_text(json.dumps(partial, indent=1))
    log(f"wrote partial provenance -> {PROVENANCE_DST}")

    # -- step 5: embed locally -------------------------------------------
    vecs, model_used, cost_note = embed_local(survivors)

    # -- step 6: k-means ------------------------------------------------------
    section("steps 6-8: cluster, draw, write")
    X = np.asarray(vecs, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # cosine geometry
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    log(f"k-means: {X.shape[0]} words, dim={X.shape[1]}, k={N_CLUSTERS}, seed={SEED}")

    # -- step 7: one random word per cluster, rng(0) --------------------------
    rng = np.random.default_rng(SEED)
    clusters, chosen = {}, []
    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            raise StepFailure(f"cluster {c} is empty")
        # deterministic member order: nearer the centroid first (tie-break only)
        d = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
        order = idx[np.lexsort((idx, d))]
        members = [survivors[i] for i in order]
        pick = members[int(rng.integers(len(members)))]  # uniform random draw
        clusters[f"cluster_{c:02d}"] = {
            "size": len(members),
            "chosen": pick,
            "chosen_rank_by_centroid_distance": members.index(pick),
            "members": members,
        }
        chosen.append((pick, f"cluster_{c:02d}"))

    sizes = [clusters[f"cluster_{c:02d}"]["size"] for c in range(N_CLUSTERS)]
    log(f"cluster sizes: min={min(sizes)} max={max(sizes)} "
        f"mean={np.mean(sizes):.1f} total={sum(sizes)}")
    centroid_ranks = [clusters[k]["chosen_rank_by_centroid_distance"] for k in clusters]
    log(f"chosen-word centroid ranks: {sorted(centroid_ranks)[:12]}... "
        f"(0 would mean 'the centroid word'; n_at_rank0={centroid_ranks.count(0)})")

    # -- step 8: write --------------------------------------------------------
    # Capitalised, to match the schema of traits_primary.json ("Extraverted", ...)
    secondary = [{"trait": w.capitalize(), "factor": "Lexicon", "keyed": "+"} for w, _ in chosen]
    SECONDARY_DST.write_text(json.dumps(secondary, indent=1))
    log(f"wrote -> {SECONDARY_DST}")

    provenance = dict(partial)
    provenance.update({
        "status": "complete",
        "embedding_model": model_used,
        "embedding_dim": int(X.shape[1]),
        "embedding_cost": cost_note,
        "n_embedded": len(survivors),
        "cluster_sizes": sizes,
        "draw_rule": ("numpy.default_rng(0); members ordered by centroid distance "
                      "(tie-break only), then a uniform random index -- a RANDOM "
                      "MEMBER, not the centroid word"),
        "clusters": clusters,
        "chosen": [{"trait": w, "cluster": c} for w, c in chosen],
    })
    PROVENANCE_DST.write_text(json.dumps(provenance, indent=1))
    log(f"wrote -> {PROVENANCE_DST}")

    return secondary, clusters, survivors, primary_lower, primary_stems, freq_threshold


# ----------------------------------------------------------------------------
# step 5: local embedding (sentence-transformers/all-mpnet-base-v2)
# ----------------------------------------------------------------------------
def embed_local(words):
    section(f"step 5: embed locally with {EMBED_MODEL}")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise StepFailure(f"sentence_transformers not available: {e}")

    try:
        model = SentenceTransformer(EMBED_MODEL)
        vecs = model.encode(words, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        raise StepFailure(f"local embedding with {EMBED_MODEL} failed: {e}")

    if len(vecs) != len(words):
        raise StepFailure(f"got {len(vecs)} embeddings for {len(words)} words")
    cost_note = "local model, no API cost"
    log(f"embedded {len(vecs)} words with {EMBED_MODEL} (dim={vecs.shape[1]}); cost: {cost_note}")
    return vecs, EMBED_MODEL, cost_note


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
def verify_secondary(secondary, clusters, survivors, primary_lower, primary_stems, freq_threshold):
    section("VERIFICATION -- secondary")
    ps = PorterStemmer()
    ok = True

    n = len(secondary)
    log(f"exactly 40             : {n} -> {'PASS' if n == 40 else 'FAIL'}")
    ok &= (n == 40)

    words = [e["trait"].lower() for e in secondary]
    dup = sorted({w for w in words if words.count(w) > 1})
    log(f"duplicates within set  : {len(dup)} -> {'PASS' if not dup else 'FAIL ' + str(dup)}")
    ok &= not dup

    exact_hits = [w for w in words if w in primary_lower]
    stem_hits = [w for w in words if ps.stem(w) in primary_stems]
    log(f"overlap w/ primary (exact): {len(exact_hits)} {exact_hits} -> "
        f"{'PASS' if not exact_hits else 'FAIL'}")
    log(f"overlap w/ primary (stem) : {len(stem_hits)} {stem_hits} -> "
        f"{'PASS' if not stem_hits else 'FAIL'}")
    ok &= not exact_hits and not stem_hits

    unfamiliar = [w for w in words if word_frequency(w, "en") < freq_threshold]
    log(f"failed familiarity filter : {len(unfamiliar)} {unfamiliar} -> "
        f"{'PASS' if not unfamiliar else 'FAIL'}")
    ok &= not unfamiliar

    untraceable = [w for w in words
                   if not any(w == c["chosen"] and w in c["members"] for c in clusters.values())]
    log(f"untraceable to a cluster  : {len(untraceable)} {untraceable} -> "
        f"{'PASS' if not untraceable else 'FAIL'}")
    ok &= not untraceable

    from_source = [w for w in words if w not in set(survivors)]
    log(f"not from the source list  : {len(from_source)} {from_source} -> "
        f"{'PASS' if not from_source else 'FAIL'}")
    ok &= not from_source

    schema_bad = [e for e in secondary
                  if set(e) != {"trait", "factor", "keyed"}
                  or e["factor"] != "Lexicon" or e["keyed"] != "+"]
    log(f"schema (Lexicon/+)        : {'PASS' if not schema_bad else 'FAIL'}")
    ok &= not schema_bad

    # none of the words that made the discarded draws unusable may survive
    survivors_flagged = sorted(set(words) & set(PREVIOUSLY_FLAGGED))
    log(f"previously-flagged words surviving     : {len(survivors_flagged)} "
        f"{survivors_flagged} -> {'PASS' if not survivors_flagged else 'FAIL'}")
    ok &= not survivors_flagged

    log()
    log("ALL 40 SECONDARY WORDS:")
    for i in range(0, 40, 5):
        log("  " + "  ".join(f"{e['trait']:<18}" for e in secondary[i:i + 5]))

    # overlap with the discarded draws -- different source, so overlap should be small
    for d in DISCARDED_DRAWS:
        p = HERE / d["file"]
        if not p.exists():
            continue
        old = [e["trait"].strip().lower() for e in json.loads(p.read_text())]
        same = sorted(set(words) & set(old))
        log(f"shared with {d['file']}: {len(same)}/40  {same}")

    log()
    log(f"OVERALL: {'ALL SECONDARY CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    try:
        primary_traits = build_primary()
        secondary, clusters, survivors, plow, pstem, famsrc = build_secondary(primary_traits)
        good = verify_secondary(secondary, clusters, survivors, plow, pstem, famsrc)
        sys.exit(0 if good else 1)
    except StepFailure as e:
        section("STEP FAILED -- STOPPING (no silent substitution)")
        log(str(e))
        sys.exit(1)
