#!/usr/bin/env python
"""
analyse_gradprobe.py -- fit-free analysis of the gradient-content probe.

QUESTION: does a per-document gradient encode the document's CONTENT (which fact it
states) as opposed to its SURFACE (genre)?

Everything here is FIT-FREE: cosine similarities and rank statistics only. No trained
classifier, no fitted linear map, no learned projection. The only "model" is the
count-sketch that upstream applied to the raw gradients.

INFERENCE INSTRUMENT: a permutation test over FACT LABELS that PRESERVES GENRE.
The 79,800 document pairs are not independent (each document sits in 399 of them), so
no naive t-test over pairs is quoted anywhere as inference. See permute_within_genre().

Outputs: results/gradprobe.json, results/gradprobe.md
Self-test:  python analyse_gradprobe.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------------------
# Sketch error budget measured upstream (meta.json -> verification.sketch_fidelity_on_real
# _gradients, measured against exact gradients on k=12 real documents). We refuse to
# interpret any cosine DIFFERENCE smaller than the relevant rms figure.
# --------------------------------------------------------------------------------------
NOISE_FLOOR = {
    "pooled": 0.009,     # rms error on a per-pair pooled cosine
    "per_layer": 0.030,  # rms error on a per-pair per-layer cosine
    "per_type": 0.030,   # rms error on a per-pair per-type cosine
    "max_pair": 0.120,   # worst single-pair error observed (per-layer)
}


# ======================================================================================
# statistics implemented directly (no scipy on this box); each is verified in --selftest
# against a published reference value.
# ======================================================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_sf(x: float) -> float:
    """Upper tail 1 - Phi(x), computed with erfc for tail accuracy."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation. NOTE: we never quote a p-value for this over pairs -- the
    pairs are dependent, so only the point estimate is meaningful."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    den = math.sqrt(float(xc @ xc) * float(yc @ yc))
    if den == 0.0:
        return float("nan")
    return float(xc @ yc / den)


def ks_uniform_stat(p: np.ndarray) -> float:
    """Two-sided one-sample Kolmogorov-Smirnov statistic D against Uniform(0,1)."""
    p = np.sort(np.asarray(p, dtype=np.float64))
    n = p.size
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(i / n - p))
    d_minus = float(np.max(p - (i - 1) / n))
    return max(d_plus, d_minus)


def ks_crit_05(n: int) -> float:
    """Asymptotic 5% critical value for the one-sample KS statistic: 1.35810 / sqrt(n)."""
    return 1.35810 / math.sqrt(n)


def describe(v: np.ndarray) -> dict:
    """Full distribution summary of a set of cosines. The 'se' is the NAIVE
    sd/sqrt(n); it is descriptive only and is NOT used for inference for any group whose
    pairs share documents."""
    v = np.asarray(v, dtype=np.float64)
    n = int(v.size)
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "n": n,
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
        "se_naive": float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "min": float(q[0]), "p01": float(q[1]), "p05": float(q[2]),
        "q25": float(q[3]), "median": float(q[4]), "q75": float(q[5]),
        "p95": float(q[6]), "p99": float(q[7]), "max": float(q[8]),
    }


def hist(v: np.ndarray, lo: float, hi: float, nbins: int = 24) -> dict:
    counts, edges = np.histogram(np.asarray(v, dtype=np.float64), bins=nbins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


# ======================================================================================
# design: labels, pair masks, permutations
# ======================================================================================

class Design:
    """Fact/genre design shared by every readout and both modes.

    Documents are indexed by ROW of the sketch arrays. Row i corresponds to
    doc_id_order[i] from meta.json -- NOT to doc_id i.
    """

    def __init__(self, fact: np.ndarray, genre: np.ndarray, doc_ids: np.ndarray,
                 tokens: np.ndarray | None = None):
        self.n = int(fact.size)
        self.fact = fact.astype(np.int64)
        self.genre = genre.astype(np.int64)
        self.doc_ids = doc_ids.astype(np.int64)
        self.tokens = tokens

        iu, ju = np.triu_indices(self.n, 1)
        self.iu, self.ju = iu, ju
        self.n_pairs = int(iu.size)

        self.same_fact = self.fact[iu] == self.fact[ju]
        self.same_genre = self.genre[iu] == self.genre[ju]
        self.m_SF = self.same_fact                      # same fact (all cross-genre by design)
        self.m_DF_SG = (~self.same_fact) & self.same_genre
        self.m_DF_DG = (~self.same_fact) & (~self.same_genre)
        self.m_cross_genre = ~self.same_genre

        # the true matching: 2 docs per fact -> 200 ordered (a,b) pairs
        byf = defaultdict(list)
        for r in range(self.n):
            byf[int(self.fact[r])].append(r)
        sizes = set(len(v) for v in byf.values())
        assert sizes == {2}, f"expected exactly 2 docs per fact, got sizes {sizes}"
        pa, pb = [], []
        for f in sorted(byf):
            a, b = byf[f]
            pa.append(a)
            pb.append(b)
        self.pair_a = np.array(pa, dtype=np.int64)
        self.pair_b = np.array(pb, dtype=np.int64)
        self.n_facts = self.pair_a.size

        assert bool(np.all(self.genre[self.pair_a] != self.genre[self.pair_b])), \
            "design violated: some same-fact pair shares a genre"

        # genre blocks, used by the permutation
        self.genre_blocks = [np.where(self.genre == g)[0] for g in range(int(self.genre.max()) + 1)]
        # multiset of (genre_a, genre_b) over the true matching -- must be invariant
        self.genre_pair_signature = Counter(
            tuple(sorted((int(self.genre[a]), int(self.genre[b]))))
            for a, b in zip(self.pair_a, self.pair_b)
        )

    def permute_within_genre(self, nperm: int, rng: np.random.Generator):
        """THE PERMUTATION. We permute the assignment of DOCUMENTS TO FACTS.

        Concretely: draw a bijection sigma of documents onto documents that maps every
        document to a document OF THE SAME GENRE (a uniform permutation inside each genre
        block). The true matching {(a_k, b_k)} is carried through sigma to a null matching
        {(sigma(a_k), sigma(b_k))}.

        Because sigma is genre-preserving and a bijection:
          * every document keeps its own genre and its own sketch (documents are fixed
            objects; only the fact labelling moves);
          * the null matching is again a perfect matching on all 400 documents;
          * the multiset of (genre_a, genre_b) combinations across the 200 null pairs is
            IDENTICAL to the observed one, so every null pair is still cross-genre and the
            genre composition of the "same-fact" set is exactly preserved;
          * which documents state the same FACT is destroyed.

        So the permutation isolates fact identity with genre held fixed by construction.
        """
        sigma = np.empty((nperm, self.n), dtype=np.int32)
        for blk in self.genre_blocks:
            m = blk.size
            if m == 0:
                continue
            order = np.argsort(rng.random((nperm, m)), axis=1)
            sigma[:, blk] = blk[order]
        return sigma[:, self.pair_a], sigma[:, self.pair_b]

    def permute_genre_labels(self, nperm: int, rng: np.random.Generator):
        """Secondary permutation, used ONLY to put a z on the GENRE effect: shuffle genre
        labels across documents (fact structure untouched). Returns (nperm, n) label
        matrix."""
        order = np.argsort(rng.random((nperm, self.n)), axis=1)
        return self.genre[order]


# ======================================================================================
# per-readout analysis
# ======================================================================================

def cosine_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Xn = X / nrm
    C = Xn @ Xn.T
    np.clip(C, -1.0, 1.0, out=C)
    return C


def three_way_contrast(C: np.ndarray, d: Design, perm_a: np.ndarray, perm_b: np.ndarray,
                       want_hist: bool = False) -> dict:
    cos = C[d.iu, d.ju]
    sf = cos[d.m_SF]
    dfsg = cos[d.m_DF_SG]
    dfdg = cos[d.m_DF_DG]

    # sum over ALL cross-genre pairs is invariant under the permutation; the null
    # matching is always a subset of the cross-genre pairs, so the two group means are a
    # closed-form function of the matching's cosine sum.
    S_cross = float(cos[d.m_cross_genre].sum())
    N_cross = int(d.m_cross_genre.sum())
    K = d.n_facts

    def stat_from_sum(s):
        return s / K - (S_cross - s) / (N_cross - K)

    T_obs = stat_from_sum(float(sf.sum()))

    # permutation null
    nperm = perm_a.shape[0]
    T_null = np.empty(nperm, dtype=np.float64)
    step = 4000
    for s0 in range(0, nperm, step):
        s1 = min(nperm, s0 + step)
        sums = C[perm_a[s0:s1], perm_b[s0:s1]].sum(axis=1)
        T_null[s0:s1] = stat_from_sum(sums)

    mu, sd = float(T_null.mean()), float(T_null.std(ddof=1))
    z = (T_obs - mu) / sd if sd > 0 else float("nan")
    ge = int(np.sum(T_null >= T_obs))
    p_one = (1 + ge) / (nperm + 1)
    p_two = (1 + int(np.sum(np.abs(T_null - mu) >= abs(T_obs - mu)))) / (nperm + 1)

    # genre effect in the same units
    G_obs = float(dfsg.mean() - dfdg.mean())

    out = {
        "same_fact": describe(sf),
        "diff_fact_same_genre": describe(dfsg),
        "diff_fact_diff_genre": describe(dfdg),
        "fact_effect": {
            "statistic": "mean(SAME-FACT) - mean(DIFF-FACT/DIFF-GENRE)  [genre controlled: both sets are cross-genre]",
            "observed": T_obs,
            "perm_null_mean": mu,
            "perm_null_sd": sd,
            "z": z,
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "n_perm": nperm,
            "p_floor": 1.0 / (nperm + 1),
            "n_perm_ge_obs": ge,
        },
        "genre_effect": {
            "statistic": "mean(DIFF-FACT/SAME-GENRE) - mean(DIFF-FACT/DIFF-GENRE)",
            "observed": G_obs,
        },
        "fact_vs_genre_ratio": (T_obs / G_obs) if G_obs != 0 else float("nan"),
    }
    if want_hist:
        lo = float(min(sf.min(), dfsg.min(), dfdg.min()))
        hi = float(max(sf.max(), dfsg.max(), dfdg.max()))
        out["hist"] = {
            "range": [lo, hi],
            "same_fact": hist(sf, lo, hi),
            "diff_fact_same_genre": hist(dfsg, lo, hi),
            "diff_fact_diff_genre": hist(dfdg, lo, hi),
        }
    return out


def genre_effect_permutation(C: np.ndarray, d: Design, glabels: np.ndarray) -> dict:
    """z for the GENRE effect, by shuffling genre labels over documents."""
    cos = C[d.iu, d.ju]
    nf = ~d.same_fact
    cos_df = cos[nf]
    iu_df, ju_df = d.iu[nf], d.ju[nf]
    obs = float(cos_df[d.same_genre[nf]].mean() - cos_df[~d.same_genre[nf]].mean())

    nperm = glabels.shape[0]
    S = float(cos_df.sum())
    N = cos_df.size
    vals = np.empty(nperm, dtype=np.float64)
    for k in range(nperm):
        gl = glabels[k]
        m = gl[iu_df] == gl[ju_df]
        s_same = float(cos_df[m].sum())
        n_same = int(m.sum())
        vals[k] = s_same / n_same - (S - s_same) / (N - n_same)
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    return {
        "observed": obs,
        "perm_null_mean": mu,
        "perm_null_sd": sd,
        "z": (obs - mu) / sd if sd > 0 else float("nan"),
        "p_one_sided": (1 + int(np.sum(vals >= obs))) / (nperm + 1),
        "n_perm": nperm,
        "note": "genre labels shuffled over documents; statistic computed on DIFF-FACT pairs only",
    }


def rank_matrices(C: np.ndarray, d: Design):
    """R[i,j] = rank of j when all 399 non-self documents are ranked by cosine to i
    (1 = most similar). R_dg[i,j] = same, but the candidate pool is only documents whose
    genre differs from i's."""
    n = d.n
    Cm = C.copy()
    np.fill_diagonal(Cm, -np.inf)
    order = np.argsort(-Cm, axis=1, kind="stable")
    R = np.empty_like(order)
    np.put_along_axis(R, order, np.arange(1, n + 1)[None, :].repeat(n, 0), axis=1)

    same_g = d.genre[:, None] == d.genre[None, :]
    Cd = C.copy()
    Cd[same_g] = -np.inf
    order_d = np.argsort(-Cd, axis=1, kind="stable")
    R_dg = np.empty_like(order_d)
    np.put_along_axis(R_dg, order_d, np.arange(1, n + 1)[None, :].repeat(n, 0), axis=1)
    pool_dg = (~same_g).sum(axis=1)
    return R, R_dg, pool_dg


def retrieval(C: np.ndarray, d: Design, perm_a: np.ndarray, perm_b: np.ndarray) -> dict:
    R, R_dg, pool_dg = rank_matrices(C, d)
    n = d.n
    a, b = d.pair_a, d.pair_b
    # 400 directed queries: each document looks for its partner
    ranks = np.concatenate([R[a, b], R[b, a]]).astype(np.float64)
    ranks_dg = np.concatenate([R_dg[a, b], R_dg[b, a]]).astype(np.float64)
    pool = np.concatenate([pool_dg[a], pool_dg[b]]).astype(np.float64)

    def block(rk, chance_pool):
        out = {
            "n_queries": int(rk.size),
            "median_rank": float(np.median(rk)),
            "mean_rank": float(rk.mean()),
            "mrr": float(np.mean(1.0 / rk)),
        }
        for k in (1, 5, 10, 50):
            out[f"top{k}"] = float(np.mean(rk <= k))
            out[f"top{k}_chance"] = float(np.mean(np.minimum(k, chance_pool) / chance_pool))
        # chance MRR: rank uniform on 1..pool  =>  E[1/rank] = H_pool / pool
        harm = {int(p): float(np.sum(1.0 / np.arange(1, int(p) + 1)) / p)
                for p in np.unique(chance_pool)}
        out["mrr_chance"] = float(np.mean([harm[int(p)] for p in chance_pool]))
        out["median_rank_chance"] = float(np.mean((chance_pool + 1.0) / 2.0))
        return out

    full_pool = np.full(ranks.size, n - 1, dtype=np.float64)
    res = {
        "all_candidates": block(ranks, full_pool),
        "diff_genre_only": block(ranks_dg, pool),
        "diff_genre_pool_sizes": {"min": int(pool.min()), "max": int(pool.max()),
                                  "mean": float(pool.mean())},
    }

    # permutation null on MRR and top-1, using the same genre-preserving fact permutation
    nperm = perm_a.shape[0]
    mrr_null = np.empty(nperm)
    top1_null = np.empty(nperm)
    step = 4000
    for s0 in range(0, nperm, step):
        s1 = min(nperm, s0 + step)
        pa, pb = perm_a[s0:s1], perm_b[s0:s1]
        rr = np.concatenate([R[pa, pb], R[pb, pa]], axis=1).astype(np.float64)
        mrr_null[s0:s1] = np.mean(1.0 / rr, axis=1)
        top1_null[s0:s1] = np.mean(rr <= 1, axis=1)
    obs_mrr = res["all_candidates"]["mrr"]
    mu, sd = float(mrr_null.mean()), float(mrr_null.std(ddof=1))
    res["mrr_permutation"] = {
        "observed": obs_mrr, "perm_null_mean": mu, "perm_null_sd": sd,
        "z": (obs_mrr - mu) / sd if sd > 0 else float("nan"),
        "p_one_sided": (1 + int(np.sum(mrr_null >= obs_mrr))) / (nperm + 1),
        "n_perm": nperm,
    }
    obs_t1 = res["all_candidates"]["top1"]
    mu1, sd1 = float(top1_null.mean()), float(top1_null.std(ddof=1))
    res["top1_permutation"] = {
        "observed": obs_t1, "perm_null_mean": mu1, "perm_null_sd": sd1,
        "z": (obs_t1 - mu1) / sd1 if sd1 > 0 else float("nan"),
        "p_one_sided": (1 + int(np.sum(top1_null >= obs_t1))) / (nperm + 1),
        "n_perm": nperm,
    }
    return res


def norm_control(C: np.ndarray, d: Design, norms: np.ndarray,
                 perm_a: np.ndarray, perm_b: np.ndarray, nbins: int = 10,
                 tokens: np.ndarray | None = None) -> dict:
    """1) correlation of pair cosine with |norm_i - norm_j|
       2) the headline contrast recomputed WITHIN |delta-norm| strata (norm-matched)."""
    cos = C[d.iu, d.ju]
    dn = np.abs(norms[d.iu] - norms[d.ju])
    r_all = pearson_r(cos, dn)
    cg = d.m_cross_genre
    r_cross = pearson_r(cos[cg], dn[cg])

    out = {
        "cos_vs_absdiff_norm_pearson_all_pairs": r_all,
        "cos_vs_absdiff_norm_pearson_cross_genre_pairs": r_cross,
        "same_fact_absdiff_norm_mean": float(dn[d.m_SF].mean()),
        "diff_fact_diff_genre_absdiff_norm_mean": float(dn[d.m_DF_DG].mean()),
        "note": "cosine is already scale-invariant per document; |delta-norm| can only act "
                "as a proxy for length/content, which is what the stratified test removes.",
    }
    if tokens is not None:
        dt = np.abs(tokens[d.iu].astype(np.float64) - tokens[d.ju].astype(np.float64))
        out["cos_vs_absdiff_tokens_pearson_all_pairs"] = pearson_r(cos, dt)
        out["same_fact_absdiff_tokens_mean"] = float(dt[d.m_SF].mean())
        out["diff_fact_diff_genre_absdiff_tokens_mean"] = float(dt[d.m_DF_DG].mean())

    # --- stratified (norm-matched) contrast -------------------------------------------
    # bins defined on |delta-norm| quantiles of the cross-genre pairs, so the SAME-FACT
    # pairs and their DIFF-FACT/DIFF-GENRE comparators sit in the same strata.
    edges = np.quantile(dn[cg], np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    bin_of_pair = np.digitize(dn, edges) - 1
    np.clip(bin_of_pair, 0, nbins - 1, out=bin_of_pair)

    # full n x n bin lookup so permuted pairs can be binned too
    DN = np.abs(norms[:, None] - norms[None, :])
    BIN = np.clip(np.digitize(DN, edges) - 1, 0, nbins - 1)

    dfdg_sum = np.zeros(nbins)
    dfdg_cnt = np.zeros(nbins)
    np.add.at(dfdg_sum, bin_of_pair[d.m_DF_DG], cos[d.m_DF_DG])
    np.add.at(dfdg_cnt, bin_of_pair[d.m_DF_DG], 1.0)
    dfdg_mean = np.where(dfdg_cnt > 0, dfdg_sum / np.maximum(dfdg_cnt, 1), np.nan)

    def stratified(bins_k, cos_k):
        """bins_k, cos_k: (nperm, K). Returns (nperm,) weighted within-stratum contrast."""
        npm = bins_k.shape[0]
        num = np.zeros(npm)
        wsum = np.zeros(npm)
        for bb in range(nbins):
            m = bins_k == bb
            c = m.sum(axis=1).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                sm = np.where(c > 0, (cos_k * m).sum(axis=1) / np.maximum(c, 1), 0.0)
            ok = (c > 0) & np.isfinite(dfdg_mean[bb])
            num += np.where(ok, c * (sm - dfdg_mean[bb]), 0.0)
            wsum += np.where(ok, c, 0.0)
        return num / np.maximum(wsum, 1e-12)

    obs_bins = BIN[d.pair_a, d.pair_b][None, :]
    obs_cos = C[d.pair_a, d.pair_b][None, :]
    T_obs = float(stratified(obs_bins, obs_cos)[0])

    nperm = perm_a.shape[0]
    T_null = np.empty(nperm)
    step = 2000
    for s0 in range(0, nperm, step):
        s1 = min(nperm, s0 + step)
        pa, pb = perm_a[s0:s1], perm_b[s0:s1]
        T_null[s0:s1] = stratified(BIN[pa, pb], C[pa, pb])
    mu, sd = float(T_null.mean()), float(T_null.std(ddof=1))
    out["norm_matched_contrast"] = {
        "statistic": "|delta-norm|-stratified mean(SAME-FACT) - mean(DIFF-FACT/DIFF-GENRE), "
                     "weighted by same-fact count per stratum",
        "n_bins": nbins,
        "observed": T_obs,
        "perm_null_mean": mu,
        "perm_null_sd": sd,
        "z": (T_obs - mu) / sd if sd > 0 else float("nan"),
        "p_one_sided": (1 + int(np.sum(T_null >= T_obs))) / (nperm + 1),
        "n_perm": nperm,
        "same_fact_per_bin": [int(x) for x in np.bincount(bin_of_pair[d.m_SF], minlength=nbins)],
        "bin_edges_absdiff_norm": [float(e) for e in edges],
    }
    return out


# ======================================================================================
# readout assembly
# ======================================================================================

def build_readouts(pooled, per_layer, per_type, type_order, norms):
    """Yields (name, kind, cosine_matrix, norm_vector_for_that_readout)."""
    n_layers = per_layer.shape[1]
    n_types = per_type.shape[1]
    total_norm = norms["total"]
    layer_norm = norms["per_layer"]
    type_norm = norms["per_type"]

    yield ("pooled", "pooled", cosine_matrix(pooled), total_norm)

    layer_C = []
    for l in range(n_layers):
        C = cosine_matrix(per_layer[:, l, :])
        layer_C.append(C)
        yield (f"layer{l:02d}", "per_layer", C, layer_norm[:, l])

    type_C = []
    for t in range(n_types):
        C = cosine_matrix(per_type[:, t, :])
        type_C.append(C)
        yield (f"type:{type_order[t]}", "per_type", C, type_norm[:, t])

    # aggregates
    yield ("mean_over_layers", "aggregate", np.mean(layer_C, axis=0), total_norm)
    yield ("mean_over_types", "aggregate", np.mean(type_C, axis=0), total_norm)
    yield ("concat_layers", "aggregate",
           cosine_matrix(per_layer.reshape(per_layer.shape[0], -1)), total_norm)
    yield ("concat_types", "aggregate",
           cosine_matrix(per_type.reshape(per_type.shape[0], -1)), total_norm)


def floor_for(kind: str, budget: dict | None = None) -> float:
    b = budget or NOISE_FLOOR
    return {"pooled": b["pooled"], "per_layer": b["per_layer"], "per_type": b["per_type"],
            "aggregate": b["pooled"]}[kind]


def budget_from_meta(meta: dict) -> dict:
    """Take the noise floor from the run's OWN measured sketch fidelity where available,
    falling back to the stated per-pair rms figures. We use the larger of the two so the
    gate is never loosened relative to the brief."""
    f = (meta.get("verification", {}) or {}).get("sketch_fidelity_on_real_gradients") or {}
    out = dict(NOISE_FLOOR)
    for key, src in (("pooled", "pooled"), ("per_layer", "per_layer"), ("per_type", "per_type")):
        if src in f and "rms_err" in f[src]:
            out[key] = max(float(f[src]["rms_err"]), NOISE_FLOOR[key])
    mx = [f[s]["max_abs_err"] for s in ("pooled", "per_layer", "per_type")
          if s in f and "max_abs_err" in f[s]]
    if mx:
        out["max_pair"] = max(max(float(m) for m in mx), NOISE_FLOOR["max_pair"])
    out["measured"] = {k: {kk: f[k][kk] for kk in ("rms_err", "max_abs_err", "pearson_r", "mean_err")
                           if kk in f.get(k, {})} for k in ("pooled", "per_layer", "per_type")
                       if k in f}
    return out


# ======================================================================================
# data loading
# ======================================================================================

def load_mode(root: str, mode: str, docs_by_id: dict):
    mdir = os.path.join(root, "out", mode)
    meta = json.load(open(os.path.join(mdir, "meta.json")))
    norms_raw = json.load(open(os.path.join(mdir, "norms.json")))
    pooled = np.load(os.path.join(mdir, "sketches_pooled.npy"))
    per_layer = np.load(os.path.join(mdir, "sketches_per_layer.npy"))
    per_type = np.load(os.path.join(mdir, "sketches_per_type.npy"))

    order = list(meta["doc_id_order"])
    n = len(order)
    assert pooled.shape[0] == n and per_layer.shape[0] == n and per_type.shape[0] == n
    assert sorted(order) == list(range(n)), "doc_id_order is not a permutation of 0..n-1"
    assert list(norms_raw["doc_id"]) == order, "norms.json row order != sketch row order"
    assert meta.get("shuffled") is False
    assert meta["verification"].get("attribution_step_i_is_doc_i", False)

    # labels taken from docs.jsonl, keyed by doc_id, laid out in SKETCH ROW ORDER
    fact_ids = np.array([docs_by_id[i]["fact_id"] for i in order], dtype=np.int64)
    genres = [docs_by_id[i]["genre"] for i in order]
    # cross-check against the labels the training run itself recorded
    dm = {d["doc_id"]: d for d in meta["doc_meta"]}
    mism = [i for i in order if dm[i]["fact_id"] != docs_by_id[i]["fact_id"]
            or dm[i]["genre"] != docs_by_id[i]["genre"]]
    assert not mism, f"docs.jsonl disagrees with meta.doc_meta for {len(mism)} docs"

    gmap = {g: k for k, g in enumerate(sorted(set(genres)))}
    fmap = {f: k for k, f in enumerate(sorted(set(int(x) for x in fact_ids)))}
    fact = np.array([fmap[int(f)] for f in fact_ids], dtype=np.int64)
    genre = np.array([gmap[g] for g in genres], dtype=np.int64)
    tokens = np.array(meta["tokenisation"]["n_tokens"], dtype=np.int64)

    norms = {
        "total": np.array(norms_raw["total"], dtype=np.float64),
        "per_layer": np.array(norms_raw["per_layer"], dtype=np.float64),
        "per_type": np.array(norms_raw["per_type"], dtype=np.float64),
    }
    d = Design(fact, genre, np.array(order), tokens)
    return dict(meta=meta, pooled=pooled, per_layer=per_layer, per_type=per_type,
                norms=norms, design=d, genre_names=sorted(set(genres)),
                type_order=meta["type_order"], mdir=mdir)


# ======================================================================================
# self-test: synthetic ground truth
# ======================================================================================

def synth_sketches(d: Design, rng, fact_amp: float, genre_amp: float, noise_amp: float,
                   dim: int = 256) -> np.ndarray:
    F = rng.standard_normal((int(d.fact.max()) + 1, dim))
    G = rng.standard_normal((int(d.genre.max()) + 1, dim))
    X = noise_amp * rng.standard_normal((d.n, dim))
    if genre_amp:
        X += genre_amp * G[d.genre]
    if fact_amp:
        X += fact_amp * F[d.fact]
    return X


def run_selftest(d: Design, seed: int = 20260812, nperm: int = 10000,
                 n_null_reps: int = 200, null_nperm: int = 2000, verbose=True) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    log = []

    def say(s):
        log.append(s)
        if verbose:
            print(s, flush=True)

    # --- reference checks on the hand-rolled statistics -------------------------------
    ref = {
        "norm_cdf(1.959964)": (norm_cdf(1.959963985), 0.975),
        "norm_cdf(2.575829)": (norm_cdf(2.5758293035), 0.995),
        "norm_cdf(0)": (norm_cdf(0.0), 0.5),
        "norm_cdf(1.6448536)": (norm_cdf(1.6448536269), 0.95),
    }
    say("[selftest] statistic implementations vs published values")
    ok_all = True
    for k, (got, want) in ref.items():
        ok = abs(got - want) < 1e-9
        ok_all &= ok
        say(f"   {k:24s} = {got:.10f}  published {want}  {'OK' if ok else 'FAIL'}")
    # Pearson r against Anscombe quartet set I (published r = 0.816)
    ax = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    ay = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    r_ans = pearson_r(ax, ay)
    ok = abs(r_ans - 0.8164205) < 1e-6
    ok_all &= ok
    say(f"   pearson_r(Anscombe I)    = {r_ans:.7f}  published 0.8164205  {'OK' if ok else 'FAIL'}")
    # KS statistic on a deterministic sample with a hand-computable answer
    # p = (.1,.2,.5,.7,.9): D+ = max(i/n - p_i) = 0.2 ; D- = max(p_i - (i-1)/n) = 0.1
    ks_d = ks_uniform_stat(np.array([0.1, 0.2, 0.5, 0.7, 0.9]))
    ok = abs(ks_d - 0.2) < 1e-12
    ok_all &= ok
    say(f"   ks_uniform_stat(det)     = {ks_d:.7f}  hand-computed 0.2  {'OK' if ok else 'FAIL'}")
    say(f"   KS 5% critical value n=200 = {ks_crit_05(200):.5f}  (1.35810/sqrt(n), published)")
    out["reference_checks"] = {"all_passed": bool(ok_all), "pearson_anscombe_I": r_ans,
                               "ks_det": ks_d, "ks_crit_05_n200": ks_crit_05(200)}

    # --- permutation preserves genre structure ----------------------------------------
    say("")
    say("[selftest] does the permutation preserve genre structure?")
    pa, pb = d.permute_within_genre(200, rng)
    genre_ok = bool(np.all(d.genre[pa] == d.genre[d.pair_a][None, :]) and
                    np.all(d.genre[pb] == d.genre[d.pair_b][None, :]))
    sigs_equal = True
    for k in range(200):
        sig = Counter(tuple(sorted((int(d.genre[a]), int(d.genre[b]))))
                      for a, b in zip(pa[k], pb[k]))
        if sig != d.genre_pair_signature:
            sigs_equal = False
            break
    perfect_matching = all(np.array_equal(np.sort(np.concatenate([pa[k], pb[k]])),
                                          np.arange(d.n)) for k in range(50))
    cross = bool(np.all(d.genre[pa] != d.genre[pb]))
    frac_true = float(np.mean([
        np.mean([(d.fact[a] == d.fact[b]) for a, b in zip(pa[k], pb[k])]) for k in range(200)
    ]))
    say(f"   genre of every permuted doc == genre of the doc it replaces : {genre_ok}")
    say(f"   multiset of (genre_a,genre_b) over 200 null pairs identical : {sigs_equal}")
    say(f"   null matching is still a perfect matching on all 400 docs   : {perfect_matching}")
    say(f"   every null pair is still cross-genre                        : {cross}")
    say(f"   fraction of null pairs that are accidentally TRUE partners  : {frac_true:.4f} "
        f"(makes the test slightly conservative, not anti-conservative)")
    out["permutation_structure"] = {
        "genre_preserved": genre_ok, "genre_pair_signature_identical": sigs_equal,
        "perfect_matching_preserved": perfect_matching, "all_pairs_cross_genre": cross,
        "frac_null_pairs_truly_same_fact": frac_true,
    }

    # --- signal / null recovery --------------------------------------------------------
    say("")
    say("[selftest] recovery on synthetic sketches with known ground truth "
        f"({nperm} permutations)")
    cases = {
        "SIGNAL  fact+genre+noise": dict(fact_amp=0.45, genre_amp=0.50, noise_amp=1.0),
        "SIGNAL  weak fact       ": dict(fact_amp=0.15, genre_amp=0.50, noise_amp=1.0),
        "NULL    genre+noise only": dict(fact_amp=0.00, genre_amp=0.50, noise_amp=1.0),
        "NULL    pure noise      ": dict(fact_amp=0.00, genre_amp=0.00, noise_amp=1.0),
    }
    pa_big, pb_big = d.permute_within_genre(nperm, rng)
    glab = d.permute_genre_labels(500, rng)
    say(f"   {'case':26s} {'fact d':>8s} {'z_fact':>9s} {'p_fact':>9s} "
        f"{'genre d':>8s} {'z_genre':>8s} {'MRR':>7s} {'MRRdg':>7s} {'top1':>7s} {'med.rank':>9s}")
    cres = {}
    for name, kw in cases.items():
        X = synth_sketches(d, rng, dim=256, **kw)
        C = cosine_matrix(X)
        tw = three_way_contrast(C, d, pa_big, pb_big)
        ge = genre_effect_permutation(C, d, glab)
        rt = retrieval(C, d, pa_big[:2000], pb_big[:2000])
        say(f"   {name:26s} {tw['fact_effect']['observed']:8.4f} "
            f"{tw['fact_effect']['z']:9.2f} {tw['fact_effect']['p_one_sided']:9.5f} "
            f"{tw['genre_effect']['observed']:8.4f} {ge['z']:8.2f} "
            f"{rt['all_candidates']['mrr']:7.4f} {rt['diff_genre_only']['mrr']:7.4f} "
            f"{rt['all_candidates']['top1']:7.4f} "
            f"{rt['all_candidates']['median_rank']:9.1f}")
        cres[name.strip()] = {
            "params": kw,
            "fact_effect": {k: tw["fact_effect"][k] for k in
                            ("observed", "z", "p_one_sided", "perm_null_mean", "perm_null_sd")},
            "genre_effect_observed": tw["genre_effect"]["observed"],
            "genre_effect_z": ge["z"],
            "mrr": rt["all_candidates"]["mrr"], "top1": rt["all_candidates"]["top1"],
            "median_rank": rt["all_candidates"]["median_rank"],
            "mrr_diff_genre_only": rt["diff_genre_only"]["mrr"],
        }
    out["cases"] = cres
    say(f"   chance: MRR={0.01646:.5f} (MRRdg~{0.0180:.4f}) top1={1/399:.5f} median rank=200.0")
    say("   note the NULL genre+noise row: overall MRR (0.006) sits BELOW chance because "
        "genre similarity outranks the cross-genre partner -- exactly the artefact the "
        "different-genre-only pool controls for.")

    # --- is the permutation p uniform under the null? ----------------------------------
    say("")
    say(f"[selftest] null p-value uniformity: {n_null_reps} independent null datasets "
        f"(genre+noise, no fact component), {null_nperm} permutations each")
    ps = np.empty(n_null_reps)
    zs = np.empty(n_null_reps)
    pa_s, pb_s = pa_big[:null_nperm], pb_big[:null_nperm]
    for i in range(n_null_reps):
        X = synth_sketches(d, rng, dim=256, fact_amp=0.0, genre_amp=0.5, noise_amp=1.0)
        C = cosine_matrix(X)
        tw = three_way_contrast(C, d, pa_s, pb_s)
        ps[i] = tw["fact_effect"]["p_one_sided"]
        zs[i] = tw["fact_effect"]["z"]
    D = ks_uniform_stat(ps)
    crit = ks_crit_05(n_null_reps)
    say(f"   mean p = {ps.mean():.4f} (want 0.5)   frac p<0.05 = {np.mean(ps < 0.05):.4f} "
        f"(want 0.05)   frac p<0.10 = {np.mean(ps < 0.10):.4f} (want 0.10)")
    say(f"   mean z = {zs.mean():+.4f} (want 0)    sd z = {zs.std(ddof=1):.4f} (want ~1)")
    say(f"   KS D vs Uniform(0,1) = {D:.4f}   5% critical value = {crit:.4f}   "
        f"{'UNIFORM (not rejected)' if D < crit else 'REJECTED'}")
    out["null_uniformity"] = {
        "n_datasets": n_null_reps, "n_perm_each": null_nperm,
        "mean_p": float(ps.mean()), "frac_p_lt_05": float(np.mean(ps < 0.05)),
        "frac_p_lt_10": float(np.mean(ps < 0.10)),
        "mean_z": float(zs.mean()), "sd_z": float(zs.std(ddof=1)),
        "ks_D": D, "ks_crit_05": crit, "uniform_not_rejected": bool(D < crit),
    }

    # --- genre effect is invariant under the FACT permutation --------------------------
    say("")
    say("[selftest] the fact permutation destroys the fact effect but leaves genre alone")
    X = synth_sketches(d, rng, dim=256, fact_amp=0.45, genre_amp=0.50, noise_amp=1.0)
    C = cosine_matrix(X)
    cos = C[d.iu, d.ju]
    nf = ~d.same_fact
    # the DIFF-FACT/SAME-GENRE set is bit-identical under a within-genre permutation,
    # because no null pair is ever same-genre. Show the genre contrast across permutations.
    gvals = []
    fvals = []
    S_cross = float(cos[d.m_cross_genre].sum())
    N_cross = int(d.m_cross_genre.sum())
    sg_mean = float(cos[d.m_DF_SG].mean())
    for k in range(200):
        s = float(C[pa_big[k], pb_big[k]].sum())
        dd_mean = (S_cross - s) / (N_cross - d.n_facts)
        gvals.append(sg_mean - dd_mean)
        fvals.append(s / d.n_facts - dd_mean)
    gvals = np.array(gvals)
    fvals = np.array(fvals)
    g_obs = float(cos[d.m_DF_SG].mean() - cos[d.m_DF_DG].mean())
    f_obs = float(cos[d.m_SF].mean() - cos[d.m_DF_DG].mean())
    say(f"   genre effect : observed {g_obs:+.5f}  across 200 permutations "
        f"{gvals.mean():+.5f} +/- {gvals.std(ddof=1):.2e}  (drift {abs(gvals.mean()-g_obs):.2e})")
    say(f"   fact  effect : observed {f_obs:+.5f}  across 200 permutations "
        f"{fvals.mean():+.5f} +/- {fvals.std(ddof=1):.2e}  -> destroyed")
    say(f"   sd(genre)/sd(fact) = {gvals.std(ddof=1)/fvals.std(ddof=1):.4f} "
        f"(genre is ~invariant; it moves only because 200 of {N_cross} cross-genre pairs "
        f"change membership)")
    out["genre_invariance"] = {
        "genre_observed": g_obs, "genre_perm_mean": float(gvals.mean()),
        "genre_perm_sd": float(gvals.std(ddof=1)),
        "fact_observed": f_obs, "fact_perm_mean": float(fvals.mean()),
        "fact_perm_sd": float(fvals.std(ddof=1)),
        "same_genre_pair_set_bit_identical_under_permutation": True,
    }
    out["log"] = log
    return out


# ======================================================================================
# main analysis
# ======================================================================================

def analyse_mode(data: dict, nperm: int, seed: int, verbose=True) -> dict:
    d: Design = data["design"]
    rng = np.random.default_rng(seed)
    perm_a, perm_b = d.permute_within_genre(nperm, rng)
    glab = d.permute_genre_labels(500, rng)

    readouts = {}
    order = []
    headline = {"pooled", "mean_over_layers", "concat_layers"}
    budget = budget_from_meta(data["meta"])
    t0 = time.time()
    for name, kind, C, nrm in build_readouts(data["pooled"], data["per_layer"],
                                             data["per_type"], data["type_order"],
                                             data["norms"]):
        r = {"kind": kind, "noise_floor": floor_for(kind, budget)}
        r["contrast"] = three_way_contrast(C, d, perm_a, perm_b,
                                           want_hist=(name in headline))
        r["retrieval"] = retrieval(C, d, perm_a[:4000], perm_b[:4000])
        r["norm_control"] = norm_control(C, d, nrm, perm_a[:5000], perm_b[:5000],
                                         tokens=d.tokens if name in headline else None)
        r["genre_effect_perm"] = genre_effect_permutation(C, d, glab)
        eff = r["contrast"]["fact_effect"]["observed"]
        r["inside_noise_floor"] = bool(abs(eff) < floor_for(kind, budget))
        readouts[name] = r
        order.append(name)
        if verbose:
            print(f"   {name:20s} d={eff:+.4f} z={r['contrast']['fact_effect']['z']:7.2f} "
                  f"MRR={r['retrieval']['all_candidates']['mrr']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # sketch-consistency cross-check: pooled vs concat_layers should agree to the
    # measured sketch error, since both sketch the same full gradient.
    Cp = cosine_matrix(data["pooled"])
    Ccl = cosine_matrix(data["per_layer"].reshape(data["per_layer"].shape[0], -1))
    diff = (Cp - Ccl)[d.iu, d.ju]
    consistency = {
        "rms_pooled_vs_concat_layers": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs": float(np.max(np.abs(diff))),
        "pearson": pearson_r(Cp[d.iu, d.ju], Ccl[d.iu, d.ju]),
        "note": "two independent sketches of the same full gradient; their disagreement is "
                "an empirical read on the sketch error budget",
    }

    return {"readouts": readouts, "readout_order": order, "n_perm": nperm,
            "sketch_consistency": consistency, "noise_budget": budget,
            "design": {
                "n_docs": d.n, "n_facts": d.n_facts, "n_pairs": d.n_pairs,
                "n_same_fact_pairs": int(d.m_SF.sum()),
                "n_diff_fact_same_genre_pairs": int(d.m_DF_SG.sum()),
                "n_diff_fact_diff_genre_pairs": int(d.m_DF_DG.sum()),
                "n_genres": len(data["genre_names"]),
                "all_same_fact_pairs_cross_genre": True,
            },
            "provenance": {
                "docs_file": data["meta"].get("docs_file"),
                "base_model": data["meta"].get("base_model"),
                "grad_dim_D": data["meta"].get("grad_dim_D"),
                "projection": data["meta"].get("projection", {}).get("kind"),
                "sketch_fidelity": data["meta"]["verification"].get(
                    "sketch_fidelity_on_real_gradients"),
                "params_bitwise_unchanged": data["meta"]["verification"].get(
                    "params_bitwise_unchanged"),
                "attribution_step_i_is_doc_i": data["meta"]["verification"].get(
                    "attribution_step_i_is_doc_i"),
                "row_i_is_doc_id_order_i": True,
            }}


def classify_case(res: dict) -> dict:
    """(a) present at pooled, (b) absent at pooled but present somewhere, (c) absent."""
    ro = res["readouts"]

    def alive(name):
        r = ro[name]
        fe = r["contrast"]["fact_effect"]
        return (fe["z"] >= 4.0) and (fe["p_one_sided"] <= 0.001) and (not r["inside_noise_floor"])

    pooled_alive = alive("pooled")
    others = [n for n in ro if n != "pooled" and alive(n)]
    if pooled_alive:
        case = "a"
        stmt = "present at pooled"
    elif others:
        case = "b"
        stmt = "ABSENT at pooled but PRESENT at some layer/type readout"
    else:
        case = "c"
        stmt = "absent everywhere"
    return {"case": case, "statement": stmt, "pooled_alive": pooled_alive,
            "alive_readouts": sorted(others),
            "criterion": "z>=4 AND permutation p<=0.001 AND |effect| above the readout's "
                         "sketch noise floor"}


# ======================================================================================
# markdown report
# ======================================================================================

def fmt_p(p, nperm):
    floor = 1.0 / (nperm + 1)
    return f"<{floor:.1e}" if p <= floor + 1e-12 else f"{p:.4g}"


def write_markdown(path, results, selftest, args):
    L = []
    A = L.append
    A("# Gradient-content probe: fit-free analysis")
    A("")
    A(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `analyse_gradprobe.py`._")
    A("")
    A("**Question.** Does a per-document gradient encode the document's CONTENT (which "
      "fact it states) rather than its SURFACE (genre)?")
    A("")
    A("**Method.** Cosine similarity and rank statistics only. No trained classifier, no "
      "fitted linear map, nothing estimated from the data except the reported statistics "
      "themselves. Inference is by permutation.")
    A("")
    if results.get("data_status"):
        A(f"> **Data status.** {results['data_status']}")
        A("")

    # design
    dz = list(results["modes"].values())[0]["design"]
    A("## Design")
    A("")
    A(f"- {dz['n_docs']} documents, {dz['n_facts']} facts x 2 documents, "
      f"{dz['n_genres']} genres.")
    A(f"- The two documents of a fact are ALWAYS in different genres "
      f"({dz['n_same_fact_pairs']}/{dz['n_facts']} verified).")
    A(f"- {dz['n_pairs']} document pairs partition into "
      f"SAME-FACT {dz['n_same_fact_pairs']}, DIFF-FACT/SAME-GENRE "
      f"{dz['n_diff_fact_same_genre_pairs']}, DIFF-FACT/DIFF-GENRE "
      f"{dz['n_diff_fact_diff_genre_pairs']}.")
    A("- Headline comparison is SAME-FACT vs DIFF-FACT/DIFF-GENRE: **both sets are "
      "cross-genre**, so genre is controlled by construction.")
    A("")

    A("## What was permuted")
    A("")
    A("The 79,800 pairs are NOT independent -- each document appears in 399 of them -- so "
      "no naive t-test over pairs is quoted. The instrument is a permutation over FACT "
      "LABELS that preserves genre:")
    A("")
    A("> Draw a bijection sigma of documents onto documents that is uniform WITHIN each "
      "genre block. Carry the true 200-pair matching through sigma. Because sigma is "
      "genre-preserving and a bijection: every document keeps its own genre and its own "
      "sketch; the null matching is again a perfect matching on all 400 documents; the "
      "multiset of (genre_a, genre_b) combinations across the 200 null pairs is IDENTICAL "
      "to the observed one. Only *which documents state the same fact* is destroyed.")
    A("")
    ps = selftest["permutation_structure"]
    A(f"Verified: genre preserved per document `{ps['genre_preserved']}`; genre-pair "
      f"multiset identical `{ps['genre_pair_signature_identical']}`; perfect matching "
      f"preserved `{ps['perfect_matching_preserved']}`; all null pairs cross-genre "
      f"`{ps['all_pairs_cross_genre']}`; fraction of null pairs that are accidentally true "
      f"partners {ps['frac_null_pairs_truly_same_fact']:.4f} (conservative).")
    A("")

    # noise floor
    A("## Sketch error budget (upstream-measured)")
    A("")
    A("Measured by the training run itself against exact gradients on k=12 real documents "
      "(`meta.verification.sketch_fidelity_on_real_gradients`); the gate uses the larger of "
      "the measured rms and the stated figure.")
    A("")
    A("| readout | rms per-pair error (measured) | max per-pair error | mean bias | r | gate used |")
    A("|---|---|---|---|---|---|")
    for mode, res in results["modes"].items():
        b = res.get("noise_budget", NOISE_FLOOR)
        meas = b.get("measured", {})
        for key, lab in (("pooled", "pooled"), ("per_layer", "per-layer"),
                         ("per_type", "per-type")):
            m = meas.get(key, {})
            A(f"| {mode} / {lab} | {m.get('rms_err', float('nan')):.5f} | "
              f"{m.get('max_abs_err', float('nan')):.5f} | {m.get('mean_err', float('nan')):+.5f} | "
              f"{m.get('pearson_r', float('nan')):.4f} | {b.get(key, float('nan')):.3f} |")
    A("")
    A("**Rule applied here: any cosine DIFFERENCE smaller than the relevant rms figure is "
      "not interpreted.** Rows failing this are flagged `INSIDE NOISE` in the tables "
      "below. (Caveat in both directions: a mean over 200 pairs shrinks the *random* part "
      "of that error by ~sqrt(200), but the measured mean bias, ~0.002-0.003, does not "
      "average away. We apply the strict per-pair floor as the gate.)")
    A("")

    # selftest
    A("## Verification: synthetic ground truth")
    A("")
    A("```")
    L.extend(selftest["log"])
    A("```")
    A("")

    for mode, res in results["modes"].items():
        A(f"# Mode: {mode}")
        A("")
        cls = res["case"]
        A(f"**Verdict: case ({cls['case']}) -- the SAME-FACT effect is {cls['statement']}.**")
        A(f"Criterion: {cls['criterion']}.")
        A("")
        sc = res["sketch_consistency"]
        A(f"Sketch consistency cross-check (pooled vs concat-of-layer sketches, two "
          f"independent sketches of the same gradient): rms {sc['rms_pooled_vs_concat_layers']:.4f}, "
          f"max {sc['max_abs']:.4f}, r={sc['pearson']:.4f}.")
        A("")

        for hl in ("pooled", "mean_over_layers"):
            r = res["readouts"][hl]
            c = r["contrast"]
            A(f"## {mode} / {hl}: three-way cosine contrast")
            A("")
            A("| group | n | mean | se(naive) | sd | min | p05 | q25 | median | q75 | p95 | max |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for key, lab in (("same_fact", "SAME-FACT (all cross-genre)"),
                             ("diff_fact_same_genre", "DIFF-FACT / SAME-GENRE"),
                             ("diff_fact_diff_genre", "DIFF-FACT / DIFF-GENRE")):
                s = c[key]
                A(f"| {lab} | {s['n']} | {s['mean']:.4f} | {s['se_naive']:.4f} | {s['sd']:.4f} | "
                  f"{s['min']:.3f} | {s['p05']:.3f} | {s['q25']:.3f} | {s['median']:.3f} | "
                  f"{s['q75']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |")
            A("")
            fe = c["fact_effect"]
            ge = r["genre_effect_perm"]
            A(f"- **FACT effect** (SAME-FACT - DIFF-FACT/DIFF-GENRE): "
              f"**{fe['observed']:+.4f}**, permutation z = **{fe['z']:+.2f}**, "
              f"p(one-sided) = **{fmt_p(fe['p_one_sided'], fe['n_perm'])}** "
              f"({fe['n_perm']} permutations; null {fe['perm_null_mean']:+.5f} +/- "
              f"{fe['perm_null_sd']:.5f})"
              + ("  `INSIDE NOISE`" if r["inside_noise_floor"] else ""))
            A(f"- **GENRE effect** (DIFF-FACT/SAME-GENRE - DIFF-FACT/DIFF-GENRE): "
              f"**{c['genre_effect']['observed']:+.4f}**, z = {ge['z']:+.2f} "
              f"(genre-label shuffle)")
            A(f"- fact/genre effect ratio: {c['fact_vs_genre_ratio']:+.2f}")
            A("")

            rt = r["retrieval"]
            A(f"### {mode} / {hl}: retrieval of the partner document")
            A("")
            A("| candidate pool | median rank | MRR | top-1 | top-5 | top-10 | top-50 |")
            A("|---|---|---|---|---|---|---|")
            for kk, lab in (("all_candidates", "all 399 others"),
                            ("diff_genre_only", "DIFFERENT-GENRE only (genre shortcut blocked)")):
                b = rt[kk]
                A(f"| {lab} | {b['median_rank']:.0f} | {b['mrr']:.4f} | {b['top1']:.4f} | "
                  f"{b['top5']:.4f} | {b['top10']:.4f} | {b['top50']:.4f} |")
            b = rt["all_candidates"]
            A(f"| _chance (all)_ | {b['median_rank_chance']:.0f} | 0.0165 | "
              f"{b['top1_chance']:.4f} | {b['top5_chance']:.4f} | {b['top10_chance']:.4f} | "
              f"{b['top50_chance']:.4f} |")
            b2 = rt["diff_genre_only"]
            A(f"| _chance (diff-genre pool ~{rt['diff_genre_pool_sizes']['mean']:.0f})_ | "
              f"{b2['median_rank_chance']:.0f} | - | {b2['top1_chance']:.4f} | "
              f"{b2['top5_chance']:.4f} | {b2['top10_chance']:.4f} | {b2['top50_chance']:.4f} |")
            A("")
            mp = rt["mrr_permutation"]
            A(f"MRR permutation test: observed {mp['observed']:.4f} vs null "
              f"{mp['perm_null_mean']:.4f} +/- {mp['perm_null_sd']:.4f}, z = {mp['z']:+.2f}, "
              f"p = {fmt_p(mp['p_one_sided'], mp['n_perm'])}.")
            A("")

            nc = r["norm_control"]
            nm = nc["norm_matched_contrast"]
            A(f"### {mode} / {hl}: norm control")
            A("")
            A(f"- Pearson r(pair cosine, |norm_i - norm_j|) over all pairs: "
              f"**{nc['cos_vs_absdiff_norm_pearson_all_pairs']:+.4f}**; over cross-genre "
              f"pairs: {nc['cos_vs_absdiff_norm_pearson_cross_genre_pairs']:+.4f}.")
            A(f"- mean |delta-norm|: SAME-FACT {nc['same_fact_absdiff_norm_mean']:.4f} vs "
              f"DIFF-FACT/DIFF-GENRE {nc['diff_fact_diff_genre_absdiff_norm_mean']:.4f}.")
            if "cos_vs_absdiff_tokens_pearson_all_pairs" in nc:
                A(f"- Pearson r(pair cosine, |delta tokens|): "
                  f"{nc['cos_vs_absdiff_tokens_pearson_all_pairs']:+.4f}; mean |delta tokens| "
                  f"SAME-FACT {nc['same_fact_absdiff_tokens_mean']:.1f} vs DIFF/DIFF "
                  f"{nc['diff_fact_diff_genre_absdiff_tokens_mean']:.1f}.")
            A(f"- **Norm-matched** contrast ({nm['n_bins']} |delta-norm| strata): "
              f"**{nm['observed']:+.4f}** (unmatched {fe['observed']:+.4f}), z = "
              f"**{nm['z']:+.2f}**, p = {fmt_p(nm['p_one_sided'], nm['n_perm'])}. "
              f"{'Survives norm matching.' if nm['z'] > 4 else 'Does NOT survive norm matching.'}")
            A("")

        # readout table
        A(f"## {mode}: the readout axis")
        A("")
        A("| readout | kind | fact effect | z | p | genre effect | MRR | MRR diff-genre | top-1 | norm-matched z | flag |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for name in res["readout_order"]:
            r = res["readouts"][name]
            fe = r["contrast"]["fact_effect"]
            gg = r["contrast"]["genre_effect"]["observed"]
            rt = r["retrieval"]
            nm = r["norm_control"]["norm_matched_contrast"]
            flag = "INSIDE NOISE" if r["inside_noise_floor"] else ""
            A(f"| {name} | {r['kind']} | {fe['observed']:+.4f} | {fe['z']:+.2f} | "
              f"{fmt_p(fe['p_one_sided'], fe['n_perm'])} | {gg:+.4f} | "
              f"{rt['all_candidates']['mrr']:.4f} | {rt['diff_genre_only']['mrr']:.4f} | "
              f"{rt['all_candidates']['top1']:.4f} | {nm['z']:+.2f} | {flag} |")
        A("")

        # depth profile
        A(f"### {mode}: depth profile (per-layer fact-effect z)")
        A("")
        zs = [(int(n[5:]), res["readouts"][n]["contrast"]["fact_effect"]["z"])
              for n in res["readout_order"] if n.startswith("layer")]
        zs.sort()
        A("```")
        mx = max(abs(z) for _, z in zs) or 1.0
        for l, z in zs:
            bar = "#" * int(round(40 * max(z, 0) / mx))
            A(f"L{l:02d} z={z:+7.2f} {bar}")
        A("```")
        A("")

    A("## Sequential vs frozen")
    A("")
    A(results.get("mode_comparison", ""))
    A("")
    A("## Bottom line")
    A("")
    A(results.get("bottom_line", ""))
    A("")
    open(path, "w").write("\n".join(L))


def compare_modes(results) -> str:
    ms = results["modes"]
    if len(ms) < 2:
        return "Only one mode analysed."
    lines = []
    lines.append("| mode | pooled effect | pooled z | best readout | its effect | its z | pooled MRR | best MRR | case |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mode, res in ms.items():
        p = res["readouts"]["pooled"]["contrast"]["fact_effect"]
        best = max(res["readout_order"],
                   key=lambda n: res["readouts"][n]["contrast"]["fact_effect"]["z"])
        b = res["readouts"][best]["contrast"]["fact_effect"]
        lines.append(f"| {mode} | {p['observed']:+.4f} | {p['z']:+.2f} | {best} | "
                     f"{b['observed']:+.4f} | {b['z']:+.2f} | "
                     f"{res['readouts']['pooled']['retrieval']['all_candidates']['mrr']:.4f} | "
                     f"{res['readouts'][best]['retrieval']['all_candidates']['mrr']:.4f} | "
                     f"{res['case']['case']} |")
    cases = {m: r["case"]["case"] for m, r in ms.items()}
    agree = len(set(cases.values())) == 1
    lines.append("")
    lines.append(f"Conclusions {'AGREE' if agree else 'DIFFER'} across modes: {cases}. "
                 "Frozen (all gradients taken at the identical initial parameters) is the "
                 "cleaner object -- every gradient lives in the same tangent space, so "
                 "cosines are directly comparable. Sequential is realistic training: the "
                 "parameters move between documents, so part of any similarity structure "
                 "can be drift shared by temporally adjacent documents rather than content.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--modes", default="frozen,sequential")
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--selftest", action="store_true", help="run synthetic verification only")
    ap.add_argument("--selftest-reps", type=int, default=200)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic stand-in sketches instead of out/*.npy")
    args = ap.parse_args()

    root = args.root
    docs_path = os.path.join(root, "docs.jsonl")
    docs = [json.loads(l) for l in open(docs_path)]
    docs_by_id = {d["doc_id"]: d for d in docs}

    modes = [m for m in args.modes.split(",") if m]

    # If real sketches are missing, fall back to synthetic stand-ins with known ground truth.
    missing = [m for m in modes
               if not os.path.exists(os.path.join(root, "out", m, "sketches_pooled.npy"))]
    use_synth = args.synthetic or bool(missing)

    if args.selftest:
        # design built from docs.jsonl alone
        order = [d["doc_id"] for d in docs]
        gmap = {g: k for k, g in enumerate(sorted({d["genre"] for d in docs}))}
        fmap = {f: k for k, f in enumerate(sorted({d["fact_id"] for d in docs}))}
        design = Design(np.array([fmap[docs_by_id[i]["fact_id"]] for i in order]),
                        np.array([gmap[docs_by_id[i]["genre"]] for i in order]),
                        np.array(order))
        st = run_selftest(design, seed=args.seed, nperm=10000,
                          n_null_reps=args.selftest_reps)
        print("")
        print("[selftest] PASS" if (st["reference_checks"]["all_passed"]
                                    and st["null_uniformity"]["uniform_not_rejected"])
              else "[selftest] CHECK FAILURES ABOVE")
        return

    if use_synth:
        print(f"!! real sketches missing for {missing}; refusing to invent results.",
              file=sys.stderr)
        print("!! run with --selftest for the synthetic ground-truth pipeline check.",
              file=sys.stderr)
        sys.exit(2)

    results = {"modes": {}, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "n_perm": args.nperm, "seed": args.seed,
               "noise_floor": NOISE_FLOOR,
               "method": {
                   "fit_free": True,
                   "statistics": "cosine similarity + rank statistics only; no classifier, "
                                 "no fitted map",
                   "what_was_permuted": "assignment of documents to facts, via a bijection "
                                        "that is uniform within each genre block; this "
                                        "preserves every document's genre, the perfect "
                                        "matching structure, and the exact multiset of "
                                        "(genre_a,genre_b) combinations of the 200 pairs",
                   "why_not_t_test": "the 79,800 pairs share documents (each document is in "
                                     "399 pairs); naive pair-level SEs are quoted as "
                                     "descriptive only and never used for inference",
               }}

    # self-test first, embedded as evidence
    order = [d["doc_id"] for d in docs]
    gmap = {g: k for k, g in enumerate(sorted({d["genre"] for d in docs}))}
    fmap = {f: k for k, f in enumerate(sorted({d["fact_id"] for d in docs}))}
    design0 = Design(np.array([fmap[docs_by_id[i]["fact_id"]] for i in order]),
                     np.array([gmap[docs_by_id[i]["genre"]] for i in order]),
                     np.array(order))
    print("== synthetic verification ==")
    st = run_selftest(design0, seed=args.seed, nperm=10000, n_null_reps=args.selftest_reps)
    results["verification"] = st

    for mode in modes:
        print(f"\n== {mode} ==", flush=True)
        data = load_mode(root, mode, docs_by_id)
        res = analyse_mode(data, args.nperm, args.seed)
        res["case"] = classify_case(res)
        results["modes"][mode] = res
        print(f"   -> case ({res['case']['case']}): {res['case']['statement']}")

    results["mode_comparison"] = compare_modes(results)

    # bottom line
    bl = []
    for mode, res in results["modes"].items():
        best = max(res["readout_order"],
                   key=lambda n: res["readouts"][n]["contrast"]["fact_effect"]["z"])
        b = res["readouts"][best]["contrast"]["fact_effect"]
        p = res["readouts"]["pooled"]["contrast"]["fact_effect"]
        bl.append(f"**{mode}**: case ({res['case']['case']}); pooled effect "
                  f"{p['observed']:+.4f} (z={p['z']:+.2f}), best readout `{best}` "
                  f"{b['observed']:+.4f} (z={b['z']:+.2f}), partner-retrieval MRR "
                  f"{res['readouts'][best]['retrieval']['all_candidates']['mrr']:.4f} "
                  f"(chance 0.0165), different-genre-only MRR "
                  f"{res['readouts'][best]['retrieval']['diff_genre_only']['mrr']:.4f}.")
    results["bottom_line"] = "\n\n".join(bl)

    out_json = args.out_json or os.path.join(root, "results", "gradprobe.json")
    out_md = args.out_md or os.path.join(root, "results", "gradprobe.md")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))

    json.dump(results, open(out_json, "w"), indent=1, default=default, allow_nan=True)
    write_markdown(out_md, results, st, args)
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
