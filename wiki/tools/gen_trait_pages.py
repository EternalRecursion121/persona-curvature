#!/usr/bin/env python3
"""Generate one wiki page per trait, plus two index pages.

Emits into wiki/pages/traits/:
    trait-<slug>.md      141 pages: the 134 zoo traits, 4 alignment probes, 3 hole probes
    traits-index.md      table of all 141
    traits-by-factor.md  the same, grouped by Big Five axis and by recovered FA factor

Every number on a page is copied from a JSON or log in the repo and the page cites
the file and, for JSON, the dotted key path.  Nothing is recomputed: no means are
taken, no cosines derived, no ranks inferred.  Where a per-trait number does not
exist in any source the section is omitted rather than filled with a placeholder;
the omissions are printed as a coverage summary at the end of the run.

Re-runnable and idempotent: it rewrites every page each run and deletes any
trait-*.md that is no longer in the generated set.  It never touches _report.md.

    /home/vibe12/projects/persona-curvature/qwen35/.venv/bin/python gen_trait_pages.py
"""
import glob
import json
import os
import re

ROOT = "/home/vibe12/projects/persona-curvature"
Q = os.path.join(ROOT, "qwen35")
OUT = os.path.join(ROOT, "wiki", "pages", "traits")
LAST_VERIFIED = "2026-09-07"

ROUND_NOTE = ("Values are printed as the source stores them; where a source float "
              "carries more digits it is shown to four significant figures, or to "
              "the nearest whole number above 9,999, and marked (rounded).")
TABLE_NOTE = ("Table values are rounded to four significant figures, or to the "
              "nearest whole number above 9,999, where the source holds more "
              "digits.")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def jload(rel):
    with open(os.path.join(Q, rel)) as fh:
        return json.load(fh)


def slug(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def sig4(x):
    """Four significant figures, without exponent noise for ordinary magnitudes."""
    import math
    if x == 0:
        return "0.0"
    if not (1e-6 <= abs(x) < 1e9):
        return f"{x:.4g}"
    d = 3 - int(math.floor(math.log10(abs(x))))
    if d <= 0:
        return f"{round(x, 0):.0f}"
    return f"{x:.{d}f}"


def n4(x):
    """Table/inline number: exact if the source is already short, else 4 s.f."""
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return str(x)
    if float(x) == int(float(x)) and abs(float(x)) < 1e6:
        return str(x)
    if len(repr(float(x)).replace("-", "").replace(".", "").lstrip("0")) <= 4:
        return repr(float(x))
    return sig4(float(x))


def num(x):
    """Prose number: as n4, but says so when it rounded."""
    if isinstance(x, (int, bool)) or x is None:
        return str(x)
    exact = repr(float(x))
    short = n4(x)
    if short != exact and float(short) != float(exact):
        return f"{short} (rounded)"
    return short


EMOJI_RE = re.compile(
    "[" "\U0001F000-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE0F" "\U000024C2"
    "\U0000203C" "\U00002049" "]+")


def has_emoji(s):
    return bool(EMOJI_RE.search(s))


def strip_emoji(s):
    return EMOJI_RE.sub("", s)


def sample_text(s, limit=300):
    """One short quotable sample: whitespace collapsed, truncated, emoji removed."""
    had = has_emoji(s)
    s = strip_emoji(s)
    s = re.sub(r"\s+", " ", s).strip()
    cut = len(s) > limit
    s = s[:limit].rstrip()
    return s, cut, had


def yq(v):
    """A YAML double-quoted scalar: summaries and source paths carry colons."""
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def blockquote(text):
    text = strip_emoji(text)
    out = []
    for line in text.split("\n"):
        out.append("> " + line if line.strip() else ">")
    return "\n".join(out)


# --------------------------------------------------------------------------
# load every source once
# --------------------------------------------------------------------------
PRIMARY = jload("traits_primary.json")
SECONDARY = jload("traits_secondary.json")
ALIGNMENT = jload("traits_alignment.json")
HOLE = jload("traits_hole.json")
SEC_PROV = jload("traits_secondary_provenance.json")
CONS = jload("constitutions.json")

NXN = jload("analysis/nxn_summary.json")
ZOO = list(NXN["names"])                       # the 134 that exist as adapters
ZOO_SET = set(ZOO)

VIZ = jload("analysis/viz.json")
FA = jload("results/fa_qwen35.json")
FA_SUM = jload("analysis/fa_summary.json")
TG = jload("analysis/trait_graph.json")
RUNMETA = jload("results/runmeta_sweep.json")
MERGE_AUDIT = jload("analysis/merge_audit.json")
CORPUS_SCAN = jload("analysis/corpus_scan_all.json")
CORPUS_DEGEN = jload("analysis/corpus_degeneration.json")
ADAPTER_EFFECT = jload("analysis/adapter_effect.json")
SITE = jload("site_traits/data.json")
HF_AUDIT = jload("analysis/hf_dataset_audit.json")
ALIGN_GEO = jload("analysis/alignment_geometry.json")
ALIGN_GEO_COMMON = jload("analysis/alignment_geometry_aligncommon.json")
HOLE_GEO = jload("analysis/hole_geometry.json")
ALIGN_RESULTS = jload("phase10_runs/alignment_results.json")
EVAL100 = jload("phase10_runs/eval_100traits.json")
ACTCROSS = jload("analysis/actspace_cross_geometry.json")

# display name -> slug, and the provenance set each trait belongs to
DISPLAY, SETOF, FACTOR, KEYED, WHY = {}, {}, {}, {}, {}
for rows, setname in ((PRIMARY, "primary"), (SECONDARY, "lexicon"),
                      (ALIGNMENT, "alignment"), (HOLE, "hole")):
    for r in rows:
        s = slug(r["trait"])
        DISPLAY[s] = r["trait"]
        SETOF[s] = setname
        FACTOR[s] = r["factor"]
        KEYED[s] = r["keyed"]
        if r.get("why"):
            WHY[s] = r["why"]

ALIGN_SLUGS = [slug(r["trait"]) for r in ALIGNMENT]
HOLE_SLUGS = [slug(r["trait"]) for r in HOLE]
PAGES = ZOO + ALIGN_SLUGS + HOLE_SLUGS
LEXICON_DRAWN = [slug(r["trait"]) for r in SECONDARY]
LEXICON_DROPPED = [s for s in LEXICON_DRAWN if s not in ZOO_SET]

# constitutions are keyed by display name
CONS_BY_SLUG = {slug(k): v for k, v in CONS.items()}

# ---- PC scores (viz.json), PC1..PC8, centred PCA over the stage-1 sketches
VIZ_IDX = {t: i for i, t in enumerate(VIZ["traits"])}
PC_LOADINGS = jload("analysis/pc_loadings.json")

# ---- FA: per-trait oblimin loadings on the k=5 centred solution
FA_PER_TRAIT = FA["per_trait"]                     # keyed by display name
# Display-name override, 2026-09-11: the third factor is shown as "Timidity".
# Nothing internal moves - the key FA_FearfulWithdrawal, the npz slug, the
# steering job, the wiki slug factor-fearful-withdrawal and the solution's own
# label in fa_summary.json are all unchanged; this maps the label at the point
# where it is written into a page.
FA_DISPLAY_RENAME = {"Fearful withdrawal": "Timidity"}


def fa_disp(name):
    return FA_DISPLAY_RENAME.get(name, name)


FA_NAMES = [fa_disp(f["name"]) for f in FA_SUM["centred_k5"]["factors"]]
FA_SLUGS = ["factor-warmth", "factor-competence", "factor-fearful-withdrawal",
            "factor-arousal", "factor-imagination"]

# ---- the factor chart (qwen35/fa_chart.py via qwen35/build_viz_data_fa.py).
# Primary frame since 2026-09-08: five coordinates on an orthonormal basis of the
# span of the five oblimin factor directions, Gram-Schmidt in the exact Gram in
# the order Warmth, Competence, Fearful withdrawal, Arousal, Imagination.
VIZ_FA = jload("analysis/viz_fa.json")
VIZ_FA_IDX = {t: i for i, t in enumerate(VIZ_FA["traits"])}
FA_TITLES = [fa_disp(t) for t in VIZ_FA["factor_titles"]]

AXIS_SLUG = {"Extraversion": "factor-axis-extraversion",
             "Agreeableness": "factor-axis-agreeableness",
             "Conscientiousness": "factor-axis-conscientiousness",
             "EmotionalStability": "factor-axis-emotional-stability",
             "Intellect": "factor-axis-intellect"}

# ---- nearest neighbours from the K=5 kNN graph
NODES = [slug(n["t"]) for n in TG["stage1"]["nodes"]]
NEIGH = {s: [] for s in NODES}
for i, j, c in TG["stage1"]["edges"]:
    NEIGH[NODES[i]].append((NODES[j], c))
    NEIGH[NODES[j]].append((NODES[i], c))
for s in NEIGH:
    seen, keep = set(), []
    for t, c in sorted(NEIGH[s], key=lambda e: -e[1]):
        if t not in seen:
            seen.add(t)
            keep.append((t, c))
    NEIGH[s] = keep[:5]

# ---- N x N ranks
RANK_RAW = NXN["raw"]["ranks"]
RANK_COLZ = NXN["column-z"]["ranks"]

# ---- cross-seed stage 1 (40 traits), from the site_traits build
SEEDPAIRED = dict(zip(SITE["seedpaired"]["names"], SITE["seedpaired"]["self_cos"]))
SEEDPAIRED_NORMA = dict(zip(SITE["seedpaired"]["names"], SITE["seedpaired"]["norm_a"]))
SEEDPAIRED_NORMB = dict(zip(SITE["seedpaired"]["names"], SITE["seedpaired"]["norm_b"]))

# ---- stage 2 second seed: the 15 traits of the seed-1 OCT run
S2_SEED1 = jload("phase10_runs/results_oct2_15traits_"
                 "v1-n1000-ni1000-k10-bugsfaithful.json")["traits"]

# ---- stage 2 per-trait records, seed 0 and seed 1, from every OCT-2 results file.
# A resumed run writes {"trait": t, "skipped": true} for a stage it did not redo,
# so a stage's real record may live in a different file from the same trait's
# other stages. Records are merged stage by stage, a real record always winning
# over a skip, and the file each real record came from is kept for the citation.
SEED1_FILE = ("results_oct2_15traits_v1-n1000-ni1000-k10-bugsfaithful.json")
S2 = {}          # slug -> {stage: (record, source file)}
S2_SEED1_REC = {}


def _real(rec):
    return rec is not None and not rec.get("skipped")


# results_oct2_1traits_v1-n40-ni8-k4-bugsfaithful.json is a smoke run (40
# reflections, 8 interactions, k=4, 416 SFT rows) and is excluded: its records
# would otherwise stand in for extraverted's production stage 2.
PROD_GEN_KEY = "v1-n1000-ni1000-k10-bugsfaithful.json"
for path in sorted(glob.glob(os.path.join(Q, "phase10_runs",
                                          "results_oct2_*traits_*.json"))):
    if not path.endswith(PROD_GEN_KEY):
        continue
    doc = json.load(open(path))
    rel = os.path.relpath(path, Q)
    base = os.path.basename(path)
    stages = doc.get("stages", {})
    if "sft" not in stages:
        continue
    target = S2_SEED1_REC if base == SEED1_FILE else S2
    for st in ("merge", "assemble", "sft", "final"):
        for rec in stages.get(st, []):
            t = rec["trait"]
            bundle = target.setdefault(t, {})
            if _real(rec):
                bundle[st] = (rec, rel)
            else:
                bundle.setdefault(st, None)

# ---- adapter effect (100 traits)
AE = {r["trait"]: r for r in ADAPTER_EFFECT}
MA = {r["trait"]: r for r in MERGE_AUDIT}
STEER = SITE["steering"]["per_trait"]
SITE_TRAITS = {t["slug"]: t for t in SITE["traits"]}

# ---- lexicon provenance
CLUSTER_OF = {slug(c["trait"]): c["cluster"] for c in SEC_PROV["chosen"]}
CLUSTERS = SEC_PROV["clusters"]

# ---- example generations
EVAL_BY_TRAIT = {r["trait"]: r for r in EVAL100}
ACTGEN = {}
with open(os.path.join(Q, "analysis/actspace_generations_adapters.jsonl")) as fh:
    for line in fh:
        r = json.loads(line)
        ACTGEN[r["trait"]] = r["responses"]
ALIGN_GEN = {r["name"]: r for r in ALIGN_RESULTS}

# ---- the 16 cross traits, read out of act_space.py so the list cannot drift
src = open(os.path.join(Q, "act_space.py")).read()
m = re.search(r"CROSS_TRAITS\s*=\s*\[(.*?)\]", src, re.S)
CROSS_TRAITS = re.findall(r'"([a-z_]+)"', m.group(1)) if m else []
ACT_MATCHED = {r["t"]: r for r in ACTCROSS["resp"]["matched"]}
ACT_MATCHED_SPEC = {r["t"]: r for r in ACTCROSS["resp_specific"]["matched"]}

# ---- HF dataset audit
HF_QUARANTINED = set(HF_AUDIT["missing_quarantined"])
HF_NOT_UPLOADED = set(HF_AUDIT["missing_not_yet_uploaded"])
HF_PARTIAL = {p["trait"]: p for p in HF_AUDIT["partial_trait_filesets"]}


# ---- alignment / hole training records, parsed from the run logs
def parse_train_log(rel):
    out = {}
    line_re = re.compile(r"^\s+([a-z_]+): corpus=(\S+) targeted=(\d+) excl=(\d+) "
                         r"loss ([0-9.eE+-]+) -> ([0-9.eE+-]+)\s*$")
    pair_re = re.compile(r"^\[data\] ([a-z_]+): (\d+) pairs\s+corpus=(\S+) "
                         r"sha256=(\S+)\s+pool_sha=(\S+)")
    done_re = re.compile(r"^\[done\] ([a-z_]+) in (\d+)s")
    path = os.path.join(Q, rel)
    if not os.path.exists(path):
        return out
    for raw in open(path, errors="replace"):
        line = raw.rstrip("\n")
        mm = line_re.match(line)
        if mm:
            out.setdefault(mm.group(1), {}).update(
                corpus=mm.group(2), targeted=int(mm.group(3)),
                excluded=int(mm.group(4)), loss_first=float(mm.group(5)),
                loss_last=float(mm.group(6)))
            continue
        mm = pair_re.match(line)
        if mm:
            out.setdefault(mm.group(1), {}).update(
                n_pairs=int(mm.group(2)), corpus=mm.group(3),
                data_sha256=mm.group(4), prompt_pool_sha256=mm.group(5))
            continue
        mm = done_re.match(line)
        if mm:
            out.setdefault(mm.group(1), {}).update(train_seconds=int(mm.group(2)))
    return out


LOG_ALIGN = parse_train_log("phase10_runs/aligntrain.log")
LOG_ALIGN_COMMON = parse_train_log("phase10_runs/aligncommon.log")
LOG_HOLE = parse_train_log("phase10_runs/holetrain.log")

HF_MODEL = "EternalRecursion/persona-lora-zoo-qwen35"
HF_DATA = "EternalRecursion/persona-curvature-oct-transcripts"


# --------------------------------------------------------------------------
# per-trait facts
# --------------------------------------------------------------------------
def fa_assignment(s):
    """(index, loadings) of the k=5 centred oblimin solution, or None."""
    disp = DISPLAY[s]
    rec = FA_PER_TRAIT.get(disp)
    if not rec or "oblimin_loadings_centred_k5" not in rec:
        return None
    load = rec["oblimin_loadings_centred_k5"]
    idx = max(range(len(load)), key=lambda i: abs(load[i]))
    return idx, load, rec


def factor_tag(s):
    f = FACTOR[s]
    return {"EmotionalStability": "emotional-stability"}.get(f, f.lower())


def set_label(s):
    return {"primary": "Goldberg 100 primary markers",
            "lexicon": "lexicon draw (Condon TDA)",
            "alignment": "alignment probe",
            "hole": "hole-word probe"}[SETOF[s]]


# --------------------------------------------------------------------------
# section builders. each returns (markdown, sources) or (None, []) when absent.
# --------------------------------------------------------------------------
def sec_identity(s):
    L, src = [], []
    setname = SETOF[s]
    tf = {"primary": "traits_primary.json", "lexicon": "traits_secondary.json",
          "alignment": "traits_alignment.json", "hole": "traits_hole.json"}[setname]
    L.append(f"- Trait word: **{DISPLAY[s]}** (slug `{s}`)")
    L.append(f"- Factor as recorded in the trait file: {FACTOR[s]}")
    L.append(f"- Keying: `{KEYED[s]}`")
    L.append(f"- Provenance set: {set_label(s)}")
    src.append(f"qwen35/{tf}")
    if setname == "primary":
        L.append("- One of the 100 Goldberg marker adjectives, 20 per Big Five "
                 "factor, 10 positively and 10 negatively keyed.")
    if setname == "lexicon":
        cl = CLUSTER_OF.get(s)
        if cl and cl in CLUSTERS:
            c = CLUSTERS[cl]
            L.append(f"- Drawn from {cl} of a 40-cluster k-means over "
                     f"{SEC_PROV['n_embedded']} trait adjectives; cluster size "
                     f"{c['size']}, chosen at rank "
                     f"{c['chosen_rank_by_centroid_distance']} by distance from the "
                     f"centroid (the draw takes a random member, not the centroid word).")
            members = [w for w in c["members"] if w != s][:8]
            if members:
                L.append("- Other words in the same cluster: "
                         + ", ".join(members) + ".")
            src.append(f"qwen35/traits_secondary_provenance.json#clusters.{cl}")
            src.append("qwen35/traits_secondary_provenance.json#chosen")
        L.append(f"- Source list: {SEC_PROV['source_description']}")
        src.append("qwen35/traits_secondary_provenance.json#source_description")
    if s in WHY:
        L.append(f"- Why this trait was added: {WHY[s]}")
    if setname in ("alignment", "hole"):
        L.append("- Not one of the 134 zoo adapters; trained afterwards as a probe "
                 "and measured against the zoo.")
    L.append("- Opposite-pole partner: no source in the repo names a per-trait "
             "opposite, so none is asserted here.")
    return "\n".join(L), src


def sec_constitution(s):
    rec = CONS_BY_SLUG.get(s)
    if not rec or not rec.get("constitution"):
        return None, []
    L = ["The constitution is the instruction given to the teacher model that "
         "generated this trait's DPO preference pairs. It is the primary "
         "definition of the trait in this project.", "",
         blockquote(rec["constitution"])]
    src = [f"qwen35/constitutions.json#{DISPLAY[s]}.constitution"]
    extra = [k for k in ("constitution_unanchored", "constitution_enumerated")
             if rec.get(k)]
    if extra:
        L += ["", "Two variant texts are stored alongside it and are not quoted "
              "here: `" + "`, `".join(extra) + "`."]
    if rec.get("anchor"):
        L += ["", "Anchor note recorded with the constitution: " + rec["anchor"]]
        src.append(f"qwen35/constitutions.json#{DISPLAY[s]}.anchor")
    return "\n".join(L), src


def sec_weight_space(s):
    """Factor chart first, principal components second.

    Reordered 2026-09-08 with the rest of the project: the five oblimin factors
    are the primary frame and the six principal components are the secondary one,
    so a reader who stops after the first table has read the factor answer. The
    PC scores are unchanged and still here, below.
    """
    L, src = [], []
    if s in VIZ_FA_IDX:
        i = VIZ_FA_IDX[s]
        co = VIZ_FA["coords"][i]
        L.append("Factor-chart coordinates. The chart is an orthonormal basis of "
                 "the span of the five oblimin factor directions, obtained by "
                 "Gram-Schmidt in the exact Gram inner product in the fixed order "
                 "below, so a coordinate is this adapter's inner product with a "
                 "unit basis vector (`qwen35/fa_chart.py`):")
        L.append("")
        L.append("| " + " | ".join(FA_TITLES) + " |")
        L.append("| " + " | ".join("---" for _ in FA_TITLES) + " |")
        L.append("| " + " | ".join(n4(x) for x in co) + " |")
        L.append("")
        L.append(TABLE_NOTE)
        L.append("")
        L.append(f"Chart length {num(VIZ_FA['chart_len'][i])} against an adapter "
                 f"norm of {num(VIZ_FA['adapter_norm'][i])}: the five factors see "
                 f"a fraction {num(VIZ_FA['chart_frac_of_norm'][i])} of this "
                 f"adapter's weight change, where the mean over the 134 is "
                 f"{num(VIZ_FA['summary']['chart_captures_frac_of_norm_mean'])}.")
        src.append(f"qwen35/analysis/viz_fa.json#coords[{i}]")
        src.append(f"qwen35/analysis/viz_fa.json#chart_len[{i}]")
        src.append("qwen35/analysis/viz_fa.json#traits")
        src.append("qwen35/fa_chart.py")

    fa = fa_assignment(s)
    if fa:
        idx, load, rec = fa
        L.append("")
        L.append("Loadings on the k=5 centred oblimin factor solution. These are "
                 "the solution's own pattern coefficients, not chart coordinates: "
                 "the two agree in sign and rank but not in scale.")
        L.append("")
        L.append("| " + " | ".join(FA_NAMES) + " |")
        L.append("| " + " | ".join("---" for _ in FA_NAMES) + " |")
        L.append("| " + " | ".join(n4(x) for x in load) + " |")
        L.append("")
        L.append(TABLE_NOTE)
        sign = "positively" if load[idx] >= 0 else "negatively"
        L.append("")
        L.append(f"Largest absolute loading: **{FA_NAMES[idx]}**, "
                 f"{num(load[idx])}, loading {sign}.")
        L.append(f"Communality {num(rec['communality_centred_k5'])}, uniqueness "
                 f"{num(rec['uniqueness_centred_k5'])}, squared multiple "
                 f"correlation {num(rec['smc'])}.")
        src.append(f"qwen35/results/fa_qwen35.json#per_trait.{DISPLAY[s]}."
                   "oblimin_loadings_centred_k5")
        src.append("qwen35/analysis/fa_summary.json#centred_k5.factors")

    if s in VIZ_IDX:
        sc = VIZ["scores"][VIZ_IDX[s]]
        L.append("")
        L.append("The secondary frame. PC scores, centred PCA over the 134 "
                 "stage-1 sketches (253,952 dimensions):")
        L.append("")
        L.append("| PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        L.append("| " + " | ".join(n4(x) for x in sc) + " |")
        L.append("")
        L.append(TABLE_NOTE)
        src.append(f"qwen35/analysis/viz.json#scores[{VIZ_IDX[s]}]")
        src.append("qwen35/analysis/viz.json#traits")
        L.append("")
        L.append("The poles of the first three components, as listed by the "
                 "loadings file: PC1 positive "
                 + ", ".join(PC_LOADINGS["pcs"]["PC1"]["pos"]) + "; PC1 negative "
                 + ", ".join(PC_LOADINGS["pcs"]["PC1"]["neg"]) + ".")
        src.append("qwen35/analysis/pc_loadings.json#pcs.PC1")

    nb = NEIGH.get(s)
    if nb:
        L.append("")
        L.append("Nearest neighbours: the five highest-cosine edges this trait has "
                 "in the K=5 nearest-neighbour graph over the stage-1 sketch "
                 "cosines. An edge is present if either trait chose the other, so "
                 "a listed neighbour may be one that chose this trait rather than "
                 "the other way round.")
        L.append("")
        L.append("")
        L.append("| neighbour | cosine |")
        L.append("| --- | --- |")
        for t, c in nb:
            L.append(f"| [[trait-{t}]] | {n4(c)} |")
        src.append("qwen35/analysis/trait_graph.json#stage1.edges")

    if s in RANK_RAW:
        L.append("")
        L.append(f"N x N scoring: the trait's own adapter is ranked "
                 f"**{RANK_RAW[s]}** of 134 on raw scores and "
                 f"**{RANK_COLZ[s]}** of 134 after column z-scoring. "
                 f"Across the zoo, top-1 is {NXN['raw']['top1']}/134 raw and "
                 f"{NXN['column-z']['top1']}/134 column-z. The identity of the "
                 f"runner-up adapter is not stored per trait, only the aggregate "
                 f"share of runners-up sharing factor and keying, so none is named.")
        src.append(f"qwen35/analysis/nxn_summary.json#raw.ranks.{s}")
        src.append(f"qwen35/analysis/nxn_summary.json#column-z.ranks.{s}")

    if s in ALIGN_GEO:
        a, b = ALIGN_GEO[s], ALIGN_GEO_COMMON[s]
        L.append("")
        L.append("Angle to the zoo. Each adapter is treated as a line, so the "
                 "angle is the arccos of the absolute cosine. Two arms were "
                 "trained: `data_alignment` uses this trait's own prompt pool, "
                 "`data_alignment_common` the pool the zoo shares.")
        L.append("")
        L.append("| arm | nearest of the 134 | degrees | next two |")
        L.append("| --- | --- | --- | --- |")
        for label, g in (("own pool", a), ("common pool", b)):
            nxt = ", ".join(f"{t} ({n4(d)})" for t, d in g["next"])
            L.append(f"| {label} | [[trait-{g['nearest']}]] "
                     f"| {n4(g['deg'])} | {nxt} |")
        L.append("")
        L.append(TABLE_NOTE)
        L.append("")
        L.append("Big Five chart coordinates (unit norm), own-pool arm, in the "
                 "order Extraversion, Agreeableness, Conscientiousness, "
                 "EmotionalStability, Intellect: "
                 + ", ".join(n4(x) for x in a["chart"]) + ".")
        L.append("")
        L.append("Cosine with named directions, own-pool arm: "
                 + "; ".join(f"{k} {n4(v)}" for k, v in a["cos"].items()) + ".")
        L.append("")
        L.append(f"Sycophantic and obsequious, the deliberate near-synonym pair "
                 f"that sets a within-batch noise floor, sit "
                 f"{num(ALIGN_GEO['_pair_deg'])} degrees apart on the own-pool arm "
                 f"and {num(ALIGN_GEO_COMMON['_pair_deg'])} on the common-pool arm.")
        src += [f"qwen35/analysis/alignment_geometry.json#{s}",
                f"qwen35/analysis/alignment_geometry_aligncommon.json#{s}",
                "qwen35/analysis/alignment_geometry.json#_pair_deg"]

    if s in HOLE_GEO:
        g = HOLE_GEO[s]
        L.append("")
        L.append("The hole is the widest empty direction in the top-5 PC subspace "
                 "of the zoo; u is that direction at k=5 and v its full-space "
                 "counterpart. Angles treat each adapter as a line.")
        L.append("")
        L.append(f"- Angle to u in the k=5 subspace: {num(g['deg_to_u_k5'])} "
                 f"degrees, sign {g['sign_u']}")
        L.append(f"- Angle to v in the full space: {num(g['deg_to_v_full'])} degrees")
        L.append(f"- Fraction of its norm inside the top-5 subspace: "
                 f"{num(g['frac_in_top5'])}")
        L.append(f"- Nearest of the 134: [[trait-{g['nearest']}|{g['nearest']}]] at "
                 f"{num(g['deg_nearest'])} degrees")
        L.append(f"- Coordinates in the top-5 PC subspace (unit norm): "
                 + ", ".join(n4(x) for x in g["pc_coords"]))
        L.append("")
        L.append(f"For reference, the nearest existing adapter to the hole "
                 f"direction is {num(HOLE_GEO['_hole_nearest_existing_deg'])} "
                 f"degrees from u at k=5 and "
                 f"{num(HOLE_GEO['_hole_nearest_existing_full_deg'])} degrees in "
                 f"the full space. Pairwise angles among the three hole words: "
                 + "; ".join(f"{k} {n4(v)}"
                             for k, v in HOLE_GEO["_pairwise"].items()) + ".")
        L.append(f"The probability that one named word lands within the "
                 f"observed angle by chance is "
                 f"{num(HOLE_GEO['_p_one'])}, and that any of the three does, "
                 f"{num(HOLE_GEO['_p_any_of_three'])}.")
        src += [f"qwen35/analysis/hole_geometry.json#{s}",
                "qwen35/analysis/hole_geometry.json#_pairwise",
                "qwen35/analysis/hole_geometry.json#_p_any_of_three"]

    if s in SEEDPAIRED:
        L.append("")
        L.append(f"Cross-seed replication of stage 1: this trait was retrained at a "
                 f"second seed, and the cosine between the two sketches of the same "
                 f"trait is {num(SEEDPAIRED[s])}. Sketch norms "
                 f"{num(SEEDPAIRED_NORMA[s])} and {num(SEEDPAIRED_NORMB[s])}. "
                 f"These come from the earlier site_traits build, whose aggregate "
                 f"({num(SITE['cross_seed_factor']['same_trait']['mean'])} mean over "
                 f"40 traits) matches the current "
                 f"analysis/crossseed_arms.json same-trait mean.")
        src += [f"qwen35/site_traits/data.json#seedpaired.self_cos "
                f"(index of {s} in seedpaired.names)",
                "qwen35/analysis/crossseed_arms.json"]

    if not L:
        return None, []
    return "\n".join(L), src


def sec_behaviour(s):
    L, src = [], []
    st = STEER.get(s)
    if st:
        L.append("Steering the base model along this adapter's direction. "
                 "Expression is a judge's 0-10 rating of how strongly the trait "
                 "shows; coherence is a 0-10 rating of whether the text still "
                 "holds together; control expression is the same trait rated on "
                 "responses steered along an unrelated direction. Judge: "
                 f"{SITE['steering']['judge_model']}.")
        L.append("")
        L.append("| alpha | expression | n | coherence | n | control expression |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for d in st["doses"]:
            L.append(f"| {n4(d['alpha'])} | {n4(d['expression'])} | {d['n_expr']} "
                     f"| {n4(d['coherence'])} | {d['n_coh']} "
                     f"| {n4(d['ctrl_expression'])} |")
        L.append("")
        L.append(f"Baseline expression with no steering: "
                 f"{num(st['baseline_expression'])}.")
        L.append("")
        L.append(TABLE_NOTE + " This is the earlier site_traits build "
                 "(historical); it is the only per-trait steering record in the repo.")
        src.append(f"qwen35/site_traits/data.json#steering.per_trait.{s}.doses")

    ae = AE.get(s)
    if ae:
        L.append("")
        L.append("`qwen35/analysis/adapter_effect.json` carries a five-field record "
                 "for this trait. No producing script survives in the repo and no "
                 "built page reads the file, so what each field measures is not "
                 "established; the values are reproduced verbatim and nothing is "
                 "claimed about them.")
        L.append("")
        L.append("| sim_base | sim_s1 | rep | leak | chars |")
        L.append("| --- | --- | --- | --- | --- |")
        L.append(f"| {n4(ae['sim_base'])} | {n4(ae['sim_s1'])} | {n4(ae['rep'])} "
                 f"| {n4(ae['leak'])} | {n4(ae['chars'])} |")
        L.append("")
        L.append(TABLE_NOTE)
        src.append(f"qwen35/analysis/adapter_effect.json (record with trait={s})")

    if s in ALIGN_GEN:
        g = ALIGN_GEN[s]
        L.append("")
        L.append(f"Steered at alphas {', '.join(n4(a) for a in g['alphas'])} over "
                 f"{len(g['prompts'])} prompts; raw direction norm "
                 f"{num(g['dir_norm_raw'])}. The generations are stored but no "
                 f"per-trait judged aggregate over them is.")
        src.append(f"qwen35/phase10_runs/alignment_results.json "
                   f"(record with name={s})")

    if not [x for x in L if x.strip()]:
        return None, []
    if s in EVAL_BY_TRAIT:
        L.append("")
        L.append("Judged Big Five scores: `qwen35/phase10_runs/judged_100.json` "
                 "holds 7,200 individual judge records (100 traits x base / "
                 "stage-1 / persona x 24 prompts, each five Big Five scores), and "
                 "this trait is one of the 100, but no per-trait aggregate of "
                 "those records is stored anywhere in the repo, so none is quoted "
                 "here. See [[judged-evaluations]] for the zoo-level result.")
        src.append("qwen35/phase10_runs/judged_100.json#records")
    else:
        L.append("")
        L.append("Judged Big Five scores: the 100-trait judged run "
                 "(`qwen35/phase10_runs/judged_100.json`) does not cover this "
                 "trait. See [[judged-evaluations]].")
    if not [x for x in L if x.strip()]:
        return None, []
    return "\n".join(L), src


def sec_examples(s):
    L, src = [], []
    ev = EVAL_BY_TRAIT.get(s)
    picked = []
    if ev:
        for cond in ("stage1", "persona"):
            responses = ev["generations"].get(cond) or []
            for i, r in enumerate(responses[:4]):
                if not has_emoji(r):
                    picked.append((cond, i, r, ev["prompts"][i]))
                    break
            else:
                if responses:
                    picked.append((cond, 0, responses[0], ev["prompts"][0]))
        for cond, i, r, prompt in picked:
            txt, cut, had = sample_text(r)
            L.append(f"Prompt: {prompt}")
            L.append("")
            L.append(f"Condition `{cond}`:")
            L.append("")
            L.append("```")
            L.append(txt + (" ..." if cut else ""))
            L.append("```")
            note = ["truncated to 300 characters" if cut else "full response",
                    "whitespace collapsed"]
            if had:
                note.append("emoji removed")
            L.append(f"({', '.join(note)})")
            L.append("")
            src.append(f"qwen35/phase10_runs/eval_100traits.json "
                       f"(record with trait={s}).generations.{cond}[{i}]")
    elif s in ACTGEN:
        for i, r in enumerate(ACTGEN[s][:2]):
            txt, cut, had = sample_text(r)
            L.append("```")
            L.append(txt + (" ..." if cut else ""))
            L.append("```")
            note = ["truncated to 300 characters" if cut else "full response",
                    "whitespace collapsed"]
            if had:
                note.append("emoji removed")
            L.append(f"({', '.join(note)})")
            L.append("")
            src.append(f"qwen35/analysis/actspace_generations_adapters.jsonl "
                       f"(line with trait={s}).responses[{i}]")
        L.insert(0, "Greedy responses from the base model with this trait's stage-1 "
                    "adapter applied, over the shared activation-space prompt pool.")
        L.insert(1, "")
    elif s in ALIGN_GEN:
        g = ALIGN_GEN[s]
        gens = g["generations"].get("1.0") or []
        for i, r in enumerate(gens[:2]):
            txt, cut, had = sample_text(r)
            L.append(f"Prompt: {g['prompts'][i]}")
            L.append("")
            L.append("```")
            L.append(txt + (" ..." if cut else ""))
            L.append("```")
            note = ["truncated to 300 characters" if cut else "full response",
                    "whitespace collapsed"]
            if had:
                note.append("emoji removed")
            L.append(f"({', '.join(note)})")
            L.append("")
            src.append(f"qwen35/phase10_runs/alignment_results.json "
                       f"(record with name={s}).generations.1.0[{i}]")
        if L:
            L.insert(0, "Base model steered along this adapter's direction at "
                        "alpha 1.0. These are steered generations, not the adapter "
                        "applied at its trained strength.")
            L.insert(1, "")
    if not L:
        return None, []
    return "\n".join(L).rstrip(), src


def sec_actspace(s):
    if s not in CROSS_TRAITS:
        return None, []
    L = [f"`{s}` is one of the {len(CROSS_TRAITS)} CROSS_TRAITS: the traits for "
         f"which the constitution was also run as a system prompt on the base "
         f"model, so that the activation-space direction P and the weight-space "
         f"adapter A can be compared on the same trait. Layer "
         f"{jload('analysis/actspace_geometry.json')['primary_layer']} of 33, "
         f"response-token window."]
    src = ["qwen35/act_space.py#CROSS_TRAITS",
           "qwen35/analysis/actspace_geometry.json#primary_layer"]
    r = ACT_MATCHED.get(s)
    if r:
        L.append("")
        L.append("Matched entry (adapter t and constitution s both this trait). "
                 "Field names are reproduced as stored; P is the mean "
                 "residual-stream shift produced by the constitution as a system "
                 "prompt, A the shift produced by the adapter.")
        L.append("")
        L.append("| field | value |")
        L.append("| --- | --- |")
        for k in ("cos_P_t", "cos_A_t", "cos_P_s", "cos_A_s", "resid_add",
                  "resid_prompt_only", "resid_adapter_only", "resid_fit",
                  "a", "b", "norm_ratio", "along_P_t", "adapter_contrib_cos",
                  "prompt_contrib_cos"):
            if k in r:
                L.append(f"| `{k}` | {n4(r[k])} |")
        L.append("")
        L.append(TABLE_NOTE)
        src.append(f"qwen35/analysis/actspace_cross_geometry.json#resp.matched "
                   f"(entry with t={s})")
    rs = ACT_MATCHED_SPEC.get(s)
    if rs:
        L.append("")
        L.append("The same entry with the component every trait shares removed "
                 "(`resp_specific`): "
                 + ", ".join(f"`{k}` {n4(rs[k])}" for k in
                             ("cos_P_s", "cos_A_t", "resid_add", "resid_fit")
                             if k in rs) + ".")
        src.append(f"qwen35/analysis/actspace_cross_geometry.json#resp_specific."
                   f"matched (entry with t={s})")
    return "\n".join(L), src


def sec_training(s):
    L, src = [], []
    rm = RUNMETA.get(s)
    if rm:
        L.append("Stage 1, DPO on constitution-generated preference pairs:")
        L.append("")
        L.append(f"- Base model {rm['resolved_base_model']} at commit "
                 f"`{rm['base_model_commit']}`")
        L.append(f"- LoRA rank {rm['lora_r']}, alpha {rm['lora_alpha']}, scaling "
                 f"{n4(rm['expected_scaling'])}, {rm['n_targeted']} targeted linear "
                 f"modules of {rm['n_total_linear']}")
        L.append(f"- {rm['n_pairs']} preference pairs, {rm['epochs']} epoch, "
                 f"effective batch {rm['effective_batch']}, "
                 f"{rm['optimizer_steps']} optimizer steps, learning rate "
                 f"{n4(rm['learning_rate'])}, beta {n4(rm['beta'])}, seed "
                 f"{rm['seed']}")
        L.append(f"- Loss {num(rm['loss_first'])} to {num(rm['loss_last'])}; "
                 f"reward margin {num(rm['reward_margin'])}; reward accuracy "
                 f"{num(rm['reward_accuracy'])}; {num(rm['train_seconds'])} seconds "
                 f"on {rm['stack']['gpu']}")
        L.append(f"- Pair corpus sha256 `{rm['data_sha256']}`, shared prompt pool "
                 f"sha256 `{rm['prompt_pool_sha256']}`")
        src.append(f"qwen35/results/runmeta_sweep.json#{s}")
        L.append("")
        L.append("`qwen35/phase5_margins.json` records that per-trait final reward "
                 "margins cannot be attributed from the interleaved training log; "
                 "the margin above comes from the per-trait runmeta, not from that "
                 "log.")
        src.append("qwen35/phase5_margins.json#note")

    for label, log in (("own pool (`data_alignment`)", LOG_ALIGN),
                       ("shared pool (`data_alignment_common`)", LOG_ALIGN_COMMON),
                       ("shared pool (`data_hole_common`)", LOG_HOLE)):
        rec = log.get(s)
        if not rec:
            continue
        L.append("")
        L.append(f"Stage 1 DPO, {label}:")
        L.append("")
        if "n_pairs" in rec:
            L.append(f"- {rec['n_pairs']} preference pairs, corpus "
                     f"`{rec.get('corpus')}`")
        L.append(f"- {rec.get('targeted')} targeted modules, "
                 f"{rec.get('excluded')} excluded")
        L.append(f"- Loss {num(rec['loss_first'])} to {num(rec['loss_last'])}"
                 + (f"; {rec['train_seconds']} seconds" if "train_seconds" in rec
                    else ""))
        logfile = ("aligntrain" if log is LOG_ALIGN else
                   "aligncommon" if log is LOG_ALIGN_COMMON else "holetrain")
        src.append(f"qwen35/phase10_runs/{logfile}.log (summary line for {s})")

    b = S2.get(s) or {}
    if any(b.get(st) for st in ("merge", "assemble", "sft", "final")):
        L.append("")
        L.append("Stage 2, OCT introspection (generate reflection and interaction "
                 "transcripts from the stage-1 model, SFT on them, merge back):")
        L.append("")
        if b.get("merge"):
            r, rel = b["merge"]
            L.append(f"- Stage-1 fold into base: {r['n_patched']} modules "
                     f"patched, max relative norm error "
                     f"{num(r['max_rel_norm_err'])}")
            src.append(f"qwen35/{rel}#stages.merge "
                       f"(record with trait={s})")
        if b.get("assemble"):
            a, rel = b["assemble"]
            L.append(f"- SFT corpus assembled: {a['n_rows']} rows, "
                     f"{a['n_kept_at_max_len']} kept at max length, "
                     f"{a['n_dropped_at_max_len']} dropped")
            src.append(f"qwen35/{rel}#stages.assemble "
                       f"(record with trait={s})")
        if b.get("sft"):
            f, rel = b["sft"]
            L.append(f"- SFT: base `{f['base']}`, {f['n_rows_trained']} rows "
                     f"trained of {f['n_rows_in']} "
                     f"({f['n_dropped_at_max_len']} dropped at max length), "
                     f"{f['n_targeted']} targeted modules, LoRA rank "
                     f"{f['lora']['lora_rank']} alpha {f['lora']['lora_alpha']}, "
                     f"learning rate {n4(f['hp']['learning_rate'])}, max length "
                     f"{f['hp']['max_len']}, seed {f['hp']['seed']}")
            L.append(f"- SFT loss {num(f['loss_first'])} to {num(f['loss_last'])} "
                     f"over {f['optimizer_steps']} optimizer steps, "
                     f"{num(f['train_seconds'])} seconds")
            src.append(f"qwen35/{rel}#stages.sft "
                       f"(record with trait={s})")
        if b.get("final"):
            w = b["final"][0]["weights"]
            L.append(f"- Persona merge weights: DPO {n4(w['dpo'])}, "
                     f"SFT {n4(w['sft'])}")
            src.append(f"qwen35/{b['final'][1]}#stages.final "
                       f"(record with trait={s})")
        gone = [st for st in ("merge", "assemble", "sft", "final")
                if not b.get(st)]
        if gone:
            L.append(f"- No unskipped record survives for the "
                     f"{', '.join(gone)} stage; the runs that redid it wrote "
                     f"`skipped: true` for this trait.")

    if s in ZOO_SET and not any((S2.get(s) or {}).get(st)
                               for st in ("merge", "assemble", "sft", "final")):
        L.append("")
        L.append("Stage 2: an adapter exists for every one of the 134, but no "
                 "unskipped per-trait OCT-2 stage record for this trait survives "
                 "in the production results files "
                 "(`phase10_runs/results_oct2_*traits_"
                 "v1-n1000-ni1000-k10-bugsfaithful.json`), so no stage-2 training "
                 "numbers are quoted. The one file that does carry a record for "
                 "some of these traits, `results_oct2_1traits_v1-n40-ni8-k4-"
                 "bugsfaithful.json`, is a smoke run at 40 reflections and 8 "
                 "interactions and is not used.")

    b1 = S2_SEED1_REC.get(s) or {}
    if b1.get("sft"):
        f, rel = b1["sft"]
        L.append("")
        L.append(f"Stage 2 was re-run at a second seed for this trait: SFT seed "
                 f"{f.get('sft_seed')}, outputs under `{f.get('oct_root')}`, "
                 f"{f['n_rows_trained']} rows trained of {f['n_rows_in']}, loss "
                 f"{num(f['loss_first'])} to {num(f['loss_last'])} over "
                 f"{f['optimizer_steps']} optimizer steps.")
        src.append(f"qwen35/{rel}#stages.sft "
                   f"(record with trait={s})")

    ma = MA.get(s)
    if ma:
        L.append("")
        L.append(f"Persona merge audit: {ma['n_modules']} modules; published "
                 f"persona norm {num(ma['norm_persona_published'])}, intended "
                 f"{num(ma['norm_intended'])}, cross term "
                 f"{num(ma['norm_cross'])}; cross over published "
                 f"{num(ma['cross_over_published'])}; cosine between published and "
                 f"intended {num(ma['cos_published_intended'])}.")
        src.append(f"qwen35/analysis/merge_audit.json (record with trait={s})")

    cs = CORPUS_SCAN.get(s)
    if cs:
        L.append("")
        L.append(f"Degeneration scan of this trait's stage-2 SFT corpus. The "
                 f"score per row is the 5-gram repetition rate of the assistant "
                 f"turns, one minus the share of distinct 5-grams; rows under 40 "
                 f"words are not scored. {cs['rows']} rows read, {cs['scored']} "
                 f"scored, mean {num(cs['mean'])}, fraction above 0.3 "
                 f"{num(cs['frac_over_03'])}, above 0.5 {num(cs['frac_over_05'])}.")
        src.append(f"qwen35/analysis/corpus_scan_all.json#{s}")
    cd = CORPUS_DEGEN.get(s)
    if cd:
        L.append(f"An earlier matched-pair scan of the same corpus, capped at "
                 f"4,000 rows, records {cd['n']} scored rows, mean "
                 f"{num(cd['mean'])}, fraction above 0.3 "
                 f"{num(cd['frac_over_03'])}.")
        src.append(f"qwen35/analysis/corpus_degeneration.json#{s}")

    stt = SITE_TRAITS.get(s)
    if stt and stt.get("desc"):
        L.append("")
        L.append("What the preference pairs actually contrast, from the earlier "
                 "site_traits build (historical):")
        L.append("")
        L.append(blockquote(stt["desc"]))
        src.append(f"qwen35/site_traits/data.json#traits (record with slug={s}).desc")

    if not L:
        return None, []
    L.append("")
    L.append(ROUND_NOTE)
    return "\n".join(L), src


def sec_artefacts(s):
    L, src = [], []
    inzoo = s in ZOO_SET
    L.append("Repository naming convention, from the uploader "
             "`qwen35/upload_zoo_batched.py`: one model repo with four subfolders, "
             "one directory per trait slug. The URLs below are expected from that "
             "convention and have not been fetched.")
    L.append("")
    if inzoo:
        L.append(f"- Stage-1 DPO adapter: "
                 f"https://huggingface.co/{HF_MODEL}/tree/main/stage1_dpo/{s}")
        L.append(f"- Stage-2 introspection adapter: "
                 f"https://huggingface.co/{HF_MODEL}/tree/main/"
                 f"stage2_introspection/{s}")
        L.append(f"- Persona merge as OCT specifies it: "
                 f"https://huggingface.co/{HF_MODEL}/tree/main/persona_merged/{s}")
        L.append(f"- Corrected persona merge: "
                 f"https://huggingface.co/{HF_MODEL}/tree/main/persona_exact/{s}")
        src.append("qwen35/upload_zoo_batched.py#REPO")
    else:
        L.append(f"- Not covered by any job in `upload_zoo_batched.py`, which "
                 f"enumerates only the sweep and OCT-2 volumes. The local adapter "
                 f"file is "
                 f"`qwen35/{'hole_files' if SETOF[s] == 'hole' else 'align_files'}"
                 f"/{s}.safetensors`"
                 + (f" and `qwen35/align_common_files/{s}.safetensors` for the "
                    f"shared-pool arm" if SETOF[s] == "alignment" else "")
                 + ". Whether it is on the Hub is not recorded, so nothing is "
                   "asserted either way.")
        src.append("qwen35/upload_zoo_batched.py#main")

    if inzoo:
        L.append("")
        L.append("Transcript dataset (stage-2 generations), expected paths in "
                 f"https://huggingface.co/datasets/{HF_DATA} : "
                 f"`self_reflection/{s}.jsonl`, `self_interaction/{s}.jsonl`, "
                 f"`self_interaction/{s}-leading.jsonl`, `sft_data/{s}.jsonl`.")
    else:
        L.append("")
        L.append("No OCT stage 2 was run for the probes, so there are no "
                 "reflection or interaction transcripts for this trait.")
    if inzoo:
        if s in HF_QUARANTINED:
            L.append("")
            L.append("The audit of 2026-08-29 lists this trait as quarantined and "
                     "therefore absent from the dataset repo.")
        elif s in HF_NOT_UPLOADED:
            L.append("")
            L.append("The audit of 2026-08-29 lists this trait among the 83 with a "
                     "complete file set on the Modal volume but not yet uploaded to "
                     "the dataset repo.")
        else:
            L.append("")
            L.append("The audit of 2026-08-29 lists this trait as neither "
                     "quarantined nor pending upload, so its files were on the "
                     "dataset repo at that date "
                     f"({HF_AUDIT['traits_on_repo']} of "
                     f"{HF_AUDIT['traits_on_volume']} traits were).")
        if s in HF_PARTIAL:
            p = HF_PARTIAL[s]
            L.append(f"Partial file set: has {', '.join(p['have'])}; missing "
                     f"{', '.join(p['missing'])}. Cause recorded as: {p['cause']}")
        src.append("qwen35/analysis/hf_dataset_audit.json#missing_not_yet_uploaded")
        src.append("qwen35/analysis/hf_dataset_audit.json#dataset_repo")

    L.append("")
    if s in SEEDPAIRED:
        L.append("- Stage 1 exists at a second seed (one of 40).")
    elif inzoo:
        L.append("- Stage 1 exists at one seed only; this trait is not among the 40 "
                 "retrained for the seed floor.")
    if s in S2_SEED1:
        L.append("- Stage 2 exists at a second seed (one of the 15 of the seed-1 "
                 "OCT run).")
    elif inzoo:
        L.append("- Stage 2 at a second seed was not run for this trait; the "
                 "seed-1 OCT run covered 15 traits.")
    if s in CROSS_TRAITS:
        L.append(f"- Activation-space constitution run exists (one of the "
                 f"{len(CROSS_TRAITS)} CROSS_TRAITS).")
    src.append("qwen35/phase10_runs/results_oct2_15traits_"
               "v1-n1000-ni1000-k10-bugsfaithful.json#traits")
    return "\n".join(L), src


def sec_links(s):
    L = []
    fa = fa_assignment(s)
    if fa:
        idx = fa[0]
        L.append(f"- Recovered factor it loads on most: [[{FA_SLUGS[idx]}]]")
    else:
        L.append("- No per-trait FA loading exists for this trait, so no recovered "
                 "factor page is linked.")
    if FACTOR[s] in AXIS_SLUG:
        L.append(f"- Big Five axis it was drawn from: [[{AXIS_SLUG[FACTOR[s]]}]]")
    L.append("- [[trait-provenance]] -- how the trait lists were built")
    L.append("- [[geometry-overview]] -- the weight-space geometry these numbers "
             "sit inside")
    L.append("- [[n-by-n-scoring]] -- what the N x N rank means")
    L.append("- [[judged-evaluations]] -- the judged Big Five protocol")
    L.append("- [[actspace-adapters]] -- activation space against weight space")
    L.append("- [[traits-index]] -- every trait in one table")
    nb = NEIGH.get(s) or []
    if nb:
        L.append("- Neighbours: " + ", ".join(f"[[trait-{t}]]" for t, _ in nb))
    for key, g in (("alignment", ALIGN_GEO.get(s)), ("hole", HOLE_GEO.get(s))):
        if g and g.get("nearest"):
            L.append(f"- Nearest zoo adapter: [[trait-{g['nearest']}]]")
            break
    return "\n".join(L), []


# --------------------------------------------------------------------------
# page assembly
# --------------------------------------------------------------------------
SECTIONS = [("Identity", sec_identity),
            ("Constitution", sec_constitution),
            ("Where it sits in weight space", sec_weight_space),
            ("Behaviour", sec_behaviour),
            ("Example generations", sec_examples),
            ("Activation space", sec_actspace),
            ("Training record", sec_training),
            ("Artefacts", sec_artefacts),
            ("Links", sec_links)]


def summary_sentence(s):
    fa = fa_assignment(s)
    bits = []
    if fa:
        idx, load, _ = fa
        bits.append(f"loads most strongly on the recovered {FA_NAMES[idx]} factor "
                    f"({n4(load[idx])})")
    nb = NEIGH.get(s) or []
    if nb:
        bits.append(f"nearest neighbour {nb[0][0]} at cosine {n4(nb[0][1])}")
    if s in ALIGN_GEO:
        g = ALIGN_GEO[s]
        bits.append(f"nearest of the 134 is {g['nearest']} at {n4(g['deg'])} degrees")
    if s in HOLE_GEO:
        g = HOLE_GEO[s]
        bits.append(f"{n4(g['deg_to_u_k5'])} degrees from the hole direction at k=5, "
                    f"nearest existing adapter {g['nearest']}")
    if not bits:
        return "Its place in the zoo geometry is recorded on this page."
    return "In weight space it " + "; ".join(bits) + "."


def summary_line(s):
    keying = {"+": "positively keyed", "-": "negatively keyed"}[KEYED[s]]
    setw = {"primary": "Goldberg primary marker",
            "lexicon": "lexicon draw",
            "alignment": "alignment probe",
            "hole": "hole-word probe"}[SETOF[s]]
    return (f"{DISPLAY[s]}: {FACTOR[s]} {keying}, {setw}. "
            + summary_sentence(s))


def build_page(s):
    body, sources, missing = [], [], []
    for heading, fn in SECTIONS:
        text, src = fn(s)
        if text is None:
            missing.append(heading)
            continue
        text = re.sub(r"\n{3,}", "\n\n", text.strip("\n"))
        body.append(f"## {heading}\n\n{text}")
        sources.extend(src)
    seen, uniq = set(), []
    for x in sources:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    tags = ["trait", factor_tag(s), SETOF[s]]
    tags = list(dict.fromkeys(tags))
    fm = ["---",
          f"title: {yq(DISPLAY[s])}",
          f"summary: {yq(summary_line(s))}",
          "status: current",
          "sources:"]
    fm += [f"  - {yq(x)}" for x in uniq]
    fm += [f"last_verified: {LAST_VERIFIED}",
           f"tags: [{', '.join(tags)}]",
           "---", ""]
    page = ("\n".join(fm) + f"# {DISPLAY[s]}\n\n"
            + "\n\n".join(body) + "\n")
    return page, missing


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def main():
    os.makedirs(OUT, exist_ok=True)
    coverage = {}
    for s in PAGES:
        page, missing = build_page(s)
        write(os.path.join(OUT, f"trait-{s}.md"), page)
        coverage[s] = missing

    # ---- index -----------------------------------------------------------
    rows = []
    for s in PAGES:
        fa = fa_assignment(s)
        faname = FA_NAMES[fa[0]] if fa else "-"
        nb = NEIGH.get(s) or []
        near = nb[0][0] if nb else (ALIGN_GEO.get(s) or HOLE_GEO.get(s) or
                                    {}).get("nearest", "-")
        rank = RANK_RAW.get(s, "-")
        rows.append(f"| [[trait-{s}]] | {DISPLAY[s]} | {FACTOR[s]} | `{KEYED[s]}` "
                    f"| {set_label(s)} | {faname} | {near} | {rank} |")
    idx = ["---",
           "title: Trait index",
           'summary: "Every trait with a page: the 134 zoo adapters, 4 '
           'alignment probes and 3 hole-word probes, with factor, keying, '
           'provenance set, recovered factor and nearest neighbour."',
           "status: current",
           "sources:",
           "  - qwen35/traits_primary.json",
           "  - qwen35/traits_secondary.json",
           "  - qwen35/traits_alignment.json",
           "  - qwen35/traits_hole.json",
           "  - qwen35/analysis/nxn_summary.json#names",
           "  - qwen35/analysis/nxn_summary.json#raw.ranks",
           "  - qwen35/results/fa_qwen35.json#per_trait",
           "  - qwen35/analysis/trait_graph.json#stage1.edges",
           f"last_verified: {LAST_VERIFIED}",
           "tags: [trait, index]",
           "---",
           "",
           "# Trait index",
           "",
           f"{len(PAGES)} pages: the {len(ZOO)} adapters of the zoo, the "
           f"{len(ALIGN_SLUGS)} alignment probes and the {len(HOLE_SLUGS)} "
           "hole-word probes trained afterwards.",
           "",
           "The recovered-factor column is the k=5 centred oblimin factor on which "
           "the trait has its largest absolute loading; it exists only for the 134 "
           "zoo traits, since the factor solution was fitted on those. The "
           "neighbour column is the highest-cosine edge in the K=5 nearest-neighbour "
           "graph, or for the probes the nearest of the 134 by angle. The rank "
           "column is the trait's own adapter's rank in raw N x N scoring.",
           "",
           "| page | trait | factor | keyed | set | recovered factor | nearest "
           "| N x N rank |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    idx += rows
    idx += ["",
            "## Drawn but not in the zoo",
            "",
            f"`traits_secondary.json` lists {len(LEXICON_DRAWN)} lexicon words, one "
            f"per cluster of the 40-cluster draw, and `constitutions.json` holds a "
            f"constitution for every one of them. Only {len(LEXICON_DRAWN) - len(LEXICON_DROPPED)} "
            f"appear in the zoo of {len(ZOO)}. The {len(LEXICON_DROPPED)} with no "
            f"adapter, and so no page, are: "
            + ", ".join(sorted(LEXICON_DROPPED)) + ". No source in the repo records "
            "why they were dropped.",
            "",
            "## Links",
            "",
            "- [[traits-by-factor]]",
            "- [[trait-provenance]]",
            "- [[geometry-overview]]"]
    write(os.path.join(OUT, "traits-index.md"), "\n".join(idx) + "\n")

    # ---- by factor -------------------------------------------------------
    byaxis = {}
    for s in PAGES:
        byaxis.setdefault(FACTOR[s], []).append(s)
    byfa = {}
    for s in PAGES:
        fa = fa_assignment(s)
        byfa.setdefault(fa[0] if fa else None, []).append(s)

    bf = ["---",
          "title: Traits by factor",
          'summary: "The 141 trait pages grouped twice: by the Big Five axis '
          'each marker was drawn from, and by the recovered factor it loads on '
          'most in the k=5 centred oblimin solution."',
          "status: current",
          "sources:",
          "  - qwen35/traits_primary.json",
          "  - qwen35/traits_secondary.json",
          "  - qwen35/traits_alignment.json",
          "  - qwen35/traits_hole.json",
          "  - qwen35/results/fa_qwen35.json#per_trait",
          "  - qwen35/analysis/fa_summary.json#centred_k5.factors",
          f"last_verified: {LAST_VERIFIED}",
          "tags: [trait, index, factors]",
          "---",
          "",
          "# Traits by factor",
          "",
          "Two groupings of the same 141 pages. The first is the label the trait "
          "arrived with: which Big Five axis Goldberg's marker list assigns it to, "
          "or `Lexicon` / `Alignment` / `Hole` for the traits that were not drawn "
          "from that list. The second is measured: which of the five factors "
          "recovered from the adapter geometry the trait loads on most strongly.",
          "",
          "## By the axis it was drawn from",
          ""]
    for f in ["Extraversion", "Agreeableness", "Conscientiousness",
              "EmotionalStability", "Intellect", "Lexicon", "Alignment", "Hole"]:
        if f not in byaxis:
            continue
        head = f"### {f}"
        if f in AXIS_SLUG:
            head += f"  ([[{AXIS_SLUG[f]}]])"
        bf.append(head)
        bf.append("")
        pos = [s for s in byaxis[f] if KEYED[s] == "+"]
        neg = [s for s in byaxis[f] if KEYED[s] == "-"]
        if neg:
            bf.append("Positively keyed: "
                      + ", ".join(f"[[trait-{s}|{DISPLAY[s]}]]" for s in pos))
            bf.append("")
            bf.append("Negatively keyed: "
                      + ", ".join(f"[[trait-{s}|{DISPLAY[s]}]]" for s in neg))
        else:
            bf.append(", ".join(f"[[trait-{s}|{DISPLAY[s]}]]" for s in byaxis[f]))
        bf.append("")

    bf.append("## By the recovered factor it loads on most")
    bf.append("")
    bf.append("Sum of squared loadings for each factor, from the oblimin rotation: "
              + "; ".join(f"{FA_NAMES[i]} "
                          f"{FA_SUM['centred_k5']['factors'][i]['ss']}"
                          for i in range(5)) + ".")
    bf.append("")
    for i in range(5):
        members = byfa.get(i, [])
        bf.append(f"### {FA_NAMES[i]}  ([[{FA_SLUGS[i]}]])")
        bf.append("")
        bf.append(f"{len(members)} traits.")
        bf.append("")
        signed = []
        for s in sorted(members, key=lambda t: -abs(
                FA_PER_TRAIT[DISPLAY[t]]["oblimin_loadings_centred_k5"][i])):
            v = FA_PER_TRAIT[DISPLAY[s]]["oblimin_loadings_centred_k5"][i]
            signed.append(f"[[trait-{s}|{DISPLAY[s]}]] ({n4(v)})")
        bf.append(", ".join(signed))
        bf.append("")
    if byfa.get(None):
        bf.append("### No FA loading")
        bf.append("")
        bf.append("The factor solution was fitted on the 134 zoo adapters only, so "
                  "the probes trained afterwards have no loading: "
                  + ", ".join(f"[[trait-{s}|{DISPLAY[s]}]]"
                              for s in byfa[None]) + ".")
        bf.append("")
    bf += ["## Links", "", "- [[traits-index]]", "- [[geometry-overview]]"]
    write(os.path.join(OUT, "traits-by-factor.md"), "\n".join(bf) + "\n")

    # ---- prune stale pages ----------------------------------------------
    want = {f"trait-{s}.md" for s in PAGES} | {"traits-index.md",
                                               "traits-by-factor.md", "_report.md"}
    removed = []
    for p in glob.glob(os.path.join(OUT, "*.md")):
        if os.path.basename(p) not in want:
            os.remove(p)
            removed.append(os.path.basename(p))

    # ---- coverage summary, printed for the section report ---------------
    heads = [h for h, _ in SECTIONS]
    print(f"pages: {len(PAGES)} trait + 2 index")
    print(f"removed stale: {removed if removed else 'none'}")
    for h in heads:
        absent = [s for s in PAGES if h in coverage[s]]
        print(f"  {h:32s} present {len(PAGES) - len(absent):3d}  absent "
              f"{len(absent):3d}"
              + (("  " + ", ".join(absent[:12])
                  + (" ..." if len(absent) > 12 else "")) if absent else ""))


if __name__ == "__main__":
    main()
