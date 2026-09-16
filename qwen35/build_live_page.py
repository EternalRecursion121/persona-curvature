#!/usr/bin/env python3
"""Regenerate the live persona-geometry page from analysis/live_components.json.

Rerun whenever new sketches land; it reads whatever is on disk and marks any
source that is still partial, because a factor-unbalanced sample produced a
confidently wrong PC1 earlier in this project and must never render as a result.
"""
import json, os, subprocess, datetime, collections

Q = "/home/vibe12/projects/persona-curvature/qwen35"
OUT = f"{Q}/live_page/index.html"
FACTORS = ["Extraversion", "Agreeableness", "Conscientiousness",
           "EmotionalStability", "Intellect"]


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def status():
    units = ["zoo-eval", "zoo-lex", "zoo-steer", "zoo-sketch-persona"]
    st = {u: (sh("systemctl", "is-active", u + ".service") or "unknown") for u in units}
    spend = containers = "n/a"
    p = f"{Q}/phase10_runs/zoo40_meter.log"
    if os.path.exists(p):
        ls = [l for l in open(p) if "est_total_spend" in l]
        if ls:
            for tok in ls[-1].split():
                if tok.startswith("est_total_spend="):
                    spend = tok.split("=", 1)[1]
                if tok.startswith("containers="):
                    containers = tok.split("=")[1]
    done = 0
    try:
        done = len([x for x in os.listdir(f"{Q}/analysis/sketches/persona_k32")
                    if x.endswith(".npz")])
    except Exception:
        pass
    return dict(units=st, spend=spend, containers=containers, persona_sketched=done)


def build():
    D = json.load(open(f"{Q}/analysis/live_components.json"))
    FA = json.load(open(f"{Q}/analysis/fa_summary.json")) if os.path.exists(
        f"{Q}/analysis/fa_summary.json") else {}
    UM = json.load(open(f"{Q}/analysis/umap_test.json")) if os.path.exists(
        f"{Q}/analysis/umap_test.json") else {}
    RS = json.load(open(f"{Q}/analysis/live_results.json")) if os.path.exists(
        f"{Q}/analysis/live_results.json") else {}
    GR = json.load(open(f"{Q}/analysis/trait_graph.json")) if os.path.exists(
        f"{Q}/analysis/trait_graph.json") else {}
    # coverage per source, so a partial sample is visibly partial
    for src, v in D.items():
        if not v.get("ready"):
            continue
        cnt = collections.Counter()
        for l in v["components"][0]["loadings"]:
            cnt[l["f"]] += 1
        v["coverage"] = {f: cnt.get(f, 0) for f in FACTORS}
        v["balanced"] = len(set(v["coverage"].values())) == 1 and v["n"] >= 100
    S = status()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tpl = open(f"{Q}/live_page/template.html").read()
    html = (tpl.replace("__DATA__", json.dumps(D))
               .replace("__STATUS__", json.dumps(S))
               .replace("__FA__", json.dumps(FA))
               .replace("__UMAP__", json.dumps(UM))
               .replace("__RESULTS__", json.dumps(RS))
               .replace("__GRAPH__", json.dumps(GR))
               .replace("__STAMP__", stamp))
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  sources: " +
          ", ".join(f"{k}={v.get('n')}{'' if v.get('balanced') else ' (partial)'}"
                    for k, v in D.items()))


if __name__ == "__main__":
    build()
