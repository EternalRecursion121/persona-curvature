#!/usr/bin/env python3
"""Regenerate the zoo40 analysis page from the live volume + run log.

Deliberately recomputes the honest final loss from log_history rather than
trusting runmeta's `loss_last`, which HF Trainer under-reports on any resumed
run (accumulated loss / TOTAL steps, but the accumulator only covers
post-resume steps). Same for train_seconds, which counts only the resumed leg.
"""
import json, os, subprocess, glob, statistics as st, datetime, shutil, tempfile

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
VOL = "pc-qwen35-oct2"
OUT = f"{Q}/zoo_page/index.html"
TOTAL = 51   # batch 1 (40 balanced + `warm` from the pilot) + batch 2 (10 balanced)


def sh(*a, t=120):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=t).stdout
    except Exception:
        return ""


def completed():
    out = sh(MODAL, "volume", "ls", VOL, "/loras_introspection")
    return sorted(l.split("/")[-1].replace(".done.json", "")
                  for l in out.splitlines() if l.strip().endswith(".done.json"))


def fetch(traits):
    d = tempfile.mkdtemp()
    recs = []
    for t in traits:
        p = os.path.join(d, f"{t}.json")
        sh(MODAL, "volume", "get", VOL, f"/loras_introspection/{t}/runmeta.json", p, t=90)
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        h = m.get("log_history", [])
        real = [e["loss"] for e in h if "loss" in e]
        if not real:
            continue
        tail = sum(real[-20:]) / len(real[-20:])
        recs.append(dict(trait=m["trait"], loss_first=real[0], loss_true=round(tail, 3),
                         loss_reported=round(m["loss_last"], 3),
                         resumed=abs(m["loss_last"] - tail) > 0.15,
                         steps=m["optimizer_steps"], rows_in=m["n_rows_in"],
                         rows_trained=m["n_rows_trained"],
                         dropped=m["n_dropped_at_max_len"],
                         gpu_h=round(m["train_seconds"] / 3600, 2),
                         hp=json.dumps(m["hp"], sort_keys=True),
                         lora=json.dumps(m["lora"], sort_keys=True),
                         targeted=m["n_targeted"]))
    shutil.rmtree(d, ignore_errors=True)
    return recs


def status():
    meter = ""
    p = f"{Q}/phase10_runs/zoo40_meter.log"
    if os.path.exists(p):
        lines = [l for l in open(p) if "est_total_spend" in l]
        if lines:
            meter = lines[-1].strip()
    spend = containers = None
    if meter:
        # parse by KEY, not by leading '$' -- the line ends '... = $590.78 / $900.00'
        # and a naive '$' scan picks up the BUDGET and reports it as the spend.
        for tok in meter.split():
            if tok.startswith("est_total_spend="):
                spend = tok.split("=", 1)[1]
            if tok.startswith("containers="):
                containers = tok.split("=")[1]
    log = f"{Q}/phase10_runs/zoo40_systemd.log"
    errs = 0
    if os.path.exists(log):
        errs = sum(1 for l in open(log, errors="ignore")
                   if any(k in l for k in ("Traceback", "FunctionTimeoutError",
                                           "OutOfMemory", "CUDA out of memory")))
    active = sh("systemctl", "is-active", "zoo40-oct.service").strip()
    return dict(spend=spend or "n/a", containers=containers or "0",
                errors=errs, service=active or "unknown")


def factor_stats(R):
    """Mean measured loss by Big Five factor, with a permutation test per factor.

    Exploratory: the pattern was found by looking, and five factors were tested,
    so the Bonferroni column is the one to read.
    """
    import random
    meta = {}
    for f in ("traits_primary.json", "traits_secondary.json"):
        try:
            for r in json.load(open(os.path.join(Q, f))):
                meta[r["trait"].lower().replace(" ", "_").replace("-", "_")] = (r["factor"], r["keyed"])
        except Exception:
            pass
    pts = [(meta.get(r["trait"], ("?", "?"))[0], r["loss_true"]) for r in R]
    pts = [(f, l) for f, l in pts if f != "?"]
    if len(pts) < 6:
        return []
    rng, out = random.Random(0), []
    allv = [l for _, l in pts]
    for fa in sorted({f for f, _ in pts}):
        a = [l for f, l in pts if f == fa]
        if len(a) < 2:
            continue
        b = [l for f, l in pts if f != fa]
        obs = st.mean(a) - st.mean(b)
        v, cnt = list(allv), 0
        for _ in range(20000):
            rng.shuffle(v)
            if abs(st.mean(v[:len(a)]) - st.mean(v[len(a):])) >= abs(obs):
                cnt += 1
        p = cnt / 20000
        out.append(dict(factor=fa, n=len(a), mean=round(st.mean(a), 3),
                        delta=round(obs, 4), p=round(p, 4),
                        bonf=round(min(1.0, p * 5), 3)))
    out.sort(key=lambda d: d["mean"])
    return out


def eval_stats():
    """Behavioural check from the eval generations, if they exist.

    Reports repetition BOTH raw and after truncating at a chat-template marker,
    because the raw figure is dominated by the model running past its stop token
    into a fabricated next turn -- a harness defect, not an adapter defect.
    """
    p = f"{Q}/phase10_runs/eval_51traits.json"
    if not os.path.exists(p):
        return None
    import re
    leak_rx = re.compile(r"(\buser\b.{0,80}\bassistant\b|<think>|<\|im_start\|>|<\|im_end\|>)", re.S)

    def rep(t):
        w = t.split()
        if len(w) < 40:
            return 0.0
        g = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
        return 1 - len(set(g)) / len(g)

    def jac(a, b):
        A, B = set(a.lower().split()), set(b.lower().split())
        return len(A & B) / max(1, len(A | B))

    raw, cut, leak, sim = [], [], [], []
    for e in json.load(open(p)):
        per = e["generations"].get("persona", [])
        base = e["generations"].get("base", [])
        if not per:
            continue
        raw.append(st.mean([rep(x) for x in per]))
        cut.append(st.mean([rep(leak_rx.split(x)[0]) for x in per]))
        leak.append(sum(1 for x in per if leak_rx.search(x)) / len(per))
        n = min(len(per), len(base))
        if n:
            sim.append(st.mean([jac(base[i], per[i]) for i in range(n)]))
    if not raw:
        return None
    return dict(n=len(raw), rep_raw=round(st.mean(raw), 3), rep_cut=round(st.mean(cut), 3),
                leak_traits=sum(1 for x in leak if x > 0), leak_frac=round(st.mean(leak), 3),
                sim=round(st.mean(sim), 3), sim_max=round(max(sim), 3),
                still_high=sum(1 for x in cut if x > 0.3))


def build():
    tr = completed()
    R = fetch(tr)
    R.sort(key=lambda r: -r["dropped"])
    S = status()
    lt = [r["loss_true"] for r in R]
    dr = [r["dropped"] for r in R]
    agg = dict(n=len(R), total=TOTAL,
               loss_mean=round(st.mean(lt), 3) if lt else 0,
               loss_sd=round(st.stdev(lt), 3) if len(lt) > 1 else 0,
               loss_min=round(min(lt), 3) if lt else 0,
               loss_max=round(max(lt), 3) if lt else 0,
               resumed=sum(r["resumed"] for r in R),
               drop_med=int(st.median(dr)) if dr else 0,
               drop_max=max(dr) if dr else 0,
               gpu_h=round(sum(r["gpu_h"] for r in R), 1),
               cfg_hp=len({r["hp"] for r in R}), cfg_lora=len({r["lora"] for r in R}),
               cfg_targ=sorted({r["targeted"] for r in R}),
               drop_mean=int(st.mean(dr)) if dr else 0,
               **S)
    FS = factor_stats(R)
    EV = eval_stats()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tpl = open(f"{Q}/zoo_page/template.html").read()
    html = (tpl.replace("__DATA__", json.dumps(R))
               .replace("__AGG__", json.dumps(agg))
               .replace("__FACTORS__", json.dumps(FS))
               .replace("__EVAL__", json.dumps(EV))
               .replace("__STAMP__", stamp))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(html)
    print(f"wrote {OUT}: {len(R)}/{TOTAL} traits, spend {S['spend']}, errors {S['errors']}")


if __name__ == "__main__":
    build()
