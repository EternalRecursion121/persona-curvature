"""Pull the 134 per-trait runmeta.json files off the volume into one local file.

WHY THIS EXISTS, and it is a correction. I reported that per-trait training health
was unrecoverable because the sweep's merged log interleaves four containers and its
metric lines carry no trait name. That was true OF THE LOG and false of the run: each
container writes `runmeta.json` beside its adapter, and that file already carries
`log_history` -- the full list of trl metric dicts -- plus a `reward_margin` taken
from the last per-step entry. The signal was never lost; I was reading the wrong
artefact and generalised "not in this log" to "not anywhere".

The metric lines are now trait-tagged as well, so future sweeps are legible from the
log alone. This script recovers what the completed one already wrote.

Each runmeta is a few KB, so this is 134 small reads inside one container rather
than anything moved across the network at size.

usage:  modal run fetch_runmeta.py   -> writes results/runmeta_sweep.json
"""
import json
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
VOLUME = os.environ.get("PC_ADAPTER_VOLUME", "pc-qwen35-sweep")

app = modal.App(os.environ.get("PC_APP_NAME", "pc-qwen35-phase6-runmeta"))
vol = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/adapters": vol}, timeout=600)
def collect(subdir: str = ""):
    # The subdirectory arrives as an ARGUMENT, not the environment — the same
    # boundary rule every sibling script follows.
    root = "/adapters" + (f"/{subdir}" if subdir else "")
    out, missing = {}, []
    for d in sorted(os.listdir(root)):
        p = f"{root}/{d}/runmeta.json"
        if not os.path.isdir(f"{root}/{d}") or d.startswith("_"):
            continue
        if not os.path.exists(p):
            missing.append(d)
            continue
        out[d] = json.load(open(p))
    return {"runmeta": out, "missing": missing}


@app.local_entrypoint()
def main(subdir: str = ""):
    subdir = subdir.strip("/")
    r = collect.remote(subdir)
    metas, missing = r["runmeta"], r["missing"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    # Namespaced whenever the input is, so a null arm's runmeta can never
    # overwrite the analysed sweep file in place.
    p = os.path.join(HERE, "results",
                     "runmeta_sweep.json" if not subdir
                     else f"runmeta_{subdir}.json")
    json.dump(metas, open(p, "w"), indent=2)
    print(f"wrote {p}: {len(metas)} traits"
          + (f", {len(missing)} MISSING runmeta: {missing[:5]}" if missing else ""))

    # A distribution is only a health check if it is ATTRIBUTED: "some trait hit
    # 18.5" and "aloof hit 18.5" answer different questions, and only the second
    # can be followed up.
    rows = [(t, m.get("reward_margin"), m.get("reward_accuracy"),
             (m.get("log_history") or [{}])[-1].get("loss"))
            for t, m in metas.items()]
    have = [r for r in rows if isinstance(r[1], (int, float))]
    print(f"  reward_margin present for {len(have)}/{len(rows)} traits")
    if have:
        have.sort(key=lambda r: r[1])
        print("  LOWEST  margin: " + ", ".join(f"{t} {v:.2f}" for t, v, _, _ in have[:5]))
        print("  HIGHEST margin: " + ", ".join(f"{t} {v:.2f}" for t, v, _, _ in have[-5:]))
        vals = [v for _, v, _, _ in have]
        import statistics
        print(f"  min {min(vals):.2f}  median {statistics.median(vals):.2f}  "
              f"max {max(vals):.2f}")
        # The gate this feeds: nothing dead, nothing collapsed.
        dead = [t for t, v, _, _ in have if v < 1.0]
        blown = [t for t, v, _, _ in have if v > 30.0]
        print(f"  margin < 1 (did not learn): {dead or 'none'}")
        print(f"  margin > 30 (collapsed):    {blown or 'none'}")
