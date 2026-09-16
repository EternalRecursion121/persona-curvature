#!/usr/bin/env python
"""
Download the Modal Volume "persona-curvature-adapters" into local ./adapters/
WITHOUT recursively listing the volume.

The volume now holds 34 runs and the recursive walk in fetch_adapters.py issues
one VolumeListFiles RPC per directory, which trips Modal's rate limiter
(ResourceExhaustedError).  The condition names are known ahead of time, so we
construct the remote paths directly and read them -- the same DIRECT-READ fast
path that drift/fetch_drift_adapters.py uses.

Usage:
    ~/cartovenv/bin/python fetch_adapters_direct.py
    ~/cartovenv/bin/python fetch_adapters_direct.py --only O,C,O_C
    ~/cartovenv/bin/python fetch_adapters_direct.py --dest /tmp/adapters --workers 2

Resumable: a file is written to <name>.part and renamed into place only after a
complete read, so any file already present is known-complete and is skipped.
"""

import argparse
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal

VOLUME = "persona-curvature-adapters"

CONDITIONS = [
    # singles
    "O", "C", "E", "A", "N",
    # compositional pairs (jointly trained)
    "O_C", "O_E", "O_A", "O_N", "C_E", "C_A", "C_N", "E_A", "E_N", "A_N",
    # union pairs
    "O_C_union", "O_E_union", "O_A_union", "O_N_union", "C_E_union",
    "C_A_union", "C_N_union", "E_A_union", "E_N_union", "A_N_union",
    # half-splits (noise floor)
    "O_h1", "O_h2", "C_h1", "C_h2", "E_h1", "E_h2",
    # reseeds (noise floor)
    "O_s1", "C_s1", "E_s1",
]

# (filename, required?)
FILES = [
    ("adapter_model.safetensors", True),
    ("adapter_config.json", True),
    ("trainmeta.json", False),
    ("train_meta.json", False),
]

MAX_TRIES = 7
BASE_SLEEP = 4.0

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def is_rate_limit(exc) -> bool:
    name = type(exc).__name__
    if "ResourceExhausted" in name:
        return True
    txt = str(exc).lower()
    return ("rate limit" in txt or "resource_exhausted" in txt
            or "resourceexhausted" in txt or "too many requests" in txt)


def is_missing(exc) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    txt = str(exc).lower()
    return "does not exist" in txt or "not found" in txt


def fetch_file(vol, remote, local):
    """Return (status, nbytes).  status in {'have', 'got', 'missing'}."""
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return "have", os.path.getsize(local)

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = local + ".part"
    for attempt in range(MAX_TRIES):
        try:
            n = 0
            with open(tmp, "wb") as f:
                for chunk in vol.read_file(remote):
                    f.write(chunk)
                    n += len(chunk)
            os.replace(tmp, local)          # atomic -> present means complete
            return "got", n
        except Exception as e:                                  # noqa: BLE001
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if is_missing(e):
                return "missing", 0
            if attempt == MAX_TRIES - 1:
                raise
            wait = BASE_SLEEP * (2 ** attempt) + random.uniform(0, 2.0)
            kind = "rate limit" if is_rate_limit(e) else type(e).__name__
            log(f"    retry {attempt + 1}/{MAX_TRIES - 1} on {remote} "
                f"({kind}) -- sleeping {wait:.1f}s")
            time.sleep(wait)
    return "missing", 0


def fetch_condition(vol, cond, dest):
    """Return dict describing what happened for one condition."""
    res = {"cond": cond, "got": 0, "have": 0, "bytes": 0,
           "missing": [], "error": None, "ok": False}
    for fname, required in FILES:
        remote = f"/{cond}/{fname}"
        local = os.path.join(dest, cond, fname)
        try:
            status, n = fetch_file(vol, remote, local)
        except Exception as e:                                  # noqa: BLE001
            res["error"] = f"{fname}: {type(e).__name__}: {e}"
            return res
        if status == "missing":
            res["missing"].append(fname)
            if required:
                return res
        else:
            res[status] += 1
            res["bytes"] += n
    # both meta names absent is tolerated; required files are present here
    res["ok"] = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="adapters")
    ap.add_argument("--volume", default=VOLUME)
    ap.add_argument("--only", default="", help="comma-separated condition names")
    ap.add_argument("--workers", type=int, default=3,
                    help="concurrent conditions (keep small: rate limits)")
    args = ap.parse_args()

    conds = CONDITIONS
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [c for c in want if c not in CONDITIONS]
        if unknown:
            print(f"warning: not in the known list: {unknown}", file=sys.stderr)
        conds = want

    dest = os.path.abspath(args.dest)
    vol = modal.Volume.from_name(args.volume, create_if_missing=False)

    print(f"volume {args.volume} -> {dest}  ({len(conds)} conditions, "
          f"{args.workers} workers)\n")

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(fetch_condition, vol, c, dest): c for c in conds}
        done = 0
        for fut in as_completed(futs):
            cond = futs[fut]
            try:
                r = fut.result()
            except Exception as e:                              # noqa: BLE001
                r = {"cond": cond, "got": 0, "have": 0, "bytes": 0,
                     "missing": [], "error": f"{type(e).__name__}: {e}",
                     "ok": False}
            results[cond] = r
            done += 1
            if r["error"]:
                tag = f"ERROR {r['error']}"
            elif not r["ok"]:
                tag = f"MISSING ({', '.join(r['missing'])})"
            else:
                miss = f"  no {'/'.join(r['missing'])}" if r["missing"] else ""
                tag = (f"ok  {r['got']} new + {r['have']} cached, "
                       f"{r['bytes'] / 1e6:.1f} MB{miss}")
            log(f"[{done:>2}/{len(conds)}] {cond:<12} {tag}")

    ok = [c for c in conds if results[c]["ok"]]
    missing = [c for c in conds if not results[c]["ok"] and not results[c]["error"]]
    errored = [c for c in conds if results[c]["error"]]
    total = sum(results[c]["bytes"] for c in conds)

    print(f"\n{len(ok)}/{len(conds)} conditions complete, "
          f"{total / 1e6:.1f} MB transferred this run, {time.time() - t0:.0f}s")
    if missing:
        print(f"NOT ON VOLUME ({len(missing)}): {', '.join(missing)}")
    if errored:
        print(f"ERRORED ({len(errored)}):")
        for c in errored:
            print(f"  {c}: {results[c]['error']}")
    no_meta = [c for c in ok if len(results[c]["missing"]) == 2]
    if no_meta:
        print(f"no trainmeta.json/train_meta.json ({len(no_meta)}): "
              f"{', '.join(no_meta)}")
    return 0 if not errored else 1


if __name__ == "__main__":
    sys.exit(main())
