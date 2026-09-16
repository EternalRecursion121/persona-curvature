#!/usr/bin/env python3
"""Scan one trait's four corpus files for HALT-class matches, before the uploader
reaches it.

Exists because an adjudication is keyed by (file, pattern, row), and every
transcript appears TWICE: once in its own file and again in sft_data, which is
the concatenation of the other three, at a different row index. Clearing a row
in self_interaction therefore does nothing for its twin in sft_data, and the
uploader quarantines a transcript a human has already read. Running this first
turns that into one review pass covering every copy.

Usage: prescan_trait.py <trait> [<trait> ...]
"""
import importlib.util, os, subprocess, sys, tempfile, shutil

Q = "/home/vibe12/projects/persona-curvature/qwen35"
MODAL = "/home/vibe12/cartovenv/bin/modal"
VOL = "pc-qwen35-oct2"

spec = importlib.util.spec_from_file_location("ud", f"{Q}/upload_datasets.py")
ud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ud)


def main(traits):
    for t in traits:
        d = tempfile.mkdtemp(prefix="prescan_")
        try:
            print(f"===== {t} =====", flush=True)
            for remote, dest in ud.remote_files(t):
                local = os.path.join(d, dest.replace("/", "__"))
                subprocess.run([MODAL, "volume", "get", VOL, remote, local],
                               capture_output=True, text=True, timeout=1800)
                if not os.path.exists(local):
                    print(f"  {dest}: FETCH FAILED", flush=True)
                    continue
                n, hits = ud.scan(local)
                halts = {k: v for k, v in hits.items() if k in ud.HALT}
                if not halts:
                    print(f"  {dest}: {n} rows, clean", flush=True)
                    continue
                for k, v in halts.items():
                    new = [h for h in v if (dest, k, h[0]) not in ud.ADJUDICATED]
                    print(f"  {dest}: {n} rows | {k}: {len(v)} match(es), "
                          f"{len(new)} UNREVIEWED -> rows {[h[0] for h in new]}", flush=True)
                    for row, m, ctx in new:
                        print(f"      row {row} {m!r}\n        {ctx[:260]}", flush=True)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
