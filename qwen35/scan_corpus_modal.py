#!/usr/bin/env python3
"""Measure 5-gram degeneration in every trait's SFT corpus, on Modal.

Run where the volume is MOUNTED rather than downloading ~15 GB to a 7 GB box.
One container reads every file; nothing is copied.

Motivation: behavioural repetition correlates with corpus repetition at only
+0.515, and two traits (timid, shy) show ~75x amplification from a clean corpus.
So corpus quality has to be measured directly per trait, not inferred from the
adapters, or retraining targets the wrong set.
"""
import json, os
import modal

app = modal.App(os.environ.get("PC_APP_NAME", "pc-qwen35-phase10-scancorpus"))
oct_vol = modal.Volume.from_name("pc-qwen35-oct2", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/oct": oct_vol}, timeout=60 * 60, cpu=4.0)
def scan(cap: int = 12000) -> dict:
    import glob
    out = {}
    for p in sorted(glob.glob("/oct/sft_data/*.jsonl")):
        t = os.path.basename(p)[:-6]
        vals, n = [], 0
        with open(p) as fh:
            for i, line in enumerate(fh):
                if i >= cap:
                    break
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                txt = "\n".join(m.get("content") or "" for m in (r.get("messages") or [])
                                if m.get("role") == "assistant")
                w = txt.split()
                n += 1
                if len(w) < 40:
                    continue
                g = [" ".join(w[i2:i2 + 5]) for i2 in range(len(w) - 4)]
                vals.append(1 - len(set(g)) / len(g))
        if vals:
            out[t] = dict(rows=n, scored=len(vals),
                          mean=sum(vals) / len(vals),
                          frac_over_03=sum(1 for x in vals if x > 0.3) / len(vals),
                          frac_over_05=sum(1 for x in vals if x > 0.5) / len(vals))
            print(f"{t:20s} n={n:6d} mean {out[t]['mean']:.4f} "
                  f">0.3 {100*out[t]['frac_over_03']:5.2f}%", flush=True)
    return out


@app.local_entrypoint()
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res = scan.remote()
    with open(os.path.join(here, "analysis", "corpus_scan_all.json"), "w") as f:
        json.dump(res, f, indent=1)
    bad = sorted(((v["frac_over_03"], t) for t, v in res.items()), reverse=True)
    print(f"\n{len(res)} traits scanned. Worst 20 by fraction of degenerate rows:")
    for fr, t in bad[:20]:
        print(f"  {t:20s} {100*fr:6.2f}%  mean {res[t]['mean']:.4f}")
    print(f"\n>5% degenerate: {sum(1 for fr,_ in bad if fr>0.05)} traits")
    print(f">10% degenerate: {sum(1 for fr,_ in bad if fr>0.10)} traits")
