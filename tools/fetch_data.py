#!/usr/bin/env python3
"""Fetch the persona-curvature data that is not kept in git.

The analysis outputs, Gram matrices, training corpora and run provenance live in the
Hugging Face dataset  EternalRecursion/persona-curvature-results  with repo-relative
paths, so downloading the dataset into the repository root puts every file exactly
where the scripts expect it (qwen35/analysis/..., qwen35/results/..., ...).

    python tools/fetch_data.py                       # everything, ~3.9 GB, 3,384 files
    python tools/fetch_data.py --only qwen35/results # one directory family (repeatable)
    python tools/fetch_data.py --dry-run             # list what would be fetched, offline
    python tools/fetch_data.py --verify              # sha256 every file against the manifest
    python tools/fetch_data.py --verify --only qwen35/analysis
    python tools/fetch_data.py --zoo stage1_dpo/curious   # one adapter from the zoo model repo

The manifest next to this script (tools/data_manifest.json) is the source of truth for
what the dataset holds; --dry-run and --verify read it and need no network.

Requires huggingface_hub >= 0.23 (real files under local_dir, no symlinks into the
cache). Tested with 1.27. No token is needed: the dataset is public.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys

DATASET = "EternalRecursion/persona-curvature-results"
ZOO = "EternalRecursion/persona-lora-zoo-qwen35"
ZOO_DEST = "qwen35/adapters_zoo"
# dataset-root files that are not part of the repo layout
CARD_FILES = ["README.md", ".gitattributes"]

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)          # tools/ -> repo root
DEFAULT_MANIFEST = os.path.join(HERE, "data_manifest.json")


def human(n):
    """Decimal units, so the numbers match the sizes quoted in docs/DATA.md."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000


def load_manifest(path):
    with open(path) as fh:
        return json.load(fh)


def norm_prefix(p):
    return p.strip().lstrip("./").rstrip("/")


def matches(path, prefixes):
    if not prefixes:
        return True
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def allow_patterns(prefixes):
    """fnmatch patterns for snapshot_download; '*' crosses '/' in the hub's matcher."""
    if not prefixes:
        return None
    pats = []
    for p in prefixes:
        pats += [p, p + "/*"]
    return pats


def select(manifest, prefixes):
    return [f for f in manifest["files"] if matches(f["path"], prefixes)]


def sha256_of(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def hub():
    try:
        import huggingface_hub
    except ImportError:
        sys.exit("huggingface_hub is not installed:  pip install 'huggingface_hub>=0.23'")
    return huggingface_hub


def snapshot_kwargs(hf):
    """Keep real files under local_dir on every hub version.

    hub < 0.23 symlinked local_dir into the cache unless told not to; the flag was
    deprecated in 0.23 and removed in 1.0, so pass it only when the signature has it."""
    params = inspect.signature(hf.snapshot_download).parameters
    return {"local_dir_use_symlinks": False} if "local_dir_use_symlinks" in params else {}


def cmd_dry_run(manifest, prefixes, root):
    files = select(manifest, prefixes)
    total = sum(f["bytes"] for f in files)
    present = 0
    for f in files:
        p = os.path.join(root, f["path"])
        have = os.path.isfile(p) and os.path.getsize(p) == f["bytes"]
        present += have
        print(f"{'have ' if have else 'fetch'}  {human(f['bytes']):>10}  {f['path']}")
    print(f"\n{len(files)} files, {human(total)} ({total} bytes) from {manifest['dataset_repo_id']}"
          f" -> {root}; {present} already present with the right size")


def cmd_fetch(manifest, prefixes, root, revision):
    hf = hub()
    files = select(manifest, prefixes)
    if not files:
        sys.exit(f"no manifest entries match {prefixes}; see --dry-run for the layout")
    total = sum(f["bytes"] for f in files)
    print(f"fetching {len(files)} files, {human(total)} from {manifest['dataset_repo_id']} into {root}")
    os.makedirs(root, exist_ok=True)
    path = hf.snapshot_download(
        repo_id=manifest["dataset_repo_id"], repo_type="dataset", revision=revision,
        local_dir=root, allow_patterns=allow_patterns(prefixes),
        ignore_patterns=CARD_FILES, **snapshot_kwargs(hf))
    print(f"done -> {path}")
    print("note: huggingface_hub keeps download metadata in <root>/.cache/huggingface/;"
          " it is safe to delete and is not part of the repository.")


def cmd_verify(manifest, prefixes, root):
    files = select(manifest, prefixes)
    ok, missing, bad = 0, [], []
    for f in files:
        p = os.path.join(root, f["path"])
        if not os.path.isfile(p):
            missing.append(f["path"])
            continue
        if os.path.getsize(p) != f["bytes"] or sha256_of(p) != f["sha256"]:
            bad.append(f["path"])
            continue
        ok += 1
    for m in missing:
        print(f"MISSING   {m}")
    for b in bad:
        print(f"MISMATCH  {b}")
    print(f"verified {ok}/{len(files)} files ok, {len(missing)} missing, {len(bad)} mismatched"
          f" (root {root})")
    if missing and not prefixes:
        print("hint: files that were never fetched count as missing; after a subset fetch pass the same"
              " --only prefixes to --verify, or run fetch_data.py with no arguments to restore everything")
    return 0 if not missing and not bad else 1


def cmd_zoo(subsets, root, dry_run):
    dest = os.path.join(root, ZOO_DEST)
    pats = []
    for s in subsets:
        s = norm_prefix(s)
        pats += [s, s + "/*"]
    print(f"adapter zoo: {ZOO}  subsets {subsets}  -> {dest}")
    print("note: the Gram, steering and self-id jobs read adapters from the Modal volume"
          " pc-qwen35-sweep at /adapters/<trait>, not from a local path; the only local readers"
          " (analyse_hole.py, analyse_bigfive.py, analyse_alignment_fa.py) expect the flat"
          " qwen35/*_files/<trait>.safetensors layout, not this one. Convenience download only;"
          " see docs/DATA.md.")
    if dry_run:
        print("would run:")
        print(f"  snapshot_download({ZOO!r}, repo_type='model', local_dir={dest!r},"
              f" allow_patterns={pats!r})")
        print("or, with the CLI:")
        for s in subsets:
            print(f"  hf download {ZOO} --include '{norm_prefix(s)}/*' --local-dir {dest}")
        return
    hf = hub()
    os.makedirs(dest, exist_ok=True)
    path = hf.snapshot_download(repo_id=ZOO, repo_type="model", local_dir=dest,
                                allow_patterns=pats, **snapshot_kwargs(hf))
    print(f"done -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="repository root to restore into (default: parent of tools/)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="path to data_manifest.json (default: next to this script)")
    ap.add_argument("--only", action="append", default=[], metavar="PREFIX",
                    help="restrict to a repo-relative prefix, e.g. qwen35/analysis (repeatable)")
    ap.add_argument("--verify", action="store_true",
                    help="recompute sha256 of the files under --root against the manifest; no download")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be fetched, with sizes, and exit (offline)")
    ap.add_argument("--revision", default=None, help="dataset revision (default: main)")
    ap.add_argument("--zoo", action="append", default=[], metavar="SUBSET",
                    help=f"also download SUBSET (e.g. stage1_dpo/curious, persona_exact) of the"
                         f" adapter zoo {ZOO} into <root>/{ZOO_DEST}/ (repeatable)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    prefixes = [norm_prefix(p) for p in args.only if norm_prefix(p)]
    manifest = load_manifest(args.manifest)

    if args.zoo:
        # --zoo is its own mode: it never also pulls the dataset. Run the script twice
        # (once without --zoo) to get both.
        cmd_zoo(args.zoo, root, args.dry_run)
        return 0
    if args.verify:
        return cmd_verify(manifest, prefixes, root)
    if args.dry_run:
        cmd_dry_run(manifest, prefixes, root)
        return 0
    cmd_fetch(manifest, prefixes, root, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
