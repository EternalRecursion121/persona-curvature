#!/usr/bin/env python3
"""Wiki health check: duplicate slugs, missing frontmatter, broken wikilinks,
orphans (no inbound links), pages without sources, status values."""
import os
import re
import sys

import yaml

WIKI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(WIKI, "pages")
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
WL_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
REQUIRED = ["title", "summary", "status", "sources", "last_verified"]
STATUSES = {"current", "superseded", "unconfirmed", "withdrawn", "historical"}

pages, dupes, problems = {}, [], []
for root, _, files in os.walk(PAGES):
    for fn in files:
        if not fn.endswith(".md") or fn.startswith("_"):
            continue
        slug = fn[:-3]
        path = os.path.relpath(os.path.join(root, fn), WIKI)
        raw = open(os.path.join(root, fn), encoding="utf-8").read()
        m = FM_RE.match(raw)
        fm = {}
        if not m:
            problems.append(f"{path}: no frontmatter")
        else:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError as e:
                problems.append(f"{path}: bad yaml ({str(e).splitlines()[0]})")
        if slug in pages:
            dupes.append((slug, path, pages[slug]["path"]))
        pages[slug] = {"path": path, "fm": fm if isinstance(fm, dict) else {}, "body": raw[m.end():] if m else raw}
        for k in REQUIRED:
            if isinstance(fm, dict) and not fm.get(k):
                problems.append(f"{path}: missing frontmatter field {k}")
        if isinstance(fm, dict) and fm.get("status") not in STATUSES:
            problems.append(f"{path}: status {fm.get('status')!r} not in {sorted(STATUSES)}")
        if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", raw):
            problems.append(f"{path}: contains an emoji")

inbound = {s: set() for s in pages}
broken = []
for slug, p in pages.items():
    for m in WL_RE.finditer(p["body"]):
        t = m.group(1).strip()
        if t in pages:
            if t != slug:
                inbound[t].add(slug)
        else:
            broken.append((slug, t))
# index.md links count as inbound for orphan purposes only if the page is a trait page
orphans = [s for s, ins in inbound.items() if not ins and not s.startswith("trait-") and s not in ("home",)]

print(f"pages: {len(pages)}")
print(f"duplicate slugs: {len(dupes)}")
for d in dupes:
    print("  ", d)
print(f"frontmatter problems: {len(problems)}")
for pr in problems[:80]:
    print("  ", pr)
if len(problems) > 80:
    print(f"   ... {len(problems) - 80} more")
print(f"broken wikilinks: {len(broken)}")
from collections import Counter
for t, n in Counter(t for _, t in broken).most_common(60):
    print(f"   {t}  ({n})")
print(f"orphans (no inbound link, excluding trait pages): {len(orphans)}")
for o in sorted(orphans):
    print("  ", o)
sys.exit(1 if (dupes or broken) else 0)
