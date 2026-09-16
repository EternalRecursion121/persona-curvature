#!/usr/bin/env python
"""Query OpenRouter's model list and print candidate instruct models by size tier.

Never prints the API key.
"""
import asyncio
import os
import re
import sys

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import MODELS_URL, headers  # noqa: E402


async def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    async with aiohttp.ClientSession() as s:
        async with s.get(MODELS_URL, headers=headers()) as r:
            r.raise_for_status()
            data = (await r.json())["data"]
    rows = []
    for m in data:
        mid = m.get("id", "")
        if pat and not re.search(pat, mid, re.I):
            continue
        p = m.get("pricing", {})
        pin = float(p.get("prompt") or 0)
        pout = float(p.get("completion") or 0)
        rows.append((mid, pin * 1e6, pout * 1e6, m.get("context_length")))
    rows.sort(key=lambda r: r[1])
    for mid, pin, pout, ctx in rows:
        print(f"{mid:<60} ${pin:>8.4f}/M in  ${pout:>8.4f}/M out  ctx={ctx}")
    print(f"({len(rows)} models)")


asyncio.run(main())
