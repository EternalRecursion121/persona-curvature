#!/usr/bin/env python3
"""Render wiki/pages/**/*.md (+ index.md, log.md) to a static site.

Usage: build_site.py [--out DIR] [--force]
Default out: /var/www/persona-wiki.  Skips the build when no source is newer
than the previous build stamp unless --force.
"""
import argparse
import html
import json
import os
import re
import shutil
import sys
import time

import markdown
import yaml

WIKI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(WIKI, "pages")
SECTION_ORDER = ["overview", "zoo", "geometry", "factors", "behaviour",
                 "actspace", "history", "conversation", "traits"]
SECTION_TITLES = {
    "overview": "Overview", "zoo": "Building the zoo", "geometry": "Weight-space geometry",
    "factors": "Factors and axes", "behaviour": "Behaviour and interventions",
    "actspace": "Activation space", "history": "History and literature",
    "conversation": "From the conversation", "traits": "Traits",
}
STATUS_ORDER = ["current", "unconfirmed", "superseded", "withdrawn", "historical"]
# built HTML pages copied verbatim into the site root: (source relative to the repo root, target name)
STATIC_FILES = [("qwen35/spider_page/index.html", "spider.html"),
                ("qwen35/sphere_page/index_fa.html", "sphere-fa.html")]

FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
WL_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


def parse(path):
    raw = open(path, encoding="utf-8").read()
    m = FM_RE.match(raw)
    fm, body = {}, raw
    if m:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError as e:
            fm = {"title": os.path.basename(path), "_fm_error": str(e)}
        body = raw[m.end():]
    if not isinstance(fm, dict):
        fm = {"title": str(fm)}
    return fm, body


def load_pages():
    pages = {}
    dupes = []
    for root, _, files in os.walk(PAGES):
        for fn in sorted(files):
            if not fn.endswith(".md") or fn.startswith("_"):
                continue
            slug = fn[:-3]
            section = os.path.relpath(root, PAGES).split(os.sep)[0]
            path = os.path.join(root, fn)
            fm, body = parse(path)
            if slug in pages:
                dupes.append((slug, path, pages[slug]["path"]))
                continue
            pages[slug] = {"slug": slug, "section": section, "path": path,
                           "fm": fm, "body": body,
                           "title": str(fm.get("title") or slug),
                           "summary": str(fm.get("summary") or ""),
                           "status": str(fm.get("status") or "unknown"),
                           "mtime": os.path.getmtime(path)}
    for slug, path, other in dupes:
        print(f"WARN duplicate slug {slug}: {path} (kept {other})", file=sys.stderr)
    return pages


def link_pass(body, pages, slug, backlinks, broken):
    def sub(m):
        target, text = m.group(1).strip(), m.group(2)
        key = target.replace(" ", "-")
        if key in pages:
            backlinks.setdefault(key, set()).add(slug)
            label = text or pages[key]["title"]
            return f'<a class="wl" href="{key}.html">{html.escape(label)}</a>'
        broken.setdefault(slug, []).append(target)
        return f'<span class="wl-broken" title="no page {html.escape(target)}">{html.escape(text or target)}</span>'
    return WL_RE.sub(sub, body)


MD = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists",
                                   "attr_list", "def_list", "footnotes"],
                       extension_configs={"toc": {"toc_depth": "2-3"}})


def render_md(text):
    MD.reset()
    return MD.convert(text)


CSS = """
:root{--bg:#f7f5f0;--fg:#1e1d1a;--muted:#6b675e;--line:#dedad0;--accent:#7a3e1d;--accent-soft:#f2e4d8;--code:#efece4;--nav:#fbf9f5;
--ok:#2f6b3a;--warn:#8a5a00;--bad:#8d2b2b;--hist:#4a5670}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--fg:#e6e2d8;--muted:#a09a8c;--line:#33353a;--accent:#e0a077;--accent-soft:#2b241f;--code:#20222a;--nav:#1d1f22;--ok:#7fc48b;--warn:#e0b45a;--bad:#e08282;--hist:#9aa8c8}}
:root[data-theme="dark"]{--bg:#17181a;--fg:#e6e2d8;--muted:#a09a8c;--line:#33353a;--accent:#e0a077;--accent-soft:#2b241f;--code:#20222a;--nav:#1d1f22;--ok:#7fc48b;--warn:#e0b45a;--bad:#e08282;--hist:#9aa8c8}
*{box-sizing:border-box}html{font-size:16px}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,"Iowan Old Style","Palatino Linotype",serif}
a{color:var(--accent)}a.wl{text-decoration-color:var(--line)}
.wl-broken{color:var(--bad);border-bottom:1px dotted var(--bad)}
.layout{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}
nav{background:var(--nav);border-right:1px solid var(--line);padding:1.2rem 1rem;font-family:system-ui,sans-serif;font-size:.85rem;position:sticky;top:0;height:100vh;overflow:auto}
nav h1{font:700 1rem/1.2 system-ui,sans-serif;margin:0 0 .8rem}nav h1 a{color:var(--fg);text-decoration:none}
nav .sec{margin:.9rem 0 .2rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:.7rem}
nav ul{list-style:none;margin:0;padding:0}nav li{margin:.15rem 0}nav a{text-decoration:none;color:var(--fg)}nav a:hover{color:var(--accent)}
nav .meta{margin-top:1rem;color:var(--muted)}
#q{width:100%;padding:.45rem .6rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
#hits{list-style:none;padding:0;margin:.4rem 0 0}#hits li{padding:.25rem 0;border-bottom:1px solid var(--line)}#hits small{display:block;color:var(--muted)}
main{padding:2rem 3rem 4rem;max-width:78ch}
main img{max-width:100%;height:auto;display:block;margin:1rem 0;border:1px solid var(--line)}
main h1{font-size:1.9rem;line-height:1.2;margin:.2rem 0 .4rem;text-wrap:balance}
main h2{font-size:1.3rem;margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.2rem}
main h3{font-size:1.05rem;margin-top:1.4rem}
.crumbs{font-family:system-ui,sans-serif;font-size:.8rem;color:var(--muted)}.crumbs a{color:var(--muted)}
.summary{color:var(--muted);font-style:italic;margin:.2rem 0 1rem}
.badges{font-family:system-ui,sans-serif;font-size:.72rem;display:flex;gap:.5rem;flex-wrap:wrap;margin:.3rem 0 1rem}
.badge{border:1px solid var(--line);border-radius:999px;padding:.1rem .6rem;color:var(--muted)}
.badge.current{color:var(--ok);border-color:var(--ok)}.badge.unconfirmed{color:var(--warn);border-color:var(--warn)}
.badge.superseded,.badge.withdrawn{color:var(--bad);border-color:var(--bad)}.badge.historical{color:var(--hist);border-color:var(--hist)}
table{border-collapse:collapse;font-size:.9rem;font-variant-numeric:tabular-nums}th,td{border:1px solid var(--line);padding:.3rem .55rem;text-align:left;vertical-align:top}
.tablewrap,pre{overflow-x:auto}pre,code{font:.85em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code);padding:.7rem .9rem;border-radius:6px}code{background:var(--code);padding:.05rem .3rem;border-radius:3px}pre code{background:none;padding:0}
blockquote{border-left:3px solid var(--accent);margin:1rem 0;padding:.2rem 1rem;background:var(--accent-soft)}
.aside{margin-top:3rem;border-top:1px solid var(--line);padding-top:1rem;font-family:system-ui,sans-serif;font-size:.85rem;color:var(--muted)}
.aside h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;border:0;margin:.8rem 0 .3rem;color:var(--muted)}
.aside ul{margin:.2rem 0;padding-left:1.1rem}
.cat h2{margin-top:1.6rem}.cat li{margin:.25rem 0}.cat .s{color:var(--muted)}
.toc{float:right;margin:0 0 1rem 1.5rem;padding:.6rem .9rem;border:1px solid var(--line);border-radius:6px;font-family:system-ui,sans-serif;font-size:.8rem;max-width:16rem;background:var(--nav)}
.toc ul{margin:0;padding-left:1rem}
@media (max-width:900px){.layout{grid-template-columns:1fr}nav{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}main{padding:1.2rem}.toc{float:none;max-width:none}}
"""

JS = """
(function(){var q=document.getElementById('q'),hits=document.getElementById('hits'),idx=null;
if(!q)return;
function load(cb){if(idx){cb();return}fetch('search.json').then(function(r){return r.json()}).then(function(d){idx=d;cb()})}
q.addEventListener('input',function(){var v=q.value.trim().toLowerCase();hits.innerHTML='';if(v.length<2)return;
load(function(){var terms=v.split(/\\s+/),scored=[];
for(var i=0;i<idx.length;i++){var p=idx[i],s=0,t=p.title.toLowerCase(),b=p.text;
for(var j=0;j<terms.length;j++){var w=terms[j];if(t.indexOf(w)>=0)s+=10;if(p.slug.indexOf(w)>=0)s+=6;if(p.summary.toLowerCase().indexOf(w)>=0)s+=4;var c=b.split(w).length-1;s+=Math.min(c,20)*0.5;if(t.indexOf(w)<0&&p.slug.indexOf(w)<0&&p.summary.toLowerCase().indexOf(w)<0&&c===0){s=-1;break}}
if(s>0)scored.push([s,p])}
scored.sort(function(a,b){return b[0]-a[0]});
scored.slice(0,25).forEach(function(x){var p=x[1],li=document.createElement('li');li.innerHTML='<a href="'+p.slug+'.html">'+p.title+'</a><small>'+p.section+' - '+p.summary.replace(/</g,'&lt;')+'</small>';hits.appendChild(li)})})});
})();
"""


def nav_html(pages, active=None):
    parts = ['<h1><a href="index.html">persona-curvature wiki</a></h1>',
             '<input id="q" type="search" placeholder="Search pages" aria-label="Search">',
             '<ul id="hits"></ul>',
             '<div class="sec">Start</div><ul>',
             '<li><a href="index.html">Home</a></li>',
             '<li><a href="catalogue.html">Catalogue (index.md)</a></li>',
             '<li><a href="log.html">Log</a></li>',
             '<li><a href="traits-index.html">All traits</a></li></ul>']
    for sec in SECTION_ORDER:
        ps = [p for p in pages.values() if p["section"] == sec]
        if not ps:
            continue
        parts.append(f'<div class="sec">{SECTION_TITLES.get(sec, sec)} ({len(ps)})</div><ul>')
        if sec == "traits":
            parts.append('<li><a href="traits-index.html">Trait index</a></li>'
                         '<li><a href="traits-by-factor.html">Traits by factor</a></li></ul>')
            continue
        for p in sorted(ps, key=lambda p: p["title"].lower()):
            cls = ' class="on"' if p["slug"] == active else ""
            parts.append(f'<li{cls}><a href="{p["slug"]}.html">{html.escape(p["title"])}</a></li>')
        parts.append("</ul>")
    parts.append(f'<div class="meta">{len(pages)} pages. Built {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}.</div>')
    return "\n".join(parts)


def shell(title, body, nav, extra_head=""):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} - persona-curvature wiki</title>
<style>{CSS}</style>{extra_head}</head><body><div class="layout"><nav>{nav}</nav><main>{body}</main></div>
<script>{JS}</script></body></html>"""


def page_html(p, pages, backlinks, nav):
    fm = p["fm"]
    crumbs = f'<div class="crumbs"><a href="index.html">wiki</a> / <a href="section-{p["section"]}.html">{SECTION_TITLES.get(p["section"], p["section"])}</a></div>'
    badges = [f'<span class="badge {html.escape(p["status"])}">{html.escape(p["status"])}</span>']
    if fm.get("last_verified"):
        badges.append(f'<span class="badge">verified {html.escape(str(fm["last_verified"]))}</span>')
    for t in (fm.get("tags") or []):
        badges.append(f'<span class="badge">{html.escape(str(t))}</span>')
    head = f'{crumbs}<h1>{html.escape(p["title"])}</h1>'
    if p["summary"]:
        head += f'<p class="summary">{html.escape(p["summary"])}</p>'
    head += f'<div class="badges">{"".join(badges)}</div>'
    body = render_md(p["_linked"])
    toc = ""
    if getattr(MD, "toc_tokens", None) and len(MD.toc_tokens) >= 3:
        toc = f'<div class="toc">{MD.toc}</div>'
    aside = ['<div class="aside">']
    srcs = fm.get("sources") or []
    if srcs:
        aside.append("<h2>Sources</h2><ul>" + "".join(
            f"<li><code>{html.escape(str(s))}</code></li>" for s in srcs) + "</ul>")
    bl = sorted(backlinks.get(p["slug"], set()))
    if bl:
        aside.append("<h2>Linked from</h2><ul>" + "".join(
            f'<li><a href="{b}.html">{html.escape(pages[b]["title"])}</a></li>' for b in bl) + "</ul>")
    aside.append(f'<h2>File</h2><code>{html.escape(os.path.relpath(p["path"], WIKI))}</code></div>')
    return shell(p["title"], head + toc + body + "".join(aside), nav)


def catalogue_html(pages, nav):
    parts = ['<h1>Catalogue</h1><p class="summary">Every page, by section, with its one-line summary and status.</p><div class="cat">']
    for sec in SECTION_ORDER:
        ps = sorted([p for p in pages.values() if p["section"] == sec], key=lambda p: p["title"].lower())
        if not ps:
            continue
        parts.append(f'<h2 id="{sec}">{SECTION_TITLES.get(sec, sec)} <span class="s">({len(ps)})</span></h2><ul>')
        for p in ps:
            parts.append(f'<li><a href="{p["slug"]}.html">{html.escape(p["title"])}</a> <span class="badge {p["status"]}">{p["status"]}</span> <span class="s">{html.escape(p["summary"])}</span></li>')
        parts.append("</ul>")
    parts.append("</div>")
    return shell("Catalogue", "".join(parts), nav)


def section_html(sec, pages, nav):
    ps = sorted([p for p in pages.values() if p["section"] == sec], key=lambda p: p["title"].lower())
    parts = [f'<h1>{SECTION_TITLES.get(sec, sec)}</h1><div class="cat"><ul>']
    for p in ps:
        parts.append(f'<li><a href="{p["slug"]}.html">{html.escape(p["title"])}</a> <span class="badge {p["status"]}">{p["status"]}</span> <span class="s">{html.escape(p["summary"])}</span></li>')
    parts.append("</ul></div>")
    return shell(SECTION_TITLES.get(sec, sec), "".join(parts), nav)


def write_index_md(pages):
    """Regenerate index.md from frontmatter (content-oriented catalogue)."""
    lines = ["# Index", "", "Catalogue of every page, generated by tools/build_site.py from page frontmatter.",
             f"Regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. {len(pages)} pages.", ""]
    for sec in SECTION_ORDER:
        ps = sorted([p for p in pages.values() if p["section"] == sec], key=lambda p: p["title"].lower())
        if not ps:
            continue
        lines.append(f"## {SECTION_TITLES.get(sec, sec)} ({len(ps)})")
        lines.append("")
        for p in ps:
            lines.append(f"- [[{p['slug']}|{p['title']}]] ({p['status']}) - {p['summary']}")
        lines.append("")
    open(os.path.join(WIKI, "index.md"), "w", encoding="utf-8").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/www/persona-wiki")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-index", action="store_true", help="do not regenerate index.md")
    a = ap.parse_args()
    stamp = os.path.join(WIKI, ".build_stamp")
    newest = 0.0
    for root, _, files in os.walk(WIKI):
        if "/.venv" in root or "/_site" in root:
            continue
        for fn in files:
            if fn.endswith((".md", ".py")):
                newest = max(newest, os.path.getmtime(os.path.join(root, fn)))
    if not a.force and os.path.exists(stamp) and os.path.getmtime(stamp) >= newest:
        print("up to date")
        return
    pages = load_pages()
    if not a.no_index:
        write_index_md(pages)
    backlinks, broken = {}, {}
    for p in pages.values():
        p["_linked"] = link_pass(p["body"], pages, p["slug"], backlinks, broken)
    nav = nav_html(pages)
    os.makedirs(a.out, exist_ok=True)
    search = []
    for p in pages.values():
        open(os.path.join(a.out, f"{p['slug']}.html"), "w", encoding="utf-8").write(
            page_html(p, pages, backlinks, nav))
        text = re.sub(r"\s+", " ", WL_RE.sub(lambda m: m.group(2) or m.group(1), p["body"]))
        search.append({"slug": p["slug"], "title": p["title"], "summary": p["summary"],
                       "section": p["section"], "text": text[:4000].lower()})
    json.dump(search, open(os.path.join(a.out, "search.json"), "w"))
    open(os.path.join(a.out, "catalogue.html"), "w").write(catalogue_html(pages, nav))
    for sec in SECTION_ORDER:
        open(os.path.join(a.out, f"section-{sec}.html"), "w").write(section_html(sec, pages, nav))
    # home = overview/home if present, else catalogue
    if "home" in pages:
        shutil.copy(os.path.join(a.out, "home.html"), os.path.join(a.out, "index.html"))
    else:
        shutil.copy(os.path.join(a.out, "catalogue.html"), os.path.join(a.out, "index.html"))
    for src, dst in STATIC_FILES:
        sp = os.path.join(os.path.dirname(WIKI), src)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(a.out, dst))
    logp = os.path.join(WIKI, "log.md")
    logbody = open(logp).read() if os.path.exists(logp) else "# Log\n\n(empty)"
    open(os.path.join(a.out, "log.html"), "w").write(
        shell("Log", render_md(link_pass(logbody, pages, "log", backlinks, broken)), nav))
    open(stamp, "w").write(str(time.time()))
    nb = sum(len(v) for v in broken.values())
    print(f"built {len(pages)} pages -> {a.out}; broken wikilinks: {nb}")
    if nb:
        for slug, ts in sorted(broken.items()):
            print(f"  {slug}: {', '.join(sorted(set(ts)))}")


if __name__ == "__main__":
    main()
