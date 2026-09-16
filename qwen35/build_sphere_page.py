#!/usr/bin/env python3
"""Fold a sphere sweep and its blind judging into the page's data.

Each sampled direction gets: its eight generations, the judge's mean Big Five
profile over those eight, the scale it scored highest on (which colours the
point), and the angle to the nearest of the 134 trait lines -- so a reader can
see whether the place they clicked is one English already has a word for.

Two spheres now exist and this script builds either:

  --suffix ""     the 2026-09-01 principal-component sphere (unchanged paths)
  --suffix _fa    the 2026-09-08 factor-chart sphere (fa_chart.FAChart basis[:3])

With --suffix _fa it additionally compares the two judged fields: every factor
sphere point is matched to the principal-component sphere point that lands
nearest to it once the PC direction is projected into the factor chart, and the
five Big Five scales are correlated across those 72 matched pairs. That is the
only way to ask whether changing the subspace changed what the model does, as
opposed to changing only which directions are called axes.

The looping ("coherence") statistics use the same detector as every other
degeneration number in this study -- analyse_alien_steer.looping, a 10-word
window repeated 4 times -- so the two spheres and the alien direction are
counted the same way.
"""
import argparse
import json
import os

import numpy as np

from analyse_alien_steer import looping

Q = os.path.dirname(os.path.abspath(__file__))
F5 = ["Extraversion", "Agreeableness", "Conscientiousness", "EmotionalStability", "Intellect"]
CLOSE_DEG = 20.0


def profiles(judged_path):
    """Mean Big Five profile per point, from the blind judge's raw records."""
    if not os.path.exists(judged_path):
        return {}
    recs = json.load(open(judged_path))["records"]
    by = {}
    for r in recs:
        by.setdefault(r["trait"], []).append(r["scores"])
    out = {}
    for n, rows in by.items():
        sc = {}
        for f in F5:
            v = [s[f] for s in rows if s.get(f) is not None]
            sc[f] = float(np.mean(v)) if v else None
        got = {f: v for f, v in sc.items() if v is not None}
        out[n] = {"scores": sc, "top": max(got, key=got.get) if got else None,
                  "n": len(rows)}
    return out


def coherence(gen):
    """How much of the sphere survives alpha = +1.5 with intact output."""
    rates, lens = [], []
    for n, g in sorted(gen.items()):
        rates.append(sum(looping(t) for t in g) / len(g))
        lens.extend(len(t) for t in g)
    rates = np.array(rates)
    return {"none": int((rates == 0).sum()), "any": int((rates > 0).sum()),
            "worst": float(rates.max()), "mean": float(rates.mean()),
            "len_mean": float(np.mean(lens))}


def smoothness(points, judged):
    """Does judged personality vary continuously with position on the sphere?

    Sampling lets one ask what a chosen direction cannot. If nearby directions
    give nearby profiles the space behaves like a map; if not, "position in
    weight space" is not a description of character at all.
    """
    scored = [p["name"] for p in points if judged.get(p["name"], {}).get("top")]
    if len(scored) < 20:
        return None
    byname = {p["name"]: p["u"] for p in points}
    U = np.array([byname[n] for n in scored])
    P = np.array([[judged[n]["scores"][f] for f in F5] for n in scored])
    ang, prof = [], []
    for i in range(len(scored)):
        for j in range(i + 1, len(scored)):
            ang.append(np.degrees(np.arccos(np.clip(U[i] @ U[j], -1, 1))))
            prof.append(float(np.linalg.norm(P[i] - P[j])))
    ang, prof = np.array(ang), np.array(prof)
    ra = np.argsort(np.argsort(ang)).astype(float)
    rp = np.argsort(np.argsort(prof)).astype(float)
    ra -= ra.mean(); rp -= rp.mean()
    rho = float(ra @ rp / (np.linalg.norm(ra) * np.linalg.norm(rp)))
    near = prof[ang < 30].mean() if (ang < 30).any() else float("nan")
    far = prof[ang > 120].mean() if (ang > 120).any() else float("nan")
    return {"rho": rho, "n_pairs": len(ang), "near_mean": float(near),
            "far_mean": float(far), "n_scored": len(scored)}


def field_vs_pc(points, judged):
    """Match every factor-sphere point to the nearest PC-sphere point, in the
    factor chart, and correlate the two judged fields scale by scale.

    The PC sphere's directions are coefficient vectors over the same 134
    adapters, so they have a position in the factor chart; a PC direction whose
    projection into the top-3 factor space is short is barely in this sphere at
    all, and the match angle and projection length are reported alongside the
    correlations so a small overlap cannot masquerade as agreement.

    Signed, not absolute: steering at +u and -u give different personalities, so
    matching on |cos| would pair a direction with its opposite.
    """
    from fa_chart import FAChart
    pcspec = f"{Q}/phase10_runs/sphere_spec.json"
    pcjud = f"{Q}/phase10_runs/judged_sphere.json"
    if not (os.path.exists(pcspec) and os.path.exists(pcjud)):
        return None
    ch = FAChart()
    jp = profiles(pcjud)
    J = json.load(open(pcspec))["jobs"]
    pcn, pcu, pccos = [], [], []
    for j in J:
        c = np.array([j["coef"].get(t, 0.0) for t in ch.names])
        x = (ch.basis @ ch.G @ c)[:3]
        r = float(np.linalg.norm(x))
        if r <= 0 or not jp.get(j["name"], {}).get("top"):
            continue
        pcn.append(j["name"]); pcu.append(x / r); pccos.append(r / ch.norm(c))
    if len(pcn) < 20:
        return None
    PU = np.array(pcu)
    rows, angs = [], []
    for p in points:
        if not judged.get(p["name"], {}).get("top"):
            continue
        cs = PU @ np.array(p["u"])
        i = int(np.argmax(cs))
        angs.append(float(np.degrees(np.arccos(np.clip(cs[i], -1, 1)))))
        rows.append((p["name"], pcn[i], pccos[i]))
    if not rows:
        return None
    angs = np.array(angs)

    def corr_over(mask):
        idx = [i for i in range(len(rows)) if mask[i]]
        if len(idx) < 8:
            return None
        out = {}
        for f in F5:
            a = np.array([judged[rows[i][0]]["scores"][f] for i in idx], dtype=float)
            b = np.array([jp[rows[i][1]]["scores"][f] for i in idx], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
            out[f] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return out

    # PC3 has cosine 0.30 with this three-space, so the projected principal-
    # component lattice is squashed towards a great circle and some factor-sphere
    # points have no PC direction anywhere near them. A weak correlation there is
    # poor overlap, not disagreement, so the close matches are reported separately.
    close = angs < CLOSE_DEG
    corr = corr_over(np.ones(len(rows), bool))
    return {"n_matched": len(rows), "n_pc_points": len(pcn),
            "pearson": corr,
            "close_deg": CLOSE_DEG, "n_close": int(close.sum()),
            "pearson_close": corr_over(close),
            "median_match_deg": float(np.median(angs)),
            "max_match_deg": float(angs.max()),
            "median_pc_cos_into_fa3": float(np.median([r[2] for r in rows])),
            "min_pc_cos_into_fa3": float(min(r[2] for r in rows)),
            "matches": [{"fa": r[0], "pc": r[1], "pc_cos": r[2], "deg": float(d)}
                        for r, d in zip(rows, angs)]}


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--paper:#F4F5F7;--ink:#16181D;--dim:#5B6070;--faint:#8A90A0;
 --rule:#DDE0E6;--card:#FFFFFF;--well:#ECEEF2;
 --ex:#B85E1E;--ag:#1B8168;--co:#3A56A4;--es:#764AA0;--in:#A93555;--lx:#8A90A0;
 --mark:#16181D;color-scheme:light}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:0 20px 90px;
 font:400 17px/1.65 Literata,Georgia,"Times New Roman",serif}
.w{max-width:680px;margin:0 auto}.wide{max-width:1060px;margin:0 auto}
h1{font:600 30px/1.2 Fraunces,Georgia,serif;margin:52px 0 6px}
h2{font:600 20px/1.3 Fraunces,Georgia,serif;margin:34px 0 8px}
.kick{font:600 11px/1 "IBM Plex Mono",monospace;letter-spacing:.16em;
 text-transform:uppercase;color:var(--faint);margin:0 0 16px}
p{margin:0 0 15px}
.panel{background:var(--card);border:1px solid var(--rule);border-radius:2px;
 padding:22px 24px;margin:26px 0}
.ctl{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;margin:0 0 14px}
.ctl label{font:600 10.5px/1 "IBM Plex Mono",monospace;letter-spacing:.13em;
 text-transform:uppercase;color:var(--faint);display:flex;align-items:center;gap:8px}
select{font:400 13px/1.4 "IBM Plex Mono",monospace;background:var(--well);
 color:var(--ink);border:1px solid var(--rule);border-radius:2px;padding:5px 8px;max-width:100%}
.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,340px);gap:22px}
@media(max-width:820px){.split{grid-template-columns:1fr}}
canvas{width:100%;display:block;touch-action:none;cursor:grab}
.read{background:var(--well);border-radius:2px;padding:15px 17px;font-size:15px;
 white-space:pre-wrap;max-height:430px;overflow:auto}
.readmeta{font:500 11px/1.5 "IBM Plex Mono",monospace;color:var(--faint);
 display:flex;justify-content:space-between;gap:10px;margin:0 0 8px}
.bars{display:grid;grid-template-columns:auto 1fr auto;gap:4px 9px;align-items:center;
 font:500 11px/1.4 "IBM Plex Mono",monospace;margin:14px 0 0}
.bars .bn{color:var(--dim);text-align:right}
.bars .bt{height:9px;background:var(--well);border-radius:999px;overflow:hidden}
.bars .bt i{display:block;height:100%;border-radius:999px}
.bars .bv{color:var(--ink);font-variant-numeric:tabular-nums}
.hintline{font:500 11.5px/1.5 "IBM Plex Mono",monospace;color:var(--faint);margin:10px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 12px;
 font:500 11px/1.4 "IBM Plex Mono",monospace}
.legend span{display:inline-flex;align-items:center;gap:6px;color:var(--dim)}
.legend i{width:11px;height:11px;border-radius:50%;display:inline-block}
table{border-collapse:collapse;width:100%;font:500 12px/1.5 "IBM Plex Mono",monospace;
 margin:0 0 8px}
th{text-align:right;padding:6px 10px;border-bottom:1px solid var(--rule);
 color:var(--faint);font-weight:600;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--rule);
 font-variant-numeric:tabular-nums;white-space:nowrap}
b{font-weight:600}
</style>
<div class="w">
<p class="kick">persona curvature &middot; __KICK__</p>
<h1>__H1__</h1>
__INTRO__
</div>
<div class="wide"><div class="panel" id="sphere">
<div class="ctl"><label>question <select class="sph-prompt"></select></label></div>
<div class="split">
<div><canvas></canvas><p class="hintline maptip"></p></div>
<div><p class="readmeta"></p><div class="read"></div>
<div class="legend">__LEGEND__</div><div class="bars"></div></div>
</div>
<p class="hintline">__CAPTION__</p>
</div></div>
<div class="w">__STATS__</div>
<script>window.PS=__DATA__;</script>
<script>
(function(){"use strict";
var SPH=window.PS;
var F5=["Extraversion","Agreeableness","Conscientiousness","EmotionalStability","Intellect"];
var HUE={Extraversion:"ex",Agreeableness:"ag",Conscientiousness:"co",
 EmotionalStability:"es",Intellect:"in"};
var cssv=function(n){return getComputedStyle(document.documentElement)
  .getPropertyValue("--"+n).trim();};
var col=function(f){return cssv(HUE[f]||"lx");};
var $=function(s,r){return (r||document).querySelector(s);};
var el=function(t,c,x){var n=document.createElement(t);if(c)n.className=c;
 if(x!=null)n.textContent=x;return n;};
var reduce=matchMedia("(prefers-reduced-motion: reduce)").matches;
var TAU=Math.PI*2;
function wrap(x){x%=TAU;return x<0?x+TAU:x;}
function Rot(){this.a=0.62;this.b=-0.34;}
Rot.prototype.spinBy=function(da,db){this.a=wrap(this.a+da);this.b=wrap(this.b+db);};
Rot.prototype.apply=function(p){
 var ca=Math.cos(this.a),sa=Math.sin(this.a),cb=Math.cos(this.b),sb=Math.sin(this.b);
 var x=p[0]*ca-p[2]*sa,z=p[0]*sa+p[2]*ca;
 var y=p[1]*cb-z*sb; z=p[1]*sb+z*cb; return [x,y,z];};
/* Landmark styling: the three sphere axes are the factors this space is built
   from and are drawn solid and labelled in full; everything else -- the two
   factors outside the span, the Big Five axes, the grand mean, the old
   principal components -- is a visitor from elsewhere and is drawn hollow, so
   nobody reads a short projection as a real position on this sphere. */
var LAB=SPH.landmark_label||{}, AX=SPH.axis_names||[];
function lmColour(k){
  var m=/^axis_(.+)$/.exec(k); if(m) return col(m[1]);
  if(k.indexOf("factor_")===0) return cssv("mark");
  return cssv("faint");
}
function makeSphere(root){
 if(!SPH||!SPH.gen){return;}
 var cv=$("canvas",root),ctx=cv.getContext("2d");
 var out=$(".read",root),meta=$(".readmeta",root),bars=$(".bars",root);
 var psel=$(".sph-prompt",root),tip=$(".maptip",root);
 (SPH.prompts||[]).forEach(function(p,i){
   var o=el("option",null,p.length>62?p.slice(0,62)+"\\u2026":p);
   o.value=i;psel.appendChild(o);});
 var rot=new Rot(),drag=null,spin=!reduce,sel=SPH.points[0].name,hover=null;
 function size(){var w=cv.clientWidth,h=Math.round(Math.min(w*0.85,440));
  var dpr=Math.min(devicePixelRatio||1,2);
  cv.width=w*dpr;cv.height=h*dpr;cv.style.height=h+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);return [w,h];}
 function draw(){
  var wh=size(),w=wh[0],h=wh[1],cx=w/2,cy=h/2,R=Math.min(w,h)*0.40;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle=cssv("rule");ctx.lineWidth=1;
  ctx.beginPath();ctx.arc(cx,cy,R,0,6.2832);ctx.stroke();
  SPH.traits&&Object.keys(SPH.traits).forEach(function(t){
   var o=SPH.traits[t],q=rot.apply(o.u);
   if(q[2]<-0.05)return;
   ctx.globalAlpha=0.16+0.2*((q[2]+1)/2);
   ctx.fillStyle=col(o.factor);
   ctx.beginPath();ctx.arc(cx+q[0]*R,cy-q[1]*R,2.1,0,6.2832);ctx.fill();});
  ctx.globalAlpha=1;root._hit=[];
  var P=SPH.points.map(function(p){return {n:p.name,q:rot.apply(p.u)};});
  P.sort(function(a,b){return a.q[2]-b.q[2];});
  P.forEach(function(o){
   var x=cx+o.q[0]*R,y=cy-o.q[1]*R,dep=(o.q[2]+1)/2;
   var sc=SPH.judged&&SPH.judged[o.n];var f=sc?sc.top:null;
   ctx.globalAlpha=0.30+0.70*dep;
   ctx.fillStyle=f?col(f):cssv("faint");
   var r=(o.n===sel?6.5:4.2)*(0.62+0.5*dep);
   ctx.beginPath();ctx.arc(x,y,r,0,6.2832);ctx.fill();
   if(o.n===sel){ctx.globalAlpha=1;ctx.strokeStyle=cssv("ink");ctx.lineWidth=1.6;
    ctx.beginPath();ctx.arc(x,y,r+4,0,6.2832);ctx.stroke();}
   root._hit.push({n:o.n,x:x,y:y,r:Math.max(r,7)});ctx.globalAlpha=1;});
  Object.keys(SPH.landmarks||{}).forEach(function(k){
   var q=rot.apply(SPH.landmarks[k]);if(q[2]<0)return;
   var x=cx+q[0]*R,y=cy-q[1]*R;
   var isAxis=AX.indexOf(k.replace("factor_",""))>=0;
   ctx.strokeStyle=lmColour(k);ctx.fillStyle=lmColour(k);
   ctx.lineWidth=isAxis?1.8:1.1;
   if(!isAxis){ctx.setLineDash([3,3]);ctx.beginPath();ctx.arc(x,y,8,0,6.2832);
    ctx.stroke();ctx.setLineDash([]);}
   ctx.beginPath();ctx.moveTo(x-5,y);ctx.lineTo(x+5,y);
   ctx.moveTo(x,y-5);ctx.lineTo(x,y+5);ctx.stroke();
   ctx.font=(isAxis?'600 10px ':'500 9px ')+'"IBM Plex Mono",monospace';
   ctx.fillText(LAB[k]||k,x+8,y-6);});
  tip.textContent=(hover||sel)+"  \\u00b7  click a point to read the model there";
  show();}
 function show(){
  var g=(SPH.gen||{})[sel];var pi=+psel.value;
  out.textContent=g?g[pi]:"(not sampled)";
  var sc=(SPH.judged||{})[sel];
  meta.innerHTML="";
  meta.appendChild(el("span",null,sel+"  \\u00b7  alpha +"+SPH.alpha));
  meta.appendChild(el("span",null,sc&&sc.nearest?
    "nearest trait "+sc.nearest+" ("+sc.nearest_deg.toFixed(0)+"\\u00b0)":""));
  bars.innerHTML="";
  F5.forEach(function(f){
   var v=sc?sc.scores[f]:null;
   bars.appendChild(el("span","bn",f.slice(0,5)));
   var t=el("span","bt"),i=el("i");
   i.style.width=((v==null?0:(v-1)/6*100))+"%";i.style.background=col(f);
   t.appendChild(i);bars.appendChild(t);
   bars.appendChild(el("span","bv",v==null?"--":v.toFixed(2)));});}
 function at(ev){var r=cv.getBoundingClientRect();
  var x=ev.clientX-r.left,y=ev.clientY-r.top,best=null,bd=1e9;
  (root._hit||[]).forEach(function(h){var d=Math.hypot(h.x-x,h.y-y);
   if(d<h.r+4&&d<bd){bd=d;best=h.n;}});return best;}
 cv.addEventListener("pointerdown",function(ev){cv.setPointerCapture(ev.pointerId);
  drag={x:ev.clientX,y:ev.clientY,moved:false,n:at(ev)};spin=false;});
 cv.addEventListener("pointermove",function(ev){
  if(drag){if(Math.abs(ev.clientX-drag.x)+Math.abs(ev.clientY-drag.y)>3)drag.moved=true;
   rot.spinBy((ev.clientX-drag.x)*0.008,(ev.clientY-drag.y)*0.008);
   drag.x=ev.clientX;drag.y=ev.clientY;draw();}
  else{var n=at(ev);if(n!==hover){hover=n;draw();}}});
 cv.addEventListener("pointerup",function(){if(drag&&!drag.moved&&drag.n)sel=drag.n;
  drag=null;draw();});
 psel.addEventListener("change",show);
 addEventListener("resize",draw);
 draw();
 (function loop(){if(spin){rot.spinBy(0.0020,0);draw();}requestAnimationFrame(loop);})();}
var s=$("#sphere");if(s)makeSphere(s);
})();
</script>
"""

FA_LABEL = {"FA_Warmth": "Warmth", "FA_Competence": "Competence",
            "FA_FearfulWithdrawal": "Timidity", "FA_Arousal": "Arousal",
            "FA_Imagination": "Imagination"}


def esc(x):
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_html(P, s):
    """A standalone, self-contained sphere page. It shares no asset with the
    blog builder on purpose: the blog page is edited by other hands and its
    _js.txt colours landmarks by Big Five name, which is wrong here, where the
    axes are factors."""
    fa = (s == "_fa")
    lab = {}
    for k in P["landmarks"]:
        if k.startswith("factor_"):
            lab[k] = FA_LABEL.get(k[7:], k[7:])
        elif k.startswith("axis_"):
            lab[k] = "Big Five " + k[5:]
        elif k == "mean_assistant_axis":
            lab[k] = "grand mean"
        else:
            lab[k] = k
    D = dict(P)
    D["landmark_label"] = lab
    D.pop("vs_pc", None)          # the comparison is prose, not a canvas layer
    axes = P.get("axis_names") or ["PC1", "PC2", "PC3"]
    axnames = ", ".join(FA_LABEL.get(x, x) for x in axes)
    C, S = P.get("coherence") or {}, P.get("smooth") or {}
    V = P.get("var3") or []
    n_judged = sum(1 for v in P.get("judged", {}).values() if v.get("top"))
    tot = len(P.get("points", []))
    tops = {}
    for v in P.get("judged", {}).values():
        if v.get("top"):
            tops[v["top"]] = tops.get(v["top"], 0) + 1
    dist = ", ".join(f"{k} {v}" for k, v in sorted(tops.items(), key=lambda kv: -kv[1]))

    intro = (f"<p>Seventy-two directions spread near-uniformly over the unit sphere of "
             f"the space spanned by the first three factor-chart basis vectors "
             f"&mdash; <b>{esc(axnames)}</b> &mdash; each steered at "
             f"&alpha;&nbsp;=&nbsp;+{P['alpha']} and generated blind on the same eight "
             f"questions. Click any point to read what the model becomes there. The "
             f"faint dots behind are the 134 trait words on the same sphere, coloured "
             f"by the Big Five factor they were drawn for.</p>"
             f"<p>Every direction elsewhere in this study was chosen for a reason. "
             f"These were not: a lattice covers the sphere whether or not anything "
             f"interesting is there, which is the only way to find out whether a "
             f"coherent persona along a chosen direction means anything.</p>"
             if fa else
             f"<p>Seventy-two directions spread near-uniformly over the unit sphere of "
             f"the top three principal components, each steered at "
             f"&alpha;&nbsp;=&nbsp;+{P['alpha']}.</p>")
    legend = "".join(f'<span><i style="background:var(--{h})"></i>{f}</span>'
                     for f, h in [("Extraversion", "ex"), ("Agreeableness", "ag"),
                                  ("Conscientiousness", "co"),
                                  ("EmotionalStability", "es"), ("Intellect", "in")])
    cap = (f"Points are coloured by the Big Five scale the blind judge rated highest "
           f"there ({n_judged} of {tot} scored). Solid crosses are the three factors "
           f"this sphere is built from; hollow crosses are directions that live partly "
           f"outside it &mdash; the other two factors, the five Big Five axes, the "
           f"grand mean, and the three principal components of the earlier sphere.")

    st = ['<h2>What the sampling measures</h2>']
    if C:
        st.append(
            f"<p>At &alpha;&nbsp;=&nbsp;+{P['alpha']}, <b>{C['none']} of the {tot}</b> "
            f"sampled directions produce no looping at all; the worst loses "
            f"{C['worst']*100:.0f}% of its responses, a mean rate of {C['mean']*100:.2f}% "
            f"across the sphere, with a mean response of {C['len_mean']:.0f} characters. "
            f"Most directions through this space give intact output, so the coherence "
            f"of the directions we chose is not by itself evidence of anything.</p>")
    if S:
        st.append(
            f"<p>Over the {S['n_scored']} scored directions and all {S['n_pairs']:,} "
            f"pairs of them, angular distance and judged Big Five profile distance "
            f"correlate at <b>&rho; = {S['rho']:+.2f}</b> (Spearman). Directions less "
            f"than 30&deg; apart differ by {S['near_mean']:.2f} on the profile; "
            f"directions more than 120&deg; apart differ by {S['far_mean']:.2f}. "
            f"Personality varies continuously with position: this is a map, not a "
            f"list.</p>")
    if V:
        st.append(
            f"<p>The three axes carry {sum(V)*100:.1f}% of the total centred adapter "
            f"variance ({', '.join(f'{v*100:.1f}%' for v in V)}). Counting which scale "
            f"the judge rated highest at each sampled point: {esc(dist)}.</p>")
    Cmp = P.get("vs_pc")
    if Cmp:
        cl = Cmp.get("pearson_close")
        rows = "".join(
            f"<tr><td>{f}</td><td>{Cmp['pearson'][f]:+.3f}</td>"
            + (f"<td>{cl[f]:+.3f}</td>" if cl else "") + "</tr>" for f in F5)
        head = ("<tr><th>scale</th><th>all "
                f"{Cmp['n_matched']}</th>"
                + (f"<th>within {Cmp['close_deg']:.0f}&deg; ({Cmp['n_close']})</th>"
                   if cl else "") + "</tr>")
        st.append(
            f"<h2>Against the principal-component sphere</h2>"
            f"<p>Each of these {Cmp['n_matched']} directions was matched to the nearest "
            f"of the {Cmp['n_pc_points']} directions of the 2026-09-01 sphere, after "
            f"projecting those into this chart (median match "
            f"{Cmp['median_match_deg']:.1f}&deg;, worst {Cmp['max_match_deg']:.1f}&deg;; "
            f"the median principal-component direction has cosine "
            f"{Cmp['median_pc_cos_into_fa3']:.2f} with this three-space, the worst "
            f"{Cmp['min_pc_cos_into_fa3']:.2f}). Correlating the two judged fields "
            f"scale by scale over those matched pairs:</p>"
            f"<table><thead>{head}</thead>"
            f"<tbody>{rows}</tbody></table>"
            f"<p class=\"hintline\">Matching uses the signed cosine: +u and &minus;u are "
            f"different personalities, so a direction is never paired with its "
            f"opposite. The second column restricts to pairs that actually land near "
            f"each other, because a principal-component direction whose projection "
            f"into this space is short has no close counterpart here and a weak "
            f"correlation for it means poor overlap, not disagreement.</p>")

    return (PAGE.replace("__TITLE__", "The personality sphere on the factor chart"
                         if fa else "The personality sphere")
                .replace("__KICK__", "factor chart" if fa else "principal components")
                .replace("__H1__", "Sampling the space")
                .replace("__INTRO__", intro)
                .replace("__LEGEND__", legend)
                .replace("__CAPTION__", cap)
                .replace("__STATS__", "".join(st))
                .replace("__DATA__", json.dumps(D)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help='"" for the PC sphere, "_fa" for the factor sphere')
    ap.add_argument("--html", default=None, help="also write a standalone page here")
    a = ap.parse_args()
    s = a.suffix
    L = json.load(open(f"{Q}/analysis/sphere_layout{s}.json"))

    res = f"{Q}/phase10_runs/sphere_results{s}.json"
    if not os.path.exists(res):
        raise SystemExit(f"sphere_results{s}.json not present yet")
    R = json.load(open(res))
    gen = R["generations"] if "generations" in R else R

    judged = profiles(f"{Q}/phase10_runs/judged_sphere{s}.json")
    if not judged:
        print(f"no judged_sphere{s}.json yet; points will render uncoloured")

    T = np.stack([L["traits"][t]["u"] for t in sorted(L["traits"])])
    tn = sorted(L["traits"])
    for p in L["points"]:
        u = np.array(p["u"])
        d = np.abs(T @ u)
        i = int(np.argmax(d))
        e = judged.setdefault(p["name"], {"scores": {f: None for f in F5}, "top": None})
        e["nearest"] = tn[i]
        e["nearest_deg"] = float(np.degrees(np.arccos(min(d[i], 1.0))))

    smooth = smoothness(L["points"], judged)
    coh = coherence(gen)
    cmp_pc = field_vs_pc(L["points"], judged) if s == "_fa" else None

    out = dict(L)
    out["smooth"] = smooth
    out["coherence"] = coh
    out["gen"] = gen
    out["judged"] = judged
    if cmp_pc:
        out["vs_pc"] = cmp_pc
    json.dump(out, open(f"{Q}/analysis/sphere_page{s}.json", "w"))
    if a.html:
        p = a.html if os.path.isabs(a.html) else f"{Q}/{a.html}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        h = render_html(out, s).encode("ascii", "xmlcharrefreplace").decode("ascii")
        open(p, "w").write(h)
        print(f"wrote {p}  ({len(h)/1e6:.2f} MB)")

    n_top = {}
    for v in judged.values():
        if v.get("top"):
            n_top[v["top"]] = n_top.get(v["top"], 0) + 1
    print(f"{len(gen)} sampled directions with generations, "
          f"{sum(1 for v in judged.values() if v.get('top'))} judged")
    print(f"coherence: {coh['none']} of {len(gen)} directions loop on nothing, "
          f"worst {coh['worst']*100:.0f}%, mean {coh['mean']*100:.2f}%, "
          f"mean response {coh['len_mean']:.1f} chars")
    if smooth:
        print(f"\nsmoothness over {smooth['n_scored']} scored directions, "
              f"{smooth['n_pairs']} pairs:")
        print(f"  Spearman(angular distance, Big Five profile distance) = {smooth['rho']:+.3f}")
        print(f"  mean profile distance for pairs under 30 degrees apart : {smooth['near_mean']:.3f}")
        print(f"  mean profile distance for pairs over 120 degrees apart : {smooth['far_mean']:.3f}")
    print("highest-scoring scale, counted over the sphere: "
          + ", ".join(f"{k} {v}" for k, v in sorted(n_top.items(), key=lambda kv: -kv[1])))
    deg = [judged[p["name"]]["nearest_deg"] for p in L["points"]]
    print(f"angle to nearest trait line: median {np.median(deg):.1f}d  max {max(deg):.1f}d")
    if cmp_pc:
        print(f"\nfactor field vs PC field, {cmp_pc['n_matched']} matched points "
              f"(median match {cmp_pc['median_match_deg']:.1f}d, max "
              f"{cmp_pc['max_match_deg']:.1f}d; median PC cosine into the factor "
              f"3-space {cmp_pc['median_pc_cos_into_fa3']:.2f}):")
        for f in F5:
            print(f"  r({f:<18}) = {cmp_pc['pearson'][f]:+.3f}")
    print(f"wrote analysis/sphere_page{s}.json")


if __name__ == "__main__":
    main()
