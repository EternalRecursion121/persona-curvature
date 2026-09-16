#!/usr/bin/env python3
"""Replication of the OCEAN dial spider plots, on a different model and a
different construction of the dials.

The original (Llama-3.1-8B-Instruct, ten OCEAN LoRAs, LLM-judged) reports that
every dial shifts its own trait more than the others, without much impact on
capabilities. We can test that twice over, because our adapters are per
ADJECTIVE rather than per factor:

  A  steering axes -- mean(+keyed adapters) minus mean(-keyed), applied to the
     base model at alpha +2 and -2
  B  trait adapters -- the judged profile of the 10 positively-keyed adapters
     for a factor, averaged, and likewise the 10 negatively-keyed

Scale is stated rather than fitted: the original's +/-100% is "maximally
amplified/suppressed" against an unnamed reference, so here it is the judged
shift as a fraction of the room left on the scale from the base model.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/spider_page"
D = json.dumps(json.load(open(f"{Q}/analysis/spider.json")), separators=(",", ":"))

HTML = """<title>Do the Dials Turn One Thing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Public+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{--paper:#ECEEEA;--ink:#151A18;--dim:#5A625E;--rule:#CFD4CE;--card:#F5F6F3;
 --teal:#0E6E6A;--ochre:#A8482A;--grid:#C6CCC6;
 --c0:#2E6FB7;--c1:#C98A1E;--c2:#4A8B3B;--c3:#8B4FA8;--c4:#B33A2B}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --paper:#111413;--ink:#E6E9E4;--dim:#949B95;--rule:#262B29;--card:#181C1A;
 --teal:#4FB3AC;--ochre:#D97A55;--grid:#2E3432;
 --c0:#6BA6E0;--c1:#E0B45A;--c2:#7FC46A;--c3:#B888D0;--c4:#E07A65}}
:root[data-theme="dark"]{--paper:#111413;--ink:#E6E9E4;--dim:#949B95;--rule:#262B29;
 --card:#181C1A;--teal:#4FB3AC;--ochre:#D97A55;--grid:#2E3432;
 --c0:#6BA6E0;--c1:#E0B45A;--c2:#7FC46A;--c3:#B888D0;--c4:#E07A65}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:0 22px 90px;
 font:400 17px/1.65 "Public Sans",system-ui,sans-serif}
.w{max-width:1000px;margin:0 auto}
header{padding:64px 0 26px;border-bottom:1.5px solid var(--ink)}
h1{font:400 clamp(40px,7vw,74px)/1 "Instrument Serif",Georgia,serif;margin:0 0 16px;
 letter-spacing:-.015em;text-wrap:balance}
.stand{font-size:19px;color:var(--dim);max-width:66ch;margin:0}
h2{font:600 12px/1 "IBM Plex Mono",monospace;letter-spacing:.19em;text-transform:uppercase;
 color:var(--dim);margin:56px 0 0;padding-bottom:11px;border-bottom:1px solid var(--rule)}
p{max-width:70ch}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:26px 0 0}
@media(max-width:760px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:var(--card);border:1px solid var(--rule);padding:12px}
figcaption{font:400 12.5px/1.5 "IBM Plex Mono",monospace;color:var(--dim);
 margin-top:8px;text-align:center}
.leg{display:flex;flex-wrap:wrap;gap:8px 18px;justify-content:center;margin:14px 0 0;
 font:400 12.5px/1 "IBM Plex Mono",monospace;color:var(--dim)}
.leg i{display:inline-block;width:16px;height:2.5px;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;width:100%;font-size:14.5px;margin-top:20px}
th{font:600 11px/1.3 "IBM Plex Mono",monospace;letter-spacing:.07em;text-transform:uppercase;
 color:var(--dim);text-align:left;padding:6px 10px 6px 0}
td{padding:6px 10px 6px 0;border-top:1px solid var(--rule)}
td.r{text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.own{font-weight:700;color:var(--teal)}
.bad{color:var(--ochre)}
.note{border-left:3px solid var(--ochre);background:var(--card);padding:14px 18px;margin:24px 0 0}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:14px;color:var(--dim)}
code{font-family:"IBM Plex Mono",monospace;font-size:13px}
svg{display:block;width:100%;height:auto;overflow:visible}
</style>
<div class="w">
<header>
<h1>Do the Dials Turn One Thing</h1>
<p class="stand">A replication of the OCEAN amplifier/suppressor spider plots, on a different
base model and with dials built two different ways. The headline claim holds. The interesting
part is where it does not, and why the suppressors behave worse than the amplifiers.</p>
</header>

<h2>A &mdash; steering axes, &alpha; = &plusmn;2</h2>
<p>Each axis is the mean of a factor's positively-keyed trait adapters minus the mean of its
negatively-keyed ones, applied to the base model as a weighted merge. Positive &alpha; is the
amplifier, negative the suppressor. Judged blind on all five scales by a different model
family.</p>
<div class="pair">
<figure><svg id="sA" viewBox="0 0 300 300" height="300"></svg>
<figcaption>Amplifier &mdash; &alpha; = +2</figcaption></figure>
<figure><svg id="sB" viewBox="0 0 300 300" height="300"></svg>
<figcaption>Suppressor &mdash; &alpha; = &minus;2</figcaption></figure>
</div>
<div class="leg" id="leg"></div>

<h2>B &mdash; the trait adapters themselves, no steering</h2>
<p>Our adapters are per adjective, not per factor, so the same question can be asked without
any merging at all: average the judged profile of the ten positively-keyed adapters for a
factor, and separately the ten negatively-keyed. Ten independently trained models per point.</p>
<div class="pair">
<figure><svg id="sC" viewBox="0 0 300 300" height="300"></svg>
<figcaption>Positively-keyed adapters, averaged</figcaption></figure>
<figure><svg id="sD" viewBox="0 0 300 300" height="300"></svg>
<figcaption>Negatively-keyed adapters, averaged</figcaption></figure>
</div>

<h2>Where it holds and where it breaks</h2>
<div id="tbl"></div>

<div class="note" id="caveat"></div>

<footer>
Base is Qwen3.5-4B; the original is Llama-3.1-8B-Instruct, so nothing here is a like-for-like
reproduction of magnitudes. Judge is a different model family from the subject, blind to
condition, 24 open-ended prompts per condition, greedy decoding with thinking disabled.
Scale: the judged shift as a share of the room remaining on the 1&ndash;7 scale from the base
model &mdash; <code>(x&minus;base)/(7&minus;base)</code> upward and
<code>(x&minus;base)/(base&minus;1)</code> downward &mdash; so +100% is the judge's ceiling.
The original's &plusmn;100% is described as "maximally amplified/suppressed" without a stated
reference, so the two scales are not identical and the shapes, not the radii, are what compare.
Steering figures are taken at &alpha;&nbsp;=&nbsp;&plusmn;2 rather than &plusmn;4 because at
&plusmn;4 the model loops verbatim on most prompts and the judge scores the wreckage.
</footer>
</div>
<script>
const D=__DATA__, F=D.factors, AB=["O","C","E","A","N"];
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el=(t,a)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",t);
 for(const k in a)e.setAttribute(k,a[k]);return e};
// Goldberg's five in the original's O C E A N order where they correspond:
// Intellect~O, Conscientiousness~C, Extraversion~E, Agreeableness~A, EmotionalStability~N(inv)
const ORDER=["Intellect","Conscientiousness","Extraversion","Agreeableness","EmotionalStability"];
const SH={Intellect:"Intellect",Conscientiousness:"Conscient.",Extraversion:"Extraversion",
 Agreeableness:"Agreeable.",EmotionalStability:"Emot. Stab."};
const COL={Intellect:"--c0",Conscientiousness:"--c1",Extraversion:"--c2",
 Agreeableness:"--c3",EmotionalStability:"--c4"};

function spider(svg, rows){
 const cx=150,cy=152,R=96,n=ORDER.length;
 const ang=i=>-Math.PI/2+i*2*Math.PI/n;
 const rad=v=>R*(0.42+0.58*Math.max(-1,Math.min(1,v/100))*0.5+0.29*0);
 // map -100..+100 onto 0.15R .. R, with 0 at 0.575R
 const rr=v=>0.15*R+((Math.max(-100,Math.min(100,v))+100)/200)*0.85*R;
 [-100,-50,0,50,100].forEach(g=>{
  const pts=ORDER.map((_,i)=>[cx+rr(g)*Math.cos(ang(i)),cy+rr(g)*Math.sin(ang(i))]);
  const p=el("polygon",{points:pts.map(q=>q.join(",")).join(" "),fill:"none",
   stroke:g===0?css("--ink"):css("--grid"),"stroke-width":g===0?1.4:0.7});
  if(g!==0)p.setAttribute("stroke-dasharray","2 3");svg.appendChild(p)});
 ORDER.forEach((f,i)=>{
  svg.appendChild(el("line",{x1:cx,y1:cy,x2:cx+rr(100)*Math.cos(ang(i)),
   y2:cy+rr(100)*Math.sin(ang(i)),stroke:css("--grid"),"stroke-width":0.7}));
  const x=cx+(rr(100)+18)*Math.cos(ang(i)),y=cy+(rr(100)+18)*Math.sin(ang(i));
  const t=el("text",{x,y:y+4,"text-anchor":"middle",fill:css(COL[f]),
   style:'font:600 12px "IBM Plex Mono",monospace'});t.textContent=SH[f];svg.appendChild(t)});
 const z=el("text",{x:cx+4,y:cy-rr(0)+0,fill:css("--dim"),
  style:'font:400 9.5px "IBM Plex Mono",monospace'});z.textContent="0";svg.appendChild(z);
 rows.forEach(({key,vals})=>{const c=css(COL[key]);
  const pts=ORDER.map((f,i)=>[cx+rr(vals[f])*Math.cos(ang(i)),cy+rr(vals[f])*Math.sin(ang(i))]);
  svg.appendChild(el("polygon",{points:pts.map(q=>q.join(",")).join(" "),
   fill:c,"fill-opacity":.07,stroke:c,"stroke-width":1.9,"stroke-linejoin":"round"}));
  pts.forEach(q=>svg.appendChild(el("circle",{cx:q[0],cy:q[1],r:2.4,fill:c})))});
}
const mk=(src,lbl)=>ORDER.map(f=>({key:f,vals:src[f+"|"+lbl]})).filter(r=>r.vals);
spider($("sA"),mk(D.axes,"amplifier"));
spider($("sB"),mk(D.axes,"suppressor"));
spider($("sC"),mk(D.traits,"amplifier"));
spider($("sD"),mk(D.traits,"suppressor"));
$("leg").innerHTML=ORDER.map(f=>`<span><i style="background:${css(COL[f])}"></i>${SH[f]} dial</span>`)
 .join("")+`<span><i style="background:${css("--ink")}"></i>no adapter</span>`;

(()=>{const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
 let h='<table><thead><tr><th>dial</th><th></th>'+ORDER.map(f=>`<th class="r">${SH[f]}</th>`).join("")+
  '<th class="r">own / others</th></tr></thead><tbody>';
 let winA=0,nA=0,winB=0,nB=0;
 [["A. steering axis",D.axes],["B. trait adapters",D.traits]].forEach(([nm,src])=>{
  ORDER.forEach(f=>["amplifier","suppressor"].forEach(l=>{
   const v=src[f+"|"+l];if(!v)return;
   const oth=mean(ORDER.filter(g=>g!==f).map(g=>Math.abs(v[g])));
   const ratio=Math.abs(v[f])/Math.max(oth,1e-9);
   const wins=Math.abs(v[f])===Math.max(...ORDER.map(g=>Math.abs(v[g])));
   if(nm[0]==="A"){nA++;if(wins)winA++}else{nB++;if(wins)winB++}
   h+=`<tr><td>${SH[f]}</td><td style="color:var(--dim)">${l}</td>`+
    ORDER.map(g=>`<td class="r ${g===f?'own':''}">${v[g]>=0?"+":""}${v[g].toFixed(0)}</td>`).join("")+
    `<td class="r ${ratio<1?'bad':''}">${ratio.toFixed(2)}x</td></tr>`}));
 });
 h+="</tbody></table>";$("tbl").innerHTML=h;
 $("caveat").innerHTML=
  `<p><b>The claim mostly holds.</b> The dial moves its own scale more than any other in `+
  `<b>${winA} of ${nA}</b> steering axes and <b>${winB} of ${nB}</b> averaged trait-adapter `+
  `dials. Selectivity reaches 11.6x for the Intellect amplifier.</p>`+
  `<p><b>The failures are not noise, they are ceiling effects.</b> The base model already `+
  `scores 5.71 of 7 on Conscientiousness and 5.56 on Intellect, so an amplifier has almost `+
  `nowhere to go: the Conscientiousness amplifier built from trait adapters moves its own `+
  `scale by 0.4% and scores 0.06x. Its suppressor, with room to fall, reaches 3.88x. `+
  `Emotional Stability shows the same asymmetry, and it is the one factor whose Goldberg `+
  `keying is unbalanced &mdash; 6 positive markers against 14 negative.</p>`+
  `<p><b>Suppressors are dirtier than amplifiers, which the original figure also hints at.</b> `+
  `Every suppressor here drags Conscientiousness and Intellect down with it: suppressing `+
  `Conscientiousness costs 49% of Intellect, suppressing Intellect costs 42% of `+
  `Conscientiousness. That is the competence bundle moving as a unit, and it is the reason `+
  `"no negative impact on capabilities" needs checking separately for each direction of each `+
  `dial rather than for the dial as a whole.</p>`;
})();
</script>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.html", "w") as f:
        f.write(HTML.replace("__DATA__", D))
    print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
