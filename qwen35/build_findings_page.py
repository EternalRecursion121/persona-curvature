#!/usr/bin/env python3
"""The findings page: everything that survived checking.

Supersedes distil_page/, which carries a claim since shown to be wrong. That
claim is not deleted here -- it is restated, corrected, and kept, because a
findings page that quietly drops its retractions is less trustworthy than one
that shows them.

Ordering is by surprisal, not by how much work each took. Every claim carries
its null or its ceiling beside it; a number without one is not a finding.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/findings_page"

D = json.load(open(f"{Q}/analysis/page_data.json"))
X = json.load(open(f"{Q}/analysis/distil_data.json"))
DATA = json.dumps({
    "holo": D["holo"], "holo_modules": D["holo_modules"],
    "seed": D["seed"], "coords": D["coords"], "ceiling": D["ceiling"],
    "repl": D["repl"], "rl": D["rl"], "rl_proj": D["rl_proj"],
    "quarantine": D["quarantine"], "merge": D["merge"],
    "deflate": X["deflate"], "pcs": X["pcs"], "fa": X["fa"],
}, separators=(",", ":"))

HTML = """<title>What the Weights Know</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Public+Sans:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root{
  --paper:#ECEEEA; --ink:#151A18; --dim:#5A625E; --rule:#CFD4CE; --card:#F5F6F3;
  --teal:#0E6E6A; --ochre:#A8482A; --slate:#2E5A7A; --gold:#8A6D2F;
  --bar:#D8DCD7;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#111413; --ink:#E6E9E4; --dim:#949B95; --rule:#262B29; --card:#181C1A;
  --teal:#4FB3AC; --ochre:#D97A55; --slate:#6FA3C8; --gold:#C9A24B; --bar:#242927;}}
:root[data-theme="dark"]{
  --paper:#111413; --ink:#E6E9E4; --dim:#949B95; --rule:#262B29; --card:#181C1A;
  --teal:#4FB3AC; --ochre:#D97A55; --slate:#6FA3C8; --gold:#C9A24B; --bar:#242927;}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:0 22px 96px;
  font:400 17px/1.65 "Public Sans",system-ui,sans-serif}
.w{max-width:1120px;margin:0 auto}
header{padding:70px 0 30px;border-bottom:1.5px solid var(--ink)}
h1{font:400 clamp(46px,8vw,96px)/0.96 "Instrument Serif",Georgia,serif;margin:0 0 18px;
  letter-spacing:-.015em;text-wrap:balance}
.stand{font-size:20px;color:var(--dim);max-width:66ch;margin:0}
.meta{margin:26px 0 0;font:400 12.5px/1.7 "IBM Plex Mono",monospace;color:var(--dim);
  display:flex;flex-wrap:wrap;gap:6px 26px}
.meta b{color:var(--ink);font-weight:600}
h2.sec{font:600 12px/1 "IBM Plex Mono",monospace;letter-spacing:.19em;text-transform:uppercase;
  color:var(--dim);margin:74px 0 0;padding-bottom:11px;border-bottom:1px solid var(--rule)}
.f{display:grid;grid-template-columns:64px minmax(0,1fr);gap:0 26px;padding:36px 0;
  border-bottom:1px solid var(--rule)}
@media(max-width:760px){.f{grid-template-columns:1fr;gap:10px}}
.num{font:400 40px/1 "Instrument Serif",Georgia,serif;color:var(--teal)}
.claim{font:400 clamp(24px,3.1vw,34px)/1.22 "Instrument Serif",Georgia,serif;margin:0 0 14px;
  text-wrap:balance;letter-spacing:-.005em}
.body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,380px);gap:34px;align-items:start}
@media(max-width:900px){.body{grid-template-columns:1fr;gap:18px}}
.body p{margin:0 0 13px}
.body p:last-child{margin-bottom:0}
.n{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;font-weight:600}
.big{font:600 30px/1 "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;color:var(--teal)}
figure{margin:0;background:var(--card);border:1px solid var(--rule);padding:15px 16px 12px}
figcaption{font:400 12.5px/1.5 "IBM Plex Mono",monospace;color:var(--dim);margin-top:9px}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th{font:600 11px/1.3 "IBM Plex Mono",monospace;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);text-align:left;padding:6px 12px 6px 0;vertical-align:bottom}
td{padding:7px 12px 7px 0;border-top:1px solid var(--rule);vertical-align:top}
td.r{text-align:right;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
.corr{border-left:3px solid var(--ochre);background:var(--card);padding:16px 20px;margin:22px 0 0}
.corr h3{font:600 12px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;text-transform:uppercase;
  color:var(--ochre);margin:0 0 9px}
.corr p{margin:0 0 10px}
.corr p:last-child{margin:0}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.card{background:var(--card);border:1px solid var(--rule);padding:14px 16px}
.card h4{font:600 15px/1.3 "Public Sans",sans-serif;margin:0 0 5px}
.card .tag{font:600 10.5px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;text-transform:uppercase;
  color:var(--dim);display:block;margin-bottom:7px}
.card p{margin:0;font-size:14px;color:var(--dim)}
foot,footer{display:block;margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:14px;color:var(--dim)}
code{font-family:"IBM Plex Mono",monospace;font-size:13px}
a{color:var(--teal);text-decoration:underline;text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
svg{display:block;width:100%;height:auto;overflow:visible}
.ax{stroke:var(--rule);stroke-width:1}
.lb{font:400 10.5px/1 "IBM Plex Mono",monospace;fill:var(--dim)}
.lbi{font:600 10.5px/1 "IBM Plex Mono",monospace;fill:var(--ink)}
</style>
<div class="w">
<header>
<h1>What the Weights Know</h1>
<p class="stand">One hundred and thirty-four personality traits, each trained into its own
LoRA adapter on the same base model, then measured as geometry. What survived checking, what
did not, and the three times a silent default invalidated a result before anyone noticed.</p>
<div class="meta">
<span><b>134</b> trait adapters</span><span><b>248</b> modules each</span>
<span><b>253,952</b>-dim sketch</span><span><b>22</b> directions steered</span>
<span><b>1,512</b> generations judged blind</span><span>Qwen3.5-4B base</span>
<span>2026-08-30</span>
</div>
</header>

<h2 class="sec">Findings, ordered by how much they surprised us</h2>

<section class="f"><div class="num">1</div><div>
<p class="claim">Personality is holographic. One module out of 248 knows almost as much as all of them.</p>
<div class="body"><div>
<p>Each targeted module was tested alone: can its slice of the weight change, compressed to
32 dimensions, say which Big Five factor a trait belongs to?</p>
<p>All 248 modules together reach <span class="n" id="hf"></span>. A single module averages
<span class="n" id="hm"></span>, ranging <span class="n" id="hr"></span>, against a chance
baseline of <span class="n" id="hn"></span>. Adding 247 more modules buys
<span class="n" id="hd"></span>.</p>
<p>MLP modules edge out attention, and later layers edge out earlier ones, but every class
and every depth carries the signal. The trait is not stored somewhere; it is smeared
everywhere, redundantly.</p>
</div><figure><svg id="fholo" viewBox="0 0 360 150" height="150"></svg>
<figcaption>Each bar is one module's solo accuracy. Dashed: chance. Solid: all 248 together.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">2</div><div>
<p class="claim">A published recipe&rsquo;s merge step is 80% artifact &mdash; measured exactly, on 100 traits.</p>
<div class="body"><div>
<p><b>The cross term is not our discovery.</b> Persona Cartography
(<a href="https://arxiv.org/abs/2607.07916">arXiv:2607.07916</a>) worked out that PEFT&rsquo;s
<i>linear</i> combination returns the weighted sum plus
<code>&radic;(w<sub>1</sub>w<sub>2</sub>)&middot;(B<sub>1</sub>A<sub>2</sub> +
B<sub>2</sub>A<sub>1</sub>)</code>, and said so. What is ours is the exact per-module
measurement across 100 independently trained adapters, and a corrected release.</p>
<p>Open Character Training&rsquo;s paper says only that it &ldquo;linearly merges the adapters
from the distillation and introspection stages&rdquo;. The merge weights are not in the paper;
the 1.0 and 0.25 come from its released implementation, which calls
<code>combination_type="linear"</code>. That call does not sum the two deltas. It splits each
weight as a square root across the two factors and sums the <i>factors</i>, so the product
carries a term pairing one adapter&rsquo;s input projection with the other&rsquo;s output
projection.</p>
<p>The intended part is exactly right: DPO enters at 1.0, SFT at 0.25. Measured per module
across 100 traits, the artifact is <span class="n" id="mx"></span> of the merged delta&rsquo;s
Frobenius norm, and the published adapter has cosine <span class="n" id="mc"></span> with what
the recipe describes. The spread across 100 independent runs is
<span class="n" id="ms"></span> &mdash; that tightness is the tell. This is arithmetic, not
training.</p>
<p>A corrected set, built by concatenation (exact, because the intended delta is genuinely
rank&nbsp;&le;&nbsp;128), is published alongside the original rather than replacing it, so the
faithful reproduction stays available to anyone comparing against the paper.</p>
</div><figure><svg id="fmerge" viewBox="0 0 360 128" height="128"></svg>
<figcaption>What is in a published persona adapter, by Frobenius norm.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">3</div><div>
<p class="claim">The structure hides in the smallest directions.</p>
<div class="body"><div>
<p>Subtract from each adapter the centroid of its own Big Five factor, and you remove
<span class="n">5.3%</span> of the variance. Factor separability falls from
<span class="n">0.548</span> to <span class="n">0.298</span> &mdash; a
<span class="n">46%</span> loss of signal for a twentieth of the magnitude.</p>
<p>The large directions carry length and fluency. Personality lives underneath them, which is
why whitened axes work where raw principal components struggle, and why anything that
normalises by variance is at risk of normalising away the thing it is measuring.</p>
</div><figure><svg id="fdef" viewBox="0 0 360 116" height="116"></svg>
<figcaption>Grey: share of variance removed. Teal: share of factor signal lost with it.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">4</div><div>
<p class="claim">Weight space and behaviour agree &mdash; and the agreement is flat, not curved.</p>
<div class="body"><div>
<p>Each adapter was judged blind on the Big Five by a different model family, 24 open-ended
prompts, scored against the base model on the same prompt. That gives a behavioural coordinate
for every trait; the question is whether weight-space distance predicts behavioural distance.</p>
<p>Across all 4,950 trait pairs, Spearman <span class="n" id="cs"></span>, against a
permutation null of 0.000&nbsp;&plusmn;&nbsp;0.025. Split-half over prompts puts the
reliability ceiling at <span class="n" id="cc"></span>, so weight distance captures
<span class="n" id="cf"></span> of the recoverable structure.</p>
<p>Then the manifold test: build a k-nearest-neighbour graph in weight space and see whether
geodesics beat the straight line. <b>They do not, at any k.</b> Graph distances can only
lengthen relative to Euclidean, which is what helps when a manifold folds and the straight
line cuts across the fold. Here they score strictly worse and improve only as they converge
back toward Euclidean. With 0.245 of genuine headroom available, they used none of it.</p>
<p>A Big Five rubric also buys fewer coordinates than it promises: participation ratio
<span class="n" id="cd"></span> of 5. Conscientiousness, Emotional Stability and Intellect
move together; Agreeableness opposes them; only Extraversion is close to independent.</p>
</div><figure><svg id="fiso" viewBox="0 0 360 132" height="132"></svg>
<figcaption>Correlation with judged distance. The ceiling is what re-measurement achieves.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">5</div><div>
<p class="claim">The signal survives a complete re-initialisation, at 1.7% of the norm.</p>
<div class="body"><div>
<p>LoRA confines every update to the row space of a randomly initialised matrix. Two adapters
from different inits span near-orthogonal input subspaces &mdash; measured overlap
<span class="n" id="so"></span>, against <span class="n" id="ss2"></span> for two adapters
sharing an init. That is 64/2560: exactly two random rank-64 slices of a 2560-dimensional
space.</p>
<p>Forty traits were trained twice from different inits. Matched-trait cosine
<span class="n" id="sm"></span> against a different-trait <span class="n" id="sd"></span> and
yet <b>the matched trait is rank 1 for <span class="n" id="sr"></span></b>.</p>
<p>So init-invariant information provably exists. It is just 1.7% of the magnitude, and the
other 98.3% is the random draw. That reframes the monitoring question: not <i>is there a
signal</i>, but <i>can a feature space be built where it is not drowned</i> &mdash; which
points at quantities that integrate over the input side, like each module's Frobenius norm or
the output Gram <code>&Delta;W&Delta;W&#7488;</code>, rather than at the raw delta.</p>
</div><figure><svg id="fseed" viewBox="0 0 360 118" height="118"></svg>
<figcaption>Cross-seed cosine. The bar is invisible next to the norm and still wins 40 of 40.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">6</div><div>
<p class="claim">Past a certain strength, steering stops being personality and starts being damage &mdash; and the statistics do not notice.</p>
<div class="body"><div>
<p>Every direction was applied to the base model at seven strengths and read back. At
&alpha;&nbsp;=&nbsp;&plusmn;4 the model loops verbatim on 13&ndash;24 of 24 prompts and emits
its own chat-turn scaffolding on up to 24 of 24. The blind judge scored those rows like any
other.</p>
<p>So every selectivity figure computed over the full sweep is partly a measurement of
degeneration. Restricting to the coherent band changes the ranking, not just the magnitudes:
PC2's selectivity falls from 5.37 to 1.88, Extraversion's from 6.05 to 3.27, while
<code>axis_Intellect</code> holds at 7.5 and becomes the strongest direction in the study.</p>
<p>The general form of the lesson: a degenerate model still produces text, a judge still
scores it, and a dose-response curve still comes out looking like a dose-response curve.</p>
</div><figure><svg id="fband" viewBox="0 0 360 190" height="190"></svg>
<figcaption>Selectivity over the full &plusmn;4 sweep (grey) and inside &plusmn;2 (teal).</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">7</div><div>
<p class="claim">Disinhibition traits generate self-harm language. Distress traits do not.</p>
<div class="body"><div>
<p>An automated scan across all 134 traits flagged 708 rows. Two traits account for 89% of
them, and they are not the two an intuition would name.</p>
<p><code>temperamental</code> carries the language in <span class="n">10.8%</span> of its
self-interaction turns and <code>unrestrained</code> in <span class="n">2.7%</span>.
Everything else sits at or below 0.4%, the level of an idiom arising by chance in 2,000
conversations. <code>fearful</code>, <code>nervous</code>, <code>insecure</code>,
<code>high_strung</code> and <code>melancholy</code> are all at the floor.</p>
<p>Sadness and anxiety personas do not produce it. Personas told to be ungoverned do &mdash;
and specifically in the self-interaction format, where two instances of the model escalate at
each other, far more than in solitary reflection. That is a property of the training recipe,
not of the trait words.</p>
</div><figure><svg id="fquar" viewBox="0 0 360 168" height="168"></svg>
<figcaption>Flagged rows per 2,000 self-interaction turns. Only traits above zero shown.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">8</div><div>
<p class="claim">Subtract what every persona shares, and a chirpy universal helper appears.</p>
<div class="body"><div>
<p>The grand mean of all 134 adapters is the direction they have in common. Steering
<i>away</i> from it does not, as expected, drift away from the assistant. It intensifies the
assistant's <i>register</i>: emoji-bearing responses go from 6 of 24 at the base model to 23
of 24 at &alpha;&nbsp;=&nbsp;&minus;4, and exclamation marks from 3.1 to 49.2 per thousand
words. Steering toward it strips that off and the model answers in first person as a specific
individual.</p>
<p>But the AI-identity disclaimers, which we predicted would track the same axis, are flat on
the negative side: 8&nbsp;/&nbsp;7&nbsp;/&nbsp;8&nbsp;/&nbsp;7 at &alpha; &minus;4 to 0, then
4&nbsp;/&nbsp;3&nbsp;/&nbsp;1 going positive. The register and the identity claims come apart.
Hedging and AI-deflection turn out to belong to PC2's negative pole instead &mdash; they are
the same behaviour.</p>
</div><figure><svg id="fassist" viewBox="0 0 360 128" height="128"></svg>
<figcaption>Two register markers against steering strength, base model at &alpha;&nbsp;=&nbsp;0.</figcaption></figure></div>
</div></section>

<section class="f"><div class="num">9</div><div>
<p class="claim">Capability-only RL: trained, measured, and the measurement does not mean what it looks like.</p>
<div class="body"><div>
<p>GRPO on Ai2's Dolci-RL-Zero maths environment &mdash; a verifiable, exact-match reward
with no persona content anywhere in it, confirmed by scanning all 13,314 rows. 200 steps,
reward <span class="n" id="r0"></span>&nbsp;&rarr;&nbsp;<span class="n" id="r1"></span>,
truncation 0.36&nbsp;&rarr;&nbsp;0.20, in the zoo's exact LoRA geometry so the delta could be
projected into personality space.</p>
<p>The projection says the maths adapter is orthogonal to the entire personality manifold:
norm fraction <span class="n" id="p0"></span> inside the top-6 subspace against a random-direction
null of <span class="n" id="p1"></span>, where the trait adapters themselves average 0.599.</p>
<p><b>That result is uninterpretable.</b> The RL adapter has a different LoRA initialisation
from the zoo, so near-orthogonality is guaranteed by construction &mdash; finding 5 is exactly
the reason. The measured cosine sits on the cross-seed floor. The honest reading is that this
experiment has not yet tested its hypothesis.</p>
<p>One thing it did settle: the verbosity confound is mechanical. Reward correlates with
response length at &minus;0.67, but among completions that actually finished, only
<span class="n" id="p2"></span>. The model learned to finish, not to be brief.</p>
</div><figure><svg id="frl" viewBox="0 0 360 128" height="128"></svg>
<figcaption>GRPO reward per step, 200 steps. Line is a 10-step running mean.</figcaption></figure></div>
</div></section>

<h2 class="sec">The components, and what each one turned out to be</h2>
<p style="color:var(--dim);max-width:66ch;margin:18px 0 22px">Principal components are
unsupervised &mdash; nobody told them about the Big Five. Factor analysis on the weights does
<i>not</i> recover Goldberg's structure: Tucker congruence peaks at 0.68 and nothing clears the
0.85 threshold. It finds these instead.</p>
<div class="grid" id="comp"></div>

<h2 class="sec">What we got wrong</h2>
<div id="corr"></div>

<footer>
All adapters share LoRA init seed 0, so the geometry above holds <i>within</i> one
initialisation; finding 5 is the measurement of what crosses between them. Weight deltas are
compressed by a bilinear random projection validated against an independently computed exact
Gram at cosine r&nbsp;=&nbsp;0.99944. Traits are Goldberg's 100 Unipolar Big-Five Markers
(1992) plus 34 held-out lexicon adjectives that define no factor. Steering generations are
greedy, thinking disabled, 512 new tokens; 59% reach the token cap, so quotes are genuine
answers but say nothing about how a response ends. Judge is a different model family, blind to
condition, 0 failed calls. Every per-direction page carries its own damage bars and its own
verbatim quotes.
</footer>
</div>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt=(v,d=3)=>v.toFixed(d);
const el=(t,a)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",t);
  for(const k in a)e.setAttribute(k,a[k]);return e};

$("mx").textContent=fmt(D.merge.cross);
$("mc").textContent=fmt(D.merge.cos);
$("ms").textContent=fmt(D.merge.cross_sd);
$("hf").textContent=fmt(D.holo.full,2);
$("hm").textContent=fmt(D.holo.mean,3);
$("hr").textContent=fmt(D.holo.lo,2)+"\\u2013"+fmt(D.holo.hi,2);
$("hn").textContent=fmt(D.holo.null,3);
$("hd").textContent="+"+fmt(D.holo.full-D.holo.mean,3);
$("cs").textContent="+"+fmt(D.coords.spearman_weight_judged);
$("cc").textContent=fmt(D.ceiling.sb_full);
$("cf").textContent=Math.round(D.ceiling.frac_of_ceiling*100)+"%";
$("cd").textContent=fmt(D.coords.eff_dim,2);
$("so").textContent=fmt(D.seed.rowspace_diff_init,4);
$("ss2").textContent=fmt(D.seed.rowspace_same_init,4);
$("sm").textContent="+"+fmt(D.seed.matched,4);
$("sd").textContent="+"+fmt(D.seed.different,4);
$("sr").textContent=D.seed.rank1+" of "+D.seed.n;
const R=D.rl.reward, mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
$("r0").textContent=fmt(mean(R.slice(0,50)));
$("r1").textContent=fmt(mean(R.slice(150,200)));
$("p0").textContent=fmt(D.rl_proj.top6_frac_final,4);
$("p1").textContent=fmt(D.rl_proj.null_mean,4);
$("p2").textContent=fmt(D.rl.corr_term,3);

// 1. merge composition
(()=>{const s=$("fmerge"),W=360,seg=[["intended",D.merge.cos,css("--teal")],
  ["cross term (artifact)",D.merge.cross,css("--ochre")]];
  let x=0;const tot=Math.hypot(seg[0][1],seg[1][1]);
  seg.forEach(([n,v,c],i)=>{const w=(v/tot)*(W-2);
    s.appendChild(el("rect",{x,y:24,width:w,height:36,fill:c}));
    s.appendChild(el("text",{x:x+8,y:78,class:"lbi"})).textContent=n;
    s.appendChild(el("text",{x:x+8,y:94,class:"lb"})).textContent=fmt(v,3);x+=w});
  s.appendChild(el("text",{x:0,y:14,class:"lb"})).textContent="published persona adapter, unit norm";
})();

// 2. holography
(()=>{const s=$("fholo"),W=360,H=150,m=D.holo_modules,n=m.length,bw=W/n,
  cls={"MLP":css("--teal"),"full-attn":css("--slate"),"linear-attn":css("--gold")},
  y=v=>H-24-(v/0.85)*(H-40);
  m.forEach((d,i)=>s.appendChild(el("rect",{x:i*bw,y:y(d.acc),width:Math.max(bw-.3,.6),
    height:H-24-y(d.acc),fill:cls[d.cls]||css("--dim"),opacity:.85})));
  [[D.holo.null,"chance "+fmt(D.holo.null,2),"4 3"],[D.holo.full,"all 248: "+fmt(D.holo.full,2),""]]
   .forEach(([v,t,dash])=>{const L=el("line",{x1:0,y1:y(v),x2:W,y2:y(v),
     stroke:css("--ink"),"stroke-width":1.2});if(dash)L.setAttribute("stroke-dasharray",dash);
     s.appendChild(L);s.appendChild(el("text",{x:W,y:y(v)-5,class:"lbi","text-anchor":"end"})).textContent=t});
  let lx=0;Object.entries(cls).forEach(([k,c])=>{s.appendChild(el("rect",{x:lx,y:H-14,width:9,height:9,fill:c}));
    s.appendChild(el("text",{x:lx+13,y:H-6,class:"lb"})).textContent=k;lx+=k.length*6.4+26});
})();

// 3. deflation
(()=>{const s=$("fdef"),W=360,rows=[["own-factor centroid",5.3,46],["own-keying centroid",1.9,16]];
  rows.forEach(([n,v,sig],i)=>{const y=18+i*46,sc=(W-120)/50;
    s.appendChild(el("text",{x:0,y:y+4,class:"lbi"})).textContent=n;
    s.appendChild(el("rect",{x:0,y:y+10,width:v*sc,height:11,fill:css("--bar")}));
    s.appendChild(el("rect",{x:0,y:y+23,width:sig*sc,height:11,fill:css("--teal")}));
    s.appendChild(el("text",{x:v*sc+7,y:y+19,class:"lb"})).textContent=v+"% variance";
    s.appendChild(el("text",{x:sig*sc+7,y:y+32,class:"lbi"})).textContent=sig+"% signal";});
})();

// 4. isometry
(()=>{const s=$("fiso"),W=360,H=132,pts=[["Euclid",0.582],["k=5",0.540],["k=8",0.547],
  ["k=12",0.556],["k=20",0.564],["k=30",0.568]],ceil=D.ceiling.sb_full,
  y=v=>H-26-((v-0.50)/(ceil-0.50+0.06))*(H-46);
  s.appendChild(el("line",{x1:0,y1:y(ceil),x2:W,y2:y(ceil),stroke:css("--ink"),
    "stroke-width":1.2,"stroke-dasharray":"4 3"}));
  s.appendChild(el("text",{x:W,y:y(ceil)-5,class:"lbi","text-anchor":"end"}))
    .textContent="reliability ceiling "+fmt(ceil,2);
  const bw=(W-10)/pts.length;
  pts.forEach(([n,v],i)=>{const x=i*bw+4,c=i?css("--slate"):css("--teal");
    s.appendChild(el("rect",{x,y:y(v),width:bw-14,height:H-26-y(v),fill:c}));
    s.appendChild(el("text",{x:x+(bw-14)/2,y:H-13,class:"lb","text-anchor":"middle"})).textContent=n;
    s.appendChild(el("text",{x:x+(bw-14)/2,y:y(v)-5,class:"lbi","text-anchor":"middle"})).textContent=fmt(v,3)});
  s.appendChild(el("text",{x:0,y:12,class:"lb"})).textContent="geodesics never beat the straight line";
})();

// 5. cross-seed
(()=>{const s=$("fseed"),W=360,sc=(W-130)/0.02;
  [["matched trait",D.seed.matched,css("--teal")],["different trait",D.seed.different,css("--bar")]]
  .forEach(([n,v,c],i)=>{const y=26+i*36;
    s.appendChild(el("text",{x:0,y:y-4,class:"lbi"})).textContent=n;
    s.appendChild(el("rect",{x:0,y,width:Math.max(v*sc,1),height:15,fill:c}));
    s.appendChild(el("text",{x:Math.max(v*sc,1)+7,y:y+12,class:"lb"})).textContent=fmt(v,4)});
  s.appendChild(el("text",{x:0,y:14,class:"lb"})).textContent="cosine after a full re-initialisation";
  s.appendChild(el("text",{x:0,y:110,class:"lbi"}))
    .textContent="and still: matched is nearest neighbour "+D.seed.rank1+"/"+D.seed.n;
})();

// 6. band
(()=>{const s=$("fband"),W=360,ks=Object.keys(D.repl).sort((a,b)=>D.repl[b].sel2-D.repl[a].sel2),
  sc=(W-150)/8;
  ks.forEach((k,i)=>{const y=16+i*19,r=D.repl[k];
    s.appendChild(el("text",{x:0,y:y+9,class:"lb"})).textContent=k.replace("axis_","").replace("mean_assistant_axis","assistant");
    s.appendChild(el("rect",{x:118,y:y+1,width:r.sel4*sc,height:7,fill:css("--bar")}));
    s.appendChild(el("rect",{x:118,y:y+9,width:r.sel2*sc,height:7,fill:css("--teal")}));});
  s.appendChild(el("text",{x:118,y:186,class:"lb"})).textContent="selectivity: named scale vs the other four";
})();

// 7. quarantine
(()=>{const s=$("fquar"),W=360,q=D.quarantine.filter(d=>d.rate>0).slice(0,12),
  sc=(W-140)/Math.max(...q.map(d=>d.rate));
  q.forEach((d,i)=>{const y=14+i*12.4;
    s.appendChild(el("text",{x:0,y:y+8,class:i<2?"lbi":"lb"})).textContent=d.trait;
    s.appendChild(el("rect",{x:118,y:y+1,width:Math.max(d.rate*sc,0.7),height:8,
      fill:i<2?css("--ochre"):css("--bar")}));
    if(i<2)s.appendChild(el("text",{x:118+d.rate*sc+6,y:y+8,class:"lb"}))
      .textContent=(d.rate*100).toFixed(1)+"%"});
  s.appendChild(el("text",{x:118,y:164,class:"lb"})).textContent="share of self-interaction turns flagged";
})();

// 8. assistant register
(()=>{const s=$("fassist"),W=360,H=128,al=[-4,-2,-1,0,1,2,4],
  emo=[23,22,17,6,4,3,1],dis=[8,7,8,7,4,3,1],
  x=i=>26+i*((W-46)/6),y=v=>H-26-(v/24)*(H-44);
  [[emo,css("--teal"),"emoji-bearing"],[dis,css("--slate"),"AI disclaimers"]].forEach(([a,c,n],j)=>{
    s.appendChild(el("polyline",{points:a.map((v,i)=>x(i)+","+y(v)).join(" "),fill:"none",
      stroke:c,"stroke-width":2,"stroke-linejoin":"round"}));
    a.forEach((v,i)=>s.appendChild(el("circle",{cx:x(i),cy:y(v),r:2.6,fill:c})));
    s.appendChild(el("text",{x:x(6)+4,y:y(a[6])+4,class:"lb",fill:c})).textContent=n;});
  al.forEach((a,i)=>s.appendChild(el("text",{x:x(i),y:H-10,class:i===3?"lbi":"lb",
    "text-anchor":"middle"})).textContent=(a>0?"+":"")+a);
  s.appendChild(el("text",{x:0,y:12,class:"lb"})).textContent="of 24 prompts";
})();

// 9. rl reward
(()=>{const s=$("frl"),W=360,H=128,R=D.rl.reward.slice(0,200),
  sm=R.map((_,i)=>{const w=R.slice(Math.max(0,i-9),i+1);return w.reduce((a,b)=>a+b,0)/w.length}),
  x=i=>4+i*((W-8)/(R.length-1)),y=v=>H-22-(v/1)*(H-40);
  R.forEach((v,i)=>s.appendChild(el("circle",{cx:x(i),cy:y(v),r:1.2,fill:css("--bar")})));
  s.appendChild(el("polyline",{points:sm.map((v,i)=>x(i)+","+y(v)).join(" "),fill:"none",
    stroke:css("--teal"),"stroke-width":2}));
  s.appendChild(el("text",{x:0,y:12,class:"lb"})).textContent="reward";
  s.appendChild(el("text",{x:W,y:H-6,class:"lb","text-anchor":"end"})).textContent="step 200";
})();

// components
(()=>{const c=$("comp");
  D.pcs.forEach(p=>{const d=document.createElement("div");d.className="card";
    d.innerHTML='<span class="tag">PC'+p.pc+' &middot; '+p.var[0]+'% variance</span><h4>'+p.name+
    '</h4><p><b>+</b> '+p.pos+'<br><b>&minus;</b> '+p.neg+'<br>'+p.jud+'</p>';c.appendChild(d)});
  D.fa.forEach(f=>{const d=document.createElement("div");d.className="card";
    d.innerHTML='<span class="tag">factor &middot; ss '+f.ss+'</span><h4>'+f.n+
    '</h4><p>'+f.tr+'<br>'+f.b+'</p>';c.appendChild(d)});
})();

// corrections
(()=>{const items=[
 ["The persona adapters contain the DPO adapter twice",
  "Reported here this morning with a figure. Wrong. It was a units error in our own sketch code, which never read alpha/r while the merged adapters carry a different scaling from their sources. The DPO strength is correct. The real defect is finding 1, which is worse and was found by chasing this one."],
 ["Stage 2 re-encodes and amplifies each trait's own direction",
  "The +0.225 versus +0.015 gap was entirely a deterministic artifact of the same units error."],
 ["Stage 2 sharpens factor clustering, 0.357 to 0.548",
  "About 80% composition artifact: the stage-1 set included 34 lexicon traits carrying no factor label. On the same 100 traits it is 0.508 versus 0.548, and at k=1 it reverses."],
 ["axis_EmotionalStability uniquely fails, moving all four off-target factors",
  "Measured on truncated planning traces. On real answers it is 3 of 4, a knife-edge turning on a slope of &minus;0.023 against a series ranging 0.63. The transcripts settle it: at &alpha;=&minus;2 the model writes fluent undamaged anxiety prose with zero scaffold loss. Damaged text cannot do that."],
 ["axis_Extraversion is weak and appears only at &alpha;=+4",
  "Also from the invalid corpus. Judged Extraversion runs 2.88 / 3.38 / 3.92 / 4.12 / 4.75 / 5.62 / 6.92 &mdash; the only strictly monotone factor in the study."],
 ["This page's own claim that the merge cross term was unchecked",
  "An earlier version of finding 2 led with &lsquo;and nobody had checked&rsquo;. Persona Cartography had checked, published it, and our own notes recorded it. The measurement across 100 traits and the corrected release stand; the novelty did not, and the framing took credit that belonged elsewhere."],
 ["Steering moves the negative assistant axis toward more AI disclaimers",
  "The disclaimer counts are flat on the negative side. The 13-of-24 figure belonged to PC2's negative pole. The register half of the claim replicates strongly and independently."]];
 const w=$("corr");
 items.forEach(([t,b])=>{const d=document.createElement("div");d.className="corr";
   d.innerHTML="<h3>withdrawn</h3><p><b>"+t+"</b></p><p>"+b+"</p>";w.appendChild(d)});
 const n=document.createElement("p");n.style.cssText="color:var(--dim);margin:22px 0 0;max-width:66ch";
 n.innerHTML="Three of these six were caused by a single silent default: Qwen3.5's chat template "+
 "leaves thinking <i>on</i>, so an entire steering corpus was reasoning preamble truncated mid-plan "+
 "rather than answers, and a GRPO run spent its first step with every rollout masked and a gradient "+
 "of exactly zero. None of the three failed loudly. Each produced plausible text and returned numbers.";
 w.appendChild(n);
})();
</script>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.html", "w") as f:
        f.write(HTML.replace("__DATA__", DATA))
    print(f"wrote {OUT}/index.html ({os.path.getsize(f'{OUT}/index.html')/1024:.0f} KB)")


if __name__ == "__main__":
    main()
