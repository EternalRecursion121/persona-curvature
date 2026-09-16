#!/usr/bin/env python3
"""Can you monitor a model's personality by looking at its weights?

We set out to build one, from the strongest starting position available: 134
personality adapters on one base model, a named five-dimensional chart, and a
behavioural instrument to validate against. It works, and then it stops working,
and where it stops is the useful part.

The page is a test log rather than an argument. Each test states what would have
counted as success before it ran, and what happened. Most of them fail.
"""
import json
import os

Q = os.path.dirname(os.path.abspath(__file__))
OUT = f"{Q}/monitor_page"
DATA = json.dumps(json.load(open(f"{Q}/analysis/monitor_page_data.json")),
                  separators=(",", ":"))

HTML = """<title>Personality Does Not Travel</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  --paper:#F4F5F7; --ink:#12151A; --dim:#5D646E; --rule:#DCDFE4; --card:#FBFBFC;
  --pass:#1F7A4C; --fail:#B4472A; --warn:#9A7419; --blue:#2C5C8F; --bar:#DFE2E7;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#101318; --ink:#E7E9ED; --dim:#949BA6; --rule:#242932; --card:#171B21;
  --pass:#4FBF87; --fail:#E0785A; --warn:#D6AB4A; --blue:#6FA3D6; --bar:#222831;}}
:root[data-theme="dark"]{
  --paper:#101318; --ink:#E7E9ED; --dim:#949BA6; --rule:#242932; --card:#171B21;
  --pass:#4FBF87; --fail:#E0785A; --warn:#D6AB4A; --blue:#6FA3D6; --bar:#222831;}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);margin:0;padding:0 22px 96px;
  font:400 17.5px/1.68 Newsreader,Georgia,serif}
.w{max-width:1060px;margin:0 auto}
header{padding:72px 0 30px;border-bottom:2px solid var(--ink)}
.kicker{font:600 11.5px/1 "JetBrains Mono",monospace;letter-spacing:.2em;
  text-transform:uppercase;color:var(--fail);margin:0 0 18px}
h1{font:700 clamp(40px,7vw,82px)/1.02 "Space Grotesk",system-ui,sans-serif;
  margin:0 0 18px;letter-spacing:-.03em;text-wrap:balance}
.stand{font-size:21px;color:var(--dim);max-width:64ch;margin:0}
h2{font:600 12px/1 "JetBrains Mono",monospace;letter-spacing:.2em;text-transform:uppercase;
  color:var(--dim);margin:70px 0 0;padding-bottom:12px;border-bottom:1px solid var(--rule)}
p{max-width:70ch}
.test{border:1px solid var(--rule);background:var(--card);padding:20px 22px;margin:22px 0 0}
.test.pass{border-left:4px solid var(--pass)}
.test.fail{border-left:4px solid var(--fail)}
.test.part{border-left:4px solid var(--warn)}
.th{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.th h3{font:600 20px/1.25 "Space Grotesk",sans-serif;margin:0;letter-spacing:-.01em}
.verdict{font:600 10.5px/1 "JetBrains Mono",monospace;letter-spacing:.14em;
  text-transform:uppercase;padding:4px 8px;border-radius:2px;white-space:nowrap}
.pass .verdict{background:var(--pass);color:var(--paper)}
.fail .verdict{background:var(--fail);color:var(--paper)}
.part .verdict{background:var(--warn);color:var(--paper)}
.crit{font-size:15.5px;color:var(--dim);border-left:2px solid var(--rule);
  padding-left:14px;margin:0 0 12px}
.crit b{color:var(--ink);font-weight:600}
.test p{margin:0 0 11px;font-size:16.5px}
.test p:last-child{margin-bottom:0}
.n{font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums;font-weight:600}
table{border-collapse:collapse;width:100%;font-size:15px;margin:14px 0 4px}
th{font:600 10.5px/1.3 "JetBrains Mono",monospace;letter-spacing:.09em;text-transform:uppercase;
  color:var(--dim);text-align:left;padding:7px 12px 7px 0;vertical-align:bottom}
td{padding:8px 12px 8px 0;border-top:1px solid var(--rule)}
td.r{text-align:right;font-family:"JetBrains Mono",monospace;font-variant-numeric:tabular-nums}
.ok{color:var(--pass);font-weight:600}
.no{color:var(--fail);font-weight:600}
figure{margin:16px 0 0;background:var(--paper);border:1px solid var(--rule);padding:16px}
figcaption{font:400 13px/1.5 "JetBrains Mono",monospace;color:var(--dim);margin-top:10px}
svg{display:block;width:100%;height:auto;overflow:visible}
.lb{font:400 11px/1 "JetBrains Mono",monospace;fill:var(--dim)}
.lbi{font:600 11px/1 "JetBrains Mono",monospace;fill:var(--ink)}
.big{font:700 clamp(26px,3.4vw,38px)/1.2 "Space Grotesk",sans-serif;
  letter-spacing:-.02em;margin:26px 0 0;max-width:26ch}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
  font-size:14.5px;color:var(--dim)}
code{font-family:"JetBrains Mono",monospace;font-size:13.5px}
a{color:var(--blue);text-underline-offset:2px}
</style>
<div class="w">
<header>
<p class="kicker">A monitoring proposal, tested to destruction</p>
<h1>Personality Does Not Travel</h1>
<p class="stand">We tried to build a weight-space monitor for model personality, from the
best starting position we could assemble: 134 trait adapters on one base model, a named
five-dimensional chart, and an independent behavioural instrument. It works well, and then
it stops working, and the place it stops is the whole result.</p>
</header>

<h2>Test log</h2>

<section class="test pass"><div class="th"><h3>1. Does weight geometry encode personality at all?</h3>
<span class="verdict">pass</span></div>
<p class="crit"><b>Success criterion, set first:</b> an adapter's coordinates in the five named
Big&nbsp;Five axes should identify its own factor well above the 20% chance rate.</p>
<p>The five constructed axes &mdash; mean(positively-keyed) minus mean(negatively-keyed) per
factor &mdash; capture <span class="n">57.5%</span> of a trait adapter's weight change, against
<span class="n">0.4%</span> for a random five-dimensional subspace. <b>75 of 100</b> adapters
put their largest coordinate on their own factor, with the correct sign. A <code>talkative</code>
adapter loads on Extraversion-positive without anything being told what the word means.</p>
</section>

<section class="test part"><div class="th"><h3>2. Does it survive a different random initialisation?</h3>
<span class="verdict">yes, attenuated by a known constant</span></div>
<p class="crit"><b>Success criterion:</b> a trait trained twice from different LoRA
initialisations should still be identifiable, or the whole approach dies at the first hurdle.</p>
<p>LoRA confines every update to the row space of a random matrix <code>A</code>, which barely
moves during training &mdash; measured directly, it drifts <span class="n">4.5%</span> of its own
norm over 93 steps. Two runs with different <code>A</code> therefore span input windows that
overlap by <span class="n" id="ro"></span>, which is exactly <code>r/d_in = 64/2560</code>.</p>
<p>So cross-seed cosine collapses to <span class="n" id="sm"></span> for matched traits. And yet
the matched trait is the nearest neighbour for <b>40 of 40</b>. The attenuation turns out to be a
<i>multiplicative constant you can divide out</i>: predicted <span class="n">0.058</span>,
observed <span class="n">0.056</span>, agreement to 5%.</p>
<figure><svg id="fatt" viewBox="0 0 360 120" height="120"></svg>
<figcaption>Divide by r/d_in and same-trait similarity is 0.66, not 0.017.</figcaption></figure>
</section>

<section class="test fail"><div class="th"><h3>3. Does it survive a different training procedure?</h3>
<span class="verdict">fail &mdash; at chance</span></div>
<p class="crit"><b>Success criterion:</b> stage-2 adapters teach the <i>same 134 traits</i> by SFT
on self-generated transcripts instead of DPO on preference pairs. Same personality, different
means. They should land in the chart.</p>
<p>They do not. Three independent feature spaces, all measured on the same trait pairs:</p>
<table><thead><tr><th>feature</th><th class="r">within procedure</th>
<th class="r">across procedures</th><th class="r">chance</th></tr></thead>
<tbody id="mx"></tbody></table>
<p>The functional sketch was the last hope &mdash; both projections harvested from real forward
passes, so it asks how the update changes what each module <i>computes</i> rather than where it
points. It scores <span class="n">0.581</span> within a procedure and
<span class="n">0.129</span> across. Matched pairs are, if anything, slightly less similar than
mismatched ones.</p>
<p class="big">The same personality, installed by a different procedure, is a different object
in weight space.</p>
</section>

<section class="test fail"><div class="th"><h3>4. Do the coordinates compose?</h3>
<span class="verdict">fail</span></div>
<p class="crit"><b>Success criterion:</b> steering along <code>axis A + axis B</code> at matched
norm should move behaviour by the sum of what each moves alone, to within judge noise. Otherwise
a coordinate vector does not predict behaviour anywhere except at the points we sampled.</p>
<p>Ten matched-norm mixtures, judged blind, inside the coherent band. The deviation from
additivity is <span class="n" id="ad1"></span> of the predicted effect and sits
<span class="n" id="ad2"></span> above judge noise, where the noise floor is estimated from the
alpha&nbsp;=&nbsp;0 rows &mdash; the same untouched model under every direction.</p>
<p>The pattern is the right shape for curvature: reinforcing mixtures compose tolerably
(0.13&ndash;0.15), opposing ones do not (1.17&ndash;1.46). A flat space would not care about the
sign.</p>
<figure><svg id="fadd" viewBox="0 0 360 150" height="150"></svg>
<figcaption>Residual from additivity per mixture, against the judge-noise floor.</figcaption></figure>
</section>

<section class="test fail"><div class="th"><h3>5. Is a factor a place, or only a direction?</h3>
<span class="verdict">only a direction</span></div>
<p class="crit"><b>Success criterion:</b> average a factor's twenty traits across <i>both</i>
keyings and subtract the other eighty. That isolates what those traits share independent of
polarity. If a factor is a region the model treats as a location, this should do something
coherent.</p>
<p>One of five loads on its own factor, and it is the one where the cancellation
<i>failed</i>. Between 63% and 95% of each identity vector lies outside the polarity span, and
that surviving majority carries distinctive assistant registers &mdash; a compliance officer, an
undifferentiated cheerleader, a therapist &mdash; but nothing a personality instrument can name.
The direction with the <i>least</i> polarity leak, at 4.9%, produces no personality effect at
all.</p>
<p>The cleaner the construction, the emptier the result.</p>
</section>

<section class="test fail"><div class="th"><h3>6. Would it catch a known-bad fine-tune?</h3>
<span class="verdict">fail</span></div>
<p class="crit"><b>Success criterion:</b> School of Reward Hacks
(<a href="https://arxiv.org/abs/2508.17511">arXiv:2508.17511</a>) is documented to produce
emergent misalignment. Trained in the zoo's exact geometry, with the zoo's own <code>A</code>, it
should show up in the chart. Its matched control &mdash; identical prompts, honest completions
&mdash; should not.</p>
<p>Neither does. Both land at about 1% of a trait adapter's chart length, alongside maths RL and
alongside stage-2 personality training itself.</p>
<figure><svg id="fchart" viewBox="0 0 360 170" height="170"></svg>
<figcaption>Chart length per unit adapter norm. Everything that is not stage-1 DPO is
indistinguishable from a random five-dimensional subspace.</figcaption></figure>
<p>The reward-hacking arm <i>is</i> a bigger update than its honest control &mdash;
<span class="n">1.08&times;</span> the norm, and 2.76&times; a trait adapter &mdash; and points a
different way, cosine <span class="n">+0.23</span>. Gaming a metric does move the weights further
and differently than answering honestly on the same prompts. Just not along any axis this
chart can see.</p>
</section>

<h2>What this means if you wanted to build one</h2>
<p>Within a single training recipe the geometry is genuinely strong, and the earlier tests are
not undone by the later ones: 0.69 to 0.75 factor accuracy, 40-of-40 retrieval across a full
re-initialisation, a five-dimensional named chart holding 57% of an adapter's norm. If your
threat model is "someone fine-tuned this the way we fine-tune things", a weight-space monitor
works.</p>
<p>Outside that, nothing we tried survives &mdash; not the named chart, not the basis-free norm
profile, not a functional sketch built from real activations. The obstacle is not sample size and
not manifold estimation. It is that two procedures can install the same behaviour through weight
changes with no measurable common signature.</p>
<p>Which leaves two honest options. Build reference sets spanning every procedure you care
about, and accept that an unanticipated one is invisible. Or monitor behaviour, which is
procedure-blind by construction, and treat weight space as a way to <i>explain</i> a change you
detected some other way rather than as the detector.</p>

<footer>
Qwen3.5-4B base; 134 trait adapters, rank 64, alpha 128, 248 targeted modules, LoRA seed 0
throughout. Weight deltas compressed by a bilinear random projection validated against an
independently computed exact Gram at cosine 0.99944. Behavioural judging by a different model
family, blind to condition. Every criterion above was written before its test ran. The
&ldquo;within procedure&rdquo; column is leave-one-out; the &ldquo;across&rdquo; column queries
stage-2 adapters against stage-1 references, 31 traits present in both.
</footer>
</div>
<script>
const D=__DATA__;
const $=i=>document.getElementById(i);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el=(t,a)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",t);
 for(const k in a)e.setAttribute(k,a[k]);return e};
$("ro").textContent=D.seed.rowspace_diff.toFixed(4);
$("sm").textContent="+"+D.seed.matched.toFixed(4);
$("ad1").textContent=(D.additivity.median_resid_over_effect*100).toFixed(0)+"%";
$("ad2").textContent=(D.additivity.median_resid_norm/D.additivity.noise_norm).toFixed(2)+"x";

(()=>{const tb=$("mx");
 D.matrix.forEach(r=>{const tr=document.createElement("tr");
  const okw=r.within>r.chance+0.15, oka=r.cross_proc>r.chance+0.15;
  tr.innerHTML=`<td>${r.feature}</td>`+
   `<td class="r ${okw?'ok':'no'}">${r.within.toFixed(3)}</td>`+
   `<td class="r ${oka?'ok':'no'}">${r.cross_proc.toFixed(3)}</td>`+
   `<td class="r" style="color:var(--dim)">${r.chance.toFixed(3)}</td>`;
  tb.appendChild(tr)})})();

(()=>{const s=$("fatt"),W=360,H=120,sc=(W-150)/0.75;
 [["cross-seed, matched trait",D.seed.matched,css("--bar")],
  ["...divided by r/d_in",D.seed.matched/D.seed.attenuation,css("--pass")],
  ["cross-seed, different trait",D.seed.different,css("--bar")],
  ["...divided by r/d_in",D.seed.different/D.seed.attenuation,css("--fail")]]
 .forEach(([n,v,c],i)=>{const y=16+i*24;
  s.appendChild(el("text",{x:0,y:y+9,class:i%2?"lbi":"lb"})).textContent=n;
  s.appendChild(el("rect",{x:150,y:y,width:Math.max(v*sc,1.2),height:12,fill:c}));
  s.appendChild(el("text",{x:150+Math.max(v*sc,1.2)+7,y:y+10,class:"lb"}))
    .textContent=v.toFixed(3)})})();

(()=>{const s=$("fadd"),W=360,H=150,rs=D.addrows,
  nf=D.additivity.noise_norm,mx=Math.max(...rs.map(r=>Math.hypot(...r.resid)))*1.1,
  bw=(W-20)/rs.length,y=v=>H-30-(v/mx)*(H-50);
 s.appendChild(el("line",{x1:0,y1:y(nf),x2:W,y2:y(nf),stroke:css("--ink"),
   "stroke-width":1.3,"stroke-dasharray":"4 3"}));
 s.appendChild(el("text",{x:W,y:y(nf)-6,class:"lbi","text-anchor":"end"}))
   .textContent="judge noise "+nf.toFixed(2);
 rs.forEach((r,i)=>{const v=Math.hypot(...r.resid);
  s.appendChild(el("rect",{x:10+i*bw,y:y(v),width:bw-2.5,height:H-30-y(v),
   fill:v>nf?css("--fail"):css("--pass"),opacity:.85}))});
 s.appendChild(el("text",{x:10,y:H-12,class:"lb"})).textContent="10 mixtures x 4 alphas";
})();

(()=>{const s=$("fchart"),W=360,H=170,C=D.charts,
  rows=[["stage-1 DPO (the chart's own set)",C.stage1,css("--pass")],
        ["stage-2 SFT, SAME traits",C.stage2,css("--fail")],
        ["reward hacking",C.sorh_hack,css("--fail")],
        ["honest control",C.sorh_control,css("--fail")],
        ["maths RL",C.math,css("--fail")],
        ["random 5-dim subspace",C.random5d,css("--bar")]],
  sc=(W-185)/C.stage1;
 rows.forEach(([n,v,c],i)=>{const y=14+i*24;
  s.appendChild(el("text",{x:0,y:y+10,class:i?"lb":"lbi"})).textContent=n;
  s.appendChild(el("rect",{x:185,y:y,width:Math.max(v*sc,1.2),height:13,fill:c}));
  s.appendChild(el("text",{x:185+Math.max(v*sc,1.2)+7,y:y+11,class:"lb"}))
    .textContent=v.toFixed(4)});
 s.appendChild(el("text",{x:185,y:H-6,class:"lb"}))
   .textContent="chart length, per unit adapter norm";
})();
</script>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/index.html", "w") as f:
        f.write(HTML.replace("__DATA__", DATA))
    print(f"wrote {OUT}/index.html ({os.path.getsize(f'{OUT}/index.html')/1024:.0f} KB)")


if __name__ == "__main__":
    main()
