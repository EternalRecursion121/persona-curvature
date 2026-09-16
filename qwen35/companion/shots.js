/* Screenshot every page of the companion site at three widths in both themes,
   and report layout defects (horizontal scroll, elements past the viewport,
   svg text rendering below 8 css px or escaping its own box). */
const { chromium } = require('/home/vibe12/projects/agent-harness/agents/retreat/node_modules/playwright');
const BASE = process.env.BASE || 'http://127.0.0.1:8731/';
const OUT = process.env.OUT || '/home/vibe12/projects/persona-curvature/qwen35/companion/shots';
const ROUND = process.env.ROUND || 'r1';
const PAGES = (process.env.PAGES || 'index,chart,behaviour,stage-two,traits,data,methods,traits/warm').split(',');
const VPS = [
  { name: '360', width: 360, height: 900 },
  { name: '1024', width: 1024, height: 900 },
  { name: '1440', width: 1440, height: 1000 },
];
const SCHEMES = (process.env.SCHEMES || 'light,dark').split(',');

const PROBE = () => {
  const res = { horizontalScroll: document.documentElement.scrollWidth > window.innerWidth + 1,
                scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth,
                overflow: [], tinyText: [], clipped: [], errors: [], empty: [] };
  document.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && (r.right > window.innerWidth + 1 || r.left < -1)) {
      if (el.closest('.scrollx') || el.closest('.tablewrap')) return;
      res.overflow.push(el.tagName + '.' + (el.className.baseVal ?? el.className ?? '') + ' r=' + Math.round(r.right));
    }
  });
  document.querySelectorAll('svg').forEach(svg => {
    const sr = svg.getBoundingClientRect();
    if (!sr.width || !svg.viewBox || !svg.viewBox.baseVal.width) return;
    const scale = sr.width / svg.viewBox.baseVal.width;
    svg.querySelectorAll('text').forEach(t => {
      const fs = parseFloat(getComputedStyle(t).fontSize) * scale;
      if (fs < 8) res.tinyText.push((t.textContent || '').slice(0, 24) + ' @' + fs.toFixed(1) + 'px');
      const b = t.getBoundingClientRect();
      if (b.width && (b.right > sr.right + 2 || b.left < sr.left - 2))
        res.clipped.push((t.textContent || '').slice(0, 24));
    });
  });
  /* functional assertions: a widget that renders nothing but throws nothing is
     still broken, and round 1's probe could not see that */
  const need = [
    ['#scatter-wrap svg circle, #scatter-wrap svg path', 60, 'scatter marks'],
    ['#loadings svg rect', 20, 'loading bars'],
    ['#neighbours table', 3, 'neighbour tables'],
    ['#trait-list a', 20, 'trait cards'],
    ['#dose-chart svg polyline', 5, 'dose-response lines'],
    ['#dose-text .gen', 1, 'dose-response generations'],
    ['#s2-treat .gen', 1, 'stage-two treatment generation'],
    ['#s2-ctrl .gen', 1, 'stage-two control generation'],
  ];
  need.forEach(([sel, min, what]) => {
    const anchor = sel.split(' ')[0];
    if (!document.querySelector(anchor)) return;   // widget not on this page
    const n = document.querySelectorAll(sel).length;
    if (n < min) res.empty.push(what + ': ' + n + ' < ' + min);
  });
  return res;
};

(async () => {
  const browser = await chromium.launch();
  const report = {};
  for (const scheme of SCHEMES) {
    for (const vp of VPS) {
      const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height },
                                             deviceScaleFactor: 1, colorScheme: scheme });
      for (const p of PAGES) {
        const page = await ctx.newPage();
        const errs = [];
        page.on('pageerror', e => errs.push(String(e)));
        page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
        const url = BASE + p + '.html';
        const resp = await page.goto(url, { waitUntil: 'networkidle' }).catch(e => ({ status: () => 'ERR ' + e }));
        await page.waitForTimeout(400);
        const key = `${p}|${vp.name}|${scheme}`;
        const name = `${ROUND}_${p.replace(/\//g, '-')}_${vp.name}_${scheme}.png`;
        await page.screenshot({ path: `${OUT}/${name}`, fullPage: true }).catch(() => {});
        const probe = await page.evaluate(PROBE).catch(e => ({ error: String(e) }));
        probe.status = resp && resp.status ? resp.status() : '?';
        probe.errors = errs;
        probe.shot = name;
        report[key] = probe;
        await page.close();
      }
      await ctx.close();
    }
  }
  await browser.close();
  const bad = {};
  for (const k in report) {
    const r = report[k];
    if (r.horizontalScroll || (r.overflow || []).length || (r.tinyText || []).length ||
        (r.clipped || []).length || (r.errors || []).length || (r.empty || []).length ||
        r.status !== 200) bad[k] = r;
  }
  console.log(JSON.stringify({ pages: Object.keys(report).length, problems: bad }, null, 1));
})();
