// Render the page headlessly, collect console errors, and screenshot the
// interactive panels. The canvases are the parts a build cannot type-check.
const { chromium } = require('/home/vibe12/projects/agent-harness/agents/retreat/node_modules/playwright');
const OUT = '/home/vibe12/projects/persona-curvature/qwen35/shots';
(async () => {
  const b = await chromium.launch();
  for (const vp of [{n:'light',w:1440,h:1000,s:'light'},
                    {n:'dark',w:1440,h:1000,s:'dark'},
                    {n:'mobile',w:390,h:844,s:'light'}]) {
    const ctx = await b.newContext({viewport:{width:vp.w,height:vp.h},
      deviceScaleFactor:2, colorScheme:vp.s});
    const p = await ctx.newPage();
    const errs = [];
    p.on('console', m => { if (m.type()==='error') errs.push(m.text()); });
    p.on('pageerror', e => errs.push('PAGEERROR ' + e.message));
    await p.goto(process.env.URL || 'http://127.0.0.1:8731/', {waitUntil:'load', timeout:60000});
    await p.waitForTimeout(2500);
    const info = await p.evaluate(() => {
      const cv = [...document.querySelectorAll('canvas')].map(c => {
        const x = c.getContext('2d');
        const d = x.getImageData(0,0,c.width,c.height).data;
        let ink = 0; for (let i=3;i<d.length;i+=4) if (d[i] > 8) ink++;
        return {w:c.width, h:c.height, painted: ink};
      });
      const body = document.body;
      return {canvases: cv,
              hscroll: body.scrollWidth > window.innerWidth + 1,
              scrollW: body.scrollWidth, winW: window.innerWidth,
              readLen: (document.querySelector('#explorer .read')||{}).textContent?.length||0,
              bars: document.querySelectorAll('#explorer .bars .bv').length,
              bg: getComputedStyle(body).backgroundColor,
              fg: getComputedStyle(body).color};
    });
    console.log(vp.n, JSON.stringify(info));
    if (errs.length) console.log('  ERRORS:', errs.slice(0,6).join(' | '));
    for (const sel of ['#explorer','#map','#sphere']) {
      const el = await p.$(sel);
      if (el) await el.screenshot({path:`${OUT}/${vp.n}${sel.slice(1)}.png`});
    }
    await p.screenshot({path:`${OUT}/${vp.n}-top.png`});
    await ctx.close();
  }
  await b.close();
})();
