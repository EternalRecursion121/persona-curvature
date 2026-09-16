const { chromium } = require('/home/vibe12/projects/agent-harness/agents/retreat/node_modules/playwright');
(async () => {
  const b = await chromium.launch();
  for (const [scheme, w] of [['light',1440],['dark',1440],['light',360]]) {
    const ctx = await b.newContext({ viewport:{width:w,height:1000}, deviceScaleFactor:2, colorScheme:scheme });
    const p = await ctx.newPage();
    await p.goto('http://127.0.0.1:8731/', {waitUntil:'load'});
    const figs = await p.$$('.figure');
    for (let i=0;i<figs.length;i++) await figs[i].screenshot({path:`shots/fig${i}-${w}-${scheme}.png`});
    const card = await p.$('#phase-10');
    await card.screenshot({path:`shots/card10-${w}-${scheme}.png`});
    await ctx.close();
  }
  await b.close();
})();
