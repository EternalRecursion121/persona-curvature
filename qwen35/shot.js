const { chromium } = require('/home/vibe12/projects/agent-harness/agents/retreat/node_modules/playwright');
const URL = process.env.URL || 'http://127.0.0.1:8731/';
const OUT = '/home/vibe12/projects/persona-curvature/qwen35/shots';

const viewports = [
  { name: 'mobile-360', width: 360, height: 800, scheme: 'light' },
  { name: 'desktop-1440', width: 1440, height: 1000, scheme: 'light' },
  { name: 'desktop-1440-dark', width: 1440, height: 1000, scheme: 'dark' },
];

async function main() {
  const browser = await chromium.launch();
  const report = {};
  for (const vp of viewports) {
    const ctx = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 2,
      colorScheme: vp.scheme,
    });
    const page = await ctx.newPage();
    await page.goto(URL, { waitUntil: 'load' });
    await page.evaluate(() => document.querySelectorAll('details').forEach(d => d.open = true));
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${OUT}/${vp.name}.png`, fullPage: true });

    report[vp.name] = await page.evaluate(() => {
      const res = { horizontalScroll: document.documentElement.scrollWidth > window.innerWidth,
                    scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth,
                    overflow: [], clipped: [], svgs: [] };
      // any element extending past the viewport
      document.querySelectorAll('*').forEach(el => {
        const r = el.getBoundingClientRect();
        if (r.width > 0 && (r.right > window.innerWidth + 1 || r.left < -1)) {
          if (el.closest('.tablewrap')) return; // intentionally scrollable
          res.overflow.push(el.tagName + '.' + (el.className.baseVal ?? el.className ?? '') + ' r=' + Math.round(r.right));
        }
      });
      // svg text that renders below 8 css px, or escapes its own svg box
      document.querySelectorAll('svg').forEach(svg => {
        const sr = svg.getBoundingClientRect();
        let minFs = 99, escapes = 0;
        svg.querySelectorAll('text').forEach(t => {
          const b = t.getBoundingClientRect();
          const scale = sr.width / svg.viewBox.baseVal.width;
          const fs = parseFloat(getComputedStyle(t).fontSize) * scale;
          if (fs < minFs) minFs = fs;
          if (b.right > sr.right + 2 || b.left < sr.left - 2 || b.bottom > sr.bottom + 2 || b.top < sr.top - 2) {
            escapes++;
            res.clipped.push((t.textContent || '').slice(0, 30) + ' | svg=' + Math.round(sr.left) + '-' + Math.round(sr.right) + ' txt=' + Math.round(b.left) + '-' + Math.round(b.right));
          }
        });
        if (sr.width > 0) res.svgs.push({ w: Math.round(sr.width), h: Math.round(sr.height), minTextPx: +minFs.toFixed(1), escapes });
      });
      return res;
    });
    await ctx.close();
  }
  console.log(JSON.stringify(report, null, 1));
  await browser.close();
}
main();
