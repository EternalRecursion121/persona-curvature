/* Steering dose-response: judged Big Five means against alpha, with the model's
   actual answers at the selected dose. Data from /data/steer/*.json. */
(function () {
  "use strict";
  var BASE = (document.querySelector('link[href$="assets/site.css"]').getAttribute('href') || '')
    .replace(/assets\/site\.css$/, '');
  var SVGNS = 'http://www.w3.org/2000/svg';
  var FC = ['var(--f0)', 'var(--f1)', 'var(--f2)', 'var(--f3)', 'var(--f4)'];
  var SCALES = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'EmotionalStability', 'Intellect'];
  var SHORT = { Extraversion: 'E', Agreeableness: 'A', Conscientiousness: 'C', EmotionalStability: 'ES', Intellect: 'I' };
  var B5COL = {};

  function el(n, a) { var e = document.createElementNS(SVGNS, n); for (var k in a) if (a[k] != null) e.setAttribute(k, a[k]); return e; }
  function get(u) { return fetch(BASE + u).then(function (r) { return r.json(); }); }
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }

  var IDX = null, CUR = null, ALPHAS = [], AKEYS = [];
  var dirSel, alphaIn, alphaLab, chartHost, textHost;

  function leadScale(name) {
    /* which Big Five scale the direction is meant to move, from its own name */
    if (name.indexOf('identity_') === 0) return name.slice(9);
    var m = { FA_Warmth: 'Agreeableness', FA_Competence: 'Conscientiousness',
              FA_FearfulWithdrawal: 'EmotionalStability', FA_Arousal: 'Extraversion',
              FA_Imagination: 'Intellect' };
    return m[name] || null;
  }

  function drawChart(name, data) {
    var W = Math.max(300, Math.round(chartHost.getBoundingClientRect().width) || 520);
    var narrow = W < 460;
    var H = narrow ? 300 : 340, L = 42, R = narrow ? 16 : 158, T = 26, B = narrow ? 62 : 44;
    var lead = leadScale(name);
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });
    svg.setAttribute('aria-label', 'judged Big Five means against alpha for ' + name);
    var lo = 1, hi = 7;
    var X = function (i) { return L + (W - L - R) * i / (ALPHAS.length - 1); };
    var Y = function (v) { return H - B - (H - T - B) * (v - lo) / (hi - lo); };
    for (var t = lo; t <= hi; t++) {
      svg.appendChild(el('line', { x1: L, x2: W - R, y1: Y(t), y2: Y(t), stroke: 'var(--rule-soft)' }));
      var lb = el('text', { x: L - 7, y: Y(t) + 3.5, class: 'axis', 'text-anchor': 'end' });
      lb.textContent = t; svg.appendChild(lb);
    }
    ALPHAS.forEach(function (a, i) {
      var lb = el('text', { x: X(i), y: H - B + 15, class: 'axis', 'text-anchor': 'middle' });
      lb.textContent = a; svg.appendChild(lb);
    });
    var ax = el('text', { x: (L + W - R) / 2, y: narrow ? H - 26 : H - 8, class: 'axis', 'text-anchor': 'middle' });
    ax.textContent = 'alpha (weight-space dose)'; svg.appendChild(ax);
    var ay = el('text', { x: L - 6, y: T - 10, class: 'axis' });
    ay.textContent = 'judged score, 1 to 7'; svg.appendChild(ay);

    var cur = +alphaIn.value;
    svg.appendChild(el('line', { x1: X(cur), x2: X(cur), y1: T, y2: H - B, stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));

    SCALES.forEach(function (s) {
      var pts = [], i;
      for (i = 0; i < ALPHAS.length; i++) {
        var m = data.means[AKEYS[i]];
        if (m && m[s] != null) pts.push([X(i), Y(m[s])]);
      }
      if (!pts.length) return;
      var isLead = s === lead;
      var col = B5COL[s] || 'var(--muted)';
      svg.appendChild(el('polyline', {
        fill: 'none', stroke: col, 'stroke-width': isLead ? 2.6 : 1.2,
        'stroke-opacity': isLead ? 1 : 0.42,
        points: pts.map(function (p) { return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' ')
      }));
      pts.forEach(function (p) {
        svg.appendChild(el('circle', { cx: p[0], cy: p[1], r: isLead ? 3.2 : 2, fill: col, 'fill-opacity': isLead ? 1 : 0.42 }));
      });
      var last = pts[pts.length - 1];
      var lab = el('text', { class: 'axis' });
      if (narrow) {
        lab.setAttribute('x', L + SCALES.indexOf(s) * ((W - L) / 5));
        lab.setAttribute('y', H - 8);
        lab.textContent = SHORT[s];
      } else {
        lab.setAttribute('x', W - R + 8);
        lab.setAttribute('y', last[1] + 3.5);
        lab.textContent = SHORT[s] + ' ' + s.replace('EmotionalStability', 'Emot. stability');
      }
      lab.setAttribute('fill', col);
      lab.setAttribute('font-weight', isLead ? '700' : '400');
      lab.setAttribute('opacity', isLead ? 1 : 0.6);
      svg.appendChild(lab);
      /* fall back to the initial if the name does not fit, whatever font loaded */
      if (!narrow) {
        try {
          var x0 = parseFloat(lab.getAttribute('x'));
          if (x0 + lab.getComputedTextLength() > W - 4) lab.textContent = SHORT[s];
        } catch (e) { /* getComputedTextLength unavailable */ }
      }
    });
    if (!narrow) {
      /* two series ending at nearly the same score would print over each other */
      var labs = [].slice.call(svg.querySelectorAll('text'))
        .filter(function (t) { return +t.getAttribute('x') > W - R; })
        .sort(function (a, b) { return +a.getAttribute('y') - +b.getAttribute('y'); });
      for (var q = 1; q < labs.length; q++) {
        var prev = +labs[q - 1].getAttribute('y'), cur = +labs[q].getAttribute('y');
        if (cur - prev < 14) labs[q].setAttribute('y', prev + 14);
      }
    }
    chartHost.innerHTML = '';
    chartHost.appendChild(svg);
    var note = document.createElement('p');
    note.className = 'small';
    note.style.color = 'var(--muted)';
    note.innerHTML = lead ? 'The bold line is ' + esc(lead.replace('EmotionalStability', 'Emotional stability')) +
      ', the scale this direction is meant to move.' : '';
    chartHost.appendChild(note);
  }

  function drawText(data) {
    var a = AKEYS[+alphaIn.value];
    var gens = data.generations[a] || [];
    var out = '<h4 style="margin-top:0">What the model said at alpha ' +
      esc(ALPHAS[+alphaIn.value]) + '</h4>';
    if (!gens.length) out += '<p class="small">No generations were shipped for this dose.</p>';
    gens.forEach(function (g, i) {
      out += '<div class="gen"><span class="q">' + esc(data.prompts[i] || '') + '</span>' +
        esc(String(g).trim()) + '</div>';
    });
    textHost.innerHTML = out;
  }

  function load(name) {
    get('data/steer/' + name + '.json').then(function (d) {
      CUR = d; drawChart(name, d); drawText(d);
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    dirSel = document.getElementById('dose-dir');
    if (!dirSel) return;
    alphaIn = document.getElementById('dose-alpha');
    alphaLab = document.getElementById('dose-alpha-label');
    chartHost = document.getElementById('dose-chart');
    textHost = document.getElementById('dose-text');
    get('data/chart.json').then(function (c) {
      c.factors.forEach(function (f, i) { B5COL[f.b5] = FC[i]; });
      return get('data/steer/index.json');
    }).then(function (ix) {
      IDX = ix; ALPHAS = ix.alphas;
      AKEYS = ix.alpha_keys || ALPHAS.map(String);
      alphaIn.max = ALPHAS.length - 1;
      alphaIn.value = ALPHAS.indexOf(0) >= 0 ? ALPHAS.indexOf(0) : 3;
      function label() { alphaLab.textContent = 'alpha = ' + ALPHAS[+alphaIn.value]; }
      label();
      dirSel.addEventListener('change', function () { load(dirSel.value); });
      alphaIn.addEventListener('input', function () {
        label();
        if (CUR) { drawChart(dirSel.value, CUR); drawText(CUR); }
      });
      load(dirSel.value);
      var t; window.addEventListener('resize', function () {
        clearTimeout(t); t = setTimeout(function () { if (CUR) drawChart(dirSel.value, CUR); }, 200);
      });
    }).catch(function (e) {
      if (chartHost) chartHost.innerHTML = '<p class="caveat">The steering data did not load: ' + esc(e) + '</p>';
    });
  });
})();
