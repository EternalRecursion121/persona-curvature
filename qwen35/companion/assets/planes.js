/* Any two axes.  A two-dimensional cut through the factor chart with the axes
   chosen by the reader: any of the five recovered factors or the six principal
   components on x and on y, one Goldberg group highlighted or all of them, the
   45% density shell of each keyed half, the pole connector (negative-pole mean to
   positive-pole mean), and non-overlapping labels with hairline leaders.  A
   facets mode draws six panels at once, one per Goldberg group plus the held-out
   Lexicon, optionally putting each group on the recovered factor its Big Five
   label is closest to.

   Vanilla SVG, no library.  Data is /data/chart.json from build_companion.py;
   the coordinates there are uncentred, and the page centres them over the 134
   zoo adapters (mean subtracted per axis), as the static figures do, so the
   shared character-register component does not pull every group to one side.

   The shell is a Gaussian kernel density estimate on the group's own points with
   Scott's bandwidth scaled by 0.62 (the same rule as figures/clusters), cut at
   the level enclosing 45% of the mass and traced by marching squares.  It is a
   picture of where the dense half of a keyed group sits, not a hull. */
(function () {
  "use strict";

  var BASE = (document.querySelector('link[href$="assets/site.css"]').getAttribute('href') || '')
    .replace(/assets\/site\.css$/, '');
  var SVGNS = 'http://www.w3.org/2000/svg';
  var GROUPS = ['Agreeableness', 'Conscientiousness', 'EmotionalStability', 'Extraversion', 'Intellect'];
  var GLABEL = { Agreeableness: 'Agreeableness', Conscientiousness: 'Conscientiousness',
    EmotionalStability: 'Emotional stability', Extraversion: 'Extraversion',
    Intellect: 'Intellect', Lexicon: 'Lexicon (held out)' };

  function el(name, attrs, text) {
    var n = document.createElementNS(SVGNS, name);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); }
  function debounce(fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }

  var root = document.querySelector('[data-planes]');
  if (!root) return;

  fetch(BASE + 'data/chart.json').then(function (r) { return r.json(); }).then(init)
    .catch(function (e) { root.textContent = 'Could not load data/chart.json: ' + e; });

  function init(DATA) {
    var zoo = DATA.rows.filter(function (r) { return r.set === 'Goldberg' || r.set === 'Lexicon'; });
    var later = DATA.rows.filter(function (r) { return r.set !== 'Goldberg' && r.set !== 'Lexicon'; });
    var nfa = DATA.nfa || 5;
    var axnames = DATA.axnames;                 // five factor titles then PC1..PC6
    var hasPC = zoo.every(function (r) { return r.p; });
    var naxes = hasPC ? axnames.length : nfa;

    // which recovered factor each Goldberg label is closest to: DATA.big5 maps a
    // Big Five scale to the factor index whose Tucker congruence is best
    var ownAxis = {};
    GROUPS.forEach(function (g) { ownAxis[g] = (DATA.big5 && DATA.big5[g] != null) ? DATA.big5[g] : 0; });
    var hue = {};                               // group -> factor css var
    GROUPS.forEach(function (g) { hue[g] = 'var(--f' + ownAxis[g] + ')'; });
    hue.Lexicon = 'var(--lx)';

    // full coordinate vector per row: factors then components
    function vec(r) { return hasPC ? r.c.concat(r.p) : r.c.slice(); }
    var V = zoo.map(vec);
    var mean = [];
    for (var a = 0; a < naxes; a++) {
      var s = 0; for (var i = 0; i < V.length; i++) s += V[i][a];
      mean.push(s / V.length);
    }
    var VL = later.map(vec);

    // ------------------------------------------------------------ controls
    function axopt(sel) {
      var o = '';
      for (var a = 0; a < naxes; a++) {
        var d = DATA.axdesc && DATA.axdesc[axnames[a]];
        var lab = axnames[a] + (a === 2 && axnames[a] === 'Timidity' ? ' (- timid / + bold)' : '');
        o += '<option value="' + a + '"' + (a === sel ? ' selected' : '') + '>' + esc(lab) +
          (a < nfa ? '' : '') + '</option>';
      }
      return o;
    }
    root.innerHTML =
      '<div class="controls">' +
      '<div class="seg" role="group" aria-label="Layout"><button type="button" class="p-single" aria-pressed="true">One panel</button>' +
      '<button type="button" class="p-facets" aria-pressed="false">Six panels</button></div>' +
      '<label class="ctl">x axis<select class="p-x">' + axopt(0) + '</select></label>' +
      '<label class="ctl">y axis<select class="p-y">' + axopt(1) + '</select></label>' +
      '<label class="ctl p-own-wrap" hidden>Per-group axes<select class="p-own">' +
      '<option value="own">y = each group\u2019s own factor</option>' +
      '<option value="best">best-separating pair per group</option>' +
      '<option value="fixed">the chosen x and y</option></select></label>' +
      '<label class="ctl p-grp-wrap">Highlight<select class="p-grp"><option value="">all five groups</option>' +
      GROUPS.map(function (g) { return '<option value="' + g + '">' + esc(GLABEL[g]) + '</option>'; }).join('') +
      '<option value="Lexicon">Lexicon (held out)</option></select></label>' +
      '<label class="ctl">Labels<select class="p-lab"><option value="hi">highlighted words</option>' +
      '<option value="all">every word</option><option value="none">none</option></select></label>' +
      '<label class="ctl"><input type="checkbox" class="p-shell" checked> shells</label>' +
      '<label class="ctl"><input type="checkbox" class="p-conn" checked> pole connectors</label>' +
      '<label class="ctl"><input type="checkbox" class="p-centre" checked> centred</label>' +
      '<label class="ctl"><input type="checkbox" class="p-later"> the 7 later adapters</label>' +
      '</div>' +
      '<div class="p-canvas"></div>' +
      '<p class="small p-note" style="color:var(--muted)"></p>';

    var q = function (c) { return root.querySelector(c); };
    var canvas = q('.p-canvas');
    var state = { layout: 'single', x: 0, y: 1, own: 'own', grp: '', lab: 'hi',
      shell: true, conn: true, centre: true, later: false };
    // the page may be opened on a URL fragment such as #x=0&y=3&grp=Extraversion
    (function () {
      var h = location.hash.replace(/^#/, '');
      if (!h) return;
      h.split('&').forEach(function (kv) {
        var p = kv.split('='); var k = p[0], v = decodeURIComponent(p[1] || '');
        if (k === 'x' || k === 'y') state[k] = Math.max(0, Math.min(naxes - 1, +v || 0));
        else if (k === 'grp') state.grp = v;
        else if (k === 'layout') state.layout = v === 'facets' ? 'facets' : 'single';
        else if (k === 'lab') state.lab = v;
        else if (k === 'own') state.own = (v === 'best' || v === 'fixed') ? v : 'own';
        else if (k in state && typeof state[k] === 'boolean') state[k] = v !== '0';
      });
    })();
    q('.p-x').value = state.x; q('.p-y').value = state.y; q('.p-grp').value = state.grp;
    q('.p-lab').value = state.lab; q('.p-shell').checked = state.shell; q('.p-conn').checked = state.conn;
    q('.p-centre').checked = state.centre; q('.p-later').checked = state.later; q('.p-own').value = state.own;

    function sync() {
      q('.p-single').setAttribute('aria-pressed', state.layout === 'single');
      q('.p-facets').setAttribute('aria-pressed', state.layout === 'facets');
      q('.p-own-wrap').hidden = state.layout !== 'facets';
      q('.p-grp-wrap').hidden = state.layout === 'facets';
      var frag = 'layout=' + state.layout + '&x=' + state.x + '&y=' + state.y +
        (state.grp ? '&grp=' + state.grp : '') + '&lab=' + state.lab +
        (state.shell ? '' : '&shell=0') + (state.conn ? '' : '&conn=0') +
        (state.centre ? '' : '&centre=0') + (state.later ? '&later=1' : '') + (state.own === 'own' ? '' : '&own=' + state.own);
      if (history.replaceState) history.replaceState(null, '', '#' + frag);
      draw();
    }
    q('.p-single').addEventListener('click', function () { state.layout = 'single'; sync(); });
    q('.p-facets').addEventListener('click', function () { state.layout = 'facets'; sync(); });
    q('.p-x').addEventListener('change', function () { state.x = +this.value; sync(); });
    q('.p-y').addEventListener('change', function () { state.y = +this.value; sync(); });
    q('.p-own').addEventListener('change', function () { state.own = this.value; sync(); });
    q('.p-grp').addEventListener('change', function () { state.grp = this.value; sync(); });
    q('.p-lab').addEventListener('change', function () { state.lab = this.value; sync(); });
    ['shell', 'conn', 'centre', 'later'].forEach(function (k) {
      q('.p-' + k).addEventListener('change', function () { state[k] = this.checked; sync(); });
    });
    window.addEventListener('resize', debounce(draw, 150));
    sync();

    // ------------------------------------------------------------ geometry
    /* the pair of recovered factors on which a group's two poles separate most:
       Mahalanobis distance between the pole means in the pooled within-pole covariance
       of the plane, every one of the ten pairs tried (the same rule as analysis/best_axis_pairs.json) */
    function bestPair(g) {
      var best = null;
      for (var i = 0; i < nfa; i++) for (var j = i + 1; j < nfa; j++) {
        var P = [], Q = [];
        zoo.forEach(function (r, k) { if (r.f !== g) return; (r.k === '+' ? P : Q).push([V[k][i], V[k][j]]); });
        if (P.length < 2 || Q.length < 2) continue;
        var mean = function (A) { var m = [0, 0]; A.forEach(function (p) { m[0] += p[0]; m[1] += p[1]; }); return [m[0] / A.length, m[1] / A.length]; };
                var mp = mean(P), mq = mean(Q);
        // pooled within-pole covariance, then the Mahalanobis distance between the means
        var cov = function (A, m) { var c = [0, 0, 0]; A.forEach(function (p) { var u = p[0] - m[0], v = p[1] - m[1]; c[0] += u * u; c[1] += u * v; c[2] += v * v; }); return c.map(function (x) { return x / (A.length - 1); }); };
        var cp = cov(P, mp), cq = cov(Q, mq);
        var a = 0.5 * (cp[0] + cq[0]), b = 0.5 * (cp[1] + cq[1]), c2 = 0.5 * (cp[2] + cq[2]);
        var det = a * c2 - b * b; if (det <= 1e-12) continue;
        var dx = mp[0] - mq[0], dy = mp[1] - mq[1];
        var d = Math.sqrt((c2 * dx * dx - 2 * b * dx * dy + a * dy * dy) / det);
        if (!best || d > best[2]) best = [i, j, d];
      }
      return best ? best : [state.x, state.y, 0];
    }
    function coords(ax, ay) {
      var out = [];
      for (var i = 0; i < V.length; i++) {
        out.push([V[i][ax] - (state.centre ? mean[ax] : 0), V[i][ay] - (state.centre ? mean[ay] : 0)]);
      }
      return out;
    }
    function coordsLater(ax, ay) {
      return VL.map(function (v) {
        return [v[ax] - (state.centre ? mean[ax] : 0), v[ay] - (state.centre ? mean[ay] : 0)];
      });
    }
    // one span for every panel, so a unit of the chart is the same length everywhere
    function commonSpan() {
      var best = 0;
      for (var a = 0; a < naxes; a++) {
        var lo = Infinity, hi = -Infinity;
        for (var i = 0; i < V.length; i++) { lo = Math.min(lo, V[i][a]); hi = Math.max(hi, V[i][a]); }
        best = Math.max(best, hi - lo);
      }
      return best * 1.12;
    }
    function midOf(a) {
      var lo = Infinity, hi = -Infinity;
      for (var i = 0; i < V.length; i++) { lo = Math.min(lo, V[i][a]); hi = Math.max(hi, V[i][a]); }
      return (lo + hi) / 2 - (state.centre ? mean[a] : 0);
    }

    /* Gaussian KDE with full covariance, Scott's factor times 0.62, cut at the
       level enclosing `frac` of the mass; returns closed loops in data space. */
    function shell(pts, frac, xlim, ylim) {
      var n = pts.length;
      if (n < 5) return null;
      var mx = 0, my = 0, i;
      for (i = 0; i < n; i++) { mx += pts[i][0]; my += pts[i][1]; }
      mx /= n; my /= n;
      var sxx = 0, syy = 0, sxy = 0;
      for (i = 0; i < n; i++) {
        var dx = pts[i][0] - mx, dy = pts[i][1] - my;
        sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
      }
      sxx /= (n - 1); syy /= (n - 1); sxy /= (n - 1);
      var f = Math.pow(n, -1 / 6) * 0.62, f2 = f * f;
      sxx *= f2; syy *= f2; sxy *= f2;
      var det = sxx * syy - sxy * sxy;
      if (det <= 1e-12) return null;
      var ixx = syy / det, iyy = sxx / det, ixy = -sxy / det;
      var G = 110;
      var x0 = xlim[0], x1 = xlim[1], y0 = ylim[0], y1 = ylim[1];
      var Z = new Float64Array(G * G), tot = 0;
      for (var gy = 0; gy < G; gy++) {
        var yy = y0 + (y1 - y0) * gy / (G - 1);
        for (var gx = 0; gx < G; gx++) {
          var xx = x0 + (x1 - x0) * gx / (G - 1), z = 0;
          for (i = 0; i < n; i++) {
            var ex = xx - pts[i][0], ey = yy - pts[i][1];
            z += Math.exp(-0.5 * (ixx * ex * ex + 2 * ixy * ex * ey + iyy * ey * ey));
          }
          Z[gy * G + gx] = z; tot += z;
        }
      }
      var sorted = Array.prototype.slice.call(Z).sort(function (a, b) { return b - a; });
      var cum = 0, level = sorted[sorted.length - 1];
      for (i = 0; i < sorted.length; i++) { cum += sorted[i]; if (cum >= frac * tot) { level = sorted[i]; break; } }
      return march(Z, G, level, x0, x1, y0, y1);
    }
    /* marching squares -> list of closed loops (arrays of [x, y]) */
    function march(Z, G, lev, x0, x1, y0, y1) {
      var segs = [];
      function X(gx) { return x0 + (x1 - x0) * gx / (G - 1); }
      function Y(gy) { return y0 + (y1 - y0) * gy / (G - 1); }
      function interp(a, b, va, vb) { var t = (lev - va) / (vb - va); return a + (b - a) * t; }
      for (var gy = 0; gy < G - 1; gy++) {
        for (var gx = 0; gx < G - 1; gx++) {
          var v0 = Z[gy * G + gx], v1 = Z[gy * G + gx + 1], v2 = Z[(gy + 1) * G + gx + 1], v3 = Z[(gy + 1) * G + gx];
          var c = (v0 > lev ? 1 : 0) | (v1 > lev ? 2 : 0) | (v2 > lev ? 4 : 0) | (v3 > lev ? 8 : 0);
          if (c === 0 || c === 15) continue;
          var pt = {
            t: [interp(X(gx), X(gx + 1), v0, v1), Y(gy)],
            r: [X(gx + 1), interp(Y(gy), Y(gy + 1), v1, v2)],
            b: [interp(X(gx), X(gx + 1), v3, v2), Y(gy + 1)],
            l: [X(gx), interp(Y(gy), Y(gy + 1), v0, v3)]
          };
          var table = { 1: [['l', 't']], 2: [['t', 'r']], 3: [['l', 'r']], 4: [['r', 'b']], 5: [['l', 't'], ['r', 'b']],
            6: [['t', 'b']], 7: [['l', 'b']], 8: [['b', 'l']], 9: [['t', 'b']], 10: [['t', 'r'], ['b', 'l']],
            11: [['r', 'b']], 12: [['r', 'l']], 13: [['t', 'r']], 14: [['l', 't']] };
          table[c].forEach(function (e) { segs.push([pt[e[0]], pt[e[1]]]); });
        }
      }
      // join segments into loops by endpoint proximity
      var key = function (p) { return p[0].toFixed(5) + ',' + p[1].toFixed(5); };
      var byStart = {};
      segs.forEach(function (s, i) { (byStart[key(s[0])] = byStart[key(s[0])] || []).push(i); (byStart[key(s[1])] = byStart[key(s[1])] || []).push(i); });
      var used = new Array(segs.length).fill(false), loops = [];
      for (var i = 0; i < segs.length; i++) {
        if (used[i]) continue;
        used[i] = true;
        var loop = [segs[i][0], segs[i][1]], cur = segs[i][1], guard = 0;
        while (guard++ < segs.length) {
          var cand = byStart[key(cur)] || [], nxt = -1;
          for (var k = 0; k < cand.length; k++) if (!used[cand[k]]) { nxt = cand[k]; break; }
          if (nxt < 0) break;
          used[nxt] = true;
          var s = segs[nxt];
          cur = key(s[0]) === key(cur) ? s[1] : s[0];
          loop.push(cur);
          if (key(cur) === key(loop[0])) break;
        }
        if (loop.length > 3) loops.push(loop);
      }
      return loops;
    }

    // ------------------------------------------------------------ drawing
    function draw() {
      canvas.innerHTML = '';
      var Wpx = Math.max(320, Math.round(canvas.getBoundingClientRect().width) || 900);
      var span = commonSpan();
      if (state.layout === 'single') {
        var H = Math.round(Math.min(Wpx * 0.86 + 74, 900));
        var svg = el('svg', { viewBox: '0 0 ' + Wpx + ' ' + H, role: 'img', class: 'planes' });
        svg.setAttribute('aria-label', 'the 134 adapters on ' + axnames[state.x] + ' against ' + axnames[state.y]);
        panel(svg, 0, 0, Wpx, H, state.x, state.y, state.grp, span, true);
        canvas.appendChild(svg);
        note(state.x, state.y, state.grp);
      } else {
        var cols = Wpx < 720 ? 1 : (Wpx < 1100 ? 2 : 3);
        var pw = Math.floor(Wpx / cols), ph = Math.round(pw + 34);
        var rows = Math.ceil(6 / cols);
        var Ht = rows * ph;
        var svg2 = el('svg', { viewBox: '0 0 ' + Wpx + ' ' + Ht, role: 'img', class: 'planes' });
        svg2.setAttribute('aria-label', 'six panels, one per Goldberg group, each on ' + axnames[state.x] + ' against its own factor');
        var order = GROUPS.concat(['Lexicon']);
        order.forEach(function (g, n) {
          var r = Math.floor(n / cols), c = n % cols;
          var axx = state.x, ay = state.y;
          if (g !== 'Lexicon' && state.own === 'own') {
            ay = ownAxis[g] != null ? ownAxis[g] : state.y;
            if (ay === state.x) ay = state.y === state.x ? (state.x + 1) % nfa : state.y;
          } else if (g !== 'Lexicon' && state.own === 'best') {
            var bp = bestPair(g); axx = bp[0]; ay = bp[1];
          }
          panel(svg2, c * pw, r * ph, pw, ph, axx, ay, g, span, false);
        });
        canvas.appendChild(svg2);
        note(state.x, null, null);
      }
    }

    function note(ax, ay, grp) {
      var n = q('.p-note');
      var src = 'Coordinates: data/chart.json (' + esc(DATA.source) + ')';
      var cen = state.centre ? 'Mean-centred over the 134 zoo adapters. ' : 'Uncentred: the shared component every adapter carries is left in. ';
      var sh = 'Shells enclose the densest 45% of one Goldberg group at one pole (solid = positively keyed, dashed = negatively keyed); the connector joins the negative-pole mean to the positive-pole mean. Colour is the Goldberg label the word carries, not a cluster found in the data. ';
      var own = state.layout === 'facets' && state.own === 'own'
        ? 'In the six-panel view each group is drawn on the recovered factor its Big Five label is closest to by Tucker congruence, so the panels do not share a plane; a unit of the chart is still the same length in every panel. '
        : (state.layout === 'facets' && state.own === 'best'
          ? 'In the six-panel view each group is drawn on the pair of recovered factors on which its two poles separate most (Mahalanobis distance between the pole means in the pooled within-pole covariance, all ten pairs tried, shown in the panel title), so the panels do not share a plane; a unit of the chart is still the same length in every panel. '
          : '');
      n.innerHTML = cen + sh + own + src + '.';
    }

    function panel(svg, X0, Y0, Wd, Hd, ax, ay, grp, span, big) {
      var m = { l: big ? 56 : 44, r: big ? 18 : 14, t: big ? 30 : 44, b: big ? 44 : 34 };
      var iw = Wd - m.l - m.r, ih = Hd - m.t - m.b;
      var g = el('g', { transform: 'translate(' + X0 + ',' + Y0 + ')' });
      svg.appendChild(g);
      var P = coords(ax, ay);
      var PL = state.later ? coordsLater(ax, ay) : [];
      var mx = midOf(ax), my = midOf(ay);
      // equal unit length on both axes; `span` is the widest data range over every
      // axis plus padding, so every point of every panel fits and a unit of the
      // chart is the same length in each
      var scale = Math.min(iw / span, ih / span);
      var xlim = [mx - iw / scale / 2, mx + iw / scale / 2];
      var ylim = [my - ih / scale / 2, my + ih / scale / 2];
      function px(x) { return m.l + (x - xlim[0]) * scale; }
      function py(y) { return m.t + ih - (y - ylim[0]) * scale; }
      var clipId = 'clip' + Math.round(X0) + '_' + Math.round(Y0);
      var defs = el('defs');
      var cp = el('clipPath', { id: clipId });
      cp.appendChild(el('rect', { x: m.l, y: m.t, width: iw, height: ih }));
      defs.appendChild(cp); g.appendChild(defs);
      var gs = el('g', { 'clip-path': 'url(#' + clipId + ')' });

      g.appendChild(el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'var(--card)', stroke: 'var(--rule)' }));
      // zero lines
      if (xlim[0] < 0 && xlim[1] > 0) g.appendChild(el('line', { x1: px(0), x2: px(0), y1: m.t, y2: m.t + ih, stroke: 'var(--rule-soft)' }));
      if (ylim[0] < 0 && ylim[1] > 0) g.appendChild(el('line', { x1: m.l, x2: m.l + iw, y1: py(0), y2: py(0), stroke: 'var(--rule-soft)' }));

      var hiGroups = grp ? [grp] : GROUPS;
      var isHi = function (r) { return grp ? r.f === grp : r.set === 'Goldberg'; };

      // the rest, grey and recessive
      zoo.forEach(function (r, i) {
        if (isHi(r)) return;
        g.appendChild(el('circle', { cx: px(P[i][0]), cy: py(P[i][1]), r: big ? 2.6 : 2.0, fill: 'var(--lx)', 'fill-opacity': 0.75 }));
      });

      // shells and connectors per highlighted keyed group
      hiGroups.forEach(function (gr) {
        if (gr === 'Lexicon') return;
        var col = hue[gr];
        var cents = {};
        ['+', '-'].forEach(function (k) {
          var idx = []; zoo.forEach(function (r, i) { if (r.f === gr && r.k === k) idx.push(i); });
          if (!idx.length) return;
          var pts = idx.map(function (i) { return P[i]; });
          var cx = 0, cy = 0; pts.forEach(function (p) { cx += p[0]; cy += p[1]; });
          cents[k] = [cx / pts.length, cy / pts.length];
          if (state.shell) {
            var padx = (xlim[1] - xlim[0]) * 0.35, pady = (ylim[1] - ylim[0]) * 0.35;
            var loops = shell(pts, 0.45, [xlim[0] - padx, xlim[1] + padx], [ylim[0] - pady, ylim[1] + pady]);
            if (loops) loops.forEach(function (lp) {
              var d = lp.map(function (p, j) { return (j ? 'L' : 'M') + px(p[0]).toFixed(1) + ' ' + py(p[1]).toFixed(1); }).join('') + 'Z';
              gs.appendChild(el('path', { d: d, fill: col, 'fill-opacity': 0.08, stroke: col, 'stroke-width': 1.1,
                'stroke-dasharray': k === '-' ? '5 3' : null, 'stroke-opacity': 0.9 }));
            });
          }
        });
        if (state.conn && cents['+'] && cents['-']) {
          var a = cents['-'], b = cents['+'];
          g.appendChild(el('line', { x1: px(a[0]), y1: py(a[1]), x2: px(b[0]), y2: py(b[1]), stroke: col, 'stroke-width': 1.6, 'stroke-opacity': 0.6, 'stroke-linecap': 'round' }));
          g.appendChild(el('circle', { cx: px(b[0]), cy: py(b[1]), r: 4.5, fill: col, stroke: 'var(--card)', 'stroke-width': 1 }));
          g.appendChild(el('circle', { cx: px(a[0]), cy: py(a[1]), r: 4.5, fill: 'var(--card)', stroke: col, 'stroke-width': 2 }));
        }
      });

      g.appendChild(gs);

      // highlighted points
      var labelIdx = [];
      zoo.forEach(function (r, i) {
        var hi = isHi(r) || (grp === 'Lexicon' && r.set === 'Lexicon');
        if (!hi) { if (state.lab === 'all') labelIdx.push({ i: i, col: 'var(--dim)', xy: P[i] }); return; }
        var col = r.set === 'Lexicon' ? 'var(--ink-soft)' : hue[r.f];
        var c;
        if (r.set === 'Lexicon') {
          var x = px(P[i][0]), y = py(P[i][1]), s = big ? 4 : 3.2;
          c = el('path', { d: 'M' + (x - s) + ' ' + (y - s) + 'L' + (x + s) + ' ' + (y + s) + 'M' + (x - s) + ' ' + (y + s) + 'L' + (x + s) + ' ' + (y - s), stroke: col, 'stroke-width': 1.5 });
        } else if (r.k === '+') {
          c = el('circle', { cx: px(P[i][0]), cy: py(P[i][1]), r: big ? 4.4 : 3.6, fill: col, stroke: 'var(--card)', 'stroke-width': 0.9 });
        } else {
          c = el('circle', { cx: px(P[i][0]), cy: py(P[i][1]), r: big ? 4.4 : 3.6, fill: 'var(--card)', stroke: col, 'stroke-width': 1.6 });
        }
        var tt = el('title', {}, r.t + ' (' + GLABEL[r.f] + (r.k ? ', keyed ' + r.k : '') + ')  ' + axnames[ax] + ' ' + P[i][0].toFixed(2) + ', ' + axnames[ay] + ' ' + P[i][1].toFixed(2));
        c.appendChild(tt);
        g.appendChild(c);
        if (state.lab !== 'none') labelIdx.push({ i: i, col: col, xy: P[i] });
      });
      // later adapters (alignment, hole) as small squares
      later.forEach(function (r, j) {
        if (!PL.length) return;
        var x = px(PL[j][0]), y = py(PL[j][1]);
        var sq = el('rect', { x: x - 3.5, y: y - 3.5, width: 7, height: 7, fill: 'var(--ink)', 'fill-opacity': 0.8, transform: 'rotate(45 ' + x + ' ' + y + ')' });
        sq.appendChild(el('title', {}, r.t + ' (' + r.set + ')'));
        g.appendChild(sq);
        labelIdx.push({ i: -1, col: 'var(--ink)', xy: PL[j], text: r.t });
      });

      // axis titles and panel title
      var fs = big ? 12 : 10.5;
      g.appendChild(el('text', { x: m.l + iw / 2, y: Hd - (big ? 14 : 10), 'text-anchor': 'middle', class: 'axis', 'font-size': fs }, axTitle(ax)));
      var yl = el('text', { x: big ? 16 : 13, y: m.t + ih / 2, 'text-anchor': 'middle', class: 'axis', 'font-size': fs,
        transform: 'rotate(-90 ' + (big ? 16 : 13) + ' ' + (m.t + ih / 2) + ')' }, axTitle(ay));
      g.appendChild(yl);
      if (!big) {
        var n = 0; zoo.forEach(function (r) { if (grp === 'Lexicon' ? r.set === 'Lexicon' : r.f === grp) n++; });
        g.appendChild(el('text', { x: m.l, y: 17, class: 'ptitle', 'font-size': 13, fill: 'var(--ink)' }, GLABEL[grp]));
        g.appendChild(el('text', { x: m.l, y: 32, class: 'psub', 'font-size': 10, fill: 'var(--dim)' },
          grp === 'Lexicon' ? n + ' held-out words, never used to define the factors'
            : (state.layout === 'facets' && state.own === 'best'
              ? n + ' markers; pole separation ' + bestPair(grp)[2].toFixed(1)
              : n + ' markers on ' + axTitle(ay))));
      } else if (grp) {
        g.appendChild(el('text', { x: m.l, y: 19, class: 'ptitle', 'font-size': 13, fill: 'var(--ink)' }, GLABEL[grp]));
      }

      // labels: greedy non-overlapping placement in rings, leader when displaced
      var placed = [];
      var fsl = big ? 11 : 9.5;
      // every point is an obstacle
      P.forEach(function (p) { var x = px(p[0]), y = py(p[1]); placed.push([x - 4, y - 4, x + 4, y + 4]); });
      var cand = [[0, 0]];
      var rings = big ? [9, 15, 22, 31, 42, 55, 70, 87] : [7, 12, 18, 25, 33, 42, 52, 63];
      rings.forEach(function (rad, n) { for (var k = 0; k < 18; k++) { var ang = 2 * Math.PI * k / 18 + 0.13 * n; cand.push([rad * Math.cos(ang), rad * Math.sin(ang)]); } });
      labelIdx.sort(function (a, b) { return (b.xy[0] * b.xy[0] + b.xy[1] * b.xy[1]) - (a.xy[0] * a.xy[0] + a.xy[1] * a.xy[1]); });
      labelIdx.forEach(function (L) {
        var text = L.text || zoo[L.i].t;
        var w = text.length * fsl * 0.58 + 4, h = fsl * 1.15;
        var x0 = px(L.xy[0]), y0 = py(L.xy[1]);
        var best = null, bestAny = null;
        for (var ci = 0; ci < cand.length; ci++) {
          var dx = cand[ci][0], dy = -cand[ci][1];
          var ha = dx > 1 ? 'start' : (dx < -1 ? 'end' : 'middle');
          var bx = x0 + dx, by = y0 + dy;
          var lx = ha === 'start' ? bx : (ha === 'end' ? bx - w : bx - w / 2);
          var ly = by - h / 2;
          if (lx < m.l + 2 || lx + w > m.l + iw - 2 || ly < m.t + 2 || ly + h > m.t + ih - 2) continue;
          var ov = 0;
          for (var k = 0; k < placed.length; k++) {
            var o = placed[k];
            var ow = Math.min(lx + w, o[2]) - Math.max(lx, o[0]), oh = Math.min(ly + h, o[3]) - Math.max(ly, o[1]);
            if (ow > 0 && oh > 0) ov += ow * oh;
          }
          if (ov <= 0) { best = [bx, by, ha, lx, ly, dx, dy]; break; }
          if (!bestAny || ov < bestAny[0]) bestAny = [ov, bx, by, ha, lx, ly, dx, dy];
        }
        var op = 1;
        if (!best) { if (!bestAny) return; best = bestAny.slice(1); op = 0.8; }
        var bx2 = best[0], by2 = best[1], ha2 = best[2], lx2 = best[3], ly2 = best[4], ddx = best[5], ddy = best[6];
        placed.push([lx2, ly2, lx2 + w, ly2 + h]);
        var rad = Math.sqrt(ddx * ddx + ddy * ddy);
        if (rad >= (big ? 15 : 12)) {
          var ex = ha2 === 'start' ? lx2 - 1 : (ha2 === 'end' ? lx2 + w + 1 : bx2);
          g.appendChild(el('line', { x1: x0, y1: y0, x2: ex, y2: by2, stroke: L.col, 'stroke-width': 0.6, 'stroke-opacity': 0.5 }));
        }
        g.appendChild(el('text', { x: bx2, y: by2, 'text-anchor': ha2, 'dominant-baseline': 'middle', 'font-size': fsl, fill: L.col, 'fill-opacity': op, class: 'plabel' }, text));
      });
    }

    function axTitle(a) {
      var n = axnames[a];
      if (n === 'Timidity') return 'Timidity (- timid / + bold)';
      return n;
    }
  }
})();
