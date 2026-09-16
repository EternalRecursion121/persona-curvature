/* The 3D map.  A port of the interactive blog's canvas chart
   (qwen35/blog_page/_js.txt, makeMap) into the companion site, parameterised by
   data URL so the same instrument draws the factor chart on index.html and
   chart.html, the small chart pinned to one adapter on all 141 trait pages, and
   the per-stage cloud on stage-two.html.

   Everything the blog's map does is here: a rotating orthographic projection
   whose tilt wraps freely in both axes, depth carried in radius and alpha,
   points z-sorted so near ones paint last, filled centres for positively keyed
   trait words and ringed centres for negatively keyed ones, full axis lines
   with the Timidity label at its negative pole, group chips, an x/y/z axis
   picker over the five factors and the six components, hover to name and click
   to pin, and the cloud centred on the mean adapter.  Two markers in the blog
   are held behind `if (false)` -- they were removed at Samuel's request on
   2026-09-14 -- and they are ported disabled, as they are there.

   The companion's own extras ride on top: colour by factor / Big Five keying /
   provenance, a set filter, a search that pins its match, and a panel that
   shows the pinned adapter's five loadings with a link to its page.

   Every colour is read from a CSS custom property at draw time, so the chart
   follows the theme; it redraws on both theme signals (the data-theme stamp and
   the OS preference), because the un-stamped state only sees the second.

   No library.  Data: /data/chart.json and /data/stages.json, both written by
   companion/build_companion.py. */
(function () {
  "use strict";

  var BASE = (document.querySelector('link[href$="assets/site.css"]').getAttribute('href') || '')
    .replace(/assets\/site\.css$/, '');

  /* the six data hues, one per Big Five scale of the trait word; the five
     recovered factors alias onto the same six in site.css */
  var HUE = {
    Extraversion: 'ex', Agreeableness: 'ag', Conscientiousness: 'co',
    EmotionalStability: 'es', Intellect: 'in', Lexicon: 'lx',
    Alignment: 'lx', Hole: 'lx'
  };
  var GROUP_LABEL = {
    Extraversion: 'Extraversion', Agreeableness: 'Agreeableness',
    Conscientiousness: 'Conscientiousness', EmotionalStability: 'Emot. stability',
    Intellect: 'Intellect', Lexicon: 'Lexicon', Alignment: 'Alignment', Hole: 'Hole'
  };

  /* resolve a custom property, following an alias chain such as --f0: var(--ag) */
  function cssv(name) {
    var cs = getComputedStyle(document.documentElement), v = '', n = 0;
    v = cs.getPropertyValue('--' + name).trim();
    while (v.indexOf('var(') === 0 && n++ < 6) {
      var m = /^var\(\s*--([A-Za-z0-9_-]+)/.exec(v);
      if (!m) break;
      v = cs.getPropertyValue('--' + m[1]).trim();
    }
    return v || '#888888';
  }

  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  function $(s, r) { return (r || document).querySelector(s); }
  function el(t, c, x) {
    var n = document.createElement(t);
    if (c) n.className = c;
    if (x != null) n.textContent = x;
    return n;
  }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  /* ------------------------------------------------------------------ 3D map */
  var TAU = Math.PI * 2;
  function wrap(x) { x %= TAU; return x < 0 ? x + TAU : x; }
  function Rot() { this.a = 0.62; this.b = -0.34; }
  /* Tilt is not clamped: both axes wrap freely, so the cloud can be rolled
     through a full revolution in either direction as many times as you like. */
  Rot.prototype.spinBy = function (da, db) { this.a = wrap(this.a + da); this.b = wrap(this.b + db); };
  Rot.prototype.apply = function (p) {
    var ca = Math.cos(this.a), sa = Math.sin(this.a), cb = Math.cos(this.b), sb = Math.sin(this.b);
    var x = p[0] * ca - p[2] * sa, z = p[0] * sa + p[2] * ca;
    var y = p[1] * cb - z * sb; z = p[1] * sb + z * cb;
    return [x, y, z];
  };

  /* ------------------------------------------------------------- the markup */
  function shell(root, D, opt) {
    var compact = opt.compact;
    var axopt = function (sel) {
      var o = [], n = D.nfa || D.factors.length;
      o.push('<optgroup label="factors">');
      for (var i = 0; i < n; i++)
        o.push('<option value="' + i + '"' + (i === sel ? ' selected' : '') + '>' +
          esc(D.axnames[i]) + '</option>');
      o.push('</optgroup>');
      if (D.axnames.length > n) {
        o.push('<optgroup label="components">');
        for (var j = n; j < D.axnames.length; j++)
          o.push('<option value="' + j + '"' + (j === sel ? ' selected' : '') + '>' +
            esc(D.axnames[j]) + '</option>');
        o.push('</optgroup>');
      }
      return o.join('');
    };
    var stageSel = '';
    if (opt.stages && D.stages) {
      stageSel = '<label class="ctl">Stage<select class="m-stage">' +
        D.stages.order.map(function (k) {
          return '<option value="' + esc(k) + '">' + esc(D.stages.label[k]) + '</option>';
        }).join('') + '</select></label>';
    }
    var ctl = '<div class="controls">' + stageSel +
      '<label class="ctl">x axis<select class="m-x">' + axopt(opt.axes[0]) + '</select></label>' +
      '<label class="ctl">y axis<select class="m-y">' + axopt(opt.axes[1]) + '</select></label>' +
      '<label class="ctl">z axis<select class="m-z">' + axopt(opt.axes[2]) + '</select></label>' +
      (compact ? '' :
        '<label class="ctl">Colour by<select class="m-col">' +
        '<option value="factor">leading factor</option>' +
        '<option value="big5">Big Five keying of the trait word</option>' +
        '<option value="set">provenance set</option></select></label>' +
        /* the per-stage file holds the 134 zoo adapters and nothing else -- the seven
           later adapters have no stage-two or persona training -- so on the stage map
           a set filter would be a control that silently does nothing */
        (opt.stages ? '' :
          '<label class="ctl">Show<select class="m-set">' +
          '<option value="all">all ' + D.rows.length + ' adapters</option>' +
          '<option value="zoo">the 134 zoo adapters</option>' +
          '<option value="goldberg">Goldberg markers only</option>' +
          '<option value="lexicon">Lexicon words only</option>' +
          '<option value="extra">alignment and hole words only</option></select></label>') +
        '<label class="ctl">Find<input type="search" class="m-find" placeholder="trait word" autocomplete="off"></label>') +
      '</div>';

    /* the Big Five order the rest of the site uses, then the unkeyed sets; the order
       the rows happen to be in means nothing and reads as an oversight */
    var ORDER = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'EmotionalStability',
                 'Intellect', 'Lexicon', 'Alignment', 'Hole'];
    var present = {};
    D.rows.forEach(function (r) { present[r.f] = 1; });
    var groups = ORDER.filter(function (g) { return present[g]; });
    Object.keys(present).forEach(function (g) { if (groups.indexOf(g) < 0) groups.push(g); });
    var chips = '<div class="chips">' + groups.map(function (g) {
      return '<button class="chip" type="button" data-f="' + esc(g) + '" aria-pressed="true" ' +
        'style="--c:var(--' + (HUE[g] || 'lx') + ')">' + esc(GROUP_LABEL[g] || g) + '</button>';
    }).join('') + '</div>';

    var panel = compact ? '' :
      '<div class="pinpanel"><h4>Pinned adapter</h4><div class="pinbody">' +
      '<p class="empty">Hover a point for its trait word. Click one to pin it here with its ' +
      'five loadings and a link to its page; click it again to let go.</p></div></div>';

    root.className = 'map' + (compact ? ' compact' : '');
    root.innerHTML = ctl + chips +
      '<div class="split"><div><canvas></canvas>' +
      '<p class="hintline maptip"></p><p class="hintline axisnames"></p>' +
      '<p class="hintline mapnote">' + esc(opt.note || '') + '</p></div>' +
      (panel ? '<div>' + panel + '</div>' : '') + '</div>';
  }

  /* ---------------------------------------------------------------- the map */
  function makeMap(root, D, opt) {
    shell(root, D, opt);
    var cv = $('canvas', root), ctx = cv.getContext('2d');
    var tip = $('.maptip', root), pinbody = $('.pinbody', root);
    var rot = new Rot(), drag = null, spin = !reduce, hover = -1, pinned = -1;
    var axes = opt.axes.slice();
    var colour = 'factor', setf = 'all', find = '';
    var stage = D.stages ? D.stages.order[opt.stage || 0] : null;

    var NFA = D.nfa || D.factors.length;
    var show = {};
    D.rows.forEach(function (r) { show[r.f] = true; });

    /* One score array with two families of columns: 0..NFA-1 are the factor-chart
       axes (fa_chart.py, the primary frame since 2026-09-08), the rest are PC1-PC6.
       They are in different units -- a chart coordinate is an inner product with a
       unit basis vector, a PC score is an eigenvector score -- so each family is
       scaled by its own maximum and never by the other's. */
    var S = [], mxf = 0, mxp = 0;
    function rebuild() {
      var C = stage && D.stages ? D.stages.coords[stage] : null;
      S = []; mxf = 0; mxp = 0;
      D.rows.forEach(function (r, i) {
        var c = C ? (C[r.s] || null) : r.c;
        var pc = r.p || null;
        S.push({ c: c, p: pc });
        if (c) for (var j = 0; j < c.length; j++) mxf = Math.max(mxf, Math.abs(c[j]));
        if (pc) for (var k = 0; k < pc.length; k++) mxp = Math.max(mxp, Math.abs(pc[k]));
      });
    }
    function val(i, j) {
      var s = S[i];
      if (j < NFA) return s.c ? s.c[j] / (mxf || 1) : null;
      return s.p ? s.p[j - NFA] / (mxp || 1) : null;
    }
    function has(i) {
      return val(i, axes[0]) !== null && val(i, axes[1]) !== null && val(i, axes[2]) !== null;
    }
    function inSet(r) {
      if (setf === 'zoo') return r.set === 'Goldberg' || r.set === 'Lexicon';
      if (setf === 'goldberg') return r.set === 'Goldberg';
      if (setf === 'lexicon') return r.set === 'Lexicon';
      if (setf === 'extra') return r.set === 'Alignment' || r.set === 'Hole';
      return true;
    }
    function colourOf(r) {
      if (colour === 'set') {
        return r.set === 'Goldberg' ? cssv('ink')
          : r.set === 'Lexicon' ? cssv('f4') : cssv('f3');
      }
      if (colour === 'big5') return cssv(HUE[r.f] || 'lx');
      return (r.lf === null || r.lf === undefined) ? cssv('lx') : cssv('f' + r.lf);
    }

    /* The chart is uncentred: every adapter carries a shared component (the grand
       mean, the character-register direction), which pushes the whole cloud off the
       origin.  Plot deviations from the centroid over the 134 zoo adapters, so the
       cross sits at the mean adapter and toggling a group does not move it. */
    function centroid() {
      var c = [0, 0, 0], n = 0;
      for (var i = 0; i < D.rows.length; i++) {
        var r = D.rows[i];
        if (r.set !== 'Goldberg' && r.set !== 'Lexicon') continue;
        if (!has(i)) continue;
        for (var k = 0; k < 3; k++) c[k] += val(i, axes[k]);
        n++;
      }
      if (n) for (var k2 = 0; k2 < 3; k2++) c[k2] /= n;
      return c;
    }
    function pts() {
      var out = [], c = centroid();
      for (var i = 0; i < D.rows.length; i++) {
        var r = D.rows[i];
        if (!show[r.f] || !inSet(r) || !has(i)) continue;
        out.push({
          i: i, p: [val(i, axes[0]) - c[0], val(i, axes[1]) - c[1], val(i, axes[2]) - c[2]]
        });
      }
      return out;
    }
    function dimmed(r) {
      return find && r.s.indexOf(find) < 0 && r.t.toLowerCase().indexOf(find) < 0;
    }
    function size() {
      var w = cv.clientWidth || root.clientWidth || 640;
      var h = Math.round(Math.min(w * (opt.compact ? 0.84 : 0.78), opt.compact ? 340 : 520));
      var dpr = Math.min(devicePixelRatio || 1, 2);
      cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return [w, h];
    }

    function draw() {
      var wh = size(), w = wh[0], h = wh[1], cxx = w / 2, cyy = h / 2, R = Math.min(w, h) * 0.40;
      ctx.clearRect(0, 0, w, h);
      /* axis cross */
      var ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      var nm = [D.axnames[axes[0]], D.axnames[axes[1]], D.axnames[axes[2]]];
      ctx.font = '600 10px "IBM Plex Mono", monospace';
      for (var a = 0; a < 3; a++) {
        var q = rot.apply(ax[a]);
        ctx.strokeStyle = cssv('rule'); ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cxx - q[0] * R * 1.16, cyy + q[1] * R * 1.16);
        ctx.lineTo(cxx + q[0] * R * 1.16, cyy - q[1] * R * 1.16);
        ctx.stroke();
        ctx.fillStyle = cssv('faint');
        /* factors are oriented so their best Goldberg congruence is positive; Timidity
           is named for its negative pole, so its label goes at the negative end */
        var sgn = (String(nm[a]).indexOf('Timidity') === 0) ? -1 : 1;
        ctx.fillText(nm[a], cxx + sgn * q[0] * R * 1.22 - 6, cyy - sgn * q[1] * R * 1.22 + 3);
      }
      var P = pts().map(function (o) { var q = rot.apply(o.p); return { i: o.i, q: q }; });
      /* scale the cloud to the canvas: the adapters carry only part of their norm in
         these three axes, so fit the plot to their outermost point rather than to the
         unit sphere, and keep a percentile radius for placing direction markers */
      var rads = P.map(function (o) { return Math.hypot(o.q[0], o.q[1], o.q[2]); })
        .sort(function (x, y) { return x - y; });
      var maxr = rads.length ? rads[rads.length - 1] : 1;
      var p90 = rads.length ? rads[Math.floor(0.9 * (rads.length - 1))] : 1;
      var SCL = R / Math.max(maxr, 1e-6);
      P.sort(function (x, y) { return x.q[2] - y.q[2]; });
      root._hit = [];
      for (var k = 0; k < P.length; k++) {
        var q2 = P[k].q, i = P[k].i, r = D.rows[i];
        var x = cxx + q2[0] * SCL, y = cyy - q2[1] * SCL;
        var dep = (q2[2] / Math.max(maxr, 1e-6) + 1) / 2;
        var rr = (opt.compact ? 2.2 : 2.6) + (opt.compact ? 2.8 : 3.4) * dep;
        var faded = dimmed(r);
        ctx.globalAlpha = (0.36 + 0.64 * dep) * (faded ? 0.16 : 1);
        ctx.fillStyle = colourOf(r);
        ctx.beginPath(); ctx.arc(x, y, rr, 0, 6.2832); ctx.fill();
        if (r.k === '-') {
          ctx.globalAlpha = faded ? 0.16 : 1;
          ctx.strokeStyle = cssv('card'); ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.arc(x, y, rr * 0.44, 0, 6.2832); ctx.stroke();
        }
        root._hit.push({ i: i, x: x, y: y, r: Math.max(rr, 6) });
        ctx.globalAlpha = 1;
      }
      /* the shared component every adapter carries: the grand-mean direction, drawn
         from the (now centred) origin at the 90th-percentile radius */
      if (false) (function () {  /* shared-component marker removed at Samuel's request 2026-09-14 */
        var c = centroid(), cn = Math.hypot(c[0], c[1], c[2]); if (!(cn > 1e-9)) return;
        var qc = rot.apply([c[0] / cn, c[1] / cn, c[2] / cn]);
        var sx = cxx + qc[0] * SCL * p90, sy = cyy - qc[1] * SCL * p90;
        ctx.setLineDash([2, 4]); ctx.strokeStyle = cssv('faint'); ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(cxx, cyy); ctx.lineTo(sx, sy); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = cssv('faint'); ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, 6.2832); ctx.fill();
        ctx.font = '600 10px "IBM Plex Mono", monospace';
        ctx.fillText('shared component', sx + 8, sy + 3);
      })();
      /* the alien direction: no hue, because it is not one of the named things */
      if (false && D.special && D.special.alien_k5) {  /* widest-gap marker removed at Samuel's request 2026-09-14 */
        var u = D.special.alien_k5.u_fa || [];
        var v = [u[axes[0]], u[axes[1]], u[axes[2]]];
        var n2 = Math.hypot(v[0], v[1], v[2]) || 1;
        var q3 = rot.apply([v[0] / n2, v[1] / n2, v[2] / n2]);
        var xx = cxx + q3[0] * SCL * p90, yy = cyy - q3[1] * SCL * p90;
        ctx.setLineDash([3, 3]); ctx.strokeStyle = cssv('ink'); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(cxx, cyy); ctx.lineTo(xx, yy); ctx.stroke();
        ctx.beginPath(); ctx.arc(xx, yy, 11, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
        ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(xx, yy, 4.4, 0, 6.2832); ctx.stroke();
        ctx.fillStyle = cssv('ink');
        ctx.font = '600 10px "IBM Plex Mono", monospace';
        ctx.fillText('widest gap', xx + 15, yy + 3);
      }
      var sel = pinned >= 0 ? pinned : hover;
      if (sel >= 0) {
        var hit = null;
        for (var z = 0; z < root._hit.length; z++) if (root._hit[z].i === sel) hit = root._hit[z];
        if (hit) {
          ctx.strokeStyle = cssv('ink'); ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.arc(hit.x, hit.y, hit.r + 4.5, 0, 6.2832); ctx.stroke();
          ctx.font = '600 12px "IBM Plex Mono", monospace'; ctx.fillStyle = cssv('ink');
          ctx.fillText(D.rows[sel].t, hit.x + hit.r + 8, hit.y + 4);
        }
      }
      var row = sel >= 0 ? D.rows[sel] : null;
      tip.textContent = row
        ? row.t + '  ·  ' + (GROUP_LABEL[row.f] || row.f) + (row.k ? ' ' + row.k : '')
        : 'drag to rotate · hover a point for its trait';
      panelFor(pinned);
    }

    function panelFor(i) {
      if (!pinbody) return;
      if (i < 0) {
        pinbody.innerHTML = '<p class="empty">Hover a point for its trait word. Click one to ' +
          'pin it here with its five loadings and a link to its page; click it again to let go.</p>';
        return;
      }
      var r = D.rows[i];
      var rows = '';
      if (r.l) {
        rows = D.factors.map(function (fa, j) {
          return '<tr><td><span class="swatch bg-f' + j + '" style="display:inline-block;' +
            'margin-right:0.4rem"></span>' + esc(fa.title) + '</td><td class="n">' +
            (r.l[j] >= 0 ? '+' : '−') + Math.abs(r.l[j]).toFixed(3) + '</td></tr>';
        }).join('');
        rows = '<table><tbody>' + rows + '</tbody></table>';
      } else {
        rows = '<p class="empty">Not part of the factored set, so it carries no loadings. ' +
          'Its chart position comes from its cross-Gram with the 134.</p>';
      }
      pinbody.innerHTML = '<span class="nm">' + esc(r.t) + '</span>' +
        '<span class="mt">' + esc(r.set === 'Goldberg' ? r.f + ', keyed ' + r.k : r.set) +
        ' · chart length ' + r.cl.toFixed(3) + '</span>' + rows +
        '<a class="go" href="' + BASE + 'traits/' + encodeURIComponent(r.s) +
        '.html">open the ' + esc(r.t) + ' page →</a>';
    }

    function at(ev) {
      var b = cv.getBoundingClientRect();
      var x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - b.left;
      var y = (ev.touches ? ev.touches[0].clientY : ev.clientY) - b.top;
      var best = -1, bd = 1e9;
      (root._hit || []).forEach(function (hh) {
        var d = Math.hypot(hh.x - x, hh.y - y);
        if (d < hh.r + 5 && d < bd) { bd = d; best = hh.i; }
      });
      return { x: x, y: y, i: best };
    }
    cv.addEventListener('pointerdown', function (ev) {
      cv.setPointerCapture(ev.pointerId);
      var a = at(ev); drag = { x: ev.clientX, y: ev.clientY, moved: false, i: a.i }; spin = false;
    });
    cv.addEventListener('pointermove', function (ev) {
      if (drag) {
        var dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        rot.spinBy(dx * 0.008, dy * 0.008);
        drag.x = ev.clientX; drag.y = ev.clientY; draw();
      } else { var hh = at(ev).i; if (hh !== hover) { hover = hh; draw(); } }
    });
    cv.addEventListener('pointerup', function () {
      if (drag && !drag.moved) pinned = (pinned === drag.i ? -1 : drag.i);
      drag = null; draw();
    });
    cv.addEventListener('pointerleave', function () { if (hover >= 0) { hover = -1; draw(); } });

    $('.chips', root).addEventListener('click', function (ev) {
      var b = ev.target.closest('.chip'); if (!b) return;
      var f = b.dataset.f; show[f] = !show[f];
      b.setAttribute('aria-pressed', show[f] ? 'true' : 'false');
      draw();
    });

    var names = $('.axisnames', root);
    function label() {
      var N = D.axdesc || {};
      names.innerHTML = '';
      ['x', 'y', 'z'].forEach(function (k, i) {
        var t = D.axnames[axes[i]];
        /* the solution's own label is worth printing only where it differs from the
           display title; otherwise the line stutters ("y Competence Competence") */
        var d = N[t] && N[t] !== t ? '  ' + N[t] : '';
        names.appendChild(el('span', null, k + '  ' + t + d));
      });
      var note = $('.mapnote', root);
      if (note && !opt.note) {
        var anyPC = axes.some(function (j) { return j >= NFA; });
        note.textContent = opt.stages
          ? 'The 134 zoo adapters. The four alignment adapters and three hole words are not ' +
            'trained past stage one, so they are not in this chart.'
          : anyPC
            ? 'A component axis is selected, so the seven adapters outside the 134 drop out: ' +
              'they have chart coordinates but no component scores.'
            : '';
      }
    }

    var sx = $('.m-x', root), sy = $('.m-y', root), sz = $('.m-z', root);
    [sx, sy, sz].forEach(function (s) {
      s.addEventListener('change', function () {
        axes = [+sx.value, +sy.value, +sz.value]; label(); draw();
      });
    });
    var sc = $('.m-col', root), ss = $('.m-set', root), sf = $('.m-find', root);
    if (sc) sc.addEventListener('change', function () { colour = sc.value; draw(); });
    if (ss) ss.addEventListener('change', function () { setf = ss.value; draw(); });
    if (sf) sf.addEventListener('input', function () {
      find = sf.value.trim().toLowerCase();
      pinned = -1;
      if (find) {
        for (var i = 0; i < D.rows.length; i++) {
          var r = D.rows[i];
          if (r.s === find || r.t.toLowerCase() === find) { pinned = i; break; }
          if (pinned < 0 && (r.s.indexOf(find) === 0 || r.t.toLowerCase().indexOf(find) === 0)) pinned = i;
        }
      }
      draw();
    });
    var st = $('.m-stage', root);
    if (st) st.addEventListener('change', function () {
      stage = st.value; rebuild(); draw();
    });

    if (opt.focus) {
      for (var i = 0; i < D.rows.length; i++) if (D.rows[i].s === opt.focus) { pinned = i; break; }
      spin = !reduce;
    }

    rebuild();
    label();
    addEventListener('resize', draw);
    /* the canvas takes its colours from the tokens at draw time, so a theme
       change only needs a redraw -- but it has two signals: the data-theme stamp
       when the reader uses the button, and the OS preference when they have not */
    new MutationObserver(draw).observe(document.documentElement,
      { attributes: true, attributeFilter: ['data-theme'] });
    var mq = matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', draw);
    else if (mq.addListener) mq.addListener(draw);

    draw();
    (function loop() {
      if (spin) { rot.spinBy(0.0022, 0); draw(); }
      requestAnimationFrame(loop);
    })();
  }

  /* -------------------------------------------------------------------- go */
  function boot() {
    var hosts = [].slice.call(document.querySelectorAll('[data-map]'));
    if (!hosts.length) return;
    var wantStages = hosts.some(function (h) { return h.dataset.stages === '1'; });
    var src = hosts[0].dataset.src || 'data/chart.json';
    var jobs = [fetch(BASE + src).then(function (r) { return r.json(); })];
    if (wantStages) jobs.push(fetch(BASE + 'data/stages.json')
      .then(function (r) { return r.json(); }).catch(function () { return null; }));
    Promise.all(jobs).then(function (res) {
      var D = res[0], ST = res[1] || null;
      if (ST) D.stages = ST;
      hosts.forEach(function (h) {
        var ax = (h.dataset.axes || '0,1,2').split(',').map(Number);
        makeMap(h, D, {
          compact: h.dataset.compact === '1',
          focus: h.dataset.focus || null,
          stages: h.dataset.stages === '1',
          axes: ax,
          note: h.dataset.note || ''
        });
      });
    }).catch(function (e) {
      hosts.forEach(function (h) {
        h.innerHTML = '<p class="caveat">The chart data did not load: ' + esc(e) + '</p>';
      });
    });
  }
  if (document.readyState !== 'loading') boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
