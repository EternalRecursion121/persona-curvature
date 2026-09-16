/* The trait index: search, filter and sort over /data/traits.json. */
(function () {
  "use strict";
  var BASE = (document.querySelector('link[href$="assets/site.css"]').getAttribute('href') || '')
    .replace(/assets\/site\.css$/, '');
  var TITLES = ['Warmth', 'Competence', 'Timidity', 'Arousal', 'Imagination'];
  var ROWS = [], host, q, ff, fk, fs, so, count;

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  function mark(r) {
    var col = (r.lf === null || r.lf === undefined) ? 'var(--f-none)' : 'var(--f' + r.lf + ')';
    if (r.set === 'Lexicon') return '<span class="swatch sq" style="background:' + col + '"></span>';
    if (r.set !== 'Goldberg') return '<span class="swatch hollow" style="border-color:' + col + '"></span>';
    if (r.k === '-') return '<span class="swatch hollow" style="border-color:' + col + '"></span>';
    return '<span class="swatch" style="background:' + col + '"></span>';
  }

  function render() {
    var term = q.value.trim().toLowerCase();
    var rows = ROWS.filter(function (r) {
      if (term && r.s.indexOf(term) < 0 && r.t.toLowerCase().indexOf(term) < 0) return false;
      if (ff.value !== '' && String(r.lf) !== ff.value) return false;
      if (fk.value === '+' && r.k !== '+') return false;
      if (fk.value === '-' && r.k !== '-') return false;
      if (fk.value === 'none' && r.k) return false;
      if (fs.value && r.set !== fs.value) return false;
      return true;
    });
    if (so.value === 'loading') rows.sort(function (a, b) { return Math.abs(b.ll || 0) - Math.abs(a.ll || 0); });
    else if (so.value === 'factor') rows.sort(function (a, b) {
      return (a.lf === b.lf) ? (b.ll || 0) - (a.ll || 0) : (a.lf == null ? 9 : a.lf) - (b.lf == null ? 9 : b.lf);
    });
    else rows.sort(function (a, b) { return a.t.localeCompare(b.t); });

    count.textContent = rows.length + ' of ' + ROWS.length + ' adapters';
    host.innerHTML = rows.map(function (r) {
      var meta = (r.lf === null || r.lf === undefined)
        ? r.set.toLowerCase()
        : TITLES[r.lf] + ' ' + (r.ll >= 0 ? '+' : '−') + Math.abs(r.ll).toFixed(2);
      return '<a class="tcard" href="' + BASE + 'traits/' + r.s + '.html">' +
        '<span class="nm">' + mark(r) + esc(r.t) + '</span>' +
        '<span class="mt">' + esc(meta) + (r.judged ? '' : ' · not judged') + '</span></a>';
    }).join('') || '<div class="tcard" style="grid-column:1/-1;color:var(--muted)">Nothing matches those filters.</div>';
  }

  document.addEventListener('DOMContentLoaded', function () {
    host = document.getElementById('trait-list');
    if (!host) return;
    q = document.getElementById('t-q'); ff = document.getElementById('t-factor');
    fk = document.getElementById('t-key'); fs = document.getElementById('t-set');
    so = document.getElementById('t-sort'); count = document.getElementById('t-count');
    fetch(BASE + 'data/traits.json').then(function (r) { return r.json(); }).then(function (d) {
      ROWS = d.traits;
      [q, ff, fk, fs, so].forEach(function (e) {
        e.addEventListener('input', render); e.addEventListener('change', render);
      });
      render();
    }).catch(function (e) {
      host.innerHTML = '<p class="caveat">The trait index did not load: ' + esc(e) + '</p>';
    });
  });
})();
