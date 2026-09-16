/* The stage-two shared direction: the same prompt answered at every dose by the
   shared direction and by the sign-balanced control. Data from /data/stage2.json. */
(function () {
  "use strict";
  var BASE = (document.querySelector('link[href$="assets/site.css"]').getAttribute('href') || '')
    .replace(/assets\/site\.css$/, '');
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }

  var D = null, ALPHAS = [], AKEYS = [], aIn, aLab, pSel, treat, ctrl;

  function render() {
    var i = +aIn.value, a = ALPHAS[i], key = AKEYS[i];
    var pi = +pSel.value;
    aLab.textContent = 'alpha = ' + a;
    [['S2_mean', treat], ['S2_balancedrandom', ctrl]].forEach(function (pair) {
      var job = D.gens[pair[0]], host = pair[1];
      if (!job) { host.innerHTML = '<p class="small">not available</p>'; return; }
      var g = (job.generations[key] || [])[pi];
      var st = D.stats && D.stats[pair[0]] && D.stats[pair[0]].per_alpha[key];
      var meta = st ? '<p class="small" style="color:var(--muted);margin:0 0 0.5rem">' +
        'first person ' + st.first_person_per_k.toFixed(1) + '/1k &middot; second person ' +
        st.second_person_per_k.toFixed(1) + '/1k &middot; markdown ' +
        (st.markdown_frac * 100).toFixed(0) + '% &middot; in character ' +
        (st.embodied_frac * 100).toFixed(0) + '% &middot; ' + st.words.toFixed(0) + ' words' +
        '</p>' : '';
      host.innerHTML = meta + '<div class="gen"><span class="q">' +
        esc(job.prompts[pi] || '') + '</span>' + esc(String(g || '(no generation)').trim()) + '</div>';
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    aIn = document.getElementById('s2-alpha');
    if (!aIn) return;
    aLab = document.getElementById('s2-alpha-label');
    pSel = document.getElementById('s2-prompt');
    treat = document.getElementById('s2-treat');
    ctrl = document.getElementById('s2-ctrl');
    fetch(BASE + 'data/stage2.json').then(function (r) { return r.json(); }).then(function (d) {
      D = d;
      var job = d.gens.S2_mean;
      AKEYS = (job.alpha_keys || Object.keys(job.generations))
        .slice().sort(function (a, b) { return parseFloat(a) - parseFloat(b); });
      ALPHAS = AKEYS.map(parseFloat);
      aIn.max = ALPHAS.length - 1;
      aIn.value = Math.max(0, ALPHAS.indexOf(0));
      job.prompts.forEach(function (p, i) {
        var o = document.createElement('option');
        o.value = i; o.textContent = (i + 1) + '. ' + p.slice(0, 52) + (p.length > 52 ? '…' : '');
        pSel.appendChild(o);
      });
      aIn.addEventListener('input', render);
      pSel.addEventListener('change', render);
      render();
    }).catch(function (e) {
      treat.innerHTML = '<p class="caveat">The stage-two data did not load: ' + esc(e) + '</p>';
    });
  });
})();
