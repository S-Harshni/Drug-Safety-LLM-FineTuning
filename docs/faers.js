// In-page version of src/drugsafety/faers.py: the same term rules (read from data.json, written from the Python module)
// and the same disproportionality maths, querying openFDA live from the visitor's browser.
const FAERS = (() => {
  const API = 'https://api.fda.gov/drug';
  const REACTIONS = 'patient.reaction.reactionmeddrapt.exact';
  let R = null;
  const cache = new Map();

  async function get(endpoint, params) {
    const q = new URLSearchParams(params).toString();
    const key = endpoint + '?' + q;
    if (!cache.has(key)) {
      cache.set(key, fetch(`${API}/${endpoint}.json?${q}`).then(async r => {
        if (r.status === 404) return null;
        if (r.status === 429) throw new Error('openFDA rate limit reached for today from this network. Try again later.');
        if (!r.ok) throw new Error(`openFDA error ${r.status}`);
        return r.json();
      }));
    }
    return cache.get(key);
  }
  const total = async search => { const got = await get('event', { search, limit: 1 }); return got ? got.meta.results.total : 0; };

  const clean = d => { const x = d.trim().toLowerCase().replace(/["\\]/g, ''); return R.aliases[x] || x; };
  const drugQuery = d => `(patient.drug.openfda.generic_name:"${clean(d)}" OR patient.drug.openfda.brand_name:"${clean(d)}")`;
  const reportedQuery = d => `patient.drug.medicinalproduct:"${clean(d)}"`;

  function toPt(effect) {
    let e = effect.trim().toLowerCase().replace(/\s+/g, ' ');
    for (const [us, uk] of R.uk) e = e.split(us).join(uk);
    e = e.replace(/(?<!a)emia\b/g, 'aemia');
    return e.toUpperCase();
  }
  function candidates(effect) {
    const pt = toPt(effect);
    const forms = [pt];
    let stripped = pt.replace(/^[A-Z0-9-]+-INDUCED /, '');
    for (let changed = true; changed;) {
      changed = false;
      for (const m of R.modifiers) if (stripped.startsWith(m)) { stripped = stripped.slice(m.length); changed = true; }
    }
    forms.push(stripped);
    for (const f of [...forms]) {
      if (f.endsWith('ES') && !f.endsWith('SES')) forms.push(f.slice(0, -1), f.slice(0, -2));
      else if (f.endsWith('S') && !/(SS|SIS|US)$/.test(f)) forms.push(f.slice(0, -1));
    }
    for (const f of [...forms]) if (R.synonyms[f]) forms.push(R.synonyms[f]);
    return [...new Set(forms)];
  }

  function prr(a, drugTotal, reactionTotal, n) {
    const b = drugTotal - a, c = reactionTotal - a, d = n - a - b - c;
    if (a === 0 || Math.min(b, c, d) <= 0) return { a, prr: 0, prr_low: 0, chi2: 0, signal: false };
    const ratio = (a / (a + b)) / (c / (c + d));
    const se = Math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d));
    const expected = (a + b) * (a + c) / n;
    const chi2 = expected > 0 ? (Math.abs(a - expected) - 0.5) ** 2 / expected : 0;
    return { a, prr: ratio, prr_low: ratio * Math.exp(-1.96 * se), chi2, signal: a >= 3 && ratio >= 2 && chi2 >= 4 };
  }

  const escape = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  function onLabel(text, effect, pt) {
    if (!text) return null;
    const forms = new Set([effect.toLowerCase().trim(), toPt(effect).toLowerCase()]);
    if (pt) forms.add(pt.toLowerCase());
    for (const [us, uk] of R.uk) for (const f of [...forms]) forms.add(f.split(uk).join(us));
    return [...forms].some(f => f.length >= 4 && new RegExp('\\b' + escape(f) + '\\b').test(text));
  }

  async function labelText(drug) {
    const d = clean(drug);
    const got = await get('label', { search: `(openfda.generic_name:"${d}" OR openfda.brand_name:"${d}")`, limit: 1 });
    if (!got || !got.results || !got.results.length) return null;
    const r = got.results[0];
    return R.label_sections.map(k => (r[k] || []).join(' ')).join(' ').toLowerCase();
  }

  async function reactionTotal(pt) {
    if (R.reaction_totals[pt] === undefined) R.reaction_totals[pt] = await total(`${REACTIONS}:"${pt}"`);
    return R.reaction_totals[pt];
  }

  // Everything about one drug: its report count, most reported reactions scored for disproportionality, the label.
  async function checkDrug(drug, top = 25) {
    const n = (await get('event', { limit: 1 })).meta.results.total;
    let query = drugQuery(drug), drugTotal = await total(query);
    if (!drugTotal) { query = reportedQuery(drug); drugTotal = await total(query); }
    if (!drugTotal) return { drug, found: false };
    const [counts, label] = await Promise.all([get('event', { search: query, count: REACTIONS, limit: 100 }), labelText(drug)]);
    const list = ((counts && counts.results) || []).slice(0, top);
    const rows = await Promise.all(list.map(async r => {
      const s = prr(r.count, drugTotal, await reactionTotal(r.term), n);
      return { term: r.term, ...s, on_label: onLabel(label, r.term.toLowerCase(), r.term) };
    }));
    return { drug, found: true, query, n, drugTotal, label, rows, counts: (counts && counts.results) || [] };
  }

  // One drug and one effect, as written by a person: map the words to a FAERS term, then score.
  async function checkPair(info, effect) {
    const options = candidates(effect);
    const listed = new Map(info.counts.map(r => [r.term, r.count]));
    let pt = options.find(o => listed.has(o) || R.reaction_totals[o]);
    if (!pt) for (const o of options) { if (await reactionTotal(o)) { pt = o; break; } }
    if (!pt) return { effect, tried: options, matched: false };
    const a = listed.has(pt) ? listed.get(pt) : await total(`${info.query} AND ${REACTIONS}:"${pt}"`);
    const s = prr(a, info.drugTotal, await reactionTotal(pt), info.n);
    return { effect, term: pt, matched: true, ...s, on_label: onLabel(info.label, effect, pt) };
  }

  return { init: rules => { R = JSON.parse(JSON.stringify(rules)); }, checkDrug, checkPair, candidates, prr, onLabel, toPt };
})();
if (typeof module !== 'undefined') module.exports = FAERS;
