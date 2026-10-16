const ci = r => r.f1_interval ? `${F.pct(r.f1_interval[0])} to ${F.pct(r.f1_interval[1])}` : '–';
const ms = v => v == null ? '–' : v < 1 ? F.dec(v, 2) + ' ms' : v < 100 ? F.dec(v, 1) + ' ms' : F.int(v) + ' ms';
const pairText = list => list.length ? list.map(p => `${p[0]} → ${p[1]}`).join('; ') : '(none)';
const verdict = (ok, yes, no) => h('span', { class: ok ? 'status' : 'status miss', text: ok ? yes : no });

const TABS = {
  Detection(D) {
    const d = D.detection, S = D.setup, best = d.sizes.at(-1), big = D.prompted_model;
    const methods = [
      { name: 'TF-IDF + logistic regression', kind: 'Classical baseline', ...d.baseline },
      { name: `${S.base_model.split('/')[1]}, no examples`, kind: 'Prompting the small model', ...d.zero_shot },
      { name: `${S.base_model.split('/')[1]}, 6 examples`, kind: 'Prompting the small model', ...d.few_shot },
      ...d.prompted.map(p => ({ name: `${big.model}, ${p.method.toLowerCase()}`, kind: 'Prompting a model 6× larger', ...p })),
      ...d.sizes.map(r => ({ name: `LoRA fine-tuned on ${F.int(r.train_sentences)} sentences`, kind: 'Fine-tuning the small model', ...r })),
    ].sort((a, b) => b.f1 - a.f1);
    const bestPrompt = [...d.prompted, d.few_shot, d.zero_shot].sort((a, b) => b.f1 - a.f1)[0];
    let pick = 'all';
    const list = h('div', { class: 'table-box scroll' });
    const draw = () => {
      const rows = D.detect_examples.filter(e => pick === 'all' || (pick === 'wrong') === (e.label !== e.predicted));
      list.replaceChildren(dataTable([
        { label: 'Sentence', key: 'text' },
        { label: 'Labelled', key: 'label', render: v => v ? 'Adverse event' : 'No event' },
        { label: 'Model', key: 'predicted', render: (v, r) => verdict(v === r.label, v ? 'Adverse event' : 'No event', v ? 'Adverse event' : 'No event') },
        { label: 'Score', key: 'score', num: true, format: v => F.dec(v, 2) },
      ], rows));
    };
    const select = h('select', { id: 'det-filter', onchange: e => { pick = e.target.value; draw(); } }, [['all', 'All sentences'], ['wrong', 'Only mistakes'], ['right', 'Only correct']].map(([v, t]) => h('option', { value: v, text: t })));
    draw();
    return section(
      intro(`Does a sentence from a medical case report describe an adverse drug event? ${F.int(S.detection.sentences)} distinct sentences, ${F.pct(S.detection.positive_share)} of them positive. Every method is scored on the same ${F.int(S.detection.test_used)} held-out sentences; decision thresholds are chosen on separate validation sentences.`),
      tiles([
        ['F1, fine-tuned small model', F.pct(best.f1), `LoRA on ${F.int(best.train_sentences)} sentences; 95% interval ${ci(best)}`],
        ['F1, best prompting result', F.pct(bestPrompt.f1), 'no training, examples in the prompt'],
        ['F1, classical baseline', F.pct(d.baseline.f1), `TF-IDF + logistic regression on ${F.int(d.baseline.train_sentences)} sentences`],
        ['Weights trained', F.pct(best.trainable_share, 2), `${F.int(best.trainable)} of ${F.int(best.total)} parameters`],
      ]),
      grid(
        card({
          title: 'F1 by method', wide: true, sub: 'F1 on the adverse-event class: the balance of how many flagged sentences are right and how many real events are found.',
          table: { columns: [{ label: 'Method', key: 'name' }, { label: 'Approach', key: 'kind' }, pctCol('F1', 'f1'), { label: '95% interval', key: 'f1_interval', num: true, render: (v, r) => ci(r) }, pctCol('Precision', 'precision'), pctCol('Recall', 'recall'), { label: 'Time per sentence', key: 'ms_per_sentence', num: true, format: ms }], rows: methods },
        }, p => hbars(p, { rows: methods.map(r => ({ label: r.name, values: [r.f1], extra: [{ name: 'precision', value: F.pct(r.precision) }, { name: 'recall', value: F.pct(r.recall) }] })), series: [{ name: 'F1', color: C.actual }], format: F.pct, max: 1 })),
        card({
          title: 'How much labelled data does fine-tuning need?', sub: 'F1 against the number of training sentences, rank 8.', wide: d.ranks.length < 2,
          table: { columns: [{ label: 'Training sentences', key: 'train_sentences', num: true, format: F.int }, pctCol('F1', 'f1'), pctCol('ROC AUC', 'roc_auc'), { label: 'Training time', key: 'training', num: true, render: v => F.int(v.seconds) + ' s' }], rows: d.sizes },
          notes: [`For reference: the classical baseline reaches ${F.pct(d.baseline.f1)} with all ${F.int(d.baseline.train_sentences)} sentences, and the same small model without training reaches ${F.pct(Math.max(d.zero_shot.f1, d.few_shot.f1))}.`],
        }, p => columnChart(p, { labels: d.sizes.map(r => F.int(r.train_sentences)), values: d.sizes.map(r => r.f1), name: 'F1', yFormat: F.pct0, label: 'F1 by training-set size' })),
        d.ranks.length < 2 ? null : card({
          title: 'How big does the adapter need to be?', sub: `LoRA rank is the width of the trained update. ${F.int(d.ranks[0].train_sentences)} training sentences each.`,
          table: { columns: [{ label: 'Rank', key: 'rank', num: true }, { label: 'Trained parameters', key: 'trainable', num: true, format: F.int }, { label: 'Share of the model', key: 'trainable_share', num: true, format: v => F.pct(v, 2) }, pctCol('F1', 'f1')], rows: d.ranks },
        }, p => columnChart(p, { labels: d.ranks.map(r => 'rank ' + r.rank), values: d.ranks.map(r => r.f1), name: 'F1', yFormat: F.pct0, label: 'F1 by LoRA rank' })),
        !D.detect_examples.length ? null : h('figure', { class: 'card wide' },
          h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: 'Try it: new sentences and what the fine-tuned model said' }), h('p', { class: 'sub', text: `${D.detect_examples.length} sentences written for this page, not taken from the corpus. The score is the model's preference for "yes" over "no"; above ${F.dec(best.threshold, 2)} it flags an event.` }))),
          h('div', { class: 'control' }, h('label', { for: 'det-filter', text: 'Show' }), select), list)));
  },

  Extraction(D) {
    const x = D.extraction, S = D.setup, big = D.prompted_model;
    const methods = [
      { name: `${S.base_model.split('/')[1]}, 4 examples in the prompt`, ...x.few_shot },
      ...x.prompted.map(p => ({ name: `${big.model}, ${p.method.toLowerCase()}`, ...p })),
      { name: `${S.base_model.split('/')[1]}, LoRA fine-tuned on ${F.int(x.lora.train_sentences)} sentences`, ...x.lora },
    ].sort((a, b) => b.f1 - a.f1);
    const bestPrompt = methods.filter(m => !m.name.includes('LoRA'))[0];
    const hasBig = x.prompted.length > 0;
    const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
    return section(
      intro(`The harder task: write out every (drug, adverse effect) pair in a sentence. ${F.int(S.extraction.sentences)} sentences with ${F.dec(S.extraction.pairs_per_sentence, 2)} pairs each on average; scored on ${S.extraction.test_used} held-out sentences. A pair counts only if both the drug and the effect match the annotation exactly.`),
      tiles([
        ['Pair F1, fine-tuned small model', F.pct(x.lora.f1), `${F.pct(x.lora.sentences_fully_right)} of sentences fully right`],
        ['Pair F1, best prompting result', F.pct(bestPrompt.f1), bestPrompt.name],
        ['Time per sentence, fine-tuned', ms(x.lora.ms_per_sentence), hasBig ? `larger model: ${ms(x.prompted.at(-1).ms_per_sentence)}` : 'batched generation on a laptop'],
        ['Training time', F.int(x.lora.training.seconds / 60) + ' min', `${x.lora.training.steps} steps on a laptop (${S.device})`],
      ]),
      grid(
        card({
          title: 'Pair F1 by method', wide: true,
          table: { columns: [{ label: 'Method', key: 'name' }, pctCol('F1', 'f1'), pctCol('Precision', 'precision'), pctCol('Recall', 'recall'), pctCol('Sentences fully right', 'sentences_fully_right'), { label: 'Time per sentence', key: 'ms_per_sentence', num: true, format: ms }], rows: methods },
          notes: ['Exact matching is strict: "skin rash" against an annotation of "rash" counts as a miss for both precision and recall. Prompted models often return a correct but differently worded span; fine-tuning teaches the annotation style as well as the task.'],
        }, p => hbars(p, { rows: methods.map(r => ({ label: r.name, values: [r.f1] })), series: [{ name: 'Pair F1', color: C.actual }], format: F.pct, max: 1 })),
        !D.extract_examples.length ? null : card({
          title: 'Try it: new sentences', wide: true, sub: `${D.extract_examples.length} sentences written for this page, not taken from the corpus, with the pairs a reader would mark and each model's output.`,
          table: { columns: [
            { label: 'Sentence', key: 'text' },
            { label: 'Expected', key: 'expected', render: v => pairText(v) },
            { label: 'Fine-tuned small model', key: 'tuned', render: (v, r) => h('span', {}, verdict(same(v, r.expected), 'Match', 'Differs'), h('br'), pairText(v)) },
            ...(hasBig && D.extract_examples.every(e => e.prompted) ? [{ label: `${big.model}, prompted`, key: 'prompted', render: (v, r) => h('span', {}, verdict(same(v, r.expected), 'Match', 'Differs'), h('br'), pairText(v)) }] : []),
          ], rows: D.extract_examples },
        })));
  },

  'Real-world signals'(D) {
    const R = D.faers;
    if (!R) return section(intro('Real-world results not generated yet: run faers_signals.py.'));
    const S = R.summary, src = R.source;
    const sets = [['model', 'Fine-tuned model'], ['human', 'Human annotators'], ['control', 'Shuffled pairs (control)']];
    const rows = sets.map(([k, name]) => ({ name, ...S[k] }));
    const pairCols = [{ label: 'Drug', key: 'drug' }, { label: 'Effect (as extracted)', key: 'effect' }, { label: 'FAERS term', key: 'term' },
      { label: 'Reports', key: 'a', num: true, format: F.int }, { label: 'PRR', key: 'prr', num: true, format: v => F.dec(v, 1) },
      { label: 'PRR, lower 95%', key: 'prr_low', num: true, format: v => F.dec(v, 1) }];
    return section(
      intro(`Do the (drug, adverse effect) pairs the fine-tuned model writes out match what happens in the real world? Each pair from ${src.sentences} held-out case-report sentences is checked against ${F.int(src.faers_reports)} FDA Adverse Event Reporting System (FAERS) reports, the post-market data drug-safety teams monitor, and against the drug's FDA label. A pair is a signal when the effect is reported with the drug disproportionately often (PRR ≥ 2, χ² ≥ 4, at least 3 reports). The same check runs on the human annotations and on shuffled pairs nobody wrote.`),
      tiles([
        ['Model pairs confirmed in FAERS', F.pct(S.model.signal_share), `${S.model.signal} of ${S.model.checked} checked pairs are signals`],
        ['Same check, human annotations', F.pct(S.human.signal_share), `${S.human.signal} of ${S.human.checked}`],
        ['Same check, shuffled pairs', F.pct(S.control.signal_share), `${S.control.signal} of ${S.control.checked}: what chance looks like`],
        ['Signals not on the FDA label', F.int(S.model.candidate), 'model pairs a safety reviewer would look at first'],
      ]),
      grid(
        card({
          title: 'Model output against real-world data', wide: true, sub: 'Share of checked pairs that are FAERS signals, and share whose effect is written in the drug\'s FDA label.',
          legend: [{ name: 'FAERS signal', color: C.actual, shape: 'box' }, { name: 'On the FDA label', color: C.model, shape: 'box' }],
          table: { columns: [{ label: 'Pairs', key: 'name' }, { label: 'Pairs', key: 'pairs', num: true }, { label: 'Checked', key: 'checked', num: true },
            pctCol('FAERS signal', 'signal_share'), pctCol('On FDA label', 'on_label_share'), { label: 'Median PRR', key: 'median_prr', num: true, format: v => F.dec(v, 1) },
            { label: 'Known', key: 'known', num: true }, { label: 'Signal, not on label', key: 'candidate', num: true }], rows },
          notes: ['Checked = the drug is in FAERS and the effect maps to a MedDRA term FAERS uses. Effects are mapped to MedDRA terms with the same small normalisation for every set (spelling, plural, modifiers such as "severe", everyday words such as fever → pyrexia); pairs that still match no FAERS term (for example "increased affect") are left out.'],
        }, p => hbars(p, { rows: rows.map(r => ({ label: r.name, values: [r.signal_share, r.on_label_share] })),
          series: [{ name: 'FAERS signal', color: C.actual }, { name: 'On the FDA label', color: C.model }], format: F.pct, max: 1 })),
        card({ title: 'Signals not yet on the FDA label', sub: 'Model pairs with a FAERS signal whose effect is not written in the label: the queue a safety reviewer would start with.', wide: true,
          table: { columns: pairCols, rows: R.examples.candidate },
          notes: ['A signal is a statistical association, not proof of cause. Some rows are wording gaps (the gentamicin label says nephrotoxicity, not renal failure) and some are confounding by indication (furosemide is given for heart failure, so heart failure is reported with it). Telling these apart is the reviewer\'s job.'] }),
        card({ title: 'Known effects the model recovered', sub: 'Model pairs that are both FAERS signals and on the label, strongest first.', wide: true,
          table: { columns: pairCols, rows: R.examples.known },
          notes: [`FAERS data as of ${src.faers_last_updated}, via openFDA. Reports are voluntary and unverified; counts reflect reporting, not incidence. A drug is matched by generic or brand name in any role on a report.`] })));
  },

  Method(D) {
    const S = D.setup, best = D.detection.sizes.at(-1);
    return section(
      intro('When is it worth fine-tuning a small model instead of prompting a large one? This project measures both on a real pharmacovigilance task, on a laptop.'),
      grid(
        card({
          title: 'Set-up', wide: true,
          table: { columns: [{ label: 'Item', key: 'k' }, { label: 'Detail', key: 'v' }], rows: [
            { k: 'Data', v: `ADE Corpus V2 (Gurulingappa et al., 2012): sentences from MEDLINE case reports, annotated for adverse drug events and for (drug, effect) pairs.` },
            { k: 'Cleaning', v: `Repeated sentences are merged before splitting (${F.int(S.detection.sentences)} distinct sentences), so no sentence is in both training and test.` },
            { k: 'Split', v: 'Train 70%, validation 10%, test 20%, decided by a hash of the sentence so it never depends on row order.' },
            { k: 'Base model', v: `${S.base_model}: an open-source chat model with ${F.int(best.total - best.trainable)} parameters, frozen.` },
            { k: 'LoRA', v: `Low-rank adapters on the attention projections (${S.lora_targets.join(', ')}). At rank ${best.rank}, ${F.int(best.trainable)} parameters are trained: ${F.pct(best.trainable_share, 2)} of the model.` },
            { k: 'Training', v: `${S.epochs} epochs, AdamW, warm-up and linear decay, gradient clipping; the loss is computed on the answer tokens only.` },
            { k: 'Read-out', v: 'Detection compares the logits of "yes" and "no" in one forward pass. Extraction generates "drug | effect" lines with greedy decoding.' },
            { k: 'Comparison', v: 'The same instructions are given to every model: the fine-tuned one, the untrained small one, and a model six times larger, with and without examples in the prompt.' },
          ] },
          notes: [
            'Sentences from the same case report can fall on both sides of the split, because the public corpus carries no document ids. That favours every trained method, the baseline included.',
            'One random seed and one base model. The 95% intervals show how much the test sample alone moves the score.',
            'This is a research exercise on public literature. It is not a medical device and makes no clinical claims.',
          ],
        })));
  },
};
const pctCol = (label, key) => ({ label, key, num: true, format: F.pct });

(async function main() {
  const D = await (await fetch('data.json')).json();
  document.getElementById('lede').textContent =
    `A ${D.setup.base_model.split('/')[1]} language model fine-tuned with LoRA to find adverse drug events in medical case reports, compared with prompting and with a classical baseline. Trained and measured on a laptop.`;
  document.getElementById('foot').textContent = 'Data: ADE Corpus V2 (Gurulingappa et al., 2012), sentences from MEDLINE case reports. Research exercise, not medical advice. Built by S Harshni.';
  const mainEl = document.getElementById('main'), nav = document.getElementById('tabs');
  const names = Object.keys(TABS), built = {};
  const slug = n => n.toLowerCase().replace(/\s+/g, '-');
  function open(name) {
    hideTip();
    for (const b of nav.children) b.setAttribute('aria-selected', String(b.textContent === name));
    for (const el of mainEl.children) el.hidden = true;
    built[name] ??= mainEl.appendChild(TABS[name](D));
    built[name].hidden = false;
    history.replaceState(null, '', '#' + slug(name));
    renderAll();
  }
  nav.append(...names.map(n => h('button', { type: 'button', role: 'tab', text: n, onclick: () => open(n) })));
  open(names.find(n => '#' + slug(n) === location.hash) || names[0]);
  let timer;
  addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  const btn = document.getElementById('theme');
  const dark = () => (document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')) === 'dark';
  btn.textContent = dark() ? 'Light mode' : 'Dark mode';
  btn.addEventListener('click', () => { document.documentElement.dataset.theme = dark() ? 'light' : 'dark'; btn.textContent = dark() ? 'Light mode' : 'Dark mode'; });
})();
