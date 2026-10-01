'use strict';

// ---------- small DOM helpers (all text goes through textContent / append, never innerHTML) ----------
const NS = 'http://www.w3.org/2000/svg';

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null || v === '') continue;
    if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  e.append(...kids.flat().filter(k => k != null && k !== ''));
  return e;
}

function s(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  if (parent) parent.append(e);
  return e;
}

// ---------- formatting ----------
const nf = new Intl.NumberFormat('en-GB');
const compact = v => {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e8 ? 0 : a >= 1e7 ? 1 : 2) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(a >= 1e5 ? 0 : 1) + 'K';
  return nf.format(Math.round(v));
};
const day = d => new Date(d + 'T00:00:00Z');
const F = {
  int: v => nf.format(Math.round(v)),
  gbp: v => '£' + compact(v),
  gbpFull: v => '£' + nf.format(Math.round(v)),
  pct: (v, d = 1) => (v * 100).toFixed(d) + '%',
  pct0: v => Math.round(v * 100) + '%',
  dec: (v, d = 2) => v.toFixed(d),
  month: d => day(d).toLocaleDateString('en-GB', { month: 'short', year: 'numeric', timeZone: 'UTC' }),
  monthShort: d => day(d).toLocaleDateString('en-GB', { month: 'short', year: '2-digit', timeZone: 'UTC' }),
  week: d => day(d).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }),
};
const C = { actual: 'var(--series-1)', model: 'var(--series-2)', baseline: 'var(--series-3)' };

function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
function ticksFor(min, max, n = 4) {
  const step = niceStep((max - min) / n || 1);
  const out = [];
  for (let v = Math.floor(min / step) * step; v < max + step * 0.999; v += step) out.push(+v.toPrecision(12));
  return out;
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---------- one tooltip for every chart ----------
const tip = h('div', { class: 'tip', hidden: true });
document.body.append(tip);
function showTip(cx, cy, title, rows) {
  tip.replaceChildren(
    h('div', { class: 'tip-title', text: title }),
    ...rows.filter(Boolean).map(r => h('div', { class: 'tip-row' },
      r.color ? h('i', { style: `background:${r.color}` }) : null, h('strong', { text: r.value }), h('span', { text: r.name }))));
  tip.hidden = false;
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  tip.style.left = clamp(cx + 14, 8, innerWidth - w - 8) + 'px';
  tip.style.top = (cy + 16 + hh > innerHeight ? cy - hh - 12 : cy + 16) + 'px';
}
const hideTip = () => { tip.hidden = true; };

// ---------- building blocks ----------
const renders = [];
const renderAll = () => renders.forEach(r => r());

function dataTable(columns, rows, best) {
  return h('table', {},
    h('thead', {}, h('tr', {}, columns.map(c => h('th', { class: c.num ? 'num' : '', text: c.label })))),
    h('tbody', {}, rows.map(r => h('tr', { class: best && best(r) ? 'best' : '' }, columns.map(c => {
      const v = r[c.key];
      const out = c.render ? c.render(v, r) : v == null ? '–' : c.format ? c.format(v) : v;
      return h('td', { class: c.num ? 'num' : '' }, out instanceof Node ? out : String(out));
    })))));
}

function tiles(items) {
  return h('div', { class: 'tiles' }, items.map(([label, value, note]) => h('div', { class: 'tile' },
    h('div', { class: 'label', text: label }), h('div', { class: 'value', text: value }), note ? h('div', { class: 'note', text: note }) : null)));
}

// A chart card: title, optional legend, the plot, and a table-view twin of the same data.
function card({ title, sub, legend, table, wide, notes }, draw) {
  const plot = h('div', { class: 'plot' });
  const tableBox = h('div', { class: 'table-box', hidden: true });
  const legendEl = legend ? h('div', { class: 'legend' }, legend.map(l => h('span', {}, h('i', { class: l.shape, style: `background:${l.color}` }), l.name))) : null;
  const paint = () => { if (draw && plot.isConnected && !plot.hidden && plot.clientWidth) draw(plot); };
  const btn = table && draw ? h('button', {
    class: 'ghost', type: 'button', text: 'Table', 'aria-pressed': 'false',
    onclick: () => {
      const on = tableBox.hidden;
      tableBox.replaceChildren(dataTable(table.columns, typeof table.rows === 'function' ? table.rows() : table.rows));
      tableBox.hidden = !on; plot.hidden = on;
      if (legendEl) legendEl.hidden = on;
      btn.textContent = on ? 'Chart' : 'Table';
      btn.setAttribute('aria-pressed', String(on));
      paint();
    },
  }) : null;
  if (table && !draw) { tableBox.hidden = false; tableBox.append(dataTable(table.columns, table.rows, table.best)); }
  renders.push(paint);
  const fig = h('figure', { class: 'card' + (wide ? ' wide' : '') },
    h('div', { class: 'card-head' }, h('div', {}, h('h3', { text: title }), sub ? h('p', { class: 'sub', text: sub }) : null), btn),
    legendEl, plot, tableBox, notes ? h('ul', { class: 'note-list' }, notes.map(n => h('li', { text: n }))) : null);
  fig.repaint = paint;
  return fig;
}

function frame(plot, height, label) {
  const W = plot.clientWidth;
  plot.replaceChildren();
  return { W, root: s('svg', { viewBox: `0 0 ${W} ${height}`, width: W, height, role: 'img', 'aria-label': label, tabindex: 0 }, plot) };
}

function yAxis(root, ticks, y, m, W, format) {
  for (const t of ticks) {
    s('line', { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t === ticks[0] ? 'axis-line' : 'grid-line' }, root);
    s('text', { x: m.l - 8, y: y(t) + 4, class: 'tick', 'text-anchor': 'end' }, root, format(t));
  }
}

// Lines over an ordered x (dates). Crosshair snaps to the nearest x and lists every series.
function lineChart(plot, { labels, series, yFormat = F.int, xFormat = x => x, tipTitle = xFormat, height = 270, area = false, mark, label }) {
  const { W, root } = frame(plot, height, label);
  const m = { l: 54, r: 14, t: 14, b: 28 };
  const ticks = ticksFor(0, Math.max(...series.flatMap(d => d.values.filter(v => v != null))), 4);
  const last = labels.length - 1, innerW = W - m.l - m.r;
  const x = i => m.l + (last ? i * innerW / last : innerW / 2);
  const y = v => m.t + (height - m.t - m.b) * (1 - v / ticks.at(-1));
  yAxis(root, ticks, y, m, W, yFormat);
  const every = Math.ceil(labels.length / Math.max(2, Math.floor(innerW / 76)));
  for (let i = 0; i <= last; i += every) {
    s('text', { x: x(i), y: height - 8, class: 'tick', 'text-anchor': i === 0 ? 'start' : 'middle' }, root, xFormat(labels[i]));
  }
  if (mark) {
    s('line', { x1: x(mark.index), x2: x(mark.index), y1: m.t, y2: height - m.b, class: 'axis-line' }, root);
    s('text', { x: x(mark.index) + 6, y: m.t + 4, class: 'axis-title' }, root, mark.label);
  }
  for (const d of series) {
    let path = '', pen = false, first = null, final = null;
    d.values.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      path += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true; first ??= i; final = i;
    });
    if (area && first != null) s('path', { d: `${path}L${x(final)},${y(0)}L${x(first)},${y(0)}Z`, class: 'area', style: `fill:${d.color}` }, root);
    s('path', { d: path, class: 'line', style: `stroke:${d.color}` }, root);
  }
  const cross = s('line', { y1: m.t, y2: height - m.b, class: 'cross', visibility: 'hidden' }, root);
  const dots = series.map(d => s('circle', { r: 4.5, class: 'ring', style: `fill:${d.color}`, visibility: 'hidden' }, root));
  let at = -1;
  const show = (i, cx, cy) => {
    at = i;
    cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
    series.forEach((d, k) => {
      const v = d.values[i];
      dots[k].setAttribute('visibility', v == null ? 'hidden' : 'visible');
      if (v != null) { dots[k].setAttribute('cx', x(i)); dots[k].setAttribute('cy', y(v)); }
    });
    showTip(cx, cy, tipTitle(labels[i]), series.map(d => d.values[i] == null ? null : { color: d.color, name: d.name, value: yFormat(d.values[i]) }));
  };
  const hide = () => { cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); hideTip(); };
  const overlay = s('rect', { x: m.l, y: m.t, width: innerW, height: height - m.t - m.b, fill: 'transparent' }, root);
  overlay.addEventListener('pointermove', e => {
    const px = e.clientX - root.getBoundingClientRect().left;
    show(clamp(Math.round((px - m.l) / (innerW / (last || 1))), 0, last), e.clientX, e.clientY);
  });
  overlay.addEventListener('pointerleave', hide);
  root.addEventListener('keydown', e => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const i = clamp((at < 0 ? last : at) + (e.key === 'ArrowRight' ? 1 : -1), 0, last);
    const box = root.getBoundingClientRect();
    show(i, box.left + x(i), box.top + m.t);
  });
  root.addEventListener('blur', hide);
}

// Columns from one baseline, thin, rounded at the data end. Optional markers for a second measure on the same scale.
function columnChart(plot, { labels, values, name, yFormat = F.int, height = 250, markers, label }) {
  const { W, root } = frame(plot, height, label);
  const m = { l: 50, r: 10, t: 18, b: 28 };
  const ticks = ticksFor(0, Math.max(...values, ...(markers ? markers.values : [])), 4);
  const band = (W - m.l - m.r) / labels.length, bw = Math.min(24, band * 0.6);
  const y = v => m.t + (height - m.t - m.b) * (1 - v / ticks.at(-1));
  yAxis(root, ticks, y, m, W, yFormat);
  labels.forEach((lab, i) => {
    const cx = m.l + band * (i + 0.5), top = y(values[i]), base = y(0), r = Math.min(4, bw / 2, base - top);
    const bar = s('path', {
      class: 'col', style: `fill:${C.actual}`,
      d: `M${cx - bw / 2},${base}V${top + r}Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top}H${cx + bw / 2 - r}Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r}V${base}Z`,
    }, root);
    if (labels.length <= 7) s('text', { x: cx, y: top - 5, class: 'cap', 'text-anchor': 'middle' }, root, yFormat(values[i]));
    s('text', { x: cx, y: height - 8, class: 'tick', 'text-anchor': 'middle' }, root, lab);
    if (markers) s('circle', { cx, cy: y(markers.values[i]), r: 5, class: 'ring', style: `fill:${markers.color}` }, root);
    const hit = s('rect', { x: m.l + band * i, y: m.t, width: band, height: height - m.t - m.b, fill: 'transparent', tabindex: 0 }, root);
    const rows = [{ color: C.actual, name, value: yFormat(values[i]) }, markers ? { color: markers.color, name: markers.name, value: yFormat(markers.values[i]) } : null];
    const on = (cx2, cy2) => { bar.classList.add('on'); showTip(cx2, cy2, lab, rows); };
    hit.addEventListener('pointermove', e => on(e.clientX, e.clientY));
    hit.addEventListener('focus', () => { const b = hit.getBoundingClientRect(); on(b.left + b.width / 2, b.top + 20); });
    for (const ev of ['pointerleave', 'blur']) hit.addEventListener(ev, () => { bar.classList.remove('on'); hideTip(); });
  });
}

// Horizontal bars in HTML: long category names wrap instead of being clipped; value sits at the bar tip.
function hbars(plot, { rows, series, format, max }) {
  const top = max ?? Math.max(...rows.flatMap(r => r.values));
  plot.replaceChildren(h('div', { class: 'hbars' }, rows.map(r => {
    const row = h('div', { class: 'hrow', tabindex: 0 }, h('div', { class: 'name', text: r.label }),
      h('div', { class: 'bars' }, r.values.map((v, i) => h('div', { class: 'track' },
        h('div', { class: 'bar', style: `width:${(v / top * 80).toFixed(2)}%;background:${series[i].color}` }),
        h('span', { class: 'val', text: format(v) })))));
    const rowsTip = r.values.map((v, i) => ({ color: series[i].color, name: series[i].name, value: format(v) })).concat(r.extra || []);
    row.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, r.label, rowsTip));
    row.addEventListener('focus', () => { const b = row.getBoundingClientRect(); showTip(b.left + 120, b.top + 8, r.label, rowsTip); });
    for (const ev of ['pointerleave', 'blur']) row.addEventListener(ev, hideTip);
    return row;
  })));
}

// Two numeric axes (one scale each), lines with markers. Each marker has a generous hit area.
function xyChart(plot, { series, xFormat, yFormat, xTitle, height = 320, label }) {
  const { W, root } = frame(plot, height, label);
  const m = { l: 54, r: 18, t: 14, b: 44 };
  const pts = series.flatMap(d => d.points);
  const xt = ticksFor(Math.min(...pts.map(p => p.x)), Math.max(...pts.map(p => p.x)), Math.max(3, Math.floor((W - m.l - m.r) / 120)));
  const yt = ticksFor(Math.min(...pts.map(p => p.y)), Math.max(...pts.map(p => p.y)), 4);
  const x = v => m.l + (W - m.l - m.r) * (v - xt[0]) / (xt.at(-1) - xt[0]);
  const y = v => m.t + (height - m.t - m.b) * (1 - (v - yt[0]) / (yt.at(-1) - yt[0]));
  yAxis(root, yt, y, m, W, yFormat);
  for (const t of xt) s('text', { x: x(t), y: height - m.b + 16, class: 'tick', 'text-anchor': 'middle' }, root, xFormat(t));
  s('text', { x: m.l + (W - m.l - m.r) / 2, y: height - 6, class: 'axis-title', 'text-anchor': 'middle' }, root, xTitle);
  for (const d of series) {
    s('path', { d: d.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(''), class: 'line', style: `stroke:${d.color}` }, root);
  }
  for (const d of series) {
    for (const p of d.points) {
      s('circle', { cx: x(p.x), cy: y(p.y), r: 4.5, class: 'ring', style: `fill:${d.color}` }, root);
      const hit = s('circle', { cx: x(p.x), cy: y(p.y), r: 13, fill: 'transparent', tabindex: 0 }, root);
      hit.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, d.name, p.tip));
      hit.addEventListener('focus', () => { const b = hit.getBoundingClientRect(); showTip(b.left + 13, b.top + 13, d.name, p.tip); });
      for (const ev of ['pointerleave', 'blur']) hit.addEventListener(ev, hideTip);
    }
  }
}

const section = (...kids) => h('section', {}, kids);
const intro = text => h('p', { class: 'intro', text });
const grid = (...kids) => h('div', { class: 'grid' }, kids);

