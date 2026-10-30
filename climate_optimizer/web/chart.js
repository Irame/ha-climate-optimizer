// Small SVG time charts: stacked panels on one time axis with a shared crosshair.

const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};

export const fmtTime = (t, tz) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: tz });
export const fmtDay = (t, tz) => new Date(t).toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'numeric', timeZone: tz });

function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const steps = [0.5, 1, 2, 2.5, 5, 10].map(m => m * mag);
  const make = step => {
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  // the roundest step near the wanted count, but always two ticks at least
  let i = steps.findIndex(s => s >= raw);
  if (i < 0) i = steps.length - 1;
  let out = make(steps[i]);
  while (out.length < 2 && i > 0) out = make(steps[--i]);
  return out;
}

function timeTicks(x0, x1, tz, px) {
  // labels at least ~64 px apart
  const perH = px / ((x1 - x0) / 36e5);
  const stepH = [1, 2, 3, 6, 12, 24, 48, 96].find(h => h * perH >= 64) || 168;
  const out = [];
  // local hour boundaries: walk hours and keep those divisible by stepH
  let t = Math.ceil(x0 / 36e5) * 36e5;
  for (; t <= x1; t += 36e5) {
    const h = +new Date(t).toLocaleString('en-GB', { hour: '2-digit', hour12: false, timeZone: tz }) % 24;
    if (h % stepH === 0) out.push({ t, midnight: h === 0 });
  }
  return out;
}

// A group of panels sharing x range, width and the hover line.
export class ChartGroup {
  constructor(container, { x0, x1, now, tz }) {
    this.container = container;
    this.x0 = x0; this.x1 = x1; this.now = now; this.tz = tz;
    this.panels = [];
    container.classList.add('chart-group');
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    container.appendChild(this.tip);
    this.width = Math.max(320, container.clientWidth || 800);
    this.left = 62; this.right = 12;
  }
  x(t) { return this.left + (t - this.x0) / (this.x1 - this.x0) * (this.width - this.left - this.right); }
  t(x) { return this.x0 + (x - this.left) / (this.width - this.left - this.right) * (this.x1 - this.x0); }

  add(opts) {
    const p = new Panel(this, opts);
    this.panels.push(p);
    return p;
  }

  hover(t, clientX, clientY) {
    if (t === null || t < this.x0 || t > this.x1) {
      this.tip.hidden = true;
      this.panels.forEach(p => p.cross(null));
      return;
    }
    this.panels.forEach(p => p.cross(t));
    const rows = this.panels.flatMap(p => p.valuesAt(t));
    const head = `${fmtDay(t, this.tz)} ${fmtTime(t, this.tz)}${t > this.now ? ' · plan' : ''}`;
    this.tip.innerHTML = `<div class="tip-head">${head}</div>` + rows.map(r =>
      `<div class="tip-row"><span class="sw" style="background:${r.color || 'transparent'}"></span><span>${r.name}</span><b>${r.text}</b></div>`).join('');
    this.tip.hidden = false;
    const box = this.container.getBoundingClientRect();
    const w = this.tip.offsetWidth;
    let left = clientX - box.left + 14;
    if (left + w > box.width) left = clientX - box.left - w - 14;
    this.tip.style.left = Math.max(0, left) + 'px';
    this.tip.style.top = Math.max(0, clientY - box.top - 20) + 'px';
  }
}

class Panel {
  constructor(g, { title, height = 140, unit = '', fmt = v => v.toFixed(1), yMin, yMax, legend = true }) {
    this.g = g; this.height = height; this.unit = unit; this.fmt = fmt;
    this.yMin = yMin; this.yMax = yMax;
    this.series = []; this.bands = []; this.rows = null; this.legend = legend;
    this.wrap = document.createElement('div');
    this.wrap.className = 'panel';
    if (title) {
      const h = document.createElement('div');
      h.className = 'panel-title';
      h.textContent = title;
      this.wrap.appendChild(h);
      this.titleEl = h;
    }
    g.container.insertBefore(this.wrap, g.tip);
  }
  line(s) { this.series.push(s); return this; }     // {name, color, t, y, step, dash, width}
  band(b) { this.bands.push(b); return this; }      // {t, lo, hi, color} (step-wise)
  gantt(rows) { this.rows = rows; return this; }    // [{name, color, spans:[{t0,t1,v,label}]}]

  draw() {
    const g = this.g, W = g.width;
    const top = 6, bottom = 20;
    if (this.rows) this.height = top + bottom + this.rows.length * 22;
    const H = this.height;
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'chart', role: 'img' });
    this.svg = svg;
    const plotB = H - bottom;
    // y scale
    if (!this.rows) {
      let lo = Infinity, hi = -Infinity;
      const see = v => { if (v !== null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } };
      for (const s of this.series) if (!s.tipOnly) s.t.forEach((t, i) => { if (t >= g.x0 && t <= g.x1) see(s.y[i]); });
      for (const b of this.bands) b.t.forEach((t, i) => { if (t >= g.x0 && t <= g.x1) { see(b.lo[i]); see(b.hi[i]); } });
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (this.yMin !== undefined) lo = Math.min(lo, this.yMin);
      if (this.yMax !== undefined) hi = Math.max(hi, this.yMax);
      const pad = (hi - lo) * 0.08 || 0.5;
      lo -= pad; hi += pad;
      this.ty = niceTicks(lo, hi, Math.max(2, Math.round(H / 45)));
      this.lo = Math.min(lo, this.ty[0]); this.hi = Math.max(hi, this.ty[this.ty.length - 1]);
      this.y = v => top + (1 - (v - this.lo) / (this.hi - this.lo)) * (plotB - top);
      for (const v of this.ty) {
        el('line', { x1: g.left, x2: W - g.right, y1: this.y(v), y2: this.y(v), class: 'grid' }, svg);
        const tx = el('text', { x: g.left - 6, y: this.y(v) + 4, class: 'tick', 'text-anchor': 'end' }, svg);
        tx.textContent = this.fmt(v);
      }
    }
    // time grid
    for (const tk of timeTicks(g.x0, g.x1, g.tz, W - g.left - g.right)) {
      const x = g.x(tk.t);
      el('line', { x1: x, x2: x, y1: top, y2: plotB, class: tk.midnight ? 'grid strong' : 'grid' }, svg);
      const tx = el('text', { x, y: H - 5, class: 'tick', 'text-anchor': 'middle' }, svg);
      tx.textContent = tk.midnight ? fmtDay(tk.t, g.tz) : fmtTime(tk.t, g.tz);
    }
    // future shading and now
    if (g.now > g.x0 && g.now < g.x1) {
      el('rect', { x: g.x(g.now), y: top, width: W - g.right - g.x(g.now), height: plotB - top, class: 'future' }, svg);
    }
    if (this.rows) this._drawRows(svg, top);
    for (const b of this.bands) this._drawBand(svg, b);
    for (const s of this.series) if (!s.tipOnly) this._drawLine(svg, s);
    if (g.now > g.x0 && g.now < g.x1) {
      el('line', { x1: g.x(g.now), x2: g.x(g.now), y1: top, y2: plotB, class: 'now' }, svg);
    }
    this.crossLine = el('line', { x1: 0, x2: 0, y1: top, y2: plotB, class: 'cross', visibility: 'hidden' }, svg);
    const hit = el('rect', { x: g.left, y: 0, width: W - g.left - g.right, height: H, fill: 'transparent' }, svg);
    const move = e => {
      const r = svg.getBoundingClientRect();
      const x = (e.clientX - r.left) * (W / r.width);
      g.hover(g.t(x), e.clientX, e.clientY);
    };
    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerdown', move);
    hit.addEventListener('pointerleave', () => g.hover(null));
    this.wrap.appendChild(svg);
    if (this.legend && this.series.filter(s => s.name && !s.noLegend && !s.tipOnly).length > 1) {
      const lg = document.createElement('div');
      lg.className = 'legend';
      const seen = new Set();
      for (const s of this.series) {
        if (!s.name || s.noLegend || s.tipOnly || seen.has(s.legendKey || s.name)) continue;
        seen.add(s.legendKey || s.name);
        lg.insertAdjacentHTML('beforeend', `<span><i style="background:${s.color}"></i>${s.legendKey || s.name}</span>`);
      }
      this.wrap.appendChild(lg);
    }
  }

  _path(s) {
    const g = this.g;
    let d = '', pen = false;
    for (let i = 0; i < s.t.length; i++) {
      const v = s.y[i];
      if (v === null || v === undefined || !isFinite(v)) { pen = false; continue; }
      const x = Math.min(g.width - g.right, Math.max(g.left, g.x(s.t[i]))), y = this.y(v);
      if (!pen) { d += `M${x.toFixed(1)},${y.toFixed(1)}`; pen = true; continue; }
      if (s.step) d += `H${x.toFixed(1)}V${y.toFixed(1)}`;
      else d += `L${x.toFixed(1)},${y.toFixed(1)}`;
    }
    if (s.step && pen && s.tEnd) d += `H${Math.min(g.width - g.right, g.x(s.tEnd)).toFixed(1)}`;
    return d;
  }
  _drawLine(svg, s) {
    if (s.fill) {
      // the area under the line, down to the baseline, for each unbroken stretch
      const base = this.y(Math.max(this.lo, 0));
      const g = this.g;
      let pts = [];
      const flush = () => {
        if (pts.length > 1) el('path', { d: `M${pts[0][0]},${base}` + pts.map(([x, y]) => `L${x},${y}`).join('') + `L${pts.at(-1)[0]},${base}Z`,
          fill: s.color, opacity: s.fillOpacity ?? 0.22, stroke: 'none' }, svg);
        pts = [];
      };
      s.t.forEach((t, i) => {
        const v = s.y[i];
        if (v === null || v === undefined || !isFinite(v)) { flush(); return; }
        pts.push([Math.min(g.width - g.right, Math.max(g.left, g.x(t))).toFixed(1), this.y(v).toFixed(1)]);
      });
      flush();
    }
    el('path', { d: this._path(s), class: 'series', stroke: s.color, 'stroke-width': s.width || 2,
      'stroke-dasharray': s.dash ? '5 4' : null, opacity: s.opacity }, svg);
  }
  _drawBand(svg, b) {
    const g = this.g;
    let d = '';
    for (let i = 0; i < b.t.length - 1; i++) {
      if (b.lo[i] === null || b.hi[i] === null) continue;
      const x0 = g.x(b.t[i]), x1 = g.x(b.t[i + 1]);
      d += `M${x0},${this.y(b.hi[i])}H${x1}V${this.y(b.lo[i])}H${x0}Z`;
    }
    el('path', { d, fill: b.color, opacity: b.opacity ?? 0.16, stroke: 'none' }, svg);
  }
  _drawRows(svg, top) {
    const g = this.g;
    this.rowY = [];
    this.rows.forEach((r, i) => {
      const y = top + i * 22;
      this.rowY.push(y);
      const lb = el('text', { x: g.left - 6, y: y + 14, class: 'tick', 'text-anchor': 'end' }, svg);
      lb.textContent = r.short || r.name;
      el('line', { x1: g.left, x2: g.width - g.right, y1: y + 20, y2: y + 20, class: 'grid' }, svg);
      // one shape per unbroken run: its height at each step is the share of the step running
      const base = y + 18, full = 15;
      const spans = r.spans.filter(s => s.t1 > g.x0 && s.t0 < g.x1).sort((a, b) => a.t0 - b.t0);
      let run = [];
      const flush = () => {
        if (!run.length) return;
        const X = t => Math.min(g.width - g.right, Math.max(g.left, g.x(t))).toFixed(1);
        let d = `M${X(run[0].t0)},${base}`;
        for (const s of run) {
          const h = (base - full * Math.min(1, Math.max(0.12, s.v))).toFixed(1);  // a sliver stays visible
          d += `L${X(s.t0)},${h}L${X(s.t1)},${h}`;
        }
        d += `L${X(run.at(-1).t1)},${base}Z`;
        el('path', { d, fill: r.color, opacity: 0.85, stroke: 'none' }, svg);
        run = [];
      };
      for (const s of spans) {
        const on = s.v > 0.01;
        if (!on || (run.length && s.t0 - run.at(-1).t1 > 1000)) flush();
        if (on) run.push(s);
      }
      flush();
    });
  }

  cross(t) {
    if (!this.crossLine) return;
    if (t === null) { this.crossLine.setAttribute('visibility', 'hidden'); return; }
    const x = this.g.x(t);
    this.crossLine.setAttribute('x1', x); this.crossLine.setAttribute('x2', x);
    this.crossLine.setAttribute('visibility', 'visible');
  }

  valuesAt(t) {
    const out = [];
    if (this.rows) {
      for (const r of this.rows) {
        const s = r.spans.find(s => s.t0 <= t && t < s.t1);
        if (s) out.push({ name: r.name, color: r.color, text: s.label || (s.v > 0.01 ? 'on' : 'off') });
      }
      return out;
    }
    for (const s of this.series) {
      if (!s.name || s.noTip) continue;
      const i = idxAt(s.t, t, s.step);
      if (i === null) continue;
      if (!s.step && (t < s.t[0] || t > s.t[s.t.length - 1])) continue;
      if (s.step && s.tEnd && t > s.tEnd) continue;
      const v = s.y[i];
      if (v === null || v === undefined || !isFinite(v)) continue;
      const unit = s.unit ?? this.unit;
      out.push({ name: s.name, color: s.color, text: (s.fmt || this.fmt)(v) + (unit ? ' ' + unit : '') });
    }
    return out;
  }
}

function idxAt(ts, t, step) {
  if (!ts.length || t < ts[0] - (step ? 0 : 6e5)) return null;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ts[m] <= t) lo = m; else hi = m - 1; }
  if (!step && lo + 1 < ts.length && Math.abs(ts[lo + 1] - t) < Math.abs(ts[lo] - t)) return lo + 1;
  return lo;
}

// A plain XY line chart (comfort cost curves).
export function curveChart(container, { series, marks = [], dots = [], hlines = [], domain, yDomain, height = 150, xFmt = v => v, yFmt = v => v.toFixed(2) }) {
  container.innerHTML = '';
  const width = Math.max(240, container.clientWidth || 420);
  const L = 48, R = 8, T = 8, B = 26;
  let xlo = Infinity, xhi = -Infinity, yhi = 0;
  for (const s of [...series, ...dots]) s.x.forEach((x, i) => { xlo = Math.min(xlo, x); xhi = Math.max(xhi, x); if (isFinite(s.y[i])) yhi = Math.max(yhi, s.y[i]); });
  for (const h of hlines) yhi = Math.max(yhi, h.y);
  if (domain) [xlo, xhi] = domain;
  if (yDomain) yhi = yDomain[1];
  yhi = yhi || 1;
  const yt = niceTicks(0, yhi, 3);
  const ymax = Math.max(yhi, yt[yt.length - 1]);
  const X = v => L + (v - xlo) / (xhi - xlo) * (width - L - R);
  const Y = v => T + (1 - Math.min(v, ymax) / ymax) * (height - T - B);
  const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width: '100%', class: 'chart', role: 'img' });
  for (const v of yt) {
    el('line', { x1: L, x2: width - R, y1: Y(v), y2: Y(v), class: 'grid' }, svg);
    el('text', { x: L - 5, y: Y(v) + 4, class: 'tick', 'text-anchor': 'end' }, svg).textContent = yFmt(v);
  }
  for (const v of niceTicks(xlo, xhi, 6)) {
    el('line', { x1: X(v), x2: X(v), y1: T, y2: height - B, class: v === 0 ? 'grid strong' : 'grid' }, svg);
    el('text', { x: X(v), y: height - 8, class: 'tick', 'text-anchor': 'middle' }, svg).textContent = xFmt(v);
  }
  for (const m of marks) {
    if (m.x < xlo || m.x > xhi) continue;
    el('line', { x1: X(m.x), x2: X(m.x), y1: T, y2: height - B, class: 'now' }, svg);
    el('text', { x: X(m.x) + (m.x < 0 ? 4 : -4), y: T + 10, class: 'tick', 'text-anchor': m.x < 0 ? 'start' : 'end' }, svg).textContent = m.label;
  }
  for (const s of dots) {
    s.x.forEach((x, i) => el('circle', { cx: X(x).toFixed(1), cy: Y(s.y[i]).toFixed(1), r: 2.2, fill: s.color, opacity: s.opacity ?? 0.35 }, svg));
  }
  for (const h of hlines) {
    el('line', { x1: L, x2: width - R, y1: Y(h.y), y2: Y(h.y), class: 'now', 'stroke-dasharray': '4 3' }, svg);
    if (h.label) el('text', { x: width - R - 4, y: Y(h.y) - 4, class: 'tick', 'text-anchor': 'end' }, svg).textContent = h.label;
  }
  for (const s of series) {
    const d = s.x.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join('');
    el('path', { d, class: 'series', stroke: s.color, 'stroke-width': 2 }, svg);
  }
  container.appendChild(svg);
  return svg;
}
