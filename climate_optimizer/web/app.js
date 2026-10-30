import { ChartGroup, curveChart, fmtTime, fmtDay } from './chart.js';

// ---- basics -------------------------------------------------------------------
const $ = (sel, root = document) => root.querySelector(sel);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const SLOTS = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const slot = i => css(SLOTS[i % SLOTS.length]);
const num = (v, d = 1) => (v === null || v === undefined || !isFinite(v)) ? '–' : Number(v).toFixed(d);
const H = 36e5, MIN = 6e4;

function token() { try { return localStorage.getItem('co-token') || ''; } catch { return ''; } }
async function api(path, opts = {}) {
  const headers = { 'Content-Type': opts.raw ? 'application/octet-stream' : 'application/json', ...(opts.headers || {}) };
  const t = token();
  if (t) headers.Authorization = 'Bearer ' + t;
  const r = await fetch('api/' + path, { ...opts, headers });
  if (r.status === 401) {
    const t2 = prompt('This optimizer needs its API token:');
    if (t2) { try { localStorage.setItem('co-token', t2); } catch { } return api(path, opts); }
    throw new Error('token required');
  }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch { }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return opts.blob ? r.blob() : r.json();
}

// ---- entity pickers ---------------------------------------------------------------
// which entities fit a field: its domains, and the ones that most likely are meant listed first
const KINDS = {
  temp: { domains: ['sensor'], likely: e => e.device_class === 'temperature' || /°/.test(e.unit || '') },
  // an energy meter (kWh) or a power sensor (W): the meter first, its readings give the more exact power
  power: { domains: ['sensor'], groups: [['Energy meters (kWh), the more exact', e => isEnergy(e)],
    ['Power sensors (W)', e => e.device_class === 'power' || /^[kM]?W$/.test(e.unit || '')]] },
  opening: { domains: ['binary_sensor'], likely: e => ['door', 'window', 'opening', 'garage_door'].includes(e.device_class) },
  climate: { domains: ['climate'] },
  switch: { domains: ['switch', 'input_boolean'] },
  weather: { domains: ['weather'] },
  price: { domains: ['sensor'], likely: e => /kwh/i.test(e.unit || '') || e.device_class === 'monetary' },
  calendar: { domains: ['calendar'] },
};
const isEnergy = e => e.device_class === 'energy' || /^[kM]?Wh$/i.test(e.unit || '');
function entityOptions(kind, value, none = '— none —') {
  const k = KINDS[kind];
  const all = (S.entities || []).filter(e => k.domains.includes(e.id.split('.')[0]))
    .sort((a, b) => (a.name || a.id).localeCompare(b.name || b.id));
  const opt = e => `<option value="${esc(e.id)}" ${e.id === value ? 'selected' : ''}>${esc(e.name ? `${e.name} (${e.id})` : e.id)}</option>`;
  let html = none === null ? '' : `<option value="">${none}</option>`;
  if (value && !all.some(e => e.id === value)) html += `<option value="${esc(value)}" selected>${esc(value)} (not in Home Assistant)</option>`;
  const groups = k.groups || (k.likely ? [['Likely', k.likely]] : []);
  const placed = new Set();
  const parts = groups.map(([label, pred]) => {
    const hit = all.filter(e => !placed.has(e) && pred(e));
    hit.forEach(e => placed.add(e));
    return [label, hit];
  }).filter(([, hit]) => hit.length);
  const other = all.filter(e => !placed.has(e));
  if (parts.length && (other.length || parts.length > 1)) {
    for (const [label, hit] of parts) html += `<optgroup label="${esc(label)}">${hit.map(opt).join('')}</optgroup>`;
    if (other.length) html += `<optgroup label="Other">${other.map(opt).join('')}</optgroup>`;
  } else html += all.map(opt).join('');
  return html;
}
// under power fields: what the optimizer makes of the entity picked
function powerCheck(sel) {
  const hint = sel.parentElement.querySelector('.kind-hint') || sel.insertAdjacentElement('afterend', Object.assign(document.createElement('div'), { className: 'hint kind-hint' }));
  const e = (S.entities || []).find(x => x.id === sel.value);
  hint.textContent = !e ? '' : isEnergy(e) ? 'energy meter: the power comes from its readings'
    : 'power sensor: works; the same device’s energy meter, if it has one, is more exact';
}
const entitySelect = (path, kind, value) => `<select data-path="${path}" data-kind="${kind}" style="width:100%">${entityOptions(kind, value)}</select>`;
const entityMulti = (path, kind, values) => `<div class="multi" data-path="${path}" data-kind="${kind}" data-multi="${esc(JSON.stringify(values || []))}"></div>`;
function initMulti(box) {
  let vals = JSON.parse(box.dataset.multi || '[]');
  const name = id => (S.entities || []).find(e => e.id === id)?.name || id;
  const render = () => {
    box.dataset.multi = JSON.stringify(vals);
    box.innerHTML = vals.map((v, i) => `<span class="chip active" title="${esc(v)}">${esc(name(v))}<button type="button" class="x" data-i="${i}" aria-label="Remove ${esc(name(v))}">×</button></span>`).join('')
      + `<select class="add" aria-label="Add">${entityOptions(box.dataset.kind, '', vals.length ? 'add another…' : 'add…')}</select>`;
    box.querySelectorAll('.x').forEach(b => b.onclick = () => { vals.splice(+b.dataset.i, 1); render(); });
    box.querySelector('.add').onchange = e => { const v = e.target.value; if (v && !vals.includes(v)) vals.push(v); render(); };
  };
  render();
}

const S = { config: null, status: null, plan: null, history: null, entities: null, tz: undefined, zoneSel: 'all' };

function zoneColor(zid) {
  const zones = S.config?.layout?.zones || [];
  const i = zones.findIndex(z => z.id === zid);
  return slot(i < 0 ? 0 : i);
}
function zoneName(zid) { return (S.config?.layout?.zones || []).find(z => z.id === zid)?.name || zid; }
function unitOf(uid) { return (S.config?.layout?.units || []).find(u => u.id === uid); }
function fnName(id) { return (S.config?.functions || []).find(f => f.id === id)?.name || id; }
function ago(ms) {
  if (!ms) return 'never';
  const s = (Date.now() - ms) / 1000;
  if (s < 90) return Math.round(s) + ' s ago';
  if (s < 5400) return Math.round(s / 60) + ' min ago';
  if (s < 172800) return Math.round(s / 3600) + ' h ago';
  return Math.round(s / 86400) + ' days ago';
}

// ---- routing ------------------------------------------------------------------
const pages = {};
let current = null;
function route() {
  const name = (location.hash || '#dashboard').slice(1);
  const page = pages[name] ? name : 'dashboard';
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('active', a.dataset.page === page));
  document.querySelectorAll('section.page').forEach(s => s.classList.toggle('active', s.id === 'page-' + page));
  current = page;
  pages[page]().catch(showError(page));
}
const showError = page => e => {
  const root = $('#page-' + page);
  root.insertAdjacentHTML('afterbegin', `<div class="msg err">${esc(e.message || e)}</div>`);
  console.error(e);
};
window.addEventListener('hashchange', route);
let resizeTimer;
// Charts are drawn for a width: redraw them when it changes. Only then: on a
// phone the keyboard changes the height, and redrawing a form under it would
// take the focus and what was typed.
const CHART_PAGES = new Set(['dashboard', 'history', 'model']);
let lastWidth = window.innerWidth;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    const typing = document.activeElement?.matches?.('input, select, textarea');
    if (current && CHART_PAGES.has(current) && !typing) pages[current](true).catch(() => { });
  }, 250);
});

async function loadBase() {
  [S.config, S.status] = await Promise.all([api('config'), api('status')]);
  pills();
}

function pills() {
  const st = S.status;
  if (!st) return;
  const p = [];
  p.push(`<span class="pill"><span class="dot ${st.ha.connected ? 'good' : 'critical'}"></span>${st.ha.connected ? 'Home Assistant' : 'Not connected'}</span>`);
  p.push(`<span class="pill"><span class="dot ${st.shadow ? 'warning' : 'good'}"></span>${st.shadow ? 'Shadow mode' : 'Controlling'}</span>`);
  const age = st.plan.created ? (Date.now() - st.plan.created) / MIN : null;
  p.push(`<span class="pill"><span class="dot ${st.plan.error ? 'critical' : age !== null && age < 11 ? 'good' : 'warning'}"></span>Plan ${st.plan.created ? ago(st.plan.created) : 'none'}</span>`);
  p.push(`<span class="pill"><span class="dot ${st.model.trained ? 'good' : 'warning'}"></span>${st.model.training ? 'Training…' : st.model.trained ? 'Model trained' : 'Model untrained'}</span>`);
  $('#pills').innerHTML = p.join('');
}

// ---- overview -----------------------------------------------------------------
pages.dashboard = async (redraw) => {
  const root = $('#page-dashboard');
  if (!redraw || !S.plan) {
    await loadBase();
    [S.plan, S.history] = await Promise.all([api('plan'), api('history?hours=12')]);
  }
  const st = S.status, P = S.plan?.plan, info = S.plan?.info || {};
  let html = '';
  if (!st.ha.connected) html += `<div class="msg err">Home Assistant: ${esc(st.ha.error || 'not connected')}. Check <a href="#settings">Settings</a>.</div>`;
  if (S.plan?.error) html += `<div class="msg err">Planning: ${esc(S.plan.error)}</div>`;
  for (const n of info.notes || []) html += `<div class="msg">${esc(n)}</div>`;
  if (st.shadow) html += `<div class="msg">Shadow mode: the optimizer plans and shows what it would do, but the proxies keep their own thermostats. Switch it off in <a href="#settings">Settings</a> once the plans look right.</div>`;

  // one card per unit
  html += '<div class="grid-cards">';
  const layout = S.config.layout;
  for (const u of layout.units) {
    const z = st.zones[u.zone] || {};
    const us = st.units[u.id] || {};
    const ui = info.units?.[u.id] || {};
    const k = 0;
    const share = P?.share?.[u.id]?.[k] ?? 0;
    const heat = P?.heat?.[u.id]?.[k] ?? 0;
    const fnId = info.schedule?.function?.[u.zone]?.[0];
    const comfort = P?.comfort?.[u.zone]?.[0];
    let state = '<span class="badge">off</span>';
    if (share >= 0.5) state = `<span class="badge ${heat < 0 ? 'cool' : 'on'}">${heat < 0 ? 'cooling' : 'heating'} ${num(Math.abs(heat), 1)} kW${us.decision?.level !== undefined ? ' · level ' + us.decision.level : ''}</span>`;
    let next = '';
    if (P) {
      const on = P.share[u.id];
      const now0 = on[0] >= 0.5;
      const j = on.findIndex((v, i) => i > 0 && (v >= 0.5) !== now0 && (now0 || v > 0.05));
      if (j > 0) next = `${now0 ? 'off' : 'on'} at ${fmtTime(P.t[j], S.tz)}`;
    }
    const notes = [];
    if (ui.note) notes.push(`<span class="badge warn">${esc(ui.note)}</span>`);
    if (ui.anyway && st.shadow) notes.push('<span class="badge">planned anyway</span>');
    if ((z.windows_open || []).length) notes.push('<span class="badge warn">window open</span>');
    const opt = us.proxy_attrs?.optimizer;
    html += `<div class="card unit-card">
      <div class="top"><span class="name"><i style="background:${zoneColor(u.zone)}"></i>${esc(u.name)}</span>${state}</div>
      <div class="temp">${num(z.temperature, 1)} °C<small>${comfort !== undefined ? `comfort ${num(comfort, 1)} °C` : 'no comfort cost'}</small></div>
      <div class="line">${comfort !== undefined && fnId ? esc(fnName(fnId)) + (info.schedule.source?.[u.zone]?.[0] && info.schedule.source[u.zone][0] !== 'default' ? ' · ' + esc(info.schedule.source[u.zone][0]) : '')
        + (P?.comfort_cost?.[u.zone] >= 0.005 ? ` · ${num(P.comfort_cost[u.zone], 2)} € off comfort` : '')
        : info.no_comfort?.[u.zone] ? 'not the plan’s to heat while the unit is off' : 'no comfort temperature'}${next ? ' · ' + next : ''}</div>
      <div class="line">${esc(us.proxy || 'no proxy')}${us.proxy_state ? ' · ' + esc(us.proxy_state) : ''}${opt ? ' · ' + esc(opt) : ''} ${notes.join(' ')}</div>
    </div>`;
  }
  html += '</div>';

  if (P) {
    const c = P.cost;
    const hours = (P.t[P.t.length - 1] - P.t[0]) / H;
    const dt = P.t.slice(1).map((t, i) => (t - P.t[i]) / H);
    const kwh = P.electric.reduce((a, v, i) => a + v * dt[i], 0);
    html += `<div class="stats" style="margin-bottom:14px">
      <div class="stat"><div class="k">Plan</div><div class="v">${P.mode === 'cool' ? 'Cooling' : 'Heating'}</div><div class="s">${num(hours, 0)} h ahead · ${ago(S.plan.created)}</div></div>
      <div class="stat"><div class="k">Electricity</div><div class="v">${num(c.energy, 2)} €</div><div class="s">${num(kwh, 1)} kWh over the plan</div></div>
      <div class="stat"><div class="k">Comfort cost</div><div class="v">${num(c.comfort, 2)} €</div><div class="s">${(() => {
        const top = Object.entries(P.comfort_cost || {}).filter(([, v]) => v >= 0.01).sort((a, b) => b[1] - a[1]).slice(0, 2);
        return top.length ? 'mostly ' + top.map(([z, v]) => `${esc(zoneName(z))} ${num(v, 2)} €`).join(', ') : 'deviations, as priced';
      })()}</div></div>
      <div class="stat"><div class="k">Starts</div><div class="v">${num(c.starts, 2)} €</div><div class="s">compressor and heater</div></div>
      ${airTile(P, info)}
      <div class="stat"><div class="k">Heat missing at the end</div><div class="v">${num(c.terminal, 2)} €</div><div class="s">solved in ${num(P.seconds, 1)} s · ${esc(P.status)}</div></div>
    </div>`;
  }
  html += `<div class="card"><div class="toolbar"><h3 style="margin:0">Next hours</h3><span class="spacer"></span><div class="chips" id="zone-chips"></div></div><div id="dash-chart"></div></div>`;
  root.innerHTML = html;
  if (!P) { $('#dash-chart').innerHTML = '<p class="hint">No plan yet.</p>'; return; }
  zoneChips($('#zone-chips'), () => pages.dashboard(true));
  drawTimeline($('#dash-chart'), P, info, S.history);
};

function airTile(P, info) {
  const ai = info.airing;
  if (!S.config.airing?.enabled || !ai) return '';
  const sug = ai.suggested || P.airing?.[0];
  const later = (P.airing || []).filter(a => !sug || a[0] > sug[0]);
  const now = Date.now();
  let v, sub;
  if (sug && sug[0] <= now && now < sug[1]) {
    v = 'Now';
    sub = `until ${fmtTime(sug[1], S.tz)}`;
  } else if (sug) {
    v = fmtTime(sug[0], S.tz);
    const day = fmtDay(sug[0], S.tz) !== fmtDay(now, S.tz) ? fmtDay(sug[0], S.tz) + ' · ' : '';
    const c = P.airing_cost;
    // what it adds; when the rooms are warmer than wanted (or it cools), it saves instead
    const cost = c === null || c === undefined ? '' : c >= 0.005 ? ` · costs ${num(c, 2)} €` : c <= -0.005 ? ` · saves ${num(-c, 2)} €` : ' · costs next to nothing';
    sub = `${day}${num(ai.duration_min, 0)} min${cost}`;
  } else {
    v = '—';
    sub = ai.none || 'none in the plan';
  }
  if (later.length) sub += ` · then ${later.map(a => (fmtDay(a[0], S.tz) !== fmtDay(now, S.tz) ? fmtDay(a[0], S.tz).split(',')[0] + ' ' : '') + fmtTime(a[0], S.tz)).join(', ')}`;
  sub += ai.done_today >= ai.per_day ? ' · done for today' : ai.done_today ? ` · ${ai.done_today} of ${ai.per_day} done today` : '';
  return `<div class="stat"><div class="k">Airing</div><div class="v">${esc(v)}</div><div class="s">${esc(sub)}</div></div>`;
}

function zoneChips(box, redraw) {
  const zones = S.config.layout.zones;
  const items = [{ id: 'all', name: 'All rooms' }, ...zones];
  box.innerHTML = items.map(z => `<button class="chip ${S.zoneSel === z.id ? 'active' : ''}" data-z="${z.id}">${z.id !== 'all' ? `<i style="background:${zoneColor(z.id)}"></i>` : ''}${esc(z.name)}</button>`).join('');
  box.querySelectorAll('button').forEach(b => b.onclick = () => { S.zoneSel = b.dataset.z; redraw(); });
}

function comfortBand(P, info, z) {
  const fns = Object.fromEntries((S.config.functions || []).map(f => [f.id, f]));
  const ids = info.schedule?.function?.[z] || [];
  const tc = P.comfort[z];
  if (!tc) return null;
  const lo = [], hi = [];
  for (let k = 0; k < tc.length; k++) {
    const f = fns[ids[k]] || {};
    lo.push(tc[k] - (f.below?.deadband ?? 0));
    hi.push(tc[k] + (f.above?.deadband ?? 0));
  }
  lo.push(lo[lo.length - 1]); hi.push(hi[hi.length - 1]);
  return { t: P.t, lo, hi };
}

function drawTimeline(box, P, info, hist) {
  box.innerHTML = '';
  const now = P.t[0];
  const x0 = hist?.t?.length ? Math.max(hist.t[0], now - 12 * H) : now - H;
  const g = new ChartGroup(box, { x0, x1: P.t[P.t.length - 1], now, tz: S.tz });
  const zones = S.config.layout.zones.filter(z => S.zoneSel === 'all' || z.id === S.zoneSel);
  // temperatures
  const tp = g.add({ title: 'Room temperature (°C): measured, then planned', unit: '°C', height: 220 });
  if (S.zoneSel !== 'all') {
    const b = comfortBand(P, info, S.zoneSel);
    if (b) tp.band({ ...b, color: zoneColor(S.zoneSel), opacity: 0.14 });
    if (P.comfort[S.zoneSel]) tp.line({ name: 'Comfort', color: css('--muted'), t: P.t, y: [...P.comfort[S.zoneSel], P.comfort[S.zoneSel].at(-1)], step: true, width: 1, noLegend: true });
  }
  for (const z of zones) {
    const c = zoneColor(z.id);
    if (hist?.temp?.[z.id]) tp.line({ name: z.name, color: c, t: hist.t, y: hist.temp[z.id] });
    if (P.temp[z.id]) tp.line({ name: z.name + ' (plan)', legendKey: z.name, color: c, t: P.t, y: P.temp[z.id], dash: true });
  }
  // units
  const rows = S.config.layout.units.map(u => {
    const spans = [];
    if (hist?.t) {
      const step = 5 * MIN;
      const hu = hist.units?.[u.id], hh = hist.heater?.[u.id];
      hist.t.forEach((t, i) => {
        const v = hu ? (hu.heat[i] || 0) + (hu.cool[i] || 0) : (hh ? (hh[i] > 0.05 ? 1 : 0) : 0);
        if (v > 0.01 && t < now) spans.push({ t0: t, t1: Math.min(now, t + step), v, label: hu ? `ran ${Math.round(v * 100)} %${hu.level[i] !== null ? ', level ' + num(hu.level[i], 1) : ''}` : `${num(hh[i], 2)} kW` });
      });
    }
    P.share[u.id].forEach((v, k) => spans.push({ t0: P.t[k], t1: P.t[k + 1], v, label: v > 0.01 ? `${num(Math.abs(P.heat[u.id][k]), 2)} kW${v < 0.99 ? ', ' + Math.round(v * 100) + ' % of the step' : ''}` : 'off' }));
    return { name: u.name, short: u.name.split(' ')[0], color: zoneColor(u.zone), spans };
  });
  if (S.config.airing?.enabled || hist?.open) {
    // windows: open as recorded (any window room), then the airing planned
    const spans = [];
    const zs = Object.keys(hist?.open || {});
    if (zs.length && hist?.t) hist.t.forEach((t, i) => {
      const v = Math.max(...zs.map(z => hist.open[z][i] || 0));
      if (v > 0.01 && t < now) spans.push({ t0: t, t1: Math.min(now, t + 5 * MIN), v, label: `open ${Math.round(v * 100)} % of the 5 minutes: ${zs.filter(z => hist.open[z][i] > 0.01).map(zoneName).join(', ')}` });
    });
    const planned = [...(P.airing || [])];
    const sug = info.airing?.suggested;
    if (sug && !planned.some(a => a[0] === sug[0])) planned.unshift(sug);
    for (const [a, b] of planned) if (b > now) spans.push({ t0: Math.max(a, now), t1: b, v: 1, label: `airing suggested ${fmtTime(a, S.tz)}–${fmtTime(b, S.tz)}` });
    if (spans.length || S.config.airing?.enabled) rows.push({ name: 'Windows', short: 'Windows', color: css('--muted'), spans });
  }
  g.add({ title: 'Units: recorded, then planned (height: share of each step running)' }).gantt(rows);
  // price
  const pp = g.add({ title: 'Electricity price (ct/kWh)', unit: 'ct', height: 120, fmt: v => v.toFixed(0) });
  const prices = (info.prices || []).filter(p => p[1] > x0).sort((a, b) => a[0] - b[0]);
  const past = (hist?.prices || []).filter(p => p[0] < (prices[0]?.[0] ?? Infinity));
  const all = [...past, ...prices];
  if (all.length) pp.line({ name: 'Price', color: slot(6), t: all.map(p => p[0]), y: all.map(p => p[2] * 100), step: true, tEnd: all.at(-1)[1] });
  pp.fmt = v => v.toFixed(v < 10 ? 1 : 0);
  // outdoor
  const op = g.add({ title: 'Outdoor temperature (°C): measured, then forecast', unit: '°C', height: 110 });
  if (hist?.tout) op.line({ name: 'Outdoor', color: slot(0), t: hist.t, y: hist.tout });
  const mids = P.t.slice(0, -1).map((t, i) => (t + P.t[i + 1]) / 2);
  if (info.tout) op.line({ name: 'Outdoor (forecast)', legendKey: 'Outdoor', color: slot(0), t: mids, y: info.tout, dash: true, noLegend: true });
  sunPanel(g, hist, info.ghi ? { t: mids, ghi: info.ghi, cloud: info.cloud } : null);
  g.panels.forEach(p => p.draw());
}

// Sun on the ground after the clouds, the way the model sees it; cloud cover in the tooltip.
// Its own panel: a second scale on the temperature chart would invent a relation.
function sunPanel(g, hist, plan) {
  const sp = g.add({ title: 'Sun (W/m², after clouds)' + (plan ? ': recorded weather, then forecast' : ''), unit: 'W/m²', height: 90, yMin: 0, fmt: v => v.toFixed(0) });
  const sun = slot(3);
  if (hist?.ghi) {
    sp.line({ name: 'Sun', color: sun, t: hist.t, y: hist.ghi, fill: true, width: 1.5 });
    if (hist.cloud) sp.line({ name: 'Clouds', color: css('--muted'), t: hist.t, y: hist.cloud, tipOnly: true, unit: '%' });
  }
  if (plan) {
    sp.line({ name: 'Sun (forecast)', color: sun, t: plan.t, y: plan.ghi, fill: true, dash: true, width: 1.5, noLegend: true });
    if (plan.cloud) sp.line({ name: 'Clouds (forecast)', color: css('--muted'), t: plan.t, y: plan.cloud, tipOnly: true, unit: '%' });
  }
  return sp;
}

// ---- history ------------------------------------------------------------------
pages.history = async (redraw) => {
  const root = $('#page-history');
  S.histHours = S.histHours || 48;
  if (!redraw || !S.histLong) {
    await loadBase();
    S.histLong = await api('history?hours=' + S.histHours);
  }
  const h = S.histLong;
  root.innerHTML = `<div class="toolbar"><h2 style="margin:0">History</h2><span class="spacer"></span>
    <div class="chips" id="hist-range">${[24, 48, 168, 336].map(x => `<button class="chip ${S.histHours === x ? 'active' : ''}" data-h="${x}">${x < 168 ? x + ' h' : x / 24 + ' days'}</button>`).join('')}</div>
    <div class="chips" id="hist-zones"></div></div>
    ${h.error ? `<div class="msg err">${esc(h.error)}</div>` : ''}
    <div class="card"><div id="hist-chart"></div></div>
    <div class="card"><h3>Energy and cost</h3><div id="hist-sum"></div></div>`;
  root.querySelectorAll('#hist-range button').forEach(b => b.onclick = () => { S.histHours = +b.dataset.h; S.histLong = null; pages.history(); });
  zoneChips($('#hist-zones'), () => pages.history(true));
  if (!h.t) return;
  const g = new ChartGroup($('#hist-chart'), { x0: h.start, x1: h.end, now: h.end, tz: S.tz });
  const zones = S.config.layout.zones.filter(z => S.zoneSel === 'all' || z.id === S.zoneSel);
  const tp = g.add({ title: 'Room temperature (°C): measured, and what the plans predicted an hour earlier', unit: '°C', height: 220 });
  for (const z of zones) {
    tp.line({ name: z.name, color: zoneColor(z.id), t: h.t, y: h.temp[z.id] });
    const pr = h.predicted_1h?.[z.id];
    if (pr?.length && S.zoneSel !== 'all') tp.line({ name: z.name + ' (predicted 1 h ahead)', legendKey: 'predicted 1 h ahead', color: css('--muted'), t: pr.map(p => p[0]), y: pr.map(p => p[1]), dash: true });
  }
  const rows = S.config.layout.units.map(u => {
    const hu = h.units?.[u.id], hh = h.heater?.[u.id];
    const spans = h.t.map((t, i) => {
      const v = hu ? (hu.heat[i] || 0) + (hu.cool[i] || 0) : (hh && hh[i] > 0.05 ? 1 : 0);
      return { t0: t, t1: t + 5 * MIN, v, label: hu ? `${Math.round(v * 100)} %${hu.level[i] !== null && v > 0 ? ', level ' + num(hu.level[i], 1) : ''}` : `${num(hh?.[i], 2)} kW` };
    });
    return { name: u.name, short: u.name.split(' ')[0], color: zoneColor(u.zone), spans };
  });
  g.add({ title: 'Units running (height: share of each 5 minutes)' }).gantt(rows);
  if (h.phase?.some(v => v !== null)) g.add({ title: 'Outdoor unit phase (kW)', unit: 'kW', height: 110, fmt: v => v.toFixed(1) }).line({ name: 'Phase', color: slot(6), t: h.t, y: h.phase });
  const pr = h.prices || [];
  if (pr.length) g.add({ title: 'Electricity price (ct/kWh)', unit: 'ct', height: 110, fmt: v => v.toFixed(0) }).line({ name: 'Price', color: slot(6), t: pr.map(p => p[0]), y: pr.map(p => p[2] * 100), step: true, tEnd: pr.at(-1)[1] });
  g.add({ title: 'Outdoor temperature (°C)', unit: '°C', height: 100 }).line({ name: 'Outdoor', color: slot(0), t: h.t, y: h.tout });
  sunPanel(g, h, null);
  g.panels.forEach(p => p.draw());

  // energy: phase-based where recorded, priced with the recorded prices
  let kwh = 0, eur = 0;
  const priceAt = t => { const p = pr.find(p => p[0] <= t && t < p[1]); return p ? p[2] : null; };
  if (h.phase) h.t.forEach((t, i) => { const v = h.phase[i]; if (v !== null) { kwh += v / 12; const p = priceAt(t); if (p !== null) eur += v / 12 * p; } });
  const heaterKwh = Object.values(h.heater || {}).reduce((a, arr) => a + arr.reduce((b, v) => b + (v || 0) / 12, 0), 0);
  $('#hist-sum').innerHTML = `<div class="stats">
    <div class="stat"><div class="k">Outdoor unit phase</div><div class="v">${num(kwh, 1)} kWh</div><div class="s">${num(eur, 2)} € at the recorded prices · includes other loads on that phase</div></div>
    <div class="stat"><div class="k">Heaters</div><div class="v">${num(heaterKwh, 1)} kWh</div><div class="s">from their plugs</div></div>
  </div>`;
};

// ---- comfort and schedules ----------------------------------------------------
const DAYS = ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU'];
const sideCost = (s, d) => {
  const x = Math.max(0, d - (s.deadband || 0));
  let c = (s.linear || 0) * x + (s.quadratic || 0) * x * x;
  if (s.limit !== null && s.limit !== undefined && s.limit !== '' && d > s.limit) c += 50 * (d - s.limit);
  return c;
};

pages.comfort = async (redraw) => {
  const root = $('#page-comfort');
  if (!redraw || !S.draft) {
    await loadBase();
    if (!S.entities) S.entities = await api('entities').catch(() => []);
    S.draft = JSON.parse(JSON.stringify({ functions: S.config.functions, defaults: S.config.defaults, comfort: S.config.comfort, rules: S.config.rules, calendar: S.config.calendar, airing: S.config.airing }));
  }
  const D = S.draft, L = S.config.layout;
  const fnOpts = sel => D.functions.map(f => `<option value="${esc(f.id)}" ${f.id === sel ? 'selected' : ''}>${esc(f.name)}</option>`).join('');
  let html = `<div class="toolbar"><h2 style="margin:0">Comfort &amp; schedules</h2><span class="spacer"></span>
    <span id="cf-msg" class="hint"></span><button id="cf-reset">Discard changes</button><button class="primary" id="cf-save">Save</button></div>
    <p class="hint">What being off the comfort temperature is worth, in euros per hour. The optimizer weighs this against electricity: a cheap deviation is taken when it saves more than it costs. Inside the deadband nothing is paid; the limit is not crossed unless it can't be helped.</p>`;
  html += '<div id="fn-list"></div><div class="row" style="margin-bottom:14px"><button id="fn-add">Add comfort function</button></div>';
  // zones
  html += `<div class="card"><h3>Rooms</h3><div class="table-wrap"><table><tr><th>Room</th><th>Comfort function when no rule applies</th><th>Comfort temperature</th></tr>`;
  for (const z of L.zones) {
    const hasProxy = L.units.some(u => u.zone === z.id && S.config.mapping?.units?.[u.id]?.proxy);
    html += `<tr><td><span class="name"><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${zoneColor(z.id)};margin-right:6px"></i>${esc(z.name)}</span></td>
      <td><select data-default="${z.id}"><option value="">Standard default</option>${fnOpts(D.defaults[z.id])}</select></td>
      <td>${hasProxy ? '<span class="hint">the proxy’s target temperature</span> ' : ''}<input type="number" step="0.5" style="width:90px" data-comfort="${z.id}" value="${D.comfort[z.id] ?? ''}" placeholder="${hasProxy ? 'fallback' : 'none'}"> °C</td></tr>`;
  }
  html += '</table></div><p class="hint">Rooms without a comfort temperature carry no comfort cost (the hallway, usually).</p></div>';
  // rules
  html += `<div class="card"><div class="toolbar"><h3 style="margin:0">Rules</h3><span class="spacer"></span><button id="rule-add">Add rule</button></div>
    <p class="hint">A rule switches a room to another comfort function, moves its comfort temperature, keeps units off, or keeps airing from being suggested, at recurring times. Where rules overlap, the higher priority wins. "Every other week, Monday to Thursday evenings": weekly, every 2 weeks, those four days, 18:00–23:00, starting on a Monday of an "on" week.</p>
    <div id="rule-list"></div></div>`;
  html += `<div class="card"><h3>Calendar</h3><div class="fields"><label class="field">Calendar entity<select id="cal">${entityOptions('calendar', D.calendar, '— no calendar —')}</select></label></div>
    <p class="hint">Events override every rule for their time. The title says what: <code>Wohnzimmer: Relaxed</code> picks a comfort function, <code>Schlafzimmer: aus</code> keeps the units there off, <code>Alle: +1</code> raises every comfort temperature by 1 K, <code>Lüften: aus</code> suggests no airing then. Rooms match by their name, id, the usual German names (Büro, Schlafzimmer, Wohnzimmer, Küche, Bad, Flur) or the names given in Settings; functions by name or id, and Normal, Entspannt, Streng, Frostschutz.</p></div>`;
  const A = D.airing, windowRooms = L.zones.filter(z => z.windows && (S.config.mapping?.zones?.[z.id]?.windows || []).length);
  const airRule = D.rules.some(r => r.action === 'no_airing' && r.enabled);
  html += `<div class="card"><h3>Airing</h3>
    <label class="check"><input type="checkbox" id="air-on" ${A.enabled ? 'checked' : ''}> Suggest when to air the whole apartment</label>
    <div class="fields" style="margin-top:10px">
      <label class="field">Minutes, all windows open<input type="number" min="2" max="60" step="1" data-air="duration_min" value="${A.duration_min}"></label>
      <label class="field">Times a day<input type="number" min="1" max="4" step="1" data-air="per_day" value="${A.per_day}"></label>
      <label class="field">At least apart (h)<input type="number" min="0.5" max="12" step="0.5" data-air="min_gap_h" value="${A.min_gap_h}"></label>
      <label class="field">Sensor in Home Assistant<input data-air="sensor" value="${esc(A.sensor)}" placeholder="none"></label>
    </div>
    <p class="hint">The plan puts it where the heat let out and heated back costs least (when cooling: where the outside air cools best), with the units in the window rooms off meanwhile, on the hour or half hour. Not when a <b>No airing</b> rule or a <code>Lüften: aus</code> event says so. Airings done today are seen at the window sensors${windowRooms.length ? ' (' + windowRooms.map(z => esc(z.name)).join(', ') + ')' : ''}. Home Assistant gets the next one as a timestamp sensor, with its end and extra cost as attributes, to notify with.</p>
    ${A.enabled && !windowRooms.length ? '<div class="msg">No room has window sensors yet: set them in <a href="#settings">Settings</a>, the plan airs the rooms that have them.</div>' : ''}
    ${A.enabled && !airRule ? '<div class="row"><span class="hint">No rule keeps it from the night.</span><button id="air-night">Add “No airing at night”</button></div>' : ''}
  </div>`;
  html += `<div class="card"><div class="toolbar"><h3 style="margin:0">The next two weeks</h3><span class="spacer"></span>
    <span class="hint" id="cal-status"></span><button id="cal-reload" ${D.calendar ? '' : 'disabled'}>Reload calendar</button></div>
    <p class="hint" style="margin-top:0">Saved rules and the calendar’s events. Hatched: units kept off; outlined: set by a calendar event.</p>
    <div id="preview"></div></div>`;
  root.innerHTML = html;

  // functions
  const list = $('#fn-list');
  D.functions.forEach((f, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    const side = (key, label) => {
      const s = f[key] || {};
      const inp = (prop, step, hint) => `<label>${hint}<input type="number" step="${step}" min="0" data-fn="${i}" data-side="${key}" data-prop="${prop}" value="${s[prop] ?? ''}" placeholder="${prop === 'limit' ? 'none' : '0'}"></label>`;
      return `<fieldset><legend>${label}</legend>${inp('deadband', 0.1, 'Deadband (K)')}${inp('linear', 0.01, 'Linear (€/K·h)')}${inp('quadratic', 0.01, 'Quadratic (€/K²·h)')}${inp('limit', 0.1, 'Limit (K)')}</fieldset>`;
    };
    card.innerHTML = `<div class="fn-card"><div>
        <div class="row" style="margin-bottom:8px"><input data-fn="${i}" data-prop="name" value="${esc(f.name)}" style="font-weight:600;flex:1">
        <span class="hint">id ${esc(f.id)}</span><button class="danger" data-del-fn="${i}" ${D.functions.length < 2 ? 'disabled' : ''}>Delete</button></div>
        <div class="sides">${side('below', 'Too cold')}${side('above', 'Too warm')}</div></div>
      <div><div class="curve"></div><p class="hint" style="margin:4px 0 0">Cost per hour against the deviation from the comfort temperature (K): left too cold, right too warm. Past a limit every further kelvin costs 50 € an hour.</p></div></div>`;
    list.appendChild(card);
    const draw = () => {
      const lim = s => (s?.limit === null || s?.limit === undefined || s?.limit === '') ? null : +s.limit;
      const lb = lim(f.below), la = lim(f.above);
      // the soft part only: past a limit the cost is prohibitive, shown as a marker
      const xs = Array.from({ length: 121 }, (_, k) => -6 + k * 0.1).filter(x => x >= -(lb ?? 99) - 1e-9 && x <= (la ?? 99) + 1e-9);
      const ys = xs.map(x => x < 0 ? sideCost({ ...f.below, limit: null }, -x) : sideCost({ ...f.above, limit: null }, x));
      const marks = [];
      if (lb !== null) marks.push({ x: -lb, label: 'limit' });
      if (la !== null) marks.push({ x: la, label: 'limit' });
      curveChart(card.querySelector('.curve'), { series: [{ x: xs, y: ys, color: slot(i) }], marks, domain: [-6, 6], xFmt: v => (v > 0 ? '+' : '') + v, yFmt: v => v.toFixed(2) + ' €', height: 170 });
    };
    draw();
    card.querySelectorAll('input[data-fn]').forEach(inp => inp.oninput = () => {
      const prop = inp.dataset.prop;
      if (prop === 'name') { f.name = inp.value; return; }
      const v = inp.value === '' ? (prop === 'limit' ? null : 0) : +inp.value;
      f[inp.dataset.side] = { ...(f[inp.dataset.side] || {}), [prop]: v };
      draw();
    });
  });
  root.querySelectorAll('[data-del-fn]').forEach(b => b.onclick = () => { D.functions.splice(+b.dataset.delFn, 1); pages.comfort(true); });
  $('#fn-add').onclick = () => {
    let n = D.functions.length + 1, id = 'custom' + n;
    while (D.functions.some(f => f.id === id)) id = 'custom' + (++n);
    D.functions.push({ id, name: 'Custom ' + n, below: { deadband: 0.5, linear: 0.1, quadratic: 0.05, limit: null }, above: { deadband: 1, linear: 0.05, quadratic: 0, limit: null } });
    pages.comfort(true);
  };
  root.querySelectorAll('[data-default]').forEach(s => s.onchange = () => { if (s.value) D.defaults[s.dataset.default] = s.value; else delete D.defaults[s.dataset.default]; });
  root.querySelectorAll('[data-comfort]').forEach(s => s.oninput = () => { if (s.value === '') delete D.comfort[s.dataset.comfort]; else D.comfort[s.dataset.comfort] = +s.value; });
  $('#cal').onchange = e => { D.calendar = e.target.value; };
  const nightRule = () => ({ id: 'night-no-airing', name: 'No airing at night', action: 'no_airing', zones: [], units: [], function: null, offset: 0, rrule: 'FREQ=DAILY', dtstart: new Date().toISOString().slice(0, 10), start: '22:00', end: '07:00', priority: 10, enabled: true });
  $('#air-on').onchange = e => {
    A.enabled = e.target.checked;
    if (A.enabled && !D.rules.some(r => r.action === 'no_airing')) D.rules.unshift(nightRule());  // nights off, to start with
    pages.comfort(true);
  };
  root.querySelectorAll('[data-air]').forEach(i => i.oninput = () => { A[i.dataset.air] = i.type === 'number' ? +i.value : i.value.trim(); });
  if ($('#air-night')) $('#air-night').onclick = () => { D.rules.unshift(nightRule()); pages.comfort(true); };

  // rules
  const rl = $('#rule-list');
  D.rules.forEach((r, i) => rl.appendChild(ruleEditor(r, i, D, fnOpts)));
  if (!D.rules.length) rl.innerHTML = '<p class="hint">No rules: every room uses its default all the time.</p>';
  $('#rule-add').onclick = () => {
    D.rules.push({ id: 'rule' + Date.now().toString(36), name: 'New rule', action: 'comfort', zones: [], units: [], function: D.functions[0].id, offset: 0, rrule: 'FREQ=DAILY', dtstart: new Date().toISOString().slice(0, 10), start: '22:00', end: '06:00', priority: 10, enabled: true });
    pages.comfort(true);
  };
  $('#cf-reset').onclick = () => { S.draft = null; pages.comfort(); };
  $('#cf-save').onclick = async () => {
    const msg = $('#cf-msg');
    try {
      S.config = await api('config', { method: 'PUT', body: JSON.stringify(D) });
      msg.textContent = 'Saved. The next plan uses it.';
      $('#cal-reload').disabled = !S.config.calendar;
      drawPreview();
    } catch (e) { msg.textContent = 'Not saved: ' + e.message; }
  };
  drawPreview();
  $('#cal-reload').onclick = async () => {
    const b = $('#cal-reload');
    b.disabled = true;
    try { await drawPreview(true); $('#cal-status').textContent += ' · the plan is being recomputed'; }
    catch (e) { $('#cal-status').textContent = e.message; }
    b.disabled = false;
  };
};

function parseRrule(s) {
  const o = {};
  for (const part of (s || '').replace(/^RRULE:/i, '').split(';')) { const [k, v] = part.split('='); if (k) o[k.toUpperCase()] = v || ''; }
  return o;
}
function buildRrule(o) { return Object.entries(o).filter(([, v]) => v !== '' && v !== undefined).map(([k, v]) => `${k}=${v}`).join(';'); }

function ruleEditor(r, i, D, fnOpts) {
  const L = S.config.layout;
  const box = document.createElement('div');
  box.className = 'rule';
  const rr = parseRrule(r.rrule);
  const freq = rr.FREQ || 'DAILY';
  const byday = (rr.BYDAY || '').split(',').filter(Boolean);
  const targets = r.action === 'no_airing' ? '' : r.action === 'lockout'
    ? L.units.map(u => `<label class="check"><input type="checkbox" data-unit="${u.id}" ${r.units.includes(u.id) ? 'checked' : ''}>${esc(u.name)}</label>`).join(' ')
    : L.zones.map(z => `<label class="check"><input type="checkbox" data-zone="${z.id}" ${r.zones.includes(z.id) ? 'checked' : ''}>${esc(z.name)}</label>`).join(' ');
  box.innerHTML = `<div class="row" style="margin-bottom:8px"><label class="check"><input type="checkbox" data-k="enabled" ${r.enabled ? 'checked' : ''}></label>
      <input data-k="name" value="${esc(r.name)}" style="font-weight:600;flex:1;min-width:160px">
      <select data-k="action"><option value="comfort" ${r.action === 'comfort' ? 'selected' : ''}>Comfort function</option><option value="offset" ${r.action === 'offset' ? 'selected' : ''}>Comfort temperature ±</option><option value="lockout" ${r.action === 'lockout' ? 'selected' : ''}>Keep units off</option><option value="no_airing" ${r.action === 'no_airing' ? 'selected' : ''}>No airing suggested</option></select>
      ${r.action === 'comfort' ? `<select data-k="function">${fnOpts(r.function)}</select>` : ''}
      ${r.action === 'offset' ? `<input type="number" step="0.5" data-k="offset" value="${r.offset}" style="width:80px"> K` : ''}
      <button class="danger" data-del>Delete</button></div>
    ${targets ? `<div class="row" style="margin-bottom:8px"><span class="hint">${r.action === 'lockout' ? 'Units (none ticked: all)' : 'Rooms (none ticked: all)'}:</span> ${targets}</div>` : ''}
    <div class="fields">
      <label class="field">Repeats<select data-r="FREQ"><option value="DAILY" ${freq === 'DAILY' ? 'selected' : ''}>daily</option><option value="WEEKLY" ${freq === 'WEEKLY' ? 'selected' : ''}>weekly</option><option value="MONTHLY" ${freq === 'MONTHLY' ? 'selected' : ''}>monthly</option></select></label>
      <label class="field">Every<input type="number" min="1" data-r="INTERVAL" value="${rr.INTERVAL || 1}"></label>
      <label class="field">From<input type="time" data-k="start" value="${r.start === '24:00' ? '23:59' : r.start}"></label>
      <label class="field">To<input type="time" data-k="end" value="${r.end === '24:00' ? '00:00' : r.end}"></label>
      <label class="field">Counting from<input type="date" data-k="dtstart" value="${r.dtstart}"></label>
      <label class="field">Priority<input type="number" data-k="priority" value="${r.priority}"></label>
    </div>
    <div class="row" style="margin-top:8px"><span class="hint">On:</span><div class="days">${DAYS.map(d => `<label><input type="checkbox" data-day="${d}" ${byday.includes(d) ? 'checked' : ''}>${d[0] + d[1].toLowerCase()}</label>`).join('')}</div>
      <span class="hint">(none: every day)</span><span class="spacer"></span><label class="field" style="flex-direction:row;align-items:center;gap:6px">RRULE<input data-k="rrule" value="${esc(r.rrule)}" style="width:280px"></label></div>`;
  const sync = () => {
    const o = parseRrule(r.rrule);
    o.FREQ = box.querySelector('[data-r=FREQ]').value;
    const iv = +box.querySelector('[data-r=INTERVAL]').value || 1;
    o.INTERVAL = iv > 1 ? iv : '';
    const days = [...box.querySelectorAll('[data-day]')].filter(c => c.checked).map(c => c.dataset.day);
    o.BYDAY = days.join(',');
    r.rrule = buildRrule({ FREQ: o.FREQ, INTERVAL: o.INTERVAL, BYDAY: o.BYDAY, ...Object.fromEntries(Object.entries(o).filter(([k]) => !['FREQ', 'INTERVAL', 'BYDAY'].includes(k))) });
    box.querySelector('[data-k=rrule]').value = r.rrule;
  };
  box.querySelectorAll('[data-r], [data-day]').forEach(e => e.onchange = sync);
  box.querySelectorAll('[data-k]').forEach(e => e.onchange = () => {
    const k = e.dataset.k;
    if (k === 'enabled') r.enabled = e.checked;
    else if (k === 'priority') r.priority = +e.value;
    else if (k === 'offset') r.offset = +e.value;
    else if (k === 'start' || k === 'end') r[k] = e.value || '00:00';
    else r[k] = e.value;
    if (k === 'action') { if (r.action === 'comfort' && !r.function) r.function = D.functions[0].id; pages.comfort(true); }
    if (k === 'rrule') pages.comfort(true);
  });
  box.querySelectorAll('[data-zone]').forEach(c => c.onchange = () => { r.zones = [...box.querySelectorAll('[data-zone]')].filter(x => x.checked).map(x => x.dataset.zone); });
  box.querySelectorAll('[data-unit]').forEach(c => c.onchange = () => { r.units = [...box.querySelectorAll('[data-unit]')].filter(x => x.checked).map(x => x.dataset.unit); });
  box.querySelector('[data-del]').onclick = () => { D.rules.splice(i, 1); pages.comfort(true); };
  return box;
}

async function drawPreview(refresh = false) {
  const box = $('#preview');
  if (!box) return;
  const pv = await api('schedule/preview?days=14&step_min=30' + (refresh ? '&refresh=true' : ''));
  const cal = pv.calendar || {};
  const status = $('#cal-status');
  if (status) status.textContent = !cal.entity ? 'no calendar set' : cal.error ? `calendar: ${cal.error}` : cal.fetched ? `calendar read ${ago(cal.fetched)}` : '';
  const L = S.config.layout;
  const fnIdx = Object.fromEntries(S.config.functions.map((f, i) => [f.id, i]));
  const per = 24 * 60 / pv.step_min;
  let html = `<div class="legend" style="margin:0 0 8px">${S.config.functions.map((f, i) => `<span><i style="background:${slot(i)};height:10px;width:10px"></i>${esc(f.name)}</span>`).join('')}</div>`;
  if (S.config.airing?.enabled && pv.no_airing) {
    // when the plan may suggest airing: the rules' and the calendar's "no airing" times
    const evAir = (cal.events || []).filter(e => e.understood && e.effect === 'no airing suggested');
    html += `<div style="margin-bottom:12px"><h3 style="margin:6px 0">Airing</h3><div class="legend" style="margin:0 0 6px"><span><i style="background:var(--good);opacity:.55;height:10px;width:10px"></i>may be suggested</span><span><i style="background:var(--grid);height:10px;width:10px"></i>not then</span></div><div class="heat">`;
    for (let d = 0; d < pv.no_airing.length / per; d++) {
      html += `<div class="heat-row"><span>${fmtDay(pv.start + d * 864e5 + 6 * H, S.tz)}</span><div class="heat-cells" style="grid-template-columns:repeat(${per},1fr)">`;
      for (let k = d * per; k < (d + 1) * per; k++) {
        const t = pv.start + k * pv.step_min * MIN, no = pv.no_airing[k];
        const ev = no && evAir.some(e => e.start < t + pv.step_min * MIN && e.end > t);
        html += `<span class="${ev ? 'ev' : ''}" style="background:${no ? 'var(--grid)' : 'color-mix(in srgb, var(--good) 55%, transparent)'}" title="${fmtTime(t, S.tz)} ${no ? 'no airing' + (ev ? ' (calendar)' : '') : 'airing may be suggested'}"></span>`;
      }
      html += `</div></div>`;
    }
    html += `<div class="heat-row"><span></span><div class="row hint" style="justify-content:space-between"><span>0:00</span><span>6:00</span><span>12:00</span><span>18:00</span><span>24:00</span></div></div></div></div>`;
  }
  html += `<div class="two">`;
  for (const z of L.zones) {
    const fn = pv.function[z.id];
    const lockUnits = L.units.filter(u => u.zone === z.id).map(u => pv.locked[u.id]);
    const hasComfort = L.units.some(u => u.zone === z.id && S.config.mapping?.units?.[u.id]?.proxy) || S.config.comfort?.[z.id] !== undefined;
    if (!hasComfort) { html += `<div><h3 style="margin:6px 0">${esc(z.name)}</h3><p class="hint">No comfort temperature: no comfort cost here.</p></div>`; continue; }
    html += `<div><h3 style="margin:6px 0">${esc(z.name)}</h3><div class="heat">`;
    for (let d = 0; d < fn.length / per; d++) {
      const day = fmtDay(pv.start + d * 864e5 + 6 * H, S.tz);
      html += `<div class="heat-row"><span>${day}</span><div class="heat-cells" style="grid-template-columns:repeat(${per},1fr)">`;
      for (let k = d * per; k < (d + 1) * per; k++) {
        const locked = lockUnits.some(l => l?.[k]);
        const fromEvent = pv.event_zone?.[z.id]?.[k] || L.units.some(u => u.zone === z.id && pv.event_unit?.[u.id]?.[k]);
        const off = pv.offset[z.id][k];
        const t = pv.start + k * pv.step_min * MIN;
        html += `<span class="${locked ? 'locked' : ''} ${fromEvent ? 'ev' : ''}" style="background:${slot(fnIdx[fn[k]] ?? 0)}" title="${fmtTime(t, S.tz)} ${esc(fnName(fn[k]))}${off ? ' ' + (off > 0 ? '+' : '') + off + ' K' : ''}${locked ? ' · units off' : ''} (${esc(pv.source[z.id][k])})"></span>`;
      }
      html += `</div></div>`;
    }
    html += `<div class="heat-row"><span></span><div class="row hint" style="justify-content:space-between"><span>0:00</span><span>6:00</span><span>12:00</span><span>18:00</span><span>24:00</span></div></div></div></div>`;
  }
  html += '</div>';
  if (cal.entity) {
    const evs = cal.events || [];
    html += `<h3 style="margin:16px 0 6px">Calendar events</h3>`;
    html += evs.length ? `<div class="table-wrap"><table><tr><th>When</th><th>Title</th><th>What the optimizer makes of it</th></tr>${evs.map(e => {
      const sameDay = fmtDay(e.start, S.tz) === fmtDay(e.end - 1, S.tz);
      const when = `${fmtDay(e.start, S.tz)} ${fmtTime(e.start, S.tz)}–${sameDay ? '' : fmtDay(e.end, S.tz) + ' '}${fmtTime(e.end, S.tz)}`;
      return `<tr><td>${esc(when)}</td><td>${esc(e.summary)}</td><td>${e.understood ? esc(e.effect) : '<span class="hint">not understood, ignored (expected for example “Wohnzimmer: Relaxed”)</span>'}</td></tr>`;
    }).join('')}</table></div>` : '<p class="hint">No events in the next two weeks.</p>';
  }
  box.innerHTML = html;
}

// ---- model ----------------------------------------------------------------------
const STAGES = {
  export: 'Reading the recorded history',
  grid: 'Laying it onto a 5-minute grid',
  fit: 'Fitting the model to it',
  check: 'Predicting the held-back days',
};
function trainingProgress(pg) {
  // a share for the bar: the fit's evaluation budget is an upper bound, it often stops early
  const share = !pg ? 0 : pg.stage === 'export' ? 0.02 : pg.stage === 'grid' ? 0.05
    : pg.stage === 'fit' ? 0.05 + 0.9 * Math.min(1, (pg.evaluations || 0) / (pg.budget || 1)) : 0.97;
  const el = pg?.elapsed_s ?? 0;
  const detail = pg?.stage === 'fit' && pg.budget ? ` · ${pg.evaluations} of at most ${pg.budget} model runs` : '';
  return `<div class="progress"><span style="width:${Math.round(share * 100)}%"></span></div>
    <p class="hint" style="margin:6px 0 0">${esc(STAGES[pg?.stage] || 'Starting')}${detail} · ${Math.floor(el / 60)}:${String(Math.floor(el % 60)).padStart(2, '0')} min</p>`;
}

let modelPoll = null;
pages.model = async () => {
  const root = $('#page-model');
  clearTimeout(modelPoll);
  await loadBase();
  const m = await api('model');
  const rep = m.report;
  const L = S.config.layout;
  let html = `<div class="toolbar"><h2 style="margin:0">Model</h2><span class="spacer"></span>
    <button id="diag" title="The recorded history, settings without secrets, the model and its report, as one file">Download diagnostics</button>
    <button id="train" ${m.training ? 'disabled' : ''}>Retrain now</button>
    <button id="train-fresh" ${m.training ? 'disabled' : ''}>Train from scratch</button></div>
    <div id="diag-msg"></div>`;
  if (m.training) html += `<div class="card" id="train-card"><h3>Training</h3><div id="train-progress">${trainingProgress(m.progress)}</div></div>`;
  if (S.trainDone) { html += `<div class="msg ok">${esc(S.trainDone)}</div>`; S.trainDone = null; }
  if (m.error) html += `<div class="msg err">Last training failed: ${esc(m.error)}</div>`;
  if (!rep) {
    html += `<div class="msg">Not trained yet${m.training ? ' — training is running' : ''}. The optimizer uses a generic apartment until then. Training reads the recorded history (up to ${S.config.history_days} days) and runs every night at ${S.config.retrain_hour}:00.</div>`;
  } else {
    const days = rep.days || {};
    html += `<div class="stats" style="margin-bottom:14px">
      <div class="stat"><div class="k">Trained</div><div class="v">${ago(rep.finished || rep.trained_at * 1000)}</div><div class="s">${(days.train || []).length} days to learn, ${(days.val || []).length} to check · ${num(rep.seconds, 0)} s</div></div>
      <div class="stat"><div class="k">Heat pump efficiency</div><div class="v">COP ${num(rep.cop7, 1)}</div><div class="s">at 7 °C outside, 21 °C inside</div></div>
      <div class="stat"><div class="k">Electricity on check days</div><div class="v">${num(rep.energy?.sim, 1)} kWh</div><div class="s">${rep.energy?.observed ? 'measured ' + num(rep.energy.obs, 1) + ' kWh' : 'no phase measurement'}</div></div>
    </div>`;
    for (const w of rep.warnings || []) html += `<div class="msg">${esc(w)}</div>`;
    html += `<div class="card"><h3>Rooms</h3><div class="table-wrap"><table><tr><th>Room</th><th class="num">Error on check days</th><th class="num">Bias</th><th class="num">Heat stored</th><th class="num">Heat loss</th><th class="num">Time constant</th></tr>`;
    L.zones.forEach((z, i) => {
      const zs = rep.zones?.[z.id] || {};
      html += `<tr><td>${esc(z.name)}</td><td class="num">${num(rep.rmse_val?.[i], 2)} K</td><td class="num">${num(rep.bias_val?.[i], 2)} K</td><td class="num">${num(zs.C, 1)} kWh/K</td><td class="num">${num((zs.Hloss || 0) * 1000, 0)} W/K</td><td class="num">${num(zs.tau_h, 0)} h</td></tr>`;
    });
    html += `</table></div><p class="hint">Error: how far whole days predicted from midnight, from nothing but weather and what the units did, stayed from the thermometers.</p></div>`;
    html += `<div class="card"><h3>Units</h3><div class="table-wrap"><table><tr><th>Unit</th><th>Heat per fan level</th></tr>`;
    for (const u of L.units) {
      const p = m.params.units[u.id] || {};
      html += `<tr><td>${esc(u.name)}</td><td>${p.heat ? p.heat.map((v, i) => `level ${i}: ${num(v, 2)} kW`).join(' · ') : num(p.kw, 2) + ' kW'}</td></tr>`;
    }
    html += `</table></div><p class="hint">Heating output at 7 °C outside; it falls ${num((m.params.plant.cap_slope || 0) * 100, 1)} % per kelvin colder.</p></div>`;
    const pc = rep.plant;
    if (pc) {
      const pl = m.params.plant;
      html += `<div class="card"><h3>Outdoor unit</h3><div class="two"><div>
        <div class="stats" style="grid-template-columns:repeat(auto-fill,minmax(120px,1fr))">
          <div class="stat"><div class="k">Base while running</div><div class="v">${num(pl.base ?? pl.standby, 2)} kW</div></div>
          <div class="stat"><div class="k">Electric limit</div><div class="v">${pl.Pmax ? num(pl.Pmax, 2) + ' kW' : '–'}</div><div class="s">${pc.hours_at_limit > 0.5 ? `reached for ${num(pc.hours_at_limit, 1)} h` : 'not reached yet: at least this'}</div></div>
          <div class="stat"><div class="k">Extra at high load</div><div class="v">${num((pl.kq || 0) * 1000, 1)} W/kW²</div></div>
        </div>
        <p class="hint">What all units together draw on the phase, against the heat they deliver: a base whenever the outdoor unit runs at all, the heat over the efficiency, and an extra that grows with the square of the total, up to the limit. Little heat is dear by the base, a lot by the extra; the plan bundles heating where it can and keeps clear of full load. The dots are five-minute steps the phase measured, with heat as the model reckons it (it is not measured).</p>
      </div><div><div id="plant-chart"></div><p class="hint" style="margin:4px 0 0">Electricity (kW) against heat (kW), at 7 °C outside and 21 °C inside.</p></div></div></div>`;
    }
    if (rep.validation?.length) {
      html += `<div class="card"><div class="toolbar"><h3 style="margin:0">Check days: predicted from midnight against measured</h3><span class="spacer"></span><div class="chips" id="val-days"></div></div>
        <div class="toolbar"><div class="chips" id="val-zones"></div></div><div id="val-chart"></div></div>`;
    }
  }
  root.innerHTML = html;
  $('#train').onclick = () => api('model/train', { method: 'POST' }).then(() => pages.model());
  $('#train-fresh').onclick = () => api('model/train?fresh=true', { method: 'POST' }).then(() => pages.model());
  $('#diag').onclick = async () => {
    const b = $('#diag'), box = $('#diag-msg');
    b.disabled = true;
    box.innerHTML = '<div class="msg">Collecting…</div>';
    try {
      const blob = await api('diagnostics?days=' + S.config.history_days, { blob: true });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `climate-optimizer-diagnostics-${new Date().toISOString().slice(0, 10)}.json.gz`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      box.innerHTML = `<div class="msg ok">Saved ${(blob.size / 1e6).toFixed(1)} MB. It holds the recorded temperatures, unit states and power, the settings without tokens, and the model.</div>`;
    } catch (e) { box.innerHTML = `<div class="msg err">${esc(e.message)}</div>`; }
    b.disabled = false;
  };
  if (m.training) {
    // follow the training, and show the new model when it is done
    const poll = async () => {
      if (current !== 'model') return;
      try {
        const n = await api('model');
        if (n.training) {
          const box = $('#train-progress');
          if (box) box.innerHTML = trainingProgress(n.progress);
          modelPoll = setTimeout(poll, 2000);
        } else {
          S.trainDone = n.error ? null : `Training finished.`;
          pages.model();
        }
      } catch { modelPoll = setTimeout(poll, 5000); }
    };
    modelPoll = setTimeout(poll, 2000);
  }
  if (rep?.plant && $('#plant-chart')) {
    const pc = rep.plant;
    const xs = pc.points.map(p => p[0]), ys = pc.points.map(p => p[1]);
    const xmax = Math.max(1, ...xs, ...pc.q) * 1.05;
    curveChart($('#plant-chart'), {
      series: [{ x: pc.q, y: pc.p, color: slot(0) }],
      dots: [{ x: xs, y: ys, color: css('--ink-2') }],
      hlines: pc.pmax ? [{ y: pc.pmax, label: 'limit' }] : [],
      domain: [0, xmax], height: 200, xFmt: v => v + ' kW', yFmt: v => v.toFixed(1),
    });
  }
  if (rep?.validation?.length) {
    S.valZone = S.valZone || L.zones[0].id;
    const nd = rep.validation.length;
    S.valDay = Math.min(S.valDay ?? nd - 1, nd - 1);
    const zi = L.zones.findIndex(z => z.id === S.valZone);
    const rmse = d => {
      let s2 = 0, n = 0;
      d.pred.forEach((r, k) => { const mv = d.meas[k]?.[zi]; if (mv !== null && mv !== undefined) { s2 += (r[zi] - mv) ** 2; n++; } });
      return n ? Math.sqrt(s2 / n) : null;
    };
    const dchips = $('#val-days');
    dchips.innerHTML = rep.validation.map((d, i) => `<button class="chip ${S.valDay === i ? 'active' : ''}" data-d="${i}">${fmtDay(d.t0 + 12 * H, S.tz)} · ${num(rmse(d), 2)} K</button>`).join('');
    dchips.querySelectorAll('button').forEach(b => b.onclick = () => { S.valDay = +b.dataset.d; pages.model(); });
    const chips = $('#val-zones');
    chips.innerHTML = L.zones.map(z => `<button class="chip ${S.valZone === z.id ? 'active' : ''}" data-z="${z.id}"><i style="background:${zoneColor(z.id)}"></i>${esc(z.name)}</button>`).join('');
    chips.querySelectorAll('button').forEach(b => b.onclick = () => { S.valZone = b.dataset.z; pages.model(); });
    const day = rep.validation[S.valDay];
    const t = day.pred.map((_, k) => day.t0 + k * 5 * MIN);
    const g = new ChartGroup($('#val-chart'), { x0: t[0], x1: t[t.length - 1], now: t[t.length - 1] + 1, tz: S.tz });
    g.add({ title: `${zoneName(S.valZone)} (°C)`, unit: '°C', height: 200 })
      .line({ name: 'Measured', color: css('--muted'), t, y: day.meas.map(r => r?.[zi] ?? null) })
      .line({ name: 'Predicted', color: zoneColor(S.valZone), t, y: day.pred.map(r => r[zi]) });
    g.panels.forEach(p => p.draw());
  }
};

// ---- settings ---------------------------------------------------------------------
const SETTING_FIELDS = [
  ['horizon_min_h', 'Plan at least (h)', 1], ['horizon_max_h', 'Plan at most (h)', 1],
  ['min_run_min', 'Minimum run (min)', 1], ['min_off_min', 'Minimum pause (min)', 1],
  ['start_cost', 'Cost of a compressor start (€)', 0.01], ['heater_start_cost', 'Cost of a heater start (€)', 0.001],
  ['terminal_quantile', 'Value heat left at the end at this price quantile', 0.05],
  ['window_assume_open_min', 'An open window closes after (min)', 1],
  ['heat_below_outdoor', 'Heat only if colder outside than (°C)', 0.5], ['cool_above_outdoor', 'Cool only if warmer outside than (°C)', 0.5],
  ['time_limit_s', 'Solver time limit (s)', 1], ['binary_h', 'On/off decided as such for (h)', 0.05],
];

pages.settings = async () => {
  const root = $('#page-settings');
  await loadBase();
  if (!S.entities) S.entities = await api('entities').catch(() => []);
  const C = JSON.parse(JSON.stringify(S.config));
  const L = C.layout;
  const M = C.mapping = C.mapping || {};
  M.zones = M.zones || {}; M.units = M.units || {};
  const field = (label, path, kind) => `<label class="field">${label}${entitySelect(path, kind, get(C, path) || '')}</label>`;
  let html = `<div class="toolbar"><h2 style="margin:0">Settings</h2><span class="spacer"></span><span id="st-msg" class="hint"></span><button class="primary" id="st-save">Save</button></div>`;
  html += `<div class="card"><h3>Mode</h3>
    <label class="check"><input type="checkbox" data-path="shadow" ${C.shadow ? 'checked' : ''}> Shadow mode: plan and show, but let the proxies’ own thermostats decide</label>
    <p class="hint">Set each proxy to <b>auto</b> and point its <code>optimizer_url</code> here: <code>${esc(location.origin + location.pathname.replace(/\/$/, ''))}</code>. In shadow mode the optimizer answers <code>follow: false</code>.</p>
    <h3 style="margin-top:14px">In shadow mode</h3>
    <p class="hint">To watch the optimizer beside another automation that switches the units and moves their setpoints: plan these units even when their proxy is off or not in auto, and use these comfort temperatures instead of the proxies’ targets. Outside shadow mode the proxies decide both.</p>
    <div class="table-wrap"><table><tr><th>Unit</th><th>Plan it anyway</th><th>Comfort temperature of its room</th></tr>
    ${L.units.map((u, i) => {
      const first = L.units.findIndex(v => v.zone === u.zone) === i;
      const zname = (L.zones.find(z => z.id === u.zone) || {}).name || u.zone;
      return `<tr><td>${esc(u.name)}</td><td><label class="check"><input type="checkbox" data-shadow-unit="${u.id}" ${(C.shadow_units || []).includes(u.id) ? 'checked' : ''}> even when its proxy is off</label></td>
        <td>${first ? `<input type="number" step="0.5" style="width:90px" data-shadow-comfort="${u.zone}" value="${C.shadow_comfort?.[u.zone] ?? ''}" placeholder="proxy’s"> °C <span class="hint">${esc(zname)}</span>` : ''}</td></tr>`;
    }).join('')}</table></div>
    <div class="fields" style="margin-top:12px"><label class="field">API token for proxies and this page<input type="password" data-path="api_token" value="${esc(C.api_token)}" placeholder="none"></label></div></div>`;
  html += `<div class="card"><h3>Home Assistant</h3><div class="fields wide">
    <label class="field">URL<input data-path="ha.url" value="${esc(C.ha.url)}" placeholder="http://homeassistant.local:8123" ${C.env?.ha_url ? 'disabled' : ''}>${C.env?.ha_url ? '<span class="hint">set by the container’s HA_URL</span>' : ''}</label>
    <label class="field">Long-lived access token<input type="password" data-path="ha.token" value="${esc(C.ha.token)}" ${C.env?.ha_token ? 'disabled' : ''}>${C.env?.ha_token ? '<span class="hint">set by the container’s HA_TOKEN</span>' : ''}</label></div>
    <label class="check" style="margin-top:8px"><input type="checkbox" data-path="ha.verify_ssl" ${C.ha.verify_ssl ? 'checked' : ''}> Verify TLS certificate</label>
    <p class="hint">${S.status.ha.connected ? 'Connected to Home Assistant ' + esc(S.status.ha.version) + '.' : 'Not connected: ' + esc(S.status.ha.error || '') + '. The entity lists below fill once it is.'}</p></div>`;
  html += `<div class="card"><h3>Rooms</h3><div class="table-wrap"><table><tr><th>Room</th><th>Thermometer</th><th>Window and door sensors</th><th>Windows face</th><th>Names in calendar events</th></tr>`;
  for (const z of L.zones) {
    M.zones[z.id] = M.zones[z.id] || {};
    const fac = z.facades || [];
    html += `<tr><td>${esc(z.name)}</td><td>${entitySelect(`mapping.zones.${z.id}.thermometer`, 'temp', M.zones[z.id].thermometer || '')}</td>
      <td>${z.windows ? entityMulti(`mapping.zones.${z.id}.windows`, 'opening', M.zones[z.id].windows) : '<span class="hint">none in the layout</span>'}</td>
      <td>${z.solar === false ? '<span class="hint">no windows</span>' : `<div class="days" style="flex-wrap:nowrap">${['N', 'E', 'S', 'W'].map(f => `<label><input type="checkbox" data-facade="${z.id}" value="${f}" ${fac.includes(f) ? 'checked' : ''}>${f}</label>`).join('')}</div>`}</td>
      <td><input data-alias="${z.id}" value="${esc((z.aliases || []).join(', '))}" placeholder="${esc({ office: 'Büro', bedroom: 'Schlafzimmer', living: 'Wohnzimmer, Küche', bath: 'Bad', hall: 'Flur' }[z.id] || 'more names')}" style="width:100%;min-width:120px"></td></tr>`;
  }
  html += `</table></div><p class="hint">Windows face: where each room gets its sun from. Ticked, the model only looks for direct sun from those sides (north gets none), which it learns much faster; none ticked, it looks everywhere. Retrain after changing it.</p></div>`;
  html += `<div class="card"><h3>Units</h3><div class="table-wrap"><table><tr><th>Unit</th><th>Proxy (what the optimizer answers)</th><th>Device under the proxy</th><th>${L.units.some(u => u.kind === 'heater') ? 'Heater energy (kWh) or power (W)' : ''}</th></tr>`;
  for (const u of L.units) {
    M.units[u.id] = M.units[u.id] || {};
    html += `<tr><td>${esc(u.name)}<div class="hint">${u.kind === 'ac' ? u.levels + ' fan levels' : 'heater'}</div></td>
      <td>${entitySelect(`mapping.units.${u.id}.proxy`, 'climate', M.units[u.id].proxy || '')}</td>
      <td>${entitySelect(`mapping.units.${u.id}.source`, u.kind === 'ac' ? 'climate' : 'switch', M.units[u.id].source || '')}
      ${u.kind === 'ac' ? `<input data-path="mapping.units.${u.id}.fan_modes" data-list="1" data-fan-for="${u.id}" value="${esc((M.units[u.id].fan_modes || []).join(', '))}" style="width:100%;margin-top:4px">
        <div class="hint" data-fan-hint="${u.id}"></div>` : ''}</td>
      <td>${u.kind === 'heater' ? entitySelect(`mapping.units.${u.id}.power`, 'power', M.units[u.id].power || '') : ''}</td></tr>`;
  }
  html += `</table></div><p class="hint">Fan modes: the device’s fan modes the proxy uses, quietest first (its <code>source_fan_modes</code>); level 0 is the first. Left empty, the optimizer takes the list the proxy sends.</p></div>`;
  html += `<div class="card"><h3>Outside, power and prices</h3><div class="fields wide">
    ${field('Outdoor thermometer', 'mapping.outdoor', 'temp')}
    ${field('Weather (forecast)', 'mapping.weather', 'weather')}
    ${field('Phase with the outdoor unit: energy (kWh) or power (W)', 'mapping.phase', 'power')}
    <label class="field">Other phases: energy (kWh) or power (W)${entityMulti('mapping.phases_other', 'power', M.phases_other)}</label>
    ${field('Car charger: energy (kWh) or power (W), optional', 'mapping.car', 'power')}
    <label class="field">Prices from<select data-path="prices.source"><option value="tibber_prices" ${C.prices.source === 'tibber_prices' ? 'selected' : ''}>tibber_prices.get_price</option><option value="attributes" ${C.prices.source === 'attributes' ? 'selected' : ''}>a sensor's attributes</option></select></label>
    <label class="field">tibber_prices entry id (if several)<input data-path="prices.entry_id" value="${esc(C.prices.entry_id || '')}"></label>
    ${field('Price sensor', 'prices.sensor', 'price')}
    </div><p class="hint">For the phases, the car and the heater an energy meter (kWh) is the better choice where there is one: the power over each five minutes comes from the difference of its readings, which counts everything, where a power sensor's reports miss what happens between them. Outdoor thermometer: best one in the shade. One in the sun reads high in the afternoon (on the south balcony up to 5 K), and the model takes that for warm weather; without one, the weather entity's temperature is used. The price sensor is also recorded for the history. With "a sensor's attributes", any list of entries with a start time and a price is understood (Nord Pool, EPEX Spot, Tibber).</p></div>`;
  html += `<div class="card"><h3>Optimizer</h3><div class="fields">${SETTING_FIELDS.map(([k, label, step]) => `<label class="field">${label}<input type="number" step="${step}" data-path="settings.${k}" value="${C.settings[k]}"></label>`).join('')}
    <label class="field">Season<select data-path="settings.season">${['auto', 'heat', 'cool'].map(v => `<option ${C.settings.season === v ? 'selected' : ''}>${v}</option>`).join('')}</select></label></div>
    <p class="hint">Starts cost wear: a compressor start is priced so the plan does not switch for a few cents. On/off is decided as such for the first quarter hour; after that the plan counts what share of each step a unit runs.</p></div>`;
  html += `<div class="card"><h3>Training</h3><div class="fields">
    <label class="field">Retrain every night at (hour)<input type="number" min="0" max="23" data-path="retrain_hour" value="${C.retrain_hour}"></label>
    <label class="field">History kept and trained on (days)<input type="number" min="3" max="365" data-path="history_days" value="${C.history_days}"></label></div>
    <div class="row" style="margin-top:10px"><label class="field">Load an export (browser snippet or diagnostics file)<input type="file" id="import-file" accept=".json,.gz,application/json,application/gzip"></label><span id="import-msg" class="hint"></span></div></div>`;
  html += `<div class="card"><h3>Layout (advanced)</h3><p class="hint">Rooms, how they connect, and which units heat them. Changing it means retraining.</p>
    <textarea id="layout-json" rows="12">${esc(JSON.stringify(C.layout, null, 2))}</textarea>
    <div class="fields" style="margin-top:8px"><label class="field">Latitude (default: Home Assistant’s)<input type="number" step="0.1" id="lat" value="${C.location?.lat ?? ''}"></label><label class="field">Longitude<input type="number" step="0.1" id="lon" value="${C.location?.lon ?? ''}"></label></div></div>`;
  root.innerHTML = html;
  root.querySelectorAll('[data-multi]').forEach(initMulti);
  root.querySelectorAll('select[data-kind="power"]').forEach(sel => { powerCheck(sel); sel.addEventListener('change', () => powerCheck(sel)); });
  // what each AC device offers, next to its fan modes
  const fanHint = uid => {
    const src = root.querySelector(`[data-path="mapping.units.${uid}.source"]`)?.value;
    const offered = (S.entities || []).find(e => e.id === src)?.fan_modes;
    const inp = root.querySelector(`[data-fan-for="${uid}"]`);
    const hint = root.querySelector(`[data-fan-hint="${uid}"]`);
    inp.placeholder = 'empty: as the proxy reports them';
    hint.innerHTML = offered?.length ? `The device offers: ${offered.map(esc).join(', ')}` : '';
  };
  for (const u of L.units.filter(u => u.kind === 'ac')) {
    fanHint(u.id);
    root.querySelector(`[data-path="mapping.units.${u.id}.source"]`).addEventListener('change', () => fanHint(u.id));
  }

  $('#import-file').onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;
    $('#import-msg').textContent = 'Loading…';
    try {
      const r = await api('import', { method: 'POST', body: await f.arrayBuffer(), raw: true });
      $('#import-msg').textContent = `${r.rows} rows loaded. Retrain on the Model page.`;
    } catch (err) { $('#import-msg').textContent = 'Failed: ' + err.message; }
  };
  $('#st-save').onclick = async () => {
    const msg = $('#st-msg');
    root.querySelectorAll('[data-path]').forEach(inp => {
      if (inp.dataset.multi !== undefined) { set(C, inp.dataset.path, JSON.parse(inp.dataset.multi)); return; }
      let v = inp.type === 'checkbox' ? inp.checked : inp.type === 'number' ? (inp.value === '' ? null : +inp.value) : inp.value.trim();
      if (inp.dataset.list) v = String(v).split(',').map(s => s.trim()).filter(Boolean);
      set(C, inp.dataset.path, v);
    });
    try {
      C.layout = JSON.parse($('#layout-json').value);
    } catch (e) { msg.textContent = 'Layout is not valid JSON'; return; }
    for (const z of C.layout.zones || []) {
      const boxes = [...root.querySelectorAll(`[data-facade="${z.id}"]`)];
      if (boxes.length) z.facades = boxes.filter(b => b.checked).map(b => b.value);
      const al = root.querySelector(`[data-alias="${z.id}"]`);
      if (al) z.aliases = al.value.split(',').map(x => x.trim()).filter(Boolean);
    }
    C.shadow_units = [...root.querySelectorAll('[data-shadow-unit]')].filter(b => b.checked).map(b => b.dataset.shadowUnit);
    C.shadow_comfort = Object.fromEntries([...root.querySelectorAll('[data-shadow-comfort]')].filter(i => i.value !== '').map(i => [i.dataset.shadowComfort, +i.value]));
    const lat = $('#lat').value, lon = $('#lon').value;
    C.location = lat !== '' && lon !== '' ? { lat: +lat, lon: +lon } : null;
    try {
      S.config = await api('config', { method: 'PUT', body: JSON.stringify(C) });
      msg.textContent = 'Saved.';
      setTimeout(() => pages.settings(), 800);
    } catch (e) { msg.textContent = 'Not saved: ' + e.message; }
  };
};

function get(o, path) { return path.split('.').reduce((a, k) => a?.[k], o); }
function set(o, path, v) {
  const ks = path.split('.');
  let a = o;
  for (const k of ks.slice(0, -1)) a = a[k] = a[k] || {};
  a[ks.at(-1)] = v;
}

// ---- start ----------------------------------------------------------------------
(async () => {
  try {
    await loadBase();
    S.tz = undefined;
    try {
      const hc = S.status?.tz;
      if (hc) S.tz = hc;
    } catch { }
  } catch (e) { console.error(e); }
  route();
  setInterval(async () => {
    try {
      S.status = await api('status');
      pills();
      if (current === 'dashboard' && S.plan && S.status.plan.created && S.status.plan.created !== S.plan.created) pages.dashboard();
    } catch { }
  }, 30000);
})();
