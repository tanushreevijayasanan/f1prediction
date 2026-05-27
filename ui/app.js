'use strict';

/* ── Config ──────────────────────────────────────────── */
const WS_URL    = 'ws://localhost:8000/ws';
const LAP_WIN   = 20;
const TOP_N     = 5;

/* ── State ───────────────────────────────────────────── */
let ws;
let charts      = {};
let winHistory  = {};
let topDrivers  = [];
let latestState = null;
let renderScheduled = false;
const RENDER_INTERVAL_MS = 250;
let lastRenderMs = 0;

/* ── Driver colour palette ────────────────────────────── */
const DRIVER_COLORS = {
  VER:'#3b82f6', NOR:'#f97316', LEC:'#ef4444', HAM:'#94a3b8',
  RUS:'#c0c0c0', SAI:'#e53e3e', ALO:'#22c55e', PIA:'#fbbf24',
  GAS:'#8b5cf6', OCO:'#f43f5e', STR:'#10b981', HUL:'#06b6d4',
  BOT:'#64748b', ZHO:'#a3e635', MAG:'#f87171', TSU:'#818cf8',
  ALB:'#38bdf8', SAR:'#e879f9', DEV:'#fb923c', RIC:'#facc15',
};
const PALETTE = [
  '#4da6ff','#f97316','#22c55e','#f59e0b','#a78bfa',
  '#f472b6','#34d399','#fb923c','#60a5fa','#e879f9',
];
function driverColor(code) {
  return DRIVER_COLORS[code] || PALETTE[[...code].reduce((h,c)=>h+c.charCodeAt(0),0) % PALETTE.length];
}

/* ── Chart.js defaults ────────────────────────────────── */
Chart.defaults.color          = '#96a0af';
Chart.defaults.borderColor    = 'rgba(255,255,255,0.08)';
Chart.defaults.font.family    = "'Rajdhani', 'Segoe UI', sans-serif";
Chart.defaults.font.size      = 10;

/* ── Bootstrap ───────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  buildStaticCharts();
  connectWS();
});

/* ═══════════════════════════════════════════════════════
   WEBSOCKET
═══════════════════════════════════════════════════════ */
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen    = () => setConn('live', 'Live');
  ws.onmessage = ({ data }) => {
    try { scheduleRender(JSON.parse(data)); }
    catch (e) { console.error('Parse error', e); }
  };
  ws.onclose = () => {
    setConn('', 'Reconnecting');
    setTimeout(connectWS, 2500);
  };
  ws.onerror = () => setConn('error', 'Error');
}

function scheduleRender(state) {
  latestState = state;
  if (renderScheduled) return;
  renderScheduled = true;
  const now = Date.now();
  const wait = Math.max(0, RENDER_INTERVAL_MS - (now - lastRenderMs));
  setTimeout(() => {
    renderScheduled = false;
    lastRenderMs = Date.now();
    if (latestState) render(latestState);
  }, wait);
}

function setConn(cls, label) {
  const dot = document.getElementById('conn-dot');
  const lbl = document.getElementById('conn-label');
  dot.className = 'conn-dot' + (cls ? ' ' + cls : '');
  lbl.textContent = label;
}

/* ═══════════════════════════════════════════════════════
   MAIN RENDER
═══════════════════════════════════════════════════════ */
function render(state) {
  try { renderHeader(state);       } catch(e){ console.warn('header',e); }
  try { renderLeaderboard(state);  } catch(e){ console.warn('lb',e); }
  try { renderCommentary(state);   } catch(e){ console.warn('commentary',e); }
  try { renderPitStrategy(state);  } catch(e){ console.warn('pit',e); }
  try { renderAggregates(state);   } catch(e){ console.warn('agg',e); }
  try { renderWinProbChart(state); } catch(e){ console.warn('winprob',e); }
  try { renderLapTimeChart(state); } catch(e){ console.warn('laptime',e); }
  try { renderTireChart(state);    } catch(e){ console.warn('tire',e); }
  try { renderCornerChart(state);  } catch(e){ console.warn('corner',e); }
  try { renderAggressionChart(state); } catch(e){ console.warn('aggr',e); }
  try { renderSpeedTrace(state);   } catch(e){ console.warn('speedtrace',e); }
}

/* ── Header ──────────────────────────────────────────── */
function renderHeader(s) {
  setText('hdr-lap',   s.lap ?? '—');
  setText('race-name', s.event || '— Grand Prix');
  setText('race-meta', `${s.year ?? ''} · Race · Lap ${s.lap ?? '—'} / 50`);

  const scBanner = document.getElementById('sc-banner');
  if (scBanner) scBanner.classList.toggle('visible', !!s.sc_active);

  if (Number.isFinite(Number(s.avg_speed))) {
    setText('hdr-avgspeed', Number(s.avg_speed).toFixed(1));
  }
  if (Number.isFinite(Number(s.top_speed))) {
    setText('hdr-tspeed', Math.round(Number(s.top_speed)));
  }
  if (Number.isFinite(Number(s.avg_lap_ms))) {
    setText('hdr-avglap', (Number(s.avg_lap_ms) / 1000).toFixed(2));
  }
}

/* ── Leaderboard ─────────────────────────────────────── */
function renderLeaderboard(s) {
  const tbody = document.getElementById('lb-body');
  if (!tbody) return;
  const rows = s.leaderboard_rows || [];
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="13" class="empty-state">Awaiting data...</td></tr>';
    return;
  }
  tbody.innerHTML = rows.slice(0, 22).map((r, i) => {
    const driver = r.driver || '---';
    const pos = Number(r.position || (i + 1));
    const prevPos = Number(r.prev_position || pos);
    const predPos = Number(r.pred_position || pos);
    const pace = Number(r.pace_score || 0);
    const tireSym = (r.tire_symbol || '?').toUpperCase();
    const tireName = (r.tire_name || 'UNKNOWN').toUpperCase();
    const age = Number(r.tire_age || 0);
    const deg = Number(r.deg_delta || 0);
    const currLt = Number(r.curr_laptime_ms || 0);
    const nextLt = Number(r.next_laptime_ms || 0);
    const gapS = Number(r.gap_s || 0);
    const pitProb = Number(r.pit_prob || 0);
    const pitAlert = !!r.pit_alert;
    const winPct = Number(r.win_prob || 0) * 100;

    const posClass = pos===1?'pos-1':pos===2?'pos-2':pos===3?'pos-3':'pos-n';
    const tyreClass= tireSym==='S'?'tyre-s':tireSym==='M'?'tyre-m':tireSym==='H'?'tyre-h':tireSym==='I'?'tyre-i':'tyre-w';
    const pitPct = Math.round(pitProb * 100);
    const gap = pos === 1 || gapS < 0.05 ? '+Leader' : `+${gapS.toFixed(3)}s`;
    const currText = currLt > 0 ? formatLap(currLt) : '—';
    const nextText = nextLt > 0 ? formatLap(nextLt) : '—';
    let predPText = ` ${predPos}`;
    if (predPos < pos) predPText = `▲${predPos}`;
    if (predPos > pos) predPText = `▼${predPos}`;
    const predPClass = predPos < pos ? 'urg-low' : predPos > pos ? 'urg-high' : 'urg-med';
    const alertText = pitAlert ? '!' : '';

    return `<tr>
      <td><span class="pos-num ${posClass}">${pos}</span></td>
      <td><div class="driver-cell"><span class="driver-code" style="color:${driverColor(driver)}">${driver}</span></div></td>
      <td><span class="win-pct">${winPct.toFixed(1)}%</span></td>
      <td><span class="${predPClass}">${predPText}</span></td>
      <td><span class="gap-val">${pace.toFixed(2)}</span></td>
      <td><span class="${tyreClass}">${tireSym}</span> <span class="driver-team">${tireName}</span></td>
      <td><span class="tyre-age">${age}</span></td>
      <td><span class="gap-val">${deg >= 0 ? '+' : ''}${deg.toFixed(1)}</span></td>
      <td><span class="laptime-val">${currText}</span></td>
      <td><span class="laptime-val">${nextText}</span></td>
      <td><span class="gap-val">${gap}</span></td>
      <td><span class="win-pct">${pitPct}%</span></td>
      <td><span class="laptime-val">${alertText}</span></td>
    </tr>`;
  }).join('');
}

/* ── Commentary ──────────────────────────────────────── */
let lastCommentary = [];
function renderCommentary(s) {
  const feed  = document.getElementById('commentary-feed');
  if (!feed) return;
  const items = s.commentary || [];
  if (!items.length) return;
  const key = items.join('|');
  if (key === lastCommentary.join('|')) return;
  lastCommentary = items;
  feed.innerHTML = items.map((msg, i) => {
    const cls = msg.toLowerCase().includes('pit') ? 'pit-event'
               : msg.toLowerCase().includes('safety') || msg.toLowerCase().includes('yellow') ? 'incident'
               : i === 0 ? 'fresh' : '';
    return `<div class="commentary-item ${cls}">
      <span class="comm-lap">LAP ${s.lap ?? '—'}</span>
      <span class="comm-text">${escHtml(msg)}</span>
    </div>`;
  }).join('');
}

/* ── Pit strategy ────────────────────────────────────── */
function renderPitStrategy(s) {
  const strategies = s.pit_strategies || {};
  const alerts     = document.getElementById('pit-alerts');
  const list       = document.getElementById('pit-list');
  if (!alerts || !list) return;
  const high = Object.entries(strategies).filter(([,p])=>p&&p.pit_urgency==='HIGH');
  alerts.innerHTML = high.length
    ? `<div class="pit-alert-banner">HIGH: ${high.map(([d])=>d).join(', ')} - pit window open</div>` : '';
  const candidates = Object.entries(strategies)
    .filter(([,p])=>p&&p.pit_probability>0.25)
    .sort(([,a],[,b])=>b.pit_probability-a.pit_probability).slice(0, 6);
  if (!candidates.length) { list.innerHTML = '<div class="pit-empty">No imminent stops flagged</div>'; return; }
  list.innerHTML = candidates.map(([driver, pit]) => {
    const prob    = pit.pit_probability || 0;
    const pct     = Math.round(prob * 100);
    const urgency = pit.pit_urgency || 'LOW';
    const color   = urgency==='HIGH'?'var(--red)':urgency==='MEDIUM'?'var(--amber)':'var(--green)';
    const compound= (pit.compound||'?')[0].toUpperCase();
    const deg     = typeof pit.tire_degradation==='number' ? pit.tire_degradation.toFixed(3) : '—';
    return `<div class="pit-card">
      <div class="pit-driver-info">
        <span class="pit-driver-name" style="color:${driverColor(driver)}">${driver}</span>
        <span class="pit-driver-detail">${compound} · Age ${pit.tire_age ?? '—'} laps · Deg ${deg}</span>
      </div>
      <div class="pit-right">
        <span class="pit-prob-num" style="color:${color}">${pct}%</span>
        <div class="pit-prob-track"><div class="pit-prob-fill" style="width:${pct}%;background:${color}"></div></div>
      </div>
    </div>`;
  }).join('');
}

/* ── Aggregates ──────────────────────────────────────── */
function renderAggregates(s) {
  if (!s.positions || !s.positions.length) return;
  const deltas  = s.positions.map(([,d])=>typeof d==='number'?d:0);
  const avg     = deltas.reduce((a,b)=>a+b,0) / deltas.length;
  const leader  = s.positions[0];
  const second  = s.positions[1];
  setText('avg-speed',  `${avg >= 0 ? '+' : ''}${avg.toFixed(1)}`);
  setText('leader-gap', leader&&second ? `${(leader[1]-second[1]).toFixed(1)} s` : '—');
  setText('pit-count',  Object.values(s.pit_strategies||{}).filter(p=>p&&p.pit_probability>0.5).length);
  setText('sc-active',  s.sc_active ? 'YES' : 'No');
}

/* ═══════════════════════════════════════════════════════
   LIVE CHART RENDERS
═══════════════════════════════════════════════════════ */

/* ── Win probability ─────────────────────────────────── */
function renderWinProbChart(s) {
  if (!s.predictions || !s.lap) return;
  const lap = s.lap;
  topDrivers = Object.entries(s.predictions)
    .sort(([,a],[,b])=>(b.win_prob||0)-(a.win_prob||0))
    .slice(0, TOP_N).map(([code])=>code);
  topDrivers.forEach(code => {
    if (!winHistory[code]) winHistory[code] = [];
    winHistory[code].push({ lap, pct: (s.predictions[code].win_prob || 0) * 100 });
    if (winHistory[code].length > LAP_WIN) winHistory[code].shift();
  });
  const allLaps = [...new Set(Object.values(winHistory).flat().map(p=>p.lap))].sort((a,b)=>a-b);
  const c = charts.winProb;
  if (!c) return;
  c.data.labels   = allLaps;
  c.data.datasets = topDrivers.map(code => {
    const map = Object.fromEntries((winHistory[code]||[]).map(p=>[p.lap, p.pct]));
    return { label: code, spanGaps: true, tension: .4, fill: false,
      borderColor: driverColor(code), borderWidth: 2, pointRadius: 0,
      data: allLaps.map(l => map[l] ?? null) };
  });
  c.update('none');
  buildLegend('winprob-legend', topDrivers.map(code=>({label:code, color:driverColor(code), type:'line'})));
}

/* ── Lap time evolution ──────────────────────────────── */
function renderLapTimeChart(s) {
  if (!s.lap_times_history) return;
  const c = charts.lapTime;
  if (!c) return;
  const history = s.lap_times_history;
  const codes   = Object.keys(history).slice(0, 5);
  const allLaps = [...new Set(codes.flatMap(code=>history[code].map(p=>p.lap)))].sort((a,b)=>a-b);
  c.data.labels   = allLaps;
  c.data.datasets = codes.map((code, i) => {
    const map = Object.fromEntries(history[code].map(p=>[p.lap, p.ms/1000]));
    return { label: code, spanGaps: true, tension: .4, fill: false,
      borderColor: driverColor(code), borderWidth: 2, pointRadius: 0,
      borderDash: i % 2 === 1 ? [4,2] : [],
      data: allLaps.map(l=>map[l]??null) };
  });
  c.update('none');
  buildLegend('laptime-legend', codes.map((code,i)=>({label:code, color:driverColor(code), type:i%2===1?'dash':'line'})));
}

/* ── Tire pace ───────────────────────────────────────── */
function renderTireChart(s) {
  if (!s.tire_history) return;
  const c = charts.tire;
  if (!c) return;
  const keys    = Object.keys(s.tire_history).slice(0, 6);
  const allLaps = [...new Set(keys.flatMap(k=>s.tire_history[k].map(p=>p.lap)))].sort((a,b)=>a-b);
  const COMPOUND_COLORS = { SOFT:'#ef4444', MEDIUM:'#f59e0b', HARD:'#9ca3af', INTER:'#3b82f6', WET:'#60a5fa' };
  c.data.labels   = allLaps;
  c.data.datasets = keys.map((key, i) => {
    const [driver, compound] = key.split('_');
    const map   = Object.fromEntries(s.tire_history[key].map(p=>[p.lap, p.speed/1000]));
    const color = COMPOUND_COLORS[compound] || driverColor(driver);
    return { label: `${driver} · ${compound}`, spanGaps: true, tension: .4, fill: false,
      borderColor: color, borderWidth: 2, pointRadius: 0,
      borderDash: i % 2 === 1 ? [5,3] : [],
      data: allLaps.map(l=>map[l]??null) };
  });
  c.update('none');
  buildLegend('tire-legend', keys.map((key,i)=>{
    const [driver,compound] = key.split('_');
    const CC = { SOFT:'#ef4444', MEDIUM:'#f59e0b', HARD:'#9ca3af', INTER:'#3b82f6', WET:'#60a5fa' };
    return { label:`${driver}·${compound}`, color: CC[compound]||driverColor(driver), type:i%2===1?'dash':'line' };
  }));
}

/* ── Corner chart (live) ─────────────────────────────── */
function renderCornerChart(s) {
  const data = s.corner_data;
  if (!data || !data.length) return;
  const c = charts.corner;
  if (!c) return;
  const labels   = data.map(d => `T${d.corner_id}`);
  const speeds   = data.map(d => d.speed   || 0);
  const throttle = data.map(d => d.throttle || 0);
  c.data.labels                = labels;
  c.data.datasets[0].data      = speeds;
  c.data.datasets[1].data      = throttle;
  c.update('none');
}

/* ── Speed trace (live) ──────────────────────────────── */
function renderSpeedTrace(s) {
  const trace = s.speed_trace;
  if (!trace || !trace.buckets || !trace.buckets.length) return;
  const c = charts.speedTrace;
  if (!c) return;

  const labels = trace.buckets.map(b => b.dist);
  const speeds = trace.buckets.map(b => b.speed);

  // Straight zone overlay as filled dataset
  const maxSpeed = Math.max(...speeds, 340);
  const zoneData = labels.map(dist => {
    const inZone = (trace.straight_zones || []).some(z => dist >= z.start && dist <= z.end);
    return inZone ? maxSpeed : null;
  });

  c.data.labels            = labels;
  c.data.datasets[0].data  = zoneData;   // straight zone fill
  c.data.datasets[1].data  = speeds;     // speed trace
  c.update('none');
}

/* -- Driver aggression (live) --------------------------------------------- */
function renderAggressionChart(s) {
  const c = charts.aggr;
  const data = s.aggression_data;
  if (!c || !data || !data.length) return;

  const rows = data
    .filter(d => d && d.driver)
    .slice(0, 10);

  c.data.labels = rows.map(d => d.driver);
  c.data.datasets[0].data = rows.map(d => Number(d.throttle_avg || 0));
  c.data.datasets[1].data = rows.map(d => Number(d.hard_brakes || 0));
  c.update('none');
}

/* ═══════════════════════════════════════════════════════
   STATIC CHART BUILDERS (called once on load)
═══════════════════════════════════════════════════════ */
function buildStaticCharts() {
  buildWinProbChart();
  buildLapTimeChart();
  buildCornerChart();
  buildSpeedTraceChart();
  buildTireChart();
  buildAggressionChart();
}

function buildWinProbChart() {
  const ctx = document.getElementById('winProbChart');
  if (!ctx) return;
  charts.winProb = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 60, title: { display: true, text: 'Win probability (%)', color:'#505060', font:{size:10} }, ticks:{color:'#505060'}, grid:{color:'rgba(255,255,255,0.04)'} },
        x: { title: { display: true, text: 'Lap', color:'#505060', font:{size:10} }, ticks:{color:'#505060'}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

function buildLapTimeChart() {
  const ctx = document.getElementById('lapTimeChart');
  if (!ctx) return;
  charts.lapTime = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { title:{display:true,text:'Lap time (s)',color:'#505060',font:{size:10}}, ticks:{color:'#505060',callback:v=>v.toFixed(1)}, grid:{color:'rgba(255,255,255,0.04)'} },
        x: { title:{display:true,text:'Lap',color:'#505060',font:{size:10}}, ticks:{color:'#505060',autoSkip:true,maxTicksLimit:10}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

function buildCornerChart() {
  const ctx = document.getElementById('cornerChart');
  if (!ctx) return;
  charts.corner = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: [],
      datasets: [
        { type: 'line', label: 'Avg speed', data: [], borderColor: '#4da6ff', tension: .4, pointRadius: 0, borderWidth: 2, yAxisID: 'y',  fill: false },
        { type: 'bar',  label: 'Throttle %', data: [], backgroundColor: 'rgba(249,115,22,0.65)', yAxisID: 'y1', borderRadius: 2 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y:  { position: 'left',  min: 0, max: 350, title:{display:true,text:'Speed (km/h)',color:'#505060',font:{size:10}}, ticks:{color:'#505060'}, grid:{color:'rgba(255,255,255,0.04)'} },
        y1: { position: 'right', min: 0, max: 110, grid:{drawOnChartArea:false}, title:{display:true,text:'Throttle %',color:'#505060',font:{size:10}}, ticks:{color:'#505060'} },
        x:  { title:{display:true,text:'Corner',color:'#505060',font:{size:10}}, ticks:{color:'#505060',autoSkip:true,maxTicksLimit:16}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

function buildSpeedTraceChart() {
  const ctx = document.getElementById('drsChart');
  if (!ctx) return;
  charts.speedTrace = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Straight zones', data: [], fill: true, backgroundColor: 'rgba(249,115,22,0.12)', borderColor: 'transparent', tension: 0, pointRadius: 0, spanGaps: false },
        { label: 'Speed',          data: [], borderColor: '#4da6ff', tension: .3, pointRadius: 0, borderWidth: 2, fill: false }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 80, max: 340, title:{display:true,text:'Speed (km/h)',color:'#505060',font:{size:10}}, ticks:{color:'#505060'}, grid:{color:'rgba(255,255,255,0.04)'} },
        x: { ticks:{color:'#505060',callback:(_,i,ticks)=>{const v=ticks[i]?.value;return v!==undefined&&v%1000===0?v+'m':null;},autoSkip:false,maxRotation:0}, grid:{color:'rgba(255,255,255,0.04)'}, title:{display:true,text:'Distance (m)',color:'#505060',font:{size:10}} }
      }
    }
  });
}

function buildTireChart() {
  const ctx = document.getElementById('tireChart');
  if (!ctx) return;
  charts.tire = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { title:{display:true,text:'Lap time (s)',color:'#505060',font:{size:10}}, ticks:{color:'#505060',callback:v=>v.toFixed(1)}, grid:{color:'rgba(255,255,255,0.04)'} },
        x: { title:{display:true,text:'Lap',color:'#505060',font:{size:10}}, ticks:{color:'#505060',autoSkip:true,maxTicksLimit:10}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

function buildAggressionChart() {
  const ctx = document.getElementById('aggrChart');
  if (!ctx) return;
  charts.aggr = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [
      { label:'Throttle avg', data:[], backgroundColor:'rgba(77,166,255,0.75)', borderRadius:3 },
      { label:'Hard brakes',  data:[], backgroundColor:'rgba(249,115,22,0.75)', borderRadius:3 },
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { min:0, title:{display:true,text:'Count / avg %',color:'#505060',font:{size:10}}, ticks:{color:'#505060'}, grid:{color:'rgba(255,255,255,0.04)'} },
        x: { ticks:{color:'#606070'}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

/* ═══════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════ */
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function escHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatLap(ms) {
  if (!ms) return '—';
  const s   = ms / 1000;
  const min = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3);
  return `${min}:${sec.padStart(6,'0')}`;
}

function buildLegend(id, items) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = items.map(item => {
    const indicator = item.type === 'dash'
      ? `<span class="leg-dash" style="border-color:${item.color}"></span>`
      : item.type === 'block'
      ? `<span class="leg-block" style="background:${item.color}"></span>`
      : `<span class="leg-line" style="background:${item.color}"></span>`;
    return `<span class="leg-item">${indicator}${escHtml(item.label)}</span>`;
  }).join('');
}
