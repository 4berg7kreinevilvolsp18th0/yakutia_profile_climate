"""Собрать офлайн HTML-дашборд для отправки без сервера/туннеля."""
from __future__ import annotations

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "gdex_outputs" / "результаты-алдан" / "daily_profiles.json"
OUT_PATH = ROOT / "gdex_outputs" / "результаты-алдан" / "aldan_dashboard.html"
REQUIRED_SCHEMA = "observations_v1"


def main() -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if data.get("schema") != REQUIRED_SCHEMA:
        raise ValueError(
            f"Нужна схема {REQUIRED_SCHEMA}, получена {data.get('schema')!r}. "
            "Сначала пересоберите daily_profiles.json."
        )
    payload = {
        "station": data.get("station_name", "Aldan"),
        "schema": data["schema"],
        "months": data["months"],
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    plotly_js = get_plotlyjs()

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Алдан — суточные профили</title>
<script>{plotly_js}</script>
<style>
  body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f7f8fa;color:#1a2332}}
  header{{padding:16px 20px;background:#fff;border-bottom:1px solid #e5e7eb}}
  h1{{margin:0;font-size:20px}}
  .sub{{margin-top:4px;color:#5b6573;font-size:13px}}
  .wrap{{display:grid;grid-template-columns:280px 1fr;gap:16px;padding:16px}}
  .panel{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px}}
  label{{display:block;font-size:12px;color:#5b6573;margin:8px 0 4px}}
  select,button{{width:100%;padding:8px;margin-bottom:8px;box-sizing:border-box}}
  .days{{max-height:55vh;overflow:auto;border-top:1px solid #eee;padding-top:8px}}
  .day{{display:flex;gap:8px;align-items:center;font-size:13px;margin:4px 0}}
  #chart{{height:75vh}}
  .stats{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px}}
  .stat{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;min-width:120px}}
  .stat b{{display:block;font-size:18px}}
  .stat span{{font-size:12px;color:#5b6573}}
  @media (max-width:900px){{.wrap{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <h1>Алдан — суточные температурные профили</h1>
  <div class="sub">Автономный офлайн-файл: выберите месяц и отключите дни-выбросы. Интернет и сервер не требуются.</div>
</header>
<div class="wrap">
  <div class="panel">
    <label>Год</label>
    <select id="year"></select>
    <label>Месяц</label>
    <select id="month"></select>
    <button id="all" type="button">Все дни</button>
    <button id="none" type="button">Сброс</button>
    <button id="outliers" type="button">Выключить выбросы (RMS≥8°C)</button>
    <div class="days" id="days"></div>
  </div>
  <div>
    <div class="stats" id="stats"></div>
    <div class="panel"><div id="chart"></div></div>
  </div>
</div>
<script>
const DATA = {payload_json};
const THRESH = 8.0;
const years = [...new Set(Object.keys(DATA.months).map(m => m.slice(0,4)))].sort();
const yearSel = document.getElementById('year');
const monthSel = document.getElementById('month');
const daysBox = document.getElementById('days');
years.forEach(y => {{
  const o = document.createElement('option');
  o.value = y; o.textContent = y; yearSel.appendChild(o);
}});
yearSel.value = years[years.length - 1];

function monthsForYear(y) {{
  return Object.keys(DATA.months).filter(m => m.startsWith(y)).sort();
}}
function fillMonths() {{
  monthSel.innerHTML = '';
  monthsForYear(yearSel.value).forEach(m => {{
    const o = document.createElement('option');
    o.value = m; o.textContent = m.slice(5); monthSel.appendChild(o);
  }});
}}
fillMonths();

function daySeries(day) {{
  const source = day.day_mean || day;
  const heights = Array.isArray(source.heights_m) ? source.heights_m : [];
  const temps = Array.isArray(source.temperature_c) ? source.temperature_c : [];
  const byHeight = new Map();
  for (let i = 0; i < Math.min(heights.length, temps.length); i++) {{
    const h = Number(heights[i]);
    const t = Number(temps[i]);
    if (Number.isFinite(h) && Number.isFinite(t)) byHeight.set(h, t);
  }}
  const pairs = [...byHeight.entries()].sort((a, b) => a[0] - b[0]);
  if (pairs.length < 2) return null;
  return {{
    heights_m: pairs.map(pair => pair[0]),
    temperature_c: pairs.map(pair => pair[1]),
  }};
}}

function interp(xh, xt, grid) {{
  const y = new Array(grid.length).fill(null);
  for (let i = 0; i < grid.length; i++) {{
    const g = grid[i];
    if (g <= xh[0]) {{ y[i] = xt[0]; continue; }}
    if (g >= xh[xh.length - 1]) {{ y[i] = xt[xt.length - 1]; continue; }}
    let j = 0;
    while (j < xh.length - 1 && xh[j + 1] < g) j++;
    const t = (g - xh[j]) / (xh[j + 1] - xh[j]);
    y[i] = xt[j] * (1 - t) + xt[j + 1] * t;
  }}
  return y;
}}
function monthMean(days, enabled) {{
  const active = days
    .filter(d => enabled.has(d.date))
    .map(daySeries)
    .filter(series => series !== null);
  if (!active.length) return null;
  const minH = Math.min(...active.map(series => series.heights_m[0]));
  const maxH = Math.max(...active.map(series => series.heights_m[series.heights_m.length - 1]));
  const grid = Array.from({{length: 40}}, (_, i) => minH + (maxH - minH) * i / 39);
  const stack = active.map(series => interp(series.heights_m, series.temperature_c, grid));
  const mean = grid.map((_, i) => {{
    let s = 0, n = 0;
    for (const row of stack) {{
      const v = row[i];
      if (Number.isFinite(v)) {{ s += v; n++; }}
    }}
    return n ? s / n : null;
  }});
  return {{grid, mean}};
}}
function rms(day, mean) {{
  const series = daySeries(day);
  if (!series) return Infinity;
  const t = interp(series.heights_m, series.temperature_c, mean.grid);
  let s = 0, n = 0;
  for (let i = 0; i < t.length; i++) {{
    if (Number.isFinite(t[i]) && Number.isFinite(mean.mean[i])) {{
      const d = t[i] - mean.mean[i];
      s += d * d; n++;
    }}
  }}
  return n ? Math.sqrt(s / n) : Infinity;
}}

function render() {{
  const key = monthSel.value;
  const days = DATA.months[key].days;
  const enabled = new Set([...daysBox.querySelectorAll('input[type=checkbox]:checked')].map(c => c.value));
  const mean = monthMean(days, enabled);
  const traces = [];
  const palette = ['#4C78A8','#F58518','#E45756','#72B7B2','#54A24B','#EECA3B','#B279A2','#FF9DA6','#9D755D','#BAB0AC'];
  days.forEach((d, i) => {{
    if (!enabled.has(d.date)) return;
    const series = daySeries(d);
    if (!series) return;
    traces.push({{
      x: series.temperature_c, y: series.heights_m, mode: 'lines', name: d.date.slice(8),
      line: {{width: 1.5, color: palette[i % palette.length]}}, opacity: 0.85
    }});
  }});
  if (mean) {{
    traces.push({{
      x: mean.mean, y: mean.grid, mode: 'lines', name: 'Среднее',
      line: {{width: 3.5, color: '#C44E52'}}
    }});
  }}
  Plotly.newPlot('chart', traces, {{
    title: DATA.station + ' — ' + key + ' (суточные средние)',
    xaxis: {{title: 'Температура, °C'}},
    yaxis: {{title: 'Высота, м'}},
    margin: {{l: 50, r: 20, t: 50, b: 40}},
    legend: {{x: 1.02, y: 1}},
  }}, {{responsive: true}});

  const inv = days.filter(d => enabled.has(d.date) && d.inversion_detected).length;
  const nProf = days.filter(d => enabled.has(d.date)).reduce((a, d) => a + d.n_profiles, 0);
  const ts = mean && mean.mean[0] != null ? mean.mean[0].toFixed(1) : '—';
  document.getElementById('stats').innerHTML =
    '<div class="stat"><b>' + enabled.size + '/' + days.length + '</b><span>дней включено</span></div>' +
    '<div class="stat"><b>' + nProf + '</b><span>профилей (сроков)</span></div>' +
    '<div class="stat"><b>' + inv + '</b><span>с инверсией</span></div>' +
    '<div class="stat"><b>' + ts + '</b><span>Ts среднего, °C</span></div>';
}}

function fillDays() {{
  const days = DATA.months[monthSel.value].days;
  daysBox.innerHTML = '';
  days.forEach(d => {{
    const row = document.createElement('label');
    row.className = 'day';
    const inv = d.inversion_detected ? ' · inv' : '';
    row.innerHTML = '<input type="checkbox" value="' + d.date + '" checked> <span>' +
      d.date.slice(8) + ' · Ts=' + d.t_surface_c + '°C · n=' + d.n_profiles + inv + '</span>';
    row.querySelector('input').addEventListener('change', render);
    daysBox.appendChild(row);
  }});
  render();
}}

yearSel.onchange = () => {{ fillMonths(); fillDays(); }};
monthSel.onchange = fillDays;
document.getElementById('all').onclick = () => {{
  daysBox.querySelectorAll('input').forEach(c => c.checked = true); render();
}};
document.getElementById('none').onclick = () => {{
  daysBox.querySelectorAll('input').forEach(c => c.checked = false); render();
}};
document.getElementById('outliers').onclick = () => {{
  const days = DATA.months[monthSel.value].days;
  const all = new Set(days.map(d => d.date));
  const mean = monthMean(days, all);
  daysBox.querySelectorAll('input').forEach(c => {{
    const day = days.find(d => d.date === c.value);
    c.checked = !(mean && rms(day, mean) >= THRESH);
  }});
  render();
}};
fillDays();
</script>
</body>
</html>
"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "size_mb": round(OUT_PATH.stat().st_size / 1e6, 2)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
