"""
MRI Calibration Plotter & Temperature Estimator
================================================
Reads a GSP/PVP phantom calibration xlsx and provides three modes:

  plot      — interactive HTML showing all calibration curves
  estimate  — estimate temperature from a measured ADC value for one formulation
  vials     — batch estimate temperatures from a CSV of vial ADC measurements,
               with a visualisation overlaid on the calibration curves

Usage:
    python calibration_plotter.py plot <calibration.xlsx> [output.html]

    python calibration_plotter.py estimate <calibration.xlsx> <formulation> --D <val>
        <formulation> can be an exact name, case-insensitive substring, or index number.
        Use "list" to print available formulations.
        --D accepts both mm²/s (e.g. 0.00196) and 10⁻³ mm²/s (e.g. 1.96) — auto-detected.

    python calibration_plotter.py vials <calibration.xlsx> <vials.csv> <phantom_config.json> --phantom <name> [output.html]
        vials.csv           — two-column CSV: vial, ADC  (mm²/s or 10⁻³ mm²/s, auto-detected)
        phantom_config.json — JSON describing all phantoms:
                              {
                                "SPIRIT": {"5% Aqueous PVP NiCl2": ["E"], ...},
                                "120E":   {"5% Aqueous PVP NiCl2": ["G","M"], ...}
                              }
        --phantom           — which phantom to use (must match a key in phantom_config.json)

Notes:
    - Temperature estimation uses ADC only (most sensitive and monotonic parameter).
    - T1 and T2 are present in the parsed data but not plotted or used for estimation.
    - ADC units are auto-detected: values < 0.1 treated as mm²/s and converted to 10⁻³ mm²/s.
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParameterEstimate:
    """Temperature estimate from ADC."""

    param: str  # always "D"
    measured: float  # measured ADC in 10⁻³ mm²/s
    temperature: float  # estimated temperature (°C)
    temp_unc: float  # ±1σ uncertainty (°C)
    fit_residual: float  # RMS residual of polynomial fit (10⁻³ mm²/s)
    poly_degree: int  # polynomial degree used
    t_range: tuple  # (T_min, T_max) of calibration data


@dataclass
class TemperatureEstimate:
    """Temperature estimate for a single formulation from ADC."""

    formulation: str
    vial: Optional[str] = None
    estimate: Optional[ParameterEstimate] = None

    def summary(self) -> str:
        prefix = f"Vial {self.vial} · " if self.vial else ""
        lines = [f"{prefix}Formulation : {self.formulation}"]
        lines.append("-" * 60)
        if self.estimate:
            e = self.estimate
            lines.append(
                f"  ADC = {e.measured:.4f} × 10⁻³ mm²/s"
                f"  →  T = {e.temperature:.2f} ± {e.temp_unc:.2f} °C"
            )
            lines.append(
                f"  (degree-{e.poly_degree} poly, fit residual = {e.fit_residual:.4g}, "
                f"valid T range {e.t_range[0]:.0f}–{e.t_range[1]:.0f} °C)"
            )
        else:
            lines.append("  No estimate available.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing — calibration spreadsheet
# ---------------------------------------------------------------------------


def parse_calibration_xlsx(filepath: str) -> dict:
    """
    Parse the multi-formulation calibration spreadsheet.
    Returns dict: { formulation_name: [ {"temperature": float, "D_val": float, ...} ] }
    """
    xl = pd.ExcelFile(filepath)
    sheet = xl.parse(xl.sheet_names[0], header=None)

    header_row = None
    for i, row in sheet.iterrows():
        if any("Formulation" in str(v) for v in row.values if pd.notna(v)):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row containing 'Formulation Name'")

    data_start = header_row + 2
    headers = sheet.iloc[header_row].tolist()

    col_map = {"formulation": None, "temperature": None}
    for idx, h in enumerate(headers):
        h_str = str(h).strip().lower()
        if "formulation" in h_str:
            col_map["formulation"] = idx
        elif "temperature" in h_str:
            col_map["temperature"] = idx

    temp_col = col_map["temperature"]
    numeric_cols = [
        i
        for i in range(len(headers))
        if i > temp_col and str(headers[i]).strip() not in ("", "nan")
    ]

    for pi, pair in enumerate(["D", "T1", "T2"]):
        col_map[f"{pair}_val"] = (
            numeric_cols[2 * pi] if 2 * pi < len(numeric_cols) else None
        )
        col_map[f"{pair}_unc"] = (
            numeric_cols[2 * pi + 1] if 2 * pi + 1 < len(numeric_cols) else None
        )

    formulations = {}
    current_name = None

    for _, row in sheet.iloc[data_start:].reset_index(drop=True).iterrows():
        vals = row.tolist()
        fname = (
            vals[col_map["formulation"]] if col_map["formulation"] is not None else None
        )
        if pd.notna(fname) and str(fname).strip():
            current_name = str(fname).strip()
        if current_name is None:
            continue
        try:
            temp = float(vals[col_map["temperature"]])
        except (ValueError, TypeError):
            continue
        record = {"temperature": temp}
        for param in ["D", "T1", "T2"]:
            for suffix in ["_val", "_unc"]:
                key = f"{param}{suffix}"
                try:
                    record[key] = (
                        float(vals[col_map[key]]) if col_map[key] is not None else None
                    )
                except (ValueError, TypeError):
                    record[key] = None
        formulations.setdefault(current_name, []).append(record)

    return formulations


# ---------------------------------------------------------------------------
# Parsing — vials CSV and vial map JSON
# ---------------------------------------------------------------------------


def parse_vials_csv(filepath: str) -> dict:
    """
    Parse a two-column CSV (vial, ADC).
    Returns dict: { vial_label: adc_raw_float }
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    return {
        str(row[df.columns[0]]).strip(): float(row[df.columns[1]])
        for _, row in df.iterrows()
    }


def load_phantom_config(filepath: str) -> dict:
    """Load phantom config JSON.  Returns the full dict (all phantoms)."""
    with open(filepath) as f:
        return json.load(f)


def build_vial_map(phantom_config: dict, phantom_name: str) -> dict:
    """
    Resolve a flat {vial_label: formulation_name} dict for a given phantom.
    Raises KeyError with a helpful message if the phantom name is not found.
    """
    available = list(phantom_config.keys())
    if phantom_name not in phantom_config:
        raise KeyError(
            f"Phantom {phantom_name!r} not found in config. " f"Available: {available}"
        )
    flat = {}
    for formulation, vials in phantom_config[phantom_name].items():
        for vial in vials:
            flat[vial.strip()] = formulation
    return flat


def autoconvert_adc(value: float) -> float:
    """Convert mm²/s → 10⁻³ mm²/s if the value looks like mm²/s (< 0.1)."""
    return value * 1000.0 if value < 0.1 else value


# ---------------------------------------------------------------------------
# Formulation resolution
# ---------------------------------------------------------------------------


def list_formulations(formulations: dict) -> None:
    print("Available formulations:")
    for i, (name, rows) in enumerate(formulations.items()):
        temps = [r["temperature"] for r in rows if np.isfinite(r["temperature"])]
        print(
            f"  [{i}] {name!r}  ({len(temps)} points, T = {min(temps):.0f}–{max(temps):.0f} °C)"
        )


def resolve_formulation(formulations: dict, query: str) -> str:
    """Match by exact name, case-insensitive substring, or integer index."""
    names = list(formulations.keys())
    try:
        return names[int(query)]
    except (ValueError, IndexError):
        pass
    if query in formulations:
        return query
    matches = [n for n in names if query.lower() in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous query {query!r}. Matches:\n"
            + "\n".join(f"  {m}" for m in matches)
        )
    raise ValueError(
        f"No formulation matching {query!r}.\nAvailable:\n"
        + "\n".join(f"  {n}" for n in names)
    )


# ---------------------------------------------------------------------------
# Temperature estimation — ADC only
# ---------------------------------------------------------------------------


def _best_poly_degree(n: int) -> int:
    if n <= 3:
        return 1
    if n <= 6:
        return 2
    return 3


def _fit_and_invert(
    temps: np.ndarray,
    values: np.ndarray,
    uncertainties: np.ndarray,
    measured: float,
) -> ParameterEstimate:
    """
    Fit polynomial to ADC vs T calibration data, invert to find T.
    Uncertainty estimated via Monte Carlo (2000 samples).
    """
    mask = np.isfinite(temps) & np.isfinite(values) & np.isfinite(uncertainties)
    temps, values, uncertainties = temps[mask], values[mask], uncertainties[mask]

    n = len(temps)
    if n < 2:
        raise ValueError(f"Insufficient valid calibration points ({n})")

    degree = _best_poly_degree(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(temps, values, degree)
    poly = np.poly1d(coeffs)

    residual = float(np.sqrt(np.mean((poly(temps) - values) ** 2)))
    t_min, t_max = float(temps.min()), float(temps.max())

    cal_vals = [float(poly(t)) for t in temps]
    cal_min, cal_max = min(cal_vals), max(cal_vals)
    margin = 0.1 * (cal_max - cal_min)
    if not (cal_min - margin <= measured <= cal_max + margin):
        raise ValueError(
            f"Measured ADC {measured:.4f} is outside calibration range "
            f"[{cal_min:.4f}, {cal_max:.4f}] for T = {t_min:.1f}–{t_max:.1f} °C"
        )

    def solve_T(val: float, p: np.poly1d) -> Optional[float]:
        f = lambda t: p(t) - val
        lo, hi = t_min - 5, t_max + 5
        try:
            if f(lo) * f(hi) < 0:
                return float(brentq(f, lo, hi))
        except Exception:
            pass
        return None

    t_est = solve_T(measured, poly)
    if t_est is None:
        raise ValueError(f"Could not bracket root for ADC = {measured:.4f}")

    N_MC = 2000
    t_samples = []
    rng = np.random.default_rng(42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(N_MC):
            v_p = values + rng.normal(0, uncertainties)
            m_p = measured + rng.normal(0, max(residual, 1e-9))
            c_p = np.polyfit(temps, v_p, degree)
            t2 = solve_T(m_p, np.poly1d(c_p))
            if t2 is not None:
                t_samples.append(t2)

    if len(t_samples) < 10:
        raise ValueError("Monte Carlo uncertainty estimation failed (too few samples)")

    return ParameterEstimate(
        param="D",
        measured=measured,
        temperature=t_est,
        temp_unc=float(np.std(t_samples)),
        fit_residual=residual,
        poly_degree=degree,
        t_range=(t_min, t_max),
    )


def estimate_temperature(
    formulations: dict,
    formulation_query: str,
    D: float,
    vial: Optional[str] = None,
) -> TemperatureEstimate:
    """
    Estimate temperature from a measured ADC value.

    Parameters
    ----------
    formulations : dict
        From parse_calibration_xlsx().
    formulation_query : str
        Name, substring, or integer index.
    D : float
        Measured ADC — auto-converted from mm²/s if needed.
    vial : str, optional
        Label for display/output purposes.
    """
    D_conv = autoconvert_adc(D)
    name = resolve_formulation(formulations, formulation_query)
    rows = formulations[name]

    pts = [
        (r["temperature"], r["D_val"], r["D_unc"])
        for r in rows
        if r["D_val"] is not None and r["D_unc"] is not None
    ]

    result = TemperatureEstimate(formulation=name, vial=vial)
    if len(pts) < 2:
        print(f"  Warning: insufficient calibration data for {name}")
        return result

    temps_arr = np.array([p[0] for p in pts])
    vals_arr = np.array([p[1] for p in pts])
    uncs_arr = np.array([p[2] for p in pts])

    try:
        result.estimate = _fit_and_invert(temps_arr, vals_arr, uncs_arr, D_conv)
    except ValueError as e:
        print(f"  Warning: {e}")

    return result


# ---------------------------------------------------------------------------
# Polynomial curve helper (dense fit line for HTML)
# ---------------------------------------------------------------------------


def _poly_curve(formulations: dict) -> dict:
    """Return dense (T, D) polynomial fit curve per formulation."""
    curves = {}
    for name, rows in formulations.items():
        pts = [
            (r["temperature"], r["D_val"])
            for r in rows
            if r["D_val"] is not None and np.isfinite(r["temperature"])
        ]
        if len(pts) < 2:
            continue
        temps = np.array([p[0] for p in pts])
        vals = np.array([p[1] for p in pts])
        mask = np.isfinite(temps) & np.isfinite(vals)
        temps, vals = temps[mask], vals[mask]
        degree = _best_poly_degree(len(temps))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(temps, vals, degree)
        poly = np.poly1d(coeffs)
        t_dense = np.linspace(temps.min(), temps.max(), 120)
        curves[name] = {"t": t_dense.tolist(), "d": poly(t_dense).tolist()}
    return curves


# ---------------------------------------------------------------------------
# Shared HTML fragments
# ---------------------------------------------------------------------------

_PALETTE = [
    "#378ADD",
    "#D85A30",
    "#1D9E75",
    "#D4537E",
    "#7F77DD",
    "#BA7517",
    "#639922",
    "#888780",
    "#185FA5",
    "#993C1D",
]

_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #ffffff; --bg2: #f4f4f2; --bg3: #e9e9e6;
    --text: #1a1a18; --text2: #5f5e5a; --border: rgba(0,0,0,0.12);
    --radius: 8px; --font: system-ui, -apple-system, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1c1c1a; --bg2: #252523; --bg3: #2e2e2b;
      --text: #e8e8e4; --text2: #888780; --border: rgba(255,255,255,0.10);
    }
  }
  body { font-family: var(--font); background: var(--bg); color: var(--text); padding: 24px; }
  h1   { font-size: 18px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: var(--text2); margin-bottom: 20px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 20px; }
  .ctrl { display: flex; flex-direction: column; gap: 4px; }
  label { font-size: 12px; color: var(--text2); font-weight: 500; }
  select {
    padding: 6px 10px; border: 0.5px solid var(--border); border-radius: var(--radius);
    background: var(--bg2); color: var(--text); font-size: 13px; cursor: pointer;
    appearance: none; padding-right: 28px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 8px center;
  }
  .toggle-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    display: flex; align-items: center; gap: 6px; padding: 5px 10px;
    border-radius: 99px; border: 0.5px solid var(--border); cursor: pointer;
    font-size: 12px; background: var(--bg2); user-select: none; transition: opacity 0.15s;
  }
  .chip.off { opacity: 0.35; }
  .chip-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .charts { display: grid; grid-template-columns: 1fr; gap: 24px; }
  @media (min-width: 900px) { .charts { grid-template-columns: 1fr 1fr; } }
  .chart-card {
    background: var(--bg2); border: 0.5px solid var(--border); border-radius: 12px; padding: 16px;
  }
  .chart-card.wide { grid-column: 1 / -1; }
  .chart-title { font-size: 13px; font-weight: 500; margin-bottom: 4px; }
  .chart-unit  { font-size: 11px; color: var(--text2); margin-bottom: 12px; }
  .chart-wrap  { position: relative; width: 100%; height: 300px; }
  .stats-section { margin-top: 28px; }
  .stats-title  { font-size: 13px; font-weight: 500; color: var(--text2); margin-bottom: 10px; }
  .stats-table  { width: 100%; border-collapse: collapse; font-size: 12px; }
  .stats-table th {
    text-align: left; padding: 6px 10px; background: var(--bg3);
    border-bottom: 0.5px solid var(--border); color: var(--text2); font-weight: 500;
  }
  .stats-table td { padding: 5px 10px; border-bottom: 0.5px solid var(--border); }
  .stats-table tr:last-child td { border-bottom: none; }
  .swatch { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
  .muted  { font-size: 11px; color: var(--text2); }
  .legend-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .legend-item {
    display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2);
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: var(--radius); padding: 4px 10px;
    cursor: pointer; user-select: none; transition: opacity 0.15s;
  }
  .legend-item.off { opacity: 0.3; }
  .vial-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    font-size: 10px; font-weight: 500; color: #fff; flex-shrink: 0;
  }
"""

_ERROR_BAR_PLUGIN_JS = """
const errorBarPlugin = {
  id: "errorBar",
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ctx.save(); ctx.strokeStyle = ds.borderColor; ctx.lineWidth = 1.2;
      Object.entries(ds.errorBars).forEach(([i, eb]) => {
        const el = meta.data[+i]; if (!el) return;
        const x = el.x, cap = 4;
        const yMin = chart.scales.y.getPixelForValue(eb.yMin);
        const yMax = chart.scales.y.getPixelForValue(eb.yMax);
        ctx.beginPath();
        ctx.moveTo(x, yMin); ctx.lineTo(x, yMax);
        ctx.moveTo(x-cap, yMin); ctx.lineTo(x+cap, yMin);
        ctx.moveTo(x-cap, yMax); ctx.lineTo(x+cap, yMax);
        ctx.stroke();
      });
      ctx.restore();
    });
  }
};
"""

_BASE_OPTS_JS = """
function baseOpts() {
  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const grid = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
  const tick = "#888780";
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}` } },
      zoom: {
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
        pan:  { enabled: true, mode: "xy" },
      }
    },
    scales: {
      x: { type: "linear",
           title: { display: true, text: "Temperature (°C)", color: tick, font: { size: 11 } },
           ticks: { color: tick, font: { size: 11 } }, grid: { color: grid } },
      y: { ticks: { color: tick, font: { size: 11 } }, grid: { color: grid } }
    }
  };
}
"""


# ---------------------------------------------------------------------------
# HTML — plot mode
# ---------------------------------------------------------------------------


def build_html(formulations: dict) -> str:
    data_json = json.dumps(formulations, indent=2)
    curves_json = json.dumps(_poly_curve(formulations))
    form_names = list(formulations.keys())
    colour_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(form_names)}
    colour_json = json.dumps(colour_map)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MRI Calibration Plotter</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/1.2.1/chartjs-plugin-zoom.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<h1>MRI Calibration — GSP / PVP Phantom</h1>
<p class="subtitle">ADC vs temperature · error bars = ±1× uncertainty · dashed line = polynomial fit · scroll to zoom · drag to pan · double-click to reset</p>

<div class="controls">
  <div class="ctrl">
    <label>Error bars</label>
    <select id="errMode">
      <option value="1">± 1×</option>
      <option value="2">± 2×</option>
      <option value="0">Hide</option>
    </select>
  </div>
  <div class="ctrl">
    <label>ADC fit line</label>
    <select id="fitMode">
      <option value="1">Show</option>
      <option value="0">Hide</option>
    </select>
  </div>
  <div class="ctrl">
    <label>Formulations</label>
    <div class="toggle-row" id="chips"></div>
  </div>
</div>

<div class="charts">
  <div class="chart-card wide">
    <div class="chart-title">Apparent Diffusion Coefficient (ADC)</div>
    <div class="chart-unit">10⁻³ mm²/s</div>
    <div class="chart-wrap" style="height:320px"><canvas id="chartD"></canvas></div>
  </div>

</div>

<div class="stats-section">
  <div class="stats-title">Summary at 20 °C (or nearest)</div>
  <table class="stats-table">
    <thead><tr>
      <th>Formulation</th><th>Temp (°C)</th>
      <th>ADC (10⁻³ mm²/s)</th>
    </tr></thead>
    <tbody id="statsBody"></tbody>
  </table>
</div>

<script>
const RAW     = {data_json};
const CURVES  = {curves_json};
const COLOURS = {colour_json};
const NAMES   = Object.keys(RAW);
let active    = new Set(NAMES);

const chipsEl = document.getElementById("chips");
NAMES.forEach(name => {{
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.innerHTML = `<span class="chip-dot" style="background:${{COLOURS[name]}}"></span>${{name}}`;
  chip.addEventListener("click", () => {{
    if (active.has(name)) {{ active.delete(name); chip.classList.add("off"); }}
    else {{ active.add(name); chip.classList.remove("off"); }}
    updateAll();
  }});
  chipsEl.appendChild(chip);
}});

document.getElementById("errMode").addEventListener("change", updateAll);
document.getElementById("fitMode").addEventListener("change", updateAll);

{_ERROR_BAR_PLUGIN_JS}
{_BASE_OPTS_JS}

function makeChart(id) {{
  const c = new Chart(document.getElementById(id).getContext("2d"), {{
    type: "line", data: {{ datasets: [] }}, options: baseOpts(), plugins: [errorBarPlugin],
  }});
  document.getElementById(id).addEventListener("dblclick", () => c.resetZoom());
  return c;
}}

const charts = {{ D: makeChart("chartD") }};

function datasetsForParam(param, errMult, showFit) {{
  const ds = [];
  NAMES.filter(n => active.has(n)).forEach(name => {{
    const pts = RAW[name].filter(r => r[`${{param}}_val`] !== null)
                         .sort((a, b) => a.temperature - b.temperature);
    const scatter = {{
      label: name,
      data: pts.map(r => ({{ x: r.temperature, y: r[`${{param}}_val`] }})),
      borderColor: COLOURS[name], backgroundColor: COLOURS[name] + "22",
      pointBackgroundColor: COLOURS[name], pointRadius: 4, pointHoverRadius: 6,
      borderWidth: 0, showLine: false,
    }};
    if (errMult > 0) {{
      scatter.errorBars = {{}};
      pts.forEach((r, i) => {{
        const u = r[`${{param}}_unc`];
        if (u != null) scatter.errorBars[i] = {{ yMin: r[`${{param}}_val`] - errMult*u, yMax: r[`${{param}}_val`] + errMult*u }};
      }});
    }}
    ds.push(scatter);
    if (showFit && param === "D" && CURVES[name]) {{
      const c = CURVES[name];
      ds.push({{
        label: name + " (fit)",
        data: c.t.map((t, i) => ({{ x: t, y: c.d[i] }})),
        borderColor: COLOURS[name] + "88", backgroundColor: "transparent",
        pointRadius: 0, borderWidth: 1.5, borderDash: [4, 3], tension: 0,
      }});
    }}
  }});
  return ds;
}}

function updateAll() {{
  const errMult = parseFloat(document.getElementById("errMode").value);
  const showFit = document.getElementById("fitMode").value === "1";
  charts.D.data.datasets = datasetsForParam("D", errMult, showFit);
  charts.D.update();
  updateStats();
}}

function updateStats() {{
  const tbody = document.getElementById("statsBody"); tbody.innerHTML = "";
  NAMES.filter(n => active.has(n)).forEach(name => {{
    const pts = RAW[name].filter(r => r.D_val !== null);
    if (!pts.length) return;
    const c = pts.reduce((b, r) => Math.abs(r.temperature-20) < Math.abs(b.temperature-20) ? r : b);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="swatch" style="background:${{COLOURS[name]}}"></span>${{name}}</td>
      <td>${{c.temperature.toFixed(1)}}</td>
      <td>${{c.D_val?.toFixed(4) ?? "—"}} <span class="muted">± ${{c.D_unc?.toFixed(4) ?? "?"}}</span></td>`;
    tbody.appendChild(tr);
  }});
}}

updateAll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTML — vials mode
# ---------------------------------------------------------------------------


def build_vials_html(
    formulations: dict, vial_results: list, phantom_name: str = ""
) -> str:
    data_json = json.dumps(formulations, indent=2)
    curves_json = json.dumps(_poly_curve(formulations))
    form_names = list(formulations.keys())
    colour_map = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(form_names)}
    colour_json = json.dumps(colour_map)

    # Vial marker colours — distinct from formulation colours
    vial_palette = [
        "#1a1a18",
        "#c0392b",
        "#16a085",
        "#8e44ad",
        "#d35400",
        "#2980b9",
        "#27ae60",
        "#f39c12",
    ]

    vial_data = []
    for idx, res in enumerate(vial_results):
        if res.estimate is None:
            continue
        vial_data.append(
            {
                "vial": res.vial,
                "formulation": res.formulation,
                "adc": res.estimate.measured,
                "T": res.estimate.temperature,
                "T_unc": res.estimate.temp_unc,
                "colour": vial_palette[idx % len(vial_palette)],
                "formColour": colour_map.get(res.formulation, "#888"),
            }
        )
    vial_json = json.dumps(vial_data)

    table_rows = []
    for res in vial_results:
        e = res.estimate
        if e:
            table_rows.append(
                f"<tr><td><strong>{res.vial}</strong></td>"
                f"<td>{res.formulation}</td>"
                f"<td>{e.measured:.4f}</td>"
                f"<td><strong>{e.temperature:.2f}</strong></td>"
                f"<td class='muted'>± {e.temp_unc:.2f} °C</td></tr>"
            )
        else:
            table_rows.append(
                f"<tr><td><strong>{res.vial}</strong></td>"
                f"<td>{res.formulation}</td>"
                f"<td colspan='3' class='muted'>No estimate</td></tr>"
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MRI Calibration — {phantom_name} Vial Temperature Estimates</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/1.2.1/chartjs-plugin-zoom.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<h1>MRI Calibration — {phantom_name + " " if phantom_name else ""}Vial Temperature Estimates</h1>
<p class="subtitle">ADC calibration curves with measured vial values overlaid · scroll to zoom · drag to pan · double-click to reset</p>

<div class="controls">
  <div class="ctrl">
    <label>Calibration curves</label>
    <div class="toggle-row" id="chips"></div>
  </div>
</div>

<div class="legend-grid" id="vialLegend"></div>

<div class="charts">
  <div class="chart-card wide">
    <div class="chart-title">ADC vs temperature — calibration + vial measurements</div>
    <div class="chart-unit">10⁻³ mm²/s</div>
    <div class="chart-wrap" style="height:440px"><canvas id="chartD"></canvas></div>
  </div>
</div>

<div class="stats-section">
  <div class="stats-title">Estimated temperatures by vial</div>
  <table class="stats-table">
    <thead><tr>
      <th>Vial</th><th>Formulation</th>
      <th>ADC (10⁻³ mm²/s)</th><th>Est. T (°C)</th><th>Uncertainty</th>
    </tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
</div>

<script>
const RAW     = {data_json};
const CURVES  = {curves_json};
const COLOURS = {colour_json};
const VIALS   = {vial_json};
const NAMES   = Object.keys(RAW);
let active    = new Set(NAMES);

// Chips
const chipsEl = document.getElementById("chips");
NAMES.forEach(name => {{
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.innerHTML = `<span class="chip-dot" style="background:${{COLOURS[name]}}"></span>${{name}}`;
  chip.addEventListener("click", () => {{
    if (active.has(name)) {{ active.delete(name); chip.classList.add("off"); }}
    else {{ active.add(name); chip.classList.remove("off"); }}
    updateChart();
  }});
  chipsEl.appendChild(chip);
}});

// Vial legend — toggleable chips
const legEl = document.getElementById("vialLegend");
let activeVials = new Set(VIALS.map(v => v.vial));

VIALS.forEach(v => {{
  const item = document.createElement("div");
  item.className = "legend-item";
  item.innerHTML =
    `<span class="vial-badge" style="background:${{v.colour}}">${{v.vial}}</span>` +
    `${{v.formulation}} &nbsp;·&nbsp; ${{v.adc.toFixed(4)}} &nbsp;→&nbsp; ` +
    `<strong>${{v.T.toFixed(2)}} °C</strong>` +
    `<span class="muted">&nbsp;± ${{v.T_unc.toFixed(2)}}</span>`;
  item.addEventListener("click", () => {{
    if (activeVials.has(v.vial)) {{ activeVials.delete(v.vial); item.classList.add("off"); }}
    else {{ activeVials.add(v.vial); item.classList.remove("off"); }}
    chart.update();   // triggers crosshairPlugin redraw
  }});
  legEl.appendChild(item);
}});

{_ERROR_BAR_PLUGIN_JS}
{_BASE_OPTS_JS}

// Crosshair + annotation plugin
const crosshairPlugin = {{
  id: "crosshair",
  afterDatasetsDraw(chart) {{
    const ctx  = chart.ctx;
    const xAx  = chart.scales.x;
    const yAx  = chart.scales.y;
    const top = yAx.top, bottom = yAx.bottom;
    const left = xAx.left, right = xAx.right;

    VIALS.filter(v => activeVials.has(v.vial)).forEach(v => {{
      const xPx = xAx.getPixelForValue(v.T);
      const yPx = yAx.getPixelForValue(v.adc);
      if (xPx < left || xPx > right || yPx < top || yPx > bottom) return;

      const col = v.colour;
      ctx.save();

      // Uncertainty shading
      const xLo = xAx.getPixelForValue(v.T - v.T_unc);
      const xHi = xAx.getPixelForValue(v.T + v.T_unc);
      ctx.fillStyle = col + "1a";
      ctx.fillRect(xLo, top, xHi - xLo, bottom - top);

      // Dashed vertical at T estimate
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = col + "99";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xPx, top); ctx.lineTo(xPx, bottom); ctx.stroke();

      // Dashed horizontal at measured ADC
      ctx.beginPath(); ctx.moveTo(left, yPx); ctx.lineTo(right, yPx); ctx.stroke();
      ctx.setLineDash([]);

      // Circle at intersection
      ctx.fillStyle = col;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(xPx, yPx, 5, 0, 2*Math.PI);
      ctx.fill(); ctx.stroke();

      // Label badge
      ctx.font = "500 11px system-ui, sans-serif";
      const tw = ctx.measureText(v.vial).width;
      const pad = 4, fs = 11;
      const bx = xPx + 8, by = yPx - 10;
      const bw = tw + pad*2, bh = fs + pad*2;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.roundRect(bx, by - bh/2, bw, bh, 3);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(v.vial, bx + pad, by);

      ctx.restore();
    }});
  }}
}};

const chart = new Chart(document.getElementById("chartD").getContext("2d"), {{
  type: "line",
  data: {{ datasets: [] }},
  options: baseOpts(),
  plugins: [errorBarPlugin, crosshairPlugin],
}});
document.getElementById("chartD").addEventListener("dblclick", () => chart.resetZoom());

function updateChart() {{
  const ds = [];
  NAMES.filter(n => active.has(n)).forEach(name => {{
    const pts = RAW[name].filter(r => r.D_val !== null).sort((a,b) => a.temperature - b.temperature);

    const scatter = {{
      label: name,
      data: pts.map(r => ({{ x: r.temperature, y: r.D_val }})),
      borderColor: COLOURS[name], backgroundColor: COLOURS[name] + "22",
      pointBackgroundColor: COLOURS[name], pointRadius: 3, borderWidth: 0, showLine: false,
    }};
    scatter.errorBars = {{}};
    pts.forEach((r, i) => {{
      if (r.D_unc != null)
        scatter.errorBars[i] = {{ yMin: r.D_val - r.D_unc, yMax: r.D_val + r.D_unc }};
    }});
    ds.push(scatter);

    if (CURVES[name]) {{
      const c = CURVES[name];
      ds.push({{
        label: name + " (fit)",
        data: c.t.map((t, i) => ({{ x: t, y: c.d[i] }})),
        borderColor: COLOURS[name] + "99", backgroundColor: "transparent",
        pointRadius: 0, borderWidth: 1.5, borderDash: [4, 3], tension: 0,
      }});
    }}
  }});

  chart.data.datasets = ds;
  chart.update();
}}

updateChart();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MRI Calibration Plotter & Temperature Estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser(
        "plot", help="Generate interactive HTML calibration plotter"
    )
    p_plot.add_argument("xlsx", help="Calibration spreadsheet (.xlsx)")
    p_plot.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Output HTML (default: <xlsx>_plotter.html)",
    )

    p_est = sub.add_parser(
        "estimate", help="Estimate temperature from a measured ADC value"
    )
    p_est.add_argument("xlsx", help="Calibration spreadsheet (.xlsx)")
    p_est.add_argument("formulation", help="Name, substring, index, or 'list'")
    p_est.add_argument(
        "--D",
        type=float,
        required=False,
        metavar="VAL",
        help="Measured ADC (mm²/s or 10⁻³ mm²/s — auto-detected)",
    )

    p_vials = sub.add_parser(
        "vials", help="Batch estimate temperatures from a vials CSV"
    )
    p_vials.add_argument("xlsx", help="Calibration spreadsheet (.xlsx)")
    p_vials.add_argument("csv", help="Vials CSV (columns: vial, ADC)")
    p_vials.add_argument("config", help="Phantom config JSON (all phantoms)")
    p_vials.add_argument(
        "--phantom", required=True, help="Phantom name to use (e.g. SPIRIT or 120E)"
    )
    p_vials.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Output HTML (default: vials_estimates.html)",
    )

    args = parser.parse_args()

    print(f"Parsing calibration: {args.xlsx}")
    formulations = parse_calibration_xlsx(args.xlsx)
    print(f"Found {len(formulations)} formulations.\n")

    if args.command == "plot":
        out = args.output or Path(args.xlsx).stem + "_plotter.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_html(formulations))
        print(f"HTML plotter written to: {out}")

    elif args.command == "estimate":
        if args.formulation.lower() == "list":
            list_formulations(formulations)
            return
        if args.D is None:
            parser.error("--D is required for the estimate command")
        result = estimate_temperature(formulations, args.formulation, D=args.D)
        print(result.summary())

    elif args.command == "vials":
        vials_adc = parse_vials_csv(args.csv)
        phantom_cfg = load_phantom_config(args.config)
        try:
            vial_map = build_vial_map(phantom_cfg, args.phantom)
        except KeyError as e:
            parser.error(str(e))

        print(f"Phantom : {args.phantom}  ({len(vial_map)} vials)\n")

        results = []
        for vial, adc_raw in vials_adc.items():
            form_query = vial_map.get(vial)
            if form_query is None:
                print(f"  Vial {vial}: not in phantom {args.phantom!r} — skipping")
                continue
            try:
                res = estimate_temperature(
                    formulations, str(form_query), D=adc_raw, vial=vial
                )
                print(res.summary())
                print()
                results.append(res)
            except ValueError as e:
                print(f"  Vial {vial}: {e}\n")

        out = args.output or "vials_estimates.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_vials_html(formulations, results, phantom_name=args.phantom))
        print(f"\nHTML estimates written to: {out}")


if __name__ == "__main__":
    main()
