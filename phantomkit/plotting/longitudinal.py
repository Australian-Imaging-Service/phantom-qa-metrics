#!/usr/bin/env python3
"""
longitudinal.py — Track phantom vial measurements over time.

Reads the embedded phantomkit-data JSON from each HTML, extracts per-vial
measurements (ADC, fitted T1, or fitted T2), and produces a new self-contained
interactive HTML showing measurements across scan dates.

Features
--------
* X-axis: date of scan (from embedded scan_date or --date CLI option)
* Y-axis: metric value (ADC ×10⁻³ mm²/s, T1 ms, or T2 ms)
* Single-vial panel: vial selector dropdown with session toggle buttons
* All-vials panel: all vials shown simultaneously; each session = unique colour,
  each vial = unique shape; per-vial reference lines; toggleable sessions & vials
* Reference line(s): temperature-dependent from calibration data
* Toggle buttons to show/hide individual sessions (synced across both panels)

Registered as ``phantomkit plot longitudinal`` via CLI auto-discovery.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import click

from phantomkit.plotting.compare_plots import (
    _REFERENCE_COLOR,
    _SESSION_PALETTE,
    _Y_LABELS,
    _auto_template_dir,
    _extract,
    _load_embedded,
    _load_reference,
)

# Chart.js pointStyle names and matching Unicode glyphs for vial shapes
_VIAL_SHAPES = [
    "circle",       # ●
    "rect",         # ■
    "triangle",     # ▲
    "rectRot",      # ◆
    "star",         # ★
    "cross",        # ✚
    "crossRot",     # ✕
    "rectRounded",  # ▣
]
_VIAL_SYMBOLS = ["●", "■", "▲", "◆", "★", "✚", "✕", "▣"]


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------


def _date_to_ms(iso_str: str) -> int:
    """Convert YYYY-MM-DD to milliseconds since Unix epoch (UTC midnight)."""
    d = date.fromisoformat(iso_str)
    return int(datetime(d.year, d.month, d.day).timestamp() * 1000)


def _derive_labels(paths: list[str]) -> list[str]:
    """Same label-derivation logic as compare_plots."""
    if len(paths) == 1:
        return [Path(paths[0]).stem]
    parts_list = [Path(p).resolve().parts for p in paths]
    min_len = min(len(p) for p in parts_list)
    common_suffix_len = 0
    for i in range(1, min_len + 1):
        if len({p[-i] for p in parts_list}) == 1:
            common_suffix_len = i
        else:
            break
    labels = []
    for parts in parts_list:
        idx = len(parts) - common_suffix_len - 1
        labels.append(parts[idx] if idx >= 0 else parts[-1])
    return labels


def _vial_order(sessions: list[dict], ref_vials: list[str]) -> list[str]:
    """Reference vials first, then any measured-only vials."""
    seen: set[str] = set()
    result: list[str] = []
    for v in ref_vials:
        vu = v.upper()
        if vu not in seen:
            result.append(vu)
            seen.add(vu)
    for sess in sessions:
        for v in sess["vals"]:
            if v.upper() not in seen:
                result.append(v.upper())
                seen.add(v.upper())
    return result


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _build_html(
    metric: str,
    sessions: list[dict],     # [{label, color, date_ms, date_str, vals}]
    vial_list: list[str],
    ref_data: dict | None,
) -> str:
    from phantomkit.plotting._html_common import html_head

    y_label = _Y_LABELS.get(metric, metric)
    title = f"Longitudinal: {metric} per vial"

    # -- JS data blobs
    sessions_json    = json.dumps(sessions)
    vial_list_json   = json.dumps(vial_list)
    y_label_json     = json.dumps(y_label)
    ref_color_json   = json.dumps(_REFERENCE_COLOR)
    vial_shapes_json = json.dumps(_VIAL_SHAPES)

    ref_by_temp: dict[str, dict] = {}
    temperatures: list = []
    default_temp: str | None = None
    if ref_data is not None:
        default_temp = ref_data.get("default_temp")
        temperatures = ref_data.get("temperatures", [])
        ref_by_temp  = ref_data.get("values_by_temp", {})

    ref_by_temp_json  = json.dumps(ref_by_temp)
    default_temp_json = json.dumps(default_temp)

    # -- Temperature selector
    temp_selector_html = ""
    if ref_data is not None and temperatures:
        ref_units = ref_data.get("units", "")
        temp_options = "".join(
            f'<option value="{t}"{" selected" if str(t) == default_temp else ""}>{t} °C</option>'
            for t in temperatures
        )
        temp_selector_html = (
            '\n    <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;">'
            + '<span style="font-size:14px;color:var(--text2);">Reference temperature:</span>'
            + '<select id="refTempSelect" onchange="pkSetRefTemp(this.value)"'
            + ' style="background:var(--bg3);color:var(--text);border:1px solid var(--border);'
            + 'border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;">'
            + temp_options
            + '</select>'
            + f'<span style="font-size:12px;color:var(--text2);">{ref_units}</span>'
            + '</div>'
        )

    # -- Per-session toggle buttons
    session_controls_html = ""
    for i, sess in enumerate(sessions):
        c = sess["color"]
        lbl = sess["label"]
        date_str = sess.get("date_str", "")
        subtitle = f' <span style="font-size:12px;color:var(--text2);">({date_str})</span>' if date_str else ""
        session_controls_html += (
            f'\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'<span style="width:14px;height:14px;border-radius:50%;background:{c};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<button id="pk-sess-btn-{i}" data-visible="1" onclick="pkToggleSession({i},this)"'
            f' style="padding:5px 16px;border-radius:99px;border:1.5px solid var(--border);'
            f'background:var(--bg3);color:var(--text);font-size:14px;font-weight:500;'
            f'cursor:pointer;user-select:none;transition:opacity .15s;">{lbl}</button>'
            f'{subtitle}'
            f'</div>'
        )

    # -- Vial toggle buttons (single row, wrapping)
    vial_controls_html = ""
    for vi, vial in enumerate(vial_list):
        sym = _VIAL_SYMBOLS[vi % len(_VIAL_SYMBOLS)]
        vial_controls_html += (
            f'<button id="pk-vial-btn-{vi}" data-visible="1" onclick="pkToggleVial({vi},this)"'
            f' style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:5px 14px;border-radius:99px;border:1.5px solid var(--border);'
            f'background:var(--bg3);color:var(--text);font-size:14px;font-weight:500;'
            f'cursor:pointer;user-select:none;transition:opacity .15s;">'
            f'<span style="font-size:15px;color:var(--text2);">{sym}</span>Vial {vial}</button>'
        )

    # -- Reference legend
    ref_legend_html = ""
    if ref_data is not None:
        ref_legend_html = (
            f'\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'<span style="width:28px;height:3px;background:transparent;'
            f'border-top:2.5px dashed {_REFERENCE_COLOR};display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:14px;color:var(--text2);">Reference (one line per vial)</span></div>'
        )

    head = html_head(title)

    return f"""{head}
<body>
<div class="page-wrap" style="max-width:1100px;margin:0 auto;">
  <h1>{title}</h1>
  <p class="subtitle">{y_label}</p>

  <div class="chart-card" style="margin-bottom:20px;">
    <p class="chart-title" style="font-size:15px;">Controls</p>
{temp_selector_html}
    <p style="font-size:13px;font-weight:500;color:var(--text2);margin-bottom:10px;">Sessions</p>
    <div id="pk-session-controls">
{session_controls_html}
    </div>

    <p style="font-size:13px;font-weight:500;color:var(--text2);margin-bottom:10px;margin-top:16px;">Vials</p>
    <div id="pk-vial-controls" style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;">
{vial_controls_html}
{ref_legend_html}
    </div>
  </div>

  <div class="chart-card">
    <p class="chart-title" style="font-size:15px;">{metric} over time &mdash; All Vials</p>
    <div class="chart-wrap" style="height:520px;">
      <canvas id="allVialsChart"></canvas>
    </div>
    <p style="font-size:13px;color:var(--text2);margin-top:8px;">
      Colour = session &middot; shape = vial &middot;
      toggle sessions/vials above &middot; scroll to zoom &middot; drag to pan
    </p>
  </div>
</div>

<script>
const SESSIONS    = {sessions_json};
const VIAL_LIST   = {vial_list_json};
const REF_BY_TEMP = {ref_by_temp_json};
const REF_COLOR   = {ref_color_json};
const VIAL_SHAPES = {vial_shapes_json};
let _pkRefTemp    = {default_temp_json};
const METRIC_LABEL = {y_label_json};

const isDark  = window.matchMedia("(prefers-color-scheme: dark)").matches;
const gridCol = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
const tickCol = "#888780";

function _fmtDate(ms) {{
  const d = new Date(ms);
  return d.toLocaleDateString("en-US", {{year: "numeric", month: "short", day: "numeric"}});
}}

function _dateRange() {{
  const ts = SESSIONS.map(function(s) {{ return s.date_ms; }});
  if (!ts.length) return {{xMin: 0, xMax: 1, pad: 86400000}};
  const xMin = Math.min.apply(null, ts);
  const xMax = Math.max.apply(null, ts);
  return {{xMin: xMin, xMax: xMax, pad: Math.max((xMax - xMin) * 0.12, 7 * 86400000)}};
}}

// ---------------------------------------------------------------------------
// Visibility state
// ---------------------------------------------------------------------------

let _sessVisible = new Array(SESSIONS.length).fill(true);
let _vialVisible = new Array(VIAL_LIST.length).fill(true);

// ---------------------------------------------------------------------------
// Dataset builders
// ---------------------------------------------------------------------------

function _buildDataDatasets() {{
  const datasets = [];
  SESSIONS.forEach(function(sess, si) {{
    VIAL_LIST.forEach(function(vial, vi) {{
      const y = sess.vals[vial];
      if (y === null || y === undefined) return;
      datasets.push({{
        pk_type: "data",
        pk_session_idx: si,
        pk_vial_idx: vi,
        label: sess.label + " — Vial " + vial,
        data: [{{x: sess.date_ms, y: parseFloat(y.toFixed(4))}}],
        backgroundColor: sess.color,
        borderColor: sess.color,
        pointBackgroundColor: sess.color,
        pointBorderColor: sess.color,
        pointBorderWidth: 2.5,
        pointStyle: VIAL_SHAPES[vi % VIAL_SHAPES.length],
        pointRadius: 9,
        pointHoverRadius: 11,
        showLine: false,
        borderWidth: 0,
        hidden: false,
      }});
    }});
  }});
  return datasets;
}}

function _buildRefDatasets() {{
  if (!REF_BY_TEMP || !_pkRefTemp) return [];
  const byVial = REF_BY_TEMP[_pkRefTemp] || {{}};
  const r = _dateRange();
  const datasets = [];
  VIAL_LIST.forEach(function(vial, vi) {{
    const refVal = byVial[vial.toUpperCase()];
    if (refVal === undefined || refVal === null) return;
    datasets.push({{
      pk_type: "ref",
      pk_vial_idx: vi,
      label: "Ref Vial " + vial,
      data: [{{x: r.xMin - r.pad, y: refVal}}, {{x: r.xMax + r.pad, y: refVal}}],
      borderColor: REF_COLOR,
      borderWidth: 1.5,
      borderDash: [8, 4],
      pointRadius: 0,
      pointHoverRadius: 0,
      showLine: true,
      fill: false,
      backgroundColor: "transparent",
      tension: 0,
      hidden: false,
    }});
  }});
  return datasets;
}}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

let chart;

(function() {{
  const ctx = document.getElementById("allVialsChart").getContext("2d");
  chart = new Chart(ctx, {{
    type: "scatter",
    data: {{ datasets: [..._buildDataDatasets(), ..._buildRefDatasets()] }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              if (ctx.dataset.pk_type === "ref") {{
                return ctx.dataset.label + ": " + ctx.parsed.y.toFixed(3);
              }}
              return ctx.dataset.label + " — " + _fmtDate(ctx.parsed.x) + ": " + ctx.parsed.y.toFixed(3);
            }}
          }}
        }},
        zoom: {{
          zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: "xy" }},
          pan:  {{ enabled: true, mode: "xy" }},
        }}
      }},
      scales: {{
        x: {{
          type: "linear",
          title: {{ display: true, text: "Date", color: tickCol, font: {{ size: 15 }} }},
          ticks: {{
            color: tickCol,
            font: {{ size: 13 }},
            maxTicksLimit: 8,
            callback: function(v) {{ return _fmtDate(v); }}
          }},
          grid: {{ color: gridCol }},
        }},
        y: {{
          title: {{ display: true, text: METRIC_LABEL, color: tickCol, font: {{ size: 15 }} }},
          ticks: {{ color: tickCol, font: {{ size: 14 }} }},
          grid: {{ color: gridCol }},
        }}
      }}
    }},
  }});
}})();

// ---------------------------------------------------------------------------
// Interactivity
// ---------------------------------------------------------------------------

function _updateVisibility() {{
  chart.data.datasets.forEach(function(ds) {{
    if (ds.pk_type === "data") {{
      ds.hidden = !_sessVisible[ds.pk_session_idx] || !_vialVisible[ds.pk_vial_idx];
    }} else if (ds.pk_type === "ref") {{
      ds.hidden = !_vialVisible[ds.pk_vial_idx];
    }}
  }});
  chart.update();
}}

function _updateRefDatasets() {{
  const byVial = (REF_BY_TEMP && _pkRefTemp) ? (REF_BY_TEMP[_pkRefTemp] || {{}}) : {{}};
  const r = _dateRange();
  chart.data.datasets.forEach(function(ds) {{
    if (ds.pk_type !== "ref") return;
    const vial = VIAL_LIST[ds.pk_vial_idx];
    if (!vial) return;
    const refVal = byVial[vial.toUpperCase()];
    ds.data = (refVal !== undefined && refVal !== null)
      ? [{{x: r.xMin - r.pad, y: refVal}}, {{x: r.xMax + r.pad, y: refVal}}]
      : [];
  }});
  chart.update("none");
}}

function pkSetRefTemp(temp) {{
  _pkRefTemp = temp;
  _updateRefDatasets();
}}

function pkToggleSession(idx, btn) {{
  _sessVisible[idx] = !_sessVisible[idx];
  btn.setAttribute("data-visible", _sessVisible[idx] ? "1" : "0");
  btn.style.opacity = _sessVisible[idx] ? "1.0" : "0.35";
  _updateVisibility();
}}

function pkToggleVial(idx, btn) {{
  _vialVisible[idx] = !_vialVisible[idx];
  btn.setAttribute("data-visible", _vialVisible[idx] ? "1" : "0");
  btn.style.opacity = _vialVisible[idx] ? "1.0" : "0.35";
  _updateVisibility();
}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command("longitudinal")
@click.argument("html_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output", "-o", required=True, type=click.Path(),
    help="Output HTML file path.",
)
@click.option(
    "--date", "dates", multiple=True, metavar="YYYY-MM-DD",
    help=(
        "Date of scan for the corresponding input file (repeat once per file). "
        "Overrides any scan_date embedded in the HTML. "
        "Required for files that have no embedded scan_date."
    ),
)
@click.option(
    "--template-dir", default=None, metavar="DIR",
    help="Path to template_data/ directory (auto-detected if omitted).",
)
@click.option(
    "--phantom", default=None, metavar="NAME",
    help="Phantom name, e.g. SPIRIT (auto-detected from input files if omitted).",
)
@click.option(
    "--label", "labels", multiple=True, metavar="TEXT",
    help="Label for each input file (repeat once per file; defaults to directory name).",
)
def main(
    html_files: tuple[str, ...],
    output: str,
    dates: tuple[str, ...],
    template_dir: str | None,
    phantom: str | None,
    labels: tuple[str, ...],
) -> None:
    """Track per-vial measurements from phantomkit HTML files over time.

    Reads the embedded JSON from each HTML, overlays all time points on one
    scatter plot with a switchable vial selector and temperature-adjusted
    reference line.

    Files must all be the same metric type (all ADC, all T1, or all T2).

    \b
    Example:
        phantomkit plot longitudinal \\
            session01/metrics/plots/T2_mapping.html \\
            session02/metrics/plots/T2_mapping.html \\
            session03/metrics/plots/T2_mapping.html \\
            -o longitudinal_T2.html

    \b
    For HTML files without an embedded scan date:
        phantomkit plot longitudinal \\
            session01/ADC.html \\
            session02/ADC.html \\
            --date 2024-01-15 --date 2024-06-10 \\
            -o longitudinal_ADC.html
    """
    if dates and len(dates) != len(html_files):
        raise click.ClickException(
            f"--date provided {len(dates)} time(s) but "
            f"{len(html_files)} input file(s) given — counts must match."
        )
    if labels and len(labels) != len(html_files):
        raise click.ClickException(
            f"--label provided {len(labels)} time(s) but "
            f"{len(html_files)} input file(s) given — counts must match."
        )

    # ---- Load and parse each HTML file
    sessions_raw: list[dict] = []
    metrics_seen: list[str] = []

    for idx, path in enumerate(html_files):
        data = _load_embedded(path)
        metric, vials_order, vals, se_vals = _extract(data)
        metrics_seen.append(metric)

        # Resolve scan date: explicit --date overrides embedded scan_date
        scan_date: str | None = dates[idx] if dates else data.get("scan_date")
        if scan_date is None:
            raise click.ClickException(
                f"No scan date found for {path!r}.\n"
                "Either re-run the pipeline to embed the date, or supply "
                "--date YYYY-MM-DD for each input file."
            )
        # Validate date format
        try:
            date.fromisoformat(scan_date)
        except ValueError:
            raise click.ClickException(
                f"Invalid date {scan_date!r} for {path!r} — expected YYYY-MM-DD format."
            )

        sessions_raw.append({
            "path": path,
            "data": data,
            "metric": metric,
            "vials_order": vials_order,
            "vals": vals,
            "se_vals": se_vals,
            "scan_date": scan_date,
        })

    # ---- Validate that all inputs share the same metric
    unique_metrics = list(dict.fromkeys(metrics_seen))
    if len(unique_metrics) > 1:
        raise click.ClickException(
            f"Input files contain mixed metrics: {unique_metrics}. "
            "All files must be the same type (all ADC, all T1, or all T2)."
        )
    metric = unique_metrics[0]

    # ---- Session labels
    if labels:
        session_labels = list(labels)
    else:
        session_labels = _derive_labels(list(html_files))

    # ---- Assign colours
    colors = [_SESSION_PALETTE[i % len(_SESSION_PALETTE)] for i in range(len(html_files))]

    # ---- Build session dicts (sort chronologically)
    sessions = [
        {
            "label": session_labels[i],
            "color": colors[i],
            "date_ms": _date_to_ms(sessions_raw[i]["scan_date"]),
            "date_str": sessions_raw[i]["scan_date"],
            "vals": {v.upper(): val for v, val in sessions_raw[i]["vals"].items() if val is not None},
        }
        for i in range(len(html_files))
    ]
    sessions.sort(key=lambda s: s["date_ms"])

    # ---- Auto-detect phantom from embedded data
    if phantom is None:
        for sr in sessions_raw:
            p = sr["data"].get("phantom")
            if p:
                phantom = p
                break

    # ---- Load reference values
    ref_data_loaded: dict | None = None
    resolved_template_dir = template_dir or _auto_template_dir()

    if resolved_template_dir and phantom is None:
        measured_upper: set[str] = set()
        for sess in sessions:
            measured_upper.update(sess["vals"].keys())
        candidates = [
            d.name for d in Path(resolved_template_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        best_overlap, best_candidate = 0, None
        for cand in candidates:
            cand_ref = _load_reference(metric, resolved_template_dir, cand)
            cand_vials = cand_ref["vials"] if cand_ref else []
            overlap = len({v.upper() for v in cand_vials} & measured_upper)
            if overlap > best_overlap:
                best_overlap, best_candidate = overlap, cand
        if best_candidate:
            phantom = best_candidate

    if resolved_template_dir and phantom:
        ref_data_loaded = _load_reference(metric, resolved_template_dir, phantom)

    if ref_data_loaded is None:
        click.echo(
            "[WARN] Reference values not loaded — provide --template-dir and --phantom "
            "to overlay a reference line.",
            err=True,
        )

    # ---- Build vial order (reference order first)
    ref_vials = ref_data_loaded["vials"] if ref_data_loaded else []
    vial_list = _vial_order(sessions, ref_vials)

    # ---- Generate and write HTML
    html = _build_html(metric, sessions, vial_list, ref_data_loaded)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"[INFO] Longitudinal plot saved to: {out_path}")
