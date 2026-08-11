#!/usr/bin/env python3
"""
compare_plots.py — Compare measurements across multiple phantomkit HTML output files.

Reads the embedded phantomkit-data JSON from each HTML, extracts per-vial
measurements (ADC, fitted T1, or fitted T2), and produces a new self-contained
interactive HTML showing all sessions on a single scatter plot.

Features
--------
* Each input file gets a unique colour (filled circles).
* Reference values shown as red open circles (loaded from template_data/ if available).
* Toggle buttons show/hide individual sessions.
* Checkboxes select which sessions contribute to a live group mean (filled squares).

Registered as ``phantomkit plot compare-plots`` via CLI auto-discovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Colour palette (session colours — red #C62828 is reserved for reference)
# ---------------------------------------------------------------------------

_SESSION_PALETTE = [
    "#378ADD",  # blue
    "#1D9E75",  # teal-green
    "#D4537E",  # rose
    "#7F77DD",  # violet
    "#BA7517",  # amber
    "#639922",  # lime
    "#185FA5",  # navy
    "#D85A30",  # burnt orange
    "#993C1D",  # dark sienna
    "#888780",  # stone grey
]

_REFERENCE_COLOR = "#C62828"


# ---------------------------------------------------------------------------
# Embedded data extraction
# ---------------------------------------------------------------------------


def _load_embedded(html_path: str) -> dict:
    """Extract the phantomkit-data JSON block from a phantomkit HTML file."""
    text = Path(html_path).read_text(encoding="utf-8")
    m = re.search(
        r'<script\s+id="phantomkit-data"\s+type="application/json">\s*(\{.*?\})\s*</script>',
        text,
        re.DOTALL,
    )
    if not m:
        raise click.ClickException(
            f"No phantomkit-data block found in {html_path!r}. "
            "Is this a phantomkit HTML output file?"
        )
    return json.loads(m.group(1))


def _extract(
    data: dict,
) -> tuple[str, list[str], dict[str, float | None], dict[str, float | None]]:
    """Return (metric_key, vials_in_order, {vial_upper: value_or_None}, {vial_upper: se_or_None}).

    metric_key is one of: "ADC", "FA", "Intensity", "T1", "T2".
    se_vals contains the curve-fit standard error of the fitted relaxation time
    (T1_se_ms or T2_se_ms) for T1/T2 types; empty dict for ADC/FA/Intensity.
    """
    dtype = data.get("type", "")

    if dtype == "vial_intensity":
        vials = data.get("vials", [])
        means = data.get("means", [])
        mode = data.get("contrast_mode", "generic")
        metric = {"adc": "ADC", "fa": "FA"}.get(mode, "Intensity")

        def _scalar(m):
            """Return a single float from a value that may be a list (multi-volume)."""
            if isinstance(m, list):
                valid = [x for x in m if x is not None]
                return sum(valid) / len(valid) if valid else None
            return float(m) if m is not None else None

        vals = {v.upper(): _scalar(m) for v, m in zip(vials, means)}
        return metric, vials, vals, {}

    if dtype == "maps_ir":
        fit_results = data.get("fit_results", [])
        vials_order, vals, se_vals = [], {}, {}
        for r in fit_results:
            v = r.get("Vial", "")
            t1 = r.get("T1_ms")
            se = r.get("T1_se_ms")
            vals[v.upper()] = float(t1) if t1 is not None else None
            se_vals[v.upper()] = float(se) if se is not None else None
            vials_order.append(v)
        return "T1", vials_order, vals, se_vals

    if dtype == "maps_te":
        fit_results = data.get("fit_results", [])
        vials_order, vals, se_vals = [], {}, {}
        for r in fit_results:
            v = r.get("Vial", "")
            t2 = r.get("T2_ms")
            se = r.get("T2_se_ms")
            vals[v.upper()] = float(t2) if t2 is not None else None
            se_vals[v.upper()] = float(se) if se is not None else None
            vials_order.append(v)
        return "T2", vials_order, vals, se_vals

    raise click.ClickException(
        f"Unknown embedded data type {dtype!r}. "
        "Expected: vial_intensity, maps_ir, or maps_te."
    )


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def _auto_template_dir() -> str | None:
    """Locate template_data/ from the installed phantomkit package."""
    import importlib.util

    spec = importlib.util.find_spec("phantomkit")
    if not spec:
        return None
    pkg_dir = Path(spec.origin).parent
    for candidate in [pkg_dir.parent / "template_data", pkg_dir / "template_data"]:
        if candidate.is_dir():
            return str(candidate)
    return None


def _load_reference(metric: str, template_dir: str, phantom: str) -> dict | None:
    """Load temperature-dependent reference values for *metric* from the calibration xlsx.

    Returns the ``load_calibration_reference`` dict or ``None`` if unavailable.
    ADC values are in ×10⁻³ mm²/s; T1/T2 values are in ms.
    """
    if metric not in ("ADC", "T1", "T2"):
        return None
    from phantomkit.plotting._calibration_reference import load_calibration_reference
    return load_calibration_reference(template_dir, phantom, metric)


# ---------------------------------------------------------------------------
# X-axis vial ordering
# ---------------------------------------------------------------------------


def _vial_axis(sessions: list[dict], ref_vials: list[str]) -> list[str]:
    """Return an ordered vial list for the x-axis.

    Reference vials come first (preserving their prescribed order), then any
    measured vials that are absent from the reference.
    """
    if ref_vials:
        seen = {v.upper() for v in ref_vials}
        result = list(ref_vials)
    else:
        seen: set = set()
        result: list = []

    for sess in sessions:
        for v in sess["vials_order"]:
            if v.upper() not in seen:
                result.append(v)
                seen.add(v.upper())

    return result


# ---------------------------------------------------------------------------
# Session label derivation
# ---------------------------------------------------------------------------


def _derive_labels(paths: list[str]) -> list[str]:
    """Derive a unique, human-readable label for each input path.

    Finds the longest common path suffix shared by all files (walking inward
    from the filename) and uses the first path component that *differs* between
    files as the label.  This naturally surfaces the subject/session directory
    regardless of how deeply the file is nested.

    Example::

        P000009_testing/native_contrasts/metrics/plots/T1_mapping.html
        P000008_html/native_contrasts/metrics/plots/T1_mapping.html
        → labels: ["P000009_testing", "P000008_html"]
    """
    if len(paths) == 1:
        return [Path(paths[0]).stem]

    parts_list = [Path(p).resolve().parts for p in paths]
    min_len = min(len(p) for p in parts_list)

    # Count how many components from the right are identical across all files.
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


# ---------------------------------------------------------------------------
# Y-axis labels
# ---------------------------------------------------------------------------

_Y_LABELS: dict[str, str] = {
    "ADC": "ADC \u00d710\u207b\u00b3 mm\u00b2/s",
    "FA": "Fractional Anisotropy",
    "T1": "T\u2081 (ms)",
    "T2": "T\u2082 (ms)",
    "Intensity": "Intensity",
}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _build_html(
    metric: str,
    vial_axis: list[str],
    sessions: list[dict],  # [{label, color, vials_order, vals, se_vals}]
    ref_data: dict | None, # from load_calibration_reference, or None
) -> str:
    from phantomkit.plotting._html_common import html_head

    y_label = _Y_LABELS.get(metric, metric)
    title = f"Comparison: {metric} per vial"

    vial_upper_to_x = {v.upper(): i for i, v in enumerate(vial_axis)}

    # -- Per-session point data
    session_datasets = []
    for sess in sessions:
        pts = [
            {"x": vial_upper_to_x[v_up], "y": round(val, 4)}
            for v_up, val in sess["vals"].items()
            if val is not None and v_up in vial_upper_to_x
        ]
        pts.sort(key=lambda p: p["x"])
        session_datasets.append({
            "label": sess["label"],
            "color": sess["color"],
            "data": pts,
        })

    # -- Reference point data (default temperature)
    ref_by_temp: dict[str, list] = {}
    temperatures: list = []
    default_temp: str | None = None
    if ref_data is not None:
        default_temp = ref_data.get("default_temp")
        temperatures = ref_data.get("temperatures", [])
        for temp_str, vals_dict in ref_data.get("values_by_temp", {}).items():
            ref_by_temp[temp_str] = [
                {"x": vial_upper_to_x[v.upper()], "y": round(vals_dict[v.upper()], 4)}
                for v in vial_axis
                if vals_dict.get(v.upper()) is not None
            ]
    ref_vals_at_default = (
        ref_data["values_by_temp"].get(default_temp, {})
        if ref_data and default_temp else {}
    )
    ref_pts = [
        {"x": vial_upper_to_x[v.upper()], "y": round(ref_vals_at_default[v.upper()], 4)}
        for v in vial_axis
        if ref_vals_at_default.get(v.upper()) is not None
    ]
    has_ref = bool(ref_pts) or bool(ref_by_temp)

    sessions_json    = json.dumps(session_datasets)
    ref_pts_json     = json.dumps(ref_pts)
    vial_labels_json = json.dumps(vial_axis)
    y_label_json     = json.dumps(y_label)
    ref_color_json   = json.dumps(_REFERENCE_COLOR)

    # -- Session control rows (toggle button + include-in-mean checkbox)
    session_controls_html = (
        '\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;'
        'padding-bottom:12px;border-bottom:1px solid var(--border);">'
        '<button onclick="pkToggleAllSessions()" '
        'style="padding:5px 16px;border-radius:99px;border:1.5px solid var(--border);'
        'background:var(--bg3);color:var(--text);font-size:14px;font-weight:500;'
        'cursor:pointer;user-select:none;transition:opacity .15s;">Toggle all</button>'
        '<button onclick="pkToggleAllMean()" '
        'style="padding:5px 16px;border-radius:99px;border:1.5px solid var(--border);'
        'background:var(--bg3);color:var(--text);font-size:14px;font-weight:500;'
        'cursor:pointer;user-select:none;transition:opacity .15s;">Toggle all in mean</button>'
        '</div>'
    )
    for i, sess in enumerate(sessions):
        c = sess["color"]
        lbl = sess["label"]
        session_controls_html += (
            f'\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'<span style="width:14px;height:14px;border-radius:50%;background:{c};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<button id="pk-sess-btn-{i}" data-visible="1" onclick="pkToggleSession({i},this)"'
            f' style="padding:5px 16px;border-radius:99px;border:1.5px solid var(--border);'
            f'background:var(--bg3);color:var(--text);font-size:14px;font-weight:500;'
            f'cursor:pointer;user-select:none;transition:opacity .15s;">{lbl}</button>'
            f'<label style="display:flex;align-items:center;gap:6px;font-size:14px;'
            f'color:var(--text2);cursor:pointer;">'
            f'<input type="checkbox" id="pk-incl-{i}" checked onchange="pkUpdateMean()">'
            f'include in mean</label>'
            f'</div>'
        )

    # -- Reference legend entry
    ref_legend_html = ""
    if has_ref:
        ref_legend_html = (
            f'\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'<span style="width:14px;height:14px;border-radius:50%;background:transparent;'
            f'border:2px solid {_REFERENCE_COLOR};display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:14px;color:var(--text2);">Reference</span></div>'
        )

    mean_legend_html = (
        '\n    <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
        '<span id="pk-mean-swatch" style="width:14px;height:14px;'
        'display:inline-block;flex-shrink:0;"></span>'
        '<span style="font-size:14px;color:var(--text2);">Group mean (&#x25a0;)</span></div>'
    )

    temp_selector_html = ""
    if ref_data is not None and temperatures:
        temp_options_parts = []
        for t in temperatures:
            sel = " selected" if str(t) == default_temp else ""
            temp_options_parts.append(f'<option value="{t}"{sel}>{t} °C</option>')
        temp_options = "".join(temp_options_parts)
        ref_units = ref_data.get("units", "")
        temp_selector_html = (
            '\n    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            + '<span style="font-size:14px;color:var(--text2);">Reference temperature:</span>'
            + '<select id="refTempSelect" onchange="pkSetRefTemp(this.value)"'
            + ' style="background:var(--bg3);color:var(--text);border:1px solid var(--border);'
            + 'border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;">'
            + temp_options
            + '</select>'
            + f'<span style="font-size:12px;color:var(--text2);">{ref_units}</span>'
            + '</div>'
        )

    ref_by_temp_json = json.dumps(ref_by_temp)
    default_temp_json = json.dumps(default_temp)

    head = html_head(title)

    return f"""{head}
<body>
<div class="page-wrap" style="max-width:960px;margin:0 auto;">
  <h1>{title}</h1>
  <p class="subtitle">{y_label}</p>

  <div class="chart-card" style="margin-bottom:20px;">
    <p class="chart-title" style="font-size:15px;">Sessions</p>
    <div id="pk-session-controls">
{session_controls_html}
{ref_legend_html}
{temp_selector_html}
{mean_legend_html}
    </div>
  </div>

  <div class="chart-card">
    <p class="chart-title" style="font-size:15px;">{metric} per vial</p>
    <div class="chart-wrap" style="height:500px;">
      <canvas id="compChart"></canvas>
    </div>
    <p style="font-size:13px;color:var(--text2);margin-top:8px;">
      Click session buttons to toggle visibility &middot;
      check/uncheck &ldquo;include in mean&rdquo; to update the group mean (&squ;)
    </p>
  </div>
</div>

<script>
const SESSIONS    = {sessions_json};
const REF_PTS     = {ref_pts_json};
const VIAL_LABELS = {vial_labels_json};
const REF_COLOR   = {ref_color_json};
const REF_BY_TEMP = {ref_by_temp_json};
let _pkRefTemp    = {default_temp_json};

const isDark   = window.matchMedia("(prefers-color-scheme: dark)").matches;
const gridCol  = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
const tickCol  = "#888780";
const MEAN_COLOR = isDark ? "#e0e0e0" : "#222222";

document.getElementById("pk-mean-swatch").style.background = MEAN_COLOR;

// ---------------------------------------------------------------------------
// Build Chart.js datasets
// ---------------------------------------------------------------------------
const datasets = [];

SESSIONS.forEach(function(sess) {{
  datasets.push({{
    label: sess.label,
    data: sess.data,
    backgroundColor: sess.color,
    borderColor: sess.color,
    pointBackgroundColor: sess.color,
    pointBorderColor: sess.color,
    pointRadius: 6,
    pointHoverRadius: 8,
    pointStyle: "circle",
    showLine: false,
    borderWidth: 0,
  }});
}});

// Reference — red open circles
datasets.push({{
  label: "Reference",
  data: REF_PTS,
  backgroundColor: "transparent",
  borderColor: "transparent",
  pointBackgroundColor: "transparent",
  pointBorderColor: REF_COLOR,
  pointBorderWidth: 2.5,
  pointRadius: 8,
  pointHoverRadius: 10,
  pointStyle: "circle",
  showLine: false,
  borderWidth: 0,
}});

// Group mean — filled squares
const MEAN_IDX = datasets.length;
datasets.push({{
  label: "Group Mean",
  data: [],
  backgroundColor: MEAN_COLOR,
  borderColor: MEAN_COLOR,
  pointBackgroundColor: MEAN_COLOR,
  pointBorderColor: MEAN_COLOR,
  pointRadius: 8,
  pointHoverRadius: 10,
  pointStyle: "rect",
  showLine: false,
  borderWidth: 0,
}});

const chart = new Chart(
  document.getElementById("compChart").getContext("2d"),
  {{
    type: "scatter",
    data: {{ datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const vialName = VIAL_LABELS[ctx.parsed.x] !== undefined
                ? VIAL_LABELS[ctx.parsed.x] : ctx.parsed.x;
              return ctx.dataset.label + " \u2014 " + vialName + ": " + ctx.parsed.y.toFixed(3);
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
          title: {{ display: true, text: "Vial", color: tickCol, font: {{ size: 15 }} }},
          ticks: {{
            color: tickCol,
            font: {{ size: 14 }},
            stepSize: 1,
            callback: function(v) {{
              return VIAL_LABELS[v] !== undefined ? VIAL_LABELS[v] : v;
            }}
          }},
          grid: {{ color: gridCol }},
        }},
        y: {{
          title: {{ display: true, text: {y_label_json}, color: tickCol, font: {{ size: 15 }} }},
          ticks: {{ color: tickCol, font: {{ size: 14 }} }},
          grid: {{ color: gridCol }},
        }}
      }}
    }},
  }}
);

// Seed group mean from all sessions (all checked by default)
pkUpdateMean();

// ---------------------------------------------------------------------------
// Interactivity
// ---------------------------------------------------------------------------

function pkToggleSession(idx, btn) {{
  const ds = chart.data.datasets[idx];
  ds.hidden = !ds.hidden;
  btn.setAttribute("data-visible", ds.hidden ? "0" : "1");
  btn.style.opacity = ds.hidden ? "0.35" : "1.0";
  chart.update();
}}

function pkUpdateMean() {{
  const byX = {{}};
  SESSIONS.forEach(function(sess, i) {{
    const cb = document.getElementById("pk-incl-" + i);
    if (!cb || !cb.checked) return;
    sess.data.forEach(function(pt) {{
      if (byX[pt.x] === undefined) byX[pt.x] = [];
      byX[pt.x].push(pt.y);
    }});
  }});
  const meanPts = [];
  Object.keys(byX).map(Number).sort(function(a,b){{return a-b;}})
    .forEach(function(x) {{
      const ys = byX[x];
      const mu = ys.reduce(function(a,b){{return a+b;}}, 0) / ys.length;
      meanPts.push({{ x: x, y: mu }});
    }});
  chart.data.datasets[MEAN_IDX].data = meanPts;
  chart.update();
}}

function pkToggleAllSessions() {{
  const n = SESSIONS.length;
  let anyHidden = false;
  for (let i = 0; i < n; i++) {{
    if (chart.data.datasets[i].hidden) {{ anyHidden = true; break; }}
  }}
  const makeVisible = anyHidden;
  for (let i = 0; i < n; i++) {{
    chart.data.datasets[i].hidden = !makeVisible;
    const btn = document.getElementById("pk-sess-btn-" + i);
    btn.setAttribute("data-visible", makeVisible ? "1" : "0");
    btn.style.opacity = makeVisible ? "1.0" : "0.35";
  }}
  chart.update();
}}

function pkToggleAllMean() {{
  const n = SESSIONS.length;
  let anyUnchecked = false;
  for (let i = 0; i < n; i++) {{
    const cb = document.getElementById("pk-incl-" + i);
    if (cb && !cb.checked) {{ anyUnchecked = true; break; }}
  }}
  const makeChecked = anyUnchecked;
  for (let i = 0; i < n; i++) {{
    const cb = document.getElementById("pk-incl-" + i);
    if (cb) cb.checked = makeChecked;
  }}
  pkUpdateMean();
}}

function pkSetRefTemp(temp) {{
  _pkRefTemp = temp;
  const refDs = chart.data.datasets.find(d => d.label === "Reference");
  if (!refDs || !REF_BY_TEMP) return;
  refDs.data = REF_BY_TEMP[temp] || [];
  chart.update("none");
}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command("compare-plots")
@click.argument("html_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output", "-o", required=True, type=click.Path(),
    help="Output HTML file path.",
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
    help="Label for each input file (repeat once per file; defaults to filename stem).",
)
def main(
    html_files: tuple[str, ...],
    output: str,
    template_dir: str | None,
    phantom: str | None,
    labels: tuple[str, ...],
) -> None:
    """Compare measurements from multiple phantomkit HTML output files.

    Reads the embedded JSON from each HTML, overlays all sessions on one
    scatter plot, and provides a toggleable group-mean marker (filled square).

    \b
    Example:
        phantomkit plot compare-plots \\
            session01/metrics/plots/ADC.html \\
            session02/metrics/plots/ADC.html \\
            -o comparison_ADC.html
    """
    if labels and len(labels) != len(html_files):
        raise click.ClickException(
            f"--label provided {len(labels)} time(s) but "
            f"{len(html_files)} input file(s) given — counts must match."
        )

    # ---- Load and parse each HTML file
    sessions_raw: list[dict] = []
    metrics_seen: list[str] = []
    for path in html_files:
        data = _load_embedded(path)
        metric, vials_order, vals, se_vals = _extract(data)
        metrics_seen.append(metric)
        sessions_raw.append({
            "path": path,
            "data": data,
            "metric": metric,
            "vials_order": vials_order,
            "vals": vals,
            "se_vals": se_vals,
        })

    # ---- Validate that all inputs share the same metric
    unique_metrics = list(dict.fromkeys(metrics_seen))
    if len(unique_metrics) > 1:
        raise click.ClickException(
            f"Input files contain mixed metrics: {unique_metrics}. "
            "All files must be the same type (all ADC, all T1, or all T2)."
        )
    metric = unique_metrics[0]

    # ---- Determine session labels
    if labels:
        session_labels = list(labels)
    else:
        session_labels = _derive_labels(list(html_files))

    # ---- Assign colours
    colors = [_SESSION_PALETTE[i % len(_SESSION_PALETTE)] for i in range(len(html_files))]

    sessions = [
        {
            "label": session_labels[i],
            "color": colors[i],
            "vials_order": sessions_raw[i]["vials_order"],
            "vals": sessions_raw[i]["vals"],
            "se_vals": sessions_raw[i].get("se_vals", {}),
        }
        for i in range(len(html_files))
    ]

    # ---- Auto-detect phantom from embedded data
    if phantom is None:
        for sr in sessions_raw:
            p = sr["data"].get("phantom")
            if p:
                phantom = p
                break

    # ---- Load reference values
    ref_vials: list[str] = []
    ref_data_loaded: dict | None = None
    resolved_template_dir = template_dir or _auto_template_dir()

    if resolved_template_dir and phantom is None:
        # Auto-select phantom by finding the candidate whose reference vials best
        # overlap with the measured vials across all input sessions.
        measured_upper = set()
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

    ref_vials = ref_data_loaded["vials"] if ref_data_loaded else []
    ref_vals_check: dict = (
        ref_data_loaded["values_by_temp"].get(ref_data_loaded["default_temp"], {})
        if ref_data_loaded else {}
    )

    if not ref_vals_check:
        click.echo(
            "[WARN] Reference values not loaded — provide --template-dir and --phantom "
            "to overlay reference data.",
            err=True,
        )

    # ---- Build x-axis vial order
    vial_axis = _vial_axis(sessions, ref_vials)

    # ---- Generate and write HTML
    html = _build_html(metric, vial_axis, sessions, ref_data_loaded)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"[INFO] Comparison plot saved to: {out_path}")
