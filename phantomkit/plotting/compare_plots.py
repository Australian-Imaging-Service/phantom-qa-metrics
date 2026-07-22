#!/usr/bin/env python3
"""
compare_plots.py — Compare measurements across multiple phantomkit HTML output files.

Reads the embedded phantomkit-data JSON from each HTML, extracts per-vial
measurements (ADC, fitted T1, T2, or PET CRC/SOR/Uniformity), and produces a
new self-contained interactive HTML.

For MRI plots (ADC, T1, T2, Intensity):
* Each input file gets a unique colour (filled circles).
* Reference values shown as red open circles (loaded from template_data/ if available).
* Toggle buttons show/hide individual sessions.
* Checkboxes select which sessions contribute to a live group mean (filled squares).

For PET plots (pet_vial_activity):
* CRC scatter + line chart (sphere vials), with CRCmax 3D / axial toggle.
* SOR table with per-session values and mean ± SD.
* Uniformity table with per-session values and mean (SD).
* Unified session toggle buttons controlling all three panels.

Registered as ``phantomkit plot compare-plots`` via CLI auto-discovery.
"""

from __future__ import annotations

import json
import math
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
        "Expected: vial_intensity, maps_ir, maps_te, or pet_vial_activity."
    )


# ---------------------------------------------------------------------------
# PET-specific extraction
# ---------------------------------------------------------------------------


def _extract_pet(data: dict) -> dict:
    """Extract CRC, SOR, and uniformity data from a pet_vial_activity HTML.

    Requires that the HTML was generated with a version of pet_html.py that
    embeds the ``crc``, ``sor``, and ``uniformity`` keys in the data tag.

    Returns
    -------
    dict with keys: crc_vials, crc_3d, crc_axial, sor_vials, sor_values,
                    uniformity_vial, uniformity_pct
    """
    crc = data.get("crc", {})
    sor = data.get("sor", {})
    uni = data.get("uniformity", {})

    if not crc:
        raise click.ClickException(
            "No 'crc' block found in the embedded data. "
            "Re-generate the PET HTML with the current version of phantomkit."
        )

    def _clean_list(lst):
        return [
            float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None
            for v in (lst or [])
        ]

    return {
        "crc_vials":      crc.get("vials", []),
        "crc_3d":         _clean_list(crc.get("crc_3d", [])),
        "crc_axial":      _clean_list(crc.get("crc_axial", [])),
        "sor_vials":      sor.get("vials", []),
        "sor_values":     _clean_list(sor.get("values", [])),
        "uniformity_vial": uni.get("vial", ""),
        "uniformity_pct": (
            float(uni["pct"])
            if uni.get("pct") is not None and not (isinstance(uni.get("pct"), float) and math.isnan(uni["pct"]))
            else None
        ),
    }


# ---------------------------------------------------------------------------
# PET comparison HTML builder
# ---------------------------------------------------------------------------


def _build_pet_compare_html(
    sessions: list[dict],
    title: str = "PET Phantom — Comparison",
) -> str:
    """Build a self-contained comparison HTML for multiple PET sessions.

    Parameters
    ----------
    sessions:
        List of dicts, each with keys: label, color, crc_vials, crc_3d,
        crc_axial, sor_vials, sor_values, uniformity_vial, uniformity_pct.
    title:
        HTML page title.
    """
    from phantomkit.plotting._html_common import (
        ERROR_BAR_PLUGIN_JS,
        base_opts_js,
        html_head,
    )

    _SPHERE_ORDER = ["1mm", "2mm", "3mm", "4mm", "5mm"]
    _SOR_ORDER    = ["Air", "Water"]

    # ── Canonical vial orders ────────────────────────────────────────────────────
    _all_crc = {v for sess in sessions for v in sess["crc_vials"]}
    all_crc_vials = [v for v in _SPHERE_ORDER if v in _all_crc]
    crc_vial_to_x = {v: i for i, v in enumerate(all_crc_vials)}

    _all_sor = {v for sess in sessions for v in sess["sor_vials"]}
    all_sor_vials = [v for v in _SOR_ORDER if v in _all_sor]
    sor_vial_to_x = {v: i for i, v in enumerate(all_sor_vials)}

    # ── Build session data for JS ────────────────────────────────────────────────
    def _crc_pts(sess, key):
        pts = []
        vals = sess[key]
        for i, v in enumerate(sess["crc_vials"]):
            if v not in crc_vial_to_x or i >= len(vals):
                continue
            val = vals[i]
            if val is not None:
                pts.append({"x": crc_vial_to_x[v], "y": round(float(val), 4)})
        return sorted(pts, key=lambda p: p["x"])

    def _sor_pts(sess, x_offset=0.0):
        pts = []
        for v, x in sor_vial_to_x.items():
            idx = sess["sor_vials"].index(v) if v in sess["sor_vials"] else -1
            val = sess["sor_values"][idx] if 0 <= idx < len(sess["sor_values"]) else None
            if val is not None:
                pts.append({"x": round(x + x_offset, 3), "y": round(float(val), 4)})
        return sorted(pts, key=lambda p: p["x"])

    n = len(sessions)
    _step = 0.08 if n <= 6 else round(0.4 / max(n - 1, 1), 3)
    _uni_offsets = [round((i - (n - 1) / 2) * _step, 3) for i in range(n)]

    sessions_data = []
    for i, sess in enumerate(sessions):
        uni_pct = sess.get("uniformity_pct")
        sessions_data.append({
            "label": sess["label"],
            "color": sess["color"],
            "crc": {
                "crc_3d":    _crc_pts(sess, "crc_3d"),
                "crc_axial": _crc_pts(sess, "crc_axial"),
            },
            "sor_pts":    _sor_pts(sess, _uni_offsets[i]),
            "sor_vials":  sess["sor_vials"],
            "sor_values": [round(float(v), 4) if v is not None else None for v in sess["sor_values"]],
            "uniformity_pct": round(float(uni_pct), 3) if uni_pct is not None else None,
            "uni_x":          _uni_offsets[i],
        })

    sessions_json  = json.dumps(sessions_data)
    crc_vials_json = json.dumps(all_crc_vials)
    sor_vials_json = json.dumps(all_sor_vials)
    n_sess = len(sessions)

    # ── Session toggle buttons ───────────────────────────────────────────────────
    sess_btns_html = ""
    for i, sess in enumerate(sessions):
        c   = sess["color"]
        lbl = sess["label"]
        sess_btns_html += (
            f'<button id="pk-sess-{i}" onclick="pkToggleSession({i},this)" '
            f'class="pk-btn" style="background:{c};color:#fff;border-color:{c};">'
            f'{lbl}</button> '
        )

    opts_js = base_opts_js(x_label="Vial", y_label="CRCmax (3D)", enable_zoom=True)
    head    = html_head(title)

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Comparing {n_sess} PET phantom session{"s" if n_sess != 1 else ""}</p>

<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Sessions</span>
    {sess_btns_html}
  </div>
</div>

<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">CRC mode</span>
    <button class="pk-btn pk-crc-mode-btn active" onclick="crcSetMode('crc_3d',this)">CRC<sub>max</sub> 3D</button>
    <button class="pk-btn pk-crc-mode-btn" onclick="crcSetMode('crc_axial',this)">CRC<sub>max</sub> axial</button>
  </div>
</div>

<div class="chart-card" style="margin-bottom:8px;">
  <div class="chart-wrap" style="height:340px"><canvas id="crcCanvas"></canvas></div>
</div>
<p style="font-size:11px;color:var(--text2);margin:4px 0 24px 4px;">
  &#9632; Mean &plusmn; SD across active sessions &middot; double-click chart to reset zoom
</p>

<div class="chart-card" style="margin-bottom:8px;">
  <div class="chart-wrap" style="height:260px"><canvas id="sorCanvas"></canvas></div>
</div>
<p style="font-size:11px;color:var(--text2);margin:4px 0 24px 4px;">
  &#9632; Mean &plusmn; SD across active sessions &middot; double-click chart to reset zoom
</p>

<div class="chart-card" style="margin-bottom:8px;">
  <div class="chart-wrap" style="height:260px"><canvas id="uniCanvas"></canvas></div>
</div>
<p style="font-size:11px;color:var(--text2);margin:4px 0 8px 4px;">
  &#9632; Mean &plusmn; SD across active sessions &middot; double-click chart to reset zoom
</p>

<script>
const SESSIONS   = {sessions_json};
const CRC_VIALS  = {crc_vials_json};
const SOR_VIALS  = {sor_vials_json};

const isDark     = window.matchMedia("(prefers-color-scheme: dark)").matches;
const MEAN_COLOR = isDark ? "#e8e8e4" : "#1a1a18";
var _crcMode = "crc_3d";
const _active = new Array(SESSIONS.length).fill(true);

{ERROR_BAR_PLUGIN_JS}
{opts_js}

// ---- Shared mean-dataset factory ---------------------------------------------
function _meanDs(extraOpts) {{
  return Object.assign({{
    label: "Mean ± SD",
    _isMean: true,
    data: [],
    errorBars: {{}},
    backgroundColor: MEAN_COLOR,
    borderColor: MEAN_COLOR + "cc",
    pointBackgroundColor: MEAN_COLOR,
    pointBorderColor: MEAN_COLOR,
    pointRadius: 7,
    pointHoverRadius: 9,
    pointStyle: "rect",
    showLine: true,
    borderWidth: 2,
    borderDash: [5, 3],
  }}, extraOpts || {{}});
}}

// ---- Shared session-dataset factory -----------------------------------------
function _sessDs(sess, i, pts, extraOpts) {{
  return Object.assign({{
    label: sess.label,
    _sessIdx: i,
    data: pts,
    backgroundColor: sess.color,
    borderColor: sess.color + "bb",
    pointBackgroundColor: sess.color,
    pointBorderColor: sess.color,
    pointRadius: 5,
    pointHoverRadius: 7,
    showLine: true,
    borderWidth: 1.5,
    tension: 0,
  }}, extraOpts || {{}});
}}

// ---- CRC chart ---------------------------------------------------------------
const crcDatasets = [];
SESSIONS.forEach((sess, i) => crcDatasets.push(
  _sessDs(sess, i, sess.crc.crc_3d.map(p => ({{x: p.x, y: p.y}})))
));
const CRC_MEAN_IDX = crcDatasets.length;
crcDatasets.push(_meanDs());

const crcOpts = baseOpts("Vial", "CRCmax (3D)");
crcOpts.scales.x.type = "linear";
crcOpts.scales.x.ticks.callback = v => CRC_VIALS[v] ?? v;
crcOpts.scales.x.ticks.stepSize = 1;

const crcChart = new Chart(
  document.getElementById("crcCanvas").getContext("2d"),
  {{ type: "line", data: {{ datasets: crcDatasets }}, options: crcOpts, plugins: [errorBarPlugin] }}
);
document.getElementById("crcCanvas").addEventListener("dblclick", () => crcChart.resetZoom());

// ---- SOR chart ---------------------------------------------------------------
const sorDatasets = [];
SESSIONS.forEach((sess, i) => sorDatasets.push(
  _sessDs(sess, i, sess.sor_pts.map(p => ({{x: p.x, y: p.y}})), {{showLine: false}})
));
const SOR_MEAN_IDX = sorDatasets.length;
sorDatasets.push(_meanDs({{showLine: false}}));

const sorOpts = baseOpts("Vial", "Spill-Over Ratio (SOR)");
sorOpts.scales.x.type = "linear";
sorOpts.scales.x.ticks.callback = v => SOR_VIALS[v] ?? v;
sorOpts.scales.x.ticks.stepSize = 1;

const sorChart = new Chart(
  document.getElementById("sorCanvas").getContext("2d"),
  {{ type: "line", data: {{ datasets: sorDatasets }}, options: sorOpts, plugins: [errorBarPlugin] }}
);
document.getElementById("sorCanvas").addEventListener("dblclick", () => sorChart.resetZoom());

// ---- Uniformity chart --------------------------------------------------------
const uniDatasets = [];
SESSIONS.forEach((sess, i) => uniDatasets.push(
  _sessDs(sess, i,
    sess.uniformity_pct !== null && sess.uniformity_pct !== undefined
      ? [{{x: sess.uni_x, y: sess.uniformity_pct}}] : [],
    {{showLine: false}}
  )
));
const UNI_MEAN_IDX = uniDatasets.length;
uniDatasets.push(_meanDs({{showLine: false}}));

const uniOpts = baseOpts("", "Uniformity (SD / Mean × 100, %)");
uniOpts.scales.x.type = "linear";
uniOpts.scales.x.min = -0.5;
uniOpts.scales.x.max = 0.5;
uniOpts.scales.x.ticks.callback = v => (v === 0 ? "Uniform" : "");
uniOpts.scales.x.ticks.stepSize = 1;

const uniChart = new Chart(
  document.getElementById("uniCanvas").getContext("2d"),
  {{ type: "line", data: {{ datasets: uniDatasets }}, options: uniOpts, plugins: [errorBarPlugin] }}
);
document.getElementById("uniCanvas").addEventListener("dblclick", () => uniChart.resetZoom());

// ---- Helpers ----------------------------------------------------------------
function _meanAndErrBars(pointsPerSession) {{
  // pointsPerSession: array of [{{x, y}}] arrays (one per active session)
  const byX = {{}};
  pointsPerSession.forEach(pts => pts.forEach(pt => {{
    if (!byX[pt.x]) byX[pt.x] = [];
    byX[pt.x].push(pt.y);
  }}));
  const pts = [], bars = {{}};
  Object.keys(byX).map(Number).sort((a, b) => a - b).forEach(x => {{
    const ys = byX[x];
    const mu = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sd = ys.length > 1
      ? Math.sqrt(ys.reduce((s, b) => s + (b - mu) ** 2, 0) / (ys.length - 1))
      : 0;
    const k = pts.length;
    pts.push({{x, y: mu}});
    if (sd > 0) bars[String(k)] = {{yMin: mu - sd, yMax: mu + sd}};
  }});
  return {{pts, bars}};
}}

// ---- Session toggle ---------------------------------------------------------
function pkToggleSession(idx, btn) {{
  _active[idx] = !_active[idx];
  const sess = SESSIONS[idx];
  if (_active[idx]) {{
    btn.style.background  = sess.color;
    btn.style.color       = "#fff";
    btn.style.borderColor = sess.color;
    btn.style.opacity     = "1";
  }} else {{
    btn.style.background  = "var(--bg3)";
    btn.style.color       = "var(--text2)";
    btn.style.borderColor = "var(--border)";
    btn.style.opacity     = "0.55";
  }}
  [crcChart, sorChart, uniChart].forEach(ch => {{
    const ds = ch.data.datasets.find(d => d._sessIdx === idx);
    if (ds) ds.hidden = !_active[idx];
  }});
  pkUpdateMeans();
}}

// ---- CRC mode toggle --------------------------------------------------------
function crcSetMode(mode, btn) {{
  _crcMode = mode;
  document.querySelectorAll(".pk-crc-mode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  crcChart.options.scales.y.title.text =
    {{ crc_3d: "CRCmax (3D)", crc_axial: "CRCmax (axial avg)" }}[mode];
  SESSIONS.forEach((sess, i) => {{
    const ds = crcChart.data.datasets.find(d => d._sessIdx === i);
    if (ds) ds.data = sess.crc[mode].map(p => ({{x: p.x, y: p.y}}));
  }});
  pkUpdateMeans();
}}

// ---- Mean + SD computation --------------------------------------------------
function pkUpdateMeans() {{
  // CRC
  const crcPts = SESSIONS.map((s, i) => _active[i] ? s.crc[_crcMode] : []);
  const crc = _meanAndErrBars(crcPts);
  crcChart.data.datasets[CRC_MEAN_IDX].data       = crc.pts;
  crcChart.data.datasets[CRC_MEAN_IDX].errorBars  = crc.bars;

  // SOR — group by nominal vial index so stagger doesn't affect the mean position
  const sorMeanPts = [], sorErrBars = {{}};
  SOR_VIALS.forEach((v, j) => {{
    const ys = [];
    SESSIONS.forEach((sess, i) => {{
      if (!_active[i]) return;
      const idx = sess.sor_vials.indexOf(v);
      const val = idx >= 0 ? sess.sor_values[idx] : null;
      if (val !== null && val !== undefined) ys.push(val);
    }});
    if (!ys.length) return;
    const mu = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sd = ys.length > 1
      ? Math.sqrt(ys.reduce((s, b) => s + (b - mu) ** 2, 0) / (ys.length - 1))
      : 0;
    const k = sorMeanPts.length;
    sorMeanPts.push({{x: j, y: mu}});
    if (sd > 0) sorErrBars[String(k)] = {{yMin: mu - sd, yMax: mu + sd}};
  }});
  sorChart.data.datasets[SOR_MEAN_IDX].data      = sorMeanPts;
  sorChart.data.datasets[SOR_MEAN_IDX].errorBars = sorErrBars;

  // Uniformity
  const uniPts = SESSIONS.map((s, i) =>
    _active[i] && s.uniformity_pct !== null && s.uniformity_pct !== undefined
      ? [{{x: 0, y: s.uniformity_pct}}] : []
  );
  const uni = _meanAndErrBars(uniPts);
  uniChart.data.datasets[UNI_MEAN_IDX].data      = uni.pts;
  uniChart.data.datasets[UNI_MEAN_IDX].errorBars = uni.bars;

  crcChart.update("none");
  sorChart.update("none");
  uniChart.update("none");
}}

pkUpdateMeans();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Reference data loading (MRI only)
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
    """Load temperature-dependent reference values for *metric* from the calibration xlsx."""
    if metric not in ("ADC", "T1", "T2"):
        return None
    from phantomkit.plotting._calibration_reference import load_calibration_reference
    return load_calibration_reference(template_dir, phantom, metric)


# ---------------------------------------------------------------------------
# X-axis vial ordering (MRI)
# ---------------------------------------------------------------------------


def _vial_axis(sessions: list[dict], ref_vials: list[str]) -> list[str]:
    """Return an ordered vial list for the x-axis."""
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
    """Derive a unique, human-readable label for each input path."""
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


# ---------------------------------------------------------------------------
# Y-axis labels (MRI)
# ---------------------------------------------------------------------------

_Y_LABELS: dict[str, str] = {
    "ADC": "ADC ×10⁻³ mm²/s",
    "FA": "Fractional Anisotropy",
    "T1": "T₁ (ms)",
    "T2": "T₂ (ms)",
    "Intensity": "Intensity",
}


# ---------------------------------------------------------------------------
# MRI HTML generation
# ---------------------------------------------------------------------------


def _build_html(
    metric: str,
    vial_axis: list[str],
    sessions: list[dict],
    ref_data: dict | None,
) -> str:
    from phantomkit.plotting._html_common import html_head

    y_label = _Y_LABELS.get(metric, metric)
    title = f"Comparison: {metric} per vial"

    vial_upper_to_x = {v.upper(): i for i, v in enumerate(vial_axis)}

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

    session_controls_html = ""
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

    ref_by_temp_json  = json.dumps(ref_by_temp)
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
              return ctx.dataset.label + " — " + vialName + ": " + ctx.parsed.y.toFixed(3);
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

pkUpdateMean();

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

    For MRI plots (ADC, T1, T2): overlays all sessions on one scatter plot
    with a toggleable group-mean marker.

    For PET plots: shows a CRC chart (scatter + line per session), a SOR
    table, and a Uniformity table, all with session toggle controls.

    \b
    Example:
        phantomkit plot compare-plots \\
            session01/metrics/plots/PET.html \\
            session02/metrics/plots/PET.html \\
            -o comparison_PET.html
    """
    if labels and len(labels) != len(html_files):
        raise click.ClickException(
            f"--label provided {len(labels)} time(s) but "
            f"{len(html_files)} input file(s) given — counts must match."
        )

    # ---- Load embedded data from each file
    raw_data: list[dict] = []
    for path in html_files:
        raw_data.append(_load_embedded(path))

    # ---- Detect data type (PET vs MRI)
    dtypes = [d.get("type", "") for d in raw_data]
    is_pet = [t == "pet_vial_activity" for t in dtypes]

    if any(is_pet) and not all(is_pet):
        raise click.ClickException(
            "Cannot mix PET and non-PET HTML files. "
            "All input files must be the same type."
        )

    # ---- Determine session labels
    if labels:
        session_labels = list(labels)
    else:
        session_labels = _derive_labels(list(html_files))

    colors = [_SESSION_PALETTE[i % len(_SESSION_PALETTE)] for i in range(len(html_files))]

    # ---- PET path
    if all(is_pet):
        sessions = []
        for i, (path, data) in enumerate(zip(html_files, raw_data)):
            pet = _extract_pet(data)
            sessions.append({
                "label": session_labels[i],
                "color": colors[i],
                **pet,
            })

        # Auto-detect title from phantom name
        phantom_name = phantom or raw_data[0].get("phantom", "PET")
        page_title   = f"{phantom_name} Phantom — Comparison"

        html = _build_pet_compare_html(sessions, title=page_title)
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        click.echo(f"[INFO] PET comparison plot saved to: {out_path}")
        return

    # ---- MRI path
    sessions_raw: list[dict] = []
    metrics_seen: list[str] = []
    for path, data in zip(html_files, raw_data):
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

    unique_metrics = list(dict.fromkeys(metrics_seen))
    if len(unique_metrics) > 1:
        raise click.ClickException(
            f"Input files contain mixed metrics: {unique_metrics}. "
            "All files must be the same type (all ADC, all T1, or all T2)."
        )
    metric = unique_metrics[0]

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

    if phantom is None:
        for sr in sessions_raw:
            p = sr["data"].get("phantom")
            if p:
                phantom = p
                break

    ref_vials: list[str] = []
    ref_data_loaded: dict | None = None
    resolved_template_dir = template_dir or _auto_template_dir()

    if resolved_template_dir and phantom is None:
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

    vial_axis = _vial_axis(sessions, ref_vials)
    html = _build_html(metric, vial_axis, sessions, ref_data_loaded)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"[INFO] Comparison plot saved to: {out_path}")
