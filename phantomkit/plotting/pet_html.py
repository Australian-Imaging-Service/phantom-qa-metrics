"""
pet_html.py
Build interactive HTML for PET phantom vial activity measurements.

Reads from a metrics xlsx (mean / std / median / max / min / p25 / p75 / count
sheets) and generates a self-contained HTML page with:

  - NiiVue MRI viewer panel (base64-embedded NIfTI)
  - Measured vial activity chart with Mean / Median / Max toggle
  - Error bars: ±SD, ±SE, ±2 SE, MAD, IQR, Min–Max  (same as ADC HTML)
  - Per-vial stats table

There is no reference dataset — this is a "measured only" report.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PET-specific controls HTML (adds Max button alongside Mean / Median)
# ---------------------------------------------------------------------------

_PET_CONTROLS_HTML = """\
<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Measure</span>
    <button class="pk-btn pk-measure-btn active" onclick="pkSetMeasure('mean',this)">Mean</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('median',this)">Median</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('max',this)">Max</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('crc_3d',this)">CRC<sub>max</sub> 3D</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('crc_axial',this)">CRC<sub>max</sub> axial</button>
  </div>
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Error bars</span>
    <button class="pk-btn pk-err-btn active" onclick="pkSetErrMode('sd',this)">&plusmn;SD</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('se',this)">&plusmn;SE</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('2se',this)">&plusmn;2&thinsp;SE</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('mad',this)">&plusmn;MAD</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('iqr',this)">IQR</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('minmax',this)">Min&ndash;Max</button>
    <button class="pk-btn pk-err-btn" onclick="pkSetErrMode('none',this)">None</button>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# PET toggle JS — identical logic to PK_TOGGLE_JS but also handles "max"
# (max has no errBounds entry, so error bars are automatically suppressed)
# ---------------------------------------------------------------------------

_PET_TOGGLE_JS = """
window._pkMeasure = "mean";
window._pkErrMode = "sd";
window._pkAfterUpdate = null;
var _pkCharts = [];

function pkSetMeasure(mode, btn) {
  window._pkMeasure = mode;
  document.querySelectorAll(".pk-measure-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  pkUpdateAllCharts();
}

function pkSetErrMode(mode, btn) {
  window._pkErrMode = mode;
  document.querySelectorAll(".pk-err-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  pkUpdateAllCharts();
}

const _CRC_MODES = new Set(["crc_3d", "crc_axial"]);
const _CRC_Y_LABELS = {crc_3d: "CRCmax (3D)", crc_axial: "CRCmax (axial avg)"};

function pkUpdateAllCharts() {
  const m = window._pkMeasure, e = window._pkErrMode;
  const isCRC = _CRC_MODES.has(m);
  _pkCharts.forEach(ch => {
    ch.data.datasets.forEach((ds) => {
      if (ds._row !== undefined) {
        const vals = PK_DATA.measure[m][ds._row];
        ds.data.forEach((pt, k) => { pt.y = vals[k]; });
        const eb = (
          !isCRC &&
          e !== "none" &&
          PK_DATA.errBounds[m] &&
          PK_DATA.errBounds[m][e]
        );
        if (eb) {
          const lo = PK_DATA.errBounds[m][e].lower[ds._row];
          const hi = PK_DATA.errBounds[m][e].upper[ds._row];
          ds.errorBars = {};
          lo.forEach((l, k) => {
            ds.errorBars[String(k)] = { yMin: l, yMax: hi[k] };
          });
        } else {
          ds.errorBars = {};
        }
      }
    });
    ch.options.scales.y.title.text = _CRC_Y_LABELS[m] ?? Y_LABEL;
    ch.update("none");
  });
  if (typeof window._pkAfterUpdate === "function") window._pkAfterUpdate();
}
"""


# ---------------------------------------------------------------------------
# Low-level HTML builder
# ---------------------------------------------------------------------------


def build_pet_html(
    *,
    vials: list,
    mean_values: np.ndarray,
    std_values: Optional[np.ndarray] = None,
    median_values: Optional[np.ndarray] = None,
    max_values: Optional[np.ndarray] = None,
    count_values: Optional[np.ndarray] = None,
    p25_values: Optional[np.ndarray] = None,
    p75_values: Optional[np.ndarray] = None,
    min_values: Optional[np.ndarray] = None,
    mean_mad_values: Optional[np.ndarray] = None,
    median_mad_values: Optional[np.ndarray] = None,
    phantom: str = "PET",
    title: str = "PET Phantom — Vial Activity",
    y_label: str = "Activity (Bq/mL)",
    nifti_image: Optional[str] = None,
    vial_niftis: Optional[dict] = None,
    embedded_data: Optional[dict] = None,
    viewer_cal_max: Optional[float] = None,
    crc_vials: Optional[list] = None,
    crc_3d_values: Optional[list] = None,
    crc_slice_avg_values: Optional[list] = None,
    uniformity_vial: Optional[str] = None,
    uniformity_pct: Optional[float] = None,
    sor_vials: Optional[list] = None,
    sor_values: Optional[list] = None,
    crc_3d_chart: Optional[list] = None,
    crc_slice_avg_chart: Optional[list] = None,
) -> str:
    """Build a self-contained interactive HTML page for PET vial measurements.

    Parameters
    ----------
    vials:
        Vial labels (e.g. ``["A", "B", "C", "D", "E"]``).
    mean_values:
        Shape ``(n_vials,)`` — mean activity per vial.
    std_values, median_values, max_values, ...:
        Additional statistics from the metrics xlsx.  All shape ``(n_vials,)``
        or ``None`` if the sheet was not found.
    phantom:
        Phantom name used in the title.
    title:
        Full HTML page/plot title.
    y_label:
        Y-axis label (e.g. ``"Activity (Bq/mL)"`` or ``"Normalised Activity"``).
    nifti_image:
        Path to the background NIfTI for the MRI viewer panel.
    vial_niftis:
        ``{vial_name: path}`` mapping for vial ROI overlay NIfTIs.
    embedded_data:
        Optional extra dict to embed as machine-readable JSON in the HTML.
    crc_vials:
        Ordered list of vial labels to show in the CRC table (e.g. A–E).
    crc_3d_values:
        CRC values computed from the full 3D ROI (max/mean), same order as
        ``crc_vials``.
    crc_slice_avg_values:
        CRC values computed from the max of per-axial-slice means divided by
        the 3D mean, same order as ``crc_vials``.
    uniformity_vial:
        Label of the uniformity vial (e.g. ``"Uniform"``).
    uniformity_pct:
        Uniformity value in percent (SD/mean × 100) for ``uniformity_vial``.
    sor_vials:
        Ordered list of vial labels for the SOR table (e.g. ``["Air", "Water"]``).
    sor_values:
        SOR values (mean(ROI)/mean(Uniform)) in the same order as ``sor_vials``.

    Returns
    -------
    str
        Complete self-contained HTML string.
    """
    from phantomkit.plotting._html_common import (
        ERROR_BAR_PLUGIN_JS,
        _compute_pk_err_bounds,
        _niivue_viewer_panel,
        base_opts_js,
        html_head,
        phantomkit_data_tag,
    )

    # Reshape 1-D (n_vials,) → (n_vials, 1) so the error-bounds helper
    # (which expects shape (n_vials, n_vols)) works without special-casing.
    def _col(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return arr[:, None] if arr.ndim == 1 else arr

    mean_2d    = _col(mean_values)
    std_2d     = _col(std_values)    if std_values    is not None else np.zeros_like(mean_2d)
    med_2d     = _col(median_values) if median_values is not None else mean_2d
    max_2d     = _col(max_values)    if max_values    is not None else mean_2d
    cnt_2d     = _col(count_values)  if count_values  is not None else np.full_like(mean_2d, 1000.0)
    p25_2d     = _col(p25_values)    if p25_values    is not None else mean_2d - std_2d
    p75_2d     = _col(p75_values)    if p75_values    is not None else mean_2d + std_2d
    mn_2d      = _col(min_values)    if min_values    is not None else p25_2d
    mx_2d      = _col(max_values)    if max_values    is not None else p75_2d
    mmad_2d    = _col(mean_mad_values)   if mean_mad_values   is not None else None
    medmad_2d  = _col(median_mad_values) if median_mad_values is not None else None

    # Transpose to (1, n_vials) — _compute_pk_err_bounds rows = volume index.
    pk_err_bounds = _compute_pk_err_bounds(
        mean_2d.T, med_2d.T, std_2d.T, cnt_2d.T,
        p25_2d.T, p75_2d.T, mn_2d.T, mx_2d.T,
        mean_mad_m=mmad_2d.T   if mmad_2d   is not None else None,
        median_mad_m=medmad_2d.T if medmad_2d is not None else None,
    )

    # "max" and CRC modes are included in PK_DATA.measure but omitted from
    # errBounds — the toggle JS falls through to ds.errorBars = {} for those.
    # CRC arrays are full-length; None entries render as missing chart points.
    _n = len(vials)
    _crc_3d_row  = crc_3d_chart        if crc_3d_chart        is not None else [None] * _n
    _crc_axl_row = crc_slice_avg_chart if crc_slice_avg_chart is not None else [None] * _n
    pk_data_json = json.dumps({
        "measure": {
            "mean":      mean_2d.T.tolist(),
            "median":    med_2d.T.tolist(),
            "max":       max_2d.T.tolist(),
            "crc_3d":    [_crc_3d_row],
            "crc_axial": [_crc_axl_row],
        },
        "errBounds": pk_err_bounds,
    })

    # ---- NiiVue viewer -------------------------------------------------------
    _has_viewer = bool(nifti_image and Path(nifti_image).exists())
    if _has_viewer:
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,  # type: ignore[arg-type]
            vial_niftis or {},
            bg_cal_min=0.0,
            bg_cal_max=viewer_cal_max,
        )
    else:
        viewer_html = viewer_js = ""

    # ---- Single Chart.js scatter dataset ------------------------------------
    means = mean_2d[:, 0].tolist()
    stds  = std_2d[:, 0].tolist()
    scatter_ds = {
        "label": "Measured activity",
        "_row": 0,
        "data": [{"x": j, "y": means[j]} for j in range(len(vials))],
        "borderColor": "#378ADD",
        "backgroundColor": "#378ADD33",
        "pointBackgroundColor": "#378ADD",
        "pointRadius": 6,
        "pointHoverRadius": 8,
        "borderWidth": 0,
        "showLine": False,
        "errorBars": {
            str(j): {"yMin": means[j] - stds[j], "yMax": means[j] + stds[j]}
            for j in range(len(vials))
        },
    }

    datasets_json = json.dumps([scatter_ds])
    vials_json    = json.dumps(list(vials))
    y_label_json  = json.dumps(y_label)

    # ---- CRC section (A–E) --------------------------------------------------
    def _fmt(v):
        return f"{v:.3f}" if v is not None and not math.isnan(v) else "&mdash;"

    if crc_vials and crc_3d_values is not None:
        _sa = crc_slice_avg_values or [None] * len(crc_vials)
        rows = "".join(
            f"<tr><td>{v}</td><td>{_fmt(c3d)}</td><td>{_fmt(csa)}</td></tr>"
            for v, c3d, csa in zip(crc_vials, crc_3d_values, _sa)
        )
        crc_section = f"""<div class="stats-section">
  <div class="stats-title">Contrast Recovery Coefficient (CRC)</div>
  <table class="stats-table">
    <thead><tr><th>Vial</th><th>CRC<sub>max</sub> (3D)</th><th>CRC<sub>max</sub> (slice-avg)</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
    else:
        crc_section = ""

    # ---- Uniformity section (Uniform ROI) -----------------------------------
    if (
        uniformity_vial
        and uniformity_pct is not None
        and not math.isnan(uniformity_pct)
    ):
        uniformity_section = f"""<div class="stats-section">
  <div class="stats-title">Uniformity &mdash; {uniformity_vial} ROI</div>
  <table class="stats-table">
    <thead><tr><th>ROI</th><th>Uniformity (SD / Mean &times; 100)</th></tr></thead>
    <tbody><tr><td>{uniformity_vial}</td><td>{uniformity_pct:.2f}%</td></tr></tbody>
  </table>
</div>"""
    else:
        uniformity_section = ""

    # ---- SOR section (Air / Water) ------------------------------------------
    if sor_vials and sor_values is not None:
        sor_rows = "".join(
            f"<tr><td>{v}</td><td>{_fmt(s)}</td></tr>"
            for v, s in zip(sor_vials, sor_values)
        )
        sor_section = f"""<div class="stats-section">
  <div class="stats-title">Spill-Over Ratio (SOR)</div>
  <table class="stats-table">
    <thead><tr><th>ROI</th><th>SOR (mean ROI / mean Uniform)</th></tr></thead>
    <tbody>{sor_rows}</tbody>
  </table>
</div>"""
    else:
        sor_section = ""

    data_tag = phantomkit_data_tag(embedded_data or {
        "type": "pet_vial_activity",
        "phantom": phantom,
        "vials": list(vials),
    })
    opts_js = base_opts_js(x_label="Vial", y_label=y_label, enable_zoom=True)
    head    = html_head(title, include_niivue=_has_viewer)

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Interactive plot &middot; scroll to zoom &middot; drag to pan &middot; double-click to reset view</p>

{viewer_html}

{_PET_CONTROLS_HTML}

<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-title">Measured vial activity</div>
  <div class="chart-wrap" style="height:340px"><canvas id="intensityChart"></canvas></div>
</div>

<div class="stats-section">
  <div class="stats-title">Per-vial values</div>
  <table class="stats-table" id="statsTable">
    <thead><tr id="statsHead"><th>Vial</th><th>Mean</th><th>&plusmn;SD</th></tr></thead>
    <tbody id="statsBody"></tbody>
  </table>
</div>

{crc_section}

{uniformity_section}

{sor_section}

{data_tag}

<script>
const VIALS    = {vials_json};
const DATASETS = {datasets_json};
const PK_DATA  = {pk_data_json};
const Y_LABEL  = {y_label_json};

{ERROR_BAR_PLUGIN_JS}
{_PET_TOGGLE_JS}
{opts_js}

const opts = baseOpts("Vial", {y_label_json});
opts.scales.x.type = "linear";
opts.scales.x.ticks.callback = (v) => VIALS[v] ?? v;
opts.scales.x.ticks.stepSize = 1;

const chart = new Chart(
  document.getElementById("intensityChart").getContext("2d"),
  {{ type: "line", data: {{ datasets: DATASETS }}, options: opts, plugins: [errorBarPlugin] }}
);
_pkCharts.push(chart);
document.getElementById("intensityChart").addEventListener("dblclick", () => chart.resetZoom());

function pkUpdateStatsTable() {{
  const m = window._pkMeasure, e = window._pkErrMode;
  const vals = PK_DATA.measure[m][0];
  const isCRC = _CRC_MODES.has(m);
  const measureLabel = {{
    mean:"Mean", median:"Median", max:"Max",
    crc_3d:"CRC<sub>max</sub> (3D)", crc_axial:"CRC<sub>max</sub> (axial avg)"
  }}[m] ?? m;
  if (isCRC) {{
    document.getElementById("statsHead").innerHTML =
      `<th>Vial</th><th>${{measureLabel}}</th>`;
  }} else {{
    const errLabel = (m === "max")
      ? "&mdash;"
      : e === "none"   ? "None"
      : e === "sd"     ? "&plusmn;SD"
      : e === "se"     ? "&plusmn;SE"
      : e === "2se"    ? "&plusmn;2&thinsp;SE"
      : e === "mad"    ? "&plusmn;MAD"
      : e === "iqr"    ? "IQR [Q25&ndash;Q75]"
      : "Min&ndash;Max";
    document.getElementById("statsHead").innerHTML =
      `<th>Vial</th><th>${{measureLabel}}</th><th>${{errLabel}}</th>`;
  }}
  const tbody = document.getElementById("statsBody");
  tbody.innerHTML = "";
  VIALS.forEach((v, j) => {{
    const val = vals[j];
    const hasVal = (val !== null && val !== undefined && !isNaN(val));
    const valStr = hasVal ? val.toFixed(isCRC ? 3 : 2) : "&mdash;";
    if (isCRC) {{
      tbody.innerHTML += `<tr><td>${{v}}</td><td>${{valStr}}</td></tr>`;
    }} else {{
      let errStr = "&mdash;";
      if (m !== "max" && e !== "none" && PK_DATA.errBounds[m] && PK_DATA.errBounds[m][e]) {{
        const lo = PK_DATA.errBounds[m][e].lower[0][j];
        const hi = PK_DATA.errBounds[m][e].upper[0][j];
        errStr = `[${{lo.toFixed(2)}}, ${{hi.toFixed(2)}}]`;
      }}
      tbody.innerHTML += `<tr><td>${{v}}</td><td>${{valStr}}</td><td>${{errStr}}</td></tr>`;
    }}
  }});
}}

window._pkAfterUpdate = pkUpdateStatsTable;
pkUpdateStatsTable();

{viewer_js}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# High-level entry point — reads an xlsx and writes the HTML
# ---------------------------------------------------------------------------


def plot_pet_vials(
    xlsx_path: str,
    output: str = "PET.html",
    phantom: str = "PET",
    nifti_image: Optional[str] = None,
    vial_niftis: Optional[dict] = None,
    y_label: str = "Activity (Bq/mL)",
) -> str:
    """Read a metrics xlsx and write the PET vial activity HTML report.

    The xlsx is expected to have at least a ``mean`` sheet.  Additional sheets
    (``std``, ``median``, ``max``, ``min``, ``p25``, ``p75``, ``count``,
    ``mean_mad``, ``median_mad``) are read when present and used for the
    interactive error-bar and measure-mode toggles.

    Parameters
    ----------
    xlsx_path:
        Path to the metrics xlsx produced by the phantom processor.
    output:
        Output HTML file path (extension is forced to ``.html``).
    phantom:
        Phantom name shown in the page title.
    nifti_image:
        Path to the background NIfTI for the MRI viewer panel.
    vial_niftis:
        ``{vial_name: path}`` dict for vial ROI overlay NIfTIs.
    y_label:
        Y-axis label.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    _VIAL_ORDER = ["1mm", "2mm", "3mm", "4mm", "5mm", "Air", "Water", "Uniform"]

    # Sort vial_niftis so viewer buttons follow the same order as the plot.
    if vial_niftis:
        vial_niftis = dict(
            sorted(
                vial_niftis.items(),
                key=lambda kv: _VIAL_ORDER.index(kv[0]) if kv[0] in _VIAL_ORDER else len(_VIAL_ORDER),
            )
        )

    xlsx = Path(xlsx_path)

    def _sheet(name: str) -> Optional[np.ndarray]:
        try:
            df = pd.read_excel(xlsx, sheet_name=name)
            return df.iloc[:, 1:].to_numpy()[:, 0]
        except Exception:
            return None

    mean_df  = pd.read_excel(xlsx, sheet_name="mean")
    vials    = (
        mean_df.iloc[:, 0]
        .astype(str)
        .str.replace(r"\.mif$", "", regex=True)
        .tolist()
    )
    mean_arr = mean_df.iloc[:, 1:].to_numpy()[:, 0]

    max_arr = _sheet("max")
    cal_max = float(max_arr.max()) if max_arr is not None and max_arr.size else None

    # ── CRC (sphere vials 1mm–5mm) ────────────────────────────────────────────
    crc_3d_arr    = _sheet("crc_3d")
    crc_sa_arr    = _sheet("crc_slice_avg")
    _crc_names    = {"1mm", "2mm", "3mm", "4mm", "5mm"}
    _crc_idx      = [i for i, v in enumerate(vials) if v in _crc_names]
    # Sort by sphere diameter descending (5mm → 1mm) for the static table
    _crc_idx      = sorted(_crc_idx, key=lambda i: vials[i], reverse=True)
    crc_vials_out = [vials[i] for i in _crc_idx] if _crc_idx else None
    crc_3d_out    = [float(crc_3d_arr[i]) for i in _crc_idx] if crc_3d_arr is not None and _crc_idx else None
    crc_sa_out    = [float(crc_sa_arr[i]) for i in _crc_idx] if crc_sa_arr is not None and _crc_idx else None

    # Full-length nullable arrays for the interactive chart (None = missing point)
    def _nullable(arr):
        if arr is None:
            return [None] * len(vials)
        return [None if math.isnan(float(v)) else float(v) for v in arr]

    crc_3d_chart = _nullable(crc_3d_arr)
    crc_sa_chart = _nullable(crc_sa_arr)

    # ── Uniformity (Uniform ROI) ──────────────────────────────────────────────
    uni_arr  = _sheet("uniformity")
    _u_idx   = next((i for i, v in enumerate(vials) if v == "Uniform"), None)
    uni_pct  = float(uni_arr[_u_idx]) if uni_arr is not None and _u_idx is not None else None
    uni_vial = "Uniform" if _u_idx is not None else None

    # ── SOR (Air and Water ROIs) ──────────────────────────────────────────────
    sor_arr       = _sheet("sor")
    _sor_names    = {"Air", "Water"}
    _sor_idx      = [i for i, v in enumerate(vials) if v in _sor_names]
    sor_vials_out = [vials[i] for i in _sor_idx] if _sor_idx else None
    sor_vals_out  = [float(sor_arr[i]) for i in _sor_idx] if sor_arr is not None and _sor_idx else None

    html = build_pet_html(
        vials=vials,
        mean_values=mean_arr,
        std_values=_sheet("std"),
        median_values=_sheet("median"),
        max_values=max_arr,
        count_values=_sheet("count"),
        p25_values=_sheet("p25"),
        p75_values=_sheet("p75"),
        min_values=_sheet("min"),
        mean_mad_values=_sheet("mean_mad"),
        median_mad_values=_sheet("median_mad"),
        phantom=phantom,
        title=f"{phantom} Phantom — Vial Activity",
        y_label=y_label,
        nifti_image=nifti_image,
        vial_niftis=vial_niftis,
        embedded_data={
            "type": "pet_vial_activity",
            "phantom": phantom,
            "vials": vials,
        },
        viewer_cal_max=cal_max,
        crc_vials=crc_vials_out,
        crc_3d_values=crc_3d_out,
        crc_slice_avg_values=crc_sa_out,
        uniformity_vial=uni_vial,
        uniformity_pct=uni_pct,
        sor_vials=sor_vials_out,
        sor_values=sor_vals_out,
        crc_3d_chart=crc_3d_chart,
        crc_slice_avg_chart=crc_sa_chart,
    )

    output_path = Path(output).with_suffix(".html")
    output_path.write_text(html, encoding="utf-8")
    return str(output_path.resolve())
