"""
pet_html.py
Build interactive HTML for PET phantom vial activity measurements.

Reads from a metrics xlsx and generates a self-contained HTML page with:

  - NiiVue MRI viewer panel (base64-embedded NIfTI)
  - CRC chart for sphere vials (1mm–5mm) with CRCmax 3D / axial toggle
  - Static summary tables for Uniformity and SOR
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# HTML builder
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
    """Build a self-contained interactive HTML page for PET vial measurements."""
    from phantomkit.plotting._html_common import (
        _niivue_viewer_panel,
        base_opts_js,
        html_head,
        phantomkit_data_tag,
    )

    _n = len(vials)

    # ── CRC chart data ──────────────────────────────────────────────────────────
    _sphere_names = {"1mm", "2mm", "3mm", "4mm", "5mm"}
    sphere_idx    = [i for i, v in enumerate(vials) if v in _sphere_names]
    sphere_vials  = [vials[i] for i in sphere_idx]

    _c3 = crc_3d_chart        or [None] * _n
    _ca = crc_slice_avg_chart or [None] * _n

    def _clean(v):
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

    crc_3d_data    = [_clean(_c3[i]) for i in sphere_idx]
    crc_axial_data = [_clean(_ca[i]) for i in sphere_idx]

    crc_chart_data_json = json.dumps({"crc_3d": crc_3d_data, "crc_axial": crc_axial_data})
    sphere_vials_json   = json.dumps(sphere_vials)

    crc_dataset = {
        "label": "CRCmax",
        "data": [{"x": j, "y": crc_3d_data[j]} for j in range(len(sphere_vials))],
        "borderColor": "#378ADD",
        "backgroundColor": "#378ADD33",
        "pointBackgroundColor": "#378ADD",
        "pointRadius": 6,
        "pointHoverRadius": 8,
        "borderWidth": 0,
        "showLine": False,
    }
    crc_datasets_json = json.dumps([crc_dataset])

    # ── NiiVue viewer ───────────────────────────────────────────────────────────
    _has_viewer = bool(nifti_image and Path(nifti_image).exists())
    if _has_viewer:
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,
            vial_niftis or {},
            bg_cal_min=0.0,
            bg_cal_max=viewer_cal_max,
        )
    else:
        viewer_html = viewer_js = ""

    # ── Static summary tables ───────────────────────────────────────────────────
    def _fmt(v):
        return f"{v:.3f}" if v is not None and not math.isnan(v) else "&mdash;"

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

    _embed = dict(embedded_data or {})
    _embed.update({
        "type": "pet_vial_activity",
        "phantom": phantom,
        "vials": list(vials),
        "crc": {
            "vials": sphere_vials,
            "crc_3d": crc_3d_data,
            "crc_axial": crc_axial_data,
        },
        "sor": {
            "vials": sor_vials or [],
            "values": list(sor_values or []),
        },
        "uniformity": {
            "vial": uniformity_vial or "",
            "pct": (
                float(uniformity_pct)
                if uniformity_pct is not None and not math.isnan(float(uniformity_pct))
                else None
            ),
        },
    })
    data_tag = phantomkit_data_tag(_embed)
    opts_js = base_opts_js(x_label="Vial", y_label="CRCmax", enable_zoom=True)
    head    = html_head(title, include_niivue=_has_viewer)

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Interactive plot &middot; scroll to zoom &middot; drag to pan &middot; double-click to reset view</p>

{viewer_html}

<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">CRC mode</span>
    <button class="pk-btn pk-crc-mode-btn active" onclick="crcSetMode('crc_3d',this)">CRC<sub>max</sub> 3D</button>
    <button class="pk-btn pk-crc-mode-btn" onclick="crcSetMode('crc_axial',this)">CRC<sub>max</sub> axial</button>
  </div>
</div>

<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-wrap" style="height:300px"><canvas id="crcCanvas"></canvas></div>
</div>

{uniformity_section}

{sor_section}

{data_tag}

<script>
const SPHERE_VIALS   = {sphere_vials_json};
const CRC_CHART_DATA = {crc_chart_data_json};

{opts_js}

const _CRC_MODE_LABELS = {{
  crc_3d:    "CRCmax (3D)",
  crc_axial: "CRCmax (axial avg)",
}};

const crcOpts = baseOpts("Vial", "CRCmax (3D)");
crcOpts.scales.x.type = "linear";
crcOpts.scales.x.ticks.callback = (v) => SPHERE_VIALS[v] ?? v;
crcOpts.scales.x.ticks.stepSize = 1;

const crcChart = new Chart(
  document.getElementById("crcCanvas").getContext("2d"),
  {{ type: "line", data: {{ datasets: {crc_datasets_json} }}, options: crcOpts }}
);
document.getElementById("crcCanvas").addEventListener("dblclick", () => crcChart.resetZoom());

function crcSetMode(mode, btn) {{
  document.querySelectorAll(".pk-crc-mode-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  const vals = CRC_CHART_DATA[mode];
  crcChart.data.datasets[0].data.forEach((pt, i) => {{
    pt.y = (vals[i] === null || vals[i] === undefined) ? NaN : vals[i];
  }});
  crcChart.options.scales.y.title.text = _CRC_MODE_LABELS[mode];
  crcChart.update("none");
}}

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
    (``crc_3d``, ``crc_slice_avg``, ``uniformity``, ``sor``) are read when
    present and used for the CRC chart and summary tables.

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
        Y-axis label (unused by the current charts; kept for API compatibility).

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    _VIAL_ORDER = ["1mm", "2mm", "3mm", "4mm", "5mm", "Air", "Water", "Uniform"]

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

    mean_df = pd.read_excel(xlsx, sheet_name="mean")
    vials   = (
        mean_df.iloc[:, 0]
        .astype(str)
        .str.replace(r"\.mif$", "", regex=True)
        .tolist()
    )
    mean_arr = mean_df.iloc[:, 1:].to_numpy()[:, 0]

    max_arr = _sheet("max")
    cal_max = float(max_arr.max()) if max_arr is not None and max_arr.size else None

    # ── CRC (sphere vials 1mm–5mm) ────────────────────────────────────────────
    crc_3d_arr = _sheet("crc_3d")
    crc_sa_arr = _sheet("crc_slice_avg")

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
