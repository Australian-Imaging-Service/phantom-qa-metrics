#!/usr/bin/env python3
"""
plot_vial_intensity.py
Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, with optional ROI overlay.

Plot mode is detected automatically from the csv_file filename:

  ADC mode  – filename contains 'ADC' (case-insensitive)
    · Y-axis: ADC ×10⁻³ mm²/s
    · Reference values + ±5%/±10% tolerance bands drawn (blue crosshairs / shading)
    · Measured values plotted as filled red circles
    · Requires --phantom and --template_dir so the reference JSON can be loaded

  FA mode   – filename contains standalone 'FA' (case-insensitive, whole-word)
    · Y-axis: Fractional Anisotropy, fixed range 0–1

  Generic   – all other filenames → standard intensity plot

Output formats
  html  (default) – interactive HTML with NiiVue MRI viewer and Chart.js plot
  png             – static matplotlib figure (original behaviour)
"""

import click
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import csv
import os
import json
import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_separator(file_path):
    """Detects the most likely CSV separator by sniffing the first line."""
    try:
        with open(file_path, "r", newline="") as f:
            first_line = f.readline()
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(first_line)
            return dialect.delimiter
    except Exception as e:
        sys.exit(f"Error detecting separator: {e}")


def overlay_adc_reference(ax: plt.Axes, ref_data: dict):
    """Draw per-vial reference ADC values (open circles) onto *ax*.

    Accepts the multi-temperature format returned by load_calibration_reference.
    Values at the default temperature are used; they are already in ×10⁻³ mm²/s.
    """
    vials = ref_data["vials"]
    default_temp = ref_data.get("default_temp")
    vals = ref_data.get("values_by_temp", {}).get(default_temp, {})
    x_pos, y_pos = [], []
    for j, v in enumerate(vials):
        val = vals.get(v.upper())
        if val is not None:
            x_pos.append(j)
            y_pos.append(val)  # already in ×10⁻³ mm²/s
    if not x_pos:
        return
    ax.scatter(
        x_pos,
        y_pos,
        marker="o",
        s=110,
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.5,
        zorder=6,
    )


def build_adc_legend(has_measured: bool) -> list:
    """Return legend handles for the ADC overlay."""
    ref_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markeredgecolor="steelblue",
        markeredgewidth=1.5,
        linestyle="None",
        markersize=10,
        label="Reference ADC",
    )
    handles = [ref_handle]
    if has_measured:
        meas_handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="crimson",
            linestyle="None",
            markersize=7,
            label="Mean (SD) ADC",
        )
        handles.append(meas_handle)
    return handles


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------


def _build_vial_intensity_html(
    *,
    vials: list,
    display_values: np.ndarray,
    display_stds,
    display_medians=None,
    display_counts=None,
    display_p25=None,
    display_p75=None,
    display_mins=None,
    display_maxs=None,
    display_mean_mad=None,
    display_median_mad=None,
    n_vols: int,
    contrast_mode: str,
    ref_data: dict | None,
    plot_type: str,
    phantom: str | None,
    title: str,
    y_label: str,
    embedded_data: dict,
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
) -> str:
    """Build a self-contained HTML string for the vial intensity plot."""
    from phantomkit.plotting._html_common import (
        html_head,
        ERROR_BAR_PLUGIN_JS,
        PK_CONTROLS_HTML,
        PK_TOGGLE_JS,
        _compute_pk_err_bounds,
        base_opts_js,
        phantomkit_data_tag,
        chart_color,
        _niivue_viewer_panel,
    )
    from pathlib import Path as _Path

    _has_viewer = bool(nifti_image and _Path(nifti_image).exists())
    if _has_viewer:
        _adc_cal = contrast_mode == "adc"
        # For ADC mode, only show the vials defined in the reference (E–L);
        # the other vials are unrelated to ADC calibration.
        _viewer_vials = dict(vial_niftis or {})
        if _adc_cal and ref_data is not None:
            _adc_set = {v.upper() for v in ref_data.get("vials", [])}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _adc_set}
        elif _adc_cal:
            _adc_set = {"E", "F", "G", "H", "I", "J", "K", "L"}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _adc_set}
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,  # type: ignore[arg-type]
            _viewer_vials,
            bg_cal_min=0.0 if _adc_cal else None,
            bg_cal_max=0.004 if _adc_cal else None,
        )
    else:
        viewer_html = viewer_js = ""

    # ---- Compute PK_DATA for measure/error-bar toggles --------------------
    import numpy as _np
    _med      = display_medians    if display_medians    is not None else display_values
    _cnt      = display_counts     if display_counts     is not None else _np.full_like(display_values, 1000.0)
    _p25      = display_p25        if display_p25        is not None else (display_values - (_np.zeros_like(display_values) if display_stds is None else display_stds))
    _p75      = display_p75        if display_p75        is not None else (display_values + (_np.zeros_like(display_values) if display_stds is None else display_stds))
    _mn       = display_mins       if display_mins       is not None else _p25
    _mx       = display_maxs       if display_maxs       is not None else _p75
    _std      = display_stds       if display_stds       is not None else _np.zeros_like(display_values)
    _mean_mad  = display_mean_mad  if display_mean_mad   is not None else None
    _med_mad   = display_median_mad if display_median_mad is not None else None

    # Shape is (n_vials, n_vols); transpose to (n_vols, n_vials) so _row = vol_idx
    pk_err_bounds = _compute_pk_err_bounds(
        display_values.T, _med.T, _std.T, _cnt.T, _p25.T, _p75.T, _mn.T, _mx.T,
        mean_mad_m=_mean_mad.T if _mean_mad is not None else None,
        median_mad_m=_med_mad.T if _med_mad is not None else None,
    )
    pk_data_json = json.dumps({
        "measure": {
            "mean":   display_values.T.tolist(),
            "median": _med.T.tolist(),
        },
        "errBounds": pk_err_bounds,
    })

    # ---- Build Chart.js datasets -------------------------------------------
    datasets = []
    for vol_idx in range(n_vols):
        means = display_values[:, vol_idx].tolist()
        stds = _std[:, vol_idx].tolist() if display_stds is not None else None

        if contrast_mode == "adc":
            color = "#378ADD"  # blue filled circles for measured ADC
            vol_label = f"Vol {vol_idx}" if n_vols > 1 else "Mean (SD) ADC"
        else:
            color = chart_color(vol_idx)
            vol_label = f"Vol {vol_idx}"

        scatter_ds = {
            "label": vol_label,
            "_row": vol_idx,
            "data": [{"x": j, "y": means[j]} for j in range(len(vials))],
            "borderColor": color,
            "backgroundColor": color + "33",
            "pointBackgroundColor": color,
            "pointRadius": 5,
            "pointHoverRadius": 7,
            "borderWidth": 1 if plot_type == "scatter" else 2,
            "showLine": plot_type != "scatter",
            "tension": 0.3 if plot_type == "line" else 0,
        }
        if stds is not None:
            scatter_ds["errorBars"] = {
                str(j): {"yMin": means[j] - stds[j], "yMax": means[j] + stds[j]}
                for j in range(len(vials))
            }
        datasets.append(scatter_ds)

    ref_by_temp_json = "null"
    temp_selector_html = ""
    default_temp_js = "null"

    # Reference ADC overlay dataset (multi-temperature)
    if ref_data is not None:
        default_temp = ref_data["default_temp"]
        temp_vals = ref_data["values_by_temp"].get(default_temp, {})
        ref_pts = []
        for j, v in enumerate(vials):
            val = temp_vals.get(v.upper())
            if val is not None:
                ref_pts.append({"x": j, "y": val})
        datasets.append({
            "label": "Reference ADC",
            "data": ref_pts,
            "borderColor": "transparent",
            "backgroundColor": "transparent",
            "pointBackgroundColor": "transparent",
            "pointBorderColor": "#C62828",
            "pointBorderWidth": 2,
            "pointRadius": 8,
            "pointStyle": "circle",
            "borderWidth": 0,
            "showLine": False,
        })
        ref_by_temp: dict[str, list] = {}
        for temp_str, vals_dict in ref_data["values_by_temp"].items():
            pts = []
            for j, v in enumerate(vials):
                val = vals_dict.get(v.upper())
                if val is not None:
                    pts.append({"x": j, "y": val})
            ref_by_temp[temp_str] = pts
        ref_by_temp_json = json.dumps(ref_by_temp)
        default_temp_js = json.dumps(default_temp)
        temp_options_parts = []
        for t in ref_data["temperatures"]:
            sel = " selected" if str(t) == default_temp else ""
            temp_options_parts.append(f'<option value="{t}"{sel}>{t} °C</option>')
        temp_options = "".join(temp_options_parts)
        temp_selector_html = (
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            + '<span style="font-size:13px;color:var(--text2);">Reference temperature:</span>'
            + '<select id="refTempSelect" onchange="pkSetRefTemp(this.value)"'
            + ' style="background:var(--bg3);color:var(--text);border:1px solid var(--border);'
            + 'border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;">'
            + temp_options
            + '</select>'
            + '</div>'
        )

    datasets_json = json.dumps(datasets)
    vials_json = json.dumps(list(vials))
    title_json = json.dumps(title)
    y_label_json = json.dumps(y_label)

    data_tag = phantomkit_data_tag(embedded_data)
    opts_js = base_opts_js(x_label="Vial", y_label=y_label, enable_zoom=True)
    head = html_head(title, include_niivue=_has_viewer)

    tick_step_js = "opts.scales.x.ticks.stepSize = 1;"
    fa_ylim_js = "opts.scales.y.min = 0; opts.scales.y.max = 1;" if contrast_mode == "fa" else ""
    adc_legend_js = (
        "opts.plugins.legend.display = true;"
        " opts.plugins.legend.labels = { color: '#888780', font: { size: 12 } };"
        if contrast_mode == "adc" else ""
    )

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Interactive plot · scroll to zoom · drag to pan · double-click to reset view</p>

{viewer_html}

{PK_CONTROLS_HTML}

<div class="chart-card">
  <div class="chart-title">{title}</div>
  {temp_selector_html}
  <div class="chart-wrap" style="height:340px"><canvas id="intensityChart"></canvas></div>
</div>

<div class="stats-section">
  <div class="stats-title">Per-vial values</div>
  <table class="stats-table" id="statsTable">
    <thead><tr id="statsHead"><th>Vial</th><th>Mean</th><th>&plusmn;SD</th></tr></thead>
    <tbody id="statsBody"></tbody>
  </table>
</div>

{data_tag}

<script>
const VIALS = {vials_json};
const DATASETS = {datasets_json};
const PK_DATA = {pk_data_json};
const N_VOLS = {n_vols};
const REF_BY_TEMP = {ref_by_temp_json};
let _pkRefTemp = {default_temp_js};

function pkSetRefTemp(temp) {{
    _pkRefTemp = temp;
    const refDs = chart.data.datasets.find(d => d.label === "Reference ADC");
    if (!refDs || REF_BY_TEMP === null) return;
    refDs.data = REF_BY_TEMP[temp] || [];
    chart.update("none");
}}

{ERROR_BAR_PLUGIN_JS}
{PK_TOGGLE_JS}
{opts_js}

const opts = baseOpts("Vial", {y_label_json});
opts.scales.x.type = "linear";
opts.scales.x.ticks.callback = (v) => VIALS[v] ?? v;
{tick_step_js}
{fa_ylim_js}
{adc_legend_js}

const chart = new Chart(
  document.getElementById("intensityChart").getContext("2d"),
  {{ type: "line", data: {{ datasets: DATASETS }}, options: opts, plugins: [errorBarPlugin] }}
);
_pkCharts.push(chart);
document.getElementById("intensityChart").addEventListener("dblclick", () => chart.resetZoom());

function pkUpdateStatsTable() {{
  const m = window._pkMeasure, e = window._pkErrMode;
  const vals = PK_DATA.measure[m][0];
  const errLabel = e === "none" ? "None" : e === "sd" ? "±SD" : e === "se" ? "±SE"
    : e === "2se" ? "±2·SE" : e === "mad" ? "±MAD" : e === "iqr" ? "IQR [Q25–Q75]" : "Min–Max";
  document.getElementById("statsHead").innerHTML =
    `<th>Vial</th><th>${{m === "mean" ? "Mean" : "Median"}}</th><th>${{errLabel}}</th>`;
  const tbody = document.getElementById("statsBody");
  tbody.innerHTML = "";
  VIALS.forEach((v, j) => {{
    let errStr = "—";
    if (e !== "none" && PK_DATA.errBounds[m] && PK_DATA.errBounds[m][e]) {{
      const lo = PK_DATA.errBounds[m][e].lower[0][j];
      const hi = PK_DATA.errBounds[m][e].upper[0][j];
      errStr = `[${{lo.toFixed(4)}}, ${{hi.toFixed(4)}}]`;
    }}
    tbody.innerHTML += `<tr><td>${{v}}</td><td>${{(vals[j] ?? 0).toFixed(4)}}</td><td>${{errStr}}</td></tr>`;
  }});
}}

window._pkAfterUpdate = pkUpdateStatsTable;
pkUpdateStatsTable();

{viewer_js}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, "
            "with optional ROI overlay.\n\n"
            "Plot mode (ADC / FA / generic) is detected automatically from "
            "the csv_file filename — no flags required."
        )
    )
    # ---- positional --------------------------------------------------------
    parser.add_argument(
        "csv_file",
        help="CSV file containing mean values (vial, mean[, vol1, vol2...]).",
    )
    parser.add_argument(
        "plot_type",
        choices=["line", "bar", "scatter"],
        help="Type of plot to generate.",
    )
    # ---- standard options --------------------------------------------------
    parser.add_argument(
        "--std_csv",
        help="Optional CSV file containing standard deviations (same shape as mean CSV).",
    )
    parser.add_argument(
        "--roi_image",
        help="Optional PNG image (e.g. mrview screenshot with ROI overlay). PNG mode only.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with mean ± std values.",
    )
    parser.add_argument(
        "--output",
        default="vial_subplot.png",
        help="Filename for saving the plot (default: vial_subplot.png).",
    )
    # ---- phantom options (used when ADC mode is auto-detected) -------------
    parser.add_argument(
        "--phantom",
        default=None,
        help=(
            "Phantom name (e.g. SPIRIT).  Used to locate ADC reference data "
            "and to label plot titles.  Required when the csv_file name "
            "contains 'ADC'."
        ),
    )
    parser.add_argument(
        "--template_dir",
        default=None,
        help=(
            "Path to the TemplateData directory.  "
            "ADC reference is read from the calibration xlsx via phantom_config.json.  "
            "Required when the csv_file name contains 'ADC'."
        ),
    )

    args = parser.parse_args()
    plot_vial_intensity(
        csv_file=args.csv_file,
        plot_type=args.plot_type,
        std_csv=args.std_csv,
        roi_image=args.roi_image,
        annotate=args.annotate,
        output=args.output,
        phantom=args.phantom,
        template_dir=args.template_dir,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_vial_intensity(
    csv_file: str,
    plot_type: str = "scatter",
    std_csv: str | None = None,
    roi_image: str | None = None,
    annotate: bool = False,
    output: str = "vial_subplot.png",
    phantom: str | None = None,
    template_dir: str | None = None,
    output_format: str = "html",
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
):
    """Plot vial vs intensity. Mode (ADC/FA/generic) auto-detected from csv_file name.

    Can be called programmatically or via the CLI entry point (main()).

    Parameters
    ----------
    csv_file : str
        Path to mean values CSV.
    plot_type : str
        "scatter", "line", or "bar".
    std_csv : str, optional
        Path to standard deviations CSV (same shape as mean CSV).
    roi_image : str, optional
        Path to ROI overlay PNG. Used in PNG mode only.
    annotate : bool
        Annotate points (PNG mode only).
    output : str
        Output file path. Extension is replaced with .html when output_format="html".
    phantom : str, optional
        Phantom name for ADC reference lookup.
    template_dir : str, optional
        Directory containing phantom_config.json and calibration xlsx files.
    output_format : str
        "html" (default) for interactive HTML, or "png" for static matplotlib figure.
    """

    # ---- Auto-detect contrast mode from csv_file filename ------------------
    csv_stem = Path(csv_file).stem
    if re.search(r"ADC", csv_stem, re.IGNORECASE):
        contrast_mode = "adc"
    elif re.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", csv_stem):
        contrast_mode = "fa"
    else:
        contrast_mode = "generic"

    # ---- Validate: ADC mode needs phantom + template_dir -------------------
    if contrast_mode == "adc":
        if not phantom:
            sys.exit(
                "Error: --phantom is required when the csv_file name contains 'ADC'."
            )
        if not template_dir:
            sys.exit(
                "Error: --template_dir is required when the csv_file name contains 'ADC'."
            )

    # ---- Load mean data -----------------------------------------------------
    _is_xlsx = Path(csv_file).suffix.lower() == ".xlsx"
    if _is_xlsx:
        mean_df = pd.read_excel(csv_file, sheet_name="mean")
    else:
        sep = detect_separator(csv_file)
        mean_df = pd.read_csv(csv_file, sep=sep)
    if mean_df.shape[1] < 2:
        raise ValueError(
            "mean data must have at least two columns (vial + at least one volume)."
        )

    vials = mean_df.iloc[:, 0].astype(str).str.replace(r"\.mif$", "", regex=True)
    mean_values = mean_df.iloc[:, 1:].to_numpy()  # shape (n_vials, n_vols)
    n_vols = mean_values.shape[1]

    # ---- Load std / extra sheets -------------------------------------------
    std_values = None
    median_values = count_values = p25_values = p75_values = None
    min_values = max_values = mean_mad_values = median_mad_values = None
    if _is_xlsx:
        std_df = pd.read_excel(csv_file, sheet_name="std")
        std_values = std_df.iloc[:, 1:].to_numpy()
        try:
            median_values     = pd.read_excel(csv_file, sheet_name="median").iloc[:, 1:].to_numpy()
            count_values      = pd.read_excel(csv_file, sheet_name="count").iloc[:, 1:].to_numpy()
            p25_values        = pd.read_excel(csv_file, sheet_name="p25").iloc[:, 1:].to_numpy()
            p75_values        = pd.read_excel(csv_file, sheet_name="p75").iloc[:, 1:].to_numpy()
            min_values        = pd.read_excel(csv_file, sheet_name="min").iloc[:, 1:].to_numpy()
            max_values        = pd.read_excel(csv_file, sheet_name="max").iloc[:, 1:].to_numpy()
        except Exception:
            pass  # older xlsx without extra sheets — graceful degradation
        try:
            mean_mad_values   = pd.read_excel(csv_file, sheet_name="mean_mad").iloc[:, 1:].to_numpy()
            median_mad_values = pd.read_excel(csv_file, sheet_name="median_mad").iloc[:, 1:].to_numpy()
        except Exception:
            pass  # older xlsx without MAD sheets
    elif std_csv:
        sep_std = detect_separator(std_csv)
        std_df = pd.read_csv(std_csv, sep=sep_std)
        if std_df.shape[1] < 2:
            raise ValueError(
                "std CSV must have at least two columns (vial + at least one volume)."
            )
        std_values = std_df.iloc[:, 1:].to_numpy()  # shape (n_vials, n_vols)

    # ---- Load ADC reference (ADC mode only) --------------------------------
    ref_data = None
    if contrast_mode == "adc" and template_dir:
        from phantomkit.plotting._calibration_reference import load_calibration_reference
        ref_data = load_calibration_reference(template_dir, phantom, "ADC")

    # ---- In ADC mode, restrict to vials with known calibration reference ------
    if contrast_mode == "adc" and ref_data is not None:
        adc_vials = [v.upper() for v in ref_data["vials"]]
        mask = vials.str.upper().isin(adc_vials)
        vials = vials[mask].reset_index(drop=True)
        mean_values = mean_values[mask.values]
        if std_values is not None:
            std_values = std_values[mask.values]
        if median_values is not None:
            median_values = median_values[mask.values]
        if count_values is not None:
            count_values = count_values[mask.values]
        if p25_values is not None:
            p25_values = p25_values[mask.values]
        if p75_values is not None:
            p75_values = p75_values[mask.values]
        if min_values is not None:
            min_values = min_values[mask.values]
        if max_values is not None:
            max_values = max_values[mask.values]
        if mean_mad_values is not None:
            mean_mad_values = mean_mad_values[mask.values]
        if median_mad_values is not None:
            median_mad_values = median_mad_values[mask.values]

    # ---- In ADC mode, scale measured values to ×10⁻³ for display ----------
    _scale = 1e3 if contrast_mode == "adc" else 1.0
    display_values      = mean_values * _scale
    display_stds        = std_values        * _scale if std_values        is not None else None
    display_medians     = median_values     * _scale if median_values     is not None else None
    display_counts      = count_values               if count_values      is not None else None
    display_p25         = p25_values        * _scale if p25_values        is not None else None
    display_p75         = p75_values        * _scale if p75_values        is not None else None
    display_mins        = min_values        * _scale if min_values        is not None else None
    display_maxs        = max_values        * _scale if max_values        is not None else None
    display_mean_mad    = mean_mad_values   * _scale if mean_mad_values   is not None else None
    display_median_mad  = median_mad_values * _scale if median_mad_values is not None else None

    # ---- Build axis labels and title ---------------------------------------
    phantom_label = f"{phantom} Phantom – " if phantom else ""
    if contrast_mode == "adc":
        y_label = "ADC ×10⁻³ mm²/s"
        title = f"{phantom_label}Measured vs Reference ADC"
    elif contrast_mode == "fa":
        y_label = "Fractional Anisotropy"
        title = f"{phantom_label}Fractional Anisotropy (Mean ± Std)"
    else:
        y_label = "Intensity"
        title = f"{phantom_label}Vial vs Intensity (Mean ± Std)"

    # ========================================================================
    # HTML output path
    # ========================================================================
    if output_format == "html":
        output_file = os.path.abspath(
            re.sub(r"\.(png|html)$", "", output, flags=re.IGNORECASE) + ".html"
        )

        embedded_data = {
            "type": "vial_intensity",
            "contrast_mode": contrast_mode,
            "phantom": phantom,
            "vials": list(vials),
            "means": display_values.tolist(),
            "stds": display_stds.tolist() if display_stds is not None else None,
        }

        html = _build_vial_intensity_html(
            vials=list(vials),
            display_values=display_values,
            display_stds=display_stds,
            display_medians=display_medians,
            display_counts=display_counts,
            display_p25=display_p25,
            display_p75=display_p75,
            display_mins=display_mins,
            display_maxs=display_maxs,
            display_mean_mad=display_mean_mad,
            display_median_mad=display_median_mad,
            n_vols=n_vols,
            contrast_mode=contrast_mode,
            ref_data=ref_data,
            plot_type=plot_type,
            phantom=phantom,
            title=title,
            y_label=y_label,
            embedded_data=embedded_data,
            nifti_image=nifti_image,
            vial_niftis=vial_niftis,
        )

        Path(output_file).write_text(html, encoding="utf-8")
        return output_file

    # ========================================================================
    # PNG output path (original matplotlib behaviour)
    # ========================================================================
    # ---- Setup figure ------------------------------------------------------
    ncols = 2 if roi_image else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), squeeze=False)
    axes = axes[0]  # flatten row
    ax = axes[0]

    # ---- Plot each volume --------------------------------------------------
    cmap = plt.get_cmap("tab10")

    for vol_idx in range(n_vols):
        means = display_values[:, vol_idx]
        stds = display_stds[:, vol_idx] if display_stds is not None else None

        if contrast_mode == "adc":
            color = "crimson"
            fmt_line = "-o"
            fmt_scatter = "o"
            marker_kw = dict(
                markersize=7, markerfacecolor="crimson", markeredgecolor="crimson"
            )
            vol_label = f"Vol {vol_idx}" if n_vols > 1 else "Mean (SD) ADC"
        else:
            color = cmap(vol_idx % 10)
            fmt_line = "-o"
            fmt_scatter = "o"
            marker_kw = {}
            vol_label = f"Vol {vol_idx}"

        if plot_type == "line":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt=fmt_line,
                capsize=5,
                color=color,
                label=vol_label,
                **marker_kw,
            )
        elif plot_type == "bar":
            x = np.arange(len(vials)) + (vol_idx - n_vols / 2) * 0.1
            ax.bar(
                x, means, yerr=stds, capsize=5, color=color, width=0.1, label=vol_label
            )
            ax.set_xticks(np.arange(len(vials)))
            ax.set_xticklabels(vials)
        elif plot_type == "scatter":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt=fmt_scatter,
                capsize=5,
                color=color,
                label=vol_label,
                **marker_kw,
            )

        if annotate:
            for vial, mean, std in zip(
                vials, means, stds if stds is not None else [0] * len(means)
            ):
                ax.text(
                    vial,
                    mean + (max(means) * 0.02),
                    f"{mean:.3f}±{std:.3f}" if stds is not None else f"{mean:.3f}",
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    # ---- ADC reference overlay (drawn after measured values so crosshairs
    #      are visible even when they coincide with measured data) -----------
    if ref_data is not None:
        overlay_adc_reference(ax, ref_data)

    # ---- Axis labels -------------------------------------------------------
    ax.set_xlabel("Vial", fontsize=12)

    if contrast_mode == "adc":
        ax.set_ylabel("ADC ×10⁻³ mm²/s", fontsize=12)
        ax.set_title(
            f"{phantom_label}Measured vs Reference ADC", fontsize=13, fontweight="bold"
        )
        # Plain numeric ticks (values are already in ×10⁻³)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
    elif contrast_mode == "fa":
        ax.set_ylabel("Fractional Anisotropy", fontsize=12)
        ax.set_title(
            f"{phantom_label}Fractional Anisotropy (Mean ± Std)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title(f"{phantom_label}Vial vs Intensity (Mean ± Std)", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Legend ------------------------------------------------------------
    if contrast_mode == "adc":
        extra_handles = build_adc_legend(has_measured=True)
        if n_vols > 1:
            vol_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="crimson",
                    linestyle="-" if plot_type == "line" else "None",
                    markersize=7,
                    label=f"Vol {i}",
                    alpha=0.4 + 0.6 * i / max(n_vols - 1, 1),
                )
                for i in range(n_vols)
            ]
            ax.legend(handles=extra_handles + vol_handles, fontsize=10)
        else:
            ax.legend(handles=extra_handles, fontsize=10)
    else:
        ax.legend(title="Volumes", fontsize=10)

    # ---- ROI overlay subplot -----------------------------------------------
    if roi_image:
        ax_img = axes[1]
        img = mpimg.imread(roi_image)
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title("Contrast with Vial ROIs")

    plt.tight_layout()
    output_file = os.path.abspath(output)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_file


@click.command("vial-intensity")
@click.argument("csv_file")
@click.argument("plot_type", default="scatter")
@click.option("--std-csv", default=None)
@click.option("--roi-image", default=None, help="ROI overlay PNG (PNG mode only).")
@click.option("--annotate", is_flag=True, default=False)
@click.option("--output", default="vial_subplot.png")
@click.option("--phantom", default=None)
@click.option("--template-dir", default=None)
@click.option(
    "--format", "output_format",
    default="html",
    type=click.Choice(["html", "png"]),
    show_default=True,
    help="Output format: interactive HTML (default) or static PNG.",
)
def main(csv_file, plot_type, std_csv, roi_image, annotate, output,
         phantom, template_dir, output_format):
    """Plot vial vs intensity (mean ± std) for 3D or 4D contrasts."""
    plot_vial_intensity(
        csv_file=csv_file,
        plot_type=plot_type,
        std_csv=std_csv,
        roi_image=roi_image,
        annotate=annotate,
        output=output,
        phantom=phantom,
        template_dir=template_dir,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
