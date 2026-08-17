"""
vendor_compare_html.py
=======================
Build the vendor-comparison HTML report: an interactive viewer (vendor image
+ vial overlays, per-vial and toggle-all) plus a per-vial comparison chart
of phantomkit's own value against the vendor's, with Mean/Median and
error-bar-variant toggles.

phantomkit's own series has no mean/median distinction (ADC: a single
displayed mean+std pulled from its own report, not a raw distribution; T1/T2:
a single curve-fit value) so its point never moves under the Mean/Median
toggle — only the freshly-computed vendor series does. The error-mode
buttons (SD/SE/2SE/MAD/IQR/Min-Max) still apply consistently to both series,
using phantomkit's own std/SE as the single spread value fed into the same
formula the vendor side uses.

Not registered as a CLI command — called directly by ``phantomkit
vendor-compare`` (phantomkit/cli.py) after computation is complete, the same
way ``phantomkit.plotting.pet_html.plot_pet_vials`` is called from the PET
pipeline stage rather than run standalone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_Y_LABELS = {
    "ADC": "ADC ×10⁻³ mm²/s",
    "T1": "T₁ (ms)",
    "T2": "T₂ (ms)",
    "Intensity": "Intensity",
}

# Colors follow the app-wide convention: red is reserved for the
# calibration-config reference value (compare_plots._REFERENCE_COLOR,
# vial_intensity.py's "Reference ADC"), so the freshly-computed vendor
# series uses yellow instead to avoid colliding with that meaning.
_PHANTOMKIT_COLOR = "#378ADD"
_VENDOR_COLOR = "#E6B800"
_REFERENCE_COLOR = "#C62828"


def build_vendor_compare_html(
    vendor_image: str,
    vial_masks: dict[str, str],
    vendor_stats: dict,
    reference_html: str,
    map_type: str,
    output: str,
    phantom: str = "",
    template_dir: str | None = None,
) -> None:
    """Build the self-contained vendor-comparison HTML report.

    Parameters
    ----------
    vendor_image:
        Path to the vendor NIfTI (used as the viewer background).
    vial_masks:
        ``{vial_name: mask_path}`` — phantomkit's own vial masks (as
        located by :func:`phantomkit.vendor_compare.locate_reference`),
        used for the viewer overlay.
    vendor_stats:
        ``{vial_name: {mean, median, std, min, max, count, p25, p75,
        mean_mad, median_mad}}`` from
        :func:`phantomkit.vendor_compare.compute_vendor_vial_stats`.
    reference_html:
        Path to phantomkit's own existing report for this session/map-type.
    map_type:
        ``"adc"`` | ``"t1"`` | ``"t2"``.
    template_dir:
        Path to ``template_data/`` for calibration reference lookup
        (auto-detected if omitted). If no reference is available for this
        phantom/metric, the reference series is simply omitted.
    """
    from phantomkit.plotting._html_common import (
        html_head,
        ERROR_BAR_PLUGIN_JS,
        PK_CONTROLS_HTML,
        PK_TOGGLE_JS,
        _compute_pk_err_bounds,
        base_opts_js,
        phantomkit_data_tag,
        _niivue_viewer_panel,
    )
    from phantomkit.plotting.compare_plots import _load_embedded, _extract, _auto_template_dir
    from phantomkit.plotting._calibration_reference import load_calibration_reference

    metric, ref_vials_order, ref_vals, ref_se = _extract(_load_embedded(reference_html))
    y_label = _Y_LABELS.get(metric, metric)
    scale = 1e3 if map_type == "adc" else 1.0

    # Common vial axis: vials present in both phantomkit's reference and the
    # freshly-computed vendor stats (matched case-insensitively).
    vendor_by_upper = {v.upper(): v for v in vendor_stats}
    vials = [
        v for v in ref_vials_order
        if v.upper() in vendor_by_upper and ref_vals.get(v.upper()) is not None
    ]
    if not vials:
        raise ValueError(
            "No vials in common between the phantomkit reference "
            f"({reference_html}) and the vendor image's vial masks."
        )
    n = len(vials)

    # Calibration-config reference values (the same "Reference" series shown
    # in vial_intensity.py/compare_plots.py) — omitted gracefully if no
    # calibration xlsx is available for this phantom/metric.
    resolved_template_dir = template_dir or _auto_template_dir()
    ref_config = (
        load_calibration_reference(resolved_template_dir, phantom, metric)
        if resolved_template_dir and phantom else None
    )
    ref_pts = []
    if ref_config is not None:
        default_temp = ref_config.get("default_temp")
        temp_vals = ref_config.get("values_by_temp", {}).get(default_temp, {})
        for j, v in enumerate(vials):
            val = temp_vals.get(v.upper())
            if val is not None:
                ref_pts.append({"x": j, "y": round(val, 4)})

    # Row 0 = phantomkit (fixed point ± one spread value), row 1 = vendor
    # (full mean/median/percentile distribution from fresh voxel stats).
    mean_m       = np.zeros((2, n))
    median_m     = np.zeros((2, n))
    std_m        = np.zeros((2, n))
    count_m      = np.ones((2, n))
    p25_m        = np.zeros((2, n))
    p75_m        = np.zeros((2, n))
    min_m        = np.zeros((2, n))
    max_m        = np.zeros((2, n))
    mean_mad_m   = np.zeros((2, n))
    median_mad_m = np.zeros((2, n))

    for j, v in enumerate(vials):
        vu = v.upper()

        pk_val = ref_vals[vu]
        pk_width = ref_se.get(vu) or 0.0
        mean_m[0, j] = median_m[0, j] = pk_val
        std_m[0, j] = pk_width
        p25_m[0, j], p75_m[0, j] = pk_val - pk_width, pk_val + pk_width
        min_m[0, j], max_m[0, j] = pk_val - pk_width, pk_val + pk_width
        mean_mad_m[0, j] = median_mad_m[0, j] = pk_width

        s = vendor_stats[vendor_by_upper[vu]]
        mean_m[1, j]   = s["mean"] * scale
        median_m[1, j] = s["median"] * scale
        std_m[1, j]    = s["std"] * scale
        count_m[1, j]  = max(s["count"], 1.0)
        p25_m[1, j]    = s["p25"] * scale
        p75_m[1, j]    = s["p75"] * scale
        min_m[1, j]    = s["min"] * scale
        max_m[1, j]    = s["max"] * scale
        mean_mad_m[1, j]   = (s["mean_mad"] or 0.0) * scale
        median_mad_m[1, j] = (s["median_mad"] or 0.0) * scale

    err_bounds = _compute_pk_err_bounds(
        mean_m, median_m, std_m, count_m, p25_m, p75_m, min_m, max_m,
        mean_mad_m=mean_mad_m, median_mad_m=median_mad_m,
    )
    pk_data = {
        "measure": {"mean": mean_m.tolist(), "median": median_m.tolist()},
        "errBounds": err_bounds,
    }

    def _dataset(row: int, label: str, color: str, filled: bool) -> dict:
        pts = [{"x": j, "y": round(float(mean_m[row, j]), 4)} for j in range(n)]
        error_bars = {
            str(j): {
                "yMin": round(float(mean_m[row, j] - std_m[row, j]), 4),
                "yMax": round(float(mean_m[row, j] + std_m[row, j]), 4),
            }
            for j in range(n)
        }
        return {
            "label": label,
            "_row": row,
            "data": pts,
            "errorBars": error_bars,
            "borderColor": color,
            "backgroundColor": color if filled else "transparent",
            "pointBackgroundColor": color if filled else "transparent",
            "pointBorderColor": color,
            "pointBorderWidth": 2,
            "pointRadius": 6,
            "pointHoverRadius": 8,
            "borderWidth": 1,
            "showLine": False,
        }

    datasets = [
        _dataset(0, "phantomkit", _PHANTOMKIT_COLOR, filled=True),
        _dataset(1, "Vendor", _VENDOR_COLOR, filled=False),
    ]
    if ref_pts:
        # Static overlay, not part of the _row/PK_DATA measure system — the
        # calibration reference has no mean/median or spread concept at all,
        # matching how vial_intensity.py's own "Reference ADC" is rendered.
        datasets.append({
            "label": "Reference",
            "data": ref_pts,
            "borderColor": "transparent",
            "backgroundColor": "transparent",
            "pointBackgroundColor": "transparent",
            "pointBorderColor": _REFERENCE_COLOR,
            "pointBorderWidth": 2,
            "pointRadius": 8,
            "pointStyle": "circle",
            "borderWidth": 0,
            "showLine": False,
        })

    # Only load/overlay the vials actually being compared — not every mask
    # in vial_segmentations/ (which may include vials irrelevant to this
    # map type), to keep the viewer's per-vial toggle list meaningful and
    # avoid loading more volumes than necessary.
    vial_masks_by_upper = {k.upper(): v for k, v in vial_masks.items()}
    viewer_vial_masks = {
        v: str(vial_masks_by_upper[v.upper()])
        for v in vials if v.upper() in vial_masks_by_upper
    }
    viewer_html, viewer_js = _niivue_viewer_panel(vendor_image, viewer_vial_masks)

    title = f"{phantom + ' — ' if phantom else ''}Vendor vs phantomkit: {metric} per vial"
    embedded = {
        "type": "vendor_compare",
        "map_type": map_type,
        "phantom": phantom,
        "vials": vials,
        "phantomkit": {v: ref_vals[v.upper()] for v in vials},
        "vendor": {v: vendor_stats[vendor_by_upper[v.upper()]] for v in vials},
    }

    datasets_json = json.dumps(datasets)
    vials_json = json.dumps(vials)
    pk_data_json = json.dumps(pk_data)
    y_label_json = json.dumps(y_label)
    data_tag = phantomkit_data_tag(embedded)
    opts_js = base_opts_js(x_label="Vial", y_label=y_label, enable_zoom=True)
    head = html_head(title, include_niivue=True)

    html = f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Interactive plot &middot; scroll to zoom &middot; drag to pan &middot; double-click to reset view</p>

{viewer_html}

{PK_CONTROLS_HTML}

<div class="chart-card">
  <div class="chart-title">{title}</div>
  <div class="chart-wrap" style="height:420px"><canvas id="vendorCompareChart"></canvas></div>
  <p style="font-size:13px;color:var(--text2);margin-top:8px;">
    phantomkit's own value has a single fixed spread (no per-vial voxel
    distribution to select mean/median from); only the Vendor series moves
    with the Mean/Median toggle.
  </p>
</div>

{data_tag}

<script>
const VIALS = {vials_json};
const DATASETS = {datasets_json};
const PK_DATA = {pk_data_json};

{ERROR_BAR_PLUGIN_JS}
{PK_TOGGLE_JS}
{opts_js}

const opts = baseOpts("Vial", {y_label_json});
opts.scales.x.type = "linear";
opts.scales.x.ticks.stepSize = 1;
opts.scales.x.ticks.callback = (v) => VIALS[v] ?? v;
opts.plugins.legend.display = true;
opts.plugins.legend.labels = {{ color: "#888780", font: {{ size: 12 }} }};

const chart = new Chart(
  document.getElementById("vendorCompareChart").getContext("2d"),
  {{ type: "line", data: {{ datasets: DATASETS }}, options: opts, plugins: [errorBarPlugin] }}
);
_pkCharts.push(chart);
document.getElementById("vendorCompareChart").addEventListener("dblclick", () => chart.resetZoom());

{viewer_js}
</script>
</body>
</html>"""

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
