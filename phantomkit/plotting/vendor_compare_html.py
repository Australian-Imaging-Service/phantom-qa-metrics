"""
vendor_compare_html.py
=======================
Build the vendor-comparison HTML report: an interactive viewer (switchable
background — phantomkit's own map image when available, plus every vendor
image — with vial overlays, per-vial and toggle-all) plus a per-vial
comparison chart of phantomkit's own value against one or more vendor
images', with Mean/Median and error-bar-variant toggles.

For ADC, phantomkit's own series gets the SAME full mean/median/percentile
distribution the freshly-computed vendor series does (read from
phantom_processor.py's own per-vial xlsx, when available) — both series
respond to the Mean/Median and error-bar toggle buttons. For T1/T2, there is
no such distribution (a curve fit only produces one value per vial), so
phantomkit's point stays at a single fixed value ± its curve-fit SE, fed
into the same error-mode formula the vendor side uses, while every vendor
series (which DOES have real per-voxel stats) still responds fully.

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
# vial_intensity.py's "Reference ADC"), so vendor series use the palette
# below instead to avoid colliding with that meaning. Vendor #1 keeps the
# original yellow so the single-vendor case looks identical to before;
# additional vendors cycle through the rest of _html_common's _PALETTE,
# skipping phantomkit's own blue.
_PHANTOMKIT_COLOR = "#378ADD"
_VENDOR_COLORS = [
    "#E6B800", "#D85A30", "#7F77DD", "#1D9E75", "#BA7517",
    "#D4537E", "#639922", "#888780", "#185FA5", "#993C1D",
]
_REFERENCE_COLOR = "#C62828"


def build_vendor_compare_html(
    vendor_images: list[str],
    vendor_labels: list[str],
    vial_masks: dict[str, str],
    vendor_stats_list: list[dict],
    reference_html: str,
    map_type: str,
    output: str,
    phantom: str = "",
    template_dir: str | None = None,
    reference_xlsx: str | None = None,
    scales: list[float | None] | None = None,
    phantomkit_image: str | None = None,
) -> None:
    """Build the self-contained vendor-comparison HTML report.

    Parameters
    ----------
    vendor_images:
        Paths to the vendor NIfTIs (each becomes a selectable viewer
        background), one per vendor.
    vendor_labels:
        Display label per vendor image, same order/length as
        ``vendor_images`` — used as the chart legend entry, table column
        header, and viewer background dropdown option.
    vial_masks:
        ``{vial_name: mask_path}`` — phantomkit's own vial masks (as
        located by :func:`phantomkit.vendor_compare.locate_reference`),
        used for the viewer overlay.
    vendor_stats_list:
        One ``{vial_name: {mean, median, std, min, max, count, p25, p75,
        mean_mad, median_mad}}`` dict per vendor image, same order as
        ``vendor_images``, from
        :func:`phantomkit.vendor_compare.compute_vendor_vial_stats`.
    reference_html:
        Path to phantomkit's own existing report for this session/map-type.
    map_type:
        ``"adc"`` | ``"t1"`` | ``"t2"``.
    template_dir:
        Path to ``template_data/`` for calibration reference lookup
        (auto-detected if omitted). If no reference is available for this
        phantom/metric, the reference series is simply omitted.
    reference_xlsx:
        For ADC only: path to phantomkit's own per-vial xlsx (as located by
        :func:`phantomkit.vendor_compare.locate_reference`), giving
        phantomkit's series the same full mean/median/percentile
        distribution the vendor series has. Falls back to a fixed point ±
        std (from reference_html) if omitted or unreadable.
    scales:
        Per-vendor multiplier, same order/length as ``vendor_images``,
        applied to that vendor's series only, to bring it in line with
        phantomkit's ×10⁻³ mm²/s display convention. Vendor ADC maps show
        up in several different unit conventions in the wild (see
        :func:`phantomkit.vendor_compare.infer_adc_scale`, used to
        auto-detect a `None` entry). phantomkit's own xlsx-derived stats
        (when reference_xlsx is given) always use a fixed ×1000 —
        `_task_extract_metrics` always writes raw mrstats output
        (mm²/s), a known, fixed convention independent of whatever any
        vendor file happens to use. Unused (1.0) for T1/T2. Pass `None`
        (the whole parameter) to auto-infer every vendor's scale.
    phantomkit_image:
        Path to phantomkit's own computed map image (currently ADC only —
        `dwi_processing.py` writes `ADC.nii.gz`; T1/T2 have no equivalent
        per-voxel map), offered as an extra viewer background option.
        Omitted from the viewer entirely when `None`.
    """
    from phantomkit.plotting._html_common import (
        html_head,
        ERROR_BAR_PLUGIN_JS,
        PK_CONTROLS_HTML,
        PK_TOGGLE_JS,
        _compute_pk_err_bounds,
        base_opts_js,
        phantomkit_data_tag,
        _niivue_multi_bg_viewer_panel,
    )
    from phantomkit.plotting.compare_plots import _load_embedded, _extract, _auto_template_dir
    from phantomkit.plotting._calibration_reference import load_calibration_reference
    from phantomkit.vendor_compare import infer_adc_scale, load_full_stats_from_xlsx

    n_vendors = len(vendor_images)
    if not (len(vendor_labels) == len(vendor_stats_list) == n_vendors):
        raise ValueError(
            "vendor_images, vendor_labels, and vendor_stats_list must all "
            f"have the same length (got {n_vendors}, {len(vendor_labels)}, "
            f"{len(vendor_stats_list)})."
        )
    if scales is None:
        scales = [None] * n_vendors
    elif len(scales) != n_vendors:
        raise ValueError(
            f"scales has {len(scales)} entries but {n_vendors} vendor "
            "images were given."
        )

    metric, ref_vials_order, ref_vals, ref_se = _extract(_load_embedded(reference_html))
    y_label = _Y_LABELS.get(metric, metric)

    resolved_scales: list[float] = []
    for i in range(n_vendors):
        s = scales[i]
        if s is None:
            s = infer_adc_scale(vendor_stats_list[i]) if map_type == "adc" else 1.0
        resolved_scales.append(s)

    # phantomkit's own xlsx is always raw mrstats output (mm²/s) — a fixed,
    # known convention, independent of whichever convention any vendor
    # file happens to use. Must NOT share a vendor's `resolved_scales[i]`.
    pk_xlsx_scale = 1e3 if map_type == "adc" else 1.0

    pk_full_stats = None
    if map_type == "adc" and reference_xlsx:
        try:
            pk_full_stats = load_full_stats_from_xlsx(Path(reference_xlsx))
        except Exception:
            pk_full_stats = None

    # Common vial axis: vials present in phantomkit's reference AND every
    # vendor's stats (matched case-insensitively).
    vendor_by_upper_list = [{v.upper(): v for v in vs} for vs in vendor_stats_list]
    vials = [
        v for v in ref_vials_order
        if ref_vals.get(v.upper()) is not None
        and all(v.upper() in vbu for vbu in vendor_by_upper_list)
    ]
    if not vials:
        raise ValueError(
            "No vials in common between the phantomkit reference "
            f"({reference_html}) and all {n_vendors} vendor image(s)' vial masks."
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
    ref_value_by_vial: dict[str, float] = {}
    ref_by_temp: dict[str, list] = {}
    temperatures: list = []
    default_temp = None
    if ref_config is not None:
        default_temp = ref_config.get("default_temp")
        temperatures = ref_config.get("temperatures", [])
        values_by_temp = ref_config.get("values_by_temp", {})
        for temp_key, temp_map in values_by_temp.items():
            ref_by_temp[str(temp_key)] = [
                round(temp_map[v.upper()], 4) if temp_map.get(v.upper()) is not None else None
                for v in vials
            ]
        temp_vals = values_by_temp.get(default_temp, {})
        for j, v in enumerate(vials):
            val = temp_vals.get(v.upper())
            if val is not None:
                ref_pts.append({"x": j, "y": round(val, 4)})
                ref_value_by_vial[v] = round(val, 4)

    # Row 0 = phantomkit, rows 1..n_vendors = each vendor. Every row gets
    # the full mean/median/percentile treatment when real distribution
    # data is available (always true for vendors; true for phantomkit only
    # when reference_xlsx was found for ADC) — otherwise a row falls back
    # to a single fixed point ± one spread value, replicated across every
    # error-mode variant.
    n_rows = 1 + n_vendors
    mean_m       = np.zeros((n_rows, n))
    median_m     = np.zeros((n_rows, n))
    std_m        = np.zeros((n_rows, n))
    count_m      = np.ones((n_rows, n))
    p25_m        = np.zeros((n_rows, n))
    p75_m        = np.zeros((n_rows, n))
    min_m        = np.zeros((n_rows, n))
    max_m        = np.zeros((n_rows, n))
    mean_mad_m   = np.zeros((n_rows, n))
    median_mad_m = np.zeros((n_rows, n))

    def _fill_row(row: int, j: int, s: dict, row_scale: float) -> None:
        mean = s.get("mean")
        mean = mean if mean is not None else 0.0
        median = s.get("median")
        median = median if median is not None else mean
        std = s.get("std") or 0.0
        count = s.get("count") or 1.0
        p25 = s.get("p25")
        p25 = p25 if p25 is not None else mean - std
        p75 = s.get("p75")
        p75 = p75 if p75 is not None else mean + std
        vmin = s.get("min")
        vmin = vmin if vmin is not None else mean - std
        vmax = s.get("max")
        vmax = vmax if vmax is not None else mean + std
        mean_mad = s.get("mean_mad")
        mean_mad = mean_mad if mean_mad is not None else std
        median_mad = s.get("median_mad")
        median_mad = median_mad if median_mad is not None else std

        mean_m[row, j]       = mean * row_scale
        median_m[row, j]     = median * row_scale
        std_m[row, j]        = std * row_scale
        count_m[row, j]      = max(count, 1.0)
        p25_m[row, j]        = p25 * row_scale
        p75_m[row, j]        = p75 * row_scale
        min_m[row, j]        = vmin * row_scale
        max_m[row, j]        = vmax * row_scale
        mean_mad_m[row, j]   = mean_mad * row_scale
        median_mad_m[row, j] = median_mad * row_scale

    has_full_pk_stats = False
    for j, v in enumerate(vials):
        vu = v.upper()

        pk_full = pk_full_stats.get(vu) if pk_full_stats else None
        if pk_full and pk_full.get("mean") is not None:
            # Raw (un-scaled) xlsx stats — always phantomkit's own fixed
            # ×1000 convention, NOT any vendor's auto-detected scale.
            _fill_row(0, j, pk_full, pk_xlsx_scale)
            has_full_pk_stats = True
        else:
            # Fallback: a single fixed value ± spread, already in
            # phantomkit's display-scaled units (no further scaling).
            pk_val = ref_vals[vu]
            pk_width = ref_se.get(vu) or 0.0
            _fill_row(0, j, {"mean": pk_val, "std": pk_width}, 1.0)

        for i in range(n_vendors):
            _fill_row(
                1 + i, j,
                vendor_stats_list[i][vendor_by_upper_list[i][vu]],
                resolved_scales[i],
            )

    err_bounds = _compute_pk_err_bounds(
        mean_m, median_m, std_m, count_m, p25_m, p75_m, min_m, max_m,
        mean_mad_m=mean_mad_m, median_mad_m=median_mad_m,
    )
    pk_data = {
        "measure": {"mean": mean_m.tolist(), "median": median_m.tolist()},
        "errBounds": err_bounds,
    }

    # Horizontal offset so series don't sit exactly on top of each other at
    # each vial's x position, which made them hard to distinguish when
    # their values were close. Step shrinks as the number of rows grows so
    # the cluster never spills into the neighboring vial's space; for a
    # single vendor (n_rows=2) this reproduces the original ±0.15 layout.
    step = min(0.3, 0.6 / (n_rows - 1)) if n_rows > 1 else 0.0
    offsets = [(i - (n_rows - 1) / 2) * step for i in range(n_rows)]

    def _dataset(row: int, label: str, color: str, x_offset: float) -> dict:
        pts = [
            {"x": j + x_offset, "y": round(float(mean_m[row, j]), 4)} for j in range(n)
        ]
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
            "backgroundColor": color,
            "pointBackgroundColor": color,
            "pointBorderColor": color,
            "pointBorderWidth": 2,
            "pointRadius": 6,
            "pointHoverRadius": 8,
            "borderWidth": 1,
            "showLine": False,
        }

    datasets = [_dataset(0, "phantomkit", _PHANTOMKIT_COLOR, offsets[0])]
    for i in range(n_vendors):
        datasets.append(_dataset(
            1 + i, vendor_labels[i], _VENDOR_COLORS[i % len(_VENDOR_COLORS)], offsets[1 + i],
        ))
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

    # Per-dataset show/hide toggle buttons, matching the Compare tab's
    # session toggle pattern (compare_plots.py's pkToggleSession /
    # pkToggleAllSessions) — a colored pill button per series plus a
    # "Toggle all" button, independent of the Mean/Median/error-bar and
    # reference-temperature controls above.
    toggle_buttons_html = "".join(
        f'<button id="vc-toggle-btn-{idx}" data-visible="1" onclick="_vcToggleDataset({idx},this)"'
        f' style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;'
        f'border-radius:99px;border:1.5px solid var(--border);background:var(--bg3);'
        f'color:var(--text);font-size:12px;font-weight:500;cursor:pointer;user-select:none;'
        f'transition:opacity .15s;">'
        f'<span style="width:10px;height:10px;border-radius:50%;'
        f'background:{ds.get("pointBorderColor") or ds.get("borderColor")};flex-shrink:0;"></span>'
        f'{ds["label"]}</button>'
        for idx, ds in enumerate(datasets)
    )
    toggle_controls_html = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;">'
        '<button onclick="_vcToggleAllDatasets()" '
        'style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        'background:var(--bg3);color:var(--text2);font-size:12px;font-weight:500;'
        'cursor:pointer;user-select:none;transition:opacity .15s;">Toggle all</button>'
        f'{toggle_buttons_html}</div>'
    )

    # Only load/overlay the vials actually being compared — not every mask
    # in vial_segmentations/ (which may include vials irrelevant to this
    # map type), to keep the viewer's per-vial toggle list meaningful and
    # avoid loading more volumes than necessary.
    vial_masks_by_upper = {k.upper(): v for k, v in vial_masks.items()}
    viewer_vial_masks = {
        v: str(vial_masks_by_upper[v.upper()])
        for v in vials if v.upper() in vial_masks_by_upper
    }
    backgrounds = []
    if phantomkit_image:
        backgrounds.append({"name": "phantomkit", "path": str(phantomkit_image)})
    for i in range(n_vendors):
        backgrounds.append({"name": vendor_labels[i], "path": str(vendor_images[i])})
    # Default selection stays the first vendor image, matching the
    # single-vendor report's previous behavior, even when phantomkit's
    # image is also offered (and thus listed first in the dropdown).
    default_bg_index = 1 if phantomkit_image else 0
    viewer_html, viewer_js = _niivue_multi_bg_viewer_panel(
        backgrounds, viewer_vial_masks, default_index=default_bg_index,
    )

    vendor_word = "vendor" if n_vendors == 1 else f"{n_vendors} vendors"
    title = f"{phantom + ' — ' if phantom else ''}{vendor_word} vs phantomkit: {metric} per vial"
    embedded = {
        "type": "vendor_compare",
        "map_type": map_type,
        "phantom": phantom,
        "vials": vials,
        "phantomkit": {v: ref_vals[v.upper()] for v in vials},
        "vendors": [
            {
                "label": vendor_labels[i],
                "image": str(vendor_images[i]),
                "stats": {v: vendor_stats_list[i][vendor_by_upper_list[i][v.upper()]] for v in vials},
            }
            for i in range(n_vendors)
        ],
    }

    if has_full_pk_stats:
        footnote = (
            "Every series responds to the Mean/Median and error-bar toggles — "
            "phantomkit's own per-vial voxel distribution, from its own xlsx."
        )
    else:
        footnote = (
            "phantomkit's own value has a single fixed spread (no per-vial "
            "voxel distribution available for this map type); only the "
            "vendor series move with the Mean/Median toggle."
        )

    # ------------------------------------------------------------------
    # Per-vial comparison table: reference / vendor(s) / phantomkit values
    # and each series' percentage difference from the reference value.
    # Static cells are seeded with the initial (mean) values and refreshed
    # by _pkAfterUpdate whenever the Mean/Median toggle changes.
    # ------------------------------------------------------------------
    def _fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4g}"

    def _fmt_pct(v: float | None, ref: float | None) -> str:
        if v is None or ref is None or ref == 0:
            return "—"
        return f"{(v - ref) / ref * 100:+.1f}%"

    table_rows = ""
    for j, v in enumerate(vials):
        ref_val = ref_value_by_vial.get(v)
        pk_val = float(mean_m[0, j])
        vendor_vals = [float(mean_m[1 + i, j]) for i in range(n_vendors)]
        cells = [f'<td>{v}</td>', f'<td id="vc-ref-{j}">{_fmt(ref_val)}</td>']
        for i in range(n_vendors):
            cells.append(f'<td id="vc-vendor-{i}-{j}">{_fmt(vendor_vals[i])}</td>')
        cells.append(f'<td id="vc-pk-{j}">{_fmt(pk_val)}</td>')
        for i in range(n_vendors):
            cells.append(
                f'<td id="vc-pct-vendor-{i}-{j}">{_fmt_pct(vendor_vals[i], ref_val)}</td>'
            )
        cells.append(f'<td id="vc-pct-pk-{j}">{_fmt_pct(pk_val, ref_val)}</td>')
        table_rows += "    <tr>" + "".join(cells) + "</tr>\n"

    if temperatures:
        temp_options = "".join(
            f'<option value="{t}"{" selected" if str(t) == str(default_temp) else ""}>{t} °C</option>'
            for t in temperatures
        )
        temp_selector_html = (
            '<div style="display:flex;align-items:center;gap:10px;margin:0 0 10px;">'
            '<span style="font-size:13px;color:var(--text2);">Reference temperature:</span>'
            '<select id="vcTempSelect" onchange="_vcSetTemp(this.value)"'
            ' style="background:var(--bg3);color:var(--text);border:1px solid var(--border);'
            'border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;">'
            f'{temp_options}</select></div>'
        )
    else:
        temp_selector_html = ""

    vendor_value_headers = "".join(f"<th>{lbl}</th>" for lbl in vendor_labels)
    vendor_pct_headers = "".join(f"<th>&Delta; {lbl}</th>" for lbl in vendor_labels)

    table_html = f"""<div class="stats-section">
  <div class="stats-title">Per-vial comparison vs reference</div>
  <div style="overflow-x:auto;">
  <table class="stats-table">
    <thead>
      <tr><th>Vial</th><th>Reference</th>{vendor_value_headers}<th>phantomkit</th>{vendor_pct_headers}<th>&Delta; phantomkit</th></tr>
    </thead>
    <tbody>
{table_rows}    </tbody>
  </table>
  </div>
  <p style="font-size:11px;color:var(--text2);margin-top:8px;">
    &Delta; columns are percentage difference from the reference value. Table follows the Mean/Median toggle and reference temperature above.
  </p>
</div>"""

    ref_by_temp_json = json.dumps(ref_by_temp)
    default_temp_json = json.dumps(default_temp)

    datasets_json = json.dumps(datasets)
    vials_json = json.dumps(vials)
    pk_data_json = json.dumps(pk_data)
    y_label_json = json.dumps(y_label)
    n_vendors_json = json.dumps(n_vendors)
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
  {temp_selector_html}
  {toggle_controls_html}
  <div class="chart-wrap" style="height:420px"><canvas id="vendorCompareChart"></canvas></div>
  <p style="font-size:13px;color:var(--text2);margin-top:8px;">
    {footnote}
  </p>
</div>

{table_html}

{data_tag}

<script>
const VIALS = {vials_json};
const DATASETS = {datasets_json};
const PK_DATA = {pk_data_json};
const VC_REF_BY_TEMP = {ref_by_temp_json};
const VC_N_VENDORS = {n_vendors_json};
let _vcCurrentTemp = {default_temp_json};

{ERROR_BAR_PLUGIN_JS}
{PK_TOGGLE_JS}
{opts_js}

function _vcRefValues() {{ return VC_REF_BY_TEMP[_vcCurrentTemp] || []; }}
function _vcFmt(v) {{ return (v === null || v === undefined) ? "—" : String(Number(v.toPrecision(4))); }}
function _vcFmtPct(v, ref) {{
  if (v === null || v === undefined || ref === null || ref === undefined || ref === 0) return "—";
  const pct = (v - ref) / ref * 100;
  return (pct >= 0 ? "+" : "") + pct.toFixed(1) + "%";
}}
function _vcUpdateTable() {{
  const m = window._pkMeasure;
  const refVals = _vcRefValues();
  VIALS.forEach((v, j) => {{
    const pk = PK_DATA.measure[m][0][j];
    const ref = refVals[j];
    const refEl = document.getElementById("vc-ref-" + j);
    const pkEl = document.getElementById("vc-pk-" + j);
    const pctPkEl = document.getElementById("vc-pct-pk-" + j);
    if (refEl) refEl.textContent = _vcFmt(ref);
    if (pkEl) pkEl.textContent = _vcFmt(pk);
    if (pctPkEl) pctPkEl.textContent = _vcFmtPct(pk, ref);
    for (let i = 0; i < VC_N_VENDORS; i++) {{
      const val = PK_DATA.measure[m][1 + i][j];
      const valEl = document.getElementById("vc-vendor-" + i + "-" + j);
      const pctEl = document.getElementById("vc-pct-vendor-" + i + "-" + j);
      if (valEl) valEl.textContent = _vcFmt(val);
      if (pctEl) pctEl.textContent = _vcFmtPct(val, ref);
    }}
  }});
}}
window._pkAfterUpdate = _vcUpdateTable;

function _vcSetTemp(temp) {{
  _vcCurrentTemp = temp;
  const refDs = chart.data.datasets.find(d => d.label === "Reference");
  if (refDs) {{
    const vals = _vcRefValues();
    refDs.data = VIALS
      .map((v, j) => ({{ x: j, y: vals[j] }}))
      .filter(p => p.y !== null && p.y !== undefined);
    chart.update("none");
  }}
  _vcUpdateTable();
}}

function _vcToggleDataset(idx, btn) {{
  const ds = chart.data.datasets[idx];
  ds.hidden = !ds.hidden;
  btn.setAttribute("data-visible", ds.hidden ? "0" : "1");
  btn.style.opacity = ds.hidden ? "0.35" : "1.0";
  chart.update();
}}

function _vcToggleAllDatasets() {{
  const n = DATASETS.length;
  let anyHidden = false;
  for (let i = 0; i < n; i++) {{
    if (chart.data.datasets[i].hidden) {{ anyHidden = true; break; }}
  }}
  const makeVisible = anyHidden;
  for (let i = 0; i < n; i++) {{
    chart.data.datasets[i].hidden = !makeVisible;
    const btn = document.getElementById("vc-toggle-btn-" + i);
    if (btn) {{
      btn.setAttribute("data-visible", makeVisible ? "1" : "0");
      btn.style.opacity = makeVisible ? "1.0" : "0.35";
    }}
  }}
  chart.update();
}}

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
