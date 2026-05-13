"""
Interactive T1T2_SNRCNR.html builder for phantomkit.

Produces a self-contained HTML page with:
  1. NiiVue MRI viewer — two panels (first TE left, first IR right)
  2. SNR scatter plot — vials on x-axis, per-contrast dropdown selector
  3. CNR scatter plot — contrasts on x-axis, two-vial selector
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Selector HTML helpers
# ---------------------------------------------------------------------------


def _contrast_selector_html(contrast_labels: list[str]) -> str:
    """Dropdown selector for contrast choice."""
    options = "".join(
        f'<option value="{i}"{" selected" if i == 0 else ""}>{label}</option>'
        for i, label in enumerate(contrast_labels)
    )
    return (
        f'<select id="snr-contrast-sel" onchange="pkSetSnrContrast(this)" '
        f'style="padding:4px 8px;border:1px solid var(--border);border-radius:4px;'
        f'background:var(--bg2);color:var(--text);font-size:12px;'
        f'max-width:340px;">{options}</select>'
    )


def _cnr_vial_selector_html(vials: list[str]) -> str:
    """Two rows of vial-selector buttons for the CNR plot."""
    def _row(cls: str, which: int, active_idx: int) -> str:
        return "".join(
            f'<button class="pk-btn {cls}{" active" if i == active_idx else ""}" '
            f'data-which="{which}" data-vial="{v}" '
            f'onclick="pkSetCnrVial(this)">{v}</button>'
            for i, v in enumerate(vials)
        )

    v1 = _row("cnr-v1-btn", 1, 0)
    v2 = _row("cnr-v2-btn", 2, 1 if len(vials) > 1 else 0)
    return (
        '<div style="margin-bottom:10px;">'
        '<div style="margin-bottom:6px;">'
        '<span class="pk-ctrl-label" style="margin-right:6px;">Vial 1</span>'
        f'<span style="display:inline-flex;flex-wrap:wrap;gap:4px;">{v1}</span>'
        '</div><div>'
        '<span class="pk-ctrl-label" style="margin-right:6px;">Vial 2</span>'
        f'<span style="display:inline-flex;flex-wrap:wrap;gap:4px;">{v2}</span>'
        '</div></div>'
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_t1t2_html(
    *,
    te_nii: str | None,
    ir_nii: str | None,
    vial_niftis: dict[str, str],
    snr_data: dict,
    cnr_data: dict,
    session_name: str,
    output_file: str,
) -> str:
    """Build a self-contained interactive HTML page for T1/T2 SNR/CNR QA.

    Parameters
    ----------
    te_nii:
        Path to the first TE contrast NIfTI (left viewer panel background).
        May be None if no TE contrasts are available.
    ir_nii:
        Path to the first IR contrast NIfTI (right viewer panel background).
        May be None if no IR contrasts are available.
    vial_niftis:
        ``{vial_name: path}`` mapping for ROI overlay NIfTIs.
    snr_data:
        ``{"vials": [...], "contrasts": [...], "snr": [[c0v0,...], [c1v0,...], ...]}``.
        ``snr[contrast_idx][vial_idx]`` gives the SNR value.
    cnr_data:
        ``{"contrasts": [...], "cnr": {v1: {v2: [c0, c1, ...]}}}``.
    session_name:
        Label shown in page title.
    output_file:
        Path to write the HTML.
    """
    from phantomkit.plotting._html_common import (
        html_head,
        nifti_to_base64,
        base_opts_js,
        ERROR_BAR_PLUGIN_JS,
    )
    from phantomkit.plotting.dwi_html import _dual_viewer_panels

    title = f"T1/T2 SNR/CNR – {session_name}" if session_name else "T1/T2 SNR/CNR"

    # ── NiiVue viewer ─────────────────────────────────────────────────────────
    vials_data = [
        {"name": name, "b64": nifti_to_base64(vpath)}
        for name, vpath in sorted(vial_niftis.items())
        if vpath and Path(vpath).exists()
    ]

    _has_te = bool(te_nii and Path(te_nii).exists())
    _has_ir = bool(ir_nii and Path(ir_nii).exists())

    if _has_te and _has_ir:
        bg_te_b64 = nifti_to_base64(te_nii)
        bg_ir_b64 = nifti_to_base64(ir_nii)
        viewer_html, viewer_js = _dual_viewer_panels(
            bg_te_b64, bg_ir_b64, vials_data,
            label_left="First TE contrast",
            label_right="First IR contrast",
        )
    elif _has_te:
        from phantomkit.plotting._html_common import _niivue_viewer_panel
        viewer_html, viewer_js = _niivue_viewer_panel(te_nii, vial_niftis)
    elif _has_ir:
        from phantomkit.plotting._html_common import _niivue_viewer_panel
        viewer_html, viewer_js = _niivue_viewer_panel(ir_nii, vial_niftis)
    else:
        viewer_html = viewer_js = ""

    # ── Data ─────────────────────────────────────────────────────────────────
    vial_names: list = snr_data["vials"]
    contrast_labels: list = snr_data["contrasts"]
    n_contrasts: int = len(contrast_labels)
    snr_matrix: list = snr_data["snr"]   # [n_contrasts][n_vials]
    cnr_dict: dict = cnr_data["cnr"]

    # ── Selectors ─────────────────────────────────────────────────────────────
    contrast_selector_html = _contrast_selector_html(contrast_labels)
    cnr_selector_html = _cnr_vial_selector_html(vial_names)

    # ── JS base options ───────────────────────────────────────────────────────
    opts_js = base_opts_js(x_label="Vial", y_label="SNR", enable_zoom=False)
    head = html_head(title, include_niivue=bool(viewer_html))

    # ── SNR global y-range (fixed across all contrasts) ───────────────────────
    all_snr_vals = [v for row in snr_matrix for v in row if v is not None]
    if all_snr_vals:
        snr_ymin = min(all_snr_vals)
        snr_ymax = max(all_snr_vals)
        pad = (snr_ymax - snr_ymin) * 0.1 or 1.0
        snr_ymin = max(0.0, snr_ymin - pad)
        snr_ymax = snr_ymax + pad
    else:
        snr_ymin, snr_ymax = 0.0, 10.0

    html = f"""{head}<body>
<h1>{title}</h1>
<p class="subtitle">T1/T2 signal-to-noise and contrast-to-noise analysis</p>

{viewer_html}

<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-title">Signal-to-Noise Ratio (SNR)</div>
  <p style="font-size:12px;color:var(--text2);margin:0 0 10px;">
    SNR&thinsp;=&thinsp;vial mean&thinsp;/&thinsp;noise&thinsp;&sigma;
    &middot; noise estimated from Noise ROI
    &middot; y-axis fixed across all contrasts
  </p>
  <div style="margin-bottom:12px;">
    <span class="pk-ctrl-label" style="margin-right:6px;">Contrast</span>
    {contrast_selector_html}
  </div>
  <div class="chart-wrap" style="height:320px"><canvas id="snrChart"></canvas></div>
</div>

<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-title">Contrast-to-Noise Ratio (CNR)</div>
  <p style="font-size:12px;color:var(--text2);margin:0 0 10px;">
    CNR&thinsp;=&thinsp;|mean(vial&thinsp;1)&thinsp;&minus;&thinsp;mean(vial&thinsp;2)|&thinsp;/&thinsp;noise&thinsp;&sigma;
    &middot; across all contrasts
  </p>
  {cnr_selector_html}
  <div class="chart-wrap" style="height:320px"><canvas id="cnrChart"></canvas></div>
</div>

<script>
const SNR_MATRIX  = {json.dumps(snr_matrix)};
const CNR_DATA    = {json.dumps(cnr_dict)};
const SNR_VIALS   = {json.dumps(vial_names)};
const CNR_LABELS  = {json.dumps(contrast_labels)};
const N_CONTRASTS = {n_contrasts};
const SNR_YMIN    = {snr_ymin:.6g};
const SNR_YMAX    = {snr_ymax:.6g};
const COLOR_MAIN  = "#378ADD";

{opts_js}
{ERROR_BAR_PLUGIN_JS}

// ── SNR chart ─────────────────────────────────────────────────────────────
function _snrDatasets(contrastIdx) {{
  var vals = SNR_MATRIX[contrastIdx] || [];
  return [{{
    label: "SNR",
    data: SNR_VIALS.map(function(v, j) {{
      return {{x: j, y: vals[j] != null ? vals[j] : null}};
    }}),
    borderColor: COLOR_MAIN,
    backgroundColor: COLOR_MAIN + "44",
    pointBackgroundColor: COLOR_MAIN,
    pointRadius: 6,
    pointHoverRadius: 8,
    showLine: false,
    borderWidth: 0,
  }}];
}}

(function() {{
  var el = document.getElementById("snrChart");
  if (!el) return;
  var opts = baseOpts("Vial", "SNR");
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.callback = function(v) {{
    return SNR_VIALS[v] != null ? SNR_VIALS[v] : v;
  }};
  opts.scales.x.ticks.stepSize = 1;
  opts.scales.y.min = SNR_YMIN;
  opts.scales.y.max = SNR_YMAX;
  opts.plugins.tooltip.callbacks.label = function(ctx) {{
    return "SNR: " + ctx.parsed.y.toFixed(2);
  }};
  opts.plugins.legend.display = false;
  window._snrChart = new Chart(el.getContext("2d"), {{
    type: "scatter",
    data: {{datasets: _snrDatasets(0)}},
    options: opts,
  }});
}})();

function pkSetSnrContrast(el) {{
  var idx = parseInt(el.value);
  if (window._snrChart) {{
    window._snrChart.data.datasets = _snrDatasets(idx);
    window._snrChart.update("none");
  }}
}}

// ── CNR chart ─────────────────────────────────────────────────────────────
var _cnrVial1 = SNR_VIALS[0] || null;
var _cnrVial2 = SNR_VIALS.length > 1 ? SNR_VIALS[1] : null;

function _getCnrVals(v1, v2) {{
  if (!v1 || !v2 || v1 === v2) return null;
  if (CNR_DATA[v1] && CNR_DATA[v1][v2] !== undefined) return CNR_DATA[v1][v2];
  if (CNR_DATA[v2] && CNR_DATA[v2][v1] !== undefined) return CNR_DATA[v2][v1];
  return null;
}}

function _cnrDatasets(v1, v2) {{
  var vals = _getCnrVals(v1, v2);
  if (!vals) return [];
  return [{{
    label: "CNR",
    data: vals.map(function(y, i) {{
      return {{x: i, y: y != null ? y : null}};
    }}),
    borderColor: COLOR_MAIN,
    backgroundColor: COLOR_MAIN + "33",
    pointBackgroundColor: COLOR_MAIN,
    pointRadius: 5,
    pointHoverRadius: 7,
    showLine: false,
    borderWidth: 0,
  }}];
}}

(function() {{
  var el = document.getElementById("cnrChart");
  if (!el) return;
  var opts = baseOpts("Contrast", "CNR");
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.stepSize = 1;
  opts.scales.x.ticks.maxRotation = 60;
  opts.scales.x.ticks.minRotation = 30;
  opts.scales.x.ticks.callback = function(v) {{
    return CNR_LABELS[v] != null ? CNR_LABELS[v] : v;
  }};
  opts.plugins.tooltip.callbacks.title = function(items) {{
    return CNR_LABELS[items[0].parsed.x] || "";
  }};
  opts.plugins.tooltip.callbacks.label = function(ctx) {{
    return "CNR: " + ctx.parsed.y.toFixed(2);
  }};
  opts.plugins.legend.display = false;
  window._cnrChart = new Chart(el.getContext("2d"), {{
    type: "scatter",
    data: {{datasets: _cnrDatasets(_cnrVial1, _cnrVial2)}},
    options: opts,
  }});
}})();

function pkSetCnrVial(el) {{
  var which = parseInt(el.dataset.which);
  var vial  = el.dataset.vial;
  if (which === 1) {{
    _cnrVial1 = vial;
    document.querySelectorAll(".cnr-v1-btn").forEach(function(b) {{
      b.classList.remove("active");
    }});
  }} else {{
    _cnrVial2 = vial;
    document.querySelectorAll(".cnr-v2-btn").forEach(function(b) {{
      b.classList.remove("active");
    }});
  }}
  el.classList.add("active");
  if (window._cnrChart) {{
    window._cnrChart.data.datasets = _cnrDatasets(_cnrVial1, _cnrVial2);
    window._cnrChart.update("none");
  }}
}}

{viewer_js}
</script>
</body>
</html>"""

    Path(output_file).write_text(html, encoding="utf-8")
    return output_file
