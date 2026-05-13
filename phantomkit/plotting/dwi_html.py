"""
Interactive DWI.html builder for phantomkit.

Produces a self-contained HTML page with:
  1. NiiVue MRI viewer(s) — single or two-column (raw vs preprocessed)
  2. SNR scatter plot — vials on x-axis, fixed y-range, per-volume selector
  3. CNR scatter plot — volumes on x-axis, two-vial selector
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Selector HTML helpers
# ---------------------------------------------------------------------------

def _vol_selector_html(n_vols: int) -> str:
    """Volume selector — buttons for ≤20 vols, dropdown otherwise."""
    if n_vols <= 20:
        buttons = "".join(
            f'<button class="pk-btn snr-vol-btn{" active" if i == 0 else ""}" '
            f'data-vol="{i}" onclick="pkSetSnrVol(this)">Vol {i}</button>'
            for i in range(n_vols)
        )
        return f'<div style="display:inline-flex;flex-wrap:wrap;gap:4px;">{buttons}</div>'
    options = "".join(f'<option value="{i}">Vol {i}</option>' for i in range(n_vols))
    return (
        f'<select onchange="pkSetSnrVol(this)" '
        f'style="padding:4px 8px;border:1px solid var(--border);border-radius:4px;'
        f'background:var(--bg2);color:var(--text);font-size:12px;">{options}</select>'
    )


def _cnr_vial_selector_html(vials: list[str]) -> str:
    """Two rows of vial-selector buttons for CNR plot.

    Uses data-which and data-vial attributes to avoid HTML quoting issues.
    """
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
# NiiVue viewer panel builders
# ---------------------------------------------------------------------------

def _single_viewer_panel(
    bg_b64: str, vials_data: list[dict], canvas_id: str = "nv-canvas",
) -> tuple[str, str]:
    """Build HTML + JS for a single NiiVue viewer panel."""
    from phantomkit.plotting._html_common import _niivue_script_tag

    chips = "".join(
        f'<button id="pk-chip-{i}" data-active="1" onclick="pkToggleVial({i+1},this)"'
        f' style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        f'background:var(--bg3);color:var(--text);font-size:11px;font-weight:500;'
        f'cursor:pointer;user-select:none;transition:opacity .15s;">'
        f'{v["name"]}</button>'
        for i, v in enumerate(vials_data)
    )
    html = (
        f'<canvas id="{canvas_id}" style="width:100%;height:300px;display:block;'
        f'background:#000;border-radius:6px;cursor:crosshair;"></canvas>'
    )
    vials_js = json.dumps(vials_data)
    bg_js = json.dumps(bg_b64)
    js = f"""
var _NV_BG_{canvas_id} = {bg_js};
var _NV_VIALS_{canvas_id} = {vials_js};

(function() {{
  var canvas = document.getElementById("{canvas_id}");
  if (!canvas) return;
  var ro = new ResizeObserver(function(entries) {{
    for (var i = 0; i < entries.length; i++) {{
      var r = entries[i].contentRect;
      var w = Math.round(r.width), h = Math.round(r.height);
      if (w > 0 && h > 0) {{ ro.disconnect(); _pkInitNv_{canvas_id}(canvas, w, h); break; }}
    }}
  }});
  ro.observe(canvas);
}})();

function _pkB64ToUrl(b64) {{
  var bin = atob(b64), arr = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([arr], {{type:"application/octet-stream"}}));
}}

function _pkInitNv_{canvas_id}(canvas, w, h) {{
  canvas.width = w; canvas.height = h;
  canvas.addEventListener("wheel", function(e) {{
    e.preventDefault();
    if ((e.ctrlKey || e.metaKey) && window._pkNv && window._pkNv.scene) {{
      e.stopPropagation();
      pkZoom(e.deltaY < 0 ? 1.15 : 1/1.15);
    }}
  }}, {{passive: false}});
  var nv = new niivue.Niivue({{
    isColorbar: false, crosshairWidth: 1, isResizeCanvas: false,
    isAntiAlias: false, multiplanarLayout: 3, multiplanarShowRender: 0,
  }});
  nv.attachToCanvas(canvas);
  nv.opts.sliceType = 3;
  var vols = [{{url: _pkB64ToUrl(_NV_BG_{canvas_id}), name: "background.nii.gz", colormap: "gray"}}];
  for (var i = 0; i < _NV_VIALS_{canvas_id}.length; i++) {{
    vols.push({{
      url: _pkB64ToUrl(_NV_VIALS_{canvas_id}[i].b64),
      name: _NV_VIALS_{canvas_id}[i].name + ".nii.gz",
      colormap: "red", opacity: 0.5,
    }});
  }}
  nv.loadVolumes(vols).then(function() {{
    nv.setRadiologicalConvention(true);
  }});
  window._pkNv = nv;
}}
"""
    return html, js


def _dual_viewer_panels(
    bg_left_b64: str, bg_right_b64: str,
    vials_data: list[dict],
    label_left: str = "Raw DWI",
    label_right: str = "DWI preproc + bias corr",
) -> tuple[str, str]:
    """Build HTML + JS for two synchronized side-by-side NiiVue viewers.

    Shared controls:
      - Vial chip buttons toggle both viewers
      - Zoom buttons / Ctrl+scroll affect both
      - Radiological convention button affects both
      - Slice position is synced via onLocationChange
    """
    _zoom_btn_style = (
        "padding:3px 8px;border-radius:99px;border:1.5px solid var(--border);"
        "background:var(--bg3);color:var(--text2);font-size:13px;font-weight:600;"
        "cursor:pointer;user-select:none;transition:opacity .15s;line-height:1;"
    )

    chips = "".join(
        f'<button id="pk-chip-{i}" data-active="1" onclick="pkToggleVial({i+1},this)"'
        f' style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        f'background:var(--bg3);color:var(--text);font-size:11px;font-weight:500;'
        f'cursor:pointer;user-select:none;transition:opacity .15s;">'
        f'{v["name"]}</button>'
        for i, v in enumerate(vials_data)
    )
    _toggle_all_btn = (
        '<button id="pk-toggle-all-btn" onclick="pkToggleAllVials()" '
        'style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        'background:var(--bg3);color:var(--text2);font-size:11px;font-weight:500;'
        'cursor:pointer;user-select:none;transition:opacity .15s;">Hide All</button>'
    ) if vials_data else ""
    chips_html = (
        '<p style="font-size:11px;color:var(--text2);margin-top:10px;margin-bottom:6px;">'
        'Toggle vial ROIs (synced)</p>'
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">'
        f'{_toggle_all_btn}{chips}</div>'
        if vials_data else ""
    )

    html = f"""<div class="chart-card" style="margin-bottom:20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <p class="chart-title" style="margin:0;">MRI Viewer</p>
    <div style="display:flex;gap:6px;align-items:center;">
      <button onclick="pkZoom(1.25)" style="{_zoom_btn_style}">+</button>
      <button onclick="pkZoom(1/1.25)" style="{_zoom_btn_style}">&minus;</button>
      <button id="pk-rad-btn" data-rad="1" onclick="pkToggleRadConvention(this)"
        style="padding:3px 10px;border-radius:99px;border:1.5px solid var(--border);
        background:var(--bg3);color:var(--text2);font-size:11px;font-weight:500;
        cursor:pointer;user-select:none;transition:opacity .15s;">Radiological</button>
    </div>
  </div>
  <div class="panel-row" style="margin-bottom:4px;">
    <div>
      <p style="font-size:11px;color:var(--text2);margin-bottom:4px;">{label_left}</p>
      <canvas id="nv-canvas-raw" style="width:100%;height:280px;display:block;background:#000;border-radius:6px;cursor:crosshair;"></canvas>
    </div>
    <div>
      <p style="font-size:11px;color:var(--text2);margin-bottom:4px;">{label_right}</p>
      <canvas id="nv-canvas-proc" style="width:100%;height:280px;display:block;background:#000;border-radius:6px;cursor:crosshair;"></canvas>
    </div>
  </div>
  <p style="font-size:11px;color:var(--text2);margin-top:4px;">Scroll: change slice &middot; Ctrl+scroll or +/&minus;: zoom (synced) &middot; drag: adjust contrast &middot; slices synced between views</p>
  {chips_html}
</div>"""

    bg_raw_js = json.dumps(bg_left_b64)
    bg_proc_js = json.dumps(bg_right_b64)
    vials_js = json.dumps(vials_data)

    js = f"""
var _NV_BG_RAW  = {bg_raw_js};
var _NV_BG_PROC = {bg_proc_js};
var _NV_VIALS   = {vials_js};
var _pkSyncLock = false;

function _pkB64ToUrl(b64) {{
  var bin = atob(b64), arr = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([arr], {{type:"application/octet-stream"}}));
}}

function _pkSyncViewers(src, dst) {{
  if (_pkSyncLock || !src || !dst || !src.scene || !dst.scene) return;
  _pkSyncLock = true;
  try {{
    dst.scene.crosshairPos = src.scene.crosshairPos.slice();
    dst.drawScene();
  }} finally {{ _pkSyncLock = false; }}
}}

function _pkInitNvCanvas(canvasId, bgB64, varName) {{
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;
  var ro = new ResizeObserver(function(entries) {{
    for (var i = 0; i < entries.length; i++) {{
      var r = entries[i].contentRect;
      var w = Math.round(r.width), h = Math.round(r.height);
      if (w > 0 && h > 0) {{ ro.disconnect(); _pkMakeNv(canvas, w, h, bgB64, varName); break; }}
    }}
  }});
  ro.observe(canvas);
}}

function _pkMakeNv(canvas, w, h, bgB64, varName) {{
  canvas.width = w; canvas.height = h;
  canvas.addEventListener("wheel", function(e) {{
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {{
      e.stopPropagation();
      pkZoom(e.deltaY < 0 ? 1.15 : 1/1.15);
    }}
  }}, {{passive: false}});
  var nv = new niivue.Niivue({{
    isColorbar: false, crosshairWidth: 1, isResizeCanvas: false,
    isAntiAlias: false, multiplanarLayout: 3, multiplanarShowRender: 0,
  }});
  nv.attachToCanvas(canvas);
  nv.opts.sliceType = 3;
  var vols = [{{url: _pkB64ToUrl(bgB64), name: "background.nii.gz", colormap: "gray"}}];
  for (var i = 0; i < _NV_VIALS.length; i++) {{
    vols.push({{
      url: _pkB64ToUrl(_NV_VIALS[i].b64),
      name: _NV_VIALS[i].name + ".nii.gz",
      colormap: "red", opacity: 0.5,
    }});
  }}
  nv.loadVolumes(vols).then(function() {{
    nv.setRadiologicalConvention(true);
    // Wire up sync after both viewers ready
    nv.onLocationChange = function() {{
      if (varName === "Raw") _pkSyncViewers(window._pkNvRaw, window._pkNvProc);
      else                   _pkSyncViewers(window._pkNvProc, window._pkNvRaw);
    }};
  }});
  if (varName === "Raw") window._pkNvRaw  = nv;
  else                   window._pkNvProc = nv;
}}

_pkInitNvCanvas("nv-canvas-raw",  _NV_BG_RAW,  "Raw");
_pkInitNvCanvas("nv-canvas-proc", _NV_BG_PROC, "Proc");

function pkToggleVial(idx, btn) {{
  var wasActive = btn.dataset.active === "1";
  btn.dataset.active = wasActive ? "0" : "1";
  btn.style.opacity = wasActive ? "0.35" : "1.0";
  var op = wasActive ? 0.0 : 0.5;
  for (var nv of [window._pkNvRaw, window._pkNvProc]) {{
    if (nv) nv.setOpacity(idx, op);
  }}
}}

function pkZoom(factor) {{
  for (var nv of [window._pkNvRaw, window._pkNvProc]) {{
    if (nv && nv.scene) {{
      nv.scene.pan2Dxyzmm[3] = Math.max(0.1, nv.scene.pan2Dxyzmm[3] * factor);
      nv.drawScene();
    }}
  }}
}}

function pkToggleRadConvention(btn) {{
  var isRad = btn.dataset.rad === "1";
  btn.dataset.rad = isRad ? "0" : "1";
  btn.textContent = isRad ? "Neurological" : "Radiological";
  for (var nv of [window._pkNvRaw, window._pkNvProc]) {{
    if (nv) nv.setRadiologicalConvention(!isRad);
  }}
}}

function pkToggleAllVials() {{
  var btns = Array.from(document.querySelectorAll('[id^="pk-chip-"]'));
  if (!btns.length) return;
  var anyActive = btns.some(function(b) {{ return b.dataset.active === "1"; }});
  btns.forEach(function(b) {{
    var isActive = b.dataset.active === "1";
    if (anyActive ? isActive : !isActive) b.click();
  }});
  var toggleBtn = document.getElementById("pk-toggle-all-btn");
  if (toggleBtn) {{
    var nowAny = btns.some(function(b) {{ return b.dataset.active === "1"; }});
    toggleBtn.textContent = nowAny ? "Hide All" : "Show All";
  }}
}}
"""
    return html, js


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dwi_html(
    *,
    meanb0_nii: str | None,
    vial_niftis: dict[str, str],
    snr_data: dict,
    cnr_data: dict,
    session_name: str,
    output_file: str,
    raw_meanb0_nii: str | None = None,
    raw_snr_data: dict | None = None,
    raw_cnr_data: dict | None = None,
) -> str:
    """Build a self-contained interactive HTML page for DWI QA.

    Parameters
    ----------
    meanb0_nii:
        Path to the mean b=0 NIfTI for the preprocessed DWI viewer background.
    vial_niftis:
        ``{vial_name: path}`` mapping for ROI overlay NIfTIs.
    snr_data / cnr_data:
        Preprocessed data dicts — ``{"vials": [...], "n_vols": int, ...}``.
    session_name:
        Label shown in page title.
    output_file:
        Path to write the HTML.
    raw_meanb0_nii / raw_snr_data / raw_cnr_data:
        Optional raw DWI equivalents — triggers two-column layout when provided.
    """
    from phantomkit.plotting._html_common import (
        html_head,
        nifti_to_base64,
        _niivue_viewer_panel,
        base_opts_js,
        ERROR_BAR_PLUGIN_JS,
    )

    title = f"DWI – {session_name}" if session_name else "DWI QA"
    two_col = bool(raw_meanb0_nii and Path(raw_meanb0_nii).exists()
                   and raw_snr_data and raw_cnr_data)

    # ── NiiVue viewer(s) ──────────────────────────────────────────────────────
    vials_data = [
        {"name": name, "b64": nifti_to_base64(vpath)}
        for name, vpath in sorted(vial_niftis.items())
        if vpath and Path(vpath).exists()
    ]

    _has_proc_bg = bool(meanb0_nii and Path(meanb0_nii).exists())
    if two_col:
        bg_proc_b64 = nifti_to_base64(meanb0_nii) if _has_proc_bg else ""
        bg_raw_b64  = nifti_to_base64(raw_meanb0_nii)
        viewer_html, viewer_js = _dual_viewer_panels(
            bg_raw_b64, bg_proc_b64, vials_data,
        )
    elif _has_proc_bg:
        # Single-column: reuse the common viewer panel
        viewer_html, viewer_js = _niivue_viewer_panel(meanb0_nii, vial_niftis)
    else:
        viewer_html = viewer_js = ""

    # ── Data ──────────────────────────────────────────────────────────────────
    snr_vials: list = snr_data["vials"]
    n_vols: int = snr_data["n_vols"]
    snr_proc: list = snr_data["snr"]    # [n_vols][n_vials]
    cnr_proc: dict = cnr_data["cnr"]

    snr_raw  = raw_snr_data["snr"]  if raw_snr_data else None
    cnr_raw  = raw_cnr_data["cnr"]  if raw_cnr_data else None

    # ── Selectors ─────────────────────────────────────────────────────────────
    vol_selector_html = _vol_selector_html(n_vols)
    cnr_selector_html = _cnr_vial_selector_html(snr_vials)

    # ── JS utils ──────────────────────────────────────────────────────────────
    opts_js = base_opts_js(x_label="Vial", y_label="SNR", enable_zoom=False)
    head = html_head(title, include_niivue=bool(viewer_html))

    # ── SNR global y-range (fixed across all volumes, both raw + proc) ────────
    all_snr_vals = [v for row in snr_proc for v in row if v is not None]
    if snr_raw:
        all_snr_vals += [v for row in snr_raw for v in row if v is not None]
    if all_snr_vals:
        snr_ymin = min(all_snr_vals)
        snr_ymax = max(all_snr_vals)
        pad = (snr_ymax - snr_ymin) * 0.1 or 1.0
        snr_ymin = max(0.0, snr_ymin - pad)
        snr_ymax = snr_ymax + pad
    else:
        snr_ymin, snr_ymax = 0.0, 10.0

    snr_section_label = '<div class="chart-title">Signal-to-Noise Ratio (SNR)</div>'
    cnr_section_label = '<div class="chart-title">Contrast-to-Noise Ratio (CNR)</div>'

    # ── Chart HTML ────────────────────────────────────────────────────────────
    snr_charts_html = '<div class="chart-wrap" style="height:320px"><canvas id="snrChart"></canvas></div>'
    cnr_charts_html = '<div class="chart-wrap" style="height:320px"><canvas id="cnrChart"></canvas></div>'

    html = f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">DWI signal-to-noise and contrast-to-noise analysis</p>

{viewer_html}

<div class="chart-card" style="margin-bottom:20px;">
  {snr_section_label}
  <p style="font-size:12px;color:var(--text2);margin:0 0 10px;">
    SNR&thinsp;=&thinsp;vial mean / noise&thinsp;&sigma; &middot; noise estimated from Noise ROI &middot; y-axis range fixed across all volumes{"&thinsp;&middot;&thinsp;<span style='color:#378ADD;font-weight:600;'>&#9679;</span> Raw DWI &thinsp;<span style='color:#27AE60;font-weight:600;'>&#9679;</span> Preprocessed &thinsp;&middot;&thinsp;click legend to toggle" if two_col else ""}
  </p>
  <div style="margin-bottom:12px;">
    <span class="pk-ctrl-label" style="margin-right:6px;">Volume</span>
    {vol_selector_html}
  </div>
  {snr_charts_html}
</div>

<div class="chart-card" style="margin-bottom:20px;">
  {cnr_section_label}
  <p style="font-size:12px;color:var(--text2);margin:0 0 10px;">
    CNR&thinsp;=&thinsp;|mean(vial 1)&thinsp;&minus;&thinsp;mean(vial 2)|&thinsp;/&thinsp;noise&thinsp;&sigma; &middot; across all DWI volumes{"&thinsp;&middot;&thinsp;<span style='color:#378ADD;font-weight:600;'>&#9679;</span> Raw DWI &thinsp;<span style='color:#27AE60;font-weight:600;'>&#9679;</span> Preprocessed &thinsp;&middot;&thinsp;click legend to toggle" if two_col else ""}
  </p>
  {cnr_selector_html}
  {cnr_charts_html}
</div>

<script>
const SNR_PROC   = {json.dumps(snr_proc)};
const SNR_RAW    = {json.dumps(snr_raw)};
const CNR_PROC   = {json.dumps(cnr_proc)};
const CNR_RAW    = {json.dumps(cnr_raw)};
const SNR_VIALS  = {json.dumps(snr_vials)};
const N_VOLS     = {json.dumps(n_vols)};
const SNR_YMIN   = {snr_ymin:.6g};
const SNR_YMAX   = {snr_ymax:.6g};
const COLOR_RAW  = "#378ADD";
const COLOR_PROC = "#27AE60";

{opts_js}
{ERROR_BAR_PLUGIN_JS}

// ── SNR chart ─────────────────────────────────────────────────────────────
function _snrDatasets(volIdx) {{
  var datasets = [];
  if (SNR_RAW) {{
    var rawVals = SNR_RAW[volIdx] || [];
    datasets.push({{
      label: "Raw DWI",
      data: SNR_VIALS.map(function(v, j) {{ return {{x: j, y: rawVals[j] != null ? rawVals[j] : null}}; }}),
      borderColor: COLOR_RAW, backgroundColor: COLOR_RAW + "44",
      pointBackgroundColor: COLOR_RAW, pointRadius: 6, pointHoverRadius: 8,
      showLine: false, borderWidth: 0,
    }});
  }}
  var procVals = SNR_PROC[volIdx] || [];
  datasets.push({{
    label: "Preprocessed",
    data: SNR_VIALS.map(function(v, j) {{ return {{x: j, y: procVals[j] != null ? procVals[j] : null}}; }}),
    borderColor: COLOR_PROC, backgroundColor: COLOR_PROC + "44",
    pointBackgroundColor: COLOR_PROC, pointRadius: 6, pointHoverRadius: 8,
    showLine: false, borderWidth: 0,
  }});
  return datasets;
}}

function _makeSnrChart(canvasId) {{
  var el = document.getElementById(canvasId);
  if (!el) return null;
  var opts = baseOpts("Vial", "SNR");
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.callback = function(v) {{ return SNR_VIALS[v] != null ? SNR_VIALS[v] : v; }};
  opts.scales.x.ticks.stepSize = 1;
  opts.scales.y.min = SNR_YMIN;
  opts.scales.y.max = SNR_YMAX;
  opts.plugins.tooltip.callbacks.label = function(ctx) {{ return ctx.dataset.label + "  SNR: " + ctx.parsed.y.toFixed(2); }};
  opts.plugins.legend.display = SNR_RAW !== null;
  opts.plugins.legend.onClick = Chart.defaults.plugins.legend.onClick;
  opts.plugins.legend.labels = {{ color: "#888780", font: {{ size: 12 }}, usePointStyle: true, pointStyle: "circle" }};
  return new Chart(el.getContext("2d"), {{type: "scatter", data: {{datasets: _snrDatasets(0)}}, options: opts}});
}}

var _snrChart = _makeSnrChart("snrChart");

function pkSetSnrVol(el) {{
  var volIdx = el.tagName === "SELECT" ? parseInt(el.value) : parseInt(el.dataset.vol);
  if (_snrChart) {{
    var hidden = _snrChart.data.datasets.map(function(_, i) {{ return _snrChart.getDatasetMeta(i).hidden; }});
    _snrChart.data.datasets = _snrDatasets(volIdx);
    hidden.forEach(function(h, i) {{ if (_snrChart.data.datasets[i]) _snrChart.getDatasetMeta(i).hidden = h; }});
    _snrChart.update("none");
  }}
  document.querySelectorAll(".snr-vol-btn").forEach(function(b) {{ b.classList.remove("active"); }});
  if (el.tagName !== "SELECT") el.classList.add("active");
}}

// ── CNR chart ─────────────────────────────────────────────────────────────
var _cnrVial1 = SNR_VIALS[0] || null;
var _cnrVial2 = SNR_VIALS.length > 1 ? SNR_VIALS[1] : null;
var _cnrVolX  = Array.from({{length: N_VOLS}}, function(_, i) {{ return i; }});

function _getCnrVals(cnrData, v1, v2) {{
  if (!cnrData || !v1 || !v2 || v1 === v2) return null;
  if (cnrData[v1] && cnrData[v1][v2] !== undefined) return cnrData[v1][v2];
  if (cnrData[v2] && cnrData[v2][v1] !== undefined) return cnrData[v2][v1];
  return null;
}}

function _cnrDatasets(v1, v2) {{
  var datasets = [];
  if (CNR_RAW) {{
    var rawVals = _getCnrVals(CNR_RAW, v1, v2);
    if (rawVals) {{
      datasets.push({{
        label: "Raw DWI",
        data: _cnrVolX.map(function(x, i) {{ return {{x: x, y: rawVals[i] != null ? rawVals[i] : null}}; }}),
        borderColor: COLOR_RAW, backgroundColor: COLOR_RAW + "33",
        pointBackgroundColor: COLOR_RAW, pointRadius: 5, pointHoverRadius: 7,
        showLine: false, borderWidth: 0,
      }});
    }}
  }}
  var procVals = _getCnrVals(CNR_PROC, v1, v2);
  if (procVals) {{
    datasets.push({{
      label: "Preprocessed",
      data: _cnrVolX.map(function(x, i) {{ return {{x: x, y: procVals[i] != null ? procVals[i] : null}}; }}),
      borderColor: COLOR_PROC, backgroundColor: COLOR_PROC + "33",
      pointBackgroundColor: COLOR_PROC, pointRadius: 5, pointHoverRadius: 7,
      showLine: false, borderWidth: 0,
    }});
  }}
  return datasets;
}}

function _makeCnrChart(canvasId) {{
  var el = document.getElementById(canvasId);
  if (!el) return null;
  var opts = baseOpts("Volume", "CNR");
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.stepSize = 1;
  opts.plugins.tooltip.callbacks.label = function(ctx) {{ return ctx.dataset.label + "  CNR: " + ctx.parsed.y.toFixed(2); }};
  opts.plugins.legend.display = CNR_RAW !== null;
  opts.plugins.legend.onClick = Chart.defaults.plugins.legend.onClick;
  opts.plugins.legend.labels = {{ color: "#888780", font: {{ size: 12 }}, usePointStyle: true, pointStyle: "circle" }};
  return new Chart(el.getContext("2d"), {{type: "scatter", data: {{datasets: _cnrDatasets(_cnrVial1, _cnrVial2)}}, options: opts}});
}}

var _cnrChart = _makeCnrChart("cnrChart");

function pkSetCnrVial(el) {{
  var which = parseInt(el.dataset.which);
  var vial  = el.dataset.vial;
  if (which === 1) {{
    _cnrVial1 = vial;
    document.querySelectorAll(".cnr-v1-btn").forEach(function(b) {{ b.classList.remove("active"); }});
  }} else {{
    _cnrVial2 = vial;
    document.querySelectorAll(".cnr-v2-btn").forEach(function(b) {{ b.classList.remove("active"); }});
  }}
  el.classList.add("active");
  if (_cnrChart) {{
    var hidden = _cnrChart.data.datasets.map(function(_, i) {{ return _cnrChart.getDatasetMeta(i).hidden; }});
    _cnrChart.data.datasets = _cnrDatasets(_cnrVial1, _cnrVial2);
    hidden.forEach(function(h, i) {{ if (_cnrChart.data.datasets[i]) _cnrChart.getDatasetMeta(i).hidden = h; }});
    _cnrChart.update("none");
  }}
}}

{viewer_js}
</script>
</body>
</html>"""

    Path(output_file).write_text(html, encoding="utf-8")
    return output_file
