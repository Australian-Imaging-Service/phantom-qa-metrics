"""
Shared HTML generation utilities for phantomkit interactive plots.

Provides CDN constants, CSS, JavaScript plugins, and data embedding
utilities used by vial_intensity, maps_ir, and maps_te.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# CDN constants (pinned versions)
# ---------------------------------------------------------------------------

CHARTJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
HAMMERJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"
ZOOM_CDN = (
    "https://cdnjs.cloudflare.com/ajax/libs/"
    "chartjs-plugin-zoom/1.2.1/chartjs-plugin-zoom.min.js"
)

_NIIVUE_CDN = (
    "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.46.0/dist/niivue.umd.min.js"
)
_NIIVUE_BUNDLE = Path(__file__).parent / "niivue.umd.min.js"


def _niivue_script_tag() -> str:
    """Return a <script> tag that loads NiiVue — inlined bundle or CDN fallback."""
    if _NIIVUE_BUNDLE.exists():
        src = _NIIVUE_BUNDLE.read_text(encoding="utf-8")
        src = re.sub(r"(?i)</script>", r"<\/script>", src)
        return f"<script>/* niivue@0.46.0 inlined */\n{src}\n</script>"
    return f'<script src="{_NIIVUE_CDN}"></script>'


def nifti_to_base64(path: str) -> str:
    """Base64-encode a NIfTI file for embedding in an HTML page.

    If the file stores float64 voxels it is re-packed as float32 in-memory
    before encoding — halving the payload with no perceptible visual change.
    The result is always a valid ``.nii.gz`` byte stream.
    """
    import gzip
    import io
    import struct

    import numpy as np  # type: ignore[import]

    raw = Path(path).read_bytes()

    # Decompress if gzipped so we can inspect/rewrite the header
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw  # already uncompressed .nii

    # NIfTI-1 datatype field is a little-endian int16 at byte offset 70
    (dtype_code,) = struct.unpack_from("<h", data, 70)

    if dtype_code == 64:  # float64 → repack as float32 (code 16)
        # Read vox_offset (float32 at offset 108) to find where voxel data starts
        (vox_offset,) = struct.unpack_from("<f", data, 108)
        vox_start = int(vox_offset) if vox_offset >= 352 else 352

        hdr = bytearray(data[:vox_start])
        # Patch datatype (offset 70) and bitpix (offset 72) for float32
        struct.pack_into("<h", hdr, 70, 16)  # datatype = float32
        struct.pack_into("<h", hdr, 72, 32)  # bitpix   = 32

        voxels = np.frombuffer(data[vox_start:], dtype="<f8").astype("<f4")
        repack = bytes(hdr) + voxels.tobytes()

        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            gz.write(repack)
        raw = buf.getvalue()

    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Color palette (Chart.js datasets)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#378ADD",
    "#D85A30",
    "#1D9E75",
    "#D4537E",
    "#7F77DD",
    "#BA7517",
    "#639922",
    "#888780",
    "#185FA5",
    "#993C1D",
]


def chart_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #ffffff; --bg2: #f4f4f2; --bg3: #e9e9e6;
    --text: #1a1a18; --text2: #5f5e5a; --border: rgba(0,0,0,0.12);
    --radius: 8px; --font: system-ui, -apple-system, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1c1c1a; --bg2: #252523; --bg3: #2e2e2b;
      --text: #e8e8e4; --text2: #888780; --border: rgba(255,255,255,0.10);
    }
  }
  body { font-family: var(--font); background: var(--bg); color: var(--text); padding: 24px; }
  h1   { font-size: 18px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: var(--text2); margin-bottom: 20px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 20px; }
  label { font-size: 12px; color: var(--text2); font-weight: 500; }
  .panel-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) { .panel-row { grid-template-columns: 1fr; } }
  .chart-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-card {
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: 12px; padding: 16px;
  }
  .chart-title { font-size: 13px; font-weight: 500; margin-bottom: 8px; }
  .chart-wrap  { position: relative; width: 100%; height: 280px; }
  /* Legend */
  .legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .legend-item {
    display: flex; align-items: center; gap: 6px; font-size: 12px;
    color: var(--text2); padding: 4px 10px; border-radius: var(--radius);
    background: var(--bg2); border: 0.5px solid var(--border);
  }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  /* Stats table */
  .stats-section { margin-top: 28px; }
  .stats-title  { font-size: 13px; font-weight: 500; color: var(--text2); margin-bottom: 10px; }
  .stats-table  { width: 100%; border-collapse: collapse; font-size: 12px; }
  .stats-table th {
    text-align: left; padding: 6px 10px; background: var(--bg3);
    border-bottom: 0.5px solid var(--border); color: var(--text2); font-weight: 500;
  }
  .stats-table td { padding: 5px 10px; border-bottom: 0.5px solid var(--border); }
  .stats-table tr:last-child td { border-bottom: none; }
  /* Measure / error-bar toggle controls */
  .pk-controls {
    display: flex; gap: 14px; flex-wrap: wrap; margin: 0 0 14px;
    align-items: center; padding: 8px 12px;
    background: var(--bg2); border-radius: 8px; border: 0.5px solid var(--border);
  }
  .pk-ctrl-group { display: flex; align-items: center; gap: 4px; }
  .pk-ctrl-label {
    font-size: 11px; color: var(--text2); margin-right: 6px;
    text-transform: uppercase; letter-spacing: 0.05em;
  }
  .pk-btn {
    padding: 3px 10px; font-size: 12px; border: 1px solid var(--border);
    border-radius: 4px; background: transparent; color: var(--text2);
    cursor: pointer; transition: background 0.1s, color 0.1s, border-color 0.1s; line-height: 1.5;
  }
  .pk-btn.active { background: #378ADD; color: #fff; border-color: #378ADD; }
  .pk-btn:hover:not(.active) { border-color: #888; color: var(--text); }
"""

# ---------------------------------------------------------------------------
# Shared JavaScript — error bar plugin and base chart options
# ---------------------------------------------------------------------------

ERROR_BAR_PLUGIN_JS = """
const errorBarPlugin = {
  id: "errorBar",
  afterDatasetsDraw(chart) {
    if (window._pkShowErrBars === false) return;
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ctx.save(); ctx.strokeStyle = ds.borderColor; ctx.lineWidth = 1.2;
      Object.entries(ds.errorBars).forEach(([i, eb]) => {
        const el = meta.data[+i]; if (!el) return;
        const x = el.x, cap = 4;
        const yMin = chart.scales.y.getPixelForValue(eb.yMin);
        const yMax = chart.scales.y.getPixelForValue(eb.yMax);
        if (!isFinite(yMin) || !isFinite(yMax)) return;
        ctx.beginPath();
        ctx.moveTo(x, yMin); ctx.lineTo(x, yMax);
        ctx.moveTo(x-cap, yMin); ctx.lineTo(x+cap, yMin);
        ctx.moveTo(x-cap, yMax); ctx.lineTo(x+cap, yMax);
        ctx.stroke();
      });
      ctx.restore();
    });
  }
};
"""


PK_CONTROLS_HTML = """\
<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Measure</span>
    <button class="pk-btn pk-measure-btn active" onclick="pkSetMeasure('mean',this)">Mean</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('median',this)">Median</button>
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

PK_CONTROLS_HTML_WITH_LEGEND = """\
<div class="pk-controls">
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Measure</span>
    <button class="pk-btn pk-measure-btn active" onclick="pkSetMeasure('mean',this)">Mean</button>
    <button class="pk-btn pk-measure-btn" onclick="pkSetMeasure('median',this)">Median</button>
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
  <div class="pk-ctrl-group">
    <span class="pk-ctrl-label">Legend</span>
    <button class="pk-btn" id="pk-legend-btn" onclick="pkToggleLegend(this)">Show</button>
  </div>
</div>"""

PK_TOGGLE_JS = """
window._pkMeasure = "mean";
window._pkErrMode = "sd";
window._pkShowLegend = false;
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

function pkToggleLegend(btn) {
  window._pkShowLegend = !window._pkShowLegend;
  btn.textContent = window._pkShowLegend ? "Hide" : "Show";
  btn.classList.toggle("active", window._pkShowLegend);
  _pkCharts.forEach(ch => {
    ch.options.plugins.legend.display = window._pkShowLegend;
    ch.update("none");
  });
}

function pkUpdateAllCharts() {
  const m = window._pkMeasure, e = window._pkErrMode;
  _pkCharts.forEach(ch => {
    ch.data.datasets.forEach((ds, di) => {
      if (ds._row !== undefined) {
        const vals = PK_DATA.measure[m][ds._row];
        ds.data.forEach((pt, k) => { pt.y = vals[k]; });
        const eb = (e !== "none" && PK_DATA.errBounds[m] && PK_DATA.errBounds[m][e]);
        if (eb) {
          const lo = PK_DATA.errBounds[m][e].lower[ds._row];
          const hi = PK_DATA.errBounds[m][e].upper[ds._row];
          ds.errorBars = {};
          lo.forEach((l, k) => { ds.errorBars[String(k)] = { yMin: l, yMax: hi[k] }; });
        } else {
          ds.errorBars = {};
        }
      }
      if (ds._fit_measure !== undefined) {
        ch.setDatasetVisibility(di, ds._fit_measure === m);
      }
      if (ds._refMeas && typeof _REF_MEAS !== "undefined") {
        const pts = _REF_MEAS[m];
        ds.data.length = 0;
        pts.forEach(p => ds.data.push(p));
      }
    });
    ch.update("none");
  });
  // Toggle fit results tables (relaxometry plots only)
  document.querySelectorAll(".pk-fit-table").forEach(el => {
    el.style.display = (el.dataset.measure === m) ? "" : "none";
  });
  if (typeof window._pkAfterUpdate === "function") window._pkAfterUpdate();
}
"""


def _compute_pk_err_bounds(
    mean_m, median_m, std_m, count_m, p25_m, p75_m, min_m, max_m,
    mean_mad_m=None, median_mad_m=None,
) -> dict:
    """Compute all error bound variants for embedding in HTML.

    All inputs are numpy arrays of the same shape (n_rows, n_cols).
    Returns a dict keyed by measure → variant → {lower, upper} lists.
    """
    import numpy as np

    se_m = std_m / np.sqrt(np.maximum(count_m, 1.0))
    _mean_mad  = mean_mad_m   if mean_mad_m   is not None else std_m
    _median_mad = median_mad_m if median_mad_m is not None else std_m

    def _b(lo, hi):
        return {"lower": lo.tolist(), "upper": hi.tolist()}

    result = {}
    for key, vals, mad in (
        ("mean",   mean_m,   _mean_mad),
        ("median", median_m, _median_mad),
    ):
        result[key] = {
            "sd":     _b(vals - std_m,     vals + std_m),
            "se":     _b(vals - se_m,      vals + se_m),
            "2se":    _b(vals - 2 * se_m,  vals + 2 * se_m),
            "mad":    _b(vals - mad,        vals + mad),
            "iqr":    _b(p25_m,            p75_m),
            "minmax": _b(min_m,            max_m),
        }
    return result


def base_opts_js(
    x_label: str = "x", y_label: str = "Intensity", enable_zoom: bool = True
) -> str:
    """Return a JS baseOpts() function for Chart.js.

    Parameters
    ----------
    enable_zoom : bool
        When True (default) zoom/pan via the chartjs-plugin-zoom is active.
        Pass False to disable zoom and pan entirely (e.g. for ADC, FA, vial
        intensity charts where scroll-to-zoom is undesirable).
    """
    if enable_zoom:
        zoom_plugin = """\
      zoom: {
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
        pan:  { enabled: true, mode: "xy" },
      }"""
    else:
        zoom_plugin = """\
      zoom: {
        zoom: { wheel: { enabled: false }, pinch: { enabled: false } },
        pan:  { enabled: false },
      }"""
    return f"""
function baseOpts(xLabel, yLabel) {{
  xLabel = xLabel || {json.dumps(x_label)};
  yLabel = yLabel || {json.dumps(y_label)};
  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const grid = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
  const tick = "#888780";
  return {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{
        display: false,
        onClick: null,
        labels: {{
          filter: (item, data) => data.datasets[item.datasetIndex]._row !== undefined,
          color: "#888780", font: {{ size: 11 }}
        }}
      }},
      tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}` }} }},
{zoom_plugin}
    }},
    scales: {{
      x: {{ type: "linear",
           title: {{ display: true, text: xLabel, color: tick, font: {{ size: 13 }} }},
           ticks: {{ color: tick, font: {{ size: 13 }} }}, grid: {{ color: grid }} }},
      y: {{ title: {{ display: true, text: yLabel, color: tick, font: {{ size: 13 }} }},
           ticks: {{ color: tick, font: {{ size: 13 }} }}, grid: {{ color: grid }} }}
    }}
  }};
}}
"""


# ---------------------------------------------------------------------------
# Embedded data tag
# ---------------------------------------------------------------------------


def phantomkit_data_tag(data: dict) -> str:
    """Return a <script> tag embedding data as JSON for downstream parsing."""
    return (
        '<script id="phantomkit-data" type="application/json">\n'
        + json.dumps(data, indent=2)
        + "\n</script>"
    )


# ---------------------------------------------------------------------------
# NiiVue viewer panel (base64-embedded NIfTI — no server required)
# ---------------------------------------------------------------------------


def _niivue_viewer_panel(
    nifti_image: str,
    vial_niftis: dict[str, str],
    bg_cal_min: float | None = None,
    bg_cal_max: float | None = None,
    overlay_nifti: str | None = None,
    overlay_label: str = "Contrast",
) -> tuple[str, str]:
    """Build an embedded NiiVue viewer panel for a static HTML page.

    NIfTI files are base64-encoded and converted to blob URLs at runtime so
    the HTML is fully self-contained with no HTTP server required.

    Parameters
    ----------
    nifti_image:
        Path to the background NIfTI file.
    vial_niftis:
        ``{vial_name: path}`` mapping for ROI overlay NIfTI files.  Only
        files that exist on disk are included.
    bg_cal_min, bg_cal_max:
        Optional intensity window for the background volume (e.g. 0/0.004 for ADC).
    overlay_nifti:
        Optional path to a second image (e.g. first IR/TE contrast) shown as a
        toggleable "hot" overlay on top of the background.
    overlay_label:
        Button label for the overlay toggle (e.g. "IR contrast").

    Returns
    -------
    html_panel : str
        The ``<div>`` element containing the canvas and vial toggle chips.
    js_block : str
        JavaScript that initialises NiiVue and defines ``pkToggleVial``.
    """
    bg_b64 = nifti_to_base64(nifti_image)

    vials_data = [
        {"name": name, "b64": nifti_to_base64(vpath)}
        for name, vpath in sorted(vial_niftis.items())
        if vpath and Path(vpath).exists()
    ]

    _has_overlay = bool(overlay_nifti and Path(overlay_nifti).exists())
    overlay_b64 = nifti_to_base64(overlay_nifti) if _has_overlay else None

    chips = "".join(
        f'<button id="pk-chip-{i}" data-active="1" onclick="pkToggleVial({i + 1},this)"'
        f' style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        f"background:var(--bg3);color:var(--text);font-size:11px;font-weight:500;"
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
    chips_section = (
        (
            '<p style="font-size:11px;color:var(--text2);margin-top:10px;margin-bottom:6px;">'
            "Toggle vial ROIs</p>"
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">'
            f'{_toggle_all_btn}{chips}</div>'
        )
        if vials_data
        else ""
    )

    overlay_btn_html = (
        f'<button id="pk-overlay-btn" data-active="0" onclick="pkToggleOverlay(this)"'
        f' style="padding:3px 10px;border-radius:99px;border:1.5px solid var(--border);'
        f"background:var(--bg3);color:var(--text2);font-size:11px;font-weight:500;"
        f'cursor:pointer;user-select:none;transition:opacity .15s;opacity:0.55;">'
        f'{overlay_label}</button>'
        if _has_overlay
        else ""
    )

    _zoom_btn_style = (
        "padding:3px 8px;border-radius:99px;border:1.5px solid var(--border);"
        "background:var(--bg3);color:var(--text2);font-size:13px;font-weight:600;"
        "cursor:pointer;user-select:none;transition:opacity .15s;line-height:1;"
    )
    html_panel = f"""<div class="chart-card" style="margin-bottom:20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <p class="chart-title" style="margin:0;">Image Viewer</p>
    <div style="display:flex;gap:6px;align-items:center;">
      {overlay_btn_html}
      <button onclick="pkZoom(1.25)" style="{_zoom_btn_style}">+</button>
      <button onclick="pkZoom(1/1.25)" style="{_zoom_btn_style}">&minus;</button>
      <button id="pk-rad-btn" data-rad="1" onclick="pkToggleRadConvention(this)"
        style="padding:3px 10px;border-radius:99px;border:1.5px solid var(--border);
        background:var(--bg3);color:var(--text2);font-size:11px;font-weight:500;
        cursor:pointer;user-select:none;transition:opacity .15s;">Radiological</button>
    </div>
  </div>
  <canvas id="nv-canvas" style="width:100%;height:300px;display:block;background:#000;border-radius:6px;cursor:crosshair;"></canvas>
  <p style="font-size:11px;color:var(--text2);margin-top:8px;">Scroll: change slice &middot; Ctrl+scroll or +/&minus;: zoom &middot; drag: adjust contrast &middot; axial / coronal / sagittal</p>
  {chips_section}
</div>"""

    bg_b64_js = json.dumps(bg_b64)
    vials_js = json.dumps(vials_data)
    overlay_b64_js = json.dumps(overlay_b64)
    cal_min_js = "null" if bg_cal_min is None else str(float(bg_cal_min))
    cal_max_js = "null" if bg_cal_max is None else str(float(bg_cal_max))

    js_block = f"""
var _NV_BG = {bg_b64_js};
var _NV_VIALS = {vials_js};
var _NV_OVERLAY = {overlay_b64_js};
var _NV_CAL_MIN = {cal_min_js};
var _NV_CAL_MAX = {cal_max_js};

(function() {{
  var canvas = document.getElementById("nv-canvas");
  if (!canvas) return;
  var ro = new ResizeObserver(function(entries) {{
    for (var i = 0; i < entries.length; i++) {{
      var r = entries[i].contentRect;
      var w = Math.round(r.width), h = Math.round(r.height);
      if (w > 0 && h > 0) {{ ro.disconnect(); _pkInitNv(canvas, w, h); break; }}
    }}
  }});
  ro.observe(canvas);
}})();

function _pkB64ToUrl(b64) {{
  var bin = atob(b64), arr = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([arr], {{type:"application/octet-stream"}}));
}}

function _pkInitNv(canvas, w, h) {{
  canvas.width = w; canvas.height = h;
  // Ctrl+scroll → zoom via scene.pan2Dxyzmm[3] (the 2-D zoom scale in NiiVue).
  // Plain scroll is left for NiiVue's own wheelListener (slice navigation).
  canvas.addEventListener("wheel", function(e) {{
    e.preventDefault();
    if ((e.ctrlKey || e.metaKey) && window._pkNv && window._pkNv.scene) {{
      e.stopPropagation();
      var factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      window._pkNv.scene.pan2Dxyzmm[3] = Math.max(0.1, window._pkNv.scene.pan2Dxyzmm[3] * factor);
      window._pkNv.drawScene();
    }}
  }}, {{ passive: false }});
  window._pkNv = new niivue.Niivue({{
    isColorbar: false, crosshairWidth: 1, isResizeCanvas: false,
    isAntiAlias: false,
    multiplanarLayout: 3,
    multiplanarShowRender: 0,
  }});
  window._pkNv.attachToCanvas(canvas);
  window._pkNv.opts.sliceType = 3;
  var vols = [{{url: _pkB64ToUrl(_NV_BG), name: "background.nii.gz", colormap: "gray"}}];
  for (var i = 0; i < _NV_VIALS.length; i++) {{
    vols.push({{
      url: _pkB64ToUrl(_NV_VIALS[i].b64),
      name: _NV_VIALS[i].name + ".nii.gz",
      colormap: "red",
      opacity: 0.5,
    }});
  }}
  if (_NV_OVERLAY !== null) {{
    vols.push({{
      url: _pkB64ToUrl(_NV_OVERLAY),
      name: "contrast_overlay.nii.gz",
      colormap: "winter",
      opacity: 0.0,
    }});
  }}
  window._pkNv.loadVolumes(vols).then(function() {{
    window._pkNv.setRadiologicalConvention(true);
    if (_NV_CAL_MIN !== null && _NV_CAL_MAX !== null) {{
      if (typeof window._pkNv.setCalMinMax === "function") {{
        window._pkNv.setCalMinMax(0, _NV_CAL_MIN, _NV_CAL_MAX);
      }} else if (window._pkNv.volumes && window._pkNv.volumes[0]) {{
        window._pkNv.volumes[0].cal_min = _NV_CAL_MIN;
        window._pkNv.volumes[0].cal_max = _NV_CAL_MAX;
        window._pkNv.updateGLVolume();
      }}
    }}
  }});
}}

function pkToggleVial(idx, btn) {{
  if (!window._pkNv) return;
  var wasActive = btn.dataset.active === "1";
  btn.dataset.active = wasActive ? "0" : "1";
  btn.style.opacity = wasActive ? "0.35" : "1.0";
  window._pkNv.setOpacity(idx, wasActive ? 0.0 : 0.5);
}}

function pkToggleOverlay(btn) {{
  if (!window._pkNv || _NV_OVERLAY === null) return;
  var wasActive = btn.dataset.active === "1";
  btn.dataset.active = wasActive ? "0" : "1";
  btn.style.opacity = wasActive ? "0.55" : "1.0";
  var overlayIdx = _NV_VIALS.length + 1;
  window._pkNv.setOpacity(overlayIdx, wasActive ? 0.0 : 0.6);
}}

function pkZoom(factor) {{
  if (!window._pkNv || !window._pkNv.scene) return;
  window._pkNv.scene.pan2Dxyzmm[3] = Math.max(0.1, window._pkNv.scene.pan2Dxyzmm[3] * factor);
  window._pkNv.drawScene();
}}

function pkToggleRadConvention(btn) {{
  if (!window._pkNv) return;
  var isRad = btn.dataset.rad === "1";
  btn.dataset.rad = isRad ? "0" : "1";
  btn.textContent = isRad ? "Neurological" : "Radiological";
  window._pkNv.setRadiologicalConvention(!isRad);
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
    return html_panel, js_block


# ---------------------------------------------------------------------------
# Relaxometry (IR / TE) 3×3 HTML builder
# ---------------------------------------------------------------------------


def build_relaxometry_html(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    fit_label: str,
    fit_key: str,
    vial_groups: list,
    contrast_numbers,
    mean_matrix,
    std_matrix,
    vial_labels: list,
    subplot_fits: list,
    embedded_data: dict,
    subplot_fits_median: list | None = None,
    fit_results_median: list | None = None,
    median_matrix=None,
    count_matrix=None,
    p25_matrix=None,
    p75_matrix=None,
    min_matrix=None,
    max_matrix=None,
    mean_mad_matrix=None,
    median_mad_matrix=None,
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
    ref_data: dict | None = None,
    overlay_contrast: str | None = None,
) -> str:
    """Build the interactive HTML for a relaxometry (IR or TE) plot.

    Parameters
    ----------
    title : str
        Page/plot title (e.g. "T1 Inversion Recovery Mapping").
    subtitle : str
        Short description line.
    x_label : str
        X-axis label (e.g. "Inversion Time (ms)").
    fit_label : str
        Legend label for fit curve (e.g. "T₁ fit").
    fit_key : str
        Key in fit_results for the relaxation time (e.g. "T1_ms").
    vial_groups : list of list of str
        Groups of vial names for the subplot grid.
    contrast_numbers : array-like
        Sorted numeric values (TI or TE) used as x-axis.
    mean_matrix : ndarray
        Shape (n_vials, n_contrasts) — mean intensities, rows=vials, cols=contrasts.
    std_matrix : ndarray
        Shape (n_vials, n_contrasts) — std deviations, same layout.
    vial_labels : list of str
        Ordered list of vial names matching row order in mean_matrix.
    subplot_fits : list of dict
        One entry per vial group, each with pre-computed fit data:
        {"vial": str, "color": str, "x_fit": list, "y_fit": list,
         "ci_upper": list or None, "ci_lower": list or None}
    embedded_data : dict
        Data to embed as machine-readable JSON in the HTML.
    overlay_contrast : str, optional
        Path to a NIfTI file (e.g. first IR or TE image) to offer as a
        toggleable "hot" overlay in the MRI viewer.
    """
    import numpy as np

    vial_to_idx = {label: i for i, label in enumerate(vial_labels)}
    # Convert to plain Python float so json.dumps handles them without error
    # (contrast_numbers may be a numpy array of int64 or float64)
    x_arr = [float(x) for x in contrast_numbers]

    # Fall back to mean-based proxies when optional matrices are absent
    _median     = median_matrix     if median_matrix     is not None else mean_matrix
    _count      = count_matrix      if count_matrix      is not None else np.ones_like(mean_matrix) * 1000
    _p25        = p25_matrix        if p25_matrix        is not None else mean_matrix - std_matrix
    _p75        = p75_matrix        if p75_matrix        is not None else mean_matrix + std_matrix
    _min        = min_matrix        if min_matrix        is not None else mean_matrix - std_matrix
    _max        = max_matrix        if max_matrix        is not None else mean_matrix + std_matrix
    _mean_mad   = mean_mad_matrix   if mean_mad_matrix   is not None else None
    _median_mad = median_mad_matrix if median_mad_matrix is not None else None

    # Pre-compute all error bound variants for the toggle controls
    pk_err_bounds = _compute_pk_err_bounds(
        mean_matrix, _median, std_matrix, _count, _p25, _p75, _min, _max,
        mean_mad_m=_mean_mad, median_mad_m=_median_mad,
    )
    pk_data_json = json.dumps({
        "measure": {
            "mean":   mean_matrix.tolist(),
            "median": _median.tolist(),
        },
        "errBounds": pk_err_bounds,
    })

    # Helper: build CI band + fit line datasets tagged with a measure
    def _fit_datasets_for(fit_entry_, color_, measure_):
        dsets = []
        if not fit_entry_ or not fit_entry_.get("x_fit"):
            return dsets
        xf_ = fit_entry_["x_fit"]
        yf_ = fit_entry_["y_fit"]
        ci_lo_ = fit_entry_.get("ci_lower")
        ci_hi_ = fit_entry_.get("ci_upper")
        if ci_hi_ and ci_lo_:
            dsets.append({
                "label": "_ci_upper", "_fit_measure": measure_,
                "data": [{"x": xf_[k], "y": ci_hi_[k]} for k in range(len(xf_))],
                "borderColor": "transparent", "backgroundColor": color_ + "22",
                "pointRadius": 0, "borderWidth": 0, "showLine": True,
                "tension": 0.3, "fill": "+1",
            })
            dsets.append({
                "label": "_ci_lower", "_fit_measure": measure_,
                "data": [{"x": xf_[k], "y": ci_lo_[k]} for k in range(len(xf_))],
                "borderColor": "transparent", "backgroundColor": "transparent",
                "pointRadius": 0, "borderWidth": 0, "showLine": True,
                "tension": 0.3, "fill": False,
            })
        dsets.append({
            "label": fit_label, "_fit_measure": measure_,
            "data": [{"x": xf_[k], "y": yf_[k]} for k in range(len(xf_))],
            "borderColor": color_ + "bb", "backgroundColor": "transparent",
            "pointRadius": 0, "borderWidth": 1.5, "borderDash": [4, 3],
            "showLine": True, "tension": 0.3, "fill": False,
        })
        return dsets

    _have_median_fits = subplot_fits_median is not None

    # Build per-subplot Chart.js dataset lists
    subplot_datasets = []
    for g_idx, group in enumerate(vial_groups):
        group_datasets = []
        group_fits_this = [f for f in subplot_fits if f["vial"] in group]
        group_fits_med  = [f for f in (subplot_fits_median or []) if f["vial"] in group]

        for j, vial in enumerate(group):
            if vial not in vial_to_idx:
                continue
            row = vial_to_idx[vial]
            color = chart_color(j if len(group) > 1 else 8)  # black-ish for single

            means = mean_matrix[row, :].tolist()
            stds = std_matrix[row, :].tolist()

            scatter_ds = {
                "label": f"Vial {vial}",
                "_row": row,
                "data": [{"x": x_arr[k], "y": means[k]} for k in range(len(x_arr))],
                "borderColor": color,
                "backgroundColor": color + "33",
                "pointBackgroundColor": color,
                "pointRadius": 5,
                "pointHoverRadius": 7,
                "borderWidth": 0,
                "showLine": False,
                "errorBars": {
                    str(k): {"yMin": means[k] - stds[k], "yMax": means[k] + stds[k]}
                    for k in range(len(x_arr))
                },
            }
            group_datasets.append(scatter_ds)

            # Mean fit datasets (always present; tagged so toggle can hide them)
            fit_entry = next((f for f in group_fits_this if f["vial"] == vial), None)
            group_datasets.extend(_fit_datasets_for(
                fit_entry, color, "mean" if _have_median_fits else None
            ))

            # Median fit datasets (only when median fits were computed)
            if _have_median_fits:
                med_fit_entry = next((f for f in group_fits_med if f["vial"] == vial), None)
                group_datasets.extend(_fit_datasets_for(med_fit_entry, color, "median"))

        subplot_datasets.append(group_datasets)

    datasets_json = json.dumps(subplot_datasets)
    x_label_json = json.dumps(x_label)
    data_tag = phantomkit_data_tag(embedded_data)
    opts_js = base_opts_js(x_label=x_label, y_label="Intensity", enable_zoom=True)

    # ------------------------------------------------------------------
    # Standardised axis bounds — shared across every subplot on the page
    # ------------------------------------------------------------------
    x_min_plot = 0.0
    x_max_plot = float(np.max(contrast_numbers)) * 1.05  # 5 % right padding

    y_lower = float(np.minimum(_p25, mean_matrix - std_matrix).min())
    y_upper = float(np.maximum(_p75, mean_matrix + std_matrix).max())
    for _fit in subplot_fits:
        for _key in ("y_fit", "ci_lower", "ci_upper"):
            if _fit.get(_key):
                y_lower = min(y_lower, min(_fit[_key]))
                y_upper = max(y_upper, max(_fit[_key]))
    _y_pad = (y_upper - y_lower) * 0.12
    y_min_raw = y_lower - _y_pad
    y_max_raw = y_upper + _y_pad

    # Round to a sensible grid unit based on the data range so axis tick
    # labels are clean integers (e.g. -200, 0, 1000, 2000 … not -187.3).
    _range = y_max_raw - y_min_raw
    _unit = 10 ** int(np.floor(np.log10(_range / 5))) if _range > 0 else 1
    y_min_plot = int(np.floor(y_min_raw / _unit)) * _unit
    y_max_plot = int(np.ceil(y_max_raw / _unit)) * _unit

    axis_consts_js = (
        f"const X_MIN = {x_min_plot:.1f};\n"
        f"const X_MAX = {x_max_plot:.1f};\n"
        f"const Y_MIN = {y_min_plot};\n"
        f"const Y_MAX = {y_max_plot};\n"
    )

    # ------------------------------------------------------------------
    # Fit-results table(s) (T1 / T2 per vial)
    # ------------------------------------------------------------------
    fit_results = embedded_data.get("fit_results", [])
    fit_unit_label = "T\u2081 (ms)" if fit_key == "T1_ms" else "T\u2082 (ms)"

    def _fit_table_body(results):
        rows = ""
        for r in sorted(results, key=lambda r: r.get(fit_key) or 0, reverse=True):
            vial = r.get("Vial", "")
            t_val = r.get(fit_key, "")
            s0_val = r.get("S0", "")
            r2_val = r.get("R2", "")
            t_str = f"{t_val:.1f}" if isinstance(t_val, (int, float)) else str(t_val)
            s0_str = f"{s0_val:.1f}" if isinstance(s0_val, (int, float)) else str(s0_val)
            r2_str = f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)
            rows += f"    <tr><td>{vial}</td><td>{t_str}</td><td>{s0_str}</td><td>{r2_str}</td></tr>\n"
        return rows

    if fit_results:
        _mean_rows = _fit_table_body(fit_results)
        if fit_results_median:
            _med_rows = _fit_table_body(fit_results_median)
            fits_table_html = f"""<div class="stats-section">
  <div class="stats-title">Relaxometry Fit Results</div>
  <div class="pk-fit-table" data-measure="mean">
  <table class="stats-table">
    <thead><tr><th>Vial</th><th>{fit_unit_label} (mean fit)</th><th>S0</th><th>R\u00b2</th></tr></thead>
    <tbody>
{_mean_rows}    </tbody>
  </table>
  </div>
  <div class="pk-fit-table" data-measure="median" style="display:none">
  <table class="stats-table">
    <thead><tr><th>Vial</th><th>{fit_unit_label} (median fit)</th><th>S0</th><th>R\u00b2</th></tr></thead>
    <tbody>
{_med_rows}    </tbody>
  </table>
  </div>
</div>"""
        else:
            fits_table_html = f"""<div class="stats-section">
  <div class="stats-title">Relaxometry Fit Results</div>
  <table class="stats-table">
    <thead>
      <tr><th>Vial</th><th>{fit_unit_label}</th><th>S0</th><th>R\u00b2</th></tr>
    </thead>
    <tbody>
{_mean_rows}    </tbody>
  </table>
</div>"""
    else:
        fits_table_html = ""

    # ------------------------------------------------------------------
    # Optional measured-vs-reference relaxometry chart
    # ------------------------------------------------------------------
    ref_chart_html = ""
    ref_chart_js = ""
    if ref_data is not None and fit_results:
        _ref_vials = ref_data.get("vials", [])

        # Support both old format {T1_ms/T2_ms: {vial: val}} and new multi-temp format
        if "values_by_temp" in ref_data:
            _default_temp = ref_data.get("default_temp")
            _temperatures = ref_data.get("temperatures", [])
            _vals_by_temp = ref_data.get("values_by_temp", {})
            _ref_vals_at_default = _vals_by_temp.get(_default_temp, {}) if _default_temp else {}
        else:
            # Legacy single-temp format (no values_by_temp key)
            _old_vals = ref_data.get(fit_key, {})
            _ref_vals_at_default = {v.upper(): float(val) for v, val in _old_vals.items() if val is not None}
            _default_temp = None
            _temperatures = []
            _vals_by_temp = {}

        def _build_meas_pts(results):
            mmap = {
                r.get("Vial", "").upper(): r.get(fit_key)
                for r in (results or [])
                if r.get(fit_key) is not None
            }
            return [
                {"x": _j, "y": round(float(mmap[_v.upper()]), 1)}
                for _j, _v in enumerate(_ref_vials)
                if mmap.get(_v.upper()) is not None
            ]

        _meas_pts = _build_meas_pts(fit_results)
        _meas_pts_median = _build_meas_pts(fit_results_median or fit_results)
        _ref_pts = [
            {"x": _j, "y": float(_ref_vals_at_default[_v.upper()])}
            for _j, _v in enumerate(_ref_vials)
            if _ref_vals_at_default.get(_v.upper()) is not None
        ]

        _ref_by_temp: dict[str, list] = {}
        for _temp_str, _vals_dict in _vals_by_temp.items():
            _ref_by_temp[_temp_str] = [
                {"x": _j, "y": float(_vals_dict[_v.upper()])}
                for _j, _v in enumerate(_ref_vials)
                if _vals_dict.get(_v.upper()) is not None
            ]

        _unit_label = "T\u2081 (ms)" if fit_key == "T1_ms" else "T\u2082 (ms)"
        _meas_color = "#378ADD"
        _ref_color = "#C62828"
        _meas_ds = {
            "label": f"Measured {_unit_label}",
            "_refMeas": True,
            "data": _meas_pts,
            "borderColor": _meas_color,
            "backgroundColor": _meas_color,
            "pointBackgroundColor": _meas_color,
            "pointRadius": 6,
            "pointHoverRadius": 8,
            "showLine": False,
            "borderWidth": 0,
        }
        _ref_ds = {
            "label": f"Reference {_unit_label}",
            "data": _ref_pts,
            "borderColor": "transparent",
            "backgroundColor": "transparent",
            "pointBackgroundColor": "transparent",
            "pointBorderColor": _ref_color,
            "pointBorderWidth": 2,
            "pointRadius": 8,
            "pointHoverRadius": 10,
            "showLine": False,
            "borderWidth": 0,
        }
        _vial_labels_json = json.dumps(_ref_vials)
        _meas_ds_json = json.dumps(_meas_ds)
        _ref_ds_json = json.dumps(_ref_ds)
        _ref_meas_json = json.dumps({"mean": _meas_pts, "median": _meas_pts_median})
        _unit_label_json = json.dumps(_unit_label)
        _ref_by_temp_json = json.dumps(_ref_by_temp)
        _default_temp_json = json.dumps(_default_temp)

        _ref_units = ref_data.get("units", "ms")
        if _temperatures:
            _temp_options_parts = []
            for _t in _temperatures:
                _sel = " selected" if str(_t) == _default_temp else ""
                _temp_options_parts.append(f'<option value="{_t}"{_sel}>{_t} °C</option>')
            _temp_options = "".join(_temp_options_parts)
            _temp_selector_html = (
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                + '<span style="font-size:13px;color:var(--text2);">Reference temperature:</span>'
                + '<select id="refChartTempSelect" onchange="pkSetRefChartTemp(this.value)"'
                + ' style="background:var(--bg3);color:var(--text);border:1px solid var(--border);'
                + 'border-radius:6px;padding:4px 10px;font-size:13px;cursor:pointer;">'
                + _temp_options
                + '</select>'
                + '</div>'
            )
        else:
            _temp_selector_html = ""

        ref_chart_html = f"""<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-title">Measured vs Reference {_unit_label}</div>
  {_temp_selector_html}
  <div class="chart-wrap" style="height:260px"><canvas id="refChart"></canvas></div>
</div>"""

        ref_chart_js = f"""
const _REF_VIALS = {_vial_labels_json};
const _REF_MEAS = {_ref_meas_json};
const _REF_BY_TEMP = {_ref_by_temp_json};
let _pkRefChartTemp = {_default_temp_json};
let refChart;
(function() {{
  const opts = baseOpts("Vial", {_unit_label_json});
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.stepSize = 1;
  opts.scales.x.ticks.callback = v => _REF_VIALS[v] ?? v;
  opts.plugins.legend.display = true;
  opts.plugins.legend.onClick = null;
  opts.plugins.legend.labels = {{ color: "#888780", font: {{ size: 12 }} }};
  refChart = new Chart(
    document.getElementById("refChart").getContext("2d"),
    {{ type: "line", data: {{ datasets: [{_meas_ds_json}, {_ref_ds_json}] }}, options: opts }}
  );
  _pkCharts.push(refChart);
}})();
function pkSetRefChartTemp(temp) {{
  _pkRefChartTemp = temp;
  const refDs = refChart.data.datasets.find(d => d.label && d.label.startsWith("Reference"));
  if (!refDs || !_REF_BY_TEMP) return;
  refDs.data = _REF_BY_TEMP[temp] || [];
  refChart.update("none");
}}"""

    # ------------------------------------------------------------------
    # Optional NiiVue viewer panel
    # ------------------------------------------------------------------
    _has_viewer = bool(nifti_image and Path(nifti_image).exists())
    if _has_viewer:
        # Only show vials present in the reference JSON (if provided)
        _viewer_vials = dict(vial_niftis or {})
        if ref_data is not None:
            _ref_set = {v.upper() for v in ref_data.get("vials", [])}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _ref_set}
        _overlay_label = "IR contrast" if fit_key == "T1_ms" else "TE contrast"
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,  # type: ignore[arg-type]
            _viewer_vials,
            overlay_nifti=overlay_contrast,
            overlay_label=_overlay_label,
        )
    else:
        viewer_html = viewer_js = ""

    head = html_head(title, include_niivue=_has_viewer)

    # Build chart card divs — 2-column grid
    chart_cards_html = ""
    for g_idx, group in enumerate(vial_groups):
        group_title = " & ".join(f"Vial {v}" for v in group)
        chart_cards_html += f"""  <div class="chart-card">
    <div class="chart-title">{group_title}</div>
    <div class="chart-wrap"><canvas id="chart{g_idx}"></canvas></div>
  </div>
"""

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>

{viewer_html}

{PK_CONTROLS_HTML_WITH_LEGEND}

<div class="chart-grid">
{chart_cards_html}
</div>
<p style="font-size:11px;color:var(--text2);margin:2px 0 18px;padding:0 4px;">
  Shaded band: 95&thinsp;% CI of fitted curve (always based on mean)
</p>

{ref_chart_html}

{fits_table_html}

{data_tag}

<script>
const SUBPLOTS = {datasets_json};
const PK_DATA = {pk_data_json};
{axis_consts_js}
{ERROR_BAR_PLUGIN_JS}
{PK_TOGGLE_JS}
{opts_js}

function makeChart(id, datasets) {{
  const opts = baseOpts({x_label_json}, "Intensity");
  opts.scales.x.min = X_MIN;
  opts.scales.x.max = X_MAX;
  opts.scales.y.min = Y_MIN;
  opts.scales.y.max = Y_MAX;
  const ctx = document.getElementById(id).getContext("2d");
  const c = new Chart(ctx, {{
    type: "line",
    data: {{ datasets }},
    options: opts,
    plugins: [errorBarPlugin],
  }});
  _pkCharts.push(c);
  return c;
}}

SUBPLOTS.forEach((datasets, i) => makeChart("chart" + i, datasets));
pkUpdateAllCharts();
{ref_chart_js}
{viewer_js}
</script>
</body>
</html>"""


def parse_phantomkit_html(html_path: str) -> dict:
    """Parse the embedded phantomkit-data JSON from an HTML output file.

    Works without BeautifulSoup by using a simple regex, making it a
    zero-dependency helper for downstream scripts.
    """
    text = Path(html_path).read_text(encoding="utf-8")
    m = re.search(
        r'<script\s+id="phantomkit-data"\s+type="application/json">\s*(\{.*?\})\s*</script>',
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(
            f"No phantomkit-data JSON block found in {html_path!r}. "
            "Is this a phantomkit HTML output file?"
        )
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# HTML head builder
# ---------------------------------------------------------------------------


def html_head(title: str, include_niivue: bool = False) -> str:
    """Return the ``<head>`` block with Chart.js CDN scripts and shared CSS.

    Parameters
    ----------
    include_niivue:
        When True, the NiiVue script is added to the head (inlined bundle or
        CDN fallback) for pages that embed an MRI viewer panel.
    """
    niivue_tag = _niivue_script_tag() if include_niivue else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="{CHARTJS_CDN}"></script>
<script src="{HAMMERJS_CDN}"></script>
<script src="{ZOOM_CDN}"></script>
{niivue_tag}
<style>{_CSS}</style>
</head>
"""
