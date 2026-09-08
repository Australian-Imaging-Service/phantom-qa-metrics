"""
Interactive MRI viewer for phantomkit QA results.

Serves a plain FastAPI/uvicorn HTTP server so NIfTI files can be fetched over
HTTP.  The viewer page is a self-contained HTML/JS file — NiiVue initialises
directly with no framework overhead.

Usage (CLI)::

    phantomkit view /path/to/contrast.nii.gz \\
        --vials /path/to/vials/ \\
        --title "T1 Inversion Recovery"

Usage (Python)::

    from phantomkit.plotting.viewer import launch_viewer
    launch_viewer(
        nifti_image="/path/to/contrast.nii.gz",
        vial_niftis={"vial_A": "/path/to/vials/vial_A.nii.gz", ...},
        title="T1 IR",
    )
"""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# NiiVue asset constants
# ---------------------------------------------------------------------------

_NIIVUE_CDN = (
    "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.46.0/dist/niivue.umd.min.js"
)
_NIIVUE_BUNDLE = Path(__file__).parent / "niivue.umd.min.js"

_COLORMAPS = [
    "red", "green", "blue", "yellow", "cyan",
    "warm", "cool", "plasma", "hot", "winter",
]


def _niivue_script_tag() -> str:
    """Inline the local NiiVue bundle if present, else fall back to CDN."""
    if _NIIVUE_BUNDLE.exists():
        src = _NIIVUE_BUNDLE.read_text(encoding="utf-8")
        src = re.sub(r"(?i)</script>", r"<\/script>", src)
        return f"<script>/* niivue@0.46.0 inlined */\n{src}\n</script>"
    return f'<script src="{_NIIVUE_CDN}"></script>'


# ---------------------------------------------------------------------------
# Plain HTML page builder
# ---------------------------------------------------------------------------

def _build_viewer_html(
    title: str,
    volumes_js: str,
    overlays: list[dict],
    dark: bool,
) -> str:
    """Return a self-contained HTML viewer page."""
    niivue_script = _niivue_script_tag()

    if dark:
        bg, bg2 = "#1c1c1a", "#252523"
        text, text2 = "#e8e8e4", "#888780"
        border = "rgba(255,255,255,0.12)"
        chip_bg, chip_border = "rgba(255,255,255,0.08)", "rgba(255,255,255,0.25)"
    else:
        bg, bg2 = "#ffffff", "#f4f4f2"
        text, text2 = "#1a1a18", "#5f5e5a"
        border = "rgba(0,0,0,0.12)"
        chip_bg, chip_border = "rgba(0,0,0,0.05)", "rgba(0,0,0,0.2)"

    chips_html = "".join(
        f'<button id="chip-{i}" data-active="1" '
        f'onclick="pkToggleVial({i + 1}, this)" '
        f'style="padding:4px 12px;border-radius:99px;'
        f'border:1.5px solid {chip_border};background:{chip_bg};'
        f'color:{text};font-size:11px;font-weight:500;cursor:pointer;'
        f'user-select:none;transition:opacity .15s;">'
        f'{ov["name"]}</button>'
        for i, ov in enumerate(overlays)
    )

    vial_section = (
        f'<p style="font-size:11px;color:{text2};margin:10px 0 6px;">'
        f"Toggle vial ROIs</p>"
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{chips_html}</div>'
    ) if overlays else ""

    # DRAG_MODE.pan == 2 in NiiVue 0.46.0
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title}</title>
{niivue_script}
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: {bg}; color: {text}; padding: 24px;
  }}
  h1 {{ font-size: 20px; font-weight: 500; margin-bottom: 20px; }}
  .card {{
    background: {bg2}; border: 0.5px solid {border};
    border-radius: 12px; padding: 16px; max-width: 1000px;
  }}
  .card-label {{ font-size: 13px; font-weight: 500; margin-bottom: 10px; }}
  #nv-canvas {{
    width: 100%; height: 500px; display: block;
    border-radius: 8px; background: #000; cursor: crosshair;
  }}
  .hint {{ font-size: 11px; color: {text2}; margin-top: 8px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="card">
  <p class="card-label">MRI Viewer</p>
  <canvas id="nv-canvas"></canvas>
  {vial_section}
  <p class="hint">Scroll to zoom &middot; drag to pan &middot; scroll wheel to change slice</p>
</div>

<script>
(function() {{
  var VOLUMES = {volumes_js};

  function _initNv(canvas, w, h) {{
    canvas.width  = w;
    canvas.height = h;
    window._pkNv = new niivue.Niivue({{
      isColorbar: false,
      crosshairWidth: 1,
      dragMode: 2,
      isResizeCanvas: false,
    }});
    window._pkNv.attachToCanvas(canvas);
    window._pkNv.opts.sliceType = 0;
    window._pkNv.loadVolumes(VOLUMES);
  }}

  var canvas = document.getElementById("nv-canvas");
  var ro = new ResizeObserver(function(entries) {{
    for (var i = 0; i < entries.length; i++) {{
      var r = entries[i].contentRect;
      var w = Math.round(r.width);
      var h = Math.round(r.height);
      if (w > 0 && h > 0) {{
        ro.disconnect();
        _initNv(canvas, w, h);
        break;
      }}
    }}
  }});
  ro.observe(canvas);
}})();

function pkToggleVial(idx, btn) {{
  if (!window._pkNv) return;
  var wasActive = btn.dataset.active === "1";
  btn.dataset.active = wasActive ? "0" : "1";
  btn.style.opacity  = wasActive ? "0.35" : "1.0";
  window._pkNv.setOpacity(idx, wasActive ? 0.0 : 0.5);
}}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------


def launch_viewer(
    nifti_image: str,
    vial_niftis: dict[str, str] | None = None,
    title: str = "MRI Viewer",
    port: int = 8080,
    dark: bool = True,
) -> None:
    """Launch an interactive MRI viewer with per-vial ROI toggle.

    Uses FastAPI + uvicorn to serve NIfTI files over HTTP so that NiiVue can
    fetch them.  The viewer page is plain HTML/JS with no framework overhead.

    Parameters
    ----------
    nifti_image : str
        Path to the background ``.nii`` or ``.nii.gz`` file.
    vial_niftis : dict, optional
        ``{vial_name: path}`` mapping for ROI overlay NIfTI files.
    title : str
        Browser tab / page title.
    port : int
        TCP port for the server (default 8080).
    dark : bool
        Dark colour scheme (default True).
    """
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    vial_niftis = vial_niftis or {}
    nifti_path = Path(nifti_image).resolve()

    # Resolve the actual file when the caller omits the NIfTI extension.
    if not nifti_path.exists():
        for _ext in (".nii.gz", ".nii"):
            _candidate = Path(str(nifti_path) + _ext)
            if _candidate.exists():
                nifti_path = _candidate
                break
        else:
            raise FileNotFoundError(f"NIfTI file not found: {nifti_image}")

    app = FastAPI()

    # ------------------------------------------------------------------
    # Mount static directories so NiiVue can fetch NIfTIs over HTTP.
    # ------------------------------------------------------------------
    app.mount("/niftis", StaticFiles(directory=str(nifti_path.parent)), name="niftis")
    bg_url = f"/niftis/{nifti_path.name}"

    _vial_dir_route: dict[str, str] = {}
    for vpath in vial_niftis.values():
        if vpath and Path(vpath).exists():
            d = str(Path(vpath).resolve().parent)
            if d not in _vial_dir_route:
                route = f"/vials{len(_vial_dir_route)}"
                app.mount(route, StaticFiles(directory=d), name=f"vials{len(_vial_dir_route)}")
                _vial_dir_route[d] = route

    overlays: list[dict] = []
    for i, (name, vpath) in enumerate(sorted(vial_niftis.items())):
        if not vpath or not Path(vpath).exists():
            continue
        d = str(Path(vpath).resolve().parent)
        url = f"{_vial_dir_route[d]}/{Path(vpath).name}"
        overlays.append({
            "name": name,
            "url": url,
            "colormap": _COLORMAPS[i % len(_COLORMAPS)],
        })

    volumes_js = json.dumps(
        [{"url": bg_url, "colormap": "gray"}]
        + [
            {
                "url": o["url"],
                "colormap": o["colormap"],
                "opacity": 0.5,
            }
            for o in overlays
        ]
    )

    html = _build_viewer_html(title, volumes_js, overlays, dark)

    @app.get("/viewer")
    async def _viewer_page() -> HTMLResponse:
        return HTMLResponse(content=html)

    @app.get("/")
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/viewer")

    viewer_url = f"http://127.0.0.1:{port}/viewer"
    threading.Timer(1.5, lambda: webbrowser.open(viewer_url)).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
