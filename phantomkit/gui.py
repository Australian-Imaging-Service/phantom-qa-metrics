"""PhantomKit web GUI — FastAPI backend + browser frontend."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_HERE  = Path(__file__).parent
_TMPL  = _HERE.parent / "template_data"
_HOST  = "0.0.0.0"
_PORT  = 7878

# When running in Docker, the container's own home directory (e.g. /root) is
# empty. PHANTOMKIT_HOME lets the launch script point the file browser at
# wherever the user's real filesystem was mounted (e.g. /hostuser).
_BROWSE_ROOT = Path(os.environ.get("PHANTOMKIT_HOME", str(Path.home())))

DWI_STEPS = [
    ("dwidenoise",     "Denoise (dwidenoise)"),
    ("mrgibbs",        "Gibbs ringing correction (mrgibbs)"),
    ("dwifslpreproc",  "Eddy current correction (dwifslpreproc)"),
    ("dwibiascorrect", "Bias field correction (dwibiascorrect)"),
    ("gradcheck",      "Gradient check"),
]


def _phantoms() -> list[str]:
    if _TMPL.is_dir():
        names = sorted(d.name for d in _TMPL.iterdir()
                       if d.is_dir() and not d.name.startswith("."))
        return names or ["SPIRIT", "120E", "PET"]
    return ["SPIRIT", "120E", "PET"]


PHANTOMS = _phantoms()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PhantomKit")


# ── Filesystem browser ───────────────────────────────────────────────────────

@app.get("/api/browse")
async def browse(path: str = "", mode: str = "dir"):
    """Return directory listing for the browser modal."""
    target = Path(path).expanduser() if path else _BROWSE_ROOT
    if not target.is_dir():
        target = target.parent
    entries = []
    try:
        items = sorted(target.iterdir(),
                       key=lambda p: (p.is_file(), p.name.lower()))
        for p in items:
            if p.name.startswith("."):
                continue
            is_dir  = p.is_dir()
            is_html = p.suffix == ".html"
            if mode == "html" and p.is_file() and not is_html:
                continue
            entries.append({"name": p.name, "is_dir": is_dir, "path": str(p)})
    except PermissionError:
        pass
    parent = str(target.parent) if target.parent != target else ""
    return JSONResponse({"path": str(target), "parent": parent, "entries": entries})


# ── HTML result helpers ──────────────────────────────────────────────────────

@app.get("/api/list-htmls")
async def list_htmls(output_dir: str = ""):
    """Return all HTML files under output_dir/**/metrics/plots/."""
    base = Path(output_dir)
    if not base.is_dir():
        return JSONResponse({"htmls": []})
    htmls = sorted(str(h) for h in base.rglob("metrics/plots/*.html"))
    return JSONResponse({"htmls": htmls})


@app.get("/api/serve-html")
async def serve_html(path: str = ""):
    """Serve an HTML file directly so the browser can open it in a new tab."""
    p = Path(path)
    if not p.is_file() or p.suffix != ".html":
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(str(p), media_type="text/html")


# ── Streaming runner ─────────────────────────────────────────────────────────

async def _stream(cmd: list[str]):
    # PYTHONUNBUFFERED: without it, the child's stdout is fully block-buffered
    # (it's a pipe, not a tty), so print()-based progress from a long-running
    # pipeline run never reaches the browser until the process exits.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout
    async for raw in proc.stdout:
        yield f"data:{json.dumps(raw.decode(errors='replace'))}\n\n"
    await proc.wait()
    yield f"data:{json.dumps(f'__done__{proc.returncode}')}\n\n"


# ── Request models ────────────────────────────────────────────────────────────

class PipelineReq(BaseModel):
    input_dir:    str
    output_dir:   str
    phantom:      str
    steps:        list[str] = []
    nocleanup:    bool = False
    dry_run:      bool = False
    readout_time: str  = ""
    eddy_options: str  = ""


class PlotReq(BaseModel):
    html_files: list[str]
    labels:     list[str] = []
    output:     str
    phantom:    str = ""


class VendorCompareReq(BaseModel):
    output_dir:     str
    vendor_images:  list[str]
    vendor_labels:  list[str] = []
    map_type:       str
    phantom:        str = ""
    output:         str


# ── Run endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/run/pipeline")
async def run_pipeline(r: PipelineReq):
    input_path = Path(r.input_dir)

    # If the supplied dir contains a 'scans' subdirectory, use that.
    scans_subdir = input_path / "scans"
    actual_input = scans_subdir if scans_subdir.is_dir() else input_path

    # Create output/<subject_name>/ from the original input dir basename.
    subject_name = input_path.name
    actual_output = Path(r.output_dir) / subject_name

    async def _stream_with_header():
        # mkdir happens inside the stream (rather than before the
        # StreamingResponse is constructed) so a failure — e.g. the path
        # isn't actually mounted into the container — shows up as a log
        # line instead of a bare 500 that the frontend's SSE reader can
        # never surface, leaving the UI looking permanently hung.
        try:
            actual_output.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            yield f"data:{json.dumps(f'ERROR: could not create output directory {actual_output}: {e}\n')}\n\n"
            yield f"data:{json.dumps('__done__1')}\n\n"
            return

        header = (
            f"Input:  {actual_input}\n"
            f"Output: {actual_output}\n"
            + ("" if actual_input == input_path
               else f"  (using scans/ subdirectory)\n")
            + "\n"
        )
        yield f"data:{json.dumps(header)}\n\n"

        cmd = [sys.executable, "-m", "phantomkit", "pipeline",
               "--input-dir",  str(actual_input),
               "--output-dir", str(actual_output),
               "--phantom",    r.phantom]
        if r.steps:        cmd += ["--processing-steps", ",".join(r.steps)]
        if r.nocleanup:    cmd.append("--nocleanup")
        if r.dry_run:      cmd.append("--dry-run")
        if r.readout_time: cmd += ["--readout-time", r.readout_time]
        if r.eddy_options: cmd += ["--eddy-options",  r.eddy_options]

        async for chunk in _stream(cmd):
            yield chunk

    return StreamingResponse(_stream_with_header(), media_type="text/event-stream")


@app.post("/api/run/compare")
async def run_compare(r: PlotReq):
    cmd = [sys.executable, "-m", "phantomkit", "plot", "compare-plots",
           *r.html_files, "-o", r.output]
    if r.phantom:
        cmd += ["--phantom", r.phantom]
    for lbl in r.labels:
        if lbl:
            cmd += ["--label", lbl]
    return StreamingResponse(_stream(cmd), media_type="text/event-stream")


@app.post("/api/run/longitudinal")
async def run_longitudinal(r: PlotReq):
    cmd = [sys.executable, "-m", "phantomkit", "plot", "longitudinal",
           *r.html_files, "-o", r.output]
    if r.phantom:
        cmd += ["--phantom", r.phantom]
    for lbl in r.labels:
        if lbl:
            cmd += ["--label", lbl]
    return StreamingResponse(_stream(cmd), media_type="text/event-stream")


@app.post("/api/run/vendor-compare")
async def run_vendor_compare(r: VendorCompareReq):
    cmd = [sys.executable, "-m", "phantomkit", "vendor-compare",
           "--output-dir", r.output_dir,
           "--map-type",   r.map_type,
           "-o",           r.output]
    for img in r.vendor_images:
        cmd += ["--vendor-image", img]
    for lbl in r.vendor_labels:
        if lbl:
            cmd += ["--vendor-label", lbl]
    if r.phantom:
        cmd += ["--phantom", r.phantom]
    return StreamingResponse(_stream(cmd), media_type="text/event-stream")


# ── HTML page ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (_PAGE
            .replace("__PHANTOMS__", json.dumps(PHANTOMS))
            .replace("__STEPS__",    json.dumps(DWI_STEPS)))


# ---------------------------------------------------------------------------
# Page template  (no external deps — everything inline)
# ---------------------------------------------------------------------------

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhantomKit</title>
<style>
/* ── reset & variables ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:      #fafafa;
  --bg2:     #ffffff;
  --bg3:     #f0f0f0;
  --border:  #e2e2e2;
  --text:    #18181b;
  --text2:   #71717a;
  --accent:  #378ADD;
  --accent2: #2563b0;
  --danger:  #dc2626;
  --ok:      #16a34a;
  --radius:  8px;
  --shadow:  0 1px 3px rgba(0,0,0,.08);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  color: var(--text);
  background: var(--bg);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:     #0f0f12;
    --bg2:    #18181b;
    --bg3:    #27272a;
    --border: #3f3f46;
    --text:   #f4f4f5;
    --text2:  #a1a1aa;
    --shadow: 0 1px 3px rgba(0,0,0,.4);
  }
}

/* ── layout ── */
body { max-width: 820px; margin: 0 auto; padding: 0 16px 60px; }

header { padding: 28px 0 20px; }
header h1 { font-size: 1.6rem; font-weight: 700; }
header p  { color: var(--text2); margin-top: 3px; }

/* ── tabs ── */
.tabs { display: flex; gap: 4px; border-bottom: 2px solid var(--border);
        margin-bottom: 20px; }
.tab  { background: none; border: none; padding: 8px 20px; font-size: 14px;
        font-weight: 500; color: var(--text2); cursor: pointer;
        border-bottom: 2px solid transparent; margin-bottom: -2px;
        border-radius: var(--radius) var(--radius) 0 0; transition: .15s; }
.tab:hover  { color: var(--text); background: var(--bg3); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }

.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── cards ── */
.card { background: var(--bg2); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 18px 20px; margin-bottom: 14px;
        box-shadow: var(--shadow); }
.card-title { font-weight: 600; margin-bottom: 14px; font-size: 13px;
              text-transform: uppercase; letter-spacing: .04em;
              color: var(--text2); }

/* ── form rows ── */
.field-row { display: flex; align-items: center; gap: 10px;
             margin-bottom: 10px; flex-wrap: wrap; }
.field-row:last-child { margin-bottom: 0; }
.field-row label { min-width: 160px; color: var(--text2); flex-shrink: 0; }
.input-group { display: flex; gap: 6px; flex: 1; }
.input-group input { flex: 1; }

input[type=text], select {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 7px 10px; color: var(--text);
  font-size: 13px; outline: none; width: 100%;
  transition: border-color .15s;
}
input[type=text]:focus, select:focus { border-color: var(--accent); }
input[type=text]::placeholder { color: var(--text2); }

select { cursor: pointer; }

/* checkboxes */
.check-row { display: flex; align-items: center; gap: 8px;
             margin-bottom: 8px; cursor: pointer; }
.check-row:last-child { margin-bottom: 0; }
.check-row input { width: 15px; height: 15px; cursor: pointer;
                   accent-color: var(--accent); }
.check-row span { user-select: none; }
.hint { font-size: 12px; color: var(--text2); font-weight: 400;
        text-transform: none; letter-spacing: 0; }

/* advanced details */
details { margin-top: 10px; }
details summary { cursor: pointer; color: var(--accent); font-size: 13px;
                  user-select: none; margin-bottom: 10px; }
details summary:hover { text-decoration: underline; }

/* ── buttons ── */
button { cursor: pointer; border: none; border-radius: 6px;
         font-size: 13px; transition: .15s; }
.btn-primary { background: var(--accent); color: #fff; padding: 8px 18px;
               font-weight: 600; }
.btn-primary:hover:not(:disabled) { background: var(--accent2); }
.btn-primary:disabled { opacity: .5; cursor: not-allowed; }
.btn-sm { background: var(--bg3); color: var(--text); padding: 6px 12px;
          border: 1px solid var(--border); }
.btn-sm:hover { background: var(--border); }
.btn-danger { background: #7f1d1d22; color: var(--danger);
              border: 1px solid var(--danger); padding: 4px 10px; }
.btn-danger:hover { background: #7f1d1d44; }
.run-row { display: flex; justify-content: flex-start; margin: 18px 0 10px; }

/* ── log box ── */
.log { background: #0d0d0d; color: #e2e8f0; border-radius: var(--radius);
       padding: 14px 16px; font-family: "Menlo", "Consolas", monospace;
       font-size: 12px; line-height: 1.6; min-height: 120px; max-height: 340px;
       overflow-y: auto; white-space: pre-wrap; word-break: break-all;
       border: 1px solid var(--border); display: none; }
.log.visible { display: block; }
.log .ok   { color: #86efac; }
.log .err  { color: #fca5a5; }
.log .cmd  { color: #93c5fd; margin-bottom: 6px; display: block; }

/* ── file list (compare / longitudinal) ── */
.file-list { display: flex; flex-direction: column; gap: 8px;
             margin-bottom: 12px; }
.file-row  { display: flex; align-items: center; gap: 8px; }
.file-row input { flex: 2; }
.file-row .lbl-input { flex: 1; }

/* ── browser modal ── */
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,.55);
           display: flex; align-items: center; justify-content: center;
           z-index: 100; }
.overlay.hidden { display: none; }
.browser-box { background: var(--bg2); border: 1px solid var(--border);
               border-radius: var(--radius); width: min(560px, 94vw);
               max-height: 80vh; display: flex; flex-direction: column;
               box-shadow: 0 8px 32px rgba(0,0,0,.3); }
.browser-head { display: flex; align-items: center; gap: 10px;
                padding: 12px 14px; border-bottom: 1px solid var(--border); }
.browser-head .path { flex: 1; font-size: 12px; color: var(--text2);
                      word-break: break-all; }
.browser-head button { background: none; color: var(--text2); font-size: 18px;
                       padding: 0 4px; }
.browser-head button:hover { color: var(--text); }
.browser-entries { overflow-y: auto; flex: 1; padding: 6px 0; }
.browser-entry { display: flex; align-items: center; gap: 10px;
                 padding: 7px 16px; cursor: pointer; transition: .1s; }
.browser-entry:hover { background: var(--bg3); }
.browser-entry .icon { font-size: 15px; width: 20px; text-align: center;
                       flex-shrink: 0; }
.browser-entry .name { flex: 1; font-size: 13px; }
.browser-entry.file-entry { color: var(--text2); }
.browser-entry.file-entry:hover { color: var(--text); }
.browser-save-row { padding: 10px 16px; border-top: 1px solid var(--border);
                    display: flex; align-items: center; gap: 8px; }
.browser-save-row label { color: var(--text2); white-space: nowrap; }
.browser-save-row input { flex: 1; }
.browser-foot { display: flex; justify-content: flex-end; gap: 8px;
                padding: 10px 14px; border-top: 1px solid var(--border); }

/* ── open-html result buttons ── */
.html-results { display: flex; flex-wrap: wrap; align-items: center;
                gap: 8px; margin-top: 10px; }
.html-results-label { font-size: 12px; color: var(--text2); flex-basis: 100%; }
.btn-open-html { background: var(--bg2); color: var(--accent);
                 border: 1px solid var(--accent); border-radius: 6px;
                 padding: 6px 14px; font-size: 13px; font-weight: 500; }
.btn-open-html:hover { background: var(--accent); color: #fff; }
</style>
</head>
<body>

<header>
  <h1>PhantomKit</h1>
  <p>Phantom QA processing suite</p>
</header>

<nav class="tabs">
  <button class="tab active" onclick="switchTab('pipeline', this)">Pipeline</button>
  <button class="tab"        onclick="switchTab('compare',  this)">Compare</button>
  <button class="tab"        onclick="switchTab('vendor',   this)">Vendor Comparison</button>
</nav>

<!-- ══════════════════════════════════ PIPELINE ══════════════════════════════ -->
<section id="tab-pipeline" class="tab-content active">

  <div class="card">
    <div class="card-title">Required</div>

    <div class="field-row">
      <label>Input directory</label>
      <div class="input-group">
        <input type="text" id="p-input-dir" placeholder="/path/to/input">
        <button class="btn-sm" onclick="openBrowser('p-input-dir','dir')">Browse…</button>
      </div>
    </div>

    <div class="field-row">
      <label>Output directory</label>
      <div class="input-group">
        <input type="text" id="p-output-dir" placeholder="/path/to/output">
        <button class="btn-sm" onclick="openBrowser('p-output-dir','dir')">Browse…</button>
      </div>
    </div>

    <div class="field-row">
      <label>Phantom</label>
      <select id="p-phantom" style="max-width:200px"></select>
    </div>
  </div>

  <div class="card">
    <div class="card-title">DWI processing <span class="hint">&nbsp;— leave unchecked if no DWI data</span></div>
    <div id="dwi-steps"></div>
  </div>

  <div class="card">
    <div class="card-title">Options</div>
    <label class="check-row">
      <input type="checkbox" id="p-nocleanup">
      <span>Keep intermediate files</span>
    </label>
    <label class="check-row">
      <input type="checkbox" id="p-dry-run">
      <span>Dry run (preview commands only)</span>
    </label>
    <details>
      <summary>Advanced options</summary>
      <div class="field-row" style="margin-top:4px">
        <label>Readout time (s)</label>
        <input type="text" id="p-readout-time" placeholder="auto-detected" style="max-width:160px">
      </div>
      <div class="field-row">
        <label>Eddy options</label>
        <input type="text" id="p-eddy-options" placeholder="--slm=linear  (default)">
      </div>
    </details>
  </div>

  <div class="run-row">
    <button class="btn-primary" id="p-run-btn" onclick="runPipeline()">▶&nbsp;Run</button>
  </div>
  <div class="log" id="p-log"></div>

</section>

<!-- ══════════════════════════════════ COMPARE ═══════════════════════════════ -->
<section id="tab-compare" class="tab-content">

  <div class="card">
    <div class="card-title">Input HTML files</div>
    <div class="file-list" id="c-file-list"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">
      <button class="btn-sm" onclick="addFileRow('c-file-list','html')">+ Add file</button>
      <button class="btn-sm" onclick="openBrowserMulti('c-file-list')">+ Add multiple…</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Output</div>
    <div class="field-row">
      <label>Output HTML</label>
      <div class="input-group">
        <input type="text" id="c-output" placeholder="/path/to/comparison.html">
        <button class="btn-sm" onclick="openBrowser('c-output','save')">Save as…</button>
      </div>
    </div>
    <div class="field-row">
      <label>Phantom <span class="hint">(optional)</span></label>
      <select id="c-phantom" style="max-width:220px"></select>
    </div>
    <label class="check-row">
      <input type="checkbox" id="c-longitudinal">
      <span>Longitudinal analysis
        <span class="hint">— plot per-vial trends across sessions instead of a side-by-side comparison</span>
      </span>
    </label>
  </div>

  <div class="run-row">
    <button class="btn-primary" id="c-run-btn" onclick="runCompare()">▶&nbsp;Run</button>
  </div>
  <div class="log" id="c-log"></div>

</section>

<!-- ═══════════════════════════════ VENDOR COMPARISON ═════════════════════════ -->
<section id="tab-vendor" class="tab-content">

  <div class="card">
    <div class="card-title">Required</div>

    <div class="field-row">
      <label>Phantom</label>
      <select id="v-phantom" style="max-width:200px"></select>
    </div>

    <div class="field-row">
      <label>Pipeline output directory</label>
      <div class="input-group">
        <input type="text" id="v-output-dir" placeholder="/path/to/output/session">
        <button class="btn-sm" onclick="openBrowser('v-output-dir','dir')">Browse…</button>
      </div>
    </div>

    <div class="field-row">
      <label>Map type</label>
      <select id="v-map-type" style="max-width:200px">
        <option value="adc">ADC (DWI space)</option>
        <option value="t1">T1 (native contrast space)</option>
        <option value="t2">T2 (native contrast space)</option>
      </select>
    </div>

  </div>

  <div class="card">
    <div class="card-title">Vendor images</div>
    <div class="file-list" id="v-vendor-list"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">
      <button class="btn-sm" onclick="addFileRow('v-vendor-list','file')">+ Add vendor image</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Output</div>
    <div class="field-row">
      <label>Output HTML</label>
      <div class="input-group">
        <input type="text" id="v-output" placeholder="/path/to/vendor_compare.html">
        <button class="btn-sm" onclick="openBrowser('v-output','save')">Save as…</button>
      </div>
    </div>
  </div>

  <div class="run-row">
    <button class="btn-primary" id="v-run-btn" onclick="runVendorCompare()">▶&nbsp;Run</button>
  </div>
  <div class="log" id="v-log"></div>

</section>

<!-- ══════════════════════════════════ FILE BROWSER ══════════════════════════ -->
<div id="browser-overlay" class="overlay hidden" onclick="overlayClick(event)">
  <div class="browser-box">
    <div class="browser-head">
      <button onclick="closeBrowser()" title="Close">✕</button>
      <span class="path" id="br-path"></span>
      <button id="br-up-btn" onclick="browseUp()" title="Up">↑</button>
    </div>
    <div class="browser-entries" id="br-entries"></div>
    <div class="browser-save-row" id="br-save-row" style="display:none">
      <label>Filename:</label>
      <input type="text" id="br-filename" placeholder="output.html">
    </div>
    <div class="browser-foot">
      <button class="btn-sm" onclick="closeBrowser()">Cancel</button>
      <button class="btn-primary" id="br-select-btn" onclick="confirmBrowser()">Select folder</button>
    </div>
  </div>
</div>

<script>
// ── Data injected by server ──────────────────────────────────────────────────
const PHANTOMS  = __PHANTOMS__;
const DWI_STEPS = __STEPS__;

// ── Initialise selects and checkboxes ────────────────────────────────────────
(function init() {
  // Phantom dropdowns — always an explicit phantom, never auto-detected
  ['p-phantom','c-phantom','v-phantom'].forEach(function(id) {
    var sel = document.getElementById(id);
    PHANTOMS.forEach(function(p) {
      var o = document.createElement('option');
      o.value = o.textContent = p;
      sel.appendChild(o);
    });
  });

  // DWI step checkboxes
  var container = document.getElementById('dwi-steps');
  DWI_STEPS.forEach(function(step) {
    var key = step[0], label = step[1];
    var lbl = document.createElement('label');
    lbl.className = 'check-row';
    lbl.innerHTML =
      '<input type="checkbox" id="step-' + key + '">' +
      '<span>' + label + '</span>';
    container.appendChild(lbl);
  });

  // Initial file rows
  addFileRow('c-file-list', 'html');
  addFileRow('v-vendor-list', 'file');
})();

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.tab-content').forEach(function(s) { s.classList.remove('active'); });
  document.querySelectorAll('.tab').forEach(function(b)        { b.classList.remove('active'); });
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}

// ── Dynamic file rows ────────────────────────────────────────────────────────
function addFileRow(listId, mode) {
  var list  = document.getElementById(listId);
  var row   = document.createElement('div');
  row.className = 'file-row';

  var pathIn = document.createElement('input');
  pathIn.type = 'text';
  pathIn.placeholder = 'path/to/session.html';

  var lblIn = document.createElement('input');
  lblIn.type = 'text';
  lblIn.className = 'lbl-input';
  lblIn.placeholder = 'Label (optional)';

  var browseBtn = document.createElement('button');
  browseBtn.className = 'btn-sm';
  browseBtn.textContent = '…';
  browseBtn.onclick = function() { openBrowser(null, mode, pathIn); };

  var removeBtn = document.createElement('button');
  removeBtn.className = 'btn-danger';
  removeBtn.textContent = '✕';
  removeBtn.onclick = function() {
    if (list.querySelectorAll('.file-row').length > 1) list.removeChild(row);
  };

  row.appendChild(browseBtn);
  row.appendChild(pathIn);
  row.appendChild(lblIn);
  row.appendChild(removeBtn);
  list.appendChild(row);
}

function getFileRows(listId) {
  var rows = document.querySelectorAll('#' + listId + ' .file-row');
  var files = [], labels = [];
  rows.forEach(function(row) {
    var inputs = row.querySelectorAll('input[type=text]');
    var f = inputs[0].value.trim();
    var l = inputs[1].value.trim();
    if (f) { files.push(f); labels.push(l); }
  });
  return { files: files, labels: labels };
}

// ── File browser modal ───────────────────────────────────────────────────────
var _brTarget    = null;   // DOM input element (single modes)
var _brMode      = 'dir';  // 'dir', 'html', 'save', 'html-multi'
var _brPath      = '';
var _brParent    = '';
var _brMultiList = null;   // listId for html-multi mode
var _brSelected  = [];     // selected paths in html-multi mode

function openBrowser(targetId, mode, targetEl) {
  _brTarget = targetEl || document.getElementById(targetId);
  _brMode   = mode;
  var startPath = (_brTarget && _brTarget.value.trim())
                ? _brTarget.value.trim()
                : '';
  document.getElementById('br-save-row').style.display =
    (mode === 'save') ? 'flex' : 'none';
  document.getElementById('br-select-btn').textContent =
    (mode === 'save')  ? 'Save here' :
    (mode === 'dir')   ? 'Select folder' : 'Select file';
  document.getElementById('browser-overlay').classList.remove('hidden');
  browseTo(startPath || '');
}

function openBrowserMulti(listId) {
  _brMultiList = listId;
  _brSelected  = [];
  _brMode      = 'html-multi';
  document.getElementById('br-save-row').style.display = 'none';
  document.getElementById('br-select-btn').textContent = 'Add 0 files';
  document.getElementById('browser-overlay').classList.remove('hidden');
  browseTo('');
}

function _updateMultiBtn() {
  var n = _brSelected.length;
  document.getElementById('br-select-btn').textContent =
    'Add ' + n + ' file' + (n !== 1 ? 's' : '');
}

function addFileRowWithPath(listId, path) {
  var list = document.getElementById(listId);
  var row  = document.createElement('div');
  row.className = 'file-row';

  var pathIn = document.createElement('input');
  pathIn.type  = 'text';
  pathIn.value = path;

  var lblIn = document.createElement('input');
  lblIn.type = 'text';
  lblIn.className   = 'lbl-input';
  lblIn.placeholder = 'Label (optional)';

  var browseBtn = document.createElement('button');
  browseBtn.className   = 'btn-sm';
  browseBtn.textContent = '…';
  browseBtn.onclick = function() { openBrowser(null, 'html', pathIn); };

  var removeBtn = document.createElement('button');
  removeBtn.className   = 'btn-danger';
  removeBtn.textContent = '✕';
  removeBtn.onclick = function() {
    if (list.querySelectorAll('.file-row').length > 1) list.removeChild(row);
  };

  row.appendChild(browseBtn);
  row.appendChild(pathIn);
  row.appendChild(lblIn);
  row.appendChild(removeBtn);
  list.appendChild(row);
}

function closeBrowser() {
  document.getElementById('browser-overlay').classList.add('hidden');
}

function overlayClick(e) {
  if (e.target === document.getElementById('browser-overlay')) closeBrowser();
}

async function browseTo(path) {
  var url = '/api/browse?mode=' + _brMode + '&path=' + encodeURIComponent(path);
  var resp = await fetch(url);
  var data = await resp.json();
  _brPath   = data.path;
  _brParent = data.parent;

  document.getElementById('br-path').textContent = _brPath;
  document.getElementById('br-up-btn').disabled = !_brParent;

  var cont = document.getElementById('br-entries');
  cont.innerHTML = '';

  data.entries.forEach(function(entry) {
    var div = document.createElement('div');
    div.className = 'browser-entry' + (entry.is_dir ? '' : ' file-entry');

    if (_brMode === 'html-multi' && !entry.is_dir) {
      var cb = document.createElement('input');
      cb.type    = 'checkbox';
      cb.checked = _brSelected.indexOf(entry.path) !== -1;
      cb.style   = 'margin-right:8px;accent-color:var(--accent);cursor:pointer;';
      cb.onchange = function() {
        var idx = _brSelected.indexOf(entry.path);
        if (cb.checked && idx === -1) _brSelected.push(entry.path);
        else if (!cb.checked && idx !== -1) _brSelected.splice(idx, 1);
        _updateMultiBtn();
      };
      div.appendChild(cb);
      div.onclick = function(e) { if (e.target !== cb) { cb.checked = !cb.checked; cb.onchange(); } };
    }

    var icon = document.createElement('span');
    icon.className   = 'icon';
    icon.textContent = entry.is_dir ? '📁' : '📄';
    var name = document.createElement('span');
    name.className   = 'name';
    name.textContent = entry.name;
    div.appendChild(icon);
    div.appendChild(name);

    if (entry.is_dir) {
      div.onclick = function() { browseTo(entry.path); };
    } else if (_brMode === 'html' || _brMode === 'file') {
      div.onclick = function() { selectPath(entry.path); };
    }
    cont.appendChild(div);
  });
}

function browseUp() {
  if (_brParent) browseTo(_brParent);
}

function selectPath(path) {
  if (_brTarget) _brTarget.value = path;
  closeBrowser();
}

function confirmBrowser() {
  if (_brMode === 'save') {
    var fname = document.getElementById('br-filename').value.trim();
    if (!fname) { alert('Enter a filename.'); return; }
    if (!fname.endsWith('.html')) fname += '.html';
    selectPath(_brPath + '/' + fname);
  } else if (_brMode === 'dir') {
    selectPath(_brPath);
  } else if (_brMode === 'html-multi') {
    if (_brSelected.length === 0) { alert('Select at least one file.'); return; }
    // Remove any empty placeholder row before adding
    var list = document.getElementById(_brMultiList);
    var emptyRows = [];
    list.querySelectorAll('.file-row').forEach(function(row) {
      if (!row.querySelector('input[type=text]').value.trim()) emptyRows.push(row);
    });
    _brSelected.forEach(function(path) { addFileRowWithPath(_brMultiList, path); });
    // Remove empty rows that were there before (if files were added)
    emptyRows.forEach(function(row) { list.removeChild(row); });
    closeBrowser();
  }
  // 'html' mode selects via entry click
}

// ── Open-HTML result buttons ─────────────────────────────────────────────────
async function showHtmlButtons(logId, paths) {
  var existing = document.getElementById(logId + '-results');
  if (existing) existing.remove();
  if (!paths || paths.length === 0) return;

  var container = document.createElement('div');
  container.id = logId + '-results';
  container.className = 'html-results';

  var lbl = document.createElement('div');
  lbl.className = 'html-results-label';
  lbl.textContent = 'Open results:';
  container.appendChild(lbl);

  paths.forEach(function(path) {
    var btn = document.createElement('button');
    btn.className = 'btn-open-html';
    btn.textContent = '↗ ' + path.split('/').pop().replace('.html', '');
    btn.onclick = function() {
      window.open('/api/serve-html?path=' + encodeURIComponent(path), '_blank');
    };
    container.appendChild(btn);
  });

  document.getElementById(logId).insertAdjacentElement('afterend', container);
}

// ── Streaming runner ─────────────────────────────────────────────────────────
async function streamRun(endpoint, payload, logId, btnId, onSuccess) {
  var log = document.getElementById(logId);
  var btn = document.getElementById(btnId);

  log.innerHTML = '';
  log.className = 'log visible';

  // Clear result buttons from a previous run
  var prevResults = document.getElementById(logId + '-results');
  if (prevResults) prevResults.remove();

  var cmdSpan = document.createElement('span');
  cmdSpan.className = 'cmd';
  cmdSpan.textContent = 'Running: ' + endpoint + '  …';
  log.appendChild(cmdSpan);

  btn.disabled = true;
  btn.textContent = '⏳ Running…';

  try {
    var resp = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    var reader  = resp.body.getReader();
    var decoder = new TextDecoder();
    var buf     = '';

    while (true) {
      var result = await reader.read();
      if (result.done) break;
      buf += decoder.decode(result.value, { stream: true });

      var parts = buf.split('\n\n');
      buf = parts.pop();

      for (var i = 0; i < parts.length; i++) {
        var evt = parts[i].trim();
        if (!evt) continue;
        if (evt.startsWith('data:')) {
          var payload2 = JSON.parse(evt.slice(5));
          if (typeof payload2 === 'string' && payload2.startsWith('__done__')) {
            var code = parseInt(payload2.slice(8));
            var tail = document.createElement('span');
            tail.className = code === 0 ? 'ok' : 'err';
            tail.textContent = code === 0
              ? '\n✓ Completed successfully.\n'
              : '\n✗ Exited with code ' + code + '.\n';
            log.appendChild(tail);
            if (code === 0 && typeof onSuccess === 'function') onSuccess();
          } else {
            log.appendChild(document.createTextNode(payload2));
          }
          log.scrollTop = log.scrollHeight;
        }
      }
    }
  } catch (err) {
    var e = document.createElement('span');
    e.className = 'err';
    e.textContent = '\n✗ Error: ' + err.message + '\n';
    log.appendChild(e);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run';
  }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────
function runPipeline() {
  var inputDir  = document.getElementById('p-input-dir').value.trim();
  var outputDir = document.getElementById('p-output-dir').value.trim();
  var phantom   = document.getElementById('p-phantom').value;
  if (!inputDir)  { alert('Please select an input directory.'); return; }
  if (!outputDir) { alert('Please select an output directory.'); return; }

  var steps = [];
  DWI_STEPS.forEach(function(s) {
    if (document.getElementById('step-' + s[0]).checked) steps.push(s[0]);
  });

  var rt   = document.getElementById('p-readout-time').value.trim();
  var eddy = document.getElementById('p-eddy-options').value.trim();

  if (rt && isNaN(parseFloat(rt))) {
    alert('Readout time must be a number (seconds).'); return;
  }

  var subjectName = inputDir.replace(/\/+$/, '').split('/').pop();
  var actualOutput = outputDir.replace(/\/+$/, '') + '/' + subjectName;

  streamRun('/api/run/pipeline', {
    input_dir:    inputDir,
    output_dir:   outputDir,
    phantom:      phantom,
    steps:        steps,
    nocleanup:    document.getElementById('p-nocleanup').checked,
    dry_run:      document.getElementById('p-dry-run').checked,
    readout_time: rt,
    eddy_options: eddy,
  }, 'p-log', 'p-run-btn', async function() {
    var resp = await fetch('/api/list-htmls?output_dir=' + encodeURIComponent(actualOutput));
    var data = await resp.json();
    showHtmlButtons('p-log', data.htmls);
  });
}

// ── Compare / Longitudinal ───────────────────────────────────────────────────
function runCompare() {
  var rows        = getFileRows('c-file-list');
  var output      = document.getElementById('c-output').value.trim();
  var phantom     = document.getElementById('c-phantom').value;
  var longitudinal = document.getElementById('c-longitudinal').checked;
  if (rows.files.length < 2) { alert('Select at least two HTML files.'); return; }
  if (!output)               { alert('Choose an output HTML path.'); return; }

  var endpoint = longitudinal ? '/api/run/longitudinal' : '/api/run/compare';

  streamRun(endpoint, {
    html_files: rows.files,
    labels:     rows.labels,
    output:     output,
    phantom:    phantom,
  }, 'c-log', 'c-run-btn', function() {
    showHtmlButtons('c-log', [output]);
  });
}

// ── Vendor Comparison ────────────────────────────────────────────────────────
function runVendorCompare() {
  var outputDir = document.getElementById('v-output-dir').value.trim();
  var rows      = getFileRows('v-vendor-list');
  var mapType   = document.getElementById('v-map-type').value;
  var phantom   = document.getElementById('v-phantom').value;
  var output    = document.getElementById('v-output').value.trim();
  if (!outputDir)            { alert('Select the pipeline output directory.'); return; }
  if (rows.files.length < 1) { alert('Select at least one vendor image.'); return; }
  if (!output)                { alert('Choose an output HTML path.'); return; }

  streamRun('/api/run/vendor-compare', {
    output_dir:     outputDir,
    vendor_images:  rows.files,
    vendor_labels:  rows.labels,
    map_type:       mapType,
    phantom:        phantom,
    output:         output,
  }, 'v-log', 'v-run-btn', function() {
    showHtmlButtons('v-log', [output]);
  });
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    local_url = f"http://127.0.0.1:{_PORT}"

    def _open():
        import time, webbrowser
        time.sleep(0.8)
        try:
            webbrowser.open(local_url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
    print(f"PhantomKit GUI → {local_url}  (Ctrl-C to quit)")
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="warning")


if __name__ == "__main__":
    main()
