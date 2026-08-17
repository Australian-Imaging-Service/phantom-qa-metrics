"""
vendor_compare.py
==================
Compare a vendor-generated parametric map (ADC/T1/T2) against phantomkit's
own already-computed values for the same session, per vial.

Locates the session's existing vial masks and phantomkit HTML report for the
given map type, regrids the vial masks onto the vendor image's own voxel
grid (nearest-neighbor — the vendor image itself is never resampled, to
avoid interpolating a continuous parametric map), and extracts per-vial
statistics via the same mrstats/mrdump approach used elsewhere in the
package (:func:`phantomkit.pet_processing.extract_pet_vial_metrics`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

_MAP_TYPE_INFO = {
    # map_type: (report_glob, vial_seg_subdir_relative_to_output_dir_or_None)
    "adc": ("metrics/plots/*ADC*.html", None),
    "t1": ("native_contrasts/metrics/plots/*T1_mapping*.html", "native_contrasts"),
    "t2": ("native_contrasts/metrics/plots/*T2_mapping*.html", "native_contrasts"),
}


def locate_reference(output_dir: Path, map_type: str) -> dict:
    """Find the existing phantomkit report + vial masks for map_type under output_dir.

    Returns {"report_html": Path, "vial_masks": {vial_name: Path}}.
    Raises click.ClickException with a clear message if zero or multiple
    candidates are found.
    """
    map_type = map_type.lower()
    if map_type not in _MAP_TYPE_INFO:
        raise click.ClickException(
            f"Unknown --map-type {map_type!r}. Expected one of: "
            f"{', '.join(_MAP_TYPE_INFO)}."
        )
    report_glob, fixed_subdir = _MAP_TYPE_INFO[map_type]

    if fixed_subdir is not None:
        candidates = sorted((output_dir / fixed_subdir).glob(
            report_glob.split("/", 1)[1]
        )) if (output_dir / fixed_subdir).is_dir() else []
        vial_seg_dirs = [output_dir / fixed_subdir / "vial_segmentations"]
    else:
        # ADC lives under a per-DWI-series subdirectory whose name isn't
        # known in advance — search every immediate subdirectory except
        # native_contrasts (which holds T1/T2, not ADC).
        candidates = []
        vial_seg_dirs = []
        for series_dir in sorted(output_dir.iterdir()):
            if not series_dir.is_dir() or series_dir.name == "native_contrasts":
                continue
            found = sorted(series_dir.glob(report_glob))
            if found:
                candidates.extend(found)
                vial_seg_dirs.append(series_dir / "vial_segmentations")

    if not candidates:
        raise click.ClickException(
            f"No existing {map_type.upper()} report found under {output_dir} "
            f"(looked for {report_glob!r}). Has the pipeline been run for "
            f"this session yet?"
        )
    if len(candidates) > 1:
        listed = "\n".join(f"  - {c}" for c in candidates)
        raise click.ClickException(
            f"Found multiple candidate {map_type.upper()} reports under "
            f"{output_dir}, expected exactly one:\n{listed}\n"
            f"Point --output-dir at a narrower session directory."
        )

    report_html = candidates[0]
    vial_seg_dir = next((d for d in vial_seg_dirs if d.is_dir()), None)
    if vial_seg_dir is None or not any(vial_seg_dir.glob("*.nii.gz")):
        raise click.ClickException(
            f"No vial_segmentations/ found alongside {report_html} "
            f"(expected {vial_seg_dirs})."
        )

    vial_masks = {
        p.name.replace(".nii.gz", ""): p for p in sorted(vial_seg_dir.glob("*.nii.gz"))
    }

    # ADC also has a per-vial xlsx (mean/median/std/count/p25/p75/min/max/
    # mean_mad/median_mad sheets, written by phantom_processor.py's
    # _task_extract_metrics) sitting alongside the report — reading it gives
    # phantomkit's own series the same full distribution the freshly
    # computed vendor stats have, instead of just a fixed mean+std. T1/T2
    # only ever have a single curve-fit value (no such xlsx exists for
    # them), so this is ADC-only.
    reference_xlsx = None
    if map_type == "adc":
        xlsx_dir = report_html.parent.parent / "xlsx"
        xlsx_candidates = sorted(xlsx_dir.glob("*ADC*.xlsx")) if xlsx_dir.is_dir() else []
        if len(xlsx_candidates) == 1:
            reference_xlsx = xlsx_candidates[0]

    return {
        "report_html": report_html,
        "vial_masks": vial_masks,
        "reference_xlsx": reference_xlsx,
    }


def ensure_nifti(image_path: Path, tmp_dir: Path) -> Path:
    """Normalize a vendor image into a well-behaved NIfTI for the viewer.

    Always runs the image through mrconvert with an explicit float32
    datatype and canonical strides — not just for MIF input — matching the
    shape every image phantomkit's own pipeline already produces (all
    mrtrix/ANTs outputs). Vendor-exported images can have quirks
    phantomkit's own images never do: MRtrix's own MIF format entirely
    (a completely different, ASCII-header-based layout), int16 storage
    with scl_slope/intercept scaling, or unusual strides/axis ordering.
    nifti_to_base64() (_html_common.py) only does a minimal float64->
    float32 repack and otherwise passes the file's bytes straight through
    to the browser's NiiVue viewer, so anything unusual in the source
    file reaches it unchanged — which can hang trying to render it.
    Converting once upfront also keeps stats computation and the viewer
    consistent on the same file.
    """
    name = image_path.name
    stem = name
    for ext in (".mif.gz", ".mif", ".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{stem}_normalized.nii.gz"
    subprocess.run(
        [
            "mrconvert", str(image_path), str(out),
            "-datatype", "float32", "-strides", "1,2,3", "-force",
        ],
        check=True, capture_output=True,
    )
    return out


def infer_adc_scale(vendor_stats: dict) -> float:
    """Guess the multiplier needed to bring vendor ADC values in line with
    phantomkit's own ×10⁻³ mm²/s display convention.

    Vendor ADC maps appear in the wild in several different unit
    conventions:
      - raw mm²/s (~0.0003-0.003)                  -> ×1000
      - already ×10⁻³ mm²/s (~0.3-3, matching
        phantomkit's own display convention)        -> ×1
      - integer-scaled, ~×10⁻⁶ (a common vendor/
        DICOM storage convention, typically
        hundreds to low thousands)                   -> ×0.001

    This is a heuristic based on typical (median-of-means) magnitude —
    there's no reliable header field to read the convention from
    directly. The chosen scale is always reported in the CLI/GUI log so
    it's never a silent guess.
    """
    means = sorted(
        abs(s["mean"]) for s in vendor_stats.values() if s.get("mean") is not None
    )
    if not means:
        return 1e3
    typical = means[len(means) // 2]
    if typical < 0.05:
        return 1e3
    if typical < 50:
        return 1.0
    return 1e-3


def _read_xlsx_sheet_means(xlsx_path: Path, sheet: str) -> dict[str, float]:
    """Read one xlsx sheet, averaging across vol0..N columns per vial."""
    import pandas as pd

    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    vol_cols = [c for c in df.columns if c != "vial"]
    result = {}
    for _, row in df.iterrows():
        vals = [row[c] for c in vol_cols if pd.notna(row[c])]
        if vals:
            result[str(row["vial"]).upper()] = float(sum(vals) / len(vals))
    return result


def load_full_stats_from_xlsx(xlsx_path: Path) -> dict:
    """Read phantomkit's own full per-vial distribution stats from its xlsx.

    Returns ``{vial_upper: {mean, median, std, count, p25, p75, min, max,
    mean_mad, median_mad}}`` — the same shape
    :func:`phantomkit.pet_processing.extract_pet_vial_metrics` returns for
    the vendor side, in the same raw (un-scaled) units.
    """
    fields = (
        "mean", "median", "std", "count", "p25", "p75",
        "min", "max", "mean_mad", "median_mad",
    )
    per_field = {}
    for f in fields:
        try:
            per_field[f] = _read_xlsx_sheet_means(xlsx_path, f)
        except Exception:
            per_field[f] = {}

    vials = set()
    for d in per_field.values():
        vials |= set(d)
    return {v: {f: per_field[f].get(v) for f in fields} for v in vials}


def compute_vendor_vial_stats(
    vendor_image: Path, vial_masks: dict[str, Path], tmp_dir: Path
) -> dict:
    """Per-vial stats on vendor_image using phantomkit's existing vial masks.

    Regrids each vial mask onto vendor_image's own voxel grid
    (nearest-neighbor, matching the pattern already used in
    phantom_processor.py) rather than resampling the vendor image itself —
    avoids interpolating a continuous parametric map. Delegates the actual
    mrstats/mrdump extraction to the existing, already-generic
    extract_pet_vial_metrics().
    """
    from phantomkit.pet_processing import extract_pet_vial_metrics

    tmp_dir.mkdir(parents=True, exist_ok=True)
    regridded: dict[str, str] = {}
    for vial, mask in sorted(vial_masks.items()):
        out = tmp_dir / f"{vial}_regrid.nii.gz"
        subprocess.run(
            [
                "mrgrid", "-template", str(vendor_image), str(mask),
                "regrid", str(out), "-interp", "nearest",
                "-datatype", "bit", "-force",
            ],
            check=True, capture_output=True,
        )
        regridded[vial] = str(out)

    return extract_pet_vial_metrics(str(vendor_image), regridded)
