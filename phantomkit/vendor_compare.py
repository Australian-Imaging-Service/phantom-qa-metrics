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
    return {"report_html": report_html, "vial_masks": vial_masks}


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
