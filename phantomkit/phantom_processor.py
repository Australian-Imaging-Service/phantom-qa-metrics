#!/usr/bin/env python3
"""
phantom_processor.py
====================
Core phantom processing engine for the phantomkit package.

Contains ``PhantomProcessor``, a class that orchestrates a Pydra workflow:

  1. ANTs registration
  2. Vial mask inverse-transform to subject space
  3. Per-vial metric extraction from all contrast images
  4. Plot generation (per-contrast scatter plots, T1/T2 parametric maps)
  5. Forward transform of all contrasts to template space

Path conventions (shared repo):
    template_data/<phantom>/ImageTemplate.nii.gz
    template_data/<phantom>/VialsLabelled/*.nii.gz
    template_data/<phantom>/adc_reference.json
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydra.compose import python, workflow
from pydra.engine import Submitter


# =============================================================================
# Module-level helpers (called inside Pydra tasks)
# =============================================================================


def _classify_contrast(contrast_file: Path) -> Optional[str]:
    """
    Classify a contrast image by its filename stem.

    Returns
    -------
    "adc"  – filename contains 'ADC' (case-insensitive)
    "fa"   – filename contains standalone 'FA' (whole-word, case-insensitive)
    None   – no special classification
    """
    stem = contrast_file.stem
    if re.search(r"ADC", stem, re.IGNORECASE):
        return "adc"
    if re.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", stem):
        return "fa"
    return None


def _generate_mrview_screenshot(
    contrast_file: Path,
    roi_overlay: str,
    output_image: str,
    intensity_range: Optional[Tuple[float, float]] = None,
) -> Optional[str]:
    """
    Capture an mrview screenshot with a vial ROI overlay.

    Parameters
    ----------
    intensity_range:
        If provided, passes ``-intensity_range min,max`` to mrview.
        Use (0, 1) for FA maps and (0, 0.005) for ADC maps.
    """
    cmd = [
        "mrview",
        str(contrast_file),
        "-mode",
        "1",
        "-plane",
        "2",
        "-interpolation",
        "0",
        "-roi.load",
        roi_overlay,
        "-roi.colour",
        "1,0,0",
        "-roi.opacity",
        "1",
        "-comments",
        "0",
        "-noannotations",
        "-fullscreen",
    ]

    if intensity_range is not None:
        cmd.extend(
            ["-intensity_range", f"{intensity_range[0]},{intensity_range[1]}"]
        )

    cmd.extend(
        [
            "-capture.folder",
            str(Path(output_image).parent),
            "-capture.prefix",
            Path(output_image).stem,
            "-capture.grab",
            "-exit",
        ]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ⚠ mrview screenshot failed: {result.stderr}")
        return None

    # mrview appends "0000" to the prefix
    actual_file = str(
        Path(output_image).parent / f"{Path(output_image).stem}0000.png"
    )
    return actual_file if Path(actual_file).exists() else None


def _build_roi_overlay(
    contrast_file: Path,
    vial_masks_list: List[Path],
    prefix: str,
    tmp_vial_dir: Path,
) -> Optional[str]:
    """Regrid each vial mask to contrast space and combine into one overlay NIfTI."""
    roi_overlay = str(tmp_vial_dir / f"{prefix}_VialsCombined.nii.gz")
    regridded_vials = []

    for vial_mask in vial_masks_list:
        vial_name = vial_mask.name.replace(".nii.gz", "").replace(".nii", "")
        regridded = str(tmp_vial_dir / f"{prefix}_{vial_name}.nii")
        cmd = [
            "mrgrid",
            "-template",
            str(contrast_file),
            str(vial_mask),
            "regrid",
            regridded,
            "-interp", "nearest",
            "-datatype", "bit",
            "-force",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        regridded_vials.append(regridded)

    if not regridded_vials:
        return None

    cmd_cat = ["mrcat"] + regridded_vials + ["-", "-axis", "3"]
    cmd_math = ["mrmath", "-", "max", roi_overlay, "-axis", "3", "-force"]

    proc_cat = subprocess.Popen(
        cmd_cat, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc_math = subprocess.Popen(
        cmd_math,
        stdin=proc_cat.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc_cat.stdout.close()
    proc_math.communicate()

    return roi_overlay if Path(roi_overlay).exists() else None


# =============================================================================
# Pydra task definitions
# =============================================================================


@python.define(outputs=["warped", "transform", "inverse_warped"])
def _task_register(
    input_image: str,
    template_phantom: str,
    output_prefix: str,
) -> tuple[str, str, str]:
    """Run ANTs rigid registration of the input image to the template phantom."""
    cmd = [
        "antsRegistrationSyN.sh",
        "-d",
        "3",
        "-f",
        template_phantom,
        "-m",
        input_image,
        "-o",
        output_prefix,
        "-t",
        "r",
        "-n",
        "8",
        "-j",
        "1",
    ]
    print("Running ANTs registration...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ANTs failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    print("  ✓ Registration complete")
    return (
        f"{output_prefix}Warped.nii.gz",
        f"{output_prefix}0GenericAffine.mat",
        f"{output_prefix}InverseWarped.nii.gz",
    )


@python.define(outputs=["out"])
def _task_save_scanner_space_template(
    inverse_warped: str,
    output_path: str,
) -> str:
    """mrconvert the inverse-warped image to the final scanner-space template path."""
    cmd = ["mrconvert", "-quiet", inverse_warped, output_path, "-force"]
    subprocess.run(cmd, check=True, capture_output=True)
    print("  ✓ Saved template in scanner space")
    return output_path


@python.define(outputs=["vial_paths"])
def _task_transform_vials(
    vial_masks: list,
    reference_image: str,
    transform_matrix: str,
    output_vial_dir: str,
    tmp_vial_dir: str,
) -> list:
    """Inverse-transform all vial masks from template space to subject space."""
    Path(output_vial_dir).mkdir(parents=True, exist_ok=True)
    Path(tmp_vial_dir).mkdir(parents=True, exist_ok=True)

    transformed = []
    for vial_mask in vial_masks:
        vial_name = (
            Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "").split(".")[0]
        )
        tmp_vial = str(Path(tmp_vial_dir) / f"{vial_name}.nii")
        output_vial = str(Path(output_vial_dir) / f"{vial_name}.nii.gz")

        cmd = [
            "antsApplyTransforms",
            "-d",
            "3",
            "-i",
            str(vial_mask),
            "-r",
            reference_image,
            "-o",
            tmp_vial,
            "-t",
            f"[{transform_matrix}, 1]",
            "-n",
            "NearestNeighbor",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Transform failed for {vial_name}: {result.stderr}")

        cmd = ["mrconvert", "-quiet", tmp_vial, output_vial, "-force"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Final conversion failed for {vial_name}: {result.stderr}"
            )

        transformed.append(output_vial)

    print(f"  ✓ Transformed {len(transformed)} vial masks")
    return transformed


def _compute_voxel_stats(
    vol_file: str, mask_file: str
) -> dict:
    """Dump masked voxels and compute p25/p75/mean_mad/median_mad in one mrdump call."""
    import numpy as _np

    nan = float("nan")
    result = {"p25": nan, "p75": nan, "mean_mad": nan, "median_mad": nan}
    try:
        r = subprocess.run(
            ["mrdump", vol_file, "-mask", mask_file],
            capture_output=True, text=True, check=True,
        )
        raw = r.stdout.strip().split()
        if not raw:
            return result
        vals = _np.array([float(x) for x in raw], dtype=float)
        result["p25"] = float(_np.percentile(vals, 25))
        result["p75"] = float(_np.percentile(vals, 75))
        result["mean_mad"] = float(_np.mean(_np.abs(vals - vals.mean())))
        result["median_mad"] = float(_np.median(_np.abs(vals - _np.median(vals))))
    except Exception:
        pass
    return result


@python.define(outputs=["sentinel"])
def _task_extract_metrics(
    contrast_files: list,
    vial_paths: list,
    adc_vials: list,
    output_metrics_dir: str,
    session_name: str,
    tmp_vols_dir: str,
) -> str:
    """Extract per-vial statistics (mean/median/std/min/max) from all contrast images."""
    adc_vials_set = set(v.upper() for v in adc_vials)
    metrics_dir = Path(output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    Path(tmp_vols_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nStep 3: Extracting metrics from all contrasts")
    print(f"  Found {len(contrast_files)} contrast image(s)")

    for contrast_file_str in contrast_files:
        contrast_file = Path(contrast_file_str)
        contrast_name = contrast_file.name
        for ext in (".nii.gz", ".nii"):
            if contrast_name.endswith(ext):
                contrast_name = contrast_name[: -len(ext)]
                break
        clean_contrast_name = contrast_name

        # Classify contrast and filter vials for ADC
        stem = contrast_file.stem
        contrast_type = None
        if re.search(r"ADC", stem, re.IGNORECASE):
            contrast_type = "adc"
        elif re.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", stem):
            contrast_type = "fa"

        active_vials = vial_paths
        if contrast_type == "adc":
            original_count = len(vial_paths)
            active_vials = [
                m
                for m in vial_paths
                if Path(m)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
                .upper()
                in adc_vials_set
            ]
            print(
                f"  [ADC mode] Restricting to {len(adc_vials_set)} ADC vials "
                f"({len(active_vials)} of {original_count})"
            )

        # Get number of volumes
        result = subprocess.run(
            ["mrinfo", "-size", str(contrast_file)], capture_output=True, text=True
        )
        size_info = result.stdout.strip().split()
        nvols = (
            int(size_info[3]) if len(size_info) >= 4 and int(size_info[3]) > 0 else 1
        )

        print(
            f"  Processing {clean_contrast_name} "
            f"({nvols} volume{'s' if nvols > 1 else ''})"
        )

        metrics_data: Dict[str, Dict[str, List[float]]] = {
            "mean": {}, "median": {}, "std": {}, "min": {}, "max": {},
            "count": {}, "p25": {}, "p75": {}, "mean_mad": {}, "median_mad": {},
        }

        for vial_mask in active_vials:
            vial_name = (
                Path(vial_mask)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
            )
            for metric in metrics_data:
                metrics_data[metric][vial_name] = []

            regridded_mask = str(
                Path(tmp_vols_dir) / f"{contrast_name}_{vial_name}.nii"
            )
            cmd = [
                "mrgrid",
                "-template",
                str(contrast_file),
                vial_mask,
                "regrid",
                regridded_mask,
                "-interp", "nearest",
                "-datatype", "bit",
                "-force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"mrgrid regrid failed for vial {vial_name}: {result.stderr}"
                )

            for vol_idx in range(nvols):
                if nvols == 1:
                    vol_file = str(contrast_file)
                else:
                    vol_file = str(
                        Path(tmp_vols_dir) / f"{contrast_name}_vol{vol_idx}.nii.gz"
                    )
                    cmd = [
                        "mrconvert",
                        str(contrast_file),
                        "-coord",
                        "3",
                        str(vol_idx),
                        vol_file,
                        "-quiet",
                        "-force",
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)

                cmd = [
                    "mrstats", "-quiet", vol_file,
                    "-output", "mean", "-output", "median", "-output", "std",
                    "-output", "min", "-output", "max", "-output", "count",
                    "-mask", regridded_mask,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0 or not result.stdout.strip():
                    raise RuntimeError(
                        f"mrstats failed for vial {vial_name}: {result.stderr}"
                    )
                values = result.stdout.strip().split()
                metrics_data["mean"][vial_name].append(float(values[0]))
                metrics_data["median"][vial_name].append(float(values[1]))
                metrics_data["std"][vial_name].append(float(values[2]))
                metrics_data["min"][vial_name].append(float(values[3]))
                metrics_data["max"][vial_name].append(float(values[4]))
                metrics_data["count"][vial_name].append(float(values[5]))
                _vstats = _compute_voxel_stats(vol_file, regridded_mask)
                metrics_data["p25"][vial_name].append(_vstats["p25"])
                metrics_data["p75"][vial_name].append(_vstats["p75"])
                metrics_data["mean_mad"][vial_name].append(_vstats["mean_mad"])
                metrics_data["median_mad"][vial_name].append(_vstats["median_mad"])

        xlsx_dir = metrics_dir / "xlsx"
        xlsx_dir.mkdir(parents=True, exist_ok=True)
        xlsx_file = xlsx_dir / f"{clean_contrast_name}.xlsx"
        sheet_order = ["mean", "median", "std", "min", "max", "count", "p25", "p75", "mean_mad", "median_mad"]
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            for metric_name in sheet_order:
                vial_data = metrics_data[metric_name]
                rows = [
                    {"vial": vn, **{f"vol{i}": v for i, v in enumerate(vals)}}
                    for vn, vals in vial_data.items()
                ]
                pd.DataFrame(rows).to_excel(writer, sheet_name=metric_name, index=False)
        print(f"    Saved: {xlsx_file.name}")

    return output_metrics_dir  # sentinel for downstream ordering


@python.define(outputs=["sentinel"])
def _task_generate_plots(
    contrast_files: list,
    metrics_dir: str,
    vial_dir: str,
    session_name: str,
    phantom_name: str,
    template_dir: str,
    metrics_sentinel: str,  # enforces Step 3 → Step 4 ordering; not used in body
    output_format: str = "html",
) -> str:
    """Generate per-contrast scatter plots and parametric map plots (IR / TE).

    Parameters
    ----------
    output_format : str
        "html" (default) — interactive HTML; NIfTI paths passed directly.
        "png"            — static matplotlib PNG; mrview used for ROI overlay.
    """
    from phantomkit.plotting.vial_intensity import plot_vial_intensity
    from phantomkit.plotting.maps_ir import plot_vial_ir_means_std
    from phantomkit.plotting.maps_te import plot_vial_te_means_std

    metrics_path = Path(metrics_dir)
    xlsx_dir = metrics_path / "xlsx"
    plots_dir = metrics_path / "plots"
    fits_dir = metrics_path / "fits"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    vial_dir_path = Path(vial_dir)
    tmp_vial_dir = vial_dir_path / "tmp"
    tmp_vial_dir.mkdir(exist_ok=True)
    vial_masks_list = list(vial_dir_path.glob("*.nii.gz"))
    contrast_file_paths = [Path(f) for f in contrast_files]
    ext = ".html" if output_format == "html" else ".png"

    def _matches(stem: str, token: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){token}(?![a-z])", stem.lower()))

    # Prefer a T1/MPRAGE image as the viewer background (same subject space as
    # vials); fall back to the first matching contrast if none is found.
    _t1_bg = next(
        (str(f) for f in contrast_file_paths
         if re.search(r"t1|mprage", f.stem, re.IGNORECASE)),
        None,
    )
    _vial_niftis_map = {
        v.name.replace(".nii.gz", "").replace(".nii", ""): str(v)
        for v in vial_masks_list
    }

    # Load T1/T2 reference values if available
    _relaxometry_ref = None
    if template_dir and phantom_name:
        _ref_path = Path(template_dir) / phantom_name / "t1t2_reference.json"
        if _ref_path.exists():
            with open(_ref_path) as _f:
                _relaxometry_ref = json.load(_f)

    print("\nStep 4: Generating plots")

    # ── Per-contrast scatter plots ────────────────────────────────────────────
    for contrast_file in contrast_file_paths:
        is_ir_or_te = (
            _matches(contrast_file.stem, "ir") or _matches(contrast_file.stem, "te")
        )
        contrast_name = contrast_file.name
        for _ext in (".nii.gz", ".nii"):
            if contrast_name.endswith(_ext):
                contrast_name = contrast_name[: -len(_ext)]
                break

        xlsx_file = xlsx_dir / f"{contrast_name}.xlsx"

        if not xlsx_file.exists():
            print(f"  ⚠ xlsx not found, skipping plot for {contrast_name}")
            continue

        # IR/TE individual contrasts always use PNG — the T1/T2 mapping HTML
        # covers the full fitted plots; per-TI/TE scatter PNGs save disk space.
        file_ext = ".png" if is_ir_or_te else ext
        output_plot = str(plots_dir / f"{contrast_name}{file_ext}")

        if output_format == "html" and not is_ir_or_te:
            try:
                plot_vial_intensity(
                    csv_file=str(xlsx_file),
                    plot_type="scatter",
                    roi_image=None,
                    output=output_plot,
                    phantom=phantom_name,
                    template_dir=template_dir,
                    output_format="html",
                    nifti_image=str(contrast_file),
                    vial_niftis=_vial_niftis_map or None,
                )
                print(f"    ✓ Generated HTML plot: {Path(output_plot).name}")
            except Exception as e:
                print(f"    ✗ HTML plot generation failed for {contrast_name}: {e}")
        else:
            roi_image = None
            if vial_masks_list and not is_ir_or_te:
                roi_overlay = _build_roi_overlay(
                    contrast_file, vial_masks_list, contrast_name, tmp_vial_dir
                )
                if roi_overlay:
                    contrast_type = _classify_contrast(contrast_file)
                    intensity_range = (
                        (0, 1)
                        if contrast_type == "fa"
                        else (0, 0.005)
                        if contrast_type == "adc"
                        else None
                    )
                    screenshot_base = str(
                        tmp_vial_dir / f"{contrast_name}_roi_overlay.png"
                    )
                    actual_screenshot = _generate_mrview_screenshot(
                        contrast_file,
                        roi_overlay,
                        screenshot_base,
                        intensity_range=intensity_range,
                    )
                    if actual_screenshot and Path(actual_screenshot).exists():
                        roi_image = actual_screenshot

            try:
                plot_vial_intensity(
                    csv_file=str(xlsx_file),
                    plot_type="scatter",
                    roi_image=roi_image,
                    output=output_plot,
                    phantom=phantom_name,
                    template_dir=template_dir,
                    output_format="png",
                )
                print(f"    ✓ Generated PNG plot: {Path(output_plot).name}")
            except Exception as e:
                print(f"    ✗ PNG plot generation failed for {contrast_name}: {e}")

    # ── Parametric map plots (IR and TE) ──────────────────────────────────────
    _mapping_names = {
        "ir": ("T1_mapping", "T1_fits"),
        "te": ("T2_mapping", "T2_fits"),
    }

    for contrast_type_key, plot_fn in [
        ("ir", plot_vial_ir_means_std),
        ("te", plot_vial_te_means_std),
    ]:
        matching = [
            f for f in contrast_file_paths if _matches(f.stem, contrast_type_key)
        ]
        if not matching:
            continue

        print(
            f"  Found {len(matching)} {contrast_type_key.upper()} contrasts: "
            f"{[f.name for f in matching]}"
        )

        _plot_name, _fits_name = _mapping_names[contrast_type_key]
        output_plot = str(plots_dir / f"{_plot_name}{ext}")
        _fits_output = str(fits_dir / f"{_fits_name}.csv")
        first_file = matching[0]

        if output_format == "html":
            try:
                plot_fn(
                    contrast_files=[str(f) for f in matching],
                    metric_dir=str(xlsx_dir),
                    output_file=output_plot,
                    roi_image=None,
                    output_format="html",
                    fits_output=_fits_output,
                    nifti_image=_t1_bg or str(first_file),
                    vial_niftis=_vial_niftis_map or None,
                    relaxometry_reference=_relaxometry_ref,
                    phantom=phantom_name,
                    overlay_contrast=str(first_file) if _t1_bg else None,
                )
                print(f"    ✓ Generated {contrast_type_key.upper()} HTML map plot")
            except Exception as e:
                print(f"    ✗ {contrast_type_key.upper()} HTML map plot failed: {e}")
        else:
            roi_image_arg = None
            if vial_masks_list:
                overlay_file = _build_roi_overlay(
                    first_file, vial_masks_list, contrast_type_key, tmp_vial_dir
                )
                if overlay_file:
                    screenshot_base = str(
                        tmp_vial_dir / f"roi_overlay_{contrast_type_key}.png"
                    )
                    actual_screenshot = _generate_mrview_screenshot(
                        first_file, overlay_file, screenshot_base
                    )
                    if actual_screenshot and Path(actual_screenshot).exists():
                        roi_image_arg = actual_screenshot

            try:
                plot_fn(
                    contrast_files=[str(f) for f in matching],
                    metric_dir=str(xlsx_dir),
                    output_file=output_plot,
                    roi_image=roi_image_arg,
                    output_format="png",
                    fits_output=_fits_output,
                )
                print(f"    ✓ Generated {contrast_type_key.upper()} PNG map plot")
            except Exception as e:
                print(f"    ✗ {contrast_type_key.upper()} PNG map plot failed: {e}")

    return metrics_dir  # sentinel for downstream ordering


@python.define(outputs=["sentinel"])
def _task_transform_contrasts(
    contrast_files: list,
    transform_matrix: str,
    template_phantom: str,
    output_dir: str,
    tmp_dir: str,
) -> str:
    """Forward-transform all contrast images from subject space to template space."""
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nStep 5: Transforming all contrasts to template space")

    for contrast_file_str in contrast_files:
        contrast_file = Path(contrast_file_str)
        contrast_name = contrast_file.stem.replace(".nii", "")
        source_image = str(contrast_file)

        # Detect 4D and single-slice
        detect = subprocess.run(
            ["mrinfo", "-size", source_image], capture_output=True, text=True
        )
        size_parts = detect.stdout.strip().split()
        is_4d = len(size_parts) >= 4 and int(size_parts[3]) > 1
        is_single_slice = len(size_parts) >= 3 and int(size_parts[2]) == 1

        if is_single_slice:
            padded = str(Path(tmp_dir) / f"{contrast_name}_padded.nii.gz")
            cmd = [
                "mrgrid",
                source_image,
                "pad",
                "-axis",
                "2",
                "1,1",
                padded,
                "-force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Padding of single-slice contrast {contrast_name} failed: "
                    f"{result.stderr}"
                )
            transform_input = padded
        else:
            transform_input = source_image

        warped_tmp = str(Path(tmp_dir) / f"{contrast_name}_template_space_tmp.nii.gz")
        warped_contrast = str(Path(tmp_dir) / f"{contrast_name}_template_space.nii.gz")

        cmd = [
            "antsApplyTransforms",
            "-d",
            "3",
            "-e",
            "3" if is_4d else "0",
            "-i",
            transform_input,
            "-r",
            template_phantom,
            "-o",
            warped_tmp,
            "-t",
            transform_matrix,
            "-n",
            "Linear",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Forward transform of contrast {contrast_name} failed: {result.stderr}"
            )

        verify = subprocess.run(
            ["mrinfo", "-size", warped_tmp], capture_output=True, text=True
        )
        if verify.returncode != 0 or not verify.stdout.strip():
            raise RuntimeError(
                f"antsApplyTransforms produced no valid output for {contrast_name}. "
                f"stderr: {result.stderr}"
            )

        shutil.move(warped_tmp, warped_contrast)
        shutil.copy2(warped_contrast, str(Path(output_dir) / contrast_file.name))
        print(f"    ✓ {contrast_file.name} → template space")

    print(f"  ✓ All contrasts saved to: {output_dir}")
    return output_dir  # sentinel for downstream ordering


@python.define(outputs=["out"])
def _task_cleanup(
    dirs_to_remove: list,
    sentinel_plots: str,     # enforces Steps 3+4 → Step 6 ordering; not used in body
    sentinel_template: str,  # enforces Step 5 → Step 6 ordering; not used in body
) -> str:
    """Remove temporary directories once all processing is complete."""
    print("\nStep 6: Cleaning up temporary directories")
    for d in dirs_to_remove:
        p = Path(d)
        if p.exists():
            try:
                shutil.rmtree(p)
                print(f"  ✓ Removed: {p.name}")
            except Exception as e:
                print(f"  ⚠ Warning: Could not remove {p.name}: {e}")
    return "done"


# =============================================================================
# Module-level Pydra workflow  (explicit parameters — no closure capture)
# =============================================================================


@workflow.define(outputs=["out"])
def PhantomSessionWf(
    input_image: str,
    template_phantom: str,
    vial_masks: list,
    adc_vials: list,
    output_prefix: str,
    output_dir_str: str,
    session_name: str,
    phantom_name: str,
    template_dir_parent: str,
    contrast_files: list,
    output_format: str = "html",
) -> str:
    """
    End-to-end phantom QC workflow.

    All inputs are passed explicitly (no closure) to avoid Pydra global
    registry collisions when the workflow is instantiated multiple times
    within the same Python process (e.g. Stage 2 + Stage 3 in parallel).
    """
    from pathlib import Path as _Path

    _output_dir = _Path(output_dir_str)
    _tmp_dir = _output_dir / "tmp"
    _vial_dir = _output_dir / "vial_segmentations"
    _metrics_dir = _output_dir / "metrics"
    _images_dir = _output_dir / "images_template_space"

    # Step 1: ANTs registration
    reg = workflow.add(
        _task_register(
            input_image=input_image,
            template_phantom=template_phantom,
            output_prefix=output_prefix,
        ),
        name="registration",
    )

    # Step 1b: Save template in scanner space (depends on reg via data)
    workflow.add(
        _task_save_scanner_space_template(
            inverse_warped=reg.inverse_warped,
            output_path=str(_output_dir / "TemplatePhantom_ScannerSpace.nii.gz"),
        ),
        name="save_scanner_space_template",
    )

    # Step 2: Transform vials to subject space (depends on reg via data)
    vials = workflow.add(
        _task_transform_vials(
            vial_masks=vial_masks,
            reference_image=input_image,
            transform_matrix=reg.transform,
            output_vial_dir=str(_vial_dir),
            tmp_vial_dir=str(_output_dir / "tmp_vials"),
        ),
        name="transform_vials",
    )

    # Step 3: Extract metrics (depends on vials via data)
    metrics = workflow.add(
        _task_extract_metrics(
            contrast_files=contrast_files,
            vial_paths=vials.vial_paths,
            adc_vials=adc_vials,
            output_metrics_dir=str(_metrics_dir),
            session_name=session_name,
            tmp_vols_dir=str(_output_dir / "tmp_vols"),
        ),
        name="extract_metrics",
    )

    # Step 4: Generate plots (depends on metrics via sentinel)
    plots = workflow.add(
        _task_generate_plots(
            contrast_files=contrast_files,
            metrics_dir=str(_metrics_dir),
            vial_dir=str(_vial_dir),
            session_name=session_name,
            phantom_name=phantom_name,
            template_dir=template_dir_parent,
            metrics_sentinel=metrics.sentinel,
            output_format=output_format,
        ),
        name="generate_plots",
    )

    # Step 5: Forward-transform contrasts to template space
    # (depends on reg via data; runs in parallel with Steps 2–4)
    template_contrasts = workflow.add(
        _task_transform_contrasts(
            contrast_files=contrast_files,
            transform_matrix=reg.transform,
            template_phantom=template_phantom,
            output_dir=str(_images_dir),
            tmp_dir=str(_tmp_dir / "template_space_contrasts"),
        ),
        name="transform_contrasts",
    )

    # Step 6: Cleanup (depends on Steps 4+5 via sentinels)
    cleanup = workflow.add(
        _task_cleanup(
            dirs_to_remove=[
                str(_tmp_dir),
                str(_output_dir / "tmp_vials"),
                str(_output_dir / "tmp_vols"),
                str(_vial_dir / "tmp"),
            ],
            sentinel_plots=plots.sentinel,
            sentinel_template=template_contrasts.sentinel,
        ),
        name="cleanup",
    )

    return cleanup.out


# =============================================================================
# PhantomProcessor
# =============================================================================


class PhantomProcessor:
    """
    Orchestrate phantom QC processing for a single session.

    Parameters
    ----------
    template_dir:
        Directory for this phantom type, e.g.
        ``<repo>/template_data/SPIRIT``.
        Must contain ``ImageTemplate.nii.gz`` and ``VialsLabelled/``.
    output_base_dir:
        Top-level output directory.  Results are written to a
        ``<session_name>/`` subdirectory within this path.
    """

    def __init__(
        self,
        template_dir: str,
        output_base_dir: str,
        output_format: str = "html",
    ):
        self.template_dir = Path(template_dir)
        self.output_base_dir = Path(output_base_dir)
        self.output_format = output_format

        # Phantom name is the last component of template_dir (e.g. "SPIRIT")
        self.phantom_name = self.template_dir.name

        self.template_phantom = self.template_dir / "ImageTemplate.nii.gz"
        self.vial_dir = self.template_dir / "VialsLabelled"
        self.vial_masks = sorted(self.vial_dir.glob("*.nii.gz"))

        # Load ADC vials from adc_reference.json
        adc_ref_path = self.template_dir / "adc_reference.json"
        if not adc_ref_path.exists():
            raise FileNotFoundError(f"adc_reference.json not found: {adc_ref_path}")
        with open(adc_ref_path) as fh:
            adc_ref = json.load(fh)
        if "vials" not in adc_ref:
            raise KeyError(f"'vials' key missing from: {adc_ref_path}")
        self.adc_vials = {v.upper() for v in adc_ref["vials"]}

        if not self.template_phantom.exists():
            raise FileNotFoundError(f"Template not found: {self.template_phantom}")
        if len(self.vial_masks) == 0:
            raise FileNotFoundError(f"No vial masks found in: {self.vial_dir}")

    def process_session(self, input_image: str) -> Dict:
        """
        Process a single phantom session end-to-end using a Pydra workflow.

        The workflow enforces the following sequence:
          1. ANTs registration
          1b. Save template in scanner space          (→ depends on 1 via data)
          2. Transform vials to subject space         (→ depends on 1 via data)
          3. Extract per-vial metrics                 (→ depends on 2 via data)
          4. Generate plots                           (→ depends on 3 via sentinel)
          5. Forward-transform contrasts to template  (→ depends on 1 via data;
                                                          runs in parallel with 2-4)
          6. Cleanup temp directories                 (→ depends on 4+5 via sentinels)

        Parameters
        ----------
        input_image:
            Path to the primary input image (T1 MPRAGE or T1 in DWI space).

        Returns
        -------
        dict
            Output paths for the processed session.
        """
        input_path = Path(input_image)
        session_name = input_path.parent.name

        output_dir = self.output_base_dir / session_name
        tmp_dir = output_dir / "tmp"
        vial_dir = output_dir / "vial_segmentations"
        metrics_dir = output_dir / "metrics"
        images_template_space_dir = output_dir / "images_template_space"

        for d in [tmp_dir, vial_dir, metrics_dir, images_template_space_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing Session: {session_name}")
        print(f"Input: {input_image}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}\n")

        # Gather contrast files before starting the workflow (deterministic glob)
        contrast_files = sorted(
            str(f)
            for f in input_path.parent.glob("*.nii.gz")
            if f.name != "TemplatePhantom_ScannerSpace.nii.gz"
        )

        wf = PhantomSessionWf(
            input_image=input_image,
            template_phantom=str(self.template_phantom),
            vial_masks=[str(m) for m in self.vial_masks],
            adc_vials=sorted(self.adc_vials),
            output_prefix=str(tmp_dir / f"{session_name}_Transformed_"),
            output_dir_str=str(output_dir),
            session_name=session_name,
            phantom_name=self.phantom_name,
            template_dir_parent=str(self.template_dir.parent),
            contrast_files=contrast_files,
            output_format=self.output_format,
        )
        cache_dir = str(output_dir / ".pydra_cache")
        sub = Submitter(worker="cf", cache_root=cache_dir)
        try:
            sub(wf, rerun=True)
        finally:
            # Cancel any pending asyncio tasks before closing the event loop.
            # Pydra's ConcurrentFuturesWorker leaves tasks pending after the
            # workflow completes; without this, Python 3.12 logs
            # "Task was destroyed but it is pending!" for each one.
            import asyncio as _asyncio
            _loop = sub.loop
            if _loop and not _loop.is_closed():
                _pending = _asyncio.all_tasks(_loop)
                if _pending:
                    for _t in _pending:
                        _t.cancel()
                    _loop.run_until_complete(
                        _asyncio.gather(*_pending, return_exceptions=True)
                    )
            sub.close()

        self._run_temperature_analysis(metrics_dir, session_name)

        print(f"\n{'=' * 60}")
        print(f"✓ Session {session_name} complete!")
        print(f"  Metrics:               {metrics_dir}")
        print(f"  Vial masks:            {vial_dir}")
        print(f"  Template-space images: {images_template_space_dir}")
        print(f"{'=' * 60}\n")

        return {
            "session": session_name,
            "output_dir": str(output_dir),
            "metrics_dir": str(metrics_dir),
            "vial_dir": str(vial_dir),
            "images_template_space_dir": str(images_template_space_dir),
            "space_image": str(output_dir / "TemplatePhantom_ScannerSpace.nii.gz"),
        }

    def _run_temperature_analysis(self, metrics_dir: Path, session_name: str) -> None:
        """Estimate phantom temperature from ADC metrics when calibration data is available.

        Writes an interactive HTML report to ``metrics_dir/plots/``.
        Silently skips if ADC metrics, calibration LUT, or phantom config are absent.
        """
        adc_xlsx = metrics_dir / "xlsx" / "ADC.xlsx"
        if not adc_xlsx.exists():
            return

        calib_lut = self.template_dir.parent / "DIFFUSION-O-3574_Calibration_GSP_PVP_20220331.xlsx"
        phantom_cfg_path = self.template_dir.parent / "phantom_config.json"

        if not calib_lut.exists() or not phantom_cfg_path.exists():
            print("  Temperature estimation skipped: calibration LUT or phantom config not found.")
            return

        print(f"\n  Running temperature estimation for {session_name}...")
        try:
            from phantomkit.plotting.calibration_plotter import (
                parse_calibration_xlsx,
                load_phantom_config,
                build_vial_map,
                estimate_temperature,
                build_vials_html,
            )

            formulations = parse_calibration_xlsx(str(calib_lut))
            _adc_df = pd.read_excel(adc_xlsx, sheet_name="mean")
            vials_adc = {
                str(row.iloc[0]).strip(): float(row.iloc[1])
                for _, row in _adc_df.iterrows()
            }
            phantom_cfg = load_phantom_config(str(phantom_cfg_path))
            vial_map = build_vial_map(phantom_cfg, self.phantom_name)

            results = []
            for vial, adc_raw in vials_adc.items():
                form_query = vial_map.get(vial)
                if form_query is None:
                    continue
                try:
                    res = estimate_temperature(
                        formulations, str(form_query), D=adc_raw, vial=vial
                    )
                    results.append(res)
                except ValueError:
                    pass

            if not results:
                print("  Temperature estimation: no matching vials found.")
                return

            plots_dir = metrics_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            output_html = plots_dir / f"{self.phantom_name}_temperature_estimates.html"
            output_html.write_text(
                build_vials_html(formulations, results, phantom_name=self.phantom_name),
                encoding="utf-8",
            )
            print(f"  ✓ Temperature estimates: {output_html.name}")
        except Exception as e:
            print(f"  Temperature estimation failed: {e}")
