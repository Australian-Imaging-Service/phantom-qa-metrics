#!/usr/bin/env python3
"""
pipeline.py
===========
End-to-end orchestrator for the phantomkit phantom QC pipeline.

Stages
------
  Stage 1  DWI processing  (phantomkit.dwi_processing)
           Runs if DWI acquisitions are found in --input-dir.
           Outputs per DWI series:
             T1_in_DWI_space.nii.gz, ADC.nii.gz, FA.nii.gz,
             DWI_preproc_biascorr.mif.gz

  Stage 2  Phantom QC in DWI space  (phantomkit.phantom_processor)
           Runs once per DWI series produced by Stage 1.
           Input image: T1_in_DWI_space.nii.gz from Stage 1.
           ADC and FA files in the same folder are picked up automatically.
           Temperature estimation runs automatically after this stage when
           ADC metrics are available.

  Stage 3  Phantom QC on native contrasts  (phantomkit.phantom_processor)
           Runs if the input directory contains IR and/or TE series.
           All contrast DICOMs (T1, IR, TE) are converted to NIfTI into a
           temporary staging folder; the phantom processor then registers
           the T1 and extracts vial metrics from every NIfTI present.
           Temperature estimation runs automatically if ADC is present.

Usage
-----
  phantomkit pipeline \\
    --input-dir  /path/to/patient/scans \\
    --output-dir /path/to/outputs \\
    --phantom    SPIRIT

  The input directory may contain series sub-directories holding DICOM files,
  NIfTI files (.nii / .nii.gz), or MRtrix MIF files (.mif / .mif.gz).
  Format detection is automatic per sub-directory.

  Optional flags:
    --denoise-degibbs   pass to DWI pipeline
    --gradcheck         pass to DWI pipeline
    --nocleanup         keep DWI tmp/ directories
    --readout-time      override TotalReadoutTime (seconds)
    --eddy-options      override FSL eddy options string
    --dry-run           plan and print; do not execute

Path conventions (shared repo)
-------------------------------
  template_data/<phantom>/ImageTemplate.nii.gz
  template_data/<phantom>/VialsLabelled/*.nii.gz
  template_data/phantom_config.json
  template_data/DIFFUSION-O-3574_Calibration_GSP_PVP_20220331.xlsx
  template_data/RELAXOMETRY - O-41770_GoldStandPhant_GSP_T1T2_20230125.xlsx
  template_data/rotations.txt

The template_data/ directory is resolved relative to the installed
package's location (``phantomkit/`` directory → ``../template_data/``).
"""

import argparse
import os
import re
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Path resolution
#
# template_data/ sits at <repo_root>/template_data/ — one level above the
# phantomkit/ package directory.
# ---------------------------------------------------------------------------


def _find_template_data_root() -> Path:
    """
    Locate the template_data/ directory relative to this file.
    Walks up from phantomkit/ to the repo root.
    """
    candidate = Path(__file__).resolve().parent  # phantomkit/
    for _ in range(3):
        td = candidate / "template_data"
        if td.is_dir():
            return td
        candidate = candidate.parent
    raise RuntimeError(
        f"Could not locate template_data/ within 3 levels of {Path(__file__).resolve()}"
    )


TEMPLATE_DATA_ROOT = _find_template_data_root()


CALIBRATION_LUT = TEMPLATE_DATA_ROOT / "DIFFUSION-O-3574_Calibration_GSP_PVP_20220331.xlsx"
PHANTOM_CONFIG = TEMPLATE_DATA_ROOT / "phantom_config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_header(title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def run_cmd(cmd: list, label: str):
    """Run a subprocess command, streaming output and raising on failure."""
    print(f"  >> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}).")


def _detect_series_format(series_dir: Path) -> str:
    """Return 'nifti', 'mif', or 'dicom' based on file contents of series_dir."""
    if any(series_dir.glob("*.nii.gz")) or any(series_dir.glob("*.nii")):
        return "nifti"
    if any(series_dir.glob("*.mif.gz")) or any(series_dir.glob("*.mif")):
        return "mif"
    return "dicom"


_FLAT_IMAGE_EXTS = (".nii.gz", ".nii", ".mif.gz", ".mif")
_SIDECAR_EXTS = (".bvec", ".bval", ".json")


def _wrap_flat_inputs(input_dir: Path, output_dir: Path) -> Path:
    """
    If input_dir contains flat NIfTI/MIF files rather than subdirectories,
    stage them into per-stem subdirectories under output_dir/_staged_input/
    so scan_input_dir can classify them by name.  Existing subdirectories
    and sidecar files (.bvec/.bval/.json) are also linked in.
    Returns the staged path, or input_dir unchanged if no flat images found.
    """
    flat_files = [
        f
        for f in sorted(input_dir.iterdir())
        if f.is_file() and any(f.name.endswith(ext) for ext in _FLAT_IMAGE_EXTS)
    ]
    if not flat_files:
        return input_dir

    staged = output_dir / "_staged_input"
    staged.mkdir(parents=True, exist_ok=True)

    for d in sorted(input_dir.iterdir()):
        if d.is_dir():
            link = staged / d.name
            if not link.exists():
                link.symlink_to(d)

    for img in flat_files:
        stem = img.name
        for ext in sorted(_FLAT_IMAGE_EXTS, key=len, reverse=True):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        sub = staged / stem
        sub.mkdir(exist_ok=True)
        img_link = sub / img.name
        if not img_link.exists():
            img_link.symlink_to(img)
        for sc_ext in _SIDECAR_EXTS:
            sc = input_dir / (stem + sc_ext)
            if sc.exists():
                sc_link = sub / sc.name
                if not sc_link.exists():
                    sc_link.symlink_to(sc)

    return staged


def stage_series_dir(series_dir: Path, out_dir: Path) -> list:
    """
    Stage a series directory into out_dir as .nii.gz files.
    Handles DICOM (dcm2niix), NIfTI (copy), and MIF (mrconvert).
    Returns list of produced .nii.gz paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _detect_series_format(series_dir)

    if fmt == "nifti":
        produced = []
        for nii in sorted(series_dir.glob("*.nii.gz")):
            dst = out_dir / nii.name
            shutil.copy2(nii, dst)
            produced.append(str(dst))
        for nii in sorted(series_dir.glob("*.nii")):
            dst = out_dir / (nii.stem + ".nii.gz")
            subprocess.run(
                ["mrconvert", str(nii), str(dst)], check=True, capture_output=True
            )
            produced.append(str(dst))
        if not produced:
            raise FileNotFoundError(f"No NIfTI files found in {series_dir}")
        return produced

    if fmt == "mif":
        produced = []
        mifs = sorted(series_dir.glob("*.mif.gz")) + sorted(series_dir.glob("*.mif"))
        for mif in mifs:
            stem = mif.name
            for ext in (".mif.gz", ".mif"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            dst = out_dir / f"{stem}.nii.gz"
            subprocess.run(
                ["mrconvert", str(mif), str(dst)], check=True, capture_output=True
            )
            produced.append(str(dst))
        if not produced:
            raise FileNotFoundError(f"No MIF files found in {series_dir}")
        return produced

    # DICOM fallback
    subprocess.run(
        ["dcm2niix", "-o", str(out_dir), "-f", "%p", "-z", "y", str(series_dir)],
        check=True,
        capture_output=True,
    )
    niis = sorted(out_dir.glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"dcm2niix produced no NIfTI from {series_dir}")
    return [str(p) for p in niis]


def scan_input_dir(input_dir: Path) -> dict:
    """
    Classify subdirectories in input_dir into:
      t1_dirs      – folders matching 't1' (case-insensitive)
      ir_dirs      – folders matching 'se_ir' or standalone 'ir'
      te_dirs      – folders matching 't2_se' or standalone 'te'
      dwi_dirs     – folders matching '_diff_' or '_DWI_' (case-insensitive)
                     AND not ending in _ADC or _FA
      other_dirs   – everything else
    """
    t1_dirs, ir_dirs, te_dirs, dwi_dirs, other_dirs = [], [], [], [], []

    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            other_dirs.append(d)
            continue

        if re.search(r"t1", name, re.IGNORECASE):
            t1_dirs.append(d)
        elif re.search(r"se_ir|(?<![a-z0-9])ir(?![a-z0-9])", name, re.IGNORECASE):
            ir_dirs.append(d)
        elif re.search(r"t2_se|(?<![a-z0-9])te(?![a-z0-9])", name, re.IGNORECASE):
            te_dirs.append(d)
        elif re.search(r"(_diff_|_DWI_)", name, re.IGNORECASE):
            dwi_dirs.append(d)
        else:
            other_dirs.append(d)

    return {
        "t1_dirs": t1_dirs,
        "ir_dirs": ir_dirs,
        "te_dirs": te_dirs,
        "dwi_dirs": dwi_dirs,
        "other_dirs": other_dirs,
        "has_dwi": bool(dwi_dirs),
        "has_native_contrasts": bool(ir_dirs or te_dirs),
    }


def derive_session_name(input_dir: Path) -> str:
    """
    Derive a clean session name from the input directory.
    Strips any leading series-number prefix (e.g. '87-PatientID' → 'PatientID').
    """
    name = input_dir.name
    stripped = re.sub(r"^\d+-", "", name)
    return stripped if stripped else name


# ---------------------------------------------------------------------------
# Stage 1: DWI processing
# ---------------------------------------------------------------------------


def run_stage1(input_dir: Path, output_dir: Path, cfg: dict, dry_run: bool) -> list:
    """
    Run DWI processing on input_dir.
    Returns list of DWI output subdirectory Paths (one per processed series).
    """
    print_header("STAGE 1 — DWI Processing")

    if dry_run:
        print("  [DRY RUN] Would run DWI processing pipeline.\n")
        return []

    from phantomkit.dwi_processing import run_pipeline as dwi_run_pipeline

    dwi_cfg: dict = {
        "scans_dir": str(input_dir),
        "output_dir": str(output_dir),
        "denoise_degibbs": cfg.get("denoise_degibbs", False),
        "gradcheck": cfg.get("gradcheck", False),
        "keep_tmp": cfg.get("nocleanup", False),
    }
    if cfg.get("readout_time") is not None:
        dwi_cfg["readout_time"] = cfg["readout_time"]
    if cfg.get("eddy_options") is not None:
        dwi_cfg["eddy_options"] = cfg["eddy_options"]

    dwi_run_pipeline(dwi_cfg)

    _t1_names = {"T1_in_DWI_space.nii.gz", "T1.nii.gz"}
    dwi_output_dirs = [
        d
        for d in sorted(output_dir.iterdir())
        if d.is_dir() and any((d / n).exists() for n in _t1_names)
    ]

    print(f"\n  Stage 1 complete. Found {len(dwi_output_dirs)} processed DWI series.")
    for d in dwi_output_dirs:
        print(f"    {d.name}")

    return dwi_output_dirs


# ---------------------------------------------------------------------------
# Stage 2: Phantom QC in DWI space
# ---------------------------------------------------------------------------


def run_stage2(
    dwi_output_dirs: list,
    output_dir: Path,
    template_dir: Path,
    dry_run: bool,
    input_identifier: str = "",
    n_threads: int = 1,
):
    """
    For each DWI output directory, run PhantomProcessor on
    T1_in_DWI_space.nii.gz (or T1.nii.gz for rpe_none).
    ADC.nii.gz and FA.nii.gz in the same folder are picked up automatically.
    """
    print_header("STAGE 2 — Phantom QC in DWI Space")

    if not dwi_output_dirs:
        print("  No DWI output directories to process — skipping Stage 2.\n")
        return

    for dwi_dir in dwi_output_dirs:
        t1_in_dwi = next(
            (
                dwi_dir / n
                for n in ("T1_in_DWI_space.nii.gz", "T1.nii.gz")
                if (dwi_dir / n).exists()
            ),
            None,
        )

        if t1_in_dwi is None:
            print(f"  WARNING: No T1 image found in {dwi_dir.name} — skipping.")
            continue

        print(f"  Processing: {dwi_dir.name}")
        print(f"    Input image: {t1_in_dwi}")
        print(f"    Contrast images found alongside T1:")
        for nii in sorted(dwi_dir.glob("*.nii.gz")):
            print(f"      {nii.name}")

        if dry_run:
            print(f"  [DRY RUN] Would run PhantomProcessor on {t1_in_dwi.name}\n")
            continue

        from phantomkit.phantom_processor import PhantomProcessor

        processor = PhantomProcessor(
            template_dir=str(template_dir),
            output_base_dir=str(output_dir),
            filename_prefix=input_identifier,
            n_threads=n_threads,
        )
        processor.process_session(str(t1_in_dwi))
        print()

    print("  Stage 2 complete.\n")


# ---------------------------------------------------------------------------
# Calibration plotter: temperature estimation from vial ADC values
# ---------------------------------------------------------------------------


def run_calibration_plot(
    dwi_output_dirs: list,
    output_dir: Path,
    phantom: str,
    dry_run: bool,
):
    """
    For each processed DWI series, estimate vial temperatures from the
    ADC_mean_matrix.csv produced by Stage 2 and write an HTML report.
    """
    print_header("STAGE 4 — Calibration Temperature Estimation")

    if not dwi_output_dirs:
        print("  No DWI output directories — skipping calibration plots.\n")
        return

    from phantomkit.plotting.calibration_plotter import (
        parse_calibration_xlsx,
        parse_vials_csv,
        load_phantom_config,
        build_vial_map,
        estimate_temperature,
        build_vials_html,
    )

    formulations = parse_calibration_xlsx(str(CALIBRATION_LUT))
    phantom_cfg = load_phantom_config(str(PHANTOM_CONFIG))

    for dwi_dir in dwi_output_dirs:
        session_name = dwi_dir.name
        metrics_dir = output_dir / session_name / "metrics"
        adc_csv = metrics_dir / "csv" / "ADC_mean_matrix.csv"
        output_html = metrics_dir / f"{phantom}_vial_temperature_estimates.html"

        if not adc_csv.exists():
            print(f"  WARNING: ADC CSV not found for {session_name} — skipping.")
            print(f"    Expected: {adc_csv}\n")
            continue

        print(f"  Series:  {session_name}")
        print(f"  ADC CSV: {adc_csv}")
        print(f"  Output:  {output_html}")

        if dry_run:
            print("  [DRY RUN] Would run calibration plotter.\n")
            continue

        try:
            vial_map = build_vial_map(phantom_cfg, phantom)
        except KeyError as e:
            print(f"  ERROR: {e} — skipping calibration plot for {session_name}.\n")
            continue

        vials_adc = parse_vials_csv(str(adc_csv))
        results = []
        for vial, adc_raw in vials_adc.items():
            form_query = vial_map.get(vial)
            if form_query is None:
                print(f"  Vial {vial}: not in phantom {phantom!r} — skipping")
                continue
            try:
                res = estimate_temperature(
                    formulations, str(form_query), D=adc_raw, vial=vial
                )
                results.append(res)
            except ValueError as e:
                print(f"  Vial {vial}: {e}")

        output_html.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_html), "w", encoding="utf-8") as f:
            f.write(build_vials_html(formulations, results, phantom_name=phantom))
        print()

    print("  Stage 4 complete.\n")


# ---------------------------------------------------------------------------
# Stage 3: Phantom QC on native contrasts
# ---------------------------------------------------------------------------


def run_stage3(
    input_dir: Path,
    output_dir: Path,
    template_dir: Path,
    scan_info: dict,
    dry_run: bool,
    input_identifier: str = "",
    n_threads: int = 1,
):
    """
    Convert T1, IR, and TE DICOMs to NIfTI into a staging folder, then
    run PhantomProcessor on the T1.  The processor picks up all NIfTIs
    in the staging folder automatically.
    """
    print_header("STAGE 3 — Phantom QC on Native Contrasts")

    t1_dirs = scan_info["t1_dirs"]
    ir_dirs = scan_info["ir_dirs"]
    te_dirs = scan_info["te_dirs"]

    if not t1_dirs:
        print("  No T1 directory found — cannot run Stage 3.\n")
        return

    session_name = derive_session_name(input_dir)
    staging_dir = output_dir / "native_contrasts_staging"

    print(f"  Session name:    {session_name}")
    print(f"  Staging folder:  {staging_dir}")
    print(f"  T1 dirs:         {[d.name for d in t1_dirs]}")
    print(f"  IR dirs:         {[d.name for d in ir_dirs]}")
    print(f"  TE dirs:         {[d.name for d in te_dirs]}")
    print()

    # Pick the T1 with the lowest series number
    def _series_num(d: Path) -> int:
        m = re.match(r"^(\d+)-", d.name)
        return int(m.group(1)) if m else 0

    primary_t1_dir = sorted(t1_dirs, key=_series_num)[0]
    contrast_dirs_to_convert = [primary_t1_dir] + ir_dirs + te_dirs

    if dry_run:
        print("  [DRY RUN] Would stage series to NIfTI and place in staging folder:")
        for d in contrast_dirs_to_convert:
            fmt = _detect_series_format(d)
            print(f"    {d.name} ({fmt})  →  {staging_dir}/")
        print("\n  [DRY RUN] Would run PhantomProcessor on staged T1 NIfTI.\n")
        return

    # Stage series to NIfTI (handles DICOM, NIfTI, and MIF inputs)
    staging_dir.mkdir(parents=True, exist_ok=True)
    t1_nii_path = None

    for series_dir in contrast_dirs_to_convert:
        fmt = _detect_series_format(series_dir)
        print(f"  Staging series ({fmt}): {series_dir.name}")
        try:
            produced = stage_series_dir(series_dir, staging_dir)
            print(f"    Produced: {[Path(p).name for p in produced]}")
            if series_dir == primary_t1_dir:
                t1_nii_path = Path(produced[0])
        except Exception as e:
            print(f"  WARNING: Staging failed for {series_dir.name}: {e}")
            if series_dir == primary_t1_dir:
                print("  Cannot proceed with Stage 3 without a T1 image.")
                shutil.rmtree(staging_dir, ignore_errors=True)
                return

    if t1_nii_path is None or not t1_nii_path.exists():
        print("  ERROR: T1 NIfTI was not produced — aborting Stage 3.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    print(f"\n  All NIfTIs in staging folder:")
    for nii in sorted(staging_dir.glob("*.nii.gz")):
        print(f"    {nii.name}")

    # Run phantom processor
    print(f"\n  Running PhantomProcessor:")
    print(f"    Input image: {t1_nii_path}")
    print(f"    Output base: {output_dir}")

    from phantomkit.phantom_processor import PhantomProcessor

    processor = PhantomProcessor(
        template_dir=str(template_dir),
        output_base_dir=str(output_dir),
        filename_prefix=input_identifier,
        n_threads=n_threads,
    )
    processor.process_session(str(t1_nii_path))

    # Clean up staging NIfTIs only — leave processed outputs in place
    print(f"\n  Removing temporary NIfTIs from staging folder: {staging_dir}")
    for nii in staging_dir.glob("*.nii.gz"):
        nii.unlink(missing_ok=True)
    for jsn in staging_dir.glob("*.json"):
        jsn.unlink(missing_ok=True)
    try:
        staging_dir.rmdir()
        print(f"  Removed empty staging directory: {staging_dir.name}")
    except OSError:
        print(
            f"  Staging directory retained (contains phantom outputs): "
            f"{staging_dir.name}"
        )

    print("\n  Stage 3 complete.\n")


# ---------------------------------------------------------------------------
# PET stage
# ---------------------------------------------------------------------------


def _find_pet_niftis(input_dir: Path) -> list:
    """Return all NIfTI files in *input_dir* (flat or one level deep in subdirs)."""
    found = []
    for pattern in ("*.nii.gz", "*.nii"):
        found.extend(sorted(input_dir.glob(pattern)))
    for sub in sorted(input_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("_"):
            for pattern in ("*.nii.gz", "*.nii"):
                found.extend(sorted(sub.glob(pattern)))
    return found


def run_pet_stage(
    input_dir: Path,
    output_dir: Path,
    template_dir: Path,
    phantom: str,
    dry_run: bool,
    input_identifier: str = "",
) -> None:
    """Register each PET NIfTI in *input_dir* to the template and generate reports."""
    print_header("PET STAGE — Registration, Metric Extraction & HTML Report")

    pet_niftis = _find_pet_niftis(input_dir)
    if not pet_niftis:
        print("  No NIfTI files found in input directory — nothing to do.\n")
        return

    print(f"  Found {len(pet_niftis)} PET image(s):")
    for p in pet_niftis:
        print(f"    {p}")
    print()

    from phantomkit.pet_processing import (
        register_pet_to_template,
        apply_transform_to_vials,
        extract_pet_vial_metrics,
        save_pet_metrics_xlsx,
    )
    from phantomkit.plotting.pet_html import plot_pet_vials

    vial_template_dir = str(template_dir / "VialsLabelled")

    for pet_nii in pet_niftis:
        stem = pet_nii.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        session = f"{input_identifier}_{stem}" if input_identifier else stem
        session_out = output_dir / session
        reg_dir     = session_out / "registration"
        metrics_dir = session_out / "metrics"
        xlsx_path   = metrics_dir / "xlsx" / "PET.xlsx"
        html_path   = metrics_dir / "plots" / "PET.html"

        print(f"  Processing: {pet_nii.name}")
        print(f"    Output:   {session_out}")

        if dry_run:
            print("  [DRY RUN] Would run PET registration + metric extraction.\n")
            continue

        # ── Registration ────────────────────────────────────────────────────
        print("  Step 1/3: registration")
        reg = register_pet_to_template(
            input_image=str(pet_nii),
            results_dir=str(reg_dir),
            template_image=str(template_dir / "ImageTemplate.nii.gz"),
            template_mask=str(template_dir / "ImageTemplate_mask.nii.gz"),
        )

        # ── Vial transform ──────────────────────────────────────────────────
        print("  Step 2/3: transforming vial labels to subject space")
        vials_out = str(session_out / "VialsLabelled")
        vial_masks = apply_transform_to_vials(
            vial_dir=vial_template_dir,
            output_dir=vials_out,
            reference_image=reg["strides_image"],
            transform_prefix=reg["transform_prefix"],
        )

        # ── Metric extraction ───────────────────────────────────────────────
        print("  Step 3/3: extracting vial metrics")
        metrics = extract_pet_vial_metrics(reg["strides_image"], vial_masks)
        save_pet_metrics_xlsx(metrics, str(xlsx_path))
        print(f"    Metrics saved: {xlsx_path}")

        # ── HTML report ─────────────────────────────────────────────────────
        html_path.parent.mkdir(parents=True, exist_ok=True)
        plot_pet_vials(
            xlsx_path=str(xlsx_path),
            output=str(html_path),
            phantom=phantom,
            nifti_image=reg["strides_image"],
            vial_niftis=vial_masks,
        )
        print(f"    HTML report:  {html_path}\n")

    print("  PET stage complete.\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_inputs(input_dir: Path, phantom: str) -> None:
    """Validate paths and phantom name before any processing. Raises RuntimeError."""
    errors = []

    if not input_dir.is_dir():
        errors.append(f"--input-dir does not exist or is not a directory: {input_dir}")

    phantom_dir = TEMPLATE_DATA_ROOT / phantom
    if not phantom_dir.is_dir():
        errors.append(
            f"Phantom template directory not found: {phantom_dir}\n"
            f"  Expected: template_data/{phantom}/"
        )
    else:
        template_img = phantom_dir / "ImageTemplate.nii.gz"
        vials_dir = phantom_dir / "VialsLabelled"
        if not template_img.exists():
            errors.append(f"ImageTemplate.nii.gz not found in: {phantom_dir}")
        if not vials_dir.is_dir() or not list(vials_dir.glob("*.nii.gz")):
            errors.append(
                f"VialsLabelled/ with .nii.gz masks not found in: {phantom_dir}"
            )

    if errors:
        raise RuntimeError(
            "Validation errors:\n" + "\n".join(f"  ✗ {e}" for e in errors)
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _cleanup_tmp_only_dirs(output_dir: Path) -> None:
    """Remove subdirectories of output_dir that contain only tmp-like directories."""
    _tmp_pattern = re.compile(r"^(tmp|\.pydra)", re.IGNORECASE)
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        children = list(d.iterdir())
        if not children:
            continue
        if all(c.is_dir() and _tmp_pattern.match(c.name) for c in children):
            shutil.rmtree(d)
            print(f"  Removed tmp-only directory: {d.name}")


def run_full_pipeline(
    input_dir: Path,
    output_dir: Path,
    phantom: str,
    *,
    denoise_degibbs: bool = False,
    gradcheck: bool = False,
    nocleanup: bool = False,
    readout_time: float | None = None,
    eddy_options: str | None = None,
    dry_run: bool = False,
) -> Path:
    """Run the full phantom QA pipeline and return the output directory.

    This is the primary Python entry point. The CLI ``phantomkit pipeline``
    and the Pydra task in ``phantomkit.pydra_task`` both call this function.
    """
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    template_dir = TEMPLATE_DATA_ROOT / phantom
    input_identifier = derive_session_name(input_dir)

    validate_inputs(input_dir, phantom)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_threads = os.cpu_count() or 1
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(n_threads))

    # PET uses its own self-contained stage — skip all MRI stages.
    if phantom == "PET":
        input_dir = _wrap_flat_inputs(input_dir, output_dir)
        run_pet_stage(
            input_dir=input_dir,
            output_dir=output_dir,
            template_dir=template_dir,
            phantom=phantom,
            dry_run=dry_run,
            input_identifier=input_identifier,
        )
        print_header("Pipeline Complete")
        print(f"  All outputs written to: {output_dir}\n")
        return output_dir

    input_dir = _wrap_flat_inputs(input_dir, output_dir)

    dwi_cfg = {
        "denoise_degibbs": denoise_degibbs,
        "gradcheck": gradcheck,
        "nocleanup": nocleanup,
        "readout_time": readout_time,
        "eddy_options": eddy_options,
    }

    # ── Discover what's in the input directory ───────────────────────────────
    print_header("Input Directory Scan")
    scan_info = scan_input_dir(input_dir)

    print(f"  Input directory:     {input_dir}")
    print(f"  Phantom:             {phantom}")
    print(f"  Template dir:        {template_dir}")
    print(f"  Output dir:          {output_dir}")
    print()
    print(f"  T1 directories:      {len(scan_info['t1_dirs'])}")
    for d in scan_info["t1_dirs"]:
        print(f"    {d.name}")
    print(f"  IR directories:      {len(scan_info['ir_dirs'])}")
    for d in scan_info["ir_dirs"]:
        print(f"    {d.name}")
    print(f"  TE directories:      {len(scan_info['te_dirs'])}")
    for d in scan_info["te_dirs"]:
        print(f"    {d.name}")
    print(f"  DWI candidate dirs:  {len(scan_info['dwi_dirs'])}")
    for d in scan_info["dwi_dirs"]:
        print(f"    {d.name}")
    print(f"  Other (ignored):     {len(scan_info['other_dirs'])}")
    print()

    run_stage1_flag = scan_info["has_dwi"]
    run_stage3_flag = bool(scan_info["t1_dirs"]) and (
        scan_info["has_native_contrasts"] or not run_stage1_flag
    )

    print(
        f"  Stage 1 (DWI):                    "
        f"{'YES' if run_stage1_flag else 'NO (no DWI found)'}"
    )
    print(
        f"  Stage 2 (phantom QC, DWI space):  "
        f"{'YES (runs per DWI series)' if run_stage1_flag else 'NO'}"
    )
    print(
        f"  Stage 3 (phantom QC, native T1):  "
        f"{'YES' if run_stage3_flag else 'NO (no T1/IR/TE found)'}"
    )
    print(
        f"  Stage 4 (calibration temp. est.): "
        f"{'YES (runs per DWI series)' if run_stage1_flag else 'NO'}"
    )
    print()

    if dry_run:
        print("  NOTE: --dry-run is active.  No processing will be performed.\n")

    # ── Execution ─────────────────────────────────────────────────────────────
    # Stage 1 — DWI processing
    dwi_output_dirs = []
    if run_stage1_flag:
        dwi_output_dirs = run_stage1(input_dir, output_dir, dwi_cfg, dry_run)
    else:
        print_header("STAGE 1 — DWI Processing")
        print("  Skipped: no DWI acquisitions found.\n")

    # Stage 2 — Phantom QC in DWI space (follows Stage 1)
    if run_stage1_flag:
        run_stage2(dwi_output_dirs, output_dir, template_dir, dry_run, input_identifier, n_threads)
    else:
        print_header("STAGE 2 — Phantom QC in DWI Space")
        print("  Skipped: Stage 1 did not run.\n")

    # Stage 3 — Phantom QC on native contrasts
    if run_stage3_flag:
        run_stage3(input_dir, output_dir, template_dir, scan_info, dry_run, input_identifier, n_threads)
    else:
        print_header("STAGE 3 — Phantom QC on Native Contrasts")
        if not scan_info["t1_dirs"]:
            print("  Skipped: no T1 directory found.\n")
        else:
            print("  Skipped: no IR or TE series found, and DWI pipeline was run.\n")

    # Stage 4 — Calibration temperature estimation (requires Stage 1/2 ADC output)
    if run_stage1_flag:
        run_calibration_plot(dwi_output_dirs, output_dir, phantom, dry_run)
    else:
        print_header("STAGE 4 — Calibration Temperature Estimation")
        print("  Skipped: no DWI acquisitions found.\n")

    # ── Cleanup tmp-only directories ─────────────────────────────────────────
    if not dry_run:
        _cleanup_tmp_only_dirs(output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_header("Pipeline Complete")
    print(f"  All outputs written to: {output_dir}\n")

    if not dry_run:
        print("  Output structure:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                print(f"    {item.name}/")
                for sub in sorted(item.iterdir()):
                    if sub.is_dir():
                        print(f"      {sub.name}/")
                    else:
                        print(f"      {sub.name}")
        print()

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end phantomkit phantom QC + DWI processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phantom", required=True)
    parser.add_argument("--denoise-degibbs", action="store_true", default=False)
    parser.add_argument("--gradcheck", action="store_true", default=False)
    parser.add_argument("--nocleanup", action="store_true", default=False)
    parser.add_argument("--readout-time", type=float, default=None)
    parser.add_argument("--eddy-options", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)

    args = parser.parse_args()

    run_full_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        phantom=args.phantom,
        denoise_degibbs=args.denoise_degibbs,
        gradcheck=args.gradcheck,
        nocleanup=args.nocleanup,
        readout_time=args.readout_time,
        eddy_options=args.eddy_options,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
