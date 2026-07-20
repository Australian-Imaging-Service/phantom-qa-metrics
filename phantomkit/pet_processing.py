"""
pet_processing.py
=================
PET phantom image preprocessing and ANTs rigid registration to a template.

Pipeline:
  1. Match voxel strides to the template (``mrtransform``).
  2. Create a binary foreground mask via Otsu thresholding (``mrthreshold``).
  3. Coarse rigid grid search for initial alignment (``antsAI``).
  4. Run ``antsRegistration`` rigid with MI metric and fixed/moving masks.
  5. Apply the inverse affine transform to bring template vial labels into
     subject space (``antsApplyTransforms``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _run(cmd: list, env: Optional[dict] = None) -> None:
    """Run a shell command; raise ``CalledProcessError`` on non-zero exit."""
    str_cmd = [str(c) for c in cmd]
    print("  $", " ".join(str_cmd))
    subprocess.run(str_cmd, check=True, env=env)


def _slice_max_of_means(pet_image: str, mask_path: str) -> float:
    """Return the maximum of per-axial-slice means within a binary mask.

    For each axial (z) slice where the mask has at least one voxel, the mean
    of the PET voxels in that slice is computed.  The function returns the
    largest of those per-slice means — used as the numerator of the
    slice-averaged CRC.
    """
    import gzip
    import struct

    import numpy as np

    def _load_vol(p: str) -> np.ndarray:
        raw = Path(p).read_bytes()
        try:
            buf = gzip.decompress(raw)
        except OSError:
            buf = raw
        dims = struct.unpack_from("<8h", buf, 40)
        nx, ny, nz = dims[1], dims[2], dims[3]
        dc = struct.unpack_from("<h", buf, 70)[0]
        (vo,) = struct.unpack_from("<f", buf, 108)
        vs = max(int(vo), 352)
        _dt_map = {2: "<u1", 4: "<i2", 8: "<i4", 16: "<f4", 64: "<f8", 512: "<u2", 768: "<u4"}
        dt = np.dtype(_dt_map.get(dc, "<f4"))
        return np.frombuffer(buf[vs:], dtype=dt)[: nx * ny * nz].reshape(
            (nx, ny, nz), order="F"
        )

    pet  = _load_vol(pet_image).astype(float)
    mask = _load_vol(mask_path) > 0.5
    slice_means = [
        float(pet[:, :, z][mask[:, :, z]].mean())
        for z in range(pet.shape[2])
        if mask[:, :, z].any()
    ]
    return float(max(slice_means)) if slice_means else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_pet_to_template(
    input_image: str,
    results_dir: str,
    template_image: Optional[str] = None,
    template_mask: Optional[str] = None,
    force: bool = True,
) -> dict:
    """Register a PET image to the NEMA phantom template using ANTs rigid.

    Steps
    -----
    1. ``mrtransform`` — match voxel strides to the template.
    2. ``mrthreshold`` — Otsu threshold to create a foreground mask.
    3. ``antsAI`` — coarse rigid grid search for initial alignment.
    4. ``antsRegistration`` — rigid MI registration with fixed and moving masks.

    Parameters
    ----------
    input_image:
        Path to the input PET NIfTI (``.nii`` or ``.nii.gz``).
    results_dir:
        Directory where intermediate and output files are written.
    template_image:
        Path to the fixed (template) NIfTI.  Defaults to the bundled
        ``template_data/PET/ImageTemplate.nii.gz``.
    template_mask:
        Path to the fixed template foreground mask.  Defaults to
        ``template_data/PET/ImageTemplate_mask.nii.gz``.
    force:
        Pass ``-force`` flag to MRtrix commands to overwrite existing files.

    Returns
    -------
    dict
        ``strides_image``    — strides-matched moving image path
        ``mask_image``       — Otsu foreground mask of moving image path
        ``warped_image``     — ANTs-warped moving image in template space path
        ``transform_prefix`` — ANTs output prefix (``<prefix>0GenericAffine.mat``)
    """
    input_path   = Path(input_image)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Resolve default template paths from the bundled template_data directory.
    if template_image is None or template_mask is None:
        _pkg    = Path(__file__).resolve().parent
        _td_pet = _pkg.parent / "template_data" / "PET"
        if template_image is None:
            template_image = str(_td_pet / "ImageTemplate.nii.gz")
        if template_mask is None:
            template_mask = str(_td_pet / "ImageTemplate_mask.nii.gz")

    # Derive output filenames from the input stem.
    stem        = input_path.name
    for _ext in (".nii.gz", ".nii"):
        if stem.endswith(_ext):
            stem = stem[: -len(_ext)]
            break

    strides_img = results_path / f"{stem}_strides.nii.gz"
    mask_img    = results_path / f"{stem}_mask.nii.gz"
    init_mat    = str(results_path / f"{stem}_init.mat")
    warped_img  = results_path / f"{stem}_warped.nii.gz"
    prefix      = str(results_path / f"{stem}_")
    force_flag  = ["-force"] if force else []

    import multiprocessing
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())

    # ── Step 1: match strides ────────────────────────────────────────────────
    _run(
        ["mrtransform", str(input_path),
         "-strides", str(template_image),
         str(strides_img)] + force_flag
    )

    # ── Step 2: Otsu threshold → foreground mask ─────────────────────────────
    _run(
        ["mrthreshold", str(strides_img), str(mask_img)] + force_flag
    )

    # ── Step 3: coarse rigid search with antsAI ──────────────────────────────
    _run([
        "antsAI",
        "-d", "3",
        "-m", f"MI[{template_image},{strides_img},32,Regular,0.2]",
        "-t", "Rigid[0.1]",
        "-s", "[20,0.12]",
        "-p", "0",
        "-x", f"[{template_mask},{mask_img}]",
        "-o", init_mat,
    ], env=env)

    # ── Step 4: ANTs rigid registration ─────────────────────────────────────
    ants_cmd = [
        "antsRegistration",
        "-d", "3",
        "-o",  f"[{prefix},{warped_img}]",
        "-r",  init_mat,
        "-m",  f"MI[{template_image},{strides_img},1,32,Regular,0.25]",
        "-t",  "Rigid[0.1]",
        "-c",  "[1000x500x250x100,1e-6,10]",
        "-s",  "3x2x1x0mm",
        "-f",  "8x4x2x1",
        "-u",
        "-x",  f"[{template_mask},{mask_img}]",
        "-v",  "1",
    ]
    _run(ants_cmd, env=env)

    return {
        "strides_image":    str(strides_img),
        "mask_image":       str(mask_img),
        "warped_image":     str(warped_img),
        "transform_prefix": prefix,
    }


def apply_transform_to_vials(
    vial_dir: str,
    output_dir: str,
    reference_image: str,
    transform_prefix: str,
    force: bool = True,
) -> dict:
    """Apply the ANTs affine transform to bring template vial masks to subject space.

    The transform produced by ``register_pet_to_template`` maps subject →
    template (forward).  To bring vial labels FROM template INTO subject space
    we apply the *inverse* transform via ``antsApplyTransforms -t [mat,1]``.

    Parameters
    ----------
    vial_dir:
        Directory containing per-vial NIfTI mask files (``*.nii.gz``).
    output_dir:
        Directory where transformed vial masks will be written.
    reference_image:
        Subject-space NIfTI used to define the output grid (typically the
        strides-matched or warped image produced by ``register_pet_to_template``).
    transform_prefix:
        ANTs output prefix from ``register_pet_to_template``; the file
        ``<prefix>0GenericAffine.mat`` must exist.
    force:
        Overwrite existing output files (passed as ``-e`` suppresses ANTs
        prompts; the output path is simply overwritten).

    Returns
    -------
    dict
        ``{vial_name: path_to_transformed_mask}``
    """
    vial_path = Path(vial_dir)
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    transform = Path(transform_prefix + "0GenericAffine.mat")
    if not transform.exists():
        raise FileNotFoundError(
            f"ANTs affine matrix not found: {transform}\n"
            "Run register_pet_to_template() first."
        )

    results: dict = {}
    for vial_file in sorted(vial_path.glob("*.nii.gz")):
        vial_name = vial_file.name.replace(".nii.gz", "")
        out_file  = out_path / vial_file.name
        _run([
            "antsApplyTransforms",
            "-d", "3",
            "-i", str(vial_file),
            "-r", str(reference_image),
            "-o", str(out_file),
            "-t", f"[{transform},1]",   # ,1 = use inverse → template → subject
            "-n", "NearestNeighbor",
        ])
        results[vial_name] = str(out_file)

    return results


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------


def run_pet_pipeline(
    input_image: str,
    results_dir: str,
    vial_template_dir: Optional[str] = None,
    template_image: Optional[str] = None,
    template_mask: Optional[str] = None,
    force: bool = True,
) -> dict:
    """Run the full PET preprocessing, registration, and vial transform pipeline.

    Parameters
    ----------
    input_image:
        Path to the input PET NIfTI.
    results_dir:
        Root output directory.
    vial_template_dir:
        Directory containing per-vial template masks.  Defaults to
        ``template_data/PET/VialsLabelled/``.
    template_image, template_mask:
        See :func:`register_pet_to_template`.
    force:
        See :func:`register_pet_to_template`.

    Returns
    -------
    dict
        All paths from the registration step plus ``vial_masks`` (dict of
        ``{vial_name: transformed_mask_path}``).
    """
    if vial_template_dir is None:
        _pkg = Path(__file__).resolve().parent
        vial_template_dir = str(_pkg.parent / "template_data" / "PET" / "VialsLabelled")

    reg = register_pet_to_template(
        input_image=input_image,
        results_dir=results_dir,
        template_image=template_image,
        template_mask=template_mask,
        force=force,
    )

    vials_out_dir = str(Path(results_dir) / "VialsLabelled")
    vial_masks = apply_transform_to_vials(
        vial_dir=vial_template_dir,
        output_dir=vials_out_dir,
        reference_image=reg["strides_image"],
        transform_prefix=reg["transform_prefix"],
        force=force,
    )

    return {**reg, "vial_masks": vial_masks}


# ---------------------------------------------------------------------------
# Metric extraction and xlsx output
# ---------------------------------------------------------------------------


def extract_pet_vial_metrics(pet_image: str, vial_masks: dict) -> dict:
    """Extract per-vial statistics from a PET image using mrstats / mrdump.

    Matches the extraction approach used by PhantomProcessor for all other
    modalities: ``mrstats`` provides mean/median/std/min/max/count and
    ``mrdump`` provides p25/p75/mean_mad/median_mad.

    Parameters
    ----------
    pet_image:
        Path to the PET NIfTI image (strides-matched, in subject space).
    vial_masks:
        ``{vial_name: mask_path}`` as returned by :func:`apply_transform_to_vials`.

    Returns
    -------
    dict
        ``{vial_name: {mean, std, median, max, min, count, p25, p75,
                        mean_mad, median_mad, uniformity, crc_3d,
                        crc_slice_avg}}``
    """
    import numpy as np

    nan = float("nan")
    metrics: dict = {}

    for vial_name, mask_path in sorted(vial_masks.items()):
        # ── mrstats: mean / median / std / min / max / count ─────────────────
        result = subprocess.run(
            [
                "mrstats", "-quiet", str(pet_image),
                "-output", "mean",
                "-output", "median",
                "-output", "std",
                "-output", "min",
                "-output", "max",
                "-output", "count",
                "-mask", str(mask_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            print(f"  WARNING: mrstats failed for vial {vial_name} — skipping.")
            print(f"    stderr: {result.stderr.strip()}")
            continue

        vals = result.stdout.strip().split()
        mean_val   = float(vals[0])
        median_val = float(vals[1])

        # ── mrdump: p25 / p75 / mean_mad / median_mad ────────────────────────
        p25 = p75 = mean_mad = median_mad = nan
        try:
            dump = subprocess.run(
                ["mrdump", str(pet_image), "-mask", str(mask_path)],
                capture_output=True, text=True, check=True,
            )
            raw = np.array([float(x) for x in dump.stdout.strip().split()])
            if raw.size:
                p25        = float(np.percentile(raw, 25))
                p75        = float(np.percentile(raw, 75))
                mean_mad   = float(np.mean(np.abs(raw - raw.mean())))
                median_mad = float(np.median(np.abs(raw - np.median(raw))))
        except Exception as exc:
            print(f"  WARNING: mrdump failed for vial {vial_name}: {exc}")

        std_val = float(vals[2])
        max_val = float(vals[4])
        uniformity    = std_val / mean_val * 100 if mean_val != 0 else nan
        crc_3d        = max_val / mean_val if mean_val != 0 else nan
        sam           = _slice_max_of_means(str(pet_image), str(mask_path))
        crc_slice_avg = sam / mean_val if mean_val != 0 else nan

        metrics[vial_name] = {
            "mean":         mean_val,
            "median":       median_val,
            "std":          std_val,
            "min":          float(vals[3]),
            "max":          max_val,
            "count":        float(vals[5]),
            "p25":          p25,
            "p75":          p75,
            "mean_mad":     mean_mad,
            "median_mad":   median_mad,
            "uniformity":   uniformity,
            "crc_3d":       crc_3d,
            "crc_slice_avg": crc_slice_avg,
        }

    return metrics


def save_pet_metrics_xlsx(metrics: dict, output_path: str) -> None:
    """Write per-vial PET metrics to an xlsx file, one sheet per statistic.

    Layout matches the PhantomProcessor xlsx format: each sheet has a
    ``vial`` column and a ``vol0`` column (single PET volume).

    Parameters
    ----------
    metrics:
        Output of :func:`extract_pet_vial_metrics`.
    output_path:
        Destination ``.xlsx`` path.
    """
    import pandas as pd

    vials = sorted(metrics.keys())
    sheet_order = [
        "mean", "median", "std", "min", "max",
        "count", "p25", "p75", "mean_mad", "median_mad",
        "uniformity", "crc_3d", "crc_slice_avg",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet in sheet_order:
            df = pd.DataFrame({
                "vial": vials,
                "vol0": [metrics[v][sheet] for v in vials],
            })
            df.to_excel(writer, sheet_name=sheet, index=False)
