"""
Pydra tasks and workflows for SPIRIT phantom DWI processing.

Implements a cumulative correction pipeline:
  MIF conversion → Denoise → Degibbs → TOPUP → b-value filter → EDDY → Bias correct

and an isolated pipeline where each correction is applied independently
to the original DWI.  ADC maps are computed at each stage across multiple
b-value shells.

The top-level workflow (:func:`DiffusionMetricsAnalysis`) exposes both pipelines and
produces step-wise difference maps plus two summary CSV reports.
"""

import logging
from pathlib import Path

from fileformats.generic import Directory, File
from fileformats.medimage import NiftiGz
from fileformats.medimage.diffusion import Bval, Bvec
from fileformats.text import Csv
from fileformats.vendor.mrtrix3.medimage import (
    ImageFormatWithDwiEncoding as MifDwi,
    ImageFormat as MifImage,
)
from pydra.compose import python, workflow
from pydra.tasks.mrtrix3.v3_1 import (
    DwiBiascorrect_Ants as DwiBiascorrectAnts,
    DwiDenoise,
    DwiExtract,
    MrCat,
    MrConvert,
    MrDegibbs,
    MrGrid,
    MrMath,
)
from pydra.tasks.fsl.v6 import TOPUP, ApplyTOPUP, Eddy, EddyQuad

logger = logging.getLogger(__name__)

# B-value shells used for multi-shell ADC computation
ADC_SHELLS = [500, 1000, 2000, 3000, 4000, 6000]

# Default phantom geometry (SPIRIT phantom)
PHANTOM_DIAMETER_MM = 191
MASK_EROSION_MM = 8


# ============================================================================
# Output directory preparation
# ============================================================================


@python.define(
    outputs=[
        "session_name",
        "dwi_steps_dir",
        "dwi_steps_iso_dir",
        "adc_maps_dir",
        "adc_maps_iso_dir",
        "diff_maps_dir",
        "topup_dir",
        "topup_iso_dir",
        "eddy_dir",
        "eddy_iso_dir",
        "masks_dir",
        "denoise_dir",
        "gibbs_dir",
        "bias_dir",
        "bias_iso_dir",
    ]
)
def PrepareOutputDirs(
    input_image: NiftiGz,
    output_base_dir: Path | None = None,
) -> tuple[
    str,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    """
    Create the full output directory tree for a DWI analysis session.

    All directories are created with ``mkdir(parents=True, exist_ok=True)``
    before being returned as path components for use by downstream tasks.

    Parameters
    ----------
    input_image : NiftiGz
        Primary AP-direction DWI NIfTI image.  The session name is taken from
        the parent directory of this file.
    output_base_dir : Path, optional
        Root output directory.  Defaults to ``./output`` if not provided.

    Returns
    -------
    session_name : str
        Name of the session (parent directory of *input_image*).
    dwi_steps_dir : Path
        Directory for cumulative pipeline DWI step images.
    dwi_steps_iso_dir : Path
        Directory for isolated pipeline DWI step images.
    adc_maps_dir : Path
        Directory for cumulative pipeline ADC maps and summary CSV.
    adc_maps_iso_dir : Path
        Directory for isolated pipeline ADC maps and summary CSV.
    diff_maps_dir : Path
        Directory for step-wise Δ ADC difference maps.
    topup_dir : Path
        Working directory for TOPUP (cumulative pipeline).
    topup_iso_dir : Path
        Working directory for TOPUP (isolated pipeline).
    eddy_dir : Path
        Working directory for EDDY (cumulative pipeline).
    eddy_iso_dir : Path
        Working directory for EDDY (isolated pipeline).
    masks_dir : Path
        Directory for phantom mask files.
    denoise_dir : Path
        Directory for denoising outputs (e.g. noise maps).
    gibbs_dir : Path
        Directory for Gibbs ringing removal outputs.
    bias_dir : Path
        Directory for bias field correction outputs (cumulative pipeline).
    bias_iso_dir : Path
        Directory for bias field correction outputs (isolated pipeline).

    Steps
    -----
    1. Resolve *output_base_dir* to ``./output`` if ``None``.
    2. Derive *session_name* from the parent directory of *input_image*.
    3. Construct all 14 sub-directory paths under ``output_base_dir /
       session_name``.
    4. Create every sub-directory with ``mkdir(parents=True, exist_ok=True)``.
    5. Return session name and all directory paths as a 15-element tuple.
    """
    from pathlib import Path as _Path

    if output_base_dir is None:
        output_base_dir = _Path.cwd() / "output"
    session_name = _Path(input_image).parent.name
    base = _Path(output_base_dir) / session_name

    dirs = {
        "dwi_steps_dir": base / "dwi_steps_cumulative",
        "dwi_steps_iso_dir": base / "dwi_steps_isolated",
        "adc_maps_dir": base / "adc_maps_native",
        "adc_maps_iso_dir": base / "adc_maps_isolated_native",
        "diff_maps_dir": base / "difference_maps",
        "topup_dir": base / "topup",
        "topup_iso_dir": base / "topup_iso",
        "eddy_dir": base / "eddy",
        "eddy_iso_dir": base / "eddy_isolated",
        "masks_dir": base / "masks",
        "denoise_dir": base / "denoise_outputs",
        "gibbs_dir": base / "gibbs_outputs",
        "bias_dir": base / "bias_outputs",
        "bias_iso_dir": base / "bias_outputs_iso",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return (
        session_name,
        dirs["dwi_steps_dir"],
        dirs["dwi_steps_iso_dir"],
        dirs["adc_maps_dir"],
        dirs["adc_maps_iso_dir"],
        dirs["diff_maps_dir"],
        dirs["topup_dir"],
        dirs["topup_iso_dir"],
        dirs["eddy_dir"],
        dirs["eddy_iso_dir"],
        dirs["masks_dir"],
        dirs["denoise_dir"],
        dirs["gibbs_dir"],
        dirs["bias_dir"],
        dirs["bias_iso_dir"],
    )


# ============================================================================
# Phantom mask creation
# ============================================================================


@python.define
def _ComputeMaskFromB0(
    b0_mean_nii: NiftiGz,
    masks_dir: Path,
    phantom_diameter_mm: float,
    erosion_mm: float,
) -> NiftiGz:
    """Compute sphere+signal phantom mask from a mean b=0 NIfTI image.

    Uses nibabel/numpy/scipy to:
    1. Find the intensity-weighted centroid.
    2. Generate a sphere of *phantom_diameter_mm* centred at the centroid.
    3. Intersect with a signal intensity threshold.
    4. Fill holes and erode by *erosion_mm*.

    Returns the path to ``phantom_mask_canonical.nii.gz``.
    """
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np
    from scipy import ndimage

    masks_dir = _Path(masks_dir)
    img = nib.load(str(b0_mean_nii))
    data = img.get_fdata()
    affine = img.affine
    voxel_sizes = np.abs(np.diag(affine)[:3])

    # Intensity-weighted centroid
    threshold = np.percentile(data[data > 0], 30) if (data > 0).any() else 1.0
    above = data > threshold
    total = data[above].sum() or 1.0
    grid = np.indices(data.shape)
    centroid = np.array(
        [(data[above] * grid[i][above]).sum() / total for i in range(3)]
    )

    # Sphere mask
    radius_mm = phantom_diameter_mm / 2.0
    xi, yi, zi = np.meshgrid(
        np.arange(data.shape[0]),
        np.arange(data.shape[1]),
        np.arange(data.shape[2]),
        indexing="ij",
    )
    dist = np.sqrt(
        ((xi - centroid[0]) * voxel_sizes[0]) ** 2
        + ((yi - centroid[1]) * voxel_sizes[1]) ** 2
        + ((zi - centroid[2]) * voxel_sizes[2]) ** 2
    )
    sphere = (dist <= radius_mm).astype(np.uint8)

    # Intersect with signal, fill holes, erode
    signal_mask = (data > threshold).astype(np.uint8)
    combined = sphere & signal_mask
    filled = ndimage.binary_fill_holes(combined).astype(np.uint8)
    erosion_voxels = max(1, int(erosion_mm / voxel_sizes.min()))
    eroded = ndimage.binary_erosion(
        filled,
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=erosion_voxels,
    ).astype(np.uint8)

    out_nii = masks_dir / "phantom_mask_canonical.nii.gz"
    nib.save(nib.Nifti1Image(eroded, affine, img.header), str(out_nii))
    return out_nii


@workflow.define
def CreatePhantomMask(
    dwi_mif: MifDwi,
    masks_dir: Path,
    phantom_diameter_mm: float = PHANTOM_DIAMETER_MM,
    erosion_mm: float = MASK_EROSION_MM,
) -> MifImage:
    """
    Create a binary phantom mask from the DWI data.

    1. **DwiExtract** extracts b=0 volumes into ``masks_dir``.
    2. **MrMath** computes the mean b=0.
    3. **_ComputeMaskFromB0** generates a sphere+signal mask with nibabel/scipy.
    4. **MrConvert** converts the NIfTI mask to MIF format (uint8).

    Returns the path to ``phantom_mask_canonical.mif``.
    """
    extract = workflow.add(
        DwiExtract(
            in_file=dwi_mif,
            bzero=True,
            out_file=masks_dir / "b0.mif",
            force=True,
            quiet=True,
        ),
        name="extract_b0",
    )

    mean_b0 = workflow.add(
        MrMath(
            in_file=extract.out_file,
            operation="mean",
            out_file=masks_dir / "b0_mean.nii.gz",
            axis=3,
            force=True,
            quiet=True,
        ),
        name="mean_b0",
    )

    mask_nii = workflow.add(
        _ComputeMaskFromB0(
            b0_mean_nii=mean_b0.out_file,
            masks_dir=masks_dir,
            phantom_diameter_mm=phantom_diameter_mm,
            erosion_mm=erosion_mm,
        ),
        name="compute_mask",
    )

    mask_mif = workflow.add(
        MrConvert(
            in_file=mask_nii.out,
            out_file=masks_dir / "phantom_mask_canonical.mif",
            datatype="uint8",
            force=True,
            quiet=True,
        ),
        name="convert_mask",
    )

    return mask_mif.out_file


# ============================================================================
# TOPUP helpers
# ============================================================================


@python.define
def WriteTopupConfig(topup_dir: Path) -> File:
    """
    Write a custom TOPUP configuration file to *topup_dir*.

    The configuration uses 9 resolution levels with 100 non-linear iterations
    at the finest resolution (3 mm voxel spacing), bending-energy
    regularisation, and spline interpolation.

    Parameters
    ----------
    topup_dir : Path
        Destination directory.  The file is written as
        ``topup_dir/topup_custom.cnf``.

    Returns
    -------
    out : File
        Absolute path to the written ``topup_custom.cnf`` configuration file.

    Steps
    -----
    1. Construct the output path ``topup_dir / "topup_custom.cnf"``.
    2. Write the 9-level warp-resolution schedule with 100 non-linear
       iterations at the finest (3 mm) level, bending-energy regularisation,
       and cubic-spline interpolation.
    3. Return the path to the written file.
    """
    from pathlib import Path as _Path

    config_path = _Path(topup_dir) / "topup_custom.cnf"
    config_path.write_text(
        "# Custom TOPUP config — 100 non-linear iterations at finest resolution (3 mm)\n"
        "warpres=20,16,14,12,10,6,4,4,3\n"
        "subsamp=2,2,2,2,2,1,1,1,1\n"
        "fwhm=8,6,4,3,3,2,1,0,0\n"
        "miter=5,5,5,5,5,10,10,20,100\n"
        "lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001\n"
        "ssqlambda=1\n"
        "regmod=bending_energy\n"
        "estmov=1,1,1,1,1,0,0,0,0\n"
        "minmet=0,0,0,0,0,1,1,1,1\n"
        "splineorder=3\n"
        "numprec=double\n"
        "interp=spline\n"
        "scale=1\n"
    )
    return config_path


@python.define
def _LoadPaB0Mean(pa_ref_dir: Path, output_path: Path) -> NiftiGz:
    """
    Load PA b=0 NIfTI files from a directory and save their mean to *output_path*.

    If the directory contains a single 4-D file, the mean is computed along
    the volume axis.  If multiple files are present, the per-voxel mean is
    computed across all volumes/files.

    Parameters
    ----------
    pa_ref_dir : Path
        Directory containing one or more PA b=0 NIfTI files (``*.nii`` or
        ``*.nii.gz``).  Files are processed in sorted order.
    output_path : Path
        Destination path for the mean PA b=0 NIfTI image.

    Returns
    -------
    out : NiftiGz
        Path to the saved mean PA b=0 NIfTI file (*output_path*).

    Steps
    -----
    1. Glob *pa_ref_dir* for all ``*.nii`` and ``*.nii.gz`` files (sorted).
    2. Raise ``FileNotFoundError`` if no files are found.
    3. Load all images with nibabel.
    4. If a single 4-D image is found, compute the mean along the last axis;
       otherwise compute the per-voxel mean across all images.
    5. Save the result to *output_path* and return the path.
    """
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np

    pa_ref_dir = _Path(pa_ref_dir)
    output_path = _Path(output_path)
    pa_niis = sorted(pa_ref_dir.glob("*.nii")) + sorted(pa_ref_dir.glob("*.nii.gz"))
    if not pa_niis:
        raise FileNotFoundError(
            f"No NIfTI files found in PA reference directory: {pa_ref_dir}"
        )
    pa_imgs = [nib.load(str(p)) for p in pa_niis]
    if len(pa_imgs) == 1 and pa_imgs[0].ndim == 4:
        pa_data = pa_imgs[0].get_fdata().mean(axis=-1)
    else:
        pa_data = np.mean([img.get_fdata() for img in pa_imgs], axis=0)
    nib.save(nib.Nifti1Image(pa_data, pa_imgs[0].affine), str(output_path))
    return output_path


@python.define
def _WriteAcqparams(topup_dir: Path, readout_time: float) -> File:
    """
    Write the AP+PA acquisition parameters file required by FSL TOPUP.

    The file contains two rows: one for the AP direction (phase-encode
    direction ``0 -1 0``) and one for the PA direction (``0 1 0``), both
    with the same total readout time.

    Parameters
    ----------
    topup_dir : Path
        Destination directory.  The file is written as
        ``topup_dir/acqparams.txt``.
    readout_time : float
        Total EPI readout time in seconds, written as the fourth column of
        each row.

    Returns
    -------
    out : File
        Absolute path to the written ``acqparams.txt`` file.

    Steps
    -----
    1. Construct the output path ``topup_dir / "acqparams.txt"``.
    2. Write two rows: AP (``0 -1 0 <readout_time>``) and PA
       (``0  1 0 <readout_time>``).
    3. Return the path to the written file.
    """
    from pathlib import Path as _Path

    acqparams = _Path(topup_dir) / "acqparams.txt"
    acqparams.write_text(f"0 -1 0 {readout_time}\n0  1 0 {readout_time}\n")
    return acqparams


@workflow.define(outputs=["both_b0_file", "acqparams_file", "n_ap_b0"])
def PrepareTopupData(
    dwi_mif: MifDwi,
    pa_ref_dir: Path,
    topup_dir: Path,
    readout_time: float = 0.033,
) -> tuple[NiftiGz, File, int]:
    """
    Prepare TOPUP input: concatenate mean AP b=0 and mean PA b=0.

    1. **DwiExtract** + **MrMath** compute the mean AP b=0.
    2. **_LoadPaB0Mean** loads and averages PA reference images.
    3. **MrCat** concatenates AP and PA along the volume axis.
    4. **_WriteAcqparams** writes the acquisition parameters file.

    Returns the 4-D concatenated NIfTI, the acqparams.txt, and the number
    of AP b=0 volumes (always 1 — we use the mean).
    """
    extract_ap = workflow.add(
        DwiExtract(
            in_file=dwi_mif,
            bzero=True,
            out_file=topup_dir / "ap_b0.mif",
            force=True,
            quiet=True,
        ),
        name="extract_ap_b0",
    )

    mean_ap = workflow.add(
        MrMath(
            in_file=extract_ap.out_file,
            operation="mean",
            out_file=topup_dir / "ap_b0_mean.nii.gz",
            axis=3,
            force=True,
            quiet=True,
        ),
        name="mean_ap_b0",
    )

    pa_mean = workflow.add(
        _LoadPaB0Mean(
            pa_ref_dir=pa_ref_dir,
            output_path=topup_dir / "pa_b0_mean.nii.gz",
        ),
        name="load_pa_b0",
    )

    concat = workflow.add(
        MrCat(
            inputs=[mean_ap.out_file, pa_mean.out],
            out_file=topup_dir / "both_b0.nii.gz",
            axis=3,
            force=True,
            quiet=True,
        ),
        name="concat_b0",
    )

    acqparams = workflow.add(
        _WriteAcqparams(topup_dir=topup_dir, readout_time=readout_time),
        name="write_acqparams",
    )

    @python.define
    def _One() -> int:
        return 1

    n_ap_b0 = workflow.add(_One(), name="n_ap_b0")

    return concat.out_file, acqparams.out, n_ap_b0.out


@python.define
def _WriteTopupIndex(dwi_nii: NiftiGz) -> list[int]:
    """
    Return a per-volume index list (all ones) for use by FSL ApplyTOPUP.

    The index value ``1`` maps every DWI volume to the first row of the
    TOPUP acquisition parameters file (the AP phase-encode direction).

    Parameters
    ----------
    dwi_nii : NiftiGz
        4-D DWI NIfTI image.  Only the last dimension (number of volumes) is
        used.

    Returns
    -------
    out : list[int]
        List of integers, all equal to ``1``, with length equal to the number
        of volumes in *dwi_nii*.

    Steps
    -----
    1. Load *dwi_nii* with nibabel and read the last dimension as ``nvols``.
    2. Return a list of ``1`` repeated ``nvols`` times.
    """
    import nibabel as nib

    nvols = nib.load(str(dwi_nii)).shape[-1]
    return [1] * nvols


@workflow.define
def ApplyTopupToDwi(
    dwi_mif: MifDwi,
    topup_fieldcoef: File,
    topup_movpar: File,
    acqparams_file: File,
    topup_dir: Path,
) -> MifDwi:
    """
    Apply TOPUP correction to the full DWI series (AP volumes, index=1).

    1. **MrConvert** exports DWI to NIfTI with FSL gradient files.
    2. **_WriteTopupIndex** generates a per-volume index list (all 1s).
    3. **ApplyTOPUP** applies Jacobian modulation correction.
    4. **MrConvert** re-embeds the gradient table and converts back to MIF.
    """
    export = workflow.add(
        MrConvert(
            in_file=dwi_mif,
            out_file=topup_dir / "dwi_for_topup.nii.gz",
            export_grad_fsl=True,
            force=True,
            quiet=True,
        ),
        name="export_dwi",
    )

    index = workflow.add(
        _WriteTopupIndex(dwi_nii=export.out_file),
        name="write_index",
    )

    apply = workflow.add(
        ApplyTOPUP(
            in_files=[export.out_file],
            in_topup_fieldcoef=topup_fieldcoef,
            in_topup_movpar=topup_movpar,
            encoding_file=acqparams_file,
            in_index=index.out,
            method="jac",
            out_corrected=topup_dir / "dwi_topup_corrected",
        ),
        name="apply_topup",
    )

    convert = workflow.add(
        MrConvert(
            in_file=apply.out_corrected,
            out_file=topup_dir / "dwi_topup_corrected.mif",
            fslgrad=export.export_grad_fsl,
            force=True,
            quiet=True,
        ),
        name="convert_to_mif",
    )

    return convert.out_file


# ============================================================================
# TOPUP pipeline sub-workflow
# ============================================================================


@workflow.define(outputs=["corrected", "fieldcoef", "movpar", "acqparams"])
def TopupCorrectionStep(
    dwi_mif: MifDwi,
    pa_ref_dir: Path,
    topup_dir: Path,
    readout_time: float = 0.033,
) -> tuple[MifDwi, File, File, File]:
    """
    Run FSL TOPUP and apply the distortion correction to the full DWI series.

    Steps
    -----
    1. **WriteTopupConfig** — write custom 100-iteration CNF file
    2. **PrepareTopupData** — extract mean AP b=0, load PA b=0, concatenate
    3. **TOPUP** — estimate the susceptibility-induced distortion field
    4. **ApplyTopupToDwi** — apply the field to all DWI volumes
    """
    config = workflow.add(
        WriteTopupConfig(topup_dir=topup_dir),
        name="write_config",
    )

    prep = workflow.add(
        PrepareTopupData(
            dwi_mif=dwi_mif,
            pa_ref_dir=pa_ref_dir,
            topup_dir=topup_dir,
            readout_time=readout_time,
        ),
        name="prep_topup_data",
    )

    topup = workflow.add(
        TOPUP(
            in_file=prep.both_b0_file,
            encoding_file=prep.acqparams_file,
            config=config.out,
            out_base=str(topup_dir / "topup_results"),
        ),
        name="topup",
    )

    apply = workflow.add(
        ApplyTopupToDwi(
            dwi_mif=dwi_mif,
            topup_fieldcoef=topup.out_fieldcoef,
            topup_movpar=topup.out_movpar,
            acqparams_file=prep.acqparams_file,
            topup_dir=topup_dir,
        ),
        name="apply_topup",
    )

    return apply.out, topup.out_fieldcoef, topup.out_movpar, prep.acqparams_file


# ============================================================================
# B-value filtering
# ============================================================================


@python.define(outputs=["filtered_nii", "bvals_file", "bvecs_file"])
def _FilterAndSaveVolumes(
    dwi_nii: NiftiGz,
    fslgrad: tuple,
    output_dir: Path,
    min_bval: float = 500.0,
) -> tuple[NiftiGz, Bval, Bvec]:
    """
    Filter DWI volumes, retaining b=0 volumes (b < 50) and shells ≥ min_bval.

    Intermediate shells (e.g. b=100, b=200) that fall between the b=0
    threshold and *min_bval* are discarded.  The filtered image and updated
    gradient files are saved as ``step03_5_filtered.*`` inside *output_dir*.

    Parameters
    ----------
    dwi_nii : NiftiGz
        4-D DWI NIfTI image to filter.
    fslgrad : tuple
        Two-element tuple ``(bvec_file, bval_file)`` containing the FSL-format
        gradient files corresponding to *dwi_nii*.
    output_dir : Path
        Directory in which the filtered NIfTI and gradient files are written.
    min_bval : float
        Minimum b-value for non-zero shells to retain (default: 500).  All
        shells with b-values in the range [50, *min_bval*) are dropped.

    Returns
    -------
    filtered_nii : NiftiGz
        Path to ``output_dir/step03_5_filtered.nii.gz``.
    bvals_file : Bval
        Path to ``output_dir/step03_5_filtered.bval``.
    bvecs_file : Bvec
        Path to ``output_dir/step03_5_filtered.bvec``.

    Steps
    -----
    1. Load b-values, b-vectors, and image data with numpy/nibabel.
    2. Build a boolean keep-mask: ``bvals < 50`` (b=0) OR
       ``bvals >= min_bval`` (target shells).
    3. Index the image data and gradient arrays with the keep-mask.
    4. Save the filtered NIfTI, b-values, and b-vectors to *output_dir*.
    5. Return the three output file paths.
    """
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np

    output_dir = _Path(output_dir)
    bvec_file, bval_file = fslgrad

    bvals = np.loadtxt(str(bval_file))
    bvecs = np.loadtxt(str(bvec_file))
    img = nib.load(str(dwi_nii))
    data = img.get_fdata()

    keep = (bvals < 50) | (bvals >= min_bval)
    filtered_nii = str(output_dir / "step03_5_filtered.nii.gz")
    bval_out = str(output_dir / "step03_5_filtered.bval")
    bvec_out = str(output_dir / "step03_5_filtered.bvec")
    nib.save(nib.Nifti1Image(data[..., keep], img.affine, img.header), filtered_nii)
    np.savetxt(bval_out, bvals[keep][np.newaxis, :], fmt="%d")
    np.savetxt(bvec_out, bvecs[:, keep], fmt="%.6f")

    return filtered_nii, bval_out, bvec_out


@workflow.define(outputs=["filtered_mif", "bvals_file", "bvecs_file"])
def FilterBvalueShells(
    dwi_mif: MifDwi,
    output_dir: Path,
    min_bval: float = 500.0,
) -> tuple[MifDwi, Bval, Bvec]:
    """
    Remove intermediate b-value shells (e.g. b=100, b=200).

    1. **MrConvert** exports to NIfTI with FSL gradient files.
    2. **_FilterAndSaveVolumes** keeps b=0 and b ≥ *min_bval* with nibabel/numpy.
    3. **MrConvert** re-embeds the filtered gradient table into MIF.

    Returns the filtered DWI as a .mif file plus separate bval/bvec files.
    """
    export = workflow.add(
        MrConvert(
            in_file=dwi_mif,
            out_file=output_dir / "tmp_for_filter.nii.gz",
            export_grad_fsl=True,
            force=True,
            quiet=True,
        ),
        name="export_nii",
    )

    filt = workflow.add(
        _FilterAndSaveVolumes(
            dwi_nii=export.out_file,
            fslgrad=export.export_grad_fsl,
            output_dir=output_dir,
            min_bval=min_bval,
        ),
        name="filter_volumes",
    )

    convert = workflow.add(
        MrConvert(
            in_file=filt.filtered_nii,
            out_file=output_dir / "step03_5_filtered.mif",
            fslgrad=(filt.bvecs_file, filt.bvals_file),
            force=True,
            quiet=True,
        ),
        name="convert_filtered",
    )

    return convert.out_file, filt.bvals_file, filt.bvecs_file


# ============================================================================
# EDDY pipeline helpers
# ============================================================================


@python.define(outputs=["index_file", "acqparams_file"])
def _WriteEddyFiles(
    dwi_nii: NiftiGz,
    fslgrad: tuple,
    topup_acqparams: File,
    eddy_dir: Path,
) -> tuple[File, File]:
    """
    Write the EDDY index file and an AP-only acquisition parameters file.

    The index file maps every DWI volume to the first row of the acqparams
    file (AP direction, index=1).  The acqparams file is derived from the
    TOPUP file by retaining only the first (AP) row.

    Parameters
    ----------
    dwi_nii : NiftiGz
        4-D DWI NIfTI image.  Only the number of volumes is used.
    fslgrad : tuple
        Two-element tuple ``(bvec_file, bval_file)``; not directly used but
        accepted for consistency with the workflow interface.
    topup_acqparams : File
        TOPUP acquisition parameters file (two rows: AP then PA).  Only the
        first row is copied to the EDDY acqparams file.
    eddy_dir : Path
        Destination directory for the written files.

    Returns
    -------
    index_file : File
        Path to ``eddy_dir/index.txt`` — space-separated column of ``1`` values,
        one per volume.
    acqparams_file : File
        Path to ``eddy_dir/acqparams.txt`` — single AP row extracted from
        *topup_acqparams*.

    Steps
    -----
    1. Load *dwi_nii* with nibabel to determine the number of volumes.
    2. Write ``index.txt`` as a single space-separated line of ``1`` values.
    3. Read the first line (AP row) from *topup_acqparams*.
    4. Write that AP row to ``eddy_dir/acqparams.txt``.
    5. Return both file paths as strings.
    """
    from pathlib import Path as _Path

    import nibabel as nib

    eddy_dir = _Path(eddy_dir)
    nvols = nib.load(str(dwi_nii)).shape[-1]

    index_file = str(eddy_dir / "index.txt")
    with open(index_file, "w") as f:
        f.write(" ".join(["1"] * nvols) + "\n")

    # Use only the AP row from the TOPUP acqparams
    acqparams_file = eddy_dir / "acqparams.txt"
    with open(topup_acqparams) as src:
        ap_row = src.readline()
    acqparams_file.write_text(ap_row)

    return index_file, str(acqparams_file)


@python.define(outputs=["bvecs_file", "bvals_file"])
def _UnpackGradFsl(grad_fsl: tuple) -> tuple[Bvec, Bval]:
    """
    Unpack an FSL gradient tuple into separately named bvecs and bvals outputs.

    Pydra tasks that export FSL gradient files return a single ``(bvec, bval)``
    tuple.  This helper splits the tuple into named outputs so that downstream
    tasks can reference each file by name.

    Parameters
    ----------
    grad_fsl : tuple
        Two-element tuple in FSL order ``(bvec_file, bval_file)`` as returned
        by MrConvert's ``export_grad_fsl`` output.

    Returns
    -------
    bvecs_file : Bvec
        Path to the FSL-format b-vectors file (first element of *grad_fsl*).
    bvals_file : Bval
        Path to the FSL-format b-values file (second element of *grad_fsl*).

    Steps
    -----
    1. Index ``grad_fsl[0]`` to extract the b-vectors file path.
    2. Index ``grad_fsl[1]`` to extract the b-values file path.
    3. Return both as named outputs.
    """
    return grad_fsl[0], grad_fsl[1]


@workflow.define(
    outputs=["dwi_nii", "bvals_file", "bvecs_file", "index_file", "acqparams_file"]
)
def PrepareEddyInputs(
    dwi_mif: MifDwi,
    topup_acqparams: File,
    eddy_dir: Path,
) -> tuple[NiftiGz, Bval, Bvec, File, File]:
    """
    Export DWI to NIfTI + FSL grads and generate the EDDY index file.

    1. **MrConvert** exports DWI to NIfTI with FSL gradient files.
    2. **_WriteEddyFiles** writes the volume index and AP-only acqparams.
    3. **_UnpackGradFsl** separates bvec/bval into named outputs.

    Reuses the AP row from the TOPUP acqparams (index=1 for all volumes).
    """
    export = workflow.add(
        MrConvert(
            in_file=dwi_mif,
            out_file=eddy_dir / "dwi_for_eddy.nii.gz",
            export_grad_fsl=True,
            force=True,
            quiet=True,
        ),
        name="export_nii",
    )

    eddy_files = workflow.add(
        _WriteEddyFiles(
            dwi_nii=export.out_file,
            fslgrad=export.export_grad_fsl,
            topup_acqparams=topup_acqparams,
            eddy_dir=eddy_dir,
        ),
        name="write_eddy_files",
    )

    unpack = workflow.add(
        _UnpackGradFsl(grad_fsl=export.export_grad_fsl),
        name="unpack_grads",
    )

    return (
        export.out_file,
        unpack.bvals_file,
        unpack.bvecs_file,
        eddy_files.index_file,
        eddy_files.acqparams_file,
    )


@workflow.define
def ConvertEddyOutputToMif(
    eddy_nii: NiftiGz,
    rotated_bvecs: Bvec,
    bvals_file: Bval,
    output_mif: Path,
) -> MifDwi:
    """
    Convert an EDDY-corrected NIfTI image with rotated b-vectors back to MIF.

    EDDY outputs a NIfTI file and a set of rotation-corrected b-vectors.
    This workflow embeds those gradient files into the MRtrix MIF format so
    that all downstream tools can access them from a single file.

    Parameters
    ----------
    eddy_nii : NiftiGz
        EDDY-corrected DWI NIfTI image.
    rotated_bvecs : Bvec
        FSL-format b-vectors file with per-volume rotation corrections applied
        by EDDY (``*.eddy_rotated_bvecs``).
    bvals_file : Bval
        FSL-format b-values file (unchanged from EDDY input).
    output_mif : Path
        Destination path for the output MIF file.

    Returns
    -------
    out : MifDwi
        Path to the MIF DWI file at *output_mif* with embedded gradient table.

    Steps
    -----
    1. **MrConvert** — embed *rotated_bvecs* and *bvals_file* into *eddy_nii*
       and write the result to *output_mif* in MRtrix MIF format.
    """
    convert = workflow.add(
        MrConvert(
            in_file=eddy_nii,
            out_file=output_mif,
            fslgrad=(rotated_bvecs, bvals_file),
            force=True,
            quiet=True,
        ),
        name="convert",
    )
    return convert.out_file


# ============================================================================
# EDDY pipeline sub-workflow
# ============================================================================


@workflow.define(outputs=["corrected_mif", "qc_dir"])
def EddyCorrectionStep(
    dwi_mif: MifDwi,
    phantom_mask: MifImage,
    topup_fieldcoef: File,
    topup_movpar: File,
    topup_acqparams: File,
    eddy_dir: Path,
) -> tuple[MifDwi, Directory]:
    """
    Run FSL EDDY with ``--repol`` outlier replacement.

    Steps
    -----
    1. **PrepareEddyInputs** — export to NIfTI, write index/acqparams
    2. **MrGrid** — regrid phantom mask to DWI voxel grid
    3. **Eddy** — eddy_openmp with ``repol`` and 4-sigma outlier threshold
    4. **EddyQuad** — generate QC PDF and JSON
    5. **ConvertEddyOutputToMif** — embed rotated bvecs into MIF
    """
    prep = workflow.add(
        PrepareEddyInputs(
            dwi_mif=dwi_mif,
            topup_acqparams=topup_acqparams,
            eddy_dir=eddy_dir,
        ),
        name="prep_eddy",
    )

    regrid_mask = workflow.add(
        MrGrid(
            in_file=phantom_mask,
            operation="regrid",
            template=prep.dwi_nii,
            out_file=str(eddy_dir / "eddy_mask.nii.gz"),
            interp="nearest",
            force=True,
            quiet=True,
        ),
        name="regrid_mask",
    )

    eddy = workflow.add(
        Eddy(
            in_file=prep.dwi_nii,
            in_mask=regrid_mask.out_file,
            in_index=prep.index_file,
            in_acqp=prep.acqparams_file,
            in_bval=prep.bvals_file,
            in_bvec=prep.bvecs_file,
            in_topup_fieldcoef=topup_fieldcoef,
            in_topup_movpar=topup_movpar,
            out_base=str(eddy_dir / "dwi_eddy"),
            repol=True,
            outlier_nstd=4,
        ),
        name="eddy",
    )

    qc = workflow.add(
        EddyQuad(
            base_name=str(eddy_dir / "dwi_eddy"),
            idx_file=prep.index_file,
            param_file=prep.acqparams_file,
            mask=regrid_mask.out_file,
            bvals=prep.bvals_file,
            bvecs=eddy.out_rotated_bvecs,
            output_dir=str(eddy_dir / "qc"),
        ),
        name="eddy_qc",
    )

    convert = workflow.add(
        ConvertEddyOutputToMif(
            eddy_nii=eddy.out_corrected,
            rotated_bvecs=eddy.out_rotated_bvecs,
            bvals_file=prep.bvals_file,
            output_mif=eddy_dir / "step04_eddy.mif",
        ),
        name="convert_eddy",
    )

    return convert.out, qc.qc_dir


# ============================================================================
# ADC computation
# ============================================================================


@python.define
def _ComputeShellAdc(
    shell_mean_nii: NiftiGz,
    b0_mean_nii: NiftiGz,
    b_value: float,
    output_path: Path,
) -> NiftiGz:
    """
    Compute the apparent diffusion coefficient map for a single b-value shell.

    Uses the mono-exponential model: ADC = ln(S0 / S_b) / b, where S0 is the
    mean b=0 signal and S_b is the mean signal at shell b.  Both signals are
    clipped to a minimum of 1e-6 to avoid division-by-zero or log-of-zero
    errors.

    Parameters
    ----------
    shell_mean_nii : NiftiGz
        Mean signal image at the target b-value shell.
    b0_mean_nii : NiftiGz
        Mean b=0 signal image (reference S0).  Must share the same voxel
        grid as *shell_mean_nii*.
    b_value : float
        The b-value (in s/mm²) of the target shell.
    output_path : Path
        Destination path for the ADC NIfTI image.

    Returns
    -------
    out : NiftiGz
        Path to the saved ADC map at *output_path*.

    Steps
    -----
    1. Load *b0_mean_nii* and *shell_mean_nii* with nibabel; clip both to
       a minimum of 1e-6.
    2. Compute ``ratio = S0 / S_b`` per voxel.
    3. Compute ``ADC = ln(ratio) / b_value`` where ratio > 0, else 0.
    4. Save the ADC map to *output_path* and return the path.
    """
    import nibabel as nib
    import numpy as np

    ref_img = nib.load(str(b0_mean_nii))
    s0 = np.maximum(ref_img.get_fdata(), 1e-6)
    s_b = np.maximum(nib.load(str(shell_mean_nii)).get_fdata(), 1e-6)
    ratio = s0 / s_b
    adc = np.where(ratio > 0, np.log(ratio) / b_value, 0.0)
    nib.save(nib.Nifti1Image(adc, ref_img.affine, ref_img.header), str(output_path))
    return output_path


@python.define(outputs=["adc_map", "stats_row"])
def _AggregateAdcMaps(
    shell_adc_files: list[NiftiGz],
    phantom_mask: NiftiGz,
    output_dir: Path,
    label: str,
) -> tuple[NiftiGz, dict]:
    """
    Average per-shell ADC maps and compute phantom-masked summary statistics.

    All shell ADC maps are averaged into a single mean ADC volume, which is
    saved to *output_dir*.  Masked (within-phantom) descriptive statistics are
    returned as a dictionary for downstream CSV reporting.

    Parameters
    ----------
    shell_adc_files : list[NiftiGz]
        Per-shell ADC NIfTI images to average.  All must share the same voxel
        grid.
    phantom_mask : NiftiGz
        Binary phantom mask in NIfTI format.  Used to restrict statistics to
        voxels inside the phantom.
    output_dir : Path
        Directory in which the aggregated ADC map is written as
        ``ADC_<label>.nii.gz``.
    label : str
        Pipeline step label embedded in the output filename and returned in
        the statistics dictionary (e.g. ``"01_denoised"``).

    Returns
    -------
    adc_map : NiftiGz
        Path to the saved mean ADC NIfTI image.
    stats_row : dict
        Dictionary with keys ``label``, ``mean_adc``, ``std_adc``,
        ``min_adc``, ``max_adc`` computed over phantom-masked voxels.

    Steps
    -----
    1. Load all *shell_adc_files* and compute their per-voxel mean.
    2. Save the mean ADC volume to ``output_dir/ADC_<label>.nii.gz``.
    3. Apply *phantom_mask* to restrict statistics to phantom voxels.
    4. Compute mean, standard deviation, minimum, and maximum ADC within the
       mask and package them into *stats_row*.
    5. Return the ADC map path and the statistics dictionary.
    """
    from pathlib import Path as _Path

    import nibabel as nib
    import numpy as np

    output_dir = _Path(output_dir)
    ref_img = nib.load(str(shell_adc_files[0]))
    adc_maps = [nib.load(str(f)).get_fdata() for f in shell_adc_files]
    mean_adc = np.mean(adc_maps, axis=0)

    adc_nii = str(output_dir / f"ADC_{label}.nii.gz")
    nib.save(nib.Nifti1Image(mean_adc, ref_img.affine, ref_img.header), adc_nii)

    mask = nib.load(str(phantom_mask)).get_fdata().astype(bool)
    masked = mean_adc[mask]
    stats_row = {
        "label": label,
        "mean_adc": float(masked.mean()),
        "std_adc": float(masked.std()),
        "min_adc": float(masked.min()),
        "max_adc": float(masked.max()),
    }
    return adc_nii, stats_row


@workflow.define(outputs=["adc_map", "stats_row"])
def ComputeAdcMaps(
    dwi_mif: MifDwi,
    phantom_mask: NiftiGz,
    output_dir: Path,
    label: str,
    shells: list[float] = ADC_SHELLS,
) -> tuple[NiftiGz, dict]:
    """
    Compute multi-shell mean ADC maps from a DWI series.

    For each shell b in *shells*:
    1. **DwiExtract** + **MrMath** compute the mean signal at that b-value.
    2. **_ComputeShellAdc** computes ADC = ln(S0 / S_b) / b per voxel.

    A combined **_AggregateAdcMaps** averages all shell ADC maps and
    returns masked statistics as a dict for the summary CSV.
    """
    # Extract and average b=0
    extract_b0 = workflow.add(
        DwiExtract(
            in_file=dwi_mif,
            bzero=True,
            out_file=output_dir / f"tmp_{label}_b0.mif",
            force=True,
            quiet=True,
        ),
        name="extract_b0",
    )

    mean_b0 = workflow.add(
        MrMath(
            in_file=extract_b0.out_file,
            operation="mean",
            out_file=output_dir / f"tmp_{label}_b0_mean.nii.gz",
            axis=3,
            force=True,
            quiet=True,
        ),
        name="mean_b0",
    )

    # Per-shell ADC computation (static loop — runs at graph construction time)
    shell_adc_files = []
    for b in shells:
        tag = f"{label}_shell_{int(b)}"

        extract_shell = workflow.add(
            DwiExtract(
                in_file=dwi_mif,
                shells=[float(b)],
                out_file=output_dir / f"tmp_{tag}.mif",
                force=True,
                quiet=True,
            ),
            name=f"extract_{tag}",
        )

        mean_shell = workflow.add(
            MrMath(
                in_file=extract_shell.out_file,
                operation="mean",
                out_file=output_dir / f"tmp_{tag}_mean.nii.gz",
                axis=3,
                force=True,
                quiet=True,
            ),
            name=f"mean_{tag}",
        )

        adc_shell = workflow.add(
            _ComputeShellAdc(
                shell_mean_nii=mean_shell.out_file,
                b0_mean_nii=mean_b0.out_file,
                b_value=float(b),
                output_path=output_dir / f"tmp_{tag}_adc.nii.gz",
            ),
            name=f"adc_{tag}",
        )

        shell_adc_files.append(adc_shell.out)

    agg = workflow.add(
        _AggregateAdcMaps(
            shell_adc_files=shell_adc_files,
            phantom_mask=phantom_mask,
            output_dir=output_dir,
            label=label,
        ),
        name="aggregate_adc",
    )

    return agg.adc_map, agg.stats_row


@python.define(outputs=["diff_map"])
def ComputeDifferenceMap(
    adc_before: NiftiGz,
    adc_after: NiftiGz,
    output_path: Path,
) -> NiftiGz:
    """
    Compute the voxelwise ADC difference map (adc_after − adc_before).

    The difference quantifies the contribution of a single correction step to
    the overall change in the apparent diffusion coefficient map.  The result
    is saved in the same voxel space as the input images.

    Parameters
    ----------
    adc_before : NiftiGz
        ADC map from the pipeline stage before the correction of interest.
    adc_after : NiftiGz
        ADC map from the pipeline stage after the correction of interest.
        Must share the same voxel grid as *adc_before*.
    output_path : Path
        Destination path for the difference NIfTI image.

    Returns
    -------
    diff_map : NiftiGz
        Path to the saved voxelwise difference map at *output_path*.

    Steps
    -----
    1. Load *adc_before* and *adc_after* with nibabel.
    2. Compute the per-voxel difference ``adc_after - adc_before``.
    3. Save the result to *output_path* using the affine and header from
       *adc_after*.
    4. Return *output_path*.
    """
    import nibabel as nib

    img_before = nib.load(str(adc_before))
    img_after = nib.load(str(adc_after))
    diff = img_after.get_fdata() - img_before.get_fdata()
    nib.save(
        nib.Nifti1Image(diff, img_after.affine, img_after.header), str(output_path)
    )
    return output_path


@python.define
def WriteSummaryCsv(
    stats_rows: list[dict],
    csv_path: Path,
    session_name: str,
    pipeline: str,
) -> Csv:
    """
    Write per-step ADC summary statistics to a CSV file.

    Each row in the output corresponds to one pipeline step and contains the
    session name, pipeline type, step label, and ADC statistics computed over
    the phantom mask.

    Parameters
    ----------
    stats_rows : list[dict]
        List of statistics dictionaries, each containing keys ``label``,
        ``mean_adc``, ``std_adc``, ``min_adc``, and ``max_adc`` as produced
        by :func:`_AggregateAdcMaps`.
    csv_path : Path
        Destination path for the output CSV file.
    session_name : str
        Session identifier added to every row as the ``Dataset`` column.
    pipeline : str
        Pipeline type string (e.g. ``"cumulative"`` or ``"isolated"``) added
        to every row as the ``Pipeline`` column.

    Returns
    -------
    out : Csv
        Path to the written CSV file at *csv_path*.

    Steps
    -----
    1. Convert each entry in *stats_rows* to a flat dictionary with columns
       ``Dataset``, ``Pipeline``, ``Step``, ``Mean_ADC``, ``StdDev_ADC``,
       ``Min_ADC``, and ``Max_ADC``.
    2. Build a pandas DataFrame from the list of row dictionaries.
    3. Write the DataFrame to *csv_path* as a CSV (without the row index).
    4. Return *csv_path*.
    """
    import pandas as pd

    rows = [
        {
            "Dataset": session_name,
            "Pipeline": pipeline,
            "Step": r["label"],
            "Mean_ADC": r["mean_adc"],
            "StdDev_ADC": r["std_adc"],
            "Min_ADC": r["min_adc"],
            "Max_ADC": r["max_adc"],
        }
        for r in stats_rows
    ]
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    return csv_path


# ============================================================================
# Cumulative DWI pipeline
# ============================================================================


@workflow.define(
    outputs=[
        "step00_original",
        "step01_denoised",
        "step02_degibbs",
        "step03_topup",
        "step03_5_filtered",
        "step04_eddy",
        "step06_biascorr",
        "phantom_mask",
        "adc_maps_dir",
        "diff_maps_dir",
        "summary_csv",
    ]
)
def CumulativeDwiPipeline(
    input_image: NiftiGz,
    bvals_file: Bval,
    bvecs_file: Bvec,
    pa_ref_dir: Path,
    output_base_dir: Path | None = None,
    readout_time: float = 0.033,
    enable_eddy: bool = True,
) -> tuple[
    MifDwi,
    MifDwi,
    MifDwi,
    MifDwi,
    MifDwi,
    MifDwi | None,
    MifDwi,
    MifImage,
    Directory,
    Directory,
    Csv,
]:
    """
    Cumulative SPIRIT phantom DWI correction pipeline.

    Corrections are applied in sequence so each step builds on the previous:
    Convert → Denoise → Degibbs → TOPUP → b-value filter → EDDY → Bias correct.

    ADC maps are computed at every stage and step-wise difference maps
    (Δ ADC) quantify the contribution of each individual correction.

    Parameters
    ----------
    input_image : NiftiGz
        Primary AP-direction DWI NIfTI image.
    bvals_file : Bval
        FSL-format b-values file.
    bvecs_file : Bvec
        FSL-format b-vectors file.
    pa_ref_dir : Path
        Directory containing the PA b=0 reference image(s) for TOPUP.
    output_base_dir : Path, optional
        Root output directory.  Session sub-directory is created automatically.
    readout_time : float
        Total EPI readout time in seconds (default: 0.033 s).
    enable_eddy : bool
        Run FSL EDDY (default: True).

    Returns
    -------
    step00_original : MifDwi
        Original DWI converted to MIF format (step 0).
    step01_denoised : MifDwi
        Denoised DWI (step 1).
    step02_degibbs : MifDwi
        Gibbs-ringing-corrected DWI (step 2).
    step03_topup : MifDwi
        TOPUP-corrected DWI (step 3).
    step03_5_filtered : MifDwi
        TOPUP-corrected DWI with intermediate b-value shells removed (step 3.5).
    step04_eddy : MifDwi or None
        EDDY-corrected DWI (step 4), or ``None`` if *enable_eddy* is ``False``.
    step06_biascorr : MifDwi
        Bias-field-corrected DWI (step 6).
    phantom_mask : MifImage
        Binary phantom mask derived from the mean b=0 image.
    adc_maps_dir : Directory
        Directory containing per-step ADC maps and the pipeline summary CSV.
    diff_maps_dir : Directory
        Directory containing step-wise Δ ADC difference maps.
    summary_csv : Csv
        Summary CSV with ADC statistics for every pipeline step.

    Steps
    -----
    1. **PrepareOutputDirs** — create all output sub-directories and derive
       the session name.
    2. **MrConvert** (step 0) — embed FSL gradient files and convert the
       input NIfTI to MIF format.
    3. **DwiDenoise** (step 1) — Marchenko-Pastur PCA denoising.
    4. **MrDegibbs** (step 2) — Gibbs ringing removal along axes [0, 1].
    5. **CreatePhantomMask** — compute a sphere+signal binary mask from the
       mean b=0 image of the degibbed data.
    6. **TopupCorrectionStep** (step 3) — estimate and apply susceptibility
       distortion correction with FSL TOPUP.
    7. **FilterBvalueShells** (step 3.5) — remove intermediate b-value
       shells, keeping b=0 and b ≥ 500.
    8. **EddyCorrectionStep** (step 4, optional) — eddy-current and motion
       correction with FSL EDDY + EddyQuad QC report.
    9. **DwiBiascorrectAnts** (step 6) — N4 bias field correction via ANTs.
    10. **ComputeAdcMaps** — compute multi-shell mean ADC maps at every
        pipeline stage (steps 0, 1, 2, 3, 3.5, 4 [optional], 6).
    11. **ComputeDifferenceMap** — compute step-wise Δ ADC maps between
        consecutive stages and the total correction difference (step 0 → 6).
    12. **WriteSummaryCsv** — aggregate ADC statistics into a single CSV.
    """
    paths = workflow.add(
        PrepareOutputDirs(
            input_image=input_image,
            output_base_dir=output_base_dir,
        ),
        name="prepare_dirs",
    )

    # Step 0: NIfTI + FSL grads → MIF
    step00 = workflow.add(
        MrConvert(
            in_file=input_image,
            out_file=paths.dwi_steps_dir / "step00_original.mif",
            fslgrad=(bvecs_file, bvals_file),
            force=True,
            quiet=True,
        ),
        name="step00_convert",
    )

    # Step 1: Denoising (Marchenko-Pastur PCA)
    step01 = workflow.add(
        DwiDenoise(
            dwi=step00.out_file,
            out=paths.dwi_steps_dir / "step01_denoised.mif",
            noise=paths.denoise_dir / "noise_map.mif",
            force=True,
            quiet=True,
        ),
        name="step01_denoise",
    )

    # Step 2: Gibbs ringing removal
    step02 = workflow.add(
        MrDegibbs(
            in_=step01.out,
            out=paths.dwi_steps_dir / "step02_denoised_degibbs.mif",
            axes=[0, 1],
            force=True,
            quiet=True,
        ),
        name="step02_degibbs",
    )

    # Create sphere phantom mask from degibbed data
    mask = workflow.add(
        CreatePhantomMask(
            dwi_mif=step02.out,
            masks_dir=paths.masks_dir,
        ),
        name="create_mask",
    )

    # Step 3: TOPUP susceptibility correction
    step03 = workflow.add(
        TopupCorrectionStep(
            dwi_mif=step02.out,
            pa_ref_dir=pa_ref_dir,
            topup_dir=paths.topup_dir,
            readout_time=readout_time,
        ),
        name="step03_topup",
    )

    # Step 3.5: Remove b=100 and b=200 shells; keep b=0 and b≥500
    step03_5 = workflow.add(
        FilterBvalueShells(
            dwi_mif=step03.corrected,
            output_dir=paths.dwi_steps_dir,
        ),
        name="step03_5_filter",
    )

    # Step 4: EDDY eddy-current and motion correction
    if enable_eddy:
        step04 = workflow.add(
            EddyCorrectionStep(
                dwi_mif=step03_5.filtered_mif,
                phantom_mask=mask.out,
                topup_fieldcoef=step03.fieldcoef,
                topup_movpar=step03.movpar,
                topup_acqparams=step03.acqparams,
                eddy_dir=paths.eddy_dir,
            ),
            name="step04_eddy",
        )
        eddy_mif = step04.corrected_mif
    else:
        eddy_mif = step03_5.filtered_mif

    # Step 6: Bias field correction (step 5 = GNC, disabled)
    step06 = workflow.add(
        DwiBiascorrectAnts(
            in_file=eddy_mif,
            mask=mask.out,
            out_file=paths.dwi_steps_dir / "step06_final_biascorr.mif",
            nocleanup=False,
            force=True,
            quiet=True,
            config=[],
        ),
        name="step06_biascorr",
    )

    # ---- ADC maps at every stage ----
    adc00 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step00.out_file,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="00_original",
        ),
        name="adc_00",
    )
    adc01 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step01.out,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="01_denoised",
        ),
        name="adc_01",
    )
    adc02 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step02.out,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="02_degibbs",
        ),
        name="adc_02",
    )
    adc03 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step03.corrected,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="03_topup",
        ),
        name="adc_03",
    )
    adc03_5 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step03_5.filtered_mif,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="03_5_filtered",
        ),
        name="adc_03_5",
    )
    adc06 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=step06.out_file,
            phantom_mask=mask.out,
            output_dir=paths.adc_maps_dir,
            label="06_final",
        ),
        name="adc_06",
    )

    # ---- Step-wise difference maps ----
    workflow.add(
        ComputeDifferenceMap(
            adc_before=adc00.adc_map,
            adc_after=adc01.adc_map,
            output_path=paths.diff_maps_dir / "diff_01_denoise.nii.gz",
        ),
        name="diff_01",
    )
    workflow.add(
        ComputeDifferenceMap(
            adc_before=adc01.adc_map,
            adc_after=adc02.adc_map,
            output_path=paths.diff_maps_dir / "diff_02_gibbs.nii.gz",
        ),
        name="diff_02",
    )
    workflow.add(
        ComputeDifferenceMap(
            adc_before=adc02.adc_map,
            adc_after=adc03.adc_map,
            output_path=paths.diff_maps_dir / "diff_03_topup.nii.gz",
        ),
        name="diff_03",
    )
    workflow.add(
        ComputeDifferenceMap(
            adc_before=adc03.adc_map,
            adc_after=adc03_5.adc_map,
            output_path=paths.diff_maps_dir / "diff_03_5_filter.nii.gz",
        ),
        name="diff_03_5",
    )
    workflow.add(
        ComputeDifferenceMap(
            adc_before=adc00.adc_map,
            adc_after=adc06.adc_map,
            output_path=paths.diff_maps_dir / "diff_total.nii.gz",
        ),
        name="diff_total",
    )

    all_stats = [
        adc00.stats_row,
        adc01.stats_row,
        adc02.stats_row,
        adc03.stats_row,
        adc03_5.stats_row,
    ]

    if enable_eddy:
        adc04 = workflow.add(
            ComputeAdcMaps(
                dwi_mif=step04.corrected_mif,
                phantom_mask=mask.out,
                output_dir=paths.adc_maps_dir,
                label="04_eddy",
            ),
            name="adc_04",
        )
        workflow.add(
            ComputeDifferenceMap(
                adc_before=adc03_5.adc_map,
                adc_after=adc04.adc_map,
                output_path=paths.diff_maps_dir / "diff_04_eddy.nii.gz",
            ),
            name="diff_04",
        )
        workflow.add(
            ComputeDifferenceMap(
                adc_before=adc04.adc_map,
                adc_after=adc06.adc_map,
                output_path=paths.diff_maps_dir / "diff_06_bias.nii.gz",
            ),
            name="diff_06",
        )
        all_stats.extend([adc04.stats_row, adc06.stats_row])
    else:
        workflow.add(
            ComputeDifferenceMap(
                adc_before=adc03_5.adc_map,
                adc_after=adc06.adc_map,
                output_path=paths.diff_maps_dir / "diff_06_bias.nii.gz",
            ),
            name="diff_06",
        )
        all_stats.append(adc06.stats_row)

    csv = workflow.add(
        WriteSummaryCsv(
            stats_rows=all_stats,
            csv_path=paths.adc_maps_dir / "pipeline_summary.csv",
            session_name=paths.session_name,
            pipeline="cumulative",
        ),
        name="summary_csv",
    )

    return (
        step00.out_file,
        step01.out,
        step02.out,
        step03.corrected,
        step03_5.filtered_mif,
        eddy_mif if enable_eddy else None,
        step06.out_file,
        mask.out,
        paths.adc_maps_dir,
        paths.diff_maps_dir,
        csv.out,
    )


# ============================================================================
# Isolated DWI pipeline
# ============================================================================


@workflow.define(
    outputs=[
        "iso_denoised",
        "iso_degibbs",
        "iso_topup",
        "iso_eddy",
        "iso_biascorr",
        "adc_maps_iso_dir",
        "summary_csv",
    ]
)
def IsolatedDwiPipeline(
    input_image: NiftiGz,
    bvals_file: Bval,
    bvecs_file: Bvec,
    pa_ref_dir: Path,
    phantom_mask: MifImage,
    output_base_dir: Path | None = None,
    readout_time: float = 0.033,
    enable_eddy: bool = True,
) -> tuple[MifDwi, MifDwi, MifDwi, MifDwi | None, MifDwi, Directory, Csv]:
    """
    Isolated correction pipeline: each correction applied independently to the
    original DWI for ablation analysis.

    Unlike :func:`CumulativeDwiPipeline`, corrections here are not accumulated.
    Instead, each correction (denoise, degibbs, TOPUP+EDDY, bias) is applied
    independently to the original converted DWI so that the isolated effect of
    each step on the ADC can be measured.  The phantom mask is inherited from
    the cumulative pipeline to ensure consistent ROI definitions across both
    analyses.

    Parameters
    ----------
    input_image : NiftiGz
        Primary AP-direction DWI NIfTI image.
    bvals_file : Bval
        FSL-format b-values file.
    bvecs_file : Bvec
        FSL-format b-vectors file.
    pa_ref_dir : Path
        Directory containing the PA b=0 reference image(s) for TOPUP.
    phantom_mask : MifImage
        Binary phantom mask inherited from :func:`CumulativeDwiPipeline`.
    output_base_dir : Path, optional
        Root output directory.  Session sub-directory is created automatically.
    readout_time : float
        Total EPI readout time in seconds (default: 0.033 s).
    enable_eddy : bool
        Run FSL EDDY in the isolated TOPUP+EDDY branch (default: True).

    Returns
    -------
    iso_denoised : MifDwi
        DWI with only denoising applied to the original.
    iso_degibbs : MifDwi
        DWI with only Gibbs ringing removal applied to the original.
    iso_topup : MifDwi
        DWI with only TOPUP susceptibility correction applied to the original.
    iso_eddy : MifDwi or None
        DWI with TOPUP+filter+EDDY applied to the original, or ``None`` if
        *enable_eddy* is ``False``.
    iso_biascorr : MifDwi
        DWI with only ANTs bias field correction applied to the original.
    adc_maps_iso_dir : Directory
        Directory containing isolated pipeline ADC maps and summary CSV.
    summary_csv : Csv
        Summary CSV with ADC statistics for each isolated correction branch.

    Steps
    -----
    1. **PrepareOutputDirs** — create isolated pipeline output sub-directories.
    2. **MrConvert** (step 0) — convert the original NIfTI to MIF format with
       embedded gradient files.
    3. **DwiDenoise** (iso step 1) — apply denoising only to the original DWI.
    4. **MrDegibbs** (iso step 2) — apply Gibbs ringing removal only to the
       original DWI.
    5. **TopupCorrectionStep** (iso step 3) — apply TOPUP only to the original
       DWI using the isolated TOPUP directory.
    6. **FilterBvalueShells** (iso step 3.5) — remove intermediate shells from
       the isolated TOPUP output.
    7. **EddyCorrectionStep** (iso step 4, optional) — apply TOPUP+filter+EDDY
       to the original DWI.
    8. **DwiBiascorrectAnts** (iso step 6) — apply bias field correction only
       to the original DWI.
    9. **ComputeAdcMaps** — compute ADC maps for each isolated branch
       (denoised, degibbs, topup, topup+eddy [optional], biascorr).
    10. **WriteSummaryCsv** — write isolated pipeline ADC statistics to CSV.
    """
    paths = workflow.add(
        PrepareOutputDirs(
            input_image=input_image,
            output_base_dir=output_base_dir,
        ),
        name="prepare_dirs",
    )

    step00 = workflow.add(
        MrConvert(
            in_file=input_image,
            out_file=paths.dwi_steps_iso_dir / "step00_original.mif",
            fslgrad=(bvecs_file, bvals_file),
            force=True,
            quiet=True,
        ),
        name="step00_convert",
    )

    # Isolated step 1: denoise only (applied to original)
    iso01 = workflow.add(
        DwiDenoise(
            dwi=step00.out_file,
            out=paths.dwi_steps_iso_dir / "step01_isolated_denoised.mif",
            force=True,
            quiet=True,
        ),
        name="iso_denoise",
    )

    # Isolated step 2: degibbs only (applied to original)
    iso02 = workflow.add(
        MrDegibbs(
            in_=step00.out_file,
            out=paths.dwi_steps_iso_dir / "step02_isolated_degibbs.mif",
            axes=[0, 1],
            force=True,
            quiet=True,
        ),
        name="iso_degibbs",
    )

    # Isolated step 3: TOPUP only (applied to original)
    iso03 = workflow.add(
        TopupCorrectionStep(
            dwi_mif=step00.out_file,
            pa_ref_dir=pa_ref_dir,
            topup_dir=paths.topup_iso_dir,
            readout_time=readout_time,
        ),
        name="iso_topup",
    )

    # Isolated step 4: TOPUP + filter + EDDY (chained from original)
    iso03_5 = workflow.add(
        FilterBvalueShells(
            dwi_mif=iso03.corrected,
            output_dir=paths.dwi_steps_iso_dir,
        ),
        name="iso_filter",
    )

    if enable_eddy:
        iso04 = workflow.add(
            EddyCorrectionStep(
                dwi_mif=iso03_5.filtered_mif,
                phantom_mask=phantom_mask,
                topup_fieldcoef=iso03.fieldcoef,
                topup_movpar=iso03.movpar,
                topup_acqparams=iso03.acqparams,
                eddy_dir=paths.eddy_iso_dir,
            ),
            name="iso_eddy",
        )
        iso04_mif = iso04.corrected_mif
    else:
        iso04_mif = iso03_5.filtered_mif

    # Isolated step 6: bias correction only (applied to original)
    iso06 = workflow.add(
        DwiBiascorrectAnts(
            in_file=step00.out_file,
            mask=phantom_mask,
            out_file=paths.dwi_steps_iso_dir / "step06_isolated_biascorr.mif",
            nocleanup=False,
            force=True,
            quiet=True,
            config=[],
        ),
        name="iso_biascorr",
    )

    # ADC maps for each isolated output
    adc_iso01 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=iso01.out,
            phantom_mask=phantom_mask,
            output_dir=paths.adc_maps_iso_dir,
            label="iso_denoised",
        ),
        name="adc_iso01",
    )
    adc_iso02 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=iso02.out,
            phantom_mask=phantom_mask,
            output_dir=paths.adc_maps_iso_dir,
            label="iso_degibbs",
        ),
        name="adc_iso02",
    )
    adc_iso03 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=iso03.corrected,
            phantom_mask=phantom_mask,
            output_dir=paths.adc_maps_iso_dir,
            label="iso_topup",
        ),
        name="adc_iso03",
    )
    adc_iso06 = workflow.add(
        ComputeAdcMaps(
            dwi_mif=iso06.out_file,
            phantom_mask=phantom_mask,
            output_dir=paths.adc_maps_iso_dir,
            label="iso_biascorr",
        ),
        name="adc_iso06",
    )

    all_iso_stats = [
        adc_iso01.stats_row,
        adc_iso02.stats_row,
        adc_iso03.stats_row,
    ]

    if enable_eddy:
        adc_iso04 = workflow.add(
            ComputeAdcMaps(
                dwi_mif=iso04.corrected_mif,
                phantom_mask=phantom_mask,
                output_dir=paths.adc_maps_iso_dir,
                label="iso_topup_eddy",
            ),
            name="adc_iso04",
        )
        all_iso_stats.append(adc_iso04.stats_row)

    all_iso_stats.append(adc_iso06.stats_row)

    csv = workflow.add(
        WriteSummaryCsv(
            stats_rows=all_iso_stats,
            csv_path=paths.adc_maps_iso_dir / "pipeline_summary_isolated.csv",
            session_name=paths.session_name,
            pipeline="isolated",
        ),
        name="summary_csv",
    )

    return (
        iso01.out,
        iso02.out,
        iso03.corrected,
        iso04_mif if enable_eddy else None,
        iso06.out_file,
        paths.adc_maps_iso_dir,
        csv.out,
    )


# ============================================================================
# Top-level analysis workflows
# ============================================================================


@workflow.define(
    outputs=[
        "adc_maps_dir",
        "adc_maps_iso_dir",
        "diff_maps_dir",
        "summary_csv",
        "summary_iso_csv",
    ]
)
def DiffusionMetricsAnalysis(
    input_image: NiftiGz,
    bvals_file: Bval,
    bvecs_file: Bvec,
    pa_ref_dir: Path,
    output_base_dir: Path | None = None,
    readout_time: float = 0.033,
    enable_eddy: bool = True,
) -> tuple[Directory, Directory, Directory, Csv, Csv]:
    """
    Full SPIRIT phantom DWI analysis workflow.

    Runs the cumulative and isolated correction pipelines, producing ADC
    maps, step-wise difference maps, and summary CSV reports for both.

    Parameters
    ----------
    input_image : NiftiGz
        Primary AP-direction DWI NIfTI image (e.g. the main DMRI acquisition).
        All b-value shells present in the file are used; intermediate shells
        (b=100, b=200) are automatically removed before EDDY.
    bvals_file : Bval
        FSL-format b-values file corresponding to *input_image*.
    bvecs_file : Bvec
        FSL-format b-vectors file corresponding to *input_image*.
    pa_ref_dir : Path
        Directory containing the PA (posterior–anterior) b=0 reference
        image(s).  Used to estimate the susceptibility-induced distortion
        field via TOPUP.  A single SBREF or a directory of NIfTI files are
        both supported (volumes are averaged).
    output_base_dir : Path, optional
        Root output directory.  A session sub-directory named after the
        parent folder of *input_image* is created automatically.
        Defaults to ``./output``.
    readout_time : float
        Total EPI readout time in seconds (default: 0.033 s).
    enable_eddy : bool
        Include FSL EDDY eddy-current and motion correction (default: True).

    Returns
    -------
    adc_maps_dir : Path
        Cumulative pipeline ADC maps directory.
    adc_maps_iso_dir : Path
        Isolated pipeline ADC maps directory.
    diff_maps_dir : Path
        Step-wise Δ ADC difference maps directory.
    summary_csv : Csv
        Cumulative pipeline summary CSV
        (columns: Dataset, Pipeline, Step, Mean_ADC, StdDev_ADC, Min_ADC, Max_ADC).
    summary_iso_csv : Csv
        Isolated pipeline summary CSV (same columns).

    Steps
    -----
    1. **CumulativeDwiPipeline** — run the full sequential correction chain
       (Convert → Denoise → Degibbs → TOPUP → b-value filter → EDDY →
       Bias correct), producing ADC maps, difference maps, and a summary CSV.
    2. **IsolatedDwiPipeline** — apply each correction independently to the
       original DWI using the phantom mask from step 1, producing isolated ADC
       maps and a separate summary CSV.
    """
    cumulative = workflow.add(
        CumulativeDwiPipeline(
            input_image=input_image,
            bvals_file=bvals_file,
            bvecs_file=bvecs_file,
            pa_ref_dir=pa_ref_dir,
            output_base_dir=output_base_dir,
            readout_time=readout_time,
            enable_eddy=enable_eddy,
        ),
        name="cumulative_pipeline",
    )

    isolated = workflow.add(
        IsolatedDwiPipeline(
            input_image=input_image,
            bvals_file=bvals_file,
            bvecs_file=bvecs_file,
            pa_ref_dir=pa_ref_dir,
            phantom_mask=cumulative.phantom_mask,
            output_base_dir=output_base_dir,
            readout_time=readout_time,
            enable_eddy=enable_eddy,
        ),
        name="isolated_pipeline",
    )

    return (
        cumulative.adc_maps_dir,
        isolated.adc_maps_iso_dir,
        cumulative.diff_maps_dir,
        cumulative.summary_csv,
        isolated.summary_csv,
    )


@workflow.define(outputs=["results"])
def DiffusionMetricsAnalysisBatch(
    input_images: list[NiftiGz],
    bvals_files: list[Bval],
    bvecs_files: list[Bvec],
    pa_ref_dirs: list[Path],
    output_base_dir: Path,
    readout_time: float = 0.033,
    enable_eddy: bool = True,
) -> list:
    """
    Batch SPIRIT phantom DWI analysis — processes multiple sessions in parallel.

    Each session is processed by :func:`DiffusionMetricsAnalysis`; sessions are
    split across ``input_images`` and results collected back into a list.

    Parameters
    ----------
    input_images : list[NiftiGz]
        One primary DWI image per session.
    bvals_files : list[Bval]
        Corresponding b-values files (same order as *input_images*).
    bvecs_files : list[Bvec]
        Corresponding b-vectors files (same order as *input_images*).
    pa_ref_dirs : list[Path]
        Corresponding PA reference directories (same order as *input_images*).
    output_base_dir : Path
        Shared root output directory; each session gets its own sub-directory.
    readout_time : float
        Total EPI readout time in seconds (default: 0.033 s).
    enable_eddy : bool
        Include FSL EDDY correction (default: True).

    Returns
    -------
    results : list
        List of cumulative pipeline ``summary_csv`` outputs from each
        :func:`DiffusionMetricsAnalysis` call, one entry per session, in the
        same order as *input_images*.

    Steps
    -----
    1. **DiffusionMetricsAnalysis** — run the full single-session analysis
       (cumulative + isolated pipelines) for every element of *input_images*
       in parallel (pydra ``split`` on ``input_image``, ``bvals_file``,
       ``bvecs_file``, and ``pa_ref_dir``).
    2. Collect the per-session ``summary_csv`` outputs into a single list
       (pydra ``combine`` on ``input_image``).
    """
    process = workflow.add(
        DiffusionMetricsAnalysis(
            input_image=input_images,
            bvals_file=bvals_files,
            bvecs_file=bvecs_files,
            pa_ref_dir=pa_ref_dirs,
            output_base_dir=output_base_dir,
            readout_time=readout_time,
            enable_eddy=enable_eddy,
        )
        .split("input_image", "bvals_file", "bvecs_file", "pa_ref_dir")
        .combine("input_image"),
        name="process_sessions",
    )

    return process.summary_csv
