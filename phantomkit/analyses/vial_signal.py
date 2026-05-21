"""
Top-level pydra workflows for phantom QA processing.

PhantomSessionWorkflow processes a single session end-to-end;
BatchWorkflow runs multiple sessions in parallel.
"""

import logging
from pathlib import Path

from fileformats.generic import Directory
from fileformats.medimage import NiftiGz
from pydra.compose import python, workflow

from phantomkit.registration import (
    RegisterToTemplate,
    SaveTemplateInScannerSpace,
)
from phantomkit.metrics import (
    TransformVialsToSubjectSpace,
    ExtractMetricsFromContrasts,
    TransformContrastsToTemplateSpace,
)
from phantomkit.plotting.visualization import GeneratePlots, Cleanup

logger = logging.getLogger(__name__)


# ============================================================================
# Session preparation tasks
# ============================================================================


@python.define(
    outputs=[
        "session_name",
        "output_dir",
        "tmp_dir",
        "vial_dir",
        "metrics_dir",
        "images_template_space_dir",
        "scanner_space_image",
    ]
)
def PrepareSessionPaths(
    input_image: NiftiGz,
    output_base_dir: Path | None = None,
) -> tuple[str, Directory, Directory, Directory, Directory, Directory, Path]:
    """
    Derive all session output paths from the input image location, create the
    directories, and return the path strings for use by downstream tasks.

    Parameters
    ----------
    input_image : NiftiGz
        Path to the primary NIfTI image for the session.  The session name is
        taken from the parent directory of this file.
    output_base_dir : Path, optional
        Root output directory.  A sub-directory named after the session
        (parent folder of *input_image*) is created automatically.
        Defaults to the current working directory.

    Returns
    -------
    session_name : str
        Name of the session (parent directory of *input_image*).
    output_dir : Directory
        Top-level session output directory (``output_base_dir / session_name``).
    tmp_dir : Directory
        Temporary working directory for intermediate files.
    vial_dir : Directory
        Directory for vial segmentation mask NIfTI files.
    metrics_dir : Directory
        Directory for per-contrast vial signal metric CSV files.
    images_template_space_dir : Directory
        Directory for contrast images warped into template space.
    scanner_space_image : Path
        Destination path for the template phantom warped into scanner space.

    Steps
    -----
    1. Resolve *output_base_dir* to the current working directory if ``None``.
    2. Derive *session_name* from the parent directory of *input_image*.
    3. Construct sub-directory paths for ``tmp``, ``vial_segmentations``,
       ``metrics``, and ``images_template_space``.
    4. Create all four sub-directories with ``mkdir(parents=True,
       exist_ok=True)``.
    5. Return all path components as a tuple for use by downstream tasks.
    """
    from pathlib import Path as _Path

    if output_base_dir is None:
        output_base_dir = _Path.cwd()
    session_name = _Path(input_image).parent.name
    output_dir = _Path(output_base_dir) / session_name
    tmp_dir = output_dir / "tmp"
    vial_dir = output_dir / "vial_segmentations"
    metrics_dir = output_dir / "metrics"
    images_template_space_dir = output_dir / "images_template_space"
    scanner_space_image = output_dir / "TemplatePhantom_ScannerSpace.nii.gz"

    for d in [tmp_dir, vial_dir, metrics_dir, images_template_space_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return (
        session_name,
        output_dir,
        tmp_dir,
        vial_dir,
        metrics_dir,
        images_template_space_dir,
        scanner_space_image,
    )


@python.define
def GetVialMasks(template_dir: Directory) -> list[NiftiGz]:
    """
    Return sorted list of vial mask NIfTI paths from the template directory.

    Parameters
    ----------
    template_dir : Directory
        Path to the GSP SPIRIT template directory.  Must contain a
        ``VialsLabelled/`` sub-directory holding per-vial mask files named
        ``*.nii.gz``.

    Returns
    -------
    out : list[NiftiGz]
        Sorted list of absolute paths to all ``*.nii.gz`` files found inside
        ``template_dir/VialsLabelled/``.

    Steps
    -----
    1. Construct the path ``template_dir / "VialsLabelled"``.
    2. Glob for all ``*.nii.gz`` files in that directory.
    3. Return the results sorted lexicographically by filename.
    """
    from pathlib import Path as _Path

    return sorted((_Path(template_dir) / "VialsLabelled").glob("*.nii.gz"))


@python.define
def GetContrastFiles(input_image: NiftiGz) -> list[NiftiGz]:
    """
    Return sorted list of all NIfTI files in the same directory as the input image.

    Parameters
    ----------
    input_image : NiftiGz
        Path to the primary NIfTI image for the session.  All ``*.nii.gz``
        files in its parent directory are treated as additional contrast images.

    Returns
    -------
    out : list[NiftiGz]
        Sorted list of absolute paths to all ``*.nii.gz`` files found in the
        same directory as *input_image*.

    Steps
    -----
    1. Resolve the parent directory of *input_image*.
    2. Glob for all ``*.nii.gz`` files in that directory.
    3. Return the results sorted lexicographically by filename.
    """
    from pathlib import Path as _Path

    return sorted(_Path(input_image).parent.glob("*.nii.gz"))


# ============================================================================
# Top-level workflows
# ============================================================================


@workflow.define(
    outputs=[
        "metrics_dir",
        "vial_dir",
        "images_template_space_dir",
        "scanner_space_image",
    ]
)
def VialSignalAnalysis(
    input_image: NiftiGz,
    template_dir: Directory,
    output_base_dir: Path | None = None,
    num_threads: int = 1,
) -> tuple[Directory, Directory, Directory, NiftiGz]:
    """
    Pydra workflow for processing a single phantom MRI session.

    Registers the phantom scan to the template using ANTs rigid SyN, extracts
    per-vial signal statistics for all contrast images, and generates plots.

    Parameters
    ----------
    input_image : NiftiGz
        Path to the primary NIfTI image for the session (e.g. the T1 MPRAGE).
        All NIfTI files in the same directory are treated as contrast images.
    template_dir : Directory
        Path to the phantom template directory.  Must contain
        ``ImageTemplate.nii.gz`` and a ``VialsLabelled/`` sub-directory of
        per-vial mask files.
    output_base_dir : Path, optional
        Root output directory.  A sub-directory named after the session
        (parent folder of *input_image*) is created automatically.
        Defaults to the current working directory.

    Returns
    -------
    metrics_dir : Directory
        Directory containing per-contrast CSV files of vial signal statistics
        (mean, median, std, min, max).
    vial_dir : Directory
        Directory containing the vial mask NIfTI files warped into subject
        (scanner) space.
    images_template_space_dir : Directory
        Directory containing all contrast images warped into template space.
    scanner_space_image : NiftiGz
        The template phantom image warped back into scanner space.

    Steps
    -----
    1. **PrepareSessionPaths**  — derive output paths and create directories
    2. **GetVialMasks**         — list template vial mask files
    3. **GetContrastFiles**     — list all contrast NIfTI files in the session
    4. **RegisterToTemplate**   — ANTs rigid SyN registration
    5. **SaveTemplateInScannerSpace** — warp template back to subject space
    6. **TransformVialsToSubjectSpace** — project vial masks into subject space
    7. **ExtractMetricsFromContrasts** — compute per-vial statistics; writes CSVs
    8. **TransformContrastsToTemplateSpace** — forward-warp all contrasts
    9. **GeneratePlots** — scatter and parametric map plots
    10. **Cleanup** — remove temporary directories
    """
    template_phantom = Path(template_dir) / "ImageTemplate.nii.gz"

    paths = workflow.add(
        PrepareSessionPaths(
            input_image=input_image,
            output_base_dir=output_base_dir,
        ),
        name="prepare_paths",
    )

    vial_masks = workflow.add(
        GetVialMasks(template_dir=template_dir),
        name="get_vial_masks",
    )

    contrast_files = workflow.add(
        GetContrastFiles(input_image=input_image),
        name="get_contrast_files",
    )

    reg = workflow.add(
        RegisterToTemplate(
            input_image=input_image,
            template_phantom=template_phantom,
            session_name=paths.session_name,
            tmp_dir=paths.tmp_dir,
            num_threads=num_threads,
        ),
        name="registration",
    )

    scanner_template = workflow.add(
        SaveTemplateInScannerSpace(
            inverse_warped=reg.inverse_warped,
            output_path=paths.scanner_space_image,
        ),
        name="save_template_scanner_space",
    )

    transform_vials = workflow.add(
        TransformVialsToSubjectSpace(
            vial_masks=vial_masks.out,
            reference_image=input_image,
            transform_matrix=reg.transform,
            output_vial_dir=paths.vial_dir,
        ),
        name="transform_vials",
    )

    extract_metrics = workflow.add(
        ExtractMetricsFromContrasts(
            contrast_files=contrast_files.out,
            vial_masks=transform_vials.out,
            output_metrics_dir=paths.metrics_dir,
            session_name=paths.session_name,
        ),
        name="extract_metrics",
    )

    warp_contrasts = workflow.add(
        TransformContrastsToTemplateSpace(
            contrast_files=contrast_files.out,
            transform_matrix=reg.transform,
            template_phantom=template_phantom,
            tmp_dir=paths.tmp_dir,
            output_dir=paths.images_template_space_dir,
        ),
        name="warp_contrasts_to_template",
    )

    workflow.add(
        GeneratePlots(
            contrast_files=contrast_files.out,
            metrics_dir=extract_metrics.out,
            vial_dir=paths.vial_dir,
            session_name=paths.session_name,
        ),
        name="generate_plots",
    )

    workflow.add(
        Cleanup(dirs=[paths.tmp_dir]),
        name="cleanup",
    )

    return (
        extract_metrics.out,
        paths.vial_dir,
        warp_contrasts.out,
        scanner_template.out,
    )


@workflow.define(outputs=["results"])
def VialSignalAnalysisBatch(
    input_images: list[NiftiGz],
    template_dir: Directory,
    output_base_dir: Path,
    num_threads: int = 1,
) -> list:
    """
    Pydra workflow for batch-processing multiple phantom sessions in parallel.

    Each session is processed by :func:`VialSignalAnalysis`; the sessions are
    split across *input_images* and results collected back into a list.

    Parameters
    ----------
    input_images : list[NiftiGz]
        One primary NIfTI image per session.  All NIfTI files in each image's
        parent directory are automatically included as contrast images.
    template_dir : Directory
        Path to the phantom template directory, passed unchanged to each
        :func:`VialSignalAnalysis` invocation.
    output_base_dir : Path
        Shared root output directory.  Each session receives its own
        sub-directory named after its parent folder.

    Returns
    -------
    results : list
        List of ``metrics_dir`` outputs from each :func:`VialSignalAnalysis`
        call, one entry per session, in the same order as *input_images*.

    Steps
    -----
    1. **VialSignalAnalysis** — run the full single-session phantom pipeline
       for every element of *input_images* in parallel (pydra ``split`` on
       ``input_image``).
    2. Collect the per-session ``metrics_dir`` outputs into a single list
       (pydra ``combine`` on ``input_image``).
    """
    process = workflow.add(
        VialSignalAnalysis(
            input_image=input_images,
            template_dir=template_dir,
            output_base_dir=output_base_dir,
            num_threads=num_threads,
        )
        .split("input_image")
        .combine("input_image"),
        name="process_sessions",
    )

    return process.metrics_dir
