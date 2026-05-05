"""
Pydra tasks and workflows for generating QA visualisations.

Wraps mrview screenshot capture (MrView shell task) and the matplotlib
plotting functions in the other plotting modules as pydra tasks.
"""

import logging
from pathlib import Path

from fileformats.generic import File
from fileformats.medimage import NiftiGz
from pydra.compose import python, shell, workflow
from pydra.tasks.mrtrix3.v3_1 import MrGrid, MrMath

logger = logging.getLogger(__name__)


# ============================================================================
# Shell Tasks
# ============================================================================


@shell.define
class MrCat(shell.Task["MrCat.Outputs"]):
    """Concatenate several images into one."""

    executable = "mrcat"

    in_files: list[NiftiGz] = shell.arg(
        argstr="",
        position=1,
        help="the input image(s).",
    )
    out_file: Path = shell.arg(
        argstr="",
        position=2,
        help="the output image.",
    )
    axis: int | None = shell.arg(
        default=None,
        argstr="-axis",
        help="specify axis along which to perform the concatenation.",
    )
    force: bool = shell.arg(
        default=False,
        argstr="-force",
        help="force image output, even if file exists.",
    )

    class Outputs(shell.Outputs):
        out_file: NiftiGz = shell.outarg(
            argstr="",
            position=2,
            path_template="{out_file}",
            help="the output image.",
        )


@shell.define
class MrView(shell.Task["MrView.Outputs"]):
    """Capture an mrview screenshot with an optional ROI overlay.

    mrview appends ``0000`` to ``capture_prefix`` when naming the output
    file, so the output is ``{capture_folder}/{capture_prefix}0000.png``.
    """

    executable = "mrview"

    image: NiftiGz = shell.arg(
        argstr="",
        position=1,
        help="Input image to display.",
    )
    mode: int = shell.arg(
        default=1,
        argstr="-mode",
        help="Display mode (1 = ortho view).",
    )
    plane: int = shell.arg(
        default=2,
        argstr="-plane",
        help="Plane to display (0=sagittal, 1=coronal, 2=axial).",
    )
    roi_load: NiftiGz | None = shell.arg(
        default=None,
        argstr="-roi.load",
        help="ROI overlay image to load.",
    )
    roi_colour: str | None = shell.arg(
        default=None,
        argstr="-roi.colour",
        help="ROI overlay colour as 'R,G,B'.",
    )
    roi_opacity: float | None = shell.arg(
        default=None,
        argstr="-roi.opacity",
        help="ROI overlay opacity (0–1).",
    )
    comments: int | None = shell.arg(
        default=None,
        argstr="-comments",
        help="Show image comments (0 = hide).",
    )
    noannotations: bool = shell.arg(
        default=False,
        argstr="-noannotations",
        help="Hide image annotations.",
    )
    fullscreen: bool = shell.arg(
        default=False,
        argstr="-fullscreen",
        help="Display fullscreen.",
    )
    capture_folder: Path = shell.arg(
        argstr="-capture.folder",
        help="Directory to save the screenshot.",
    )
    capture_prefix: str = shell.arg(
        argstr="-capture.prefix",
        help="Filename prefix for the screenshot.",
    )
    intensity_range: str | None = shell.arg(
        default=None,
        argstr="-intensity_range",
        help=(
            "Display intensity range as 'min,max'. "
            "Use '0,1' for FA maps and '0,0.005' for ADC maps."
        ),
    )
    capture_grab: bool = shell.arg(
        default=True,
        argstr="-capture.grab",
        help="Capture a screenshot.",
    )
    exit_after: bool = shell.arg(
        default=True,
        argstr="-exit",
        help="Exit mrview after capturing.",
    )

    class Outputs(shell.Outputs):
        out_png: File = shell.outarg(
            path_template="{capture_folder}/{capture_prefix}0000.png",
            help="Captured screenshot (mrview appends '0000' to the prefix).",
        )


# ============================================================================
# Workflows
# ============================================================================


@workflow.define
def BuildRoiOverlay(
    vial_masks: list[NiftiGz], reference_image: NiftiGz, prefix: str, tmp_dir: Path
) -> NiftiGz:
    """Regrid each vial mask to the reference image space and combine into one overlay."""
    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    cat_tmp = tmp / f"{prefix}_cat_tmp.nii.gz"
    roi_overlay = tmp / f"{prefix}_VialsCombined.nii.gz"

    regridded: list = []
    for vial_mask in vial_masks:  # triggers runtime fallback
        vial_name = Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "")
        out = str(tmp / f"{prefix}_{vial_name}.nii.gz")
        grid = workflow.add(
            MrGrid(
                in_file=vial_mask,
                operation="regrid",
                template=reference_image,
                out_file=out,
                interp="nearest",
                force=True,
            ),
            name=f"grid_{prefix}_{vial_name}",
        )
        regridded.append(grid.out_file)

    cat = workflow.add(
        MrCat(in_files=regridded, out_file=cat_tmp, axis=3, force=True),
        name=f"cat_{prefix}",
    )
    math = workflow.add(
        MrMath(in_file=[cat.out_file], operation="max", out_file=roi_overlay, axis=3),
        name=f"math_{prefix}",
    )
    return math.out_file


@workflow.define
def GeneratePlots(
    contrast_files: list[NiftiGz],
    metrics_dir: Path,
    vial_dir: Path,
    session_name: str,
    output_format: str = "html",
    phantom_name: str | None = None,
) -> None:
    """
    Generate per-contrast vial intensity scatter plots and parametric map
    plots (IR/TE) using the plotting API functions directly as pydra tasks.

    In HTML mode (default): NIfTI paths are passed directly to the plotting
    functions for NiiVue embedding; mrview screenshots are skipped.

    In PNG mode: Vial mask overlays are built using MrGrid + MrCat + MrMath
    pydra tasks and mrview captures screenshots as before.

    Parameters
    ----------
    output_format : str
        "html" (default) — interactive HTML output, no mrview required.
        "png"            — static matplotlib PNG, mrview used for ROI overlay.
    """
    import re as _re
    from pathlib import Path as _Path
    from phantomkit.plotting.vial_intensity import plot_vial_intensity
    from phantomkit.plotting.maps_ir import (
        plot_vial_ir_means_std as plot_ir,
    )
    from phantomkit.plotting.maps_te import (
        plot_vial_te_means_std as plot_te,
    )

    metrics_path = _Path(metrics_dir)
    xlsx_dir = metrics_path / "xlsx"
    plots_dir = metrics_path / "plots"
    fits_dir = metrics_path / "fits"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)
    vial_path = _Path(vial_dir)
    tmp_vial_dir = vial_path / "tmp"

    vial_masks_list = sorted(vial_path.glob("*.nii.gz"))  # triggers runtime fallback

    def _classify_contrast(stem: str) -> str | None:
        """Detect ADC / FA contrast types from filename stem."""
        import re as _re2
        if _re2.search(r"ADC", stem, _re2.IGNORECASE):
            return "adc"
        if _re2.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", stem):
            return "fa"
        return None

    def _intensity_range_for(contrast_type: str | None) -> str | None:
        if contrast_type == "fa":
            return "0,1"
        if contrast_type == "adc":
            return "0,0.005"
        return None

    # Resolve template_dir (parent of phantom-specific dir) from metrics_dir.
    # metrics_dir is <output_dir>/<session>/metrics/ — template_data/ is
    # resolved from the installed package location.
    import importlib.util as _ilu
    _pkg_spec = _ilu.find_spec("phantomkit")
    _pkg_dir = _Path(_pkg_spec.origin).parent if _pkg_spec else _Path(metrics_dir)
    _template_data_root = None
    for _candidate in [_pkg_dir.parent / "template_data",
                       _pkg_dir / "template_data"]:
        if _candidate.is_dir():
            _template_data_root = str(_candidate)
            break

    # Per-contrast scatter plots
    for contrast_file in contrast_files:
        contrast_path = _Path(contrast_file)
        contrast_name = contrast_path.stem
        clean_contrast_name = contrast_name.replace(".nii", "").replace(".gz", "")
        xlsx_file = str(xlsx_dir / f"{clean_contrast_name}.xlsx")
        if not _Path(xlsx_file).exists():
            continue

        # Output filename: .html or .png depending on format
        _ext = ".html" if output_format == "html" else ".png"
        output_plot = str(plots_dir / f"{contrast_name}{_ext}")

        contrast_type = _classify_contrast(contrast_path.stem)
        intensity_range_str = _intensity_range_for(contrast_type)

        if output_format == "html":
            _vial_niftis_scatter = {
                p.name.replace(".nii.gz", "").replace(".nii", ""): str(p)
                for p in vial_masks_list
            }
            workflow.add(
                python.define(plot_vial_intensity)(
                    csv_file=xlsx_file,
                    plot_type="scatter",
                    roi_image=None,
                    output=output_plot,
                    phantom=None,
                    template_dir=_template_data_root,
                    output_format="html",
                    nifti_image=str(contrast_path),
                    vial_niftis=_vial_niftis_scatter,
                ),
                name=f"scatter_{contrast_name}",
            )
        else:
            # PNG mode: build ROI overlay and capture mrview screenshot
            roi_image: File | None = None
            if vial_masks_list:
                overlay = workflow.add(
                    BuildRoiOverlay(
                        vial_masks=vial_masks_list,
                        reference_image=contrast_file,
                        prefix=contrast_name,
                        tmp_dir=tmp_vial_dir,
                    ),
                    name=f"overlay_{contrast_name}",
                )
                screenshot = workflow.add(
                    MrView(
                        image=contrast_file,
                        roi_load=overlay.out,
                        roi_colour="1,0,0",
                        roi_opacity=1.0,
                        comments=0,
                        noannotations=True,
                        fullscreen=True,
                        intensity_range=intensity_range_str,
                        capture_folder=tmp_vial_dir,
                        capture_prefix=f"{contrast_name}_roi_overlay",
                    ),
                    name=f"screenshot_{contrast_name}",
                )
                roi_image = screenshot.out_png

            workflow.add(
                python.define(plot_vial_intensity)(
                    csv_file=xlsx_file,
                    plot_type="scatter",
                    roi_image=roi_image,
                    output=output_plot,
                    phantom=None,
                    template_dir=_template_data_root,
                    output_format="png",
                ),
                name=f"scatter_{contrast_name}",
            )

    # Parametric map plots (IR and TE)
    def _matches(stem: str, token: str) -> bool:
        return bool(_re.search(rf"(?<![a-z0-9]){token}(?![a-z])", stem.lower()))

    _mapping_names = {
        "ir": ("T1_mapping", "T1_fits"),
        "te": ("T2_mapping", "T2_fits"),
    }
    # T1/MPRAGE image used as the viewer background across all map plots.
    # Falls back to the first matching contrast if no T1 is found.
    def _find_t1_bg() -> str | None:
        for f in contrast_files:
            if _re.search(r"t1|mprage", _Path(f).stem, _re.IGNORECASE):
                return str(f)
        return None

    _t1_bg = _find_t1_bg()

    for contrast_type, plot_fn in [
        ("ir", plot_ir),
        ("te", plot_te),
    ]:
        matching = [f for f in contrast_files if _matches(_Path(f).stem, contrast_type)]
        if not matching:
            continue

        _ext = ".html" if output_format == "html" else ".png"
        _plot_name, _fits_name = _mapping_names[contrast_type]
        output_plot = str(plots_dir / f"{_plot_name}{_ext}")
        _fits_output = str(fits_dir / f"{_fits_name}.csv")

        if output_format == "html":
            _vial_niftis = {
                p.name.replace(".nii.gz", "").replace(".nii", ""): str(p)
                for p in vial_masks_list
            }
            workflow.add(
                python.define(plot_fn)(
                    contrast_files=matching,
                    metric_dir=str(xlsx_dir),
                    output_file=output_plot,
                    roi_image=None,
                    output_format="html",
                    fits_output=_fits_output,
                    nifti_image=_t1_bg or str(matching[0]),
                    vial_niftis=_vial_niftis,
                    phantom=phantom_name,
                ),
                name=f"map_plot_{contrast_type}",
            )
        else:
            roi_image_arg: File | None = None
            if vial_masks_list:
                overlay = workflow.add(
                    BuildRoiOverlay(
                        vial_masks=vial_masks_list,
                        reference_image=matching[0],
                        prefix=contrast_type,
                        tmp_dir=tmp_vial_dir,
                    ),
                    name=f"overlay_{contrast_type}",
                )
                screenshot = workflow.add(
                    MrView(
                        image=matching[0],
                        roi_load=overlay.out,
                        roi_colour="1,0,0",
                        roi_opacity=1.0,
                        comments=0,
                        noannotations=True,
                        fullscreen=True,
                        capture_folder=tmp_vial_dir,
                        capture_prefix=f"roi_overlay_{contrast_type}",
                    ),
                    name=f"screenshot_{contrast_type}",
                )
                roi_image_arg = screenshot.out_png

            workflow.add(
                python.define(plot_fn)(
                    contrast_files=matching,
                    metric_dir=str(xlsx_dir),
                    output_file=output_plot,
                    roi_image=roi_image_arg,
                    output_format="png",
                    fits_output=_fits_output,
                ),
                name=f"map_plot_{contrast_type}",
            )


@python.define
def Cleanup(dirs: list[Path]) -> None:
    """Remove temporary directories."""
    import shutil
    from pathlib import Path as _Path

    for d in dirs:
        p = _Path(d)
        if p.exists():
            try:
                shutil.rmtree(p)
                logger.info("Removed: %s", p.name)
            except Exception as e:
                logger.warning("Could not remove %s: %s", p.name, e)
