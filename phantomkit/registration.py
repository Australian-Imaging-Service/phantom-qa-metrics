"""
Pydra tasks and workflows for phantom template registration.

Registers a phantom MRI scan to a reference template using ANTs rigid SyN.
"""

import logging
from pathlib import Path

from fileformats.generic import File
from fileformats.medimage import NiftiGz
from pydra.compose import python, shell, workflow
from pydra.tasks.mrtrix3.v3_0 import MrConvert

logger = logging.getLogger(__name__)


# ============================================================================
# Shell Task: ANTs SyN registration (full SyN — not in pydra-tasks-ants yet)
# ============================================================================


@shell.define
class RegistrationSynN(shell.Task["RegistrationSynN.Outputs"]):
    """
    Wrapper for ``antsRegistrationSyN.sh`` (full SyN, not SyNQuick).

    Parameters
    ----------
    fixed_image:
        The fixed (reference/template) image.
    moving_image:
        The moving (subject) image to register.
    output_prefix:
        Prefix for all output files produced by the script.
    transform_type:
        Transform type passed to ``-t``: ``'r'`` (rigid), ``'a'`` (affine),
        ``'s'`` (SyN), etc.
    num_threads:
        Number of ITK threads.
    use_histogram_matching:
        Pass ``-j 1`` to enable histogram matching.
    """

    executable = "antsRegistrationSyN.sh"

    fixed_image: NiftiGz = shell.arg(argstr="-f", help="Fixed image.")
    moving_image: NiftiGz = shell.arg(argstr="-m", help="Moving image.")
    output_prefix: Path = shell.arg(argstr="-o", help="Output prefix.")
    transform_type: str = shell.arg(
        default="s",
        argstr="-t",
        help="Transform type (r=rigid, a=affine, s=SyN).",
    )
    num_threads: int = shell.arg(
        default=1,
        argstr="-n",
        help="Number of ITK threads.",
    )
    use_histogram_matching: int = shell.arg(
        default=0,
        argstr="-j",
        help="Use histogram matching (0 or 1).",
    )

    class Outputs(shell.Outputs):
        warped_image: NiftiGz = shell.outarg(
            path_template="{output_prefix}Warped.nii.gz",
            help="Warped moving image in fixed space.",
        )
        inverse_warped_image: NiftiGz = shell.outarg(
            path_template="{output_prefix}InverseWarped.nii.gz",
            help="Inverse-warped fixed image in moving space.",
        )
        out_matrix: File = shell.outarg(
            path_template="{output_prefix}0GenericAffine.mat",
            help="Generic affine transform matrix (.mat).",
        )


# ============================================================================
# Python Tasks
# ============================================================================


@python.define
def ParseMrStatsStdout(stdout: str) -> list[float]:
    """Parse the whitespace-separated numbers written by mrstats to stdout."""
    return [float(x) for x in stdout.strip().split()]


# ============================================================================
# Registration Workflows
# ============================================================================


@workflow.define(outputs=["warped", "transform", "inverse_warped"])
def RegisterToTemplate(
    input_image: NiftiGz,
    template_phantom: NiftiGz,
    session_name: str,
    tmp_dir: Path,
    num_threads: int = 1,
) -> tuple[NiftiGz, File, NiftiGz]:
    """
    Register the input image to the template phantom using ANTs rigid SyN.

    A single **RegistrationSynN** call produces the warped image and affine
    transform matrix.
    """
    from pathlib import Path as _Path

    tmp = _Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    reg = workflow.add(
        RegistrationSynN(
            fixed_image=template_phantom,
            moving_image=input_image,
            output_prefix=tmp / f"{session_name}_Transformed_",
            transform_type="r",
            num_threads=num_threads,
            use_histogram_matching=1,
        ),
        name="reg",
    )
    return reg.warped_image, reg.out_matrix, reg.inverse_warped_image


@workflow.define
def SaveTemplateInScannerSpace(
    inverse_warped: NiftiGz,
    output_path: Path,
) -> NiftiGz:
    """Save the template phantom warped into scanner (subject) space."""
    node = workflow.add(
        MrConvert(
            in_file=inverse_warped,
            out_file=output_path,
            quiet=True,
            force=True,
        ),
        name="convert",
    )
    return node.out_file
