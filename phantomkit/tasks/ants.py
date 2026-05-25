"""ANTs shell tasks for PhantomKit."""
from pydra.compose import shell
from pydra.utils.typing import MultiInputObj
from fileformats.medimage import NiftiGz
from fileformats.medimage_mrtrix3 import ImageIn, ImageOut
from fileformats.generic import File


@shell.define
class AntsRegistrationSyN(shell.Task["AntsRegistrationSyN.Outputs"]):
    """Rigid/affine/SyN registration via antsRegistrationSyN.sh."""

    executable = "antsRegistrationSyN.sh"

    fixed_image: NiftiGz = shell.arg(argstr="-f {fixed_image}")
    moving_image: NiftiGz = shell.arg(argstr="-m {moving_image}")
    output_prefix: str = shell.arg(argstr="-o {output_prefix}")
    dimensionality: int = shell.arg(argstr="-d {dimensionality}", default=3)
    transform_type: str = shell.arg(argstr="-t {transform_type}", default="r")
    n_threads: int = shell.arg(argstr="-n {n_threads}", default=1)

    class Outputs(shell.Outputs):
        warped: NiftiGz = shell.outarg(
            path_template="{output_prefix}Warped.nii.gz"
        )
        inverse_warped: NiftiGz = shell.outarg(
            path_template="{output_prefix}InverseWarped.nii.gz"
        )
        transform: File = shell.outarg(
            path_template="{output_prefix}0GenericAffine.mat"
        )


@shell.define
class DwiCatMulti(shell.Task["DwiCatMulti.Outputs"]):
    """Concatenate multiple DWI series with intensity matching (dwicat).

    Wrapper for ``dwicat`` that accepts a list of input images via
    ``MultiInputObj``, unlike the auto-generated ``DwiCat`` task which
    is limited to a single input.
    """

    executable = "dwicat"

    inputs: MultiInputObj[ImageIn] = shell.arg(
        position=1,
        argstr="",
        sep=" ",
        help="Two or more input DWI series to concatenate.",
    )

    class Outputs(shell.Outputs):
        out_file: ImageOut = shell.outarg(
            position=2,
            argstr="",
            path_template="out_file.mif",
            help="Concatenated output DWI series.",
        )
