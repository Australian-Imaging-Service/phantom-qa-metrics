import json
import typing as ty
from pathlib import Path

from pydra2app.core.cli import make
from pydra2app.xnat import XnatApp
from frametree.core.utils import show_cli_trace
from frametree.xnat import Xnat
from pydra2app.xnat.deploy import install_and_launch_xnat_cs_command

from conftest import upload_test_dataset_to_xnat, test_data_dir

PKG_DIR = Path(__file__).parent.parent

SPEC_PATH = PKG_DIR / "specs" / "phantomkit" / "preprocess.yaml"

SKIP_BUILD = False

# The acquisition protocol this test exercises is fixed (confirmed against a
# real scan session): DWI is always rpe_all (both AP and PA are full
# multi-volume acquisitions, never a b0-only RPE), and native-contrast QC
# always has exactly 1 MPRAGE + 12 TI scans (T1 mapping) + 8 TE scans
# (T2 mapping). Every source below is therefore always provided -- there is
# no "unset source" case to pass as "" here (see PhantomKitXnatWorkflow's
# module docstring for why every FileSet-typed input is a concrete,
# non-Optional type given this fixed protocol).
PHANTOM = "SPIRIT"

TI_VALUES = [50, 100, 150, 250, 500, 1000, 1500, 2000, 3000, 5000, 7500, 9970]
TE_VALUES = [14, 20, 40, 80, 160, 320, 640, 1000]


def test_phantom_qc_preprocess_app(
    run_prefix: str,
    xnat_connect: ty.Any,
    xnat_repository: Xnat,
    cli_runner: ty.Callable[..., ty.Any],
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    build_dir.mkdir(exist_ok=True, parents=True)

    project_id = f"{run_prefix}phantomkitpreprocess"

    test_data = test_data_dir / "specs" / "phantom-qc" / "preprocess"
    upload_test_dataset_to_xnat(project_id, test_data, xnat_connect)

    # No license installation needed -- phantomkit's pipeline uses
    # mrtrix3/FSL/ANTs/dcm2niix only, never FreeSurfer/FastSurfer.
    xnat_repository.define_frameset(project_id)

    if SKIP_BUILD:
        build_arg = "--generate-only"
    else:
        build_arg = "--build"

    result = cli_runner(
        make,
        [
            "xnat",
            str(SPEC_PATH),
            "--build-dir",
            str(build_dir),
            build_arg,
            # template_data/ is a top-level subdirectory of the repo root,
            # matching pydra2app's <resources-dir>/<resource-name>/
            # convention directly -- becomes the "template_data" resource
            # declared in preprocess.yaml, mounted at
            # /phantomkit_resources/template_data.
            "--resources-dir",
            str(PKG_DIR),
            "--for-localhost",
            "--use-local-packages",
            "--raise-errors",
        ],
    )

    assert result.exit_code == 0, show_cli_trace(result)

    image_spec = XnatApp.load(SPEC_PATH)

    # Every source's match value is just its own scan type -- the test data
    # under tests/data/specs/phantom-qc/preprocess/ names each scan
    # directory to match its source directly (no DICOM headers involved),
    # matching the DWI test's own NIfTI-based approach in the pipelines repo.
    command_inputs = {
        "T1w": "T1w",
        "DWI": "DWI",
        "RPE": "RPE",
        **{f"TI_{v}": f"TI_{v}" for v in TI_VALUES},
        **{f"TE_{v}": f"TE_{v}" for v in TE_VALUES},
        "Phantom": PHANTOM,
    }

    with xnat_connect() as xlogin:
        for command_obj in image_spec.commands:
            with open(build_dir / "xnat_commands" / (command_obj.name + ".json")) as f:
                command_json = json.load(f)
            command_json["name"] = command_json["label"] = (
                image_spec.name + command_obj.name + run_prefix
            )

            test_xsession = next(iter(xlogin.projects[project_id].experiments.values()))

            inputs_json = dict(command_inputs)
            inputs_json["pydra2app_flags"] = (
                "--worker debug "
                "--work /work "  # NB: work dir moved inside container due to file-locking issue on some mounted volumes (see https://github.com/tox-dev/py-filelock/issues/147)
                "--dataset-name default "
                "--logger frametree debug "
                "--logger frametree-xnat debug "
                "--logger pydra2app debug "
                "--logger pydra2app-xnat debug "
            )

            workflow_id, status, out_str = install_and_launch_xnat_cs_command(
                command_json=command_json,
                project_id=project_id,
                session_id=test_xsession.id,
                inputs=inputs_json,
                xlogin=xlogin,
                timeout=30000,
            )
            assert status == "Complete", f"Workflow {workflow_id} failed.\n{out_str}"
