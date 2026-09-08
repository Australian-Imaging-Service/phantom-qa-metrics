# flake8: noqa: E501
import logging
import os
import tempfile
import typing as ty
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# XNAT Container Service integration-test fixtures
#
# Generic, pipeline-agnostic fixtures mirroring the proven pattern used in
# Australian-Imaging-Service/pipelines' own root conftest.py — nothing here
# is specific to phantomkit's own pipeline, so any future XNAT CS test can
# reuse these unmodified.
# ---------------------------------------------------------------------------

logger = logging.getLogger("phantomkit")

test_data_dir = Path(__file__).parent / "tests" / "data"

TEST_SUBJECT_LABEL = "TESTSUBJ"
TEST_SESSION_LABEL = "TESTSUBJ_01"


@pytest.fixture(scope="session")
def run_prefix() -> str:
    "A datetime string used to avoid stale data left over from previous test runs"
    return datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")


@pytest.fixture(scope="session")
def xnat_connect() -> ty.Any:
    import xnat4tests

    xnat4tests.start_xnat()
    yield xnat4tests.connect


@pytest.fixture(scope="session")
def xnat_repository(run_prefix: str, xnat_connect: ty.Any) -> ty.Any:
    import xnat4tests
    from frametree.xnat import Xnat

    config = xnat4tests.Config()

    repository = Xnat(
        server=config.xnat_uri,
        user=config.xnat_user,
        password=config.xnat_password,
        cache_dir=tempfile.mkdtemp(),
    )

    # Stash a project prefix in the repository object
    repository.__annotations__["run_prefix"] = run_prefix

    yield repository


@pytest.fixture
def cli_runner() -> ty.Callable[..., ty.Any]:
    def invoke(*args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        runner = CliRunner()
        return runner.invoke(*args, catch_exceptions=catch_cli_exceptions, **kwargs)

    return invoke


def make_project_id(dataset_name: str, run_prefix: ty.Optional[str] = None) -> str:
    return (run_prefix if run_prefix else "") + dataset_name


def upload_test_dataset_to_xnat(
    project_id: str, source_data_dir: Path, xnat_connect: ty.Any
) -> None:
    """Upload one test dataset (one scan dir per source, one resource dir
    per XNAT resource) to a freshly-created XNAT project/subject/session.

    For each scan directory under ``source_data_dir``: if a ``DICOM/``
    resource subdirectory exists, the scan's XNAT type/id are read from
    that DICOM's headers (SeriesDescription/SeriesNumber); otherwise the
    scan directory's own name becomes the type directly (used for NIfTI-only
    test data, where there's no DICOM header to read a type from).
    """
    from frametree.core.utils import varname2path
    from fileformats.application import Dicom
    import xnat as xnat_pkg

    with xnat_connect() as login:
        login.put(f"/data/archive/projects/{project_id}")

    with xnat_connect() as login:
        xproject = login.projects[project_id]
        xclasses = login.classes
        xsubject = xclasses.SubjectData(label=TEST_SUBJECT_LABEL, parent=xproject)
        xsession = xclasses.MrSessionData(label=TEST_SESSION_LABEL, parent=xsubject)
        for test_scan_dir in source_data_dir.iterdir():
            if test_scan_dir.name.startswith("."):
                continue
            scan_id = test_scan_dir.stem
            dicom_resource = test_scan_dir / "DICOM"
            if dicom_resource.exists():
                mdata = Dicom(next(dicom_resource.iterdir())).metadata
                scan_id = mdata["SeriesNumber"]
                scan_type = mdata["SeriesDescription"]
            else:
                scan_id = test_scan_dir.stem
                scan_type = varname2path(scan_id)
            xscan = xclasses.MrScanData(id=scan_id, type=scan_type, parent=xsession)

            for resource_path in test_scan_dir.iterdir():
                if not resource_path.is_dir():
                    continue
                xresource = xscan.create_resource(resource_path.stem)
                xresource.upload_dir(resource_path, method="tar_file")

        try:
            login.put(f"/data/experiments/{xsession.id}?pullDataFromHeaders=true")
        except xnat_pkg.exceptions.XNATResponseError as e:
            logger.warning(
                f"Failed to pull metadata from DICOM headers for session {xsession.id} "
                f"with error: {e}"
            )


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo[ty.Any]) -> None:
        if call.excinfo is not None:
            raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
        raise excinfo.value

    catch_cli_exceptions = False
else:
    catch_cli_exceptions = True
