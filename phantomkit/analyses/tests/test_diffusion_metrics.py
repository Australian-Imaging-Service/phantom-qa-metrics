"""Unit tests for phantomkit.analyses.diffusion_metrics."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fileformats.text import TextFile
from fileformats.medimage import NiftiGz, Bval, Bvec
from fileformats.vendor.mrtrix3.medimage import ImageFormat as Mif

from phantomkit.analyses.diffusion_metrics import (
    ADC_SHELLS,
    ComputeDifferenceMap,
    PrepareOutputDirs,
    WriteSummaryCsv,
    WriteTopupConfig,
    _AggregateAdcMaps,
    _ComputeMaskFromB0,
    _ComputeShellAdc,
    _FilterAndSaveVolumes,
    _LoadPaB0Mean,
    _UnpackGradFsl,
    _WriteAcqparams,
    _WriteEddyFiles,
    _WriteTopupIndex,
)

# ============================================================================
# Fixtures
# ============================================================================


def _make_nifti(path: Path, data: np.ndarray, affine=None) -> Path:
    """Save a numpy array as a NIfTI file and return the path."""
    if affine is None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def nifti_3d(tmp_path: Path) -> Path:
    """3-D NIfTI with a sphere of signal that decreases from center outward.

    Varying intensities are required so the 30th-percentile threshold used
    by _ComputeMaskFromB0 is below the maximum, allowing the centroid and
    signal mask to be computed correctly.
    """
    data = np.zeros((10, 10, 10), dtype=np.float32)
    cx, cy, cz = 5, 5, 5
    for i in range(10):
        for j in range(10):
            for k in range(10):
                d2 = (i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2
                if d2 <= 9:
                    # Intensity decreases with distance: 1000 at center, 700 at edge
                    data[i, j, k] = 1000.0 - np.sqrt(d2) * 100.0
    return _make_nifti(tmp_path / "b0_mean.nii.gz", data)


@pytest.fixture
def nifti_4d(tmp_path: Path) -> tuple[Path, Path, Path]:
    """4-D NIfTI (4 volumes) with bval/bvec files.

    b-values: [0, 100, 500, 1000]
    After filtering (keep b=0 and b≥500): volumes 0, 2, 3.
    """
    data = np.ones((5, 5, 5, 4), dtype=np.float32)
    nii_path = _make_nifti(tmp_path / "dwi.nii.gz", data)

    bval_path = tmp_path / "dwi.bval"
    bval_path.write_text("0 100 500 1000\n")

    bvec_path = tmp_path / "dwi.bvec"
    bvec_path.write_text("0 1 0 1\n" "0 0 1 0\n" "0 0 0 1\n")

    return nii_path, bval_path, bvec_path


@pytest.fixture
def acqparams_file(tmp_path: Path) -> Path:
    """Standard AP+PA acquisition parameters file."""
    p = tmp_path / "acqparams.txt"
    p.write_text("0 -1 0 0.033\n0  1 0 0.033\n")
    return p


# ============================================================================
# PrepareOutputDirs
# ============================================================================


def test_prepare_output_dirs_creates_all_dirs(tmp_path: Path) -> None:
    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="dwi")

    out = PrepareOutputDirs(input_image=image, output_base_dir=tmp_path / "output")()

    assert out.session_name == "session01"
    for attr in [
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
    ]:
        assert Path(getattr(out, attr)).exists(), f"{attr} not created"


def test_prepare_output_dirs_default_base(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="dwi")

    out = PrepareOutputDirs(input_image=image)()

    assert Path(out.dwi_steps_dir).exists()
    assert "output" in str(out.dwi_steps_dir)


# ============================================================================
# WriteTopupConfig
# ============================================================================


def test_write_topup_config_creates_file(tmp_path: Path) -> None:
    out = WriteTopupConfig(topup_dir=tmp_path)()
    config = Path(out.out)
    assert config.exists()
    assert config.name == "topup_custom.cnf"


def test_write_topup_config_content(tmp_path: Path) -> None:
    out = WriteTopupConfig(topup_dir=tmp_path)()
    content = Path(out.out).read_text()
    assert "warpres=" in content
    assert "miter=" in content
    assert "100" in content  # 100 non-linear iterations


# ============================================================================
# _ComputeMaskFromB0
# ============================================================================


def test_compute_mask_from_b0_creates_nifti(tmp_path: Path, nifti_3d: Path) -> None:
    # Use a large diameter (20 mm = 10 voxels at 2 mm/vox) to cover the sphere
    out = _ComputeMaskFromB0(
        b0_mean_nii=nifti_3d,
        masks_dir=tmp_path,
        phantom_diameter_mm=20.0,
        erosion_mm=1.0,
    )()
    mask_path = Path(out.out)
    assert mask_path.exists()
    assert mask_path.name == "phantom_mask_canonical.nii.gz"


def test_compute_mask_from_b0_binary_output(tmp_path: Path, nifti_3d: Path) -> None:
    out = _ComputeMaskFromB0(
        b0_mean_nii=nifti_3d,
        masks_dir=tmp_path,
        phantom_diameter_mm=20.0,
        erosion_mm=0.0,
    )()
    data = nib.load(str(out.out)).get_fdata()
    unique = np.unique(data)
    assert set(unique).issubset(
        {0.0, 1.0}
    ), f"Expected binary mask, got values: {unique}"


def test_compute_mask_from_b0_nonzero_mask(tmp_path: Path, nifti_3d: Path) -> None:
    out = _ComputeMaskFromB0(
        b0_mean_nii=nifti_3d,
        masks_dir=tmp_path,
        phantom_diameter_mm=20.0,
        erosion_mm=0.0,
    )()
    data = nib.load(str(out.out)).get_fdata()
    assert data.sum() > 0, "Mask should have at least one foreground voxel"


# ============================================================================
# _LoadPaB0Mean
# ============================================================================


def test_load_pa_b0_mean_single_3d(tmp_path: Path) -> None:
    pa_dir = tmp_path / "pa_ref"
    pa_dir.mkdir()
    data = np.full((5, 5, 5), 500.0, dtype=np.float32)
    _make_nifti(pa_dir / "pa_sbref.nii.gz", data)

    out_path = tmp_path / "pa_mean.nii.gz"
    out = _LoadPaB0Mean(pa_ref_dir=pa_dir, output_path=out_path)()

    assert Path(out.out).exists()
    result = nib.load(str(out.out)).get_fdata()
    assert result.shape == (5, 5, 5)
    assert np.allclose(result, 500.0)


def test_load_pa_b0_mean_4d_averages_volumes(tmp_path: Path) -> None:
    pa_dir = tmp_path / "pa_ref"
    pa_dir.mkdir()
    data = np.stack([np.full((4, 4, 4), float(v)) for v in [100, 200, 300]], axis=-1)
    _make_nifti(pa_dir / "pa.nii.gz", data)

    out_path = tmp_path / "pa_mean.nii.gz"
    out = _LoadPaB0Mean(pa_ref_dir=pa_dir, output_path=out_path)()

    result = nib.load(str(out.out)).get_fdata()
    assert np.allclose(result, 200.0)


def test_load_pa_b0_mean_missing_dir_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_pa"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _LoadPaB0Mean(pa_ref_dir=empty_dir, output_path=tmp_path / "out.nii.gz")()


# ============================================================================
# _WriteAcqparams
# ============================================================================


def test_write_acqparams_creates_file(tmp_path: Path) -> None:
    out = _WriteAcqparams(topup_dir=tmp_path, readout_time=0.033)()
    p = Path(out.out)
    assert p.exists()
    assert p.name == "acqparams.txt"


def test_write_acqparams_content(tmp_path: Path) -> None:
    out = _WriteAcqparams(topup_dir=tmp_path, readout_time=0.05)()
    lines = Path(out.out).read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 -1 0")
    assert lines[1].startswith("0  1 0")
    assert "0.05" in lines[0]
    assert "0.05" in lines[1]


# ============================================================================
# _WriteTopupIndex
# ============================================================================


def test_write_topup_index_length(tmp_path: Path, nifti_4d) -> None:
    nii_path, _, _ = nifti_4d
    out = _WriteTopupIndex(dwi_nii=nii_path)()
    assert out.out == [1, 1, 1, 1]


def test_write_topup_index_all_ones(tmp_path: Path) -> None:
    data = np.ones((3, 3, 3, 7), dtype=np.float32)
    nii_path = _make_nifti(tmp_path / "dwi7.nii.gz", data)
    out = _WriteTopupIndex(dwi_nii=nii_path)()
    assert out.out == [1] * 7


# ============================================================================
# _FilterAndSaveVolumes
# ============================================================================


def test_filter_and_save_volumes_keeps_b0_and_high_shells(
    tmp_path: Path, nifti_4d
) -> None:
    nii_path, bval_path, bvec_path = nifti_4d
    grad_fsl = (bvec_path, bval_path)

    out = _FilterAndSaveVolumes(
        dwi_nii=nii_path,
        fslgrad=grad_fsl,
        output_dir=tmp_path,
        min_bval=500.0,
    )()

    kept_bvals = np.loadtxt(str(out.bvals_file))
    # b=0 and b=500, b=1000 → 3 volumes kept, b=100 dropped
    assert len(kept_bvals) == 3
    assert 0 in kept_bvals
    assert 100 not in kept_bvals
    assert 500 in kept_bvals
    assert 1000 in kept_bvals


def test_filter_and_save_volumes_output_nifti_shape(tmp_path: Path, nifti_4d) -> None:
    nii_path, bval_path, bvec_path = nifti_4d
    out = _FilterAndSaveVolumes(
        dwi_nii=nii_path,
        fslgrad=(bvec_path, bval_path),
        output_dir=tmp_path,
    )()
    filtered = nib.load(str(out.filtered_nii))
    assert filtered.shape[-1] == 3  # 3 kept volumes


def test_filter_and_save_volumes_creates_files(tmp_path: Path, nifti_4d) -> None:
    nii_path, bval_path, bvec_path = nifti_4d
    out = _FilterAndSaveVolumes(
        dwi_nii=nii_path,
        fslgrad=(bvec_path, bval_path),
        output_dir=tmp_path,
    )()
    assert Path(out.filtered_nii).exists()
    assert Path(out.bvals_file).exists()
    assert Path(out.bvecs_file).exists()


# ============================================================================
# _WriteEddyFiles
# ============================================================================


def test_write_eddy_files_creates_index(
    tmp_path: Path, nifti_4d, acqparams_file: Path
) -> None:
    nii_path, bval_path, bvec_path = nifti_4d
    eddy_dir = tmp_path / "eddy"
    eddy_dir.mkdir()

    out = _WriteEddyFiles(
        dwi_nii=nii_path,
        fslgrad=(bvec_path, bval_path),
        topup_acqparams=acqparams_file,
        eddy_dir=eddy_dir,
    )()

    index_content = Path(out.index_file).read_text().strip()
    assert index_content == "1 1 1 1"  # 4 volumes → 4 ones


def test_write_eddy_files_acqparams_ap_only(
    tmp_path: Path, nifti_4d, acqparams_file: Path
) -> None:
    nii_path, bval_path, bvec_path = nifti_4d
    eddy_dir = tmp_path / "eddy"
    eddy_dir.mkdir()

    out = _WriteEddyFiles(
        dwi_nii=nii_path,
        fslgrad=(bvec_path, bval_path),
        topup_acqparams=acqparams_file,
        eddy_dir=eddy_dir,
    )()

    # Only the AP row (first line) should be written
    lines = Path(out.acqparams_file).read_text().strip().splitlines()
    assert len(lines) == 1
    assert "-1" in lines[0]


# ============================================================================
# _UnpackGradFsl
# ============================================================================


def test_unpack_grad_fsl_separates_tuple() -> None:
    bvec = Bvec.sample()
    bval = Bval.sample()

    out = _UnpackGradFsl(grad_fsl=(bvec, bval))()

    assert out.bvecs_file != bvec
    assert out.bvecs_file.hash() == bvec.hash()


# ============================================================================
# _ComputeShellAdc
# ============================================================================


def test_compute_shell_adc_formula(tmp_path: Path) -> None:
    """ADC = ln(S0 / S_b) / b for uniform images."""
    s0_data = np.full((4, 4, 4), 1000.0, dtype=np.float32)
    sb_data = np.full((4, 4, 4), 500.0, dtype=np.float32)
    b0_path = _make_nifti(tmp_path / "b0.nii.gz", s0_data)
    sb_path = _make_nifti(tmp_path / "shell.nii.gz", sb_data)
    adc_path = tmp_path / "adc.nii.gz"

    out = _ComputeShellAdc(
        shell_mean_nii=sb_path,
        b0_mean_nii=b0_path,
        b_value=1000.0,
        output_path=adc_path,
    )()

    adc = nib.load(str(out.out)).get_fdata()
    expected = np.log(1000.0 / 500.0) / 1000.0
    assert np.allclose(adc, expected, rtol=1e-4)


def test_compute_shell_adc_output_exists(tmp_path: Path) -> None:
    data = np.ones((3, 3, 3), dtype=np.float32) * 100
    b0_path = _make_nifti(tmp_path / "b0.nii.gz", data)
    sb_path = _make_nifti(tmp_path / "sb.nii.gz", data)
    adc_path = tmp_path / "adc.nii.gz"

    out = _ComputeShellAdc(
        shell_mean_nii=sb_path, b0_mean_nii=b0_path, b_value=500.0, output_path=adc_path
    )()

    assert Path(out.out).exists()


# ============================================================================
# _AggregateAdcMaps
# ============================================================================


def _make_adc_and_mask(tmp_path: Path) -> tuple[list[Path], Path]:
    """Create two uniform ADC maps and a full mask."""
    mask_data = np.ones((4, 4, 4), dtype=np.uint8)
    mask_path = _make_nifti(tmp_path / "mask.nii.gz", mask_data.astype(np.float32))

    adc_files = []
    for i, val in enumerate([1e-3, 2e-3]):
        adc_data = np.full((4, 4, 4), val, dtype=np.float32)
        p = _make_nifti(tmp_path / f"adc_{i}.nii.gz", adc_data)
        adc_files.append(p)
    return adc_files, mask_path


def test_aggregate_adc_maps_creates_output(tmp_path: Path) -> None:
    adc_files, mask_path = _make_adc_and_mask(tmp_path)
    out_dir = tmp_path / "adc_out"
    out_dir.mkdir()

    out = _AggregateAdcMaps(
        shell_adc_files=adc_files,
        phantom_mask=mask_path,
        output_dir=out_dir,
        label="test",
    )()

    assert Path(out.adc_map).exists()
    assert Path(out.adc_map).name == "ADC_test.nii.gz"


def test_aggregate_adc_maps_correct_mean(tmp_path: Path) -> None:
    adc_files, mask_path = _make_adc_and_mask(tmp_path)
    out_dir = tmp_path / "adc_out"
    out_dir.mkdir()

    out = _AggregateAdcMaps(
        shell_adc_files=adc_files,
        phantom_mask=mask_path,
        output_dir=out_dir,
        label="test",
    )()

    # Mean of [1e-3, 2e-3] = 1.5e-3
    assert np.isclose(out.stats_row["mean_adc"], 1.5e-3, rtol=1e-4)


def test_aggregate_adc_maps_stats_row_keys(tmp_path: Path) -> None:
    adc_files, mask_path = _make_adc_and_mask(tmp_path)
    out_dir = tmp_path / "adc_out"
    out_dir.mkdir()

    out = _AggregateAdcMaps(
        shell_adc_files=adc_files,
        phantom_mask=mask_path,
        output_dir=out_dir,
        label="mytest",
    )()

    assert out.stats_row["label"] == "mytest"
    assert {"mean_adc", "std_adc", "min_adc", "max_adc"}.issubset(out.stats_row)


# ============================================================================
# ComputeDifferenceMap
# ============================================================================


def test_compute_difference_map_subtraction(tmp_path: Path) -> None:
    before = np.full((4, 4, 4), 1.0, dtype=np.float32)
    after = np.full((4, 4, 4), 3.0, dtype=np.float32)
    before_path = _make_nifti(tmp_path / "before.nii.gz", before)
    after_path = _make_nifti(tmp_path / "after.nii.gz", after)
    out_path = tmp_path / "diff.nii.gz"

    out = ComputeDifferenceMap(
        adc_before=before_path, adc_after=after_path, output_path=out_path
    )()

    diff = nib.load(str(out.diff_map)).get_fdata()
    assert np.allclose(diff, 2.0)


def test_compute_difference_map_output_exists(tmp_path: Path) -> None:
    data = np.ones((3, 3, 3), dtype=np.float32)
    a = _make_nifti(tmp_path / "a.nii.gz", data)
    b = _make_nifti(tmp_path / "b.nii.gz", data * 2)

    out = ComputeDifferenceMap(
        adc_before=a, adc_after=b, output_path=tmp_path / "d.nii.gz"
    )()
    assert Path(out.diff_map).exists()


# ============================================================================
# WriteSummaryCsv
# ============================================================================


def test_write_summary_csv_creates_file(tmp_path: Path) -> None:
    rows = [
        {
            "label": "step0",
            "mean_adc": 1e-3,
            "std_adc": 1e-4,
            "min_adc": 0.5e-3,
            "max_adc": 1.5e-3,
        },
        {
            "label": "step1",
            "mean_adc": 1.1e-3,
            "std_adc": 1e-4,
            "min_adc": 0.6e-3,
            "max_adc": 1.6e-3,
        },
    ]
    csv_path = tmp_path / "summary.csv"

    out = WriteSummaryCsv(
        stats_rows=rows, csv_path=csv_path, session_name="sess01", pipeline="cumulative"
    )()

    assert Path(out.out).exists()


def test_write_summary_csv_columns(tmp_path: Path) -> None:
    import csv

    rows = [
        {
            "label": "step0",
            "mean_adc": 1e-3,
            "std_adc": 0.0,
            "min_adc": 1e-3,
            "max_adc": 1e-3,
        },
    ]
    csv_path = tmp_path / "summary.csv"
    out = WriteSummaryCsv(
        stats_rows=rows, csv_path=csv_path, session_name="mysess", pipeline="isolated"
    )()

    with open(out.out) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
    assert "Dataset" in header
    assert "Pipeline" in header
    assert "Step" in header
    assert "Mean_ADC" in header


def test_write_summary_csv_row_values(tmp_path: Path) -> None:
    import csv

    rows = [
        {
            "label": "s0",
            "mean_adc": 2e-3,
            "std_adc": 0.0,
            "min_adc": 2e-3,
            "max_adc": 2e-3,
        },
    ]
    out = WriteSummaryCsv(
        stats_rows=rows,
        csv_path=tmp_path / "out.csv",
        session_name="s001",
        pipeline="cumulative",
    )()

    with open(out.out) as f:
        data = list(csv.DictReader(f))
    assert data[0]["Dataset"] == "s001"
    assert data[0]["Pipeline"] == "cumulative"
    assert data[0]["Step"] == "s0"


# ============================================================================
# Workflow graph construction (no execution)
# ============================================================================


def _dummy_file(tmp_path: Path, name: str = "dummy.nii.gz") -> Path:
    """Create an empty placeholder file."""
    p = tmp_path / name
    p.touch()
    return p


def test_create_phantom_mask_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import CreatePhantomMask

    wf = CreatePhantomMask(dwi_mif=Mif.sample(), masks_dir=tmp_path)
    assert wf is not None


def test_prepare_topup_data_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import PrepareTopupData

    wf = PrepareTopupData(
        dwi_mif=Mif.sample(),
        pa_ref_dir=tmp_path,
        topup_dir=tmp_path,
    )
    assert wf is not None


def test_apply_topup_to_dwi_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import ApplyTopupToDwi

    wf = ApplyTopupToDwi(
        dwi_mif=Mif.sample(),
        topup_fieldcoef=NiftiGz.sample(),
        topup_movpar=TextFile.sample(),
        acqparams_file=TextFile.sample(),
        topup_dir=tmp_path,
    )
    assert wf is not None


def test_filter_bvalue_shells_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import FilterBvalueShells

    wf = FilterBvalueShells(dwi_mif=Mif.sample(), output_dir=tmp_path)
    assert wf is not None


def test_prepare_eddy_inputs_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import PrepareEddyInputs

    wf = PrepareEddyInputs(
        dwi_mif=Mif.sample(),
        topup_acqparams=TextFile.sample(),
        eddy_dir=tmp_path,
    )
    assert wf is not None


def test_convert_eddy_output_to_mif_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import ConvertEddyOutputToMif

    bvecs = Bvec.sample()

    wf = ConvertEddyOutputToMif(
        eddy_nii=NiftiGz.sample(),
        rotated_bvecs=Path(bvecs),
        bvals_file=bvecs.b_values_file,
        output_mif=tmp_path / "out.mif",
    )
    assert wf is not None


def test_compute_adc_maps_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import ComputeAdcMaps

    wf = ComputeAdcMaps(
        dwi_mif=Mif.sample(),
        phantom_mask=NiftiGz.sample(),
        output_dir=tmp_path,
        label="test",
        shells=[1000.0, 2000.0],
    )
    assert wf is not None


def test_topup_correction_step_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import TopupCorrectionStep

    wf = TopupCorrectionStep(
        dwi_mif=Mif.sample(),
        pa_ref_dir=tmp_path,
        topup_dir=tmp_path,
    )
    assert wf is not None


def test_eddy_correction_step_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import EddyCorrectionStep

    wf = EddyCorrectionStep(
        dwi_mif=Mif.sample(),
        phantom_mask=Mif.sample(seed=0),
        topup_fieldcoef=Mif.sample(seed=1),
        topup_movpar=TextFile.sample(),
        topup_acqparams=TextFile.sample(),
        eddy_dir=tmp_path,
    )
    assert wf is not None


def test_cumulative_pipeline_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import CumulativeDwiPipeline

    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="dwi")

    bvecs = Bvec.sample()

    wf = CumulativeDwiPipeline(
        input_image=image,
        bvecs_file=Path(bvecs),
        bvals_file=bvecs.b_values_file,
        pa_ref_dir=tmp_path,
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None


def test_isolated_pipeline_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import IsolatedDwiPipeline

    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="dwi")

    bvecs = Bvec.sample()

    wf = IsolatedDwiPipeline(
        input_image=image,
        bvecs_file=Path(bvecs),
        bvals_file=bvecs.b_values_file,
        pa_ref_dir=tmp_path,
        phantom_mask=Mif.sample(),
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None


def test_diffusion_metrics_analysis_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import DiffusionMetricsAnalysis

    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="dwi")

    bvecs = Bvec.sample()

    wf = DiffusionMetricsAnalysis(
        input_image=image,
        bvecs_file=Path(bvecs),
        bvals_file=bvecs.b_values_file,
        pa_ref_dir=tmp_path,
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None


def test_diffusion_metrics_analysis_batch_build(tmp_path: Path) -> None:
    from phantomkit.analyses.diffusion_metrics import DiffusionMetricsAnalysisBatch

    images = [
        NiftiGz.sample(dest_dir=tmp_path / f"sess{i}", stem="dwi") for i in range(2)
    ]

    wf = DiffusionMetricsAnalysisBatch(
        input_images=images,
        bvals_files=[Bval.sample() for i in range(2)],
        bvecs_files=[Bvec.sample() for i in range(2)],
        pa_ref_dirs=[tmp_path / f"pa{i}" for i in range(2)],
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None


def test_compute_adc_maps_varying_shells_build(tmp_path: Path) -> None:
    """ComputeAdcMaps builds successfully with different shell counts."""
    from phantomkit.analyses.diffusion_metrics import ComputeAdcMaps

    for shells in ([1000.0], [500.0, 1000.0, 2000.0], ADC_SHELLS):
        wf = ComputeAdcMaps(
            dwi_mif=Mif.sample(),
            phantom_mask=NiftiGz.sample(),
            output_dir=tmp_path,
            label="test",
            shells=shells,
        )
        assert wf is not None
