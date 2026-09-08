"""Unit tests for phantomkit.pipeline."""

import json
import subprocess
from pathlib import Path

import pytest

from phantomkit.pipeline import (
    _has_classifiable_series,
    _relabel_dwi_pe_collisions,
    _stage_input,
    _validate_te_ti_header,
    scan_input_dir,
    stage_series_dir,
)
from phantomkit.dwi_processing import classify_candidates


# ── _has_classifiable_series ────────────────────────────────────────────────


def test_has_classifiable_series_true_for_mprage(tmp_path: Path) -> None:
    (tmp_path / "SEQ-A_NIF_SAG_MPRAGE").mkdir()
    assert _has_classifiable_series(tmp_path) is True


def test_has_classifiable_series_false_for_unrecognized_dirs(tmp_path: Path) -> None:
    (tmp_path / "SEQ-C_NIF_THERM1_fl2d12_e1").mkdir()
    assert _has_classifiable_series(tmp_path) is False


def test_has_classifiable_series_false_for_empty_dir(tmp_path: Path) -> None:
    assert _has_classifiable_series(tmp_path) is False


# ── scan_input_dir: TE/TI classification regex ──────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "SEQ-D_NIF_AX_T1_MAPS_TI_1000",  # existing underscore-separated style
        "SEQ-D_NIF_AX_T1_MAPS_TI_50",
        "AEP_T1map_MP2RAGE_Sag_AP-_INV2_SIM-TI1100ms",  # no-separator style
        "AEP_T1map_MP2RAGE_Sag_AP-_INV2_SIM-TI410ms",
    ],
)
def test_scan_input_dir_classifies_ti_series(tmp_path: Path, name: str) -> None:
    (tmp_path / name).mkdir()
    info = scan_input_dir(tmp_path)
    assert [d.name for d in info["ti_dirs"]] == [name]


@pytest.mark.parametrize(
    "name",
    [
        "SEQ-E_NIF_AX_T2_MAPS_TE_14_34",  # existing underscore-separated style
        "SEQ-E_NIF_AX_T2_MAPS_TE_1000",
        "AEP_T2map_CorOblHipp_RL-_SIM-TE83ms",  # no-separator style
        "knee_t2map_sag_32sl_10echo_384_SIM-TE34ms",
    ],
)
def test_scan_input_dir_classifies_te_series(tmp_path: Path, name: str) -> None:
    (tmp_path / name).mkdir()
    info = scan_input_dir(tmp_path)
    assert [d.name for d in info["te_dirs"]] == [name]


def test_scan_input_dir_does_not_misclassify_unrelated_names(tmp_path: Path) -> None:
    (tmp_path / "AEP_T1map_MP2RAGE_Sag_AP-_INV1").mkdir()
    (tmp_path / "SEQ-C_NIF_THERM1_fl2d12_e1").mkdir()
    info = scan_input_dir(tmp_path)
    assert info["ti_dirs"] == []
    assert info["te_dirs"] == []
    assert len(info["other_dirs"]) == 2


# ── _relabel_dwi_pe_collisions ──────────────────────────────────────────────


def _write_dwi_series(
    staged_dir: Path, stem: str, phase_encoding_direction: str
) -> None:
    (staged_dir / f"{stem}.nii.gz").write_bytes(b"")
    (staged_dir / f"{stem}.bvec").write_text("0\n0\n0\n")
    (staged_dir / f"{stem}.bval").write_text("0\n")
    (staged_dir / f"{stem}.json").write_text(
        json.dumps({"PhaseEncodingDirection": phase_encoding_direction})
    )


def test_relabel_dwi_pe_collisions_disambiguates_ap_pa(tmp_path: Path) -> None:
    # Same SeriesDescription for both — dcm2niix appends a bare disambiguation
    # letter to the second one, exactly as observed against real data.
    _write_dwi_series(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_PA", "j")
    _write_dwi_series(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_PAa", "j-")

    _relabel_dwi_pe_collisions(tmp_path)

    names = sorted(p.stem.replace(".nii", "") for p in tmp_path.glob("*.nii.gz"))
    assert names == ["SEQ-B_NIF_AX_DWI_b1000_AP", "SEQ-B_NIF_AX_DWI_b1000_PA"]


def test_relabel_dwi_pe_collisions_leaves_already_tagged_series_alone(
    tmp_path: Path,
) -> None:
    _write_dwi_series(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_AP", "j-")
    _write_dwi_series(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_PA", "j")

    _relabel_dwi_pe_collisions(tmp_path)

    names = sorted(p.stem.replace(".nii", "") for p in tmp_path.glob("*.nii.gz"))
    assert names == ["SEQ-B_NIF_AX_DWI_b1000_AP", "SEQ-B_NIF_AX_DWI_b1000_PA"]


def test_relabel_dwi_pe_collisions_ignores_non_ap_axis(tmp_path: Path) -> None:
    _write_dwi_series(tmp_path, "SEQ-X_NIF_AX_DWI_b1000", "i-")
    _relabel_dwi_pe_collisions(tmp_path)
    assert (tmp_path / "SEQ-X_NIF_AX_DWI_b1000.nii.gz").exists()


def test_relabel_dwi_pe_collisions_ignores_non_dwi_files(tmp_path: Path) -> None:
    # No bvec/bval — not a DWI-with-gradients acquisition, leave untouched.
    (tmp_path / "SEQ-A_NIF_SAG_MPRAGE.nii.gz").write_bytes(b"")
    (tmp_path / "SEQ-A_NIF_SAG_MPRAGE.json").write_text(
        json.dumps({"PhaseEncodingDirection": "j-"})
    )
    _relabel_dwi_pe_collisions(tmp_path)
    assert (tmp_path / "SEQ-A_NIF_SAG_MPRAGE.nii.gz").exists()


# ── _stage_input ─────────────────────────────────────────────────────────────


def test_stage_input_returns_unchanged_when_already_classifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    (input_dir / "SEQ-A_NIF_SAG_MPRAGE").mkdir(parents=True)
    output_dir.mkdir()

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dcm2niix fallback should not run when already classifiable")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)

    result = _stage_input(input_dir, output_dir)

    assert result == input_dir
    assert not (output_dir / "_staged_dicom").exists()


def test_stage_input_falls_back_to_dcm2niix_for_foreign_dicom_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    # Foreign dump: some file with no recognizable series-name subdirectory.
    (input_dir / "IM0001").write_bytes(b"fake dicom bytes")

    def _fake_dcm2niix(cmd, check, capture_output):
        assert cmd[0] == "dcm2niix"
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (out_dir / "SEQ-A_NIF_SAG_MPRAGE.nii.gz").write_bytes(b"")
        (out_dir / "SEQ-A_NIF_SAG_MPRAGE.json").write_text("{}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", _fake_dcm2niix)

    result = _stage_input(input_dir, output_dir)

    assert _has_classifiable_series(result)
    assert (output_dir / "_staged_dicom").exists()


def test_stage_input_handles_missing_dcm2niix_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "IM0001").write_bytes(b"fake dicom bytes")

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("dcm2niix not found")

    monkeypatch.setattr(subprocess, "run", _raise_not_found)

    result = _stage_input(input_dir, output_dir)

    assert result == input_dir
    assert not _has_classifiable_series(result)


def test_stage_input_handles_dcm2niix_finding_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "not_a_dicom.txt").write_text("hello")

    def _fake_dcm2niix_no_output(cmd, check, capture_output):
        return subprocess.CompletedProcess(cmd, 0)  # produces no NIfTIs

    monkeypatch.setattr(subprocess, "run", _fake_dcm2niix_no_output)

    result = _stage_input(input_dir, output_dir)

    assert result == input_dir


# ── _validate_te_ti_header ───────────────────────────────────────────────────


def _write_nii_with_json(tmp_path: Path, stem: str, header: dict) -> list:
    nii = tmp_path / f"{stem}.nii.gz"
    nii.write_bytes(b"")
    (tmp_path / f"{stem}.json").write_text(json.dumps(header))
    return [str(nii)]


def test_validate_te_ti_header_matching_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    produced = _write_nii_with_json(tmp_path, "SEQ-E_NIF_AX_T2_MAPS_TE_14", {"EchoTime": 0.014})
    _validate_te_ti_header(tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_14", produced, "te")
    assert "WARNING" not in capsys.readouterr().out


def test_validate_te_ti_header_mismatch_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    produced = _write_nii_with_json(tmp_path, "SEQ-E_NIF_AX_T2_MAPS_TE_99", {"EchoTime": 0.014})
    _validate_te_ti_header(tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_99", produced, "te")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "TE=99ms" in out
    assert "14ms" in out


def test_validate_ti_header_mismatch_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    produced = _write_nii_with_json(
        tmp_path, "SEQ-D_NIF_AX_T1_MAPS_TI_1000", {"InversionTime": 0.5}
    )
    _validate_te_ti_header(tmp_path / "SEQ-D_NIF_AX_T1_MAPS_TI_1000", produced, "ti")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "TI=1000ms" in out
    assert "500ms" in out


def test_validate_te_ti_header_no_json_is_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    nii = tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_14.nii.gz"
    nii.write_bytes(b"")
    _validate_te_ti_header(tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_14", [str(nii)], "te")
    assert capsys.readouterr().out == ""


def test_validate_te_ti_header_missing_field_is_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    produced = _write_nii_with_json(tmp_path, "SEQ-E_NIF_AX_T2_MAPS_TE_14", {})
    _validate_te_ti_header(tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_14", produced, "te")
    assert capsys.readouterr().out == ""


# ── stage_series_dir: MIF branch now exports JSON ───────────────────────────


def test_stage_series_dir_mif_produces_json_sidecar(tmp_path: Path) -> None:
    import shutil as _shutil

    if _shutil.which("mrconvert") is None:
        pytest.skip("mrconvert not available")

    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    series_dir = tmp_path / "SEQ-E_NIF_AX_T2_MAPS_TE_14"
    series_dir.mkdir()
    out_dir = tmp_path / "out"

    fake_nii = tmp_path / "fake.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((4, 4, 4), dtype="float32"), numpy.eye(4)),
        str(fake_nii),
    )
    mif_path = series_dir / "SEQ-E_NIF_AX_T2_MAPS_TE_14.mif.gz"
    subprocess.run(
        ["mrconvert", str(fake_nii), str(mif_path), "-force"],
        check=True, capture_output=True,
    )

    produced = stage_series_dir(series_dir, out_dir)

    assert len(produced) == 1
    json_path = Path(produced[0]).with_name(
        Path(produced[0]).name.replace(".nii.gz", ".json")
    )
    assert json_path.exists()


# ── classify_candidates: AP/PA header cross-check ───────────────────────────


def _make_dwi_conversion(tmp_path: Path, name: str, phase_encoding_direction: str) -> dict:
    d = tmp_path / name
    d.mkdir()
    bvec = d / "dwi.bvec"
    bval = d / "dwi.bval"
    json_path = d / "dwi.json"
    bvec.write_text("1 0 0\n0 1 0\n0 0 1\n")
    bval.write_text("1000 1000 1000\n")
    json_path.write_text(json.dumps({"PhaseEncodingDirection": phase_encoding_direction}))
    return {
        "dir": str(d),
        "conv": {"nii": str(d / "dwi.nii.gz"), "json": str(json_path),
                 "bvec": str(bvec), "bval": str(bval)},
    }


def test_classify_candidates_matching_ap_pa_no_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    entry = _make_dwi_conversion(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_AP", "j-")
    classify_candidates([entry["dir"]], {entry["dir"]: entry["conv"]})
    assert "WARNING" not in capsys.readouterr().out


def test_classify_candidates_mismatched_ap_pa_warns_but_still_classifies_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Folder name says AP, but the header's PhaseEncodingDirection ("j",
    # not "j-") actually corresponds to PA.
    entry = _make_dwi_conversion(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_AP", "j")
    result = classify_candidates([entry["dir"]], {entry["dir"]: entry["conv"]})
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "AP" in out and "PA" in out
    # Classification itself is unaffected — still goes into pending_fwd by name.
    assert entry["dir"] in result["pending_fwd"]


def test_classify_candidates_no_json_is_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    entry = _make_dwi_conversion(tmp_path, "SEQ-B_NIF_AX_DWI_b1000_AP", "j-")
    entry["conv"]["json"] = ""
    classify_candidates([entry["dir"]], {entry["dir"]: entry["conv"]})
    assert "WARNING" not in capsys.readouterr().out
