"""Unit tests for phantomkit.analyses.vial_signal."""

from pathlib import Path

from fileformats.medimage import NiftiGz

from phantomkit.analyses.vial_signal import (
    GetContrastFiles,
    GetVialMasks,
    PrepareSessionPaths,
)

# ── PrepareSessionPaths ───────────────────────────────────────────────────────


def test_prepare_session_paths_creates_directories(tmp_path: Path) -> None:
    session_dir = tmp_path / "20240101"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="t1")

    out = PrepareSessionPaths(input_image=image, output_base_dir=tmp_path / "output")()

    assert out.session_name == "20240101"
    assert Path(out.tmp_dir).exists()
    assert Path(out.vial_dir).exists()
    assert Path(out.metrics_dir).exists()
    assert Path(out.images_template_space_dir).exists()
    assert str(out.scanner_space_image).endswith("TemplatePhantom_ScannerSpace.nii.gz")


# ── GetVialMasks ──────────────────────────────────────────────────────────────


def test_get_vial_masks_returns_sorted_list(tmp_path: Path) -> None:
    vials_dir = tmp_path / "VialsLabelled"
    vials_dir.mkdir()
    for stem in ["VialC", "VialA", "VialB"]:
        NiftiGz.sample(dest_dir=vials_dir, stem=stem)

    out = GetVialMasks(template_dir=tmp_path)()

    assert [Path(p).name for p in out.out] == [
        "VialA.nii.gz",
        "VialB.nii.gz",
        "VialC.nii.gz",
    ]


def test_get_vial_masks_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "VialsLabelled").mkdir()
    out = GetVialMasks(template_dir=tmp_path)()
    assert out.out == []


# ── GetContrastFiles ──────────────────────────────────────────────────────────


def test_get_contrast_files_returns_nifti_siblings(tmp_path: Path) -> None:
    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    NiftiGz.sample(dest_dir=session_dir, stem="t2")
    image = NiftiGz.sample(dest_dir=session_dir, stem="t1")
    (session_dir / "notes.txt").touch()

    out = GetContrastFiles(input_image=image)()

    assert [Path(p).name for p in out.out] == ["t1.nii.gz", "t2.nii.gz"]


def test_get_contrast_files_excludes_non_nifti(tmp_path: Path) -> None:
    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    image = NiftiGz.sample(dest_dir=session_dir, stem="scan")
    (session_dir / "report.pdf").touch()

    out = GetContrastFiles(input_image=image)()

    assert len(out.out) == 1
    assert Path(out.out[0]).name == "scan.nii.gz"


# ── Top-level workflow graph construction (no execution) ──────────────────────


def test_vial_signal_build(tmp_path: Path) -> None:
    from phantomkit.analyses.vial_signal import VialSignalAnalysis

    image = NiftiGz.sample(dest_dir=tmp_path, stem="t1")
    template_dir = tmp_path / "template"
    template_dir.mkdir()

    wf = VialSignalAnalysis(
        input_image=image,
        template_dir=template_dir,
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None


def test_vial_signal_batch_build(tmp_path: Path) -> None:
    from phantomkit.analyses.vial_signal import VialSignalAnalysisBatch

    images = [
        NiftiGz.sample(dest_dir=tmp_path / f"session{i}", stem="t1") for i in range(2)
    ]
    template_dir = tmp_path / "template"
    template_dir.mkdir()

    wf = VialSignalAnalysisBatch(
        input_images=images,
        template_dir=template_dir,
        output_base_dir=tmp_path / "output",
    )
    assert wf is not None
