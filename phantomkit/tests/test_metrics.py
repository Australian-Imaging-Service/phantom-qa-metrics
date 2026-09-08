"""Unit tests for phantomkit.metrics."""

from pathlib import Path

from fileformats.generic import File
from fileformats.medimage import NiftiGz

from phantomkit.metrics import (
    GatherList,
    ParseMrInfoSize,
    PrepVialTransformPaths,
    TransformContrastsToTemplateSpace,
    TransformVialsToSubjectSpace,
    ExtractMetricsFromContrasts,
)

# ── ParseMrInfoSize ───────────────────────────────────────────────────────────


def test_parse_mrinfo_size_3d() -> None:
    out = ParseMrInfoSize(stdout="256 256 50")()
    assert out.is_4d is False
    assert out.is_single_slice is False
    assert out.nvols == 1


def test_parse_mrinfo_size_4d() -> None:
    out = ParseMrInfoSize(stdout="256 256 50 10")()
    assert out.is_4d is True
    assert out.is_single_slice is False
    assert out.nvols == 10


def test_parse_mrinfo_size_single_slice() -> None:
    out = ParseMrInfoSize(stdout="256 256 1")()
    assert out.is_4d is False
    assert out.is_single_slice is True


# ── GatherList ────────────────────────────────────────────────────────────────


def test_gather_list_preserves_order(tmp_path: Path) -> None:
    items = [NiftiGz.sample(dest_dir=tmp_path, stem=f"img{i}") for i in range(3)]
    out = GatherList(items=items)()
    assert [f.fspath for f in out.out] == [f.fspath for f in items]


def test_gather_list_empty() -> None:
    out = GatherList(items=[])()
    assert out.out == []


# ── PrepVialTransformPaths ────────────────────────────────────────────────────


def test_prep_vial_transform_paths(tmp_path: Path) -> None:
    mask = NiftiGz.sample(dest_dir=tmp_path, stem="VialA")
    vial_dir = tmp_path / "vials"
    out = PrepVialTransformPaths(vial_mask=mask, output_vial_dir=vial_dir)()
    assert out.vial_name == "VialA"
    assert out.vial_mask_out.fspath == mask.fspath
    assert str(out.output_path).endswith("VialA.nii.gz")
    assert Path(vial_dir).exists()


# ── Workflow graph construction (no execution) ────────────────────────────────


def test_transform_vials_to_subject_space_builds(tmp_path: Path) -> None:
    vial = NiftiGz.sample(dest_dir=tmp_path, stem="VialA")
    ref = NiftiGz.sample(dest_dir=tmp_path, stem="ref")
    affine = File.sample(dest_dir=tmp_path, stem="affine")
    wf = TransformVialsToSubjectSpace(
        vial_masks=[vial],
        reference_image=ref,
        transform_matrix=affine,
        output_vial_dir=tmp_path / "vials",
    )
    assert wf is not None


def test_extract_metrics_from_contrasts_builds_empty() -> None:
    # Empty contrast list — loop body never runs; graph trivially builds
    wf = ExtractMetricsFromContrasts(
        contrast_files=[],
        vial_masks=[],
        output_metrics_dir="/fake/metrics",
        session_name="test_session",
    )
    assert wf is not None


def test_transform_contrasts_to_template_space_builds_empty(tmp_path: Path) -> None:
    template = NiftiGz.sample(dest_dir=tmp_path, stem="template")
    affine = File.sample(dest_dir=tmp_path, stem="affine")
    wf = TransformContrastsToTemplateSpace(
        contrast_files=[],
        transform_matrix=affine,
        template_phantom=template,
        tmp_dir=tmp_path / "tmp",
        output_dir=tmp_path / "output",
    )
    assert wf is not None
