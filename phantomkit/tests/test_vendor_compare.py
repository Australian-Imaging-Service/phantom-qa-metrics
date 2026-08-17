"""Unit tests for phantomkit.vendor_compare and phantomkit.plotting.vendor_compare_html."""

import json
import re
import shutil
from pathlib import Path

import click
import pytest

from phantomkit.vendor_compare import locate_reference, compute_vendor_vial_stats, ensure_nifti
from phantomkit.plotting.vendor_compare_html import build_vendor_compare_html


# ── locate_reference ─────────────────────────────────────────────────────────


def _write_adc_report(series_dir: Path, name: str = "ADC.html") -> None:
    plots_dir = series_dir / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / name).write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.7], "stds": [0.02]}</script></body></html>'
    )
    vial_dir = series_dir / "vial_segmentations"
    vial_dir.mkdir(parents=True, exist_ok=True)
    (vial_dir / "E.nii.gz").write_bytes(b"")


def _write_t1_report(output_dir: Path) -> None:
    plots_dir = output_dir / "native_contrasts" / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "sess_T1_mapping.html").write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "maps_ir", "fit_results": [{"Vial": "E", "T1_ms": 800, "T1_se_ms": 20}]}'
        "</script></body></html>"
    )
    vial_dir = output_dir / "native_contrasts" / "vial_segmentations"
    vial_dir.mkdir(parents=True, exist_ok=True)
    (vial_dir / "E.nii.gz").write_bytes(b"")


def test_locate_reference_adc(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_adc_report(output_dir / "DWI_series")

    result = locate_reference(output_dir, "adc")

    assert result["report_html"].name == "ADC.html"
    assert "E" in result["vial_masks"]


def test_locate_reference_t1(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_t1_report(output_dir)

    result = locate_reference(output_dir, "t1")

    assert result["report_html"].name == "sess_T1_mapping.html"
    assert "E" in result["vial_masks"]


def test_locate_reference_unknown_map_type(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException):
        locate_reference(tmp_path, "adc2")


def test_locate_reference_no_match_raises(tmp_path: Path) -> None:
    (tmp_path / "output").mkdir()
    with pytest.raises(click.ClickException, match="No existing ADC report"):
        locate_reference(tmp_path / "output", "adc")


def test_locate_reference_multiple_matches_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_adc_report(output_dir / "series_a", name="ADC_a.html")
    _write_adc_report(output_dir / "series_b", name="ADC_b.html")

    with pytest.raises(click.ClickException, match="multiple candidate ADC reports"):
        locate_reference(output_dir, "adc")


# ── compute_vendor_vial_stats ────────────────────────────────────────────────


def test_compute_vendor_vial_stats(tmp_path: Path) -> None:
    if shutil.which("mrgrid") is None or shutil.which("mrstats") is None:
        pytest.skip("mrgrid/mrstats not available")

    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(10, 10, 10).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {}
    for vial in ("E", "F"):
        mask_path = tmp_path / f"{vial}.nii.gz"
        mask = numpy.zeros((10, 10, 10), dtype="uint8")
        mask[2:5, 2:5, 2:5] = 1
        nibabel.save(nibabel.Nifti1Image(mask, affine), str(mask_path))
        vial_masks[vial] = mask_path

    stats = compute_vendor_vial_stats(vendor_image, vial_masks, tmp_path / "tmp")

    assert set(stats) == {"E", "F"}
    for vial_stats in stats.values():
        for key in ("mean", "median", "std", "min", "max", "count", "p25", "p75"):
            assert key in vial_stats


# ── build_vendor_compare_html ────────────────────────────────────────────────


def _fake_vendor_stats(vials):
    return {
        v: {
            "mean": 0.7, "median": 0.69, "std": 0.03, "min": 0.5, "max": 0.9,
            "count": 500, "p25": 0.68, "p75": 0.72, "mean_mad": 0.02, "median_mad": 0.02,
        }
        for v in vials
    }


def test_build_vendor_compare_html(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E", "F"], '
        '"means": [0.70, 0.90], "stds": [0.02, 0.03]}</script></body></html>'
    )

    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {}
    for vial in ("E", "F"):
        p = tmp_path / f"{vial}.nii.gz"
        nibabel.save(
            nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), str(p)
        )
        vial_masks[vial] = str(p)

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_image=str(vendor_image),
        vial_masks=vial_masks,
        vendor_stats=_fake_vendor_stats(["E", "F"]),
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        phantom="SPIRIT",
    )

    assert output.exists()
    html = output.read_text()
    assert "pk-measure-btn" in html  # PK_CONTROLS_HTML present
    assert '"phantomkit"' in html
    assert '"Vendor"' in html
    assert "niivue" in html.lower()


def test_build_vendor_compare_html_no_common_vials_raises(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.70], "stds": [0.02]}</script></body></html>'
    )
    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(4, 4, 4).astype("float32"), affine),
        str(vendor_image),
    )

    with pytest.raises(ValueError, match="No vials in common"):
        build_vendor_compare_html(
            vendor_image=str(vendor_image),
            vial_masks={},
            vendor_stats=_fake_vendor_stats(["Z"]),  # no overlap with "E"
            reference_html=str(ref_html),
            map_type="adc",
            output=str(tmp_path / "out.html"),
        )


def test_build_vendor_compare_html_only_shows_plotted_vials_in_viewer(
    tmp_path: Path,
) -> None:
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.70], "stds": [0.02]}</script></body></html>'
    )
    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    # vial_masks includes "A", which isn't in the ADC comparison set — it
    # should never reach the viewer's toggle chips.
    vial_masks = {}
    for vial in ("E", "A"):
        p = tmp_path / f"{vial}.nii.gz"
        nibabel.save(
            nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), str(p)
        )
        vial_masks[vial] = str(p)

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_image=str(vendor_image),
        vial_masks=vial_masks,
        vendor_stats=_fake_vendor_stats(["E"]),
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
    )

    html = output.read_text()
    chips = re.findall(r'pk-chip-\d+[^>]*>([A-Z])<', html)
    assert chips == ["E"]


def test_build_vendor_compare_html_includes_calibration_reference(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    vials = ["E", "F", "G", "H", "I", "J", "K", "L"]
    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        + json.dumps({
            "type": "vial_intensity", "contrast_mode": "adc", "vials": vials,
            "means": [1.85, 1.61, 1.38, 1.19, 1.01, 0.86, 0.56, 0.30],
            "stds": [0.03] * 8,
        })
        + "</script></body></html>"
    )

    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {}
    for vial in vials:
        p = tmp_path / f"{vial}.nii.gz"
        nibabel.save(
            nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), str(p)
        )
        vial_masks[vial] = str(p)

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_image=str(vendor_image),
        vial_masks=vial_masks,
        vendor_stats=_fake_vendor_stats(vials),
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        phantom="SPIRIT",
        template_dir="template_data",
    )

    html = output.read_text()
    m = re.search(r"const DATASETS = (\[.*?\]);\s*const PK_DATA", html, re.DOTALL)
    datasets = json.loads(m.group(1))
    labels = [d["label"] for d in datasets]
    assert labels == ["phantomkit", "Vendor", "Reference"]

    by_label = {d["label"]: d for d in datasets}
    assert by_label["phantomkit"]["borderColor"] == "#378ADD"
    assert by_label["Vendor"]["borderColor"] == "#E6B800"
    assert by_label["Reference"]["pointBorderColor"] == "#C62828"


def test_build_vendor_compare_html_omits_reference_when_unavailable(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.70], "stds": [0.02]}</script></body></html>'
    )
    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {"E": str(tmp_path / "E.nii.gz")}
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine),
        vial_masks["E"],
    )

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_image=str(vendor_image),
        vial_masks=vial_masks,
        vendor_stats=_fake_vendor_stats(["E"]),
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        # no phantom / template_dir given -> no calibration reference
    )

    html = output.read_text()
    m = re.search(r"const DATASETS = (\[.*?\]);\s*const PK_DATA", html, re.DOTALL)
    datasets = json.loads(m.group(1))
    assert [d["label"] for d in datasets] == ["phantomkit", "Vendor"]


# ── ensure_nifti ─────────────────────────────────────────────────────────────


def test_ensure_nifti_passes_through_nifti_unchanged(tmp_path: Path) -> None:
    p = tmp_path / "image.nii.gz"
    p.write_bytes(b"")
    assert ensure_nifti(p, tmp_path / "tmp") == p


def test_ensure_nifti_converts_mif(tmp_path: Path) -> None:
    if shutil.which("mrconvert") is None:
        pytest.skip("mrconvert not available")

    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")
    import subprocess

    affine = numpy.eye(4)
    nii = tmp_path / "fake.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine), str(nii)
    )
    mif = tmp_path / "vendor.mif.gz"
    subprocess.run(["mrconvert", str(nii), str(mif), "-force"], check=True, capture_output=True)

    converted = ensure_nifti(mif, tmp_path / "conv")

    assert converted != mif
    assert converted.name == "vendor.nii.gz"
    assert converted.exists()
