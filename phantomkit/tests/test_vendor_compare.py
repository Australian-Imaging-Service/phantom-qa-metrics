"""Unit tests for phantomkit.vendor_compare and phantomkit.plotting.vendor_compare_html."""

import json
import re
import shutil
from pathlib import Path

import click
import pytest

from phantomkit.vendor_compare import (
    locate_reference, compute_vendor_vial_stats, ensure_nifti,
    infer_adc_scale, load_full_stats_from_xlsx,
)
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
    # No ADC.nii.gz written alongside vial_segmentations/ in this fixture.
    assert result["phantomkit_image"] is None


def test_locate_reference_adc_finds_phantomkit_image_when_present(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    series_dir = output_dir / "DWI_series"
    _write_adc_report(series_dir)
    (series_dir / "ADC.nii.gz").write_bytes(b"")

    result = locate_reference(output_dir, "adc")

    assert result["phantomkit_image"] == series_dir / "ADC.nii.gz"


def test_locate_reference_t1(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    _write_t1_report(output_dir)

    result = locate_reference(output_dir, "t1")

    assert result["report_html"].name == "sess_T1_mapping.html"
    assert "E" in result["vial_masks"]
    # T1/T2 never have a per-voxel map image (only per-vial curve-fit CSVs).
    assert result["phantomkit_image"] is None


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
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E", "F"])],
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
            vendor_images=[str(vendor_image)],
            vendor_labels=["Vendor"],
            vial_masks={},
            vendor_stats_list=[_fake_vendor_stats(["Z"])],  # no overlap with "E"
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
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
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
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(vials)],
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

    # Temperature dropdown: multiple options, driving both the chart's
    # Reference series and the table via VC_REF_BY_TEMP / _vcSetTemp.
    assert '<select id="vcTempSelect"' in html
    assert html.count("<option value=") > 1
    ref_by_temp_m = re.search(r"const VC_REF_BY_TEMP = (\{.*?\});", html)
    ref_by_temp = json.loads(ref_by_temp_m.group(1))
    assert len(ref_by_temp) > 1
    assert all(len(v) == len(vials) for v in ref_by_temp.values())

    # Table's Reference column is seeded from the default temperature.
    assert 'id="vc-ref-0"' in html and 'id="vc-ref-1"' in html


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
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        # no phantom / template_dir given -> no calibration reference
    )

    html = output.read_text()
    m = re.search(r"const DATASETS = (\[.*?\]);\s*const PK_DATA", html, re.DOTALL)
    datasets = json.loads(m.group(1))
    assert [d["label"] for d in datasets] == ["phantomkit", "Vendor"]
    assert '<select id="vcTempSelect"' not in html


# ── build_vendor_compare_html: multiple vendor images ───────────────────────


def _setup_multi_vendor_case(tmp_path: Path):
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E", "F"], '
        '"means": [0.70, 0.90], "stds": [0.02, 0.03]}</script></body></html>'
    )

    affine = numpy.eye(4)
    vial_masks = {}
    for vial in ("E", "F"):
        p = tmp_path / f"{vial}.nii.gz"
        nibabel.save(
            nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), str(p)
        )
        vial_masks[vial] = str(p)

    vendor_images = []
    for name in ("siemens_ADC", "ge_ADC"):
        p = tmp_path / f"{name}.nii.gz"
        nibabel.save(
            nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
            str(p),
        )
        vendor_images.append(str(p))

    return ref_html, vial_masks, vendor_images


def test_build_vendor_compare_html_multi_vendor_datasets_and_offsets(
    tmp_path: Path,
) -> None:
    ref_html, vial_masks, vendor_images = _setup_multi_vendor_case(tmp_path)
    vendor_stats_list = [
        _fake_vendor_stats(["E", "F"]),
        {
            v: {
                "mean": 0.75, "median": 0.74, "std": 0.04, "min": 0.5, "max": 1.0,
                "count": 400, "p25": 0.7, "p75": 0.8, "mean_mad": 0.03, "median_mad": 0.03,
            }
            for v in ("E", "F")
        },
    ]

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_images=vendor_images,
        vendor_labels=["VendorA", "VendorB"],
        vial_masks=vial_masks,
        vendor_stats_list=vendor_stats_list,
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
    )

    html = output.read_text()
    m = re.search(r"const DATASETS = (\[.*?\]);\s*const PK_DATA", html, re.DOTALL)
    datasets = json.loads(m.group(1))
    assert [d["label"] for d in datasets] == ["phantomkit", "VendorA", "VendorB"]

    by_label = {d["label"]: d for d in datasets}
    assert by_label["phantomkit"]["borderColor"] == "#378ADD"
    assert by_label["VendorA"]["borderColor"] == "#E6B800"
    assert by_label["VendorB"]["borderColor"] == "#D85A30"

    # Each row is offset to a distinct x position for the same vial (j=0),
    # symmetric around the vial's integer tick.
    xs = [ds["data"][0]["x"] for ds in datasets]
    assert len(set(xs)) == 3
    assert sum(xs) == pytest.approx(0.0, abs=1e-9)

    # Table gets one value+delta column pair per vendor, indexed by vendor
    # position (i) and vial position (j).
    assert 'id="vc-vendor-0-0"' in html and 'id="vc-vendor-1-0"' in html
    assert 'id="vc-pct-vendor-0-0"' in html and 'id="vc-pct-vendor-1-0"' in html
    assert "<th>VendorA</th>" in html and "<th>VendorB</th>" in html


def test_build_vendor_compare_html_mismatched_list_lengths_raises(
    tmp_path: Path,
) -> None:
    ref_html, vial_masks, vendor_images = _setup_multi_vendor_case(tmp_path)

    with pytest.raises(ValueError, match="same length"):
        build_vendor_compare_html(
            vendor_images=vendor_images,
            vendor_labels=["OnlyOneLabel"],
            vial_masks=vial_masks,
            vendor_stats_list=[_fake_vendor_stats(["E", "F"])] * 2,
            reference_html=str(ref_html),
            map_type="adc",
            output=str(tmp_path / "out.html"),
        )


def test_build_vendor_compare_html_viewer_includes_phantomkit_background(
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
    vial_masks = {"E": str(tmp_path / "E.nii.gz")}
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), vial_masks["E"],
    )
    phantomkit_image = tmp_path / "ADC.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(phantomkit_image),
    )

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        phantomkit_image=str(phantomkit_image),
    )

    html = output.read_text()
    assert '<select id="pk-bg-select"' in html
    assert ">phantomkit</option>" in html


def test_build_vendor_compare_html_viewer_omits_background_dropdown_for_single_source(
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
    vial_masks = {"E": str(tmp_path / "E.nii.gz")}
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), vial_masks["E"],
    )

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
        reference_html=str(ref_html),
        map_type="adc",
        output=str(output),
        # no phantomkit_image -> only one background candidate -> no dropdown
    )

    html = output.read_text()
    assert '<select id="pk-bg-select"' not in html


# ── ensure_nifti ─────────────────────────────────────────────────────────────


def test_ensure_nifti_normalizes_nifti_input(tmp_path: Path) -> None:
    """Always runs through mrconvert now (dtype/strides normalization for
    the viewer), even for already-NIfTI input — not a pure passthrough."""
    if shutil.which("mrconvert") is None:
        pytest.skip("mrconvert not available")

    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    affine = numpy.eye(4)
    p = tmp_path / "image.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image((numpy.random.rand(6, 6, 6) * 1000).astype("int16"), affine),
        str(p),
    )

    normalized = ensure_nifti(p, tmp_path / "tmp")

    assert normalized != p
    assert normalized.name == "image_normalized.nii.gz"
    assert normalized.exists()


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
    assert converted.name == "vendor_normalized.nii.gz"
    assert converted.exists()


# ── infer_adc_scale ──────────────────────────────────────────────────────────


def _stats_with_means(means):
    return {str(i): {"mean": m} for i, m in enumerate(means)}


def test_infer_adc_scale_raw_mm2_per_s() -> None:
    assert infer_adc_scale(_stats_with_means([0.0007, 0.0009, 0.0011])) == 1e3


def test_infer_adc_scale_already_scaled() -> None:
    assert infer_adc_scale(_stats_with_means([0.7, 0.9, 1.1])) == 1.0


def test_infer_adc_scale_integer_scaled() -> None:
    assert infer_adc_scale(_stats_with_means([700, 900, 1100])) == 1e-3


def test_infer_adc_scale_empty_defaults_to_1000() -> None:
    assert infer_adc_scale({}) == 1e3


# ── load_full_stats_from_xlsx ────────────────────────────────────────────────


def test_load_full_stats_from_xlsx(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    openpyxl = pytest.importorskip("openpyxl")  # noqa: F841 -- required by ExcelWriter

    xlsx_path = tmp_path / "ADC.xlsx"
    with pandas.ExcelWriter(xlsx_path) as writer:
        for sheet, vals in [
            ("mean", [0.0007, 0.0009]),
            ("median", [0.00068, 0.00088]),
            ("std", [0.00002, 0.00003]),
            ("count", [500, 480]),
            ("p25", [0.00065, 0.00085]),
            ("p75", [0.0007, 0.0009]),
            ("min", [0.0005, 0.0007]),
            ("max", [0.0009, 0.0011]),
            ("mean_mad", [0.00002, 0.00003]),
            ("median_mad", [0.00002, 0.00003]),
        ]:
            pandas.DataFrame({"vial": ["E", "F"], "vol0": vals}).to_excel(
                writer, sheet_name=sheet, index=False
            )

    stats = load_full_stats_from_xlsx(xlsx_path)

    assert set(stats) == {"E", "F"}
    assert stats["E"]["mean"] == pytest.approx(0.0007)
    assert stats["E"]["median"] == pytest.approx(0.00068)
    assert stats["F"]["count"] == pytest.approx(480)


# ── build_vendor_compare_html: full stats for both series (ADC) ─────────────


def test_build_vendor_compare_html_both_series_have_mean_median_when_xlsx_given(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.70], "stds": [0.02]}</script></body></html>'
    )
    xlsx_path = tmp_path / "ADC.xlsx"
    with pandas.ExcelWriter(xlsx_path) as writer:
        for sheet, val in [
            ("mean", 0.0007), ("median", 0.00068), ("std", 0.00002), ("count", 500),
            ("p25", 0.00065), ("p75", 0.0007), ("min", 0.0005), ("max", 0.0009),
            ("mean_mad", 0.00002), ("median_mad", 0.00002),
        ]:
            pandas.DataFrame({"vial": ["E"], "vol0": [val]}).to_excel(
                writer, sheet_name=sheet, index=False
            )

    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {"E": str(tmp_path / "E.nii.gz")}
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), vial_masks["E"],
    )

    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
        reference_html=str(ref_html),
        reference_xlsx=str(xlsx_path),
        map_type="adc",
        output=str(output),
        scales=[1e3],
    )

    html = output.read_text()
    m = re.search(r"const PK_DATA = (\{.*?\});", html, re.DOTALL)
    pk_data = json.loads(m.group(1))
    # Both rows now have distinct mean vs median (not collapsed to the same
    # fixed point) — row 0 is phantomkit, row 1 is vendor.
    assert pk_data["measure"]["mean"][0] != pk_data["measure"]["median"][0]
    assert "Every series responds" in html


def test_build_vendor_compare_html_pk_xlsx_scale_independent_of_vendor_scale(
    tmp_path: Path,
) -> None:
    """phantomkit's own xlsx-derived series must always be scaled by its
    fixed x1000 mm^2/s convention, never by the vendor image's own
    (independently detected) scale -- regression test for a bug where both
    series were scaled by the same `scale` value, corrupting phantomkit's
    own numbers whenever the vendor's convention differed from x1000."""
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    numpy = pytest.importorskip("numpy")
    nibabel = pytest.importorskip("nibabel")

    ref_html = tmp_path / "ref.html"
    ref_html.write_text(
        '<html><body><script id="phantomkit-data" type="application/json">'
        '{"type": "vial_intensity", "contrast_mode": "adc", "vials": ["E"], '
        '"means": [0.70], "stds": [0.02]}</script></body></html>'
    )
    xlsx_path = tmp_path / "ADC.xlsx"
    with pandas.ExcelWriter(xlsx_path) as writer:
        for sheet, val in [
            ("mean", 0.0007), ("median", 0.00068), ("std", 0.00002), ("count", 500),
            ("p25", 0.00065), ("p75", 0.0007), ("min", 0.0005), ("max", 0.0009),
            ("mean_mad", 0.00002), ("median_mad", 0.00002),
        ]:
            pandas.DataFrame({"vial": ["E"], "vol0": [val]}).to_excel(
                writer, sheet_name=sheet, index=False
            )

    affine = numpy.eye(4)
    vendor_image = tmp_path / "vendor.nii.gz"
    nibabel.save(
        nibabel.Nifti1Image(numpy.random.rand(6, 6, 6).astype("float32"), affine),
        str(vendor_image),
    )
    vial_masks = {"E": str(tmp_path / "E.nii.gz")}
    nibabel.save(
        nibabel.Nifti1Image(numpy.zeros((6, 6, 6), dtype="uint8"), affine), vial_masks["E"],
    )

    # Vendor's own detected scale is deliberately different from
    # phantomkit's fixed x1000 xlsx convention (pk_xlsx_scale). If the two
    # were ever conflated, phantomkit's row would come out scaled by this
    # value instead (~0.0000007) rather than by the correct fixed x1000
    # (~0.7).
    output = tmp_path / "report.html"
    build_vendor_compare_html(
        vendor_images=[str(vendor_image)],
        vendor_labels=["Vendor"],
        vial_masks=vial_masks,
        vendor_stats_list=[_fake_vendor_stats(["E"])],
        reference_html=str(ref_html),
        reference_xlsx=str(xlsx_path),
        map_type="adc",
        output=str(output),
        scales=[1e-3],
    )

    html = output.read_text()
    m = re.search(r"const PK_DATA = (\{.*?\});", html, re.DOTALL)
    pk_data = json.loads(m.group(1))

    # Row 0 = phantomkit (xlsx mean 0.0007 x fixed pk_xlsx_scale 1e3 = 0.7),
    # row 1 = vendor (fake stats mean 0.7 x vendor scale 1e-3 = 0.0007).
    assert pk_data["measure"]["mean"][0][0] == pytest.approx(0.7, rel=1e-6)
    assert pk_data["measure"]["mean"][1][0] == pytest.approx(0.0007, rel=1e-6)
