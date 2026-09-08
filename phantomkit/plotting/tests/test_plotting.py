"""Unit tests for phantomkit.plotting functions."""

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# All vial labels expected by maps_ir / maps_te (12 vials)
_VIALS = ["S", "D", "P", "M", "C", "N", "B", "T", "A", "R", "O", "Q"]

# Realistic-ish signal values: higher-intensity vials (A,O,Q) get larger values
_MEANS = {
    "S": 200,
    "D": 400,
    "P": 380,
    "M": 600,
    "C": 900,
    "N": 950,
    "B": 1200,
    "T": 1100,
    "A": 2800,
    "R": 2600,
    "O": 3000,
    "Q": 2900,
}


def _write_mean_std_csvs(metrics_dir: Path, base_name: str, scale: float = 1.0) -> None:
    """Write a per-contrast xlsx with all 8 metric sheets."""
    means = np.array([_MEANS[v] * scale for v in _VIALS], dtype=float)
    stds = means * 0.05
    counts = np.full(len(_VIALS), 1000.0)
    p25 = means * 0.92
    p75 = means * 1.08
    mins = means * 0.80
    maxs = means * 1.20
    medians = means * 1.01

    xlsx_path = metrics_dir / f"{base_name}.xlsx"
    mean_mad   = means * 0.04   # ~4% of mean as mean MAD
    median_mad = means * 0.035  # ~3.5% as median MAD

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, vals in [
            ("mean", means), ("std", stds), ("median", medians),
            ("count", counts), ("p25", p25), ("p75", p75),
            ("min", mins), ("max", maxs),
            ("mean_mad", mean_mad), ("median_mad", median_mad),
        ]:
            pd.DataFrame({"vial": _VIALS, "vol0": vals}).to_excel(
                writer, sheet_name=sheet_name, index=False
            )


def _write_vial_intensity_csv(path: Path, n_vols: int = 1) -> None:
    """Write a multi-volume mean CSV for plot_vial_intensity."""
    data = {"vial": _VIALS}
    for v in range(n_vols):
        data[f"vol{v}"] = [_MEANS[vial] * (1 + v * 0.1) for vial in _VIALS]
    pd.DataFrame(data).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# vial_intensity
# ---------------------------------------------------------------------------


class TestPlotVialIntensity:
    """Tests for phantomkit.plotting.vial_intensity.plot_vial_intensity."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from phantomkit.plotting.vial_intensity import plot_vial_intensity

        self.fn = plot_vial_intensity
        self.tmp = tmp_path
        self.mean_csv = tmp_path / "mean.csv"
        self.std_csv = tmp_path / "std.csv"
        _write_vial_intensity_csv(self.mean_csv, n_vols=1)
        _write_vial_intensity_csv(self.std_csv, n_vols=1)  # reuse as std

    def test_scatter_returns_html_path(self) -> None:
        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="scatter",
            output=str(self.tmp / "out.html"),
        )
        assert out.endswith(".html")
        assert os.path.exists(out)

    def test_line_plot(self) -> None:
        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="line",
            output=str(self.tmp / "out.html"),
        )
        assert os.path.exists(out)

    def test_bar_plot(self) -> None:
        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="bar",
            output=str(self.tmp / "out.html"),
        )
        assert os.path.exists(out)

    def test_with_std_csv(self) -> None:
        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="scatter",
            std_csv=str(self.std_csv),
            output=str(self.tmp / "out.html"),
        )
        assert os.path.exists(out)

    def test_with_annotate(self) -> None:
        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="scatter",
            std_csv=str(self.std_csv),
            annotate=True,
            output=str(self.tmp / "out.html"),
        )
        assert os.path.exists(out)

    def test_multivol(self) -> None:
        mean_csv = self.tmp / "mean4d.csv"
        _write_vial_intensity_csv(mean_csv, n_vols=3)
        out = self.fn(
            csv_file=str(mean_csv),
            plot_type="line",
            output=str(self.tmp / "out.html"),
        )
        assert os.path.exists(out)

    def test_with_roi_image(self, tmp_path) -> None:
        import matplotlib.pyplot as plt

        roi_png = tmp_path / "roi.png"
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        ax.plot([0, 1], [0, 1])
        fig.savefig(str(roi_png))
        plt.close(fig)

        out = self.fn(
            csv_file=str(self.mean_csv),
            plot_type="scatter",
            roi_image=str(roi_png),
            output=str(tmp_path / "out.html"),
        )
        assert os.path.exists(out)

    def test_invalid_csv_too_few_columns(self, monkeypatch) -> None:
        # Patch detect_separator to return comma so pandas sees one real column.
        from phantomkit.plotting import vial_intensity

        monkeypatch.setattr(vial_intensity, "detect_separator", lambda _: ",")
        bad = self.tmp / "bad.csv"
        bad.write_text("vial\n" + "\n".join(_VIALS))
        with pytest.raises(ValueError, match="at least two columns"):
            self.fn(csv_file=str(bad), plot_type="scatter")


# ---------------------------------------------------------------------------
# maps_ir helpers
# ---------------------------------------------------------------------------


class TestMapsIrHelpers:
    """Tests for pure helper functions in maps_ir."""

    def test_extract_numeric(self) -> None:
        from phantomkit.plotting.maps_ir import extract_numeric

        assert extract_numeric("IR_500") == 500
        assert extract_numeric("contrast_100_extra_200") == 200
        assert extract_numeric("no_numbers") is None

    def test_inv_rec_zero_ti(self) -> None:
        from phantomkit.plotting.maps_ir import inv_rec

        # At TI=0: |S0*(1-2)| = S0
        assert inv_rec(0, 1000, 800) == pytest.approx(1000)

    def test_inv_rec_large_ti(self) -> None:
        from phantomkit.plotting.maps_ir import inv_rec

        # At very large TI the signal approaches S0
        assert inv_rec(1e9, 1000, 800) == pytest.approx(1000, rel=1e-3)

    def test_calc_r2_perfect_fit(self) -> None:
        from phantomkit.plotting.maps_ir import calc_r2

        y = np.array([1.0, 2.0, 3.0])
        assert calc_r2(y, y) == pytest.approx(1.0)

    def test_calc_r2_mean_fit(self) -> None:
        from phantomkit.plotting.maps_ir import calc_r2

        y = np.array([1.0, 2.0, 3.0])
        assert calc_r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)

    def test_find_xlsx_file(self, tmp_path) -> None:
        from phantomkit.plotting.maps_ir import find_xlsx_file

        (tmp_path / "ir_500.xlsx").touch()
        result = find_xlsx_file(str(tmp_path), "ir_500")
        assert result.endswith("ir_500.xlsx")

    def test_find_xlsx_file_not_found(self, tmp_path) -> None:
        from phantomkit.plotting.maps_ir import find_xlsx_file

        with pytest.raises(FileNotFoundError):
            find_xlsx_file(str(tmp_path), "missing")


# ---------------------------------------------------------------------------
# maps_ir plot function
# ---------------------------------------------------------------------------


class TestMapsIrPlot:
    """Tests for plot_vial_means_std_pub_from_nifti in maps_ir."""

    # Use 4 contrasts so curve_fit has enough points (model has 2 params)
    _TIS = [100, 300, 600, 1200]

    @pytest.fixture()
    def setup(self, tmp_path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        contrast_files = []
        for ti in self._TIS:
            name = f"ir_{ti}"
            _write_mean_std_csvs(metrics, name, scale=ti / 600)
            contrast_files.append(str(tmp_path / f"{name}.nii.gz"))
        return contrast_files, metrics, tmp_path

    def test_produces_output_file(self, setup) -> None:
        from phantomkit.plotting.maps_ir import plot_vial_ir_means_std

        contrast_files, metrics, tmp_path = setup
        out = plot_vial_ir_means_std(
            contrast_files=contrast_files,
            metric_dir=str(metrics),
            output_file=str(tmp_path / "ir_plot.html"),
        )
        assert os.path.exists(out)

    def test_also_writes_fit_csv(self, setup) -> None:
        from phantomkit.plotting.maps_ir import plot_vial_ir_means_std

        contrast_files, metrics, tmp_path = setup
        plot_vial_ir_means_std(
            contrast_files=contrast_files,
            metric_dir=str(metrics),
            output_file=str(tmp_path / "ir_plot.html"),
        )
        # The function should write a fit CSV alongside the plot
        all_csvs = list(tmp_path.rglob("*.csv")) + list(metrics.rglob("*.csv"))
        assert len(all_csvs) > 0, "Expected a fitted-parameters CSV to be written"

    def test_with_roi_image(self, setup, tmp_path) -> None:
        import matplotlib.pyplot as plt
        from phantomkit.plotting.maps_ir import plot_vial_ir_means_std

        contrast_files, metrics, _ = setup
        roi_png = tmp_path / "roi.png"
        fig, ax = plt.subplots()
        fig.savefig(str(roi_png))
        plt.close(fig)

        out = plot_vial_ir_means_std(
            contrast_files=contrast_files,
            metric_dir=str(metrics),
            output_file=str(tmp_path / "ir_plot.html"),
            roi_image=str(roi_png),
        )
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# maps_te helpers
# ---------------------------------------------------------------------------


class TestMapsTeHelpers:
    def test_extract_numeric(self) -> None:
        from phantomkit.plotting.maps_te import extract_numeric

        assert extract_numeric("SE_80") == 80
        assert extract_numeric("no_numbers") is None

    def test_mono_exp_zero_te(self) -> None:
        from phantomkit.plotting.maps_te import mono_exp

        assert mono_exp(0, 1000, 50) == pytest.approx(1000)

    def test_mono_exp_large_te(self) -> None:
        from phantomkit.plotting.maps_te import mono_exp

        assert mono_exp(1e9, 1000, 50) == pytest.approx(0.0, abs=1e-10)

    def test_calc_r2_perfect_fit(self) -> None:
        from phantomkit.plotting.maps_te import calc_r2

        y = np.array([3.0, 2.0, 1.0])
        assert calc_r2(y, y) == pytest.approx(1.0)

    def test_find_xlsx_file_not_found(self, tmp_path) -> None:
        from phantomkit.plotting.maps_te import find_xlsx_file

        with pytest.raises(FileNotFoundError):
            find_xlsx_file(str(tmp_path), "missing")


# ---------------------------------------------------------------------------
# maps_te plot function
# ---------------------------------------------------------------------------


class TestMapsTePlot:
    _TES = [20, 60, 100, 160]

    @pytest.fixture()
    def setup(self, tmp_path):
        metrics = tmp_path / "metrics"
        metrics.mkdir()
        contrast_files = []
        for te in self._TES:
            name = f"te_{te}"
            # Mono-exp decay: signal decreases with TE
            scale = np.exp(-te / 80)
            _write_mean_std_csvs(metrics, name, scale=scale)
            contrast_files.append(str(tmp_path / f"{name}.nii.gz"))
        return contrast_files, metrics, tmp_path

    def test_produces_output_file(self, setup) -> None:
        from phantomkit.plotting.maps_te import plot_vial_te_means_std

        contrast_files, metrics, tmp_path = setup
        out = plot_vial_te_means_std(
            contrast_files=contrast_files,
            metric_dir=str(metrics),
            output_file=str(tmp_path / "te_plot.html"),
        )
        assert os.path.exists(out)

    def test_with_roi_image(self, setup, tmp_path) -> None:
        import matplotlib.pyplot as plt
        from phantomkit.plotting.maps_te import plot_vial_te_means_std

        contrast_files, metrics, _ = setup
        roi_png = tmp_path / "roi.png"
        fig, ax = plt.subplots()
        fig.savefig(str(roi_png))
        plt.close(fig)

        out = plot_vial_te_means_std(
            contrast_files=contrast_files,
            metric_dir=str(metrics),
            output_file=str(tmp_path / "te_plot.html"),
            roi_image=str(roi_png),
        )
        assert os.path.exists(out)
