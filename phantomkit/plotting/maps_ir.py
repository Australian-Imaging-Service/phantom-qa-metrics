#!/usr/bin/env python3
"""
================================================================================
T1 Inversion Recovery Plotting Script
================================================================================
This script reads mean and standard deviation data from CSV files for multiple
inversion time (TI) contrast images, plots them grouped by vials, and fits an
inversion recovery model to estimate T1 relaxation times.

Key outputs:
- Interactive HTML (default) or publication-quality PNG with 3x3 grid plot
- CSV file with fitted T1 values and R² statistics
================================================================================
"""

import matplotlib

import logging
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

logger = logging.getLogger(__name__)


def find_xlsx_file(metric_dir, contrast_name):
    """
    Find a per-contrast xlsx file in metric_dir.

    Raises:
        FileNotFoundError: If no matching file is found
    """
    target = contrast_name + ".xlsx"
    for f in os.listdir(metric_dir):
        if f == target:
            return os.path.join(metric_dir, f)
    raise FileNotFoundError(
        f"No xlsx file found for contrast '{contrast_name}' in {metric_dir}"
    )


def find_csv_file(metric_dir, contrast_name, suffix):
    """Kept for backwards compatibility — prefer find_xlsx_file."""
    exact_tail = contrast_name + suffix
    for f in os.listdir(metric_dir):
        if f.endswith(exact_tail):
            return os.path.join(metric_dir, f)
    raise FileNotFoundError(
        f"No CSV file found for contrast '{contrast_name}' with suffix '{suffix}' in {metric_dir}"
    )


def _compute_fits_ir(vial_groups, vial_to_idx, data_matrix, contrast_numbers):
    """Fit T1 inversion recovery for all vials; return (subplot_fits, fit_results)."""
    subplot_fits = []
    fit_results = []
    for group in vial_groups:
        for vial in group:
            if vial not in vial_to_idx:
                continue
            i = vial_to_idx[vial]
            try:
                popt, pcov = curve_fit(
                    inv_rec, contrast_numbers, data_matrix[i, :],
                    p0=(data_matrix[i, -1], 1000), maxfev=5000,
                )
                S0_fit, T1_fit = popt
                r2 = calc_r2(data_matrix[i, :], inv_rec(contrast_numbers, *popt))
                try:
                    T1_se = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else None
                except Exception:
                    T1_se = None
                fit_results.append({"Vial": vial, "S0": S0_fit, "T1_ms": T1_fit, "T1_se_ms": T1_se, "R2": r2})
                x_fit = np.linspace(0, max(contrast_numbers), 200)
                y_fit = inv_rec(x_fit, *popt)
                ci_lower = ci_upper = None
                try:
                    samples = np.random.multivariate_normal(popt, pcov, 1000)
                    preds = np.array([inv_rec(x_fit, *s) for s in samples])
                    ci_lower = np.percentile(preds, 2.5, axis=0)
                    ci_upper = np.percentile(preds, 97.5, axis=0)
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning("Could not calculate CI for vial %s: %s", vial, e)
                subplot_fits.append({
                    "vial": vial,
                    "x_fit": x_fit.tolist(),
                    "y_fit": y_fit.tolist(),
                    "ci_lower": ci_lower.tolist() if ci_lower is not None else None,
                    "ci_upper": ci_upper.tolist() if ci_upper is not None else None,
                })
            except RuntimeError:
                logger.warning("Could not fit T₁ for vial %s", vial)
    return subplot_fits, fit_results


def extract_numeric(label):
    """
    Extract the last numeric value from a string label.

    Used to extract inversion times from filenames (e.g., 'IR_500' → 500)

    Args:
        label: String containing numbers (e.g., 'contrast_100')

    Returns:
        Last integer found in the string, or None if no numbers found
    """
    numbers = re.findall(r"\d+", label)
    return int(numbers[-1]) if numbers else None


# Inversion recovery model for magnitude data
def inv_rec(ti, S0, T1):
    """
    Inversion recovery signal model for magnitude MRI data.

    Model: |S0 * (1 - 2 * exp(-TI/T1))|

    Args:
        ti: Inversion time (ms) - can be scalar or array
        S0: Equilibrium signal intensity
        T1: Longitudinal relaxation time (ms)

    Returns:
        Signal intensity at given inversion time(s)
    """
    return np.abs(S0 * (1 - 2 * np.exp(-ti / T1)))


def calc_r2(y_true, y_pred):
    """
    Calculate coefficient of determination (R²) for model fit quality.

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares

    Args:
        y_true: Observed data values
        y_pred: Predicted values from model

    Returns:
        R² value (1.0 = perfect fit, 0.0 = no better than mean)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_vial_ir_means_std(
    contrast_files: list[str],
    metric_dir: str,
    output_file: str = "vial_summary_T1.png",
    annotate: bool = False,
    roi_image: str | None = None,
    output_format: str = "html",
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
    fits_output: str | None = None,
    relaxometry_reference: dict | None = None,
    phantom: str | None = None,
    overlay_contrast: str | None = None,
    scan_date: str | None = None,
) -> str:
    """
    Create interactive HTML or publication-quality PNG plots of vial intensity
    data with T1 curve fitting.

    Generates a 3x3 grid of subplots showing intensity vs inversion time for
    different vial groups, with fitted T1 curves overlaid on the measured data.

    Args:
        contrast_files: List of NIfTI file paths for different inversion times
        metric_dir: Directory containing mean/std CSV files
        output_file: Output filename for the plot (default: 'vial_summary_T1.png')
        annotate: Whether to annotate points with mean ± std (PNG mode only)
        roi_image: Optional path to ROI overlay image (PNG mode only)
        output_format: "html" (default) for interactive HTML, "png" for static figure
        nifti_image: Path to the background NIfTI file to embed in the HTML viewer.
        vial_niftis: ``{vial_name: path}`` mapping for ROI overlay NIfTIs (all shown in red).

    Returns:
        Absolute path to the saved output file.
    """

    # ========================================================================
    # CONFIGURATION: Vial groupings for subplots
    # ========================================================================
    # Define which vials appear together in each subplot
    # Single-element lists get their own subplot (plotted in black)
    # Multi-element lists share a subplot (each vial gets a different color)
    vial_groups = [
        ["S"],  # Subplot 1: Vial S alone
        ["D", "P"],  # Subplot 2: Vials D and P together
        ["M"],  # Subplot 3: Vial M alone
        ["C", "N"],  # Subplot 4: Vials C and N together
        ["B", "T"],  # Subplot 5: Vials B and T together
        ["A", "R"],  # Subplot 6: Vials A and R together
        ["O"],  # Subplot 7: Vial O alone
        ["Q"],  # Subplot 8: Vial Q alone
        # Subplot 9: Reserved for MRI viewer (HTML) or ROI overlay image (PNG)
    ]

    # ========================================================================
    # DATA LOADING: Read mean and std deviation from CSV files
    # ========================================================================
    vial_labels = None  # Vial identifiers (e.g., ['A', 'B', 'C', ...])
    contrast_numbers = []  # Inversion times extracted from filenames
    mean_matrix = []  # Mean intensity values for each vial at each TI
    std_matrix = []  # Standard deviation values
    median_matrix = []
    count_matrix = []
    p25_matrix = []
    p75_matrix = []
    min_matrix = []
    max_matrix = []
    mean_mad_matrix = []
    median_mad_matrix = []

    # Loop through each contrast file (different inversion times)
    for nifti_path in contrast_files:
        # Extract base filename without extension
        base_name = os.path.basename(nifti_path).replace(".nii.gz", "")

        # Find the per-contrast xlsx and read mean/std sheets
        xlsx_path = find_xlsx_file(metric_dir, base_name)
        mean_df = pd.read_excel(xlsx_path, sheet_name="mean")
        std_df = pd.read_excel(xlsx_path, sheet_name="std")

        # Get vial labels from first file (column 0)
        if vial_labels is None:
            vial_labels = mean_df.iloc[:, 0].astype(str).tolist()

        # Extract intensity values (column 1) and inversion time from filename
        mean_matrix.append(mean_df.iloc[:, 1].to_numpy())
        std_matrix.append(std_df.iloc[:, 1].to_numpy())
        contrast_numbers.append(extract_numeric(base_name))

        # Optional sheets — graceful degradation for older xlsx files
        _mean_col = mean_df.iloc[:, 1].to_numpy()
        for _attr, _sheet in [
            (median_matrix, "median"),
            (count_matrix, "count"),
            (p25_matrix, "p25"),
            (p75_matrix, "p75"),
            (min_matrix, "min"),
            (max_matrix, "max"),
            (mean_mad_matrix, "mean_mad"),
            (median_mad_matrix, "median_mad"),
        ]:
            try:
                _df = pd.read_excel(xlsx_path, sheet_name=_sheet)
                vals = _df.iloc[:, 1].to_numpy().astype(float)
                nan_mask = np.isnan(vals)
                if nan_mask.any():
                    vals[nan_mask] = _mean_col[nan_mask]
                _attr.append(vals)
            except Exception:
                _attr.append(_mean_col.copy())

    # Convert lists to numpy arrays for easier manipulation
    mean_matrix = np.array(mean_matrix)
    std_matrix = np.array(std_matrix)
    median_matrix = np.array(median_matrix)
    count_matrix = np.array(count_matrix)
    p25_matrix = np.array(p25_matrix)
    p75_matrix = np.array(p75_matrix)
    min_matrix = np.array(min_matrix)
    max_matrix = np.array(max_matrix)
    mean_mad_matrix = np.array(mean_mad_matrix)
    median_mad_matrix = np.array(median_mad_matrix)

    # ========================================================================
    # DATA ORGANIZATION: Sort by inversion time and transpose
    # ========================================================================
    # Sort all data by inversion time (ascending order)
    sort_idx = np.argsort(contrast_numbers)
    contrast_numbers = np.array(contrast_numbers)[sort_idx]
    mean_matrix       = mean_matrix[sort_idx].T    # rows=vials, cols=TI
    std_matrix        = std_matrix[sort_idx].T
    median_matrix     = median_matrix[sort_idx].T
    count_matrix      = count_matrix[sort_idx].T
    p25_matrix        = p25_matrix[sort_idx].T
    p75_matrix        = p75_matrix[sort_idx].T
    min_matrix        = min_matrix[sort_idx].T
    max_matrix        = max_matrix[sort_idx].T
    mean_mad_matrix   = mean_mad_matrix[sort_idx].T
    median_mad_matrix = median_mad_matrix[sort_idx].T

    # Create mapping from vial label to row index for quick lookup
    vial_to_idx = {label: i for i, label in enumerate(vial_labels)}

    # Define y-axis limits for all subplots (consistent across all plots)
    yticks = [-100, 1000, 2000, 3000, 4000]

    # ========================================================================
    # CURVE FITTING (mean and median \u2014 shared by both PNG and HTML paths)
    # ========================================================================
    cmap = plt.get_cmap("tab10")
    subplot_fits, fit_results = _compute_fits_ir(
        vial_groups, vial_to_idx, mean_matrix, contrast_numbers
    )
    subplot_fits_median, fit_results_median = _compute_fits_ir(
        vial_groups, vial_to_idx, median_matrix, contrast_numbers
    )

    # ========================================================================
    # SAVE: Fitted T1 CSV (mean-based; always, regardless of output format)
    # ========================================================================
    csv_stem = re.sub(r"\.(png|html)$", "", output_file, flags=re.IGNORECASE)
    csv_output = fits_output if fits_output else csv_stem + "_T1_fits.csv"
    pd.DataFrame(fit_results).to_csv(csv_output, index=False)
    logger.info("Fitted parameters saved to %s", csv_output)

    # ========================================================================
    # HTML OUTPUT
    # ========================================================================
    if output_format == "html":
        from phantomkit.plotting._html_common import build_relaxometry_html

        html_file = re.sub(r"\.(png|html)$", "", output_file, flags=re.IGNORECASE) + ".html"

        embedded_data = {
            "type": "maps_ir",
            "phantom": phantom,
            "scan_date": scan_date,
            "vial_groups": vial_groups,
            "contrast_numbers": contrast_numbers.tolist(),
            "mean_matrix": mean_matrix.tolist(),
            "std_matrix": std_matrix.tolist(),
            "fit_results": fit_results,
        }

        html = build_relaxometry_html(
            title="T1 Inversion Recovery Mapping",
            subtitle="Intensity vs inversion time with T₁ curve fitting",
            x_label="Inversion Time (ms)",
            fit_label="T₁ fit",
            fit_key="T1_ms",
            vial_groups=vial_groups,
            contrast_numbers=contrast_numbers,
            mean_matrix=mean_matrix,
            std_matrix=std_matrix,
            median_matrix=median_matrix,
            count_matrix=count_matrix,
            p25_matrix=p25_matrix,
            p75_matrix=p75_matrix,
            min_matrix=min_matrix,
            max_matrix=max_matrix,
            mean_mad_matrix=mean_mad_matrix,
            median_mad_matrix=median_mad_matrix,
            vial_labels=vial_labels,
            subplot_fits=subplot_fits,
            subplot_fits_median=subplot_fits_median,
            fit_results_median=fit_results_median,
            embedded_data=embedded_data,
            nifti_image=nifti_image,
            vial_niftis=vial_niftis,
            ref_data=relaxometry_reference,
            overlay_contrast=overlay_contrast,
        )

        from pathlib import Path
        Path(html_file).write_text(html, encoding="utf-8")
        logger.info("Interactive HTML saved to %s", html_file)
        return os.path.abspath(html_file)

    # ========================================================================
    # PNG OUTPUT (original matplotlib behaviour)
    # ========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for g_idx, group in enumerate(vial_groups):
        ax = axes[g_idx]

        # --------------------------------------------------------------------
        # CASE 1: Single vial subplot
        # --------------------------------------------------------------------
        if len(group) == 1:
            vial = group[0]
            if vial not in vial_to_idx:
                continue
            i = vial_to_idx[vial]

            ax.errorbar(
                contrast_numbers,
                mean_matrix[i, :],
                yerr=std_matrix[i, :],
                fmt="none",
                capsize=5,
                color="black",
                alpha=0.5,
            )
            ax.scatter(
                contrast_numbers,
                mean_matrix[i, :],
                s=50,
                color="black",
                marker="o",
                label=f"Vial {vial}",
                zorder=3,
            )

            fit_entry = next((f for f in subplot_fits if f["vial"] == vial), None)
            if fit_entry:
                x_fit = np.array(fit_entry["x_fit"])
                y_fit = np.array(fit_entry["y_fit"])
                if fit_entry.get("ci_lower") and fit_entry.get("ci_upper"):
                    ax.fill_between(
                        x_fit,
                        fit_entry["ci_lower"],
                        fit_entry["ci_upper"],
                        color="gray",
                        alpha=0.2,
                        zorder=1,
                        label="95% CI",
                    )
                ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.8, zorder=2, label="T₁ fit")

        # --------------------------------------------------------------------
        # CASE 2: Multiple vials in one subplot
        # --------------------------------------------------------------------
        else:
            for j, vial in enumerate(group):
                if vial not in vial_to_idx:
                    continue
                i = vial_to_idx[vial]

                ax.errorbar(
                    contrast_numbers,
                    mean_matrix[i, :],
                    yerr=std_matrix[i, :],
                    fmt="none",
                    capsize=5,
                    color=cmap(j % 10),
                    alpha=0.5,
                )
                ax.scatter(
                    contrast_numbers,
                    mean_matrix[i, :],
                    s=50,
                    color=cmap(j % 10),
                    marker="o",
                    label=f"Vial {vial}",
                    zorder=3,
                )

                fit_entry = next((f for f in subplot_fits if f["vial"] == vial), None)
                if fit_entry:
                    x_fit = np.array(fit_entry["x_fit"])
                    y_fit = np.array(fit_entry["y_fit"])
                    if fit_entry.get("ci_lower") and fit_entry.get("ci_upper"):
                        ax.fill_between(
                            x_fit,
                            fit_entry["ci_lower"],
                            fit_entry["ci_upper"],
                            color=cmap(j % 10),
                            alpha=0.15,
                            zorder=1,
                        )
                    ax.plot(
                        x_fit, y_fit, "--", color=cmap(j % 10), alpha=0.8, zorder=2
                    )

            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(" & ".join(group), fontsize=10)
        ax.set_ylim(min(yticks), max(yticks))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        if g_idx % 3 == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks])
            ax.set_ylabel("Intensity", fontsize=9)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([])

    # ROI overlay in last subplot
    ax = axes[-1]
    if roi_image and os.path.exists(roi_image):
        img = plt.imread(roi_image)
        ax.imshow(
            img,
            extent=[
                contrast_numbers[0],
                contrast_numbers[-1],
                min(yticks),
                max(yticks),
            ],
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ROI Overlay", fontsize=10)
    else:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info("Publication-ready plot saved to %s", output_file)
    plt.close(fig)
    return os.path.abspath(output_file)


@click.command("maps-ir")
@click.argument("contrast_files", nargs=-1, required=True)
@click.option("-m", "--metric-dir", required=True, help="Directory with mean/std CSVs.")
@click.option("-o", "--output", default="vial_summary_T1.png", show_default=True)
@click.option("--annotate", is_flag=True, default=False)
@click.option("--roi-image", default=None, help="ROI overlay PNG (PNG mode only).")
@click.option(
    "--format", "output_format",
    default="html",
    type=click.Choice(["html", "png"]),
    show_default=True,
    help="Output format: interactive HTML (default) or static PNG.",
)
def main(contrast_files, metric_dir, output, annotate, roi_image, output_format):
    """Plot vial mean ± std for inversion recovery with T1 fitting."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plot_vial_ir_means_std(
        list(contrast_files),
        metric_dir=metric_dir,
        output_file=output,
        annotate=annotate,
        roi_image=roi_image,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
