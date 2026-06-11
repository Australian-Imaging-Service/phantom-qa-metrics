"""
Pydra workflows for per-vial metric extraction.

Transforms vial masks into subject space, extracts per-vial signal
statistics from contrast images, and forward-warps contrasts to
template space.
"""

import logging
import os.path
from pathlib import Path

from fileformats.generic import File
from fileformats.medimage import NiftiGz
from pydra.compose import python, workflow
from pydra.tasks.ants.v2.resampling.apply_transforms import ApplyTransforms
from pydra.tasks.mrtrix3.v3_0 import MrConvert, MrGrid, MrInfo, MrStats

from phantomkit.registration import ParseMrStatsStdout


@python.define
def ComputePercentile(
    vol_file: NiftiGz, mask_file: NiftiGz, percentile: int, tmp_dir: Path
) -> float:
    """Compute a per-voxel percentile within a mask using mrthreshold + mrdump."""
    import hashlib
    import subprocess

    h = hashlib.md5(f"{vol_file}{mask_file}{percentile}".encode()).hexdigest()[:8]
    from pathlib import Path as _P
    mask1 = str(_P(tmp_dir) / f"pct{percentile}_{h}_m1.mif")
    mask2 = str(_P(tmp_dir) / f"pct{percentile}_{h}_m2.mif")
    try:
        subprocess.run(
            ["mrthreshold", str(vol_file), "-mask", str(mask_file),
             "-percentile", str(percentile), mask1, "-force", "-quiet"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["mrthreshold", str(vol_file), "-mask", mask1,
             "-top", "1", mask2, "-force", "-quiet"],
            check=True, capture_output=True,
        )
        r = subprocess.run(
            ["mrdump", str(vol_file), "-mask", mask2],
            capture_output=True, text=True, check=True,
        )
        vals = [float(x) for x in r.stdout.strip().split() if x]
        return vals[0] if vals else float("nan")
    except Exception:
        return float("nan")

logger = logging.getLogger(__name__)


# ============================================================================
# Python Tasks
# ============================================================================


@python.define(outputs=["is_4d", "is_single_slice", "nvols"])
def ParseMrInfoSize(stdout: str) -> tuple[bool, bool, int]:
    """Parse ``mrinfo -size`` stdout into dimensionality flags and volume count."""
    parts = stdout.strip().split()
    is_4d = len(parts) >= 4 and int(parts[3]) > 1
    is_single_slice = len(parts) >= 3 and int(parts[2]) == 1
    nvols = int(parts[3]) if len(parts) >= 4 and int(parts[3]) > 0 else 1
    return is_4d, is_single_slice, nvols


@python.define
def CopyFile(src: File, dst: Path) -> File:
    """Copy *src* to *dst* and return *dst*."""
    import shutil

    shutil.copy2(src, dst)
    return dst


@python.define(outputs=["vial_name", "tmp_path", "output_path", "vial_mask_out"])
def PrepVialTransformPaths(
    vial_mask: NiftiGz, output_vial_dir: Path
) -> tuple[str, Path, Path, NiftiGz]:
    """Derive per-vial output paths and pass vial_mask through."""
    from pathlib import Path as _Path

    output_dir = _Path(output_vial_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_vial_dir = output_dir.parent / "tmp_vials"
    tmp_vial_dir.mkdir(parents=True, exist_ok=True)
    vial_name = (
        _Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "").split(".")[0]
    )
    return (
        vial_name,
        tmp_vial_dir / f"{vial_name}.nii",
        output_dir / f"{vial_name}.nii.gz",
        vial_mask,
    )


@python.define
def GatherList(items: list[NiftiGz]) -> list[NiftiGz]:
    """Identity task used to collect a combined split into a typed list output."""
    return list(items)


# ============================================================================
# Metrics Workflows
# ============================================================================


@workflow.define
def TransformVialsToSubjectSpace(
    vial_masks: list[NiftiGz],
    reference_image: NiftiGz,
    transform_matrix: File,
    output_vial_dir: Path,
) -> list[NiftiGz]:
    """
    Transform each vial mask from template space into subject (scanner) space.

    Split over each vial mask:
    1. **PrepVialTransformPaths** derives per-vial output paths.
    2. **ApplyTransforms** applies the inverse affine.
    3. **CopyFile** moves the output to the per-vial tmp path.
    4. **MrConvert** writes the final output.

    A combined **GatherList** collects all per-vial output paths.
    """
    prep = workflow.add(
        PrepVialTransformPaths(
            vial_mask=vial_masks, output_vial_dir=output_vial_dir
        ).split("vial_mask"),
        name="prep",
    )

    at = workflow.add(
        ApplyTransforms(
            input_image=prep.vial_mask_out,
            reference_image=reference_image,
            transforms=[f"[{transform_matrix}, 1]"],
            interpolation="Linear",
            out_postfix="_warped",
        ),
        name="apply_transforms",
    )

    copy = workflow.add(
        CopyFile(src=at.output_image, dst=prep.tmp_path),
        name="copy",
    )

    final = workflow.add(
        MrConvert(
            in_file=copy.out,
            out_file=prep.output_path,
            quiet=True,
            force=True,
        ),
        name="final",
    )

    gather = workflow.add(
        GatherList(items=final.out_file).combine("prep"),
        name="gather",
    )
    return gather.out


@workflow.define
def ExtractMetricsFromContrasts(
    contrast_files: list[NiftiGz],
    vial_masks: list[NiftiGz],
    output_metrics_dir: Path,
    session_name: str,
) -> Path:
    """
    Extract per-vial statistics (mean, median, std, min, max) from every
    contrast image and write per-metric CSV files.

    For each (contrast, vial) pair:
    - **MrInfo** + **ParseMrInfoSize** detect the volume count.
    - **MrGrid** regrids the vial mask to the contrast image grid.
    - **MrConvert** extracts individual volumes from 4-D series.
    - **MrStats** + **ParseMrStatsStdout** compute per-volume statistics.

    Iterating over ``contrast_files`` triggers pydra's runtime fallback, so
    all node outputs are concrete values available for loop control.

    Returns the metrics directory path.
    """
    import pandas as pd
    from pathlib import Path as _Path

    metrics_dir = _Path(output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_vol_dir = metrics_dir.parent / "tmp_vols"
    tmp_vol_dir.mkdir(parents=True, exist_ok=True)

    for contrast_file in contrast_files:
        contrast_path = _Path(contrast_file)
        contrast_name = contrast_path.stem
        clean_name = contrast_name.replace(".nii", "").replace(".gz", "")

        info = workflow.add(
            MrInfo(image_=[contrast_file], size=True, quiet=True),
            name=f"info_{clean_name}",
        )
        parse_size = workflow.add(
            ParseMrInfoSize(stdout=info.stdout),
            name=f"parse_size_{clean_name}",
        )
        nvols = parse_size.nvols

        logger.info(
            "Processing %s (%d volume%s)", clean_name, nvols, "s" if nvols > 1 else ""
        )
        metrics_data: dict[str, dict[str, list[float]]] = {
            "mean": {}, "median": {}, "std": {}, "min": {}, "max": {},
            "count": {}, "p25": {}, "p75": {},
        }

        for vial_mask in vial_masks:
            vial_name = (
                _Path(vial_mask)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
            )
            for metric in metrics_data:
                metrics_data[metric][vial_name] = []

            grid = workflow.add(
                MrGrid(
                    in_file=vial_mask,
                    operation="regrid",
                    template=contrast_file,
                    out_file=str(tmp_vol_dir / f"{clean_name}_{vial_name}.nii.gz"),
                    interp="nearest",
                    force=True,
                ),
                name=f"grid_{clean_name}_{vial_name}",
            )
            regridded_mask = grid.out_file

            for vol_idx in range(nvols):
                tag = f"{clean_name}_{vial_name}_v{vol_idx}"

                if nvols == 1:
                    vol_file = contrast_file
                else:
                    conv = workflow.add(
                        MrConvert(
                            in_file=contrast_file,
                            coord=[(3, vol_idx)],
                            out_file=str(
                                tmp_vol_dir / f"{clean_name}_vol{vol_idx}.nii.gz"
                            ),
                            quiet=True,
                            force=True,
                        ),
                        name=f"conv_{tag}",
                    )
                    vol_file = conv.out_file

                stats = workflow.add(
                    MrStats(
                        image_=vol_file,
                        mask=regridded_mask,
                        output=["mean", "median", "std", "min", "max", "count"],
                        quiet=True,
                    ),
                    name=f"stats_{tag}",
                )
                parse_stats = workflow.add(
                    ParseMrStatsStdout(stdout=stats.stdout),
                    name=f"parse_stats_{tag}",
                )
                values = parse_stats.out
                metrics_data["mean"][vial_name].append(values[0])
                metrics_data["median"][vial_name].append(values[1])
                metrics_data["std"][vial_name].append(values[2])
                metrics_data["min"][vial_name].append(values[3])
                metrics_data["max"][vial_name].append(values[4])
                metrics_data["count"][vial_name].append(values[5])

                p25_task = workflow.add(
                    ComputePercentile(
                        vol_file=vol_file, mask_file=regridded_mask,
                        percentile=25, tmp_dir=tmp_vol_dir,
                    ),
                    name=f"p25_{tag}",
                )
                p75_task = workflow.add(
                    ComputePercentile(
                        vol_file=vol_file, mask_file=regridded_mask,
                        percentile=75, tmp_dir=tmp_vol_dir,
                    ),
                    name=f"p75_{tag}",
                )
                metrics_data["p25"][vial_name].append(p25_task.out)
                metrics_data["p75"][vial_name].append(p75_task.out)

        xlsx_dir = metrics_dir / "xlsx"
        xlsx_dir.mkdir(parents=True, exist_ok=True)
        xlsx_file = xlsx_dir / f"{clean_name}.xlsx"
        sheet_order = ["mean", "median", "std", "min", "max", "count", "p25", "p75"]
        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            for metric_name in sheet_order:
                vial_data = metrics_data[metric_name]
                rows = [
                    {"vial": vn, **{f"vol{i}": v for i, v in enumerate(vals)}}
                    for vn, vals in vial_data.items()
                ]
                pd.DataFrame(rows).to_excel(writer, sheet_name=metric_name, index=False)
        logger.info("Saved: %s", xlsx_file.name)

    return metrics_dir


@workflow.define
def TransformContrastsToTemplateSpace(
    contrast_files: list[NiftiGz],
    transform_matrix: File,
    template_phantom: NiftiGz,
    tmp_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Forward-transform every contrast image into template space.

    For each contrast:
    1. **MrInfo** + **ParseMrInfoSize** detect dimensionality (task outputs
       are concrete at runtime after the for-loop triggers the fallback).
    2. If single-slice, **MrGrid** pads the z-axis.
    3. **ApplyTransforms** applies the forward ANTs affine.
    4. **CopyFile** writes the result to the output directory.

    Returns the template-space output directory path.
    """
    from pathlib import Path as _Path
    from fileformats.generic import File as _File

    output_path = _Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for contrast_file in contrast_files:
        contrast_path = _Path(contrast_file)
        contrast_name = contrast_path.stem.replace(".nii", "")
        logger.info("Transforming: %s", contrast_path.name)

        info = workflow.add(
            MrInfo(image_=[contrast_file], size=True, quiet=True),
            name=f"info_{contrast_name}",
        )
        parse_size = workflow.add(
            ParseMrInfoSize(stdout=info.stdout),
            name=f"parse_size_{contrast_name}",
        )

        transform_input = contrast_file

        if parse_size.is_single_slice:
            pad = workflow.add(
                MrGrid(
                    in_file=contrast_file,
                    operation="pad",
                    axis=[(2, (1, 1))],
                    out_file=f"{contrast_name}_padded.nii.gz",
                    force=True,
                ),
                name=f"pad_{contrast_name}",
            )
            transform_input = pad.out_file

        at = workflow.add(
            ApplyTransforms(
                input_image=transform_input,
                reference_image=template_phantom,
                transforms=[transform_matrix],
                input_image_type=3 if parse_size.is_4d else 0,
                interpolation="Linear",
                out_postfix="_warped",
            ),
            name=f"at_{contrast_name}",
        )

        copy_file = workflow.add(
            CopyFile(
                src=at.output_image,
                dst=output_path / contrast_path.name,
            ),
            name=f"copy_{contrast_name}",
        )

        logger.info("%s -> template space", contrast_path.name)
        copied_files.append(copy_file.out)

    @python.define
    def CommonPath(file_paths: list[_File]) -> _Path:
        return _Path(os.path.commonpath(file_paths))

    common_path = workflow.add(CommonPath(copied_files))

    logger.info("All contrasts saved to: %s", output_path)
    return common_path.out
