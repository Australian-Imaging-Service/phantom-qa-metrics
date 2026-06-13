"""Top-level PhantomKit Pydra workflow and helper tasks.

This module is the single entry-point for pydra2app / frametree-xnat:

    phantomkit.pydra_workflow:PhantomKitWorkflow

``PhantomKitWorkflow`` accepts concrete typed scan inputs (no directory paths)
and wires all processing stages as a Pydra task graph.  Execution parameters
(worker, cache root) live only in the CLI via ``Submitter``.

Design rules
------------
- Subprocess calls: ``@shell.define`` tasks (pydra-tasks-mrtrix3/fsl, or custom
  tasks in ``phantomkit.tasks``).
- Pure Python logic: ``@python.define`` tasks.
- ``@python.define`` path inputs: ``Path`` or ``Path | None``.
- ``@python.define`` path outputs: fileformats types in the return tuple.
- ``@workflow.define`` wires tasks via ``workflow.add()``.
- No ``Submitter`` calls inside any task or workflow body.
"""
from __future__ import annotations

from pathlib import Path

from pydra.compose import python, workflow
from fileformats.medimage import NiftiGz, Bvec, Bval
from fileformats.vendor.mrtrix3.medimage import ImageIn, ImageOut
from fileformats.generic import File


# ---------------------------------------------------------------------------
# StageSeries — convert a raw series directory to typed NIfTI objects
# ---------------------------------------------------------------------------


@python.define(outputs=["nii", "bvec", "bval", "json_file"])
def StageSeries(
    series_dir: Path,
    out_dir: Path,
) -> tuple[NiftiGz, Bvec | None, Bval | None, File | None]:
    """Convert a DICOM / NIfTI / MIF series directory to typed fileformats objects.

    Wraps ``convert_series_to_nii()`` from ``dwi_processing``.
    """
    from phantomkit.dwi_processing import convert_series_to_nii

    result = convert_series_to_nii(str(series_dir), str(out_dir))
    nii = NiftiGz(result["nii"])
    bvec = Bvec(result["bvec"]) if result.get("bvec") else None
    bval = Bval(result["bval"]) if result.get("bval") else None
    json_file = File(result["json"]) if result.get("json") else None
    return nii, bvec, bval, json_file


# ---------------------------------------------------------------------------
# RunCalibration — temperature estimation from vial ADC values
# ---------------------------------------------------------------------------


@python.define(outputs=["html_report"])
def RunCalibration(
    adc_csv: Path,
    calibration_lut: Path,
    phantom_config: Path,
    phantom: str,
    output_html: Path,
) -> File:
    """Estimate vial temperatures from ADC values and write an HTML report.

    Wraps the ``calibration_plotter`` functions directly (no subprocess).
    """
    from phantomkit.plotting.calibration_plotter import (
        parse_calibration_xlsx,
        parse_vials_csv,
        load_phantom_config,
        build_vial_map,
        estimate_temperature,
        build_vials_html,
    )

    formulations = parse_calibration_xlsx(str(calibration_lut))
    phantom_cfg = load_phantom_config(str(phantom_config))
    vial_map = build_vial_map(phantom_cfg, phantom)
    vials_adc = parse_vials_csv(str(adc_csv))

    results = []
    for vial, adc_raw in vials_adc.items():
        form_query = vial_map.get(vial)
        if form_query is None:
            continue
        try:
            res = estimate_temperature(formulations, str(form_query), D=adc_raw, vial=vial)
            results.append(res)
        except ValueError:
            pass

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(build_vials_html(formulations, results, phantom_name=phantom))
    return File(str(output_html))


# ---------------------------------------------------------------------------
# RunPhantomSession — phantom QC for one image via PhantomProcessor
# ---------------------------------------------------------------------------


@python.define(outputs=["out"])
def RunPhantomSession(
    input_image: Path,
    template_dir: Path,
    output_dir: Path,
    rayleigh_correction: bool = False,
    cpu_threads: int = 1,
) -> str:
    """Run PhantomProcessor (ANTs registration + vial metrics + plots) on one image.

    This is a ``@python.define`` adapter that bridges the legacy
    ``PhantomProcessor.process_session()`` API with the Pydra task graph.
    The Submitter inside ``process_session()`` is the legacy fallback path;
    future work will refactor ``PhantomSessionWf`` to expose typed file
    outputs directly so it can be added as a sub-workflow without this wrapper.
    """
    from phantomkit.phantom_processor import PhantomProcessor

    processor = PhantomProcessor(
        template_dir=str(template_dir),
        output_base_dir=str(output_dir),
        rayleigh_correction=rayleigh_correction,
        n_threads=cpu_threads,
    )
    result = processor.process_session(str(input_image))
    return result.get("output_dir", str(output_dir))


# ---------------------------------------------------------------------------
# PhantomKitWorkflow — top-level @workflow.define
# ---------------------------------------------------------------------------


@workflow.define(outputs=["t1_in_dwi", "adc", "fa", "dwi_preproc"])
def PhantomKitWorkflow(
    # ── Core anatomical & phantom identity ───────────────────────────────────
    t1w: NiftiGz,
    phantom: str,
    # ── DWI inputs (optional; Stages 1+2 skipped when dwi is None) ───────────
    dwi: ImageIn | None = None,
    dwi_pe_dir: str | None = None,
    preproc_mode: str = "rpe_none",
    rpe: ImageIn | None = None,
    fwd_b0: NiftiGz | None = None,
    readout_time: float | None = None,
    eddy_options: str | None = None,
    denoise_degibbs: bool = False,
    gradcheck: bool = False,
) -> tuple[NiftiGz | None, NiftiGz | None, NiftiGz | None, ImageOut | None]:
    """End-to-end PhantomKit MRI phantom QA workflow.

    All inputs are concrete typed fileformats objects — no directory paths.
    Directory scanning and series classification live in the CLI
    (``phantomkit pipeline``), which instantiates this workflow with resolved
    scan objects and submits via ``Submitter``.

    Stage 1 runs DWI preprocessing (``DWISeriesWorkflow``) and is skipped when
    ``dwi`` is ``None``.  Outputs are ``None`` when the corresponding stage is
    skipped.
    """
    from phantomkit.pipeline import TEMPLATE_DATA_ROOT
    from phantomkit.dwi_processing import DWISeriesWorkflow

    template_dir = TEMPLATE_DATA_ROOT / phantom

    if dwi is not None:
        # ── Stage 1: DWI preprocessing ────────────────────────────────────────
        dwi_wf = workflow.add(
            DWISeriesWorkflow(
                t1w=t1w,
                dwi=dwi,
                dwi_pe_dir=dwi_pe_dir or "AP",
                preproc_mode=preproc_mode,
                rpe=rpe,
                fwd_b0=fwd_b0,
                readout_time=readout_time if readout_time is not None else 0.05,
                eddy_options=eddy_options if eddy_options is not None else " --slm=linear",
                denoise_degibbs=denoise_degibbs,
                gradcheck=gradcheck,
            ),
            name="dwi_processing",
        )
        return (
            dwi_wf.t1_in_dwi,
            dwi_wf.adc,
            dwi_wf.fa,
            dwi_wf.dwi_preproc,
        )
    else:
        # No DWI — Stage 1+2 skipped; Stage 3 (native QC) is handled by
        # the CLI via pipeline.py stages until PhantomSessionWf is refactored
        # to expose typed file outputs for direct workflow.add() integration.
        return t1w, None, None, None
