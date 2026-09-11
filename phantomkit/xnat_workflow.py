"""XNAT Container Service entry-point for phantomkit's full pipeline.

This module is a dedicated, separate entry-point for pydra2app /
frametree-xnat — deliberately not a modification of
``phantomkit.pydra_workflow.PhantomKitWorkflow`` (which stays as-is for the
CLI's more flexible, partially-optional flows):

    phantomkit.xnat_workflow:PhantomKitXnatWorkflow

The acquisition protocol this module targets is fixed (confirmed against a
real scan session): DWI is always ``rpe_all`` (one full forward-phase-encode
AP volume + one full reverse-phase-encode PA volume, both always provided),
and native-contrast QC always has exactly 1 MPRAGE + 12 TI scans (T1
mapping) + 8 TE scans (T2 mapping). Because every input is always provided,
every FileSet-typed field here is a single concrete type — never
``Optional``/``Union`` — which sidesteps a confirmed upstream pydra/
frametree bug where Optional or Union file-type fields crash pipeline
construction or type coercion.

frametree-xnat only supports one-source-per-one-exact-scan matching (no
regex source can deliver multiple scans as a list), so the 12 TI + 8 TE
scans are 20 separately-named parameters rather than a compact list input.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from pydra.compose import python, workflow
from fileformats.medimage import NiftiGz
from fileformats.medimage.diffusion import NiftiGzXBvec
from fileformats.vendor.mrtrix3.medimage import ImageOut, ImageFormatGz
from fileformats.generic import Directory


@python.define(outputs=["mif"])
def _EmbedGradients(dwi: NiftiGzXBvec, out_name: str) -> ImageFormatGz:
    """Convert a NIfTI+bvec/bval(+json) bundle into a .mif.gz with the
    gradient table embedded in its header.

    DWISeriesWorkflow's internal mrtrix3 DWI tools (dwiextract, dwicat,
    dwidenoise, ...) need gradients embedded in the image itself, not
    sitting in separate sidecar files next to a plain NIfTI — mrtrix3's
    auto-discovery for raw NIfTI input doesn't pick up sibling .bvec/.bval
    files the way dcm2niix-produced NIfTI conventionally implies. This
    mirrors cli.py's own ``mrconvert ... -fslgrad <bvec> <bval>`` step for
    the CLI path exactly — DWISeriesWorkflow always expects an
    already-gradient-embedded input, regardless of caller.
    """
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    bvec = next(p for p in dwi.fspaths if p.suffix == ".bvec")
    bval = next(p for p in dwi.fspaths if p.suffix == ".bval")
    json_sidecars = [p for p in dwi.fspaths if p.suffix == ".json"]

    out_dir = _Path(tempfile.mkdtemp(prefix="embed_grad_"))
    out_path = out_dir / f"{out_name}.mif.gz"

    cmd = [
        "mrconvert", str(dwi.fspath), str(out_path),
        "-fslgrad", str(bvec), str(bval), "-force",
    ]
    if json_sidecars:
        cmd += ["-json_import", str(json_sidecars[0])]
    subprocess.run(cmd, check=True, capture_output=True)
    return ImageFormatGz(str(out_path))


def _rename_contrasts(named: dict) -> list:
    """Copy each contrast to a fresh path stemmed by its own source name.

    phantom_processor.py's per-contrast classification (TI vs TE vs
    anatomical vs ADC/FA) and output naming both derive identity purely
    from each input file's own local path stem — frametree/pydra2app
    stage downloaded XNAT sources under their own (non-descriptive, and
    not guaranteed distinct) local cache filenames, so without this
    rename every contrast collides on whatever generic stem staging used,
    silently overwriting all but the last-processed contrast's output.
    Renaming here (not in phantom_processor.py) keeps the CLI's own
    already-correct file-naming untouched.
    """
    import shutil
    import tempfile
    from pathlib import Path as _Path
    from fileformats.medimage import NiftiGz as _NiftiGz

    out_dir = _Path(tempfile.mkdtemp(prefix="named_contrasts_"))
    renamed = []
    for name, fileset in named.items():
        dest = out_dir / f"{name}.nii.gz"
        shutil.copy2(str(fileset), dest)
        renamed.append(_NiftiGz(dest))
    return renamed


@python.define(outputs=["files"])
def _CombineStage2Contrasts(
    t1_in_dwi: NiftiGz, adc: NiftiGz, fa: NiftiGz
) -> list[NiftiGz]:
    """Bundle Stage 1's three lazy outputs into one list[NiftiGz] field.

    Pydra can't coerce a plain Python list containing several separate
    lazy sub-workflow outputs directly into a list[NiftiGz]-typed
    parameter at another task's call site (it expects the whole list to
    come from one upstream task) — this task exists purely to be that one
    upstream source. Also renames each to its own identity (T1_in_DWI /
    ADC / FA — "FA" uppercase to match phantom_processor.py's
    case-sensitive FA-classification regex) via _rename_contrasts.
    """
    return _rename_contrasts({"T1_in_DWI": t1_in_dwi, "ADC": adc, "FA": fa})


@python.define(outputs=["files"])
def _CombineStage3Contrasts(
    t1w: NiftiGz,
    ti_50: NiftiGz, ti_100: NiftiGz, ti_150: NiftiGz, ti_250: NiftiGz,
    ti_500: NiftiGz, ti_1000: NiftiGz, ti_1500: NiftiGz, ti_2000: NiftiGz,
    ti_3000: NiftiGz, ti_5000: NiftiGz, ti_7500: NiftiGz, ti_9970: NiftiGz,
    te_14: NiftiGz, te_20: NiftiGz, te_40: NiftiGz, te_80: NiftiGz,
    te_160: NiftiGz, te_320: NiftiGz, te_640: NiftiGz, te_1000: NiftiGz,
) -> list[NiftiGz]:
    """Bundle the MPRAGE + 12 TI + 8 TE workflow inputs into one list[NiftiGz]
    field — same reason as _CombineStage2Contrasts above, and likewise
    renames each to its own source name (T1w, TI_50, ..., TE_1000) via
    _rename_contrasts."""
    return _rename_contrasts({
        "T1w": t1w,
        "TI_50": ti_50, "TI_100": ti_100, "TI_150": ti_150, "TI_250": ti_250,
        "TI_500": ti_500, "TI_1000": ti_1000, "TI_1500": ti_1500, "TI_2000": ti_2000,
        "TI_3000": ti_3000, "TI_5000": ti_5000, "TI_7500": ti_7500, "TI_9970": ti_9970,
        "TE_14": te_14, "TE_20": te_20, "TE_40": te_40, "TE_80": te_80,
        "TE_160": te_160, "TE_320": te_320, "TE_640": te_640, "TE_1000": te_1000,
    })


@python.define(outputs=["out_dir"])
def FinalizeOutputs(
    dwi_preproc: ImageOut,
    stage2_dir: Directory,
    stage3_dir: Directory,
    output_dir_str: str,
) -> Directory:
    """Bundle DWI preprocessing + Stage 2/3 QC outputs into one directory.

    Mirrors the output-bundling pattern already used for multi-output XNAT
    sinks elsewhere (one ``Directory`` sink beats several flat,
    unnamespaced ones).
    """
    import shutil

    out = Path(output_dir_str)
    out.mkdir(parents=True, exist_ok=True)

    dwi_dest = out / "dwi_preproc"
    dwi_dest.mkdir(exist_ok=True)
    shutil.copy2(str(dwi_preproc), dwi_dest / Path(str(dwi_preproc)).name)

    # Per-contrast PNG scatter plots are redundant with the T1/T2 mapping
    # HTML reports (which cover the same fitted data) -- excluded from the
    # uploaded XNAT resource to cut noise/size, without touching
    # phantom_processor.py's own PNG generation (still used by the CLI).
    stage2_dest = out / "stage2_dwi_space_qc"
    shutil.copytree(
        str(stage2_dir), stage2_dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.png"),
    )

    stage3_dest = out / "stage3_native_contrast_qc"
    shutil.copytree(
        str(stage3_dir), stage3_dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.png"),
    )

    return Directory(str(out))


@python.define(outputs=["out_dir"])
def FinalizeNativeContrastOnly(
    stage3_dir: Directory,
    output_dir_str: str,
) -> Directory:
    """Bundle Stage 3 (native-contrast QC) alone into one Directory sink.

    Used when ``run_dwi_qc=False`` skips Stage 1/2 entirely -- kept as a
    separate task (rather than making FinalizeOutputs' dwi_preproc/
    stage2_dir Optional) since Optional FileSet-typed pydra fields are a
    confirmed crash in this pydra/frametree version stack.
    """
    import shutil

    out = Path(output_dir_str)
    out.mkdir(parents=True, exist_ok=True)

    stage3_dest = out / "stage3_native_contrast_qc"
    shutil.copytree(
        str(stage3_dir), stage3_dest, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.png"),
    )

    return Directory(str(out))


@workflow.define(outputs=["out_dir"])
def PhantomKitXnatWorkflow(
    # ── Anatomical / phantom identity ─────────────────────────────────────
    t1w: NiftiGz,
    phantom: str,
    # ── DWI (always rpe_all: both AP and PA are full volumes) ────────────
    # NiftiGzXBvec (not the DWISeriesWorkflow-internal ImageIn union type,
    # which crashes frametree's Pipeline.inputs_validator -- issubclass()
    # requires its first argument to be a real class, not a Union alias)
    # -- a single concrete class, and a real member of the ImageIn union,
    # so passing it into DWISeriesWorkflow's own dwi/rpe params is valid.
    dwi: NiftiGzXBvec,
    rpe: NiftiGzXBvec,
    # ── Native-contrast relaxometry (fixed: 12 TI + 8 TE) ─────────────────
    ti_50: NiftiGz, ti_100: NiftiGz, ti_150: NiftiGz, ti_250: NiftiGz,
    ti_500: NiftiGz, ti_1000: NiftiGz, ti_1500: NiftiGz, ti_2000: NiftiGz,
    ti_3000: NiftiGz, ti_5000: NiftiGz, ti_7500: NiftiGz, ti_9970: NiftiGz,
    te_14: NiftiGz, te_20: NiftiGz, te_40: NiftiGz, te_80: NiftiGz,
    te_160: NiftiGz, te_320: NiftiGz, te_640: NiftiGz, te_1000: NiftiGz,
    # ── Template data (mounted via pydra2app resources:, not the base
    # image — mirrors the T1w pipeline's own resources_dir/configuration
    # pattern for FastSurfer/parcellation atlases, decoupling template_data
    # from needing a new base-image push whenever it changes) ─────────────
    template_data_root: str,
    # ── Processing options ─────────────────────────────────────────────────
    readout_time: float = 0.05,
    eddy_options: str = " --slm=linear",
    # do_fslpreproc is NOT exposed/optional -- it's the eddy/topup
    # correction step rpe_all (FPE+RPE) data exists for; leaving it off
    # (DWISeriesWorkflow's own default) would make collecting RPE data
    # pointless. The others default to off, matching the CLI's own
    # --processing-steps default ("none" -- only tensor fitting), but are
    # exposed so a launch can opt in without a code change.
    #
    # `bool | None` (not plain `bool`) because pydra2app's XNAT command
    # builder serializes a `False` Python default as `default-value: ""`
    # in the command JSON (it treats the default via `if default`, which
    # can't tell False from unset), and at launch time converts any
    # empty-string value for a non-str parameter into None -- so an
    # unset boolean parameter arrives here as None, not False. A plain
    # `bool` field rejects that None as a missing mandatory value; coerce
    # it back to bool below instead.
    do_denoise: bool | None = False,
    do_degibbs: bool | None = False,
    do_biascorrect: bool | None = False,
    gradcheck: bool | None = False,
    # Skips Stage 1 (DWI preprocessing) and Stage 2 (DWI-space QC)
    # entirely when False -- output has only stage3_native_contrast_qc/.
    # dwi/rpe stay mandatory sources regardless (fixed protocol), just
    # unused when off. `bool | None` for the same reason as the four
    # flags above.
    run_dwi_qc: bool | None = True,
    rayleigh_correction: bool = False,
    cpu_threads: int = 1,
) -> Directory:
    """End-to-end phantom QC for the XNAT Container Service.

    Stage 1: DWI preprocessing (``DWISeriesWorkflow``, ``preproc_mode``
    fixed to ``"rpe_all"`` — not exposed as a parameter, since the
    acquisition protocol never varies).
    Stage 2: phantom QC in DWI space (``PhantomSessionWf`` on
    ``t1_in_dwi``/``adc``/``fa``), runs in parallel with Stage 3.
    Stage 3: native-contrast QC (``PhantomSessionWf`` on the MPRAGE + all
    12 TI + all 8 TE scans).
    All three stages' outputs are bundled into one ``Directory`` sink by
    ``FinalizeOutputs``.

    ``template_data_root`` is a fixed, non-source ``configuration:`` value
    in the YAML spec (mounted from the ``template_data`` pydra2app
    resource) — not read from ``phantomkit.pipeline.TEMPLATE_DATA_ROOT``,
    which instead resolves relative to wherever phantomkit itself happens
    to be installed.
    """
    from phantomkit.dwi_processing import DWISeriesWorkflow
    from phantomkit.phantom_processor import PhantomSessionWf, resolve_phantom_template

    do_denoise = bool(do_denoise)
    do_degibbs = bool(do_degibbs)
    do_biascorrect = bool(do_biascorrect)
    gradcheck = bool(gradcheck)
    run_dwi_qc = bool(run_dwi_qc) if run_dwi_qc is not None else True

    template_dir = Path(template_data_root) / phantom
    resolved = resolve_phantom_template(template_dir)
    template_phantom = str(resolved["template_phantom"])
    vial_masks = [str(m) for m in resolved["vial_masks"]]
    adc_vials = sorted(resolved["adc_vials"])
    phantom_name = resolved["phantom_name"]
    template_dir_parent = str(template_dir.parent)

    def _new_qc_output_dir(prefix: str) -> str:
        """Create a fresh output tree for one PhantomSessionWf invocation.

        PhantomSessionWf assumes its tmp/vial/metrics/images subdirectories
        already exist (matching PhantomProcessor.process_session()'s own
        pre-creation of these dirs before constructing the workflow).
        """
        out_dir = Path(tempfile.mkdtemp(prefix=prefix))
        for sub in ("tmp", "vial_segmentations", "metrics", "images_template_space"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)
        return str(out_dir)

    # ── Stage 1 + 2: DWI preprocessing and DWI-space QC (skippable) ───────
    # Branching on a concrete bool here (not a lazy field, since
    # run_dwi_qc arrives as a plain Python value at workflow-construction
    # time) mirrors the same pattern fileformats' own ExtendedDcm2niix
    # converter workflow uses for its to_4d/extract_volume options.
    if run_dwi_qc:
        # DWISeriesWorkflow's internal mrtrix3 DWI tools need the gradient
        # table embedded in the image itself (matches cli.py's own
        # `mrconvert -fslgrad ...` step) -- the raw NiftiGzXBvec sources
        # (nii + separate .bvec/.bval sidecars) aren't auto-discovered by
        # mrtrix3's lower-level DWI tools the way dcm2niix output implicitly
        # is, so embed them explicitly first.
        dwi_mif = workflow.add(_EmbedGradients(dwi=dwi, out_name="DWI_AP"), name="embed_dwi_grad")
        rpe_mif = workflow.add(_EmbedGradients(dwi=rpe, out_name="DWI_PA"), name="embed_rpe_grad")

        dwi_wf = workflow.add(
            DWISeriesWorkflow(
                t1w=t1w,
                dwi=dwi_mif.mif,
                dwi_pe_dir="AP",
                preproc_mode="rpe_all",
                rpe=rpe_mif.mif,
                readout_time=readout_time,
                eddy_options=eddy_options,
                do_denoise=do_denoise,
                do_degibbs=do_degibbs,
                do_fslpreproc=True,
                do_biascorrect=do_biascorrect,
                gradcheck=gradcheck,
            ),
            name="dwi_processing",
        )

        # ── Stage 2: phantom QC in DWI space (parallel with Stage 3) ──────
        stage2_dir = _new_qc_output_dir("phantomkit_stage2_")
        stage2_contrasts = workflow.add(
            _CombineStage2Contrasts(
                t1_in_dwi=dwi_wf.t1_in_dwi, adc=dwi_wf.adc, fa=dwi_wf.fa,
            ),
            name="stage2_combine",
        )
        stage2 = workflow.add(
            PhantomSessionWf(
                input_image=dwi_wf.t1_in_dwi,
                template_phantom=template_phantom,
                vial_masks=vial_masks,
                adc_vials=adc_vials,
                output_prefix=str(Path(stage2_dir) / "tmp" / "stage2_Transformed_"),
                output_dir_str=stage2_dir,
                session_name="stage2_dwi_space",
                phantom_name=phantom_name,
                template_dir_parent=template_dir_parent,
                contrast_files=stage2_contrasts.files,
                rayleigh_correction=rayleigh_correction,
                cpu_threads=cpu_threads,
            ),
            name="stage2_qc",
        )

    # ── Stage 3: native-contrast QC (MPRAGE + 12 TI + 8 TE) ────────────────
    stage3_dir = _new_qc_output_dir("phantomkit_stage3_")
    stage3_contrasts = workflow.add(
        _CombineStage3Contrasts(
            t1w=t1w,
            ti_50=ti_50, ti_100=ti_100, ti_150=ti_150, ti_250=ti_250,
            ti_500=ti_500, ti_1000=ti_1000, ti_1500=ti_1500, ti_2000=ti_2000,
            ti_3000=ti_3000, ti_5000=ti_5000, ti_7500=ti_7500, ti_9970=ti_9970,
            te_14=te_14, te_20=te_20, te_40=te_40, te_80=te_80,
            te_160=te_160, te_320=te_320, te_640=te_640, te_1000=te_1000,
        ),
        name="stage3_combine",
    )
    stage3 = workflow.add(
        PhantomSessionWf(
            input_image=t1w,
            template_phantom=template_phantom,
            vial_masks=vial_masks,
            adc_vials=adc_vials,
            output_prefix=str(Path(stage3_dir) / "tmp" / "stage3_Transformed_"),
            output_dir_str=stage3_dir,
            session_name="stage3_native_contrast",
            phantom_name=phantom_name,
            template_dir_parent=template_dir_parent,
            contrast_files=stage3_contrasts.files,
            rayleigh_correction=rayleigh_correction,
            cpu_threads=cpu_threads,
        ),
        name="stage3_qc",
    )

    # ── Finalize: bundle everything into one Directory sink ───────────────
    if run_dwi_qc:
        final = workflow.add(
            FinalizeOutputs(
                dwi_preproc=dwi_wf.dwi_preproc,
                stage2_dir=stage2.out_dir,
                stage3_dir=stage3.out_dir,
                output_dir_str=tempfile.mkdtemp(prefix="phantomkit_out_"),
            ),
            name="finalize",
        )
    else:
        final = workflow.add(
            FinalizeNativeContrastOnly(
                stage3_dir=stage3.out_dir,
                output_dir_str=tempfile.mkdtemp(prefix="phantomkit_out_"),
            ),
            name="finalize",
        )

    return final.out_dir
