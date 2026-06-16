 ```mermaid
flowchart TD
    INPUT[/"--input-dir  --phantom  --output-dir<br/>--processing-steps dwidenoise,mrgibbs,dwifslpreproc,dwibiascorrect,gradcheck  (any subset; default: none)<br/>--nocleanup  --readout-time  --eddy-options  --dry-run  --worker cf|debug<br/>Sub-directories may contain DICOM · NIfTI (.nii/.nii.gz) · MIF (.mif/.mif.gz)"/]

    INPUT --> VALIDATE["validate_inputs()<br/>check template_data/{phantom}/ exists"]
    VALIDATE --> PET_FORK{phantom == PET?}

    %% ── PET fork ──────────────────────────────────────────────────────────────
    PET_FORK -->|yes| PET_STAGE["run_pet_stage()  (pipeline.py)"]
    PET_STAGE --> PET_REG["register_pet_to_template()<br/>ANTs SyNRA"]
    PET_REG --> PET_VIALS["apply_transform_to_vials()<br/>antsApplyTransforms"]
    PET_VIALS --> PET_METRICS["extract_pet_vial_metrics()<br/>mrstats"]
    PET_METRICS --> PET_HTML["plot_pet_vials()<br/>HTML report"]
    PET_HTML --> DONE

    %% ── MRI fork: dual-scan ───────────────────────────────────────────────────
    PET_FORK -->|no| SCAN["Scan input directory<br/>scan_directory()    → candidate DWI dirs<br/>scan_input_dir()    → T1 / IR / TE dirs<br/>_wrap_flat_inputs() → stage flat NIfTIs"]
    SCAN --> S1
    SCAN --> S3

    subgraph S1["Stage 1 — DWI Processing  (cli.py → PhantomKitWorkflow)"]
        direction TB
        CLASSIFY["classify series<br/>convert_all_candidates → staged NIfTI/MIF<br/>classify_candidates  → DWI / fwd-PE / RPE dirs<br/>match_ap_pa_pairs + build_pe_assignment_map<br/>plan_workflow() per series"]
        CLASSIFY --> TYPED["Wrap inputs as typed fileformats objects<br/>T1 → NiftiGz  ·  DWI → ImageFormatGz (mrconvert + bvec/bval/JSON)<br/>RPE → ImageFormatGz  (optional)<br/>fwd-b0 → NiftiGz  (optional)"]
        TYPED --> SUBMIT["PhantomKitWorkflow  (pydra_workflow.py)<br/>↳ DWISeriesWorkflow  (dwi_processing.py)<br/>steps passed as individual bools  (do_denoise, do_degibbs, do_fslpreproc, do_biascorrect, gradcheck)<br/>submitted via pydra Submitter"]

        SUBMIT --> GRADCHECK["Optional  (gradcheck in --processing-steps)<br/>DwiGradcheck → embed corrected gradients  (MrConvert)<br/>applied to DWI and RPE independently for rpe_all"]
        GRADCHECK --> DENOISE["Optional  (dwidenoise in --processing-steps)<br/>DwiDenoise"]
        DENOISE --> DEGIBBS["Optional  (mrgibbs in --processing-steps)<br/>MrDegibbs"]
        DEGIBBS --> PREPROC["Optional  (dwifslpreproc in --processing-steps)<br/>RunDwifslpreproc<br/>rpe_none  — no distortion correction<br/>rpe_pair  — spin-echo EPI pair  (RPE b0 + optional fwd-b0)<br/>rpe_split — SE pair from split series<br/>rpe_all   — full concatenated AP+PA  (RunDwicat first)"]
        PREPROC --> BIASCORR["Optional  (dwibiascorrect in --processing-steps)<br/>mrthreshold (Otsu) → DwiBiascorrect_Ants"]
        BIASCORR --> B0NII["DwiExtract → MrMath (mean, axis 3) → MrConvert<br/>→ b0_mean.nii.gz"]
        B0NII --> FLIRT["FlirtCoregister<br/>b0→T1 (6 DOF, flirt)<br/>invert xfm → T1→DWI space (flirt -applyxfm)<br/>→ T1_in_DWI_space.nii.gz"]
        B0NII --> TENSOR["Dwi2Tensor → Tensor2Metric → CastToMif → MrConvert<br/>→ ADC.nii.gz  FA.nii.gz"]
        FLIRT --> COPY["Copy outputs from pydra cache → series_out/<br/>T1_in_DWI_space.nii.gz · ADC.nii.gz · FA.nii.gz<br/>DWI_{steps}.mif.gz  (name reflects steps applied)<br/>DWI_raw.mif.gz  (for dual-viewer in DWI.html)"]
    end

    subgraph S3["Stage 3 — Native Contrast QC  (pipeline.py)"]
        direction TB
        STAGE3["run_stage3()<br/>stage T1 + IR + TE → NIfTI staging folder<br/>DICOM → dcm2niix  ·  NIfTI → copy  ·  MIF → mrconvert"]
        STAGE3 --> WF3["PhantomProcessor.process_session(T1.nii.gz)<br/>— see PhantomSessionWf detail —"]
    end

    COPY --> S2

    subgraph S2["Stage 2 — DWI-space Phantom QC  (pipeline.py)"]
        direction TB
        WF2["run_stage2()<br/>PhantomProcessor.process_session(T1_in_DWI_space.nii.gz)<br/>— see PhantomSessionWf detail —"]
    end

    subgraph WF["PhantomSessionWf — task dependency graph  (phantom_processor.py)"]
        direction TB
        REG["①  antsRegistrationSyN.sh  (rigid, -t r)<br/>T1 → template  ►  0GenericAffine.mat<br/>InverseWarped.nii.gz"]

        REG -->|inverse_warped| SAVE["①b  mrconvert<br/>TemplatePhantom_ScannerSpace.nii.gz"]
        REG -->|transform_matrix| VIALS["②  antsApplyTransforms  (inverse, NearestNeighbor)<br/>vial masks → subject space"]
        REG -->|transform_matrix| FWDXFM["⑤  antsApplyTransforms<br/>all contrasts → template space"]

        VIALS -->|vial_paths| METRICS["③  mrgrid + mrstats + mrdump + numpy<br/>per-vial mean / median / std / min / max / count<br/>p25 / p75  ·  mean_mad / median_mad<br/>→ xlsx (one sheet per metric)"]
        REFDATA[/"template_data/{phantom}/<br/>adc_reference  ·  t1t2_reference<br/>(SPIRIT: 12 vials · 120E: 24 vials)"/]
        REFDATA --> PLOTS
        METRICS -->|sentinel| PLOTS["④  plot_vial_intensity<br/>plot_vial_ir_means_std  ← if IR series present<br/>plot_vial_te_means_std  ← if TE series present<br/>→ Interactive HTML  (NiiVue viewer + Chart.js)<br/>PNG fallback available"]

        PLOTS -->|sentinel| CLEANUP["⑥  shutil.rmtree<br/>tmp  tmp_vials  tmp_vols  vial_dir/tmp"]
        FWDXFM -->|sentinel| CLEANUP
    end

    S1 -.->|"runs in parallel"| S3
    S2 -.->|"runs after Stage 1"| DONE
    S3 -.-> DONE

    DONE(["outputs/<br/>  {session}/metrics/plots/*.html<br/>     scatter plots · T1_mapping · T2_mapping<br/>  {session}/metrics/fits/*.csv<br/>  {session}/metrics/csv/<br/>  {session}/vial_segmentations/<br/>  {session}/images_template_space/<br/>  {session}/TemplatePhantom_ScannerSpace.nii.gz"])
```
