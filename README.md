<img src="https://raw.githubusercontent.com/australian-imaging-service/phantomkit/main/docs/source/_static/logo.svg" alt="PhantomKit" width="600"/>

[![CI/CD](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/phantomkit/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/phantomkit)
[![PyPI version](https://img.shields.io/pypi/v/phantomkit.svg)](https://pypi.python.org/pypi/phantomkit/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://australian-imaging-service.github.io/phantomkit/)

# PhantomKit

**PhantomKit** is a Python toolkit for automated quality assurance (QA) of MRI scanners using physical phantoms. It provides a high-level processing engine and pydra-based workflows that register phantom scans to a reference template, extract per-vial signal statistics across multiple contrast types, and generate publication-quality plots — with full support for DWI preprocessing and diffusion metric (ADC, FA) extraction.

## Features

- **End-to-end pipeline** — single command processes a session directory through DWI preprocessing, phantom QC in DWI space, and native contrast QC
- **Format-agnostic input** — each series sub-directory may contain DICOM, NIfTI (`.nii`/`.nii.gz`), or MRtrix MIF (`.mif`/`.mif.gz`) files; format is detected automatically per sub-directory
- **Automatic series classification** — DWI, reverse phase-encode, T1, IR, and TE series are detected and paired automatically from folder names and sidecar metadata; no manual configuration required
- **DWI preprocessing** — FSL `dwifslpreproc` with automatic phase-encoding correction mode selection (`rpe_none`, `rpe_pair`, `rpe_all`), optional denoising/Gibbs correction, tensor fitting, and T1-to-DWI co-registration via FLIRT
- **Template-based registration** — iterative ANTs rigid registration with automatic orientation search across a rotation library; vial masks propagated to subject space via inverse transform
- **Vial metric extraction** — per-vial mean, median, std, min and max across all contrast images (T1, IR, TE, ADC, FA), written to CSV
- **Plotting** — ADC/FA scatter plots with SPIRIT reference values, T1/IR and T2/TE parametric map plots with mrview ROI overlays and Monte Carlo 95% CI bands
- **Checkpoint-based resumption** — re-running the pipeline skips stages whose outputs already exist
- **Parallel batch processing** — pydra-native splitting and combining for multi-session datasets

## Installation

```bash
python -m pip install phantomkit
```

### External dependencies

The pipeline requires FSL, MRtrix3, ANTs, and dcm2niix to be available on `PATH`:

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) — `dwifslpreproc`, `flirt`, `convert_xfm`
- [MRtrix3](https://www.mrtrix.org/) — `mrconvert`, `dwi2tensor`, `tensor2metric`, `dwidenoise`, `mrdegibbs`, `mrstats`, `mrview`
- [ANTs](http://stnava.github.io/ANTs/) — `antsRegistrationSyN.sh`, `antsApplyTransforms`
- [dcm2niix](https://github.com/rordenlab/dcm2niix) — DICOM to NIfTI conversion (required only when input is DICOM)

## Basic usage

### End-to-end pipeline

Point the pipeline at a session directory whose sub-directories contain DICOM, NIfTI, or MIF series. The format is detected automatically per sub-directory — no conversion needed beforehand.

```bash
# SPIRIT phantom (accepts DICOM, NIfTI, or MIF sub-directories)
phantomkit pipeline \
    --input-dir  /data/session01 \
    --output-dir /results/session01 \
    --phantom    SPIRIT

# 120E phantom (T1 and T2 mapping supported)
phantomkit pipeline \
    --input-dir  /data/session01 \
    --output-dir /results/session01 \
    --phantom    120E
```

Supported phantoms and their reference data:

| Phantom | ADC | T1 mapping | T2 mapping |
|---------|-----|------------|------------|
| SPIRIT  | ✓ (vials E–L) | ✓ (12 vials) | ✓ (12 vials) |
| 120E    | ✓ (all 24 vials) | ✓ (24 vials) | ✓ (24 vials) |

### Series naming conventions

Series sub-directories are classified automatically by **word-boundary token matching** on the folder or file name (case-insensitive). No manual tagging is required.

| Series type | Radiology term | Token match |
|---|---|---|
| Inversion Recovery | IR / MPRAGE | `ir` |
| Multi-Echo Spin Echo | MESE / T2-SE | `te` |
| T1-weighted | T1 / MPRAGE | `t1` or `mprage` |
| DWI | Diffusion-weighted | auto-detected via bvec/bval |
| Reverse phase-encode | B0 field map | paired with DWI |

For IR and MESE series, the **numeric suffix** in the filename is interpreted as the physical parameter: inversion time TI (ms) for IR, and echo time TE (ms) for MESE. Example: `se_ir_500.nii.gz` → TI = 500 ms; `t2_se_TE_80.nii.gz` → TE = 80 ms.

Optional flags:

```
--denoise-degibbs    Apply dwidenoise + mrdegibbs before preprocessing
--gradcheck          Run dwigradcheck to verify gradient orientations
--nocleanup          Keep intermediate tmp/ directories after completion
--readout-time       Override TotalReadoutTime (seconds) for dwifslpreproc
--eddy-options       Override FSL eddy options string
--dry-run            Print the planned workflow without executing
```

Output structure:

```
/results/session01/
  <DWI_series>/
    DWI_preproc_biascorr.mif.gz
    ADC.nii.gz
    FA.nii.gz
    T1_in_DWI_space.nii.gz
    metrics/                      ← per-vial CSVs and QA plots
    vial_segmentations/
  native_contrasts/
    metrics/                      ← per-vial CSVs and parametric map plots
    vial_segmentations/
    images_template_space/
```

### Python API

```python
from phantomkit.phantom_processor import PhantomProcessor

processor = PhantomProcessor(
    template_dir="/templates/SPIRIT",
    output_base_dir="/results",
    rotation_library_file="/templates/rotations.txt",
)
results = processor.process_session("/data/session01/T1.nii.gz")
```

Or via the pydra workflow API:

```python
from phantomkit.analyses.vial_signal import VialSignalAnalysis

wf = VialSignalAnalysis(
    input_image="/data/session01/T1.nii.gz",
    template_dir="/templates/SPIRIT",
    rotation_library_file="/templates/rotations.txt",
)
outputs = wf(cache_root="/data/pydra-cache")
```

### CLI — run a protocol directly

```bash
# Single session
phantomkit run vial-signal /data/session01/T1.nii.gz \
    --template-dir          /templates/SPIRIT \
    --rotation-library-file /templates/rotations.txt \
    --output-base-dir       /results

# Batch mode — process every matching image under /data/
phantomkit run vial-signal /data/ \
    --template-dir          /templates/SPIRIT \
    --rotation-library-file /templates/rotations.txt \
    --output-base-dir       /results \
    --pattern               "*T1*.nii.gz"

# List available protocols
phantomkit list
```

### Plotting

Generate QA plots from existing CSV metric files:

```bash
# Generic scatter plot
phantomkit plot vial-intensity \
    /results/session01/metrics/session01_T1_mean_matrix.csv scatter \
    --std-csv /results/session01/metrics/session01_T1_std_matrix.csv \
    --output  /results/session01/metrics/session01_T1_PLOTmeanstd.png

# ADC scatter plot with SPIRIT reference values
phantomkit plot vial-intensity \
    /results/session01/metrics/session01_ADC_mean_matrix.csv scatter \
    --std-csv      /results/session01/metrics/session01_ADC_std_matrix.csv \
    --phantom      SPIRIT \
    --template-dir /templates \
    --output       /results/session01/metrics/session01_ADC_PLOTmeanstd.png

# T1 inversion-recovery parametric map plot
phantomkit plot maps-ir \
    /results/session01/images_template_space/se_ir_*.nii.gz \
    --metric-dir /results/session01/metrics \
    --output     /results/session01/metrics/session01_T1map_plot.png

# T2 spin-echo parametric map plot
phantomkit plot maps-te \
    /results/session01/images_template_space/t2_se_*.nii.gz \
    --metric-dir /results/session01/metrics \
    --output     /results/session01/metrics/session01_T2map_plot.png
```

See the [CLI documentation](https://australian-imaging-service.github.io/phantomkit/cli.html) for the full option reference.

## License

Copyright 2026 Australian Imaging Service. Released under the [Apache License 2.0](LICENSE).