# phantomkit XNAT CS test data — scaffold only, no data yet

This directory is the expected layout for `tests/test_phantom_qc_preprocess.py`'s
test dataset — 23 scan directories, one per source declared in
`specs/phantomkit/preprocess.yaml`, matching the fixed acquisition protocol
(1 MPRAGE + AP/PA DWI (`rpe_all`) + 12 TI + 8 TE scans).

Each scan directory's name is the exact XNAT scan type the test uploads it
as (`upload_test_dataset_to_xnat` in the root `conftest.py` derives the type
directly from the directory name for non-DICOM data) — do not rename these
directories without also updating `TI_VALUES`/`TE_VALUES`/`command_inputs`
in the test file.

**The `NIFTI/` subdirectories are currently empty placeholders.** The test
is not runnable until real phantom scan data (SPIRIT phantom, or whichever
`Phantom` the test is pointed at) is dropped in:

- `T1w/NIFTI/`: one MPRAGE volume, e.g. `t1w.nii.gz`.
- `DWI/NIFTI/`, `RPE/NIFTI/`: forward/reverse phase-encode DWI volumes —
  each needs `dwi.nii.gz` **and** `dwi.bvec`/`dwi.bval` (and ideally
  `dwi.json` for the sidecar metadata `DWISeriesWorkflow` may read). Both
  must be full multi-volume DWI acquisitions (equal number of volumes),
  matching `rpe_all` mode.
- `TI_*/NIFTI/`, `TE_*/NIFTI/`: one relaxometry contrast volume each, e.g.
  `TI_50/NIFTI/ti_50.nii.gz`.

Any single-vial or synthetic phantom NIfTI works for a first pass at
exercising the pipeline end-to-end; a real SPIRIT-phantom acquisition is
needed to get meaningful per-vial QC metrics out the other end.
