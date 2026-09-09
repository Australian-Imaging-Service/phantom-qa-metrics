# phantomkit XNAT CS test data

`tests/test_phantom_qc_preprocess.py` uploads whatever is under
`scans/` here to a fresh XNAT project via `upload_test_dataset_to_xnat`
(root `conftest.py`), then launches the pipeline against it.

`scans/` itself isn't committed to git (real scan data — copy it in
separately, e.g. onto the target VM at
`tests/data/specs/phantom-qc/preprocess/scans/`). Its expected layout is
XNAT-archive-style, one directory per scan, resources nested underneath:

```
scans/
  <SeriesNumber>-<SeriesDescription>/
    resources/
      DICOM/
        files/
          *.dcm
```

`upload_test_dataset_to_xnat` reads each scan's XNAT type/id directly from
its DICOM headers (`SeriesDescription`/`SeriesNumber`) — the directory name
itself doesn't need to match anything (it's just there for humans; the
folder-naming convention sanitizes descriptions differently than the DICOM
header does — e.g. a header of `SEQ-B_NIF_AX_DWI_b1000_AP` shows up in a
folder named `27-SEQ_B_NIF_AX_DWI_b1000_AP`, hyphen swapped for
underscore). A flat, non-DICOM layout (`<scan>/<resource>/*`, resource type
taken directly from the scan directory's own name) also works, for
hand-authored synthetic NIfTI test data.

The fixed protocol this test exercises needs, at minimum, one scan per
source declared in `specs/phantomkit/preprocess.yaml`: one MPRAGE, one
forward-phase-encode DWI, one reverse-phase-encode DWI (both full
multi-volume acquisitions — `rpe_all`), 12 TI scans (T1 mapping), and 8 TE
scans (T2 mapping). `tests/test_phantom_qc_preprocess.py`'s
`T1W_SCAN_TYPE`/`DWI_SCAN_TYPE`/`RPE_SCAN_TYPE`/`TI_SCAN_TYPES`/
`TE_SCAN_TYPES` constants are the exact `SeriesDescription` match strings
the test launches with — update them if a different dataset's series
descriptions differ from the ones currently hardcoded there.
