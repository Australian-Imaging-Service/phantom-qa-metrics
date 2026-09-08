Quick start
===========

PhantomKit can be used through the ``phantomkit`` command-line interface or
directly via its Python API.


End-to-end pipeline (CLI)
--------------------------

The fastest way to process a session is the ``pipeline`` command.  Point it at
a directory of DICOM subdirectories, specify an output location, and name the
phantom:

.. code-block:: console

    $ phantomkit pipeline \
          --input-dir  /data/session01 \
          --output-dir /results/session01 \
          --phantom    SPIRIT

PhantomKit will:

1. Scan ``--input-dir`` and classify subdirectories into T1, IR, TE, and DWI
   series automatically.
2. Run DWI preprocessing (Stage 1) if DWI series are found.
3. Run phantom QC in DWI space (Stage 2) on each preprocessed DWI output.
4. Run phantom QC on native T1/IR/TE contrasts (Stage 3) if those series are
   present.

See :doc:`cli` for the full option reference.


Python API — single session
----------------------------

The :class:`~phantomkit.phantom_processor.PhantomProcessor` class orchestrates
phantom QC for a single session end-to-end without the pipeline scaffolding:

.. code-block:: python

    from phantomkit.phantom_processor import PhantomProcessor

    processor = PhantomProcessor(
        template_dir="/templates/SPIRIT",
        output_base_dir="/results",
        rotation_library_file="/templates/rotations.txt",
    )
    results = processor.process_session("/data/session01/T1.nii.gz")

    print(results["metrics_dir"])
    print(results["vial_dir"])
    print(results["images_template_space_dir"])

All ``*.nii.gz`` files in the same directory as the input image are treated as
additional contrast images.


Python API — pydra workflow
----------------------------

For finer-grained control, :class:`~phantomkit.analyses.vial_signal.VialSignalAnalysis`
exposes the same processing as a pydra workflow, enabling lazy execution and
caching:

.. code-block:: python

    from phantomkit.analyses.vial_signal import VialSignalAnalysis

    wf = VialSignalAnalysis(
        input_image="/data/session01/T1.nii.gz",
        template_dir="/templates/SPIRIT",
        rotation_library_file="/templates/rotations.txt",
        output_base_dir="/results",
    )
    outputs = wf(cache_root="/data/pydra-cache")

    print(outputs.metrics_dir)

Run ``phantomkit list`` to discover all available protocol workflows.  Batch
processing is available via :class:`~phantomkit.analyses.vial_signal.VialSignalAnalysisBatch`.


Output structure
-----------------

A completed session produces:

.. code-block:: text

    /results/session01/
      metrics/
        session01_<contrast>_mean_matrix.csv
        session01_<contrast>_std_matrix.csv
        session01_<contrast>_PLOTmeanstd.png
        session01_ir_map_PLOTmeanstd_T1mapping.png   (if IR contrasts present)
        session01_TE_map_PLOTmeanstd_TEmapping.png   (if TE contrasts present)
      vial_segmentations/
        A.nii.gz … T.nii.gz
      images_template_space/
        <contrast>.nii.gz …
      TemplatePhantom_ScannerSpace.nii.gz

When called through the pipeline, DWI series each get their own subdirectory:

.. code-block:: text

    /results/session01/
      <DWI_series_name>/
        DWI_preproc_biascorr.mif.gz
        ADC.nii.gz
        FA.nii.gz
        T1_in_DWI_space.nii.gz
        metrics/
        vial_segmentations/
      native_contrasts/
        metrics/
        vial_segmentations/
        images_template_space/