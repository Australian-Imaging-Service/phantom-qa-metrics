Command-line interface
======================

PhantomKit installs the ``phantomkit`` command, which groups four subcommand
families:

- ``phantomkit pipeline`` — end-to-end phantom QC + DWI orchestrator
- ``phantomkit run`` — execute a single pydra protocol workflow
- ``phantomkit plot`` — generate individual QA plots from existing CSVs
- ``phantomkit list`` — list all available protocol workflows

Use ``--help`` at any level for a full option listing::

    phantomkit --help
    phantomkit pipeline --help
    phantomkit run --help
    phantomkit run vial-signal --help
    phantomkit plot --help
    phantomkit plot vial-intensity --help


End-to-end pipeline
--------------------

The ``pipeline`` command is the primary entry point for processing a complete
scanner session.  It scans ``--input-dir`` for acquisition subdirectories,
classifies them automatically, and runs up to three stages:

- **Stage 1** — DWI preprocessing (runs if DWI acquisitions are found)
- **Stage 2** — Phantom QC in DWI space (runs per DWI series after Stage 1)
- **Stage 3** — Phantom QC on native T1/IR/TE contrasts (runs if those series
  are present)

Stages 1 and 3 execute in parallel where possible.

.. code-block:: console

    $ phantomkit pipeline \
          --input-dir  /data/session01 \
          --output-dir /results/session01 \
          --phantom    SPIRIT

Options:

.. code-block:: text

    --input-dir PATH       Root directory containing acquisition subdirectories
                           (DICOM folders).  [required]
    --output-dir PATH      Top-level output directory.  All results are written
                           here.  [required]
    --phantom TEXT         Phantom name, e.g. SPIRIT.  Used to locate
                           template_data/<phantom>/.  [required]
    --denoise-degibbs      Apply dwidenoise + mrdegibbs before preprocessing.
    --gradcheck            Run dwigradcheck to verify gradient orientations.
    --nocleanup            Keep DWI tmp/ intermediate directories.
    --readout-time FLOAT   Override TotalReadoutTime (seconds) for dwifslpreproc.
    --eddy-options TEXT    Override FSL eddy options string.
    --dry-run              Plan and print commands; do not execute any processing.

Output structure:

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


Discovering available protocols
---------------------------------

.. code-block:: console

    $ phantomkit list
    Available protocols:
      vial-signal (batch supported)
        Pydra workflow for processing a single GSP SPIRIT phantom MRI session.
      diffusion-metrics (batch supported)
        Full SPIRIT phantom DWI analysis workflow.


Running a protocol workflow
----------------------------

The ``run`` subgroup executes pydra protocol workflows directly, bypassing the
pipeline orchestrator.

Single session
~~~~~~~~~~~~~~

Pass a path to a NIfTI file as ``INPUT`` to process one session:

.. code-block:: console

    $ phantomkit run vial-signal /data/session01/T1.nii.gz \
          --template-dir          /templates/SPIRIT \
          --rotation-library-file /templates/rotations.txt \
          --output-base-dir       /results

The session name is taken from the parent directory of the image
(``session01`` in the example above).  All ``*.nii.gz`` files in the same
directory are treated as additional contrast images.

Batch mode
~~~~~~~~~~

Pass a *directory* as ``INPUT`` to process every matching image found
recursively inside it:

.. code-block:: console

    $ phantomkit run vial-signal /data \
          --template-dir          /templates/SPIRIT \
          --rotation-library-file /templates/rotations.txt \
          --output-base-dir       /results \
          --pattern               "*T1*.nii.gz" \
          --worker                cf

``--pattern`` controls which files are selected (default ``*.nii.gz``).
``--worker`` selects the pydra execution backend: ``cf`` (concurrent futures,
default) for parallel execution or ``serial`` for sequential.

Protocol options
~~~~~~~~~~~~~~~~

All parameters of the underlying workflow are exposed as CLI options.  Run
``phantomkit run <protocol> --help`` to see the full list.  For
``vial-signal``:

.. code-block:: text

    Options:
      --template-dir PATH             Path to the GSP SPIRIT template directory.
                                      [required]
      --rotation-library-file PATH    Path to the rotation library text file.
                                      [required]
      --output-base-dir PATH          Root output directory (default: cwd).
      --worker [cf|serial]            Pydra execution backend.  [default: cf]
      --pattern TEXT                  Glob pattern for batch mode.
                                      [default: *.nii.gz]


Generating plots
-----------------

The ``plot`` subcommands generate standalone QA figures from CSV metric files
already written to disk (e.g. after a completed ``run`` or ``pipeline``).

Vial intensity scatter plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots per-vial mean ± std as a scatter, line, or bar chart for a single
contrast.  In ADC mode (auto-detected from filename), measured values are
overlaid against SPIRIT reference values:

.. code-block:: console

    $ phantomkit plot vial-intensity \
          /results/session01/metrics/session01_T1_mean_matrix.csv \
          scatter \
          --std-csv    /results/session01/metrics/session01_T1_std_matrix.csv \
          --output     /results/session01/metrics/session01_T1_PLOTmeanstd.png

    # ADC mode — reference values overlaid automatically
    $ phantomkit plot vial-intensity \
          /results/session01/metrics/session01_ADC_mean_matrix.csv \
          scatter \
          --std-csv      /results/session01/metrics/session01_ADC_std_matrix.csv \
          --phantom      SPIRIT \
          --template-dir /templates \
          --output       /results/session01/metrics/session01_ADC_PLOTmeanstd.png

Options:

.. code-block:: text

    Arguments:
      CSV_FILE              Mean-value CSV produced by metric extraction.
      {line|bar|scatter}    Plot style.

    Options:
      --std-csv PATH        CSV file containing standard deviations.
      --roi-image PATH      PNG screenshot of the contrast with vial ROI overlay.
      --annotate            Annotate each point with mean ± std.
      --phantom TEXT        Phantom name (required in ADC mode).
      --template-dir PATH   TemplateData directory (required in ADC mode).
      --output PATH         Output filename.  [default: vial_subplot.png]

Inversion-recovery (T1) map plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots vial mean ± std across multiple IR contrasts with T₁ inversion-recovery
curve fitting and 95% Monte Carlo confidence intervals:

.. code-block:: console

    $ phantomkit plot maps-ir \
          /results/session01/images_template_space/se_ir_*.nii.gz \
          --metric-dir /results/session01/metrics \
          --output     /results/session01/metrics/session01_T1map_plot.png

Options:

.. code-block:: text

    Arguments:
      CONTRAST_FILES...   IR NIfTI images in template space (one per TI).

    Options:
      -m, --metric-dir PATH   Directory containing mean/std CSV files.  [required]
      -o, --output PATH       Output filename.  [default: vial_summary_T1.png]
      --annotate              Annotate each point with mean ± std.
      --roi-image PATH        PNG ROI overlay for the extra subplot.

Spin-echo (T2/TE) map plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots vial mean ± std across multiple TE contrasts with mono-exponential T₂
decay fitting and 95% Monte Carlo confidence intervals:

.. code-block:: console

    $ phantomkit plot maps-te \
          /results/session01/images_template_space/t2_se_*.nii.gz \
          --metric-dir /results/session01/metrics \
          --output     /results/session01/metrics/session01_T2map_plot.png

Options are identical to ``maps-ir``, except the fitted parameter is T₂ and
the output CSV records T₂ values.