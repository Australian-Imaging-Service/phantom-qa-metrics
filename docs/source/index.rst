.. _home:

PhantomKit
==========

.. image:: https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml/badge.svg
   :target: https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml
.. image:: https://codecov.io/gh/australian-imaging-service/phantomkit/branch/main/graph/badge.svg?token=UIS0OGPST7
   :target: https://codecov.io/gh/australian-imaging-service/phantomkit
.. image:: https://img.shields.io/pypi/pyversions/phantomkit.svg
   :target: https://pypi.python.org/pypi/phantomkit/
   :alt: Supported Python versions
.. image:: https://img.shields.io/pypi/v/phantomkit.svg
   :target: https://pypi.python.org/pypi/phantomkit/
   :alt: Latest Version

*PhantomKit* is a Python toolkit for automated quality assurance (QA) of MRI
scanners using physical phantoms. It provides pydra-based workflows and a
high-level processing engine that register phantom scans to a reference
template, extract per-vial signal statistics across multiple contrast types,
and generate publication-quality plots — with full support for DWI
preprocessing and diffusion metric (ADC, FA) extraction.

Key features:

- **End-to-end pipeline** — single ``phantomkit pipeline`` command processes a
  raw DICOM session directory through DWI preprocessing (Stage 1), phantom QC
  in DWI space (Stage 2), and native contrast QC (Stage 3)
- **Automatic series classification** — DWI, reverse phase-encode, T1, IR, and
  TE series are detected from folder names and DICOM sidecar metadata; no manual
  configuration required
- **DWI preprocessing** — FSL ``dwifslpreproc`` with automatic phase-encoding
  correction mode selection (``rpe_none``, ``rpe_pair``, ``rpe_all``), optional
  denoising/Gibbs correction, tensor fitting, and T1-to-DWI co-registration via
  FLIRT
- **Template-based registration** — iterative ANTs rigid registration with
  automatic orientation search across a rotation library; vial masks propagated
  to subject space via inverse transform
- **Vial metric extraction** — per-vial mean, median, std, min and max across
  all contrast images, written to CSV
- **Plotting** — ADC/FA scatter plots with SPIRIT reference values, T1/IR and
  T2/TE parametric map plots with mrview ROI overlays
- **Parallel batch processing** — pydra-native splitting and combining for
  multi-session datasets
- **Checkpoint-based resumption** — re-running the pipeline skips stages whose
  outputs already exist


Installation
------------

*PhantomKit* can be installed for Python >=3.11 using *pip*:

.. code-block:: console

    $ python3 -m pip install phantomkit

External dependencies (must be on ``PATH``):

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_ — ``dwifslpreproc``,
  ``flirt``, ``convert_xfm``
- `MRtrix3 <https://www.mrtrix.org/>`_ — ``mrconvert``, ``dwi2tensor``,
  ``tensor2metric``, ``dwidenoise``, ``mrdegibbs``, ``mrstats``, ``mrview``
- `ANTs <http://stnava.github.io/ANTs/>`_ — ``antsRegistrationSyN.sh``,
  ``antsApplyTransforms``
- `dcm2niix <https://github.com/rordenlab/dcm2niix>`_ — DICOM to NIfTI
  conversion


License
-------

*PhantomKit* is released under the
`Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Copyright 2026 Australian Imaging Service.


.. toctree::
    :maxdepth: 2
    :hidden:

    quick_start
    cli
    api