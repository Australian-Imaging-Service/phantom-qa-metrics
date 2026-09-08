API
===

This page documents the public Python API of PhantomKit.  The package is
organised into five sub-packages:

- :mod:`phantomkit.analyses` — top-level pydra workflows (user-facing entry
  points)
- :mod:`phantomkit.phantom_processor` — imperative processing engine
  (:class:`~phantomkit.phantom_processor.PhantomProcessor`)
- :mod:`phantomkit.registration` — ANTs registration tasks and workflows
- :mod:`phantomkit.metrics` — vial mask transform and metric extraction
  workflows
- :mod:`phantomkit.plotting` — QA plot generation functions and pydra tasks


Analyses
---------

The ``analyses`` sub-package contains the pydra workflow classes that serve as
the primary programmatic entry points.

Vial signal analysis
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: phantomkit.analyses.vial_signal.VialSignalAnalysis

.. autoclass:: phantomkit.analyses.vial_signal.VialSignalAnalysisBatch

Diffusion metrics analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: phantomkit.analyses.diffusion_metrics.DiffusionMetricsAnalysis

.. autoclass:: phantomkit.analyses.diffusion_metrics.DiffusionMetricsAnalysisBatch


Phantom processor
------------------

:class:`~phantomkit.phantom_processor.PhantomProcessor` is the imperative
counterpart to :class:`~phantomkit.analyses.vial_signal.VialSignalAnalysis`.
It is used internally by the pipeline orchestrator (Stages 2 and 3) and can
also be used directly for scripted single-session processing.

.. autoclass:: phantomkit.phantom_processor.PhantomProcessor
   :members: process_session


Registration
------------

.. autoclass:: phantomkit.registration.IterativeRegistration

.. autoclass:: phantomkit.registration.CheckRegistration

.. autoclass:: phantomkit.registration.SaveTemplateInScannerSpace

.. autoclass:: phantomkit.registration.RegistrationSynN


Metrics
-------

.. autoclass:: phantomkit.metrics.TransformVialsToSubjectSpace

.. autoclass:: phantomkit.metrics.ExtractMetricsFromContrasts

.. autoclass:: phantomkit.metrics.TransformContrastsToTemplateSpace


Plotting
--------

.. autofunction:: phantomkit.plotting.vial_intensity.plot_vial_intensity

.. autofunction:: phantomkit.plotting.maps_ir.plot_vial_ir_means_std

.. autofunction:: phantomkit.plotting.maps_te.plot_vial_te_means_std

.. autoclass:: phantomkit.plotting.visualization.GeneratePlots

.. autoclass:: phantomkit.plotting.visualization.BuildRoiOverlay