"""Command-line interface for phantom QA processing."""

import importlib
import logging
import pkgutil
import shutil
from pathlib import Path
from typing import get_args, get_origin

import attrs
import click

import phantomkit.analyses

logger = logging.getLogger(__name__)

# Fields that are internal to pydra or used for input routing
_SKIP_FIELDS = frozenset({"constructor"})
_INPUT_FIELD_NAMES = frozenset({"input_image", "input_images"})


# ---------------------------------------------------------------------------
# Protocol discovery
# ---------------------------------------------------------------------------


def _discover_protocols() -> dict[str, tuple]:
    """
    Scan ``phantomkit.analyses`` for (single, batch) workflow pairs.

    Returns a dict mapping a hyphenated slug (e.g. ``"vial-signal"``) to a
    ``(single_cls, batch_cls_or_None)`` tuple.  Only non-private modules that
    contain at least one pydra workflow class (detected by ``Outputs``
    attribute) are included.
    """
    found: dict[str, tuple] = {}
    for info in pkgutil.iter_modules(phantomkit.analyses.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"phantomkit.analyses.{info.name}")
        for attr_name, obj in vars(mod).items():
            if attr_name.startswith("_") or not attr_name.endswith("Analysis"):
                continue
            if isinstance(obj, type) and hasattr(obj, "Outputs"):
                slug = info.name.replace("_", "-")
                batch = getattr(mod, f"{attr_name}Batch", None)
                found[slug] = (obj, batch)
                break  # one primary workflow per module
    return found


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _is_required(field: attrs.Attribute) -> bool:
    """Return True if an attrs field has no real default (pydra NOTHING)."""
    if isinstance(field.default, attrs.Factory):
        return getattr(field.default.factory, "__name__", "") == "nothing_factory"
    return field.default is attrs.NOTHING


def _workflow_fields(cls) -> list[attrs.Attribute]:
    """Return user-visible attrs fields, excluding internal pydra fields."""
    return [
        f
        for f in attrs.fields(cls)
        if not f.name.startswith("_") and f.name not in _SKIP_FIELDS
    ]


def _annotation_to_click(ann) -> tuple[click.ParamType, bool]:
    """
    Map a Python type annotation to a ``(click_type, multiple)`` pair.

    ``multiple=True`` means the option should be repeatable (for ``list[X]``).
    """
    import types as _types
    import typing

    origin = get_origin(ann)
    args = get_args(ann)

    # Union / Optional  (X | None  or  Optional[X])
    if origin is typing.Union or isinstance(ann, _types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _annotation_to_click(non_none[0])

    # list[X]
    if origin is list:
        inner, _ = _annotation_to_click(args[0]) if args else (click.STRING, False)
        return inner, True

    # fileformats FileSet subclasses → paths
    try:
        from fileformats.core import FileSet

        if isinstance(ann, type) and issubclass(ann, FileSet):
            return click.Path(), False
    except ImportError:
        pass

    if ann is Path or (isinstance(ann, type) and issubclass(ann, Path)):
        return click.Path(), False
    if ann is int:
        return click.INT, False
    if ann is float:
        return click.FLOAT, False
    if ann is bool:
        return click.BOOL, False
    return click.STRING, False


# ---------------------------------------------------------------------------
# Dynamic command builder
# ---------------------------------------------------------------------------


def _build_command(slug: str, single_cls, batch_cls) -> click.Command:
    """Build a click Command whose options mirror the workflow's parameters."""

    single_fields = {f.name: f for f in _workflow_fields(single_cls)}

    def _callback(**kwargs):
        input_val: str = kwargs.pop("input")
        worker: str = kwargs.pop("worker")
        pattern: str = kwargs.pop("pattern")
        # Convert click tuples (from multiple=True) to lists
        wf_kwargs = {
            k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }
        # Remove None values for optional params (keep explicit None only if needed)
        wf_kwargs = {
            k: v
            for k, v in wf_kwargs.items()
            if v is not None
            or (k in single_fields and not _is_required(single_fields[k]))
        }

        path = Path(input_val)

        if path.is_dir():
            # Batch mode: glob for images
            if batch_cls is None:
                raise click.UsageError(
                    f"Protocol '{slug}' does not support batch mode."
                )
            images = sorted(str(p) for p in path.rglob(pattern))
            if not images:
                raise click.UsageError(
                    f"No files matching '{pattern}' found under {path}"
                )
            logger.info("Batch mode: found %d image(s):", len(images))
            for img in images:
                logger.info("  %s", img)
            wf = batch_cls(input_images=images, **wf_kwargs)
        else:
            # Single mode
            wf = single_cls(input_image=input_val, **wf_kwargs)

        from pydra.engine import Submitter

        with Submitter(worker=worker) as sub:
            sub(wf)
        logger.info("Done.")

    # Build click params from the workflow's attrs fields
    click_params: list[click.Parameter] = []
    for name, field in single_fields.items():
        if name in _INPUT_FIELD_NAMES:
            continue
        ann = field.type if field.type is not None else str
        click_type, multiple = _annotation_to_click(ann)
        required = _is_required(field) and not multiple
        default = None if _is_required(field) else field.default
        click_params.append(
            click.Option(
                [f"--{name.replace('_', '-')}"],
                type=click_type,
                default=default,
                required=required,
                multiple=multiple,
                show_default=default is not None,
                help=f"{name}.",
            )
        )

    # Common options
    click_params += [
        click.Option(
            ["--worker"],
            type=click.Choice(["cf", "serial"]),
            default="cf",
            show_default=True,
            help="Pydra worker type.",
        ),
        click.Option(
            ["--pattern"],
            default="*.nii.gz",
            show_default=True,
            help=(
                "Glob pattern used when INPUT is a directory (batch mode) "
                "to find NIfTI images recursively."
            ),
        ),
        click.Argument(["input"]),
    ]

    # Compose docstring from workflow
    doc = getattr(single_cls, "__doc__", None) or ""
    first_line = (
        doc.strip().splitlines()[0] if doc.strip() else f"Run the {slug} protocol."
    )
    help_text = (
        f"{first_line}\n\n"
        "INPUT may be a single NIfTI file (single-session mode) or a directory "
        "(batch mode — images are found with --pattern)."
    )

    return click.Command(
        name=slug,
        callback=_callback,
        params=click_params,
        help=help_text,
    )


# ---------------------------------------------------------------------------
# Main CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Phantom QA processing CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


@main.command("list")
def list_protocols() -> None:
    """List all available protocol workflows."""
    protocols = _discover_protocols()
    if not protocols:
        click.echo("No protocols found.")
        return
    click.echo("Available protocols:")
    for slug, (single_cls, batch_cls) in protocols.items():
        batch_info = " (batch supported)" if batch_cls is not None else ""
        doc = getattr(single_cls, "__doc__", "") or ""
        summary = doc.strip().splitlines()[0] if doc.strip() else ""
        click.echo(f"  {slug}{batch_info}")
        if summary:
            click.echo(f"    {summary}")


@main.group()
def run() -> None:
    """Run a phantom QA protocol workflow.

    Use ``phantomkit run <protocol> --help`` to see the options for a specific
    protocol.  Pass a single NIfTI file as INPUT for single-session mode, or a
    directory for batch mode.
    """


def _register_commands() -> None:
    protocols = _discover_protocols()
    for slug, (single_cls, batch_cls) in protocols.items():
        run.add_command(_build_command(slug, single_cls, batch_cls))


# ---------------------------------------------------------------------------
# Plot subgroup — collects the per-module click commands from phantomkit.plotting
# ---------------------------------------------------------------------------


@main.group()
def plot() -> None:
    """Generate QA plots.

    Use ``phantom-process plot <subcommand> --help`` for options.
    """


def _register_plot_commands() -> None:
    import phantomkit.plotting as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"phantomkit.plotting.{info.name}")
        cmd = getattr(mod, "main", None)
        if isinstance(cmd, (click.Command, click.Group)):
            plot.add_command(cmd, name=info.name.replace("_", "-"))


_register_commands()
_register_plot_commands()


# ---------------------------------------------------------------------------
# View command — NiceGUI MRI viewer
# ---------------------------------------------------------------------------


@main.command("view")
@click.argument("nifti_image", metavar="NIFTI")
@click.option(
    "--vials",
    "vials_dir",
    default=None,
    metavar="DIR",
    help="Directory containing vial ROI NIfTI masks (.nii.gz).",
)
@click.option("--title", default="MRI Viewer", show_default=True)
@click.option(
    "--port", default=8080, show_default=True, help="TCP port for the NiceGUI server."
)
def view_mri(nifti_image: str, vials_dir: str | None, title: str, port: int) -> None:
    """Launch an interactive MRI viewer for a NIfTI image.

    NIFTI is the path to the background .nii or .nii.gz file.

    Use --vials to specify a directory of vial ROI masks that can be
    toggled on/off as overlays.
    """
    from phantomkit.plotting.viewer import launch_viewer

    vial_niftis: dict[str, str] = {}
    if vials_dir:
        vials_path = Path(vials_dir)
        for p in sorted(vials_path.glob("*.nii.gz")):
            name = p.name.replace(".nii.gz", "").replace(".nii", "")
            vial_niftis[name] = str(p)

    launch_viewer(
        nifti_image=nifti_image,
        vial_niftis=vial_niftis or None,
        title=title,
        port=port,
    )


# ---------------------------------------------------------------------------
# Pipeline command — end-to-end phantom QC + DWI orchestrator
# ---------------------------------------------------------------------------


@main.command("pipeline")
@click.option(
    "--input-dir",
    required=True,
    help="Root directory containing acquisition subdirectories (DICOM folders).",
)
@click.option(
    "--output-dir",
    required=True,
    help="Top-level output directory. All results are written here.",
)
@click.option(
    "--phantom",
    required=True,
    help="Phantom name, e.g. SPIRIT. Used to locate template_data/<phantom>/.",
)
@click.option(
    "--denoise-degibbs",
    is_flag=True,
    default=False,
    help="Apply dwidenoise + mrdegibbs before preprocessing.",
)
@click.option(
    "--gradcheck",
    is_flag=True,
    default=False,
    help="Run dwigradcheck to verify gradient orientations.",
)
@click.option(
    "--nocleanup",
    is_flag=True,
    default=False,
    help="Keep DWI tmp/ intermediate directories.",
)
@click.option(
    "--readout-time",
    type=float,
    default=None,
    help="Override TotalReadoutTime (seconds) for dwifslpreproc.",
)
@click.option(
    "--eddy-options",
    type=str,
    default=None,
    help="Override FSL eddy options string.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Plan and print commands; do not execute any processing.",
)
def run_pipeline(
    input_dir,
    output_dir,
    phantom,
    denoise_degibbs,
    gradcheck,
    nocleanup,
    readout_time,
    eddy_options,
    dry_run,
):
    """Run the end-to-end phantom QC + DWI processing pipeline.

    Orchestrates DWI preprocessing (Stage 1), phantom QC in DWI space
    (Stage 2), and phantom QC on native T1/IR/TE contrasts (Stage 3).
    Stages 1 and 3 run in parallel; Stage 2 follows Stage 1.

    INPUT_DIR should contain acquisition subdirectories (DICOM folders).
    """
    import sys
    from pathlib import Path
    from phantomkit.pipeline import (
        scan_input_dir,
        validate_inputs,
        run_stage1,
        run_stage2,
        run_stage3,
        print_header,
        TEMPLATE_DATA_ROOT,
    )
    import concurrent.futures
    import threading

    # Build a minimal args-like namespace so validate_inputs() can be reused
    class _Args:
        pass

    _args = _Args()
    _args.input_dir = input_dir
    _args.output_dir = output_dir
    _args.phantom = phantom

    validate_inputs(_args)

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    template_dir = TEMPLATE_DATA_ROOT / phantom

    output_path.mkdir(parents=True, exist_ok=True)

    dwi_cfg = {
        "denoise_degibbs": denoise_degibbs,
        "gradcheck": gradcheck,
        "nocleanup": nocleanup,
        "readout_time": readout_time,
        "eddy_options": eddy_options,
    }

    print_header("Input Directory Scan")
    scan_info = scan_input_dir(input_path)

    run_stage1_flag = scan_info["has_dwi"]
    run_stage3_flag = bool(scan_info["t1_dirs"]) and (
        scan_info["has_native_contrasts"] or not run_stage1_flag
    )

    _print_lock = threading.Lock()

    def _locked_header(title):
        with _print_lock:
            print_header(title)

    dwi_output_dirs = []
    stage1_error = stage3_error = None

    def _s1():
        nonlocal dwi_output_dirs, stage1_error
        try:
            if run_stage1_flag:
                dwi_output_dirs[:] = run_stage1(
                    input_path, output_path, dwi_cfg, dry_run
                )
            else:
                _locked_header("STAGE 1 — DWI Processing")
                print("  Skipped: no DWI acquisitions found.\n")
        except Exception as exc:
            stage1_error = exc

    def _s3():
        nonlocal stage3_error
        try:
            if run_stage3_flag:
                run_stage3(input_path, output_path, template_dir, scan_info, dry_run)
            else:
                _locked_header("STAGE 3 — Phantom QC on Native Contrasts")
                print("  Skipped.\n")
        except Exception as exc:
            stage3_error = exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        concurrent.futures.wait([executor.submit(_s1), executor.submit(_s3)])

    if stage1_error:
        raise click.ClickException(f"Stage 1 failed: {stage1_error}")
    if stage3_error:
        raise click.ClickException(f"Stage 3 failed: {stage3_error}")

    if run_stage1_flag:
        run_stage2(dwi_output_dirs, output_path, template_dir, dry_run)
    else:
        print_header("STAGE 2 — Phantom QC in DWI Space")
        print("  Skipped: Stage 1 did not run.\n")

    # Remove staging-only DWI directories (contain only tmp/, no final outputs).
    # These are created by convert_all_candidates for candidate series that were
    # not selected as the main DWI output (e.g. RPE pair b=0 volumes).
    if run_stage1_flag and not cfg.get("nocleanup"):
        _t1_markers = {"T1_in_DWI_space.nii.gz", "T1.nii.gz"}
        for d in sorted(output_path.iterdir()):
            if not d.is_dir():
                continue
            if any((d / m).exists() for m in _t1_markers):
                continue  # real DWI output dir — leave it alone
            if (d / "tmp").exists() and not any(
                p for p in d.iterdir()
                if p.name != "tmp" and not p.name.startswith(".")
            ):
                try:
                    shutil.rmtree(d)
                    print(f"  Removed staging dir: {d.name}")
                except Exception as e:
                    print(f"  Warning: could not remove {d.name}: {e}")

    print_header("Pipeline Complete")
    print(f"  All outputs written to: {output_path}\n")
