"""
pet_orientation.py
===================
Automatic gross-orientation correction for multi-site PET phantom images.

Problem
-------
PET NIfTI headers exported by different sites/scanners frequently do **not**
encode the true scan orientation: the sform/qform is often left at a generic,
near-identity value regardless of how the phantom was actually positioned in
the scanner. Because of this, simply matching voxel strides to the template
(``mrtransform -strides``, the first step of :func:`phantomkit.pet_processing
.register_pet_to_template`) is not sufficient on its own — strides-matching
only changes how the voxel array is *stored* to mirror a reference layout, it
does not alter image content. An image whose phantom is genuinely rotated by
a multiple of 90 degrees relative to the template will still be off by that
amount afterwards, which is well outside the capture range of a rigid ANTs
registration.

Approach
--------
Rather than trust the header, this module identifies the phantom's own
internal landmarks — the small high-signal ("hot") vials and the larger
low-signal ("cold") inserts baked into the physical phantom design, as
segmented in ``template_data/PET/VialsLabelled/*.nii.gz`` — and matches them
against equivalent blobs detected in the input image's own array-index space.

For a given input image:

1. ``mrthreshold`` (default Otsu) creates a whole-body foreground mask.
2. Compact local-contrast blobs (positive residual = hot vial, negative
   residual = cold insert) are located inside that mask.
3. Candidate blobs are matched against the template's known vial positions
   under every one of the 24 physically-realisable 90-degree-multiple
   rotations (the chiral octahedral group: permutations of the three voxel
   axes combined with an even number of sign flips — i.e. every rotation
   reachable by physically turning the phantom, as opposed to a mirror
   image), via optimal (Hungarian) point-to-point assignment.
4. The best-fitting rotation is applied directly to the voxel array
   (an exact transpose + flip — no interpolation) and the image is re-saved
   with a simple diagonal, zero-offset affine matching the template's own
   header convention.

If landmark detection is too sparse or ambiguous to confidently distinguish
between candidate rotations, the best available fit is still applied (it
remains the most probable answer), but ``low_confidence`` is set on the
result so callers can surface a warning and recommend a visual sanity check.
"""

from __future__ import annotations

import itertools
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# The 24 proper (physically-realisable) axis-permutation rotation matrices
# ---------------------------------------------------------------------------


def _generate_proper_rotations() -> list:
    """All 3x3 matrices that permute/flip the coordinate axes with det = +1.

    These are exactly the rotations reachable by physically turning a
    rectangular object onto one of its 24 axis-aligned orientations (the
    rotation group of the cube). Matrices with det = -1 are mirror images
    and are excluded, since no physical rotation can produce them.
    """
    mats = []
    for perm in itertools.permutations(range(3)):
        perm_matrix = np.zeros((3, 3))
        for i, p in enumerate(perm):
            perm_matrix[i, p] = 1
        for signs in itertools.product((1, -1), repeat=3):
            rot = perm_matrix @ np.diag(signs)
            if abs(np.linalg.det(rot) - 1) < 1e-6:
                mats.append(rot.astype(int))
    return mats


PROPER_ROTATIONS: list = _generate_proper_rotations()


# ---------------------------------------------------------------------------
# Template landmark loading
# ---------------------------------------------------------------------------


def load_template_landmarks(template_mask: str, vial_dir: str) -> tuple:
    """Load per-vial centroid positions (mm, relative to body centroid).

    Parameters
    ----------
    template_mask:
        Whole-phantom foreground mask for the template
        (``ImageTemplate_mask.nii.gz``).
    vial_dir:
        Directory of per-vial binary masks (``VialsLabelled/*.nii.gz``); the
        vial name is taken from the filename stem.

    Returns
    -------
    tuple
        ``(points, volumes, voxel_size)`` where ``points`` maps vial name ->
        3-vector position in mm relative to the whole-body mask centroid,
        ``volumes`` maps vial name -> volume in mm^3, and ``voxel_size`` is
        the template's voxel size in mm.
    """
    mask_img = nib.load(str(template_mask))
    voxel_size = np.array(mask_img.header.get_zooms()[:3], dtype=float)
    mask = np.asarray(mask_img.dataobj) > 0
    if not mask.any():
        raise ValueError(f"Template mask is empty: {template_mask}")
    body_centroid = np.argwhere(mask).mean(axis=0)

    voxel_vol = float(np.prod(voxel_size))
    points: dict = {}
    volumes: dict = {}
    for vial_path in sorted(Path(vial_dir).glob("*.nii.gz")):
        name = vial_path.name[: -len(".nii.gz")]
        vial_mask = np.asarray(nib.load(str(vial_path)).dataobj) > 0
        idx = np.argwhere(vial_mask)
        if idx.size == 0:
            continue
        points[name] = (idx.mean(axis=0) - body_centroid) * voxel_size
        volumes[name] = len(idx) * voxel_vol

    if not points:
        raise ValueError(f"No non-empty vial masks found in {vial_dir}")

    return points, volumes, voxel_size


# ---------------------------------------------------------------------------
# Candidate landmark detection in an input image
# ---------------------------------------------------------------------------


def detect_landmark_candidates(
    data: np.ndarray,
    mask: np.ndarray,
    voxel_size: np.ndarray,
    min_vol_mm3: float,
    max_vol_mm3: float,
    contrast_sigma_mm: float = 8.0,
) -> list:
    """Locate compact hot/cold blobs inside *mask* via local intensity contrast.

    A local background estimate is formed by box-filtering the (masked)
    image with a window a few times larger than the expected vial size; the
    residual (``data - local_background``) highlights small hot vials
    (positive) and cold inserts (negative) regardless of the image's overall
    intensity scale, which varies a great deal between reconstruction
    algorithms and sites.

    Parameters
    ----------
    data:
        3D intensity array (raw array-index space, no reslicing).
    mask:
        Boolean whole-body foreground mask, same shape as *data*.
    voxel_size:
        3-vector of voxel dimensions in mm.
    min_vol_mm3, max_vol_mm3:
        Size range (in mm^3) used to keep candidate blobs and discard both
        noise specks and large compartments (e.g. the main background fill).
    contrast_sigma_mm:
        Approximate half-width (mm) of the local-background box filter.

    Returns
    -------
    list
        ``[(position_mm, volume_mm3, kind), ...]`` where ``position_mm`` is
        relative to the whole-body mask centroid and ``kind`` is ``"hot"``
        or ``"cold"``.
    """
    if not mask.any():
        return []

    body_centroid = np.argwhere(mask).mean(axis=0)

    # Crop to the mask's bounding box: these images often carry a large
    # amount of empty air padding, and cropping keeps the filtering fast.
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - 4, 0)
    hi = np.minimum(idx.max(axis=0) + 5, np.array(mask.shape))
    sl = tuple(slice(lo[i], hi[i]) for i in range(3))
    sub_data = data[sl]
    sub_mask = mask[sl]

    size_vox = np.maximum(1, np.round(2 * contrast_sigma_mm / voxel_size)).astype(int)
    maskf = sub_mask.astype(np.float32)
    num = ndimage.uniform_filter(sub_data * maskf, size=tuple(size_vox))
    den = ndimage.uniform_filter(maskf, size=tuple(size_vox))
    den[den < 1e-6] = 1e-6
    smoothed = num / den
    residual = np.zeros_like(sub_data, dtype=np.float32)
    residual[sub_mask] = sub_data[sub_mask] - smoothed[sub_mask]

    rvals = residual[sub_mask]
    if rvals.size == 0:
        return []
    hot_thr = np.percentile(rvals, 97)
    cold_thr = np.percentile(rvals, 3)

    voxel_vol = float(np.prod(voxel_size))
    candidates: list = []

    for kind, comp_mask in (
        ("hot", (residual > hot_thr) & sub_mask),
        ("cold", (residual < cold_thr) & sub_mask),
    ):
        labelled, n_labels = ndimage.label(comp_mask, structure=np.ones((3, 3, 3)))
        if n_labels == 0:
            continue
        counts = np.bincount(labelled.ravel())
        labels = np.arange(1, n_labels + 1)
        vols = counts[1 : n_labels + 1] * voxel_vol
        keep = (vols >= min_vol_mm3) & (vols <= max_vol_mm3)
        keep_labels = labels[keep]
        if keep_labels.size == 0:
            continue
        centroids = ndimage.center_of_mass(comp_mask, labelled, keep_labels)
        for centroid, vol in zip(centroids, vols[keep]):
            position = (np.array(centroid) + lo - body_centroid) * voxel_size
            candidates.append((position, float(vol), kind))

    return candidates


# ---------------------------------------------------------------------------
# Rotation search
# ---------------------------------------------------------------------------


def _match_score(
    rotation: np.ndarray,
    candidates: list,
    template_points: dict,
    tol_mm: float,
):
    """Optimally assign rotated candidates to template points; score the fit."""
    if not candidates:
        return None
    labels = list(template_points.keys())
    target = np.array([template_points[label] for label in labels])
    source = np.array([c[0] for c in candidates])
    rotated = (rotation @ source.T).T
    cost = np.linalg.norm(rotated[:, None, :] - target[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    dists = cost[row, col]
    good = dists < tol_mm
    if not good.any():
        return None
    return float(dists[good].mean()), int(good.sum())


def estimate_orientation(
    data: np.ndarray,
    mask: np.ndarray,
    voxel_size: np.ndarray,
    template_points: dict,
    template_volumes: dict,
    z_bin_mm: float = 3.0,
    z_half_window_mm: float = 4.0,
    match_tol_mm: float = 8.0,
    volume_margin: float = 1.5,
) -> dict:
    """Search for the best-fitting axis-aligned rotation of *data* onto the template.

    The template landmarks typically cluster into a small number of axial
    "planes" (e.g. the small hot vials sit at one end of the phantom, the
    cold inserts at the other). Restricting each fit attempt to candidates
    that share a similar z position keeps unrelated blobs elsewhere in the
    volume from diluting the point-matching, which is what makes the
    Hungarian assignment discriminate cleanly between rotation candidates.

    Parameters
    ----------
    data, mask, voxel_size:
        See :func:`detect_landmark_candidates`.
    template_points, template_volumes:
        See :func:`load_template_landmarks`.
    z_bin_mm:
        Bin width used to group candidates into distinct z-windows to try.
    z_half_window_mm:
        Half-width (mm) of each z-window.
    match_tol_mm:
        Maximum distance (mm) for a candidate/template pairing to count as
        a match.
    volume_margin:
        Multiplicative margin applied to the template's vial volume range
        when filtering candidate blob sizes.

    Returns
    -------
    dict
        ``rotation`` (3x3 int ndarray, one of :data:`PROPER_ROTATIONS`),
        ``mean_error_mm``, ``n_matched``, ``margin_mm`` (gap to the
        runner-up rotation — a rough confidence indicator) and
        ``low_confidence`` (bool).
    """
    min_vol = min(template_volumes.values()) / volume_margin
    max_vol = max(template_volumes.values()) * volume_margin

    candidates = detect_landmark_candidates(
        data, mask, voxel_size, min_vol_mm3=min_vol, max_vol_mm3=max_vol
    )

    if not candidates:
        return {
            "rotation": np.eye(3, dtype=int),
            "mean_error_mm": float("nan"),
            "n_matched": 0,
            "margin_mm": float("nan"),
            "low_confidence": True,
        }

    body_centroid_z = np.argwhere(mask).mean(axis=0)[2]
    z_candidates = sorted({round(c[0][2] / z_bin_mm) * z_bin_mm for c in candidates})
    half_window_vox = max(4, int(round(z_half_window_mm / voxel_size[2])))

    results = []
    for z_rel in z_candidates:
        z_abs = body_centroid_z + z_rel / voxel_size[2]
        lo_vox, hi_vox = z_abs - half_window_vox, z_abs + half_window_vox
        window_candidates = [
            c
            for c in candidates
            if lo_vox <= (body_centroid_z + c[0][2] / voxel_size[2]) <= hi_vox
        ]
        if len(window_candidates) < 3:
            continue
        for rotation in PROPER_ROTATIONS:
            score = _match_score(rotation, window_candidates, template_points, match_tol_mm)
            if score is not None:
                mean_err, n_good = score
                results.append((mean_err, n_good, rotation))

    if not results:
        return {
            "rotation": np.eye(3, dtype=int),
            "mean_error_mm": float("nan"),
            "n_matched": 0,
            "margin_mm": float("nan"),
            "low_confidence": True,
        }

    results.sort(key=lambda t: (-t[1], t[0]))
    best_mean, best_n, best_rotation = results[0]
    runner_up = next(
        (r for r in results[1:] if not np.array_equal(r[2], best_rotation)), None
    )
    margin = (runner_up[0] - best_mean) if runner_up is not None else float("inf")
    low_confidence = best_mean > 6.0 or margin < 1.0

    return {
        "rotation": best_rotation,
        "mean_error_mm": best_mean,
        "n_matched": best_n,
        "margin_mm": margin,
        "low_confidence": low_confidence,
    }


# ---------------------------------------------------------------------------
# Applying a rotation to a voxel array
# ---------------------------------------------------------------------------


def apply_rotation_to_array(data: np.ndarray, rotation: np.ndarray) -> tuple:
    """Apply an axis-permutation/flip rotation directly to a 3D voxel array.

    ``rotation`` must be one of the 24 matrices in :data:`PROPER_ROTATIONS`
    (each row and column has exactly one non-zero entry, +-1). This is
    implemented as an exact ``numpy.transpose`` + ``numpy.flip`` — no
    interpolation is involved, so the output voxels are a lossless
    re-indexing of the input.

    Returns
    -------
    tuple
        ``(rotated_array, axis_permutation, axis_signs)``.
    """
    perm = [int(np.nonzero(rotation[i])[0][0]) for i in range(3)]
    signs = [int(rotation[i, perm[i]]) for i in range(3)]
    out = np.transpose(data, axes=perm)
    flip_axes = tuple(i for i in range(3) if signs[i] == -1)
    if flip_axes:
        out = np.flip(out, axis=flip_axes)
    return out, perm, signs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reorient_pet_image(
    input_image: str,
    output_image: str,
    template_image: Optional[str] = None,
    template_mask: Optional[str] = None,
    vial_template_dir: Optional[str] = None,
    force: bool = True,
) -> dict:
    """Correct gross (90-degree-multiple) orientation errors in a PET image.

    This is the entry point used as Step 0 of
    :func:`phantomkit.pet_processing.register_pet_to_template`. It writes a
    new NIfTI file whose voxel content has been rotated (via exact
    re-indexing, not interpolation) to match the template's orientation
    convention, with a simple diagonal, zero-offset affine.

    Parameters
    ----------
    input_image:
        Path to the input PET NIfTI.
    output_image:
        Path to write the reoriented NIfTI.
    template_image:
        Unused directly (kept for interface symmetry with
        :func:`~phantomkit.pet_processing.register_pet_to_template`); reserved
        for future use (e.g. voxel-size-aware matching).
    template_mask, vial_template_dir:
        Paths to the template's whole-body mask and per-vial label
        directory. Both default to the bundled
        ``template_data/PET/ImageTemplate_mask.nii.gz`` and
        ``template_data/PET/VialsLabelled/``.
    force:
        Pass ``-force`` to ``mrthreshold``.

    Returns
    -------
    dict
        ``output_image``, ``rotation`` (3x3 list of ints), ``axis_permutation``,
        ``axis_signs``, ``mean_error_mm``, ``n_matched``, ``margin_mm``,
        ``low_confidence`` and ``identity`` (True if no rotation was needed).
    """
    input_path = Path(input_image)
    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if template_mask is None or vial_template_dir is None:
        _pkg = Path(__file__).resolve().parent
        _td_pet = _pkg.parent / "template_data" / "PET"
        if template_mask is None:
            template_mask = str(_td_pet / "ImageTemplate_mask.nii.gz")
        if vial_template_dir is None:
            vial_template_dir = str(_td_pet / "VialsLabelled")

    template_points, template_volumes, _ = load_template_landmarks(
        template_mask, vial_template_dir
    )

    img = nib.load(str(input_path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    voxel_size = np.array(img.header.get_zooms()[:3], dtype=float)

    # Whole-body foreground mask, via mrthreshold (default Otsu threshold).
    with tempfile.TemporaryDirectory() as tmp_dir:
        body_mask_path = Path(tmp_dir) / "body_mask.nii.gz"
        cmd = ["mrthreshold", "-quiet", str(input_path), str(body_mask_path)]
        if force:
            cmd.append("-force")
        subprocess.run(cmd, check=True)
        mask = np.asarray(nib.load(str(body_mask_path)).dataobj) > 0

    if not mask.any():
        raise RuntimeError(f"mrthreshold produced an empty foreground mask for {input_path}")

    result = estimate_orientation(
        data, mask, voxel_size, template_points, template_volumes
    )
    rotation = result["rotation"]

    rotated, perm, signs = apply_rotation_to_array(data, rotation)
    new_voxel_size = voxel_size[perm]

    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = new_voxel_size
    out_img = nib.Nifti1Image(rotated.astype(np.float32), affine)
    out_img.header.set_zooms(tuple(new_voxel_size))
    nib.save(out_img, str(output_path))

    if result["low_confidence"]:
        print(
            f"  WARNING: orientation fit for {input_path.name} has low confidence "
            f"(mean_error={result['mean_error_mm']:.1f}mm, "
            f"margin={result['margin_mm']:.1f}mm, n_matched={result['n_matched']}). "
            "Recommend a visual check against the template before trusting "
            "downstream registration."
        )

    return {
        "output_image": str(output_path),
        "rotation": np.asarray(rotation).tolist(),
        "axis_permutation": perm,
        "axis_signs": signs,
        "mean_error_mm": result["mean_error_mm"],
        "n_matched": result["n_matched"],
        "margin_mm": result["margin_mm"],
        "low_confidence": result["low_confidence"],
        "identity": bool(np.array_equal(rotation, np.eye(3, dtype=int))),
    }
