"""Unit tests for phantomkit.pet_orientation."""

from __future__ import annotations

import numpy as np
import pytest

from phantomkit.pet_orientation import (
    PROPER_ROTATIONS,
    apply_rotation_to_array,
    detect_landmark_candidates,
    estimate_orientation,
    load_template_landmarks,
)


# ---------------------------------------------------------------------------
# PROPER_ROTATIONS
# ---------------------------------------------------------------------------


def test_proper_rotations_has_24_unique_matrices() -> None:
    assert len(PROPER_ROTATIONS) == 24
    seen = {tuple(r.ravel()) for r in PROPER_ROTATIONS}
    assert len(seen) == 24


def test_proper_rotations_are_orthogonal_with_det_one() -> None:
    for r in PROPER_ROTATIONS:
        assert np.allclose(r @ r.T, np.eye(3))
        assert np.isclose(np.linalg.det(r), 1.0)


def test_proper_rotations_include_identity_and_exclude_mirrors() -> None:
    identity = np.eye(3, dtype=int)
    assert any(np.array_equal(r, identity) for r in PROPER_ROTATIONS)
    # A pure single-axis flip has det = -1 and must not appear.
    mirror = np.diag([1, 1, -1]).astype(int)
    assert not any(np.array_equal(r, mirror) for r in PROPER_ROTATIONS)


# ---------------------------------------------------------------------------
# apply_rotation_to_array
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_array() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((6, 7, 8)).astype(np.float32)


def test_apply_rotation_identity_is_noop(sample_array) -> None:
    out, perm, signs = apply_rotation_to_array(sample_array, np.eye(3, dtype=int))
    assert perm == [0, 1, 2]
    assert signs == [1, 1, 1]
    assert np.array_equal(out, sample_array)


def test_apply_rotation_flip_only(sample_array) -> None:
    rotation = np.diag([1, -1, -1]).astype(int)
    out, perm, signs = apply_rotation_to_array(sample_array, rotation)
    assert perm == [0, 1, 2]
    assert signs == [1, -1, -1]
    assert np.array_equal(out, sample_array[:, ::-1, ::-1])


def test_apply_rotation_swap_and_flip() -> None:
    # Non-cubic array to make an incorrect axis mapping obvious via shape mismatch.
    rng = np.random.default_rng(1)
    data = rng.random((5, 9, 4)).astype(np.float32)
    rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    out, perm, signs = apply_rotation_to_array(data, rotation)
    assert perm == [1, 0, 2]
    assert signs == [1, 1, -1]
    expected = np.transpose(data, (1, 0, 2))[:, :, ::-1]
    assert out.shape == (9, 5, 4)
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("rotation", PROPER_ROTATIONS)
def test_apply_rotation_roundtrip_via_transpose(rotation, sample_array) -> None:
    """Applying R then R^T (its inverse, since R is orthogonal) restores the array."""
    forward, _, _ = apply_rotation_to_array(sample_array, rotation)
    back, _, _ = apply_rotation_to_array(forward, rotation.T)
    assert np.array_equal(back, sample_array)


# ---------------------------------------------------------------------------
# load_template_landmarks
# ---------------------------------------------------------------------------


def _write_nifti(path, data, voxel_size=(1.0, 1.0, 1.0)) -> None:
    import nibabel as nib

    affine = np.diag(list(voxel_size) + [1.0])
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), str(path))


def test_load_template_landmarks(tmp_path) -> None:
    shape = (20, 20, 20)
    body_mask = np.zeros(shape, dtype=np.uint8)
    body_mask[2:18, 2:18, 2:18] = 1  # centroid = (9.5, 9.5, 9.5)
    mask_path = tmp_path / "mask.nii.gz"
    _write_nifti(mask_path, body_mask)

    vial_dir = tmp_path / "VialsLabelled"
    vial_dir.mkdir()

    vial_a = np.zeros(shape, dtype=np.uint8)
    vial_a[10:12, 10:12, 10:12] = 1  # 8 voxels, centroid (10.5, 10.5, 10.5)
    _write_nifti(vial_dir / "A.nii.gz", vial_a)

    vial_b = np.zeros(shape, dtype=np.uint8)
    vial_b[4:6, 4:6, 4:6] = 1  # 8 voxels, centroid (4.5, 4.5, 4.5)
    _write_nifti(vial_dir / "B.nii.gz", vial_b)

    points, volumes, voxel_size = load_template_landmarks(str(mask_path), str(vial_dir))

    assert set(points) == {"A", "B"}
    assert points["A"] == pytest.approx([1.0, 1.0, 1.0])
    assert points["B"] == pytest.approx([-5.0, -5.0, -5.0])
    assert volumes["A"] == pytest.approx(8.0)
    assert volumes["B"] == pytest.approx(8.0)
    assert voxel_size == pytest.approx([1.0, 1.0, 1.0])


def test_load_template_landmarks_empty_mask_raises(tmp_path) -> None:
    shape = (10, 10, 10)
    mask_path = tmp_path / "mask.nii.gz"
    _write_nifti(mask_path, np.zeros(shape, dtype=np.uint8))
    vial_dir = tmp_path / "VialsLabelled"
    vial_dir.mkdir()

    with pytest.raises(ValueError):
        load_template_landmarks(str(mask_path), str(vial_dir))


# ---------------------------------------------------------------------------
# estimate_orientation — synthetic end-to-end recovery
# ---------------------------------------------------------------------------


# An asymmetric 5-point "vial constellation": four points clustered at one
# end of the phantom (mimicking the small hot vials) and one at the other
# end (mimicking a cold insert), matching the real PET template's layout
# closely enough to exercise the same z-windowing logic.
_TEMPLATE_POINTS = {
    "A": np.array([5.0, 3.0, -8.0]),
    "B": np.array([-4.0, 6.0, -8.0]),
    "C": np.array([2.0, -5.0, -8.0]),
    "D": np.array([-6.0, -2.0, -8.0]),
    "E": np.array([1.0, 7.0, 9.0]),
}
_TEMPLATE_VOLUMES = {"A": 27.0, "B": 27.0, "C": 27.0, "D": 27.0, "E": 27.0}


def _synthetic_volume(points: dict, shape=(40, 40, 40), background=1.0, blob_value=200.0):
    """Build a volume with a small cube blob at each landmark position."""
    data = np.full(shape, background, dtype=np.float32)
    center = (np.array(shape) - 1) / 2.0
    for pos in points.values():
        voxel = np.round(center + pos).astype(int)
        sl = tuple(
            slice(max(0, c - 1), min(s, c + 2)) for c, s in zip(voxel, shape)
        )
        data[sl] = blob_value
    mask = np.ones(shape, dtype=bool)
    return data, mask


@pytest.mark.parametrize(
    "true_rotation",
    [
        np.eye(3, dtype=int),
        np.diag([1, -1, -1]).astype(int),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    ],
)
def test_estimate_orientation_recovers_known_rotation(true_rotation) -> None:
    canonical_data, mask = _synthetic_volume(_TEMPLATE_POINTS)

    # Simulate a mis-oriented acquisition: apply true_rotation to a
    # template-matching volume, exactly as reorient_pet_image would need to
    # undo.
    misoriented_data, _, _ = apply_rotation_to_array(canonical_data, true_rotation)

    result = estimate_orientation(
        misoriented_data,
        mask,
        voxel_size=np.array([1.0, 1.0, 1.0]),
        template_points=_TEMPLATE_POINTS,
        template_volumes=_TEMPLATE_VOLUMES,
    )

    # The lone point at the opposite end of the phantom (z=+9) falls in its
    # own z-window with too few neighbours to be scored on its own; the
    # 4-point cluster is enough to recover the rotation unambiguously.
    assert result["n_matched"] >= 4
    assert result["mean_error_mm"] < 1.0

    corrected, _, _ = apply_rotation_to_array(misoriented_data, result["rotation"])
    assert np.array_equal(corrected, canonical_data)


def test_estimate_orientation_no_candidates_falls_back_to_identity() -> None:
    shape = (10, 10, 10)
    data = np.ones(shape, dtype=np.float32)  # no landmark contrast at all
    mask = np.ones(shape, dtype=bool)

    result = estimate_orientation(
        data,
        mask,
        voxel_size=np.array([1.0, 1.0, 1.0]),
        template_points=_TEMPLATE_POINTS,
        template_volumes=_TEMPLATE_VOLUMES,
    )

    assert np.array_equal(result["rotation"], np.eye(3, dtype=int))
    assert result["low_confidence"] is True


# ---------------------------------------------------------------------------
# detect_landmark_candidates
# ---------------------------------------------------------------------------


def test_detect_landmark_candidates_finds_all_blobs() -> None:
    data, mask = _synthetic_volume(_TEMPLATE_POINTS)
    candidates = detect_landmark_candidates(
        data, mask, voxel_size=np.array([1.0, 1.0, 1.0]),
        min_vol_mm3=5.0, max_vol_mm3=200.0,
    )
    assert len(candidates) >= len(_TEMPLATE_POINTS)
    kinds = {c[2] for c in candidates}
    assert kinds <= {"hot", "cold"}
