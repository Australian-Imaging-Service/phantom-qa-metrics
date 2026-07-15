#!/usr/bin/env python3
"""
DWI processing and tensor metrics pipeline (Pydra 0.23+)

Usage:
    python Functions/pydra_dwi_processing.py --config pydra_dwi_processing.yaml
    python Functions/pydra_dwi_processing.py --scans-dir /path/to/scans --output-dir /path/to/out [options]

Options:
    --nocleanup     Keep the tmp/ intermediate file directory after processing
                    (default: tmp/ is removed once final outputs are copied)

Output structure:
    <output_dir>/
        <DWI_series_name>/
            ADC.nii.gz
            FA.nii.gz
            DWI_preproc_biascorr.mif.gz   (or DWI_denoise_gibbs_preproc_biascorr.mif.gz)
            T1_in_DWI_space.nii.gz
            tmp/                           (intermediate files — removed unless --nocleanup)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from pydra.compose import python, workflow
from pydra.engine import Submitter
import yaml
from fileformats.medimage import NiftiGz
from fileformats.vendor.mrtrix3.medimage import ImageIn, ImageOut, ImageFormatGz


# =============================================================================
# Utility helpers
# =============================================================================


def run_cmd(cmd: list, cwd: str = None):
    """Run a shell command, raising on failure."""
    print(f"  >> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def sanitise_name(name: str) -> str:
    """Convert a folder name to a valid Python identifier for Pydra task names."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def strip_series_number(name: str) -> str:
    """Strip leading series number and SEQ prefix, e.g. '87-SEQ5_ep2d_diff_...' -> 'ep2d_diff_...'"""
    name = re.sub(r"^\d+-", "", name)
    return re.sub(r"^SEQ\d+_", "", name, flags=re.IGNORECASE)


def get_series_number(name: str) -> int:
    """Extract leading series number from folder name, e.g. '87-ep2d...' -> 87."""
    m = re.match(r"^(\d+)-", name)
    return int(m.group(1)) if m else 0


def common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def get_nvols(path: str) -> int:
    """Return number of volumes using mrinfo."""
    result = subprocess.run(
        ["mrinfo", path, "-ndim"], capture_output=True, text=True, check=True
    )
    ndim = int(result.stdout.strip())
    if ndim == 3:
        return 1
    result2 = subprocess.run(
        ["mrinfo", path, "-size"], capture_output=True, text=True, check=True
    )
    return int(result2.stdout.strip().split()[-1])


def read_bvals(bval_path: str) -> list:
    """Read bval file and return list of float values. Returns [] if missing/empty."""
    try:
        with open(bval_path) as f:
            content = f.read().strip()
        if not content:
            return []
        return [float(v) for v in content.split()]
    except Exception:
        return []


def has_nonzero_bvals(bval_path: str) -> bool:
    """Return True if bval file exists, is non-empty, and contains non-zero values."""
    vals = read_bvals(bval_path)
    return bool(vals) and any(v > 10 for v in vals)  # threshold >10 to ignore rounding


def all_zero_bvals(bval_path: str) -> bool:
    """Return True if bval file exists and all values are effectively zero."""
    vals = read_bvals(bval_path)
    return bool(vals) and all(v <= 10 for v in vals)


def get_readout_time(json_path: str, fallback: float = 0.0342002) -> float:
    """Extract TotalReadoutTime from dcm2niix JSON sidecar."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        val = data.get("TotalReadoutTime")
        if val is not None:
            return float(val)
    except Exception:
        pass
    print(f"  Warning: TotalReadoutTime not found in {json_path} -- using {fallback}")
    return fallback


def get_pe_from_json(json_path: str) -> tuple:
    """
    (#9) Fallback: read PhaseEncodingDirection from JSON sidecar and map to
    MRtrix3-compatible direction string.
    Returns (pe_dir, rpe_dir) or raises ValueError if not found/unrecognised.
    """
    mapping = {
        "j-": ("AP", "PA"),
        "j": ("PA", "AP"),
        "i": ("LR", "RL"),
        "i-": ("RL", "LR"),
        "k": ("SI", "IS"),
        "k-": ("IS", "SI"),
    }
    try:
        with open(json_path) as f:
            data = json.load(f)
        ped = data.get("PhaseEncodingDirection", "").strip()
        if ped in mapping:
            return mapping[ped]
    except Exception:
        pass
    raise ValueError(f"PhaseEncodingDirection not found or unrecognised in {json_path}")


def detect_pe_direction(folder_name: str) -> tuple:
    """
    Return (pe_dir, rpe_dir) from a folder name.
    Raises ValueError if not found — caller should fall back to get_pe_from_json (#9).
    """
    pairs = [
        (r"_AP(_|$)", "AP", "PA"),
        (r"_A_P(_|$)", "AP", "PA"),
        (r"_PA(_|$)", "PA", "AP"),
        (r"_P_A(_|$)", "PA", "AP"),
        (r"_LR(_|$)", "LR", "RL"),
        (r"_L_R(_|$)", "LR", "RL"),
        (r"_RL(_|$)", "RL", "LR"),
        (r"_R_L(_|$)", "RL", "LR"),
        (r"_SI(_|$)", "SI", "IS"),
        (r"_S_I(_|$)", "SI", "IS"),
        (r"_IS(_|$)", "IS", "SI"),
        (r"_I_S(_|$)", "IS", "SI"),
    ]
    for pattern, pe, rpe in pairs:
        if re.search(pattern, folder_name, re.IGNORECASE):
            return pe, rpe
    raise ValueError(
        f"Could not determine phase encoding direction from folder name: {folder_name}"
    )


# =============================================================================
# Directory scanning constants
# =============================================================================

FWD_DIRS = ["AP", "LR", "SI", "A_P", "L_R", "S_I"]
RPE_DIRS = ["PA", "RL", "IS", "P_A", "R_L", "I_S"]
ALL_PE_DIRS = FWD_DIRS + RPE_DIRS


def has_terminal_pe(name: str, directions: list) -> bool:
    """
    Returns True if a PE direction tag appears as the final token in the folder
    name (nothing follows, or only a single short alphanumeric token with no
    underscores). This identifies dedicated b0 PE images vs full DWI series.
    """
    for d_ in directions:
        for pat in [rf"_{d_}$", rf"_{d_}_[A-Za-z0-9]{{1,10}}$"]:
            if re.search(pat, name, re.IGNORECASE):
                return True
        # Underscore-separated variant (e.g. A_P, P_A)
        if len(d_) == 3 and d_[1] == "_":
            for pat in [
                rf"_{d_[0]}_{d_[2]}$",
                rf"_{d_[0]}_{d_[2]}_[A-Za-z0-9]{{1,10}}$",
            ]:
                if re.search(pat, name, re.IGNORECASE):
                    return True
    return False


def get_acq_stem_and_suffix(folder_name: str) -> tuple:
    """
    Split acquisition name into (prefix, suffix) around the PE direction tag.
    Both must match exactly for two series to be considered valid AP/PA partners
    (#14 — prevents pairing of MONO/BIPOLAR variants or different parameters).

    e.g. '12-ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R_AP_ESpt54_BW2264_BIPOLAR'
      -> prefix: 'ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R'
         suffix: 'ESpt54_BW2264_BIPOLAR'

    Returns (full_name_no_series, "", "") if no PE direction tag found.
    """
    name = strip_series_number(folder_name)
    # Try longer variants first (A_P before AP) to avoid partial matches
    all_dirs_ordered = [
        "A_P",
        "P_A",
        "L_R",
        "R_L",
        "S_I",
        "I_S",
        "AP",
        "PA",
        "LR",
        "RL",
        "SI",
        "IS",
    ]
    for d in all_dirs_ordered:
        m = re.search(rf"_({re.escape(d)})(_|$)", name, re.IGNORECASE)
        if m:
            prefix = name[: m.start()]
            # suffix is everything after the direction tag (and the following _)
            suffix_start = m.end()
            suffix = name[suffix_start:] if suffix_start < len(name) else ""
            return prefix, suffix
    return name, ""


# =============================================================================
# DICOM conversion helper (used in planning stage)
# =============================================================================


def _detect_series_format(series_dir: str) -> str:
    """Return 'nifti', 'mif', or 'dicom' based on file contents of series_dir."""
    p = Path(series_dir)
    if any(p.glob("*.nii.gz")) or any(p.glob("*.nii")):
        return "nifti"
    if any(p.glob("*.mif.gz")) or any(p.glob("*.mif")):
        return "mif"
    return "dicom"


def convert_series_to_nii(series_dir: str, out_dir: str) -> dict:
    """
    Stage a series directory to NIfTI, handling DICOM, NIfTI, and MIF inputs.
    Returns dict with paths to: nii, json, bvec, bval (empty string if absent).
    """
    fmt = _detect_series_format(series_dir)
    os.makedirs(out_dir, exist_ok=True)

    if fmt == "nifti":
        niis = sorted(Path(series_dir).glob("*.nii.gz"))
        if not niis:
            niis = [p for p in sorted(Path(series_dir).glob("*.nii"))
                    if not p.name.endswith(".nii.gz")]
        if not niis:
            raise FileNotFoundError(f"No NIfTI files in {series_dir}")
        nii_src = niis[0]
        base_src = str(nii_src)
        for _ext in (".nii.gz", ".nii"):
            if base_src.endswith(_ext):
                base_src = base_src[: -len(_ext)]
                break
        if str(nii_src).endswith(".nii.gz"):
            dst_nii = str(Path(out_dir) / nii_src.name)
            shutil.copy2(nii_src, dst_nii)
        else:
            dst_nii = str(Path(out_dir) / (nii_src.stem + ".nii.gz"))
            subprocess.run(
                ["mrconvert", str(nii_src), dst_nii], check=True, capture_output=True
            )
        base_dst = dst_nii.replace(".nii.gz", "")

        def _copy_sidecar(ext: str) -> str:
            src = base_src + ext
            if Path(src).exists():
                dst = base_dst + ext
                shutil.copy2(src, dst)
                return dst
            return ""

        return {
            "nii": dst_nii,
            "json": _copy_sidecar(".json"),
            "bvec": _copy_sidecar(".bvec"),
            "bval": _copy_sidecar(".bval"),
        }

    if fmt == "mif":
        mifs = sorted(Path(series_dir).glob("*.mif.gz")) + sorted(
            Path(series_dir).glob("*.mif")
        )
        if not mifs:
            raise FileNotFoundError(f"No MIF files in {series_dir}")
        mif_src = mifs[0]
        stem = mif_src.name
        for _ext in (".mif.gz", ".mif"):
            if stem.endswith(_ext):
                stem = stem[: -len(_ext)]
                break
        nii = str(Path(out_dir) / f"{stem}.nii.gz")
        bvec = str(Path(out_dir) / f"{stem}.bvec")
        bval = str(Path(out_dir) / f"{stem}.bval")
        json_out = str(Path(out_dir) / f"{stem}.json")
        subprocess.run(
            [
                "mrconvert", str(mif_src), nii,
                "-export_grad_fsl", bvec, bval,
                "-json_export", json_out,
            ],
            check=True,
            capture_output=True,
        )
        return {
            "nii": nii,
            "json": json_out if Path(json_out).exists() else "",
            "bvec": bvec if Path(bvec).exists() else "",
            "bval": bval if Path(bval).exists() else "",
        }

    # DICOM fallback
    subprocess.run(
        ["dcm2niix", "-o", out_dir, "-f", "%p", "-z", "y", series_dir],
        check=True,
        capture_output=True,
    )
    niis = sorted(Path(out_dir).glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"No NIfTI produced from DICOM: {series_dir}")
    nii = str(niis[0])
    base = nii.replace(".nii.gz", "")
    return {
        "nii": nii,
        "json": base + ".json" if Path(base + ".json").exists() else "",
        "bvec": base + ".bvec" if Path(base + ".bvec").exists() else "",
        "bval": base + ".bval" if Path(base + ".bval").exists() else "",
    }


# =============================================================================
# Directory scanning and candidate classification
# =============================================================================


def scan_directory(scans_dir: str) -> dict:
    """
    First pass: classify subdirectories into mprage_dirs, candidate_dwi (all series
    containing _diff_ or _DWI_), and ignored. PE classification is PROVISIONAL
    at this stage — final assignment happens after series staging in
    classify_candidates().
    """
    scans_path = Path(scans_dir)
    mprage_dirs = []
    candidate_dwi = []  # all _diff_ / _DWI_ series, including PE images
    ignored = []

    for d in sorted(scans_path.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # MPRAGE structural reference
        if re.search(r"MPRAGE", name, re.IGNORECASE):
            mprage_dirs.append(str(d))
            continue

        # Skip scanner-derived maps
        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        # Only consider DWI-like acquisitions
        if not re.search(r"(_diff_|_DWI_)", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        candidate_dwi.append(str(d))

    if not mprage_dirs:
        raise ValueError(f"Could not identify any MPRAGE directory in {scans_dir}")
    if not candidate_dwi:
        raise ValueError(f"Could not identify any DWI directories in {scans_dir}")

    return {
        "mprage_dirs": mprage_dirs,
        "candidate_dwi": candidate_dwi,
        "ignored": ignored,
    }


def convert_all_candidates(candidate_dwi: list, output_dir: str) -> dict:
    """
    (#18 ordering step 2) Stage all candidate DWI series upfront.
    Handles DICOM, NIfTI, and MIF inputs.
    Returns a dict: {series_dir: {nii, json, bvec, bval}}
    """
    conversions = {}
    for series_dir in candidate_dwi:
        name = Path(series_dir).name
        conv_dir = str(Path(output_dir) / name / "tmp" / "dwi_nii")
        fmt = _detect_series_format(series_dir)
        print(f"  Staging ({fmt}): {name}")
        try:
            result = convert_series_to_nii(series_dir, conv_dir)
            conversions[series_dir] = result
        except Exception as e:
            print(f"  WARNING: Series staging failed for {name}: {e}")
            conversions[series_dir] = None
    return conversions


def classify_candidates(candidate_dwi: list, conversions: dict) -> dict:
    """
    (#18) Final classification of candidate DWI series into:
      - dwi_dirs:     full DWI series eligible for tensor processing (non-zero bvals)
      - fwd_pe_dirs:  forward PE correction images (all b-values zero)
      - rpe_dirs:     reverse PE correction images (all b-values zero)
      - pending_fwd:  series with non-zero bvals and a fwd PE direction tag anywhere
                      in name — reclassified to dwi_dirs or fwd_pe_dirs after pairing
      - pending_rpe:  same but reverse PE direction tag
      - skipped:      list of (path, reason) tuples

    Classification rules (applied in order after DICOM conversion):
      1. No bvec/bval files → skip (#16)
      2. Empty bval file → skip (#16)
      3. All b-values zero → PE correction image candidate, classified by PE
         direction tag found ANYWHERE in folder name (direction tag position
         does not affect classification — only bval content matters)
      4. Has non-zero b-values AND a PE direction tag → pending (reclassified
         after rpe_all pairing check)
      5. Has non-zero b-values, no PE direction tag → dwi_dirs
    """
    dwi_dirs = []
    fwd_pe_dirs = []
    rpe_dirs = []
    skipped = []
    pending_fwd = []  # non-zero bvals + fwd PE tag — needs pairing check
    pending_rpe = []  # non-zero bvals + rpe PE tag — needs pairing check

    for dicom_dir in candidate_dwi:
        name = Path(dicom_dir).name
        conv = conversions.get(dicom_dir)

        if conv is None:
            skipped.append((dicom_dir, "DICOM conversion failed"))
            continue

        bvec = conv["bvec"]
        bval = conv["bval"]

        # #16 — check bvec/bval exist and are non-empty
        if not bvec or not bval or not Path(bvec).exists() or not Path(bval).exists():
            skipped.append(
                (dicom_dir, "no bvec/bval files — not a raw DWI acquisition")
            )
            continue
        if not read_bvals(bval):
            skipped.append((dicom_dir, "empty bval file — not a raw DWI acquisition"))
            continue

        is_b0_only = all_zero_bvals(bval)

        # Detect PE direction tag anywhere in the folder name (not just terminal)
        has_fwd_tag = any(
            re.search(rf"_{re.escape(d)}(_|$)", name, re.IGNORECASE) for d in FWD_DIRS
        )
        has_rpe_tag = any(
            re.search(rf"_{re.escape(d)}(_|$)", name, re.IGNORECASE) for d in RPE_DIRS
        )

        if is_b0_only:
            # All b-values zero: classify as PE correction image by direction tag
            if has_fwd_tag:
                fwd_pe_dirs.append(dicom_dir)
            elif has_rpe_tag:
                rpe_dirs.append(dicom_dir)
            else:
                skipped.append(
                    (
                        dicom_dir,
                        "all b-values zero and no phase encoding direction in folder name — "
                        "cannot use for correction",
                    )
                )
        else:
            # Has non-zero b-values: pending if PE tag present, dwi_dirs otherwise
            if has_fwd_tag:
                pending_fwd.append(dicom_dir)
            elif has_rpe_tag:
                pending_rpe.append(dicom_dir)
            else:
                dwi_dirs.append(dicom_dir)

    return {
        "dwi_dirs": dwi_dirs,
        "fwd_pe_dirs": fwd_pe_dirs,
        "rpe_dirs": rpe_dirs,
        "pending_fwd": pending_fwd,
        "pending_rpe": pending_rpe,
        "skipped": skipped,
    }


# =============================================================================
# AP/PA pairing
# =============================================================================


def match_ap_pa_pairs(
    dwi_dirs: list, pending_fwd: list, pending_rpe: list, conversions: dict
) -> tuple:
    """
    (#14, #18) Detect AP/PA (rpe_all/rpe_split) pairs across dwi_dirs,
    pending_fwd, and pending_rpe.

    Pairing requires:
      - Same acquisition prefix (everything before the PE direction tag)
      - Same acquisition suffix (everything after the PE direction tag) — #14
      - Opposing PE directions

    Pending series with non-zero bvals that find a partner are moved to dwi_dirs.
    Pending series that find no partner are moved to fwd_pe_dirs / rpe_dirs.

    Returns:
      (dwi_dirs, fwd_pe_dirs, rpe_dirs, rpe_all_map, tie_warnings)
      rpe_all_map: {fwd_path: rpe_path}
      tie_warnings: list of (pe_image_path, [matched_dwi_paths]) for #20
    """
    fwd_set = {"AP", "LR", "SI"}
    rpe_set = {"PA", "RL", "IS"}

    # Build stem map over all candidates (dwi + pending)
    all_candidates = dwi_dirs + pending_fwd + pending_rpe
    stem_map = {}  # (prefix, suffix) -> list of (path, pe_dir)
    for d in all_candidates:
        name = Path(d).name
        try:
            pe, _ = detect_pe_direction(name)
        except ValueError:
            pe = None
        prefix, suffix = get_acq_stem_and_suffix(name)
        key = (prefix, suffix)
        stem_map.setdefault(key, []).append((d, pe))

    rpe_all_map = {}
    tie_warnings = []
    paired_paths = set()

    for (prefix, suffix), entries in stem_map.items():
        fwds = [(p, pe) for p, pe in entries if pe in fwd_set]
        rpes = [(p, pe) for p, pe in entries if pe in rpe_set]

        if not (fwds and rpes):
            continue

        for fwd_path, _ in fwds:
            fwd_num = get_series_number(Path(fwd_path).name)
            # Find closest RPE by series number
            best_rpe_path = min(
                rpes, key=lambda x: abs(get_series_number(Path(x[0]).name) - fwd_num)
            )[0]
            rpe_all_map[fwd_path] = best_rpe_path
            paired_paths.add(fwd_path)
            paired_paths.add(best_rpe_path)

            # #20 — check for tie (multiple equal-prefix matches)
            if len(fwds) > 1 or len(rpes) > 1:
                tie_warnings.append((fwd_path, [p for p, _ in rpes]))

    # Move paired pending series into dwi_dirs; unpaired pending into PE dirs
    final_dwi = list(dwi_dirs)
    final_fwd = list(conversions.get("fwd_pe_dirs", []))
    final_rpe = list(conversions.get("rpe_dirs", []))

    for p in pending_fwd:
        if p in paired_paths:
            final_dwi.append(p)
        else:
            final_fwd.append(p)

    for p in pending_rpe:
        if p in paired_paths:
            # PA partner of a paired series — recorded in rpe_all_map, not in dwi_dirs
            pass
        else:
            final_rpe.append(p)

    return sorted(final_dwi), final_fwd, final_rpe, rpe_all_map, tie_warnings


# =============================================================================
# T1 and PE assignment helpers
# =============================================================================


def assign_mprage(dwi_name: str, mprage_dirs: list) -> str:
    """
    Assign the nearest preceding MPRAGE by series number.
    Falls back to the first MPRAGE if none precedes the DWI.
    """
    dwi_num = get_series_number(dwi_name)
    best = None
    best_num = -1
    for mprage in mprage_dirs:
        mprage_num = get_series_number(Path(mprage).name)
        if mprage_num < dwi_num and mprage_num > best_num:
            best_num = mprage_num
            best = mprage
    return best if best is not None else mprage_dirs[0]


def build_pe_assignment_map(
    dwi_dirs: list, fwd_pe_dirs: list, rpe_dirs: list, rpe_all_map: dict
) -> dict:
    """
    Build a map of {dwi_dir: {"rpe": rpe_dir_or_None, "fwd": fwd_dir_or_None}}
    for all DWI series not already in rpe_all_map.

    A single b0 PE correction image is assigned to ALL DWI series sharing the
    longest common prefix with it — not just the nearest one. This correctly
    handles the case where one b0 image is intended to correct an entire block
    of acquisitions that vary only in a trailing parameter (e.g. bandwidth sweep).

    If multiple PE images share the same best prefix length against a DWI block,
    the one with the nearest series number to the block is chosen, with a warning.

    Priority: rpe_all (already in rpe_all_map) > rpe_pair > rpe_none.
    """
    unpaired = [d for d in dwi_dirs if d not in rpe_all_map]
    assignment = {d: {"rpe": None, "fwd": None, "tie_warning": False} for d in unpaired}

    def assign_pe_images(pe_dirs, key):
        """
        For each PE image, find all DWI series sharing the longest common prefix.
        Assign this PE image to that entire group.
        """
        # Build prefix lengths: pe_dir -> {dwi_dir: prefix_len}
        pe_scores = {}
        for pe_dir in pe_dirs:
            pe_stem = strip_series_number(Path(pe_dir).name)
            scores = {}
            for dwi_dir in unpaired:
                dwi_stem = strip_series_number(Path(dwi_dir).name)
                scores[dwi_dir] = common_prefix_len(dwi_stem, pe_stem)
            pe_scores[pe_dir] = scores

        # For each PE image, identify the group of DWI series with max prefix length
        # Only assign if prefix length is meaningful (>10 chars)
        pe_groups = {}  # pe_dir -> set of dwi_dirs it should serve
        for pe_dir, scores in pe_scores.items():
            max_len = max(scores.values()) if scores else 0
            if max_len <= 0:
                continue  # no common prefix at all — skip
            group = {d for d, l in scores.items() if l == max_len}
            pe_groups[pe_dir] = group

        # For each DWI series, find which PE image has the best (longest) prefix
        # If multiple PE images tie for a DWI series, pick nearest by series number
        dwi_to_best_pe = {}
        for dwi_dir in unpaired:
            candidates = []
            best_len = 0
            for pe_dir, group in pe_groups.items():
                if dwi_dir in group:
                    pe_stem = strip_series_number(Path(pe_dir).name)
                    dwi_stem = strip_series_number(Path(dwi_dir).name)
                    length = common_prefix_len(dwi_stem, pe_stem)
                    if length > best_len:
                        best_len = length
                        candidates = [pe_dir]
                    elif length == best_len:
                        candidates.append(pe_dir)
            if not candidates:
                continue
            tie = len(candidates) > 1
            dwi_num = get_series_number(Path(dwi_dir).name)
            best_pe = min(
                candidates, key=lambda p: abs(get_series_number(Path(p).name) - dwi_num)
            )
            dwi_to_best_pe[dwi_dir] = (best_pe, tie)

        # Now assign: use the winning PE image for each DWI, extend to whole group
        # Group DWI series that share the same winning PE image
        pe_to_dwi_group = {}
        for dwi_dir, (pe_dir, tie) in dwi_to_best_pe.items():
            pe_to_dwi_group.setdefault(pe_dir, []).append((dwi_dir, tie))

        for pe_dir, dwi_list in pe_to_dwi_group.items():
            any_tie = any(tie for _, tie in dwi_list)
            for dwi_dir, _ in dwi_list:
                if assignment[dwi_dir][key] is None:
                    assignment[dwi_dir][key] = pe_dir
                    assignment[dwi_dir]["tie_warning"] = any_tie

    assign_pe_images(rpe_dirs, "rpe")
    assign_pe_images(fwd_pe_dirs, "fwd")

    return assignment


# =============================================================================
# PE table header-based mode inference
# =============================================================================

# Mapping from (i, j, k) unit vector (rounded) to direction string.
# MRtrix convention: j- = AP, j = PA, i = LR, i- = RL, k = SI, k- = IS
_PETABLE_VEC_TO_DIR = {
    (0, 1, 0): "PA",
    (0, -1, 0): "AP",
    (1, 0, 0): "LR",
    (-1, 0, 0): "RL",
    (0, 0, 1): "SI",
    (0, 0, -1): "IS",
}


def _petable_vec_to_dir(vec: tuple) -> str:
    """
    Map a (i, j, k) unit vector from mrinfo -petable to a direction string.
    Returns 'UNKNOWN' if the vector doesn't match any known direction.
    """
    rounded = tuple(round(float(v)) for v in vec)
    return _PETABLE_VEC_TO_DIR.get(rounded, f"UNKNOWN{rounded}")


def get_petable_mode(
    nii: str, bvec: str, bval: str, json_path: str, tmp_dir: str, has_rpe: bool
) -> tuple:
    """
    Infer the preproc mode for a DWI series by reading the phase encoding table
    embedded in the MIF header produced by mrconvert.

    Steps:
      1. mrconvert NIfTI+bvec+bval+JSON → temporary MIF
      2. mrinfo -petable on the MIF
      3. Parse rows to extract PE directions per volume
      4. Apply mode inference rules:
           - All same direction, no RPE available  → rpe_none
           - All same direction, RPE available     → rpe_pair
           - Mixed directions, equal counts        → rpe_all
           - Mixed directions, unequal counts      → rpe_split

    Returns:
      (mode, directions_found, petable_rows)
      mode:             one of 'rpe_none', 'rpe_pair', 'rpe_all', 'rpe_split'
      directions_found: list of direction strings per volume (e.g. ['AP','AP','PA'])
      petable_rows:     raw petable as list of lists (for diagnostics)

    Raises RuntimeError if mrconvert or mrinfo fails.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_mif = str(Path(tmp_dir) / "petable_check.mif")

    # Build mrconvert command — include JSON sidecar if available (embeds PE info)
    cmd = ["mrconvert", nii, tmp_mif, "-fslgrad", bvec, bval, "-force", "-quiet"]
    if json_path and Path(json_path).exists():
        cmd += ["-json_import", json_path]

    try:
        subprocess.run(
            [str(c) for c in cmd], check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mrconvert failed during petable check: {e.stderr.strip()}")

    # Read petable — columns: i  j  k  readout_time  (one row per volume)
    try:
        result = subprocess.run(
            ["mrinfo", tmp_mif, "-petable"], capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mrinfo -petable failed: {e.stderr.strip()}")
    finally:
        # Always clean up the temporary MIF
        try:
            Path(tmp_mif).unlink(missing_ok=True)
        except Exception:
            pass

    raw = result.stdout.strip()
    if not raw:
        raise RuntimeError(
            "mrinfo -petable returned no output — PE information "
            "may not be embedded in this image."
        )

    petable_rows = []
    directions = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        vec = (parts[0], parts[1], parts[2])
        petable_rows.append(parts)
        directions.append(_petable_vec_to_dir(vec))

    if not directions:
        raise RuntimeError("Could not parse any rows from mrinfo -petable output.")

    unique_dirs = set(directions)

    if len(unique_dirs) == 1:
        # All volumes have the same PE direction
        mode = "rpe_pair" if has_rpe else "rpe_none"
    else:
        # Mixed PE directions in a single series
        counts = {d: directions.count(d) for d in unique_dirs}
        if len(set(counts.values())) == 1:
            mode = "rpe_all"  # equal counts
        else:
            mode = "rpe_split"  # unequal counts

    return mode, directions, petable_rows


# =============================================================================
# Plan construction
# =============================================================================


def resolve_pe_direction(dwi_name: str, dwi_json: str) -> tuple:
    """
    (#9) Try to determine PE direction from folder name first.
    Fall back to JSON sidecar if not found. Returns (pe_dir, rpe_dir, source)
    where source is 'folder_name' or 'json_sidecar'.
    Raises ValueError if neither source yields a direction.
    """
    try:
        pe_dir, rpe_dir = detect_pe_direction(dwi_name)
        return pe_dir, rpe_dir, "folder_name"
    except ValueError:
        pass

    if dwi_json and Path(dwi_json).exists():
        try:
            pe_dir, rpe_dir = get_pe_from_json(dwi_json)
            return pe_dir, rpe_dir, "json_sidecar"
        except ValueError:
            pass

    raise ValueError(
        f"Could not determine phase encoding direction for {dwi_name} "
        f"from folder name or JSON sidecar."
    )


def plan_workflow(
    dwi_dir: str,
    mprage_dirs: list,
    fwd_pe_dirs: list,
    rpe_dirs: list,
    rpe_all_map: dict,
    pe_assignment_map: dict,
    conversions: dict,
    cfg: dict,
) -> dict:
    """
    Build a complete plan dict for a single DWI series.
    All DICOM conversion has already been done — we use results from conversions.
    PE correction image assignments come from pe_assignment_map (pre-built so that
    one b0 image can be shared across a whole block of DWI series).
    """
    dwi_name = Path(dwi_dir).name
    conv = conversions[dwi_dir]
    dwi_nii = conv["nii"]
    dwi_json = conv["json"]
    dwi_bvec = conv["bvec"]
    dwi_bval = conv["bval"]
    dwi_nvols = get_nvols(dwi_nii)

    out_dir = str(Path(cfg["output_dir"]) / dwi_name)
    tmp_dir = str(Path(out_dir) / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # MPRAGE assignment
    t1_dir = assign_mprage(dwi_name, mprage_dirs)

    # PE direction (#9)
    warnings = []
    try:
        pe_dir, rpe_dir, pe_source = resolve_pe_direction(dwi_name, dwi_json)
    except ValueError as e:
        raise ValueError(str(e))

    if pe_source == "json_sidecar":
        warnings.append(
            f"Phase encoding direction ({pe_dir}) inferred from JSON sidecar "
            f"— not found in folder name. Please verify."
        )

    # RPE assignment — priority: rpe_all > rpe_pair > rpe_none
    rpe_all_partner = rpe_all_map.get(dwi_dir)

    rpe_nii = None
    rpe_nvols = 0
    fwd_pe_nii = None
    fwd_pe_dir_used = None
    rpe_dir_path = None

    if rpe_all_partner:
        # rpe_all: full-volume PA partner — most preferred
        rpe_dir_path = rpe_all_partner
        rpe_conv = conversions.get(rpe_all_partner)
        if rpe_conv:
            rpe_nii = rpe_conv["nii"]
            rpe_nvols = get_nvols(rpe_nii)

        if rpe_nvols == dwi_nvols:
            warnings.append(
                f"{dwi_name} and {Path(rpe_all_partner).name} have equal volumes "
                f"({dwi_nvols}). Assuming rpe_all (full repeats). If directions are "
                f"split across PE directions, set mode: rpe_split in config YAML."
            )
        fwd_pe_dir_used = None  # b0 PE images not needed for rpe_all

    else:
        # Use pe_assignment_map for b0 PE correction (rpe_pair)
        assignment = pe_assignment_map.get(dwi_dir, {})
        rpe_dir_path = assignment.get("rpe")
        fwd_pe_dir_used = assignment.get("fwd")
        tie_warning = assignment.get("tie_warning", False)

        # #20 — tie warning
        if tie_warning:
            warnings.append(
                f"PE correction image matched equally to multiple DWI series. "
                f"Assigned by nearest series number. If incorrect, consider adding "
                f"a distinguishing suffix (e.g. _repeat) to the second acquisition "
                f"block and its correction pair."
            )

        # Ensure RPE conversion is available
        if rpe_dir_path:
            rpe_conv = conversions.get(rpe_dir_path)
            if rpe_conv is None:
                rpe_conv_dir = str(Path(tmp_dir) / "rpe_nii")
                rpe_conv = convert_series_to_nii(rpe_dir_path, rpe_conv_dir)
                conversions[rpe_dir_path] = rpe_conv
            rpe_nii = rpe_conv["nii"]
            rpe_nvols = get_nvols(rpe_nii)

        # Ensure FWD PE conversion is available
        if fwd_pe_dir_used:
            fwd_conv = conversions.get(fwd_pe_dir_used)
            if fwd_conv is None:
                fwd_conv_dir = str(Path(tmp_dir) / "fwd_pe_nii")
                fwd_conv = convert_series_to_nii(fwd_pe_dir_used, fwd_conv_dir)
                conversions[fwd_pe_dir_used] = fwd_conv
            fwd_pe_nii = fwd_conv["nii"]

        # #17 — orphaned RPE handling (no forward PE partner)
        if rpe_dir_path and not fwd_pe_dir_used:
            if rpe_nvols == 1:
                warnings.append(
                    f"{Path(rpe_dir_path).name} has no matching forward PE image. "
                    f"Will use mean b0 from main DWI as forward PE image (rpe_pair)."
                )
            elif rpe_nvols > 1:
                warnings.append(
                    f"{Path(rpe_dir_path).name} has no matching forward PE image "
                    f"and RPE has multiple volumes. Cannot use for correction. "
                    f"Falling back to rpe_none."
                )
                rpe_dir_path = None
                rpe_nii = None
                rpe_nvols = 0

    # Determine preproc mode
    if rpe_nii is None:
        preproc_mode = "rpe_none"
    elif rpe_nvols == dwi_nvols:
        preproc_mode = "rpe_all"
    elif rpe_nvols > 1 and fwd_pe_nii is not None:
        preproc_mode = "rpe_split"
    else:
        preproc_mode = "rpe_pair"

    # ── Header-based mode sanity check ──────────────────────────────────────
    # Run mrconvert + mrinfo -petable on the main DWI (and RPE if present) to
    # derive the mode independently from the MIF header PE table.  If the result
    # disagrees with the filename-derived mode, warn and override.
    #
    # For rpe_all: the check requires a concatenated AP+PA MIF which does not
    # exist at planning time.  We skip the header check for rpe_all and note it.
    petable_tmp = str(Path(tmp_dir) / "petable_check")
    if preproc_mode == "rpe_all":
        notes_petable = (
            "Header PE check: skipped for rpe_all "
            "(requires concatenated AP+PA MIF — verified during processing)"
        )
    else:
        try:
            header_mode, header_dirs, _ = get_petable_mode(
                nii=dwi_nii,
                bvec=dwi_bvec,
                bval=dwi_bval,
                json_path=dwi_json,
                tmp_dir=petable_tmp,
                has_rpe=rpe_nii is not None,
            )

            # Also check RPE series if present
            rpe_header_mode = None
            if rpe_nii:
                rpe_conv_for_check = conversions.get(rpe_dir_path, {})
                try:
                    rpe_header_mode, _, _ = get_petable_mode(
                        nii=rpe_nii,
                        bvec=rpe_conv_for_check.get("bvec", ""),
                        bval=rpe_conv_for_check.get("bval", ""),
                        json_path=rpe_conv_for_check.get("json", ""),
                        tmp_dir=petable_tmp + "_rpe",
                        has_rpe=False,  # checking the RPE series itself in isolation
                    )
                except RuntimeError as e:
                    warnings.append(
                        f"Header check on RPE series could not be completed: {e} "
                        f"— RPE series assignment retained."
                    )

            if header_mode != preproc_mode:
                warnings.append(
                    f"Preproc mode mismatch — filename-derived: {preproc_mode}, "
                    f"header-derived: {header_mode}. "
                    f"Overriding with header-derived mode. "
                    f"PE directions found in header: {sorted(set(header_dirs))}. "
                    f"Please verify series naming and acquisition protocol."
                )
                preproc_mode = header_mode

            notes_petable = (
                f"Header PE check: {header_mode} "
                f"(directions: {', '.join(sorted(set(header_dirs)))})"
            )
            if rpe_header_mode:
                notes_petable += f"; RPE header: {rpe_header_mode}"

        except RuntimeError as e:
            warnings.append(
                f"Header-based PE mode check could not be completed: {e} "
                f"— falling back to filename-derived mode ({preproc_mode})."
            )
            notes_petable = "Header PE check: unavailable"

    # #21 — note unused PE images when rpe_all is used
    notes = [notes_petable]
    if rpe_all_partner:
        unused = [d for d in fwd_pe_dirs + rpe_dirs if d != rpe_all_partner]
        for d in unused:
            notes.append(f"{Path(d).name} — not used (DWI series has rpe_all partner)")

    # Resolve readout time
    readout_time_override = cfg.get("readout_time", None)
    if readout_time_override is not None:
        readout_time = float(readout_time_override)
        readout_time_source = "config override"
    elif dwi_json and Path(dwi_json).exists():
        readout_time = get_readout_time(dwi_json)
        readout_time_source = f"JSON ({Path(dwi_json).name})"
    else:
        readout_time = 0.0342002
        readout_time_source = "fallback (no JSON found)"

    processing_steps = cfg.get("processing_steps", [])
    do_gradcheck = cfg.get("gradcheck", False) or "gradcheck" in processing_steps
    eddy_options = cfg.get("eddy_options", " --slm=linear")
    keep_tmp = cfg.get("keep_tmp", False)

    _name_parts = []
    if "dwidenoise" in processing_steps:
        _name_parts.append("denoise")
    if "mrgibbs" in processing_steps:
        _name_parts.append("gibbs")
    if "dwifslpreproc" in processing_steps:
        _name_parts.append("preproc")
    if "dwibiascorrect" in processing_steps:
        _name_parts.append("biascorr")
    dwi_preproc_name = ("DWI_" + "_".join(_name_parts) if _name_parts else "DWI") + ".mif.gz"

    # T1 output filename depends on whether registration was performed.
    t1_output_name = "T1_in_DWI_space.nii.gz"

    rpe_conv = conversions.get(rpe_dir_path) if rpe_dir_path else None
    fwd_conv = conversions.get(fwd_pe_dir_used) if fwd_pe_dir_used else None

    return {
        "dwi_name": dwi_name,
        "dwi_dir": dwi_dir,
        "t1_dir": t1_dir,
        "fwd_pe_dir": fwd_pe_dir_used,
        "rpe_dir_path": rpe_dir_path,
        "pe_dir": pe_dir,
        "rpe_dir": rpe_dir,
        "pe_source": pe_source,
        "dwi_nii": dwi_nii,
        "dwi_json": dwi_json,
        "dwi_bvec": dwi_bvec,
        "dwi_bval": dwi_bval,
        "dwi_nvols": dwi_nvols,
        "rpe_nii": rpe_nii,
        "rpe_json": rpe_conv["json"] if rpe_conv else "",
        "rpe_bvec": rpe_conv["bvec"] if rpe_conv else "",
        "rpe_bval": rpe_conv["bval"] if rpe_conv else "",
        "rpe_nvols": rpe_nvols,
        "fwd_pe_nii": fwd_pe_nii,
        "fwd_pe_json": fwd_conv["json"] if fwd_conv else "",
        "fwd_pe_bvec": fwd_conv["bvec"] if fwd_conv else "",
        "fwd_pe_bval": fwd_conv["bval"] if fwd_conv else "",
        "preproc_mode": preproc_mode,
        "readout_time": readout_time,
        "readout_time_src": readout_time_source,
        "processing_steps": processing_steps,
        "do_gradcheck": do_gradcheck,
        "eddy_options": eddy_options,
        "keep_tmp": keep_tmp,
        "dwi_preproc_name": dwi_preproc_name,
        "t1_output_name": t1_output_name,
        "out_dir": out_dir,
        "tmp_dir": tmp_dir,
        "warnings": warnings,
        "notes": notes,
    }


def print_plan(plans: list, skipped: list):
    """Print workflow summary including all warnings and notes."""
    print("\n" + "=" * 70)
    print("WORKFLOW PLAN")
    print("=" * 70)

    if skipped:
        print("\nSKIPPED SERIES:")
        for path, reason in skipped:
            print(f"  SKIPPED: {Path(path).name}")
            print(f"           ({reason})")

    for p in plans:
        rpe_name = Path(p["rpe_dir_path"]).name if p["rpe_dir_path"] else "not found"
        fwd_name = Path(p["fwd_pe_dir"]).name if p["fwd_pe_dir"] else "not used"

        print(f"\nDWI series:       {p['dwi_name']}")
        print(f"  T1:             {Path(p['t1_dir']).name}")

        if p["preproc_mode"] == "rpe_all":
            print(f"  AP/PA pair:     {p['dwi_name']}  ({p['dwi_nvols']} vols)")
            print(f"                  {rpe_name}  ({p['rpe_nvols']} vols)")
            print(f"  Forward PE b0:  not used (rpe_all)")
            print(f"  Reverse PE b0:  not used (rpe_all)")
        else:
            print(f"  Forward PE:     {fwd_name}")
            print(f"  Reverse PE:     {rpe_name}")
            print(f"  DWI volumes:    {p['dwi_nvols']}")
            print(f"  RPE volumes:    {p['rpe_nvols']}")

        pe_str = p["pe_dir"]
        if p["pe_source"] == "json_sidecar":
            pe_str += (
                "  (WARNING: inferred from JSON sidecar — not found in folder name)"
            )
        print(f"  PE direction:   {pe_str}")
        print(f"  Preproc mode:   {p['preproc_mode']}")
        print(f"  Readout time:   {p['readout_time']} ({p['readout_time_src']})")
        _steps = p.get("processing_steps") or []
        _pipeline = ["raw DWI"]
        if p["do_gradcheck"]:
            _pipeline.append("gradcheck")
        if "dwidenoise" in _steps:
            _pipeline.append("dwidenoise")
        if "mrgibbs" in _steps:
            _pipeline.append("mrgibbs")
        if "dwifslpreproc" in _steps:
            _pipeline.append("dwifslpreproc")
        if "dwibiascorrect" in _steps:
            _pipeline.append("dwibiascorrect")
        _pipeline.append("tensor")
        print(f"  Pipeline:       {' → '.join(_pipeline)}")
        print(f"  Output MIF:     {p['dwi_preproc_name']}")

        if p["do_gradcheck"]:
            targets = [f"DWI ({p['pe_dir']})"]
            if p["preproc_mode"] == "rpe_all":
                targets.append(f"DWI ({p['rpe_dir']})")
            elif p["fwd_pe_nii"]:
                targets.append("Forward PE b0")
            if p["rpe_nii"] and p["preproc_mode"] != "rpe_all":
                targets.append("Reverse PE b0")
            print(f"    Gradcheck on: {', '.join(targets)}")

        if "dwidenoise" in (p.get("processing_steps") or []) and p["preproc_mode"] == "rpe_all":
            print(
                f"    Denoise on:   DWI ({p['pe_dir']}) and DWI ({p['rpe_dir']}) separately"
            )

        print(f"  Output:         {p['out_dir']}")

        if p["preproc_mode"] == "rpe_none":
            print(
                f"  NOTE: preproc_mode is rpe_none — no distortion correction applied.\n"
                f"        B0-to-T1 co-registration will still be performed, but EPI\n"
                f"        distortion may reduce registration accuracy.\n"
                f"        Consider acquiring a reverse PE image for optimal results."
            )

        for w in p["warnings"]:
            print(f"  WARNING: {w}")
        for n in p["notes"]:
            print(f"  NOTE:    {n}")

    print("\n" + "=" * 70)
    print("Proceed? (Ctrl+C to abort)")
    print("=" * 70 + "\n")


# =============================================================================
# Pydra tasks
# =============================================================================


@python.define(outputs=["nii", "json_file", "bvec", "bval", "sentinel"])
def stage_series(series_dir: str, out_dir: str) -> tuple[str, str, str, str, str]:
    """Stage a series (DICOM/NIfTI/MIF) to NIfTI. Returns NIfTI + sidecar paths."""
    result = convert_series_to_nii(series_dir, out_dir)
    return (
        result["nii"],
        result.get("json", ""),
        result.get("bvec", ""),
        result.get("bval", ""),
        result["nii"],  # sentinel: forces true lazy dependency in downstream tasks
    )


@python.define
def convert_to_mif_initial(
    nii: str, json_file: str, bvec: str, bval: str, out_path: str
) -> str:
    """Step 1 of gradcheck: mrconvert NIfTI -> MIF with original gradients embedded."""
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    return out_path


@python.define(outputs=["bvec", "bval"])
def run_gradcheck(mif_path: str, out_dir: str, prefix: str) -> tuple[str, str]:
    """Run dwigradcheck on a MIF, export corrected bvec/bval."""
    out_bvec = str(Path(out_dir) / f"{prefix}_corrected.bvec")
    out_bval = str(Path(out_dir) / f"{prefix}_corrected.bval")
    run_cmd(["dwigradcheck", mif_path, "-export_grad_fsl", out_bvec, out_bval])
    return (out_bvec, out_bval)


@python.define
def convert_to_mif_final(
    nii: str,
    json_file: str,
    bvec: str,
    bval: str,
    out_path: str,
    sentinel: str = "",
) -> str:
    """mrconvert NIfTI -> MIF with (optionally corrected) gradients."""
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    result = subprocess.run(
        ["mrinfo", out_path, "-petable"], capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"  Warning: PE table not found in MIF header: {out_path}")
    return out_path


@python.define
def run_dwidenoise(in_mif: str, out_mif: str) -> str:
    run_cmd(["dwidenoise", in_mif, out_mif])
    return out_mif


@python.define
def run_mrdegibbs(in_mif: str, out_mif: str) -> str:
    run_cmd(["mrdegibbs", in_mif, out_mif])
    return out_mif


@python.define
def concatenate_ap_pa(ap_mif: str, pa_mif: str, out_mif: str) -> str:
    """Concatenate AP and PA DWI series (AP first) for rpe_all."""
    run_cmd(["dwicat", ap_mif, pa_mif, out_mif])
    return out_mif


@python.define
def build_se_epi(
    dwi_mif: str,
    rpe_nii: str,
    rpe_json: str,
    rpe_bvec: str,
    rpe_bval: str,
    fwd_pe_nii: str,
    fwd_pe_json: str,
    fwd_pe_bvec: str,
    fwd_pe_bval: str,
    pe_dir: str,
    rpe_dir: str,
    preproc_mode: str,
    tmp_dir: str,
) -> str:
    """Build se_epi pair for rpe_pair / rpe_split. Returns bzero_pair path."""
    if preproc_mode not in ("rpe_pair", "rpe_split"):
        return ""

    rpe_mif = str(Path(tmp_dir) / f"DWI_ref_{rpe_dir}.mif.gz")
    run_cmd(
        [
            "mrconvert",
            rpe_nii,
            rpe_mif,
            "-json_import",
            rpe_json,
            "-fslgrad",
            rpe_bvec,
            rpe_bval,
        ]
    )

    if fwd_pe_nii:
        fwd_mif = str(Path(tmp_dir) / f"DWI_ref_{pe_dir}.mif.gz")
        run_cmd(
            [
                "mrconvert",
                fwd_pe_nii,
                fwd_mif,
                "-json_import",
                fwd_pe_json,
                "-fslgrad",
                fwd_pe_bvec,
                fwd_pe_bval,
            ]
        )
    else:
        # #17 — use mean b0 from main DWI as forward PE
        print("  No forward PE image — computing mean b0 from DWI...")
        mean_b0 = str(Path(tmp_dir) / f"mean_bzero_{pe_dir}.mif.gz")
        extract = subprocess.Popen(
            ["dwiextract", dwi_mif, "-", "-bzero"], stdout=subprocess.PIPE
        )
        subprocess.run(
            ["mrmath", "-", "mean", mean_b0, "-axis", "3"],
            stdin=extract.stdout,
            check=True,
        )
        extract.stdout.close()
        extract.wait()
        fwd_mif = mean_b0

    bzero_pair = str(Path(tmp_dir) / "bzero_pair.mif.gz")
    run_cmd(["dwicat", fwd_mif, rpe_mif, bzero_pair])
    return bzero_pair


@python.define
def run_dwifslpreproc(
    dwi_mif: str,
    out_mif: str,
    pe_dir: str,
    preproc_mode: str,
    se_epi: str,
    readout_time: float,
    eddy_options: str,
    scratch_dir: str = "",
) -> str:
    import os
    if "--nthr" not in eddy_options:
        ncores = os.cpu_count() or 1
        eddy_options = eddy_options.rstrip() + f" --nthr={ncores}"
        print(f"    eddy: using {ncores} CPU thread(s) (--nthr={ncores})")
    scratch_dir = scratch_dir or str(Path(out_mif).parent / "dwifslpreproc_scratch")
    cmd = [
        "dwifslpreproc",
        dwi_mif,
        out_mif,
        "-pe_dir",
        pe_dir,
        f"-{preproc_mode}",
        "-eddy_options",
        eddy_options,
        "-scratch",
        scratch_dir,
        "-nocleanup",
    ]
    if preproc_mode in ("rpe_pair", "rpe_split"):
        cmd += ["-se_epi", se_epi, "-readout_time", str(readout_time), "-align_seepi"]
    elif preproc_mode == "rpe_all":
        cmd += ["-readout_time", str(readout_time)]
    run_cmd(cmd)
    # Clean up scratch after successful completion (dwifslpreproc -nocleanup keeps it)
    import shutil as _shutil

    if Path(scratch_dir).exists():
        _shutil.rmtree(scratch_dir)
    return out_mif


@python.define
def run_dwi2mask(in_mif: str, out_mif: str) -> str:
    run_cmd(["dwi2mask", in_mif, out_mif])
    return out_mif


@python.define
def run_dwibiascorrect(in_mif: str, out_mif: str, mask_mif: str, bias_mif: str) -> str:
    run_cmd(
        [
            "dwibiascorrect",
            "ants",
            in_mif,
            out_mif,
            "-mask",
            mask_mif,
            "-bias",
            bias_mif,
        ]
    )
    return out_mif


@python.define
def extract_mean_b0(dwi_biascorr_mif: str, out_nii: str) -> str:
    extract = subprocess.Popen(
        ["dwiextract", dwi_biascorr_mif, "-", "-bzero"], stdout=subprocess.PIPE
    )
    subprocess.run(
        ["mrmath", "-", "mean", out_nii, "-axis", "3"], stdin=extract.stdout, check=True
    )
    extract.stdout.close()
    extract.wait()
    return out_nii


@python.define
def register_b0_to_t1(b0_nii: str, t1_nii: str, out_b0_in_t1: str, out_mat: str) -> str:
    run_cmd(
        [
            "flirt",
            "-in",
            b0_nii,
            "-ref",
            t1_nii,
            "-out",
            out_b0_in_t1,
            "-omat",
            out_mat,
            "-dof",
            "6",
        ]
    )
    return out_mat


@python.define(outputs=["t1_in_dwi", "mrtrix_xfm"])
def invert_and_apply_transform(
    b02t1_mat: str, t1_nii: str, b0_nii: str, tmp_dir: str
) -> tuple[str, str]:
    t12b0_mat = str(Path(tmp_dir) / "T12b0.mat")
    t1_in_dwi = str(Path(tmp_dir) / "T1_in_DWI_space.nii.gz")
    mrtrix_txt = str(Path(tmp_dir) / "struct2diff_mrtrix.txt")
    run_cmd(["convert_xfm", "-omat", t12b0_mat, "-inverse", b02t1_mat])
    run_cmd(
        [
            "flirt",
            "-in",
            t1_nii,
            "-ref",
            b0_nii,
            "-out",
            t1_in_dwi,
            "-init",
            t12b0_mat,
            "-applyxfm",
        ]
    )
    run_cmd(["transformconvert", t12b0_mat, t1_nii, b0_nii, "flirt_import", mrtrix_txt])
    return (t1_in_dwi, mrtrix_txt)


@python.define
def copy_t1_as_dwi_space(t1_nii: str, tmp_dir: str) -> str:
    """
    rpe_none fallback: copy the T1 directly to
    T1_in_DWI_space.nii.gz without performing any registration.

    B0-to-T1 co-registration is unreliable when no distortion correction
    is applied (rpe_none), so the native T1 is used as-is.  Stage 2 of the
    orchestrator (phantom QC in DWI space) will therefore operate on the
    native T1 geometry rather than a registered approximation.
    """

    t1_out = str(Path(tmp_dir) / "T1.nii.gz")
    shutil.copy2(t1_nii, t1_out)
    print(f"  Copied T1 (no registration): {Path(t1_out).name}")
    return t1_out


@python.define(outputs=["adc", "fa"])
def compute_tensor_metrics(dwi_biascorr_mif: str, tmp_dir: str) -> tuple[str, str]:
    tensor_mif = str(Path(tmp_dir) / "tensor.mif.gz")
    adc_mif = str(Path(tmp_dir) / "ADC.mif.gz")
    fa_mif = str(Path(tmp_dir) / "FA.mif.gz")
    adc_nii = str(Path(tmp_dir) / "ADC.nii.gz")
    fa_nii = str(Path(tmp_dir) / "FA.nii.gz")
    run_cmd(["dwi2tensor", dwi_biascorr_mif, tensor_mif])
    run_cmd(["tensor2metric", "-adc", adc_mif, "-fa", fa_mif, tensor_mif])
    run_cmd(["mrconvert", adc_mif, adc_nii])
    run_cmd(["mrconvert", fa_mif, fa_nii])
    return (adc_nii, fa_nii)


@python.define(outputs=["out"])
def copy_final_outputs_task(
    dwi_biascorr_mif: str,
    t1_in_dwi: str,
    adc_nii: str,
    fa_nii: str,
    out_dir: str,
    dwi_preproc_name: str,
    t1_output_name: str,
    dwi_raw_mif: str = "",
):
    os.makedirs(out_dir, exist_ok=True)
    dst_dwi = str(Path(out_dir) / dwi_preproc_name)
    dst_t1 = str(Path(out_dir) / t1_output_name)
    dst_adc = str(Path(out_dir) / "ADC.nii.gz")
    dst_fa = str(Path(out_dir) / "FA.nii.gz")
    copies = [
        (dwi_biascorr_mif, dst_dwi),
        (t1_in_dwi, dst_t1),
        (adc_nii, dst_adc),
        (fa_nii, dst_fa),
    ]
    if dwi_raw_mif and Path(dwi_raw_mif).exists():
        copies.append((dwi_raw_mif, str(Path(out_dir) / "DWI_raw.mif.gz")))
    for src, dst in copies:
        shutil.copy2(src, dst)
        print(f"  Copied: {Path(dst).name}")
    return dst_dwi  # sentinel: used by cleanup ordering


def cleanup_tmp(sentinel: str, tmp_dir: str, keep_tmp: bool) -> str:
    """
    Remove the tmp directory after all outputs have been copied.
    Skipped if keep_tmp is True (--nocleanup flag).
    sentinel: any upstream output — used only to enforce task ordering.
    """
    if keep_tmp:
        print(f"  --nocleanup set: retaining {tmp_dir}")
        return tmp_dir
    if Path(tmp_dir).exists():
        shutil.rmtree(tmp_dir)
        print(f"  Removed tmp directory: {tmp_dir}")
    return tmp_dir


# =============================================================================
# FSL / MRtrix3 subprocess wrappers — bypass pydra 1.0a9 type-checking issues
# (mandatory-field rules, xor semantics, eddy_options quoting, MultiInputObj)
# =============================================================================


@python.define(outputs=["out"])
def RunDwicat(input1: object, input2: object) -> ImageFormatGz:
    """Concatenate two DWI series using dwicat via subprocess.

    DwiCatMulti passes inputs as a Python list to MultiInputObj, which pydra
    1.0a9 cannot coerce when the elements are lazy fields.  Running via
    subprocess avoids the type-checking issue entirely.
    """
    import subprocess
    from pathlib import Path as _Path
    from fileformats.vendor.mrtrix3.medimage import ImageFormatGz as _MifGz

    out_file = str(_Path.cwd() / "dwicat_out.mif.gz")
    subprocess.run(["dwicat", str(input1), str(input2), out_file], check=True)
    return _MifGz(out_file)


@python.define(outputs=["out"])
def RunDwifslpreproc(
    in_file: object,
    pe_dir: str,
    preproc_mode: str,
    eddy_options: str | None = None,
    readout_time: float | None = None,
    se_epi: object = None,
) -> ImageFormatGz:
    """Run dwifslpreproc via subprocess with correct eddy_options quoting.

    pydra-tasks-fsl formats the command as a string then splits on whitespace,
    stripping any leading space from eddy_options values.  Without the leading
    space, MRtrix3's argparse mistakes '--slm=linear' for a new option flag and
    raises 'expected one argument'.  Running via subprocess list preserves the
    leading space and avoids the xor/mandatory-field rule breakage in
    pydra-tasks-fsl 0.3.1 under pydra 1.0a9.
    """
    import subprocess
    from pathlib import Path as _Path
    from fileformats.vendor.mrtrix3.medimage import ImageFormatGz as _MifGz

    out_file = str(_Path.cwd() / "dwi_preproc.mif.gz")
    cmd = ["dwifslpreproc", str(in_file), out_file, "-pe_dir", pe_dir]

    if preproc_mode == "rpe_all":
        cmd.append("-rpe_all")
    elif preproc_mode in ("rpe_pair", "rpe_split") and se_epi is not None:
        cmd += ["-rpe_pair", "-se_epi", str(se_epi)]
    else:
        cmd.append("-rpe_none")

    if readout_time is not None:
        cmd += ["-readout_time", str(readout_time)]

    if eddy_options:
        import os as _os
        if "--nthr" not in eddy_options:
            ncores = _os.cpu_count() or 1
            eddy_options = eddy_options.rstrip() + f" --nthr={ncores}"
        # Prepend a space so MRtrix3's argparse treats the value as a positional
        # string and not an option flag (values starting with '--' are misread).
        cmd += ["-eddy_options", f" {eddy_options.strip()}"]

    subprocess.run(cmd, check=True)
    return _MifGz(out_file)


@python.define(outputs=["out"])
def MrThresholdMask(b0: object) -> ImageFormatGz:
    """Create a binary phantom mask from mean b0 using mrthreshold (Otsu).

    dwi2mask fslbet uses FSL BET which requires brain-like morphology and
    fails on phantom data.  mrthreshold with Otsu reliably separates the
    phantom object (high signal) from background regardless of shape.
    """
    import subprocess
    from pathlib import Path as _Path
    from fileformats.vendor.mrtrix3.medimage import ImageFormatGz as _MifGz

    out_file = str(_Path.cwd() / "phantom_mask.mif.gz")
    subprocess.run(["mrthreshold", str(b0), out_file], check=True)
    return _MifGz(out_file)


@python.define(outputs=["out"])
def FlirtCoregister(b0: object, t1: object) -> NiftiGz:
    """Register b0 to T1 (6 DOF), invert transform, warp T1 into DWI space.

    Runs flirt and convert_xfm via subprocess, avoiding pydra-tasks-fsl
    compatibility breakage with pydra 1.0a9 (xor-group mandatory-field rules).
    Returns T1_in_DWI_space.nii.gz as a NiftiGz.
    """
    import subprocess
    from pathlib import Path as _Path
    from fileformats.medimage import NiftiGz as _NiftiGz

    cwd = _Path.cwd()
    b0_str = str(b0)
    t1_str = str(t1)
    mat_fwd = str(cwd / "b0_to_t1.mat")
    mat_inv = str(cwd / "t1_to_b0.mat")
    out_path = str(cwd / "T1_in_DWI_space.nii.gz")

    subprocess.run(
        ["flirt", "-in", b0_str, "-ref", t1_str, "-dof", "6", "-omat", mat_fwd],
        check=True,
    )
    subprocess.run(
        ["convert_xfm", "-omat", mat_inv, "-inverse", mat_fwd],
        check=True,
    )
    subprocess.run(
        ["flirt", "-in", t1_str, "-ref", b0_str, "-applyxfm", "-init", mat_inv, "-out", out_path],
        check=True,
    )
    return _NiftiGz(out_path)


# =============================================================================
# Type-narrowing adapter for Tensor2Metric dual-use outputs
# =============================================================================


@python.define(outputs=["out"])
def CastToMif(path: object) -> ImageFormatGz:
    """Coerce a Tensor2Metric dual-use output (Union[image, bool, None]) to ImageFormatGz.

    Tensor2Metric's adc/fa outputs are declared as Union[image_types, bool, NoneType]
    because the same field doubles as the boolean enable-flag input.  Passing the lazy
    field directly to MrConvert.in_file fails pydra's static type check.  This adapter
    accepts `object` (compatible with any upstream type) and re-wraps the resolved path
    as an explicit ImageFormatGz so downstream tasks see a clean image type.
    """
    return ImageFormatGz(str(path))


# =============================================================================
# DWISeriesWorkflow — canonical @workflow.define for a single DWI series
# =============================================================================


@workflow.define(outputs=["t1_in_dwi", "adc", "fa", "dwi_preproc"])
def DWISeriesWorkflow(
    t1w: NiftiGz,
    dwi: ImageIn,
    dwi_pe_dir: str,
    preproc_mode: str,
    rpe: ImageIn | None = None,
    fwd_b0: NiftiGz | None = None,
    readout_time: float = 0.05,
    eddy_options: str = " --slm=linear",
    do_denoise: bool = False,
    do_degibbs: bool = False,
    do_fslpreproc: bool = False,
    do_biascorrect: bool = False,
    gradcheck: bool = False,
) -> tuple[NiftiGz, NiftiGz, NiftiGz, ImageOut]:
    """End-to-end DWI preprocessing + T1 coregistration + tensor metrics.

    Inputs are concrete typed fileformats objects — directory scanning and
    series classification happen in the CLI before this workflow is instantiated.
    Conditional branches (``do_*``, ``gradcheck``, ``preproc_mode``) are
    evaluated at graph-construction time using the concrete values provided.
    Individual bool flags are used instead of list[str] to avoid pydra
    serialisation issues with list types across nested workflow boundaries.
    """
    from pydra.tasks.mrtrix3.v3_1 import (
        DwiDenoise,
        DwiBiascorrect_Ants,
        DwiExtract,
        MrMath,
        MrConvert,
        Dwi2Tensor,
        Tensor2Metric,
        DwiGradcheck,
        MrDegibbs,
    )
    # ── Optional gradient-table correction ───────────────────────────────────
    if gradcheck:
        gc = workflow.add(
            DwiGradcheck(in_file=dwi, export_grad_mrtrix=True, config=[]),
            name="gradcheck",
        )
        dwi_in = workflow.add(
            MrConvert(in_file=dwi, grad=gc.export_grad_mrtrix),
            name="embed_corrected_grads",
        ).out_file
        # For rpe_all the RPE has the full multi-direction DWI series —
        # apply gradcheck to it independently so each image carries its own
        # corrected gradient table into the subsequent processing steps.
        if rpe is not None and preproc_mode == "rpe_all":
            gc_rpe = workflow.add(
                DwiGradcheck(in_file=rpe, export_grad_mrtrix=True, config=[]),
                name="gradcheck_rpe",
            )
            rpe_in = workflow.add(
                MrConvert(in_file=rpe, grad=gc_rpe.export_grad_mrtrix),
                name="embed_corrected_grads_rpe",
            ).out_file
        else:
            rpe_in = rpe
    else:
        dwi_in = dwi
        rpe_in = rpe

    # ── Optional denoising ────────────────────────────────────────────────────
    if do_denoise:
        dn = workflow.add(DwiDenoise(dwi=dwi_in), name="denoise")
        dwi_after_dn = dn.out
        if rpe is not None and preproc_mode == "rpe_all":
            dn_rpe = workflow.add(DwiDenoise(dwi=rpe_in), name="denoise_rpe")
            rpe_after_dn = dn_rpe.out
        else:
            rpe_after_dn = rpe_in
    else:
        dwi_after_dn = dwi_in
        rpe_after_dn = rpe_in

    # ── Optional Gibbs ringing removal ───────────────────────────────────────
    if do_degibbs:
        dg = workflow.add(MrDegibbs(in_=dwi_after_dn), name="degibbs")
        dwi_for_preproc = dg.out
        if rpe is not None and preproc_mode == "rpe_all":
            dg_rpe = workflow.add(MrDegibbs(in_=rpe_after_dn), name="degibbs_rpe")
            rpe_for_preproc = dg_rpe.out
        else:
            rpe_for_preproc = rpe_after_dn
    else:
        dwi_for_preproc = dwi_after_dn
        rpe_for_preproc = rpe_after_dn

    # ── Optional dwifslpreproc (phase-encoding correction) ────────────────────
    if do_fslpreproc:
        if preproc_mode == "rpe_all" and rpe is not None:
            cat = workflow.add(
                RunDwicat(input1=dwi_for_preproc, input2=rpe_for_preproc),
                name="concat_ap_pa",
            )
            preproc = workflow.add(
                RunDwifslpreproc(
                    in_file=cat.out,
                    pe_dir=dwi_pe_dir,
                    preproc_mode="rpe_all",
                    readout_time=readout_time,
                    eddy_options=eddy_options,
                ),
                name="preproc",
            )
        elif preproc_mode in ("rpe_pair", "rpe_split") and rpe is not None:
            rpe_b0 = workflow.add(
                DwiExtract(in_file=rpe, bzero=True), name="rpe_b0_extract"
            )
            if fwd_b0 is not None:
                fwd_as_mif = workflow.add(
                    MrConvert(in_file=fwd_b0), name="fwd_b0_to_mif"
                )
                se_epi = workflow.add(
                    RunDwicat(input1=fwd_as_mif.out_file, input2=rpe_b0.out_file),
                    name="se_epi_pair",
                ).out
            else:
                se_epi = rpe_b0.out_file
            preproc = workflow.add(
                RunDwifslpreproc(
                    in_file=dwi_for_preproc,
                    pe_dir=dwi_pe_dir,
                    preproc_mode=preproc_mode,
                    se_epi=se_epi,
                    readout_time=readout_time,
                    eddy_options=eddy_options,
                ),
                name="preproc",
            )
        else:  # rpe_none
            preproc = workflow.add(
                RunDwifslpreproc(
                    in_file=dwi_for_preproc,
                    pe_dir=dwi_pe_dir,
                    preproc_mode="rpe_none",
                    eddy_options=eddy_options,
                ),
                name="preproc",
            )
        preproc_out = preproc.out
    else:
        preproc_out = dwi_for_preproc

    # ── Mean b0 → phantom mask + FLIRT reference ─────────────────────────────
    b0_vols = workflow.add(
        DwiExtract(in_file=preproc_out, bzero=True), name="b0_extract"
    )
    b0_mean = workflow.add(
        MrMath(in_file=b0_vols.out_file, operation="mean", axis=3), name="b0_mean"
    )
    b0_nii = workflow.add(
        MrConvert(in_file=b0_mean.out_file, out_file="b0_mean.nii.gz"),
        name="b0_to_nii",
    )

    # ── Optional bias field correction ────────────────────────────────────────
    if do_biascorrect:
        mask = workflow.add(MrThresholdMask(b0=b0_mean.out_file), name="mask")
        biascorr = workflow.add(
            DwiBiascorrect_Ants(in_file=preproc_out, mask=mask.out, config=[]),
            name="biascorr",
        )
        dwi_for_tensor = biascorr.out_file
    else:
        dwi_for_tensor = preproc_out

    # ── T1 → DWI space coregistration (FLIRT rigid, 6 DOF) ───────────────────
    t1_in_dwi = workflow.add(
        FlirtCoregister(b0=b0_nii.out_file, t1=t1w),
        name="coreg",
    )

    # ── Diffusion tensor + metrics ────────────────────────────────────────────
    tensor = workflow.add(Dwi2Tensor(dwi=dwi_for_tensor), name="tensor")
    metrics = workflow.add(
        Tensor2Metric(tensor=tensor.dt, adc="ADC.mif.gz", fa="FA.mif.gz"), name="metrics"
    )
    adc_cast = workflow.add(CastToMif(path=metrics.adc), name="adc_cast")
    fa_cast = workflow.add(CastToMif(path=metrics.fa), name="fa_cast")
    adc_nii = workflow.add(
        MrConvert(in_file=adc_cast.out, out_file="ADC.nii.gz"), name="adc_to_nii"
    )
    fa_nii = workflow.add(
        MrConvert(in_file=fa_cast.out, out_file="FA.nii.gz"), name="fa_to_nii"
    )

    return (
        t1_in_dwi.out,
        adc_nii.out_file,
        fa_nii.out_file,
        dwi_for_tensor,
    )


# =============================================================================
# Per-DWI workflow builder (legacy path — kept for backward compat)
# =============================================================================


def _exists(*paths) -> bool:
    """Return True only if every path exists on disk."""
    return all(Path(p).exists() for p in paths)


def build_dwi_workflow(plan: dict):
    """
    Build and execute a pydra 1.0 workflow for a single DWI series.

    Design decisions:
    - cache_root is set per-series to <out_dir>/.pydra_cache — isolated,
      avoids cross-series contamination.
    - rerun=True ensures pydra re-executes rather than replaying stale cache.
    - Conditional branching (preproc_mode, processing_steps, do_gradcheck) is
      resolved at workflow-class-definition time using concrete plan values
      captured in the closure. Each unique combination gets a distinctly-named
      class so pydra's cache correctly distinguishes them.
    - stage_series returns a sentinel output (=nii) so downstream tasks that
      need T1-before-DWI ordering have a true lazy dependency edge, not a
      resolved string literal that pydra treats as a known constant.
    - CP1 (all final outputs exist) is checked before building the workflow —
      fast early-return with no pydra overhead.
    - cleanup_tmp runs as plain Python after sub() returns, guaranteeing it
      only executes after all outputs are confirmed copied.
    """
    tmp_dir = plan["tmp_dir"]
    out_dir = plan["out_dir"]
    pe_dir = plan["pe_dir"]
    rpe_dir = plan["rpe_dir"]
    preproc_mode = plan["preproc_mode"]
    processing_steps = plan.get("processing_steps") or []
    do_denoise = "dwidenoise" in processing_steps
    do_degibbs = "mrgibbs" in processing_steps
    do_fslpreproc = "dwifslpreproc" in processing_steps
    do_biascorrect = "dwibiascorrect" in processing_steps
    do_gradcheck = plan["do_gradcheck"]
    readout_time = plan["readout_time"]
    eddy_options = plan["eddy_options"]
    keep_tmp = plan["keep_tmp"]
    dwi_preproc_name = plan["dwi_preproc_name"]
    t1_output_name = plan["t1_output_name"]
    t1_dir = plan["t1_dir"]
    dwi_nii = plan["dwi_nii"]
    dwi_json = plan["dwi_json"]
    dwi_bvec = plan["dwi_bvec"]
    dwi_bval = plan["dwi_bval"]
    rpe_nii = plan["rpe_nii"]
    rpe_json = plan["rpe_json"]
    rpe_bvec = plan["rpe_bvec"]
    rpe_bval = plan["rpe_bval"]
    fwd_pe_nii = plan["fwd_pe_nii"]
    fwd_pe_json = plan["fwd_pe_json"]
    fwd_pe_bvec = plan["fwd_pe_bvec"]
    fwd_pe_bval = plan["fwd_pe_bval"]

    t1_conv_dir = str(Path(tmp_dir) / "t1_nii")
    dwi_raw_mif = str(Path(tmp_dir) / f"DWI_raw_{pe_dir}.mif.gz")
    preproc_mif = str(Path(tmp_dir) / "DWI_preproc.mif.gz")
    biascorr_mif = str(Path(tmp_dir) / dwi_preproc_name)
    t1_in_dwi = str(Path(tmp_dir) / t1_output_name)

    out_dwi = str(Path(out_dir) / dwi_preproc_name)
    out_adc = str(Path(out_dir) / "ADC.nii.gz")
    out_fa = str(Path(out_dir) / "FA.nii.gz")
    out_t1 = str(Path(out_dir) / t1_output_name)

    # ── CP1: all final outputs exist — nothing to do ─────────────────────────
    if _exists(out_dwi, out_adc, out_fa, out_t1):
        print(f"  [skip] All outputs already exist in {Path(out_dir).name}")
        return

    # ── CP2: bias-corrected MIF exists in out_dir — skip DWI preprocessing ───
    # dwifslpreproc + dwibiascorrect are by far the most expensive steps.
    # If the bias-corrected MIF already exists in out_dir (e.g. from a prior
    # successful run where only downstream steps failed), copy it into tmp/ so
    # the pydra workflow can use it directly and skip straight to coregistration
    # and tensor metrics.
    skip_preproc = _exists(out_dwi)
    if skip_preproc:
        print(f"  [skip] DWI preprocessing already done: {Path(out_dwi).name}")
        os.makedirs(tmp_dir, exist_ok=True)
        if not _exists(biascorr_mif):
            shutil.copy2(out_dwi, biascorr_mif)

    # ── Workflow class name encodes the branch combination ────────────────────
    # Each unique combination gets a distinct class name so pydra caches them
    # independently. skip_preproc adds an extra tag to distinguish the two paths.
    steps_tag = "_".join(sorted(processing_steps)) if processing_steps else "nosteps"
    gc_tag = "gc" if do_gradcheck else "nogc"
    sp_tag = "skippreproc" if skip_preproc else "fullpreproc"
    wf_class_name = f"DwiWf_{preproc_mode}_{steps_tag}_{gc_tag}_{sp_tag}"

    if skip_preproc:
        # ── Reduced workflow: T1 conversion + registration + tensor + copy ───
        # biascorr_mif is already in tmp/ (copied from out_dir above).
        @workflow.define(outputs=["result"])
        def DwiWf(x: int) -> str:

            # T1 series staging (always needed for registration)
            conv_t1 = workflow.add(
                stage_series(series_dir=t1_dir, out_dir=t1_conv_dir),
                name="convert_t1",
            )

            # ── T1-to-DWI registration ────────────────────────────────────────
            b0 = workflow.add(
                extract_mean_b0(
                    dwi_biascorr_mif=biascorr_mif,
                    out_nii=str(Path(tmp_dir) / "bzero_f.nii.gz"),
                ),
                name="mean_b0",
            )
            flirt = workflow.add(
                register_b0_to_t1(
                    b0_nii=b0.out,
                    t1_nii=conv_t1.nii,
                    out_b0_in_t1=str(Path(tmp_dir) / "b0_to_T1.nii.gz"),
                    out_mat=str(Path(tmp_dir) / "b02T1.mat"),
                ),
                name="register",
            )
            xfm = workflow.add(
                invert_and_apply_transform(
                    b02t1_mat=flirt.out,
                    t1_nii=conv_t1.nii,
                    b0_nii=b0.out,
                    tmp_dir=tmp_dir,
                ),
                name="invert_xfm",
            )
            t1_result = xfm.t1_in_dwi

            # ── Tensor metrics ────────────────────────────────────────────────
            tensor = workflow.add(
                compute_tensor_metrics(
                    dwi_biascorr_mif=biascorr_mif,
                    tmp_dir=tmp_dir,
                ),
                name="tensor",
            )

            # ── Copy final outputs ────────────────────────────────────────────
            copy = workflow.add(
                copy_final_outputs_task(
                    dwi_biascorr_mif=biascorr_mif,
                    t1_in_dwi=t1_result,
                    adc_nii=tensor.adc,
                    fa_nii=tensor.fa,
                    out_dir=out_dir,
                    dwi_preproc_name=dwi_preproc_name,
                    t1_output_name=t1_output_name,
                ),
                name="copy_outputs",
            )
            return copy.out

    else:
        # ── Full workflow: all stages ─────────────────────────────────────────
        @workflow.define(outputs=["result"])
        def DwiWf(x: int) -> str:

            # ── T1 series staging ─────────────────────────────────────────────
            # Always run — needed for registration regardless of skip_preproc.
            conv_t1 = workflow.add(
                stage_series(series_dir=t1_dir, out_dir=t1_conv_dir),
                name="convert_t1",
            )
            # conv_t1.sentinel is a true lazy output — used to gate DWI chain

            # ── DWI → MIF (with optional gradcheck) ──────────────────────────
            if do_gradcheck:
                mif_init = workflow.add(
                    convert_to_mif_initial(
                        nii=dwi_nii,
                        json_file=dwi_json,
                        bvec=dwi_bvec,
                        bval=dwi_bval,
                        out_path=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_init.mif.gz"),
                    ),
                    name="dwi_to_mif_init",
                )
                gc_dwi = workflow.add(
                    run_gradcheck(mif_path=mif_init.out, out_dir=tmp_dir, prefix="dwi"),
                    name="gradcheck_dwi",
                )
                dwi_bvec_lazy = gc_dwi.bvec
                dwi_bval_lazy = gc_dwi.bval

                if preproc_mode == "rpe_all" and rpe_nii:
                    rpe_init = workflow.add(
                        convert_to_mif_initial(
                            nii=rpe_nii,
                            json_file=rpe_json,
                            bvec=rpe_bvec,
                            bval=rpe_bval,
                            out_path=str(
                                Path(tmp_dir) / f"DWI_raw_{rpe_dir}_init.mif.gz"
                            ),
                        ),
                        name="rpe_to_mif_init",
                    )
                    gc_rpe = workflow.add(
                        run_gradcheck(
                            mif_path=rpe_init.out, out_dir=tmp_dir, prefix="rpe"
                        ),
                        name="gradcheck_rpe",
                    )
                    rpe_bvec_lazy = gc_rpe.bvec
                    rpe_bval_lazy = gc_rpe.bval
                else:
                    rpe_bvec_lazy = rpe_bvec
                    rpe_bval_lazy = rpe_bval

                if fwd_pe_nii and preproc_mode in ("rpe_pair", "rpe_split"):
                    fwd_init = workflow.add(
                        convert_to_mif_initial(
                            nii=fwd_pe_nii,
                            json_file=fwd_pe_json,
                            bvec=fwd_pe_bvec,
                            bval=fwd_pe_bval,
                            out_path=str(
                                Path(tmp_dir) / f"DWI_ref_{pe_dir}_init.mif.gz"
                            ),
                        ),
                        name="fwd_to_mif_init",
                    )
                    gc_fwd = workflow.add(
                        run_gradcheck(
                            mif_path=fwd_init.out, out_dir=tmp_dir, prefix="fwd"
                        ),
                        name="gradcheck_fwd",
                    )
                    fwd_bvec_lazy = gc_fwd.bvec
                    fwd_bval_lazy = gc_fwd.bval
                else:
                    fwd_bvec_lazy = fwd_pe_bvec
                    fwd_bval_lazy = fwd_pe_bval
            else:
                dwi_bvec_lazy = dwi_bvec
                dwi_bval_lazy = dwi_bval
                rpe_bvec_lazy = rpe_bvec
                rpe_bval_lazy = rpe_bval
                fwd_bvec_lazy = fwd_pe_bvec
                fwd_bval_lazy = fwd_pe_bval

            dwi_mif = workflow.add(
                convert_to_mif_final(
                    nii=dwi_nii,
                    json_file=dwi_json,
                    bvec=dwi_bvec_lazy,
                    bval=dwi_bval_lazy,
                    out_path=dwi_raw_mif,
                    sentinel=conv_t1.sentinel,
                ),
                name="dwi_to_mif",
            )

            if preproc_mode == "rpe_all" and rpe_nii:
                rpe_mif = workflow.add(
                    convert_to_mif_final(
                        nii=rpe_nii,
                        json_file=rpe_json,
                        bvec=rpe_bvec_lazy,
                        bval=rpe_bval_lazy,
                        out_path=str(Path(tmp_dir) / f"DWI_raw_{rpe_dir}.mif.gz"),
                        sentinel=conv_t1.sentinel,
                    ),
                    name="rpe_to_mif",
                )
                rpe_mif_out = rpe_mif.out
            else:
                rpe_mif_out = None

            # ── Optional denoising ────────────────────────────────────────────
            if do_denoise:
                dn_dwi = workflow.add(
                    run_dwidenoise(
                        in_mif=dwi_mif.out,
                        out_mif=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_denoise.mif.gz"),
                    ),
                    name="denoise_dwi",
                )
                dwi_after_dn = dn_dwi.out
                if preproc_mode == "rpe_all" and rpe_nii:
                    dn_rpe = workflow.add(
                        run_dwidenoise(
                            in_mif=rpe_mif_out,
                            out_mif=str(
                                Path(tmp_dir) / f"DWI_raw_{rpe_dir}_denoise.mif.gz"
                            ),
                        ),
                        name="denoise_rpe",
                    )
                    rpe_after_dn = dn_rpe.out
                else:
                    rpe_after_dn = rpe_mif_out
            else:
                dwi_after_dn = dwi_mif.out
                rpe_after_dn = rpe_mif_out

            # ── Optional Gibbs ringing removal ───────────────────────────────
            if do_degibbs:
                dg_dwi = workflow.add(
                    run_mrdegibbs(
                        in_mif=dwi_after_dn,
                        out_mif=str(
                            Path(tmp_dir) / f"DWI_raw_{pe_dir}_gibbs.mif.gz"
                        ),
                    ),
                    name="degibbs_dwi",
                )
                dwi_for_preproc = dg_dwi.out
                if preproc_mode == "rpe_all" and rpe_nii:
                    dg_rpe = workflow.add(
                        run_mrdegibbs(
                            in_mif=rpe_after_dn,
                            out_mif=str(
                                Path(tmp_dir) / f"DWI_raw_{rpe_dir}_gibbs.mif.gz"
                            ),
                        ),
                        name="degibbs_rpe",
                    )
                    rpe_for_preproc = dg_rpe.out
                else:
                    rpe_for_preproc = rpe_after_dn
            else:
                dwi_for_preproc = dwi_after_dn
                rpe_for_preproc = rpe_after_dn

            # ── Optional dwifslpreproc ────────────────────────────────────────
            if do_fslpreproc:
                # ── Concatenate AP+PA for rpe_all ─────────────────────────────
                if preproc_mode == "rpe_all" and rpe_nii:
                    cat = workflow.add(
                        concatenate_ap_pa(
                            ap_mif=dwi_for_preproc,
                            pa_mif=rpe_for_preproc,
                            out_mif=str(Path(tmp_dir) / "DWI_AP_PA_concat.mif.gz"),
                        ),
                        name="concat_ap_pa",
                    )
                    dwi_input = cat.out
                else:
                    dwi_input = dwi_for_preproc

                # ── se_epi pair (rpe_pair / rpe_split) ────────────────────────
                se = workflow.add(
                    build_se_epi(
                        dwi_mif=dwi_for_preproc,
                        rpe_nii=rpe_nii or "",
                        rpe_json=rpe_json,
                        rpe_bvec=rpe_bvec_lazy,
                        rpe_bval=rpe_bval_lazy,
                        fwd_pe_nii=fwd_pe_nii or "",
                        fwd_pe_json=fwd_pe_json,
                        fwd_pe_bvec=fwd_bvec_lazy,
                        fwd_pe_bval=fwd_bval_lazy,
                        pe_dir=pe_dir,
                        rpe_dir=rpe_dir,
                        preproc_mode=preproc_mode,
                        tmp_dir=tmp_dir,
                    ),
                    name="se_epi",
                )
                fslpreproc = workflow.add(
                    run_dwifslpreproc(
                        dwi_mif=dwi_input,
                        out_mif=preproc_mif,
                        pe_dir=pe_dir,
                        preproc_mode=preproc_mode,
                        se_epi=se.out,
                        readout_time=readout_time,
                        eddy_options=eddy_options,
                        scratch_dir=str(Path(tmp_dir) / "dwifslpreproc_scratch"),
                    ),
                    name="fslpreproc",
                )
                preproc_out = fslpreproc.out
            else:
                preproc_out = dwi_for_preproc

            # ── Optional bias field correction ────────────────────────────────
            if do_biascorrect:
                dwi_mask = workflow.add(
                    run_dwi2mask(
                        in_mif=preproc_out,
                        out_mif=str(Path(tmp_dir) / "DWI_preproc_mask.mif.gz"),
                    ),
                    name="dwi_mask",
                )
                biascorr = workflow.add(
                    run_dwibiascorrect(
                        in_mif=preproc_out,
                        out_mif=biascorr_mif,
                        mask_mif=dwi_mask.out,
                        bias_mif=str(Path(tmp_dir) / "DWI_preproc_bias.mif.gz"),
                    ),
                    name="dwi_biascorr",
                )
                biascorr_out = biascorr.out
            else:
                biascorr_out = preproc_out

            # ── T1-to-DWI registration ────────────────────────────────────────
            b0 = workflow.add(
                extract_mean_b0(
                    dwi_biascorr_mif=biascorr_out,
                    out_nii=str(Path(tmp_dir) / "bzero_f.nii.gz"),
                ),
                name="mean_b0",
            )
            flirt = workflow.add(
                register_b0_to_t1(
                    b0_nii=b0.out,
                    t1_nii=conv_t1.nii,
                    out_b0_in_t1=str(Path(tmp_dir) / "b0_to_T1.nii.gz"),
                    out_mat=str(Path(tmp_dir) / "b02T1.mat"),
                ),
                name="register",
            )
            xfm = workflow.add(
                invert_and_apply_transform(
                    b02t1_mat=flirt.out,
                    t1_nii=conv_t1.nii,
                    b0_nii=b0.out,
                    tmp_dir=tmp_dir,
                ),
                name="invert_xfm",
            )
            t1_result = xfm.t1_in_dwi

            # ── Tensor metrics ────────────────────────────────────────────────
            tensor = workflow.add(
                compute_tensor_metrics(
                    dwi_biascorr_mif=biascorr_out,
                    tmp_dir=tmp_dir,
                ),
                name="tensor",
            )

            # ── Copy final outputs ────────────────────────────────────────────
            copy = workflow.add(
                copy_final_outputs_task(
                    dwi_biascorr_mif=biascorr_out,
                    t1_in_dwi=t1_result,
                    adc_nii=tensor.adc,
                    fa_nii=tensor.fa,
                    out_dir=out_dir,
                    dwi_preproc_name=dwi_preproc_name,
                    t1_output_name=t1_output_name,
                    dwi_raw_mif=dwi_raw_mif,
                ),
                name="copy_outputs",
            )
            return copy.out

    # Rename the class so pydra treats each branch combination as a distinct type
    DwiWf.__name__ = wf_class_name
    DwiWf.__qualname__ = wf_class_name

    cache_dir = str(Path(out_dir) / ".pydra_cache")
    wf = DwiWf(x=1)
    return wf, cache_dir, tmp_dir, keep_tmp


# =============================================================================
# Top-level pipeline
# =============================================================================


def run_pipeline(cfg: dict):
    scans_dir = cfg["scans_dir"]
    output_dir = cfg.get("output_dir", str(Path(scans_dir).name))
    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scans directory:  {scans_dir}")
    print(f"Processing steps: {', '.join(cfg.get('processing_steps', [])) or '(none)'}")
    print(f"Gradcheck:        {cfg.get('gradcheck', False)}")
    print(f"Keep tmp:         {cfg.get('keep_tmp', False)}")
    print(f"Output:           {Path(output_dir).resolve()}")

    # Step 1: classify directories (provisional)
    dirs = scan_directory(scans_dir)
    print(f"\nFound:")
    print(f"  {len(dirs['mprage_dirs'])} MPRAGE series")
    print(f"  {len(dirs['candidate_dwi'])} candidate DWI series (before filtering)")
    if dirs["ignored"]:
        print(f"  {len(dirs['ignored'])} ignored (non-DWI)")

    # Step 2: convert all candidate DICOMs
    print("\nConverting DICOMs for all candidate series...")
    conversions = convert_all_candidates(dirs["candidate_dwi"], output_dir)

    # Steps 3-6: validate bvec/bval, check b-values, finalise classification
    classified = classify_candidates(dirs["candidate_dwi"], conversions)

    # Steps 7: detect AP/PA pairs (across dwi_dirs and pending series)
    print("\nDetecting AP/PA pairs...")
    # Pass existing fwd/rpe dirs into match so they are preserved
    conversions["fwd_pe_dirs"] = classified["fwd_pe_dirs"]
    conversions["rpe_dirs"] = classified["rpe_dirs"]
    dwi_dirs, fwd_pe_dirs, rpe_dirs, rpe_all_map, tie_warnings = match_ap_pa_pairs(
        classified["dwi_dirs"],
        classified["pending_fwd"],
        classified["pending_rpe"],
        conversions,
    )

    # Step 8: build PE assignment map — one b0 PE image → all DWI series in same block
    print("Assigning PE correction images to DWI blocks...")
    pe_assignment_map = build_pe_assignment_map(
        dwi_dirs, fwd_pe_dirs, rpe_dirs, rpe_all_map
    )

    print(f"\nAfter classification:")
    print(f"  {len(dwi_dirs)} DWI series to process")
    print(f"  {len(fwd_pe_dirs)} forward PE images")
    print(f"  {len(rpe_dirs)} reverse PE images")
    print(f"  {len(rpe_all_map)} rpe_all pairs detected")
    if classified["skipped"]:
        print(f"  {len(classified['skipped'])} series skipped")

    if not dwi_dirs:
        raise ValueError("No processable DWI series found after filtering.")

    # Steps 9-12: plan each workflow
    print("\nPlanning workflows...")
    plans = []
    for dwi_dir in dwi_dirs:
        plan = plan_workflow(
            dwi_dir=dwi_dir,
            mprage_dirs=dirs["mprage_dirs"],
            fwd_pe_dirs=fwd_pe_dirs,
            rpe_dirs=rpe_dirs,
            rpe_all_map=rpe_all_map,
            pe_assignment_map=pe_assignment_map,
            conversions=conversions,
            cfg=cfg,
        )
        plans.append(plan)

    # Step 13: print plan and pause
    print_plan(plans, classified["skipped"])

    for plan in plans:
        wf, cache_dir, tmp_dir, keep_tmp = build_dwi_workflow(plan)
        with Submitter(worker="cf", cache_root=cache_dir) as sub:
            sub(wf, rerun=True)
        cleanup_tmp(sentinel="", tmp_dir=tmp_dir, keep_tmp=keep_tmp)

    print("\n=== All done ===")
    print(f"Outputs in: {Path(output_dir).resolve()}")


# =============================================================================
# CLI
# =============================================================================


def load_config(args) -> dict:
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    if args.scans_dir:
        cfg["scans_dir"] = args.scans_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if getattr(args, "processing_steps", None):
        cfg["processing_steps"] = [s.strip() for s in args.processing_steps.split(",") if s.strip()]
    if args.gradcheck:
        cfg["gradcheck"] = True
    if args.nocleanup:
        cfg["keep_tmp"] = True
    if args.readout_time is not None:
        cfg["readout_time"] = args.readout_time
    if args.eddy_options is not None:
        cfg["eddy_options"] = args.eddy_options
    if "scans_dir" not in cfg:
        raise ValueError("scans_dir must be provided via --scans-dir or config YAML")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="DWI processing and tensor metrics pipeline"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--scans-dir", type=str, help="Path to scans directory")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--processing-steps", type=str, default=None,
                        help="Comma-separated steps: dwidenoise,mrgibbs,dwifslpreproc")
    parser.add_argument("--gradcheck", action="store_true", default=None)
    parser.add_argument(
        "--nocleanup",
        action="store_true",
        default=False,
        help="Keep the tmp/ directory after processing (default: remove)",
    )
    parser.add_argument("--readout-time", type=float, default=None)
    parser.add_argument("--eddy-options", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
