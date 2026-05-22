"""
Shared loader for temperature-dependent reference values.

Maps phantom vial labels → formulation names (via phantom_config.json), then
looks up ADC, T1, or T2 values at each calibrated temperature from the
appropriate calibration spreadsheet:

  ADC       – DIFFUSION-O-3574 xlsx  (PVP solutions; D, T1, T2 columns)
  T1 / T2   – RELAXOMETRY xlsx       (MnCl2 solutions; T1, T2 columns)

Only vials whose formulation appears in the selected xlsx are included,
which prevents ADC lookups pulling in MnCl2 vials and vice versa.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_METRIC_KEY = {"ADC": "D_val", "T1": "T1_val", "T2": "T2_val"}
_METRIC_UNITS = {"ADC": "×10⁻³ mm²/s", "T1": "ms", "T2": "ms"}

_DEFAULT_TEMP = 20


# ---------------------------------------------------------------------------
# xlsx finders
# ---------------------------------------------------------------------------

def _find_diffusion_xlsx(template_data_dir: str) -> Optional[Path]:
    base = Path(template_data_dir)
    candidates = [
        p for p in base.glob("DIFFUSION-O-3574*.xlsx")
        if not p.name.startswith("~$")
    ]
    return candidates[0] if candidates else None


def _find_relaxometry_xlsx(template_data_dir: str) -> Optional[Path]:
    base = Path(template_data_dir)
    candidates = [
        p for p in base.glob("RELAXOMETRY*.xlsx")
        if not p.name.startswith("~$")
    ]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# xlsx parsers
# ---------------------------------------------------------------------------

def _parse_relaxometry_xlsx(path: Path) -> dict:
    """Parse the RELAXOMETRY xlsx into {formulation: [{temperature, T1_val, T2_val}]}.

    Expected column layout (0-indexed):
      0  Formulation Name
      1  Temperature (°C)
      3  T1 reported (ms)
      6  T2 reported (ms)
    """
    import openpyxl
    wb = openpyxl.load_workbook(str(path), data_only=True)
    ws = wb.active
    result: dict = {}
    for row in ws.iter_rows(min_row=3, values_only=True):  # skip 2 header rows
        form = row[0]
        temp_raw = row[1]
        t1_raw = row[3]
        t2_raw = row[6]
        if not isinstance(form, str) or not form.strip():
            continue
        if temp_raw is None:
            continue
        try:
            temp = int(temp_raw)
        except (ValueError, TypeError):
            continue
        rec: dict = {"temperature": temp}
        if t1_raw is not None:
            try:
                rec["T1_val"] = float(t1_raw)
            except (ValueError, TypeError):
                pass
        if t2_raw is not None:
            try:
                rec["T2_val"] = float(t2_raw)
            except (ValueError, TypeError):
                pass
        result.setdefault(form.strip(), []).append(rec)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_calibration_reference(
    template_data_dir: str,
    phantom: str,
    metric: str,
) -> Optional[dict]:
    """Load temperature-dependent reference values for a given phantom and metric.

    Parameters
    ----------
    template_data_dir:
        Path to the ``template_data/`` directory (contains ``phantom_config.json``
        and the calibration xlsx files).
    phantom:
        Phantom name, e.g. ``"SPIRIT"`` or ``"120E"``.
    metric:
        One of ``"ADC"``, ``"T1"``, ``"T2"``.

    Returns
    -------
    dict or None
        ``{
            "vials"         : list[str],
            "temperatures"  : list[int],
            "values_by_temp": {str(temp): {vial_upper: float}},
            "units"         : str,
            "default_temp"  : str,
        }``
        Returns ``None`` if the required xlsx or config is unavailable.
    """
    if metric not in _METRIC_KEY:
        return None

    base = Path(template_data_dir)
    config_path = base / "phantom_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as _f:
        phantom_config = json.load(_f)

    phantom_map: dict = phantom_config.get(phantom, phantom_config.get(phantom.upper(), {}))
    if not phantom_map:
        return None

    # Select the right xlsx and parser for the metric
    if metric == "ADC":
        xlsx_path = _find_diffusion_xlsx(template_data_dir)
        if xlsx_path is None:
            return None
        from phantomkit.plotting.calibration_plotter import parse_calibration_xlsx
        formulations = parse_calibration_xlsx(str(xlsx_path))
    else:
        xlsx_path = _find_relaxometry_xlsx(template_data_dir)
        if xlsx_path is None:
            return None
        formulations = _parse_relaxometry_xlsx(xlsx_path)

    val_key = _METRIC_KEY[metric]

    # Build {formulation: {temp_int: value}} — only for formulations in the xlsx
    form_by_temp: dict[str, dict[int, float]] = {}
    for form_name, records in formulations.items():
        for rec in records:
            val = rec.get(val_key)
            raw_temp = rec.get("temperature")
            if val is None or raw_temp is None:
                continue
            try:
                temp = int(raw_temp)
            except (ValueError, TypeError):
                continue
            form_by_temp.setdefault(form_name, {})[temp] = float(val)

    # Map phantom vials → formulation, restricted to formulations in this xlsx.
    # This prevents ADC lookups pulling in MnCl2 vials (and vice versa) when
    # phantom_config.json lists both PVP and MnCl2 entries for the same phantom.
    vial_to_form: dict[str, str] = {}
    vials_ordered: list[str] = []
    seen: set[str] = set()
    for formulation, vial_list in phantom_map.items():
        if formulation not in form_by_temp:
            continue
        for v in vial_list:
            vu = v.upper()
            vial_to_form[vu] = formulation
            if vu not in seen:
                vials_ordered.append(vu)
                seen.add(vu)

    if not vials_ordered:
        return None

    # Collect temperatures covered by the matched vials
    relevant_forms = set(vial_to_form.values())
    all_temps: set[int] = set()
    for form in relevant_forms:
        if form in form_by_temp:
            all_temps.update(form_by_temp[form].keys())

    temperatures = sorted(all_temps)
    if not temperatures:
        return None

    # Build values_by_temp: str(temp) → {vial_upper: value}
    values_by_temp: dict[str, dict[str, float]] = {}
    for temp in temperatures:
        entry: dict[str, float] = {}
        for vial in vials_ordered:
            form = vial_to_form.get(vial)
            if form and form in form_by_temp and temp in form_by_temp[form]:
                entry[vial] = form_by_temp[form][temp]
        if entry:
            values_by_temp[str(temp)] = entry

    default_temp = str(min(temperatures, key=lambda t: abs(t - _DEFAULT_TEMP)))

    return {
        "vials"         : vials_ordered,
        "temperatures"  : temperatures,
        "values_by_temp": values_by_temp,
        "units"         : _METRIC_UNITS[metric],
        "default_temp"  : default_temp,
    }
