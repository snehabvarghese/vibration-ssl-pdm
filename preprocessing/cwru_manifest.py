"""
CWRU Bearing Data Center -- file manifest.

WHAT THIS IS
------------
The CWRU dataset is distributed as a set of MATLAB `.mat` files, each named by
an opaque number (e.g. `105.mat`). The mapping from file number to
(fault type, fault diameter, motor load, sampling rate, sensor position) lives
only on the download web page, so we encode it here explicitly.

Encoding the mapping in code -- rather than relying on file names -- means the
labels used in this project are the *real* dataset labels, not invented ones.

Slice covered here: the widely used benchmark slice
    * Normal baseline (4 motor loads)
    * 12 kHz drive-end faults: inner race (IR), ball (B), outer race (OR)
      at 0.007", 0.014", 0.021" and 0.028" diameters,
      with outer-race faults further split by load-zone clock position
      (@3, @6, @12 o'clock).

Source: https://engineering.case.edu/bearingdatacenter/download-data-file
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

# Approximate shaft speed (rpm) reported by CWRU for each motor load (HP).
LOAD_TO_RPM: Dict[int, int] = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

BASE_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"


@dataclass(frozen=True)
class CWRURecord:
    """One CWRU `.mat` file and everything we know about it."""

    file_id: str              # e.g. "105"
    subset: str               # "normal" | "12k_drive_end"
    fault_type: str           # "Normal" | "IR" | "B" | "OR"
    fault_diameter_in: float  # 0.0 for normal, else 0.007 / 0.014 / 0.021 / 0.028
    orientation: Optional[str]  # outer-race clock position: "3", "6", "12"; else None
    load_hp: int              # 0..3
    sampling_rate: int        # Hz

    @property
    def rpm(self) -> int:
        return LOAD_TO_RPM[self.load_hp]

    @property
    def url(self) -> str:
        return BASE_URL.format(file_id=self.file_id)

    @property
    def filename(self) -> str:
        return f"{self.file_id}.mat"

    def label(self, mode: str = "fault_type_size") -> str:
        """Class label for this record.

        mode="fault_type"       -> "Normal" / "IR" / "B" / "OR"          (4 classes)
        mode="fault_type_size"  -> "Normal" / "IR007" / "OR021@6" / ...  (classic benchmark)
        """
        if self.fault_type == "Normal":
            return "Normal"
        if mode == "fault_type":
            return self.fault_type
        size = f"{int(round(self.fault_diameter_in * 1000)):03d}"
        base = f"{self.fault_type}{size}"
        if self.fault_type == "OR" and self.orientation is not None:
            base = f"{base}@{self.orientation}"
        return base


def _group(
    file_ids: List[str],
    subset: str,
    fault_type: str,
    diameter: float,
    orientation: Optional[str],
    sampling_rate: int,
) -> List[CWRURecord]:
    """Helper: four consecutive file ids correspond to motor loads 0,1,2,3."""
    return [
        CWRURecord(
            file_id=fid,
            subset=subset,
            fault_type=fault_type,
            fault_diameter_in=diameter,
            orientation=orientation,
            load_hp=load,
            sampling_rate=sampling_rate,
        )
        for load, fid in enumerate(file_ids)
    ]


# --------------------------------------------------------------------------
# The manifest itself
# --------------------------------------------------------------------------
_RECORDS: List[CWRURecord] = []

# Normal baseline. CWRU recorded these at 48 kHz, unlike the 12 kHz fault
# files, so the loader resamples them to the common target rate. (Verified
# empirically: 98.mat holds 483 903 samples = ~10 s at 48 kHz.)
_RECORDS += _group(["97", "98", "99", "100"], "normal", "Normal", 0.0, None, 48_000)

# 12 kHz drive-end faults
_RECORDS += _group(["105", "106", "107", "108"], "12k_drive_end", "IR", 0.007, None, 12_000)
_RECORDS += _group(["118", "119", "120", "121"], "12k_drive_end", "B", 0.007, None, 12_000)
_RECORDS += _group(["130", "131", "132", "133"], "12k_drive_end", "OR", 0.007, "6", 12_000)
_RECORDS += _group(["144", "145", "146", "147"], "12k_drive_end", "OR", 0.007, "3", 12_000)
_RECORDS += _group(["156", "158", "159", "160"], "12k_drive_end", "OR", 0.007, "12", 12_000)

_RECORDS += _group(["169", "170", "171", "172"], "12k_drive_end", "IR", 0.014, None, 12_000)
_RECORDS += _group(["185", "186", "187", "188"], "12k_drive_end", "B", 0.014, None, 12_000)
_RECORDS += _group(["197", "198", "199", "200"], "12k_drive_end", "OR", 0.014, "6", 12_000)

_RECORDS += _group(["209", "210", "211", "212"], "12k_drive_end", "IR", 0.021, None, 12_000)
_RECORDS += _group(["222", "223", "224", "225"], "12k_drive_end", "B", 0.021, None, 12_000)
_RECORDS += _group(["234", "235", "236", "237"], "12k_drive_end", "OR", 0.021, "6", 12_000)
_RECORDS += _group(["246", "247", "248", "249"], "12k_drive_end", "OR", 0.021, "3", 12_000)
_RECORDS += _group(["258", "259", "260", "261"], "12k_drive_end", "OR", 0.021, "12", 12_000)

_RECORDS += _group(["3001", "3002", "3003", "3004"], "12k_drive_end", "IR", 0.028, None, 12_000)
_RECORDS += _group(["3005", "3006", "3007", "3008"], "12k_drive_end", "B", 0.028, None, 12_000)


def all_records() -> List[CWRURecord]:
    """Every record in the manifest."""
    return list(_RECORDS)


def select_records(
    subsets: Optional[List[str]] = None,
    loads: Optional[List[int]] = None,
    diameters: Optional[List[float]] = None,
) -> List[CWRURecord]:
    """Filter the manifest. `None` means 'no filter on this field'."""
    out = all_records()
    if subsets is not None:
        out = [r for r in out if r.subset in subsets]
    if loads is not None:
        out = [r for r in out if r.load_hp in loads]
    if diameters is not None:
        out = [r for r in out if r.fault_type == "Normal" or r.fault_diameter_in in diameters]
    return out
