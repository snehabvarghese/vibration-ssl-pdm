"""
Dataset loading layer.

WHAT IT DOES
------------
Turns raw dataset files on disk into a uniform in-memory representation:

    LoadedSignal(signal: np.ndarray[float64], sampling_rate: int, label: str, ...)

Everything downstream (filtering, segmentation, the model) only ever sees a
`LoadedSignal`, never a `.mat` file. That is deliberate: in Phase 2 a live
ESP32/MQTT stream can produce `LoadedSignal` objects and the rest of the
pipeline keeps working unchanged (see `StreamingSource` at the bottom).

WHY A SEPARATE LOADER
---------------------
* CWRU `.mat` variable names are irregular -- the 0.028" files store their data
  under a *different* number than the file name (`3001.mat` -> `X056_DE_time`),
  so we resolve variables by suffix, not by name.
* The normal-baseline files are sampled at 48 kHz while the fault files are at
  12 kHz; the loader resamples everything to one common rate so that a fixed
  window length always means the same physical duration.
* Corrupt/missing files must not crash a 64-file run.

INPUTS  : a `CWRURecord` (from `cwru_manifest.py`) + the raw data directory
OUTPUTS : `LoadedSignal`
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol

import numpy as np

from config import CFG, DATA_RAW_DIR
from preprocessing.cwru_manifest import CWRURecord, select_records


class SignalLoadError(RuntimeError):
    """Raised when a record exists but cannot be parsed into a usable signal."""


@dataclass
class LoadedSignal:
    """A single raw vibration time series plus its metadata.

    Attributes
    ----------
    signal          1-D float array of acceleration samples
    sampling_rate   Hz, after any resampling done by the loader
    record_id       unique id of the source recording (used for leak-free splits)
    label           class label, e.g. "Normal", "IR007", "OR021@6"
    fault_type      coarse label: "Normal" / "IR" / "B" / "OR"
    load_hp         motor load in HP (operating condition)
    rpm             nominal shaft speed
    meta            anything else worth carrying around
    """

    signal: np.ndarray
    sampling_rate: int
    record_id: str
    label: str
    fault_type: str
    load_hp: int
    rpm: int
    meta: Dict[str, object]

    @property
    def duration_s(self) -> float:
        return len(self.signal) / float(self.sampling_rate)


# --------------------------------------------------------------------------
# CWRU .mat parsing
# --------------------------------------------------------------------------
_CHANNEL_SUFFIX = {"DE": "_DE_time", "FE": "_FE_time", "BA": "_BA_time"}


def _find_channel_key(mat: Dict[str, object], channel: str) -> Optional[str]:
    """Find the variable holding the requested accelerometer channel.

    CWRU keys look like `X105_DE_time`, but the numeric part is not always the
    file number, so we match on the suffix only.
    """
    suffix = _CHANNEL_SUFFIX[channel]
    candidates = [k for k in mat if isinstance(k, str) and k.endswith(suffix)]
    if not candidates:
        return None
    # Deterministic choice when a file (rarely) contains more than one match.
    return sorted(candidates)[0]


def _resample_to(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample by an integer ratio when possible, else by polyphase filtering."""
    if src_rate == dst_rate:
        return signal
    from math import gcd

    from scipy.signal import decimate, resample_poly

    if src_rate % dst_rate == 0:
        factor = src_rate // dst_rate
        # `decimate` applies an anti-aliasing filter before dropping samples.
        return np.asarray(decimate(signal, factor, ftype="fir", zero_phase=True))
    g = gcd(src_rate, dst_rate)
    return np.asarray(resample_poly(signal, dst_rate // g, src_rate // g))


def load_signal(
    record: CWRURecord,
    raw_dir: Path = DATA_RAW_DIR,
    channel: Optional[str] = None,
    target_rate: Optional[int] = None,
    label_mode: Optional[str] = None,
) -> LoadedSignal:
    """Load one CWRU record into a `LoadedSignal`.

    Parameters
    ----------
    record       manifest entry describing the file
    raw_dir      directory containing the `.mat` files
    channel      "DE" (drive end), "FE" (fan end) or "BA" (base); defaults to config
    target_rate  resample everything to this rate; defaults to config
    label_mode   "fault_type" or "fault_type_size"; defaults to config

    Raises
    ------
    FileNotFoundError  if the `.mat` file is missing
    SignalLoadError    if it is present but unusable (bad channel, empty, NaNs)
    """
    from scipy.io import loadmat

    channel = channel or CFG.dataset.channel
    target_rate = target_rate or CFG.dataset.sampling_rate
    label_mode = label_mode or CFG.dataset.label_mode

    path = Path(raw_dir) / record.filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m preprocessing.download_cwru` first"
        )

    try:
        mat = loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - scipy raises many different types
        raise SignalLoadError(f"could not read {path}: {exc}") from exc

    key = _find_channel_key(mat, channel)
    if key is None:
        available = sorted(k for k in mat if isinstance(k, str) and not k.startswith("__"))
        raise SignalLoadError(
            f"{path.name}: channel {channel!r} not present (available: {available})"
        )

    raw = np.asarray(mat[key], dtype=np.float64).reshape(-1)
    if raw.size == 0:
        raise SignalLoadError(f"{path.name}: channel {channel!r} is empty")

    # Handle missing/corrupt samples gracefully instead of poisoning the model
    # with NaNs: linearly interpolate short gaps, and refuse the record if the
    # signal is mostly invalid.
    finite = np.isfinite(raw)
    if not finite.all():
        bad_frac = 1.0 - finite.mean()
        if bad_frac > 0.01:
            raise SignalLoadError(f"{path.name}: {bad_frac:.1%} non-finite samples")
        warnings.warn(f"{path.name}: interpolating {int((~finite).sum())} non-finite samples")
        idx = np.arange(raw.size)
        raw = np.interp(idx, idx[finite], raw[finite])

    signal = _resample_to(raw, record.sampling_rate, target_rate)

    # Prefer the RPM stored in the file when present; fall back to the nominal
    # value from the manifest.
    rpm_key = [k for k in mat if isinstance(k, str) and k.endswith("RPM")]
    rpm = int(np.asarray(mat[rpm_key[0]]).reshape(-1)[0]) if rpm_key else record.rpm

    return LoadedSignal(
        signal=signal.astype(np.float64, copy=False),
        sampling_rate=target_rate,
        record_id=record.file_id,
        label=record.label(label_mode),
        fault_type=record.fault_type,
        load_hp=record.load_hp,
        rpm=rpm,
        meta={
            "source_file": path.name,
            "mat_variable": key,
            "native_sampling_rate": record.sampling_rate,
            "subset": record.subset,
            "fault_diameter_in": record.fault_diameter_in,
            "orientation": record.orientation,
            "channel": channel,
        },
    )


def load_dataset(
    raw_dir: Path = DATA_RAW_DIR,
    subsets: Optional[List[str]] = None,
    loads: Optional[List[int]] = None,
    skip_errors: bool = True,
) -> List[LoadedSignal]:
    """Load every selected record, skipping (and reporting) unreadable ones."""
    subsets = subsets if subsets is not None else CFG.dataset.subsets
    loads = loads if loads is not None else CFG.dataset.loads
    records = select_records(subsets=subsets, loads=loads)

    signals: List[LoadedSignal] = []
    for rec in records:
        try:
            signals.append(load_signal(rec, raw_dir=raw_dir))
        except (FileNotFoundError, SignalLoadError) as exc:
            if not skip_errors:
                raise
            warnings.warn(f"skipping record {rec.file_id}: {exc}")
    return signals


# --------------------------------------------------------------------------
# Phase 2 hook
# --------------------------------------------------------------------------
class SignalSource(Protocol):
    """The only interface the rest of the pipeline depends on.

    Phase 1 implementation: `FileSignalSource` (CWRU files on disk).
    Phase 2 implementation: an MQTT/serial reader that yields `LoadedSignal`
    chunks from an ESP32 + accelerometer. Nothing downstream needs to change.
    """

    def __iter__(self) -> Iterator[LoadedSignal]: ...


class FileSignalSource:
    """Iterates over the offline dataset as a `SignalSource`."""

    def __init__(self, raw_dir: Path = DATA_RAW_DIR, **kwargs) -> None:
        self.raw_dir = raw_dir
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[LoadedSignal]:
        yield from load_dataset(raw_dir=self.raw_dir, **self.kwargs)
