"""Quality control and minimal preprocessing for RSNC waveform windows.

The shared functions validate three-component SAC windows and apply the
conditioning used by the labelling and fine-tuning stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from obspy import Stream, read

# ---------------------------------------------------------------------------
# Window layout expected from acquisition.
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ: float = 100.0        # native HH broadband rate, not resampled
DELTA_S: float = 1.0 / SAMPLING_RATE_HZ
WINDOW_SECONDS: int = 150              # [P - 30 s, P + 120 s]
NPTS: int = 15001                      # samples per component (verified)
TIME_BEFORE_P_S: int = 30              # seconds before the theoretical P
THEORETICAL_P_SAMPLE: int = int(round(TIME_BEFORE_P_S * SAMPLING_RATE_HZ))  # 3000

TAPER_MAX_PERCENTAGE: float = 0.05     # 5% cosine taper on each end
TAPER_TYPE: str = "cosine"

COMPONENT_ORDER: str = "ZNE"           # SeisBench data_format component_order
CHANNEL_BY_COMPONENT = {"Z": "HHZ", "N": "HHN", "E": "HHE"}

# Tolerances.
_SAMPLING_TOL_HZ: float = 1e-3         # accept 100.0 +/- 0.001 Hz
_START_TOL_S: float = DELTA_S          # components must start within one sample


@dataclass
class WindowResult:
    """Outcome of QC + preprocessing for a single event-station window."""

    trace_name: str
    station_code: str
    timestamp: str
    status: str = "discarded"                 # "ok" | "discarded"
    reason: str = ""                          # empty when status == "ok"
    n_components_found: int = 0
    channels_found: str = ""                  # e.g. "HHE,HHN,HHZ"
    window_start_utc: str = ""
    npts: int = 0
    sampling_rate_hz: float = 0.0
    # Populated only when status == "ok"; (3, NPTS) float32 in ZNE order.
    data: Optional[np.ndarray] = field(default=None, repr=False)

    def as_record(self) -> dict:
        """Manifest row (waveform payload excluded)."""
        return {
            "trace_name": self.trace_name,
            "station_code": self.station_code,
            "timestamp": self.timestamp,
            "window_start_utc": self.window_start_utc,
            "n_components_found": self.n_components_found,
            "channels_found": self.channels_found,
            "npts": self.npts,
            "sampling_rate_hz": self.sampling_rate_hz,
            "status": self.status,
            "reason": self.reason,
        }


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Walk up until the folder that contains both ``data/raw`` and ``src``."""
    start = (start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if (p / "data" / "raw").is_dir() and (p / "src").is_dir():
            return p
    return start


def parse_sac_name(path: Path) -> Optional[tuple[str, str, str]]:
    """Parse ``<STA>.<CHANNEL>.<YYYY.DDD.HH.MM.SS>.SAC``.

    Returns ``(station_code, channel, timestamp)`` or ``None`` if the name does
    not follow the acquisition convention.
    """
    parts = path.name.split(".")
    # STA, CHAN, YYYY, DDD, HH, MM, SS, SAC  -> 8 fields
    if len(parts) != 8 or parts[-1].upper() != "SAC":
        return None
    station_code, channel = parts[0], parts[1]
    timestamp = ".".join(parts[2:7])  # YYYY.DDD.HH.MM.SS
    return station_code, channel, timestamp


def group_station_windows(station_dir: Path) -> dict[str, dict[str, Path]]:
    """Group one station's SAC files as ``{timestamp: {channel: path}}``."""
    windows: dict[str, dict[str, Path]] = {}
    for path in station_dir.glob("*.SAC"):
        parsed = parse_sac_name(path)
        if parsed is None:
            continue
        _, channel, timestamp = parsed
        windows.setdefault(timestamp, {})[channel] = path
    return windows


def _component_of(channel: str) -> str:
    """Last character of the channel code ('HHZ' -> 'Z')."""
    return channel[-1].upper()


def preprocess_stream(st: Stream) -> Stream:
    """Demean, linearly detrend, and taper a QC-passed stream in place."""
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=TAPER_MAX_PERCENTAGE, type=TAPER_TYPE)
    return st


def qc_preprocess_window(
    files: dict[str, Path],
    station_code: str,
    timestamp: str,
    *,
    return_data: bool = True,
    apply_conditioning: bool = True,
) -> WindowResult:
    """Validate one three-component window and return its QC outcome.

    Parameters
    ----------
    files
        ``{channel: path}`` for a single event-station window.
    station_code, timestamp
        Identifiers used to build ``trace_name`` (``"<STA>.<timestamp>"``).
    return_data
        If ``True`` and the window passes, attach the ``(3, NPTS)`` float32
        array in ZNE order. Set to ``False`` for a QC-only pass (manifest build)
        to avoid holding waveforms in memory.
    apply_conditioning
        If ``True`` (default) apply demean + detrend + taper. Set to ``False``
        to obtain the QC-aligned array for before/after diagnostics.
    """
    trace_name = f"{station_code}.{timestamp}"
    res = WindowResult(
        trace_name=trace_name,
        station_code=station_code,
        timestamp=timestamp,
        n_components_found=len(files),
        channels_found=",".join(sorted(files)),
    )

    # --- 1. Require Z, N, and E components ---------------------------------
    comp_to_path: dict[str, Path] = {}
    for channel, path in files.items():
        comp_to_path[_component_of(channel)] = path
    if not {"Z", "N", "E"}.issubset(comp_to_path):
        res.reason = "missing_components"
        return res

    # --- Read only the three expected components --------------------------
    try:
        st = Stream()
        for comp in ("Z", "N", "E"):
            st += read(str(comp_to_path[comp]), format="SAC")
    except Exception as exc:  # unreadable / corrupt SAC
        res.reason = f"read_error:{type(exc).__name__}"
        return res

    # --- 2. Merge segmentation; a surviving gap is a discard --------------
    try:
        # method=1 interpolates overlaps; fill_value=None leaves real gaps
        # masked so they can be detected below.
        st.merge(method=1, fill_value=None)
    except Exception as exc:
        res.reason = f"merge_error:{type(exc).__name__}"
        return res
    if len(st) != 3:  # more than one non-mergeable segment per channel
        res.reason = "gap"
        return res
    if any(isinstance(tr.data, np.ma.MaskedArray) and tr.data.mask.any() for tr in st):
        res.reason = "gap"
        return res

    # --- Sampling rate ----------------------------------------------------
    if any(abs(tr.stats.sampling_rate - SAMPLING_RATE_HZ) > _SAMPLING_TOL_HZ for tr in st):
        res.sampling_rate_hz = float(st[0].stats.sampling_rate)
        res.reason = "sampling_mismatch"
        return res
    res.sampling_rate_hz = SAMPLING_RATE_HZ

    # --- Alignment: same start time within one sample ---------------------
    starts = [tr.stats.starttime for tr in st]
    if max(starts) - min(starts) > _START_TOL_S:
        res.reason = "start_misaligned"
        return res
    res.window_start_utc = str(min(starts))

    # --- Length: enforce exactly NPTS -------------------------------------
    # Preserve the window start so the theoretical P remains at its expected
    # sample. Discard short windows rather than padding them.
    npts_each = [tr.stats.npts for tr in st]
    res.npts = int(min(npts_each))
    if res.npts < NPTS:
        res.reason = "too_short"
        return res
    if max(npts_each) > NPTS:
        end = min(starts) + (NPTS - 1) * DELTA_S
        st.trim(min(starts), end, nearest_sample=True)
        if any(tr.stats.npts != NPTS for tr in st):
            res.reason = "length_mismatch"
            return res
        res.npts = NPTS

    # --- 3. Conditioning: demean + detrend + taper ------------------------
    if apply_conditioning:
        try:
            preprocess_stream(st)
        except Exception as exc:
            res.reason = f"preprocess_error:{type(exc).__name__}"
            return res

    # --- Assemble the (3, NPTS) array in strict ZNE order -----------------
    by_comp = {tr.stats.channel[-1].upper(): tr for tr in st}
    try:
        data = np.stack(
            [np.asarray(by_comp[c].data, dtype=np.float32)[:NPTS] for c in COMPONENT_ORDER]
        )
    except KeyError:
        res.reason = "component_order_error"
        return res
    if data.shape != (3, NPTS):
        res.reason = "shape_error"
        return res

    res.status = "ok"
    res.reason = ""
    if return_data:
        res.data = data
    return res


def read_window_sac_header(
    station_dir: Path,
    timestamp: str,
    station_code: Optional[str] = None,
) -> Optional[dict]:
    """Read event geometry from the vertical SAC header without waveform samples.

    Return ``{evla, evlo, evdp, dist, baz}`` or ``None`` if the file is unavailable.
    """
    station_code = station_code or station_dir.name
    zpath = station_dir / f"{station_code}.{CHANNEL_BY_COMPONENT['Z']}.{timestamp}.SAC"
    if not zpath.exists():
        return None
    try:
        tr = read(str(zpath), format="SAC", headonly=True)[0]
    except Exception:
        return None
    sac = tr.stats.sac
    return {k: (float(sac[k]) if sac.get(k) is not None else None)
            for k in ("evla", "evlo", "evdp", "dist", "baz")}


def load_window_array(
    station_dir: Path,
    timestamp: str,
    station_code: Optional[str] = None,
    *,
    preprocess: bool = True,
) -> Optional[np.ndarray]:
    """Load a QC-passed ``(3, NPTS)`` ZNE window, or return ``None``.

    Set ``preprocess=False`` to inspect the aligned waveform before conditioning.
    """
    station_code = station_code or station_dir.name
    files: dict[str, Path] = {}
    for comp, channel in CHANNEL_BY_COMPONENT.items():
        path = station_dir / f"{station_code}.{channel}.{timestamp}.SAC"
        if path.exists():
            files[channel] = path
    res = qc_preprocess_window(
        files, station_code, timestamp,
        return_data=True, apply_conditioning=preprocess,
    )
    return res.data if res.status == "ok" else None
