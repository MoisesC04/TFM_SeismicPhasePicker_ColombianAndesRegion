"""Consensus pseudo-labelling utilities for seismic waveform windows.

The module loads pretrained pickers, combines phase picks, applies quality
checks, and returns labels for dataset construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from obspy import Stream, Trace, UTCDateTime

# ---------------------------------------------------------------------------
# Constants shared with the preprocessing stage.
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ: float = 100.0
DELTA_S: float = 1.0 / SAMPLING_RATE_HZ
NPTS: int = 15001
COMPONENT_ORDER: str = "ZNE"
CHANNEL_BY_COMPONENT = {"Z": "HHZ", "N": "HHN", "E": "HHE"}

# A neutral, fixed reference start so pick times map linearly to samples.
_REF_START = UTCDateTime(0)

# ---------------------------------------------------------------------------
# Consensus hyper-parameters (overridable from the notebook).
# ---------------------------------------------------------------------------
#: Minimum peak probability for a pick to enter the phase-wise vote.
DEFAULT_THRESHOLDS = {"P": 0.30, "S": 0.15}
#: Two picks of the same phase agree if they fall within this many samples.
TOL_SAMPLES: int = 50          # 0.5 s at 100 Hz (legacy coupled rule)
#: Minimum number of models that must agree on a phase (legacy coupled rule).
MIN_AGREEMENT: int = 2
#: Maximum probability among retained picks for a noise candidate.
NOISE_MAX_PROB: float = 0.15

# --- Independent P and S consensus used by label_window_decoupled ----------
#: P vote count, probability threshold, and temporal tolerance.
DEFAULT_MIN_P: int = 3
DEFAULT_THR_P: float = 0.30
DEFAULT_TOL_P: int = 20         # 0.2 s at 100 Hz
#: S vote count, probability threshold, and temporal tolerance.
DEFAULT_MIN_S: int = 2
DEFAULT_THR_S: float = 0.15
DEFAULT_TOL_S: int = 30         # 0.3 s at 100 Hz
#: Per-phase RMS SNR floor in dB; lower-SNR picks are dropped individually.
DEFAULT_SNR_CUT_DB: float = 0.0

#: Default SeisBench weights; the loader reports unavailable names.
DEFAULT_MODEL_WEIGHTS = {
    "PhaseNet": "original",
    "EQTransformer": "original",
    # Use a combined EQCCT class if available, otherwise its P and S pickers.
    "EQCCT": "original",
    "EQCCT_P": "original",
    "EQCCT_S": "original",
    # GPD contributes a fourth architecture to the consensus.
    "GPD": "original",
}


# ===========================================================================
# Array <-> Stream helpers
# ===========================================================================
def array_to_stream(
    arr: np.ndarray,
    *,
    network: str = "CM",
    station: str = "XXXX",
    starttime: UTCDateTime = _REF_START,
) -> Stream:
    """Wrap a ``(3, NPTS)`` ZNE array in an ObsPy Stream for SeisBench.

    The absolute start time is fixed and arbitrary; only relative timing
    matters, so a pick at epoch+t seconds maps to sample ``round(t * fs)``.
    """
    if arr.shape != (3, arr.shape[1]):
        raise ValueError(f"expected (3, N) array, got {arr.shape}")
    st = Stream()
    for comp, row in zip(COMPONENT_ORDER, arr):
        tr = Trace(data=np.ascontiguousarray(row, dtype=np.float32))
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.channel = CHANNEL_BY_COMPONENT[comp]
        tr.stats.sampling_rate = SAMPLING_RATE_HZ
        tr.stats.starttime = starttime
        st += tr
    return st


def _peak_time_to_sample(peak_time: UTCDateTime, starttime: UTCDateTime) -> int:
    return int(round(float(peak_time - starttime) * SAMPLING_RATE_HZ))


# ===========================================================================
# Model loading
# ===========================================================================
class _EQCCTCombined:
    """Combine EQCCT's separate P and S pickers into one consensus voter."""

    def __init__(self, p_model, s_model):
        self.p_model = p_model
        self.s_model = s_model

    def classify(self, st, **kwargs):
        from types import SimpleNamespace

        p_out = self.p_model.classify(st, **kwargs)
        s_out = self.s_model.classify(st, **kwargs)
        picks = [pk for pk in p_out.picks if str(pk.phase).upper() == "P"]
        picks += [pk for pk in s_out.picks if str(pk.phase).upper() == "S"]
        return SimpleNamespace(picks=picks)


def load_models(
    weights: Optional[dict[str, str]] = None,
    *,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Load available pretrained architectures onto ``device``.

    Return ``{model_name: model}``; combine EQCCTP and EQCCTS when needed.
    Unavailable weights are reported with the available alternatives.
    """
    import torch
    import seisbench.models as sbm

    weights = {**DEFAULT_MODEL_WEIGHTS, **(weights or {})}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _from_pretrained(cls, name, weight):
        try:
            model = cls.from_pretrained(weight)
        except Exception as exc:  # unknown weight name / download failure
            available = []
            try:
                available = cls.list_pretrained()
            except Exception:
                pass
            print(f"[WARN] could not load {name}('{weight}'): {exc}")
            if available:
                print(f"       available weights for {name}: {available}")
            return None
        model.to(device)
        model.eval()
        return model

    models: dict = {}

    # Load architectures that return both P and S picks.
    single = [("PhaseNet", sbm.PhaseNet), ("EQTransformer", sbm.EQTransformer)]
    if getattr(sbm, "GPD", None) is not None and "GPD" in weights:
        single.append(("GPD", sbm.GPD))
    for name, cls in single:
        model = _from_pretrained(cls, name, weights[name])
        if model is not None:
            models[name] = model
            if verbose:
                print(f"[ok] {name} <- '{weights[name]}' on {device}")

    # EQCCT: use a combined class if present, otherwise assemble EQCCTP + EQCCTS.
    eqcct_cls = getattr(sbm, "EQCCT", None)
    if eqcct_cls is not None:
        model = _from_pretrained(eqcct_cls, "EQCCT", weights.get("EQCCT", "original"))
        if model is not None:
            models["EQCCT"] = model
            if verbose:
                print(f"[ok] EQCCT <- '{weights.get('EQCCT', 'original')}' on {device}")
    else:
        eqcctp_cls = getattr(sbm, "EQCCTP", None)
        eqccts_cls = getattr(sbm, "EQCCTS", None)
        if eqcctp_cls is None or eqccts_cls is None:
            print("[WARN] EQCCT (nor EQCCTP/EQCCTS) available in this SeisBench version; skipping.")
        else:
            p_model = _from_pretrained(eqcctp_cls, "EQCCTP", weights.get("EQCCT_P", "original"))
            s_model = _from_pretrained(eqccts_cls, "EQCCTS", weights.get("EQCCT_S", "original"))
            if p_model is not None and s_model is not None:
                models["EQCCT"] = _EQCCTCombined(p_model, s_model)
                if verbose:
                    print(f"[ok] EQCCT <- EQCCTP+EQCCTS ('original') on {device}")

    if len(models) < MIN_AGREEMENT:
        raise RuntimeError(
            f"Only {len(models)} model(s) loaded; need at least {MIN_AGREEMENT} "
            "for a consensus. Check the weight names printed above."
        )
    return models


# ===========================================================================
# Per-window inference
# ===========================================================================
@dataclass
class ModelPick:
    model: str
    phase: str          # "P" | "S"
    sample: int
    prob: float


def run_models_on_window(
    arr: np.ndarray,
    models: dict,
    *,
    thresholds: Optional[dict[str, float]] = None,
    station: str = "XXXX",
) -> list[ModelPick]:
    """Run every model on one window and return the picks above threshold.

    Picks are expressed as integer sample indices relative to the window start.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    st = array_to_stream(arr, station=station)

    # Request picks at the vote thresholds before applying the shared filter.
    classify_args = {"P_threshold": thresholds["P"], "S_threshold": thresholds["S"]}

    picks: list[ModelPick] = []
    for name, model in models.items():
        out = model.classify(st, **classify_args)
        for pk in out.picks:
            phase = str(pk.phase).upper()
            if phase not in thresholds:
                continue
            prob = float(pk.peak_value)
            if prob < thresholds[phase]:
                continue
            sample = _peak_time_to_sample(pk.peak_time, st[0].stats.starttime)
            if 0 <= sample < arr.shape[1]:
                picks.append(ModelPick(name, phase, sample, prob))
    return picks


# ===========================================================================
# Consensus
# ===========================================================================
@dataclass
class ConsensusPick:
    phase: str
    sample: int
    n_models: int
    models: tuple[str, ...]
    mean_prob: float


def consensus_from_picks(
    picks: list[ModelPick],
    *,
    tol_samples: int = TOL_SAMPLES,
    min_agreement: int = MIN_AGREEMENT,
) -> list[ConsensusPick]:
    """Cluster same-phase picks by consecutive sample gaps.

    Keep clusters with enough distinct models and return their probability-
    weighted sample positions.
    """
    out: list[ConsensusPick] = []
    for phase in ("P", "S"):
        ph = sorted((p for p in picks if p.phase == phase), key=lambda p: p.sample)
        if not ph:
            continue
        cluster: list[ModelPick] = [ph[0]]
        for pk in ph[1:]:
            if pk.sample - cluster[-1].sample <= tol_samples:
                cluster.append(pk)
            else:
                out.extend(_finalise_cluster(cluster, phase, min_agreement))
                cluster = [pk]
        out.extend(_finalise_cluster(cluster, phase, min_agreement))
    return out


def _finalise_cluster(
    cluster: list[ModelPick], phase: str, min_agreement: int
) -> list[ConsensusPick]:
    # Keep one vote per model (the most confident) before counting agreement.
    best_by_model: dict[str, ModelPick] = {}
    for pk in cluster:
        cur = best_by_model.get(pk.model)
        if cur is None or pk.prob > cur.prob:
            best_by_model[pk.model] = pk
    votes = list(best_by_model.values())
    if len(votes) < min_agreement:
        return []
    weights = np.array([v.prob for v in votes], dtype=float)
    samples = np.array([v.sample for v in votes], dtype=float)
    sample = int(round(float(np.average(samples, weights=weights))))
    return [ConsensusPick(
        phase=phase,
        sample=sample,
        n_models=len(votes),
        models=tuple(sorted(best_by_model)),
        mean_prob=float(weights.mean()),
    )]


# ===========================================================================
# SNR and classification of a window
# ===========================================================================
@dataclass
class WindowLabel:
    trace_name: str
    station_code: str
    timestamp: str
    category: str = "unlabelled"        # "earthquake" | "noise" | "unlabelled"
    p_sample: Optional[int] = None
    s_sample: Optional[int] = None
    p_n_models: int = 0
    s_n_models: int = 0
    p_prob: float = 0.0
    s_prob: float = 0.0
    snr_db: float = float("nan")        # alias of snr_p (kept for compatibility)
    snr_p: float = float("nan")         # RMS SNR (dB) around P on the vertical
    snr_s: float = float("nan")         # RMS SNR (dB) around S on the stronger horizontal
    all_picks: list = field(default_factory=list, repr=False)


def snr_db_at(
    arr: np.ndarray, sample: int, comp_index: int, *, win_s: float = 5.0
) -> float:
    """RMS-based SNR in dB on component ``comp_index`` around ``sample``.

    Noise = ``win_s`` seconds before the pick; signal = ``win_s`` seconds after.
    Returns NaN if there is not enough room on either side.
    """
    x = arr[comp_index]
    n = int(win_s * SAMPLING_RATE_HZ)
    if sample is None or sample - n < 0 or sample + n > x.shape[0]:
        return float("nan")
    rms_n = float(np.sqrt(np.mean(x[sample - n:sample] ** 2))) + 1e-12
    rms_s = float(np.sqrt(np.mean(x[sample:sample + n] ** 2))) + 1e-12
    return 20.0 * float(np.log10(rms_s / rms_n))


def snr_db_around(
    arr: np.ndarray, p_sample: int, *, noise_s: float = 5.0, signal_s: float = 5.0
) -> float:
    """Legacy P-wave SNR wrapper; ``noise_s`` sets both window lengths.

    ``signal_s`` is retained for compatibility but is not used here.
    """
    return snr_db_at(arr, p_sample, 0, win_s=noise_s)


def stronger_horizontal_index(arr: np.ndarray, s_sample: int, half: int = 100) -> int:
    """Index (1=N or 2=E) of the horizontal with the larger amplitude near S."""
    w = slice(max(0, s_sample - half), min(arr.shape[1], s_sample + half))
    return 1 if np.abs(arr[1][w]).max() >= np.abs(arr[2][w]).max() else 2


def synth_noise(
    arr: np.ndarray,
    n_out: int,
    *,
    pre_a: int = 200,
    pre_b: int = 2700,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a phase-randomized noise surrogate from a pre-P segment.

    Preserve each component's spectral amplitude and rescale to its original
    standard deviation. The caller sets the required output length.
    """
    if rng is None:
        rng = np.random.default_rng()
    seg = arr[:, pre_a:pre_b]
    res = np.zeros((3, n_out), dtype=np.float32)
    for i in range(3):
        s = seg[i] - seg[i].mean()
        amp = np.abs(np.fft.rfft(s))
        amp = np.interp(np.linspace(0, 1, n_out // 2 + 1),
                        np.linspace(0, 1, amp.size), amp)
        n = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, amp.size)), n=n_out)
        res[i] = (n / (n.std() + 1e-9) * (s.std() + 1e-9)).astype(np.float32)
    return np.ascontiguousarray(res)


def label_window(
    arr: np.ndarray,
    trace_name: str,
    station_code: str,
    timestamp: str,
    models: dict,
    *,
    thresholds: Optional[dict[str, float]] = None,
    tol_samples: int = TOL_SAMPLES,
    min_agreement: int = MIN_AGREEMENT,
    noise_max_prob: float = NOISE_MAX_PROB,
) -> WindowLabel:
    """Apply the legacy coupled P/S decision to one window.

    An earthquake needs consensus P and a later S from a P-voting model.
    Otherwise the window is noise only when all retained picks fall below
    ``noise_max_prob``; unresolved windows remain unlabelled.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    picks = run_models_on_window(arr, models, thresholds=thresholds, station=station_code[:4])
    label = WindowLabel(trace_name, station_code, timestamp, all_picks=picks)

    consensus = consensus_from_picks(
        picks, tol_samples=tol_samples, min_agreement=min_agreement
    )
    cp = {c.phase: c for c in consensus}

    if "P" in cp:
        # Require a later S from a model that voted for the consensus P.
        p_models = set(cp["P"].models)
        p_sample = cp["P"].sample
        s_from_p_models = [
            pk for pk in picks
            if pk.phase == "S" and pk.model in p_models and pk.sample > p_sample
        ]
        if s_from_p_models:
            best_s = max(s_from_p_models, key=lambda pk: pk.prob)
            near = [pk for pk in s_from_p_models
                    if abs(pk.sample - best_s.sample) <= tol_samples]
            label.category = "earthquake"
            label.p_sample = p_sample
            label.p_n_models = cp["P"].n_models
            label.p_prob = cp["P"].mean_prob
            label.s_sample = best_s.sample
            label.s_n_models = len({pk.model for pk in near})
            label.s_prob = float(np.mean([pk.prob for pk in near]))
            label.snr_db = snr_db_around(arr, label.p_sample)
            return label
        # A P vote alone does not meet this legacy event rule.
        label.category = "unlabelled"
        return label

    # Without consensus P, only low-probability retained picks permit noise.
    if all(p.prob < noise_max_prob for p in picks):
        label.category = "noise"
    else:
        label.category = "unlabelled"
    return label


def label_window_decoupled(
    arr: np.ndarray,
    trace_name: str,
    station_code: str,
    timestamp: str,
    models: dict,
    *,
    min_p: int = DEFAULT_MIN_P,
    thr_p: float = DEFAULT_THR_P,
    tol_p: int = DEFAULT_TOL_P,
    min_s: int = DEFAULT_MIN_S,
    thr_s: float = DEFAULT_THR_S,
    tol_s: int = DEFAULT_TOL_S,
    noise_max_prob: float = NOISE_MAX_PROB,
    snr_cut_db: float = DEFAULT_SNR_CUT_DB,
    apply_snr_qc: bool = True,
) -> WindowLabel:
    """Label a window using separate P and S votes and optional SNR checks.

    An event requires consensus P; an accepted S must follow it. Low-SNR picks
    are removed independently, and a window becomes unlabelled if neither phase
    remains. With no consensus P, low-probability retained picks permit noise.
    """
    picks = run_models_on_window(
        arr, models, thresholds={"P": thr_p, "S": thr_s}, station=station_code[:4]
    )
    label = WindowLabel(trace_name, station_code, timestamp, all_picks=picks)

    p_picks = [p for p in picks if p.phase == "P"]
    s_picks = [p for p in picks if p.phase == "S"]
    cp = consensus_from_picks(p_picks, tol_samples=tol_p, min_agreement=min_p)
    cs = consensus_from_picks(s_picks, tol_samples=tol_s, min_agreement=min_s)
    P = max(cp, key=lambda c: (c.n_models, c.mean_prob)) if cp else None
    S = max(cs, key=lambda c: (c.n_models, c.mean_prob)) if cs else None
    if P is not None and S is not None and S.sample <= P.sample:
        S = None  # S must follow P

    if P is None:
        if all(p.prob < noise_max_prob for p in picks):
            label.category = "noise"
        else:
            label.category = "unlabelled"
        return label

    # Consensus P (and maybe S) -> earthquake. Record picks and per-phase SNR.
    label.category = "earthquake"
    label.p_sample, label.p_n_models, label.p_prob = P.sample, P.n_models, P.mean_prob
    label.snr_p = snr_db_at(arr, P.sample, 0)
    label.snr_db = label.snr_p
    if S is not None:
        label.s_sample, label.s_n_models, label.s_prob = S.sample, S.n_models, S.mean_prob
        label.snr_s = snr_db_at(arr, S.sample, stronger_horizontal_index(arr, S.sample))

    if apply_snr_qc:
        if label.p_sample is not None and label.snr_p == label.snr_p and label.snr_p < snr_cut_db:
            label.p_sample, label.p_n_models, label.p_prob = None, 0, 0.0
        if label.s_sample is not None and label.snr_s == label.snr_s and label.snr_s < snr_cut_db:
            label.s_sample, label.s_n_models, label.s_prob = None, 0, 0.0
        if label.p_sample is None and label.s_sample is None:
            label.category = "unlabelled"
    return label
