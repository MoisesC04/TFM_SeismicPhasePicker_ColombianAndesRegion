"""Evaluation utilities for stored phase-picking records.

The module derives metrics, statistical summaries, and figures from the raw
records produced during inference without re-running the models.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import training_utils as tu

SAMPLING_RATE = 100          # Hz -> 1 sample = 10 ms
PHASES = ("P", "S")

# Consistent colors for baseline and fine-tuned results.
COLOR_BASE = "#E8590C"
COLOR_FT = "#4C6EF5"
COLOR_GRID = "0.85"


# ===========================================================================
# RECORD UTILITIES
# ===========================================================================
def enrich_records(recs, metadata, cols=("trace_synthetic", "source_depth_km",
                                         "trace_name", "split")):
    """Add metadata fields to inference records by position, in place.

    Require one metadata row per record to preserve the collection order.
    """
    if len(recs) != len(metadata):
        raise ValueError(
            f"Misaligned: {len(recs)} records vs {len(metadata)} metadata rows. "
            "The records must come from the same subset collect() was run on."
        )
    present = [c for c in cols if c in metadata.columns]
    missing = [c for c in cols if c not in metadata.columns]
    if missing:
        print(f"  warning: columns absent from the metadata, skipped: {missing}")
    if not present:
        return recs
    block = metadata[present].reset_index(drop=True)
    for i, rec in enumerate(recs):
        row = block.iloc[i]
        for c in present:
            rec[c] = row[c]
    return recs


# ---------------------------------------------------------------------------
# Resumable inference
# ---------------------------------------------------------------------------
def _collect_range(model, gen, md, lo, hi, decode_ps, *, sampling_rate,
                   peak_floor, peak_dist, batch_size, device, extra_cols):
    """Collect inference records for the half-open index range ``[lo, hi)``."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from scipy.signal import find_peaks

    loader = DataLoader(Subset(gen, list(range(lo, hi))), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    for _m in (model if isinstance(model, (list, tuple)) else [model]):
        _m.eval()

    recs, idx = [], lo
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            P, S = decode_ps(model, x)
            y = batch["y"].numpy()
            for i in range(x.shape[0]):
                row = md.iloc[idx]
                idx += 1
                rec = {"station": str(row.get("station_code", "?")),
                       "category": row.get("trace_category", "?"),
                       "magnitude": float(row.get("source_magnitude", np.nan)),
                       "distance": float(row.get("path_ep_distance_km", np.nan)),
                       "snr_p": float(row.get("trace_snr_p_db", np.nan)),
                       "snr_s": float(row.get("trace_snr_s_db", np.nan)),
                       "true": {}, "peaks": {}}
                for c in extra_cols:
                    if c in md.columns:
                        rec[c] = row[c]
                for ph, prob in (("P", P[i]), ("S", S[i])):
                    ch = 0 if ph == "P" else 1
                    rec["true"][ph] = (int(y[i, ch].argmax())
                                       if y[i, ch].max() > 0.5 else None)
                    pk, props = find_peaks(prob, height=peak_floor, distance=peak_dist)
                    rec["peaks"][ph] = list(zip(pk.tolist(),
                                                props["peak_heights"].tolist()))
                recs.append(rec)
    return recs


def collect_resumable(model, subset, decode_ps, *, part_dir, chunk=2000,
                      in_samples, sigma, sampling_rate=SAMPLING_RATE,
                      peak_floor=0.05, peak_dist=50, batch_size=128, device=None,
                      norm_type="peak",
                      extra_cols=("trace_synthetic", "source_depth_km",
                                  "trace_name", "split"),
                      verbose=True):
    """Collect inference in resumable chunks and return records in dataset order.

    Completed chunks are reused only when the saved parameter manifest matches
    the current call. ``extra_cols`` copies selected metadata into each record.
    """
    device = device or tu.DEVICE
    part_dir = Path(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    n = len(subset)
    md = subset.metadata.reset_index(drop=True)
    gen = tu.build_eval_generator(subset, in_samples, sigma, norm_type=norm_type)

    fingerprint = {"n": int(n), "in_samples": int(in_samples), "sigma": int(sigma),
                   "norm_type": norm_type, "chunk": int(chunk),
                   "peak_floor": float(peak_floor), "peak_dist": int(peak_dist)}
    man = part_dir / "manifest.json"
    if man.exists():
        stored = json.loads(man.read_text(encoding="utf-8"))
        if stored != fingerprint:
            raise RuntimeError(
                "The manifest of the stored chunks does not match this run.\n"
                f"  stored:  {stored}\n  current: {fingerprint}\n"
                f"Delete {part_dir} and start over, or fix the configuration."
            )
    else:
        man.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    bounds = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    already = sum(1 for k in range(len(bounds))
                  if (part_dir / f"chunk_{k:04d}.pkl").exists())
    if verbose:
        print(f"  {n} windows in {len(bounds)} chunks of {chunk} "
              f"| already stored: {already}")

    t0 = time.time()
    ran = 0
    for k, (lo, hi) in enumerate(bounds):
        target = part_dir / f"chunk_{k:04d}.pkl"
        if target.exists():
            continue
        recs = _collect_range(model, gen, md, lo, hi, decode_ps,
                              sampling_rate=sampling_rate, peak_floor=peak_floor,
                              peak_dist=peak_dist, batch_size=batch_size,
                              device=device, extra_cols=extra_cols)
        tmp = target.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(recs, f)
        os.replace(tmp, target)              # replace only after writing the chunk

        ran += 1
        if verbose:
            elapsed = time.time() - t0
            remaining = len(bounds) - already - ran
            eta = elapsed / ran * remaining
            print(f"    chunk {k + 1}/{len(bounds)}  [{lo}:{hi}]  "
                  f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                  flush=True)

    out = []
    for k in range(len(bounds)):
        with open(part_dir / f"chunk_{k:04d}.pkl", "rb") as f:
            out.extend(pickle.load(f))
    if len(out) != n:
        raise RuntimeError(f"Expected {n} records, assembled {len(out)}. "
                           f"Delete {part_dir} and run again.")
    if verbose:
        print(f"  done: {len(out)} windows")
    return out


def clear_parts(part_dir, verbose=True):
    """Delete the intermediate chunks of a run that is already assembled and saved."""
    part_dir = Path(part_dir)
    if not part_dir.exists():
        return
    for p in sorted(part_dir.iterdir()):
        p.unlink()
    part_dir.rmdir()
    if verbose:
        print(f"  chunks deleted: {part_dir}")


def save_records(tag, arch, rec_base, rec_ft, config, results_dir=None):
    """Save baseline and fine-tuned records without deriving metrics."""
    results_dir = results_dir or tu.RESULTS_DIR
    path = results_dir / f"records_{tag}.pkl"
    payload = {"baseline": rec_base, "finetuned": rec_ft,
               "config": {**config, "arch": arch}}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved: {path}  ({len(rec_base)} windows x 2 models)")
    return path


def load_all(tags):
    """Load several `records_<tag>.pkl` files and return them as {arch: payload}."""
    out = {}
    for tag in tags:
        payload = tu.load_records(tag)
        arch = payload.get("config", {}).get("arch", tag)
        out[arch] = payload
        n = len(payload["baseline"])
        print(f"  {arch:14s} tag={tag:18s} {n} windows")
    return out


# ===========================================================================
# PART 1 — PICKING CORE
# ===========================================================================
@dataclass
class PhaseResult:
    """Result for one phase (P or S) over a set of records."""
    phase: str
    tp: int
    fp: int
    fn: int
    n_ignored: int              # earthquake windows without a label for this phase
    residuals_ms: np.ndarray    # (pred - true) in ms, over true positives only
    hits: list                  # [(record_index, hit_bool)] for the paired tests

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    def fbeta(self, beta=1.0):
        p, r = self.precision, self.recall
        if not (p == p and r == r) or (p + r) == 0:
            return float("nan")
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)

    @property
    def f1(self):
        return self.fbeta(1.0)

    @property
    def f2(self):
        return self.fbeta(2.0)


def _peaks_above(rec, phase, thr):
    return [s for (s, p) in rec["peaks"][phase] if p >= thr]


def evaluate(recs, thr=0.30, tol_s=0.50, sampling_rate=SAMPLING_RATE,
             unlabeled="negative"):
    """Score predicted P/S peaks against pseudo-arrival references.

    A hit is the closest same-phase peak above ``thr`` within ``tol_s``. Other
    peaks count as false positives. For earthquake windows without a reference
    for a phase, ``unlabeled`` counts peaks as negative or ignores that phase;
    noise-window peaks always count as false positives.
    """
    if unlabeled not in ("negative", "ignore"):
        raise ValueError("unlabeled must be 'negative' or 'ignore'")

    out = {}
    for ph in PHASES:
        tp = fp = fn = n_ign = 0
        res, hits = [], []
        for i, rec in enumerate(recs):
            preds = _peaks_above(rec, ph, thr)
            true = rec["true"][ph]

            if true is None:
                is_noise = str(rec.get("category", "")).lower() == "noise"
                if is_noise or unlabeled == "negative":
                    fp += len(preds)
                else:
                    n_ign += 1
                continue

            near = sorted((abs(s - true), s) for s in preds
                          if abs(s - true) / sampling_rate <= tol_s)
            if near:
                tp += 1
                res.append((near[0][1] - true) / sampling_rate * 1000.0)
                fp += len(preds) - 1
                hits.append((i, True))
            else:
                fn += 1
                fp += len(preds)
                hits.append((i, False))

        out[ph] = PhaseResult(ph, tp, fp, fn, n_ign,
                              np.asarray(res, dtype=float), hits)
    return out


def timing_stats(residuals_ms):
    """Summarize signed and absolute pick residuals in milliseconds."""
    a = np.asarray(residuals_ms, dtype=float)
    if a.size == 0:
        keys = ["n", "mae_ms", "medae_ms", "rmse_ms", "bias_ms", "std_ms", "mad_ms",
                "p90_ms", "p95_ms", "pct_10ms", "pct_20ms", "pct_50ms", "pct_100ms"]
        return {k: (0 if k == "n" else float("nan")) for k in keys}
    absa = np.abs(a)
    med = float(np.median(a))
    return {
        "n": int(a.size),
        "mae_ms": float(np.mean(absa)),
        "medae_ms": float(np.median(absa)),
        "rmse_ms": float(np.sqrt(np.mean(a ** 2))),
        "bias_ms": med,                                   # signed median
        "std_ms": float(np.std(a)),
        "mad_ms": float(np.median(np.abs(a - med))),      # robust to tails
        "p90_ms": float(np.percentile(absa, 90)),
        "p95_ms": float(np.percentile(absa, 95)),
        "pct_10ms": float(np.mean(absa <= 10) * 100),
        "pct_20ms": float(np.mean(absa <= 20) * 100),
        "pct_50ms": float(np.mean(absa <= 50) * 100),
        "pct_100ms": float(np.mean(absa <= 100) * 100),
    }


#: Metrics shown in the main comparison table, in order.
CORE_METRICS = ["recall", "precision", "f2", "mae_ms", "medae_ms", "bias_ms"]

#: Direction of improvement: +1 if higher is better, -1 if lower is better.
_BETTER = {"recall": +1, "precision": +1, "f2": +1, "f1": +1,
           "mae_ms": -1, "medae_ms": -1, "rmse_ms": -1, "p90_ms": -1,
           "bias_ms": 0}          # 0 -> the absolute value is compared

#: (label used in the tables, key inside the records payload)
MODELS = (("baseline", "baseline"), ("finetuned", "finetuned"))


def _row(arch, model, ph, r: PhaseResult):
    d = {"architecture": arch, "model": model, "phase": ph,
         "TP": r.tp, "FP": r.fp, "FN": r.fn, "unlabeled": r.n_ignored,
         "recall": r.recall, "precision": r.precision,
         "f1": r.f1, "f2": r.f2}
    d.update(timing_stats(r.residuals_ms))
    return d


def summary_table(payloads, thr=0.30, tol_s=0.50, unlabeled="negative",
                  metrics=None):
    """Compare baseline and fine-tuned records across architectures.

    Return detailed per-phase metrics and their signed percentage changes.
    """
    metrics = metrics or CORE_METRICS
    rows = []
    for arch, payload in payloads.items():
        for model, key in MODELS:
            res = evaluate(payload[key], thr, tol_s, unlabeled=unlabeled)
            for ph in PHASES:
                rows.append(_row(arch, model, ph, res[ph]))
    detail = pd.DataFrame(rows)

    imp = []
    for (arch, ph), g in detail.groupby(["architecture", "phase"], sort=False):
        b = g[g.model == "baseline"].iloc[0]
        f = g[g.model == "finetuned"].iloc[0]
        rec = {"architecture": arch, "phase": ph}
        for m in metrics:
            vb, vf = float(b[m]), float(f[m])
            rec[f"{m}_baseline"] = vb
            rec[f"{m}_finetuned"] = vf
            rec[f"{m}_improvement_%"] = _improvement_pct(m, vb, vf)
        imp.append(rec)
    improvement = pd.DataFrame(imp)
    return detail, improvement


def _improvement_pct(metric, v_base, v_ft):
    """% improvement with the correct sign for the metric."""
    sign = _BETTER.get(metric, +1)
    if sign == 0:                       # bias: improving means moving towards zero
        v_base, v_ft, sign = abs(v_base), abs(v_ft), -1
    if not (v_base == v_base and v_ft == v_ft) or v_base == 0:
        return float("nan")
    return sign * (v_ft - v_base) / abs(v_base) * 100.0


# ---------------------------------------------------------------------------
# Pick-level 3x3 confusion matrix
# ---------------------------------------------------------------------------
CONF_ROWS = ("P real", "S real", "Ruido")
CONF_COLS = ("P predicha", "S predicha", "No detectada")


def confusion_pick_level(recs, thr=0.30, tol_s=0.50, sampling_rate=SAMPLING_RATE):
    """Build a 3x3 matrix of reference P, reference S, and noise outcomes.

    Each pseudo-arrival contributes one count. Noise contributes one count per
    window, assigned to the highest-probability predicted phase when present.
    """
    M = np.zeros((3, 3), dtype=int)
    tol = tol_s * sampling_rate

    for rec in recs:
        is_noise = str(rec.get("category", "")).lower() == "noise"
        if is_noise:
            best = {}
            for ph in PHASES:
                above = [(p, s) for (s, p) in rec["peaks"][ph] if p >= thr]
                if above:
                    best[ph] = max(above)[0]
            if not best:
                M[2, 2] += 1
            else:
                ph = max(best, key=best.get)
                M[2, 0 if ph == "P" else 1] += 1
            continue

        for r_i, ph in enumerate(PHASES):
            true = rec["true"][ph]
            if true is None:
                continue
            other = "S" if ph == "P" else "P"
            same_ok = any(abs(s - true) <= tol for s in _peaks_above(rec, ph, thr))
            if same_ok:
                M[r_i, r_i] += 1
                continue
            other_ok = any(abs(s - true) <= tol for s in _peaks_above(rec, other, thr))
            if other_ok:
                M[r_i, 0 if other == "P" else 1] += 1
            else:
                M[r_i, 2] += 1

    return pd.DataFrame(M, index=list(CONF_ROWS), columns=list(CONF_COLS))


def confusion_derived(conf: pd.DataFrame):
    """Balanced accuracy, MCC and phase-confusion rate from the 3x3 matrix."""
    M = conf.values.astype(float)
    per_row = np.divide(np.diag(M), M.sum(axis=1),
                        out=np.full(3, np.nan), where=M.sum(axis=1) > 0)
    bal_acc = float(np.nanmean(per_row))

    n = M.sum()
    if n == 0:
        return {"balanced_accuracy": np.nan, "mcc": np.nan,
                "phase_confusion_%": np.nan}
    p_k = M.sum(axis=0) / n          # predicted per class
    t_k = M.sum(axis=1) / n          # true per class
    c = np.trace(M) / n
    den = (np.sqrt(max(1 - (p_k ** 2).sum(), 0.0))
           * np.sqrt(max(1 - (t_k ** 2).sum(), 0.0)))
    mcc = float((c - (p_k * t_k).sum()) / den) if den > 0 else float("nan")

    phase_true = M[0, :2].sum() + M[1, :2].sum()
    phase_swap = M[0, 1] + M[1, 0]
    conf_ps = float(phase_swap / phase_true * 100) if phase_true else float("nan")
    return {"balanced_accuracy": bal_acc, "mcc": mcc, "phase_confusion_%": conf_ps}


# ---------------------------------------------------------------------------
# Part 1 figures
# ---------------------------------------------------------------------------
def plot_confusion(ax, conf: pd.DataFrame, title=""):
    """Plot row-normalized confusion values with counts in each cell."""
    M = conf.values.astype(float)
    row_sum = M.sum(axis=1, keepdims=True)
    frac = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)

    ax.imshow(frac, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(M.shape[1]), conf.columns, fontsize=8)
    ax.set_yticks(range(M.shape[0]), conf.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{int(M[i, j])}\n{frac[i, j]*100:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.55 else "0.15")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=10)
    return ax


def plot_residuals(ax, res_base_ms, res_ft_ms, phase="", bin_ms=10,
                   lim_ms=500, show_legend=True):
    """Plot baseline and fine-tuned pick residuals in milliseconds."""
    bins = np.arange(-lim_ms, lim_ms + bin_ms, bin_ms)
    if np.size(res_base_ms):
        ax.hist(res_base_ms, bins=bins, color=COLOR_BASE, alpha=0.55,
                label="baseline", edgecolor="none")
    if np.size(res_ft_ms):
        ax.hist(res_ft_ms, bins=bins, facecolor="none", edgecolor=COLOR_FT,
                hatch="///", linewidth=1.0, label="fine-tuned")
    ax.axvline(0, color="0.35", lw=1.2, zorder=0)
    ax.set_xlabel("residual (ms)   pred - true", fontsize=9)
    ax.set_ylabel("picks", fontsize=9)
    ax.set_title(f"Phase {phase}", fontsize=10)
    ax.grid(axis="y", color=COLOR_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=8)
    return ax


# ===========================================================================
# PART 2 — SUPPORTING ANALYSIS
# ===========================================================================
def pr_curve(recs, phase, thresholds=None, tol_s=0.50, unlabeled="negative"):
    """Threshold sweep -> (recall, precision, thresholds) for one phase."""
    thresholds = np.asarray(
        thresholds if thresholds is not None
        else np.round(np.arange(0.05, 0.96, 0.05), 2))
    R, P = [], []
    for t in thresholds:
        r = evaluate(recs, float(t), tol_s, unlabeled=unlabeled)[phase]
        R.append(r.recall)
        P.append(r.precision)
    return np.array(R), np.array(P), thresholds


def pr_auc(R, P):
    """Area under the precision-recall curve (trapezoid rule, sorted by recall)."""
    R, P = np.asarray(R, float), np.asarray(P, float)
    m = ~np.isnan(R) & ~np.isnan(P)
    R, P = R[m], P[m]
    if R.size < 2:
        return float("nan")
    o = np.argsort(R)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(P[o], R[o]))


def best_threshold(recs, phase, beta=2.0, thresholds=None, tol_s=0.50,
                   unlabeled="negative"):
    """Find the threshold that maximizes F-beta on the supplied records."""
    thresholds = (thresholds if thresholds is not None
                  else np.round(np.arange(0.05, 0.96, 0.05), 2))
    best, best_v = float("nan"), -1.0
    for t in thresholds:
        v = evaluate(recs, float(t), tol_s, unlabeled=unlabeled)[phase].fbeta(beta)
        if v == v and v > best_v:
            best, best_v = float(t), v
    return best, best_v


def bootstrap_ci(recs, stat, n_boot=1000, alpha=0.05, seed=42, **kw):
    """Estimate a confidence interval by resampling whole windows."""
    rng = np.random.default_rng(seed)
    n = len(recs)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat([recs[i] for i in idx], **kw)
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(np.nanmean(vals)), "ci_low": float(lo),
            "ci_high": float(hi), "n_boot": n_boot}


def mcnemar(recs_base, recs_ft, phase, thr=0.30, tol_s=0.50, unlabeled="negative"):
    """Compare paired detection outcomes using discordant counts and McNemar's test."""
    from scipy.stats import binomtest

    hb = dict(evaluate(recs_base, thr, tol_s, unlabeled=unlabeled)[phase].hits)
    hf = dict(evaluate(recs_ft, thr, tol_s, unlabeled=unlabeled)[phase].hits)
    common = sorted(set(hb) & set(hf))
    b = sum(1 for i in common if hb[i] and not hf[i])
    c = sum(1 for i in common if hf[i] and not hb[i])
    if b + c == 0:
        return {"only_baseline": b, "only_finetuned": c, "p": float("nan"),
                "n": len(common)}
    p = binomtest(c, b + c, 0.5, alternative="two-sided").pvalue
    return {"only_baseline": b, "only_finetuned": c, "p": float(p), "n": len(common)}


def wilcoxon_abs(recs_base, recs_ft, phase, thr=0.30, tol_s=0.50,
                 unlabeled="negative", sampling_rate=SAMPLING_RATE):
    """Compare absolute residuals for arrivals detected by both versions."""
    from scipy.stats import wilcoxon

    def per_index(recs):
        d = {}
        for i, rec in enumerate(recs):
            true = rec["true"][phase]
            if true is None:
                continue
            near = sorted((abs(s - true), s) for s in _peaks_above(rec, phase, thr)
                          if abs(s - true) / sampling_rate <= tol_s)
            if near:
                d[i] = abs(near[0][1] - true) / sampling_rate * 1000.0
        return d

    a, b = per_index(recs_base), per_index(recs_ft)
    common = sorted(set(a) & set(b))
    if len(common) < 10:
        return {"n": len(common), "p": float("nan"),
                "median_baseline_ms": float("nan"),
                "median_finetuned_ms": float("nan"),
                "median_diff_ms": float("nan")}
    va = np.array([a[i] for i in common])
    vb = np.array([b[i] for i in common])
    p = 1.0 if np.allclose(va, vb) else float(wilcoxon(va, vb).pvalue)
    return {"n": len(common), "p": p,
            "median_baseline_ms": float(np.median(va)),
            "median_finetuned_ms": float(np.median(vb)),
            "median_diff_ms": float(np.median(vb - va))}


# Distance and magnitude intervals used by the stratified summaries.
BINS_DISTANCE = [0, 50, 100, 150, 200, 250, 300, 350]
BINS_MAGNITUDE = [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 10.0]


def recall_by_bin(recs, phase, key, edges, thr=0.30, tol_s=0.50,
                  unlabeled="negative"):
    """Recall per bin of `key` ('distance', 'magnitude', 'snr_p', ...)."""
    res = evaluate(recs, thr, tol_s, unlabeled=unlabeled)[phase]
    hits = np.array([h for _, h in res.hits], dtype=float)
    vals = np.array([recs[i].get(key, np.nan) for i, _ in res.hits], dtype=float)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (vals >= lo) & (vals < hi) & ~np.isnan(vals)
        rows.append({"bin": f"{lo:g}-{hi:g}", "n": int(m.sum()),
                     "recall": float(hits[m].mean()) if m.sum() else np.nan})
    return pd.DataFrame(rows)


def noise_false_alarms(recs, thr=0.30):
    """Count noise-window false alarms, separating real and synthetic samples.

    If ``trace_synthetic`` is absent, return the combined rate only.
    """
    noise = [r for r in recs if str(r.get("category", "")).lower() == "noise"]
    if not noise:
        return pd.DataFrame()

    has_flag = "trace_synthetic" in noise[0]
    groups = {"all": noise}
    if has_flag:
        # trace_synthetic is True for synthetic, NaN/False for real (dataset convention)
        groups = {
            "real": [r for r in noise if r.get("trace_synthetic") != True],
            "synthetic": [r for r in noise if r.get("trace_synthetic") == True],
            "all": noise,
        }
    else:
        print("  warning: 'trace_synthetic' missing. Use enrich_records() to split it.")

    rows = []
    for name, g in groups.items():
        n = len(g)
        if n == 0:
            continue
        fp_p = sum(any(p >= thr for _, p in r["peaks"]["P"]) for r in g)
        fp_s = sum(any(p >= thr for _, p in r["peaks"]["S"]) for r in g)
        fp_any = sum(any(p >= thr for _, p in r["peaks"]["P"])
                     or any(p >= thr for _, p in r["peaks"]["S"]) for r in g)
        rows.append({"noise_type": name, "windows": n,
                     "FP_P_%": 100 * fp_p / n, "FP_S_%": 100 * fp_s / n,
                     "FP_any_%": 100 * fp_any / n})
    return pd.DataFrame(rows)
