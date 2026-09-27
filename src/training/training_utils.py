"""Shared training and inference utilities for the local SeisBench workflow.

This module centralizes runtime setup, dataset loading, fine-tuning helpers,
and common result paths used by the architecture-specific notebooks.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Register Windows DLL directories before importing PyTorch.
# ---------------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _register_torch_dll_dirs():
    dirs = [os.path.join(sys.prefix, "Library", "bin"),
            os.path.join(sys.prefix, "Library", "mingw-w64", "bin"),
            os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
            sys.prefix]
    for p in list(sys.path):
        if p and os.path.isdir(os.path.join(p, "torch", "lib")):
            dirs.append(os.path.join(p, "torch", "lib"))
            env_root = os.path.dirname(os.path.dirname(p))
            dirs += [os.path.join(env_root, "Library", "bin"),
                     os.path.join(env_root, "Library", "mingw-w64", "bin"), env_root]
    seen = set()
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass


_register_torch_dll_dirs()

import json
import pickle
from pathlib import Path

import torch                     # noqa: E402  (torch first, after the DLL fix)
from torch.utils.data import DataLoader  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import seisbench.data as sbd     # noqa: E402
import seisbench.generate as sbg  # noqa: E402
import logging                   # noqa: E402
logging.getLogger("seisbench").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO        = Path(r"M:\Github\TFM_SeismicPhasePicker_ColombianAndesRegion")
DATASET_DIR = REPO / "data" / "processed" / "dataset"
MODELS_DIR  = REPO / "data" / "processed" / "models"
RESULTS_DIR = REPO / "data" / "processed" / "eval_results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Arrival columns -> phase (common to all architectures)
PHASE_DICT = {"trace_p_arrival_sample": "P", "trace_s_arrival_sample": "S"}

# ---------------------------------------------------------------------------
# Shared, reproducible subsampling for the training and collection notebooks.
# Values below 1.0 sample within each split and trace category.
# ---------------------------------------------------------------------------
SUBSAMPLE_FRACTION = 1.0        # 1.0 = full dataset (final run). 1/6 = quick test.
SUBSAMPLE_SEED     = 42


def banner():
    import seisbench
    print("seisbench", seisbench.__version__, "| torch", torch.__version__, "| device:", DEVICE)


# ===========================================================================
# DATASET
# ===========================================================================
def load_dataset(stations=None, sampling_rate=100, fraction=None, seed=None, verbose=True):
    """Load local waveforms with optional station filtering and stratified sampling.

    ``stations=None`` retains every station. ``fraction=None`` uses the shared
    ``SUBSAMPLE_FRACTION`` setting.
    """
    data = sbd.WaveformDataset(str(DATASET_DIR), sampling_rate=sampling_rate)
    n_all = len(data)
    if stations:
        data.filter(data.metadata["station_code"].isin(stations), inplace=True)
        if verbose:
            print(f"Station filter ACTIVE {stations}  ->  {len(data)}/{n_all} traces")
    elif verbose:
        print(f"No filter (all stations): {n_all} traces")

    frac = SUBSAMPLE_FRACTION if fraction is None else fraction
    sd = SUBSAMPLE_SEED if seed is None else seed
    if frac and frac < 1.0:
        n_before = len(data)
        md = data.metadata.reset_index(drop=True)
        rng = np.random.default_rng(sd)
        mask = np.zeros(len(md), dtype=bool)
        gcols = [c for c in ("split", "trace_category") if c in md.columns]
        groups = md.groupby(gcols) if gcols else [(None, md)]
        for _, g in groups:
            pos = g.index.values
            k = max(1, int(round(len(pos) * frac)))
            mask[rng.choice(pos, size=k, replace=False)] = True
        data.filter(mask, inplace=True)
        if verbose:
            print(f"[SUBSAMPLE {frac:g}] ACTIVE -> {len(data)}/{n_before} traces "
                  f"(stratified by split/category, seed {sd}). "
                  f"Set SUBSAMPLE_FRACTION=1.0 for the full dataset.")

    if verbose:
        print("stations:", sorted(data.metadata["station_code"].unique().tolist()))
    return data


def split_summary(sub, name):
    """Print the size and earthquake/noise breakdown of one split (train/dev/test)."""
    vc = sub.metadata["trace_category"].value_counts().to_dict()
    print(f"  {name:5s}: {len(sub):5d}  {vc}")


def make_tag(arch, stations):
    """Build the `<arch>_<all|Nsta>` tag used to name saved eval_results files."""
    return f"{arch.lower()}_" + ("all" if not stations else f"{len(stations)}sta")


# ===========================================================================
# FINE-TUNING (generic loop + early stopping; the loss is passed as an arg)
# ===========================================================================
def fit(model, train_loader, dev_loader, loss_fn, *, epochs, patience,
        min_delta, lr, device=DEVICE, verbose=True):
    """Fine-tune with an architecture-specific loss and restore best dev weights."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(loader, train):
        model.train(train)
        torch.set_grad_enabled(train)
        tot, n = 0.0, 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["X"])
            loss = loss_fn(out, batch)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            bs = batch["X"].shape[0]
            tot += loss.item() * bs; n += bs
        torch.set_grad_enabled(True)
        return tot / max(n, 1)

    hist = {"train": [], "dev": []}
    best, best_ep, best_state, no_improve = float("inf"), 0, None, 0
    stop_reason = f"reached the {epochs} epochs"
    for ep in range(1, epochs + 1):
        tr = run(train_loader, True)
        dv = run(dev_loader, False)
        hist["train"].append(tr); hist["dev"].append(dv)
        if dv < best - min_delta:
            best, best_ep = dv, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0; flag = "  *best dev*"
        else:
            no_improve += 1; flag = f"  (no improvement {no_improve}/{patience})"
        if verbose:
            print(f"epoch {ep:02d}/{epochs}   train={tr:.4f}   dev={dv:.4f}{flag}")
        if no_improve >= patience:
            stop_reason = f"early stopping at epoch {ep} (dev did not improve for {patience} epochs)"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"\n{stop_reason}.\nBest dev loss: {best:.4f} (epoch {best_ep}) -> weights restored.")
    return hist, best, best_ep, stop_reason


def plot_loss(hist, model_name):
    """Plot training and dev losses and report a simple late-rise check."""
    import matplotlib.pyplot as plt
    ep = list(range(1, len(hist["train"]) + 1))
    plt.figure(figsize=(7, 4.5))
    plt.plot(ep, hist["train"], "-o", label="train")
    plt.plot(ep, hist["dev"],   "-s", label="dev / validation")
    best_ep = int(np.argmin(hist["dev"])) + 1
    best_dev = min(hist["dev"])
    gap = hist["train"][-1] - hist["dev"][-1]
    plt.title(f"Fine-tuning {model_name}   |   final train-val gap = {gap:+.4f}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
    rising = hist["dev"][-1] - best_dev
    if best_ep >= len(ep) - 1 or rising < 1e-3:
        print(f"[OK] No sign of overfitting (best dev at epoch {best_ep}/{len(ep)}, "
              f"final gap {gap:+.4f}).")
    else:
        print(f"[!] Possible overfitting: dev reached its minimum at epoch {best_ep} and rose "
              f"{rising:.4f}. Early stopping already restores that epoch.")


# ===========================================================================
# EVALUATION
# ===========================================================================
def build_eval_generator(subset, in_samples, sigma, norm_type="peak"):
    """Build evaluation windows and soft P/S labels from dataset pseudo-arrivals.

    Window length, label width, and normalization are supplied by the caller.
    """
    gen = sbg.GenericGenerator(subset)
    gen.add_augmentations([
        sbg.WindowAroundSample(list(PHASE_DICT.keys()), samples_before=in_samples // 2,
                               windowlen=in_samples, selection="first", strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=norm_type),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=PHASE_DICT, sigma=sigma, dim=0,
                                  model_labels="PSN"),
    ])
    return gen


def collect(model, subset, decode_ps, *, in_samples, sigma, sampling_rate=100,
            peak_floor=0.05, peak_dist=50, batch_size=128, device=DEVICE,
            norm_type="peak"):
    """Collect reference arrivals, predicted peaks, and metadata per window.

    ``decode_ps`` returns P/S probability arrays. ``model`` may be a pair of
    single-phase pickers, as with EQCCT.
    """
    gen = build_eval_generator(subset, in_samples, sigma, norm_type=norm_type)
    loader = DataLoader(gen, batch_size=batch_size, shuffle=False, num_workers=0)
    md_ = subset.metadata
    recs, idx = [], 0
    for _m in (model if isinstance(model, (list, tuple)) else [model]):
        _m.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            P, S = decode_ps(model, x)
            y = batch["y"].numpy()
            for i in range(x.shape[0]):
                row = md_.iloc[idx]; idx += 1
                rec = {"station": str(row.get("station_code", "?")),
                       "category": row.get("trace_category", "?"),
                       "magnitude": float(row.get("source_magnitude", np.nan)),
                       "distance":  float(row.get("path_ep_distance_km", np.nan)),
                       "snr_p":     float(row.get("trace_snr_p_db", np.nan)),
                       "snr_s":     float(row.get("trace_snr_s_db", np.nan)),
                       "true": {}, "peaks": {}}
                for ph, prob in (("P", P[i]), ("S", S[i])):
                    c = 0 if ph == "P" else 1
                    rec["true"][ph] = int(y[i, c].argmax()) if y[i, c].max() > 0.5 else None
                    pk, props = find_peaks(prob, height=peak_floor, distance=peak_dist)
                    rec["peaks"][ph] = list(zip(pk.tolist(), props["peak_heights"].tolist()))
                recs.append(rec)
    return recs


def metrics_at(recs, thr, tol_s=0.50, sampling_rate=100):
    """Compare predicted peaks with pseudo-arrival references at one threshold.

    Return detection counts, derived scores, and timing residuals for P and S.
    """
    out = {}
    for ph in ("P", "S"):
        TP = FP = FN = 0
        res, det_flags = [], []
        for r in recs:
            preds = [s for (s, p) in r["peaks"][ph] if p >= thr]
            true = r["true"][ph]
            if true is not None:
                within = sorted((abs(s - true) / sampling_rate, s) for s in preds
                                if abs(s - true) / sampling_rate < tol_s)
                if within:
                    TP += 1; res.append((within[0][1] - true) / sampling_rate)
                    FP += len(preds) - 1; det = True
                else:
                    FN += 1; FP += len(preds); det = False
                snr = r["snr_p"] if ph == "P" else r["snr_s"]
                det_flags.append((det, r["magnitude"], r["distance"], snr))
            else:
                FP += len(preds)
        prec = TP / (TP + FP) if (TP + FP) else float("nan")
        rec_ = TP / (TP + FN) if (TP + FN) else float("nan")
        f1 = (2 * prec * rec_ / (prec + rec_)
              if prec == prec and rec_ == rec_ and (prec + rec_) > 0 else float("nan"))
        arr = np.array(res)
        out[ph] = {"TP": TP, "FP": FP, "FN": FN, "precision": prec, "recall": rec_, "f1": f1,
                   "mae": float(np.mean(np.abs(arr))) if arr.size else float("nan"),
                   "rmse": float(np.sqrt(np.mean(arr ** 2))) if arr.size else float("nan"),
                   "median": float(np.median(arr)) if arr.size else float("nan"),
                   "std": float(np.std(arr)) if arr.size else float("nan"),
                   "p90": float(np.percentile(np.abs(arr), 90)) if arr.size else float("nan"),
                   "residuals": arr, "det_flags": det_flags}
    return out


def print_det(m, title, thr, tol_s):
    """Print the P/S detection table (TP, FP, FN, precision, recall, F1) from `metrics_at`."""
    print(title, f"[THR={thr}, TOL={tol_s}s]")
    print(f"  {'phase':4s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>7s} {'recall':>7s} {'F1':>7s}")
    for ph in ("P", "S"):
        d = m[ph]
        print(f"  {ph:4s} {d['TP']:4d} {d['FP']:4d} {d['FN']:4d} "
              f"{d['precision']:7.3f} {d['recall']:7.3f} {d['f1']:7.3f}")


def print_timing(m, title):
    """Print the P/S timing table (MAE, RMSE, median, std, P90) from `metrics_at`."""
    print(title)
    print(f"  {'phase':4s} {'MAE(s)':>8s} {'RMSE(s)':>8s} {'median':>8s} {'std':>7s} {'P90(s)':>8s}")
    for ph in ("P", "S"):
        d = m[ph]
        print(f"  {ph:4s} {d['mae']:8.3f} {d['rmse']:8.3f} {d['median']:+8.3f} "
              f"{d['std']:7.3f} {d['p90']:8.3f}")


def noise_fp(recs, thr):
    """Count false positives on noise windows at threshold `thr` (P, S, and either)."""
    noise = [r for r in recs if r["category"] == "noise"]
    n = len(noise)
    fp = sum(any(p >= thr for _, p in r["peaks"]["P"]) for r in noise)
    fs = sum(any(p >= thr for _, p in r["peaks"]["S"]) for r in noise)
    fa = sum(any(p >= thr for _, p in r["peaks"]["P"]) or
             any(p >= thr for _, p in r["peaks"]["S"]) for r in noise)
    return {"n": n, "P": fp, "S": fs, "any": fa}


def pr_points(recs, ph, thrs, tol_s=0.50):
    """Sweep thresholds `thrs` for phase `ph` and return (recall, precision) arrays."""
    P, R = [], []
    for t in thrs:
        m = metrics_at(recs, t, tol_s)[ph]
        P.append(m["precision"]); R.append(m["recall"])
    return np.array(R), np.array(P)


def pr_auc(R, P):
    """Area under the precision-recall curve via the trapezoidal rule, sorted by recall."""
    mask = ~np.isnan(R) & ~np.isnan(P)
    R, P = R[mask], P[mask]
    if R.size < 2:
        return float("nan")
    order = np.argsort(R)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trap(P[order], R[order]))


def recall_by_bin(det_flags, key_idx, edges):
    """Bin `det_flags` (from `metrics_at`) by the value at `key_idx` (magnitude/distance/SNR)
    and return per-bin recall and window count for the breakdown plots."""
    dets = np.array([d[0] for d in det_flags], dtype=float)
    vals = np.array([d[key_idx] for d in det_flags], dtype=float)
    rec, cnt = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vals >= lo) & (vals < hi) & ~np.isnan(vals)
        cnt.append(int(mask.sum()))
        rec.append(float(dets[mask].mean()) if mask.sum() else np.nan)
    return rec, cnt


def metrics_dataframe(arch, m_base, m_ft):
    """Flatten baseline vs. fine-tuned `metrics_at` output into one tidy DataFrame."""
    rows = []
    for tag, m in (("baseline", m_base), ("fine-tuned", m_ft)):
        for ph in ("P", "S"):
            d = m[ph]
            rows.append({"arch": arch, "model": tag, "phase": ph,
                         "TP": d["TP"], "FP": d["FP"], "FN": d["FN"],
                         "precision": d["precision"], "recall": d["recall"], "f1": d["f1"],
                         "mae_s": d["mae"], "rmse_s": d["rmse"], "median_s": d["median"],
                         "std_s": d["std"], "p90_s": d["p90"]})
    return pd.DataFrame(rows)


def save_results(tag, arch, m_base, m_ft, pr_rows, auc_summary, noise_summary,
                 rec_base, rec_ft, config):
    """Save derived metrics and inference records for later analysis."""
    metrics_df = metrics_dataframe(arch, m_base, m_ft)
    metrics_df.to_csv(RESULTS_DIR / f"metrics_{tag}.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(RESULTS_DIR / f"pr_{tag}.csv", index=False)
    summary = {"tag": tag, "arch": arch, "auc": auc_summary, "noise_fp": noise_summary,
               "config": config}
    with open(RESULTS_DIR / f"summary_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(RESULTS_DIR / f"records_{tag}.pkl", "wb") as f:
        pickle.dump({"baseline": rec_base, "finetuned": rec_ft, "config": config}, f)
    print("Saved to:", RESULTS_DIR)
    for p, ext in (("metrics", "csv"), ("pr", "csv"), ("summary", "json"), ("records", "pkl")):
        print("  ", f"{p}_{tag}.{ext}")
    return metrics_df


def load_records(tag):
    """Load saved per-window inference records by tag."""
    with open(RESULTS_DIR / f"records_{tag}.pkl", "rb") as f:
        return pickle.load(f)
