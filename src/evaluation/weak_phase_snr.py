"""Summarize recovery rates by SNR and baseline-to-fine-tuned metrics.

The script reads stored evaluation records and writes derived tables to the
evaluation-results directory without running model inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import training_utils as tu          # noqa: E402
import evaluation_utils as eu        # noqa: E402

TAGS = ["phasenet_all", "eqtransformer_all", "eqcct_all", "gpd_all"]
ARCHS = ["PhaseNet", "EQTransformer", "EQCCT", "GPD"]

THR, TOL_S = 0.30, 0.50
WEAK_DB = 6.0
#: Phase-specific SNR intervals in dB; the final interval is open-ended.
EDGES = {"P": [0.0, 6.0, 12.0, 18.0, np.inf],
         "S": [0.0, 6.0, 12.0, np.inf]}
BIN_LABELS = {"P": ["0-6", "6-12", "12-18", ">18"],
              "S": ["0-6", "6-12", ">12"]}

#: Summary metric, display label, and decimal precision.
METRICS = [
    ("recall",    "Sensibilidad",           3),
    ("precision", "Precisión",              3),
    ("f2",        "F2",                     3),
    ("mae_ms",    "Error medio",            1),
    ("medae_ms",  "Error mediano",           0),
    ("weak",      "Fases débiles",          3),
]


def load():
    payloads = eu.load_all(TAGS)
    ref = payloads["PhaseNet"]["baseline"]
    common = {i for i, r in enumerate(ref) if r["true"]["S"] is not None}
    return payloads, common


def _is_noise(rec):
    return str(rec.get("category", "")).lower() == "noise"


def for_metrics(recs, phase, common):
    """Select the evaluated records and missing-label policy for each phase."""
    if phase == "P":
        return recs, "negative"
    return [r for i, r in enumerate(recs) if i in common or _is_noise(r)], "ignore"


def for_recall(recs, phase, common):
    """Keep reference-labelled arrivals for the phase-specific recall breakdown."""
    return [r for i, r in enumerate(recs) if i in common] if phase == "S" else recs


def hits_and_snr(recs, phase):
    res = eu.evaluate(recs, THR, TOL_S, unlabeled="ignore")[phase]
    key = "snr_p" if phase == "P" else "snr_s"
    hits = np.array([h for _, h in res.hits], dtype=float)
    snr = np.array([recs[i].get(key, np.nan) for i, _ in res.hits], dtype=float)
    return hits, snr


def recall_by_snr(recs, phase):
    hits, snr = hits_and_snr(recs, phase)
    edges, labels = EDGES[phase], BIN_LABELS[phase]
    rows = []
    for (lo, hi), label in zip(zip(edges[:-1], edges[1:]), labels):
        m = (snr >= lo) & (snr < hi) & ~np.isnan(snr)
        rows.append({"bin_db": label, "n": int(m.sum()),
                     "recall": float(hits[m].mean()) if m.sum() else np.nan})
    return pd.DataFrame(rows)


def weak_recall(recs, phase):
    hits, snr = hits_and_snr(recs, phase)
    m = (snr < WEAK_DB) & ~np.isnan(snr)
    return (float(hits[m].mean()) if m.sum() else np.nan), int(m.sum())


def build(payloads, common):
    breakdown, summary = [], []
    for arch in ARCHS:
        for phase in ("P", "S"):
            vals = {}
            for label, key in (("baseline", "baseline"), ("finetuned", "finetuned")):
                recs_m, unlabeled = for_metrics(payloads[arch][key], phase, common)
                r = eu.evaluate(recs_m, THR, TOL_S, unlabeled=unlabeled)[phase]
                t = eu.timing_stats(r.residuals_ms)

                recs_r = for_recall(payloads[arch][key], phase, common)
                wr, n_weak = weak_recall(recs_r, phase)
                vals[label] = {"recall": r.recall, "precision": r.precision,
                               "f1": r.f1, "f2": r.f2, "mae_ms": t["mae_ms"],
                               "medae_ms": t["medae_ms"], "weak": wr}

                b = recall_by_snr(recs_r, phase)
                b.insert(0, "model", label)
                b.insert(0, "phase", phase)
                b.insert(0, "architecture", arch)
                breakdown.append(b)

            row = {"architecture": arch, "phase": phase, "n_weak": n_weak}
            for metric, _, _ in METRICS:
                vb, vf = vals["baseline"][metric], vals["finetuned"][metric]
                # Weak-phase recovery follows the recall improvement direction.
                pct_key = "recall" if metric == "weak" else metric
                row[f"{metric}_baseline"] = vb
                row[f"{metric}_finetuned"] = vf
                row[f"{metric}_improvement_%"] = eu._improvement_pct(pct_key, vb, vf)
            row["f1_baseline"] = vals["baseline"]["f1"]
            row["f1_finetuned"] = vals["finetuned"]["f1"]
            row["f1_improvement_%"] = eu._improvement_pct(
                "f1", vals["baseline"]["f1"], vals["finetuned"]["f1"])
            summary.append(row)
    return pd.concat(breakdown, ignore_index=True), pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _num(v, dec):
    if v != v:                                   # NaN
        return "---"
    s = f"{v:.{dec}f}"
    if abs(v) >= 1000:                           # thousands separator, LaTeX style
        s = f"{v:,.{dec}f}".replace(",", "\\,")
    return s


def _pct(v):
    if v != v:
        return "---"
    return f"{v:+.1f}"


def summary_latex(summary, phase):
    d = summary[summary.phase == phase].set_index("architecture")
    head = " & ".join(f"\\textbf{{{lab}}}" for _, lab, _ in METRICS)
    out = [
        r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.3}",
        f"\\caption{{Resumen comparativo de la fase {phase} entre el modelo preentrenado "
        f"y el ajustado. Los errores temporales se expresan en milisegundos y las fases "
        f"débiles son las que presentan una relación señal/ruido inferior a "
        f"{WEAK_DB:g}\\,dB. Un valor positivo de $\\Delta$ indica mejora en todas las "
        f"columnas, de modo que en las de error corresponde a una reducción.}}",
        f"\\label{{tab:resumen_mejora_{phase.lower()}}}",
        r"\begin{tabular}{ll" + "c" * len(METRICS) + "}", r"\hline",
        f"\\textbf{{Arquitectura}} & \\textbf{{Modelo}} & {head} \\\\", r"\hline",
    ]
    for arch in ARCHS:
        r = d.loc[arch]
        for label, suffix in (("Preentrenado", "baseline"), ("Ajustado", "finetuned")):
            cells = " & ".join(
                (f"\\dg{{{_num(r[f'{m}_{suffix}'], dec)}}}"
                 if arch == "GPD" and m in ("mae_ms", "medae_ms")
                 else _num(r[f"{m}_{suffix}"], dec))
                for m, _, dec in METRICS)
            prefix = arch if label == "Preentrenado" else ""
            out.append(f"{prefix:14s} & {label:12s} & {cells} \\\\")
        cells = " & ".join(
            (f"\\dg{{{_pct(r[f'{m}_improvement_%'])}}}"
             if arch == "GPD" and m in ("mae_ms", "medae_ms")
             else _pct(r[f"{m}_improvement_%"]))
            for m, _, _ in METRICS)
        out.append(f"{'':14s} & $\\Delta$ (\\%) & {cells} \\\\")
        out.append(r"\hline" if arch != ARCHS[-1] else "")
    out += [r"\hline",
            r"\multicolumn{8}{p{0.95\textwidth}}{\scriptsize $^{\dagger}$ Resolución "
            r"temporal limitada a 200\,ms, no comparable.} \\",
            r"\multicolumn{8}{p{0.95\textwidth}}{\scriptsize Las columnas de error se "
            r"calculan sobre las detecciones de cada modelo. Su variación no es "
            r"directamente interpretable cuando la sensibilidad cambia de forma "
            r"sustancial, como ocurre en EQCCT.} \\",
            r"\end{tabular}", r"\end{table}"]
    return "\n".join(l for l in out if l != "")


def breakdown_latex(breakdown, phase):
    """Format SNR bins as rows with baseline and fine-tuned columns."""
    d = breakdown[breakdown.phase == phase]
    labels = BIN_LABELS[phase]
    n = d[(d.architecture == "PhaseNet") & (d.model == "baseline")].set_index("bin_db")["n"]

    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{a}}}}}" for a in ARCHS)
    out = [
        r"\begin{table}[ht!]", r"\centering", r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.3}",
        f"\\caption{{Sensibilidad de la onda {phase} por tramo de relación señal/ruido, "
        f"expresada en decibelios, "
        + ("sobre el subconjunto de prueba.}" if phase == "P"
           else "sobre el conjunto común de referencia.}"),
        f"\\label{{tab:desglose_snr_{phase.lower()}}}",
        r"\begin{tabular}{ll" + "cc" * len(ARCHS) + "}", r"\hline",
        r"\textbf{SNR} & \textbf{Vent.} & " + head + r" \\",
        " & & " + " & ".join(["Pre. & Aj."] * len(ARCHS)) + r" \\", r"\hline",
    ]
    for b in labels:
        cells = []
        for arch in ARCHS:
            g = d[(d.architecture == arch)].set_index(["model", "bin_db"])["recall"]
            cells += [_num(g[("baseline", b)], 3), _num(g[("finetuned", b)], 3)]
        bin_label = b.replace("-", "--").replace(">", "$>$")
        nb = f"{int(n[b]):,}".replace(",", "\\,")
        out.append(f"{bin_label:9s} & {nb:7s} & " + " & ".join(cells) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main():
    payloads, common = load()
    print(f"conjunto comun (S): {len(common)}")
    breakdown, summary = build(payloads, common)

    out = tu.RESULTS_DIR
    breakdown.to_csv(out / "p2_breakdown_snr.csv", index=False)
    summary.to_csv(out / "p1_summary_por_fase.csv", index=False)
    print("saved:", out / "p2_breakdown_snr.csv")
    print("saved:", out / "p1_summary_por_fase.csv")

    for phase in ("P", "S"):
        print(f"\n% ===== FASE {phase} =====")
        print(breakdown_latex(breakdown, phase))
        print()
        print(summary_latex(summary, phase))


if __name__ == "__main__":
    main()
