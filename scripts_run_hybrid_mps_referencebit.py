from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.eta_hybrid_mps_referencebit import (
    HybridMPSConfig,
    crossing_x,
    run_point,
    simulate_trajectory,
    summarize_convergence,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
LOG = ROOT / "logs"
ART.mkdir(exist_ok=True)
LOG.mkdir(exist_ok=True)


def write_svg(path: Path, title: str, xlabel: str, ylabel: str, series: Sequence[Tuple[str, List[Tuple[float, float]], str]]) -> None:
    w, h, m = 980, 580, 80
    xs = [x for _, pts, _ in series for x, _ in pts]
    ys = [y for _, pts, _ in series for _, y in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1.0

    def tx(x: float) -> float:
        return m + (x - xmin) * (w - 2 * m) / (xmax - xmin)

    def ty(y: float) -> float:
        return h - m - (y - ymin) * (h - 2 * m) / (ymax - ymin)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{m}' y1='{h-m}' x2='{w-m}' y2='{h-m}' stroke='black'/>",
        f"<line x1='{m}' y1='{m}' x2='{m}' y2='{h-m}' stroke='black'/>",
        f"<text x='{w/2}' y='30' text-anchor='middle' font-size='22'>{title}</text>",
        f"<text x='{w/2}' y='{h-20}' text-anchor='middle' font-size='16'>{xlabel}</text>",
        f"<text x='25' y='{h/2}' transform='rotate(-90 25 {h/2})' text-anchor='middle' font-size='16'>{ylabel}</text>",
    ]

    for i, (name, pts, color) in enumerate(series):
        d = " ".join(("M" if j == 0 else "L") + f" {tx(x):.2f} {ty(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.0' />")
        ly = m + 20 * i
        parts.append(f"<line x1='{w-310}' y1='{ly}' x2='{w-270}' y2='{ly}' stroke='{color}' stroke-width='2.0'/><text x='{w-260}' y='{ly+5}' font-size='13'>{name}</text>")

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    cfg = HybridMPSConfig()
    L_values = [16, 24, 32, 40, 48]
    etas = [1.0, 0.9, 0.75, 0.5, 0.25]
    g_values = [0.30 + 0.08 * i for i in range(9)]
    ntraj = 8
    base_seed = 8081

    raw_likelihoods: List[Dict[str, object]] = []
    dataset: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    convergence_rows: List[Dict[str, float]] = []
    crossing_rows: List[Dict[str, float]] = []

    for eta in etas:
        keta = f"eta_{eta:.2f}"
        dataset[keta] = {}
        for L in L_values:
            kL = f"L{L}"
            dataset[keta][kL] = {}
            T_main = L
            T_long = 2 * L
            t_list = [T_main, T_long] if L == max(L_values) else [T_main]
            for T in t_list:
                kT = f"T{T}"
                dataset[keta][kL][kT] = {}
                for g_idx, g in enumerate(g_values):
                    # convergence in chi
                    rows = []
                    for chi in cfg.chi_values:
                        seed = base_seed + int(1000 * eta) + 31 * L + 7 * T + g_idx + chi
                        out = run_point(L, T, g, eta, chi, ntraj, seed, cfg)
                        rows.append(out)
                    conv = summarize_convergence(rows, cfg.chi_values)
                    convergence_rows.append(
                        {
                            "eta": eta,
                            "L": float(L),
                            "T": float(T),
                            "g": g,
                            "max_delta_P_dec": conv["max_delta_P_dec"],
                            "max_delta_S_ref": conv["max_delta_S_ref"],
                        }
                    )
                    final = rows[-1]
                    dataset[keta][kL][kT][f"{g:.3f}"] = final

                    # keep a small raw trajectory subset for reproducibility/auditing
                    if L in (16, 32, 48) and T == T_main and g_idx in (2, 6, 10):
                        for tr in range(10):
                            seed = base_seed + int(1e5 * eta) + 13 * L + 17 * T + 23 * g_idx + tr
                            rr = simulate_trajectory(L, T, g, eta, cfg.chi_values[-1], seed, cfg)
                            raw_likelihoods.append(
                                {
                                    "eta": eta,
                                    "L": L,
                                    "T": T,
                                    "g": g,
                                    "seed": seed,
                                    "b_true": rr["b_true"],
                                    "decoded": rr["decoded"],
                                    "logp0": rr["logp0"],
                                    "logp1": rr["logp1"],
                                    "llr_total": rr["llr_total"],
                                    "S_ref": rr["S_ref"],
                                }
                            )

        # crossings (main time only)
        for i in range(len(L_values) - 1):
            Ls, Lb = L_values[i], L_values[i + 1]
            ys_p_s = [dataset[keta][f"L{Ls}"][f"T{Ls}"][f"{g:.3f}"]["P_dec"] for g in g_values]
            ys_p_b = [dataset[keta][f"L{Lb}"][f"T{Lb}"][f"{g:.3f}"]["P_dec"] for g in g_values]
            ys_s_s = [dataset[keta][f"L{Ls}"][f"T{Ls}"][f"{g:.3f}"]["S_ref"] for g in g_values]
            ys_s_b = [dataset[keta][f"L{Lb}"][f"T{Lb}"][f"{g:.3f}"]["S_ref"] for g in g_values]
            gc_rec = crossing_x(g_values, ys_p_s, ys_p_b)
            gc_state = crossing_x(g_values, ys_s_s, ys_s_b)
            crossing_rows.append(
                {
                    "eta": eta,
                    "L_pair": f"{Ls}-{Lb}",
                    "gc_record": -1.0 if gc_rec is None else gc_rec,
                    "gc_state": -1.0 if gc_state is None else gc_state,
                    "Delta_c": -1.0 if (gc_rec is None or gc_state is None) else (gc_state - gc_rec),
                }
            )

    # plots: crossings for each eta
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for eta in etas:
        keta = f"eta_{eta:.2f}"
        series_rec = []
        series_state = []
        for ci, L in enumerate(L_values):
            rec = [(g, dataset[keta][f"L{L}"][f"T{L}"][f"{g:.3f}"]["P_dec"]) for g in g_values]
            st = [(g, dataset[keta][f"L{L}"][f"T{L}"][f"{g:.3f}"]["S_ref"]) for g in g_values]
            series_rec.append((f"L={L}", rec, colors[ci % len(colors)]))
            series_state.append((f"L={L}", st, colors[ci % len(colors)]))
        write_svg(ART / f"hybrid_record_crossings_eta_{eta:.2f}.svg", f"Record-only decoder crossings (eta={eta:.2f})", "g", "P_dec", series_rec)
        write_svg(ART / f"hybrid_state_crossings_eta_{eta:.2f}.svg", f"Reference-qubit entropy crossings (eta={eta:.2f})", "g", "S_ref", series_state)

    # critical-point and Delta_c plots per eta
    for eta in etas:
        rows = [r for r in crossing_rows if abs(r["eta"] - eta) < 1e-12]
        pts_rec = []
        pts_st = []
        pts_d = []
        for i, r in enumerate(rows):
            x = i + 1
            if r["gc_record"] > 0:
                pts_rec.append((x, r["gc_record"]))
            if r["gc_state"] > 0:
                pts_st.append((x, r["gc_state"]))
            if r["Delta_c"] > -0.5:
                pts_d.append((x, r["Delta_c"]))
        if pts_rec and pts_st:
            write_svg(ART / f"hybrid_gc_eta_{eta:.2f}.svg", f"Apparent critical points (eta={eta:.2f})", "size-pair index", "g_c", [("record", pts_rec, "#1f77b4"), ("state", pts_st, "#d62728")])
        if pts_d:
            write_svg(ART / f"hybrid_delta_c_eta_{eta:.2f}.svg", f"Diagnostic splitting Delta_c (eta={eta:.2f})", "size-pair index", "Delta_c", [("Delta_c", pts_d, "#2ca02c")])

    # convergence/collapse figure
    conv_by_eta = []
    for eta in etas:
        vals = [r["max_delta_P_dec"] + r["max_delta_S_ref"] for r in convergence_rows if abs(r["eta"] - eta) < 1e-12]
        avg = sum(vals) / len(vals)
        conv_by_eta.append((eta, avg))
    write_svg(ART / "hybrid_convergence_vs_eta.svg", "Bond-dimension convergence proxy", "eta", "avg |Δchi| (P_dec + S_ref)", [("convergence", conv_by_eta, "#9467bd")])

    # save logs
    (LOG / "hybrid_mps_dataset.json").write_text(json.dumps({"config": {"L_values": L_values, "etas": etas, "g_values": g_values, "ntraj": ntraj, "chi_values": cfg.chi_values, "base_seed": base_seed}, "data": dataset}, indent=2), encoding="utf-8")
    (LOG / "hybrid_mps_convergence.json").write_text(json.dumps(convergence_rows, indent=2), encoding="utf-8")
    (LOG / "hybrid_mps_crossings.json").write_text(json.dumps(crossing_rows, indent=2), encoding="utf-8")
    (LOG / "hybrid_mps_raw_likelihoods.json").write_text(json.dumps(raw_likelihoods, indent=2), encoding="utf-8")

    valid = [r["Delta_c"] for r in crossing_rows if r["Delta_c"] > -0.5]
    mean_delta = sum(valid) / len(valid) if valid else float("nan")
    if not valid:
        tag = "full rounding / no-go"
    elif mean_delta > 0.04:
        tag = "diagnostic bifurcation"
    elif abs(mean_delta) < 0.02:
        tag = "common hidden-scale crossover"
    else:
        tag = "full rounding / no-go"

    note = [
        "# Hybrid MPS reference-bit summary",
        "",
        f"Conclusion: {tag}",
        f"Mean Delta_c (valid crossings): {mean_delta if valid else 'N/A'}",
        "",
        "Checklist:",
        "- eta=1 baseline included",
        "- record-only Bayes decoder from trajectory likelihoods",
        "- state proxy from reference-qubit entropy",
        "- finite-size crossing extraction for both observables",
        "- bond-dimension convergence table saved",
        "",
        "Artifacts:",
        "- logs/hybrid_mps_dataset.json",
        "- logs/hybrid_mps_raw_likelihoods.json",
        "- logs/hybrid_mps_convergence.json",
        "- logs/hybrid_mps_crossings.json",
    ]
    (ART / "hybrid_mps_summary_report.md").write_text("\n".join(note), encoding="utf-8")


if __name__ == "__main__":
    main()
