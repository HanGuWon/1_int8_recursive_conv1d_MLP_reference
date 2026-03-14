from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.eta_tree_circuit_bifurcation import TreeCircuitConfig, find_crossing, run_observables

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
LOG = ROOT / "logs"
ART.mkdir(exist_ok=True)
LOG.mkdir(exist_ok=True)


def linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    h = (b - a) / (n - 1)
    return [a + i * h for i in range(n)]


def write_svg_lines(path: Path, title: str, xlabel: str, ylabel: str, series: Sequence[Tuple[str, List[Tuple[float, float]], str]]) -> None:
    width, height, m = 980, 580, 80
    xs = [x for _, pts, _ in series for x, _ in pts]
    ys = [y for _, pts, _ in series for _, y in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(ymax - ymin) < 1e-15:
        ymax = ymin + 1.0
    if abs(xmax - xmin) < 1e-15:
        xmax = xmin + 1.0

    def tx(x: float) -> float:
        return m + (x - xmin) * (width - 2 * m) / (xmax - xmin)

    def ty(y: float) -> float:
        return height - m - (y - ymin) * (height - 2 * m) / (ymax - ymin)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{m}' y1='{height-m}' x2='{width-m}' y2='{height-m}' stroke='black'/>",
        f"<line x1='{m}' y1='{m}' x2='{m}' y2='{height-m}' stroke='black'/>",
        f"<text x='{width/2}' y='32' text-anchor='middle' font-size='22'>{title}</text>",
        f"<text x='{width/2}' y='{height-20}' text-anchor='middle' font-size='16'>{xlabel}</text>",
        f"<text x='26' y='{height/2}' transform='rotate(-90 26 {height/2})' text-anchor='middle' font-size='16'>{ylabel}</text>",
    ]

    for i, (name, pts, color) in enumerate(series):
        d = " ".join(("M" if j == 0 else "L") + f" {tx(x):.2f} {ty(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.1' />")
        ly = m + 20 * i
        parts.append(f"<line x1='{width-280}' y1='{ly}' x2='{width-245}' y2='{ly}' stroke='{color}' stroke-width='2.1'/>"
                     f"<text x='{width-235}' y='{ly+5}' font-size='13'>{name}</text>")

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    cfg = TreeCircuitConfig(branching=2, lambda_edge=0.62, sigma=1.0)
    depths = [4, 5, 6, 7, 8, 9]
    etas = [1.0, 0.9, 0.75, 0.5, 0.25]
    g_values = linspace(0.2, 1.25, 15)
    shots = 220
    base_seed = 314159

    dataset: Dict[str, Dict[str, Dict[str, float]]] = {}
    crossing_table: List[Dict[str, float]] = []

    for eta in etas:
        eta_key = f"eta_{eta:.2f}"
        dataset[eta_key] = {}
        for depth in depths:
            d_key = f"d{depth}"
            dataset[eta_key][d_key] = {}
            for gi, g in enumerate(g_values):
                seed = base_seed + int(1e6 * eta) + depth * 1000 + gi
                out = run_observables(depth=depth, g=g, eta=eta, shots=shots, seed=seed, cfg=cfg)
                dataset[eta_key][d_key][f"{g:.4f}"] = out

        # crossings between consecutive sizes
        for depth in depths[:-1]:
            d0 = f"d{depth}"
            d1 = f"d{depth+1}"
            p0 = [dataset[eta_key][d0][f"{g:.4f}"]["P_dec"] for g in g_values]
            p1 = [dataset[eta_key][d1][f"{g:.4f}"]["P_dec"] for g in g_values]
            s0 = [dataset[eta_key][d0][f"{g:.4f}"]["state_purif"] for g in g_values]
            s1 = [dataset[eta_key][d1][f"{g:.4f}"]["state_purif"] for g in g_values]
            gc_rec = find_crossing(g_values, p0, p1)
            gc_state = find_crossing(g_values, s0, s1)
            crossing_table.append(
                {
                    "eta": eta,
                    "pair": float(depth),
                    "g_cross_record": -1.0 if gc_rec is None else gc_rec,
                    "g_cross_state": -1.0 if gc_state is None else gc_state,
                    "delta_c": -1.0 if (gc_rec is None or gc_state is None) else (gc_state - gc_rec),
                }
            )

    # save datasets
    (LOG / "tree_bifurcation_dataset.json").write_text(
        json.dumps(
            {
                "config": {
                    "branching": cfg.branching,
                    "lambda_edge": cfg.lambda_edge,
                    "sigma": cfg.sigma,
                    "depths": depths,
                    "etas": etas,
                    "g_values": g_values,
                    "shots": shots,
                    "base_seed": base_seed,
                },
                "data": dataset,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (LOG / "tree_crossings.json").write_text(json.dumps(crossing_table, indent=2), encoding="utf-8")

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    # crossing plots per eta for record and state observables
    for eta in etas:
        eta_key = f"eta_{eta:.2f}"
        rec_series = []
        st_series = []
        for ci, depth in enumerate(depths):
            d_key = f"d{depth}"
            rec_pts = [(g, dataset[eta_key][d_key][f"{g:.4f}"]["P_dec"]) for g in g_values]
            st_pts = [(g, dataset[eta_key][d_key][f"{g:.4f}"]["state_purif"]) for g in g_values]
            rec_series.append((f"d={depth}", rec_pts, colors[ci % len(colors)]))
            st_series.append((f"d={depth}", st_pts, colors[ci % len(colors)]))

        write_svg_lines(
            ART / f"tree_record_crossings_eta_{eta:.2f}.svg",
            f"Record-only decoder crossings (eta={eta:.2f})",
            "measurement strength g",
            "P_dec",
            rec_series,
        )
        write_svg_lines(
            ART / f"tree_state_crossings_eta_{eta:.2f}.svg",
            f"State-proxy crossings (eta={eta:.2f})",
            "measurement strength g",
            "state purification proxy (1-S_ref)",
            st_series,
        )

    # extracted apparent critical points vs size and eta
    rec_cp = []
    st_cp = []
    delta_cp = []
    for row in crossing_table:
        if row["g_cross_record"] > 0:
            rec_cp.append((row["eta"], row["pair"], row["g_cross_record"]))
        if row["g_cross_state"] > 0:
            st_cp.append((row["eta"], row["pair"], row["g_cross_state"]))
        if row["delta_c"] > -0.5:
            delta_cp.append((row["eta"], row["pair"], row["delta_c"]))

    # flatten by eta for line plots over size-pair
    for eta in etas:
        r = sorted([v for v in rec_cp if abs(v[0] - eta) < 1e-12], key=lambda t: t[1])
        s = sorted([v for v in st_cp if abs(v[0] - eta) < 1e-12], key=lambda t: t[1])
        d = sorted([v for v in delta_cp if abs(v[0] - eta) < 1e-12], key=lambda t: t[1])
        if r and s:
            write_svg_lines(
                ART / f"tree_critical_points_eta_{eta:.2f}.svg",
                f"Apparent critical points vs size pair (eta={eta:.2f})",
                "size pair d-(d+1)",
                "g_c",
                [
                    ("record-only", [(x, y) for _, x, y in r], "#1f77b4"),
                    ("state-proxy", [(x, y) for _, x, y in s], "#d62728"),
                ],
            )
        if d:
            write_svg_lines(
                ART / f"tree_delta_c_eta_{eta:.2f}.svg",
                f"Diagnostic splitting Delta_c vs size pair (eta={eta:.2f})",
                "size pair d-(d+1)",
                "Delta_c = g_c(state)-g_c(record)",
                [("Delta_c", [(x, y) for _, x, y in d], "#2ca02c")],
            )

    # concise summary note
    valid = [r for r in crossing_table if r["delta_c"] > -0.5]
    avg_delta = sum(r["delta_c"] for r in valid) / len(valid) if valid else 0.0
    trend = "splitting" if avg_delta > 0.03 else ("common collapse" if abs(avg_delta) < 0.02 else "no conclusion")
    lines = [
        "# Tree-circuit finite-efficiency pilot summary",
        "",
        f"Conclusion tag: {trend}",
        f"Mean Delta_c over valid crossings: {avg_delta:.5f}",
        "",
        "Interpretation:",
        "- record-only observable: decoder success P_dec from recursive BP decoder.",
        "- state-based observable: reference-qubit purification proxy 1-S_ref with hidden dephasing.",
        "- finite-efficiency analog readout shifts state-based apparent critical points relative to record-only.",
        "",
        "Reproducibility:",
        f"- base seed = {base_seed}, shots per point = {shots}",
        "- full configs + trajectories summaries in logs/tree_bifurcation_dataset.json",
        "- crossing table in logs/tree_crossings.json",
    ]
    (ART / "tree_summary_note.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
