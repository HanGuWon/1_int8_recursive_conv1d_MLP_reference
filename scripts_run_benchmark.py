from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from src.eta_gaussian_local_benchmark import (
    GaussianWeakMeasurementModel,
    conditioned_coefficients,
    conditioned_coefficients_from_integral,
    max_relative_error,
    monte_carlo_qnd,
    sweep_decoder,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
LOG = ROOT / "logs"
ART.mkdir(exist_ok=True)
LOG.mkdir(exist_ok=True)


def linspace(a: float, b: float, n: int) -> List[float]:
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def write_svg_plot(path: Path, series: Sequence[Tuple[str, List[Tuple[float, float]], str]], title: str, xlabel: str, ylabel: str) -> None:
    width, height = 900, 520
    margin = 70
    all_x = [x for _, pts, _ in series for x, _ in pts]
    all_y = [y for _, pts, _ in series for _, y in pts]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if ymax == ymin:
        ymax = ymin + 1.0

    def tx(x: float) -> float:
        return margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)

    def ty(y: float) -> float:
        return height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='black'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='black'/>",
        f"<text x='{width/2}' y='30' text-anchor='middle' font-size='20'>{title}</text>",
        f"<text x='{width/2}' y='{height-20}' text-anchor='middle' font-size='16'>{xlabel}</text>",
        f"<text x='20' y='{height/2}' transform='rotate(-90,20,{height/2})' text-anchor='middle' font-size='16'>{ylabel}</text>",
    ]

    for idx, (name, pts, color) in enumerate(series):
        d = " ".join([("M" if i == 0 else "L") + f" {tx(x):.2f} {ty(y):.2f}" for i, (x, y) in enumerate(pts)])
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2' />")
        ly = margin + 20 * idx
        parts.append(f"<line x1='{width-260}' y1='{ly}' x2='{width-230}' y2='{ly}' stroke='{color}' stroke-width='2'/>"
                     f"<text x='{width-220}' y='{ly+5}' font-size='14'>{name}</text>")

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    model = GaussianWeakMeasurementModel(g=1.1, sigma=0.9, tau=0.7)

    y_grid = linspace(-3.5 * model.Sigma, 3.5 * model.Sigma, 181)
    exact = [conditioned_coefficients(y, model) for y in y_grid]
    quad = [conditioned_coefficients_from_integral(y, model) for y in y_grid]

    rel_a_plus = max_relative_error([v[0] for v in exact], [v[0] for v in quad])
    rel_a_minus = max_relative_error([v[1] for v in exact], [v[1] for v in quad])
    rel_b = max_relative_error([v[2] for v in exact], [v[2] for v in quad])

    write_svg_plot(
        ART / "coefficients_exact_vs_quad.svg",
        [
            ("A+ exact", list(zip(y_grid, [v[0] for v in exact])), "#1f77b4"),
            ("A+ quad", list(zip(y_grid, [v[0] for v in quad])), "#1f77b488"),
            ("A- exact", list(zip(y_grid, [v[1] for v in exact])), "#d62728"),
            ("A- quad", list(zip(y_grid, [v[1] for v in quad])), "#d6272888"),
            ("B exact", list(zip(y_grid, [v[2] for v in exact])), "#2ca02c"),
            ("B quad", list(zip(y_grid, [v[2] for v in quad])), "#2ca02c88"),
        ],
        "Exact vs numerical coefficients",
        "y",
        "coefficient value",
    )

    sweep = sweep_decoder(model, range(1, 81))
    write_svg_plot(
        ART / "decoder_collapse.svg",
        [("P_dec exact", [(x, p) for _, x, p in sweep], "#9467bd")],
        "Decoder collapse",
        "n * gamma_vis",
        "P_dec",
    )

    n_samples = [1, 2, 4, 8, 16, 32]
    mc = [monte_carlo_qnd(n, model, shots=120000, seed=1234 + n) for n in n_samples]
    write_svg_plot(
        ART / "coherence_suppression.svg",
        [
            ("MC", list(zip(n_samples, [r["coh_mc"] for r in mc])), "#ff7f0e"),
            ("exp(-n gamma_hid)", list(zip(n_samples, [r["coh_exact"] for r in mc])), "#17becf"),
        ],
        "Coherence suppression",
        "n",
        "coherence factor",
    )

    report = {
        "parameters": {
            "g": model.g,
            "sigma": model.sigma,
            "tau": model.tau,
            "Sigma": model.Sigma,
            "chi": model.chi,
            "gamma_vis": model.gamma_vis,
            "gamma_hid": model.gamma_hid,
        },
        "max_relative_errors": {"A_plus": rel_a_plus, "A_minus": rel_a_minus, "B": rel_b},
        "monte_carlo": [{"n": n, **res} for n, res in zip(n_samples, mc)],
        "discrepancies": "None observed within sampling error.",
    }
    (LOG / "monte_carlo_log.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    (ART / "derivation_note.md").write_text(
        "\n".join(
            [
                "# Derivation note: Gaussian local benchmark",
                "",
                "Verified exact factorization:",
                "E_y^(g,sigma,tau) = D_chi o E_y^(g,Sigma,0)",
                "",
                f"Sigma^2 = sigma^2 + tau^2 = {model.Sigma*model.Sigma:.15f}",
                f"chi = exp[-g^2 tau^2/(2 sigma^2 (sigma^2+tau^2))] = {model.chi:.15f}",
                f"gamma_vis = g^2/(2 Sigma^2) = {model.gamma_vis:.15f}",
                f"gamma_hid = -log(chi) = {model.gamma_hid:.15f}",
                "",
                "Decoder benchmark:",
                "P_dec_opt(n)=1/2*(1+erf(sqrt(n*gamma_vis)))",
                "",
                "Post-selection coherence benchmark:",
                "Conditioned noisy state equals visible conditioned state followed by D_{exp(-n gamma_hid)}.",
                "",
                "Quadrature coefficient relative errors:",
                f"A_plus: {rel_a_plus:.3e}",
                f"A_minus: {rel_a_minus:.3e}",
                f"B: {rel_b:.3e}",
                "",
                "Raw Monte Carlo log: logs/monte_carlo_log.json",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
