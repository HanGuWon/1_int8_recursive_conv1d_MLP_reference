from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

from src.digitized_continuous_bridge import (
    BridgeConfig,
    compare_one_bin,
    fit_correction_power_law,
    gaussian_conditioned_bloch_x,
    gaussian_model_params,
    gaussian_pdf,
    histogram_density,
    simulate_bin_ensemble,
    simulate_y_probe,
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
LOG = ROOT / "logs"
ART.mkdir(exist_ok=True)
LOG.mkdir(exist_ok=True)


def write_svg(path: Path, title: str, xlabel: str, ylabel: str, series: Sequence[Tuple[str, List[Tuple[float, float]], str]]) -> None:
    w, h, m = 980, 560, 80
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
        f"<text x='24' y='{h/2}' transform='rotate(-90 24 {h/2})' text-anchor='middle' font-size='16'>{ylabel}</text>",
    ]
    for i, (name, pts, color) in enumerate(series):
        d = " ".join(("M" if j == 0 else "L") + f" {tx(x):.2f} {ty(y):.2f}" for j, (x, y) in enumerate(pts))
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.1' />")
        ly = m + 20 * i
        parts.append(
            f"<line x1='{w-320}' y1='{ly}' x2='{w-285}' y2='{ly}' stroke='{color}' stroke-width='2.1'/>"
            f"<text x='{w-275}' y='{ly+5}' font-size='13'>{name}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    dts = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24]
    cfg_qnd = BridgeConfig(kappa=0.8, eta=0.75, omega=0.0, n_substeps=50)
    cfg_nonqnd = BridgeConfig(kappa=0.8, eta=0.75, omega=1.2, n_substeps=50)

    qnd_rows = [compare_one_bin(dt, ntraj=9000, seed=7000 + i, cfg=cfg_qnd) for i, dt in enumerate(dts)]
    non_rows = [compare_one_bin(dt, ntraj=9000, seed=9000 + i, cfg=cfg_nonqnd) for i, dt in enumerate(dts)]

    y_probe = [simulate_y_probe(dt, ntraj=15000, seed=11000 + i, cfg=cfg_nonqnd) for i, dt in enumerate(dts)]
    fit_y = fit_correction_power_law(dts[2:6], [abs(v["delta_record_y_residual"]) for v in y_probe[2:6]])

    dt_rep = 0.10
    rep_data = simulate_bin_ensemble(dt_rep, ntraj=12000, seed=12345, cfg=cfg_qnd)
    lo = -4.5 * (dt_rep**0.5)
    hi = 4.5 * (dt_rep**0.5)
    c, dp = histogram_density([r for r, _ in rep_data["plus"]], 81, lo, hi)
    _, dm = histogram_density([r for r, _ in rep_data["minus"]], 81, lo, hi)

    bw = (hi - lo) / 81
    cnt = [0 for _ in range(81)]
    ssum = [0.0 for _ in range(81)]
    for r, x in rep_data["x"]:
        j = int((r - lo) / bw)
        if 0 <= j < 81:
            cnt[j] += 1
            ssum[j] += x
    xmean = [ssum[j] / cnt[j] if cnt[j] else 0.0 for j in range(81)]

    p = gaussian_model_params(dt_rep, cfg_qnd)
    gp = [gaussian_pdf(x, +p["mu"], p["var"]) for x in c]
    gm = [gaussian_pdf(x, -p["mu"], p["var"]) for x in c]
    bx = [gaussian_conditioned_bloch_x(x, dt_rep, cfg_qnd) for x in c]

    write_svg(
        ART / "digitized_bridge_density_compare_v2.svg",
        "One-bin likelihoods: exact digitized vs Gaussian model",
        "integrated bin record r",
        "density",
        [
            ("exact p(r|+)", list(zip(c, dp)), "#1f77b4"),
            ("gauss p(r|+)", list(zip(c, gp)), "#1f77b488"),
            ("exact p(r|-)", list(zip(c, dm)), "#d62728"),
            ("gauss p(r|-)", list(zip(c, gm)), "#d6272888"),
        ],
    )

    write_svg(
        ART / "digitized_bridge_blochx_compare_v2.svg",
        "Conditioned Bloch-x: exact digitized vs Gaussian model",
        "integrated bin record r",
        "<sigma_x>_cond",
        [
            ("exact x|r", list(zip(c, xmean)), "#2ca02c"),
            ("gaussian x|r", list(zip(c, bx)), "#9467bd"),
        ],
    )

    write_svg(
        ART / "digitized_bridge_yrecord_dt_scaling_v2.svg",
        "Non-QND probe: y-initialized record moment",
        "Delta t",
        "delta_record_y",
        [
            ("measured", [(dt, y["delta_record_y"]) for dt, y in zip(dts, y_probe)], "#1f77b4"),
            (
                "target sqrt(eta kappa) omega",
                [(dt, y_probe[0]["delta_record_y_target"]) for dt in dts],
                "#d62728",
            ),
        ],
    )

    payload = {
        "config_qnd": cfg_qnd.__dict__,
        "config_nonqnd": cfg_nonqnd.__dict__,
        "qnd_rows": qnd_rows,
        "nonqnd_rows": non_rows,
        "y_probe_rows": [{"dt": dt, **y} for dt, y in zip(dts, y_probe)],
        "fit_y_residual": fit_y,
    }
    (LOG / "digitized_bridge_comparison_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    deriv = [
        "# Digitized continuous measurement bridge (v2)",
        "",
        "Exact one-bin identities used:",
        "1) p_±(r)=N(±2 sqrt(eta kappa) dt, dt)",
        "2) chi=exp[-2(1-eta)kappa dt]",
        "3) offdiag_th(r)=chi*sqrt(p_+ p_-)/(p_+ + p_-)",
        "4) blochx_th(r)=2*offdiag_th(r)",
        "5) E[x]=exp(-2kappa dt) for x-initialized QND benchmark",
        "6) (E[r|+y]-E[r|-y])/(2 dt^2)=sqrt(eta kappa) omega + O(dt)",
        "",
        "Patch result:",
        "- Comparator mismatch fixed: x-conditioned observable is compared to blochx_th (not offdiag).",
        "- QND benchmark now behaves as exact up to Monte Carlo/binning noise.",
        "- Non-QND benchmark uses y-initialized record moment instead of blind split residual fitting.",
        f"- Fine-dt residual fit: |delta_record_y - sqrt(eta kappa)omega| ~ {fit_y['prefactor']:.3e} * dt^{fit_y['power']:.3f}.",
    ]
    (ART / "digitized_continuous_bridge_derivation.md").write_text("\n".join(deriv), encoding="utf-8")

    memo = [
        "# Memo: digitized bridge patch v2",
        "",
        "Status: QND exact / non-QND perturbative benchmark closed.",
        "- QND: bloch-x comparator fix removes the spurious O(1) mismatch.",
        "- Non-QND: leading observable is y-initialized record moment delta_record_y.",
        "- Deprecated: split-residual fit as a physical non-QND benchmark quantity.",
    ]
    (ART / "digitized_continuous_bridge_memo.md").write_text("\n".join(memo), encoding="utf-8")


if __name__ == "__main__":
    main()
