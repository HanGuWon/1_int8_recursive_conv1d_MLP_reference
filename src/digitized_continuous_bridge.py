from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class BridgeConfig:
    kappa: float = 0.8
    eta: float = 0.75
    omega: float = 0.0
    n_substeps: int = 40


def gaussian_pdf(x: float, mu: float, var: float) -> float:
    return math.exp(-0.5 * (x - mu) * (x - mu) / var) / math.sqrt(2.0 * math.pi * var)


def gaussian_model_params(dt: float, cfg: BridgeConfig) -> Dict[str, float]:
    mu = 2.0 * math.sqrt(cfg.eta * cfg.kappa) * dt
    var = dt
    chi = math.exp(-2.0 * (1.0 - cfg.eta) * cfg.kappa * dt)
    return {"mu": mu, "var": var, "chi": chi}


def gaussian_conditioned_coherence(r: float, dt: float, cfg: BridgeConfig) -> float:
    p = gaussian_model_params(dt, cfg)
    p_plus = gaussian_pdf(r, p["mu"], p["var"])
    p_minus = gaussian_pdf(r, -p["mu"], p["var"])
    norm = 0.5 * (p_plus + p_minus)
    return p["chi"] * math.sqrt(p_plus * p_minus) / (2.0 * norm)


def _clip(v: float, lo: float = -0.999999999, hi: float = 0.999999999) -> float:
    return max(lo, min(hi, v))


def _simulate_bin_for_initial(
    x0: float,
    y0: float,
    z0: float,
    dt: float,
    cfg: BridgeConfig,
    rng: random.Random,
) -> Tuple[float, float, float, float]:
    x, y, z = x0, y0, z0
    ds = dt / cfg.n_substeps
    r = 0.0
    sq = math.sqrt(cfg.eta * cfg.kappa)
    for _ in range(cfg.n_substeps):
        dW = rng.gauss(0.0, math.sqrt(ds))
        dY = 2.0 * sq * z * ds + dW

        dx = -2.0 * cfg.kappa * x * ds - 2.0 * sq * x * z * dW
        dy = -2.0 * cfg.kappa * y * ds - cfg.omega * z * ds - 2.0 * sq * y * z * dW
        dz = cfg.omega * y * ds + 2.0 * sq * (1.0 - z * z) * dW

        x = _clip(x + dx)
        y = _clip(y + dy)
        z = _clip(z + dz)
        r += dY
    return r, x, y, z


def simulate_bin_ensemble(dt: float, ntraj: int, seed: int, cfg: BridgeConfig) -> Dict[str, List[Tuple[float, float]]]:
    rng = random.Random(seed)
    out_plus: List[Tuple[float, float]] = []
    out_minus: List[Tuple[float, float]] = []
    out_coh: List[Tuple[float, float]] = []

    for _ in range(ntraj):
        rp, *_ = _simulate_bin_for_initial(0.0, 0.0, +1.0, dt, cfg, rng)
        rm, *_ = _simulate_bin_for_initial(0.0, 0.0, -1.0, dt, cfg, rng)
        rc, xc, *_ = _simulate_bin_for_initial(1.0, 0.0, 0.0, dt, cfg, rng)
        out_plus.append((rp, 1.0))
        out_minus.append((rm, 1.0))
        out_coh.append((rc, xc))

    return {"plus": out_plus, "minus": out_minus, "coh": out_coh}


def histogram_density(samples: Sequence[float], nbin: int, lo: float, hi: float) -> Tuple[List[float], List[float]]:
    bw = (hi - lo) / nbin
    bins = [0 for _ in range(nbin)]
    for s in samples:
        j = int((s - lo) / bw)
        if 0 <= j < nbin:
            bins[j] += 1
    norm = len(samples) * bw
    centers = [lo + (j + 0.5) * bw for j in range(nbin)]
    dens = [b / norm for b in bins]
    return centers, dens


def binned_conditioned_mean(pairs: Sequence[Tuple[float, float]], nbin: int, lo: float, hi: float) -> Tuple[List[float], List[float]]:
    bw = (hi - lo) / nbin
    cnt = [0 for _ in range(nbin)]
    ssum = [0.0 for _ in range(nbin)]
    for r, val in pairs:
        j = int((r - lo) / bw)
        if 0 <= j < nbin:
            cnt[j] += 1
            ssum[j] += val
    centers = [lo + (j + 0.5) * bw for j in range(nbin)]
    mean = [ssum[j] / cnt[j] if cnt[j] > 0 else 0.0 for j in range(nbin)]
    return centers, mean


def l2_error(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(1, len(a)))


def compare_one_bin(dt: float, ntraj: int, seed: int, cfg: BridgeConfig) -> Dict[str, float]:
    data = simulate_bin_ensemble(dt, ntraj, seed, cfg)
    params = gaussian_model_params(dt, cfg)

    lo = -4.5 * math.sqrt(dt)
    hi = +4.5 * math.sqrt(dt)
    nbin = 81

    rp = [r for r, _ in data["plus"]]
    rm = [r for r, _ in data["minus"]]
    rc_pairs = data["coh"]

    c, dp = histogram_density(rp, nbin, lo, hi)
    _, dm = histogram_density(rm, nbin, lo, hi)
    _, cmean = binned_conditioned_mean(rc_pairs, nbin, lo, hi)

    gp = [gaussian_pdf(x, +params["mu"], params["var"]) for x in c]
    gm = [gaussian_pdf(x, -params["mu"], params["var"]) for x in c]
    gb = [gaussian_conditioned_coherence(x, dt, cfg) for x in c]

    e_p = l2_error(dp, gp)
    e_m = l2_error(dm, gm)
    e_b = l2_error(cmean, gb)

    # visible/hidden split diagnostic (unconditioned coherence)
    mean_x = sum(x for _, x in rc_pairs) / len(rc_pairs)
    predicted_uncond = math.exp(-2.0 * cfg.kappa * dt)

    return {
        "dt": dt,
        "err_density_plus": e_p,
        "err_density_minus": e_m,
        "err_coherence": e_b,
        "mean_x_exact": mean_x,
        "mean_x_pred": predicted_uncond,
        "split_residual": mean_x - predicted_uncond,
    }


def fit_correction_power_law(dts: Sequence[float], errs: Sequence[float]) -> Dict[str, float]:
    # fit log(err)=a + p log(dt)
    pairs = [(math.log(x), math.log(max(y, 1e-18))) for x, y in zip(dts, errs)]
    sx = sum(x for x, _ in pairs)
    sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs)
    sxy = sum(x * y for x, y in pairs)
    n = len(pairs)
    den = n * sxx - sx * sx
    p = (n * sxy - sx * sy) / den if abs(den) > 1e-15 else 0.0
    a = (sy - p * sx) / n
    return {"power": p, "prefactor": math.exp(a)}


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
