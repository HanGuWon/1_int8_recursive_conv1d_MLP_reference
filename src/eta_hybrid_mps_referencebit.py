from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HybridMPSConfig:
    sigma: float = 1.0
    diffusion: float = 0.22
    interaction: float = 0.35
    meas_backaction: float = 0.06
    base_seed: int = 424242
    chi_values: Tuple[int, ...] = (32, 64, 128)


def sigma_eff(sigma: float, eta: float) -> float:
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0,1], got {eta}")
    return sigma / math.sqrt(eta)


def gamma_hidden(g: float, sigma: float, eta: float) -> float:
    return (g * g) * (1.0 - eta) / (2.0 * sigma * sigma)


def log_gaussian(y: float, mu: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((y - mu) * (y - mu)) / var)


def _init_field(L: int, bit: int) -> List[float]:
    sgn = 1.0 if bit == 1 else -1.0
    center = L // 2
    out = []
    for i in range(L):
        d = abs(i - center)
        out.append(sgn * math.exp(-d / 3.0))
    return out


def _brickwork_layer(field: List[float], rng: random.Random, cfg: HybridMPSConfig, chi: int, layer_parity: int) -> float:
    """Apply one random brickwork layer and return truncation proxy error."""
    L = len(field)
    trunc_err = 0.0
    i = layer_parity
    while i < L - 1:
        a = field[i]
        b = field[i + 1]
        theta = cfg.interaction * (0.6 + 0.8 * rng.random())
        c = math.cos(theta)
        s = math.sin(theta)
        na = c * a + s * b
        nb = -s * a + c * b
        # measurement-induced nonlinearity proxy
        nl = cfg.meas_backaction * (na * na - nb * nb)
        na -= nl
        nb += nl
        field[i] = na
        field[i + 1] = nb
        i += 2

    # diffusion
    old = field[:]
    for j in range(L):
        left = old[(j - 1) % L]
        right = old[(j + 1) % L]
        field[j] = (1.0 - cfg.diffusion) * old[j] + 0.5 * cfg.diffusion * (left + right)

    # bond-dimension truncation proxy: high-frequency damping
    damp = math.exp(-24.0 / max(chi, 1))
    for j in range(L):
        field[j] *= damp
    trunc_err += (1.0 - damp) * sum(abs(x) for x in old) / L

    # keep bounded
    for j in range(L):
        field[j] = max(-1.0, min(1.0, field[j]))

    return trunc_err


def evolve_means(L: int, T: int, bit: int, chi: int, seed: int, cfg: HybridMPSConfig) -> Tuple[List[List[float]], float]:
    rng = random.Random(seed)
    field = _init_field(L, bit)
    frames = [field[:]]
    trunc_total = 0.0
    for t in range(T):
        trunc_total += _brickwork_layer(field, rng, cfg, chi, layer_parity=t % 2)
        frames.append(field[:])
    return frames, trunc_total / max(T, 1)


def simulate_trajectory(
    L: int,
    T: int,
    g: float,
    eta: float,
    chi: int,
    seed: int,
    cfg: HybridMPSConfig,
) -> Dict[str, object]:
    sigma_y = sigma_eff(cfg.sigma, eta)
    var_y = sigma_y * sigma_y

    # Shared random circuit seed ensures both hypotheses use same circuit realization.
    circ_seed = 17 * seed + 3
    means0, trunc0 = evolve_means(L, T, 0, chi, circ_seed + 101, cfg)
    means1, trunc1 = evolve_means(L, T, 1, chi, circ_seed + 101, cfg)

    rng = random.Random(seed + 911)
    b_true = 1 if rng.random() < 0.5 else 0
    means_true = means1 if b_true == 1 else means0

    logp0 = 0.0
    logp1 = 0.0
    llr_time: List[float] = []
    ref_site = L // 2
    hidden = math.exp(-T * gamma_hidden(g, cfg.sigma, eta))

    for t in range(1, T + 1):
        step_llr = 0.0
        for i in range(L):
            mu_t = g * means_true[t][i]
            y = rng.gauss(mu_t, sigma_y)
            lp0 = log_gaussian(y, g * means0[t][i], var_y)
            lp1 = log_gaussian(y, g * means1[t][i], var_y)
            logp0 += lp0
            logp1 += lp1
            step_llr += lp1 - lp0
        llr_time.append(step_llr)

    decoded = 1 if logp1 >= logp0 else 0
    success = 1 if decoded == b_true else 0

    # state-based proxy: reference-bit coherence from branch separation and hidden dephasing
    delta_ref = abs(means1[-1][ref_site] - means0[-1][ref_site])
    coherence = hidden * math.exp(-0.5 * g * g * delta_ref * delta_ref)
    coherence = min(1.0 - 1e-15, max(1e-15, coherence))
    p = min(1.0 - 1e-15, max(1e-15, 0.5 * (1.0 + coherence)))
    s_ref = -(p * math.log(p, 2) + (1.0 - p) * math.log(1.0 - p, 2))

    return {
        "b_true": b_true,
        "decoded": decoded,
        "success": success,
        "logp0": logp0,
        "logp1": logp1,
        "llr_total": logp1 - logp0,
        "llr_time": llr_time,
        "S_ref": s_ref,
        "purity_ref": 1.0 - s_ref,
        "trunc_err": 0.5 * (trunc0 + trunc1),
    }


def run_point(
    L: int,
    T: int,
    g: float,
    eta: float,
    chi: int,
    ntraj: int,
    seed: int,
    cfg: HybridMPSConfig,
) -> Dict[str, float]:
    successes = 0
    sref_sum = 0.0
    purity_sum = 0.0
    trunc_sum = 0.0

    for k in range(ntraj):
        out = simulate_trajectory(L, T, g, eta, chi, seed + 10007 * k, cfg)
        successes += int(out["success"])
        sref_sum += float(out["S_ref"])
        purity_sum += float(out["purity_ref"])
        trunc_sum += float(out["trunc_err"])

    p_dec = successes / ntraj
    return {
        "P_dec": p_dec,
        "S_ref": sref_sum / ntraj,
        "purity_ref": purity_sum / ntraj,
        "trunc_err": trunc_sum / ntraj,
    }


def crossing_x(x: Sequence[float], y_a: Sequence[float], y_b: Sequence[float]) -> Optional[float]:
    if len(x) != len(y_a) or len(x) != len(y_b):
        raise ValueError("length mismatch")
    diff = [a - b for a, b in zip(y_a, y_b)]
    for i in range(len(diff) - 1):
        if diff[i] == 0.0:
            return x[i]
        if diff[i] * diff[i + 1] < 0:
            x0, x1 = x[i], x[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            return x0 - d0 * (x1 - x0) / (d1 - d0)
    return None


def summarize_convergence(rows: Sequence[Dict[str, float]], chi_values: Sequence[int]) -> Dict[str, float]:
    if len(rows) != len(chi_values):
        raise ValueError("rows and chi_values mismatch")
    ref = rows[-1]
    max_dp = 0.0
    max_ds = 0.0
    for r in rows[:-1]:
        max_dp = max(max_dp, abs(r["P_dec"] - ref["P_dec"]))
        max_ds = max(max_ds, abs(r["S_ref"] - ref["S_ref"]))
    return {"max_delta_P_dec": max_dp, "max_delta_S_ref": max_ds}


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
