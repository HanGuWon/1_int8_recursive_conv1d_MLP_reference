from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TreeCircuitConfig:
    branching: int = 2
    lambda_edge: float = 0.62
    sigma: float = 1.0

    def num_nodes(self, depth: int) -> int:
        return 2 ** (depth + 1) - 1


def sigma_eff_from_eta(sigma: float, eta: float) -> float:
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must be in (0,1], got {eta}")
    return sigma / math.sqrt(eta)


def hidden_dephasing_gamma(g: float, sigma: float, eta: float) -> float:
    # tau^2 = sigma^2 (1-eta)/eta and Sigma^2 = sigma^2/eta
    # gamma_hid = g^2 tau^2 / (2 sigma^2 Sigma^2) = g^2 (1-eta)/(2 sigma^2)
    return (g * g) * (1.0 - eta) / (2.0 * sigma * sigma)


def local_llr(y: float, g: float, sigma_eff: float) -> float:
    return 2.0 * g * y / (sigma_eff * sigma_eff)


def child_to_parent_llr(child_llr: float, lambda_edge: float) -> float:
    # exact message passing for binary symmetric broadcast edge
    a = 0.5 * (1.0 + lambda_edge)
    b = 0.5 * (1.0 - lambda_edge)
    ex = math.exp(0.5 * child_llr)
    em = math.exp(-0.5 * child_llr)
    num = a * ex + b * em
    den = b * ex + a * em
    return math.log(num / den)


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-15), 1.0 - 1e-15)
    return -(p * math.log(p, 2) + (1.0 - p) * math.log(1.0 - p, 2))


def sample_tree_measurements(
    depth: int,
    g: float,
    eta: float,
    cfg: TreeCircuitConfig,
    rng: random.Random,
) -> Tuple[int, List[int], List[float]]:
    n = cfg.num_nodes(depth)
    spins = [0] * n
    y = [0.0] * n

    sigma_eff = sigma_eff_from_eta(cfg.sigma, eta)

    root = 1 if rng.random() < 0.5 else -1
    spins[0] = root
    y[0] = rng.gauss(g * root, sigma_eff)

    for idx in range(2 ** depth - 1):
        s = spins[idx]
        left = 2 * idx + 1
        right = 2 * idx + 2
        for child in (left, right):
            if child >= n:
                continue
            same = rng.random() < 0.5 * (1.0 + cfg.lambda_edge)
            cspin = s if same else -s
            spins[child] = cspin
            y[child] = rng.gauss(g * cspin, sigma_eff)

    return root, spins, y


def decode_root_llr(depth: int, y: Sequence[float], g: float, eta: float, cfg: TreeCircuitConfig) -> float:
    n = cfg.num_nodes(depth)
    sigma_eff = sigma_eff_from_eta(cfg.sigma, eta)
    msg = [0.0] * n

    first_leaf = 2**depth - 1
    for idx in range(n - 1, -1, -1):
        llr_local = local_llr(y[idx], g, sigma_eff)
        if idx >= first_leaf:
            msg[idx] = llr_local
        else:
            l = 2 * idx + 1
            r = 2 * idx + 2
            contrib = child_to_parent_llr(msg[l], cfg.lambda_edge) + child_to_parent_llr(msg[r], cfg.lambda_edge)
            msg[idx] = llr_local + contrib
    return msg[0]


def conditional_state_proxy(llr_root: float, depth: int, g: float, eta: float, cfg: TreeCircuitConfig) -> float:
    """Reference-qubit entropy proxy from posterior confidence + hidden dephasing.

    We map posterior confidence |m|=|tanh(LLR/2)| and hidden dephasing c=exp(-N gamma_hid)
    to an effective Bloch length r=c|m|, then return S_ref = H2((1+r)/2).
    """
    m_abs = abs(math.tanh(0.5 * llr_root))
    gamma_hid = hidden_dephasing_gamma(g, cfg.sigma, eta)
    c = math.exp(-cfg.num_nodes(depth) * gamma_hid)
    r = min(1.0, max(0.0, c * m_abs))
    return binary_entropy(0.5 * (1.0 + r))


def run_observables(
    depth: int,
    g: float,
    eta: float,
    shots: int,
    seed: int,
    cfg: TreeCircuitConfig,
) -> Dict[str, float]:
    rng = random.Random(seed)
    correct = 0
    state_entropy_sum = 0.0

    for _ in range(shots):
        root, _, y = sample_tree_measurements(depth, g, eta, cfg, rng)
        llr = decode_root_llr(depth, y, g, eta, cfg)
        dec = 1 if llr >= 0.0 else -1
        if dec == root:
            correct += 1
        state_entropy_sum += conditional_state_proxy(llr, depth, g, eta, cfg)

    p_dec = correct / shots
    s_ref = state_entropy_sum / shots
    return {
        "P_dec": p_dec,
        "state_entropy": s_ref,
        "state_purif": 1.0 - s_ref,
    }


def find_crossing(x: Sequence[float], y_small: Sequence[float], y_large: Sequence[float]) -> Optional[float]:
    if len(x) != len(y_small) or len(x) != len(y_large):
        raise ValueError("x and y arrays must have same length")
    diff = [a - b for a, b in zip(y_small, y_large)]
    for i in range(len(diff) - 1):
        if diff[i] == 0.0:
            return x[i]
        if diff[i] * diff[i + 1] < 0.0:
            x0, x1 = x[i], x[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            return x0 - d0 * (x1 - x0) / (d1 - d0)
    return None


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
