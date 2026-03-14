from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class GaussianWeakMeasurementModel:
    g: float
    sigma: float
    tau: float

    @property
    def Sigma(self) -> float:
        return math.sqrt(self.sigma * self.sigma + self.tau * self.tau)

    @property
    def chi(self) -> float:
        s2 = self.sigma * self.sigma
        return math.exp(-(self.g * self.g) * (self.tau * self.tau) / (2.0 * s2 * (s2 + self.tau * self.tau)))

    @property
    def gamma_vis(self) -> float:
        return (self.g * self.g) / (2.0 * self.Sigma * self.Sigma)

    @property
    def gamma_hid(self) -> float:
        return -math.log(self.chi)


def gaussian_pdf(x: float, mu: float, var: float) -> float:
    return math.exp(-((x - mu) * (x - mu)) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)


def p_y_given_z(y: float, z: int, model: GaussianWeakMeasurementModel) -> float:
    return gaussian_pdf(y, model.g * z, model.Sigma * model.Sigma)


def conditioned_coefficients(y: float, model: GaussianWeakMeasurementModel) -> Tuple[float, float, float]:
    p_plus = p_y_given_z(y, +1, model)
    p_minus = p_y_given_z(y, -1, model)
    norm = 0.5 * (p_plus + p_minus)
    a_plus = p_plus / (2.0 * norm)
    a_minus = p_minus / (2.0 * norm)
    b = model.chi * math.sqrt(p_plus * p_minus) / (2.0 * norm)
    return a_plus, a_minus, b


def _simpson(f, a: float, b: float) -> float:
    c = 0.5 * (a + b)
    return (b - a) * (f(a) + 4.0 * f(c) + f(b)) / 6.0


def _adaptive_simpson(f, a: float, b: float, eps: float, whole: float, depth: int) -> float:
    c = 0.5 * (a + b)
    left = _simpson(f, a, c)
    right = _simpson(f, c, b)
    if depth <= 0 or abs(left + right - whole) <= 15.0 * eps:
        return left + right + (left + right - whole) / 15.0
    return _adaptive_simpson(f, a, c, eps / 2.0, left, depth - 1) + _adaptive_simpson(
        f, c, b, eps / 2.0, right, depth - 1
    )


def integrate_real_line(f, sigma_scale: float, eps: float = 1e-14) -> float:
    bound = 16.0 * sigma_scale
    whole = _simpson(f, -bound, bound)
    return _adaptive_simpson(f, -bound, bound, eps, whole, depth=28)


def conditioned_coefficients_from_integral(y: float, model: GaussianWeakMeasurementModel) -> Tuple[float, float, float]:
    s2 = model.sigma * model.sigma
    t2 = model.tau * model.tau

    def p_x_given_z(x: float, z: int) -> float:
        return gaussian_pdf(x, model.g * z, s2)

    def p_y_given_x(x: float) -> float:
        return gaussian_pdf(y, x, t2)

    def i_plus(x: float) -> float:
        return p_y_given_x(x) * p_x_given_z(x, +1)

    def i_minus(x: float) -> float:
        return p_y_given_x(x) * p_x_given_z(x, -1)

    def i_off(x: float) -> float:
        amp = math.exp(-((x - model.g) * (x - model.g) + (x + model.g) * (x + model.g)) / (4.0 * s2))
        amp /= math.sqrt(2.0 * math.pi * s2)
        return p_y_given_x(x) * amp

    scale = math.sqrt(s2 + t2)
    p_plus = integrate_real_line(i_plus, scale)
    p_minus = integrate_real_line(i_minus, scale)
    off = integrate_real_line(i_off, scale)

    norm = 0.5 * (p_plus + p_minus)
    return p_plus / (2.0 * norm), p_minus / (2.0 * norm), off / (2.0 * norm)


def decoder_success_probability(n: int, model: GaussianWeakMeasurementModel) -> float:
    return 0.5 * (1.0 + math.erf(math.sqrt(n * model.gamma_vis)))


def monte_carlo_qnd(n: int, model: GaussianWeakMeasurementModel, shots: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    success = 0
    sum_ratio = 0.0

    for _ in range(shots):
        z = 1 if rng.random() < 0.5 else -1
        s = 0.0
        ratio_prod = 1.0
        for __ in range(n):
            y = rng.gauss(model.g * z, model.Sigma)
            s += y
            p_plus = p_y_given_z(y, +1, model)
            p_minus = p_y_given_z(y, -1, model)
            norm = 0.5 * (p_plus + p_minus)
            b_noisy = model.chi * math.sqrt(p_plus * p_minus) / (2.0 * norm)
            b_visible = math.sqrt(p_plus * p_minus) / (2.0 * norm)
            ratio_prod *= b_noisy / b_visible
        decode = 1 if s >= 0.0 else -1
        if decode == z:
            success += 1
        sum_ratio += ratio_prod

    p_dec_mc = success / shots
    coh_mc = sum_ratio / shots
    return {
        "p_dec_mc": p_dec_mc,
        "p_dec_exact": decoder_success_probability(n, model),
        "coh_mc": coh_mc,
        "coh_exact": math.exp(-n * model.gamma_hid),
    }


def sweep_decoder(model: GaussianWeakMeasurementModel, n_values: Iterable[int]) -> List[Tuple[int, float, float]]:
    out: List[Tuple[int, float, float]] = []
    for n in sorted(n_values):
        out.append((n, n * model.gamma_vis, decoder_success_probability(n, model)))
    return out


def max_relative_error(a: Sequence[float], b: Sequence[float]) -> float:
    m = 0.0
    for ai, bi in zip(a, b):
        denom = max(abs(ai), 1e-16)
        m = max(m, abs((bi - ai) / denom))
    return m
