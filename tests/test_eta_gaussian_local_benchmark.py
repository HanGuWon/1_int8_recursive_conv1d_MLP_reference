import math

from src.eta_gaussian_local_benchmark import (
    GaussianWeakMeasurementModel,
    conditioned_coefficients,
    conditioned_coefficients_from_integral,
    decoder_success_probability,
    monte_carlo_qnd,
)


def test_closed_form_parameters():
    model = GaussianWeakMeasurementModel(g=1.3, sigma=0.8, tau=0.5)
    assert math.isclose(model.Sigma**2, model.sigma**2 + model.tau**2, rel_tol=0, abs_tol=1e-15)
    chi_target = math.exp(-(model.g**2) * (model.tau**2) / (2 * model.sigma**2 * (model.sigma**2 + model.tau**2)))
    assert math.isclose(model.chi, chi_target, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(model.gamma_vis, model.g**2 / (2 * model.Sigma**2), rel_tol=0, abs_tol=1e-15)
    assert math.isclose(model.gamma_hid, -math.log(model.chi), rel_tol=0, abs_tol=1e-15)


def test_exact_factorization_coefficients_match_quadrature():
    model = GaussianWeakMeasurementModel(g=1.1, sigma=0.9, tau=0.7)
    ys = [(-3.5 + 7.0 * i / 40.0) * model.Sigma for i in range(41)]
    for y in ys:
        exact = conditioned_coefficients(y, model)
        quad = conditioned_coefficients_from_integral(y, model)
        rel = [abs((q - e) / max(abs(e), 1e-16)) for e, q in zip(exact, quad)]
        assert max(rel) < 1e-10


def test_decoder_formula_matches_monte_carlo():
    model = GaussianWeakMeasurementModel(g=1.0, sigma=1.1, tau=0.6)
    n = 24
    shots = 180000
    res = monte_carlo_qnd(n=n, model=model, shots=shots, seed=2026)
    se = math.sqrt(res["p_dec_exact"] * (1 - res["p_dec_exact"]) / shots)
    assert abs(res["p_dec_mc"] - res["p_dec_exact"]) < 5.0 * se


def test_coherence_formula_matches_monte_carlo():
    model = GaussianWeakMeasurementModel(g=0.9, sigma=1.0, tau=0.4)
    n = 16
    res = monte_carlo_qnd(n=n, model=model, shots=200000, seed=77)
    assert abs(res["coh_mc"] - res["coh_exact"]) < 3e-3


def test_decoder_closed_form_expression():
    model = GaussianWeakMeasurementModel(g=0.7, sigma=0.8, tau=0.3)
    for n in [1, 2, 5, 10, 20]:
        p = decoder_success_probability(n, model)
        target = 0.5 * (1 + math.erf(math.sqrt(n * model.gamma_vis)))
        assert math.isclose(p, target, rel_tol=0, abs_tol=1e-15)
