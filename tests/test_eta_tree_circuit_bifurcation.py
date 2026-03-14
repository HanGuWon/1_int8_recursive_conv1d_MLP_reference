import math

from src.eta_tree_circuit_bifurcation import (
    TreeCircuitConfig,
    child_to_parent_llr,
    find_crossing,
    hidden_dephasing_gamma,
    run_observables,
    sigma_eff_from_eta,
)


def test_sigma_eff_and_gamma_hid_identities():
    sigma = 1.2
    eta = 0.75
    g = 0.9
    sigma_eff = sigma_eff_from_eta(sigma, eta)
    assert math.isclose(sigma_eff * sigma_eff, sigma * sigma / eta, rel_tol=0, abs_tol=1e-15)
    gamma = hidden_dephasing_gamma(g, sigma, eta)
    assert math.isclose(gamma, g * g * (1 - eta) / (2 * sigma * sigma), rel_tol=0, abs_tol=1e-15)


def test_child_to_parent_message_is_odd_and_monotone():
    lam = 0.62
    vals = [child_to_parent_llr(x, lam) for x in [-3.0, -1.0, 0.0, 1.0, 3.0]]
    assert vals[0] < vals[1] < vals[2] < vals[3] < vals[4]
    assert math.isclose(vals[0], -vals[4], rel_tol=0, abs_tol=1e-12)


def test_ideal_efficiency_baseline_beats_noisy_readout_for_decoder():
    cfg = TreeCircuitConfig(lambda_edge=0.62, sigma=1.0)
    g = 0.55
    ideal = run_observables(depth=7, g=g, eta=1.0, shots=1400, seed=100, cfg=cfg)
    noisy = run_observables(depth=7, g=g, eta=0.5, shots=1400, seed=100, cfg=cfg)
    assert ideal["P_dec"] > noisy["P_dec"]


def test_finite_efficiency_reduces_state_purification():
    cfg = TreeCircuitConfig(lambda_edge=0.62, sigma=1.0)
    g = 0.8
    high_eta = run_observables(depth=6, g=g, eta=1.0, shots=1600, seed=17, cfg=cfg)
    low_eta = run_observables(depth=6, g=g, eta=0.5, shots=1600, seed=17, cfg=cfg)
    assert low_eta["state_purif"] < high_eta["state_purif"]


def test_crossing_interpolation():
    x = [0.0, 1.0, 2.0]
    y1 = [0.0, 0.5, 1.0]
    y2 = [1.0, 0.5, 0.0]
    c = find_crossing(x, y1, y2)
    assert math.isclose(c, 1.0, rel_tol=0, abs_tol=1e-15)
