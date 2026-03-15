import math

from src.eta_hybrid_mps_referencebit import (
    HybridMPSConfig,
    crossing_x,
    gamma_hidden,
    run_point,
    sigma_eff,
    simulate_trajectory,
    summarize_convergence,
)


def test_eta_identities():
    s = sigma_eff(1.3, 0.5)
    assert math.isclose(s * s, 1.3 * 1.3 / 0.5, rel_tol=0, abs_tol=1e-15)
    g = gamma_hidden(0.8, 1.1, 0.75)
    assert math.isclose(g, 0.8 * 0.8 * (1 - 0.75) / (2 * 1.1 * 1.1), rel_tol=0, abs_tol=1e-15)


def test_likelihood_is_finite_and_decoder_stable():
    cfg = HybridMPSConfig()
    out = simulate_trajectory(L=16, T=16, g=0.7, eta=0.6, chi=128, seed=1234, cfg=cfg)
    assert math.isfinite(out["logp0"])
    assert math.isfinite(out["logp1"])
    assert out["decoded"] in (0, 1)


def test_eta_one_baseline_beats_low_eta_in_decoder():
    cfg = HybridMPSConfig()
    ideal = run_point(L=24, T=24, g=0.72, eta=1.0, chi=128, ntraj=36, seed=200, cfg=cfg)
    noisy = run_point(L=24, T=24, g=0.72, eta=0.5, chi=128, ntraj=36, seed=200, cfg=cfg)
    assert ideal["P_dec"] >= noisy["P_dec"]


def test_state_rounding_with_hidden_dephasing():
    cfg = HybridMPSConfig()
    high = run_point(L=24, T=24, g=0.8, eta=1.0, chi=128, ntraj=32, seed=77, cfg=cfg)
    low = run_point(L=24, T=24, g=0.8, eta=0.25, chi=128, ntraj=32, seed=77, cfg=cfg)
    assert low["S_ref"] > high["S_ref"]


def test_convergence_summary_and_crossing():
    rows = [{"P_dec": 0.7, "S_ref": 0.4}, {"P_dec": 0.73, "S_ref": 0.38}, {"P_dec": 0.74, "S_ref": 0.37}]
    c = summarize_convergence(rows, [32, 64, 128])
    assert c["max_delta_P_dec"] > 0
    x = [0.2, 0.4, 0.6]
    y1 = [0.2, 0.5, 0.8]
    y2 = [0.7, 0.5, 0.3]
    assert math.isclose(crossing_x(x, y1, y2), 0.4, rel_tol=0, abs_tol=1e-15)
