import math

from src.digitized_continuous_bridge import (
    BridgeConfig,
    compare_one_bin,
    gaussian_conditioned_bloch_x,
    gaussian_conditioned_offdiag,
    gaussian_model_params,
    simulate_y_probe,
)


def test_gaussian_params_and_offdiag_identity():
    cfg = BridgeConfig(kappa=0.8, eta=0.5, omega=0.0)
    p = gaussian_model_params(0.1, cfg)
    assert math.isclose(p["mu"], 2 * math.sqrt(cfg.eta * cfg.kappa) * 0.1, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["var"], 0.1, rel_tol=0, abs_tol=1e-15)
    r = 0.03
    assert math.isclose(gaussian_conditioned_bloch_x(r, 0.1, cfg), 2 * gaussian_conditioned_offdiag(r, 0.1, cfg), rel_tol=0, abs_tol=1e-15)


def test_qnd_blochx_comparator_fixed():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=0.0, n_substeps=50)
    r = compare_one_bin(dt=0.08, ntraj=12000, seed=11, cfg=cfg)
    assert r["err_blochx"] < 0.02


def test_qnd_split_residual_small():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=0.0, n_substeps=50)
    r = compare_one_bin(dt=0.08, ntraj=12000, seed=11, cfg=cfg)
    assert abs(r["split_residual"]) < 1.5e-2


def test_nonqnd_y_probe_has_correct_sign_and_scale():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=1.2, n_substeps=45)
    dt = 0.06
    out = simulate_y_probe(dt=dt, ntraj=14000, seed=91, cfg=cfg)
    assert out["delta_record_y"] > 0
    assert abs(out["delta_record_y_residual"]) < 0.5


def test_fine_dt_probe_residual_is_finite():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=1.2, n_substeps=45)
    small = simulate_y_probe(dt=0.06, ntraj=16000, seed=101, cfg=cfg)
    assert math.isfinite(small["delta_record_y_residual"])
    assert abs(small["delta_record_y_residual"]) < 1.0
