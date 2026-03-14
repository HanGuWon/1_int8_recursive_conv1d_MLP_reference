import math

from src.digitized_continuous_bridge import (
    BridgeConfig,
    compare_one_bin,
    fit_correction_power_law,
    gaussian_model_params,
)


def test_gaussian_params_identity():
    cfg = BridgeConfig(kappa=0.8, eta=0.5, omega=0.0)
    p = gaussian_model_params(0.1, cfg)
    assert math.isclose(p["mu"], 2 * math.sqrt(cfg.eta * cfg.kappa) * 0.1, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["var"], 0.1, rel_tol=0, abs_tol=1e-15)


def test_qnd_split_residual_small():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=0.0, n_substeps=50)
    r = compare_one_bin(dt=0.08, ntraj=12000, seed=11, cfg=cfg)
    assert abs(r["split_residual"]) < 8e-3


def test_nonqnd_has_larger_coherence_error_than_qnd():
    qnd = BridgeConfig(kappa=0.8, eta=0.75, omega=0.0, n_substeps=45)
    non = BridgeConfig(kappa=0.8, eta=0.75, omega=1.2, n_substeps=45)
    r0 = compare_one_bin(dt=0.12, ntraj=10000, seed=31, cfg=qnd)
    r1 = compare_one_bin(dt=0.12, ntraj=10000, seed=31, cfg=non)
    assert r1["err_coherence"] > r0["err_coherence"]


def test_correction_fit_is_finite():
    cfg = BridgeConfig(kappa=0.8, eta=0.75, omega=1.2, n_substeps=40)
    dts = [0.04, 0.08, 0.12]
    errs = [abs(compare_one_bin(dt=d, ntraj=9000, seed=50 + i, cfg=cfg)["err_coherence"]) for i, d in enumerate(dts)]
    fit = fit_correction_power_law(dts, errs)
    assert math.isfinite(fit["power"])
    assert math.isfinite(fit["prefactor"])
