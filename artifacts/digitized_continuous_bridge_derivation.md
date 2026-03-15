# Digitized continuous measurement bridge (v2)

Exact one-bin identities used:
1) p_±(r)=N(±2 sqrt(eta kappa) dt, dt)
2) chi=exp[-2(1-eta)kappa dt]
3) offdiag_th(r)=chi*sqrt(p_+ p_-)/(p_+ + p_-)
4) blochx_th(r)=2*offdiag_th(r)
5) E[x]=exp(-2kappa dt) for x-initialized QND benchmark
6) (E[r|+y]-E[r|-y])/(2 dt^2)=sqrt(eta kappa) omega + O(dt)

Patch result:
- Comparator mismatch fixed: x-conditioned observable is compared to blochx_th (not offdiag).
- QND benchmark now behaves as exact up to Monte Carlo/binning noise.
- Non-QND benchmark uses y-initialized record moment instead of blind split residual fitting.
- Fine-dt residual fit: |delta_record_y - sqrt(eta kappa)omega| ~ 3.032e-02 * dt^-0.657.