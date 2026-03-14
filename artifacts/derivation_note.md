# Derivation note: Gaussian local benchmark

Verified exact factorization:
E_y^(g,sigma,tau) = D_chi o E_y^(g,Sigma,0)

Sigma^2 = sigma^2 + tau^2 = 1.300000000000000
chi = exp[-g^2 tau^2/(2 sigma^2 (sigma^2+tau^2))] = 0.754629057631985
gamma_vis = g^2/(2 Sigma^2) = 0.465384615384615
gamma_hid = -log(chi) = 0.281528964862298

Decoder benchmark:
P_dec_opt(n)=1/2*(1+erf(sqrt(n*gamma_vis)))

Post-selection coherence benchmark:
Conditioned noisy state equals visible conditioned state followed by D_{exp(-n gamma_hid)}.

Quadrature coefficient relative errors:
A_plus: 7.774e-12
A_minus: 7.774e-12
B: 5.335e-13

Raw Monte Carlo log: logs/monte_carlo_log.json