"""2D ILT using JAX + jaxopt LBFGS (optional)."""

from typing import List
import numpy as np

# JAX imports are optional — guard them to allow the package to import even if JAX isn't present.
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
    from jaxopt import LBFGS
    JAX_AVAILABLE = True
except Exception as e:
    JAX_AVAILABLE = False

from ..types import ILT2DConfig, ILTResult

if JAX_AVAILABLE:
    def pack_jax(A, Ulist):
        tri_idx = jnp.triu_indices(A.shape[0])
        parts = [A.ravel()]
        for U in Ulist:
            parts.append(U[tri_idx])
        return jnp.concatenate(parts)

    def unpack_jax(theta, tau_bins, delay_bins):
        tri_idx = jnp.triu_indices(tau_bins)
        tri_len = tri_idx[0].size
        A = theta[:tau_bins * tau_bins].reshape(tau_bins, tau_bins)
        Ulist = []
        offset = tau_bins * tau_bins
        for _ in range(delay_bins):
            u_flat = theta[offset: offset + tri_len]
            U = jnp.zeros((tau_bins, tau_bins))
            U = U.at[tri_idx].set(u_flat)
            Ulist.append(U)
            offset += tri_len
        return A, Ulist

    @jit
    def model_maps(A, Ulist, K_irf):
        G_stack = jnp.stack([U @ U.T for U in Ulist])   # (ΔT, τ, τ)
        M_tau = A @ G_stack @ A.T                       # (ΔT, τ, τ)
        return K_irf @ M_tau @ K_irf.T                  # (ΔT, t, t)

    @jit
    def entropy_jax(A, m_prior):
        return jnp.sum(jnp.where(A > 1e-12,
                                 A - m_prior - A * jnp.log(A / (m_prior + 1e-12)),
                                 0.0))

    @jit
    def loss_jax(theta, eta, m_prior, corr_maps, K_irf, sigma2=1.0):
        A, Ulist = unpack_jax(theta, m_prior.shape[0], corr_maps.shape[0])
        sim = model_maps(A, Ulist, K_irf)
        chi2 = jnp.mean((corr_maps - sim)**2 / (sim + sigma2))
        ent = entropy_jax(A, m_prior)
        return chi2 - eta * ent

    def value_and_grad_fun_jax(theta, eta, m_prior, corr_maps, K_irf):
        f = lambda th: loss_jax(th, eta, m_prior, corr_maps, K_irf)
        return float(f(theta)), jax.grad(f)(theta)

    def initialize_theta(A0, delay_bins):
        tau_bins = A0.shape[0]
        tri_idx = jnp.triu_indices(tau_bins)
        tri_len = tri_idx[0].size
        parts = [A0.ravel()]
        for _ in range(delay_bins):
            U0 = jnp.eye(tau_bins)
            parts.append(U0[tri_idx])
        return jnp.concatenate(parts)

    def initialize_theta(tau, init_states, width):
        X, Y = jnp.meshgrid(tau, tau)
        res = 0
        for p0 in init_states:
            res += jnp.exp(-((X-p0)**2 + (Y-p0)**2)/(2*width**2))
        return res

    def scan_irf_and_eta_2d(corr_maps, time, tau, irf, A0=None, config: ILT2DConfig) -> ILTResult:
        """
        Scan IRF shifts & eta for 2D maps. Requires jax+jaxopt.
        """

        if not JAX_AVAILABLE:
            raise RuntimeError("scan_irf_and_eta_2d requires jax and jaxopt. Install them first.")

        # convert to jax arrays
        corr = jnp.array(corr_maps)
        tau_bins = len(tau)
        delay_bins = corr.shape[0]
        K = jnp.exp(-jnp.outer(time, 1.0 / tau))

        if config.eta_grid is None:
            import numpy as onp
            config.eta_grid = onp.logspace(-12, -8, 3)
        if config.irf_shift_list is None:
            import numpy as onp
            config.irf_shift_list = onp.arange(0, 3)
        if config.m_prior is None:
            config.m_prior = jnp.ones((tau_bins, tau_bins))

        best_Q = jnp.inf
        best_results = None
        loss_history = []

        if not A0:
            A0 = jnp.ones((tau_bins, tau_bins))

        for shift in config.irf_shift_list:
            # align irf
            Rise_FL = int(jnp.argmax(corr[0].diagonal()))
            Rise_IRF = int(jnp.argmax(irf)) + int(shift)
            irf_shifted = jnp.roll(irf, Rise_FL - Rise_IRF)
            irf_matrix = toeplitz(np.r_[irf_shifted[0], np.zeros(tau_bins - 1)], irf_shifted)
            # simple K_irf placeholder: apply K without full conv for speed here
            K_irf = jnp.array(irf_shifted.T) @ K + 1e-2

            for eta in config.eta_grid:
                # initialize A0 and U0
                X, Y = jnp.meshgrid(tau, tau)
                U0 = [jnp.eye(tau_bins)] * delay_bins
                theta0 = initialize_theta(A0, delay_bins)

                # LBFGS solver wrapper
                solver = LBFGS(fun=lambda th: (loss_jax(th, eta, config.m_prior, corr, K_irf), jax.grad(lambda th: loss_jax(th, eta, config.m_prior, corr, K_irf))(th)),
                               value_and_grad=False, maxiter=config.maxiter)

                sol = solver.run(theta0)
                theta_opt = sol.params
                Q_val = sol.state.value
                loss_history.append(float(Q_val))
                if Q_val < best_Q:
                    best_Q = Q_val
                    best_results = (theta_opt, K_irf, irf_shifted, eta, shift)

        if best_results is None:
            raise RuntimeError("Optimization failed to find a solution.")

        theta_best, K_irf_best, irf_best, best_eta, best_shift = best_results
        A_final, U_final = unpack_jax(theta_best, tau_bins, delay_bins)
        ilt_maps = jnp.stack([U @ U.T for U in U_final])
        model_out = model_maps(A_final, U_final, K_irf_best)

        return ILTResult(
            best_params={'eta': float(best_eta)},
            best_loss=float(best_Q),
            loss_history=[float(x) for x in loss_history],
            best_irf_shift=int(best_shift),
            best_eta=float(best_eta),
            outputs={'A_final': np.array(A_final), 'U_final': [np.array(U) for U in U_final],
                     'ilt_maps': np.array(ilt_maps), 'model_out': np.array(model_out)}
        )
else:
    def scan_irf_and_eta_2d(*args, **kwargs):
        raise RuntimeError("2D ILT requires jax and jaxopt. Install them to use scan_irf_and_eta_2d.")
