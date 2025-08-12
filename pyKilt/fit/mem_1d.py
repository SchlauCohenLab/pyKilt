"""1D maximum-entropy (MEM) ILT fittings using lmfit / scipy.optimize."""

import numpy as np
from lmfit import minimize, Parameters
from scipy.signal import convolve
from typing import Tuple, Dict
from ..core.kernels import exp_kernel, make_irf_matrix_conv
from ..types import ILT1DConfig, ILTResult

def entropy_1d(A: np.ndarray, m_prior: np.ndarray) -> float:
    A_clipped = np.clip(A, 1e-12, None)
    return np.sum(A_clipped - m_prior - A_clipped * np.log(A_clipped / (m_prior + 1e-12)))

def reconvolve_1d(A: np.ndarray, K_irf: np.ndarray, y0: float) -> np.ndarray:
    return K_irf @ A + y0

def mem_objective_1d(A_arr, ExpData, K_irf, y0, m_prior, eta, sigma2=1.0):
    A = np.array(A_arr)
    model = reconvolve_1d(A, K_irf, y0)
    chi2 = np.sum((ExpData - model) ** 2 / (model + sigma2))
    ent = entropy_1d(A, m_prior)
    return chi2 - eta * ent

def fit_single_eta(ExpData, K_irf, y0, m_prior, eta, A_init=None, maxiter=1000):
    """
    Fit for a single eta value using lmfit minimize. Returns A_fit and result dict.
    """
    tau_bins = K_irf.shape[1]
    if A_init is None:
        A_init = np.ones(tau_bins)
        A_init = A_init / A_init.sum()
    params = Parameters()
    for i in range(tau_bins):
        params.add(f'A{i}', value=float(A_init[i]), min=0)
    res = minimize(mem_objective_1d, params, args=(ExpData, K_irf, y0, m_prior, eta), kws={'maxiter': maxiter})
    A_fit = np.array([res.params[f'A{i}'].value for i in range(tau_bins)])
    result = {
        'success': res.success,
        'message': res.message,
        'chisqr': getattr(res, 'chisqr', np.nan),
        'redchi': getattr(res, 'redchi', np.nan),
        'res': res,
    }
    return A_fit, result

def scan_irf_and_eta_1d(ExpData: np.ndarray, time: np.ndarray, tau: np.ndarray, irf: np.ndarray,
                        config: ILT1DConfig) -> ILTResult:
    """
    Full scan over IRF shifts and eta grid. Returns ILTResult with best A and metadata.
    """
    # defaults
    if config.eta_grid is None:
        config.eta_grid = np.logspace(-8, -4, 5)
    if config.irf_shift_list is None:
        config.irf_shift_list = np.arange(0, 8)
    if config.m_prior is None:
        config.m_prior = np.ones_like(tau) / len(tau)

    # precompute base kernel
    K = exp_kernel(time, tau)  # (t, τ)

    best_loss = np.inf
    best = None
    loss_history = []

    for s_idx, shift_val in enumerate(config.irf_shift_list):
        # align IRF by shift (roll)
        Rise_FL = int(np.argmax(ExpData))
        Rise_IRF = int(np.argmax(irf)) + int(shift_val)
        irf_shifted = np.roll(irf, Rise_FL - Rise_IRF)
        K_irf = make_irf_matrix_conv(K, irf_shifted, conv_pad=config.conv_pad)

        for eta in config.eta_grid:
            if config.verbose:
                print(f"[1D] IRF shift {shift_val} | eta {eta:.2e}")
            A_init = config.m_prior.copy()
            A_fit, info = fit_single_eta(ExpData, K_irf, config.y0, config.m_prior, eta, A_init=A_init, maxiter=config.maxiter)
            model = reconvolve_1d(A_fit, K_irf, config.y0)
            # simple loss metric (chi2 - eta*entropy)
            chi2 = np.sum((ExpData - model) ** 2 / (model + 1.0))
            ent = entropy_1d(A_fit, config.m_prior)
            loss = chi2 - eta * ent
            loss_history.append(loss)
            if config.verbose:
                print(f"   -> loss {loss:.4g}, chisq {chi2:.4g}")
            if loss < best_loss:
                best_loss = loss
                best = dict(A=A_fit, model=model, irf_shift=shift_val, eta=eta, info=info)

    result = ILTResult(
        best_params={'irf_shift': best['irf_shift'], 'eta': best['eta']},
        best_loss=best_loss,
        loss_history=loss_history,
        best_irf_shift=best['irf_shift'],
        best_eta=best['eta'],
        outputs={'A': best['A'], 'fit_curve': best['model'], 'info': best['info']},
    )
    return result
