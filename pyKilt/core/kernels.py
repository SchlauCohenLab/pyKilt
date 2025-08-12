"""IRF / kernel utilities."""

import numpy as np
from scipy.signal import convolve

def exp_kernel(time: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Exponential kernel K(t, τ) = exp(-t/τ)."""
    return np.exp(-np.outer(time, 1.0 / tau))

def gaussian_irf(time: np.ndarray, center: float, width: float) -> np.ndarray:
    irf = np.exp(-((time - center) ** 2) / (2 * width ** 2))
    s = irf.sum()
    if s == 0:
        return irf
    return irf / s

def make_irf_matrix_conv(K: np.ndarray, irf: np.ndarray, conv_pad: int = 10) -> np.ndarray:
    """
    Produce K_irf by convolving each column of K with irf.
    K : (len(time), len(tau))
    irf : 1D array same length as time
    """
    tlen, tau_bins = K.shape
    irf_matrix = np.zeros_like(K)
    # pad arrays to avoid edge artifacts
    pad = conv_pad
    irf_padded = np.pad(irf, pad, mode='constant', constant_values=0.0)
    for i in range(tau_bins):
        col = K[:, i]
        col_padded = np.pad(col, pad, mode='constant', constant_values=0.0)
        conv = convolve(col_padded, irf_padded, mode='full')
        # centered slice
        start = pad + pad
        irf_matrix[:, i] = conv[start:start + tlen]
        s = irf_matrix[:, i].sum()
        if s > 0:
            irf_matrix[:, i] /= s
    return irf_matrix

def toeplitz_irf_matrix(irf_shifted: np.ndarray, tau_bins: int) -> np.ndarray:
    """Return toeplitz IRF matrix if needed (small helper)."""
    from scipy.linalg import toeplitz
    return toeplitz(np.r_[irf_shifted[0], np.zeros(tau_bins - 1)], irf_shifted)
