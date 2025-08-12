"""Fast 2D fluorescence decay correlation (FD-2D) core routines."""

import numba as nb
import numpy as np
from typing import Dict

# numba-compiled core (identical logic to original)
@nb.njit(cache=True, fastmath=True, parallel=True)
def fdc_core(macrotimes, microtimes, dT, ddT, time, corr_maps):
    """
    Build correlation maps.

    Parameters
    ----------
    macrotimes : 1D array (float64)
    microtimes : 1D array (float64) same length as macrotimes
    dT : 1D array of delays (float64)
    ddT : float tolerance for pairing macrotimes
    time : 1D time axis (bin edges or centers)
    corr_maps : preallocated 3D array (len(dT), len(time), len(time)) to be filled
    """
    for n in nb.prange(dT.size):
        for i in nb.prange(macrotimes.size):
            t_start = macrotimes[i] + dT[n] - ddT
            t_end   = macrotimes[i] + dT[n] + ddT
            j0 = np.searchsorted(macrotimes, t_start, side='left')
            j1 = np.searchsorted(macrotimes, t_end,   side='right')

            if microtimes[i] < time[0] or microtimes[i] > time[-1]:
                continue

            k = np.searchsorted(time, microtimes[i], side='left') - 1
            if k < 0:
                continue

            for j in range(j0, j1):
                if i == j:
                    continue
                if microtimes[j] < time[0] or microtimes[j] >= time[-1]:
                    continue
                l = np.searchsorted(time, microtimes[j], side='right') - 1
                if l >= 0:
                    corr_maps[n, k, l] += 1


def corr_maps_from_pickles(ut_dict, mt_dict, delays, ddT, bins,
                                 tau_min=0.0, Ith=700, acquisition_time=30, N_hist=2000):
    """
    Wrapper that loops through molecules in ut_dict / mt_dict, filters and calls fdc_core.
    Returns corr_maps array.
    """
    tau_bins = len(bins) - 1
    corr_maps = np.zeros((len(delays), tau_bins, tau_bins), dtype=np.float64)
    # simple histogram accumulator for decay
    decay = np.zeros(tau_bins, dtype=np.float64)

    for mol in ut_dict.keys():
        macrotimes, microtimes = mt_dict[mol], ut_dict[mol]
        corr_maps_from_single_stream(macrotimes, microtimes, delays, ddT, bins,
                                         tau_min, Ith, acquisition_time, N_hist,
                                         corr_maps, decay)
    return corr_maps, decay

def corr_maps_from_single_stream(macrotimes, microtimes, delays, ddT, bins,
                                     tau_min=0.0, Ith=700, acquisition_time=30, N_hist=2000,
                                     corr_maps=None, decay=None):
    """
    Build correlation maps from a single photon stream.

    Parameters
    ----------
    macrotimes : 1D array (float64)
    microtimes : 1D array (float64) same length as macrotimes
    delays : 1D array of delays (float64)
    ddT : float tolerance for pairing macrotimes
    bins : 1D time axis (bin edges or centers)
    tau_min : minimum time threshold
    Ith : intensity threshold
    acquisition_time : acquisition time in seconds
    N_hist : number of histogram bins
    corr_maps : preallocated 3D array to accumulate correlations
    decay : preallocated 1D array to accumulate decay histogram

    Returns
    -------
    tuple : (corr_maps, decay) if corr_maps and decay were None, otherwise None
    """
    tau_bins = len(bins) - 1
    if corr_maps is None:
        corr_maps = np.zeros((len(delays), tau_bins, tau_bins), dtype=np.float64)
    if decay is None:
        decay = np.zeros(tau_bins, dtype=np.float64)

    macrotimes = np.nan_to_num(macrotimes, 0)
    macrotimes = macrotimes[macrotimes > 0]
    if macrotimes.size == 0:
        return corr_maps, decay
    macrotimes -= macrotimes.min()

    microtimes = np.nan_to_num(microtimes, 0) * 1e9  # convert if needed
    if microtimes.size == 0:
        return corr_maps, decay
    microtimes = microtimes % 12.5

    trace, tr_bins = np.histogram(macrotimes, bins=N_hist)
    trace = trace * N_hist / acquisition_time
    if np.mean(trace) < Ith:
        return corr_maps, decay

    mask = microtimes > tau_min
    macrotimes = macrotimes[mask]
    microtimes = microtimes[mask]
    microtimes -= tau_min

    if microtimes.size == 0:
        return corr_maps, decay

    decay += np.histogram(microtimes, bins=tau_bins)[0]

    macrotimes = macrotimes.astype(np.float64)
    microtimes = microtimes.astype(np.float64)

    fdc_core(macrotimes, microtimes, delays, ddT, bins, corr_maps)

    if 'corr_maps' in locals() and 'decay' in locals():
        return corr_maps, decay
    return None
