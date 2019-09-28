# MCMC diagnostic utilities.

import warnings
import numpy as np
import scipy

def autocorrelation(chain, max_lag):
    """
    Return an ndarray containing the autocorrelations for each
    dimension of the chain separately.

    The shape of the returned array is (max_lag+1, ndim).
    """
    ndim = chain.shape[1]
    acors = np.empty((max_lag+1, ndim))
    if max_lag > len(chain)/5:
        warnings.warn('max_lag is more than one fifth the chain length')
    for dim in range(ndim):
        chain1d = chain[:, dim] - np.average(chain[:, dim])
        for lag in range(max_lag+1):
            unshifted = None
            shifted = chain1d[lag:]
            if 0 == lag:
                unshifted = chain1d
            else:
                unshifted = chain1d[:-lag]
            normalization = np.sqrt(np.dot(unshifted, unshifted))
            normalization *= np.sqrt(np.dot(shifted, shifted))
            acors[lag, dim] = np.dot(unshifted, shifted) / normalization
    return acors

def exponential_function(lag, tao, offset):
    return np.exp(-lag/tao) + offset

def autocorrelation_time(acors):
    tao = scipy.optimize.curve_fit(exponential_function, lags, acors)[0][0]
    return tao

def most_correlated(acors):
    maximum = 0
    most = -1
    ndim = acors.shape[1]
    for dim in range(ndim):
        total = np.sum(np.abs(acors[:, dim]))
        if np.abs(total) > maximum:
            maximum = total
            most = dim
    return most

def relative_error(chain):
    n_samples = len(chain)
    mean = np.mean(chain, axis=0)
    dev = np.std(chain, axis=0, ddof=1)
    return dev / np.sqrt(n_samples) / np.abs(mean)
