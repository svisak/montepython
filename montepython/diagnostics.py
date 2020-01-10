# MCMC diagnostic utilities.

import warnings
import numpy as np
import scipy

def autocorrelation(chain, max_lag):
    """
    Return an ndarray containing the autocorrelations for each
    dimension of the chain separately.

    The shape of the returned array is
        -> (max_lag+1, ndim) if the shape of chain is (n_samples, ndim)
        -> (max_lag+1,) if the shape of the chain is (n_samples,).
    """
    if max_lag > len(chain)/5:
        # The calculation doesn't make sense if
        # the maximum lag is too close to the full chain length
        warnings.warn('max_lag is more than one fifth the chain length')

    # Reshape chains of shape (length,) to (length, 1),
    # and remember that we did this
    reshaped = False
    ndim = 1
    try:
        ndim = chain.shape[1]
    except IndexError:
        chain = chain.reshape(-1, ndim)
        reshaped = True

    # Storage space for the autocorrelations
    acors = np.empty((max_lag+1, ndim))

    # Treat each dimension separately
    for dim in range(ndim):
        # Subtract the average
        chain1d = chain[:, dim] - np.average(chain[:, dim])
        # Calculate the autocorrelation for 0 <= lag <= max_lag
        for lag in range(max_lag+1):
            unshifted = None
            shifted = chain1d[lag:]
            if 0 == lag:
                unshifted = chain1d
            else:
                unshifted = chain1d[:-lag]
            # Normalization
            normalization = np.sqrt(np.dot(unshifted, unshifted))
            normalization *= np.sqrt(np.dot(shifted, shifted))
            # Save the result
            acors[lag, dim] = np.dot(unshifted, shifted) / normalization
    if reshaped is True:
        # Reshape acors to match the shape of the input chain
        acors = acors.reshape(-1,)
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
