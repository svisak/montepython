#!/usr/bin/env python
# MCMC utilities.

import warnings
import numpy as np
import scipy

def autocorrelation(chain, max_lag):
    dimensions = chain.shape[1]
    acors = np.empty((max_lag+1, dimensions))
    if max_lag > len(chain)/5:
        warnings.warn('max_lag is more than one fifth the chain length')
    for dim in range(dimensions):
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
    tao = scipy.optimize.curve_fit(exp_func, lags, acors)[0][0]
    return tao

def most_correlated(acors):
    taus = autocorrelation_time(acors)
    return np.argmax(taus)
