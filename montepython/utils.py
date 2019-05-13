#!/usr/bin/env python
# MCMC utilities.

import warnings
import numpy as np

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

def autocorrelation_time(acors):
    n_dim = acors.shape[1]
    c = 5 # A constant, see Sokal
    for M in np.arange(10, 10000):
        taus = np.zeros(n_dim)
        for d in np.arange(n_dim):
            taus[d] = 1 + 2*np.sum(acors[1:M, d])
        if np.all(taus > 1.0) and M > c*taus.max():
            return taus
    raise ValueError('Could not compute autocorrelation time')

def most_correlated(acors):
    taus = autocorrelation_time(acors)
    return np.argmax(taus)
