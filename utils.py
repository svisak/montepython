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

def most_correlated(acors):
    maximum = 0
    most = -1
    dimensions = acors.shape[1]
    for dim in range(dim):
        total = np.sum(np.abs(acors[:, dim]))
        if np.abs(total) > maximum:
            maximum = total
            most = dim
    return most
