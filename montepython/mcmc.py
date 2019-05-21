#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

class MCMC(ABC):

    def __init__(self, dim, startpos, lnprior, lnlikelihood, args=[], kwargs={}):
        self.args = args
        self.kwargs = kwargs
        self.lnprior = lnprior
        self.lnlikelihood = lnlikelihood

        # Create chain and add startpos to it
        self.chain = Chain(dim)
        self.chain.extend(1)
        self.chain.accept(startpos)
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_rate(self):
        return self.chain.acceptance_rate()

    def get_chain(self):
        return self.chain.get_chain()

    def lnposterior(self, position):
        # CONVENIENCE
        lnprior_val = self.lnprior(position)
        lnlikelihood_val = self.lnlikelihood(position)

        # CHECK INF/NAN
        if np.isinf(lnprior_val):
            return lnprior_val
        elif np.isinf(lnlikelihood_val):
            return lnlikelihood_val
        if np.isnan(lnprior_val) or np.isnan(lnlikelihood_val):
            raise ValueError('NaN encountered in lnposterior')

        # OK, RETURN LN OF POSTERIOR
        return lnprior_val + lnlikelihood_val

    @abstractmethod
    def get_mcmc_type(self):
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def run(self, n_steps):
        raise NotImplementedError("Unimplemented abstract method!")


class Chain():

    def __init__(self, dim):
        self.dim = dim
        self.chain = np.empty((0, self.dim))
        self.n_accepted = 0
        self.index = -1

    def accept(self, position):
        self.chain[self.index+1, :] = position
        self.n_accepted += 1
        self.index += 1

    def reject(self):
        self.chain[self.index+1, :] = self.head()
        self.index += 1

    def head(self):
        return self.chain[self.index, :]

    def get_chain(self):
        return self.chain

    def extend(self, n):
        self.chain = np.concatenate((self.chain, np.zeros((n, self.dim))))

    def acceptance_rate(self):
        return self.n_accepted / (self.index+1)

    def dimensionality(self):
        return self.dim
