#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

class MCMC(ABC):
    """
    Abstract MCMC base class.
    This is the only class the user interacts with (beyond instantiation.)

    """

    def __init__(self, dim, startpos, lnprior, lnlikelihood, args=[], kwargs={}):
        self.args = args
        self.kwargs = kwargs
        self.lnprior = lnprior
        self.lnlikelihood = lnlikelihood

        # Create chain and add startpos to it
        self._metachain = MetaChain(dim)
        self._metachain.extend(1)
        self._metachain.accept(startpos)
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_fraction(self):
        return self._metachain.acceptance_fraction()

    def get_chain(self):
        return self._metachain.chain()

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
    def to_ugly_string(self):
        """Return a description, suitable for filenames, of the MCMC object."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def get_mcmc_type(self):
        """Return a string with the name of the MCMC algorithm."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def run(self, n_steps):
        """
        Run the MCMC algorithm.

        :param n_steps: Number of MCMC steps to perform
        :returns: Nothing
        """
        raise NotImplementedError("Unimplemented abstract method!")


class MetaChain():

    def __init__(self, dim):
        self._dim = dim
        self._chain = np.empty((0, self._dim))
        self._n_accepted = 0
        self._index = -1

    def accept(self, position):
        self._chain[self._index+1, :] = position
        self._n_accepted += 1
        self._index += 1

    def reject(self):
        self._chain[self._index+1, :] = self.head()
        self._index += 1

    def head(self):
        return self._chain[self._index, :]

    def chain(self):
        return self._chain

    def extend(self, n):
        self._chain = np.concatenate((self._chain, np.zeros((n, self._dim))))

    def acceptance_fraction(self):
        return self._n_accepted / (self._index+1)

    def dimensionality(self):
        return self._dim

    def steps_taken(self):
        return self._index+1
