#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

class MontePython(ABC):

    def __init__(self, dim, startpos, lnprior, lnlikelihood, args=[], kwargs={}):
        self.args = args
        self.kwargs = kwargs
        self.dim = dim
        self.lnprior = lnprior
        self.lnlikelihood = lnlikelihood

        # Create chain and add startpos to it
        self.chain = Chain(self.dim)
        self.chain.extend(1)
        self.chain.accept(startpos)
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_rate(self):
        return self.chain.acceptance_rate()

    def get_chain(self):
        return self.chain.get_chain()

    def lnposterior(self, position):
        if np.isinf(self.lnprior(position)):
            return self.lnprior(position)
        elif np.isinf(self.lnlikelihood(position)):
            return self.lnlikelihood(position)
        else:
            return self.lnprior(position) + self.lnlikelihood(position)

    @abstractmethod
    def propose(self):
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

    def current_index(self):
        return self.index

    def acceptance_rate(self):
        return self.n_accepted / (self.index+1)

    def dimensionality(self):
        return self.dim
