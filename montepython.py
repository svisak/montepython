#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

class MontePython(ABC):

    def __init__(self, dim, startpos, prior, likelihood, args=[], kwargs={}):
        self.dim = dim
        self.prior = prior
        self.likelihood = likelihood
        self.chain = Chain(self.dim)
        self.args = args
        self.kwargs = kwargs
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_rate(self):
        return self.chain.acceptance_rate()

    def get_chain(self):
        return self.chain.get_chain()

    def posterior(self, position):
        return self.prior(position) + self.likelihood(position)
	
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

    def extend_chain(self, n):
        self.chain = np.concatenate((self.chain, np.zeros((n, self.dim))))

    def index(self):
        return self.index

    def acceptance_rate(self):
        return self.n_accepted / len(self.chain) # Check this

    def dimensionality(self):
        return self.dim
