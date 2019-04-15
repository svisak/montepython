#!/usr/bin/env python

from abc import ABC, abstractmethod
import numpy as np

class MontePython(ABC):

    def __init__(self, dim, startpos, prior, likelihood, args=[], kwargs={}):
        self.dim = dim
        self.prior = prior
        self.likelihood = likelihood
        self.args = args
        self.kwargs = kwargs
        self._chain = np.empty((0, self.dim))
        self.status = ChainStatus(self.dim, startpos)
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_rate(self):
        return self.status.n_accepted() / len(self._chain) # Check this

    def chain(self):
        return self._chain

    def extend_chain(self, n):
        self._chain = np.concatenate((self._chain, np.zeros((n, self.dim))))

    def posterior(self, position):
        return prior(position) + likelihood(position)
	
    # Move to subclasses
    def metropolis_ratio(self, lnprob_proposed, lnprob_current):
        exponent = lnprob_proposed - lnprob_current
        if exponent >= 0.:
            return 1.
        return np.exp(exponent)
	
    # Move to subclasses
    def maybe_accept(self, proposed_position, ratio):
        if np.random.rand() < ratio:
            self.status.update_state(proposed_position)
        else:
            self.status.update_state()
        self._chain[self.status.index(), :] = self.status.position()

    # THIS SHOULD BE ABSTRACT
    def run(self, n_steps):
        extend_chain(n_steps)
        n_accepted = 0
        for i in range(0, n_steps-1):
            proposed_state = propose_state()
            ratio = metropolis_ratio(posterior(proposed_state),
                                     posterior(self.status.state()))
            maybe_accept(proposed_state, ratio)

    # Move to subclasses?
    @abstractmethod
    def propose_state(self):
        raise NotImplementedError("Unimplemented abstract method!")

class ChainStatus():

    def __init__(self, dim, startpos):
        self._dim = dim
        self._index = -1
        self._position = startpos
        self._n_accepted = 0

    def update_state(self):
        self._index++

    def update_state(self, position):
        self._index++
        self._state = position
        self._n_accepted++

    def index(self):
        return self._index

    def position(self):
        return self._position

    def n_accepted(self):
        return self._n_accepted
