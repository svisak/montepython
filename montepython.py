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
    
    def set_seed(self, seed):
        np.random.seed(seed)

    def acceptance_rate(self):
        return self.status.n_accepted() / len(self._chain)

    def chain(self):
        return self._chain

    def extend_chain(self, n):
        self._chain = np.concatenate((self._chain, np.zeros((n, self.dim))))

    def posterior(self, state):
        return prior(state) + likelihood(state)
	
    def maybe_accept(self, proposed_state, ratio):
        if np.random.rand() < ratio:
            self.status.update_state(proposed_state)
        else:
            self.status.update_state()
        self._chain[self.status.index(), :] = self.status.state()

    def run(self, n_steps):
        extend_chain(n_steps)
        for i in range(0, n_steps-1):
            proposed_state = propose_state()
            ratio = metropolis_ratio(posterior(proposed_state),
                                     posterior(self.status.state()))
            maybe_accept(proposed_state, ratio)

    @abstractmethod
    def propose_state(self):
        raise NotImplementedError("Unimplemented abstract method!")
    
    @abstractmethod
    def metropolis_ratio(self, *args, **kwargs):
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def run(self, n_steps):
        raise NotImplementedError("Unimplemented abstract method!")

class Status():

    def __init__(self, state):
        self._dim = dim
        self._index = -1
        self._state = state
        self._n_accepted = 0

    def update_state(self):
        self._index++

    def update_state(self, new_state):
        self._index++
        self._state = new_state
        self._n_accepted++

    def index(self):
        return self._index

    def state(self):
        return self._state

    def n_accepted(self):
        return self._n_accepted

class State():
    def __init__(self, position):
