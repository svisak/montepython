#!/usr/bin/env python

from .mcmc import MCMC
import numpy as np
from numpy.random import multivariate_normal

class RWM(MCMC):

    def __init__(self, stepsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._covariance = stepsize * np.eye(self.chain.dimensionality())

    def run(self, n_steps):
        self._chain.extend(n_steps)
        for i in range(n_steps):
            # PROPOSE NEW STATE
            position = self._chain.head()
            proposed_position = multivariate_normal(position, self._covariance)

            # ACCEPTANCE PROBABILITY
            current_position = self._chain.head()
            lnposterior_diff = self.lnposterior(proposed_position)
            lnposterior_diff -= self.lnposterior(current_position)
            metropolis_ratio = np.exp(lnposterior_diff)
            acceptance_probability = min(1, metropolis_ratio)

            # ACCEPT / REJECT
            if np.random.rand() < acceptance_probability:
                self._chain.accept(proposed_position)
            else:
                self._chain.reject()
