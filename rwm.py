#!/usr/bin/env python

import montepython
import numpy as np

class RWM(montepython.MontePython):

    def __init__(self, stepsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stepsize = stepsize

    def propose(self):
        cov = self.stepsize * np.eye(self.chain.dimensionality())
        return np.random.multivariate_normal(self.chain.head(), cov)

    def run(self, n_steps):
        self.chain.extend(n_steps)
        for i in range(n_steps):
            # PROPOSE NEW STATE
            proposed_position = self.propose()

            # METROPOLIS RATIO
            current_position = self.chain.head()
            exponent = self.lnposterior(proposed_position) - self.lnposterior(current_position)
            metropolis_ratio = 1.
            if exponent < 0.:
                metropolis_ratio = np.exp(exponent)

            # ACCEPT / REJECT
            if np.random.rand() < metropolis_ratio:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()
