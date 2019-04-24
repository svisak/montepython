#!/usr/bin/env python

import montepython
import numpy as np

class RWM(montepython.MontePython):

    def __init__(self, stepsize, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stepsize = stepsize

    def propose(self):
        cov = self.stepsize * np.eye(self.dim)
        return np.random.multivariate_normal(self.chain.head(), cov)

    def run(self, n_steps):
        self.chain.extend(n_steps)
        for i in range(1, n_steps):
            proposed_position = self.propose()

            # Metropolis ratio
            current_position = self.chain.head()
            exponent = self.lnposterior(proposed_position) - self.lnposterior(current_position)
            ratio = 1.
            if exponent < 0.:
                ratio = np.exp(exponent)

            # Accept/reject
            if np.random.rand() < ratio:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()
