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
            posterior_diff = self.lnposterior(proposed_position)
            posterior_diff -= self.lnposterior(current_position)
            metropolis_ratio = min(1, np.exp(posterior_diff))

            # ACCEPT / REJECT
            if np.random.rand() < metropolis_ratio:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()
