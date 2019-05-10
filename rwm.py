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

            # ACCEPTANCE PROBABILITY
            current_position = self.chain.head()
            posterior_diff = self.lnposterior(proposed_position)
            posterior_diff -= self.lnposterior(current_position)
            metropolis_ratio = np.exp(posterior_diff)
            acceptance_probability = min(1, metropolis_ratio)

            # ACCEPT / REJECT
            if np.random.rand() < acceptance_probability:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()
