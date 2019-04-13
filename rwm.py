#!/usr/bin/env python

from .montepython import MontePython

class RandomWalkMetropolis(MontePython):

    def __init__(self, *args, **kwargs, stepsize):
        super().__init__(*args, **kwargs)
        self.status = Status(self.dim, startpos)
        self.stepsize = stepsize

    def metropolis_ratio(self, proposed_position):
        current_position = self.status.state().position()
        exponent = posterior(proposed_position) - posterior(current_state)
        if exponent >= 0.:
            return 1.
        return np.exp(exponent)
	
    def propose_state(self):
        cov = stepsize * np.eye(self.dim)
        return np.random.multivariate_normal(self.status.state(), cov)

    def run(self, n_steps)
        extend_chain(n_steps)
        for i in range(0, n_steps-1):
            proposed_state = propose_state()
            ratio = metropolis_ratio(proposed_state)
            maybe_accept(proposed_state, ratio)
