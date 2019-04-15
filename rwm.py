#!/usr/bin/env python

from .montepython import MontePython

class RandomWalkMetropolis(MontePython):

    def __init__(self, *args, **kwargs, stepsize):
        super().__init__(*args, **kwargs)
        self.stepsize = stepsize

    def propose(self):
        cov = stepsize * np.eye(self.dim)
        return np.random.multivariate_normal(self.chain.head(), cov)

    def run(self, n_steps)
        extend_chain(n_steps)
        for i in range(0, n_steps-1):
            proposed_position = propose()

            # Metropolis ratio
            current_position = self.chain.head()
            exponent = posterior(proposed_position) - posterior(current_position)
            ratio = 1.
            if exponent < 0.:
                ratio = np.exp(exponent)

            # Accept/reject
            if np.random.rand() < ratio:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()
