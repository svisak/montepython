#!/usr/bin/env python
# Base sampling class for others to inherit from.

class MontePython():
    def __init__(self, dim, prior, likelihood, args=[], kwargs={}):
        self.dim = dim
        self.prior = prior
        self.likelihood = likelihood
        self.args = args
        self.kwargs = kwargs

    def sample(self, ):
        raise NotImplementedError('Method sample not implemented')

    def run(self, n_samples):
        # TODO

    def metropolis_ratio(self, pi_proposed, pi_current):
        quotient = pi_proposed - pi_current
        if quotient >= 1:
            return 1.0
        else:
            return quotient

    def posterior(self, position):
        return prior(position) + likelihood(position) # + or * depends on log or lin!
