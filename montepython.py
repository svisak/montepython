#!/usr/bin/env python
"""
Base sampling class for others to inherit from.
"""

class MontePython():

    def __init__(self, dim, prior, likelihood, args=[], kwargs={}):
        self.dim = dim
        self.prior = prior
        self.likelihood = likelihood
        self.args = args
        self.kwargs = kwargs

    def run(self, n_samples):
        # TODO
