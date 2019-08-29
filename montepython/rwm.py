#!/usr/bin/env python

from .mcmc import MCMC
import numpy as np
from numpy.random import multivariate_normal

class RWM(MCMC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stepsize = kwargs.pop('stepsize', 1.0)
        self._covariance = stepsize * np.eye(self._metachain.dimensionality())

        # CALCULATE VALUE OF POSTERIOR AT STARTPOS
        tmp = self.lnposterior(self._metachain.startpos())
        self._remember_value(tmp)

    def to_ugly_string(self):
        n = self._metachain.chain_length()
        dim = self._metachain.dimensionality()
        stepsize = self._covariance[0, 0]
        str = "rwm_N{}_dim{}_stepsize{}".format(n, dim, stepsize)
        return str

    def to_pretty_string(self):
        n = self._metachain.chain_length()
        dim = self._metachain.dimensionality()
        stepsize = self._covariance[0, 0]
        str = "RWM, {} samples, stepsize {}".format(n, dim, stepsize)
        return str

    def get_mcmc_type(self):
        return "RWM"

    def sample(self):
        # PROPOSE NEW STATE
        current_position = self._metachain.head()
        proposed_position = multivariate_normal(current_position, self._covariance)

        # ACCEPTANCE PROBABILITY
        proposed_value = self.lnposterior(proposed_position)
        current_value = self._recall_value()
        lnposterior_diff = proposed_value - current_value
        # Let 1 be the maximum value of the Metropolis ratio
        # This is to prevent numerical issues since lnposterior_diff
        # can be a large positive number
        metropolis_ratio = 1
        if 0 > lnposterior_diff:
            metropolis_ratio = np.exp(lnposterior_diff)
        # Technically the acceptance probability is min(1, metropolis_ratio)
        acceptance_probability = metropolis_ratio

        # ACCEPT / REJECT
        if np.random.rand() < acceptance_probability:
            self._metachain.accept(proposed_position)
            self._remember_value(proposed_value)
        else:
            self._metachain.reject()
