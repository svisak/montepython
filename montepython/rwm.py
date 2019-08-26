#!/usr/bin/env python

from .mcmc import MCMC
import numpy as np
from numpy.random import multivariate_normal

class RWM(MCMC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stepsize = kwargs.pop('stepsize', 1.0)
        self._covariance = stepsize * np.eye(self._metachain.dimensionality())
        self._current_lnposterior_value = None

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

    def _get_current_lnposterior_value(self, current_position):
        if self._current_lnposterior_value is None:
            return self.lnposterior(current_position)
        else:
            return self._current_lnposterior_value

    def sample(self):
        # PROPOSE NEW STATE
        position = self._metachain.head()
        proposed_position = multivariate_normal(position, self._covariance)

        # ACCEPTANCE PROBABILITY
        current_position = self._metachain.head()
        proposed_value = self.lnposterior(proposed_position)
        current_value = self._get_current_lnposterior_value(current_position)
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
            self._current_lnposterior_value = proposed_value
        else:
            self._metachain.reject()
