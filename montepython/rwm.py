from .mcmc import MCMC
import numpy as np
from numpy.random import multivariate_normal

class RWM(MCMC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        stepsize = kwargs.pop('stepsize', 1.0)
        self._covariance = stepsize * np.eye(self.get_ndim())

    # STATE PROPOSAL
    def propose_state(self, current_state):
        pos = current_state.position()
        cov = self._covariance
        proposed_position = multivariate_normal(pos, cov)
        self._bayes.evaluate(proposed_position)
        proposed_lnposterior = self._bayes.lnposterior()
        proposed_state = State(proposed_position, proposed_lnposterior)
        return proposed_state

    # JOINT LNPROB
    def joint_lnprob(self, state):
        # This is trivial for RWM, but not for HMC
        return state.lnposterior()

    # ALGORITHM INFORMATION
    def to_ugly_string(self):
        n = self._metachain.chain_length()
        ndim = self.get_ndim()
        stepsize = self._covariance[0, 0]
        str = "rwm_N{}_ndim{}_stepsize{}".format(n, ndim, stepsize)
        return str

    def to_pretty_string(self):
        n = self._metachain.chain_length()
        ndim = self.get_ndim()
        stepsize = self._covariance[0, 0]
        str = "RWM, {} samples, stepsize {}".format(n, ndim, stepsize)
        return str

    def get_mcmc_type(self):
        return "RWM"

