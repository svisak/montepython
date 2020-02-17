import numpy as np
from numpy.random import multivariate_normal

from .mcmc import MCMC
from .state import State

class RWM(MCMC):

    def __init__(self, bayes, startpos, **kwargs):
        # CALL SUPERCLASS CONSTRUCTOR
        super().__init__(bayes, startpos, **kwargs)

        # POP OPTIONAL PARAMETERS
        self._stepsize = kwargs.pop('stepsize', 1.0)

        # CREATE THE COVARIANCE MATRIX
        self._covariance = (self._stepsize ** 2) * np.eye(self.ndim())

    @property
    def stepsize(self):
        return self._stepsize

    def to_disk(self, *args, **kwargs):
        kwargs['stepsize'] = self.stepsize
        super().to_disk(*args, **kwargs)

    # STATE PROPOSAL
    def propose_state(self, current_state):
        pos = current_state.get('position')
        cov = self._covariance
        proposed_position = multivariate_normal(pos, cov)
        self._bayes.evaluate(proposed_position)
        tmp = {}
        tmp['position'] = proposed_position
        tmp['lnposterior'] = self._bayes.get_lnposterior_value()
        proposed_state = State(**tmp)
        return proposed_state

    # JOINT LNPROB
    def joint_lnprob(self, state):
        # This is trivial for RWM, but not for HMC
        return state.get('lnposterior')

    # ALGORITHM INFORMATION
    def to_ugly_string(self):
        n = self._metachain.chain_length()
        ndim = self.ndim()
        stepsize = self._covariance[0, 0]
        str = "rwm_N{}_ndim{}_stepsize{}".format(n, ndim, stepsize)
        return str

    def to_pretty_string(self):
        n = self._metachain.chain_length()
        ndim = self.ndim()
        stepsize = self._covariance[0, 0]
        str = "RWM, {} samples, stepsize {}".format(n, ndim, stepsize)
        return str

    def mcmc_type(self, uppercase=False, expand=False):
        if uppercase is True:
            return "RWM"
        elif expand is True:
            return "Random Walk Metropolis"
        else:
            return "rwm"
