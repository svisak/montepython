# HMC sampling class.

from .mcmc import MCMC
import numpy as np

class HMC(MCMC):
    """
    Implementation of the MCMC base class, using the Hamiltonian Monte Carlo
    algorithm. Users should not use the methods in this class, instead use the
    methods provided in the base class.

    :param **kwargs:
        MCMC parameters specific to HMC.

        leapfrog_ell:
            The nominal number of steps used by the leapfrog solver.

        leapfrog_epsilon:
            The nominal step length used by the leapfrog solver.

        mass_matrix: (optional)
            The HMC mass matrix to use. A ndim x ndim matrix.
            [Do not use this yet, as mass matrices other than the default will
            lead to incorrect results!]

        temperature: (optional)
            The sampling temperature. Default: 1. Use with caution for the moment!

    """

    def __init__(self, startpos, *args, **kwargs):
        super().__init__(*args)

        self._bayes.evaluate(startpos)
        lnposterior = self._bayes.lnposterior()
        momentum = self.draw_momentum()
        gradient = self._bayes.gradient()
        initial_state = HMCState(startpos, lnposterior, momentum, gradient)
        self._metachain = MetaChain(initial_state)

        # POP MANDATORY PARAMETERS
        ell = kwargs.pop('leapfrog_ell')
        epsilon = kwargs.pop('leapfrog_epsilon')

        # POP OPTIONAL PARAMETERS
        default_mass_matrix = np.eye(self._metachain.dimensionality())
        self._mass_matrix = kwargs.pop('mass_matrix', default_mass_matrix)
        self._inverse_mass_matrix = np.linalg.inv(self._mass_matrix)

        # INSTANTIATE LEAPFROG
        tmp = self._inverse_mass_matrix
        self._leapfrog = Leapfrog(self._bayes, ell, epsilon, tmp)

    # STATE PROPOSAL
    def propose_state(self, current_state):
        current_state.set_momentum(self.draw_momentum())
        return self._leapfrog.solve(current_state)

    def draw_momentum(self):
        mean = np.zeros(self._metachain.dimensionality())
        cov = np.eye(self._metachain.dimensionality())
        return np.random.multivariate_normal(mean, cov)

    # JOINT LNPROB
    def joint_lnprob(self, state):
        # The minus is because P = exp(-H), so ln P = -H
        # P is the joint probability for the position and momentum
        return -self.hamiltonian(state)

    def hamiltonian(self, state):
        potential_energy = self.potential(state)
        kinetic_energy = self.kinetic(state)
        if np.isinf(potential_energy):
            return potential_energy
        elif np.isinf(kinetic_energy):
            return kinetic_energy
        else:
            return potential_energy + kinetic_energy

    def potential(self, state):
        return -state.lnposterior()

    def kinetic(self, state):
        momentum = state.momentum()
        inv = self.get_inverse_mass_matrix()
        kinetic_energy = momentum @ inv @ momentum
        kinetic_energy /= 2
        if np.isnan(kinetic_energy):
            msg = 'NaN: Energy.kinetic at momentum = {}'.format(momentum)
            raise ValueError(msg)
        return kinetic_energy

    # ALGORITHM INFORMATION - GENERAL
    def to_ugly_string(self):
        n = self._metachain.chain_length()
        ndim = self.get_ndim()
        ell = self._leapfrog.get_ell()
        eps = self._leapfrog.get_epsilon()
        str = "hmc_N{}_ndim{}_L{}_eps{}".format(n, ndim, ell, eps)
        return str

    def to_pretty_string(self):
        n = self._metachain.chain_length()
        ndim = self.get_ndim()
        ell = self._leapfrog.get_ell()
        eps = self._leapfrog.get_epsilon()
        str = "HMC, {} samples, {} leapfrog steps of length {}".format(n, ndim, ell, eps)
        return str

    def get_mcmc_type(self):
        return "HMC"

    # ALGORITHM INFORMATION - HMC SPECIFIC
    def get_mass_matrix(self):
        return self._mass_matrix

    def get_inverse_mass_matrix(self):
        return self._inverse_mass_matrix
