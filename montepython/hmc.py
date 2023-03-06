# HMC sampling class.

import numpy as np
import sys

from .mcmc import MCMC
from .leapfrog import Leapfrog
from .utils import check_positive_semidefinite

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

        temperature: (optional)
            The sampling temperature. Default: 1. Use with caution for the moment!

    """

    def __init__(self, bayes, startpos, **kwargs):
        # CALL SUPERCLASS CONSTRUCTOR
        super().__init__(bayes, startpos, **kwargs)

        # POP MANDATORY PARAMETERS
        # n_timesteps, dt are just alternate names for ell and epsilon.
        ell = kwargs.pop('n_timesteps', None)
        if ell is None:
            ell = kwargs.pop('leapfrog_ell')
        epsilon = kwargs.pop('dt', None)
        if epsilon is None:
            epsilon = kwargs.pop('leapfrog_epsilon')

        # POP OPTIONAL PARAMETERS
        default_mass_matrix = np.eye(self.ndim())
        self._mass_matrix = kwargs.pop('mass_matrix', default_mass_matrix)
        self._inverse_mass_matrix = np.linalg.inv(self._mass_matrix)

        # CHECK THAT MASS MATRIX IS POSITIVE SEMIDEFINITE.
        # np.random.multivariate_normal MAY STILL GIVE WARNINGS; THESE CAN BE IGNORED.
        if not check_positive_semidefinite(self._mass_matrix):
            print('Mass matrix is not positive semidefinite! Exiting.')
            sys.exit(1)

        # INSTANTIATE LEAPFROG
        tmp = self._inverse_mass_matrix
        self._leapfrog = Leapfrog(self._bayes, ell, epsilon, tmp)

        # SET INITIAL VALUES
        self._metachain.head().set('momentum', self.draw_momentum())
        self._metachain.head().set('nlp_gradient', self._bayes.get_nlp_gradient_value())


    def to_disk(self, *args, **kwargs):
        kwargs['leapfrog_ell'] = self._leapfrog.get_ell()
        kwargs['leapfrog_epsilon'] = self._leapfrog.get_epsilon()
        kwargs['mass_matrix'] = self._mass_matrix
        kwargs['inverse_mass_matrix'] = self._inverse_mass_matrix
        return super().to_disk(*args, **kwargs)

    # STATE PROPOSAL
    def propose_state(self, current_state):
        current_state.set('momentum', self.draw_momentum())
        return self._leapfrog.solve(current_state)

    def draw_momentum(self):
        mean = np.zeros(self.ndim())
        cov = self.get_mass_matrix()
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
        return -state.get('lnposterior')

    def kinetic(self, state):
        momentum = state.get('momentum')
        inv = self.get_inverse_mass_matrix()
        kinetic_energy = momentum.T @ inv @ momentum
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

    def mcmc_type(self, uppercase=False, expand=False):
        if uppercase is True:
            return "HMC"
        elif expand is True:
            return "Hamiltonian Monte Carlo"
        else:
            return "hmc"

    # ALGORITHM INFORMATION - HMC SPECIFIC
    def get_mass_matrix(self):
        return self._mass_matrix

    def get_inverse_mass_matrix(self):
        return self._inverse_mass_matrix
