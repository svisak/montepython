#!/usr/bin/env python
# HMC sampling class.

from .mcmc import MCMC
import numpy as np

class HMC(MCMC):
    """
    Implementation of the MCMC base class, using the Hamiltonian Monte Carlo
    algorithm. Users should not use the methods in this class, instead use the
    methods provided in the base class.

    :param **kwargs:
        All parameters for the MCMC sampler are supplied as keyword arguments.
        The list below covers the ones that are specific to HMC; see the docu-
        mentation for the base class for parameters common to every sampler.

        Class parameters
        ----------------
        gradient:
            A function that takes a vector in parameter space as input and
            returns the gradient of the negative log posterior with respect
            to the parameters for that position.

        leapfrog_ell:
            The nominal number of steps used by the leapfrog solver.

        leapfrog_epsilon:
            The nominal step length used by the leapfrog solver.

        mass_matrix: (optional)
            The HMC mass matrix to use. A dim x dim matrix.
            [Do not use this yet, as mass matrices other than the default will
            lead to incorrect results!]

        save_momenta: (optional)
            A boolean variable. If set to true the randomly drawn momenta will
            be saved in a MetaChain object. Default: false.
            [Currently unimplemented!]

        temperature: (optional)
            The sampling temperature. Default: 1. Use with caution for the moment!

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # POP MANDATORY PARAMETERS
        gradient = kwargs.pop('gradient')
        ell = kwargs.pop('leapfrog_ell')
        epsilon = kwargs.pop('leapfrog_epsilon')

        # POP OPTIONAL PARAMETERS
        self._save_momenta = kwargs.pop('save_momenta', False)
        self._temperature = kwargs.pop('temperature', 1) # TODO Make sure temperature != 1 works as intended
        default_mass_matrix = np.eye(self._metachain.dimensionality())
        mass_matrix = kwargs.pop('mass_matrix', default_mass_matrix)

        # INSTANTIATE ENERGY AND LEAPFROG
        self._energy = Energy(self.lnposterior, mass_matrix)
        self._leapfrog = Leapfrog(gradient, ell, epsilon, self._energy)

    def to_ugly_string(self):
        n = self._metachain.steps_taken()
        dim = self._metachain.dimensionality()
        ell = self._leapfrog.get_ell()
        eps = self._leapfrog.get_epsilon()
        str = "hmc_N{}_dim{}_L{}_eps{}".format(n, dim, ell, eps)
        return str

    def to_pretty_string(self):
        n = self._metachain.steps_taken()
        dim = self._metachain.dimensionality()
        ell = self._leapfrog.get_ell()
        eps = self._leapfrog.get_epsilon()
        str = "HMC, {} samples, {} leapfrog steps of length {}".format(n, dim, ell, eps)
        return str

    def get_mcmc_type(self):
        return "HMC"

    def draw_momentum(self):
        mean = np.zeros(self._metachain.dimensionality())
        cov = np.eye(self._metachain.dimensionality())
        return np.random.multivariate_normal(mean, cov)

    def sample(self):
        # PROPOSE NEW STATE
        current_state = State(self._metachain.head(), self.draw_momentum())
        proposed_state = self._leapfrog.solve(current_state)

        # ACCEPTANCE PROBABILITY
        current_energy = self._energy.hamiltonian(current_state)
        proposed_energy = self._energy.hamiltonian(proposed_state)
        diff = current_energy - proposed_energy
        metropolis_ratio = np.exp(diff / self._temperature)
        acceptance_probability = min(1, metropolis_ratio)

        # ACCEPT / REJECT
        if np.random.rand() < acceptance_probability:
            self._metachain.accept(proposed_state.position())
        else:
            self._metachain.reject()


class State():
    """Hold a position-momentum pair. For code clarity."""

    def __init__(self, position, momentum):
        self._position = position
        self._momentum = momentum

    def position(self):
        return self._position

    def momentum(self):
        return self._momentum


class Energy():
    """Keep track of the energy expressions needed in HMC."""

    def __init__(self, lnposterior, mass_matrix):
        self._lnposterior = lnposterior
        self._mass_matrix = mass_matrix
        self._inverse_mass_matrix = np.linalg.inv(mass_matrix)

    def get_mass_matrix(self):
        return self._mass_matrix

    def get_inverse_mass_matrix(self):
        return self._inverse_mass_matrix

    def potential(self, position):
        return -self._lnposterior(position)

    def kinetic(self, momentum):
        kinetic_energy = momentum @ self._inverse_mass_matrix @ momentum
        kinetic_energy /= 2
        if np.isnan(kinetic_energy):
            raise ValueError('NaN: Energy.kinetic at momentum = {}'.format(momentum))
        return kinetic_energy

    def hamiltonian(self, state):
        potential_energy = self.potential(state.position())
        kinetic_energy = self.kinetic(state.momentum())
        if np.isinf(potential_energy):
            return potential_energy
        elif np.isinf(kinetic_energy):
            return kinetic_energy
        else:
            return potential_energy + kinetic_energy


class Leapfrog():
    """Leapfrog solver for HMC."""

    def __init__(self, gradient, ell, epsilon, energy):
        self._gradient = gradient
        self._ell = ell
        self._epsilon = epsilon
        self._energy = energy

    def draw_ell(self):
        return self._ell
        # return self.ell + np.random.randint(-self.ell // 5, self.ell // 5)

    def draw_epsilon(self):
        return self._epsilon
        # return self.epsilon + 0.1*self.epsilon*np.random.rand()

    def get_ell(self):
        """Return the nominal number of steps used by the solver."""
        return self._ell

    def get_epsilon(self):
        """Return the nominal step size used by the solver."""
        return self._epsilon

    def solve(self, initial_state):
        position = initial_state.position()
        momentum = initial_state.momentum()

        # PERTURB ELL AND EPSILON
        ell = self.draw_ell()
        epsilon = self.draw_epsilon()

        # SOLVE AND RETURN
        momentum = momentum - epsilon * self._gradient(position) / 2
        for i in range(ell):
            position += epsilon * self._energy.get_inverse_mass_matrix() @ momentum
            if (i != ell-1):
                momentum -= epsilon * self._gradient(position)
        momentum -= epsilon * self._gradient(position) / 2
        momentum = -momentum
        return State(position, momentum)
