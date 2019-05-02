#!/usr/bin/env python
# HMC sampling class.

import montepython
import numpy as np

class HMC(montepython.MontePython):

    def __init__(self, gradient, ell, epsilon, *args, **kwargs):
        mass_matrix = kwargs.pop('mass_matrix', None)
        super().__init__(*args, **kwargs)
        self.energy = None
        if None == mass_matrix:
            self.energy = Energy(self.lnposterior, np.eye(self.dim))
        else:
            self.energy = Energy(self.lnposterior, mass_matrix)
        self.leapfrog = Leapfrog(gradient, ell, epsilon, self.energy)

    def draw_momentum(self):
        mean = np.zeros(self.chain.dimensionality())
        cov = np.eye(self.chain.dimensionality())
        return np.random.multivariate_normal(mean, cov)

    def propose(self, position, momentum):
        proposed_position, proposed_momentum = self.leapfrog.solve(position, momentum)
        return (proposed_position, proposed_momentum)

    def run(self, n_steps):
        self.chain.extend(n_steps)
        for i in range(1, n_steps):
            position = self.chain.head()
            momentum = self.draw_momentum()
            proposed_position, proposed_momentum = self.propose(position, momentum)

            # METROPOLIS RATIO
            current_energy = self.energy.hamiltonian(position, momentum)
            proposed_energy = self.energy.hamiltonian(proposed_position, proposed_momentum)
            exponent = current_energy - proposed_energy
            ratio = 1.
            if exponent < 0.:
                ratio = np.exp(exponent)

            # ACCEPT / REJECT
            if np.random.rand() < ratio:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()


class Energy():

    def __init__(self, lnposterior, mass_matrix):
        self.lnposterior = lnposterior
        self.inverse_mass_matrix = np.linalg.inv(mass_matrix)

    def get_inverse_mass_matrix(self):
        return self.inverse_mass_matrix

    def potential(self, position):
        return -self.lnposterior(position)

    def kinetic(self, momentum):
        tmp = np.matmul(self.inverse_mass_matrix, momentum)
        kinetic_energy = np.matmul(momentum, tmp) / 2
        if np.isnan(kinetic_energy):
            raise ValueError('NaN: Energy.kinetic at momentum = {}'.format(momentum))
        return kinetic_energy

    def hamiltonian(self, position, momentum):
        potential_energy = self.potential(position)
        kinetic_energy = self.kinetic(momentum)
        if np.isinf(potential_energy):
            return potential_energy
        elif np.isinf(kinetic_energy):
            return kinetic_energy
        else:
            return potential_energy + kinetic_energy


class Leapfrog():

    def __init__(self, gradient, ell, epsilon, energy):
        self.gradient = gradient
        self.ell = ell
        self.epsilon = epsilon
        self.energy = energy

    def draw_ell(self):
        return self.ell
        # return self.ell + np.random.randint(-self.ell // 5, self.ell // 5)

    def draw_epsilon(self):
        return self.epsilon
        # return self.epsilon + 0.1*self.epsilon*np.random.rand()

    def solve(self, position, momentum):
        ell = self.draw_ell()
        epsilon = self.draw_epsilon()

        # SOLVE AND RETURN
        momentum = momentum - epsilon * self.gradient(position) / 2
        for i in range(ell):
            tmp = np.matmul(self.energy.get_inverse_mass_matrix(), momentum)
            position = position + epsilon * tmp
            if (i != ell-1):
                momentum = momentum - epsilon * self.gradient(position)
        momentum = momentum - epsilon * self.gradient(position) / 2
        momentum = -momentum
        return (position, momentum)
