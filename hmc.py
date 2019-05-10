#!/usr/bin/env python
# HMC sampling class.

import montepython
import numpy as np

class HMC(montepython.MontePython):

    def __init__(self, gradient, ell, epsilon, *args, **kwargs):
        # FOR TIMING LEAPFROG
        self.exec_time = 0
        # POP OPTIONAL PARAMETERS
        self.save_momenta = kwargs.pop('save_momenta', False)
        self.temp = kwargs.pop('temp', 1) # TODO Make sure temp != 1 works as intended
        mass_matrix = kwargs.pop('mass_matrix', None)

        # SEND THE REST OF THE PARAMETERS OFF TO BASE CLASS
        super().__init__(*args, **kwargs)

        # INSTANTIATE ENERGY AND LEAPFROG
        self.energy = None
        if None == mass_matrix:
            self.energy = Energy(self.lnposterior, np.eye(self.chain.dimensionality()))
        else:
            self.energy = Energy(self.lnposterior, mass_matrix)
        self.leapfrog = Leapfrog(gradient, ell, epsilon, self.energy)

    def draw_momentum(self):
        mean = np.zeros(self.chain.dimensionality())
        cov = np.eye(self.chain.dimensionality())
        return np.random.multivariate_normal(mean, cov)

    def run(self, n_steps):
        self.chain.extend(n_steps)
        for i in range(n_steps):
            # PROPOSE NEW STATE
            position = self.chain.head()
            momentum = self.draw_momentum()
            proposed_state = self.leapfrog.solve(position, momentum)
            proposed_position = proposed_state[0]
            proposed_momentum = proposed_state[1]

            # ACCEPTANCE PROBABILITY
            current_energy = self.energy.hamiltonian(position, momentum)
            proposed_energy = self.energy.hamiltonian(proposed_position, proposed_momentum)
            metropolis_ratio = np.exp((current_energy - proposed_energy) / self.temp)
            acceptance_probability = min(1, metropolis_ratio)

            # ACCEPT / REJECT
            if np.random.rand() < acceptance_probability:
                self.chain.accept(proposed_position)
            else:
                self.chain.reject()


class Energy():

    def __init__(self, lnposterior, mass_matrix):
        self.lnposterior = lnposterior
        self.mass_matrix = mass_matrix
        self.inverse_mass_matrix = np.linalg.inv(mass_matrix)

    def get_mass_matrix(self):
        return self.mass_matrix

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
