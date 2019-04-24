#!/usr/bin/env python
# HMC sampling class.

import montepython
import numpy as np

class HMC(montepython.MontePython):

    def __init__(self, gradient, ell, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leapfrog = Leapfrog(gradient, ell, epsilon)
        self.mass_matrix = np.eye(self.dim)

    # TO SET A CUSTOM MASS MATRIX
    def set_mass_matrix(self, mass_matrix):
        self.mass_matrix = mass_matrix

    # TO MULTIPLY CURRENT MASS MATRIX BY SCALAR
    def set_mass_matrix(self, scalar):
        self.mass_matrix *= scalar

    def propose(self):
        position = self.chain.head()
        mean = np.zeros(self.chain.dimensionality())
        cov = np.eye(self.chain.dimensionality())
        momentum = np.random.multivariate_normal(mean, cov)
        proposed_position, proposed_momentum = self.leapfrog.solve(position, momentum)
        return (proposed_position, proposed_momentum)

    def hamiltonian(self, position, momentum):
        potential_energy = -self.lnposterior(position)
        if np.isinf(potential_energy):
            return potential_energy
        kinetic_energy = np.matmul(momentum, np.matmul(self.mass_matrix, momentum))
        kinetic_energy /= 2.
        return potential_energy + kinetic_energy

    def run(self, n_steps):
        self.chain.extend(n_steps)
        momentum = np.empty(self.chain.dimensionality())
        for i in range(1, n_steps):
            proposed_position, proposed_momentum = self.propose()

            # METROPOLIS RATIO
            proposed_energy = self.hamiltonian(proposed_position, proposed_momentum)
            current_energy = self.hamiltonian(self.chain.head(), momentum)
            exponent = current_energy - proposed_energy
            ratio = 1.
            if exponent < 0.:
                ratio = np.exp(exponent)

            # ACCEPT / REJECT
            if np.random.rand() < ratio:
                self.chain.accept(proposed_position)
                momentum = proposed_momentum
            else:
                self.chain.reject()


class Leapfrog():

    def __init__(self, gradient, ell, epsilon):
        self.gradient = gradient
        self.ell = ell
        self.epsilon = epsilon
        self.all_positions = []
        self.all_momenta = []

    def draw_ell(self):
        return self.ell
        # return self.ell + np.random.randint(-self.ell // 5, self.ell // 5)

    def draw_epsilon(self):
        return self.epsilon
        # return self.epsilon + 0.1*self.epsilon*np.random.rand()

    def solve(self, position, momentum):
        self.all_positions = []
        self.all_momenta = []
        self.all_positions.append(position)
        self.all_momenta.append(momentum)
        ell = self.draw_ell()
        epsilon = self.draw_epsilon()

        # SOLVE AND RETURN
        momentum = momentum - epsilon * self.gradient(position) / 2
        #self.all_momenta.append(momentum)
        for i in range(1, ell):
            position = position + epsilon * momentum
            self.all_positions.append(position)
            if (i != ell):
                momentum = momentum - epsilon * self.gradient(position)
                self.all_momenta.append(momentum)
        momentum = momentum - epsilon * self.gradient(position) / 2
        #self.all_momenta.append(momentum)
        momentum = -momentum
        return (position, momentum)
