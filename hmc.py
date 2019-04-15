#!/usr/bin/env python
# HMC sampling class.

import montepython

class HMC(MontePython):

    def __init__(self, *args, **kwargs, gradient, ell, epsilon):
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
        mean = np.zeros(self.chain.dim())
        cov = np.eye(self.chain.dim())
        momentum = np.random.multivariate_normal(mean, cov)
        proposed_position, proposed_momentum = leapfrog.solve(position, momentum)
        return (proposed_position, proposed_momentum)

    def hamiltonian(self, position, momentum)
        potential_energy = posterior(position)
        kinetic_energy = np.matmul(momentum, np.matmul(self.mass_matrix, momentum))
        kinetic_energy /= 2.
        return potential_energy + kinetic_energy

    def run(self, n_steps):
        extend_chain(n_steps)
        momentum = np.zeros(self.chain.dim())
        for i in range(0, n_steps-1):
            proposed_position, proposed_momentum = propose()

            # METROPOLIS RATIO
            proposed_energy = hamiltonian(proposed_position, proposed_momentum)
            current_energy = hamiltonian(self.chain.head(), momentum)
            exponent = proposed_energy - current_energy
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

    def draw_ell(self):
        return self.ell + np.random.randint(-self.ell // 5, self.ell // 5)

    def draw_epsilon(self):
        return self.epsilon + 0.1*self.epsilon*np.random.rand()

    def solve(self, position, momentum):
        ell = draw_ell()
        epsilon = draw_epsilon()

        # SOLVE AND RETURN
        momentum = momentum - self.gradient(position) / 2
        for i in range(1, ell):
            position = position + epsilon * momentum
            if (i != ell):
                momentum = momentum - epsilon * self.gradient(position)
        momentum = momentum - epsilon * self.gradient(position) / 2
        momentum = -momentum
        return (position, momentum)
