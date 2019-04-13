#!/usr/bin/env python
# HMC sampling class.

import montepython

class HMC(MontePython):
    :param gradient:
        A function that, given a state, returns the posterior gradient at that state.

    def __init__(self, *args, **kwargs, gradient):
        super().__init__(*args, **kwargs)
        self.gradient = gradient
        self.mass_matrix = np.eye(self.dim)

    def set_mass_matrix(self, mass_matrix):
        self.mass_matrix = mass_matrix

    def metropolis_ratio(self, proposed_position, proposed_momentum, current_momentum):
        current_position = self.status.state()
        exponent = posterior(proposed_state) - posterior(current_state)

    def propose_state(self):
        position = self.status.state()
        momentum_mean = np.zeros(self.dim)
        momentum_cov = np.eye(self.dim)
        momentum = np.random.multivariate_normal(momentum_mean, momentum_cov)
        proposed_position, proposed_momentum = leapfrog(position, momentum)


    def leapfrog(self, position, momentum):
        # TODO

    def run(self, n_steps)
        extend_chain(n_steps)
        for i in range(0, n_steps-1):
            proposed_position, proposed_momentum = propose_state()
            ratio = metropolis_ratio(proposed_position, proposed_momentum, current_momentum)
            maybe_accept(proposed_position, ratio)

class Hamiltonian():
    def __init__(self, posterior, *args, **kwargs):
        self.potential = posterior
       
    def hamiltonian(self, position, momentum):
        return potential + kinetic # Watch log vs lin

class Status(montepython.Status):

    def __init__(self, *args):
        super().__init__(*args)
        self._momentum = np.zeros(self.dim)

