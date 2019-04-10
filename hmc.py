#!/usr/bin/env python
# HMC sampling class.

from .montepython import MontePython

class HMC(MontePython):
    :param gradient:
        A function that, given a state, returns the posterior gradient at that state.

    def __init__(self, *args, **kwargs, gradient):
        super().__init__(*args, **kwargs)
        self.gradient = gradient

    # ell and epsilon are fixed during the solving of Hamilton's equations
    def leapfrog(self, ell, epsilon):
        # TODO

class Hamiltonian():
    def __init__(self, posterior, *args, **kwargs):
        self.potential = posterior
       
    def hamiltonian(self, position, momentum):
        return potential + kinetic # Watch log vs lin
