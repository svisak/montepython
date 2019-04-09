#!/usr/bin/env python
"""
HMC sampling class.
"""

from .montepython import MontePython

# Hamiltonian Monte Carlo sampler
class Spamilton(MontePython):

    :param gradient:
        A function that, given a state, returns the gradient at that state.

    def __init__(self, gradient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient = gradient
