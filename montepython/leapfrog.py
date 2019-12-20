from .state import State

import numpy as np
import sys

class Leapfrog():
    """Leapfrog solver for HMC."""

    def __init__(self, bayes, ell, epsilon, inverse_mass_matrix):
        self._bayes = bayes
        self._ell = ell
        self._epsilon = epsilon
        self._inverse_mass_matrix = inverse_mass_matrix
        if self._ell < 1:
            print(f'Invalid number ({self._ell}) of leapfrog steps, exiting.',
                                                                file=sys.stderr)
            sys.exit(1)

    def get_ell(self):
        """Return the nominal number of steps used by the solver."""
        ell = np.random.normal(self._ell, self._ell/3)
        ell = int(np.ceil(ell))
        return ell

    def get_epsilon(self):
        """Return the nominal step size used by the solver."""
        return self._epsilon

    def solve(self, initial_state):
        position = initial_state.get('position')
        momentum = initial_state.get('momentum')

        ell = self.get_ell()
        epsilon = self.get_epsilon()

        # SOLVE
        nlp_gradient = self._bayes.get_nlp_gradient_value()
        momentum = momentum - epsilon * nlp_gradient / 2
        for i in range(ell):
            position += epsilon * self._inverse_mass_matrix @ momentum
            self._bayes.evaluate(position)
            if (i != ell-1):
                momentum -= epsilon * self._bayes.get_nlp_gradient_value()
        momentum -= epsilon * self._bayes.get_nlp_gradient_value() / 2
        momentum = -momentum

        # RETURN
        lnposterior = self._bayes.get_lnposterior_value()
        nlp_gradient = self._bayes.get_nlp_gradient_value()
        tmp = {}
        tmp['position'] = position
        tmp['momentum'] = momentum
        tmp['lnposterior'] = lnposterior
        tmp['nlp_gradient'] = nlp_gradient
        return State(**tmp)
