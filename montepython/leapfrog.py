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

    def get_epsilon(self):
        """Return the step size used by the solver."""
        return self._epsilon

    def get_ell(self):
        """Return the nominal number of steps used by the solver."""
        return self._ell

    def _draw_epsilon(self):
        """
        Draw a random step size from a uniform distribution
        centered around self._epsilon.
        """
        half_epsilon = self._epsilon / 2
        low = self._epsilon - half_epsilon
        high = self._epsilon + half_epsilon
        epsilon = np.random.uniform(low, high)
        return epsilon

    def _draw_ell(self):
        """
        Draw a random number of steps from a uniform distribution
        centered around self._ell.
        """
        half_ell = self._ell // 2
        low = self._ell - half_ell # The lower bound is inclusive
        high = self._ell + half_ell + 1 # The upper bound is exclusive so add 1
        ell = np.random.randint(low, high)
        return ell

    def solve(self, initial_state):
        position = initial_state.get('position')
        momentum = initial_state.get('momentum')

        ell = self._draw_ell()
        epsilon = self._draw_epsilon()

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
