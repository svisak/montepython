from .state import State

class Leapfrog():
    """Leapfrog solver for HMC."""

    def __init__(self, bayes, ell, epsilon, inverse_mass_matrix):
        self._bayes = bayes
        self._ell = ell
        self._epsilon = epsilon
        self._inverse_mass_matrix = inverse_mass_matrix

    def get_ell(self):
        """Return the nominal number of steps used by the solver."""
        return self._ell

    def get_epsilon(self):
        """Return the nominal step size used by the solver."""
        return self._epsilon

    def solve(self, initial_state):
        position = initial_state.get('position')
        momentum = initial_state.get('momentum')

        ell = self.get_ell()
        epsilon = self.get_epsilon()

        # SOLVE
        lngradient = self._bayes.get_lngradient_value()
        momentum = momentum - epsilon * lngradient / 2
        for i in range(ell):
            position += epsilon * self._inverse_mass_matrix @ momentum
            self._bayes.evaluate(position)
            if (i != ell-1):
                momentum -= epsilon * self._bayes.get_lngradient_value()
        momentum -= epsilon * self._bayes.get_lngradient_value() / 2
        momentum = -momentum

        # RETURN
        lnposterior = self._bayes.get_lnposterior_value()
        gradient = self._bayes.get_gradient_value()
        tmp = {}
        tmp['position'] = position
        tmp['momentum'] = momentum
        tmp['lnposterior'] = lnposterior
        tmp['gradient'] = gradient
        return State(**tmp)
