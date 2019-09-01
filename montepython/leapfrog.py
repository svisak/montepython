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
        position = initial_state.position()
        momentum = initial_state.momentum()

        ell = self.get_ell()
        epsilon = self.get_epsilon()

        # SOLVE AND RETURN
        lngradient = self._bayes.lngradient()
        momentum = momentum - epsilon * lngradient / 2
        for i in range(ell):
            position += epsilon * self._inverse_mass_matrix @ momentum
            self._bayes.evaluate(position)
            if (i != ell-1):
                momentum -= epsilon * self._bayes.lngradient()
        momentum -= epsilon * self._bayes.lngradient() / 2
        momentum = -momentum
        lnposterior = self._bayes.lnposterior()
        gradient = self._bayes.gradient()
        return HMCState(position, lnposterior, momentum, gradient)
