class State():

    def __init__(self, position, lnposterior):
        self._position = copy.deepcopy(position)
        self._lnposterior = copy.deepcopy(lnposterior)

    def position(self):
        return copy.deepcopy(self._position)

    def lnposterior(self):
        return copy.deepcopy(self._lnposterior)


class HMCState(State):

    def __init__(self, *args, momentum, gradient):
        super().__init__(*args)
        self._momentum = copy.deepcopy(momentum)
        self._gradient = copy.deepcopy(gradient)

    def momentum(self):
        return copy.deepcopy(self._momentum)

    def gradient(self):
        return copy.deepcopy(self._gradient)

    def set_momentum(self, momentum):
        self._momentum = copy.deepcopy(momentum)
