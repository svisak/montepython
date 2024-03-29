import numpy as np
import copy

class MetaChain():
    """
    This class is used to store relevant data about the
    sampling process, such as the resulting Markov chain
    and the acceptance rate.
    This, in combination with State, is a very versatile class,
    allowing for consistent storage of almost any information
    about the samples. For example, the posterior value of the
    latest accepted sample is stored here in order to avoid
    unnecessary recomputations.
    """

    def __init__(self, initial_state):
        self._states = []
        self._n_accepted = -1
        self.accept(initial_state)

    def reset(self):
        """
        Reset the metachain to the state it
        was in just after initialization.
        """
        self.__init__(self._states[0])

    def accept(self, state):
        """
        Add state to the chain and increment
        the number of accepted samples.
        """
        self._states.append(state)
        self._n_accepted += 1

    def reject(self):
        """Copy the previous head to the new head."""
        self._states.append(copy.deepcopy(self.head()))

    def head(self):
        """Return the latest state in the chain."""
        return self._states[-1]

    def startpos(self):
        """Return the starting position of the chain."""
        return self._states[0].get('position')

    def chain_with_startpos(self):
        """Return the Markov chain including startpos."""
        chain = np.empty((self.chain_length(), self.ndim()))
        for i in range(self.chain_length()):
            chain[i, :] = self._states[i].get('position')
        return chain

    def chain(self):
        """Return the Markov chain, excluding the start position."""
        chain = self.chain_with_startpos()
        return chain[1:, :]

    def acceptance_rate(self):
        """
        Return the current acceptance rate.
        The start position is not included in the calculation.
        """
        if 0 == self.chain_length()-1:
            return 0.
        return self._n_accepted / (self.chain_length()-1)

    def chain_length(self):
        """
        Return the length of the ndarray. This may include elements
        that are not yet filled.
        """
        return len(self._states)

    def ndim(self):
        """Return the number of dimensions in the chain."""
        return len(self._states[0].get('position'))
