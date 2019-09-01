class MetaChain():
    """
    This class is used to store relevant data about the
    sample process, such as the resulting Markov chain as
    a numpy ndarray and the acceptance fraction.
    It also handles updating of the chain after each sample.

    """

    def __init__(self, initial_state):
        self._states = []
        self._n_accepted = 0
        self._index = -1
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
        the index and accepted samples.
        """
        self._states.append(state)
        self._n_accepted += 1

    def reject(self):
        """Copy the previous head to the new head, and increment index."""
        self._states.append(copy.deepcopy(self.head()))

    def head(self):
        """Return the latest state in the chain."""
        return self._states[-1]

    def startpos(self):
        """Return the first position in the chain."""
        return copy.deepcopy(self._states[0].position())

    def chain(self):
        """Return the Markov chain."""
        chain = np.empty((self.chain_length(), self.ndim()))
        for i in range(self.chain_length()):
            chain[i, :] = self._states[i].position()
        return chain

    def acceptance_fraction(self):
        """Return the current acceptance fraction."""
        return self._n_accepted / self.chain_length()

    def chain_length(self):
        """
        Return the length of the ndarray. This may include elements
        that are not yet filled.
        """
        return len(self._states)

    def ndim(self):
        """Return the number of dimensions in the chain."""
        return len(self._states[0].position())
