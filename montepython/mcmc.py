from abc import ABC, abstractmethod
import numpy as np
import sys

from .state import State
from .metachain import MetaChain
from .utils import mcmc_to_disk
import montepython


class MCMC(ABC):
    """
    Abstract MCMC base class. All actual sampling classes (such as HMC)
    inherits the methods in this class, and implement the abstract methods.

    All pertinent information about the sampler, including the Markov chain
    itself, can be accessed with the methods provided here.

    :param initial_state:
        The starting State of the Markov chain.

    :param *args:
        Not used.

    :param **kwargs:
        Implementation-specific parameters. Please see the documentation for
        the specific sampler you want to use to learn about parameters
        specific to that sampler.

    """

    def __init__(self, bayes, startpos, **kwargs):
        # MAKE SURE STARTPOS IS AN NDARRAY
        if not isinstance(startpos, np.ndarray):
            msg = 'startpos is not an ndarray'
            raise ValueError(msg)

        # SET BAYES
        self._bayes = bayes

        # SET COMMON INITIAL STATE PARAMETERS
        self._bayes.evaluate(startpos)
        tmp = {}
        tmp['position'] = startpos
        tmp['lnposterior'] = self._bayes.get_lnposterior_value()
        initial_state = State(**tmp)

        # CREATE METACHAIN
        self._metachain = MetaChain(initial_state)

        # POP OPTIONAL PARAMETERS
        self._temperature = kwargs.pop('temperature', 1.0)

    def set_seed(self, seed):
        """Set the numpy random seed."""
        np.random.seed(seed)

    def acceptance_fraction(self):
        """Return the acceptance fraction of the samples so far."""
        return self._metachain.acceptance_fraction()

    def chain(self):
        """
        Return the Markov chain resulting from the sampling.
        This is an ndarray.
        """
        return self._metachain.chain()

    def ndim(self):
        return self._metachain.ndim()

    def to_disk(self, filename=None, dataset_name=None, **kwargs):
        """Save the MCMC chain with metadata in an HDF5 file."""

        if filename is not None:
            kwargs['filename'] = filename
        if dataset_name is not None:
            kwargs['dataset_name'] = dataset_name
        kwargs['acceptance_fraction'] = self.acceptance_fraction()
        kwargs['ndim'] = self.ndim()
        kwargs['startpos'] = self._metachain.startpos()
        kwargs['mcmc_type'] = self.mcmc_type()
        kwargs['montepython_version'] = montepython.__version__
        for key, value in kwargs.items():
            kwargs[key] = value
        mcmc_to_disk(self, **kwargs)

    def replace_bayes(self, bayes):
        self._bayes = bayes

    def run(self, n_samples):
        """Run the MCMC sampler for n_samples samples."""
        for i in range(n_samples):
            self.sample()

    def sample(self):
        # PROPOSE NEW STATE
        current_state = self._metachain.head()
        proposed_state = self.propose_state(current_state)

        # ACCEPTANCE PROBABILITY
        current_lnprob = self.joint_lnprob(current_state)
        proposed_lnprob = self.joint_lnprob(proposed_state)
        diff = proposed_lnprob - current_lnprob
        metropolis_ratio = 1
        if 0 > diff:
            metropolis_ratio = np.exp(diff / self._temperature)
        acceptance_probability = metropolis_ratio

        # ACCEPT / REJECT
        if np.random.rand() < acceptance_probability:
            self._metachain.accept(proposed_state)
        else:
            self._metachain.reject()

    @abstractmethod
    def propose_state(self, current_state):
        """Propose new state."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def joint_lnprob(self, state):
        """Return the log probability of the (joint) state."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_ugly_string(self):
        """Return a description, suitable for filenames, of the MCMC object."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_pretty_string(self):
        """
        Return a description, suitable for titles etc, of the MCMC object.
        """
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def mcmc_type(self, uppercase=False):
        """Return a string with the name of the MCMC algorithm."""
        raise NotImplementedError("Unimplemented abstract method!")
