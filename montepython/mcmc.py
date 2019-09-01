from abc import ABC, abstractmethod
import numpy as np
import h5py
import sys

class MCMC(ABC):
    """
    Abstract MCMC base class. All actual sampling classes (such as HMC)
    inherits the methods in this class, and implement the abstract methods.

    All pertinent information about the sampler, including the Markov chain
    itself, can be accessed with the methods provided here.

    :param bayes:
        # TODO

    :param startpos:
        The starting position of the Markov chain.

    :param *args:
        Not used.

    :param **kwargs:
        Implementation-specific parameters. Please see the documentation for
        the specific sampler you want to use to learn about parameters
        specific to that sampler.

    """

    def __init__(self, bayes, *args=[], **kwargs={}):
        self._bayes = bayes

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

    def to_disk(self, filename=None, dataset_name=None, *args, **kwargs):
        """Save the MCMC chain with metadata in an HDF5 file."""
        if filename is None:
            filename = 'out.hdf5'
        if dataset_name is None:
            dataset_nate = self.to_ugly_string()
        f = h5py.File(filename)
        dset = f[dataset_name]
        dset[...] = self.chain()
        dset.attrs['acceptance_fraction'] = self.acceptance_fraction()
        dset.attrs['ndim'] = self.ndim()
        dset.attrs['startpos'] = self._metachain.startpos()
        dset.attrs['mcmc_type'] = self.get_mcmc_type()
        dset.attrs['montepython_version'] = montepython.__version__
        for key, value in kwargs.items():
            dset.attrs[key] = value

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
            np.exp(diff / self._temperature)
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
    def lnprob(self, state):
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
    def get_mcmc_type(self):
        """Return a string with the name of the MCMC algorithm."""
        raise NotImplementedError("Unimplemented abstract method!")
